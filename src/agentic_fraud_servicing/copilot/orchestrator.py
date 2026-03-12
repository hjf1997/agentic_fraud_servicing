"""Central copilot orchestrator with running state.

Processes transcript events, maintains running hypothesis scores and
impersonation risk, invokes specialist agents (triage, auth, question
planner, retrieval, hypothesis), and produces CopilotSuggestion output
for each turn. This is a plain Python class — not an Agents SDK Agent —
keeping the control flow explicit and auditable.
"""

import uuid
from datetime import datetime, timezone

from agents import ModelProvider

from agentic_fraud_servicing.copilot.auth_agent import AuthAssessment, run_auth_assessment
from agentic_fraud_servicing.copilot.hypothesis_agent import HypothesisAssessment, run_hypothesis
from agentic_fraud_servicing.copilot.question_planner import QuestionPlan, run_question_planner
from agentic_fraud_servicing.copilot.retrieval_agent import RetrievalResult, run_retrieval
from agentic_fraud_servicing.copilot.triage_agent import run_triage
from agentic_fraud_servicing.gateway.tool_gateway import AuthContext, ToolGateway
from agentic_fraud_servicing.gateway.tools.write_tools import append_evidence_node
from agentic_fraud_servicing.models.case import CopilotSuggestion
from agentic_fraud_servicing.models.claims import ClaimExtraction, ClaimExtractionResult
from agentic_fraud_servicing.models.enums import EvidenceSourceType
from agentic_fraud_servicing.models.evidence import ClaimStatement
from agentic_fraud_servicing.models.transcript import TranscriptEvent

# Standard fields to gather during a dispute call
_INITIAL_MISSING_FIELDS = [
    "transaction_date",
    "merchant_name",
    "amount",
    "auth_method",
]

# Mapping from missing field names to keywords that indicate the field
# has been addressed in the transcript text (simple heuristic)
_FIELD_KEYWORDS: dict[str, list[str]] = {
    "transaction_date": ["date", "when", "day", "yesterday", "last week", "month"],
    "merchant_name": ["merchant", "store", "shop", "vendor", "company", "amazon", "walmart"],
    "amount": ["amount", "dollar", "charge", "charged", "$", "paid", "cost"],
    "auth_method": ["pin", "chip", "swipe", "tap", "contactless", "online", "signature"],
}


class CopilotOrchestrator:
    """Central orchestrator for the realtime copilot.

    Maintains running state across transcript events and invokes specialist
    agents to produce CopilotSuggestion output for each turn. Uses a
    hub-and-spoke pattern where each specialist is called explicitly via
    its run_* wrapper function.

    Attributes:
        gateway: ToolGateway for mediated data access.
        model_provider: LLM provider for all specialist agents.
        case_id: Set from the first transcript event.
        call_id: Set from the first transcript event.
        hypothesis_scores: Running 4-category investigation hypothesis scores.
        impersonation_risk: Current impersonation risk (0.0-1.0).
        missing_fields: Fields still needed from the caller.
        evidence_collected: Evidence references gathered so far.
        transcript_history: All transcript events processed.
        accumulated_claims: All claims extracted across turns.
    """

    def __init__(self, gateway: ToolGateway, model_provider: ModelProvider) -> None:
        self.gateway = gateway
        self.model_provider = model_provider
        self.case_id: str | None = None
        self.call_id: str | None = None
        self.hypothesis_scores: dict[str, float] = {
            "THIRD_PARTY_FRAUD": 0.0,
            "FIRST_PARTY_FRAUD": 0.0,
            "SCAM": 0.0,
            "DISPUTE": 0.0,
        }
        self.impersonation_risk: float = 0.0
        self.missing_fields: list[str] = list(_INITIAL_MISSING_FIELDS)
        self.evidence_collected: list[str] = []
        self.transcript_history: list[TranscriptEvent] = []
        self.accumulated_claims: list[ClaimExtraction] = []
        self._retrieval_result: RetrievalResult | None = None

    async def process_event(self, event: TranscriptEvent) -> CopilotSuggestion:
        """Process a single transcript event and return copilot suggestions.

        Runs 5-step pipeline: triage (claim extraction) → auth → question
        planner → retrieval → hypothesis scoring. Each specialist call is
        wrapped in try/except for graceful degradation.

        Args:
            event: The transcript event to process.

        Returns:
            CopilotSuggestion combining all specialist results.
        """
        # 1. Append event and set case_id/call_id on first event
        self.transcript_history.append(event)
        if self.case_id is None:
            self.case_id = f"case-{event.call_id}"
            self.call_id = event.call_id

        risk_flags: list[str] = []
        suggested_questions: list[str] = []
        retrieved_facts: list[str] = []

        # 2. Run retrieval once at the start to pre-fetch data
        if self._retrieval_result is None and self.case_id is not None:
            self._retrieval_result = await self._run_retrieval_safe(risk_flags)

        # Extract data from retrieval for downstream agents
        retrieval = self._retrieval_result
        auth_events = retrieval.auth_events if retrieval else []
        customer_profile = retrieval.customer_profile if retrieval else None

        # 3. Run triage agent (claim extraction) with full conversation history
        conversation_history = [(e.speaker.value, e.text) for e in self.transcript_history]
        triage_result = await self._run_triage_safe(event.text, risk_flags, conversation_history)
        if triage_result is not None and triage_result.claims:
            self.accumulated_claims.extend(triage_result.claims)
            self._persist_claims(triage_result.claims)

        # Build running summary from accumulated claims
        running_summary = self._build_claims_summary()

        # 4. Run auth assessment agent
        auth_result = await self._run_auth_safe(
            event.text, auth_events, customer_profile, risk_flags
        )
        if auth_result is not None:
            self.impersonation_risk = auth_result.impersonation_risk
            if auth_result.step_up_recommended:
                risk_flags.append(f"Step-up auth recommended: {auth_result.step_up_method}")
            risk_flags.extend(auth_result.risk_factors)

        # 5. Run question planner agent
        question_result = await self._run_question_planner_safe(running_summary, risk_flags)
        if question_result is not None:
            suggested_questions = question_result.questions

        # 6. Update missing fields based on event text keywords
        self._update_missing_fields(event.text)

        # 7. Collect retrieved facts from retrieval result
        if self._retrieval_result is not None:
            retrieved_facts = [self._retrieval_result.retrieval_summary]
            self.evidence_collected = [
                f"txn:{len(self._retrieval_result.transactions)}",
                f"auth:{len(self._retrieval_result.auth_events)}",
            ]

        # 8. Run hypothesis agent — scores 4 categories using all context
        hypothesis_result = await self._run_hypothesis_safe(
            auth_result=auth_result, risk_flags=risk_flags
        )
        if hypothesis_result is not None:
            self.hypothesis_scores = dict(hypothesis_result.scores)

        # 9. Build safety guidance
        safety_guidance = self._build_safety_guidance()

        return CopilotSuggestion(
            call_id=event.call_id,
            timestamp_ms=event.timestamp_ms,
            suggested_questions=suggested_questions,
            risk_flags=risk_flags,
            retrieved_facts=retrieved_facts,
            running_summary=running_summary,
            safety_guidance=safety_guidance,
            hypothesis_scores=dict(self.hypothesis_scores),
            impersonation_risk=self.impersonation_risk,
        )

    # -- Private helper methods --

    def _persist_claims(self, claims: list[ClaimExtraction]) -> None:
        """Persist extracted claims as ClaimStatement evidence nodes.

        Each ClaimExtraction is written to the evidence store as a
        ClaimStatement (source_type=ALLEGATION) so the investigator can
        access structured claim data in the evidence graph.
        """
        if self.case_id is None:
            return
        ctx = AuthContext(agent_id="copilot", case_id=self.case_id, permissions={"write"})
        for claim in claims:
            node_id = f"claim-{uuid.uuid4().hex[:12]}"
            entities_str = (
                ", ".join(f"{k}={v}" for k, v in claim.entities.items()) if claim.entities else ""
            )
            classification = claim.claim_type.value
            text = claim.claim_description
            if entities_str:
                text = f"{text} [{entities_str}]"
            node = ClaimStatement(
                node_id=node_id,
                case_id=self.case_id,
                source_type=EvidenceSourceType.ALLEGATION,
                created_at=datetime.now(tz=timezone.utc),
                text=text,
                claim_type=claim.claim_type,
                classification=classification,
            )
            try:
                append_evidence_node(self.gateway, ctx, node)
            except RuntimeError:
                pass  # Duplicate or storage error — don't block the copilot

    def _build_claims_summary(self) -> str:
        """Build a running summary string from accumulated claims."""
        if not self.accumulated_claims:
            return "No claims extracted yet."
        parts = []
        for c in self.accumulated_claims:
            parts.append(f"{c.claim_type}: {c.claim_description}")
        return "Claims: " + "; ".join(parts) + "."

    def _format_claims_for_hypothesis(self) -> str:
        """Format all accumulated claims for the hypothesis agent input."""
        if not self.accumulated_claims:
            return "No claims extracted yet."
        lines = []
        for i, c in enumerate(self.accumulated_claims, 1):
            entities_str = (
                ", ".join(f"{k}={v}" for k, v in c.entities.items()) if c.entities else "none"
            )
            lines.append(
                f"{i}. [{c.claim_type}] {c.claim_description} "
                f"(confidence: {c.confidence:.1f}, entities: {entities_str})"
            )
        return "\n".join(lines)

    def _format_auth_for_hypothesis(self, auth_result: AuthAssessment | None) -> str:
        """Format auth assessment for the hypothesis agent input."""
        if auth_result is None:
            return "No auth assessment available."
        factors = ", ".join(auth_result.risk_factors) if auth_result.risk_factors else "none"
        return (
            f"Impersonation risk: {auth_result.impersonation_risk:.2f}. "
            f"Risk factors: {factors}. "
            f"Step-up: {auth_result.step_up_method}. "
            f"{auth_result.assessment_summary}"
        )

    def _format_evidence_for_hypothesis(self) -> str:
        """Format retrieved evidence for the hypothesis agent input."""
        if self._retrieval_result is None:
            return "No evidence retrieved."
        r = self._retrieval_result
        return (
            f"Transactions: {len(r.transactions)} found. "
            f"Auth events: {len(r.auth_events)} found. "
            f"Customer profile: {'available' if r.customer_profile else 'not available'}. "
            f"{r.retrieval_summary}"
        )

    def _format_conversation_for_hypothesis(self) -> str:
        """Format a brief conversation summary for the hypothesis agent."""
        if not self.transcript_history:
            return "No conversation yet."
        last_n = self.transcript_history[-5:]  # Last 5 turns for brevity
        lines = [f"{e.speaker.value}: {e.text[:100]}" for e in last_n]
        return f"{len(self.transcript_history)} turns total. Recent:\n" + "\n".join(lines)

    def _update_missing_fields(self, text: str) -> None:
        """Remove missing fields addressed by keywords in the transcript text."""
        text_lower = text.lower()
        resolved = []
        for field_name in self.missing_fields:
            keywords = _FIELD_KEYWORDS.get(field_name, [])
            if any(kw in text_lower for kw in keywords):
                resolved.append(field_name)
        for field_name in resolved:
            self.missing_fields.remove(field_name)

    def _build_safety_guidance(self) -> str:
        """Build safety guidance string based on current state."""
        parts = []
        if self.impersonation_risk >= 0.6:
            parts.append("HIGH impersonation risk — verify caller identity before proceeding.")
        if self.missing_fields:
            parts.append(f"Still need: {', '.join(self.missing_fields)}.")
        parts.append("Never ask for full PAN or CVV.")
        return " ".join(parts)

    async def _run_retrieval_safe(self, risk_flags: list[str]) -> RetrievalResult | None:
        """Run retrieval agent with error handling."""
        try:
            return await run_retrieval(
                case_id=self.case_id,  # type: ignore[arg-type]
                call_id=self.call_id,  # type: ignore[arg-type]
                gateway=self.gateway,
                model_provider=self.model_provider,
            )
        except Exception as exc:
            risk_flags.append(f"Retrieval failed: {exc}")
            return None

    async def _run_triage_safe(
        self,
        text: str,
        risk_flags: list[str],
        conversation_history: list[tuple[str, str]] | None = None,
    ) -> ClaimExtractionResult | None:
        """Run triage agent with error handling."""
        try:
            return await run_triage(
                transcript_text=text,
                model_provider=self.model_provider,
                conversation_history=conversation_history,
            )
        except Exception as exc:
            risk_flags.append(f"Triage failed: {exc}")
            return None

    async def _run_auth_safe(
        self,
        text: str,
        auth_events: list[dict],
        customer_profile: dict | None,
        risk_flags: list[str],
    ) -> AuthAssessment | None:
        """Run auth assessment agent with error handling."""
        try:
            return await run_auth_assessment(
                transcript_text=text,
                auth_events=auth_events,
                customer_profile=customer_profile,
                model_provider=self.model_provider,
            )
        except Exception as exc:
            risk_flags.append(f"Auth assessment failed: {exc}")
            return None

    async def _run_question_planner_safe(
        self, case_summary: str, risk_flags: list[str]
    ) -> QuestionPlan | None:
        """Run question planner agent with error handling."""
        try:
            return await run_question_planner(
                case_summary=case_summary,
                missing_fields=list(self.missing_fields),
                hypothesis_scores=dict(self.hypothesis_scores),
                model_provider=self.model_provider,
            )
        except Exception as exc:
            risk_flags.append(f"Question planner failed: {exc}")
            return None

    async def _run_hypothesis_safe(
        self,
        auth_result: AuthAssessment | None,
        risk_flags: list[str],
    ) -> HypothesisAssessment | None:
        """Run hypothesis agent with error handling.

        On failure, scores remain unchanged from the previous turn.
        """
        try:
            return await run_hypothesis(
                claims_summary=self._format_claims_for_hypothesis(),
                auth_summary=self._format_auth_for_hypothesis(auth_result),
                evidence_summary=self._format_evidence_for_hypothesis(),
                current_scores=dict(self.hypothesis_scores),
                conversation_summary=self._format_conversation_for_hypothesis(),
                model_provider=self.model_provider,
            )
        except Exception as exc:
            risk_flags.append(f"Hypothesis scoring failed: {exc}")
            return None
