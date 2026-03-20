"""Central copilot orchestrator with running state.

Processes transcript events, maintains running hypothesis scores and
impersonation risk, invokes specialist agents, and produces CopilotSuggestion
output for each turn. Triage, auth, and retrieval run concurrently via
asyncio.gather(); hypothesis, case advisor, and question planner run
sequentially after them. This is a plain Python class — not an Agents SDK
Agent — keeping the control flow explicit and auditable.
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone

from agents import ModelProvider

from agentic_fraud_servicing.copilot.auth_agent import AuthAssessment, run_auth_assessment
from agentic_fraud_servicing.copilot.case_advisor import CaseAdvisory, run_case_advisor
from agentic_fraud_servicing.copilot.hypothesis_agent import HypothesisAssessment, run_hypothesis
from agentic_fraud_servicing.copilot.question_planner import QuestionPlan, run_question_planner
from agentic_fraud_servicing.copilot.retrieval_agent import RetrievalResult, run_retrieval
from agentic_fraud_servicing.copilot.triage_agent import run_triage
from agentic_fraud_servicing.gateway.tool_gateway import AuthContext, ToolGateway
from agentic_fraud_servicing.gateway.tools.write_tools import append_evidence_node
from agentic_fraud_servicing.models.allegations import (
    AllegationExtraction,
    AllegationExtractionResult,
)
from agentic_fraud_servicing.models.case import CopilotSuggestion
from agentic_fraud_servicing.models.enums import EvidenceSourceType
from agentic_fraud_servicing.models.evidence import AllegationStatement
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
        accumulated_allegations: All allegations extracted across turns.
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
        self.accumulated_allegations: list[AllegationExtraction] = []
        self._retrieval_result: RetrievalResult | None = None

    async def process_event(self, event: TranscriptEvent) -> CopilotSuggestion:
        """Process a single transcript event and return copilot suggestions.

        Runs a parallelized pipeline: triage, auth, and retrieval run
        concurrently via asyncio.gather(), then hypothesis → case advisor →
        question planner run sequentially. Each specialist call is wrapped
        in try/except for graceful degradation.

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
        case_eligibility: list[dict] = []
        case_advisory_summary: str = ""

        # 2. Prepare inputs for the parallel group
        conversation_history = [(e.speaker.value, e.text) for e in self.transcript_history]
        # Use existing retrieval data for auth inputs (defaults on first call)
        prior_retrieval = self._retrieval_result
        auth_events = prior_retrieval.auth_events if prior_retrieval else []
        customer_profile = prior_retrieval.customer_profile if prior_retrieval else None

        # 3. Run triage, auth, and retrieval in parallel
        triage_result, auth_result, retrieval_result = await asyncio.gather(
            self._run_triage_safe(event.text, risk_flags, conversation_history),
            self._run_auth_safe(event.text, auth_events, customer_profile, risk_flags),
            self._run_retrieval_safe(risk_flags),
        )

        # 4. Process parallel results: retrieval
        if retrieval_result is not None:
            self._retrieval_result = retrieval_result

        # 5. Process parallel results: triage
        if triage_result is not None and triage_result.allegations:
            self.accumulated_allegations.extend(triage_result.allegations)
            self._persist_allegations(triage_result.allegations)

        # Build running summary from accumulated allegations
        running_summary = self._build_allegations_summary()

        # 6. Process parallel results: auth
        if auth_result is not None:
            self.impersonation_risk = auth_result.impersonation_risk
            if auth_result.step_up_recommended:
                risk_flags.append(f"Step-up auth recommended: {auth_result.step_up_method}")
            risk_flags.extend(auth_result.risk_factors)

        # 7. Update missing fields based on event text keywords
        self._update_missing_fields(event.text)

        # 8. Collect retrieved facts from retrieval result
        if self._retrieval_result is not None:
            retrieved_facts = [self._retrieval_result.retrieval_summary]
            self.evidence_collected = [
                f"txn:{len(self._retrieval_result.transactions)}",
                f"auth:{len(self._retrieval_result.auth_events)}",
            ]

        # 9. Run hypothesis agent — scores 4 categories using all context
        hypothesis_result = await self._run_hypothesis_safe(
            auth_result=auth_result, risk_flags=risk_flags
        )
        if hypothesis_result is not None:
            self.hypothesis_scores = dict(hypothesis_result.scores)

        # 10. Run case advisor — evaluate case opening eligibility
        case_advisory = await self._run_case_advisor_safe(risk_flags)
        if case_advisory is not None:
            case_eligibility = [a.model_dump(mode="json") for a in case_advisory.assessments]
            case_advisory_summary = case_advisory.summary
            # Prepend unmet criteria to missing_fields for the question planner
            unmet = []
            for assessment in case_advisory.assessments:
                for criterion in assessment.unmet_criteria:
                    unmet.append(f"[{assessment.case_type}] {criterion}")
            self.missing_fields = unmet + self.missing_fields

        # 11. Run question planner agent (after case advisor so it has unmet criteria)
        question_result = await self._run_question_planner_safe(running_summary, risk_flags)
        if question_result is not None:
            suggested_questions = question_result.questions

        # 12. Build safety guidance
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
            case_eligibility=case_eligibility,
            case_advisory_summary=case_advisory_summary,
        )

    # -- Private helper methods --

    def _persist_allegations(self, allegations: list[AllegationExtraction]) -> None:
        """Persist extracted allegations as AllegationStatement evidence nodes.

        Each AllegationExtraction is written to the evidence store as an
        AllegationStatement (source_type=ALLEGATION) so the investigator can
        access structured allegation data in the evidence graph.
        """
        if self.case_id is None:
            return
        ctx = AuthContext(agent_id="copilot", case_id=self.case_id, permissions={"write"})
        for allegation in allegations:
            node_id = f"allegation-{uuid.uuid4().hex[:12]}"
            entities_str = (
                ", ".join(f"{k}={v}" for k, v in allegation.entities.items())
                if allegation.entities
                else ""
            )
            classification = allegation.detail_type.value
            text = allegation.description
            if entities_str:
                text = f"{text} [{entities_str}]"
            node = AllegationStatement(
                node_id=node_id,
                case_id=self.case_id,
                source_type=EvidenceSourceType.ALLEGATION,
                created_at=datetime.now(tz=timezone.utc),
                text=text,
                detail_type=allegation.detail_type,
                classification=classification,
                entities=allegation.entities if allegation.entities else {},
            )
            try:
                append_evidence_node(self.gateway, ctx, node)
            except RuntimeError:
                pass  # Duplicate or storage error — don't block the copilot

    def _build_allegations_summary(self) -> str:
        """Build a running summary string from accumulated allegations."""
        if not self.accumulated_allegations:
            return "No allegations extracted yet."
        parts = []
        for a in self.accumulated_allegations:
            parts.append(f"{a.detail_type}: {a.description}")
        return "Allegations: " + "; ".join(parts) + "."

    def _format_allegations_for_hypothesis(self) -> str:
        """Format all accumulated allegations for the hypothesis agent input."""
        if not self.accumulated_allegations:
            return "No allegations extracted yet."
        lines = []
        for i, a in enumerate(self.accumulated_allegations, 1):
            entities_str = (
                ", ".join(f"{k}={v}" for k, v in a.entities.items()) if a.entities else "none"
            )
            lines.append(
                f"{i}. [{a.detail_type}] {a.description} "
                f"(confidence: {a.confidence:.1f}, entities: {entities_str})"
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
        """Format retrieved evidence as structured JSON for the hypothesis agent.

        Produces a summary line followed by detailed JSON containing actual
        evidence node data (amounts, auth types, device IDs, etc.) so the
        hypothesis agent's reasoning patterns can trigger on specific evidence.
        """
        if self._retrieval_result is None:
            return "No evidence retrieved."
        r = self._retrieval_result

        # Summary counts for quick reference
        summary = (
            f"Transactions: {len(r.transactions)} found. "
            f"Auth events: {len(r.auth_events)} found. "
            f"Customer profile: {'available' if r.customer_profile else 'not available'}."
        )

        # Build structured evidence data
        evidence_data: dict = {}

        if r.transactions:
            evidence_data["transactions"] = r.transactions

        if r.auth_events:
            evidence_data["auth_events"] = r.auth_events

        if r.customer_profile:
            evidence_data["customer_profile"] = r.customer_profile

        if not evidence_data:
            return f"{summary}\n{r.retrieval_summary}"

        return f"{summary}\n{json.dumps(evidence_data, indent=2, default=str)}"

    def _format_conversation_for_hypothesis(self) -> str:
        """Format a brief conversation summary for the hypothesis agent."""
        if not self.transcript_history:
            return "No conversation yet."
        last_n = self.transcript_history[-5:]  # Last 5 turns for brevity
        lines = [f"{e.speaker.value}: {e.text[:100]}" for e in last_n]
        return f"{len(self.transcript_history)} turns total. Recent:\n" + "\n".join(lines)

    def _update_missing_fields(self, text: str) -> None:
        """Remove missing fields addressed by transcript keywords or extracted entities.

        Two resolution strategies:
        1. Keyword heuristic: check if known keywords appear in the transcript text.
        2. Entity-based: if the triage agent extracted an entity whose key matches
           a missing field name (e.g., 'merchant_name', 'amount'), resolve it.
        """
        text_lower = text.lower()

        # Collect all entity keys from accumulated allegations
        entity_keys: set[str] = set()
        for allegation in self.accumulated_allegations:
            entity_keys.update(allegation.entities.keys())

        resolved = []
        for field_name in self.missing_fields:
            # Strategy 1: keyword match in transcript text
            keywords = _FIELD_KEYWORDS.get(field_name, [])
            if any(kw in text_lower for kw in keywords):
                resolved.append(field_name)
                continue
            # Strategy 2: entity extracted by triage matches the field name
            if field_name in entity_keys:
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
        """Run retrieval agent with error handling.

        Idempotent: returns cached result immediately if retrieval has already
        run, making it safe to include in every asyncio.gather() call.
        """
        if self._retrieval_result is not None:
            return self._retrieval_result
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
    ) -> AllegationExtractionResult | None:
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
                allegations_summary=self._format_allegations_for_hypothesis(),
                auth_summary=self._format_auth_for_hypothesis(auth_result),
                evidence_summary=self._format_evidence_for_hypothesis(),
                current_scores=dict(self.hypothesis_scores),
                conversation_summary=self._format_conversation_for_hypothesis(),
                model_provider=self.model_provider,
            )
        except Exception as exc:
            risk_flags.append(f"Hypothesis scoring failed: {exc}")
            return None

    async def _run_case_advisor_safe(self, risk_flags: list[str]) -> CaseAdvisory | None:
        """Run case advisor agent with error handling."""
        try:
            return await run_case_advisor(
                allegations_summary=self._format_allegations_for_hypothesis(),
                evidence_summary=self._format_evidence_for_hypothesis(),
                hypothesis_scores=dict(self.hypothesis_scores),
                conversation_summary=self._format_conversation_for_hypothesis(),
                model_provider=self.model_provider,
            )
        except Exception as exc:
            risk_flags.append(f"Case advisor failed: {exc}")
            return None
