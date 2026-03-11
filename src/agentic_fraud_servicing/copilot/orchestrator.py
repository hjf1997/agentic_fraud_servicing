"""Central copilot orchestrator with running state.

Processes transcript events, maintains running hypothesis scores and
impersonation risk, invokes specialist agents (triage, auth, question
planner, retrieval), and produces CopilotSuggestion output for each turn.
This is a plain Python class — not an Agents SDK Agent — keeping the
control flow explicit and auditable.
"""

from agents import ModelProvider

from agentic_fraud_servicing.copilot.auth_agent import AuthAssessment, run_auth_assessment
from agentic_fraud_servicing.copilot.question_planner import QuestionPlan, run_question_planner
from agentic_fraud_servicing.copilot.retrieval_agent import RetrievalResult, run_retrieval
from agentic_fraud_servicing.copilot.triage_agent import run_triage
from agentic_fraud_servicing.gateway.tool_gateway import ToolGateway
from agentic_fraud_servicing.models.case import CopilotSuggestion
from agentic_fraud_servicing.models.claims import ClaimExtractionResult
from agentic_fraud_servicing.models.enums import AllegationType
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
        self._retrieval_result: RetrievalResult | None = None

    async def process_event(self, event: TranscriptEvent) -> CopilotSuggestion:
        """Process a single transcript event and return copilot suggestions.

        Appends the event to history, invokes specialist agents (triage,
        auth, question planner), updates running state, and assembles the
        CopilotSuggestion. Each specialist call is wrapped in try/except
        so failures are graceful — the orchestrator continues with remaining
        specialists and records the error in risk_flags.

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
        running_summary = ""

        # 2. Run retrieval once at the start to pre-fetch data
        if self._retrieval_result is None and self.case_id is not None:
            self._retrieval_result = await self._run_retrieval_safe(risk_flags)

        # Extract data from retrieval for downstream agents
        retrieval = self._retrieval_result
        auth_events = retrieval.auth_events if retrieval else []
        customer_profile = retrieval.customer_profile if retrieval else None

        # 3. Run triage agent with full conversation history
        conversation_history = [(e.speaker.value, e.text) for e in self.transcript_history]
        triage_result = await self._run_triage_safe(event.text, risk_flags, conversation_history)
        if triage_result is not None:
            self._update_hypothesis_scores(triage_result)
            # Build summary from extracted claims
            if triage_result.claims:
                parts = []
                for c in triage_result.claims:
                    if isinstance(c, str):
                        parts.append(c)
                    else:
                        parts.append(c.claim_description)
                claims_str = "; ".join(parts)
            else:
                claims_str = "none"
            running_summary = f"Claims: {claims_str}."

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

        # 8. Build safety guidance
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

    # Map 4-category hypothesis keys back to 3-value AllegationType for triage
    _HYPOTHESIS_TO_ALLEGATION: dict[str, AllegationType] = {
        "THIRD_PARTY_FRAUD": AllegationType.FRAUD,
        "FIRST_PARTY_FRAUD": AllegationType.FRAUD,
        "SCAM": AllegationType.SCAM,
        "DISPUTE": AllegationType.DISPUTE,
    }

    # Map triage's 3-value AllegationType to 4-category hypothesis key
    _ALLEGATION_TO_HYPOTHESIS: dict[AllegationType, str] = {
        AllegationType.FRAUD: "THIRD_PARTY_FRAUD",
        AllegationType.SCAM: "SCAM",
        AllegationType.DISPUTE: "DISPUTE",
    }

    def _current_allegation_type(self) -> AllegationType | None:
        """Derive the current leading allegation type from hypothesis scores.

        Maps from the 4-category hypothesis keys back to the 3-value
        AllegationType that the triage agent expects as previous_type.
        Both THIRD_PARTY_FRAUD and FIRST_PARTY_FRAUD map to AllegationType.FRAUD.
        """
        if all(v == 0.0 for v in self.hypothesis_scores.values()):
            return None
        top_key = max(self.hypothesis_scores, key=self.hypothesis_scores.get)  # type: ignore[arg-type]
        return self._HYPOTHESIS_TO_ALLEGATION.get(top_key)

    def _update_hypothesis_scores(self, triage_result: ClaimExtractionResult) -> None:
        """Update hypothesis scores from a triage result.

        Maps triage's AllegationType (3 values) to the 4-category hypothesis
        keys via _ALLEGATION_TO_HYPOTHESIS. If the triage detects a category
        shift (story inconsistency), adds a small boost to FIRST_PARTY_FRAUD
        since shifts often indicate the CM is misrepresenting.

        Note: This method will be replaced by the hypothesis agent in task 14.4.
        Guard against ClaimExtractionResult (no allegation_type) during transition.
        """
        allegation_type = getattr(triage_result, "allegation_type", None)
        if allegation_type is not None:
            # Map 3-value AllegationType to 4-category hypothesis key
            detected = self._ALLEGATION_TO_HYPOTHESIS.get(allegation_type)
            if detected is None:
                return
            confidence = getattr(triage_result, "confidence", 0.5)
            for key in self.hypothesis_scores:
                if key == detected:
                    # Weighted moving average toward the new confidence
                    self.hypothesis_scores[key] = (
                        self.hypothesis_scores[key] * 0.4 + confidence * 0.6
                    )
                else:
                    # Decay other scores
                    self.hypothesis_scores[key] *= 0.7

        # Category shifts suggest the CM may be misrepresenting — boost
        # first-party fraud hypothesis as a simple heuristic
        if getattr(triage_result, "category_shift_detected", False):
            self.hypothesis_scores["FIRST_PARTY_FRAUD"] = min(
                1.0, self.hypothesis_scores["FIRST_PARTY_FRAUD"] + 0.15
            )

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
