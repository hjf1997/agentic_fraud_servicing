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
import time
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
from agentic_fraud_servicing.models.enums import EvidenceSourceType, SpeakerType
from agentic_fraud_servicing.models.evidence import AllegationStatement
from agentic_fraud_servicing.models.transcript import TranscriptEvent

# Standard fields to gather during a dispute call
_INITIAL_MISSING_FIELDS = [
    "transaction_date",
    "merchant_name",
    "amount",
    "auth_method",
]


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

    def __init__(
        self,
        gateway: ToolGateway,
        model_provider: ModelProvider,
        assess_interval: int = 5,
    ) -> None:
        self.gateway = gateway
        self.model_provider = model_provider
        self.assess_interval: int = max(1, assess_interval)
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
        self._recent_suggestions: list[list[str]] = []
        self._turn_count: int = 0
        self._cm_turn_count: int = 0
        self._last_assessed_idx: int = 0

    async def process_event(
        self, event: TranscriptEvent, *, is_last: bool = False
    ) -> CopilotSuggestion | None:
        """Process a single transcript event and return copilot suggestions.

        Only CARDMEMBER events on assessment turns trigger the agent pipeline.
        CCP/SYSTEM events and non-assessment CARDMEMBER turns return None.
        When ``is_last=True``, the pipeline always runs regardless of the
        interval — this ensures a final assessment on the last conversation turn.

        For assessment turns the pipeline is:
        1. Parallel: triage (claim extraction) + auth (conditional) + retrieval
        2. Sequential: hypothesis → case advisor (after turn 3) → question planner

        Auth is conditional after the first 3 turns: skipped when
        impersonation risk is low (< 0.4).

        Args:
            event: The transcript event to process.
            is_last: If True, force assessment regardless of interval.

        Returns:
            CopilotSuggestion on assessment turns, None otherwise.
        """
        # 1. Append event, increment turn count, set case_id/call_id on first event
        self._turn_count += 1
        self.transcript_history.append(event)
        if self.case_id is None:
            self.case_id = f"case-{event.call_id}"
            self.call_id = event.call_id

        # 2. Non-CARDMEMBER events: record in history, no assessment needed
        if event.speaker != SpeakerType.CARDMEMBER:
            return None

        # 3. CARDMEMBER event: check assess_interval.
        # Always run on the first CM turn and the last turn. After that, run
        # every assess_interval CM turns (e.g. interval=5 → CM turns 1, 6, 11, ...).
        self._cm_turn_count += 1
        if (
            not is_last
            and self._cm_turn_count > 1
            and (self._cm_turn_count - 1) % self.assess_interval != 0
        ):
            return None

        # 4. CARDMEMBER event: run full agent pipeline
        risk_flags: list[str] = []

        # 3a. Parallel group: triage + auth (conditional) + retrieval
        # Build window: [CONTEXT] turns (trailing from previous assessment) +
        # [NEW] turns (all turns since last assessment). This ensures no CM
        # turns are missed even when assess_interval > 1.
        context_trailing = 4  # number of previously-assessed turns for context
        history = self.transcript_history
        new_start = self._last_assessed_idx
        context_start = max(0, new_start - context_trailing)
        window = history[context_start:]
        new_turn_offset = new_start - context_start
        conversation_window = [(e.speaker.value, e.text) for e in window]
        allegation_summary = (
            self._format_allegations_for_hypothesis()
            if self.accumulated_allegations
            else None
        )

        prior_retrieval = self._retrieval_result
        auth_events = prior_retrieval.auth_events if prior_retrieval else []
        customer_profile = prior_retrieval.customer_profile if prior_retrieval else None

        auth_result = None
        if self._should_run_auth(event):
            triage_result, auth_result, retrieval_result = await asyncio.gather(
                self._run_triage_safe(
                    risk_flags, conversation_window, new_turn_offset, allegation_summary
                ),
                self._run_auth_safe(event.text, auth_events, customer_profile, risk_flags),
                self._run_retrieval_safe(risk_flags),
            )
        else:
            triage_result, retrieval_result = await asyncio.gather(
                self._run_triage_safe(
                    risk_flags, conversation_window, new_turn_offset, allegation_summary
                ),
                self._run_retrieval_safe(risk_flags),
            )

        # 3b. Process parallel results: retrieval
        if retrieval_result is not None:
            self._retrieval_result = retrieval_result

        # 3c. Process parallel results: triage
        if triage_result is not None and triage_result.allegations:
            self.accumulated_allegations.extend(triage_result.allegations)
            self._persist_allegations(triage_result.allegations)

        # 3c'. Advance the assessment index so the next assessment knows
        # which turns have already been processed by triage.
        self._last_assessed_idx = len(self.transcript_history)

        # 3d. Process parallel results: auth
        if auth_result is not None:
            self.impersonation_risk = auth_result.impersonation_risk
            if auth_result.step_up_recommended:
                risk_flags.append(f"Step-up auth recommended: {auth_result.step_up_method}")
            risk_flags.extend(auth_result.risk_factors)

        # 4. Update missing fields from triage-extracted entities
        self._update_missing_fields()

        # 5. Collect retrieved facts from retrieval result
        if self._retrieval_result is not None:
            self.evidence_collected = [
                f"txn:{len(self._retrieval_result.transactions)}",
                f"auth:{len(self._retrieval_result.auth_events)}",
            ]

        # 6-7. Run hypothesis agent and case advisor.
        # On turns > 3, run both in PARALLEL — case advisor uses previous turn's
        # hypothesis_scores (acceptable since scores shift incrementally).
        case_advisory = None
        if self._turn_count > 3:
            hypothesis_result, case_advisory = await asyncio.gather(
                self._run_hypothesis_safe(
                    auth_result=auth_result, risk_flags=risk_flags
                ),
                self._run_case_advisor_safe(risk_flags),
            )
        else:
            hypothesis_result = await self._run_hypothesis_safe(
                auth_result=auth_result, risk_flags=risk_flags
            )
        if hypothesis_result is not None:
            self.hypothesis_scores = dict(hypothesis_result.scores)
        unmet_criteria: list[str] = []
        if case_advisory is not None:
            # Collect unmet criteria to pass to question planner (not accumulated
            # in self.missing_fields — they change each turn and would bloat the list)
            for assessment in case_advisory.assessments:
                for criterion in assessment.unmet_criteria:
                    unmet_criteria.append(f"[{assessment.case_type}] {criterion}")

        # 8. Run question planner agent (after case advisor so it has unmet criteria)
        running_summary = self._build_allegations_summary()
        planner_missing = unmet_criteria + self.missing_fields
        question_result = await self._run_question_planner_safe(
            running_summary, risk_flags, extra_missing=planner_missing
        )

        # 9. Track recent suggestions for dedup in subsequent turns
        suggested_questions = question_result.questions if question_result else []
        self._recent_suggestions.append(suggested_questions)
        if len(self._recent_suggestions) > 3:
            self._recent_suggestions = self._recent_suggestions[-3:]

        # 10. Build and return the suggestion
        return self._build_suggestion(
            event,
            risk_flags=risk_flags,
            suggested_questions=suggested_questions,
            case_advisory=case_advisory,
        )

    # -- Private helper methods --

    def _build_suggestion(
        self,
        event: TranscriptEvent,
        *,
        risk_flags: list[str] | None = None,
        suggested_questions: list[str] | None = None,
        case_advisory: CaseAdvisory | None = None,
    ) -> CopilotSuggestion:
        """Assemble a CopilotSuggestion from current state.

        Called after a CARDMEMBER pipeline run (with fresh results) or
        directly for SYSTEM/CCP events (returning previous state).
        """
        case_eligibility: list[dict] = []
        case_advisory_summary = ""
        if case_advisory is not None:
            case_eligibility = [a.model_dump(mode="json") for a in case_advisory.assessments]
            case_advisory_summary = case_advisory.summary

        retrieved_facts: list[str] = []
        if self._retrieval_result is not None:
            retrieved_facts = [self._retrieval_result.retrieval_summary]

        return CopilotSuggestion(
            call_id=event.call_id,
            timestamp_ms=event.timestamp_ms,
            suggested_questions=suggested_questions or [],
            risk_flags=risk_flags or [],
            retrieved_facts=retrieved_facts,
            running_summary=self._build_allegations_summary(),
            safety_guidance=self._build_safety_guidance(),
            hypothesis_scores=dict(self.hypothesis_scores),
            impersonation_risk=self.impersonation_risk,
            case_eligibility=case_eligibility,
            case_advisory_summary=case_advisory_summary,
        )

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
        disputed_count = sum(1 for t in r.transactions if t.get("is_disputed"))
        summary = (
            f"Transactions: {len(r.transactions)} found ({disputed_count} disputed). "
            f"Auth events: {len(r.auth_events)} found. "
            f"Customer profile: {'available' if r.customer_profile else 'not available'}."
        )

        # Build structured evidence data
        evidence_data: dict = {}

        if r.transactions:
            disputed = [t for t in r.transactions if t.get("is_disputed")]
            undisputed = [t for t in r.transactions if not t.get("is_disputed")]
            if disputed:
                evidence_data["disputed_transactions"] = disputed
            if undisputed:
                evidence_data["account_transactions"] = undisputed

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

    def _update_missing_fields(self) -> None:
        """Remove missing fields resolved by triage-extracted entities.

        Iterates accumulated allegations and collects entity keys. If an
        entity key matches a missing field name (e.g., 'merchant_name',
        'amount'), the field is considered resolved and removed.
        """
        entity_keys: set[str] = set()
        for allegation in self.accumulated_allegations:
            entity_keys.update(allegation.entities.keys())

        self.missing_fields = [f for f in self.missing_fields if f not in entity_keys]

    def _build_safety_guidance(self) -> str:
        """Build safety guidance string based on current state."""
        parts = []
        if self.impersonation_risk >= 0.6:
            parts.append("HIGH impersonation risk — verify caller identity before proceeding.")
        if self.missing_fields:
            parts.append(f"Still need: {', '.join(self.missing_fields)}.")
        parts.append("Never ask for full PAN or CVV.")
        return " ".join(parts)

    def _log_agent_trace(
        self, agent_name: str, action: str, duration_ms: float, status: str = "success"
    ) -> None:
        """Persist a per-agent trace record to the trace store for audit."""
        try:
            self.gateway.trace_store.log_invocation(
                trace_id=str(uuid.uuid4()),
                case_id=self.case_id or "unknown",
                agent_id=f"copilot.{agent_name}",
                action=action,
                input_data=json.dumps({"turn": self._turn_count}),
                output_data=json.dumps({"status": status}),
                duration_ms=duration_ms,
                timestamp=datetime.now(timezone.utc),
                status=status,
            )
        except Exception:
            pass  # Never let trace logging break the pipeline

    def _should_run_auth(self, event: TranscriptEvent) -> bool:
        """Determine whether to invoke the auth agent on this CARDMEMBER turn.

        Auth runs unconditionally on the first 3 turns (identity not yet
        established) or when impersonation risk is elevated (>= 0.4). Once
        identity is established with low risk, auth is skipped to save latency.

        Note: This method is only called for CARDMEMBER events — non-CARDMEMBER
        events return early in process_event() before reaching this check.
        """
        if self._turn_count <= 3:
            return True
        if self.impersonation_risk >= 0.4:
            return True
        return False

    async def _run_retrieval_safe(self, risk_flags: list[str]) -> RetrievalResult | None:
        """Run retrieval agent with error handling.

        Idempotent: returns cached result immediately if retrieval has already
        run, making it safe to include in every asyncio.gather() call.
        """
        if self._retrieval_result is not None:
            return self._retrieval_result
        t0 = time.perf_counter()
        try:
            result = await run_retrieval(
                case_id=self.case_id,  # type: ignore[arg-type]
                call_id=self.call_id,  # type: ignore[arg-type]
                gateway=self.gateway,
                model_provider=self.model_provider,
            )
            self._log_agent_trace("retrieval", "run", (time.perf_counter() - t0) * 1000)
            return result
        except Exception as exc:
            self._log_agent_trace(
                "retrieval", "run", (time.perf_counter() - t0) * 1000, status="error"
            )
            risk_flags.append(f"Retrieval failed: {exc}")
            return None

    async def _run_triage_safe(
        self,
        risk_flags: list[str],
        conversation_history: list[tuple[str, str]],
        new_turn_offset: int = 0,
        allegation_summary: str | None = None,
    ) -> AllegationExtractionResult | None:
        """Run triage agent with error handling."""
        t0 = time.perf_counter()
        try:
            result = await run_triage(
                conversation_history=conversation_history,
                model_provider=self.model_provider,
                new_turn_offset=new_turn_offset,
                allegation_summary=allegation_summary,
            )
            self._log_agent_trace("triage", "run", (time.perf_counter() - t0) * 1000)
            return result
        except Exception as exc:
            self._log_agent_trace(
                "triage", "run", (time.perf_counter() - t0) * 1000, status="error"
            )
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
        t0 = time.perf_counter()
        try:
            result = await run_auth_assessment(
                transcript_text=text,
                auth_events=auth_events,
                customer_profile=customer_profile,
                model_provider=self.model_provider,
            )
            self._log_agent_trace("auth", "run", (time.perf_counter() - t0) * 1000)
            return result
        except Exception as exc:
            self._log_agent_trace("auth", "run", (time.perf_counter() - t0) * 1000, status="error")
            risk_flags.append(f"Auth assessment failed: {exc}")
            return None

    async def _run_question_planner_safe(
        self,
        case_summary: str,
        risk_flags: list[str],
        *,
        extra_missing: list[str] | None = None,
    ) -> QuestionPlan | None:
        """Run question planner agent with error handling.

        Passes recent conversation turns and previously suggested questions
        so the planner avoids repeating questions already asked or suggested.
        ``extra_missing`` (typically unmet case-advisor criteria) is passed
        as additional missing fields for the current turn only — it is NOT
        accumulated into ``self.missing_fields``.
        """
        # Last 5 conversation turns for context
        recent_turns = [(e.speaker.value, e.text) for e in self.transcript_history[-5:]]
        # Flatten recent suggestions into a single dedup list
        recent_questions = [q for qs in self._recent_suggestions for q in qs]
        missing = extra_missing if extra_missing else list(self.missing_fields)
        t0 = time.perf_counter()
        try:
            result = await run_question_planner(
                case_summary=case_summary,
                missing_fields=missing,
                hypothesis_scores=dict(self.hypothesis_scores),
                model_provider=self.model_provider,
                recent_turns=recent_turns if recent_turns else None,
                recent_questions=recent_questions if recent_questions else None,
            )
            self._log_agent_trace("question_planner", "run", (time.perf_counter() - t0) * 1000)
            return result
        except Exception as exc:
            self._log_agent_trace(
                "question_planner", "run", (time.perf_counter() - t0) * 1000, status="error"
            )
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
        t0 = time.perf_counter()
        try:
            result = await run_hypothesis(
                allegations_summary=self._format_allegations_for_hypothesis(),
                auth_summary=self._format_auth_for_hypothesis(auth_result),
                evidence_summary=self._format_evidence_for_hypothesis(),
                current_scores=dict(self.hypothesis_scores),
                conversation_summary=self._format_conversation_for_hypothesis(),
                model_provider=self.model_provider,
            )
            self._log_agent_trace("hypothesis", "run", (time.perf_counter() - t0) * 1000)
            return result
        except Exception as exc:
            self._log_agent_trace(
                "hypothesis", "run", (time.perf_counter() - t0) * 1000, status="error"
            )
            risk_flags.append(f"Hypothesis scoring failed: {exc}")
            return None

    async def _run_case_advisor_safe(self, risk_flags: list[str]) -> CaseAdvisory | None:
        """Run case advisor agent with error handling."""
        t0 = time.perf_counter()
        try:
            result = await run_case_advisor(
                allegations_summary=self._format_allegations_for_hypothesis(),
                evidence_summary=self._format_evidence_for_hypothesis(),
                hypothesis_scores=dict(self.hypothesis_scores),
                conversation_summary=self._format_conversation_for_hypothesis(),
                model_provider=self.model_provider,
            )
            self._log_agent_trace("case_advisor", "run", (time.perf_counter() - t0) * 1000)
            return result
        except Exception as exc:
            self._log_agent_trace(
                "case_advisor", "run", (time.perf_counter() - t0) * 1000, status="error"
            )
            risk_flags.append(f"Case advisor failed: {exc}")
            return None
