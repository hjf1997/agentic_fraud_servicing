"""Central copilot orchestrator with running state.

Processes transcript events, maintains running hypothesis scores and
impersonation risk, invokes specialist agents, and produces CopilotSuggestion
output for each turn. Triage, auth, and retrieval run concurrently via
asyncio.gather(); hypothesis and case advisor run in parallel after them.
The case advisor handles both eligibility assessment and question planning.
This is a plain Python class — not an Agents SDK Agent — keeping the control
flow explicit and auditable.
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
from agentic_fraud_servicing.copilot.langfuse_tracing import (
    extract_http_error,
    get_langfuse,
    is_firewall_block,
    tag_agent_error,
    tag_firewall_block,
)
from agentic_fraud_servicing.copilot.retrieval_agent import RetrievalResult, run_retrieval
from agentic_fraud_servicing.copilot.triage_agent import run_triage
from agentic_fraud_servicing.gateway.tool_gateway import AuthContext, ToolGateway
from agentic_fraud_servicing.gateway.tools.write_tools import append_evidence_node
from agentic_fraud_servicing.ingestion.firewall_redactor import FirewallRedactor
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
        self._last_hypothesis: HypothesisAssessment | None = None
        self._recent_suggestions: list[list[str]] = []
        self._turn_count: int = 0
        self._cm_turn_count: int = 0
        self._last_assessed_idx: int = 0
        self._firewall_redactor = FirewallRedactor()

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
        2. Parallel: hypothesis + case advisor (after turn 3)

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

        # -- LangFuse observability: open a trace for this assessment turn --
        lf = get_langfuse()
        _lf_obs_ctx = None
        _lf_prop_ctx = None
        if lf is not None:
            try:
                from langfuse import propagate_attributes

                _lf_obs_ctx = lf.start_as_current_observation(
                    as_type="agent", name="copilot_turn",
                )
                _lf_obs_ctx.__enter__()
                _lf_prop_ctx = propagate_attributes(
                    session_id=self.case_id or "unknown",
                    tags=["copilot"],
                    metadata={
                        "cm_turn": str(self._cm_turn_count),
                        "assess_interval": str(self.assess_interval),
                    },
                )
                _lf_prop_ctx.__enter__()
            except Exception:
                _lf_obs_ctx = None
                _lf_prop_ctx = None

        try:
            return await self._run_pipeline(event, risk_flags)
        finally:
            if _lf_prop_ctx is not None:
                try:
                    _lf_prop_ctx.__exit__(None, None, None)
                except Exception:
                    pass
            if _lf_obs_ctx is not None:
                try:
                    _lf_obs_ctx.__exit__(None, None, None)
                except Exception:
                    pass
            if lf is not None:
                try:
                    lf.flush()
                except Exception:
                    pass

    async def _run_pipeline(
        self, event: TranscriptEvent, risk_flags: list[str]
    ) -> CopilotSuggestion:
        """Execute the full agent pipeline for an assessment turn.

        Separated from process_event so the LangFuse trace context wraps the
        entire pipeline cleanly via try/finally.
        """
        lf = get_langfuse()

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
        conversation_window = [
            (e.speaker.value, self._firewall_redactor.redact_text(e.text))
            for e in window
        ]
        allegation_summary = (
            self._format_allegations_for_hypothesis()
            if self.accumulated_allegations
            else None
        )

        prior_retrieval = self._retrieval_result
        auth_events = prior_retrieval.auth_events if prior_retrieval else []
        customer_profile = prior_retrieval.customer_profile if prior_retrieval else None

        # -- Phase 1: triage + auth (conditional) + retrieval --
        _lf_p1 = None
        if lf is not None:
            try:
                _lf_p1 = lf.start_as_current_observation(
                    as_type="chain", name="phase1_parallel",
                )
                _lf_p1.__enter__()
            except Exception:
                _lf_p1 = None

        auth_result = None
        if self._should_run_auth(event):
            triage_result, auth_result, retrieval_result = await asyncio.gather(
                self._run_triage_safe(
                    risk_flags, conversation_window, new_turn_offset, allegation_summary
                ),
                self._run_auth_safe(
                    self._firewall_redactor.redact_text(event.text),
                    auth_events, customer_profile, risk_flags,
                    conversation_window,
                ),
                self._run_retrieval_safe(risk_flags),
            )
        else:
            triage_result, retrieval_result = await asyncio.gather(
                self._run_triage_safe(
                    risk_flags, conversation_window, new_turn_offset, allegation_summary
                ),
                self._run_retrieval_safe(risk_flags),
            )

        if _lf_p1 is not None:
            try:
                _lf_p1.__exit__(None, None, None)
            except Exception:
                pass

        # 3b. Process parallel results: retrieval
        if retrieval_result is not None:
            self._retrieval_result = retrieval_result

        # 3c. Process parallel results: triage
        _invalidate_retrieval = False
        if triage_result is not None and triage_result.allegations:
            self.accumulated_allegations.extend(triage_result.allegations)
            self._persist_allegations(triage_result.allegations)
            _invalidate_retrieval = True  # Evidence store changed; refresh next turn

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

        # 6-7. Run hypothesis agent and case advisor (merged: eligibility + questions).
        # On turns > 3, run both in PARALLEL — case advisor uses previous turn's
        # hypothesis_scores (acceptable since scores shift incrementally).
        _lf_p2 = None
        if lf is not None:
            try:
                _lf_p2 = lf.start_as_current_observation(
                    as_type="chain", name="phase2_hypothesis_advisor",
                )
                _lf_p2.__enter__()
            except Exception:
                _lf_p2 = None

        case_advisory = None
        if self._turn_count > 3:
            hypothesis_result, case_advisory = await asyncio.gather(
                self._run_hypothesis_safe(
                    auth_result=auth_result, risk_flags=risk_flags
                ),
                self._run_case_advisor_safe(risk_flags, conversation_window),
            )
        else:
            hypothesis_result = await self._run_hypothesis_safe(
                auth_result=auth_result, risk_flags=risk_flags
            )

        if _lf_p2 is not None:
            try:
                _lf_p2.__exit__(None, None, None)
            except Exception:
                pass
        if hypothesis_result is not None:
            self.hypothesis_scores = dict(hypothesis_result.scores)
            self._last_hypothesis = hypothesis_result

        # 8. Track recent suggestions for dedup in subsequent turns
        suggested_questions = case_advisory.questions if case_advisory else []
        self._recent_suggestions.append(suggested_questions)
        if len(self._recent_suggestions) > 3:
            self._recent_suggestions = self._recent_suggestions[-3:]

        # 9. Invalidate retrieval cache if evidence changed this turn, so the
        # next assessment re-fetches fresh data. Done after hypothesis/advisor
        # have consumed this turn's retrieval result.
        if _invalidate_retrieval:
            self._retrieval_result = None

        # 10. Build and return the suggestion
        return self._build_suggestion(
            event,
            risk_flags=risk_flags,
            case_advisory=case_advisory,
        )

    # -- Private helper methods --

    def _build_suggestion(
        self,
        event: TranscriptEvent,
        *,
        risk_flags: list[str] | None = None,
        case_advisory: CaseAdvisory | None = None,
    ) -> CopilotSuggestion:
        """Assemble a CopilotSuggestion from current state.

        Called after a CARDMEMBER pipeline run (with fresh results) or
        directly for SYSTEM/CCP events (returning previous state).
        """
        case_eligibility: list[dict] = []
        case_advisory_summary = ""
        suggested_questions: list[str] = []
        information_sufficient = False
        if case_advisory is not None:
            case_eligibility = [a.model_dump(mode="json") for a in case_advisory.assessments]
            case_advisory_summary = case_advisory.summary
            suggested_questions = case_advisory.questions
            information_sufficient = case_advisory.information_sufficient

        retrieved_facts: list[str] = []
        if self._retrieval_result is not None:
            retrieved_facts = [self._retrieval_result.retrieval_summary]

        return CopilotSuggestion(
            call_id=event.call_id,
            timestamp_ms=event.timestamp_ms,
            suggested_questions=suggested_questions,
            risk_flags=risk_flags or [],
            retrieved_facts=retrieved_facts,
            running_summary=self._build_allegations_summary(),
            safety_guidance=self._build_safety_guidance(),
            hypothesis_scores=dict(self.hypothesis_scores),
            impersonation_risk=self.impersonation_risk,
            case_eligibility=case_eligibility,
            case_advisory_summary=case_advisory_summary,
            information_sufficient=information_sufficient,
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
        """Format a conversation summary for the hypothesis agent.

        Uses the assessment-based window (context + new turns) to ensure no
        turns are missed between assessments, matching the triage window.
        """
        if not self.transcript_history:
            return "No conversation yet."
        context_trailing = 4
        context_start = max(0, self._last_assessed_idx - context_trailing)
        window = self.transcript_history[context_start:]
        lines = [
            f"{e.speaker.value}: {self._firewall_redactor.redact_text(e.text)}"
            for e in window
        ]
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

    @staticmethod
    def _format_error(agent_name: str, exc: BaseException) -> str:
        """Format an agent error with HTTP status code for terminal output."""
        status_code, error_body = extract_http_error(exc)
        if status_code is not None:
            return f"{agent_name} failed (HTTP {status_code}): {error_body[:200]}"
        return f"{agent_name} failed: {exc}"

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
        if self._cm_turn_count <= 3:
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
            tag_agent_error("retrieval", exc)
            if is_firewall_block(exc):
                tag_firewall_block("retrieval", str(exc))
                risk_flags.append("FIREWALL BLOCKED: retrieval agent prompt rejected by policy")
            else:
                risk_flags.append(self._format_error("Retrieval", exc))
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
            tag_agent_error("triage", exc)
            if is_firewall_block(exc):
                tag_firewall_block("triage", str(exc))
                risk_flags.append("FIREWALL BLOCKED: triage agent prompt rejected by policy")
            else:
                risk_flags.append(self._format_error("Triage", exc))
            return None

    async def _run_auth_safe(
        self,
        text: str,
        auth_events: list[dict],
        customer_profile: dict | None,
        risk_flags: list[str],
        conversation_history: list[tuple[str, str]] | None = None,
    ) -> AuthAssessment | None:
        """Run auth assessment agent with error handling."""
        t0 = time.perf_counter()
        try:
            result = await run_auth_assessment(
                transcript_text=text,
                auth_events=auth_events,
                customer_profile=customer_profile,
                model_provider=self.model_provider,
                conversation_history=conversation_history,
            )
            self._log_agent_trace("auth", "run", (time.perf_counter() - t0) * 1000)
            return result
        except Exception as exc:
            self._log_agent_trace("auth", "run", (time.perf_counter() - t0) * 1000, status="error")
            tag_agent_error("auth", exc)
            if is_firewall_block(exc):
                tag_firewall_block("auth", str(exc))
                risk_flags.append("FIREWALL BLOCKED: auth agent prompt rejected by policy")
            else:
                risk_flags.append(self._format_error("Auth", exc))
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
                previous_reasoning=self._last_hypothesis,
            )
            self._log_agent_trace("hypothesis", "run", (time.perf_counter() - t0) * 1000)
            return result
        except Exception as exc:
            self._log_agent_trace(
                "hypothesis", "run", (time.perf_counter() - t0) * 1000, status="error"
            )
            tag_agent_error("hypothesis", exc)
            if is_firewall_block(exc):
                tag_firewall_block("hypothesis", str(exc))
                risk_flags.append("FIREWALL BLOCKED: hypothesis agent prompt rejected by policy")
            else:
                risk_flags.append(self._format_error("Hypothesis", exc))
            return None

    async def _run_case_advisor_safe(
        self,
        risk_flags: list[str],
        conversation_window: list[tuple[str, str]],
    ) -> CaseAdvisory | None:
        """Run case advisor agent with error handling.

        The merged case advisor handles both eligibility assessment and
        question planning. It receives the assessment-based conversation
        window, missing fields, and recent questions for deduplication.
        """
        recent_questions = [q for qs in self._recent_suggestions for q in qs]
        t0 = time.perf_counter()
        try:
            result = await run_case_advisor(
                allegations_summary=self._format_allegations_for_hypothesis(),
                evidence_summary=self._format_evidence_for_hypothesis(),
                hypothesis_scores=dict(self.hypothesis_scores),
                conversation_window=conversation_window,
                model_provider=self.model_provider,
                missing_fields=self.missing_fields if self.missing_fields else None,
                recent_questions=recent_questions if recent_questions else None,
            )
            self._log_agent_trace("case_advisor", "run", (time.perf_counter() - t0) * 1000)
            return result
        except Exception as exc:
            self._log_agent_trace(
                "case_advisor", "run", (time.perf_counter() - t0) * 1000, status="error"
            )
            tag_agent_error("case_advisor", exc)
            if is_firewall_block(exc):
                tag_firewall_block("case_advisor", str(exc))
                risk_flags.append("FIREWALL BLOCKED: case_advisor agent prompt rejected by policy")
            else:
                risk_flags.append(self._format_error("Case advisor", exc))
            return None
