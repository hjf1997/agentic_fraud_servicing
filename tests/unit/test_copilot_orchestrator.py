"""Tests for the copilot orchestrator module.

All specialist agent run_* functions are mocked to return canned results.
Tests verify state management, CopilotSuggestion assembly, graceful
degradation on specialist failure, and running state accumulation.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_fraud_servicing.copilot.orchestrator import (
    _INITIAL_MISSING_FIELDS,
    CopilotOrchestrator,
)
from agentic_fraud_servicing.models.case import CopilotSuggestion
from agentic_fraud_servicing.models.enums import AllegationType, SpeakerType
from agentic_fraud_servicing.models.transcript import TranscriptEvent

# -- Fixtures --


def _make_event(
    text: str = "I didn't make this purchase",
    call_id: str = "call-001",
    event_id: str = "evt-001",
    timestamp_ms: int = 1000,
    speaker: SpeakerType = SpeakerType.CARDMEMBER,
) -> TranscriptEvent:
    """Create a TranscriptEvent for testing."""
    return TranscriptEvent(
        call_id=call_id,
        event_id=event_id,
        timestamp_ms=timestamp_ms,
        speaker=speaker,
        text=text,
    )


def _mock_triage_result(
    allegation_type=AllegationType.FRAUD,
    confidence=0.8,
    claims=None,
    key_phrases=None,
):
    """Create a mock TriageResult."""
    result = MagicMock()
    result.allegation_type = allegation_type
    result.confidence = confidence
    result.claims = claims or ["unauthorized purchase"]
    result.category_shift_detected = False
    result.key_phrases = key_phrases or ["didn't make this purchase"]
    return result


def _mock_auth_result(
    impersonation_risk=0.2,
    step_up_recommended=False,
    step_up_method="NONE",
    risk_factors=None,
):
    """Create a mock AuthAssessment."""
    result = MagicMock()
    result.impersonation_risk = impersonation_risk
    result.step_up_recommended = step_up_recommended
    result.step_up_method = step_up_method
    result.risk_factors = risk_factors or []
    result.assessment_summary = "Low risk caller"
    return result


def _mock_question_result(questions=None):
    """Create a mock QuestionPlan."""
    result = MagicMock()
    result.questions = questions or ["When did this transaction occur?"]
    result.rationale = ["Need transaction date"]
    result.priority_field = "transaction_date"
    result.confidence = 0.7
    return result


def _mock_retrieval_result(transactions=None, auth_events=None, customer_profile=None):
    """Create a mock RetrievalResult."""
    result = MagicMock()
    result.transactions = transactions or [{"amount": 100.0}]
    result.auth_events = auth_events or [{"type": "chip"}]
    result.customer_profile = customer_profile
    result.retrieval_summary = "Found 1 transaction and 1 auth event"
    result.data_gaps = []
    return result


def _make_orchestrator() -> CopilotOrchestrator:
    """Create a CopilotOrchestrator with mock gateway and provider."""
    gateway = MagicMock()
    model_provider = MagicMock()
    return CopilotOrchestrator(gateway=gateway, model_provider=model_provider)


# Patch paths for the 4 specialist run_* functions
_TRIAGE_PATCH = "agentic_fraud_servicing.copilot.orchestrator.run_triage"
_AUTH_PATCH = "agentic_fraud_servicing.copilot.orchestrator.run_auth_assessment"
_QUESTION_PATCH = "agentic_fraud_servicing.copilot.orchestrator.run_question_planner"
_RETRIEVAL_PATCH = "agentic_fraud_servicing.copilot.orchestrator.run_retrieval"


# -- Test Classes --


class TestCopilotOrchestratorInit:
    """Tests for orchestrator initialization."""

    def test_initial_state(self):
        """Constructor sets correct defaults."""
        orch = _make_orchestrator()
        assert orch.case_id is None
        assert orch.call_id is None
        assert orch.hypothesis_scores == {"FRAUD": 0.0, "DISPUTE": 0.0, "SCAM": 0.0}
        assert orch.impersonation_risk == 0.0
        assert orch.missing_fields == list(_INITIAL_MISSING_FIELDS)
        assert orch.evidence_collected == []
        assert orch.transcript_history == []

    def test_stores_gateway_and_provider(self):
        """Gateway and model_provider are stored as attributes."""
        gateway = MagicMock()
        provider = MagicMock()
        orch = CopilotOrchestrator(gateway=gateway, model_provider=provider)
        assert orch.gateway is gateway
        assert orch.model_provider is provider


class TestProcessEvent:
    """Tests for the process_event method."""

    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_returns_copilot_suggestion(
        self, mock_triage, mock_auth, mock_question, mock_retrieval
    ):
        """process_event returns a CopilotSuggestion instance."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert isinstance(result, CopilotSuggestion)
        assert result.call_id == "call-001"
        assert result.timestamp_ms == 1000

    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_sets_case_id_from_first_event(
        self, mock_triage, mock_auth, mock_question, mock_retrieval
    ):
        """case_id and call_id are set from the first event."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()

        orch = _make_orchestrator()
        await orch.process_event(_make_event(call_id="call-42"))

        assert orch.case_id == "case-call-42"
        assert orch.call_id == "call-42"

    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_accumulates_transcript_history(
        self, mock_triage, mock_auth, mock_question, mock_retrieval
    ):
        """transcript_history grows with each event."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()

        orch = _make_orchestrator()
        await orch.process_event(_make_event(event_id="evt-1"))
        await orch.process_event(_make_event(event_id="evt-2"))

        assert len(orch.transcript_history) == 2
        assert orch.transcript_history[0].event_id == "evt-1"
        assert orch.transcript_history[1].event_id == "evt-2"

    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_updates_hypothesis_scores(
        self, mock_triage, mock_auth, mock_question, mock_retrieval
    ):
        """hypothesis_scores update from triage results."""
        mock_triage.return_value = _mock_triage_result(
            allegation_type=AllegationType.FRAUD, confidence=0.9
        )
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()

        orch = _make_orchestrator()
        await orch.process_event(_make_event())

        # FRAUD score should increase (0.0 * 0.4 + 0.9 * 0.6 = 0.54)
        assert orch.hypothesis_scores["FRAUD"] == pytest.approx(0.54, abs=0.01)
        # Other scores should stay at 0 (0.0 * 0.7 = 0.0)
        assert orch.hypothesis_scores["DISPUTE"] == 0.0
        assert orch.hypothesis_scores["SCAM"] == 0.0

    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_updates_impersonation_risk(
        self, mock_triage, mock_auth, mock_question, mock_retrieval
    ):
        """impersonation_risk updates from auth assessment."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result(impersonation_risk=0.75)
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert orch.impersonation_risk == 0.75
        assert result.impersonation_risk == 0.75

    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_includes_suggested_questions(
        self, mock_triage, mock_auth, mock_question, mock_retrieval
    ):
        """CopilotSuggestion includes questions from the planner."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result(
            questions=["What was the transaction amount?", "Where was the charge?"]
        )
        mock_retrieval.return_value = _mock_retrieval_result()

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert len(result.suggested_questions) == 2
        assert "What was the transaction amount?" in result.suggested_questions


class TestMissingFieldsUpdate:
    """Tests for missing fields keyword-based removal."""

    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_removes_field_on_keyword_match(
        self, mock_triage, mock_auth, mock_question, mock_retrieval
    ):
        """Missing fields are removed when keywords appear in text."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()

        orch = _make_orchestrator()
        # Text mentions "amount" and "merchant" — should resolve those fields
        await orch.process_event(_make_event(text="The amount was $500 at the merchant store"))

        assert "amount" not in orch.missing_fields
        assert "merchant_name" not in orch.missing_fields
        # transaction_date and auth_method should still be missing
        assert "transaction_date" in orch.missing_fields
        assert "auth_method" in orch.missing_fields


class TestGracefulDegradation:
    """Tests for specialist failure handling."""

    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_triage_failure_continues(
        self, mock_triage, mock_auth, mock_question, mock_retrieval
    ):
        """If triage fails, orchestrator continues and records error."""
        mock_triage.side_effect = RuntimeError("LLM timeout")
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert isinstance(result, CopilotSuggestion)
        assert any("Triage failed" in flag for flag in result.risk_flags)
        # Questions should still be present from the question planner
        assert len(result.suggested_questions) > 0

    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_all_specialists_fail(
        self, mock_triage, mock_auth, mock_question, mock_retrieval
    ):
        """If all specialists fail, result still returned with error flags."""
        mock_triage.side_effect = RuntimeError("triage error")
        mock_auth.side_effect = RuntimeError("auth error")
        mock_question.side_effect = RuntimeError("question error")
        mock_retrieval.side_effect = RuntimeError("retrieval error")

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert isinstance(result, CopilotSuggestion)
        assert any("Triage failed" in f for f in result.risk_flags)
        assert any("Auth assessment failed" in f for f in result.risk_flags)
        assert any("Question planner failed" in f for f in result.risk_flags)
        assert any("Retrieval failed" in f for f in result.risk_flags)


class TestStepUpAuth:
    """Tests for step-up auth risk flag propagation."""

    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_step_up_auth_in_risk_flags(
        self, mock_triage, mock_auth, mock_question, mock_retrieval
    ):
        """Step-up recommendation appears in risk_flags."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result(
            impersonation_risk=0.85,
            step_up_recommended=True,
            step_up_method="SMS_OTP",
        )
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert any("Step-up auth recommended: SMS_OTP" in f for f in result.risk_flags)


class TestSafetyGuidance:
    """Tests for safety guidance generation."""

    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_safety_guidance_includes_pan_warning(
        self, mock_triage, mock_auth, mock_question, mock_retrieval
    ):
        """Safety guidance always includes PAN/CVV warning."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert "Never ask for full PAN or CVV" in result.safety_guidance
