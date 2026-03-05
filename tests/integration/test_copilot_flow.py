"""Integration tests for the full copilot pipeline.

Tests the end-to-end flow: transcript ingestion with PII redaction, copilot
orchestrator processing, specialist agent invocation (mocked), and
CopilotSuggestion output. Uses real SQLite stores via gateway_factory and
mocked LLM calls via patched run_* functions.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agentic_fraud_servicing.copilot.auth_agent import AuthAssessment
from agentic_fraud_servicing.copilot.orchestrator import CopilotOrchestrator
from agentic_fraud_servicing.copilot.question_planner import QuestionPlan
from agentic_fraud_servicing.copilot.retrieval_agent import RetrievalResult
from agentic_fraud_servicing.copilot.triage_agent import TriageResult
from agentic_fraud_servicing.models.case import CopilotSuggestion
from agentic_fraud_servicing.models.enums import AllegationType
from agentic_fraud_servicing.ui.helpers import load_transcript_file

# Path to the sample transcript fixture
_SAMPLE_TRANSCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "sample_transcript.json"

# Canned specialist results for mocking
_TRIAGE_RESULT = TriageResult(
    claims=["Unauthorized charge at AMZN Marketplace", "Unknown merchant TechGadgets"],
    allegation_type=AllegationType.FRAUD,
    confidence=0.85,
    category_shift_detected=False,
    key_phrases=["did not make", "never heard of"],
)

_AUTH_ASSESSMENT = AuthAssessment(
    impersonation_risk=0.15,
    risk_factors=["Device recognized"],
    step_up_recommended=False,
    step_up_method="NONE",
    assessment_summary="Low impersonation risk. Device and identity verified.",
)

_QUESTION_PLAN = QuestionPlan(
    questions=[
        "Do you recall making any online purchases around March 1st?",
        "Have you shared your card details with any family members?",
    ],
    rationale=[
        "Clarify if AMZN charge could be legitimate",
        "Check for authorized user activity",
    ],
    priority_field="auth_method",
    confidence=0.80,
)

_RETRIEVAL_RESULT = RetrievalResult(
    transactions=[{"node_type": "TRANSACTION", "amount": 487.50, "merchant": "AMZN Marketplace"}],
    auth_events=[{"node_type": "AUTH_EVENT", "result": "passed"}],
    customer_profile={"name": "John Smith", "account_status": "active"},
    retrieval_summary="Found 1 transaction, 1 auth event, customer profile active.",
    data_gaps=["No delivery proof available"],
)


# -- Patch targets for the 4 specialist run_* functions --
_PATCH_TRIAGE = "agentic_fraud_servicing.copilot.orchestrator.run_triage"
_PATCH_AUTH = "agentic_fraud_servicing.copilot.orchestrator.run_auth_assessment"
_PATCH_QUESTION = "agentic_fraud_servicing.copilot.orchestrator.run_question_planner"
_PATCH_RETRIEVAL = "agentic_fraud_servicing.copilot.orchestrator.run_retrieval"


@pytest.fixture()
def _mock_specialists():
    """Patch all 4 specialist run_* functions with canned results."""
    with (
        patch(_PATCH_TRIAGE, new_callable=AsyncMock, return_value=_TRIAGE_RESULT) as m_triage,
        patch(_PATCH_AUTH, new_callable=AsyncMock, return_value=_AUTH_ASSESSMENT) as m_auth,
        patch(_PATCH_QUESTION, new_callable=AsyncMock, return_value=_QUESTION_PLAN) as m_question,
        patch(
            _PATCH_RETRIEVAL, new_callable=AsyncMock, return_value=_RETRIEVAL_RESULT
        ) as m_retrieval,
    ):
        yield {
            "triage": m_triage,
            "auth": m_auth,
            "question": m_question,
            "retrieval": m_retrieval,
        }


# -- Tests --


class TestTranscriptIngestion:
    """Verify transcript loading with PII redaction."""

    def test_load_sample_transcript_redacts_pan(self):
        """The sample transcript PAN (371449635398431) must be redacted."""
        events = load_transcript_file(_SAMPLE_TRANSCRIPT)
        assert len(events) == 8

        # Event 2 (index 1) contains the AMEX PAN in the original JSON
        cardmember_event = events[1]
        assert "[PAN_REDACTED]" in cardmember_event.text
        assert "371449635398431" not in cardmember_event.text

    def test_all_events_have_correct_call_id(self):
        """All events should share the same call_id from the sample transcript."""
        events = load_transcript_file(_SAMPLE_TRANSCRIPT)
        for event in events:
            assert event.call_id == "call-demo-001"


class TestCopilotSuggestionOutput:
    """Verify CopilotSuggestion structure and content from the orchestrator."""

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_process_event_returns_copilot_suggestion(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """Each process_event call should return a CopilotSuggestion instance."""
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider)

        result = await orch.process_event(sample_transcript_events[0])
        assert isinstance(result, CopilotSuggestion)

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_suggestion_contains_questions_from_planner(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """Suggested questions should come from the question planner mock."""
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider)

        result = await orch.process_event(sample_transcript_events[0])
        assert result.suggested_questions == _QUESTION_PLAN.questions

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_suggestion_contains_safety_guidance(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """Safety guidance must include the PAN/CVV warning."""
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider)

        result = await orch.process_event(sample_transcript_events[0])
        assert "PAN" in result.safety_guidance or "CVV" in result.safety_guidance


class TestRunningStateAccumulation:
    """Verify that orchestrator state accumulates across multiple events."""

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_case_id_and_call_id_set_from_first_event(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """case_id and call_id should be set after the first event."""
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider)

        await orch.process_event(sample_transcript_events[0])
        assert orch.call_id == "call-test-001"
        assert orch.case_id == "case-call-test-001"

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_transcript_history_accumulates(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """Transcript history should grow with each processed event."""
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider)

        for i, event in enumerate(sample_transcript_events):
            await orch.process_event(event)
            assert len(orch.transcript_history) == i + 1

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_hypothesis_scores_update(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """Hypothesis scores should be non-zero after triage runs."""
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider)

        # Process two events so scores accumulate
        await orch.process_event(sample_transcript_events[0])
        await orch.process_event(sample_transcript_events[1])

        # The triage mock returns FRAUD with confidence 0.85
        assert orch.hypothesis_scores["FRAUD"] > 0.0

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_impersonation_risk_set_from_auth(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """Impersonation risk should match the auth assessment mock value."""
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider)

        await orch.process_event(sample_transcript_events[0])
        assert orch.impersonation_risk == pytest.approx(0.15)


class TestLiveTest:
    """Live integration test requiring real Bedrock credentials."""

    @pytest.mark.live
    async def test_copilot_flow_live(self, gateway_factory, tmp_path):
        """End-to-end copilot test with real LLM provider.

        Requires AWS Bedrock credentials configured in .env
        (LLM_PROVIDER=bedrock, AWS_PROFILE, AWS_REGION, AWS_BEDROCK_MODEL_ID).
        Skipped by default — run with: pytest -m live

        Loads the sample transcript, creates a real provider, and processes
        the first event through the copilot orchestrator.
        """
        from agentic_fraud_servicing.ui.helpers import create_provider

        try:
            provider = create_provider()
        except Exception:
            pytest.skip("LLM provider not configured — skipping live test")

        events = load_transcript_file(_SAMPLE_TRANSCRIPT)
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, provider)

        result = await orch.process_event(events[0])
        assert isinstance(result, CopilotSuggestion)
        assert result.call_id == "call-demo-001"
