"""Integration tests for the full copilot pipeline.

Tests the end-to-end flow: transcript ingestion with PII redaction, copilot
orchestrator processing, specialist agent invocation (mocked), and
CopilotSuggestion output. Uses real SQLite stores via gateway_factory and
mocked LLM calls via patched run_* functions.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from agentic_fraud_servicing.models.allegations import (
    AllegationExtraction,
    AllegationExtractionResult,
)

from agentic_fraud_servicing.copilot.auth_agent import AuthAssessment
from agentic_fraud_servicing.copilot.hypothesis_agent import HypothesisAssessment
from agentic_fraud_servicing.copilot.orchestrator import CopilotOrchestrator
from agentic_fraud_servicing.copilot.question_planner import QuestionPlan
from agentic_fraud_servicing.copilot.retrieval_agent import RetrievalResult
from agentic_fraud_servicing.models.case import CopilotSuggestion
from agentic_fraud_servicing.models.enums import AllegationDetailType
from agentic_fraud_servicing.ui.helpers import load_transcript_file

# Path to the sample transcript fixture
_SAMPLE_TRANSCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "sample_transcript.json"

# Canned specialist results for mocking
_TRIAGE_RESULT = AllegationExtractionResult(
    allegations=[
        AllegationExtraction(
            detail_type=AllegationDetailType.UNRECOGNIZED_TRANSACTION,
            description="CM disputes charge at AMZN Marketplace",
            entities={"merchant_name": "AMZN Marketplace"},
            confidence=0.85,
        ),
        AllegationExtraction(
            detail_type=AllegationDetailType.MERCHANT_FRAUD,
            description="CM says TechGadgets is unknown merchant",
            entities={"merchant_name": "TechGadgets"},
            confidence=0.75,
        ),
    ]
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

_HYPOTHESIS_RESULT = HypothesisAssessment(
    scores={
        "THIRD_PARTY_FRAUD": 0.60,
        "FIRST_PARTY_FRAUD": 0.10,
        "SCAM": 0.15,
        "DISPUTE": 0.15,
    },
    reasoning={
        "THIRD_PARTY_FRAUD": "CM claims unauthorized charge, no contradictions yet.",
        "FIRST_PARTY_FRAUD": "Low — no signs of misrepresentation so far.",
        "SCAM": "No social engineering indicators detected.",
        "DISPUTE": "Could be a merchant issue but CM frames as unauthorized.",
    },
    contradictions=[],
    assessment_summary="Likely third-party fraud based on initial claim analysis.",
)


# -- Patch targets for the 5 specialist run_* functions --
_PATCH_TRIAGE = "agentic_fraud_servicing.copilot.orchestrator.run_triage"
_PATCH_AUTH = "agentic_fraud_servicing.copilot.orchestrator.run_auth_assessment"
_PATCH_QUESTION = "agentic_fraud_servicing.copilot.orchestrator.run_question_planner"
_PATCH_RETRIEVAL = "agentic_fraud_servicing.copilot.orchestrator.run_retrieval"
_PATCH_HYPOTHESIS = "agentic_fraud_servicing.copilot.orchestrator.run_hypothesis"


@pytest.fixture()
def _mock_specialists():
    """Patch all 5 specialist run_* functions with canned results."""
    with (
        patch(_PATCH_TRIAGE, new_callable=AsyncMock, return_value=_TRIAGE_RESULT) as m_triage,
        patch(_PATCH_AUTH, new_callable=AsyncMock, return_value=_AUTH_ASSESSMENT) as m_auth,
        patch(_PATCH_QUESTION, new_callable=AsyncMock, return_value=_QUESTION_PLAN) as m_question,
        patch(
            _PATCH_RETRIEVAL, new_callable=AsyncMock, return_value=_RETRIEVAL_RESULT
        ) as m_retrieval,
        patch(
            _PATCH_HYPOTHESIS, new_callable=AsyncMock, return_value=_HYPOTHESIS_RESULT
        ) as m_hypothesis,
    ):
        yield {
            "triage": m_triage,
            "auth": m_auth,
            "question": m_question,
            "retrieval": m_retrieval,
            "hypothesis": m_hypothesis,
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

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_hypothesis_scores_from_hypothesis_agent(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """Hypothesis scores in suggestion should come from the hypothesis agent mock."""
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider)

        result = await orch.process_event(sample_transcript_events[0])
        assert result.hypothesis_scores["THIRD_PARTY_FRAUD"] == pytest.approx(0.60)
        assert result.hypothesis_scores["FIRST_PARTY_FRAUD"] == pytest.approx(0.10)
        assert result.hypothesis_scores["SCAM"] == pytest.approx(0.15)
        assert result.hypothesis_scores["DISPUTE"] == pytest.approx(0.15)


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
    async def test_hypothesis_scores_match_hypothesis_agent(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """Hypothesis scores should match the hypothesis agent output (not formulaic)."""
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider)

        await orch.process_event(sample_transcript_events[0])

        assert orch.hypothesis_scores["THIRD_PARTY_FRAUD"] == pytest.approx(0.60)
        assert orch.hypothesis_scores["FIRST_PARTY_FRAUD"] == pytest.approx(0.10)

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_suggestion_has_all_four_hypothesis_keys(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """CopilotSuggestion hypothesis_scores must contain all 4 category keys."""
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider)

        result = await orch.process_event(sample_transcript_events[0])
        expected_keys = {"THIRD_PARTY_FRAUD", "FIRST_PARTY_FRAUD", "SCAM", "DISPUTE"}
        assert set(result.hypothesis_scores.keys()) == expected_keys

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_accumulated_allegations_grow(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """Accumulated allegations should grow with each processed event."""
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider)

        await orch.process_event(sample_transcript_events[0])
        allegations_after_first = len(orch.accumulated_allegations)
        assert allegations_after_first == 2  # _TRIAGE_RESULT has 2 allegations

        await orch.process_event(sample_transcript_events[1])
        assert len(orch.accumulated_allegations) == allegations_after_first + 2

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
