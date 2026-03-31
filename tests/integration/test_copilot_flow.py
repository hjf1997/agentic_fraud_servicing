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
from agentic_fraud_servicing.copilot.case_advisor import CaseAdvisory, CaseTypeAssessment
from agentic_fraud_servicing.copilot.hypothesis_agent import HypothesisAssessment
from agentic_fraud_servicing.copilot.orchestrator import CopilotOrchestrator
from agentic_fraud_servicing.models.enums import SpeakerType
from agentic_fraud_servicing.copilot.retrieval_agent import RetrievalResult
from agentic_fraud_servicing.models.allegations import (
    AllegationExtraction,
    AllegationExtractionResult,
)
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

_CASE_ADVISORY = CaseAdvisory(
    assessments=[
        CaseTypeAssessment(
            case_type="fraud",
            eligibility="incomplete",
            met_criteria=["Transaction identified", "Card status active"],
            unmet_criteria=["Identity verification pending", "Date reported missing"],
            blockers=[],
            policy_citations=["Per fraud_case_checklist.md: 'Identity must be verified'"],
        ),
        CaseTypeAssessment(
            case_type="dispute",
            eligibility="eligible",
            met_criteria=["Merchant identified", "Amount confirmed"],
            unmet_criteria=[],
            blockers=[],
            policy_citations=[],
        ),
    ],
    general_warnings=["Escalation may be required for amounts over $10K"],
    questions=[
        "Do you recall making any online purchases around March 1st?",
        "Have you shared your card details with any family members?",
    ],
    rationale=[
        "Clarify if AMZN charge could be legitimate",
        "Check for authorized user activity",
    ],
    priority_field="auth_method",
    information_sufficient=False,
    summary="Fraud case incomplete pending identity verification. Dispute case eligible.",
)


# -- Patch targets for the 5 specialist run_* functions --
_PATCH_TRIAGE = "agentic_fraud_servicing.copilot.orchestrator.run_triage"
_PATCH_AUTH = "agentic_fraud_servicing.copilot.orchestrator.run_auth_assessment"
_PATCH_RETRIEVAL = "agentic_fraud_servicing.copilot.orchestrator.run_retrieval"
_PATCH_HYPOTHESIS = "agentic_fraud_servicing.copilot.orchestrator.run_hypothesis"
_PATCH_CASE_ADVISOR = "agentic_fraud_servicing.copilot.orchestrator.run_case_advisor"


@pytest.fixture()
def _mock_specialists():
    """Patch all 5 specialist run_* functions with canned results."""
    with (
        patch(_PATCH_TRIAGE, new_callable=AsyncMock, return_value=_TRIAGE_RESULT) as m_triage,
        patch(_PATCH_AUTH, new_callable=AsyncMock, return_value=_AUTH_ASSESSMENT) as m_auth,
        patch(
            _PATCH_RETRIEVAL, new_callable=AsyncMock, return_value=_RETRIEVAL_RESULT
        ) as m_retrieval,
        patch(
            _PATCH_HYPOTHESIS, new_callable=AsyncMock, return_value=_HYPOTHESIS_RESULT
        ) as m_hypothesis,
        patch(
            _PATCH_CASE_ADVISOR, new_callable=AsyncMock, return_value=_CASE_ADVISORY
        ) as m_case_advisor,
    ):
        yield {
            "triage": m_triage,
            "auth": m_auth,
            "retrieval": m_retrieval,
            "hypothesis": m_hypothesis,
            "case_advisor": m_case_advisor,
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
        """CARDMEMBER process_event should return a CopilotSuggestion instance."""
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider, assess_interval=1)

        # Use CARDMEMBER event (index 1) — CCP/SYSTEM returns None
        result = await orch.process_event(sample_transcript_events[1])
        assert isinstance(result, CopilotSuggestion)

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_suggestion_contains_questions_from_case_advisor(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """Suggested questions should come from the case advisor mock."""
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider, assess_interval=1)
        orch._turn_count = 3  # Case advisor only runs after turn 3

        # Use CARDMEMBER event (index 1) — only CARDMEMBER triggers agents
        result = await orch.process_event(sample_transcript_events[1])
        assert result.suggested_questions == _CASE_ADVISORY.questions

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_suggestion_contains_safety_guidance(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """Safety guidance must include the PAN/CVV warning."""
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider, assess_interval=1)

        # Use CARDMEMBER event (index 1) — CCP/SYSTEM returns None
        result = await orch.process_event(sample_transcript_events[1])
        assert "PAN" in result.safety_guidance or "CVV" in result.safety_guidance

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_hypothesis_scores_from_hypothesis_agent(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """Hypothesis scores in suggestion should come from the hypothesis agent mock."""
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider, assess_interval=1)

        # Use CARDMEMBER event (index 1) — only CARDMEMBER triggers agents
        result = await orch.process_event(sample_transcript_events[1])
        assert result.hypothesis_scores["THIRD_PARTY_FRAUD"] == pytest.approx(0.60)
        assert result.hypothesis_scores["FIRST_PARTY_FRAUD"] == pytest.approx(0.10)
        assert result.hypothesis_scores["SCAM"] == pytest.approx(0.15)
        assert result.hypothesis_scores["DISPUTE"] == pytest.approx(0.15)

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_suggestion_contains_case_eligibility(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """CopilotSuggestion.case_eligibility should be populated from case advisor mock."""
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider, assess_interval=1)
        orch._turn_count = 3  # Case advisor only runs after turn 3

        # Use CARDMEMBER event (index 1) — only CARDMEMBER triggers agents
        result = await orch.process_event(sample_transcript_events[1])
        assert len(result.case_eligibility) == 2
        assert result.case_eligibility[0]["case_type"] == "fraud"
        assert result.case_eligibility[0]["eligibility"] == "incomplete"
        assert result.case_eligibility[1]["case_type"] == "dispute"
        assert result.case_eligibility[1]["eligibility"] == "eligible"

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_case_advisory_summary_in_suggestion(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """CopilotSuggestion.case_advisory_summary should be populated from case advisor mock."""
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider, assess_interval=1)
        orch._turn_count = 3  # Case advisor only runs after turn 3

        # Use CARDMEMBER event (index 1) — only CARDMEMBER triggers agents
        result = await orch.process_event(sample_transcript_events[1])
        assert "Fraud case incomplete" in result.case_advisory_summary
        assert "Dispute case eligible" in result.case_advisory_summary


class TestRunningStateAccumulation:
    """Verify that orchestrator state accumulates across multiple events."""

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_case_id_and_call_id_set_from_first_event(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """case_id and call_id should be set after the first event."""
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider, assess_interval=1)

        await orch.process_event(sample_transcript_events[0])
        assert orch.call_id == "call-test-001"
        assert orch.case_id == "case-call-test-001"

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_transcript_history_accumulates(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """Transcript history should grow with each processed event."""
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider, assess_interval=1)

        for i, event in enumerate(sample_transcript_events):
            await orch.process_event(event)
            assert len(orch.transcript_history) == i + 1

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_hypothesis_scores_match_hypothesis_agent(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """Hypothesis scores should match the hypothesis agent output (not formulaic)."""
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider, assess_interval=1)

        # Use CARDMEMBER event (index 1) — only CARDMEMBER triggers agents
        await orch.process_event(sample_transcript_events[1])

        assert orch.hypothesis_scores["THIRD_PARTY_FRAUD"] == pytest.approx(0.60)
        assert orch.hypothesis_scores["FIRST_PARTY_FRAUD"] == pytest.approx(0.10)

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_suggestion_has_all_four_hypothesis_keys(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """CopilotSuggestion hypothesis_scores must contain all 4 category keys."""
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider, assess_interval=1)

        # Use CARDMEMBER event (index 1) — CCP/SYSTEM returns None
        result = await orch.process_event(sample_transcript_events[1])
        expected_keys = {"THIRD_PARTY_FRAUD", "FIRST_PARTY_FRAUD", "SCAM", "DISPUTE"}
        assert set(result.hypothesis_scores.keys()) == expected_keys

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_accumulated_allegations_grow(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """Accumulated allegations should grow with each CARDMEMBER event.

        Uses CARDMEMBER events (index 1) since triage is skipped for CCP/SYSTEM
        speakers via the speaker-based fast path.
        """
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider, assess_interval=1)

        # sample_transcript_events[1] is CARDMEMBER — triage runs
        await orch.process_event(sample_transcript_events[1])
        allegations_after_first = len(orch.accumulated_allegations)
        assert allegations_after_first == 2  # _TRIAGE_RESULT has 2 allegations

        # Process another CARDMEMBER-like event to see growth
        await orch.process_event(sample_transcript_events[1])
        assert len(orch.accumulated_allegations) == allegations_after_first + 2

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_impersonation_risk_set_from_auth(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """Impersonation risk should match the auth assessment mock value."""
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider, assess_interval=1)

        # Use CARDMEMBER event (index 1) — only CARDMEMBER triggers agents
        await orch.process_event(sample_transcript_events[1])
        assert orch.impersonation_risk == pytest.approx(0.15)


class TestPipelineOptimizations:
    """Verify pipeline optimizations from L1-16 work end-to-end."""

    async def test_system_event_skips_all_agents(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """SYSTEM events should skip all agents — no LLM calls."""
        with (
            patch(_PATCH_TRIAGE, new_callable=AsyncMock, return_value=_TRIAGE_RESULT) as m_triage,
            patch(_PATCH_AUTH, new_callable=AsyncMock, return_value=_AUTH_ASSESSMENT) as m_auth,
            patch(
                _PATCH_RETRIEVAL, new_callable=AsyncMock, return_value=_RETRIEVAL_RESULT
            ) as m_retrieval,
            patch(
                _PATCH_HYPOTHESIS, new_callable=AsyncMock, return_value=_HYPOTHESIS_RESULT
            ) as m_hypo,
            patch(
                _PATCH_CASE_ADVISOR, new_callable=AsyncMock, return_value=_CASE_ADVISORY
            ) as m_advisor,
        ):
            gateway = gateway_factory(tmp_path)
            orch = CopilotOrchestrator(gateway, mock_model_provider, assess_interval=1)

            # Process SYSTEM event (index 2)
            await orch.process_event(sample_transcript_events[2])

            # SYSTEM events skip ALL agents
            m_triage.assert_not_awaited()
            m_auth.assert_not_awaited()
            m_hypo.assert_not_awaited()
            m_retrieval.assert_not_awaited()
            m_advisor.assert_not_awaited()

    async def test_multiple_events_retrieval_cached(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """Retrieval should only be called once across multiple events (cached)."""
        with (
            patch(_PATCH_TRIAGE, new_callable=AsyncMock, return_value=_TRIAGE_RESULT),
            patch(_PATCH_AUTH, new_callable=AsyncMock, return_value=_AUTH_ASSESSMENT),
            patch(
                _PATCH_RETRIEVAL, new_callable=AsyncMock, return_value=_RETRIEVAL_RESULT
            ) as m_retrieval,
            patch(_PATCH_HYPOTHESIS, new_callable=AsyncMock, return_value=_HYPOTHESIS_RESULT),
            patch(_PATCH_CASE_ADVISOR, new_callable=AsyncMock, return_value=_CASE_ADVISORY),
        ):
            gateway = gateway_factory(tmp_path)
            orch = CopilotOrchestrator(gateway, mock_model_provider, assess_interval=1)

            # Process 3 events
            for event in sample_transcript_events[:3]:
                await orch.process_event(event)

            # Retrieval should be called exactly once (cached after first call)
            m_retrieval.assert_awaited_once()

    async def test_cardmember_full_pipeline(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """CARDMEMBER events should invoke all 5 agents."""
        with (
            patch(_PATCH_TRIAGE, new_callable=AsyncMock, return_value=_TRIAGE_RESULT) as m_triage,
            patch(_PATCH_AUTH, new_callable=AsyncMock, return_value=_AUTH_ASSESSMENT) as m_auth,
            patch(
                _PATCH_RETRIEVAL, new_callable=AsyncMock, return_value=_RETRIEVAL_RESULT
            ) as m_retrieval,
            patch(
                _PATCH_HYPOTHESIS, new_callable=AsyncMock, return_value=_HYPOTHESIS_RESULT
            ) as m_hypo,
            patch(
                _PATCH_CASE_ADVISOR, new_callable=AsyncMock, return_value=_CASE_ADVISORY
            ) as m_advisor,
        ):
            gateway = gateway_factory(tmp_path)
            orch = CopilotOrchestrator(gateway, mock_model_provider, assess_interval=1)
            # Advance past early-turn gates so all 5 agents run:
            # turn > 3 for case advisor, impersonation_risk >= 0.4 for auth
            orch._turn_count = 3
            orch.impersonation_risk = 0.5

            # Process CARDMEMBER event (index 1) — full pipeline
            await orch.process_event(sample_transcript_events[1])

            # All 5 agents should be called
            m_triage.assert_awaited_once()
            m_auth.assert_awaited_once()
            m_retrieval.assert_awaited_once()
            m_hypo.assert_awaited_once()
            m_advisor.assert_awaited_once()

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_output_format_unchanged(
        self, sample_transcript_events, gateway_factory, tmp_path, mock_model_provider
    ):
        """CopilotSuggestion must have all expected fields with correct types."""
        gateway = gateway_factory(tmp_path)
        orch = CopilotOrchestrator(gateway, mock_model_provider, assess_interval=1)

        result = await orch.process_event(sample_transcript_events[1])

        # Verify all expected fields exist with correct types
        assert isinstance(result, CopilotSuggestion)
        assert isinstance(result.call_id, str) and result.call_id != ""
        assert isinstance(result.timestamp_ms, int)
        assert isinstance(result.suggested_questions, list)
        assert isinstance(result.risk_flags, list)
        assert isinstance(result.retrieved_facts, list)
        assert isinstance(result.running_summary, str)
        assert isinstance(result.safety_guidance, str)
        assert isinstance(result.hypothesis_scores, dict)
        assert isinstance(result.impersonation_risk, float)
        assert isinstance(result.case_eligibility, list)
        assert isinstance(result.case_advisory_summary, str)

        # Verify hypothesis_scores has all 4 keys
        expected_keys = {"THIRD_PARTY_FRAUD", "FIRST_PARTY_FRAUD", "SCAM", "DISPUTE"}
        assert set(result.hypothesis_scores.keys()) == expected_keys
        for v in result.hypothesis_scores.values():
            assert isinstance(v, float)


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
        orch = CopilotOrchestrator(gateway, provider, assess_interval=1)

        # Find first CARDMEMBER event — CCP/SYSTEM returns None
        cm_event = next(e for e in events if e.speaker == SpeakerType.CARDMEMBER)
        result = await orch.process_event(cm_event)
        assert isinstance(result, CopilotSuggestion)
        assert result.call_id == "call-demo-001"
