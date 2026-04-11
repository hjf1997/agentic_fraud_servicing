"""Tests for the copilot orchestrator module.

All specialist agent run_* functions are mocked to return canned results.
Tests verify state management, CopilotSuggestion assembly, graceful
degradation on specialist failure, and running state accumulation.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_fraud_servicing.copilot.case_advisor import CaseAdvisory, CaseTypeAssessment
from agentic_fraud_servicing.copilot.orchestrator import CopilotOrchestrator
from agentic_fraud_servicing.models.allegations import (
    AllegationExtraction,
    AllegationExtractionResult,
)
from agentic_fraud_servicing.models.case import CopilotSuggestion
from agentic_fraud_servicing.models.enums import AllegationDetailType, SpeakerType
from agentic_fraud_servicing.models.transcript import TranscriptEvent


@pytest.fixture(autouse=True)
def _disable_langfuse():
    """Disable LangFuse in all orchestrator tests."""
    with patch("agentic_fraud_servicing.copilot.orchestrator.get_langfuse", return_value=None):
        yield


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


def _mock_triage_result(allegations=None):
    """Create a mock AllegationExtractionResult with allegations."""
    if allegations is None:
        allegations = [
            AllegationExtraction(
                detail_type=AllegationDetailType.UNRECOGNIZED_TRANSACTION,
                description="CM says they did not make this purchase",
                entities={"merchant_name": "TechVault"},
                confidence=0.9,
            )
        ]
    return AllegationExtractionResult(allegations=allegations)


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


def _mock_retrieval_result(auth_events=None, customer_profile=None, transaction_summary=None):
    """Create a mock RetrievalResult."""
    result = MagicMock()
    result.transaction_summary = transaction_summary or (
        "== Disputed Transactions (1 total, $100.00) ==\n2024-01-01:\n- $100.00 at TestMerchant"
    )
    result.auth_events = auth_events or [{"type": "chip"}]
    result.customer_profile = customer_profile
    result.retrieval_summary = "Found 1 transaction and 1 auth event"
    result.data_gaps = []
    return result


def _mock_hypothesis_result(scores=None):
    """Create a mock HypothesisAssessment."""
    result = MagicMock()
    result.scores = scores or {
        "THIRD_PARTY_FRAUD": 0.5,
        "FIRST_PARTY_FRAUD": 0.2,
        "SCAM": 0.1,
        "DISPUTE": 0.2,
    }
    result.reasoning = {
        "THIRD_PARTY_FRAUD": "Likely unauthorized",
        "FIRST_PARTY_FRAUD": "Low indicators",
        "SCAM": "No scam pattern",
        "DISPUTE": "Low indicators",
    }
    result.contradictions = []
    result.assessment_summary = "Likely third-party fraud"
    return result


def _mock_specialist_outputs():
    """Create mock SpecialistAssessment outputs for the specialist panel."""
    from agentic_fraud_servicing.copilot.hypothesis_specialists import SpecialistAssessment

    return {
        "DISPUTE": SpecialistAssessment(
            category="DISPUTE",
            likelihood=0.2,
            reasoning="No merchant issue identified.",
            eligibility="blocked",
            evidence_gaps=[],
            policy_citations=["Per dispute_case_checklist.md: 'Cannot open if fraud case active'"],
        ),
        "SCAM": SpecialistAssessment(
            category="SCAM",
            likelihood=0.1,
            reasoning="No scam pattern.",
            eligibility="eligible",
        ),
        "THIRD_PARTY_FRAUD": SpecialistAssessment(
            category="THIRD_PARTY_FRAUD",
            likelihood=0.5,
            reasoning="Unfamiliar device used.",
            eligibility="eligible",
            evidence_gaps=["Identity verification pending", "Authorization method unclear"],
            policy_citations=["Per fraud_case_checklist.md: 'Identity must be verified'"],
        ),
    }


def _mock_case_advisory(
    questions=None,
    information_sufficient=False,
):
    """Create a mock CaseAdvisory with realistic assessments and questions."""
    if questions is None:
        questions = ["When did this transaction occur?"]
    return CaseAdvisory(
        assessments=[
            CaseTypeAssessment(
                case_type="fraud",
                eligibility="eligible",
                met_criteria=["Transaction identified", "Card status verified"],
                unmet_criteria=["Identity verification pending", "Authorization method unclear"],
                blockers=[],
                policy_citations=["Per fraud_case_checklist.md: 'Identity must be verified'"],
            ),
            CaseTypeAssessment(
                case_type="dispute",
                eligibility="blocked",
                met_criteria=["Transaction identified"],
                unmet_criteria=[],
                blockers=["Active fraud case exists for this transaction"],
                policy_citations=[
                    "Per dispute_case_checklist.md: 'Cannot open if fraud case active'"
                ],
            ),
        ],
        general_warnings=["High-value transaction — supervisor review may be needed"],
        questions=questions,
        rationale=["Need transaction date"],
        priority_field="transaction_date",
        information_sufficient=information_sufficient,
        summary="Fraud case eligible. Dispute is blocked.",
    )


def _make_orchestrator() -> CopilotOrchestrator:
    """Create a CopilotOrchestrator with mock gateway and provider."""
    gateway = MagicMock()
    model_provider = MagicMock()
    return CopilotOrchestrator(gateway=gateway, model_provider=model_provider, assess_interval=1)


# Patch paths for the specialist run_* functions
_TRIAGE_PATCH = "agentic_fraud_servicing.copilot.orchestrator.run_triage"
_AUTH_PATCH = "agentic_fraud_servicing.copilot.orchestrator.run_auth_assessment"
_RETRIEVAL_PATCH = "agentic_fraud_servicing.copilot.orchestrator.run_retrieval"
_SPECIALISTS_PATCH = "agentic_fraud_servicing.copilot.orchestrator.run_specialists"
_ARBITRATOR_PATCH = "agentic_fraud_servicing.copilot.orchestrator.run_arbitrator"
_CASE_ADVISOR_PATCH = "agentic_fraud_servicing.copilot.orchestrator.run_case_advisor"
_QUESTION_VALIDATOR_PATCH = (
    "agentic_fraud_servicing.copilot.orchestrator.validate_pending_questions"
)


# -- Test Classes --


class TestCopilotOrchestratorInit:
    """Tests for orchestrator initialization."""

    def test_initial_state(self):
        """Constructor sets correct defaults."""
        orch = _make_orchestrator()
        assert orch.case_id is None
        assert orch.call_id is None
        assert orch.hypothesis_scores == {
            "THIRD_PARTY_FRAUD": 0.0,
            "FIRST_PARTY_FRAUD": 0.0,
            "SCAM": 0.0,
            "DISPUTE": 0.0,
            "UNABLE_TO_DETERMINE": 0.0,
        }
        assert orch.impersonation_risk == 0.0
        assert orch.evidence_collected == []
        assert orch.transcript_history == []
        assert orch.accumulated_allegations == []

    def test_stores_gateway_and_provider(self):
        """Gateway and model_provider are stored as attributes."""
        gateway = MagicMock()
        provider = MagicMock()
        orch = CopilotOrchestrator(gateway=gateway, model_provider=provider)
        assert orch.gateway is gateway
        assert orch.model_provider is provider


class TestProcessEvent:
    """Tests for the process_event method."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_returns_copilot_suggestion(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """process_event returns a CopilotSuggestion instance."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert isinstance(result, CopilotSuggestion)
        assert result.call_id == "call-001"
        assert result.timestamp_ms == 1000

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_sets_case_id_from_first_event(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """case_id and call_id are set from the first event."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        await orch.process_event(_make_event(call_id="call-42"))

        assert orch.case_id == "case-call-42"
        assert orch.call_id == "call-42"

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_accumulates_transcript_history(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """transcript_history grows with each event."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        await orch.process_event(_make_event(event_id="evt-1"))
        await orch.process_event(_make_event(event_id="evt-2"))

        assert len(orch.transcript_history) == 2
        assert orch.transcript_history[0].event_id == "evt-1"
        assert orch.transcript_history[1].event_id == "evt-2"

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_updates_impersonation_risk(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """impersonation_risk updates from auth assessment."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result(impersonation_risk=0.75)
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert orch.impersonation_risk == 0.75
        assert result.impersonation_risk == 0.75

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_includes_suggested_questions_from_case_advisor(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """CopilotSuggestion includes questions from the case advisor."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = _mock_case_advisory(
            questions=["What was the transaction amount?", "Where was the charge?"]
        )

        orch = _make_orchestrator()
        orch._turn_count = 3  # Case advisor only runs after turn 3
        result = await orch.process_event(_make_event())

        assert len(result.suggested_questions) == 2
        assert "What was the transaction amount?" in result.suggested_questions


class TestParallelExecution:
    """Tests for the asyncio.gather() parallel execution of triage, auth, and retrieval."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_triage_auth_retrieval_all_called_on_first_event(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """Triage, auth, and retrieval are all invoked on the first event."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        await orch.process_event(_make_event())

        mock_triage.assert_awaited_once()
        mock_auth.assert_awaited_once()
        mock_retrieval.assert_awaited_once()

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_retrieval_not_reinvoked_when_no_allegations(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """Retrieval is cached when triage extracts no allegations."""
        mock_triage.return_value = AllegationExtractionResult(allegations=[])
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        await orch.process_event(_make_event(event_id="evt-1"))
        await orch.process_event(_make_event(event_id="evt-2"))

        # No allegations → cache not invalidated → retrieval called once
        mock_retrieval.assert_awaited_once()
        # Triage and auth called twice (once per event)
        assert mock_triage.await_count == 2
        assert mock_auth.await_count == 2


class TestAccumulatedAllegations:
    """Tests for accumulated allegations growing across events."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_accumulated_allegations_grow(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """accumulated_allegations grows with each process_event call."""
        allegation1 = AllegationExtraction(
            detail_type=AllegationDetailType.UNRECOGNIZED_TRANSACTION,
            description="Unauthorized purchase",
            entities={"amount": "$500"},
            confidence=0.9,
        )
        allegation2 = AllegationExtraction(
            detail_type=AllegationDetailType.CARD_POSSESSION,
            description="Card never left wallet",
            entities={},
            confidence=0.8,
        )
        mock_triage.side_effect = [
            AllegationExtractionResult(allegations=[allegation1]),
            AllegationExtractionResult(allegations=[allegation2]),
        ]
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        await orch.process_event(_make_event(event_id="evt-1"))
        assert len(orch.accumulated_allegations) == 1
        first = orch.accumulated_allegations[0]
        assert first.detail_type == AllegationDetailType.UNRECOGNIZED_TRANSACTION

        await orch.process_event(_make_event(event_id="evt-2"))
        assert len(orch.accumulated_allegations) == 2
        assert orch.accumulated_allegations[1].detail_type == AllegationDetailType.CARD_POSSESSION

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_empty_triage_does_not_add_allegations(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """No allegations added when triage returns empty list."""
        mock_triage.return_value = AllegationExtractionResult(allegations=[])
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        await orch.process_event(_make_event())
        assert len(orch.accumulated_allegations) == 0

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_running_summary_built_from_allegations(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """running_summary in CopilotSuggestion reflects accumulated allegations."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert "Allegations:" in result.running_summary
        assert "UNRECOGNIZED_TRANSACTION" in result.running_summary


class TestHypothesisScoring:
    """Tests for hypothesis agent integration."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_scores_come_from_hypothesis_agent(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """hypothesis_scores are set from hypothesis agent output."""
        custom_scores = {
            "THIRD_PARTY_FRAUD": 0.1,
            "FIRST_PARTY_FRAUD": 0.7,
            "SCAM": 0.1,
            "DISPUTE": 0.1,
        }
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result(scores=custom_scores)
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert orch.hypothesis_scores == custom_scores
        assert result.hypothesis_scores == custom_scores

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_suggestion_contains_all_four_hypothesis_keys(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """CopilotSuggestion.hypothesis_scores has all 4 category keys."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert set(result.hypothesis_scores.keys()) == {
            "THIRD_PARTY_FRAUD",
            "FIRST_PARTY_FRAUD",
            "SCAM",
            "DISPUTE",
        }

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_arbitrator_called_with_accumulated_context(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """Arbitrator receives specialist assessments, allegations, auth, and scores."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        await orch.process_event(_make_event())

        mock_arbitrator.assert_awaited_once()
        call_kwargs = mock_arbitrator.call_args.kwargs
        assert "UNRECOGNIZED_TRANSACTION" in call_kwargs["allegations_summary"]
        assert "Impersonation risk" in call_kwargs["auth_summary"]
        assert "THIRD_PARTY_FRAUD" in str(call_kwargs["current_scores"])
        # Specialist assessments are passed directly
        assert "specialist_assessments" in call_kwargs
        specs = call_kwargs["specialist_assessments"]
        assert "DISPUTE" in specs
        assert "THIRD_PARTY_FRAUD" in specs

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_specialists_receive_evidence_and_conversation(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """Specialists receive evidence and conversation context from orchestrator."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result(
            transaction_summary=(
                "== Disputed Transactions (1 total, $2,847.99) ==\n"
                "2024-01-01:\n- $2,847.99 at TechVault, chip_pin"
            ),
            auth_events=[{"auth_type": "chip_pin", "result": "success", "device_id": "dev-001"}],
            customer_profile={"customer_id": "cust-001", "name": "John Smith"},
        )
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        await orch.process_event(_make_event())

        mock_specialists.assert_awaited_once()
        call_kwargs = mock_specialists.call_args.kwargs
        # Evidence summary is passed to specialists
        evidence = call_kwargs["evidence_summary"]
        assert "Disputed Transactions" in evidence
        assert "2,847.99" in evidence
        # Conversation summary also passed to specialists
        assert "CARDMEMBER" in call_kwargs["conversation_summary"]

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_hypothesis_failure_leaves_scores_unchanged(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """When hypothesis agent fails, scores carry over from previous state."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.side_effect = RuntimeError("LLM timeout")
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        # Pre-set some scores to verify they remain unchanged
        orch.hypothesis_scores = {
            "THIRD_PARTY_FRAUD": 0.6,
            "FIRST_PARTY_FRAUD": 0.1,
            "SCAM": 0.2,
            "DISPUTE": 0.1,
        }

        result = await orch.process_event(_make_event())

        # Scores should be unchanged
        assert orch.hypothesis_scores["THIRD_PARTY_FRAUD"] == 0.6
        assert result.hypothesis_scores["THIRD_PARTY_FRAUD"] == 0.6
        assert any("Hypothesis failed" in f for f in result.risk_flags)


class TestGracefulDegradation:
    """Tests for specialist failure handling."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_triage_failure_continues(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """If triage fails, orchestrator continues and records error."""
        mock_triage.side_effect = RuntimeError("LLM timeout")
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert isinstance(result, CopilotSuggestion)
        assert any("Triage failed" in flag for flag in result.risk_flags)

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_all_specialists_fail(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """If all specialists fail, result still returned with error flags."""
        mock_triage.side_effect = RuntimeError("triage error")
        mock_auth.side_effect = RuntimeError("auth error")
        mock_retrieval.side_effect = RuntimeError("retrieval error")
        mock_specialists.side_effect = RuntimeError("specialists error")
        mock_arbitrator.side_effect = RuntimeError("hypothesis error")
        mock_advisor.side_effect = RuntimeError("case advisor error")

        orch = _make_orchestrator()
        # Advance past early-turn gates so all agents are invoked:
        # turn > 3 for case advisor, impersonation_risk >= 0.4 for auth
        orch._turn_count = 3
        orch.impersonation_risk = 0.5
        result = await orch.process_event(_make_event())

        assert isinstance(result, CopilotSuggestion)
        assert any("Triage failed" in f for f in result.risk_flags)
        assert any("Auth failed" in f for f in result.risk_flags)
        assert any("Retrieval failed" in f for f in result.risk_flags)
        assert any("Specialists failed" in f for f in result.risk_flags)
        assert any("Hypothesis failed" in f for f in result.risk_flags)
        assert any("Case advisor failed" in f for f in result.risk_flags)


class TestStepUpAuth:
    """Tests for step-up auth risk flag propagation."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_step_up_auth_in_risk_flags(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """Step-up recommendation appears in risk_flags."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result(
            impersonation_risk=0.85,
            step_up_recommended=True,
            step_up_method="SMS_OTP",
        )
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert any("Step-up auth recommended: SMS_OTP" in f for f in result.risk_flags)


class TestSafetyGuidance:
    """Tests for safety guidance generation."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_safety_guidance_includes_pan_warning(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """Safety guidance always includes PAN/CVV warning."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert "Never ask for full PAN or CVV" in result.safety_guidance


class TestSpeakerFastPath:
    """Tests for speaker-based fast path routing."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_system_event_skips_all_agents(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """SYSTEM events skip all agents — only CARDMEMBER events trigger agents."""
        orch = _make_orchestrator()
        event = _make_event(text="Call connected", speaker=SpeakerType.SYSTEM)
        result = await orch.process_event(event)

        assert result is None
        mock_triage.assert_not_awaited()
        mock_auth.assert_not_awaited()
        mock_arbitrator.assert_not_awaited()
        mock_retrieval.assert_not_awaited()
        mock_advisor.assert_not_awaited()

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_system_event_returns_previous_hypothesis_scores(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """SYSTEM events return None and do not alter hypothesis scores."""
        orch = _make_orchestrator()
        orch.hypothesis_scores = {
            "THIRD_PARTY_FRAUD": 0.6,
            "FIRST_PARTY_FRAUD": 0.1,
            "SCAM": 0.2,
            "DISPUTE": 0.1,
        }
        event = _make_event(text="Identity verified", speaker=SpeakerType.SYSTEM)
        result = await orch.process_event(event)

        assert result is None
        assert orch.hypothesis_scores["THIRD_PARTY_FRAUD"] == 0.6
        assert orch.hypothesis_scores["FIRST_PARTY_FRAUD"] == 0.1

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_ccp_event_skips_all_agents(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """CCP events skip all agents — only CARDMEMBER events trigger agents."""
        orch = _make_orchestrator()
        event = _make_event(text="Can you confirm the transaction date?", speaker=SpeakerType.CCP)
        result = await orch.process_event(event)

        assert result is None
        mock_triage.assert_not_awaited()
        mock_auth.assert_not_awaited()
        mock_arbitrator.assert_not_awaited()
        mock_retrieval.assert_not_awaited()
        mock_advisor.assert_not_awaited()

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_cardmember_event_runs_full_pipeline(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """CARDMEMBER events run all 5 agents."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        event = _make_event(speaker=SpeakerType.CARDMEMBER)
        await orch.process_event(event)

        mock_triage.assert_awaited_once()
        mock_auth.assert_awaited_once()
        mock_retrieval.assert_awaited_once()
        mock_arbitrator.assert_awaited_once()

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_transcript_history_updated_for_all_speakers(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """Transcript history grows for SYSTEM, CCP, and CARDMEMBER events."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        await orch.process_event(_make_event(event_id="e1", speaker=SpeakerType.SYSTEM))
        await orch.process_event(_make_event(event_id="e2", speaker=SpeakerType.CCP))
        await orch.process_event(_make_event(event_id="e3", speaker=SpeakerType.CARDMEMBER))

        assert len(orch.transcript_history) == 3
        assert orch.transcript_history[0].speaker == SpeakerType.SYSTEM
        assert orch.transcript_history[1].speaker == SpeakerType.CCP
        assert orch.transcript_history[2].speaker == SpeakerType.CARDMEMBER


class TestConditionalAuth:
    """Tests for conditional auth invocation after identity established."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_auth_runs_on_first_three_turns(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """Auth agent is called on turns 1, 2, and 3 unconditionally."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result(impersonation_risk=0.1)
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        for i in range(3):
            await orch.process_event(_make_event(event_id=f"evt-{i}"))

        assert mock_auth.await_count == 3

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_auth_skipped_on_turn_4_low_risk(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """Auth agent is NOT called on turn 4 when impersonation_risk < 0.4."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result(impersonation_risk=0.1)
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        # First 3 turns — auth runs
        for i in range(3):
            await orch.process_event(_make_event(event_id=f"evt-{i}"))
        assert mock_auth.await_count == 3

        # Turn 4 — low risk, auth should be skipped
        await orch.process_event(_make_event(event_id="evt-3"))
        assert mock_auth.await_count == 3  # Still 3, not 4

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_auth_runs_on_turn_4_high_risk(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """Auth agent IS called on turn 4 when impersonation_risk >= 0.4."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result(impersonation_risk=0.5)
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        # First 3 turns — auth runs, risk set to 0.5
        for i in range(3):
            await orch.process_event(_make_event(event_id=f"evt-{i}"))
        assert mock_auth.await_count == 3

        # Turn 4 — high risk (0.5 >= 0.4), auth should run
        await orch.process_event(_make_event(event_id="evt-3"))
        assert mock_auth.await_count == 4

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_impersonation_risk_preserved_when_auth_skipped(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """When auth is skipped, impersonation_risk retains its previous value."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result(impersonation_risk=0.2)
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        # 3 turns with risk 0.2 — auth runs
        for i in range(3):
            await orch.process_event(_make_event(event_id=f"evt-{i}"))
        assert orch.impersonation_risk == 0.2

        # Turn 4 — auth skipped (risk 0.2 < 0.4), risk should stay at 0.2
        result = await orch.process_event(_make_event(event_id="evt-3"))
        assert mock_auth.await_count == 3  # Auth not called on turn 4
        assert orch.impersonation_risk == 0.2
        assert result.impersonation_risk == 0.2


class TestAuthGateCountsCMTurns:
    """Tests that auth gating uses CM turn count, not total turn count."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_auth_runs_on_first_cm_turn_after_system_events(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """Auth runs on the first CM turn even if preceded by SYSTEM/CCP events."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result(impersonation_risk=0.1)
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        # 3 non-CM events first
        await orch.process_event(_make_event(event_id="sys-1", speaker=SpeakerType.SYSTEM))
        await orch.process_event(_make_event(event_id="ccp-1", speaker=SpeakerType.CCP))
        await orch.process_event(_make_event(event_id="sys-2", speaker=SpeakerType.SYSTEM))
        # _turn_count is now 3, but _cm_turn_count is 0
        assert mock_auth.await_count == 0

        # First CM turn — auth should run (cm_turn_count=1 <= 3)
        await orch.process_event(_make_event(event_id="cm-1"))
        assert mock_auth.await_count == 1

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_auth_runs_3_cm_turns_with_interleaved_ccp(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """Auth runs on first 3 CM turns regardless of interleaved CCP/SYSTEM events."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result(impersonation_risk=0.1)
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        # Interleaved: SYSTEM, CM, CCP, CM, SYSTEM, CCP, CM, CCP, CM
        await orch.process_event(_make_event(event_id="s1", speaker=SpeakerType.SYSTEM))
        await orch.process_event(_make_event(event_id="cm1"))  # CM turn 1
        await orch.process_event(_make_event(event_id="ccp1", speaker=SpeakerType.CCP))
        await orch.process_event(_make_event(event_id="cm2"))  # CM turn 2
        await orch.process_event(_make_event(event_id="s2", speaker=SpeakerType.SYSTEM))
        await orch.process_event(_make_event(event_id="ccp2", speaker=SpeakerType.CCP))
        await orch.process_event(_make_event(event_id="cm3"))  # CM turn 3
        assert mock_auth.await_count == 3

        # CM turn 4 — low risk, auth should be skipped
        await orch.process_event(_make_event(event_id="ccp3", speaker=SpeakerType.CCP))
        await orch.process_event(_make_event(event_id="cm4"))
        assert mock_auth.await_count == 3  # Still 3


class TestRetrievalCacheInvalidation:
    """Tests for retrieval cache invalidation after evidence writes."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_retrieval_not_reinvoked_after_allegations(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """Retrieval is NOT re-run after triage extracts allegations — it fetches
        all case data unconditionally, so new allegations don't change results."""
        mock_triage.return_value = _mock_triage_result()  # has allegations
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        await orch.process_event(_make_event(event_id="evt-1"))
        assert mock_retrieval.await_count == 1

        # Second event — retrieval cache is permanent, not invalidated
        await orch.process_event(_make_event(event_id="evt-2"))
        assert mock_retrieval.await_count == 1

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_retrieval_cached_when_no_allegations(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """Retrieval stays cached when triage extracts no allegations."""
        mock_triage.return_value = AllegationExtractionResult(allegations=[])
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        await orch.process_event(_make_event(event_id="evt-1"))
        assert mock_retrieval.await_count == 1

        # No allegations → cache not invalidated → retrieval still cached
        await orch.process_event(_make_event(event_id="evt-2"))
        assert mock_retrieval.await_count == 1


class TestCaseAdvisorIntegration:
    """Tests for case advisor integration in the pipeline."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_case_eligibility_populated(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """CopilotSuggestion.case_eligibility is populated from CaseAdvisory."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = _mock_case_advisory()

        orch = _make_orchestrator()
        orch._turn_count = 3  # Case advisor only runs after turn 3
        result = await orch.process_event(_make_event())

        assert len(result.case_eligibility) == 2
        assert result.case_eligibility[0]["case_type"] == "fraud"
        assert result.case_eligibility[0]["eligibility"] == "eligible"
        assert result.case_eligibility[1]["case_type"] == "dispute"
        assert result.case_eligibility[1]["eligibility"] == "blocked"

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_case_advisory_summary_populated(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """CopilotSuggestion.case_advisory_summary is populated."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = _mock_case_advisory()

        orch = _make_orchestrator()
        orch._turn_count = 3  # Case advisor only runs after turn 3
        result = await orch.process_event(_make_event())

        assert "Fraud case eligible" in result.case_advisory_summary

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_questions_from_case_advisor_in_suggestion(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """CopilotSuggestion.suggested_questions comes from case advisory questions."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = _mock_case_advisory(
            questions=["What date was this transaction?", "Do you still have the card?"]
        )

        orch = _make_orchestrator()
        orch._turn_count = 3
        result = await orch.process_event(_make_event())

        assert result.suggested_questions == [
            "What date was this transaction?",
            "Do you still have the card?",
        ]

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_information_sufficient_propagated(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """information_sufficient from case advisory propagates to CopilotSuggestion."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = _mock_case_advisory(
            questions=[],
            information_sufficient=True,
        )

        orch = _make_orchestrator()
        orch._turn_count = 3
        result = await orch.process_event(_make_event())

        assert result.information_sufficient is True
        assert result.suggested_questions == []

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_case_advisor_failure_graceful(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """Case advisor failure does not break the pipeline."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.side_effect = RuntimeError("LLM timeout")

        orch = _make_orchestrator()
        orch._turn_count = 3  # Case advisor only runs after turn 3
        result = await orch.process_event(_make_event())

        assert isinstance(result, CopilotSuggestion)
        assert result.case_eligibility == []
        assert result.case_advisory_summary == ""
        assert result.suggested_questions == []
        assert result.information_sufficient is False
        assert any("Case advisor failed" in f for f in result.risk_flags)


class TestTriageAssessmentWindow:
    """Tests for the assessment-based window applied to triage agent input."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_first_assessment_all_turns_are_new(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """On the first assessment, all turns are new (new_turn_offset=0)."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        await orch.process_event(_make_event(event_id="evt-0", text="Turn 0"))

        last_call = mock_triage.call_args
        conv_history = last_call.kwargs.get("conversation_history")
        new_turn_offset = last_call.kwargs.get("new_turn_offset")
        assert conv_history is not None
        assert len(conv_history) == 1
        assert new_turn_offset == 0
        assert last_call.kwargs.get("allegation_summary") is None

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_second_assessment_has_context_and_new_turns(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """Second assessment includes context from first + new turns."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        # assess_interval=1 so each CM event triggers assessment
        await orch.process_event(_make_event(event_id="evt-0", text="Turn 0"))
        await orch.process_event(_make_event(event_id="evt-1", text="Turn 1"))

        last_call = mock_triage.call_args
        conv_history = last_call.kwargs.get("conversation_history")
        new_turn_offset = last_call.kwargs.get("new_turn_offset")
        assert conv_history is not None
        assert len(conv_history) == 2
        assert new_turn_offset == 1

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_context_trailing_limited_to_4(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """Context trailing is capped at 4, old turns are excluded."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        # 10 events, all assessed (assess_interval=1)
        for i in range(10):
            await orch.process_event(_make_event(event_id=f"evt-{i}", text=f"Turn {i} text"))

        last_call = mock_triage.call_args
        conv_history = last_call.kwargs.get("conversation_history")
        new_turn_offset = last_call.kwargs.get("new_turn_offset")
        assert conv_history is not None
        assert len(conv_history) == 5
        assert new_turn_offset == 4
        assert "Turn 9 text" in conv_history[4][1]
        assert "Turn 5 text" in conv_history[0][1]

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_allegation_summary_passed_when_allegations_exist(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """Allegation summary is passed when accumulated_allegations is non-empty."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        await orch.process_event(_make_event(event_id="evt-0", text="Turn 0"))
        await orch.process_event(_make_event(event_id="evt-1", text="Turn 1"))

        last_call = mock_triage.call_args
        allegation_summary = last_call.kwargs.get("allegation_summary")
        assert allegation_summary is not None
        assert "UNRECOGNIZED_TRANSACTION" in allegation_summary

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_first_call_no_allegation_summary(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """First assessment has no allegation summary (no prior allegations)."""
        mock_triage.return_value = AllegationExtractionResult(allegations=[])
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        await orch.process_event(_make_event(event_id="evt-0", text="Turn 0"))

        last_call = mock_triage.call_args
        assert last_call.kwargs.get("allegation_summary") is None

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_last_assessed_idx_advances(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """_last_assessed_idx advances after each assessment."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        assert orch._last_assessed_idx == 0

        await orch.process_event(_make_event(event_id="evt-0"))
        assert orch._last_assessed_idx == 1

        await orch.process_event(_make_event(event_id="evt-1"))
        assert orch._last_assessed_idx == 2


class TestCaseAdvisorContext:
    """Tests for conversation context and dedup passed to the case advisor."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_probing_questions_tracked_and_passed(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """Probing questions accumulate and are passed to case advisor."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = _mock_case_advisory(
            questions=["When did the transaction occur?"]
        )

        orch = _make_orchestrator()
        orch._turn_count = 3  # Enable case advisor

        # First event — no probing questions yet
        await orch.process_event(_make_event(event_id="evt-1"))
        first_call_kwargs = mock_advisor.call_args.kwargs
        assert first_call_kwargs["probing_questions"] is None

        # Second event — should have the first event's questions in probing list
        await orch.process_event(_make_event(event_id="evt-2"))
        second_call_kwargs = mock_advisor.call_args.kwargs
        assert second_call_kwargs["probing_questions"] is not None
        texts = [pq.text for pq in second_call_kwargs["probing_questions"]]
        assert "When did the transaction occur?" in texts

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_probing_questions_accumulate_across_turns(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """Probing questions accumulate across turns (no arbitrary limit)."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        # Each advisory returns one question with a target category
        mock_advisor.return_value = _mock_case_advisory(questions=["Test question?"])

        orch = _make_orchestrator()
        orch._turn_count = 3  # Enable case advisor
        for i in range(5):
            await orch.process_event(_make_event(event_id=f"evt-{i}"))

        # All 5 questions accumulated (one per turn)
        assert len(orch._probing_questions) == 5
        assert all(pq.status == "pending" for pq in orch._probing_questions)

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_ARBITRATOR_PATCH, new_callable=AsyncMock)
    @patch(_SPECIALISTS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_conversation_window_passed_to_case_advisor(
        self,
        mock_triage,
        mock_auth,
        mock_retrieval,
        mock_specialists,
        mock_arbitrator,
        mock_advisor,
    ):
        """Case advisor receives the assessment-based conversation window."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_specialists.return_value = _mock_specialist_outputs()
        mock_arbitrator.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = _mock_case_advisory()

        orch = _make_orchestrator()
        orch._turn_count = 3
        await orch.process_event(_make_event(text="I didn't make this purchase"))

        call_kwargs = mock_advisor.call_args.kwargs
        conv_window = call_kwargs["conversation_window"]
        assert conv_window is not None
        speakers = [t[0] for t in conv_window]
        assert "CARDMEMBER" in speakers
