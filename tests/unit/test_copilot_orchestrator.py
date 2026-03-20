"""Tests for the copilot orchestrator module.

All specialist agent run_* functions are mocked to return canned results.
Tests verify state management, CopilotSuggestion assembly, graceful
degradation on specialist failure, and running state accumulation.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from agentic_fraud_servicing.copilot.case_advisor import CaseAdvisory, CaseTypeAssessment
from agentic_fraud_servicing.copilot.orchestrator import (
    _INITIAL_MISSING_FIELDS,
    CopilotOrchestrator,
)
from agentic_fraud_servicing.models.allegations import (
    AllegationExtraction,
    AllegationExtractionResult,
)
from agentic_fraud_servicing.models.case import CopilotSuggestion
from agentic_fraud_servicing.models.enums import AllegationDetailType, SpeakerType
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


def _mock_case_advisory():
    """Create a mock CaseAdvisory with realistic assessments."""
    return CaseAdvisory(
        assessments=[
            CaseTypeAssessment(
                case_type="fraud",
                eligibility="incomplete",
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
        next_info_needed=["Verify caller identity", "Confirm authorization method"],
        summary="Fraud case is incomplete pending identity verification. Dispute is blocked.",
    )


def _make_orchestrator() -> CopilotOrchestrator:
    """Create a CopilotOrchestrator with mock gateway and provider."""
    gateway = MagicMock()
    model_provider = MagicMock()
    return CopilotOrchestrator(gateway=gateway, model_provider=model_provider)


# Patch paths for the 6 specialist run_* functions
_TRIAGE_PATCH = "agentic_fraud_servicing.copilot.orchestrator.run_triage"
_AUTH_PATCH = "agentic_fraud_servicing.copilot.orchestrator.run_auth_assessment"
_QUESTION_PATCH = "agentic_fraud_servicing.copilot.orchestrator.run_question_planner"
_RETRIEVAL_PATCH = "agentic_fraud_servicing.copilot.orchestrator.run_retrieval"
_HYPOTHESIS_PATCH = "agentic_fraud_servicing.copilot.orchestrator.run_hypothesis"
_CASE_ADVISOR_PATCH = "agentic_fraud_servicing.copilot.orchestrator.run_case_advisor"


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
        }
        assert orch.impersonation_risk == 0.0
        assert orch.missing_fields == list(_INITIAL_MISSING_FIELDS)
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
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_returns_copilot_suggestion(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """process_event returns a CopilotSuggestion instance."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert isinstance(result, CopilotSuggestion)
        assert result.call_id == "call-001"
        assert result.timestamp_ms == 1000

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_sets_case_id_from_first_event(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """case_id and call_id are set from the first event."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        await orch.process_event(_make_event(call_id="call-42"))

        assert orch.case_id == "case-call-42"
        assert orch.call_id == "call-42"

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_accumulates_transcript_history(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """transcript_history grows with each event."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        await orch.process_event(_make_event(event_id="evt-1"))
        await orch.process_event(_make_event(event_id="evt-2"))

        assert len(orch.transcript_history) == 2
        assert orch.transcript_history[0].event_id == "evt-1"
        assert orch.transcript_history[1].event_id == "evt-2"

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_updates_impersonation_risk(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """impersonation_risk updates from auth assessment."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result(impersonation_risk=0.75)
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert orch.impersonation_risk == 0.75
        assert result.impersonation_risk == 0.75

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_includes_suggested_questions(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """CopilotSuggestion includes questions from the planner."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result(
            questions=["What was the transaction amount?", "Where was the charge?"]
        )
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert len(result.suggested_questions) == 2
        assert "What was the transaction amount?" in result.suggested_questions


class TestParallelExecution:
    """Tests for the asyncio.gather() parallel execution of triage, auth, and retrieval."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_triage_auth_retrieval_all_called_on_first_event(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """Triage, auth, and retrieval are all invoked on the first event."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        await orch.process_event(_make_event())

        mock_triage.assert_awaited_once()
        mock_auth.assert_awaited_once()
        mock_retrieval.assert_awaited_once()

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_retrieval_not_reinvoked_on_second_event(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """Retrieval is cached after the first call and not re-invoked."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        await orch.process_event(_make_event(event_id="evt-1"))
        await orch.process_event(_make_event(event_id="evt-2"))

        # Retrieval called only once — second call returns cached result
        mock_retrieval.assert_awaited_once()
        # Triage and auth called twice (once per event)
        assert mock_triage.await_count == 2
        assert mock_auth.await_count == 2


class TestAccumulatedAllegations:
    """Tests for accumulated allegations growing across events."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_accumulated_allegations_grow(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
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
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
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
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_empty_triage_does_not_add_allegations(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """No allegations added when triage returns empty list."""
        mock_triage.return_value = AllegationExtractionResult(allegations=[])
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        await orch.process_event(_make_event())
        assert len(orch.accumulated_allegations) == 0

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_running_summary_built_from_allegations(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """running_summary in CopilotSuggestion reflects accumulated allegations."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert "Allegations:" in result.running_summary
        assert "UNRECOGNIZED_TRANSACTION" in result.running_summary


class TestHypothesisScoring:
    """Tests for hypothesis agent integration."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_scores_come_from_hypothesis_agent(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
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
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result(scores=custom_scores)
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert orch.hypothesis_scores == custom_scores
        assert result.hypothesis_scores == custom_scores

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_suggestion_contains_all_four_hypothesis_keys(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """CopilotSuggestion.hypothesis_scores has all 4 category keys."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
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
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_hypothesis_called_with_accumulated_context(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """Hypothesis agent receives claims, auth, evidence, scores, and conversation."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        await orch.process_event(_make_event())

        mock_hypothesis.assert_awaited_once()
        call_kwargs = mock_hypothesis.call_args.kwargs
        assert "UNRECOGNIZED_TRANSACTION" in call_kwargs["allegations_summary"]
        assert "Impersonation risk" in call_kwargs["auth_summary"]
        # Evidence summary should contain structured JSON with actual data
        evidence = call_kwargs["evidence_summary"]
        assert "Transactions: 1 found" in evidence
        assert "amount" in evidence
        assert "THIRD_PARTY_FRAUD" in str(call_kwargs["current_scores"])
        assert "CARDMEMBER" in call_kwargs["conversation_summary"]

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_evidence_summary_contains_structured_json(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """Evidence summary includes structured JSON with transaction and auth fields."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result(
            transactions=[{"amount": 2847.99, "merchant": "TechVault", "auth_method": "chip_pin"}],
            auth_events=[{"auth_type": "chip_pin", "result": "success", "device_id": "dev-001"}],
            customer_profile={"customer_id": "cust-001", "name": "John Smith"},
        )
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        await orch.process_event(_make_event())

        call_kwargs = mock_hypothesis.call_args.kwargs
        evidence = call_kwargs["evidence_summary"]
        # Structured transaction fields
        assert "2847.99" in evidence
        assert "TechVault" in evidence
        assert "chip_pin" in evidence
        # Structured auth event fields
        assert "auth_type" in evidence
        assert "device_id" in evidence
        assert "dev-001" in evidence
        # Customer profile included
        assert "customer_id" in evidence
        assert "cust-001" in evidence

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_hypothesis_failure_leaves_scores_unchanged(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """When hypothesis agent fails, scores carry over from previous state."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.side_effect = RuntimeError("LLM timeout")
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
        assert any("Hypothesis scoring failed" in f for f in result.risk_flags)


class TestMissingFieldsUpdate:
    """Tests for entity-based missing field resolution."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_resolves_field_from_entities(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """Fields are resolved when triage extracts a matching entity key."""
        mock_triage.return_value = _mock_triage_result(
            allegations=[
                AllegationExtraction(
                    detail_type=AllegationDetailType.UNRECOGNIZED_TRANSACTION,
                    description="Unauthorized purchase at TechVault",
                    entities={"merchant_name": "TechVault", "amount": "$500"},
                    confidence=0.9,
                )
            ]
        )
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        await orch.process_event(_make_event())

        # Entity keys merchant_name and amount should resolve those fields
        assert "merchant_name" not in orch.missing_fields
        assert "amount" not in orch.missing_fields
        # transaction_date and auth_method still missing (no matching entities)
        assert "transaction_date" in orch.missing_fields
        assert "auth_method" in orch.missing_fields

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_keyword_text_does_not_resolve_field(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """Text containing keywords but no entity extraction does NOT resolve fields."""
        # Triage returns empty entities — keywords in text should not matter
        mock_triage.return_value = _mock_triage_result(
            allegations=[
                AllegationExtraction(
                    detail_type=AllegationDetailType.UNRECOGNIZED_TRANSACTION,
                    description="Mentioned merchant store and amount",
                    entities={},
                    confidence=0.5,
                )
            ]
        )
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        # Text has keywords "amount" and "merchant" but no entities extracted
        await orch.process_event(_make_event(text="What about the merchant store and the amount?"))

        # All fields should still be missing — keyword text alone doesn't resolve
        assert "merchant_name" in orch.missing_fields
        assert "amount" in orch.missing_fields
        assert "transaction_date" in orch.missing_fields
        assert "auth_method" in orch.missing_fields


class TestGracefulDegradation:
    """Tests for specialist failure handling."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_triage_failure_continues(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """If triage fails, orchestrator continues and records error."""
        mock_triage.side_effect = RuntimeError("LLM timeout")
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert isinstance(result, CopilotSuggestion)
        assert any("Triage failed" in flag for flag in result.risk_flags)
        # Questions should still be present from the question planner
        assert len(result.suggested_questions) > 0

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_all_specialists_fail(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """If all specialists fail, result still returned with error flags."""
        mock_triage.side_effect = RuntimeError("triage error")
        mock_auth.side_effect = RuntimeError("auth error")
        mock_question.side_effect = RuntimeError("question error")
        mock_retrieval.side_effect = RuntimeError("retrieval error")
        mock_hypothesis.side_effect = RuntimeError("hypothesis error")
        mock_advisor.side_effect = RuntimeError("case advisor error")

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert isinstance(result, CopilotSuggestion)
        assert any("Triage failed" in f for f in result.risk_flags)
        assert any("Auth assessment failed" in f for f in result.risk_flags)
        assert any("Question planner failed" in f for f in result.risk_flags)
        assert any("Retrieval failed" in f for f in result.risk_flags)
        assert any("Hypothesis scoring failed" in f for f in result.risk_flags)
        assert any("Case advisor failed" in f for f in result.risk_flags)


class TestStepUpAuth:
    """Tests for step-up auth risk flag propagation."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_step_up_auth_in_risk_flags(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
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
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert any("Step-up auth recommended: SMS_OTP" in f for f in result.risk_flags)


class TestSafetyGuidance:
    """Tests for safety guidance generation."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_safety_guidance_includes_pan_warning(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """Safety guidance always includes PAN/CVV warning."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert "Never ask for full PAN or CVV" in result.safety_guidance


class TestSpeakerFastPath:
    """Tests for speaker-based fast path routing."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_system_event_skips_triage_auth_hypothesis(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """SYSTEM events skip triage, auth, and hypothesis agents."""
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        event = _make_event(text="Call connected", speaker=SpeakerType.SYSTEM)
        result = await orch.process_event(event)

        assert isinstance(result, CopilotSuggestion)
        mock_triage.assert_not_awaited()
        mock_auth.assert_not_awaited()
        mock_hypothesis.assert_not_awaited()
        # Retrieval, question planner, and case advisor still run
        mock_retrieval.assert_awaited_once()
        mock_question.assert_awaited_once()

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_system_event_returns_previous_hypothesis_scores(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """SYSTEM events return the previous hypothesis scores unchanged."""
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        orch.hypothesis_scores = {
            "THIRD_PARTY_FRAUD": 0.6,
            "FIRST_PARTY_FRAUD": 0.1,
            "SCAM": 0.2,
            "DISPUTE": 0.1,
        }
        event = _make_event(text="Identity verified", speaker=SpeakerType.SYSTEM)
        result = await orch.process_event(event)

        assert result.hypothesis_scores["THIRD_PARTY_FRAUD"] == 0.6
        assert result.hypothesis_scores["FIRST_PARTY_FRAUD"] == 0.1

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_ccp_event_skips_triage(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """CCP events skip triage but run auth and hypothesis."""
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        event = _make_event(text="Can you confirm the transaction date?", speaker=SpeakerType.CCP)
        await orch.process_event(event)

        mock_triage.assert_not_awaited()
        mock_auth.assert_awaited_once()
        mock_hypothesis.assert_awaited_once()

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_cardmember_event_runs_full_pipeline(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """CARDMEMBER events run all 6 agents."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        event = _make_event(speaker=SpeakerType.CARDMEMBER)
        await orch.process_event(event)

        mock_triage.assert_awaited_once()
        mock_auth.assert_awaited_once()
        mock_retrieval.assert_awaited_once()
        mock_hypothesis.assert_awaited_once()
        mock_question.assert_awaited_once()

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_transcript_history_updated_for_all_speakers(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """Transcript history grows for SYSTEM, CCP, and CARDMEMBER events."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
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
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_auth_runs_on_first_three_turns(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """Auth agent is called on turns 1, 2, and 3 unconditionally."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result(impersonation_risk=0.1)
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        for i in range(3):
            await orch.process_event(_make_event(event_id=f"evt-{i}"))

        assert mock_auth.await_count == 3

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_auth_skipped_on_turn_4_low_risk(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """Auth agent is NOT called on turn 4 when impersonation_risk < 0.4."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result(impersonation_risk=0.1)
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
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
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_auth_runs_on_turn_4_high_risk(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """Auth agent IS called on turn 4 when impersonation_risk >= 0.4."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result(impersonation_risk=0.5)
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
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
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_auth_runs_on_system_event_after_turn_3(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """Auth agent runs on SYSTEM events regardless of turn count.

        Note: SYSTEM events already skip auth via speaker fast path (16.3),
        so this verifies _should_run_auth returns True for SYSTEM events even
        though the fast path takes precedence.
        """
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result(impersonation_risk=0.1)
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        # 3 CARDMEMBER turns to establish identity
        for i in range(3):
            await orch.process_event(_make_event(event_id=f"evt-{i}"))

        # _should_run_auth returns True for SYSTEM event (even though speaker
        # fast path skips auth for SYSTEM events at a higher level)
        system_event = _make_event(
            event_id="evt-sys", text="Identity verified", speaker=SpeakerType.SYSTEM
        )
        assert orch._should_run_auth(system_event) is True

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_impersonation_risk_preserved_when_auth_skipped(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """When auth is skipped, impersonation_risk retains its previous value."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result(impersonation_risk=0.2)
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
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


class TestCaseAdvisorIntegration:
    """Tests for case advisor integration in the pipeline."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_case_eligibility_populated(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """CopilotSuggestion.case_eligibility is populated from CaseAdvisory."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = _mock_case_advisory()

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert len(result.case_eligibility) == 2
        assert result.case_eligibility[0]["case_type"] == "fraud"
        assert result.case_eligibility[0]["eligibility"] == "incomplete"
        assert result.case_eligibility[1]["case_type"] == "dispute"
        assert result.case_eligibility[1]["eligibility"] == "blocked"

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_case_advisory_summary_populated(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """CopilotSuggestion.case_advisory_summary is populated."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = _mock_case_advisory()

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert "Fraud case is incomplete" in result.case_advisory_summary

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_unmet_criteria_prepended_to_missing_fields(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """Unmet criteria from case advisor are prepended to missing_fields."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = _mock_case_advisory()

        orch = _make_orchestrator()
        await orch.process_event(_make_event())

        # Unmet criteria should be at the front of missing_fields
        assert any("[fraud]" in f.lower() for f in orch.missing_fields)
        assert "[fraud] Identity verification pending" in orch.missing_fields
        assert "[fraud] Authorization method unclear" in orch.missing_fields

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_case_advisor_failure_graceful(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """Case advisor failure does not break the pipeline."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.side_effect = RuntimeError("LLM timeout")

        orch = _make_orchestrator()
        result = await orch.process_event(_make_event())

        assert isinstance(result, CopilotSuggestion)
        assert result.case_eligibility == []
        assert result.case_advisory_summary == ""
        assert any("Case advisor failed" in f for f in result.risk_flags)


class TestTriageSlidingWindow:
    """Tests for the sliding window applied to triage agent input."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_triage_receives_at_most_5_turns(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """After 10 events, triage receives only the last 5 turns (not all 10)."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        for i in range(10):
            await orch.process_event(_make_event(event_id=f"evt-{i}", text=f"Turn {i} text"))

        # Check the last call to triage — should have 5 turns, not 10
        last_call = mock_triage.call_args
        conv_history = last_call.kwargs.get("conversation_history")
        assert conv_history is not None
        assert len(conv_history) == 5
        # Should contain the last 5 turns (turns 5-9)
        assert "Turn 5 text" in conv_history[0][1]
        assert "Turn 9 text" in conv_history[4][1]

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_allegation_summary_passed_when_window_active(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """When sliding window is active (>5 turns), allegation summary is passed."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        # Process 6 events so the window kicks in on the 6th
        for i in range(6):
            await orch.process_event(_make_event(event_id=f"evt-{i}", text=f"Turn {i}"))

        last_call = mock_triage.call_args
        allegation_summary = last_call.kwargs.get("allegation_summary")
        # Should have an allegation summary since accumulated_allegations has data
        assert allegation_summary is not None
        assert "UNRECOGNIZED_TRANSACTION" in allegation_summary

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_short_call_sends_full_history_no_summary(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """Short calls (<=5 turns) send full history without allegation summary."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        # Process 3 events — under the window threshold
        for i in range(3):
            await orch.process_event(_make_event(event_id=f"evt-{i}", text=f"Turn {i}"))

        last_call = mock_triage.call_args
        conv_history = last_call.kwargs.get("conversation_history")
        allegation_summary = last_call.kwargs.get("allegation_summary")
        # Full history (3 turns) should be sent
        assert conv_history is not None
        assert len(conv_history) == 3
        # No allegation summary for short calls
        assert allegation_summary is None


class TestQuestionPlannerContext:
    """Tests for conversation context and dedup passed to the question planner."""

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_recent_turns_passed_to_question_planner(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """Question planner receives recent conversation turns."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        await orch.process_event(_make_event(text="I didn't make this purchase"))

        call_kwargs = mock_question.call_args.kwargs
        assert call_kwargs["recent_turns"] is not None
        speakers = [t[0] for t in call_kwargs["recent_turns"]]
        assert "CARDMEMBER" in speakers

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_recent_suggestions_tracked_and_passed(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """Previously suggested questions are passed to avoid repetition."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result(
            questions=["When did the transaction occur?"]
        )
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        # First event — no recent suggestions yet
        await orch.process_event(_make_event(event_id="evt-1"))
        first_call_kwargs = mock_question.call_args.kwargs
        assert first_call_kwargs["recent_questions"] is None

        # Second event — should have the first event's suggestions
        await orch.process_event(_make_event(event_id="evt-2"))
        second_call_kwargs = mock_question.call_args.kwargs
        assert second_call_kwargs["recent_questions"] is not None
        assert "When did the transaction occur?" in second_call_kwargs["recent_questions"]

    @patch(_CASE_ADVISOR_PATCH, new_callable=AsyncMock)
    @patch(_HYPOTHESIS_PATCH, new_callable=AsyncMock)
    @patch(_RETRIEVAL_PATCH, new_callable=AsyncMock)
    @patch(_QUESTION_PATCH, new_callable=AsyncMock)
    @patch(_AUTH_PATCH, new_callable=AsyncMock)
    @patch(_TRIAGE_PATCH, new_callable=AsyncMock)
    async def test_recent_suggestions_limited_to_three(
        self, mock_triage, mock_auth, mock_question, mock_retrieval, mock_hypothesis, mock_advisor
    ):
        """_recent_suggestions stores at most 3 entries."""
        mock_triage.return_value = _mock_triage_result()
        mock_auth.return_value = _mock_auth_result()
        mock_question.return_value = _mock_question_result()
        mock_retrieval.return_value = _mock_retrieval_result()
        mock_hypothesis.return_value = _mock_hypothesis_result()
        mock_advisor.return_value = None

        orch = _make_orchestrator()
        for i in range(5):
            await orch.process_event(_make_event(event_id=f"evt-{i}"))

        assert len(orch._recent_suggestions) == 3
