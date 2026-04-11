"""Tests for copilot/case_advisor.py — models, agent, and runner.

The case advisor is a question planner consuming specialist assessments.
It does not load policies — specialists handle policy-grounded reasoning.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_fraud_servicing.copilot.case_advisor import (
    CASE_ADVISOR_INSTRUCTIONS,
    CaseAdvisory,
    CaseTypeAssessment,
    _map_specialists_to_case_types,
    case_advisor,
    load_policies,
    run_case_advisor,
)
from agentic_fraud_servicing.copilot.hypothesis_specialists import (
    SpecialistAssessment,
)

# ---------------------------------------------------------------------------
# CaseTypeAssessment
# ---------------------------------------------------------------------------


class TestCaseTypeAssessment:
    """Tests for the CaseTypeAssessment model."""

    def test_defaults(self) -> None:
        a = CaseTypeAssessment(case_type="fraud", eligibility="incomplete")
        assert a.case_type == "fraud"
        assert a.eligibility == "incomplete"
        assert a.met_criteria == []
        assert a.unmet_criteria == []
        assert a.blockers == []
        assert a.policy_citations == []

    def test_all_fields(self) -> None:
        a = CaseTypeAssessment(
            case_type="dispute",
            eligibility="blocked",
            met_criteria=["transaction identified"],
            unmet_criteria=["merchant contact required"],
            blockers=["refund already issued"],
            policy_citations=["Per dispute_case_checklist.md: 'refund already issued'"],
        )
        assert a.case_type == "dispute"
        assert a.eligibility == "blocked"
        assert len(a.met_criteria) == 1
        assert len(a.unmet_criteria) == 1
        assert len(a.blockers) == 1
        assert len(a.policy_citations) == 1

    def test_json_round_trip(self) -> None:
        a = CaseTypeAssessment(
            case_type="fraud",
            eligibility="eligible",
            met_criteria=["identity verified"],
        )
        data = json.loads(a.model_dump_json())
        restored = CaseTypeAssessment.model_validate(data)
        assert restored == a


# ---------------------------------------------------------------------------
# CaseAdvisory
# ---------------------------------------------------------------------------


class TestCaseAdvisory:
    """Tests for the CaseAdvisory model."""

    def test_defaults(self) -> None:
        c = CaseAdvisory()
        assert c.assessments == []
        assert c.general_warnings == []
        assert c.questions == []
        assert c.rationale == []
        assert c.priority_field == ""
        assert c.information_sufficient is False
        assert c.summary == ""

    def test_with_assessments_and_questions(self) -> None:
        fraud = CaseTypeAssessment(case_type="fraud", eligibility="incomplete")
        dispute = CaseTypeAssessment(case_type="dispute", eligibility="eligible")
        c = CaseAdvisory(
            assessments=[fraud, dispute],
            general_warnings=["Elderly caller — handle with care"],
            questions=["When did the transaction occur?", "Do you still have the card?"],
            rationale=["Need transaction date", "Need card possession info"],
            priority_field="transaction_date",
            information_sufficient=False,
            summary="Fraud case is incomplete. Dispute case is eligible.",
        )
        assert len(c.assessments) == 2
        assert c.assessments[0].case_type == "fraud"
        assert len(c.general_warnings) == 1
        assert len(c.questions) == 2
        assert len(c.rationale) == 2
        assert c.priority_field == "transaction_date"
        assert c.information_sufficient is False
        assert c.summary.startswith("Fraud")

    def test_information_sufficient_with_empty_questions(self) -> None:
        c = CaseAdvisory(
            assessments=[
                CaseTypeAssessment(case_type="fraud", eligibility="eligible"),
            ],
            questions=[],
            information_sufficient=True,
            summary="All required information gathered. Fraud case is eligible.",
        )
        assert c.information_sufficient is True
        assert c.questions == []
        assert c.priority_field == ""

    def test_json_round_trip(self) -> None:
        c = CaseAdvisory(
            assessments=[
                CaseTypeAssessment(case_type="fraud", eligibility="blocked"),
            ],
            questions=["When did this happen?"],
            rationale=["Need transaction date"],
            priority_field="transaction_date",
            information_sufficient=False,
            summary="Fraud is blocked due to prior dispute.",
        )
        data = json.loads(c.model_dump_json())
        restored = CaseAdvisory.model_validate(data)
        assert restored == c


# ---------------------------------------------------------------------------
# load_policies (backward compat utility)
# ---------------------------------------------------------------------------


class TestLoadPolicies:
    """Tests for the load_policies function."""

    def test_loads_real_policies(self) -> None:
        """Loads from docs/policies/ — expects policy files in subdirectories."""
        text = load_policies()
        assert len(text) > 0
        # Check separators include subdirectory paths
        assert "--- dispute/dispute_case_checklist.md ---" in text
        assert "--- fraud/fraud_case_checklist.md ---" in text
        assert "--- scam/fraud_case_checklist.md ---" in text

    def test_concatenated_with_separators(self) -> None:
        text = load_policies()
        # Verify actual policy content is present
        assert "Fraud Case Opening Checklist" in text
        assert "Merchant Dispute Case Opening Checklist" in text
        assert "General Case Opening Guidelines" in text

    def test_missing_directory_returns_empty(self) -> None:
        text = load_policies("/nonexistent/path/to/policies")
        assert text == ""

    def test_empty_directory_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            text = load_policies(tmpdir)
            assert text == ""

    def test_custom_directory_with_md_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test_policy.md"
            p.write_text("# Test Policy\n\nSome rules here.", encoding="utf-8")
            text = load_policies(tmpdir)
            assert "--- test_policy.md ---" in text
            assert "Some rules here." in text


# ---------------------------------------------------------------------------
# _map_specialists_to_case_types
# ---------------------------------------------------------------------------


class TestMapSpecialistsToCaseTypes:
    """Tests for the specialist-to-CaseTypeAssessment mapping."""

    def test_maps_dispute_and_fraud(self):
        """Dispute and fraud specialists map to case types."""
        specs = {
            "DISPUTE": SpecialistAssessment(
                category="DISPUTE",
                likelihood=0.7,
                eligibility="eligible",
                supporting_evidence=["Merchant confirmed cancellation"],
                evidence_gaps=["Refund policy not confirmed"],
                policy_citations=["Per checklist: 'cancellation date required'"],
            ),
            "THIRD_PARTY_FRAUD": SpecialistAssessment(
                category="THIRD_PARTY_FRAUD",
                likelihood=0.2,
                eligibility="blocked",
                reasoning="CM's device was used for auth.",
                evidence_gaps=[],
                policy_citations=["Per fraud checklist: 'enrolled device auth'"],
            ),
        }
        result = _map_specialists_to_case_types(specs)
        assert len(result) == 2

        dispute = next(a for a in result if a.case_type == "dispute")
        assert dispute.eligibility == "eligible"
        assert "Merchant confirmed cancellation" in dispute.met_criteria
        assert "Refund policy not confirmed" in dispute.unmet_criteria
        assert dispute.blockers == []

        fraud = next(a for a in result if a.case_type == "fraud")
        assert fraud.eligibility == "blocked"
        assert len(fraud.blockers) == 1  # reasoning as blocker

    def test_scam_not_mapped(self):
        """Scam specialist does not produce a CaseTypeAssessment."""
        specs = {
            "SCAM": SpecialistAssessment(category="SCAM", likelihood=0.5, eligibility="eligible"),
        }
        result = _map_specialists_to_case_types(specs)
        assert len(result) == 0

    def test_empty_assessments(self):
        """Empty dict produces empty list."""
        result = _map_specialists_to_case_types({})
        assert result == []


# ---------------------------------------------------------------------------
# case_advisor Agent instance
# ---------------------------------------------------------------------------


class TestCaseAdvisorAgent:
    """Tests for the case_advisor Agent instance."""

    def test_agent_name(self) -> None:
        assert case_advisor.name == "case_advisor"

    def test_agent_output_type(self) -> None:
        assert case_advisor.output_type.output_type is CaseAdvisory

    def test_instructions_no_policy_text(self) -> None:
        """Instructions do NOT include raw policy document content."""
        assert "Fraud Case Opening Checklist" not in CASE_ADVISOR_INSTRUCTIONS
        assert "Merchant Dispute Case Opening Checklist" not in CASE_ADVISOR_INSTRUCTIONS

    def test_instructions_reference_specialist_assessments(self) -> None:
        """Instructions reference specialist assessments as input."""
        assert "Specialist Assessments" in CASE_ADVISOR_INSTRUCTIONS

    def test_instructions_contain_question_planning(self) -> None:
        """Instructions include question generation rules."""
        assert "information_sufficient" in CASE_ADVISOR_INSTRUCTIONS
        assert "evidence gap" in CASE_ADVISOR_INSTRUCTIONS.lower()

    def test_instructions_contain_stopping_condition(self) -> None:
        """Instructions include stopping condition rules."""
        assert "Stopping Condition" in CASE_ADVISOR_INSTRUCTIONS
        assert "information_sufficient = true" in CASE_ADVISOR_INSTRUCTIONS

    def test_instructions_contain_pan_cvv_guardrail(self) -> None:
        """Instructions include PAN/CVV guardrail."""
        assert (
            "NEVER ask the customer to reveal their full card number" in CASE_ADVISOR_INSTRUCTIONS
        )
        assert "CVV" in CASE_ADVISOR_INSTRUCTIONS

    def test_instructions_contain_uniqueness_rule(self) -> None:
        """Instructions include question uniqueness guidance."""
        assert "Do NOT repeat" in CASE_ADVISOR_INSTRUCTIONS
        assert "distinct evidence gap" in CASE_ADVISOR_INSTRUCTIONS

    def test_instructions_emphasize_advisory(self) -> None:
        """Instructions emphasize this is advisory, not a final decision."""
        assert "ADVISORY" in CASE_ADVISOR_INSTRUCTIONS


# ---------------------------------------------------------------------------
# run_case_advisor
# ---------------------------------------------------------------------------


class TestRunCaseAdvisor:
    """Tests for the run_case_advisor async function."""

    @pytest.fixture
    def mock_provider(self):
        return MagicMock()

    @pytest.fixture
    def mock_specialist_outputs(self):
        return {
            "DISPUTE": SpecialistAssessment(
                category="DISPUTE",
                likelihood=0.3,
                reasoning="Some merchant issue.",
                eligibility="eligible",
                evidence_gaps=["Merchant contact confirmation"],
            ),
            "SCAM": SpecialistAssessment(
                category="SCAM",
                likelihood=0.1,
                reasoning="No scam indicators.",
                eligibility="eligible",
            ),
            "THIRD_PARTY_FRAUD": SpecialistAssessment(
                category="THIRD_PARTY_FRAUD",
                likelihood=0.5,
                reasoning="Unfamiliar device detected.",
                eligibility="eligible",
                evidence_gaps=["Device fingerprint details"],
            ),
        }

    @pytest.fixture
    def default_scores(self):
        return {
            "THIRD_PARTY_FRAUD": 0.25,
            "FIRST_PARTY_FRAUD": 0.25,
            "SCAM": 0.25,
            "DISPUTE": 0.25,
        }

    async def test_returns_case_advisory_with_mapped_assessments(
        self, mock_provider, mock_specialist_outputs, default_scores
    ) -> None:
        """run_case_advisor returns CaseAdvisory with assessments mapped from specialists."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = CaseAdvisory(
            questions=["When did the transaction happen?"],
            summary="Fraud case eligible.",
        )

        with patch(
            "agentic_fraud_servicing.copilot.case_advisor.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ):
            result = await run_case_advisor(
                specialist_assessments=mock_specialist_outputs,
                hypothesis_scores=default_scores,
                conversation_window=[("CARDMEMBER", "I didn't make this purchase")],
                model_provider=mock_provider,
            )

        assert isinstance(result, CaseAdvisory)
        # Assessments mapped from specialists (dispute + fraud)
        assert len(result.assessments) == 2
        case_types = {a.case_type for a in result.assessments}
        assert case_types == {"dispute", "fraud"}

    async def test_includes_specialist_assessments_in_message(
        self, mock_provider, mock_specialist_outputs, default_scores
    ) -> None:
        """User message contains formatted specialist assessments."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = CaseAdvisory()

        with patch(
            "agentic_fraud_servicing.copilot.case_advisor.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_case_advisor(
                specialist_assessments=mock_specialist_outputs,
                hypothesis_scores=default_scores,
                conversation_window=[("CARDMEMBER", "test")],
                model_provider=mock_provider,
            )

        user_input = mock_run.call_args.kwargs.get("input") or mock_run.call_args.args[1]
        assert "## Specialist Assessments" in user_input
        assert "Dispute Specialist" in user_input
        assert "Fraud Specialist (Third-Party)" in user_input
        assert "Scam Specialist" in user_input
        assert "Evidence gaps:" in user_input
        assert "Eligibility:" in user_input

    async def test_includes_scores_in_message(
        self, mock_provider, mock_specialist_outputs
    ) -> None:
        """User message includes formatted hypothesis scores."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = CaseAdvisory()

        scores = {
            "THIRD_PARTY_FRAUD": 0.10,
            "FIRST_PARTY_FRAUD": 0.60,
            "SCAM": 0.20,
            "DISPUTE": 0.10,
        }

        with patch(
            "agentic_fraud_servicing.copilot.case_advisor.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_case_advisor(
                specialist_assessments=mock_specialist_outputs,
                hypothesis_scores=scores,
                conversation_window=[("CARDMEMBER", "test")],
                model_provider=mock_provider,
            )

        user_input = mock_run.call_args.kwargs.get("input") or mock_run.call_args.args[1]
        assert "FIRST_PARTY_FRAUD: 0.60" in user_input

    async def test_includes_probing_questions(
        self, mock_provider, mock_specialist_outputs, default_scores
    ) -> None:
        """User message includes probing question list with statuses."""
        from agentic_fraud_servicing.models.case import ProbingQuestion

        mock_run_result = MagicMock()
        mock_run_result.final_output = CaseAdvisory()

        probing_qs = [
            ProbingQuestion(
                text="When did this happen?",
                status="pending",
                turn_suggested=1,
                target_category="THIRD_PARTY_FRAUD",
            ),
            ProbingQuestion(
                text="Did you authorize the transaction?",
                status="answered",
                turn_suggested=1,
                target_category="THIRD_PARTY_FRAUD",
                reason="CM confirmed they did not authorize",
            ),
        ]

        with patch(
            "agentic_fraud_servicing.copilot.case_advisor.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_case_advisor(
                specialist_assessments=mock_specialist_outputs,
                hypothesis_scores=default_scores,
                conversation_window=[("CARDMEMBER", "test")],
                model_provider=mock_provider,
                probing_questions=probing_qs,
            )

        user_input = mock_run.call_args.kwargs.get("input") or mock_run.call_args.args[1]
        assert "Current Question List" in user_input
        assert "When did this happen?" in user_input
        assert "[pending]" in user_input
        assert "[answered]" in user_input

    async def test_omits_optional_sections_when_none(
        self, mock_provider, mock_specialist_outputs, default_scores
    ) -> None:
        """Optional sections are omitted when not provided."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = CaseAdvisory()

        with patch(
            "agentic_fraud_servicing.copilot.case_advisor.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_case_advisor(
                specialist_assessments=mock_specialist_outputs,
                hypothesis_scores=default_scores,
                conversation_window=[],
                model_provider=mock_provider,
            )

        user_input = mock_run.call_args.kwargs.get("input") or mock_run.call_args.args[1]
        assert "Recently Suggested Questions" not in user_input
        assert "## Recent Conversation" not in user_input

    async def test_wraps_exceptions_in_runtime_error(
        self, mock_provider, mock_specialist_outputs, default_scores
    ) -> None:
        """run_case_advisor wraps SDK exceptions in RuntimeError."""
        with patch(
            "agentic_fraud_servicing.copilot.case_advisor.run_with_retry",
            new_callable=AsyncMock,
            side_effect=ValueError("LLM connection failed"),
        ):
            with pytest.raises(RuntimeError, match="Case advisor agent failed"):
                await run_case_advisor(
                    specialist_assessments=mock_specialist_outputs,
                    hypothesis_scores=default_scores,
                    conversation_window=[("CARDMEMBER", "test")],
                    model_provider=mock_provider,
                )
