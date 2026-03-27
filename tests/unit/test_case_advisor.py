"""Tests for copilot/case_advisor.py — models, policy loader, agent, and runner."""

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
    case_advisor,
    load_policies,
    run_case_advisor,
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
        assert c.next_info_needed == []
        assert c.summary == ""

    def test_with_assessments(self) -> None:
        fraud = CaseTypeAssessment(case_type="fraud", eligibility="incomplete")
        dispute = CaseTypeAssessment(case_type="dispute", eligibility="eligible")
        c = CaseAdvisory(
            assessments=[fraud, dispute],
            general_warnings=["Elderly caller — handle with care"],
            next_info_needed=["Card status at time of transaction"],
            summary="Fraud case is incomplete. Dispute case is eligible.",
        )
        assert len(c.assessments) == 2
        assert c.assessments[0].case_type == "fraud"
        assert len(c.general_warnings) == 1
        assert c.summary.startswith("Fraud")

    def test_json_round_trip(self) -> None:
        c = CaseAdvisory(
            assessments=[
                CaseTypeAssessment(case_type="fraud", eligibility="blocked"),
            ],
            summary="Fraud is blocked due to prior dispute.",
        )
        data = json.loads(c.model_dump_json())
        restored = CaseAdvisory.model_validate(data)
        assert restored == c

    def test_next_info_needed_list(self) -> None:
        c = CaseAdvisory(
            next_info_needed=[
                "Transaction date",
                "Merchant name",
                "Card status",
            ],
        )
        assert len(c.next_info_needed) == 3
        assert "Transaction date" in c.next_info_needed


# ---------------------------------------------------------------------------
# load_policies
# ---------------------------------------------------------------------------


class TestLoadPolicies:
    """Tests for the load_policies function."""

    def test_loads_real_policies(self) -> None:
        """Loads from docs/policies/ — expects 3 .md files."""
        text = load_policies()
        assert len(text) > 0
        # Check separators for all 3 files
        assert "--- dispute_case_checklist.md ---" in text
        assert "--- fraud_case_checklist.md ---" in text
        assert "--- general_guidelines.md ---" in text

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
# case_advisor Agent instance
# ---------------------------------------------------------------------------


class TestCaseAdvisorAgent:
    """Tests for the case_advisor Agent instance."""

    def test_agent_name(self) -> None:
        assert case_advisor.name == "case_advisor"

    def test_agent_output_type(self) -> None:
        assert case_advisor.output_type.output_type is CaseAdvisory

    def test_instructions_contain_policy_text(self) -> None:
        """Instructions include embedded policy document content."""
        assert "fraud_case_checklist.md" in CASE_ADVISOR_INSTRUCTIONS
        assert "dispute_case_checklist.md" in CASE_ADVISOR_INSTRUCTIONS
        assert "general_guidelines.md" in CASE_ADVISOR_INSTRUCTIONS
        # Actual policy content
        assert "Fraud Case Opening Checklist" in CASE_ADVISOR_INSTRUCTIONS
        assert "Merchant Dispute Case Opening Checklist" in CASE_ADVISOR_INSTRUCTIONS

    def test_instructions_contain_investigation_categories_reference(self) -> None:
        """Instructions include the full INVESTIGATION_CATEGORIES_REFERENCE."""
        assert "THIRD_PARTY_FRAUD" in CASE_ADVISOR_INSTRUCTIONS
        assert "FIRST_PARTY_FRAUD" in CASE_ADVISOR_INSTRUCTIONS
        assert "Authorization: NO" in CASE_ADVISOR_INSTRUCTIONS

    def test_instructions_mention_eligibility_statuses(self) -> None:
        """Instructions reference eligible, blocked, and incomplete statuses."""
        lower = CASE_ADVISOR_INSTRUCTIONS.lower()
        assert "eligible" in lower
        assert "blocked" in lower
        assert "incomplete" in lower

    def test_instructions_emphasize_advisory(self) -> None:
        """Instructions emphasize this is advisory, not a final decision."""
        assert "ADVISORY" in CASE_ADVISOR_INSTRUCTIONS

    def test_instructions_require_policy_citations(self) -> None:
        """Instructions tell the agent to cite specific policy text."""
        lower = CASE_ADVISOR_INSTRUCTIONS.lower()
        assert "cite" in lower
        assert "per" in lower and "filename" in lower or "policy" in lower


# ---------------------------------------------------------------------------
# run_case_advisor
# ---------------------------------------------------------------------------


class TestRunCaseAdvisor:
    """Tests for the run_case_advisor async function."""

    @pytest.fixture
    def mock_provider(self):
        return MagicMock()

    @pytest.fixture
    def sample_advisory(self):
        return CaseAdvisory(
            assessments=[
                CaseTypeAssessment(
                    case_type="fraud",
                    eligibility="incomplete",
                    met_criteria=["Transaction identified"],
                    unmet_criteria=["Card status not confirmed"],
                ),
            ],
            general_warnings=["Elderly caller — handle with care"],
            next_info_needed=["Card status at time of transaction"],
            summary="Fraud case is incomplete. Dispute case blocked.",
        )

    @pytest.fixture
    def default_scores(self):
        return {
            "THIRD_PARTY_FRAUD": 0.25,
            "FIRST_PARTY_FRAUD": 0.25,
            "SCAM": 0.25,
            "DISPUTE": 0.25,
        }

    async def test_returns_case_advisory(
        self, mock_provider, sample_advisory, default_scores
    ) -> None:
        """run_case_advisor returns CaseAdvisory from Runner.run."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = sample_advisory

        with patch(
            "agentic_fraud_servicing.copilot.case_advisor.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ):
            result = await run_case_advisor(
                allegations_summary="UNRECOGNIZED_TRANSACTION: $2847 at TechVault",
                evidence_summary="Chip+PIN auth from enrolled device",
                hypothesis_scores=default_scores,
                conversation_summary="CM called about unauthorized charge",
                model_provider=mock_provider,
            )

        assert isinstance(result, CaseAdvisory)
        assert len(result.assessments) == 1
        assert result.assessments[0].case_type == "fraud"

    async def test_passes_model_provider(self, mock_provider, default_scores) -> None:
        """run_case_advisor passes model_provider in RunConfig."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = CaseAdvisory()

        with patch(
            "agentic_fraud_servicing.copilot.case_advisor.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_case_advisor(
                allegations_summary="test",
                evidence_summary="test",
                hypothesis_scores=default_scores,
                conversation_summary="test",
                model_provider=mock_provider,
            )

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["run_config"].model_provider is mock_provider

    async def test_includes_allegations_in_message(self, mock_provider, default_scores) -> None:
        """run_case_advisor includes allegations_summary in user message."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = CaseAdvisory()

        with patch(
            "agentic_fraud_servicing.copilot.case_advisor.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_case_advisor(
                allegations_summary="UNRECOGNIZED_TRANSACTION: $2847 at TechVault",
                evidence_summary="evidence",
                hypothesis_scores=default_scores,
                conversation_summary="summary",
                model_provider=mock_provider,
            )

        user_input = mock_run.call_args.kwargs.get("input") or mock_run.call_args.args[1]
        assert "UNRECOGNIZED_TRANSACTION: $2847 at TechVault" in user_input
        assert "## Allegations" in user_input

    async def test_includes_evidence_in_message(self, mock_provider, default_scores) -> None:
        """run_case_advisor includes evidence_summary in user message."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = CaseAdvisory()

        with patch(
            "agentic_fraud_servicing.copilot.case_advisor.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_case_advisor(
                allegations_summary="claims",
                evidence_summary="Chip+PIN auth from enrolled device ID dev-123",
                hypothesis_scores=default_scores,
                conversation_summary="summary",
                model_provider=mock_provider,
            )

        user_input = mock_run.call_args.kwargs.get("input") or mock_run.call_args.args[1]
        assert "Chip+PIN auth from enrolled device ID dev-123" in user_input
        assert "## Evidence" in user_input

    async def test_includes_scores_in_message(self, mock_provider) -> None:
        """run_case_advisor includes formatted hypothesis scores in user message."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = CaseAdvisory()

        scores = {
            "THIRD_PARTY_FRAUD": 0.10,
            "FIRST_PARTY_FRAUD": 0.60,
            "SCAM": 0.20,
            "DISPUTE": 0.10,
        }

        with patch(
            "agentic_fraud_servicing.copilot.case_advisor.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_case_advisor(
                allegations_summary="claims",
                evidence_summary="evidence",
                hypothesis_scores=scores,
                conversation_summary="summary",
                model_provider=mock_provider,
            )

        user_input = mock_run.call_args.kwargs.get("input") or mock_run.call_args.args[1]
        assert "FIRST_PARTY_FRAUD: 0.60" in user_input

    async def test_wraps_exceptions_in_runtime_error(self, mock_provider, default_scores) -> None:
        """run_case_advisor wraps SDK exceptions in RuntimeError."""
        with patch(
            "agentic_fraud_servicing.copilot.case_advisor.Runner.run",
            new_callable=AsyncMock,
            side_effect=ValueError("LLM connection failed"),
        ):
            with pytest.raises(RuntimeError, match="Case advisor agent failed"):
                await run_case_advisor(
                    allegations_summary="claims",
                    evidence_summary="evidence",
                    hypothesis_scores=default_scores,
                    conversation_summary="summary",
                    model_provider=mock_provider,
                )
