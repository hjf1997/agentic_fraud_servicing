"""Tests for copilot/case_advisor.py — models and policy loader."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agentic_fraud_servicing.copilot.case_advisor import (
    CaseAdvisory,
    CaseTypeAssessment,
    load_policies,
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
            case_type="scam",
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
