"""Tests for case and copilot domain models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agentic_fraud_servicing.models.case import (
    AuditEntry,
    Case,
    CopilotSuggestion,
    DecisionFactor,
    DecisionRecommendation,
    TimelineEvent,
    TransactionRef,
)
from agentic_fraud_servicing.models.enums import (
    AllegationType,
    CaseStatus,
    InvestigationCategory,
)

NOW = datetime(2026, 3, 6, 12, 0, 0, tzinfo=timezone.utc)


class TestTransactionRef:
    """Tests for the TransactionRef model."""

    def test_creation(self) -> None:
        ref = TransactionRef(
            transaction_id="txn-001",
            amount=99.99,
            merchant_name="ACME Corp",
            transaction_date=NOW,
        )
        assert ref.transaction_id == "txn-001"
        assert ref.amount == 99.99
        assert ref.merchant_name == "ACME Corp"
        assert ref.transaction_date == NOW


class TestTimelineEvent:
    """Tests for the TimelineEvent model."""

    def test_defaults(self) -> None:
        evt = TimelineEvent(timestamp=NOW, event_type="call_start", description="Call began")
        assert evt.source is None

    def test_all_fields(self) -> None:
        evt = TimelineEvent(
            timestamp=NOW,
            event_type="claim_made",
            description="Customer claimed fraud",
            source="triage_agent",
        )
        assert evt.source == "triage_agent"


class TestAuditEntry:
    """Tests for the AuditEntry model."""

    def test_defaults(self) -> None:
        entry = AuditEntry(timestamp=NOW, action="case_created")
        assert entry.agent_id is None
        assert entry.details == ""

    def test_all_fields(self) -> None:
        entry = AuditEntry(
            timestamp=NOW,
            action="evidence_added",
            agent_id="scheme_mapper",
            details="Added transaction node txn-001",
        )
        assert entry.agent_id == "scheme_mapper"
        assert entry.details == "Added transaction node txn-001"


class TestDecisionRecommendation:
    """Tests for DecisionFactor and DecisionRecommendation."""

    def test_with_factors(self) -> None:
        factors = [
            DecisionFactor(factor="High amount", evidence_ref="node-1", weight=0.8),
            DecisionFactor(factor="New device", evidence_ref="node-2", weight=0.6),
        ]
        rec = DecisionRecommendation(
            category=InvestigationCategory.THIRD_PARTY_FRAUD,
            confidence=0.85,
            top_factors=factors,
            uncertainties=["Merchant response pending"],
            suggested_actions=["Block card", "Notify customer"],
            required_approvals=["senior_analyst"],
        )
        assert rec.category == InvestigationCategory.THIRD_PARTY_FRAUD
        assert rec.confidence == 0.85
        assert len(rec.top_factors) == 2
        assert rec.top_factors[0].weight == 0.8
        assert rec.uncertainties == ["Merchant response pending"]

    def test_defaults(self) -> None:
        rec = DecisionRecommendation(category=InvestigationCategory.DISPUTE, confidence=0.5)
        assert rec.top_factors == []
        assert rec.uncertainties == []
        assert rec.suggested_actions == []
        assert rec.required_approvals == []

    def test_all_categories(self) -> None:
        """All 4 InvestigationCategory values are accepted."""
        for cat in InvestigationCategory:
            rec = DecisionRecommendation(category=cat, confidence=0.7)
            assert rec.category == cat


class TestCase:
    """Tests for the Case model."""

    def _make_case(self, **overrides) -> Case:
        """Helper to create a Case with required fields."""
        defaults = {
            "case_id": "case-001",
            "call_id": "call-001",
            "customer_id": "cust-001",
            "account_id": "acct-001",
            "created_at": NOW,
        }
        defaults.update(overrides)
        return Case(**defaults)

    def test_defaults(self) -> None:
        case = self._make_case()
        assert case.status == CaseStatus.OPEN
        assert case.allegation_type is None
        assert case.allegation_confidence == 0.0
        assert case.impersonation_risk == 0.0
        assert case.transactions_in_scope == []
        assert case.timeline == []
        assert case.evidence_refs == []
        assert case.decision_recommendation is None
        assert case.audit_trail == []
        assert case.updated_at is None

    def test_all_fields(self) -> None:
        txn = TransactionRef(
            transaction_id="txn-001",
            amount=250.0,
            merchant_name="Store X",
            transaction_date=NOW,
        )
        rec = DecisionRecommendation(
            category=InvestigationCategory.THIRD_PARTY_FRAUD, confidence=0.9
        )
        case = self._make_case(
            allegation_type=AllegationType.FRAUD,
            allegation_confidence=0.9,
            impersonation_risk=0.3,
            transactions_in_scope=[txn],
            timeline=[TimelineEvent(timestamp=NOW, event_type="call_start", description="Start")],
            evidence_refs=["node-1", "node-2"],
            decision_recommendation=rec,
            status=CaseStatus.INVESTIGATING,
            audit_trail=[AuditEntry(timestamp=NOW, action="opened")],
            updated_at=NOW,
        )
        assert case.allegation_type == AllegationType.FRAUD
        assert case.status == CaseStatus.INVESTIGATING
        assert len(case.transactions_in_scope) == 1
        assert case.decision_recommendation is not None
        assert case.updated_at == NOW

    def test_invalid_allegation_type(self) -> None:
        with pytest.raises(ValidationError):
            self._make_case(allegation_type="INVALID")

    def test_model_dump_roundtrip(self) -> None:
        case = self._make_case(allegation_type=AllegationType.SCAM)
        data = case.model_dump()
        restored = Case(**data)
        assert restored.case_id == case.case_id
        assert restored.allegation_type == AllegationType.SCAM
        assert restored.status == CaseStatus.OPEN

    def test_json_roundtrip(self) -> None:
        case = self._make_case(
            allegation_type=AllegationType.DISPUTE,
            status=CaseStatus.PENDING_REVIEW,
        )
        json_str = case.model_dump_json()
        restored = Case.model_validate_json(json_str)
        assert restored.case_id == case.case_id
        assert restored.allegation_type == AllegationType.DISPUTE
        assert restored.status == CaseStatus.PENDING_REVIEW
        assert restored.created_at == case.created_at


class TestCopilotSuggestion:
    """Tests for the CopilotSuggestion model."""

    def test_defaults(self) -> None:
        suggestion = CopilotSuggestion(call_id="call-001", timestamp_ms=1000)
        assert suggestion.suggested_questions == []
        assert suggestion.risk_flags == []
        assert suggestion.retrieved_facts == []
        assert suggestion.running_summary == ""
        assert suggestion.safety_guidance == ""
        assert suggestion.hypothesis_scores == {}
        assert suggestion.impersonation_risk == 0.0
        assert suggestion.case_eligibility == []
        assert suggestion.case_advisory_summary == ""

    def test_all_fields(self) -> None:
        suggestion = CopilotSuggestion(
            call_id="call-001",
            timestamp_ms=5000,
            suggested_questions=["When did you last use the card?"],
            risk_flags=["High-value transaction"],
            retrieved_facts=["Last auth: chip at POS"],
            running_summary="Customer reports unauthorized purchase",
            safety_guidance="Do not share PAN",
            hypothesis_scores={
                "THIRD_PARTY_FRAUD": 0.7,
                "FIRST_PARTY_FRAUD": 0.0,
                "SCAM": 0.1,
                "DISPUTE": 0.2,
            },
            impersonation_risk=0.15,
        )
        assert len(suggestion.suggested_questions) == 1
        assert suggestion.hypothesis_scores["THIRD_PARTY_FRAUD"] == 0.7
        assert suggestion.impersonation_risk == 0.15

    def test_case_eligibility_with_dicts(self) -> None:
        eligibility = [
            {
                "case_type": "fraud",
                "eligibility": "eligible",
                "met_criteria": ["Identity verified", "Within 120-day window"],
                "unmet_criteria": [],
                "blockers": [],
                "policy_citations": ["Per fraud_case_checklist.md: '120-day window'"],
            },
            {
                "case_type": "dispute",
                "eligibility": "incomplete",
                "met_criteria": ["Amount > $25"],
                "unmet_criteria": ["Merchant contact attempt"],
                "blockers": [],
                "policy_citations": [],
            },
        ]
        suggestion = CopilotSuggestion(
            call_id="call-003",
            timestamp_ms=4000,
            case_eligibility=eligibility,
            case_advisory_summary="Fraud case eligible; dispute needs merchant contact.",
        )
        assert len(suggestion.case_eligibility) == 2
        assert suggestion.case_eligibility[0]["eligibility"] == "eligible"
        assert suggestion.case_eligibility[1]["unmet_criteria"] == ["Merchant contact attempt"]
        assert "Fraud case eligible" in suggestion.case_advisory_summary

    def test_case_eligibility_json_roundtrip(self) -> None:
        suggestion = CopilotSuggestion(
            call_id="call-004",
            timestamp_ms=5000,
            case_eligibility=[
                {"case_type": "scam", "eligibility": "blocked", "blockers": ["Active fraud case"]}
            ],
            case_advisory_summary="Scam case blocked.",
        )
        json_str = suggestion.model_dump_json()
        restored = CopilotSuggestion.model_validate_json(json_str)
        assert restored.case_eligibility == suggestion.case_eligibility
        assert restored.case_advisory_summary == "Scam case blocked."

    def test_model_dump_roundtrip(self) -> None:
        suggestion = CopilotSuggestion(
            call_id="call-002",
            timestamp_ms=3000,
            hypothesis_scores={
                "THIRD_PARTY_FRAUD": 0.8,
                "FIRST_PARTY_FRAUD": 0.0,
                "SCAM": 0.0,
                "DISPUTE": 0.0,
            },
        )
        data = suggestion.model_dump()
        restored = CopilotSuggestion(**data)
        assert restored.call_id == suggestion.call_id
        assert restored.hypothesis_scores["THIRD_PARTY_FRAUD"] == 0.8
