"""Tests for shared enums in models/enums.py."""

from agentic_fraud_servicing.models.enums import (
    INVESTIGATION_CATEGORIES_REFERENCE,
    AllegationDetailType,
    AllegationType,
    AuthMethod,
    CaseStatus,
    EvidenceEdgeType,
    EvidenceNodeType,
    EvidenceSourceType,
    InvestigationCategory,
    RiskLevel,
    SpeakerType,
    TransactionChannel,
)


class TestSpeakerType:
    def test_members(self):
        assert set(SpeakerType) == {
            SpeakerType.CARDMEMBER,
            SpeakerType.CCP,
            SpeakerType.SYSTEM,
        }
        assert len(SpeakerType) == 3

    def test_string_serialization(self):
        assert SpeakerType.CARDMEMBER == "CARDMEMBER"
        assert SpeakerType.CCP.value == "CCP"


class TestAllegationType:
    def test_members(self):
        assert set(AllegationType) == {
            AllegationType.FRAUD,
            AllegationType.DISPUTE,
            AllegationType.SCAM,
        }
        assert len(AllegationType) == 3

    def test_string_serialization(self):
        assert AllegationType.FRAUD == "FRAUD"


class TestInvestigationCategory:
    def test_members(self):
        assert set(InvestigationCategory) == {
            InvestigationCategory.THIRD_PARTY_FRAUD,
            InvestigationCategory.FIRST_PARTY_FRAUD,
            InvestigationCategory.SCAM,
            InvestigationCategory.DISPUTE,
        }
        assert len(InvestigationCategory) == 4

    def test_string_serialization(self):
        assert InvestigationCategory.THIRD_PARTY_FRAUD == "THIRD_PARTY_FRAUD"
        assert InvestigationCategory.FIRST_PARTY_FRAUD == "FIRST_PARTY_FRAUD"
        assert InvestigationCategory.SCAM.value == "SCAM"
        assert InvestigationCategory.DISPUTE.value == "DISPUTE"

    def test_str_mixin(self):
        for member in InvestigationCategory:
            assert isinstance(member, str)


class TestInvestigationCategoriesReference:
    def test_is_string(self):
        assert isinstance(INVESTIGATION_CATEGORIES_REFERENCE, str)

    def test_length(self):
        assert len(INVESTIGATION_CATEGORIES_REFERENCE) > 500

    def test_contains_all_category_names(self):
        for cat in InvestigationCategory:
            assert cat.value in INVESTIGATION_CATEGORIES_REFERENCE

    def test_contains_key_phrases(self):
        """Each category definition includes authorization, fraud actor, etc."""
        for phrase in [
            "Authorization: NO",
            "Authorization: YES",
            "Fraud actor:",
            "CM role:",
            "Evidence focus:",
            "Investigation question:",
            "Typical scenarios:",
        ]:
            assert phrase in INVESTIGATION_CATEGORIES_REFERENCE

    def test_contains_first_party_fraud_cross_cutting_note(self):
        assert "NEVER self-report" in INVESTIGATION_CATEGORIES_REFERENCE


class TestRiskLevel:
    def test_members(self):
        assert set(RiskLevel) == {
            RiskLevel.LOW,
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
        }
        assert len(RiskLevel) == 4

    def test_string_serialization(self):
        assert RiskLevel.CRITICAL == "CRITICAL"


class TestCaseStatus:
    def test_members(self):
        assert set(CaseStatus) == {
            CaseStatus.OPEN,
            CaseStatus.INVESTIGATING,
            CaseStatus.PENDING_REVIEW,
            CaseStatus.CLOSED,
        }
        assert len(CaseStatus) == 4

    def test_string_serialization(self):
        assert CaseStatus.PENDING_REVIEW == "PENDING_REVIEW"


class TestEvidenceEdgeType:
    def test_members(self):
        assert set(EvidenceEdgeType) == {
            EvidenceEdgeType.FACT,
            EvidenceEdgeType.ALLEGATION,
            EvidenceEdgeType.SUPPORTS,
            EvidenceEdgeType.CONTRADICTS,
            EvidenceEdgeType.DERIVED_FROM,
        }
        assert len(EvidenceEdgeType) == 5

    def test_string_serialization(self):
        assert EvidenceEdgeType.DERIVED_FROM == "DERIVED_FROM"


class TestEvidenceSourceType:
    def test_members(self):
        assert set(EvidenceSourceType) == {
            EvidenceSourceType.FACT,
            EvidenceSourceType.ALLEGATION,
        }
        assert len(EvidenceSourceType) == 2

    def test_string_serialization(self):
        assert EvidenceSourceType.FACT == "FACT"
        assert EvidenceSourceType.ALLEGATION == "ALLEGATION"


class TestEvidenceNodeType:
    def test_members(self):
        expected = {
            EvidenceNodeType.TRANSACTION,
            EvidenceNodeType.AUTH_EVENT,
            EvidenceNodeType.CARD,
            EvidenceNodeType.DEVICE,
            EvidenceNodeType.CUSTOMER,
            EvidenceNodeType.MERCHANT,
            EvidenceNodeType.DELIVERY_PROOF,
            EvidenceNodeType.REFUND_RECORD,
            EvidenceNodeType.ALLEGATION_STATEMENT,
            EvidenceNodeType.INVESTIGATOR_NOTE,
        }
        assert set(EvidenceNodeType) == expected
        assert len(EvidenceNodeType) == 10

    def test_string_serialization(self):
        assert EvidenceNodeType.AUTH_EVENT == "AUTH_EVENT"
        assert EvidenceNodeType.ALLEGATION_STATEMENT == "ALLEGATION_STATEMENT"


class TestAuthMethod:
    def test_members(self):
        assert set(AuthMethod) == {
            AuthMethod.CHIP,
            AuthMethod.SWIPE,
            AuthMethod.CONTACTLESS,
            AuthMethod.CNP,
            AuthMethod.MANUAL,
        }
        assert len(AuthMethod) == 5

    def test_string_serialization(self):
        assert AuthMethod.CNP == "CNP"


class TestTransactionChannel:
    def test_members(self):
        assert set(TransactionChannel) == {
            TransactionChannel.POS,
            TransactionChannel.ONLINE,
            TransactionChannel.ATM,
            TransactionChannel.PHONE,
            TransactionChannel.MAIL,
        }
        assert len(TransactionChannel) == 5

    def test_string_serialization(self):
        assert TransactionChannel.ONLINE == "ONLINE"


def test_all_enums_importable():
    """Verify all 11 enum classes are importable from the module."""
    all_enums = [
        SpeakerType,
        AllegationType,
        InvestigationCategory,
        AllegationDetailType,
        RiskLevel,
        CaseStatus,
        EvidenceEdgeType,
        EvidenceSourceType,
        EvidenceNodeType,
        AuthMethod,
        TransactionChannel,
    ]
    assert len(all_enums) == 11
    for enum_cls in all_enums:
        assert issubclass(enum_cls, str)
