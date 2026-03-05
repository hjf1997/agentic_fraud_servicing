"""Tests for shared enums in models/enums.py."""

from agentic_fraud_servicing.models.enums import (
    AllegationType,
    AuthMethod,
    CaseStatus,
    EvidenceEdgeType,
    EvidenceNodeType,
    EvidenceSourceType,
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
            EvidenceNodeType.CLAIM_STATEMENT,
            EvidenceNodeType.INVESTIGATOR_NOTE,
        }
        assert set(EvidenceNodeType) == expected
        assert len(EvidenceNodeType) == 10

    def test_string_serialization(self):
        assert EvidenceNodeType.AUTH_EVENT == "AUTH_EVENT"
        assert EvidenceNodeType.CLAIM_STATEMENT == "CLAIM_STATEMENT"


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
    """Verify all 9 enum classes are importable from the module."""
    all_enums = [
        SpeakerType,
        AllegationType,
        RiskLevel,
        CaseStatus,
        EvidenceEdgeType,
        EvidenceSourceType,
        EvidenceNodeType,
        AuthMethod,
        TransactionChannel,
    ]
    assert len(all_enums) == 9
    for enum_cls in all_enums:
        assert issubclass(enum_cls, str)
