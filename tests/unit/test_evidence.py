"""Tests for evidence graph models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agentic_fraud_servicing.models.enums import (
    AuthMethod,
    EvidenceEdgeType,
    EvidenceNodeType,
    EvidenceSourceType,
    TransactionChannel,
)
from agentic_fraud_servicing.models.evidence import (
    AuthEvent,
    Card,
    ClaimStatement,
    Customer,
    DeliveryProof,
    Device,
    EvidenceEdge,
    EvidenceNode,
    EvidenceRef,
    InvestigatorNote,
    Merchant,
    RefundRecord,
    Transaction,
)

NOW = datetime(2026, 3, 6, 12, 0, 0, tzinfo=timezone.utc)

# Common kwargs shared by all evidence nodes
BASE_KWARGS = {
    "node_id": "n-001",
    "case_id": "case-001",
    "source_type": EvidenceSourceType.FACT,
    "created_at": NOW,
}


class TestSourceTypeRequired:
    """source_type must be explicitly provided — no default allowed."""

    def test_omitting_source_type_raises(self) -> None:
        with pytest.raises(ValidationError, match="source_type"):
            EvidenceNode(
                node_id="n-001",
                case_id="case-001",
                node_type=EvidenceNodeType.TRANSACTION,
                created_at=NOW,
            )

    def test_invalid_source_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            EvidenceNode(
                node_id="n-001",
                case_id="case-001",
                node_type=EvidenceNodeType.TRANSACTION,
                source_type="INVALID",
                created_at=NOW,
            )

    def test_concrete_node_requires_source_type(self) -> None:
        """Concrete nodes inherit the requirement from the base."""
        with pytest.raises(ValidationError, match="source_type"):
            Transaction(
                node_id="n-001",
                case_id="case-001",
                created_at=NOW,
                amount=100.0,
                merchant_name="Shop",
                transaction_date=NOW,
            )


class TestTransaction:
    def test_defaults_and_node_type(self) -> None:
        txn = Transaction(
            **BASE_KWARGS,
            amount=99.99,
            merchant_name="Amazon",
            transaction_date=NOW,
        )
        assert txn.node_type == EvidenceNodeType.TRANSACTION
        assert txn.currency == "USD"
        assert txn.merchant_id is None
        assert txn.auth_method is None
        assert txn.channel is None

    def test_all_fields(self) -> None:
        txn = Transaction(
            **BASE_KWARGS,
            amount=250.0,
            currency="EUR",
            merchant_name="Acme",
            merchant_id="m-100",
            transaction_date=NOW,
            auth_method=AuthMethod.CHIP,
            channel=TransactionChannel.POS,
        )
        assert txn.currency == "EUR"
        assert txn.auth_method == AuthMethod.CHIP
        assert txn.channel == TransactionChannel.POS


class TestAuthEvent:
    def test_creation(self) -> None:
        evt = AuthEvent(
            **BASE_KWARGS,
            auth_type="3DS",
            result="success",
            timestamp=NOW,
        )
        assert evt.node_type == EvidenceNodeType.AUTH_EVENT
        assert evt.device_id is None

    def test_with_device(self) -> None:
        evt = AuthEvent(
            **BASE_KWARGS,
            auth_type="OTP",
            result="failed",
            timestamp=NOW,
            device_id="dev-001",
        )
        assert evt.device_id == "dev-001"


class TestCard:
    def test_creation(self) -> None:
        card = Card(**BASE_KWARGS, card_id="c-001", status="active")
        assert card.node_type == EvidenceNodeType.CARD
        assert card.recent_changes == []

    def test_with_changes(self) -> None:
        card = Card(
            **BASE_KWARGS,
            card_id="c-001",
            status="blocked",
            recent_changes=["pin_changed", "address_updated"],
        )
        assert len(card.recent_changes) == 2


class TestDevice:
    def test_creation(self) -> None:
        dev = Device(**BASE_KWARGS, device_id="dev-001")
        assert dev.node_type == EvidenceNodeType.DEVICE
        assert dev.fingerprint is None
        assert dev.enrolment_date is None


class TestCustomer:
    def test_creation(self) -> None:
        cust = Customer(**BASE_KWARGS, profile_hash="abc123")
        assert cust.node_type == EvidenceNodeType.CUSTOMER
        assert cust.recent_changes == []
        assert cust.risk_indicators == []


class TestMerchant:
    def test_creation(self) -> None:
        m = Merchant(**BASE_KWARGS, merchant_id="m-001")
        assert m.node_type == EvidenceNodeType.MERCHANT
        assert m.category is None
        assert m.dispute_history == 0


class TestDeliveryProof:
    def test_creation(self) -> None:
        dp = DeliveryProof(**BASE_KWARGS, tracking_id="TK-001", status="delivered")
        assert dp.node_type == EvidenceNodeType.DELIVERY_PROOF
        assert dp.delivery_date is None


class TestRefundRecord:
    def test_creation(self) -> None:
        rr = RefundRecord(
            **BASE_KWARGS,
            refund_id="r-001",
            amount=50.0,
            refund_date=NOW,
            status="processed",
        )
        assert rr.node_type == EvidenceNodeType.REFUND_RECORD
        assert rr.amount == 50.0


class TestClaimStatement:
    def test_creation(self) -> None:
        cs = ClaimStatement(**BASE_KWARGS, text="I did not make this purchase")
        assert cs.node_type == EvidenceNodeType.CLAIM_STATEMENT
        assert cs.classification is None

    def test_allegation_source(self) -> None:
        cs = ClaimStatement(
            node_id="n-001",
            case_id="case-001",
            source_type=EvidenceSourceType.ALLEGATION,
            created_at=NOW,
            text="Card was stolen",
            classification="theft",
        )
        assert cs.source_type == EvidenceSourceType.ALLEGATION


class TestInvestigatorNote:
    def test_creation(self) -> None:
        note = InvestigatorNote(
            **BASE_KWARGS, text="Contradicts delivery proof", author="scam_detector"
        )
        assert note.node_type == EvidenceNodeType.INVESTIGATOR_NOTE
        assert note.author == "scam_detector"


class TestEvidenceEdge:
    def test_creation(self) -> None:
        edge = EvidenceEdge(
            edge_id="e-001",
            case_id="case-001",
            source_node_id="n-001",
            target_node_id="n-002",
            edge_type=EvidenceEdgeType.SUPPORTS,
            created_at=NOW,
        )
        assert edge.edge_type == EvidenceEdgeType.SUPPORTS

    def test_all_edge_types(self) -> None:
        """Every EvidenceEdgeType value is accepted."""
        for et in EvidenceEdgeType:
            edge = EvidenceEdge(
                edge_id=f"e-{et.value}",
                case_id="case-001",
                source_node_id="n-001",
                target_node_id="n-002",
                edge_type=et,
                created_at=NOW,
            )
            assert edge.edge_type == et


class TestEvidenceRef:
    def test_creation(self) -> None:
        ref = EvidenceRef(
            node_id="n-001",
            node_type=EvidenceNodeType.TRANSACTION,
            source_type=EvidenceSourceType.FACT,
        )
        assert ref.node_id == "n-001"
        assert ref.source_type == EvidenceSourceType.FACT


class TestRoundTrip:
    """Serialization round-trip tests."""

    def test_transaction_model_dump_roundtrip(self) -> None:
        txn = Transaction(
            **BASE_KWARGS,
            amount=99.99,
            merchant_name="Amazon",
            transaction_date=NOW,
            auth_method=AuthMethod.CNP,
            channel=TransactionChannel.ONLINE,
        )
        data = txn.model_dump()
        restored = Transaction(**data)
        assert restored == txn

    def test_claim_statement_model_dump_roundtrip(self) -> None:
        cs = ClaimStatement(
            node_id="n-002",
            case_id="case-001",
            source_type=EvidenceSourceType.ALLEGATION,
            created_at=NOW,
            text="I never authorized this",
            classification="unauthorized",
        )
        data = cs.model_dump()
        restored = ClaimStatement(**data)
        assert restored == cs

    def test_transaction_json_roundtrip(self) -> None:
        txn = Transaction(
            **BASE_KWARGS,
            amount=150.0,
            merchant_name="Best Buy",
            transaction_date=NOW,
        )
        json_str = txn.model_dump_json()
        restored = Transaction.model_validate_json(json_str)
        assert restored == txn

    def test_evidence_edge_model_dump_roundtrip(self) -> None:
        edge = EvidenceEdge(
            edge_id="e-001",
            case_id="case-001",
            source_node_id="n-001",
            target_node_id="n-002",
            edge_type=EvidenceEdgeType.CONTRADICTS,
            created_at=NOW,
        )
        data = edge.model_dump()
        restored = EvidenceEdge(**data)
        assert restored == edge
