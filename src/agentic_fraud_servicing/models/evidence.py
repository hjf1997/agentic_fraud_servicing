"""Evidence graph models for fraud case investigation.

Defines the evidence graph structure: a base EvidenceNode with 10 concrete
node types, EvidenceEdge for relationships between nodes, and EvidenceRef
for lightweight references. Every node must declare its source_type
(FACT vs ALLEGATION) to separate verified data from customer claims.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel

from agentic_fraud_servicing.models.enums import (
    AllegationDetailType,
    AuthMethod,
    EvidenceEdgeType,
    EvidenceNodeType,
    EvidenceSourceType,
    TransactionChannel,
    TransactionOutcome,
)

# ---------------------------------------------------------------------------
# Base evidence node
# ---------------------------------------------------------------------------


class EvidenceNode(BaseModel):
    """Base model for all evidence graph nodes.

    Every node carries a source_type (FACT or ALLEGATION) to distinguish
    system-verified data from customer-stated claims. This field is required
    with no default so callers must always make an explicit choice.
    """

    node_id: str
    case_id: str
    node_type: EvidenceNodeType
    source_type: EvidenceSourceType
    created_at: datetime


# ---------------------------------------------------------------------------
# Concrete evidence node types
# ---------------------------------------------------------------------------


class Transaction(EvidenceNode):
    """A financial transaction under investigation."""

    node_type: EvidenceNodeType = EvidenceNodeType.TRANSACTION
    amount: float
    currency: str = "USD"
    merchant_name: str
    merchant_id: str | None = None
    transaction_date: datetime
    auth_method: AuthMethod | None = None
    channel: TransactionChannel | None = None
    outcome: TransactionOutcome | None = None
    is_disputed: bool = False


class AuthEvent(EvidenceNode):
    """An authentication event (e.g. chip read, OTP, 3DS)."""

    node_type: EvidenceNodeType = EvidenceNodeType.AUTH_EVENT
    auth_type: str
    result: str
    timestamp: datetime
    device_id: str | None = None


class Card(EvidenceNode):
    """Card details relevant to the investigation."""

    node_type: EvidenceNodeType = EvidenceNodeType.CARD
    card_id: str
    status: str
    recent_changes: list[str] = []


class Device(EvidenceNode):
    """A device associated with a transaction or authentication event."""

    node_type: EvidenceNodeType = EvidenceNodeType.DEVICE
    device_id: str
    fingerprint: str | None = None
    enrolment_date: datetime | None = None


class Customer(EvidenceNode):
    """Customer profile data relevant to the investigation."""

    node_type: EvidenceNodeType = EvidenceNodeType.CUSTOMER
    profile_hash: str
    recent_changes: list[str] = []
    risk_indicators: list[str] = []


class Merchant(EvidenceNode):
    """Merchant information relevant to the dispute."""

    node_type: EvidenceNodeType = EvidenceNodeType.MERCHANT
    merchant_id: str
    category: str | None = None
    dispute_history: int = 0


class DeliveryProof(EvidenceNode):
    """Delivery proof for a disputed transaction."""

    node_type: EvidenceNodeType = EvidenceNodeType.DELIVERY_PROOF
    tracking_id: str
    status: str
    delivery_date: datetime | None = None


class RefundRecord(EvidenceNode):
    """A refund issued for a transaction."""

    node_type: EvidenceNodeType = EvidenceNodeType.REFUND_RECORD
    refund_id: str
    amount: float
    refund_date: datetime
    status: str


class AllegationStatement(EvidenceNode):
    """An allegation statement from the cardholder."""

    node_type: EvidenceNodeType = EvidenceNodeType.ALLEGATION_STATEMENT
    text: str
    detail_type: AllegationDetailType | None = None
    classification: str | None = None
    entities: dict[str, Any] = {}


class InvestigatorNote(EvidenceNode):
    """A note added by an investigator agent during case analysis."""

    node_type: EvidenceNodeType = EvidenceNodeType.INVESTIGATOR_NOTE
    text: str
    author: str


# ---------------------------------------------------------------------------
# Edge and reference models
# ---------------------------------------------------------------------------


class EvidenceEdge(BaseModel):
    """A directed relationship between two evidence nodes."""

    edge_id: str
    case_id: str
    source_node_id: str
    target_node_id: str
    edge_type: EvidenceEdgeType
    created_at: datetime


class EvidenceRef(BaseModel):
    """Lightweight reference to an evidence node (used in case models)."""

    node_id: str
    node_type: EvidenceNodeType
    source_type: EvidenceSourceType
