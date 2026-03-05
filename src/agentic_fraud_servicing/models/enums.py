"""Shared enums for domain models.

All enums use the str mixin so values serialize as JSON-friendly strings.
These are leaf dependencies imported by all model modules.
"""

from enum import Enum


class SpeakerType(str, Enum):
    """Speaker role in a call transcript."""

    CARDMEMBER = "CARDMEMBER"
    CCP = "CCP"
    SYSTEM = "SYSTEM"


class AllegationType(str, Enum):
    """Type of allegation reported by the cardholder."""

    FRAUD = "FRAUD"
    DISPUTE = "DISPUTE"
    SCAM = "SCAM"


class RiskLevel(str, Enum):
    """Risk severity classification."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CaseStatus(str, Enum):
    """Lifecycle status of a fraud servicing case."""

    OPEN = "OPEN"
    INVESTIGATING = "INVESTIGATING"
    PENDING_REVIEW = "PENDING_REVIEW"
    CLOSED = "CLOSED"


class EvidenceEdgeType(str, Enum):
    """Relationship type between evidence graph nodes."""

    FACT = "FACT"
    ALLEGATION = "ALLEGATION"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    DERIVED_FROM = "DERIVED_FROM"


class EvidenceSourceType(str, Enum):
    """Distinguishes verified system data from customer-stated claims."""

    FACT = "FACT"
    ALLEGATION = "ALLEGATION"


class EvidenceNodeType(str, Enum):
    """Type of evidence node in the evidence graph."""

    TRANSACTION = "TRANSACTION"
    AUTH_EVENT = "AUTH_EVENT"
    CARD = "CARD"
    DEVICE = "DEVICE"
    CUSTOMER = "CUSTOMER"
    MERCHANT = "MERCHANT"
    DELIVERY_PROOF = "DELIVERY_PROOF"
    REFUND_RECORD = "REFUND_RECORD"
    CLAIM_STATEMENT = "CLAIM_STATEMENT"
    INVESTIGATOR_NOTE = "INVESTIGATOR_NOTE"


class AuthMethod(str, Enum):
    """Authentication method used for a transaction."""

    CHIP = "CHIP"
    SWIPE = "SWIPE"
    CONTACTLESS = "CONTACTLESS"
    CNP = "CNP"
    MANUAL = "MANUAL"


class TransactionChannel(str, Enum):
    """Channel through which a transaction was conducted."""

    POS = "POS"
    ONLINE = "ONLINE"
    ATM = "ATM"
    PHONE = "PHONE"
    MAIL = "MAIL"
