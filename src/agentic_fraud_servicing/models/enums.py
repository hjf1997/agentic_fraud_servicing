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
    """Type of allegation reported by the cardholder.

    Represents what the cardmember claims (3 values). A cardmember will
    never self-report first-party fraud, so this enum intentionally has
    no FIRST_PARTY_FRAUD value.
    """

    FRAUD = "FRAUD"
    DISPUTE = "DISPUTE"
    SCAM = "SCAM"


class InvestigationCategory(str, Enum):
    """Investigation conclusion category (4 values).

    Represents what the system concludes after analyzing evidence. Any
    allegation type can map to any investigation category — for example,
    a CM claiming FRAUD may turn out to be FIRST_PARTY_FRAUD.
    """

    THIRD_PARTY_FRAUD = "THIRD_PARTY_FRAUD"
    FIRST_PARTY_FRAUD = "FIRST_PARTY_FRAUD"
    SCAM = "SCAM"
    DISPUTE = "DISPUTE"


INVESTIGATION_CATEGORIES_REFERENCE = """
## Investigation Category Definitions

The system uses four investigation categories to classify what actually happened,
independent of what the cardmember (CM) alleges. Any allegation type (FRAUD,
DISPUTE, SCAM) can resolve to any of these four categories.

### 1. Third-Party Fraud (THIRD_PARTY_FRAUD)
A transaction made without the cardmember's knowledge or permission by an
external criminal who gained unauthorized access to the account or card
credentials.
- Authorization: NO — the CM did not authorize the transaction.
- Fraud actor: External criminal (identity thief, card skimmer, data breach
  exploiter).
- CM role: Victim — the CM had no involvement in the transaction.
- Evidence focus: Authentication logs showing unusual device/IP/location,
  card-not-present indicators, chip vs. swipe discrepancies, device fingerprint
  mismatches, rapid sequential transactions, geographic impossibility.
- Typical scenarios: Stolen card used at POS, compromised credentials used
  online, account takeover after phishing, counterfeit card from skimmed data.
- Investigation question: "Did the cardmember actually authorize the
  transaction?"

### 2. Scam (SCAM)
A transaction authorized by the cardmember, but the authorization was obtained
through deception or manipulation by an external fraudster. The CM willingly
made the payment but was tricked into doing so.
- Authorization: YES — the CM authorized the transaction, but under false
  pretenses.
- Fraud actor: External scammer (romance scammer, investment fraudster, tech
  support impersonator, phishing operator).
- CM role: Victim of manipulation — the CM was deceived into authorizing.
- Evidence focus: Narrative consistency and social-engineering patterns,
  communication trail with the scammer, urgency or pressure tactics in the
  transcript, coached language suggesting third-party influence, payment
  patterns typical of scam (wire, gift cards, crypto).
- Typical scenarios: Romance scam (fake relationship soliciting payments),
  investment scam (fake returns promising scheme), tech support scam
  (fake alerts demanding payment), authorized push payment (APP) fraud,
  impersonation of authority (fake IRS, fake bank).
- Investigation question: "Did the cardmember authorize the payment because
  they were deceived by an external party?"

### 3. First-Party Fraud (FIRST_PARTY_FRAUD)
The legitimate cardmember intentionally provides false information or
misrepresents a transaction to obtain a refund or avoid liability. Also
known as "friendly fraud" or "customer misrepresentation."
- Authorization: YES — the CM made or authorized the transaction.
- Fraud actor: The cardmember themselves.
- CM role: Perpetrator — the CM is intentionally misrepresenting facts.
- Evidence focus: Contradictions between the CM's claims and verified evidence
  (e.g., claims "didn't make purchase" but chip+PIN auth from enrolled device),
  signed delivery proof contradicting "never received," transaction history
  showing merchant familiarity, behavioral red flags (urgency, story shifts,
  accidental knowledge of merchant details before being told).
- Typical scenarios: CM claims unauthorized transaction but evidence shows
  chip+PIN from their enrolled device, CM claims goods not received but
  delivery was signed, CM claims they were scammed but no evidence of external
  manipulator, CM has pattern of similar disputes across merchants.
- Investigation question: "Is the cardmember intentionally misrepresenting
  the facts to obtain an undeserved refund?"
- IMPORTANT: A CM will NEVER self-report first-party fraud. This category is
  always an investigation finding based on contradictions between claims and
  evidence. Any allegation type (FRAUD, DISPUTE, or SCAM) can turn out to be
  first-party fraud.

### 4. Dispute (DISPUTE)
A transaction authorized by the cardmember where the complaint is about the
merchant's performance, billing, or service — not about fraud or deception.
- Authorization: YES — the CM made or authorized the transaction.
- Fraud actor: None — no fraud involved.
- CM role: Legitimate complainant — the CM has a valid grievance with the
  merchant.
- Evidence focus: Merchant records, delivery proof, refund policy, service
  level agreements, billing statements, prior communication with merchant,
  product/service description vs. what was delivered.
- Typical scenarios: Goods not as described, services not rendered, duplicate
  billing, subscription cancellation not honored, defective product, partial
  delivery, merchant refused legitimate refund request.
- Investigation question: "Did the merchant fail to deliver what was promised,
  or is there a legitimate billing error?"
""".strip()


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


class ClaimType(str, Enum):
    """Tier 1 claim types for granular claim extraction.

    9 fraud types + 8 dispute types covering 100% of fraud calls
    and 95% of dispute calls.
    """

    # Fraud Tier 1 (9 types)
    TRANSACTION_DISPUTE = "TRANSACTION_DISPUTE"
    CARD_NOT_PRESENT_FRAUD = "CARD_NOT_PRESENT_FRAUD"
    LOST_STOLEN_CARD = "LOST_STOLEN_CARD"
    IDENTITY_VERIFICATION = "IDENTITY_VERIFICATION"
    ACCOUNT_TAKEOVER = "ACCOUNT_TAKEOVER"
    LOCATION_CLAIM = "LOCATION_CLAIM"
    CARD_POSSESSION = "CARD_POSSESSION"
    MERCHANT_FRAUD = "MERCHANT_FRAUD"
    SPENDING_PATTERN = "SPENDING_PATTERN"

    # Dispute Tier 1 (8 types)
    GOODS_NOT_RECEIVED = "GOODS_NOT_RECEIVED"
    DUPLICATE_CHARGE = "DUPLICATE_CHARGE"
    RETURN_NOT_CREDITED = "RETURN_NOT_CREDITED"
    INCORRECT_AMOUNT = "INCORRECT_AMOUNT"
    GOODS_NOT_AS_DESCRIBED = "GOODS_NOT_AS_DESCRIBED"
    RECURRING_AFTER_CANCEL = "RECURRING_AFTER_CANCEL"
    SERVICES_NOT_RENDERED = "SERVICES_NOT_RENDERED"
    DEFECTIVE_MERCHANDISE = "DEFECTIVE_MERCHANDISE"
