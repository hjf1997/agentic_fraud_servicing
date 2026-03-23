# Enterprise Evaluation: Generating Scenario Scripts from Case HTML

## Purpose

This document instructs an LLM to generate a **simulation scenario script** for
the AMEX Agentic Fraud Servicing system from a real case HTML page. The generated
script seeds evidence into the system so the copilot can process a **pre-existing
transcript** (not LLM-generated dialogue). This is an **evaluation mode** — the
transcript already happened, and we measure how well the copilot would have
assisted the CCP in real time.

---

## What You Receive

1. **This README** — instructions for generating the scenario script.
2. **A case HTML page** — an internal AMEX page displaying all relevant data for
   a single case (transactions, auth events, customer profile, merchant info,
   CCP notes, call transcript, and the final outcome tag).

---

## What You Generate

A single Python file: `scripts/scenario_{name}.py`

This file must follow the exact pattern of existing scenario modules. It contains:
1. A module docstring describing the case scenario.
2. Evidence seeding functions that inject real case data.
3. A transcript loaded from a file (NOT LLM-generated dialogue).
4. A `build_scenario()` function that returns a `Scenario` dataclass.
5. A `register_scenario()` call at module bottom.

---

## Step-by-Step Instructions

### Step 1: Extract Data from the HTML

Parse the case HTML and extract these categories of data. Map each to the
corresponding evidence node model.

#### 1a. Transaction(s)

For each transaction in the case:

```python
from agentic_fraud_servicing.models.evidence import Transaction
from agentic_fraud_servicing.models.enums import (
    AuthMethod, TransactionChannel, EvidenceSourceType
)

Transaction(
    node_id="txn-{scenario_prefix}-{seq}",     # e.g. "txn-eval001-001"
    case_id=case_id,
    source_type=EvidenceSourceType.FACT,         # transactions are always FACT
    created_at=_NOW,
    amount=<amount_from_html>,                   # float, e.g. 487.50
    merchant_name="<merchant_name_from_html>",
    merchant_id="merch-<id>",                    # or None if not available
    transaction_date=<date_from_html>,           # datetime object
    auth_method=AuthMethod.<method>,             # CHIP, SWIPE, CONTACTLESS, CNP, MANUAL
    channel=TransactionChannel.<channel>,        # POS, ONLINE, ATM, PHONE, MAIL
)
```

**Auth method mapping** from HTML terminology:
- "chip and PIN", "chip+PIN", "EMV" → `AuthMethod.CHIP`
- "swipe", "magnetic stripe" → `AuthMethod.SWIPE`
- "tap", "contactless", "NFC" → `AuthMethod.CONTACTLESS`
- "online", "e-commerce", "card not present", "CNP" → `AuthMethod.CNP`
- "manual entry", "key entered" → `AuthMethod.MANUAL`

**Channel mapping**:
- "point of sale", "in-store", "retail" → `TransactionChannel.POS`
- "online", "web", "e-commerce" → `TransactionChannel.ONLINE`
- "ATM" → `TransactionChannel.ATM`
- "phone order", "MOTO" → `TransactionChannel.PHONE`

#### 1b. AuthEvent(s)

For each authentication record:

```python
from agentic_fraud_servicing.models.evidence import AuthEvent

AuthEvent(
    node_id="auth-{scenario_prefix}-{seq}",
    case_id=case_id,
    source_type=EvidenceSourceType.FACT,
    created_at=_NOW,
    auth_type="<type>",        # "chip_pin", "3ds", "otp_sms", "password", etc.
    result="<result>",         # "success", "failed", "challenged"
    timestamp=<datetime>,
    device_id="<device_id>",   # or None if not available
)
```

#### 1c. Customer Profile

```python
from agentic_fraud_servicing.models.evidence import Customer

Customer(
    node_id="cust-{scenario_prefix}-001",
    case_id=case_id,
    source_type=EvidenceSourceType.FACT,
    created_at=_NOW,
    profile_hash="cust-<id>",
    recent_changes=[...],      # e.g. ["address_change_30d_ago"]
    risk_indicators=[...],     # e.g. ["first_dispute", "high_value_purchase"]
)
```

Risk indicators to look for in the HTML:
- First-time dispute → `"first_dispute"`
- Multiple disputes → `"repeat_disputer"`
- Recent address/phone/email change → `"address_change_Nd_ago"`
- High-value transaction for this customer → `"high_value_purchase"`
- New merchant relationship → `"new_merchant_relationship"`

#### 1d. Merchant Profile

```python
from agentic_fraud_servicing.models.evidence import Merchant

Merchant(
    node_id="merch-{scenario_prefix}-001",
    case_id=case_id,
    source_type=EvidenceSourceType.FACT,
    created_at=_NOW,
    merchant_id="merch-<id>",
    category="<category>",           # e.g. "electronics_retail", "food_delivery"
    dispute_history=<int>,           # number of prior disputes at this merchant
)
```

#### 1e. Card Status

```python
from agentic_fraud_servicing.models.evidence import Card

Card(
    node_id="card-{scenario_prefix}-001",
    case_id=case_id,
    source_type=EvidenceSourceType.FACT,
    created_at=_NOW,
    card_id="card-<id>",
    status="active",                 # or "blocked", "replaced", "closed"
    recent_changes=[...],            # e.g. ["reported_lost_2d_ago", "pin_changed_5d_ago"]
)
```

#### 1f. Delivery Proof (if available)

```python
from agentic_fraud_servicing.models.evidence import DeliveryProof

DeliveryProof(
    node_id="delivery-{scenario_prefix}-001",
    case_id=case_id,
    source_type=EvidenceSourceType.FACT,
    created_at=_NOW,
    tracking_id="<tracking>",
    status="<status>",               # "delivered_signed", "delivered_unsigned", "in_transit", etc.
    delivery_date=<datetime_or_none>,
)
```

#### 1g. Allegation Statements

For each claim the cardmember made (from CCP notes or transcript):

```python
from agentic_fraud_servicing.models.evidence import AllegationStatement
from agentic_fraud_servicing.models.enums import AllegationDetailType

AllegationStatement(
    node_id="allegation-{scenario_prefix}-{seq}",
    case_id=case_id,
    source_type=EvidenceSourceType.ALLEGATION,  # claims are always ALLEGATION
    created_at=_NOW,
    text="<what the cardmember said — verbatim or paraphrased>",
    detail_type=AllegationDetailType.<type>,     # see taxonomy below
    classification="<free_text_classification>",
)
```

**AllegationDetailType taxonomy** (17 values):

Fraud tier:
- `UNRECOGNIZED_TRANSACTION` — "I don't recognize this charge"
- `CARD_NOT_PRESENT_FRAUD` — "I didn't make this online purchase"
- `LOST_STOLEN_CARD` — "My card was lost/stolen"
- `IDENTITY_VERIFICATION` — claims about identity not matching
- `ACCOUNT_TAKEOVER` — "Someone changed my account details"
- `LOCATION_CLAIM` — "I was in a different city/country"
- `CARD_POSSESSION` — "I had my card with me the whole time"
- `MERCHANT_FRAUD` — "The merchant is fraudulent"
- `SPENDING_PATTERN` — "This doesn't match my spending pattern"

Dispute tier:
- `GOODS_NOT_RECEIVED` — "I never received the item"
- `DUPLICATE_CHARGE` — "I was charged twice"
- `RETURN_NOT_CREDITED` — "I returned the item but wasn't refunded"
- `INCORRECT_AMOUNT` — "The amount is wrong"
- `GOODS_NOT_AS_DESCRIBED` — "The item was not as described"
- `RECURRING_AFTER_CANCEL` — "I cancelled but was still charged"
- `SERVICES_NOT_RENDERED` — "The service was never provided"
- `DEFECTIVE_MERCHANDISE` — "The product was defective"

#### 1h. Evidence Edges

Create edges to capture relationships between evidence nodes:

```python
from agentic_fraud_servicing.models.evidence import EvidenceEdge
from agentic_fraud_servicing.models.enums import EvidenceEdgeType

# Types: FACT, ALLEGATION, SUPPORTS, CONTRADICTS, DERIVED_FROM
```

Common edge patterns:
- Transaction → AuthEvent (SUPPORTS): auth event proves how transaction was authenticated
- AllegationStatement → Transaction (ALLEGATION): claim is about this transaction
- AllegationStatement → AuthEvent (CONTRADICTS): claim contradicts auth evidence
- DeliveryProof → Transaction (SUPPORTS): delivery confirms the purchase happened
- InvestigatorNote → Transaction (DERIVED_FROM): note derived from transaction analysis

#### 1i. Prior Case History (if any)

If the HTML shows prior disputes, investigations, or notes on the same
transaction or customer, create InvestigatorNote nodes with `source_type=FACT`:

```python
from agentic_fraud_servicing.models.evidence import InvestigatorNote

InvestigatorNote(
    node_id="inv-note-{scenario_prefix}-{seq}",
    case_id=case_id,
    source_type=EvidenceSourceType.FACT,
    created_at=<date_of_note>,
    text="<summary of prior investigation finding>",
    author="<who_wrote_it>",
)
```

### Step 2: Extract the Transcript

The HTML contains a call transcript. Extract it as a JSON array and save it to
`scripts/transcripts/{scenario_name}.json`.

Each transcript event must have this structure:

```json
{
    "call_id": "call-eval-{scenario_name}",
    "event_id": "evt-{sequence_number}",
    "timestamp_ms": <milliseconds_from_call_start>,
    "speaker": "CCP" | "CARDMEMBER" | "SYSTEM",
    "text": "<the spoken text>",
    "confidence": 1.0,
    "meta": {"channel": "phone", "locale": "en-US"}
}
```

**Speaker mapping** from HTML terminology:
- Agent names, "CSR", "Representative", "Specialist" → `"CCP"`
- Customer names, "Cardholder", "Caller", "Member" → `"CARDMEMBER"`
- System messages, automated notes, verification status → `"SYSTEM"`

**Important**: Include ALL turns from the transcript — do not summarize or skip.
The copilot processes each turn sequentially and must see the full conversation.

If the HTML contains timestamps, convert to milliseconds from call start.
If no timestamps, use 5-second intervals (5000ms increments).

### Step 3: Extract the Ground Truth Outcome

The HTML contains an outcome tag or case resolution. Extract it as:

```python
# Ground truth from the HTML's outcome tag
GROUND_TRUTH = {
    "outcome": "<fraud|dispute|scam|first_party_fraud|legitimate>",
    "resolution": "<approved|denied|partial|pending>",
    "amount_credited": <float_or_none>,
    "notes": "<any CCP or investigator notes about the resolution>",
}
```

Map the HTML outcome to `InvestigationCategory` values:
- "Fraud", "Unauthorized" → `"THIRD_PARTY_FRAUD"`
- "Scam", "Social engineering" → `"SCAM"`
- "Friendly fraud", "First-party" → `"FIRST_PARTY_FRAUD"`
- "Dispute", "Merchant issue" → `"DISPUTE"`
- "Legitimate", "No fraud found" → `"FIRST_PARTY_FRAUD"` (CM's claim was wrong)

### Step 4: Create the Case

```python
from agentic_fraud_servicing.models.case import Case, TransactionRef, AuditEntry
from agentic_fraud_servicing.models.enums import AllegationType, CaseStatus

def _create_case(gateway, case_id, call_id):
    case = Case(
        case_id=case_id,
        call_id=call_id,
        customer_id="cust-<id>",
        account_id="acct-<id>",
        allegation_type=AllegationType.<type>,  # What the CM claims: FRAUD, DISPUTE, or SCAM
        allegation_confidence=0.5,
        status=CaseStatus.OPEN,
        transactions_in_scope=[
            TransactionRef(
                transaction_id="txn-{scenario_prefix}-001",
                amount=<amount>,
                merchant_name="<merchant>",
                transaction_date=<date>,
            ),
            # ... one per disputed transaction
        ],
        audit_trail=[
            AuditEntry(
                timestamp=_NOW,
                action="case_created",
                agent_id="evaluation",
                details="Case created from enterprise case HTML for evaluation.",
            ),
        ],
        created_at=_NOW,
    )
    create_case(gateway, ctx, case)
    return case
```

### Step 5: Build the Scenario

```python
from scripts.simulation_data import Scenario, register_scenario

def build_scenario() -> Scenario:
    return Scenario(
        name="eval_{descriptive_name}",
        title="<One-line description from the HTML case>",
        description="<2-4 line description of the case>",
        case_id="case-eval-{name}-001",
        call_id="call-eval-{name}-001",
        cm_system_prompt="",  # Empty — we use real transcripts, not LLM-generated
        system_event_auth=None,  # Not needed — real transcript has these
        system_event_evidence=None,
        max_turns=0,  # Not used in evaluation mode
        seed_evidence_fn=_seed_evidence,
        create_case_fn=_create_case,
    )

register_scenario("eval_{name}", build_scenario)
```

### Step 6: Store Ground Truth

Add the ground truth outcome to the scenario file as a module-level constant:

```python
GROUND_TRUTH = {
    "investigation_category": "THIRD_PARTY_FRAUD",  # or FIRST_PARTY_FRAUD, SCAM, DISPUTE
    "resolution": "approved",                         # approved, denied, partial, pending
    "amount_credited": 487.50,                        # or None
    "ccp_notes": "...",                               # verbatim CCP notes from HTML
}
```

---

## Complete File Template

```python
"""Scenario: {title} — Enterprise evaluation from case HTML.

{2-4 sentence description of the case, the allegation, and the outcome.}

Ground truth outcome: {INVESTIGATION_CATEGORY} ({resolution}).
"""

from datetime import datetime, timedelta, timezone

from agentic_fraud_servicing.gateway.tool_gateway import AuthContext, ToolGateway
from agentic_fraud_servicing.gateway.tools.write_tools import (
    append_evidence_edge,
    append_evidence_node,
    create_case,
)
from agentic_fraud_servicing.models.case import AuditEntry, Case, TransactionRef
from agentic_fraud_servicing.models.enums import (
    AllegationDetailType,
    AllegationType,
    AuthMethod,
    CaseStatus,
    EvidenceEdgeType,
    EvidenceSourceType,
    TransactionChannel,
)
from agentic_fraud_servicing.models.evidence import (
    AllegationStatement,
    AuthEvent,
    Card,
    Customer,
    DeliveryProof,       # if applicable
    EvidenceEdge,
    InvestigatorNote,    # if prior history
    Merchant,
    Transaction,
)
from scripts.simulation_data import Scenario, register_scenario

_NOW = datetime.now(tz=timezone.utc)
# Define relative timestamps from HTML dates:
# _TXN_DATE = _NOW - timedelta(days=<days_ago>)

# ---------------------------------------------------------------
# Ground truth from the case HTML outcome tag
# ---------------------------------------------------------------
GROUND_TRUTH = {
    "investigation_category": "<THIRD_PARTY_FRAUD|FIRST_PARTY_FRAUD|SCAM|DISPUTE>",
    "resolution": "<approved|denied|partial|pending>",
    "amount_credited": None,
    "ccp_notes": "<verbatim CCP notes from the HTML>",
}


def _seed_evidence(gateway: ToolGateway, case_id: str) -> None:
    """Seed evidence nodes and edges extracted from the case HTML."""
    ctx = AuthContext(agent_id="evaluation", case_id=case_id, permissions={"write"})

    # --- Evidence nodes ---
    # (paste Transaction, AuthEvent, Customer, Merchant, Card, etc. here)

    # --- Evidence edges ---
    # (paste EvidenceEdge instances here)


def _create_case(gateway: ToolGateway, case_id: str, call_id: str) -> Case:
    """Create the initial Case from HTML case data."""
    ctx = AuthContext(agent_id="evaluation", case_id=case_id, permissions={"write"})
    case = Case(
        case_id=case_id,
        call_id=call_id,
        customer_id="...",
        account_id="...",
        allegation_type=AllegationType.FRAUD,  # what CM claims
        status=CaseStatus.OPEN,
        transactions_in_scope=[...],
        audit_trail=[
            AuditEntry(
                timestamp=_NOW,
                action="case_created",
                agent_id="evaluation",
                details="Evaluation case from enterprise HTML.",
            ),
        ],
        created_at=_NOW,
    )
    create_case(gateway, ctx, case)
    return case


def build_scenario() -> Scenario:
    """Build the evaluation scenario."""
    return Scenario(
        name="eval_<name>",
        title="<title>",
        description="<description>",
        case_id="case-eval-<name>-001",
        call_id="call-eval-<name>-001",
        cm_system_prompt="",
        system_event_auth=None,
        system_event_evidence=None,
        max_turns=0,
        seed_evidence_fn=_seed_evidence,
        create_case_fn=_create_case,
    )


register_scenario("eval_<name>", build_scenario)
```

---

## Transcript File Format

Save the transcript to `scripts/transcripts/{scenario_name}.json`:

```json
[
    {
        "call_id": "call-eval-{name}-001",
        "event_id": "evt-001",
        "timestamp_ms": 0,
        "speaker": "CCP",
        "text": "Thank you for calling American Express, my name is Sarah. How can I help you today?",
        "confidence": 1.0,
        "meta": {"channel": "phone", "locale": "en-US"}
    },
    {
        "call_id": "call-eval-{name}-001",
        "event_id": "evt-002",
        "timestamp_ms": 5000,
        "speaker": "CARDMEMBER",
        "text": "Hi, I need to dispute a charge on my account...",
        "confidence": 1.0,
        "meta": {"channel": "phone", "locale": "en-US"}
    }
]
```

---

## Key Rules

1. **Evidence source types**: System-verified data (transactions, auth events,
   delivery proof, card status, merchant profiles) are always `FACT`. Customer
   statements and claims are always `ALLEGATION`. Never mix them.

2. **PII redaction**: Replace any real card numbers with fake AMEX test numbers
   (e.g., `371449635398431`). Replace real SSNs, DOBs, and addresses with
   placeholders. The ingestion pipeline will redact these anyway, but don't
   include real PII in the script.

3. **Timestamps**: Use relative timestamps (`_NOW - timedelta(days=N)`) rather
   than absolute dates. This keeps the scenario reproducible.

4. **One scenario per case**: Each HTML page produces one scenario file and one
   transcript file. Do not combine multiple cases.

5. **No LLM dialogue generation**: The `cm_system_prompt` should be empty string.
   The transcript comes from the real call, not from LLM simulators. The
   `max_turns` should be 0 (unused in evaluation mode).

6. **Ground truth is required**: Every evaluation scenario must include the
   `GROUND_TRUTH` dict with the actual investigation outcome from the HTML.

7. **AllegationType vs InvestigationCategory**: Use `AllegationType` (FRAUD,
   DISPUTE, SCAM) for `Case.allegation_type` — this is what the CM claims.
   Use `InvestigationCategory` values (THIRD_PARTY_FRAUD, FIRST_PARTY_FRAUD,
   SCAM, DISPUTE) for `GROUND_TRUTH["investigation_category"]` — this is the
   actual outcome.
