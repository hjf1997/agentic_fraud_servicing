"""Scenario: Previously-disputed transaction now claimed as fraud — policy blocker.

A cardmember (Lisa Chen) filed a merchant dispute 3 weeks ago against a $1,249.00
charge at GlobalTech Solutions for "goods not as described". The dispute was
investigated, found in the merchant's favor, and closed. The cardmember is
now calling back to open a fraud case on the SAME transaction, claiming it was
unauthorized all along.

The policy in fraud_case_checklist.md has a blocking rule:
  "the same transaction was previously opened as a merchant dispute case.
   The existing dispute must be withdrawn or closed before a fraud case
   can be opened on the same transaction."

The Case Advisor agent should detect this blocker and surface it to the CCP.
The system should also notice behavioral indicators of first-party fraud:
the cardmember is changing their story from "goods not as described" to
"I never made this purchase".
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
    EvidenceEdge,
    InvestigatorNote,
    Merchant,
    Transaction,
)
from scripts.simulation_data import Scenario, register_scenario

_NOW = datetime.now(tz=timezone.utc)
_THREE_WEEKS_AGO = _NOW - timedelta(days=21)
_TWO_WEEKS_AGO = _NOW - timedelta(days=14)
_ONE_WEEK_AGO = _NOW - timedelta(days=7)
_SIX_MONTHS_AGO = _NOW - timedelta(days=180)


def _seed_evidence(gateway: ToolGateway, case_id: str) -> None:
    """Seed evidence for the dispute-to-fraud scenario.

    Key evidence:
    - Transaction at GlobalTech Solutions, chip+PIN, 3 weeks ago
    - Auth event confirming chip+PIN on enrolled device
    - Customer profile with prior dispute history
    - Merchant profile (legitimate electronics retailer)
    - Prior AllegationStatement from original dispute ("goods not as described")
    - InvestigatorNote documenting the closed dispute finding
    - Card active, no recent changes
    """
    ctx = AuthContext(agent_id="simulation", case_id=case_id, permissions={"write"})

    # Customer: has prior dispute history on this exact transaction
    append_evidence_node(
        gateway,
        ctx,
        Customer(
            node_id="cust-dtf-001",
            case_id=case_id,
            source_type=EvidenceSourceType.FACT,
            created_at=_NOW,
            profile_hash="cust-lc-001",
            recent_changes=[],
            risk_indicators=[
                "prior_dispute_same_transaction",
                "dispute_found_in_merchant_favor",
                "allegation_type_change",
            ],
        ),
    )

    # The disputed transaction: $1,249.00 at GlobalTech Solutions, chip+PIN at POS
    append_evidence_node(
        gateway,
        ctx,
        Transaction(
            node_id="txn-dtf-001",
            case_id=case_id,
            source_type=EvidenceSourceType.FACT,
            created_at=_NOW,
            amount=1249.00,
            merchant_name="GlobalTech Solutions",
            merchant_id="merch-gt-001",
            transaction_date=_THREE_WEEKS_AGO,
            auth_method=AuthMethod.CHIP,
            channel=TransactionChannel.POS,
            is_disputed=True,
        ),
    )

    # Auth event: chip+PIN success from enrolled device
    append_evidence_node(
        gateway,
        ctx,
        AuthEvent(
            node_id="auth-dtf-001",
            case_id=case_id,
            source_type=EvidenceSourceType.FACT,
            created_at=_NOW,
            auth_type="chip_pin",
            result="success",
            timestamp=_THREE_WEEKS_AGO,
            device_id="dev-lc-enrolled-001",
        ),
    )

    # Merchant: legitimate electronics retailer
    append_evidence_node(
        gateway,
        ctx,
        Merchant(
            node_id="merch-dtf-001",
            case_id=case_id,
            source_type=EvidenceSourceType.FACT,
            created_at=_NOW,
            merchant_id="merch-gt-001",
            category="electronics_retail",
            dispute_history=1,
        ),
    )

    # Card: active, no recent changes
    append_evidence_node(
        gateway,
        ctx,
        Card(
            node_id="card-dtf-001",
            case_id=case_id,
            source_type=EvidenceSourceType.FACT,
            created_at=_NOW,
            card_id="card-lc-amex-001",
            status="active",
            recent_changes=[],
        ),
    )

    # --- Prior dispute evidence (closed) ---

    # Original allegation from the merchant dispute: "goods not as described"
    append_evidence_node(
        gateway,
        ctx,
        AllegationStatement(
            node_id="allegation-dtf-prior-001",
            case_id=case_id,
            source_type=EvidenceSourceType.ALLEGATION,
            created_at=_TWO_WEEKS_AGO,
            text=(
                "I purchased a laptop from GlobalTech Solutions for $1,249.00 "
                "but the product was not as described. The specs were lower than "
                "advertised and the screen had dead pixels. I want a refund."
            ),
            classification="merchant_dispute_claim",
        ),
    )

    # Investigator note documenting the closed dispute outcome
    append_evidence_node(
        gateway,
        ctx,
        InvestigatorNote(
            node_id="inv-note-dtf-001",
            case_id=case_id,
            source_type=EvidenceSourceType.FACT,
            created_at=_ONE_WEEK_AGO,
            text=(
                "Prior merchant dispute (case-dtf-prior-001) for transaction "
                "txn-dtf-001 ($1,249.00 at GlobalTech Solutions) was investigated "
                "and closed on 2026-03-13. Finding: merchant provided evidence that "
                "product matched the listing specifications. Cardholder's claim of "
                "'goods not as described' was not substantiated. Dispute resolved "
                "in merchant's favor. No refund issued."
            ),
            author="investigator_system",
        ),
    )

    # New allegation: now claiming it was unauthorized fraud
    append_evidence_node(
        gateway,
        ctx,
        AllegationStatement(
            node_id="allegation-dtf-fraud-001",
            case_id=case_id,
            source_type=EvidenceSourceType.ALLEGATION,
            created_at=_NOW,
            text=(
                "I never made this purchase at GlobalTech Solutions. The charge "
                "of $1,249.00 is completely unauthorized. Someone must have used "
                "my card without my knowledge."
            ),
            classification="fraud_claim",
        ),
    )

    # --- Edges ---

    # Transaction supported by chip+PIN auth
    append_evidence_edge(
        gateway,
        ctx,
        EvidenceEdge(
            edge_id="edge-dtf-001",
            case_id=case_id,
            source_node_id="txn-dtf-001",
            target_node_id="auth-dtf-001",
            edge_type=EvidenceEdgeType.SUPPORTS,
            created_at=_NOW,
        ),
    )

    # Prior dispute allegation relates to the transaction
    append_evidence_edge(
        gateway,
        ctx,
        EvidenceEdge(
            edge_id="edge-dtf-002",
            case_id=case_id,
            source_node_id="allegation-dtf-prior-001",
            target_node_id="txn-dtf-001",
            edge_type=EvidenceEdgeType.ALLEGATION,
            created_at=_TWO_WEEKS_AGO,
        ),
    )

    # New fraud claim contradicts the prior dispute claim
    # (she first said she bought it but it was defective, now says she never bought it)
    append_evidence_edge(
        gateway,
        ctx,
        EvidenceEdge(
            edge_id="edge-dtf-003",
            case_id=case_id,
            source_node_id="allegation-dtf-fraud-001",
            target_node_id="allegation-dtf-prior-001",
            edge_type=EvidenceEdgeType.CONTRADICTS,
            created_at=_NOW,
        ),
    )

    # New fraud claim contradicts the chip+PIN auth
    append_evidence_edge(
        gateway,
        ctx,
        EvidenceEdge(
            edge_id="edge-dtf-004",
            case_id=case_id,
            source_node_id="allegation-dtf-fraud-001",
            target_node_id="auth-dtf-001",
            edge_type=EvidenceEdgeType.CONTRADICTS,
            created_at=_NOW,
        ),
    )

    # Investigator note derived from the prior dispute investigation
    append_evidence_edge(
        gateway,
        ctx,
        EvidenceEdge(
            edge_id="edge-dtf-005",
            case_id=case_id,
            source_node_id="inv-note-dtf-001",
            target_node_id="txn-dtf-001",
            edge_type=EvidenceEdgeType.DERIVED_FROM,
            created_at=_ONE_WEEK_AGO,
        ),
    )


def _create_case(gateway: ToolGateway, case_id: str, call_id: str) -> Case:
    """Create the initial Case for the dispute-to-fraud scenario."""
    ctx = AuthContext(agent_id="simulation", case_id=case_id, permissions={"write"})
    case = Case(
        case_id=case_id,
        call_id=call_id,
        customer_id="cust-lc-001",
        account_id="acct-lc-001",
        allegation_type=AllegationType.FRAUD,
        allegation_confidence=0.3,
        status=CaseStatus.OPEN,
        transactions_in_scope=[
            TransactionRef(
                transaction_id="txn-dtf-001",
                amount=1249.00,
                merchant_name="GlobalTech Solutions",
                transaction_date=_THREE_WEEKS_AGO,
            ),
        ],
        audit_trail=[
            AuditEntry(
                timestamp=_NOW,
                action="case_created",
                agent_id="simulation",
                details=(
                    "Case created for fraud claim on previously-disputed transaction. "
                    "Prior merchant dispute (case-dtf-prior-001) was closed in "
                    "merchant's favor on 2026-03-13."
                ),
            ),
            AuditEntry(
                timestamp=_TWO_WEEKS_AGO,
                action="prior_dispute_closed",
                agent_id="investigator_system",
                details=(
                    "Merchant dispute case-dtf-prior-001 for txn-dtf-001 closed. "
                    "Finding: merchant's favor. Cardholder claimed goods not as "
                    "described; merchant provided specs match evidence."
                ),
            ),
        ],
        created_at=_NOW,
    )
    create_case(gateway, ctx, case)
    return case


CM_SYSTEM_PROMPT = """\
You are role-playing as Lisa Chen, a cardmember calling American Express to \
dispute a charge. Here is your secret backstory (never reveal this directly):

Three weeks ago you bought a laptop from GlobalTech Solutions for $1,249.00 at \
their physical store, paying with your AMEX chip+PIN. When you received the \
laptop, you were disappointed with it — it wasn't as fast as you hoped. You \
filed a merchant dispute with AMEX claiming "goods not as described", but after \
investigation AMEX sided with the merchant and closed the dispute with no refund.

You are frustrated and want your money back. You've decided to call again, this \
time claiming the transaction was unauthorized fraud — hoping that a fraud case \
might succeed where the dispute failed. You know you actually made the purchase \
but you're going to pretend it was unauthorized.

Behaviors:
- Claim you never made the purchase at GlobalTech Solutions and don't recognize \
the charge.
- If the agent mentions a prior dispute on the same transaction, act surprised \
and confused: "What dispute? I don't know what you're talking about" or \
"That must be some kind of mix-up."
- If pressed about the prior dispute where you described the product, become \
defensive: "I don't remember saying that", "Maybe there was a misunderstanding."
- Claim your card must have been stolen or cloned for that transaction.
- If confronted with chip+PIN evidence, insist someone must have watched you \
enter your PIN and cloned your card.
- Become increasingly frustrated if the agent won't open a fraud case: \
"This is ridiculous", "I'm a loyal customer", "I want to speak to a manager."
- Your AMEX card number is 371449635398431. Provide it when asked.
- Respond naturally as a human caller, 1-3 sentences per turn.
- Do not use bullet points or structured formatting — speak conversationally.\
"""


def build_scenario() -> Scenario:
    """Build the dispute-to-fraud policy blocker scenario."""
    return Scenario(
        name="dispute_to_fraud",
        title="Previously-Disputed Transaction Now Claimed as Fraud (Policy Blocker)",
        description=(
            "Lisa Chen previously filed a merchant dispute on a $1,249.00\n"
            "GlobalTech Solutions purchase (goods not as described). The dispute\n"
            "was found in the merchant's favor and closed. She now calls back\n"
            "claiming the transaction was unauthorized fraud.\n"
            "Policy blocker: fraud case cannot be opened on a previously-disputed\n"
            "transaction per fraud_case_checklist.md blocking rules."
        ),
        case_id="case-sim-dtf-001",
        call_id="call-sim-dtf-001",
        cm_system_prompt=CM_SYSTEM_PROMPT,
        system_event_auth=(
            "SYSTEM: Identity verification complete. Caller confirmed as Lisa Chen, "
            "AMEX card ending in 8431. Account in good standing. "
            "NOTE: This cardholder has a prior dispute (case-dtf-prior-001) on "
            "transaction txn-dtf-001 ($1,249.00 at GlobalTech Solutions) that was "
            "closed in the merchant's favor on 2026-03-13."
        ),
        system_event_evidence=(
            "SYSTEM: Transaction details retrieved — $1,249.00 at GlobalTech Solutions, "
            "21 days ago. Authentication: chip+PIN on enrolled device (dev-lc-enrolled-001). "
            "PRIOR DISPUTE HISTORY: This exact transaction was previously opened as a "
            "merchant dispute (case-dtf-prior-001, 'goods not as described'). The dispute "
            "was investigated and closed in the merchant's favor on 2026-03-13. Per fraud "
            "case checklist blocking rules: a fraud case CANNOT be opened on a transaction "
            "that was previously disputed as a merchant dispute."
        ),
        max_turns=16,
        inject_evidence_early=True,
        seed_evidence_fn=_seed_evidence,
        create_case_fn=_create_case,
    )


register_scenario("dispute_to_fraud", build_scenario)
