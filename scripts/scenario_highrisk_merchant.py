"""Scenario: Disputed large transactions at a high-risk merchant.

A loyal, long-term cardmember disputes a series of large CNP transactions
at a merchant they have no prior history with. The merchant is labeled
as high-risk.

No assumptions are made about whether this is genuine third-party fraud,
friendly fraud, or something else. The agents must analyze the evidence
and reach their own conclusion.
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
    Card,
    Customer,
    EvidenceEdge,
    Merchant,
    Transaction,
)
from scripts.simulation_data import Scenario, register_scenario

_NOW = datetime.now(tz=timezone.utc)


def _seed_evidence(gateway: ToolGateway, case_id: str) -> None:
    """Seed evidence: disputed transactions, customer profile, merchant, claim."""
    ctx = AuthContext(agent_id="simulation", case_id=case_id, permissions={"write"})

    # -- Customer profile: loyal, long-term relationship --
    append_evidence_node(
        gateway,
        ctx,
        Customer(
            node_id="cust-hr-001",
            case_id=case_id,
            source_type=EvidenceSourceType.FACT,
            created_at=_NOW,
            profile_hash="cust-hr-001",
            recent_changes=[],
            risk_indicators=[],
        ),
    )

    # -- Card: active, no recent changes --
    append_evidence_node(
        gateway,
        ctx,
        Card(
            node_id="card-hr-001",
            case_id=case_id,
            source_type=EvidenceSourceType.FACT,
            created_at=_NOW,
            card_id="card-hr-amex-001",
            status="active",
            recent_changes=[],
        ),
    )

    # -- Merchant: high-risk, elevated dispute history --
    append_evidence_node(
        gateway,
        ctx,
        Merchant(
            node_id="merch-hr-001",
            case_id=case_id,
            source_type=EvidenceSourceType.FACT,
            created_at=_NOW,
            merchant_id="merch-hr-001",
            category="online_electronics",
            dispute_history=47,
        ),
    )

    # -- Disputed transactions: 3 large CNP charges over 4 days --
    disputed_txns = [
        ("txn-hr-001", 1249.99, _NOW - timedelta(days=6)),
        ("txn-hr-002", 2199.00, _NOW - timedelta(days=4)),
        ("txn-hr-003", 879.50, _NOW - timedelta(days=3)),
    ]
    for node_id, amount, txn_date in disputed_txns:
        append_evidence_node(
            gateway,
            ctx,
            Transaction(
                node_id=node_id,
                case_id=case_id,
                source_type=EvidenceSourceType.FACT,
                created_at=_NOW,
                amount=amount,
                merchant_name="GlobalTech Direct",
                merchant_id="merch-hr-001",
                transaction_date=txn_date,
                auth_method=AuthMethod.CNP,
                channel=TransactionChannel.ONLINE,
            ),
        )

    # -- Allegation statement (ALLEGATION) --
    append_evidence_node(
        gateway,
        ctx,
        AllegationStatement(
            node_id="allegation-hr-001",
            case_id=case_id,
            source_type=EvidenceSourceType.ALLEGATION,
            created_at=_NOW,
            text=(
                "I did not make these purchases at GlobalTech Direct. "
                "I have never heard of this merchant. There are three "
                "charges totaling over $4,300 that I need to dispute."
            ),
            classification="fraud_claim",
        ),
    )

    # -- Edges: allegation relates to each disputed transaction --
    for i, (txn_id, _, _) in enumerate(disputed_txns, start=1):
        append_evidence_edge(
            gateway,
            ctx,
            EvidenceEdge(
                edge_id=f"edge-hr-{i:03d}",
                case_id=case_id,
                source_node_id="allegation-hr-001",
                target_node_id=txn_id,
                edge_type=EvidenceEdgeType.ALLEGATION,
                created_at=_NOW,
            ),
        )


def _create_case(gateway: ToolGateway, case_id: str, call_id: str) -> Case:
    """Create the initial Case for the high-risk merchant dispute."""
    ctx = AuthContext(agent_id="simulation", case_id=case_id, permissions={"write"})
    case = Case(
        case_id=case_id,
        call_id=call_id,
        customer_id="cust-hr-001",
        account_id="acct-hr-001",
        allegation_type=AllegationType.FRAUD,
        allegation_confidence=0.5,
        status=CaseStatus.OPEN,
        transactions_in_scope=[
            TransactionRef(
                transaction_id="txn-hr-001",
                amount=1249.99,
                merchant_name="GlobalTech Direct",
                transaction_date=_NOW - timedelta(days=6),
            ),
            TransactionRef(
                transaction_id="txn-hr-002",
                amount=2199.00,
                merchant_name="GlobalTech Direct",
                transaction_date=_NOW - timedelta(days=4),
            ),
            TransactionRef(
                transaction_id="txn-hr-003",
                amount=879.50,
                merchant_name="GlobalTech Direct",
                transaction_date=_NOW - timedelta(days=3),
            ),
        ],
        audit_trail=[
            AuditEntry(
                timestamp=_NOW,
                action="case_created",
                agent_id="simulation",
                details=(
                    "Case created for 3 disputed transactions at "
                    "GlobalTech Direct totaling $4,328.49."
                ),
            ),
        ],
        created_at=_NOW,
    )
    create_case(gateway, ctx, case)
    return case


CM_SYSTEM_PROMPT = """\
You are role-playing as a cardmember calling American Express to dispute \
charges on your account. Here is what you know:

- You noticed three large charges from "GlobalTech Direct" on your AMEX \
statement: $1,249.99, $2,199.00, and $879.50.
- You have never heard of GlobalTech Direct and did not make these purchases.
- You have been an AMEX cardmember for over 12 years and have never had \
a dispute before.
- Your AMEX card number is 379354508162306. Provide it when asked.

Behaviors:
- Answer the agent's questions naturally and honestly based on what you know.
- If you don't know something, say so — don't make things up.
- You are concerned and want these charges investigated and resolved.
- Respond naturally as a human caller, 1-3 sentences per turn.
- Do not use bullet points or structured formatting — speak conversationally.\
"""


def build_scenario() -> Scenario:
    """Build and return the high-risk merchant dispute scenario."""
    return Scenario(
        name="highrisk_merchant",
        title=(
            "Disputed Large Transactions at High-Risk Merchant (Loyal Customer, No Prior History)"
        ),
        description=(
            "Loyal, long-term cardmember disputes 3 large CNP transactions\n"
            "at GlobalTech Direct ($1,249.99 + $2,199.00 + $879.50 = $4,328.49).\n"
            "No prior history with this merchant. Merchant is high-risk\n"
            "(47 prior disputes). All transactions are card-not-present."
        ),
        case_id="case-sim-hr-001",
        call_id="call-sim-hr-001",
        cm_system_prompt=CM_SYSTEM_PROMPT,
        system_event_auth=(
            "SYSTEM: Identity verification complete. Caller confirmed as "
            "cardholder, AMEX card ending in 2306. Account in good standing. "
            "Customer since 2014 (12+ years). No prior disputes or fraud claims."
        ),
        system_event_evidence=(
            "SYSTEM: Transaction details retrieved — 3 charges at GlobalTech "
            "Direct: $1,249.99 (6 days ago), $2,199.00 (4 days ago), $879.50 "
            "(3 days ago). All card-not-present (CNP). No prior transactions "
            "with this merchant on this account. Merchant risk profile: "
            "HIGH-RISK — 47 disputes in the past 12 months, online electronics "
            "category."
        ),
        max_turns=14,
        seed_evidence_fn=_seed_evidence,
        create_case_fn=_create_case,
    )


register_scenario("highrisk_merchant", build_scenario)
