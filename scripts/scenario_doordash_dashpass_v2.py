"""Scenario: Disputed DoorDash Dash Pass (v2) — transaction info injected early.

Variant of doordash_dashpass where all transaction history is provided to the
copilot via a SYSTEM event immediately after identity verification (turn 4),
rather than waiting until turn 10. This gives the CCP full context early in
the call so the copilot can suggest more targeted questions.

No assumptions are made about whether this is fraud, a dispute, or a scam.
The agents must analyze the evidence and reach their own conclusion.
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
    ClaimStatement,
    EvidenceEdge,
    Merchant,
    Transaction,
)
from scripts.simulation_data import Scenario, register_scenario

_NOW = datetime.now(tz=timezone.utc)
_THREE_DAYS_AGO = _NOW - timedelta(days=3)


def _seed_evidence(gateway: ToolGateway, case_id: str) -> None:
    """Seed evidence: disputed Dash Pass charge, historical DoorDash orders, claim."""
    ctx = AuthContext(agent_id="simulation", case_id=case_id, permissions={"write"})

    # -- Historical DoorDash food delivery orders (various amounts, none $96) --
    historical = [
        ("txn-dd-hist-001", 27.43, _NOW - timedelta(days=14)),
        ("txn-dd-hist-002", 34.15, _NOW - timedelta(days=45)),
        ("txn-dd-hist-003", 18.99, _NOW - timedelta(days=78)),
        ("txn-dd-hist-004", 42.60, _NOW - timedelta(days=120)),
        ("txn-dd-hist-005", 31.25, _NOW - timedelta(days=200)),
    ]
    for node_id, amount, txn_date in historical:
        append_evidence_node(
            gateway,
            ctx,
            Transaction(
                node_id=node_id,
                case_id=case_id,
                source_type=EvidenceSourceType.FACT,
                created_at=_NOW,
                amount=amount,
                merchant_name="DoorDash",
                merchant_id="merch-dd-001",
                transaction_date=txn_date,
                auth_method=AuthMethod.CNP,
                channel=TransactionChannel.ONLINE,
            ),
        )

    # -- Disputed transaction: $96 DoorDash - Dash Pass, 3 days ago, CNP --
    append_evidence_node(
        gateway,
        ctx,
        Transaction(
            node_id="txn-dd-disputed",
            case_id=case_id,
            source_type=EvidenceSourceType.FACT,
            created_at=_NOW,
            amount=96.00,
            merchant_name="DoorDash - Dash Pass",
            merchant_id="merch-dd-001",
            transaction_date=_THREE_DAYS_AGO,
            auth_method=AuthMethod.CNP,
            channel=TransactionChannel.ONLINE,
        ),
    )

    # -- Merchant: DoorDash --
    append_evidence_node(
        gateway,
        ctx,
        Merchant(
            node_id="merch-dd-001",
            case_id=case_id,
            source_type=EvidenceSourceType.FACT,
            created_at=_NOW,
            merchant_id="merch-dd-001",
            category="food_delivery",
            dispute_history=0,
        ),
    )

    # -- Claim statement (ALLEGATION) --
    append_evidence_node(
        gateway,
        ctx,
        ClaimStatement(
            node_id="claim-dd-001",
            case_id=case_id,
            source_type=EvidenceSourceType.ALLEGATION,
            created_at=_NOW,
            text="I did not authorize this $96 DoorDash - Dash Pass charge.",
            classification="dispute_claim",
        ),
    )

    # -- Edge: claim relates to the disputed transaction --
    append_evidence_edge(
        gateway,
        ctx,
        EvidenceEdge(
            edge_id="edge-dd-001",
            case_id=case_id,
            source_node_id="claim-dd-001",
            target_node_id="txn-dd-disputed",
            edge_type=EvidenceEdgeType.ALLEGATION,
            created_at=_NOW,
        ),
    )


def _create_case(gateway: ToolGateway, case_id: str, call_id: str) -> Case:
    """Create the initial Case for the DoorDash Dash Pass v2 scenario."""
    ctx = AuthContext(agent_id="simulation", case_id=case_id, permissions={"write"})
    case = Case(
        case_id=case_id,
        call_id=call_id,
        customer_id="cust-dd-002",
        account_id="acct-dd-002",
        allegation_type=AllegationType.FRAUD,
        allegation_confidence=0.5,
        status=CaseStatus.OPEN,
        transactions_in_scope=[
            TransactionRef(
                transaction_id="txn-dd-disputed",
                amount=96.00,
                merchant_name="DoorDash - Dash Pass",
                transaction_date=_THREE_DAYS_AGO,
            ),
        ],
        audit_trail=[
            AuditEntry(
                timestamp=_NOW,
                action="case_created",
                agent_id="simulation",
                details="Case created for disputed DoorDash - Dash Pass charge (v2).",
            ),
        ],
        created_at=_NOW,
    )
    create_case(gateway, ctx, case)
    return case


CM_SYSTEM_PROMPT = """\
You are role-playing as a cardmember calling American Express to dispute a \
charge on your account. Here is what you know:

- A few days ago, a $96 charge from "DoorDash - Dash Pass" appeared on your \
AMEX statement.
- You use DoorDash to order food occasionally, but you do not recognize this \
specific charge. Your DoorDash orders are usually for different amounts.
- You want to dispute this charge.
- Your AMEX card number is 374245455400126. Provide it when asked.

Behaviors:
- Answer the agent's questions naturally and honestly based on what you know.
- If you don't know something, say so — don't make things up.
- You are calling because you want this charge investigated and resolved.
- Respond naturally as a human caller, 1-3 sentences per turn.
- Do not use bullet points or structured formatting — speak conversationally.\
"""


def build_scenario() -> Scenario:
    """Build the DoorDash Dash Pass v2 scenario (transaction info injected early)."""
    return Scenario(
        name="doordash_dashpass_v2",
        title="Disputed DoorDash - Dash Pass Charge v2 ($96 CNP, early transaction injection)",
        description=(
            "Variant of doordash_dashpass where all transaction history is\n"
            "injected as a SYSTEM event right after identity verification,\n"
            "giving the copilot full context early in the call."
        ),
        case_id="case-sim-ddp2-001",
        call_id="call-sim-ddp2-001",
        cm_system_prompt=CM_SYSTEM_PROMPT,
        # Identity verification — same as v1
        system_event_auth=(
            "SYSTEM: Identity verification complete. Caller confirmed as cardholder, "
            "AMEX card ending in 0126. Account in good standing. No prior disputes."
        ),
        # Transaction details — same content as v1 but will be injected at turn 4
        # (right after auth) instead of at turn 10
        system_event_evidence=(
            "SYSTEM: Transaction details retrieved — $96.00 DoorDash - Dash Pass, "
            "3 days ago, card-not-present (CNP). Historical DoorDash transactions: "
            "$27.43 (14 days ago), $34.15 (45 days ago), $18.99 (78 days ago), "
            "$42.60 (120 days ago), $31.25 (200 days ago). All prior transactions "
            "are regular DoorDash food delivery orders — none match the $96 Dash Pass "
            "amount or description."
        ),
        max_turns=14,
        inject_evidence_early=True,
        seed_evidence_fn=_seed_evidence,
        create_case_fn=_create_case,
    )


register_scenario("doordash_dashpass_v2", build_scenario)
