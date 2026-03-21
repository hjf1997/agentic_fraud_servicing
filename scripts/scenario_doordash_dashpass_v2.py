"""Scenario: Disputed DoorDash Dash Pass (v2) — no data injection during call.

The copilot has access only to the cardmember profile and historical
transactions via the evidence store. No SYSTEM events inject data during the
call — the agents must gather information purely from the conversation and
what the retrieval agent can fetch from the seeded evidence.

No assumptions are made about whether this is fraud, a dispute, or a scam.
The agents must analyze the evidence and reach their own conclusion.
"""

from datetime import datetime, timedelta, timezone

from agentic_fraud_servicing.gateway.tool_gateway import AuthContext, ToolGateway
from agentic_fraud_servicing.gateway.tools.write_tools import (
    append_evidence_node,
    create_case,
)
from agentic_fraud_servicing.models.case import AuditEntry, Case, TransactionRef
from agentic_fraud_servicing.models.enums import (
    AllegationType,
    AuthMethod,
    CaseStatus,
    EvidenceSourceType,
    TransactionChannel,
)
from agentic_fraud_servicing.models.evidence import (
    Merchant,
    Transaction,
)
from scripts.simulation_data import DisputeAction, Scenario, register_scenario

_NOW = datetime.now(tz=timezone.utc)
_THREE_DAYS_AGO = _NOW - timedelta(days=3)


def _seed_evidence(gateway: ToolGateway, case_id: str) -> None:
    """Seed evidence: historical DoorDash orders, disputed Dash Pass charge, merchant profile."""
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
    """Build the DoorDash Dash Pass v2 scenario (dispute action mid-call)."""
    return Scenario(
        name="doordash_dashpass_v2",
        title="Disputed DoorDash - Dash Pass Charge v2 ($96 CNP, dispute action at turn 6)",
        description=(
            "Copilot has access to historical transactions and merchant\n"
            "profile via the evidence store. At turn 6, the CCP marks the\n"
            "disputed transaction and links it to the cardmember's claim."
        ),
        case_id="case-sim-ddp2-001",
        call_id="call-sim-ddp2-001",
        cm_system_prompt=CM_SYSTEM_PROMPT,
        max_turns=16,
        dispute_actions=[
            DisputeAction(
                trigger_turn=6,
                transaction_node_ids=["txn-dd-disputed"],
                claim_text="Cardmember does not recognize $96 DoorDash - Dash Pass charge.",
            ),
        ],
        seed_evidence_fn=_seed_evidence,
        create_case_fn=_create_case,
    )


register_scenario("doordash_dashpass_v2", build_scenario)
