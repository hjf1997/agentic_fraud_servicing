"""Scenario: Disputed DoorDash Dash Pass transaction.

A cardmember disputes a $96 DoorDash - Dash Pass charge from a few days ago.
Transaction records show the same recurring $96 DoorDash - Dash Pass charge
yearly over the past 5 years. All transactions are card-not-present (CNP).

No assumptions are made about whether this is fraud, a dispute, or a scam.
The agents must analyze the evidence and reach their own conclusion.
"""

from datetime import datetime, timedelta, timezone

from agents import Agent

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
    EvidenceEdge,
    Merchant,
    Transaction,
)
from scripts.simulation_data import Scenario, register_scenario

_NOW = datetime.now(tz=timezone.utc)
_THREE_DAYS_AGO = _NOW - timedelta(days=3)
_ONE_YEAR_AGO = _NOW - timedelta(days=365)
_TWO_YEARS_AGO = _NOW - timedelta(days=730)
_THREE_YEARS_AGO = _NOW - timedelta(days=1095)
_FOUR_YEARS_AGO = _NOW - timedelta(days=1460)
_FIVE_YEARS_AGO = _NOW - timedelta(days=1825)


def _seed_evidence(gateway: ToolGateway, case_id: str) -> None:
    """Seed evidence: disputed transaction, 5 historical transactions, merchant, claim."""
    ctx = AuthContext(agent_id="simulation", case_id=case_id, permissions={"write"})

    # -- 5 historical DoorDash - Dash Pass transactions (one per year, $96 each, CNP) --
    historical_dates = [
        _FIVE_YEARS_AGO,
        _FOUR_YEARS_AGO,
        _THREE_YEARS_AGO,
        _TWO_YEARS_AGO,
        _ONE_YEAR_AGO,
    ]
    for i, txn_date in enumerate(historical_dates, start=1):
        append_evidence_node(
            gateway,
            ctx,
            Transaction(
                node_id=f"txn-dd-hist-{i:03d}",
                case_id=case_id,
                source_type=EvidenceSourceType.FACT,
                created_at=_NOW,
                amount=96.00,
                merchant_name="DoorDash - Dash Pass",
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

    # -- Allegation statement (ALLEGATION) --
    append_evidence_node(
        gateway,
        ctx,
        AllegationStatement(
            node_id="allegation-dd-001",
            case_id=case_id,
            source_type=EvidenceSourceType.ALLEGATION,
            created_at=_NOW,
            text="I did not authorize this $96 DoorDash - Dash Pass charge.",
            classification="dispute_claim",
        ),
    )

    # -- Edge: allegation relates to the disputed transaction --
    append_evidence_edge(
        gateway,
        ctx,
        EvidenceEdge(
            edge_id="edge-dd-001",
            case_id=case_id,
            source_node_id="allegation-dd-001",
            target_node_id="txn-dd-disputed",
            edge_type=EvidenceEdgeType.ALLEGATION,
            created_at=_NOW,
        ),
    )

    # -- Edge: historical pattern links last legitimate charge to disputed one --
    append_evidence_edge(
        gateway,
        ctx,
        EvidenceEdge(
            edge_id="edge-dd-002",
            case_id=case_id,
            source_node_id="txn-dd-hist-005",
            target_node_id="txn-dd-disputed",
            edge_type=EvidenceEdgeType.SUPPORTS,
            created_at=_NOW,
        ),
    )


def _create_case(gateway: ToolGateway, case_id: str, call_id: str) -> Case:
    """Create the initial Case for the DoorDash dispute scenario."""
    ctx = AuthContext(agent_id="simulation", case_id=case_id, permissions={"write"})
    case = Case(
        case_id=case_id,
        call_id=call_id,
        customer_id="cust-dd-001",
        account_id="acct-dd-001",
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
                details="Case created for disputed DoorDash - Dash Pass charge.",
            ),
        ],
        created_at=_NOW,
    )
    create_case(gateway, ctx, case)
    return case


# The CM agent is simply a cardmember who wants to dispute this charge.
# No assumptions about honesty, backstory, or whether this is actually fraud.
CM_SYSTEM_PROMPT = """\
You are role-playing as a cardmember calling American Express to dispute a \
charge on your account. Here is what you know:

- A few days ago, a $96 charge from "DoorDash - Dash Pass" appeared on your \
AMEX statement.
- You want to dispute this charge.
- Your AMEX card number is 374245455400126. Provide it when asked.

Behaviors:
- Answer the agent's questions naturally and honestly based on what you know.
- If you don't know something, say so — don't make things up.
- You are calling because you want this charge investigated and resolved.
- Respond naturally as a human caller, 1-3 sentences per turn.
- Do not use bullet points or structured formatting — speak conversationally.\
"""

cm_agent = Agent(name="cm_simulator", instructions=CM_SYSTEM_PROMPT)


def build_scenario() -> Scenario:
    """Build and return the DoorDash dispute scenario."""
    return Scenario(
        name="doordash_fraud",
        title="Disputed DoorDash - Dash Pass Charge ($96 CNP)",
        description=(
            "Cardmember disputes a $96 DoorDash - Dash Pass charge from days ago.\n"
            "Transaction records show the same $96 recurring charge yearly for\n"
            "the past 5 years. All transactions are card-not-present (CNP)."
        ),
        case_id="case-sim-dd-001",
        call_id="call-sim-dd-001",
        cm_system_prompt=CM_SYSTEM_PROMPT,
        system_event_auth=(
            "SYSTEM: Identity verification complete. Caller confirmed as cardholder, "
            "AMEX card ending in 0126. Account in good standing. No prior disputes."
        ),
        system_event_evidence=(
            "SYSTEM: Transaction details retrieved — $96.00 DoorDash - Dash Pass, "
            "3 days ago, card-not-present (CNP). Historical pattern: 5 prior "
            "DoorDash - Dash Pass charges of $96.00/year over the past 5 years, "
            "all CNP."
        ),
        max_turns=14,
        seed_evidence_fn=_seed_evidence,
        create_case_fn=_create_case,
    )


register_scenario("doordash_fraud", build_scenario)
