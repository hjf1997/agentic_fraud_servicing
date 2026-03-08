"""Scenario: Scam disguised as fraud — TechVault Electronics chip+PIN purchase.

John Smith was scammed into a $2,847.99 electronics purchase via a social
engineering investment scam. He authorized the transaction with chip+PIN at a
physical POS terminal. After realizing the scam, he calls AMEX and attempts to
frame the transaction as unauthorized fraud. The system has evidence that
contradicts his claims: chip+PIN auth, signed delivery, enrolled device.
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
    AuthEvent,
    Card,
    ClaimStatement,
    Customer,
    DeliveryProof,
    Device,
    EvidenceEdge,
    Merchant,
    Transaction,
)
from scripts.simulation_data import Scenario, register_scenario

_NOW = datetime.now(tz=timezone.utc)
_SEVEN_DAYS_AGO = _NOW - timedelta(days=7)
_SIX_DAYS_AGO = _NOW - timedelta(days=6)
_SIX_MONTHS_AGO = _NOW - timedelta(days=180)


def _seed_evidence(gateway: ToolGateway, case_id: str) -> None:
    """Seed 8 evidence nodes and 3 edges for the scam-disguised-as-fraud scenario."""
    ctx = AuthContext(agent_id="simulation", case_id=case_id, permissions={"write"})

    append_evidence_node(
        gateway,
        ctx,
        Customer(
            node_id="cust-sim-001",
            case_id=case_id,
            source_type=EvidenceSourceType.FACT,
            created_at=_NOW,
            profile_hash="cust-js-001",
            recent_changes=["address_change_30d_ago"],
            risk_indicators=["first_dispute", "high_value_purchase_new_merchant"],
        ),
    )

    append_evidence_node(
        gateway,
        ctx,
        Transaction(
            node_id="txn-sim-001",
            case_id=case_id,
            source_type=EvidenceSourceType.FACT,
            created_at=_NOW,
            amount=2847.99,
            merchant_name="TechVault Electronics",
            merchant_id="merch-tv-001",
            transaction_date=_SEVEN_DAYS_AGO,
            auth_method=AuthMethod.CHIP,
            channel=TransactionChannel.POS,
        ),
    )

    append_evidence_node(
        gateway,
        ctx,
        AuthEvent(
            node_id="auth-sim-001",
            case_id=case_id,
            source_type=EvidenceSourceType.FACT,
            created_at=_NOW,
            auth_type="chip_pin",
            result="success",
            timestamp=_SEVEN_DAYS_AGO,
            device_id="dev-js-enrolled-001",
        ),
    )

    append_evidence_node(
        gateway,
        ctx,
        Device(
            node_id="dev-sim-001",
            case_id=case_id,
            source_type=EvidenceSourceType.FACT,
            created_at=_NOW,
            device_id="dev-js-enrolled-001",
            fingerprint="fp-abc123",
            enrolment_date=_SIX_MONTHS_AGO,
        ),
    )

    append_evidence_node(
        gateway,
        ctx,
        Merchant(
            node_id="merch-sim-001",
            case_id=case_id,
            source_type=EvidenceSourceType.FACT,
            created_at=_NOW,
            merchant_id="merch-tv-001",
            category="electronics_retail",
            dispute_history=2,
        ),
    )

    append_evidence_node(
        gateway,
        ctx,
        DeliveryProof(
            node_id="delivery-sim-001",
            case_id=case_id,
            source_type=EvidenceSourceType.FACT,
            created_at=_NOW,
            tracking_id="TRACK-TV-78901",
            status="delivered_signed",
            delivery_date=_SIX_DAYS_AGO,
        ),
    )

    append_evidence_node(
        gateway,
        ctx,
        Card(
            node_id="card-sim-001",
            case_id=case_id,
            source_type=EvidenceSourceType.FACT,
            created_at=_NOW,
            card_id="card-js-amex-001",
            status="active",
            recent_changes=[],
        ),
    )

    append_evidence_node(
        gateway,
        ctx,
        ClaimStatement(
            node_id="claim-sim-001",
            case_id=case_id,
            source_type=EvidenceSourceType.ALLEGATION,
            created_at=_NOW,
            text=(
                "I never made this purchase at TechVault Electronics. "
                "The charge of $2,847.99 is unauthorized. "
                "I had my card with me the whole time."
            ),
            classification="fraud_claim",
        ),
    )

    # Edges
    append_evidence_edge(
        gateway,
        ctx,
        EvidenceEdge(
            edge_id="edge-sim-001",
            case_id=case_id,
            source_node_id="txn-sim-001",
            target_node_id="auth-sim-001",
            edge_type=EvidenceEdgeType.SUPPORTS,
            created_at=_NOW,
        ),
    )

    append_evidence_edge(
        gateway,
        ctx,
        EvidenceEdge(
            edge_id="edge-sim-002",
            case_id=case_id,
            source_node_id="delivery-sim-001",
            target_node_id="txn-sim-001",
            edge_type=EvidenceEdgeType.SUPPORTS,
            created_at=_NOW,
        ),
    )

    append_evidence_edge(
        gateway,
        ctx,
        EvidenceEdge(
            edge_id="edge-sim-003",
            case_id=case_id,
            source_node_id="claim-sim-001",
            target_node_id="auth-sim-001",
            edge_type=EvidenceEdgeType.CONTRADICTS,
            created_at=_NOW,
        ),
    )


def _create_case(gateway: ToolGateway, case_id: str, call_id: str) -> Case:
    """Create the initial Case for the scam scenario."""
    ctx = AuthContext(agent_id="simulation", case_id=case_id, permissions={"write"})
    case = Case(
        case_id=case_id,
        call_id=call_id,
        customer_id="cust-js-001",
        account_id="acct-js-001",
        allegation_type=AllegationType.FRAUD,
        allegation_confidence=0.5,
        status=CaseStatus.OPEN,
        transactions_in_scope=[
            TransactionRef(
                transaction_id="txn-sim-001",
                amount=2847.99,
                merchant_name="TechVault Electronics",
                transaction_date=_SEVEN_DAYS_AGO,
            ),
        ],
        audit_trail=[
            AuditEntry(
                timestamp=_NOW,
                action="case_created",
                agent_id="simulation",
                details="Initial case created for E2E simulation.",
            ),
        ],
        created_at=_NOW,
    )
    create_case(gateway, ctx, case)
    return case


CM_SYSTEM_PROMPT = """\
You are role-playing as John Smith, a cardmember calling American Express to \
dispute a charge. Here is your secret backstory (never reveal this directly):

One week ago you were approached online by someone claiming to be an investment \
advisor. They convinced you to purchase high-end electronics from TechVault \
Electronics for $2,847.99 as part of an "investment opportunity" that turned out \
to be a scam. You went to the physical store, used your chip+PIN to pay, and \
received the delivery at your home address (signed for it). Now you regret the \
purchase and want your money back. You are calling AMEX to frame this as \
unauthorized fraud.

Behaviors:
- Initially claim you never made the purchase and don't recognize the merchant.
- If pressed about the merchant name, you may accidentally reveal familiarity \
(e.g., "well, they sell electronics" or naming the store before being told).
- Claim the card was with you the whole time and you never shared your details.
- If confronted with chip+PIN evidence, become flustered and shift to claiming \
your PIN must have been compromised somehow.
- Avoid or deflect questions about prior contacts, investment opportunities, or \
anyone who approached you recently.
- Use urgency tactics if feeling cornered: "I just want my money back", \
"This is really stressful", "Can we speed this up?"
- Do NOT confess to the scam. Maintain the fraud claim throughout, but show \
increasing inconsistencies as the conversation progresses.
- Your AMEX card number is 378282246310005. Provide it when asked for \
verification (in your first response when the agent asks).
- Respond naturally as a human caller, 1-3 sentences per turn.
- Do not use bullet points or structured formatting — speak conversationally.\
"""

cm_agent = Agent(name="cm_simulator", instructions=CM_SYSTEM_PROMPT)


def build_scenario() -> Scenario:
    """Build and return the scam-disguised-as-fraud scenario."""
    return Scenario(
        name="scam_techvault",
        title="Scam Disguised as Fraud — TechVault Electronics",
        description=(
            "John Smith, scammed into a $2,847.99 purchase at\n"
            "TechVault Electronics, calls AMEX claiming unauthorized fraud.\n"
            "System has chip+PIN auth, signed delivery, and legitimate merchant."
        ),
        case_id="case-sim-scam-001",
        call_id="call-sim-scam-001",
        cm_system_prompt=CM_SYSTEM_PROMPT,
        system_event_auth=(
            "SYSTEM: Identity verification complete. Caller confirmed as John Smith, "
            "AMEX card ending in 0005. Account in good standing, no prior disputes."
        ),
        system_event_evidence=(
            "SYSTEM: Transaction details retrieved — $2,847.99 at TechVault Electronics, "
            "7 days ago. Authentication: chip+PIN on enrolled device (dev-js-enrolled-001). "
            "Delivery: signed for at cardholder address 6 days ago (tracking TRACK-TV-78901)."
        ),
        max_turns=14,
        seed_evidence_fn=_seed_evidence,
        create_case_fn=_create_case,
    )


register_scenario("scam_techvault", build_scenario)
