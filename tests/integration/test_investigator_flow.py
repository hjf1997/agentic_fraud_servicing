"""Integration tests for the full post-call investigation pipeline.

Tests the end-to-end flow: case + evidence seeding in real SQLite stores,
InvestigatorOrchestrator running all 4 specialist agents (mocked), case status
update, InvestigatorNote write-back, CONTRADICTS edge creation, and trace
logging. Uses real SQLite stores via gateway_factory and mocked LLM calls
via patched run_* functions.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from agentic_fraud_servicing.gateway.tool_gateway import AuthContext
from agentic_fraud_servicing.gateway.tools.write_tools import (
    append_evidence_node,
    create_case,
)
from agentic_fraud_servicing.investigator.case_writer import CasePack
from agentic_fraud_servicing.investigator.merchant_evidence import (
    MerchantAnalysis,
)
from agentic_fraud_servicing.investigator.orchestrator import (
    InvestigatorOrchestrator,
)
from agentic_fraud_servicing.investigator.scam_detector import ScamAnalysis
from agentic_fraud_servicing.investigator.scheme_mapper import (
    SchemeMappingResult,
)
from agentic_fraud_servicing.models.enums import (
    CaseStatus,
    EvidenceSourceType,
    InvestigationCategory,
)
from agentic_fraud_servicing.models.evidence import (
    AllegationStatement,
    Merchant,
    Transaction,
)

# Canned specialist results for mocking

_SCHEME_RESULT = SchemeMappingResult(
    reason_codes=[
        {
            "network": "AMEX",
            "code": "FR2",
            "description": "Fraud Full Recourse",
            "match_confidence": 0.9,
        }
    ],
    primary_reason_code="FR2",
    primary_network="AMEX",
    documentation_gaps=["Signed receipt needed"],
    analysis_summary="Mapped to FR2 for unauthorized transaction.",
)

_MERCHANT_RESULT = MerchantAnalysis(
    normalized_merchants=[{"original": "AMZN*Marketplace", "normalized": "Amazon Marketplace"}],
    conflicts=[],
    consolidated_summary="Single Amazon transaction, no conflicts.",
    merchant_risk_score=0.3,
    recommendations=["Verify delivery status"],
)

_SCAM_RESULT = ScamAnalysis(
    scam_likelihood=0.2,
    manipulation_indicators=[],
    contradictions=[],
    matched_patterns=[],
    analysis_summary="Low scam likelihood.",
)

_SCAM_RESULT_WITH_CONTRADICTIONS = ScamAnalysis(
    scam_likelihood=0.7,
    manipulation_indicators=["Urgency language detected"],
    contradictions=[
        {
            "claim": "Card was stolen",
            "contradicting_evidence": "Chip+PIN auth at local POS",
            "severity": "high",
            "allegation_node_id": "allegation-001",
            "evidence_node_id": "txn-001",
        }
    ],
    matched_patterns=["first-party fraud"],
    analysis_summary="High contradiction between claim and auth evidence.",
)

_CASE_PACK = CasePack(
    case_summary="Investigation of fraud case. Unauthorized charge at Amazon.",
    timeline=[
        {
            "timestamp": "2026-03-01",
            "event_type": "transaction",
            "description": "Charge $487.50",
            "source": "FACT",
        }
    ],
    evidence_list=[
        {
            "node_id": "txn-001",
            "node_type": "TRANSACTION",
            "source_type": "FACT",
            "summary": "$487.50 at Amazon",
        }
    ],
    decision_recommendation={
        "category": "THIRD_PARTY_FRAUD",
        "confidence": 0.85,
        "top_factors": [{"factor": "unauthorized charge", "weight": 0.9}],
        "uncertainties": ["delivery status unknown"],
        "suggested_actions": ["provisional credit"],
        "required_approvals": ["supervisor_review"],
    },
    investigation_notes=["Recommend provisional credit while investigating."],
)

# Patch targets for the 4 specialist run_* functions
_MOD = "agentic_fraud_servicing.investigator.orchestrator"
_PATCH_SCHEME = f"{_MOD}.run_scheme_mapping"
_PATCH_MERCHANT = f"{_MOD}.run_merchant_analysis"
_PATCH_SCAM = f"{_MOD}.run_scam_detection"
_PATCH_CASE_WRITER = f"{_MOD}.run_case_writer"


@pytest.fixture()
def seeded_gateway(sample_case, gateway_factory, tmp_path):
    """Create a gateway with seeded case and evidence nodes.

    Seeds: 1 Case (OPEN/FRAUD), 1 Transaction (FACT), 1 Merchant (FACT),
    1 AllegationStatement (ALLEGATION).
    """
    gateway = gateway_factory(tmp_path)
    ctx = AuthContext(
        agent_id="test",
        case_id=sample_case.case_id,
        permissions={"read", "write"},
    )

    # Seed the case
    create_case(gateway, ctx, sample_case)

    now = datetime.now(timezone.utc)

    # Seed evidence nodes
    txn = Transaction(
        node_id="txn-001",
        case_id=sample_case.case_id,
        source_type=EvidenceSourceType.FACT,
        created_at=now,
        amount=487.50,
        merchant_name="AMZN*Marketplace",
        transaction_date=now,
    )
    merchant = Merchant(
        node_id="merch-001",
        case_id=sample_case.case_id,
        source_type=EvidenceSourceType.FACT,
        created_at=now,
        merchant_id="amzn-123",
        category="retail",
    )
    allegation = AllegationStatement(
        node_id="allegation-001",
        case_id=sample_case.case_id,
        source_type=EvidenceSourceType.ALLEGATION,
        created_at=now,
        text="I did not make this purchase at Amazon.",
    )

    for node in [txn, merchant, allegation]:
        append_evidence_node(gateway, ctx, node)

    return gateway


@pytest.fixture()
def _mock_specialists():
    """Patch all 4 specialist run_* functions with canned results."""
    with (
        patch(
            _PATCH_SCHEME,
            new_callable=AsyncMock,
            return_value=_SCHEME_RESULT,
        ) as m_scheme,
        patch(
            _PATCH_MERCHANT,
            new_callable=AsyncMock,
            return_value=_MERCHANT_RESULT,
        ) as m_merchant,
        patch(
            _PATCH_SCAM,
            new_callable=AsyncMock,
            return_value=_SCAM_RESULT,
        ) as m_scam,
        patch(
            _PATCH_CASE_WRITER,
            new_callable=AsyncMock,
            return_value=_CASE_PACK,
        ) as m_writer,
    ):
        yield {
            "scheme": m_scheme,
            "merchant": m_merchant,
            "scam": m_scam,
            "writer": m_writer,
        }


# -- Tests --


class TestInvestigateReturns:
    """Verify investigate() returns a CasePack and updates state."""

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_returns_case_pack(self, seeded_gateway, sample_case, mock_model_provider):
        """investigate() should return a CasePack instance."""
        orch = InvestigatorOrchestrator(seeded_gateway, mock_model_provider)
        result = await orch.investigate(sample_case.case_id)
        assert isinstance(result, CasePack)
        assert result.case_summary == _CASE_PACK.case_summary

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_case_not_found_raises(self, seeded_gateway, mock_model_provider):
        """investigate() should raise RuntimeError for nonexistent case."""
        orch = InvestigatorOrchestrator(seeded_gateway, mock_model_provider)
        with pytest.raises(RuntimeError, match="Case not found"):
            await orch.investigate("nonexistent-case")


class TestDecisionCategory:
    """Verify decision recommendation uses InvestigationCategory values."""

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_case_pack_uses_investigation_category(
        self, seeded_gateway, sample_case, mock_model_provider
    ):
        """CasePack decision_recommendation.category should be a valid InvestigationCategory."""
        orch = InvestigatorOrchestrator(seeded_gateway, mock_model_provider)
        result = await orch.investigate(sample_case.case_id)

        category = result.decision_recommendation["category"]
        valid_categories = {c.value for c in InvestigationCategory}
        assert category in valid_categories


class TestCaseStatusUpdate:
    """Verify case status is updated to INVESTIGATING in the store."""

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_status_updated_to_investigating(
        self, seeded_gateway, sample_case, mock_model_provider
    ):
        """Case status should be INVESTIGATING after investigation."""
        orch = InvestigatorOrchestrator(seeded_gateway, mock_model_provider)
        await orch.investigate(sample_case.case_id)

        updated = seeded_gateway.case_store.get_case(sample_case.case_id)
        assert updated.status == CaseStatus.INVESTIGATING


class TestEvidenceWriteback:
    """Verify InvestigatorNote and edges are written back to the store."""

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_investigator_note_added(self, seeded_gateway, sample_case, mock_model_provider):
        """An InvestigatorNote should be written to the evidence store."""
        orch = InvestigatorOrchestrator(seeded_gateway, mock_model_provider)
        await orch.investigate(sample_case.case_id)

        nodes = seeded_gateway.evidence_store.get_nodes_by_case(sample_case.case_id)
        note_nodes = [n for n in nodes if n.get("node_type") == "INVESTIGATOR_NOTE"]
        assert len(note_nodes) >= 1
        assert note_nodes[0]["author"] == "investigator"
        assert note_nodes[0]["text"] == _CASE_PACK.case_summary

    async def test_contradicts_edges_created(
        self, seeded_gateway, sample_case, mock_model_provider
    ):
        """CONTRADICTS edges should be added from scam contradictions."""
        with (
            patch(
                _PATCH_SCHEME,
                new_callable=AsyncMock,
                return_value=_SCHEME_RESULT,
            ),
            patch(
                _PATCH_MERCHANT,
                new_callable=AsyncMock,
                return_value=_MERCHANT_RESULT,
            ),
            patch(
                _PATCH_SCAM,
                new_callable=AsyncMock,
                return_value=_SCAM_RESULT_WITH_CONTRADICTIONS,
            ),
            patch(
                _PATCH_CASE_WRITER,
                new_callable=AsyncMock,
                return_value=_CASE_PACK,
            ),
        ):
            orch = InvestigatorOrchestrator(seeded_gateway, mock_model_provider)
            await orch.investigate(sample_case.case_id)

        edges = seeded_gateway.evidence_store.get_edges_by_case(sample_case.case_id)
        contradicts = [e for e in edges if e.get("edge_type") == "CONTRADICTS"]
        assert len(contradicts) == 1
        assert contradicts[0]["source_node_id"] == "allegation-001"
        assert contradicts[0]["target_node_id"] == "txn-001"


class TestTraceLogging:
    """Verify trace store records invocations."""

    @pytest.mark.usefixtures("_mock_specialists")
    async def test_trace_entries_logged(self, seeded_gateway, sample_case, mock_model_provider):
        """Trace store should have entries after investigation."""
        orch = InvestigatorOrchestrator(seeded_gateway, mock_model_provider)
        await orch.investigate(sample_case.case_id)

        traces = seeded_gateway.trace_store.get_traces_by_case(sample_case.case_id)
        assert len(traces) >= 1
        actions = [t["action"] for t in traces]
        # At minimum: get_case, get_evidence, and write-back calls
        assert "get_case" in actions
        assert "get_evidence" in actions


class TestSpecialistInvocation:
    """Verify all 4 specialists are called with correct arguments."""

    async def test_all_specialists_called(self, seeded_gateway, sample_case, mock_model_provider):
        """All 4 specialist run_* functions should be invoked."""
        with (
            patch(
                _PATCH_SCHEME,
                new_callable=AsyncMock,
                return_value=_SCHEME_RESULT,
            ) as m_scheme,
            patch(
                _PATCH_MERCHANT,
                new_callable=AsyncMock,
                return_value=_MERCHANT_RESULT,
            ) as m_merchant,
            patch(
                _PATCH_SCAM,
                new_callable=AsyncMock,
                return_value=_SCAM_RESULT,
            ) as m_scam,
            patch(
                _PATCH_CASE_WRITER,
                new_callable=AsyncMock,
                return_value=_CASE_PACK,
            ) as m_writer,
        ):
            orch = InvestigatorOrchestrator(seeded_gateway, mock_model_provider)
            await orch.investigate(sample_case.case_id)

            m_scheme.assert_awaited_once()
            m_merchant.assert_awaited_once()
            m_scam.assert_awaited_once()
            m_writer.assert_awaited_once()


class TestLiveTest:
    """Live integration test requiring real Bedrock credentials."""

    @pytest.mark.live
    async def test_investigator_flow_live(self, sample_case, gateway_factory, tmp_path):
        """End-to-end investigator test with real LLM provider.

        Requires AWS Bedrock credentials configured in .env
        (LLM_PROVIDER=bedrock, AWS_PROFILE, AWS_REGION,
        AWS_BEDROCK_MODEL_ID). Skipped by default — run with: pytest -m live

        Seeds a case with evidence, creates a real provider, and runs the
        full investigation pipeline through the InvestigatorOrchestrator.
        """
        from agentic_fraud_servicing.gateway.tools.write_tools import (
            append_evidence_node as write_node,
        )
        from agentic_fraud_servicing.gateway.tools.write_tools import (
            create_case as write_case,
        )
        from agentic_fraud_servicing.ui.helpers import create_provider

        try:
            provider = create_provider()
        except Exception:
            pytest.skip("LLM provider not configured — skipping live test")

        gateway = gateway_factory(tmp_path)
        ctx = AuthContext(
            agent_id="test",
            case_id=sample_case.case_id,
            permissions={"read", "write"},
        )
        write_case(gateway, ctx, sample_case)

        now = datetime.now(timezone.utc)
        txn = Transaction(
            node_id="txn-live-001",
            case_id=sample_case.case_id,
            source_type=EvidenceSourceType.FACT,
            created_at=now,
            amount=487.50,
            merchant_name="AMZN*Marketplace",
            transaction_date=now,
        )
        write_node(gateway, ctx, txn)

        orch = InvestigatorOrchestrator(gateway, provider)
        result = await orch.investigate(sample_case.case_id)
        assert isinstance(result, CasePack)
