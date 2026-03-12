"""Tests for the investigator orchestrator module.

All specialist agent run_* functions are mocked to return canned results.
Tests verify case/evidence reading, specialist invocation, result write-back,
graceful degradation on specialist failure, and trace logging.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_fraud_servicing.investigator.case_writer import CasePack
from agentic_fraud_servicing.investigator.orchestrator import InvestigatorOrchestrator
from agentic_fraud_servicing.models.case import Case
from agentic_fraud_servicing.models.enums import (
    AllegationType,
    CaseStatus,
    EvidenceEdgeType,
    EvidenceSourceType,
)

# -- Fixtures --

_NOW = datetime.now(timezone.utc)


def _make_case(case_id: str = "case-001") -> Case:
    """Create a Case instance for testing."""
    return Case(
        case_id=case_id,
        call_id="call-001",
        customer_id="cust-001",
        account_id="acct-001",
        allegation_type=AllegationType.FRAUD,
        allegation_confidence=0.8,
        status=CaseStatus.OPEN,
        created_at=_NOW,
    )


def _mock_scheme_result():
    """Create a mock SchemeMappingResult."""
    result = MagicMock()
    result.model_dump_json.return_value = '{"primary_reason_code": "FR2"}'
    result.reason_codes = [{"code": "FR2", "network": "AMEX"}]
    return result


def _mock_merchant_result():
    """Create a mock MerchantAnalysis."""
    result = MagicMock()
    result.model_dump_json.return_value = '{"merchant_risk_score": 0.3}'
    return result


def _mock_scam_result(contradictions=None):
    """Create a mock ScamAnalysis."""
    result = MagicMock()
    result.model_dump_json.return_value = '{"scam_likelihood": 0.2}'
    result.contradictions = contradictions or []
    return result


def _mock_case_pack():
    """Create a CasePack for testing."""
    return CasePack(
        case_summary="Investigation summary for case-001.",
        timeline=[
            {
                "timestamp": "2026-01-01",
                "event_type": "transaction",
                "description": "Purchase",
                "source": "FACT",
            }
        ],
        evidence_list=[
            {
                "node_id": "n1",
                "node_type": "TRANSACTION",
                "source_type": "FACT",
                "summary": "Purchase $100",
            }
        ],
        decision_recommendation={"category": "fraud", "confidence": 0.85},
        investigation_notes=["Note 1"],
    )


def _make_evidence_nodes():
    """Create sample evidence node dicts."""
    return [
        {"node_id": "n1", "node_type": "TRANSACTION", "source_type": "FACT", "amount": 100.0},
        {"node_id": "n2", "node_type": "MERCHANT", "source_type": "FACT", "name": "Store A"},
        {
            "node_id": "n3",
            "node_type": "CLAIM_STATEMENT",
            "source_type": "ALLEGATION",
            "text": "I didn't buy this",
            "entities": {},
        },
    ]


def _make_evidence_nodes_with_related_claims():
    """Create evidence nodes with 2 related claims sharing a merchant_name."""
    return [
        {"node_id": "n1", "node_type": "TRANSACTION", "source_type": "FACT", "amount": 100.0},
        {
            "node_id": "claim-001",
            "node_type": "CLAIM_STATEMENT",
            "source_type": "ALLEGATION",
            "text": "I didn't buy this at TechVault",
            "entities": {"merchant_name": "TechVault", "amount": 2847.99},
        },
        {
            "node_id": "claim-002",
            "node_type": "CLAIM_STATEMENT",
            "source_type": "ALLEGATION",
            "text": "That TechVault purchase was online",
            "entities": {"merchant_name": "TechVault", "channel": "online"},
        },
        {
            "node_id": "claim-003",
            "node_type": "CLAIM_STATEMENT",
            "source_type": "ALLEGATION",
            "text": "I also saw a charge from Amazon",
            "entities": {"merchant_name": "Amazon", "amount": 50.0},
        },
    ]


def _make_gateway():
    """Create a mock ToolGateway with working stores."""
    gateway = MagicMock()
    gateway.case_store.get_case.return_value = _make_case()
    gateway.evidence_store.get_nodes_by_case.return_value = _make_evidence_nodes()
    gateway.evidence_store.get_edges_by_case.return_value = []
    return gateway


def _make_orchestrator(gateway=None) -> InvestigatorOrchestrator:
    """Create an InvestigatorOrchestrator with mock gateway and provider."""
    if gateway is None:
        gateway = _make_gateway()
    model_provider = MagicMock()
    return InvestigatorOrchestrator(gateway=gateway, model_provider=model_provider)


# Patch paths for the 4 specialist run_* functions
_SCHEME_PATCH = "agentic_fraud_servicing.investigator.orchestrator.run_scheme_mapping"
_MERCHANT_PATCH = "agentic_fraud_servicing.investigator.orchestrator.run_merchant_analysis"
_SCAM_PATCH = "agentic_fraud_servicing.investigator.orchestrator.run_scam_detection"
_WRITER_PATCH = "agentic_fraud_servicing.investigator.orchestrator.run_case_writer"
# Write tools patches
_APPEND_NODE_PATCH = "agentic_fraud_servicing.investigator.orchestrator.append_evidence_node"
_APPEND_EDGE_PATCH = "agentic_fraud_servicing.investigator.orchestrator.append_evidence_edge"
_UPDATE_STATUS_PATCH = "agentic_fraud_servicing.investigator.orchestrator.update_case_status"


# -- Test Classes --


class TestInvestigatorOrchestratorInit:
    """Tests for orchestrator initialization."""

    def test_stores_gateway_and_provider(self):
        """Gateway and model_provider are stored as attributes."""
        gateway = MagicMock()
        provider = MagicMock()
        orch = InvestigatorOrchestrator(gateway=gateway, model_provider=provider)
        assert orch.gateway is gateway
        assert orch.model_provider is provider


class TestInvestigate:
    """Tests for the investigate method."""

    @patch(_UPDATE_STATUS_PATCH)
    @patch(_APPEND_NODE_PATCH)
    @patch(_WRITER_PATCH, new_callable=AsyncMock)
    @patch(_SCAM_PATCH, new_callable=AsyncMock)
    @patch(_MERCHANT_PATCH, new_callable=AsyncMock)
    @patch(_SCHEME_PATCH, new_callable=AsyncMock)
    async def test_returns_case_pack(
        self,
        mock_scheme,
        mock_merchant,
        mock_scam,
        mock_writer,
        mock_append_node,
        mock_update_status,
    ):
        """investigate returns a CasePack instance."""
        mock_scheme.return_value = _mock_scheme_result()
        mock_merchant.return_value = _mock_merchant_result()
        mock_scam.return_value = _mock_scam_result()
        mock_writer.return_value = _mock_case_pack()

        orch = _make_orchestrator()
        result = await orch.investigate("case-001")

        assert isinstance(result, CasePack)
        assert result.case_summary == "Investigation summary for case-001."

    @patch(_UPDATE_STATUS_PATCH)
    @patch(_APPEND_NODE_PATCH)
    @patch(_WRITER_PATCH, new_callable=AsyncMock)
    @patch(_SCAM_PATCH, new_callable=AsyncMock)
    @patch(_MERCHANT_PATCH, new_callable=AsyncMock)
    @patch(_SCHEME_PATCH, new_callable=AsyncMock)
    async def test_reads_case_and_evidence(
        self,
        mock_scheme,
        mock_merchant,
        mock_scam,
        mock_writer,
        mock_append_node,
        mock_update_status,
    ):
        """investigate reads case and evidence from gateway stores."""
        mock_scheme.return_value = _mock_scheme_result()
        mock_merchant.return_value = _mock_merchant_result()
        mock_scam.return_value = _mock_scam_result()
        mock_writer.return_value = _mock_case_pack()

        gateway = _make_gateway()
        orch = _make_orchestrator(gateway=gateway)
        await orch.investigate("case-001")

        gateway.case_store.get_case.assert_called_once_with("case-001")
        gateway.evidence_store.get_nodes_by_case.assert_called_once_with("case-001")
        gateway.evidence_store.get_edges_by_case.assert_called_once_with("case-001")

    async def test_raises_if_case_not_found(self):
        """investigate raises RuntimeError if case not found."""
        gateway = MagicMock()
        gateway.case_store.get_case.return_value = None
        orch = _make_orchestrator(gateway=gateway)

        with pytest.raises(RuntimeError, match="Case not found: case-999"):
            await orch.investigate("case-999")

    @patch(_UPDATE_STATUS_PATCH)
    @patch(_APPEND_NODE_PATCH)
    @patch(_WRITER_PATCH, new_callable=AsyncMock)
    @patch(_SCAM_PATCH, new_callable=AsyncMock)
    @patch(_MERCHANT_PATCH, new_callable=AsyncMock)
    @patch(_SCHEME_PATCH, new_callable=AsyncMock)
    async def test_runs_all_specialists(
        self,
        mock_scheme,
        mock_merchant,
        mock_scam,
        mock_writer,
        mock_append_node,
        mock_update_status,
    ):
        """investigate calls all 4 specialist agents."""
        mock_scheme.return_value = _mock_scheme_result()
        mock_merchant.return_value = _mock_merchant_result()
        mock_scam.return_value = _mock_scam_result()
        mock_writer.return_value = _mock_case_pack()

        orch = _make_orchestrator()
        await orch.investigate("case-001")

        mock_scheme.assert_called_once()
        mock_merchant.assert_called_once()
        mock_scam.assert_called_once()
        mock_writer.assert_called_once()

    @patch(_UPDATE_STATUS_PATCH)
    @patch(_APPEND_NODE_PATCH)
    @patch(_WRITER_PATCH, new_callable=AsyncMock)
    @patch(_SCAM_PATCH, new_callable=AsyncMock)
    @patch(_MERCHANT_PATCH, new_callable=AsyncMock)
    @patch(_SCHEME_PATCH, new_callable=AsyncMock)
    async def test_writes_investigator_note(
        self,
        mock_scheme,
        mock_merchant,
        mock_scam,
        mock_writer,
        mock_append_node,
        mock_update_status,
    ):
        """investigate writes an InvestigatorNote to evidence store."""
        mock_scheme.return_value = _mock_scheme_result()
        mock_merchant.return_value = _mock_merchant_result()
        mock_scam.return_value = _mock_scam_result()
        mock_writer.return_value = _mock_case_pack()

        orch = _make_orchestrator()
        await orch.investigate("case-001")

        mock_append_node.assert_called_once()
        call_args = mock_append_node.call_args
        node = call_args[1]["node"] if "node" in call_args[1] else call_args[0][2]
        assert node.case_id == "case-001"
        assert node.text == "Investigation summary for case-001."
        assert node.author == "investigator"
        assert node.source_type == EvidenceSourceType.FACT

    @patch(_UPDATE_STATUS_PATCH)
    @patch(_APPEND_NODE_PATCH)
    @patch(_WRITER_PATCH, new_callable=AsyncMock)
    @patch(_SCAM_PATCH, new_callable=AsyncMock)
    @patch(_MERCHANT_PATCH, new_callable=AsyncMock)
    @patch(_SCHEME_PATCH, new_callable=AsyncMock)
    async def test_updates_case_status(
        self,
        mock_scheme,
        mock_merchant,
        mock_scam,
        mock_writer,
        mock_append_node,
        mock_update_status,
    ):
        """investigate updates case status to INVESTIGATING."""
        mock_scheme.return_value = _mock_scheme_result()
        mock_merchant.return_value = _mock_merchant_result()
        mock_scam.return_value = _mock_scam_result()
        mock_writer.return_value = _mock_case_pack()

        orch = _make_orchestrator()
        await orch.investigate("case-001")

        mock_update_status.assert_called_once()
        call_args = mock_update_status.call_args
        # Positional: gateway, ctx, case_id, status
        assert call_args[0][2] == "case-001"
        assert call_args[0][3] == CaseStatus.INVESTIGATING

    @patch(_UPDATE_STATUS_PATCH)
    @patch(_APPEND_NODE_PATCH)
    @patch(_WRITER_PATCH, new_callable=AsyncMock)
    @patch(_SCAM_PATCH, new_callable=AsyncMock)
    @patch(_MERCHANT_PATCH, new_callable=AsyncMock)
    @patch(_SCHEME_PATCH, new_callable=AsyncMock)
    async def test_trace_logging(
        self,
        mock_scheme,
        mock_merchant,
        mock_scam,
        mock_writer,
        mock_append_node,
        mock_update_status,
    ):
        """investigate logs gateway calls to trace store."""
        mock_scheme.return_value = _mock_scheme_result()
        mock_merchant.return_value = _mock_merchant_result()
        mock_scam.return_value = _mock_scam_result()
        mock_writer.return_value = _mock_case_pack()

        gateway = _make_gateway()
        orch = _make_orchestrator(gateway=gateway)
        await orch.investigate("case-001")

        # At least 2 log_call invocations: get_case and get_evidence
        assert gateway.log_call.call_count >= 2


class TestGracefulDegradation:
    """Tests for specialist failure handling."""

    @patch(_UPDATE_STATUS_PATCH)
    @patch(_APPEND_NODE_PATCH)
    @patch(_WRITER_PATCH, new_callable=AsyncMock)
    @patch(_SCAM_PATCH, new_callable=AsyncMock)
    @patch(_MERCHANT_PATCH, new_callable=AsyncMock)
    @patch(_SCHEME_PATCH, new_callable=AsyncMock)
    async def test_scheme_failure_continues(
        self,
        mock_scheme,
        mock_merchant,
        mock_scam,
        mock_writer,
        mock_append_node,
        mock_update_status,
    ):
        """If scheme mapper fails, orchestrator continues with other specialists."""
        mock_scheme.side_effect = RuntimeError("LLM timeout")
        mock_merchant.return_value = _mock_merchant_result()
        mock_scam.return_value = _mock_scam_result()
        mock_writer.return_value = _mock_case_pack()

        orch = _make_orchestrator()
        result = await orch.investigate("case-001")

        assert isinstance(result, CasePack)
        # Case writer should receive empty scheme result
        writer_call = mock_writer.call_args
        assert writer_call[1]["scheme_result"] == "{}"

    @patch(_UPDATE_STATUS_PATCH)
    @patch(_APPEND_NODE_PATCH)
    @patch(_WRITER_PATCH, new_callable=AsyncMock)
    @patch(_SCAM_PATCH, new_callable=AsyncMock)
    @patch(_MERCHANT_PATCH, new_callable=AsyncMock)
    @patch(_SCHEME_PATCH, new_callable=AsyncMock)
    async def test_all_pre_writer_specialists_fail(
        self,
        mock_scheme,
        mock_merchant,
        mock_scam,
        mock_writer,
        mock_append_node,
        mock_update_status,
    ):
        """If all pre-writer specialists fail, case writer still runs."""
        mock_scheme.side_effect = RuntimeError("scheme error")
        mock_merchant.side_effect = RuntimeError("merchant error")
        mock_scam.side_effect = RuntimeError("scam error")
        mock_writer.return_value = _mock_case_pack()

        orch = _make_orchestrator()
        result = await orch.investigate("case-001")

        assert isinstance(result, CasePack)
        mock_writer.assert_called_once()


class TestContradictionEdges:
    """Tests for CONTRADICTS edge creation from scam analysis."""

    @patch(_UPDATE_STATUS_PATCH)
    @patch(_APPEND_EDGE_PATCH)
    @patch(_APPEND_NODE_PATCH)
    @patch(_WRITER_PATCH, new_callable=AsyncMock)
    @patch(_SCAM_PATCH, new_callable=AsyncMock)
    @patch(_MERCHANT_PATCH, new_callable=AsyncMock)
    @patch(_SCHEME_PATCH, new_callable=AsyncMock)
    async def test_creates_contradicts_edges(
        self,
        mock_scheme,
        mock_merchant,
        mock_scam,
        mock_writer,
        mock_append_node,
        mock_append_edge,
        mock_update_status,
    ):
        """CONTRADICTS edges are created when scam analysis finds contradictions."""
        contradictions = [
            {"claim_node_id": "n3", "evidence_node_id": "n1", "severity": "high"},
        ]
        mock_scheme.return_value = _mock_scheme_result()
        mock_merchant.return_value = _mock_merchant_result()
        mock_scam.return_value = _mock_scam_result(contradictions=contradictions)
        mock_writer.return_value = _mock_case_pack()

        orch = _make_orchestrator()
        await orch.investigate("case-001")

        mock_append_edge.assert_called_once()
        edge = mock_append_edge.call_args[0][2]
        assert edge.source_node_id == "n3"
        assert edge.target_node_id == "n1"
        assert edge.edge_type == EvidenceEdgeType.CONTRADICTS

    @patch(_UPDATE_STATUS_PATCH)
    @patch(_APPEND_EDGE_PATCH)
    @patch(_APPEND_NODE_PATCH)
    @patch(_WRITER_PATCH, new_callable=AsyncMock)
    @patch(_SCAM_PATCH, new_callable=AsyncMock)
    @patch(_MERCHANT_PATCH, new_callable=AsyncMock)
    @patch(_SCHEME_PATCH, new_callable=AsyncMock)
    async def test_no_edges_without_node_ids(
        self,
        mock_scheme,
        mock_merchant,
        mock_scam,
        mock_writer,
        mock_append_node,
        mock_append_edge,
        mock_update_status,
    ):
        """Contradictions without node IDs don't create edges."""
        contradictions = [
            {"claim": "no card", "contradicting_evidence": "chip used", "severity": "high"},
        ]
        mock_scheme.return_value = _mock_scheme_result()
        mock_merchant.return_value = _mock_merchant_result()
        mock_scam.return_value = _mock_scam_result(contradictions=contradictions)
        mock_writer.return_value = _mock_case_pack()

        orch = _make_orchestrator()
        await orch.investigate("case-001")

        mock_append_edge.assert_not_called()


class TestDerivedFromEdges:
    """Tests for DERIVED_FROM edge creation between related claims."""

    @patch(_UPDATE_STATUS_PATCH)
    @patch(_APPEND_EDGE_PATCH)
    @patch(_APPEND_NODE_PATCH)
    @patch(_WRITER_PATCH, new_callable=AsyncMock)
    @patch(_SCAM_PATCH, new_callable=AsyncMock)
    @patch(_MERCHANT_PATCH, new_callable=AsyncMock)
    @patch(_SCHEME_PATCH, new_callable=AsyncMock)
    async def test_creates_derived_from_edges_for_related_claims(
        self,
        mock_scheme,
        mock_merchant,
        mock_scam,
        mock_writer,
        mock_append_node,
        mock_append_edge,
        mock_update_status,
    ):
        """Claims sharing merchant_name get linked with DERIVED_FROM edges."""
        mock_scheme.return_value = _mock_scheme_result()
        mock_merchant.return_value = _mock_merchant_result()
        mock_scam.return_value = _mock_scam_result()
        mock_writer.return_value = _mock_case_pack()

        gateway = MagicMock()
        gateway.case_store.get_case.return_value = _make_case()
        gateway.evidence_store.get_nodes_by_case.return_value = (
            _make_evidence_nodes_with_related_claims()
        )
        gateway.evidence_store.get_edges_by_case.return_value = []

        orch = _make_orchestrator(gateway=gateway)
        await orch.investigate("case-001")

        # claim-002 -> claim-001 (both have merchant_name=TechVault)
        # claim-003 has merchant_name=Amazon — no link to the others
        edge_calls = [
            c
            for c in mock_append_edge.call_args_list
            if c[0][2].edge_type == EvidenceEdgeType.DERIVED_FROM
        ]
        assert len(edge_calls) == 1
        edge = edge_calls[0][0][2]
        assert edge.source_node_id == "claim-002"
        assert edge.target_node_id == "claim-001"
        assert edge.edge_type == EvidenceEdgeType.DERIVED_FROM

    @patch(_UPDATE_STATUS_PATCH)
    @patch(_APPEND_EDGE_PATCH)
    @patch(_APPEND_NODE_PATCH)
    @patch(_WRITER_PATCH, new_callable=AsyncMock)
    @patch(_SCAM_PATCH, new_callable=AsyncMock)
    @patch(_MERCHANT_PATCH, new_callable=AsyncMock)
    @patch(_SCHEME_PATCH, new_callable=AsyncMock)
    async def test_no_derived_from_without_shared_entities(
        self,
        mock_scheme,
        mock_merchant,
        mock_scam,
        mock_writer,
        mock_append_node,
        mock_append_edge,
        mock_update_status,
    ):
        """Claims without overlapping entities don't get linked."""
        mock_scheme.return_value = _mock_scheme_result()
        mock_merchant.return_value = _mock_merchant_result()
        mock_scam.return_value = _mock_scam_result()
        mock_writer.return_value = _mock_case_pack()

        # Two claims with different merchants — no overlap
        nodes = [
            {
                "node_id": "c1",
                "node_type": "CLAIM_STATEMENT",
                "source_type": "ALLEGATION",
                "text": "Claim A",
                "entities": {"merchant_name": "StoreA"},
            },
            {
                "node_id": "c2",
                "node_type": "CLAIM_STATEMENT",
                "source_type": "ALLEGATION",
                "text": "Claim B",
                "entities": {"merchant_name": "StoreB"},
            },
        ]
        gateway = MagicMock()
        gateway.case_store.get_case.return_value = _make_case()
        gateway.evidence_store.get_nodes_by_case.return_value = nodes
        gateway.evidence_store.get_edges_by_case.return_value = []

        orch = _make_orchestrator(gateway=gateway)
        await orch.investigate("case-001")

        # No DERIVED_FROM edges should be created
        derived_calls = [
            c
            for c in mock_append_edge.call_args_list
            if c[0][2].edge_type == EvidenceEdgeType.DERIVED_FROM
        ]
        assert len(derived_calls) == 0

    @patch(_UPDATE_STATUS_PATCH)
    @patch(_APPEND_EDGE_PATCH)
    @patch(_APPEND_NODE_PATCH)
    @patch(_WRITER_PATCH, new_callable=AsyncMock)
    @patch(_SCAM_PATCH, new_callable=AsyncMock)
    @patch(_MERCHANT_PATCH, new_callable=AsyncMock)
    @patch(_SCHEME_PATCH, new_callable=AsyncMock)
    async def test_no_derived_from_without_entities(
        self,
        mock_scheme,
        mock_merchant,
        mock_scam,
        mock_writer,
        mock_append_node,
        mock_append_edge,
        mock_update_status,
    ):
        """Claims without entities dict are skipped for linking."""
        mock_scheme.return_value = _mock_scheme_result()
        mock_merchant.return_value = _mock_merchant_result()
        mock_scam.return_value = _mock_scam_result()
        mock_writer.return_value = _mock_case_pack()

        orch = _make_orchestrator()  # default nodes have empty entities
        await orch.investigate("case-001")

        derived_calls = [
            c
            for c in mock_append_edge.call_args_list
            if c[0][2].edge_type == EvidenceEdgeType.DERIVED_FROM
        ]
        assert len(derived_calls) == 0
