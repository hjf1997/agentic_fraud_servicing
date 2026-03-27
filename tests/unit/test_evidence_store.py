"""Unit tests for storage.evidence_store module."""

from datetime import datetime, timezone

import pytest

from agentic_fraud_servicing.models.enums import (
    EvidenceEdgeType,
    EvidenceSourceType,
)
from agentic_fraud_servicing.models.evidence import (
    AllegationStatement,
    EvidenceEdge,
    Transaction,
)
from agentic_fraud_servicing.storage.evidence_store import EvidenceStore

NOW = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


def _make_transaction(
    node_id: str = "NODE-TXN-001",
    case_id: str = "CASE-001",
) -> Transaction:
    """Helper to build a minimal Transaction node for testing."""
    return Transaction(
        node_id=node_id,
        case_id=case_id,
        source_type=EvidenceSourceType.FACT,
        created_at=NOW,
        amount=99.99,
        merchant_name="TestMerchant",
        transaction_date=NOW,
    )


def _make_allegation(
    node_id: str = "NODE-CLM-001",
    case_id: str = "CASE-001",
) -> AllegationStatement:
    """Helper to build a minimal AllegationStatement node for testing."""
    return AllegationStatement(
        node_id=node_id,
        case_id=case_id,
        source_type=EvidenceSourceType.ALLEGATION,
        created_at=NOW,
        text="I did not make this purchase.",
    )


def _make_edge(
    edge_id: str = "EDGE-001",
    case_id: str = "CASE-001",
    source_node_id: str = "NODE-TXN-001",
    target_node_id: str = "NODE-CLM-001",
    edge_type: EvidenceEdgeType = EvidenceEdgeType.ALLEGATION,
) -> EvidenceEdge:
    """Helper to build a minimal EvidenceEdge for testing."""
    return EvidenceEdge(
        edge_id=edge_id,
        case_id=case_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        edge_type=edge_type,
        created_at=NOW,
    )


class TestEvidenceStoreInit:
    """Tests for EvidenceStore initialization."""

    def test_wal_mode_enabled(self, tmp_path):
        """WAL journal mode should be active after init."""
        store = EvidenceStore(str(tmp_path / "test.db"))
        row = store._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"
        store.close()

    def test_creates_parent_directories(self, tmp_path):
        """Init should create parent directories if they don't exist."""
        db_path = tmp_path / "nested" / "dir" / "test.db"
        store = EvidenceStore(str(db_path))
        assert db_path.exists()
        store.close()


class TestAddAndGetNodes:
    """Tests for add_node and get_nodes_by_case."""

    def test_add_and_get_round_trip(self, tmp_path):
        """A stored Transaction node should be retrievable with all fields."""
        store = EvidenceStore(str(tmp_path / "test.db"))
        txn = _make_transaction()
        store.add_node(txn)

        nodes = store.get_nodes_by_case("CASE-001")
        assert len(nodes) == 1
        node = nodes[0]
        assert node["node_id"] == "NODE-TXN-001"
        assert node["case_id"] == "CASE-001"
        assert node["node_type"] == "TRANSACTION"
        assert node["source_type"] == "FACT"
        assert node["amount"] == 99.99
        assert node["merchant_name"] == "TestMerchant"
        store.close()

    def test_get_nodes_no_results_returns_empty(self, tmp_path):
        """Getting nodes for a nonexistent case should return empty list."""
        store = EvidenceStore(str(tmp_path / "test.db"))
        assert store.get_nodes_by_case("NONEXISTENT") == []
        store.close()

    def test_duplicate_node_raises(self, tmp_path):
        """Adding a node with a duplicate node_id should raise RuntimeError."""
        store = EvidenceStore(str(tmp_path / "test.db"))
        txn = _make_transaction()
        store.add_node(txn)

        with pytest.raises(RuntimeError, match="already exists"):
            store.add_node(txn)
        store.close()

    def test_multiple_node_types(self, tmp_path):
        """Different node types should be stored and retrieved correctly."""
        store = EvidenceStore(str(tmp_path / "test.db"))
        txn = _make_transaction()
        claim = _make_allegation()
        store.add_node(txn)
        store.add_node(claim)

        nodes = store.get_nodes_by_case("CASE-001")
        assert len(nodes) == 2
        node_types = {n["node_type"] for n in nodes}
        assert node_types == {"TRANSACTION", "ALLEGATION_STATEMENT"}

        # Verify type-specific fields are preserved
        txn_node = next(n for n in nodes if n["node_type"] == "TRANSACTION")
        assert txn_node["amount"] == 99.99
        claim_node = next(n for n in nodes if n["node_type"] == "ALLEGATION_STATEMENT")
        assert claim_node["text"] == "I did not make this purchase."
        store.close()


class TestAddAndGetEdges:
    """Tests for add_edge and get_edges_by_case."""

    def test_add_and_get_round_trip(self, tmp_path):
        """A stored edge should be retrievable with all fields."""
        store = EvidenceStore(str(tmp_path / "test.db"))
        edge = _make_edge()
        store.add_edge(edge)

        edges = store.get_edges_by_case("CASE-001")
        assert len(edges) == 1
        e = edges[0]
        assert e["edge_id"] == "EDGE-001"
        assert e["case_id"] == "CASE-001"
        assert e["source_node_id"] == "NODE-TXN-001"
        assert e["target_node_id"] == "NODE-CLM-001"
        assert e["edge_type"] == "ALLEGATION"
        store.close()

    def test_get_edges_no_results_returns_empty(self, tmp_path):
        """Getting edges for a nonexistent case should return empty list."""
        store = EvidenceStore(str(tmp_path / "test.db"))
        assert store.get_edges_by_case("NONEXISTENT") == []
        store.close()

    def test_duplicate_edge_raises(self, tmp_path):
        """Adding an edge with a duplicate edge_id should raise RuntimeError."""
        store = EvidenceStore(str(tmp_path / "test.db"))
        edge = _make_edge()
        store.add_edge(edge)

        with pytest.raises(RuntimeError, match="already exists"):
            store.add_edge(edge)
        store.close()


class TestGetConnectedNodes:
    """Tests for get_connected_nodes."""

    def test_finds_via_source_and_target(self, tmp_path):
        """Should find nodes connected via both source and target edges."""
        store = EvidenceStore(str(tmp_path / "test.db"))
        txn = _make_transaction("NODE-A")
        claim = _make_allegation("NODE-B")
        txn2 = _make_transaction("NODE-C")
        store.add_node(txn)
        store.add_node(claim)
        store.add_node(txn2)

        # NODE-A -> NODE-B (A is source)
        store.add_edge(_make_edge("E1", "CASE-001", "NODE-A", "NODE-B"))
        # NODE-C -> NODE-A (A is target)
        store.add_edge(_make_edge("E2", "CASE-001", "NODE-C", "NODE-A"))

        connected = store.get_connected_nodes("NODE-A")
        connected_ids = {n["node_id"] for n in connected}
        assert connected_ids == {"NODE-B", "NODE-C"}
        store.close()

    def test_no_connections_returns_empty(self, tmp_path):
        """A node with no edges should return empty list."""
        store = EvidenceStore(str(tmp_path / "test.db"))
        txn = _make_transaction()
        store.add_node(txn)

        assert store.get_connected_nodes("NODE-TXN-001") == []
        store.close()


class TestUpdateNode:
    """Tests for update_node."""

    def test_update_node_success(self, tmp_path):
        """update_node should replace the stored JSON data for an existing node."""
        store = EvidenceStore(str(tmp_path / "test.db"))
        txn = _make_transaction()
        store.add_node(txn)

        # Mutate and update
        txn.is_disputed = True
        store.update_node(txn)

        nodes = store.get_nodes_by_case("CASE-001")
        assert len(nodes) == 1
        assert nodes[0]["is_disputed"] is True
        store.close()

    def test_update_node_not_found_raises(self, tmp_path):
        """update_node with a nonexistent node_id should raise RuntimeError."""
        store = EvidenceStore(str(tmp_path / "test.db"))
        txn = _make_transaction(node_id="NONEXISTENT")

        with pytest.raises(RuntimeError, match="not found"):
            store.update_node(txn)
        store.close()


class TestErrorWrapping:
    """Tests for sqlite3 error wrapping as RuntimeError."""

    def test_get_nodes_on_closed_raises(self, tmp_path):
        """get_nodes_by_case on a closed connection should raise RuntimeError."""
        store = EvidenceStore(str(tmp_path / "test.db"))
        store.close()

        with pytest.raises(RuntimeError):
            store.get_nodes_by_case("CASE-001")

    def test_get_edges_on_closed_raises(self, tmp_path):
        """get_edges_by_case on a closed connection should raise RuntimeError."""
        store = EvidenceStore(str(tmp_path / "test.db"))
        store.close()

        with pytest.raises(RuntimeError):
            store.get_edges_by_case("CASE-001")

    def test_get_connected_on_closed_raises(self, tmp_path):
        """get_connected_nodes on a closed connection should raise RuntimeError."""
        store = EvidenceStore(str(tmp_path / "test.db"))
        store.close()

        with pytest.raises(RuntimeError):
            store.get_connected_nodes("NODE-001")
