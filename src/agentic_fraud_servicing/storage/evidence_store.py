"""SQLite-backed evidence graph persistence.

Stores evidence nodes and edges as JSON blobs with denormalized columns
for case_id, node_type, source_type, and edge endpoints. Uses WAL mode
for concurrent read support.
"""

import json
import os
import sqlite3

from agentic_fraud_servicing.models.evidence import EvidenceEdge, EvidenceNode


class EvidenceStore:
    """SQLite store for evidence graph persistence.

    Evidence nodes are stored in the `evidence_nodes` table with the full
    model serialized as JSON in the `data` column. Denormalized columns
    (case_id, node_type, source_type) enable efficient filtering. Edges
    are stored similarly in `evidence_edges` with indexes on case_id,
    source_node_id, and target_node_id.

    Args:
        db_path: Path to the SQLite database file. Parent directories are
            created automatically if they don't exist.
    """

    def __init__(self, db_path: str) -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        try:
            self._conn = sqlite3.connect(db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS evidence_nodes (
                    node_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_nodes_case_id
                    ON evidence_nodes (case_id);

                CREATE TABLE IF NOT EXISTS evidence_edges (
                    edge_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    source_node_id TEXT NOT NULL,
                    target_node_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_edges_case_id
                    ON evidence_edges (case_id);
                CREATE INDEX IF NOT EXISTS idx_edges_source_node_id
                    ON evidence_edges (source_node_id);
                CREATE INDEX IF NOT EXISTS idx_edges_target_node_id
                    ON evidence_edges (target_node_id);
                """
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise RuntimeError(f"EvidenceStore init failed: {exc}") from exc

    def add_node(self, node: EvidenceNode) -> None:
        """Insert an evidence node into the store.

        Args:
            node: The EvidenceNode (or subclass) to persist.

        Raises:
            RuntimeError: If the node_id already exists or a DB error occurs.
        """
        try:
            self._conn.execute(
                "INSERT INTO evidence_nodes "
                "(node_id, case_id, node_type, source_type, data, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    node.node_id,
                    node.case_id,
                    node.node_type.value,
                    node.source_type.value,
                    node.model_dump_json(),
                    node.created_at.isoformat(),
                ),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            raise RuntimeError(
                f"add_node failed: node_id '{node.node_id}' already exists"
            ) from exc
        except sqlite3.Error as exc:
            raise RuntimeError(f"add_node failed for node_id '{node.node_id}': {exc}") from exc

    def add_edge(self, edge: EvidenceEdge) -> None:
        """Insert an evidence edge into the store.

        Args:
            edge: The EvidenceEdge to persist.

        Raises:
            RuntimeError: If the edge_id already exists or a DB error occurs.
        """
        try:
            self._conn.execute(
                "INSERT INTO evidence_edges "
                "(edge_id, case_id, source_node_id, target_node_id, "
                "edge_type, data, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    edge.edge_id,
                    edge.case_id,
                    edge.source_node_id,
                    edge.target_node_id,
                    edge.edge_type.value,
                    edge.model_dump_json(),
                    edge.created_at.isoformat(),
                ),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            raise RuntimeError(
                f"add_edge failed: edge_id '{edge.edge_id}' already exists"
            ) from exc
        except sqlite3.Error as exc:
            raise RuntimeError(f"add_edge failed for edge_id '{edge.edge_id}': {exc}") from exc

    def update_node(self, node: EvidenceNode) -> None:
        """Update an existing evidence node in the store.

        Replaces the JSON data blob for the node matching node.node_id.

        Args:
            node: The EvidenceNode (or subclass) with updated fields.

        Raises:
            RuntimeError: If node_id does not exist or a DB error occurs.
        """
        try:
            cursor = self._conn.execute(
                "UPDATE evidence_nodes SET data = ? WHERE node_id = ?",
                (node.model_dump_json(), node.node_id),
            )
            if cursor.rowcount == 0:
                raise RuntimeError(
                    f"update_node failed: node_id '{node.node_id}' not found"
                )
            self._conn.commit()
        except RuntimeError:
            raise
        except sqlite3.Error as exc:
            raise RuntimeError(
                f"update_node failed for node_id '{node.node_id}': {exc}"
            ) from exc

    def get_nodes_by_case(self, case_id: str) -> list[dict]:
        """Retrieve all evidence nodes for a case.

        Args:
            case_id: The case identifier to filter by.

        Returns:
            List of node dicts parsed from the JSON data column.
            Empty list if no nodes found.

        Raises:
            RuntimeError: On DB error.
        """
        try:
            rows = self._conn.execute(
                "SELECT data FROM evidence_nodes WHERE case_id = ?",
                (case_id,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise RuntimeError(f"get_nodes_by_case failed for case_id '{case_id}': {exc}") from exc

        return [json.loads(row[0]) for row in rows]

    def get_edges_by_case(self, case_id: str) -> list[dict]:
        """Retrieve all evidence edges for a case.

        Args:
            case_id: The case identifier to filter by.

        Returns:
            List of edge dicts parsed from the JSON data column.
            Empty list if no edges found.

        Raises:
            RuntimeError: On DB error.
        """
        try:
            rows = self._conn.execute(
                "SELECT data FROM evidence_edges WHERE case_id = ?",
                (case_id,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise RuntimeError(f"get_edges_by_case failed for case_id '{case_id}': {exc}") from exc

        return [json.loads(row[0]) for row in rows]

    def get_connected_nodes(self, node_id: str) -> list[dict]:
        """Retrieve all nodes connected to a given node via edges.

        Finds nodes where the given node_id appears as either source or
        target in an edge, then returns the opposite endpoint nodes.

        Args:
            node_id: The node identifier to find connections for.

        Returns:
            List of connected node dicts. Empty list if no connections.

        Raises:
            RuntimeError: On DB error.
        """
        try:
            # Find node IDs connected via edges where node_id is source or target
            rows = self._conn.execute(
                "SELECT DISTINCT n.data FROM evidence_nodes n "
                "INNER JOIN evidence_edges e "
                "ON (e.source_node_id = ? AND n.node_id = e.target_node_id) "
                "OR (e.target_node_id = ? AND n.node_id = e.source_node_id)",
                (node_id, node_id),
            ).fetchall()
        except sqlite3.Error as exc:
            raise RuntimeError(
                f"get_connected_nodes failed for node_id '{node_id}': {exc}"
            ) from exc

        return [json.loads(row[0]) for row in rows]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
