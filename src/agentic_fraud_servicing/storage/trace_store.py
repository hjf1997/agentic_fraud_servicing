"""SQLite-backed agent trace and audit log storage.

Stores agent invocation traces with all context needed for audit replay:
trace_id, case_id, agent_id, action, input/output data, duration, and status.
Uses WAL mode for concurrent read support.
"""

import os
import sqlite3
from datetime import datetime


class TraceStore:
    """SQLite store for agent invocation traces.

    Each trace records a single agent tool call with its input, output,
    duration, and status. Traces are indexed by case_id and agent_id for
    efficient querying during audit review.

    Args:
        db_path: Path to the SQLite database file. Parent directories are
            created automatically if they don't exist.
    """

    def __init__(self, db_path: str) -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        try:
            self._conn = sqlite3.connect(db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    output_data TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'success'
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_case_id ON traces (case_id)")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_traces_agent_id ON traces (agent_id)"
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise RuntimeError(f"TraceStore init failed: {exc}") from exc

    def log_invocation(
        self,
        trace_id: str,
        case_id: str,
        agent_id: str,
        action: str,
        input_data: str,
        output_data: str,
        duration_ms: float,
        timestamp: datetime,
        status: str = "success",
    ) -> None:
        """Record an agent invocation trace.

        Args:
            trace_id: Unique identifier for this trace.
            case_id: The case this invocation belongs to.
            agent_id: The agent that was invoked.
            action: Description of the action performed.
            input_data: JSON string of the invocation input.
            output_data: JSON string of the invocation output.
            duration_ms: Execution duration in milliseconds.
            timestamp: When the invocation occurred.
            status: Outcome status (default 'success').

        Raises:
            RuntimeError: On duplicate trace_id or DB error.
        """
        try:
            self._conn.execute(
                "INSERT INTO traces "
                "(trace_id, case_id, agent_id, action, input_data, output_data, "
                "duration_ms, timestamp, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trace_id,
                    case_id,
                    agent_id,
                    action,
                    input_data,
                    output_data,
                    duration_ms,
                    timestamp.isoformat(),
                    status,
                ),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            raise RuntimeError(
                f"log_invocation failed: trace_id '{trace_id}' already exists"
            ) from exc
        except sqlite3.Error as exc:
            raise RuntimeError(f"log_invocation failed for trace_id '{trace_id}': {exc}") from exc

    def get_traces_by_case(self, case_id: str) -> list[dict]:
        """Retrieve all traces for a case, ordered by timestamp ascending.

        Args:
            case_id: The case to query traces for.

        Returns:
            List of trace dicts with all column values. Empty if none found.

        Raises:
            RuntimeError: On DB error.
        """
        try:
            rows = self._conn.execute(
                "SELECT trace_id, case_id, agent_id, action, input_data, "
                "output_data, duration_ms, timestamp, status "
                "FROM traces WHERE case_id = ? ORDER BY timestamp ASC",
                (case_id,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise RuntimeError(
                f"get_traces_by_case failed for case_id '{case_id}': {exc}"
            ) from exc

        return [dict(row) for row in rows]

    def get_trace(self, trace_id: str) -> dict | None:
        """Retrieve a single trace by its ID.

        Args:
            trace_id: The unique trace identifier.

        Returns:
            A dict with all column values, or None if not found.

        Raises:
            RuntimeError: On DB error.
        """
        try:
            row = self._conn.execute(
                "SELECT trace_id, case_id, agent_id, action, input_data, "
                "output_data, duration_ms, timestamp, status "
                "FROM traces WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise RuntimeError(f"get_trace failed for trace_id '{trace_id}': {exc}") from exc

        if row is None:
            return None
        return dict(row)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
