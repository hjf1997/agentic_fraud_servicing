"""SQLite-backed case persistence with CRUD operations.

Stores Case models as JSON blobs with denormalized status and created_at
columns for efficient filtering and ordering. Uses WAL mode for concurrent
read support.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

from agentic_fraud_servicing.models.case import Case
from agentic_fraud_servicing.models.enums import CaseStatus


class CaseStore:
    """SQLite store for Case model persistence.

    Each case is stored as a JSON blob in the `data` column, with `status`
    and `created_at` denormalized for efficient WHERE/ORDER BY queries.

    Args:
        db_path: Path to the SQLite database file. Parent directories are
            created automatically if they don't exist.
    """

    def __init__(self, db_path: str) -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        try:
            self._conn = sqlite3.connect(db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT
                )
                """
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise RuntimeError(f"CaseStore init failed: {exc}") from exc

    def create_case(self, case: Case) -> None:
        """Insert a new case into the store.

        Args:
            case: The Case model to persist.

        Raises:
            RuntimeError: If the case_id already exists or a DB error occurs.
        """
        try:
            self._conn.execute(
                "INSERT INTO cases (case_id, status, data, created_at) VALUES (?, ?, ?, ?)",
                (
                    case.case_id,
                    case.status.value,
                    case.model_dump_json(),
                    case.created_at.isoformat(),
                ),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:
            raise RuntimeError(
                f"create_case failed: case_id '{case.case_id}' already exists"
            ) from exc
        except sqlite3.Error as exc:
            raise RuntimeError(f"create_case failed for case_id '{case.case_id}': {exc}") from exc

    def get_case(self, case_id: str) -> Case | None:
        """Retrieve a case by its ID.

        Args:
            case_id: The unique case identifier.

        Returns:
            The deserialized Case, or None if not found.

        Raises:
            RuntimeError: On DB error.
        """
        try:
            row = self._conn.execute(
                "SELECT data FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        except sqlite3.Error as exc:
            raise RuntimeError(f"get_case failed for case_id '{case_id}': {exc}") from exc

        if row is None:
            return None
        return Case.model_validate_json(row[0])

    def update_case_status(self, case_id: str, status: CaseStatus) -> None:
        """Update the status of an existing case.

        Updates both the denormalized `status` column and the `status` field
        inside the JSON `data` column. Sets `updated_at` to the current time.

        Args:
            case_id: The case to update.
            status: The new CaseStatus value.

        Raises:
            RuntimeError: If case_id is not found or a DB error occurs.
        """
        try:
            row = self._conn.execute(
                "SELECT data FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        except sqlite3.Error as exc:
            raise RuntimeError(
                f"update_case_status failed for case_id '{case_id}': {exc}"
            ) from exc

        if row is None:
            raise RuntimeError(f"update_case_status failed: case_id '{case_id}' not found")

        # Update the status inside the JSON data
        data = json.loads(row[0])
        now = datetime.now(timezone.utc)
        data["status"] = status.value
        data["updated_at"] = now.isoformat()
        updated_json = json.dumps(data)

        try:
            self._conn.execute(
                "UPDATE cases SET status = ?, data = ?, updated_at = ? WHERE case_id = ?",
                (status.value, updated_json, now.isoformat(), case_id),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise RuntimeError(
                f"update_case_status failed for case_id '{case_id}': {exc}"
            ) from exc

    def list_cases_by_status(self, status: CaseStatus) -> list[Case]:
        """List all cases with the given status, newest first.

        Args:
            status: The CaseStatus to filter by.

        Returns:
            List of Case models ordered by created_at descending.

        Raises:
            RuntimeError: On DB error.
        """
        try:
            rows = self._conn.execute(
                "SELECT data FROM cases WHERE status = ? ORDER BY created_at DESC",
                (status.value,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise RuntimeError(
                f"list_cases_by_status failed for status '{status.value}': {exc}"
            ) from exc

        return [Case.model_validate_json(row[0]) for row in rows]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
