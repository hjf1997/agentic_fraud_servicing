"""Unit tests for storage.case_store module."""

from datetime import datetime, timezone

import pytest

from agentic_fraud_servicing.models.case import Case
from agentic_fraud_servicing.models.enums import AllegationType, CaseStatus
from agentic_fraud_servicing.storage.case_store import CaseStore


def _make_case(
    case_id: str = "CASE-001",
    status: CaseStatus = CaseStatus.OPEN,
) -> Case:
    """Helper to build a minimal Case for testing."""
    return Case(
        case_id=case_id,
        call_id="CALL-001",
        customer_id="CUST-001",
        account_id="ACCT-001",
        allegation_type=AllegationType.FRAUD,
        status=status,
        created_at=datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
    )


class TestCaseStoreInit:
    """Tests for CaseStore initialization."""

    def test_wal_mode_enabled(self, tmp_path):
        """WAL journal mode should be active after init."""
        store = CaseStore(str(tmp_path / "test.db"))
        row = store._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"
        store.close()

    def test_creates_parent_directories(self, tmp_path):
        """Init should create parent directories if they don't exist."""
        db_path = tmp_path / "nested" / "dir" / "test.db"
        store = CaseStore(str(db_path))
        assert db_path.exists()
        store.close()


class TestCreateAndGetCase:
    """Tests for create_case and get_case round-trip."""

    def test_create_and_get_round_trip(self, tmp_path):
        """A created case should be retrievable with all fields preserved."""
        store = CaseStore(str(tmp_path / "test.db"))
        case = _make_case()
        store.create_case(case)

        retrieved = store.get_case("CASE-001")
        assert retrieved is not None
        assert retrieved.case_id == case.case_id
        assert retrieved.call_id == case.call_id
        assert retrieved.customer_id == case.customer_id
        assert retrieved.account_id == case.account_id
        assert retrieved.allegation_type == AllegationType.FRAUD
        assert retrieved.status == CaseStatus.OPEN
        assert retrieved.created_at == case.created_at
        store.close()

    def test_get_nonexistent_returns_none(self, tmp_path):
        """Getting a case that doesn't exist should return None."""
        store = CaseStore(str(tmp_path / "test.db"))
        assert store.get_case("NONEXISTENT") is None
        store.close()

    def test_create_duplicate_raises(self, tmp_path):
        """Creating a case with a duplicate case_id should raise RuntimeError."""
        store = CaseStore(str(tmp_path / "test.db"))
        case = _make_case()
        store.create_case(case)

        with pytest.raises(RuntimeError, match="already exists"):
            store.create_case(case)
        store.close()


class TestUpdateCaseStatus:
    """Tests for update_case_status."""

    def test_update_changes_status_in_column_and_json(self, tmp_path):
        """Status update should change both the column and the JSON data."""
        store = CaseStore(str(tmp_path / "test.db"))
        store.create_case(_make_case())

        store.update_case_status("CASE-001", CaseStatus.INVESTIGATING)

        # Check via get_case (reads from JSON data column)
        retrieved = store.get_case("CASE-001")
        assert retrieved is not None
        assert retrieved.status == CaseStatus.INVESTIGATING
        assert retrieved.updated_at is not None

        # Check the denormalized column directly
        row = store._conn.execute(
            "SELECT status, updated_at FROM cases WHERE case_id = ?",
            ("CASE-001",),
        ).fetchone()
        assert row[0] == "INVESTIGATING"
        assert row[1] is not None
        store.close()

    def test_update_nonexistent_raises(self, tmp_path):
        """Updating a nonexistent case should raise RuntimeError."""
        store = CaseStore(str(tmp_path / "test.db"))
        with pytest.raises(RuntimeError, match="not found"):
            store.update_case_status("NONEXISTENT", CaseStatus.CLOSED)
        store.close()


class TestListCasesByStatus:
    """Tests for list_cases_by_status."""

    def test_filters_by_status(self, tmp_path):
        """Should return only cases matching the requested status."""
        store = CaseStore(str(tmp_path / "test.db"))
        store.create_case(_make_case("CASE-001", CaseStatus.OPEN))
        store.create_case(_make_case("CASE-002", CaseStatus.INVESTIGATING))
        store.create_case(_make_case("CASE-003", CaseStatus.OPEN))

        open_cases = store.list_cases_by_status(CaseStatus.OPEN)
        assert len(open_cases) == 2
        assert all(c.status == CaseStatus.OPEN for c in open_cases)

        investigating = store.list_cases_by_status(CaseStatus.INVESTIGATING)
        assert len(investigating) == 1
        assert investigating[0].case_id == "CASE-002"
        store.close()

    def test_no_matches_returns_empty(self, tmp_path):
        """Should return an empty list when no cases match the status."""
        store = CaseStore(str(tmp_path / "test.db"))
        store.create_case(_make_case())
        assert store.list_cases_by_status(CaseStatus.CLOSED) == []
        store.close()


class TestErrorWrapping:
    """Tests for sqlite3 error wrapping as RuntimeError."""

    def test_get_case_on_closed_raises(self, tmp_path):
        """get_case on a closed connection should raise RuntimeError."""
        store = CaseStore(str(tmp_path / "test.db"))
        store.close()

        with pytest.raises(RuntimeError):
            store.get_case("CASE-001")

    def test_create_case_on_closed_raises(self, tmp_path):
        """create_case on a closed connection should raise RuntimeError."""
        store = CaseStore(str(tmp_path / "test.db"))
        store.close()

        with pytest.raises(RuntimeError):
            store.create_case(_make_case())

    def test_list_cases_on_closed_raises(self, tmp_path):
        """list_cases_by_status on a closed connection should raise RuntimeError."""
        store = CaseStore(str(tmp_path / "test.db"))
        store.close()

        with pytest.raises(RuntimeError):
            store.list_cases_by_status(CaseStatus.OPEN)

    def test_update_status_on_closed_raises(self, tmp_path):
        """update_case_status on a closed connection should raise RuntimeError."""
        store = CaseStore(str(tmp_path / "test.db"))
        store.close()

        with pytest.raises(RuntimeError):
            store.update_case_status("CASE-001", CaseStatus.CLOSED)
