"""Unit tests for TraceStore agent audit log storage."""

from datetime import datetime, timezone

import pytest

from agentic_fraud_servicing.storage.trace_store import TraceStore


@pytest.fixture
def store(tmp_path):
    """Create a fresh TraceStore for each test."""
    db_path = str(tmp_path / "traces.db")
    s = TraceStore(db_path)
    yield s
    s.close()


def _make_timestamp(hour: int = 12, minute: int = 0) -> datetime:
    """Helper to create a UTC timestamp with a specific hour/minute."""
    return datetime(2026, 3, 6, hour, minute, 0, tzinfo=timezone.utc)


class TestTraceStoreInit:
    """Tests for TraceStore initialization."""

    def test_wal_mode_enabled(self, store):
        """WAL journal mode should be active."""
        row = store._conn.execute("PRAGMA journal_mode").fetchone()
        assert dict(row)["journal_mode"] == "wal"

    def test_parent_dir_creation(self, tmp_path):
        """Parent directories should be created if they don't exist."""
        db_path = str(tmp_path / "nested" / "deep" / "traces.db")
        s = TraceStore(db_path)
        s.close()
        assert (tmp_path / "nested" / "deep" / "traces.db").exists()


class TestLogAndGetTrace:
    """Tests for log_invocation and get_trace."""

    def test_round_trip(self, store):
        """Logged trace should be retrievable with all fields intact."""
        ts = _make_timestamp(10, 30)
        store.log_invocation(
            trace_id="t-001",
            case_id="case-abc",
            agent_id="triage_agent",
            action="classify_claim",
            input_data='{"text": "I was charged twice"}',
            output_data='{"category": "dispute"}',
            duration_ms=245.5,
            timestamp=ts,
            status="success",
        )

        result = store.get_trace("t-001")
        assert result is not None
        assert result["trace_id"] == "t-001"
        assert result["case_id"] == "case-abc"
        assert result["agent_id"] == "triage_agent"
        assert result["action"] == "classify_claim"
        assert result["input_data"] == '{"text": "I was charged twice"}'
        assert result["output_data"] == '{"category": "dispute"}'
        assert result["duration_ms"] == 245.5
        assert result["timestamp"] == ts.isoformat()
        assert result["status"] == "success"

    def test_get_trace_nonexistent_returns_none(self, store):
        """Getting a nonexistent trace should return None."""
        assert store.get_trace("nonexistent") is None

    def test_duplicate_trace_id_raises(self, store):
        """Inserting a duplicate trace_id should raise RuntimeError."""
        ts = _make_timestamp()
        kwargs = dict(
            trace_id="t-dup",
            case_id="case-1",
            agent_id="agent-1",
            action="test",
            input_data="{}",
            output_data="{}",
            duration_ms=100.0,
            timestamp=ts,
        )
        store.log_invocation(**kwargs)
        with pytest.raises(RuntimeError, match="already exists"):
            store.log_invocation(**kwargs)

    def test_all_fields_stored(self, store):
        """All 9 required fields should be present in the returned dict."""
        ts = _make_timestamp(14, 15)
        store.log_invocation(
            trace_id="t-fields",
            case_id="case-f",
            agent_id="auth_agent",
            action="verify_identity",
            input_data='{"customer_id": "c123"}',
            output_data='{"risk": 0.3}',
            duration_ms=89.2,
            timestamp=ts,
            status="error",
        )

        result = store.get_trace("t-fields")
        expected_keys = {
            "trace_id",
            "case_id",
            "agent_id",
            "action",
            "input_data",
            "output_data",
            "duration_ms",
            "timestamp",
            "status",
        }
        assert set(result.keys()) == expected_keys

    def test_default_status_is_success(self, store):
        """Status should default to 'success' when not specified."""
        ts = _make_timestamp()
        store.log_invocation(
            trace_id="t-default",
            case_id="case-d",
            agent_id="agent-d",
            action="act",
            input_data="{}",
            output_data="{}",
            duration_ms=50.0,
            timestamp=ts,
        )
        result = store.get_trace("t-default")
        assert result["status"] == "success"


class TestGetTracesByCase:
    """Tests for get_traces_by_case."""

    def test_returns_ordered_by_timestamp(self, store):
        """Traces should be returned in ascending timestamp order."""
        for i, hour in enumerate([15, 10, 12]):
            store.log_invocation(
                trace_id=f"t-ord-{i}",
                case_id="case-order",
                agent_id="agent-x",
                action="step",
                input_data="{}",
                output_data="{}",
                duration_ms=100.0,
                timestamp=_make_timestamp(hour),
            )

        results = store.get_traces_by_case("case-order")
        assert len(results) == 3
        # Should be ordered: 10:00, 12:00, 15:00
        assert results[0]["trace_id"] == "t-ord-1"
        assert results[1]["trace_id"] == "t-ord-2"
        assert results[2]["trace_id"] == "t-ord-0"

    def test_filters_by_case_id(self, store):
        """Only traces matching the case_id should be returned."""
        ts = _make_timestamp()
        for case_id, trace_id in [("case-a", "t-a"), ("case-b", "t-b"), ("case-a", "t-a2")]:
            store.log_invocation(
                trace_id=trace_id,
                case_id=case_id,
                agent_id="agent",
                action="act",
                input_data="{}",
                output_data="{}",
                duration_ms=10.0,
                timestamp=ts,
            )

        results = store.get_traces_by_case("case-a")
        assert len(results) == 2
        assert all(r["case_id"] == "case-a" for r in results)

    def test_no_results_returns_empty_list(self, store):
        """Querying a case with no traces should return an empty list."""
        assert store.get_traces_by_case("nonexistent-case") == []


class TestErrorWrapping:
    """Tests for sqlite3 error wrapping."""

    def test_log_invocation_on_closed_raises(self, store):
        """log_invocation on a closed store should raise RuntimeError."""
        store.close()
        with pytest.raises(RuntimeError):
            store.log_invocation(
                trace_id="t-fail",
                case_id="case-f",
                agent_id="agent",
                action="act",
                input_data="{}",
                output_data="{}",
                duration_ms=10.0,
                timestamp=_make_timestamp(),
            )

    def test_get_traces_on_closed_raises(self, store):
        """get_traces_by_case on a closed store should raise RuntimeError."""
        store.close()
        with pytest.raises(RuntimeError):
            store.get_traces_by_case("case-001")

    def test_get_trace_on_closed_raises(self, store):
        """get_trace on a closed store should raise RuntimeError."""
        store.close()
        with pytest.raises(RuntimeError):
            store.get_trace("t-001")
