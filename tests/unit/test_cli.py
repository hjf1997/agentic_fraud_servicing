"""Tests for the CLI interface (ui/cli.py).

Covers argument parsing, subcommand dispatch, JSON/text output formatting,
and error handling. All LLM calls are mocked — no real provider needed.
"""

import argparse
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_fraud_servicing.investigator.case_writer import CasePack
from agentic_fraud_servicing.models.case import Case, CopilotSuggestion
from agentic_fraud_servicing.models.enums import AllegationType, CaseStatus
from agentic_fraud_servicing.models.transcript import TranscriptEvent
from agentic_fraud_servicing.ui.cli import (
    _format_case_pack_text,
    _format_suggestion_text,
    build_parser,
    cmd_investigate,
    cmd_simulate,
    cmd_view_case,
)

# -- Fixtures --


def _make_suggestion() -> CopilotSuggestion:
    """Create a minimal CopilotSuggestion for testing."""
    return CopilotSuggestion(
        call_id="call-001",
        timestamp_ms=1000,
        suggested_questions=["What is the merchant name?"],
        risk_flags=["low confidence"],
        retrieved_facts=["3 transactions found"],
        running_summary="Possible fraud claim.",
        safety_guidance="Never ask for full PAN or CVV.",
        hypothesis_scores={"FRAUD": 0.7, "DISPUTE": 0.2, "SCAM": 0.1},
        impersonation_risk=0.15,
    )


def _make_case_pack() -> CasePack:
    """Create a minimal CasePack for testing."""
    return CasePack(
        case_summary="Fraud case with contradictions.",
        timeline=[{"timestamp": "2024-01-01", "description": "Card used at merchant"}],
        evidence_list=[{"node_id": "n1", "node_type": "TRANSACTION"}],
        decision_recommendation={"category": "fraud", "confidence": 0.85},
        investigation_notes=["Merchant conflict detected"],
    )


def _make_event() -> TranscriptEvent:
    """Create a minimal TranscriptEvent for testing."""
    return TranscriptEvent(
        call_id="call-001",
        event_id="evt-001",
        timestamp_ms=1000,
        speaker="CCP",
        text="Hello, how can I help?",
    )


def _make_case() -> Case:
    """Create a minimal Case for testing."""
    return Case(
        case_id="case-001",
        call_id="call-001",
        customer_id="cust-001",
        account_id="acct-001",
        allegation_type=AllegationType.FRAUD,
        status=CaseStatus.OPEN,
        created_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
    )


# -- Parser tests --


class TestBuildParser:
    """Test argument parser construction and parsing."""

    def test_simulate_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["simulate", "-t", "file.json", "-d", "/tmp/db", "-o", "text"])
        assert args.command == "simulate"
        assert args.transcript == "file.json"
        assert args.db_dir == "/tmp/db"
        assert args.output == "text"

    def test_investigate_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["investigate", "-c", "case-123", "-o", "json"])
        assert args.command == "investigate"
        assert args.case_id == "case-123"
        assert args.output == "json"

    def test_view_case_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["view-case", "-c", "case-456", "-d", "/data/x"])
        assert args.command == "view-case"
        assert args.case_id == "case-456"
        assert args.db_dir == "/data/x"

    def test_default_db_dir(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["simulate", "-t", "f.json"])
        assert args.db_dir == "data/cli"

    def test_default_output_json(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["investigate", "-c", "case-1"])
        assert args.output == "json"


# -- Text formatter tests --


class TestFormatSuggestionText:
    """Test text formatting for CopilotSuggestion."""

    def test_contains_key_sections(self) -> None:
        text = _format_suggestion_text(_make_suggestion())
        assert "Copilot Suggestion" in text
        assert "Summary:" in text
        assert "Impersonation Risk:" in text
        assert "Hypothesis Scores:" in text
        assert "Suggested Questions:" in text
        assert "Risk Flags:" in text
        assert "Safety:" in text


class TestFormatCasePackText:
    """Test text formatting for CasePack."""

    def test_contains_key_sections(self) -> None:
        text = _format_case_pack_text(_make_case_pack())
        assert "Case Pack" in text
        assert "Summary:" in text
        assert "Timeline:" in text
        assert "Evidence Items:" in text
        assert "Decision:" in text
        assert "Notes:" in text


# -- Subcommand handler tests --


class TestCmdSimulate:
    """Test the simulate subcommand handler."""

    @pytest.mark.asyncio
    async def test_simulate_json_output(self, tmp_path, capsys) -> None:
        """Simulate prints JSON for each transcript event."""
        suggestion = _make_suggestion()
        mock_orch = MagicMock()
        mock_orch.process_event = AsyncMock(return_value=suggestion)

        args = argparse.Namespace(
            transcript="transcript.json",
            db_dir=str(tmp_path / "db"),
            output="json",
        )
        events = [_make_event(), _make_event()]

        with (
            patch("agentic_fraud_servicing.ui.cli.load_transcript_file", return_value=events),
            patch("agentic_fraud_servicing.ui.cli.create_gateway"),
            patch("agentic_fraud_servicing.ui.cli.create_provider"),
            patch(
                "agentic_fraud_servicing.ui.cli.CopilotOrchestrator",
                return_value=mock_orch,
            ),
        ):
            await cmd_simulate(args)

        out = capsys.readouterr().out
        # Should contain two JSON blocks (one per event)
        assert out.count('"call_id"') == 2
        # Each should be valid JSON
        blocks = [b.strip() for b in out.strip().split("\n\n") if b.strip()]
        for block in blocks:
            parsed = json.loads(block)
            assert parsed["call_id"] == "call-001"

    @pytest.mark.asyncio
    async def test_simulate_file_not_found(self, tmp_path) -> None:
        """Simulate exits with code 1 if transcript file is missing."""
        args = argparse.Namespace(
            transcript="/nonexistent/file.json",
            db_dir=str(tmp_path / "db"),
            output="json",
        )
        with (
            patch(
                "agentic_fraud_servicing.ui.cli.load_transcript_file",
                side_effect=FileNotFoundError("not found"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            await cmd_simulate(args)
        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_simulate_text_output(self, tmp_path, capsys) -> None:
        """Simulate prints text output when --output text is used."""
        suggestion = _make_suggestion()
        mock_orch = MagicMock()
        mock_orch.process_event = AsyncMock(return_value=suggestion)

        args = argparse.Namespace(
            transcript="transcript.json",
            db_dir=str(tmp_path / "db"),
            output="text",
        )

        with (
            patch(
                "agentic_fraud_servicing.ui.cli.load_transcript_file",
                return_value=[_make_event()],
            ),
            patch("agentic_fraud_servicing.ui.cli.create_gateway"),
            patch("agentic_fraud_servicing.ui.cli.create_provider"),
            patch(
                "agentic_fraud_servicing.ui.cli.CopilotOrchestrator",
                return_value=mock_orch,
            ),
        ):
            await cmd_simulate(args)

        out = capsys.readouterr().out
        assert "Copilot Suggestion" in out


class TestCmdInvestigate:
    """Test the investigate subcommand handler."""

    @pytest.mark.asyncio
    async def test_investigate_json_output(self, tmp_path, capsys) -> None:
        """Investigate prints CasePack as JSON."""
        case_pack = _make_case_pack()
        mock_orch = MagicMock()
        mock_orch.investigate = AsyncMock(return_value=case_pack)

        args = argparse.Namespace(
            case_id="case-001",
            db_dir=str(tmp_path / "db"),
            output="json",
        )

        with (
            patch("agentic_fraud_servicing.ui.cli.create_gateway"),
            patch("agentic_fraud_servicing.ui.cli.create_provider"),
            patch(
                "agentic_fraud_servicing.ui.cli.InvestigatorOrchestrator",
                return_value=mock_orch,
            ),
        ):
            await cmd_investigate(args)

        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["case_summary"] == "Fraud case with contradictions."

    @pytest.mark.asyncio
    async def test_investigate_case_not_found(self, tmp_path) -> None:
        """Investigate exits with code 1 if the case is not found."""
        mock_orch = MagicMock()
        mock_orch.investigate = AsyncMock(side_effect=RuntimeError("Case not found: case-999"))

        args = argparse.Namespace(
            case_id="case-999",
            db_dir=str(tmp_path / "db"),
            output="json",
        )

        with (
            patch("agentic_fraud_servicing.ui.cli.create_gateway"),
            patch("agentic_fraud_servicing.ui.cli.create_provider"),
            patch(
                "agentic_fraud_servicing.ui.cli.InvestigatorOrchestrator",
                return_value=mock_orch,
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            await cmd_investigate(args)
        assert exc_info.value.code == 1


class TestCmdViewCase:
    """Test the view-case subcommand handler."""

    def test_view_case_found(self, tmp_path, capsys) -> None:
        """View-case prints case JSON when the case exists."""
        from agentic_fraud_servicing.storage.case_store import CaseStore

        store = CaseStore(tmp_path / "cases.db")
        case = _make_case()
        store.create_case(case)
        store.close()

        mock_gw = MagicMock()
        mock_gw.case_store.get_case.return_value = case

        args = argparse.Namespace(case_id="case-001", db_dir=str(tmp_path))

        with patch("agentic_fraud_servicing.ui.cli.create_gateway", return_value=mock_gw):
            cmd_view_case(args)

        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["case_id"] == "case-001"
        assert parsed["allegation_type"] == AllegationType.FRAUD.value

    def test_view_case_not_found(self, tmp_path, capsys) -> None:
        """View-case exits with code 1 when the case does not exist."""
        mock_gw = MagicMock()
        mock_gw.case_store.get_case.return_value = None

        args = argparse.Namespace(case_id="case-missing", db_dir=str(tmp_path))

        with (
            patch("agentic_fraud_servicing.ui.cli.create_gateway", return_value=mock_gw),
            pytest.raises(SystemExit) as exc_info,
        ):
            cmd_view_case(args)
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "case not found" in err
