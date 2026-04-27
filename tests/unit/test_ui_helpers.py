"""Tests for UI helpers module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentic_fraud_servicing.models.case import CopilotSuggestion
from agentic_fraud_servicing.models.transcript import TranscriptEvent
from agentic_fraud_servicing.ui.helpers import (
    create_gateway,
    create_provider,
    format_suggestion_json,
    load_transcript_file,
)


class TestCreateGateway:
    """Tests for the create_gateway helper."""

    def test_creates_stores_and_gateway(self, tmp_path: Path) -> None:
        """Gateway is returned with all three stores accessible."""
        gateway = create_gateway(tmp_path / "db")
        assert gateway.case_store is not None
        assert gateway.evidence_store is not None
        assert gateway.trace_store is not None
        # Clean up
        gateway.case_store.close()
        gateway.evidence_store.close()
        gateway.trace_store.close()

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        """Parent directories are created when they don't exist."""
        db_dir = tmp_path / "nested" / "deep" / "db"
        assert not db_dir.exists()
        gateway = create_gateway(db_dir)
        assert db_dir.exists()
        gateway.case_store.close()
        gateway.evidence_store.close()
        gateway.trace_store.close()


class TestCreateProvider:
    """Tests for the create_provider helper."""

    @patch("agentic_fraud_servicing.ui.helpers.get_model_provider")
    @patch("agentic_fraud_servicing.ui.helpers.get_settings")
    def test_returns_provider(self, mock_settings: MagicMock, mock_provider: MagicMock) -> None:
        """Provider is created from settings."""
        mock_settings.return_value = MagicMock()
        mock_provider.return_value = MagicMock()
        result = create_provider()
        mock_settings.assert_called_once()
        mock_provider.assert_called_once_with(mock_settings.return_value)
        assert result is mock_provider.return_value


class TestLoadTranscriptFile:
    """Tests for the load_transcript_file helper."""

    def test_loads_transcript_events(self, tmp_path: Path) -> None:
        """Loads a valid transcript JSON file into TranscriptEvent list."""
        data = [
            {
                "call_id": "c1",
                "event_id": "e1",
                "timestamp_ms": 1000,
                "speaker": "CCP",
                "text": "Hello, how can I help you?",
            }
        ]
        path = tmp_path / "transcript.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        events = load_transcript_file(path)
        assert len(events) == 1
        assert isinstance(events[0], TranscriptEvent)
        assert events[0].call_id == "c1"

    def test_redacts_pan(self, tmp_path: Path) -> None:
        """PAN in transcript text is redacted after loading."""
        data = [
            {
                "call_id": "c1",
                "event_id": "e1",
                "timestamp_ms": 1000,
                "speaker": "CARDMEMBER",
                "text": "My card number is 371449635398431.",
            }
        ]
        path = tmp_path / "transcript.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        events = load_transcript_file(path)
        assert "371449635398431" not in events[0].text
        assert "[PAN_REDACTED]" in events[0].text
        assert events[0].redaction.contains_pan is True

    def test_file_not_found(self) -> None:
        """Raises FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_transcript_file("/nonexistent/path/transcript.json")


class TestFormatSuggestionJson:
    """Tests for format_suggestion_json."""

    def test_formats_as_valid_json(self) -> None:
        """CopilotSuggestion is serialized to valid indented JSON."""
        suggestion = CopilotSuggestion(
            call_id="c1",
            timestamp_ms=1000,
            suggested_questions=["What happened?"],
            risk_flags=["high_amount"],
        )
        result = format_suggestion_json(suggestion)
        parsed = json.loads(result)
        assert parsed["call_id"] == "c1"
        assert parsed["suggested_questions"] == ["What happened?"]


