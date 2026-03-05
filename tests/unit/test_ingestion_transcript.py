"""Tests for ingestion/transcript.py — transcript parsing with redaction."""

import json

import pytest

from agentic_fraud_servicing.ingestion.transcript import (
    parse_transcript_batch,
    parse_transcript_event,
    parse_transcript_json,
)
from agentic_fraud_servicing.models.enums import SpeakerType
from agentic_fraud_servicing.models.transcript import TranscriptEvent


def _make_raw(
    text: str = "Hello, how can I help?",
    speaker: str = "CARDMEMBER",
    **overrides,
) -> dict:
    """Build a minimal raw event dict with optional overrides."""
    base = {
        "call_id": "call-001",
        "event_id": "evt-001",
        "timestamp_ms": 1000,
        "speaker": speaker,
        "text": text,
    }
    base.update(overrides)
    return base


# --- parse_transcript_event ---


class TestParseTranscriptEvent:
    """Tests for parse_transcript_event."""

    def test_required_fields_only(self):
        """Parse event with just required fields — defaults applied."""
        raw = _make_raw()
        event = parse_transcript_event(raw)

        assert isinstance(event, TranscriptEvent)
        assert event.call_id == "call-001"
        assert event.event_id == "evt-001"
        assert event.timestamp_ms == 1000
        assert event.speaker == SpeakerType.CARDMEMBER
        assert event.text == "Hello, how can I help?"
        assert event.confidence == 1.0

    def test_optional_fields(self):
        """Parse event with confidence and meta provided."""
        raw = _make_raw(
            confidence=0.95,
            meta={"channel": "phone", "locale": "en-US", "turn_index": 3},
        )
        event = parse_transcript_event(raw)

        assert event.confidence == 0.95
        assert event.meta.channel == "phone"
        assert event.meta.locale == "en-US"
        assert event.meta.turn_index == 3

    def test_missing_required_key_raises(self):
        """Missing required key raises ValueError with key name."""
        raw = {"call_id": "call-001", "event_id": "evt-001"}
        with pytest.raises(ValueError, match="Missing required keys.*timestamp_ms"):
            parse_transcript_event(raw)

    def test_pan_redacted(self):
        """PAN in text is replaced with [PAN_REDACTED]."""
        raw = _make_raw(text="Card number is 378282246310005")
        event = parse_transcript_event(raw)

        assert "[PAN_REDACTED]" in event.text
        assert "378282246310005" not in event.text
        assert event.redaction.contains_pan is True

    def test_cvv_redacted(self):
        """CVV following keyword is replaced with [CVV_REDACTED]."""
        raw = _make_raw(text="My CVV: 123")
        event = parse_transcript_event(raw)

        assert "[CVV_REDACTED]" in event.text
        assert event.redaction.contains_cvv is True

    def test_multiple_pii_types(self):
        """Multiple PII types detected and redacted in a single event."""
        raw = _make_raw(text="Card 4111111111111111, SSN 123-45-6789, DOB: 01/15/1990")
        event = parse_transcript_event(raw)

        assert "[PAN_REDACTED]" in event.text
        assert "[SSN_REDACTED]" in event.text
        assert "[DOB_REDACTED]" in event.text
        assert event.redaction.contains_pan is True
        assert "SSN" in event.redaction.pii_types
        assert "DOB" in event.redaction.pii_types

    def test_no_pii_text_unchanged(self):
        """Text without PII passes through unchanged."""
        raw = _make_raw(text="I need help with my account")
        event = parse_transcript_event(raw)

        assert event.text == "I need help with my account"
        assert event.redaction.contains_pan is False
        assert event.redaction.contains_cvv is False
        assert event.redaction.pii_types == []

    def test_all_speaker_types(self):
        """All SpeakerType values are accepted."""
        for speaker in SpeakerType:
            raw = _make_raw(speaker=speaker.value)
            event = parse_transcript_event(raw)
            assert event.speaker == speaker


# --- parse_transcript_batch ---


class TestParseTranscriptBatch:
    """Tests for parse_transcript_batch."""

    def test_batch_multiple_events(self):
        """Parse a batch of multiple events."""
        events = [
            _make_raw(event_id="evt-001", text="First message"),
            _make_raw(event_id="evt-002", text="Second message", speaker="CCP"),
        ]
        result = parse_transcript_batch(events)

        assert len(result) == 2
        assert result[0].event_id == "evt-001"
        assert result[1].event_id == "evt-002"
        assert result[1].speaker == SpeakerType.CCP

    def test_batch_non_list_raises(self):
        """Non-list input raises ValueError."""
        with pytest.raises(ValueError, match="Expected list"):
            parse_transcript_batch({"not": "a list"})

    def test_batch_empty_list(self):
        """Empty list returns empty result."""
        assert parse_transcript_batch([]) == []


# --- parse_transcript_json ---


class TestParseTranscriptJson:
    """Tests for parse_transcript_json."""

    def test_json_single_event(self):
        """JSON string with single event dict returns one-element list."""
        raw = _make_raw()
        result = parse_transcript_json(json.dumps(raw))

        assert len(result) == 1
        assert result[0].call_id == "call-001"

    def test_json_event_list(self):
        """JSON string with event list returns matching list."""
        events = [_make_raw(event_id="evt-001"), _make_raw(event_id="evt-002")]
        result = parse_transcript_json(json.dumps(events))

        assert len(result) == 2

    def test_json_invalid_raises(self):
        """Invalid JSON string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_transcript_json("{not valid json}")

    def test_json_with_pan(self):
        """PAN in JSON-parsed event is redacted."""
        raw = _make_raw(text="My card is 4111111111111111")
        result = parse_transcript_json(json.dumps(raw))

        assert "[PAN_REDACTED]" in result[0].text
        assert "4111111111111111" not in result[0].text
