"""Tests for transcript event models."""

import json

import pytest
from pydantic import ValidationError

from agentic_fraud_servicing.models.enums import SpeakerType
from agentic_fraud_servicing.models.transcript import (
    RedactionInfo,
    TranscriptEvent,
    TranscriptMeta,
)


def _minimal_event() -> TranscriptEvent:
    """Create a TranscriptEvent with only required fields."""
    return TranscriptEvent(
        call_id="call-001",
        event_id="evt-001",
        timestamp_ms=1700000000000,
        speaker=SpeakerType.CARDMEMBER,
        text="I want to report a fraudulent charge.",
    )


class TestRedactionInfo:
    """Tests for RedactionInfo model."""

    def test_defaults(self) -> None:
        info = RedactionInfo()
        assert info.contains_pan is False
        assert info.contains_cvv is False
        assert info.pii_types == []

    def test_populated(self) -> None:
        info = RedactionInfo(contains_pan=True, pii_types=["SSN", "DOB"])
        assert info.contains_pan is True
        assert info.contains_cvv is False
        assert info.pii_types == ["SSN", "DOB"]


class TestTranscriptMeta:
    """Tests for TranscriptMeta model."""

    def test_defaults(self) -> None:
        meta = TranscriptMeta()
        assert meta.channel is None
        assert meta.locale is None
        assert meta.turn_index is None

    def test_populated(self) -> None:
        meta = TranscriptMeta(channel="phone", locale="en-US", turn_index=3)
        assert meta.channel == "phone"
        assert meta.locale == "en-US"
        assert meta.turn_index == 3


class TestTranscriptEvent:
    """Tests for TranscriptEvent model."""

    def test_minimal_required_fields(self) -> None:
        event = _minimal_event()
        assert event.call_id == "call-001"
        assert event.event_id == "evt-001"
        assert event.timestamp_ms == 1700000000000
        assert event.speaker == SpeakerType.CARDMEMBER
        assert event.text == "I want to report a fraudulent charge."
        # Verify defaults
        assert event.confidence == 1.0
        assert event.redaction == RedactionInfo()
        assert event.meta == TranscriptMeta()

    def test_all_fields_populated(self) -> None:
        event = TranscriptEvent(
            call_id="call-002",
            event_id="evt-010",
            timestamp_ms=1700000005000,
            speaker=SpeakerType.CCP,
            text="Let me look into that for you.",
            confidence=0.95,
            redaction=RedactionInfo(contains_pan=True, pii_types=["SSN"]),
            meta=TranscriptMeta(channel="phone", locale="en-US", turn_index=5),
        )
        assert event.speaker == SpeakerType.CCP
        assert event.confidence == 0.95
        assert event.redaction.contains_pan is True
        assert event.meta.turn_index == 5

    def test_invalid_speaker_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            TranscriptEvent(
                call_id="call-001",
                event_id="evt-001",
                timestamp_ms=1700000000000,
                speaker="INVALID_SPEAKER",
                text="Hello",
            )

    def test_model_dump(self) -> None:
        event = _minimal_event()
        d = event.model_dump()
        assert isinstance(d, dict)
        assert d["call_id"] == "call-001"
        assert d["speaker"] == "CARDMEMBER"
        assert d["redaction"] == {
            "contains_pan": False,
            "contains_cvv": False,
            "pii_types": [],
        }

    def test_model_dump_json(self) -> None:
        event = _minimal_event()
        json_str = event.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["event_id"] == "evt-001"
        assert parsed["speaker"] == "CARDMEMBER"

    def test_round_trip_model_dump(self) -> None:
        original = TranscriptEvent(
            call_id="call-003",
            event_id="evt-020",
            timestamp_ms=1700000010000,
            speaker=SpeakerType.SYSTEM,
            text="Call transferred.",
            confidence=0.99,
            redaction=RedactionInfo(contains_cvv=True),
            meta=TranscriptMeta(channel="chat", locale="fr-FR", turn_index=12),
        )
        reconstructed = TranscriptEvent(**original.model_dump())
        assert reconstructed == original

    def test_round_trip_json(self) -> None:
        original = _minimal_event()
        json_str = original.model_dump_json()
        reconstructed = TranscriptEvent.model_validate_json(json_str)
        assert reconstructed == original

    def test_all_speaker_types_accepted(self) -> None:
        for speaker in SpeakerType:
            event = TranscriptEvent(
                call_id="call-001",
                event_id="evt-001",
                timestamp_ms=1700000000000,
                speaker=speaker,
                text="Test",
            )
            assert event.speaker == speaker
