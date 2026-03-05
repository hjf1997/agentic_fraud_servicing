"""Transcript parsing with automatic PII redaction.

Converts raw JSON transcript data into validated, redacted TranscriptEvent
models. Redaction is baked into the parsing pipeline — every event passes
through redact_all() before becoming a TranscriptEvent.
"""

import json

from agentic_fraud_servicing.ingestion.redaction import redact_all
from agentic_fraud_servicing.models.enums import SpeakerType
from agentic_fraud_servicing.models.transcript import TranscriptEvent, TranscriptMeta

_REQUIRED_KEYS = ("call_id", "event_id", "timestamp_ms", "speaker", "text")


def parse_transcript_event(raw: dict) -> TranscriptEvent:
    """Parse a single raw transcript event dict into a TranscriptEvent.

    Validates required keys, runs text through PII redaction, and constructs
    a TranscriptEvent with the redacted text and populated redaction field.

    Args:
        raw: Dict with keys call_id, event_id, timestamp_ms, speaker, text
            (required) and confidence, meta (optional).

    Returns:
        A TranscriptEvent with redacted text and RedactionInfo.

    Raises:
        ValueError: If required keys are missing.
        RedactionError: If redact_all encounters malformed input.
    """
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(f"Missing required keys: {', '.join(missing)}")

    # Redact PII from transcript text before it reaches the model
    redacted_text, redaction_info = redact_all(raw["text"])

    # Build optional meta if provided
    meta = TranscriptMeta(**raw["meta"]) if raw.get("meta") else TranscriptMeta()

    return TranscriptEvent(
        call_id=raw["call_id"],
        event_id=raw["event_id"],
        timestamp_ms=raw["timestamp_ms"],
        speaker=SpeakerType(raw["speaker"]),
        text=redacted_text,
        confidence=raw.get("confidence", 1.0),
        redaction=redaction_info,
        meta=meta,
    )


def parse_transcript_batch(raw_events: list[dict]) -> list[TranscriptEvent]:
    """Parse a list of raw event dicts into TranscriptEvent models.

    Args:
        raw_events: List of raw event dicts.

    Returns:
        List of TranscriptEvent models with redacted text.

    Raises:
        ValueError: If raw_events is not a list.
    """
    if not isinstance(raw_events, list):
        raise ValueError(f"Expected list, got {type(raw_events).__name__}")

    return [parse_transcript_event(event) for event in raw_events]


def parse_transcript_json(json_str: str) -> list[TranscriptEvent]:
    """Parse a JSON string into TranscriptEvent models.

    Accepts either a single event dict or a list of event dicts as JSON.

    Args:
        json_str: JSON string containing one event dict or a list of dicts.

    Returns:
        List of TranscriptEvent models (always a list, even for single events).

    Raises:
        ValueError: If json_str is invalid JSON or not a dict/list.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if isinstance(data, dict):
        return [parse_transcript_event(data)]
    if isinstance(data, list):
        return parse_transcript_batch(data)

    raise ValueError(f"Expected JSON object or array, got {type(data).__name__}")
