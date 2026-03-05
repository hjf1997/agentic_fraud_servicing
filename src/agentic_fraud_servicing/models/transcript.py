"""Transcript event models for call ingestion.

Defines Pydantic v2 models for transcript events, redaction metadata,
and transcript metadata. These models represent individual utterances
from a call transcript after ASR processing.
"""

from pydantic import BaseModel

from agentic_fraud_servicing.models.enums import SpeakerType


class RedactionInfo(BaseModel):
    """Tracks what sensitive data was redacted from a transcript segment."""

    contains_pan: bool = False
    contains_cvv: bool = False
    pii_types: list[str] = []


class TranscriptMeta(BaseModel):
    """Channel and locale metadata for a transcript event."""

    channel: str | None = None
    locale: str | None = None
    turn_index: int | None = None


class TranscriptEvent(BaseModel):
    """A single utterance from a call transcript.

    Represents one speaker turn after ASR processing and PCI redaction.
    The `speaker` field uses the SpeakerType enum to identify who spoke.
    """

    call_id: str
    event_id: str
    timestamp_ms: int
    speaker: SpeakerType
    text: str
    confidence: float = 1.0
    redaction: RedactionInfo = RedactionInfo()
    meta: TranscriptMeta = TranscriptMeta()
