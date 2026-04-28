"""Allegation extraction models for granular triage output.

Defines structured allegation types extracted from cardmember conversation.
Each allegation captures what the CM stated (not conclusions), the detail type,
structured entities, and a confidence score.
"""

from typing import Any

from pydantic import BaseModel, Field

from agentic_fraud_servicing.models.enums import AllegationDetailType


class AllegationExtraction(BaseModel):
    """A single allegation extracted from the conversation.

    Represents what the cardmember alleged — not conclusions about what
    happened. Each allegation has a detail type from the 22-value
    AllegationDetailType taxonomy, a natural-language description,
    structured entities, and a confidence score.
    """

    detail_type: AllegationDetailType
    description: str
    entities: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    context: str | None = None


class AllegationExtractionResult(BaseModel):
    """Result of allegation extraction from a conversation turn or full transcript.

    Contains zero or more extracted allegations. The triage agent produces one
    of these per invocation.
    """

    allegations: list[AllegationExtraction] = Field(default_factory=list)
