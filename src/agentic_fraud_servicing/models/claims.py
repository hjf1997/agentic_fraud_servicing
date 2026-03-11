"""Claim extraction models for granular triage output.

Defines structured claim types extracted from cardmember conversation.
Each claim captures what the CM stated (not conclusions), the claim type,
structured entities, and a confidence score.
"""

from typing import Any

from pydantic import BaseModel, Field

from agentic_fraud_servicing.models.enums import ClaimType


class ClaimExtraction(BaseModel):
    """A single claim extracted from the conversation.

    Represents what the cardmember claimed — not conclusions about what
    happened. Each claim has a type from the 17-value ClaimType taxonomy,
    a natural-language description, structured entities, and a confidence
    score.
    """

    claim_type: ClaimType
    claim_description: str
    entities: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    context: str | None = None


class ClaimExtractionResult(BaseModel):
    """Result of claim extraction from a conversation turn or full transcript.

    Contains zero or more extracted claims. The triage agent produces one
    of these per invocation.
    """

    claims: list[ClaimExtraction] = Field(default_factory=list)
