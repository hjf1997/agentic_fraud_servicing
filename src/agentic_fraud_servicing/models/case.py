"""Case, copilot suggestion, and decision recommendation models.

Defines the core case lifecycle models (Case, AuditEntry, TimelineEvent),
decision support models (DecisionRecommendation, DecisionFactor), and
realtime copilot output (CopilotSuggestion). All categorical fields use
enums from models.enums.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from agentic_fraud_servicing.models.enums import (
    AllegationType,
    CaseStatus,
    InvestigationCategory,
)


class TransactionRef(BaseModel):
    """Reference to a transaction in scope for a case."""

    transaction_id: str
    amount: float
    merchant_name: str
    transaction_date: datetime


class TimelineEvent(BaseModel):
    """A timestamped event in the case timeline."""

    timestamp: datetime
    event_type: str
    description: str
    source: str | None = None


class AuditEntry(BaseModel):
    """Immutable audit log entry for case actions."""

    timestamp: datetime
    action: str
    agent_id: str | None = None
    details: str = ""


class DecisionFactor(BaseModel):
    """A weighted factor supporting a decision recommendation."""

    factor: str
    evidence_ref: str
    weight: float


class DecisionRecommendation(BaseModel):
    """Recommendation produced by the post-call investigator."""

    category: InvestigationCategory
    confidence: float
    top_factors: list[DecisionFactor] = []
    uncertainties: list[str] = []
    suggested_actions: list[str] = []
    required_approvals: list[str] = []


class Case(BaseModel):
    """Central case entity tracking a fraud servicing investigation.

    Aggregates allegations, evidence references, timeline events,
    decision recommendations, and a full audit trail. Status defaults
    to OPEN and progresses through the CaseStatus lifecycle.
    """

    case_id: str
    call_id: str
    customer_id: str
    account_id: str
    allegation_type: AllegationType | None = None
    allegation_confidence: float = 0.0
    impersonation_risk: float = 0.0
    transactions_in_scope: list[TransactionRef] = []
    timeline: list[TimelineEvent] = []
    evidence_refs: list[str] = []
    decision_recommendation: DecisionRecommendation | None = None
    status: CaseStatus = CaseStatus.OPEN
    audit_trail: list[AuditEntry] = []
    created_at: datetime
    updated_at: datetime | None = None


class ProbingQuestion(BaseModel):
    """A probing question with lifecycle tracking.

    Tracks each suggested question from creation through resolution.
    Status transitions: pending → answered (CM addressed the topic),
    pending → invalidated (target hypothesis collapsed, evidence
    resolved the gap, or conversation moved past relevance), or
    pending → skipped (CCP chose not to ask within the staleness window).
    """

    text: str
    status: Literal["pending", "answered", "invalidated", "skipped"] = "pending"
    turn_suggested: int = 0
    assessment_suggested: int = 0
    """Which assessment cycle this question was created in."""
    target_category: str = ""
    """Which investigation category this question helps discriminate."""
    reason: str = ""
    """Why answered/invalidated/skipped (empty when pending)."""
    turn_resolved: int | None = None


class CopilotSuggestion(BaseModel):
    """Output from the realtime copilot for a single transcript turn.

    Contains suggested questions, risk flags, retrieved facts, and
    running hypothesis scores for the CCP to act on during the call.
    """

    call_id: str
    timestamp_ms: int
    suggested_questions: list[str] = []
    """All currently pending probing questions (what CCP should ask now)."""
    probing_questions: list[dict] = []
    """Full question list snapshot with lifecycle statuses at this turn."""
    risk_flags: list[str] = []
    retrieved_facts: list[str] = []
    running_summary: str = ""
    safety_guidance: str = ""
    hypothesis_scores: dict[str, float] = {}
    impersonation_risk: float = 0.0
    specialist_likelihoods: dict[str, float] = {}
    case_eligibility: list[dict] = []
    case_advisory_summary: str = ""
    information_sufficient: bool = False
