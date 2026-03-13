"""Domain models and enums for fraud servicing."""

from agentic_fraud_servicing.models.allegations import (
    AllegationExtraction,
    AllegationExtractionResult,
)
from agentic_fraud_servicing.models.case import (
    AuditEntry,
    Case,
    CopilotSuggestion,
    DecisionFactor,
    DecisionRecommendation,
    TimelineEvent,
    TransactionRef,
)
from agentic_fraud_servicing.models.enums import (
    INVESTIGATION_CATEGORIES_REFERENCE,
    AllegationDetailType,
    AllegationType,
    AuthMethod,
    CaseStatus,
    EvidenceEdgeType,
    EvidenceNodeType,
    EvidenceSourceType,
    InvestigationCategory,
    RiskLevel,
    SpeakerType,
    TransactionChannel,
)
from agentic_fraud_servicing.models.evidence import (
    AllegationStatement,
    AuthEvent,
    Card,
    Customer,
    DeliveryProof,
    Device,
    EvidenceEdge,
    EvidenceNode,
    EvidenceRef,
    InvestigatorNote,
    Merchant,
    RefundRecord,
    Transaction,
)
from agentic_fraud_servicing.models.transcript import (
    RedactionInfo,
    TranscriptEvent,
    TranscriptMeta,
)

__all__ = [
    # enums
    "AllegationDetailType",
    "AllegationType",
    "AuthMethod",
    "CaseStatus",
    "EvidenceEdgeType",
    "EvidenceNodeType",
    "EvidenceSourceType",
    "INVESTIGATION_CATEGORIES_REFERENCE",
    "InvestigationCategory",
    "RiskLevel",
    "SpeakerType",
    "TransactionChannel",
    # allegations
    "AllegationExtraction",
    "AllegationExtractionResult",
    # transcript
    "RedactionInfo",
    "TranscriptEvent",
    "TranscriptMeta",
    # evidence
    "AllegationStatement",
    "AuthEvent",
    "Card",
    "Customer",
    "DeliveryProof",
    "Device",
    "EvidenceEdge",
    "EvidenceNode",
    "EvidenceRef",
    "InvestigatorNote",
    "Merchant",
    "RefundRecord",
    "Transaction",
    # case
    "AuditEntry",
    "Case",
    "CopilotSuggestion",
    "DecisionFactor",
    "DecisionRecommendation",
    "TimelineEvent",
    "TransactionRef",
]
