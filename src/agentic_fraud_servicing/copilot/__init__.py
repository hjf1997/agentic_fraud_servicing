"""Realtime copilot agents for during-call assistance."""

from agentic_fraud_servicing.copilot.auth_agent import (
    AuthAssessment,
    auth_agent,
    run_auth_assessment,
)
from agentic_fraud_servicing.copilot.case_advisor import (
    CaseAdvisory,
    CaseTypeAssessment,
    case_advisor,
    load_policies,
    run_case_advisor,
)
from agentic_fraud_servicing.copilot.context import (
    CopilotContext,
    tool_fetch_customer_profile,
    tool_lookup_transactions,
    tool_query_auth_logs,
)
from agentic_fraud_servicing.copilot.hypothesis_agent import (
    HypothesisAssessment,
    HypothesisReasoning,
    ReasoningNoteUpdate,
    hypothesis_reasoning_agent,
    merge_reasoning_notes,
    run_arbitrator,
)
from agentic_fraud_servicing.copilot.hypothesis_specialists import (
    SpecialistAssessment,
    SpecialistNoteUpdate,
    merge_specialist_notes,
    run_specialists,
)
from agentic_fraud_servicing.copilot.orchestrator import CopilotOrchestrator
from agentic_fraud_servicing.copilot.retrieval_agent import (
    RetrievalResult,
    retrieval_agent,
    run_retrieval,
)
from agentic_fraud_servicing.copilot.triage_agent import (
    AllegationExtractionResult,
    run_triage,
    triage_agent,
)

__all__ = [
    # context
    "CopilotContext",
    "tool_lookup_transactions",
    "tool_query_auth_logs",
    "tool_fetch_customer_profile",
    # triage_agent
    "AllegationExtractionResult",
    "triage_agent",
    "run_triage",
    # auth_agent
    "AuthAssessment",
    "auth_agent",
    "run_auth_assessment",
    # retrieval_agent
    "RetrievalResult",
    "retrieval_agent",
    "run_retrieval",
    # hypothesis_agent
    "HypothesisAssessment",
    "HypothesisReasoning",
    "ReasoningNoteUpdate",
    "hypothesis_reasoning_agent",
    "merge_reasoning_notes",
    "run_arbitrator",
    # hypothesis_specialists
    "SpecialistAssessment",
    "SpecialistNoteUpdate",
    "merge_specialist_notes",
    "run_specialists",
    # case_advisor
    "CaseTypeAssessment",
    "CaseAdvisory",
    "case_advisor",
    "run_case_advisor",
    "load_policies",
    # orchestrator
    "CopilotOrchestrator",
]
