"""Realtime copilot agents for during-call assistance."""

from agentic_fraud_servicing.copilot.auth_agent import (
    AuthAssessment,
    auth_agent,
    run_auth_assessment,
)
from agentic_fraud_servicing.copilot.context import (
    CopilotContext,
    tool_fetch_customer_profile,
    tool_lookup_transactions,
    tool_query_auth_logs,
)
from agentic_fraud_servicing.copilot.orchestrator import CopilotOrchestrator
from agentic_fraud_servicing.copilot.question_planner import (
    QuestionPlan,
    question_agent,
    run_question_planner,
)
from agentic_fraud_servicing.copilot.retrieval_agent import (
    RetrievalResult,
    retrieval_agent,
    run_retrieval,
)
from agentic_fraud_servicing.copilot.triage_agent import (
    TriageResult,
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
    "TriageResult",
    "triage_agent",
    "run_triage",
    # auth_agent
    "AuthAssessment",
    "auth_agent",
    "run_auth_assessment",
    # question_planner
    "QuestionPlan",
    "question_agent",
    "run_question_planner",
    # retrieval_agent
    "RetrievalResult",
    "retrieval_agent",
    "run_retrieval",
    # orchestrator
    "CopilotOrchestrator",
]
