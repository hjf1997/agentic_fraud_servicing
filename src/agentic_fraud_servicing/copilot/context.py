"""Shared context dataclass and tool wrappers for copilot agents.

CopilotContext holds the running state for a copilot session and is passed
to Runner.run(context=...). Tool wrappers bridge the existing gateway read
tools to the Agents SDK @function_tool interface.
"""

import json
from dataclasses import dataclass, field

from agents import RunContextWrapper, function_tool

from agentic_fraud_servicing.gateway.tool_gateway import AuthContext, ToolGateway
from agentic_fraud_servicing.gateway.tools.read_tools import (
    fetch_customer_profile,
    lookup_transactions,
    query_auth_logs,
)
from agentic_fraud_servicing.ingestion.firewall_redactor import FirewallRedactor
from agentic_fraud_servicing.models.transcript import TranscriptEvent

# Shared redactor for tool responses — uses safe patterns only via redact_dict
_tool_redactor = FirewallRedactor()

# Fields to strip from evidence dicts before sending to the LLM.
# These are internal plumbing fields with no investigative value.
_STRIP_COMMON = {"node_id", "case_id", "node_type", "source_type", "created_at"}
_STRIP_TRANSACTION = _STRIP_COMMON | {"merchant_id"}
_STRIP_AUTH_EVENT = _STRIP_COMMON
_STRIP_CUSTOMER = _STRIP_COMMON | {"profile_hash"}


def _strip_fields(records: list[dict], strip: set[str]) -> list[dict]:
    """Remove internal fields from evidence dicts for LLM consumption."""
    return [{k: v for k, v in rec.items() if k not in strip} for rec in records]


@dataclass
class CopilotContext:
    """Shared running state for a copilot session.

    Passed to Runner.run(context=...) and accessed in @function_tool
    functions via RunContextWrapper[CopilotContext].

    Attributes:
        case_id: Current case being worked.
        call_id: Current call identifier.
        gateway: ToolGateway instance for mediated data access.
        hypothesis_scores: Running scores for fraud/dispute/scam hypotheses.
        impersonation_risk: Current impersonation risk score (0.0-1.0).
        missing_fields: Fields still needed from the caller.
        evidence_collected: Evidence references gathered so far.
        transcript_history: Transcript events processed so far.
    """

    case_id: str
    call_id: str
    gateway: ToolGateway
    hypothesis_scores: dict[str, float] = field(default_factory=dict)
    impersonation_risk: float = 0.0
    missing_fields: list[str] = field(default_factory=list)
    evidence_collected: list[str] = field(default_factory=list)
    transcript_history: list[TranscriptEvent] = field(default_factory=list)


def _make_auth_ctx(copilot_ctx: CopilotContext) -> AuthContext:
    """Create an AuthContext for copilot read operations."""
    return AuthContext(
        agent_id="copilot",
        case_id=copilot_ctx.case_id,
        permissions={"read"},
    )


@function_tool
async def tool_lookup_transactions(ctx: RunContextWrapper[CopilotContext]) -> str:
    """Look up transaction evidence for the current case.

    Fetches all TRANSACTION-type evidence nodes via the Tool Gateway,
    with PAN fields masked. Returns a JSON array of transaction dicts.
    """
    copilot_ctx = ctx.context
    auth = _make_auth_ctx(copilot_ctx)
    results = lookup_transactions(copilot_ctx.gateway, auth, copilot_ctx.case_id)
    stripped = _strip_fields(results, _STRIP_TRANSACTION)
    return json.dumps(_tool_redactor.redact_dict(stripped))


@function_tool
async def tool_query_auth_logs(ctx: RunContextWrapper[CopilotContext]) -> str:
    """Query authentication event logs for the current case.

    Fetches all AUTH_EVENT-type evidence nodes via the Tool Gateway.
    Returns a JSON array of auth event dicts.
    """
    copilot_ctx = ctx.context
    auth = _make_auth_ctx(copilot_ctx)
    results = query_auth_logs(copilot_ctx.gateway, auth, copilot_ctx.case_id)
    stripped = _strip_fields(results, _STRIP_AUTH_EVENT)
    return json.dumps(_tool_redactor.redact_dict(stripped))


@function_tool
async def tool_fetch_customer_profile(ctx: RunContextWrapper[CopilotContext]) -> str:
    """Fetch the customer profile for the current case.

    Fetches the first CUSTOMER-type evidence node via the Tool Gateway,
    with PAN fields masked. Returns a JSON object or 'null' if not found.
    """
    copilot_ctx = ctx.context
    auth = _make_auth_ctx(copilot_ctx)
    result = fetch_customer_profile(copilot_ctx.gateway, auth, copilot_ctx.case_id)
    if result:
        stripped = _strip_fields([result], _STRIP_CUSTOMER)[0]
        result = _tool_redactor.redact_dict(stripped)
    return json.dumps(result)
