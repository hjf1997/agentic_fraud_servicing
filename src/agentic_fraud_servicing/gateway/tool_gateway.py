"""Central Tool Gateway for mediated data access between agents and storage.

All agent data access goes through this gateway, which enforces authentication,
field-level PAN masking, and immutable audit logging via the TraceStore.
"""

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from agentic_fraud_servicing.ingestion.redaction import _PAN_RE
from agentic_fraud_servicing.storage.case_store import CaseStore
from agentic_fraud_servicing.storage.evidence_store import EvidenceStore
from agentic_fraud_servicing.storage.trace_store import TraceStore


class GatewayAuthError(RuntimeError):
    """Raised when an agent fails authentication or authorization.

    Attributes:
        agent_id: Identity of the agent that failed auth.
        action: The action that was denied.
    """

    def __init__(self, message: str, agent_id: str = "", action: str = "") -> None:
        self.agent_id = agent_id
        self.action = action
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        ctx_parts = []
        if self.agent_id:
            ctx_parts.append(f"agent_id={self.agent_id}")
        if self.action:
            ctx_parts.append(f"action={self.action}")
        if ctx_parts:
            return f"{base} [{', '.join(ctx_parts)}]"
        return base


@dataclass
class AuthContext:
    """Authentication and authorization context for a gateway call.

    Plain dataclass — this is internal infrastructure, not a domain model.

    Attributes:
        agent_id: Identity of the calling agent.
        case_id: Optional case scope for the request.
        permissions: Set of permission strings ('read', 'write', 'compliance').
    """

    agent_id: str
    case_id: str | None = None
    permissions: set[str] = field(default_factory=set)


class ToolGateway:
    """Central gateway mediating all agent access to the storage layer.

    Holds references to all three stores and provides auth checking,
    audit logging, and PAN masking utilities for gateway tool functions.

    Args:
        case_store: CaseStore instance for case persistence.
        evidence_store: EvidenceStore instance for evidence graph persistence.
        trace_store: TraceStore instance for agent audit logs.
    """

    def __init__(
        self,
        case_store: CaseStore,
        evidence_store: EvidenceStore,
        trace_store: TraceStore,
    ) -> None:
        self._case_store = case_store
        self._evidence_store = evidence_store
        self._trace_store = trace_store

    @property
    def case_store(self) -> CaseStore:
        """Read-only access to the case store."""
        return self._case_store

    @property
    def evidence_store(self) -> EvidenceStore:
        """Read-only access to the evidence store."""
        return self._evidence_store

    @property
    def trace_store(self) -> TraceStore:
        """Read-only access to the trace store."""
        return self._trace_store

    def check_auth(self, ctx: AuthContext, required_permission: str) -> None:
        """Validate that the auth context has the required permission.

        Args:
            ctx: The authentication context to check.
            required_permission: Permission string that must be present.

        Raises:
            GatewayAuthError: If agent_id is empty or permission is missing.
        """
        if not ctx.agent_id:
            raise GatewayAuthError(
                "agent_id must not be empty",
                agent_id=ctx.agent_id,
                action=required_permission,
            )
        if required_permission not in ctx.permissions:
            raise GatewayAuthError(
                f"Permission '{required_permission}' not granted",
                agent_id=ctx.agent_id,
                action=required_permission,
            )

    def log_call(
        self,
        ctx: AuthContext,
        action: str,
        input_summary: str,
        output_summary: str,
        duration_ms: float,
        status: str = "success",
    ) -> None:
        """Log a gateway tool call to the trace store for audit.

        Args:
            ctx: Auth context identifying the calling agent.
            action: Description of the action performed.
            input_summary: Summary of the call input.
            output_summary: Summary of the call output.
            duration_ms: Duration of the call in milliseconds.
            status: Outcome status (default 'success').
        """
        trace_id = str(uuid.uuid4())
        case_id = ctx.case_id or "unknown"
        self._trace_store.log_invocation(
            trace_id=trace_id,
            case_id=case_id,
            agent_id=ctx.agent_id,
            action=action,
            input_data=input_summary,
            output_data=output_summary,
            duration_ms=duration_ms,
            timestamp=datetime.now(timezone.utc),
            status=status,
        )

    def mask_pan_in_dict(self, data: dict) -> dict:
        """Deep-copy a dict and replace any PAN patterns in string values.

        Iterates over all values in the dict (including nested dicts and lists)
        and replaces PAN patterns with [PAN_REDACTED]. Does NOT mutate the
        original dict.

        Args:
            data: Dictionary potentially containing PAN strings.

        Returns:
            A new dict with all PAN patterns masked.
        """
        result = copy.deepcopy(data)
        self._mask_values(result)
        return result

    def _mask_values(self, obj: dict | list) -> None:
        """Recursively mask PAN patterns in dict values or list elements."""
        if isinstance(obj, dict):
            for key in obj:
                if isinstance(obj[key], str):
                    obj[key] = _PAN_RE.sub("[PAN_REDACTED]", obj[key])
                elif isinstance(obj[key], (dict, list)):
                    self._mask_values(obj[key])
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, str):
                    obj[i] = _PAN_RE.sub("[PAN_REDACTED]", item)
                elif isinstance(item, (dict, list)):
                    self._mask_values(item)
