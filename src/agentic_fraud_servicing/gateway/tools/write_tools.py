"""Write data access tools for agents.

Each function takes a ToolGateway and AuthContext, enforces 'write' permission,
logs the call to the trace store for audit, and wraps storage errors in
RuntimeError with context.
"""

import time

from agentic_fraud_servicing.gateway.tool_gateway import AuthContext, ToolGateway
from agentic_fraud_servicing.models.case import Case
from agentic_fraud_servicing.models.enums import CaseStatus
from agentic_fraud_servicing.models.evidence import EvidenceEdge, EvidenceNode


def create_case(gateway: ToolGateway, ctx: AuthContext, case: Case) -> None:
    """Persist a new case to the CaseStore.

    Args:
        gateway: ToolGateway instance mediating storage access.
        ctx: Auth context for the calling agent.
        case: The Case model to persist.

    Raises:
        GatewayAuthError: If auth context lacks 'write' permission.
        RuntimeError: On storage errors (e.g. duplicate case_id).
    """
    gateway.check_auth(ctx, "write")
    start = time.monotonic()
    status = "success"
    try:
        gateway.case_store.create_case(case)
    except RuntimeError:
        status = "error"
        raise
    except Exception as exc:
        status = "error"
        raise RuntimeError(f"create_case failed for case_id '{case.case_id}': {exc}") from exc
    finally:
        duration_ms = (time.monotonic() - start) * 1000
        gateway.log_call(
            ctx=ctx,
            action="create_case",
            input_summary=f'{{"case_id": "{case.case_id}"}}',
            output_summary=f'{{"status": "{status}"}}',
            duration_ms=duration_ms,
            status=status,
        )


def update_case_status(
    gateway: ToolGateway, ctx: AuthContext, case_id: str, status: CaseStatus
) -> None:
    """Update the status of an existing case.

    Args:
        gateway: ToolGateway instance mediating storage access.
        ctx: Auth context for the calling agent.
        case_id: The case identifier to update.
        status: The new CaseStatus value.

    Raises:
        GatewayAuthError: If auth context lacks 'write' permission.
        RuntimeError: If case not found or storage error occurs.
    """
    gateway.check_auth(ctx, "write")
    start = time.monotonic()
    call_status = "success"
    try:
        gateway.case_store.update_case_status(case_id, status)
    except RuntimeError:
        call_status = "error"
        raise
    except Exception as exc:
        call_status = "error"
        raise RuntimeError(f"update_case_status failed for case_id '{case_id}': {exc}") from exc
    finally:
        duration_ms = (time.monotonic() - start) * 1000
        gateway.log_call(
            ctx=ctx,
            action="update_case_status",
            input_summary=f'{{"case_id": "{case_id}", "status": "{status.value}"}}',
            output_summary=f'{{"status": "{call_status}"}}',
            duration_ms=duration_ms,
            status=call_status,
        )


def append_evidence_node(gateway: ToolGateway, ctx: AuthContext, node: EvidenceNode) -> None:
    """Add an evidence node to the EvidenceStore.

    Args:
        gateway: ToolGateway instance mediating storage access.
        ctx: Auth context for the calling agent.
        node: The EvidenceNode (or subclass) to persist.

    Raises:
        GatewayAuthError: If auth context lacks 'write' permission.
        RuntimeError: On storage errors (e.g. duplicate node_id).
    """
    gateway.check_auth(ctx, "write")
    start = time.monotonic()
    status = "success"
    try:
        gateway.evidence_store.add_node(node)
    except RuntimeError:
        status = "error"
        raise
    except Exception as exc:
        status = "error"
        raise RuntimeError(
            f"append_evidence_node failed for node_id '{node.node_id}': {exc}"
        ) from exc
    finally:
        duration_ms = (time.monotonic() - start) * 1000
        gateway.log_call(
            ctx=ctx,
            action="append_evidence_node",
            input_summary=f'{{"node_id": "{node.node_id}", "case_id": "{node.case_id}"}}',
            output_summary=f'{{"status": "{status}"}}',
            duration_ms=duration_ms,
            status=status,
        )


def append_evidence_edge(gateway: ToolGateway, ctx: AuthContext, edge: EvidenceEdge) -> None:
    """Add an evidence edge to the EvidenceStore.

    Args:
        gateway: ToolGateway instance mediating storage access.
        ctx: Auth context for the calling agent.
        edge: The EvidenceEdge to persist.

    Raises:
        GatewayAuthError: If auth context lacks 'write' permission.
        RuntimeError: On storage errors (e.g. duplicate edge_id).
    """
    gateway.check_auth(ctx, "write")
    start = time.monotonic()
    status = "success"
    try:
        gateway.evidence_store.add_edge(edge)
    except RuntimeError:
        status = "error"
        raise
    except Exception as exc:
        status = "error"
        raise RuntimeError(
            f"append_evidence_edge failed for edge_id '{edge.edge_id}': {exc}"
        ) from exc
    finally:
        duration_ms = (time.monotonic() - start) * 1000
        gateway.log_call(
            ctx=ctx,
            action="append_evidence_edge",
            input_summary=(
                f'{{"edge_id": "{edge.edge_id}", "case_id": "{edge.case_id}", '
                f'"source": "{edge.source_node_id}", "target": "{edge.target_node_id}"}}'
            ),
            output_summary=f'{{"status": "{status}"}}',
            duration_ms=duration_ms,
            status=status,
        )
