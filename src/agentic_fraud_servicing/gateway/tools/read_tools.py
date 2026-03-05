"""Read-only data access tools for agents.

Each function takes a ToolGateway and AuthContext, enforces 'read' permission,
masks PAN fields in output where applicable, and logs the call to the trace
store for audit. Storage errors are wrapped in RuntimeError with context.
"""

import time

from agentic_fraud_servicing.gateway.tool_gateway import AuthContext, ToolGateway
from agentic_fraud_servicing.models.enums import EvidenceNodeType


def lookup_transactions(gateway: ToolGateway, ctx: AuthContext, case_id: str) -> list[dict]:
    """Fetch TRANSACTION-type evidence nodes for a case.

    Retrieves all evidence nodes for the case, filters to TRANSACTION type,
    masks PAN patterns in each dict, and logs the call.

    Args:
        gateway: ToolGateway instance mediating storage access.
        ctx: Auth context for the calling agent.
        case_id: The case identifier to look up transactions for.

    Returns:
        List of transaction node dicts with PAN fields masked.

    Raises:
        GatewayAuthError: If auth context lacks 'read' permission.
        RuntimeError: On storage or other system errors.
    """
    gateway.check_auth(ctx, "read")
    start = time.monotonic()
    try:
        all_nodes = gateway.evidence_store.get_nodes_by_case(case_id)
        transactions = [n for n in all_nodes if n.get("node_type") == EvidenceNodeType.TRANSACTION]
        masked = [gateway.mask_pan_in_dict(t) for t in transactions]
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"lookup_transactions failed for case_id '{case_id}': {exc}") from exc
    finally:
        duration_ms = (time.monotonic() - start) * 1000
        gateway.log_call(
            ctx=ctx,
            action="lookup_transactions",
            input_summary=f'{{"case_id": "{case_id}"}}',
            output_summary=f'{{"count": {len(masked) if "masked" in dir() else 0}}}',
            duration_ms=duration_ms,
        )

    return masked


def query_auth_logs(gateway: ToolGateway, ctx: AuthContext, case_id: str) -> list[dict]:
    """Fetch AUTH_EVENT-type evidence nodes for a case.

    Retrieves all evidence nodes for the case and filters to AUTH_EVENT type.
    No PAN masking needed since auth events don't contain PANs.

    Args:
        gateway: ToolGateway instance mediating storage access.
        ctx: Auth context for the calling agent.
        case_id: The case identifier to query auth logs for.

    Returns:
        List of auth event node dicts.

    Raises:
        GatewayAuthError: If auth context lacks 'read' permission.
        RuntimeError: On storage or other system errors.
    """
    gateway.check_auth(ctx, "read")
    start = time.monotonic()
    try:
        all_nodes = gateway.evidence_store.get_nodes_by_case(case_id)
        auth_events = [n for n in all_nodes if n.get("node_type") == EvidenceNodeType.AUTH_EVENT]
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"query_auth_logs failed for case_id '{case_id}': {exc}") from exc
    finally:
        duration_ms = (time.monotonic() - start) * 1000
        gateway.log_call(
            ctx=ctx,
            action="query_auth_logs",
            input_summary=f'{{"case_id": "{case_id}"}}',
            output_summary=f'{{"count": {len(auth_events) if "auth_events" in dir() else 0}}}',
            duration_ms=duration_ms,
        )

    return auth_events


def fetch_customer_profile(gateway: ToolGateway, ctx: AuthContext, case_id: str) -> dict | None:
    """Fetch the first CUSTOMER-type evidence node for a case.

    Retrieves all evidence nodes for the case, finds the first CUSTOMER type,
    and masks PAN patterns in the result (customer records may contain
    embedded card references).

    Args:
        gateway: ToolGateway instance mediating storage access.
        ctx: Auth context for the calling agent.
        case_id: The case identifier to fetch the customer profile for.

    Returns:
        Customer node dict with PAN fields masked, or None if no customer
        node exists for the case.

    Raises:
        GatewayAuthError: If auth context lacks 'read' permission.
        RuntimeError: On storage or other system errors.
    """
    gateway.check_auth(ctx, "read")
    start = time.monotonic()
    result: dict | None = None
    try:
        all_nodes = gateway.evidence_store.get_nodes_by_case(case_id)
        customers = [n for n in all_nodes if n.get("node_type") == EvidenceNodeType.CUSTOMER]
        if customers:
            result = gateway.mask_pan_in_dict(customers[0])
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"fetch_customer_profile failed for case_id '{case_id}': {exc}"
        ) from exc
    finally:
        duration_ms = (time.monotonic() - start) * 1000
        gateway.log_call(
            ctx=ctx,
            action="fetch_customer_profile",
            input_summary=f'{{"case_id": "{case_id}"}}',
            output_summary=f'{{"found": {result is not None}}}',
            duration_ms=duration_ms,
        )

    return result
