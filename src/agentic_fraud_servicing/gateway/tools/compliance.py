"""Compliance tools for regulatory checks by agents.

Each function takes a ToolGateway and AuthContext, enforces 'compliance'
permission, logs the call to the trace store for audit, and wraps errors.
"""

import time
from datetime import datetime, timezone

from agentic_fraud_servicing.gateway.tool_gateway import AuthContext, ToolGateway

# Standard financial record retention: 7 years (365 * 7 = 2555 days).
RETENTION_DAYS = 2555


def check_retention(gateway: ToolGateway, ctx: AuthContext, case_id: str) -> dict:
    """Check whether case data is within the retention window.

    Args:
        gateway: ToolGateway instance mediating storage access.
        ctx: Auth context for the calling agent.
        case_id: The case identifier to check.

    Returns:
        Dict with case_id, within_retention bool, age_days, and retention_limit_days.

    Raises:
        GatewayAuthError: If auth context lacks 'compliance' permission.
        RuntimeError: If case not found or storage error occurs.
    """
    gateway.check_auth(ctx, "compliance")
    start = time.monotonic()
    status = "success"
    try:
        case = gateway.case_store.get_case(case_id)
        if case is None:
            raise RuntimeError(f"Case '{case_id}' not found")

        now = datetime.now(timezone.utc)
        age_delta = now - case.created_at
        age_days = age_delta.days

        return {
            "case_id": case_id,
            "within_retention": age_days <= RETENTION_DAYS,
            "age_days": age_days,
            "retention_limit_days": RETENTION_DAYS,
        }
    except RuntimeError:
        status = "error"
        raise
    except Exception as exc:
        status = "error"
        raise RuntimeError(f"check_retention failed for case_id '{case_id}': {exc}") from exc
    finally:
        duration_ms = (time.monotonic() - start) * 1000
        gateway.log_call(
            ctx=ctx,
            action="check_retention",
            input_summary=f'{{"case_id": "{case_id}"}}',
            output_summary=f'{{"status": "{status}"}}',
            duration_ms=duration_ms,
            status=status,
        )


def verify_consent(gateway: ToolGateway, ctx: AuthContext, case_id: str) -> dict:
    """Verify that the case has required consent for processing.

    Checks the case audit trail for entries whose action contains 'consent'
    (case-insensitive).

    Args:
        gateway: ToolGateway instance mediating storage access.
        ctx: Auth context for the calling agent.
        case_id: The case identifier to check.

    Returns:
        Dict with case_id, consent_recorded bool, and consent_entries list.

    Raises:
        GatewayAuthError: If auth context lacks 'compliance' permission.
        RuntimeError: If case not found or storage error occurs.
    """
    gateway.check_auth(ctx, "compliance")
    start = time.monotonic()
    status = "success"
    try:
        case = gateway.case_store.get_case(case_id)
        if case is None:
            raise RuntimeError(f"Case '{case_id}' not found")

        consent_entries = [
            entry.details or entry.action
            for entry in case.audit_trail
            if "consent" in entry.action.lower()
        ]

        return {
            "case_id": case_id,
            "consent_recorded": len(consent_entries) > 0,
            "consent_entries": consent_entries,
        }
    except RuntimeError:
        status = "error"
        raise
    except Exception as exc:
        status = "error"
        raise RuntimeError(f"verify_consent failed for case_id '{case_id}': {exc}") from exc
    finally:
        duration_ms = (time.monotonic() - start) * 1000
        gateway.log_call(
            ctx=ctx,
            action="verify_consent",
            input_summary=f'{{"case_id": "{case_id}"}}',
            output_summary=f'{{"status": "{status}"}}',
            duration_ms=duration_ms,
            status=status,
        )


def redact_case_fields(
    gateway: ToolGateway,
    ctx: AuthContext,
    case_id: str,
    fields_to_redact: list[str],
) -> dict:
    """Apply field-level redaction to a case's data for display or export.

    Returns a dict representation of the case with specified fields replaced
    by '[REDACTED]'. Does NOT modify the stored case — read-only operation.

    Args:
        gateway: ToolGateway instance mediating storage access.
        ctx: Auth context for the calling agent.
        case_id: The case identifier to redact.
        fields_to_redact: List of field names to replace with '[REDACTED]'.

    Returns:
        Dict of case data with specified fields redacted.

    Raises:
        GatewayAuthError: If auth context lacks 'compliance' permission.
        ValueError: If case not found or a field doesn't exist on the case model.
    """
    gateway.check_auth(ctx, "compliance")
    start = time.monotonic()
    status = "success"
    try:
        case = gateway.case_store.get_case(case_id)
        if case is None:
            raise ValueError(f"Case '{case_id}' not found")

        case_dict = case.model_dump(mode="json")

        for field_name in fields_to_redact:
            if field_name not in case_dict:
                raise ValueError(f"Field '{field_name}' does not exist on Case model")
            case_dict[field_name] = "[REDACTED]"

        return case_dict
    except (ValueError, RuntimeError):
        status = "error"
        raise
    except Exception as exc:
        status = "error"
        raise RuntimeError(f"redact_case_fields failed for case_id '{case_id}': {exc}") from exc
    finally:
        duration_ms = (time.monotonic() - start) * 1000
        gateway.log_call(
            ctx=ctx,
            action="redact_case_fields",
            input_summary=f'{{"case_id": "{case_id}", "fields": {fields_to_redact}}}',
            output_summary=f'{{"status": "{status}"}}',
            duration_ms=duration_ms,
            status=status,
        )
