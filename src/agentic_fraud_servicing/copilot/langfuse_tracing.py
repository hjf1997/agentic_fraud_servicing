"""LangFuse observability integration for the copilot pipeline.

Provides auto-instrumentation of all OpenAI Agents SDK Runner.run() calls
via the OpenInference instrumentor, plus helpers for orchestrator-level
span enrichment.

Supports both LangFuse Cloud and self-hosted deployments — controlled
entirely via environment variables (LANGFUSE_BASE_URL, LANGFUSE_PUBLIC_KEY,
LANGFUSE_SECRET_KEY).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_fraud_servicing.config import Settings

logger = logging.getLogger(__name__)

_instrumentor = None
_initialized = False


def init_langfuse(settings: Settings):
    """Initialize LangFuse + OpenAI Agents SDK auto-instrumentation.

    Call once at app startup. Safe to call when LangFuse is not configured
    (returns None, no instrumentation registered).

    Returns:
        The LangFuse client if initialization succeeded, else None.
    """
    global _instrumentor, _initialized

    if not settings.langfuse_enabled:
        logger.debug("LangFuse disabled — LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set")
        return None

    # Ensure env vars are set before langfuse client reads them at import time
    if settings.langfuse_base_url:
        os.environ.setdefault("LANGFUSE_BASE_URL", settings.langfuse_base_url)
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)

    try:
        from langfuse import get_client
        from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

        _instrumentor = OpenAIAgentsInstrumentor()
        _instrumentor.instrument()

        # The OpenInference processor hooks into the Agents SDK's built-in
        # tracing. If OPENAI_AGENTS_DISABLE_TRACING=1 was set (e.g. to
        # suppress "API key not set" warnings when using Bedrock), tracing
        # is globally disabled and no processors get called. Re-enable it
        # so prompts/completions are captured by the OpenInference → LangFuse
        # pipeline.
        try:
            from agents.tracing import set_tracing_disabled

            set_tracing_disabled(False)
        except Exception:
            pass

        client = get_client()
        if not client.auth_check():
            logger.warning("LangFuse auth check failed — disabling instrumentation")
            _instrumentor.uninstrument()
            _instrumentor = None
            return None

        _initialized = True
        logger.info("LangFuse initialized (base_url=%s)", settings.langfuse_base_url)
        return client
    except Exception:
        logger.exception("Failed to initialize LangFuse")
        if _instrumentor is not None:
            try:
                _instrumentor.uninstrument()
            except Exception:
                pass
            _instrumentor = None
        return None


def get_langfuse():
    """Return the LangFuse client if initialized, else None."""
    if not _initialized:
        return None
    try:
        from langfuse import get_client

        return get_client()
    except Exception:
        return None


def tag_firewall_block(agent_name: str, error_message: str) -> None:
    """Tag the current LangFuse trace when an agent is blocked by the enterprise firewall.

    Attaches a score (firewall_block=1) and a "firewall_block" tag to the
    current observation and its parent trace so blocked calls surface as a
    colored pill in the LangFuse traces list and are filterable by tag.
    """
    lf = get_langfuse()
    if lf is None:
        return
    try:
        obs = lf.get_current_observation()
        if obs is not None:
            # Tag the observation — tags auto-aggregate to the parent trace,
            # making blocked traces visible as colored pills in the list view.
            obs.update(tags=["firewall_block", f"blocked:{agent_name}"])
            # Score the specific observation (agent-level)
            obs.score(
                name="firewall_block",
                value=1,
                comment=f"[{agent_name}] {error_message}",
            )
            # Also score the parent trace for top-level filtering
            obs.score_trace(
                name="firewall_block",
                value=1,
                comment=f"[{agent_name}] {error_message}",
            )
    except Exception:
        pass


def is_firewall_block(exc: BaseException) -> bool:
    """Detect whether an exception chain contains a 403 firewall policy block.

    Walks the exception __cause__ chain looking for openai.PermissionDeniedError
    (status_code=403) or the enterprise firewall rejection message.
    """
    current: BaseException | None = exc
    while current is not None:
        msg = str(current).lower()
        if "403" in msg and "policy" in msg:
            return True
        # Check for openai SDK's PermissionDeniedError
        type_name = type(current).__name__
        if type_name == "PermissionDeniedError":
            return True
        if hasattr(current, "status_code") and getattr(current, "status_code") == 403:
            return True
        current = getattr(current, "__cause__", None)
    return False


def shutdown_langfuse() -> None:
    """Flush pending traces and uninstrument. Call on app shutdown."""
    global _instrumentor, _initialized
    try:
        client = get_langfuse()
        if client is not None:
            client.flush()
    except Exception:
        pass
    if _instrumentor is not None:
        try:
            _instrumentor.uninstrument()
        except Exception:
            pass
        _instrumentor = None
    _initialized = False
