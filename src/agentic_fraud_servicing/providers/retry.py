"""Retry wrapper for Agents SDK Runner.run() calls.

Retries on transient failures (malformed JSON from LLM, HTTP timeouts) but
immediately propagates non-retriable errors (firewall/DLP policy blocks,
permission errors). Uses exponential backoff between attempts.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agents import Runner
from agents.exceptions import ModelBehaviorError

logger = logging.getLogger(__name__)

# Default retry settings
DEFAULT_MAX_RETRIES = 2
DEFAULT_BASE_DELAY = 1.0


def _is_retriable(exc: BaseException) -> bool:
    """Determine whether an exception warrants a retry.

    Retriable:
        - ModelBehaviorError (invalid JSON output, unexpected content type)
        - Timeout errors (httpx, openai, asyncio)
        - HTTP 429 (rate limit) and 5xx (server errors)

    Non-retriable:
        - Firewall/DLP policy blocks (403)
        - Permission errors
        - All other errors (bad prompts, validation, etc.)
    """
    # Firewall blocks are never retriable — same prompt will fail again
    # Lazy import to avoid circular dependency (langfuse_tracing → ... → retry)
    from agentic_fraud_servicing.copilot.langfuse_tracing import is_firewall_block

    if is_firewall_block(exc):
        return False

    # ModelBehaviorError = LLM returned malformed JSON or unexpected output
    if isinstance(exc, ModelBehaviorError):
        return True

    # Walk the exception chain for timeout and HTTP status errors
    current: BaseException | None = exc
    while current is not None:
        type_name = type(current).__name__

        # Timeout errors from httpx, openai, or asyncio
        if type_name in ("TimeoutError", "APITimeoutError", "ConnectTimeout", "ReadTimeout"):
            return True
        if isinstance(current, (asyncio.TimeoutError, TimeoutError)):
            return True

        # HTTP status-based retries (rate limit, server errors)
        status = getattr(current, "status_code", None)
        if status is not None:
            if status == 429 or status >= 500:
                return True
            # 4xx other than 429 are not retriable (bad request, auth, etc.)
            if 400 <= status < 500:
                return False

        current = getattr(current, "__cause__", None)

    return False


async def run_with_retry(
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    **kwargs: Any,
) -> Any:
    """Call Runner.run() with automatic retry on transient failures.

    Wraps agents.Runner.run() and retries on ModelBehaviorError (invalid
    JSON) and timeout errors. Propagates firewall blocks and other
    non-retriable errors immediately.

    Args:
        *args: Positional arguments forwarded to Runner.run().
        max_retries: Maximum number of retry attempts (default 2).
        base_delay: Base delay in seconds for exponential backoff.
        **kwargs: Keyword arguments forwarded to Runner.run().

    Returns:
        The RunResult from Runner.run().

    Raises:
        The original exception if all retries are exhausted or the error
        is non-retriable.
    """
    last_exc: BaseException | None = None

    for attempt in range(1 + max_retries):
        try:
            return await Runner.run(*args, **kwargs)
        except Exception as exc:
            last_exc = exc

            if not _is_retriable(exc):
                raise

            if attempt < max_retries:
                delay = base_delay * (2**attempt)
                agent_name = ""
                if args:
                    agent_name = getattr(args[0], "name", "")
                logger.warning(
                    "Retry %d/%d for agent '%s' after %s: %s (backoff %.1fs)",
                    attempt + 1,
                    max_retries,
                    agent_name,
                    type(exc).__name__,
                    str(exc)[:200],
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                raise

    # Should not reach here, but satisfy type checker
    raise last_exc  # type: ignore[misc]
