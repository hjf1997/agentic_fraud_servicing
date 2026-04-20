"""Tests for providers.retry — Runner.run retry wrapper."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_fraud_servicing.providers.retry import _is_retriable, run_with_retry

# ---------------------------------------------------------------------------
# _is_retriable tests
# ---------------------------------------------------------------------------


class TestIsRetriable:
    def test_model_behavior_error_is_retriable(self):
        """ModelBehaviorError (invalid JSON) should be retried."""
        from agents.exceptions import ModelBehaviorError

        exc = ModelBehaviorError("Invalid JSON when parsing ...")
        assert _is_retriable(exc) is True

    def test_firewall_block_is_not_retriable(self):
        """403 firewall/DLP policy blocks should never be retried."""
        exc = Exception("403 policy violation")
        with patch(
            "agentic_fraud_servicing.copilot.langfuse_tracing.is_firewall_block",
            return_value=True,
        ):
            assert _is_retriable(exc) is False

    def test_timeout_error_is_retriable(self):
        """asyncio.TimeoutError should be retried."""
        exc = asyncio.TimeoutError()
        assert _is_retriable(exc) is True

    def test_builtin_timeout_error_is_retriable(self):
        """Built-in TimeoutError should be retried."""
        exc = TimeoutError("Connection timed out")
        assert _is_retriable(exc) is True

    def test_rate_limit_429_is_retriable(self):
        """HTTP 429 (rate limit) should be retried."""
        exc = Exception("Rate limited")
        exc.status_code = 429
        assert _is_retriable(exc) is True

    def test_server_error_500_is_retriable(self):
        """HTTP 500 (server error) should be retried."""
        exc = Exception("Internal server error")
        exc.status_code = 500
        assert _is_retriable(exc) is True

    def test_bad_request_400_is_not_retriable(self):
        """HTTP 400 (bad request) should not be retried."""
        exc = Exception("Bad request")
        exc.status_code = 400
        assert _is_retriable(exc) is False

    def test_generic_exception_is_not_retriable(self):
        """Generic exceptions without specific markers should not be retried."""
        exc = ValueError("some validation error")
        assert _is_retriable(exc) is False

    def test_chained_timeout_is_retriable(self):
        """Timeout in exception __cause__ chain should be retried."""
        inner = asyncio.TimeoutError()
        outer = RuntimeError("Agent failed")
        outer.__cause__ = inner
        assert _is_retriable(outer) is True


# ---------------------------------------------------------------------------
# run_with_retry tests
# ---------------------------------------------------------------------------


class TestRunWithRetry:
    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """Successful call returns immediately without retry."""
        mock_result = MagicMock()
        with patch(
            "agentic_fraud_servicing.providers.retry.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_run:
            result = await run_with_retry("agent", input="test")
            assert result is mock_result
            assert mock_run.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_model_behavior_error(self):
        """Retries on ModelBehaviorError and succeeds on second attempt."""
        from agents.exceptions import ModelBehaviorError

        mock_result = MagicMock()
        with patch(
            "agentic_fraud_servicing.providers.retry.Runner.run",
            new_callable=AsyncMock,
            side_effect=[ModelBehaviorError("Invalid JSON"), mock_result],
        ) as mock_run:
            result = await run_with_retry(
                "agent",
                input="test",
                max_retries=2,
                base_delay=0.01,
            )
            assert result is mock_result
            assert mock_run.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_firewall_block(self):
        """Firewall blocks propagate immediately without retry."""
        exc = Exception("403 policy block")
        with (
            patch(
                "agentic_fraud_servicing.providers.retry.Runner.run",
                new_callable=AsyncMock,
                side_effect=exc,
            ) as mock_run,
            patch(
                "agentic_fraud_servicing.copilot.langfuse_tracing.is_firewall_block",
                return_value=True,
            ),
        ):
            with pytest.raises(Exception, match="403 policy block"):
                await run_with_retry("agent", input="test", max_retries=2)
            assert mock_run.call_count == 1

    @pytest.mark.asyncio
    async def test_exhausts_retries_then_raises(self):
        """Raises the last exception after all retries are exhausted."""
        from agents.exceptions import ModelBehaviorError

        exc = ModelBehaviorError("Invalid JSON")
        with patch(
            "agentic_fraud_servicing.providers.retry.Runner.run",
            new_callable=AsyncMock,
            side_effect=exc,
        ) as mock_run:
            with pytest.raises(ModelBehaviorError, match="Invalid JSON"):
                await run_with_retry(
                    "agent",
                    input="test",
                    max_retries=2,
                    base_delay=0.01,
                )
            # 1 initial + 2 retries = 3 total
            assert mock_run.call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_generic_error(self):
        """Non-retriable errors propagate immediately."""
        with patch(
            "agentic_fraud_servicing.providers.retry.Runner.run",
            new_callable=AsyncMock,
            side_effect=ValueError("bad input"),
        ) as mock_run:
            with pytest.raises(ValueError, match="bad input"):
                await run_with_retry("agent", input="test", max_retries=2)
            assert mock_run.call_count == 1
