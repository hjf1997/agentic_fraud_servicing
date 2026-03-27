"""Tests for the LangFuse tracing integration module."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from agentic_fraud_servicing.copilot import langfuse_tracing


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Reset module-level state between tests."""
    langfuse_tracing._instrumentor = None
    langfuse_tracing._initialized = False
    yield
    langfuse_tracing._instrumentor = None
    langfuse_tracing._initialized = False


def _make_settings(enabled: bool = True, base_url: str = "http://localhost:3000"):
    """Create a mock Settings with LangFuse config."""
    s = MagicMock()
    s.langfuse_enabled = enabled
    s.langfuse_base_url = base_url
    s.langfuse_public_key = "pk-lf-test" if enabled else None
    s.langfuse_secret_key = "sk-lf-test" if enabled else None
    return s


@pytest.fixture()
def _mock_langfuse_imports():
    """Mock the langfuse and openinference packages for import."""
    mock_langfuse_mod = MagicMock()
    mock_openinference_mod = MagicMock()
    mock_instrumentor_cls = mock_openinference_mod.OpenAIAgentsInstrumentor

    with (
        patch.dict(sys.modules, {
            "langfuse": mock_langfuse_mod,
            "openinference": MagicMock(),
            "openinference.instrumentation": MagicMock(),
            "openinference.instrumentation.openai_agents": mock_openinference_mod,
        }),
    ):
        yield mock_langfuse_mod, mock_instrumentor_cls


class TestInitLangfuse:
    """Tests for init_langfuse()."""

    def test_enabled_instruments_and_returns_client(self, _mock_langfuse_imports):
        """When enabled, instruments agents and returns the LangFuse client."""
        mock_langfuse_mod, mock_instrumentor_cls = _mock_langfuse_imports

        mock_client = MagicMock()
        mock_client.auth_check.return_value = True
        mock_langfuse_mod.get_client.return_value = mock_client

        mock_inst = MagicMock()
        mock_instrumentor_cls.return_value = mock_inst

        result = langfuse_tracing.init_langfuse(_make_settings(enabled=True))

        mock_inst.instrument.assert_called_once()
        mock_client.auth_check.assert_called_once()
        assert result is mock_client
        assert langfuse_tracing._initialized is True

    def test_disabled_returns_none(self):
        """When disabled, returns None without importing langfuse."""
        result = langfuse_tracing.init_langfuse(_make_settings(enabled=False))

        assert result is None
        assert langfuse_tracing._initialized is False
        assert langfuse_tracing._instrumentor is None

    def test_auth_failure_uninstruments(self, _mock_langfuse_imports):
        """When auth_check fails, uninstruments and returns None."""
        mock_langfuse_mod, mock_instrumentor_cls = _mock_langfuse_imports

        mock_client = MagicMock()
        mock_client.auth_check.return_value = False
        mock_langfuse_mod.get_client.return_value = mock_client

        mock_inst = MagicMock()
        mock_instrumentor_cls.return_value = mock_inst

        result = langfuse_tracing.init_langfuse(_make_settings(enabled=True))

        mock_inst.instrument.assert_called_once()
        mock_inst.uninstrument.assert_called_once()
        assert result is None
        assert langfuse_tracing._instrumentor is None

    def test_import_error_returns_none(self):
        """When langfuse is not installed, returns None gracefully."""
        # Patch sys.modules to make langfuse import fail
        with patch.dict(sys.modules, {"langfuse": None}):
            result = langfuse_tracing.init_langfuse(_make_settings(enabled=True))
        assert result is None


class TestGetLangfuse:
    """Tests for get_langfuse()."""

    def test_returns_none_before_init(self):
        """Returns None when init_langfuse has not been called."""
        assert langfuse_tracing.get_langfuse() is None

    def test_returns_client_after_init(self, _mock_langfuse_imports):
        """Returns the client after successful initialization."""
        mock_langfuse_mod, _ = _mock_langfuse_imports
        mock_client = MagicMock()
        mock_langfuse_mod.get_client.return_value = mock_client
        langfuse_tracing._initialized = True

        result = langfuse_tracing.get_langfuse()
        assert result is mock_client

    def test_returns_none_on_exception(self):
        """Returns None if get_client raises."""
        langfuse_tracing._initialized = True
        # Without langfuse in sys.modules, import will fail
        with patch.dict(sys.modules, {"langfuse": None}):
            assert langfuse_tracing.get_langfuse() is None


class TestShutdownLangfuse:
    """Tests for shutdown_langfuse()."""

    def test_shutdown_flushes_and_uninstruments(self):
        """Shutdown flushes the client and uninstruments."""
        mock_inst = MagicMock()
        langfuse_tracing._instrumentor = mock_inst
        langfuse_tracing._initialized = True

        mock_client = MagicMock()
        with patch.object(langfuse_tracing, "get_langfuse", return_value=mock_client):
            langfuse_tracing.shutdown_langfuse()

        mock_client.flush.assert_called_once()
        mock_inst.uninstrument.assert_called_once()
        assert langfuse_tracing._instrumentor is None
        assert langfuse_tracing._initialized is False

    def test_shutdown_safe_when_not_initialized(self):
        """Shutdown is safe to call when not initialized."""
        langfuse_tracing.shutdown_langfuse()
        assert langfuse_tracing._initialized is False
