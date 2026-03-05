"""Tests for providers.base module — ProviderError and get_model_provider factory."""

from unittest.mock import MagicMock, patch

import pytest

from agentic_fraud_servicing.config import Settings
from agentic_fraud_servicing.providers.base import (
    Model,
    ModelProvider,
    ModelResponse,
    ModelSettings,
    ProviderError,
    get_model_provider,
)

# ---------------------------------------------------------------------------
# ProviderError tests
# ---------------------------------------------------------------------------


class TestProviderError:
    """Tests for the ProviderError exception class."""

    def test_stores_model_id_and_request_type(self) -> None:
        err = ProviderError("boom", model_id="gpt-4o", request_type="get_response")
        assert err.model_id == "gpt-4o"
        assert err.request_type == "get_response"

    def test_str_includes_context_when_provided(self) -> None:
        err = ProviderError("timeout", model_id="claude-3", request_type="stream")
        result = str(err)
        assert "timeout" in result
        assert "model_id=claude-3" in result
        assert "request_type=stream" in result

    def test_str_works_with_none_context(self) -> None:
        err = ProviderError("something broke")
        assert err.model_id is None
        assert err.request_type is None
        assert str(err) == "something broke"

    def test_str_with_only_model_id(self) -> None:
        err = ProviderError("fail", model_id="gpt-4o")
        result = str(err)
        assert "model_id=gpt-4o" in result
        assert "request_type" not in result

    def test_inherits_from_runtime_error(self) -> None:
        err = ProviderError("test")
        assert isinstance(err, RuntimeError)


# ---------------------------------------------------------------------------
# SDK re-export tests
# ---------------------------------------------------------------------------


class TestReExports:
    """Verify SDK types are accessible via providers.base."""

    def test_model_is_importable(self) -> None:
        assert Model is not None

    def test_model_provider_is_importable(self) -> None:
        assert ModelProvider is not None

    def test_model_response_is_importable(self) -> None:
        assert ModelResponse is not None

    def test_model_settings_is_importable(self) -> None:
        assert ModelSettings is not None


# ---------------------------------------------------------------------------
# get_model_provider tests
# ---------------------------------------------------------------------------


class TestGetModelProvider:
    """Tests for the get_model_provider factory function."""

    def _make_settings(self, provider: str = "bedrock") -> Settings:
        """Create a Settings instance with the given provider, bypassing env vars."""
        env = {
            "LLM_PROVIDER": provider,
            "AWS_PROFILE": "default",
            "AWS_REGION": "us-east-1",
            "AWS_BEDROCK_MODEL_ID": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "OPENAI_API_KEY": "sk-test-key",
        }
        with patch.dict("os.environ", env, clear=False):
            return Settings()

    def test_returns_openai_provider_for_openai(self) -> None:
        """Factory returns OpenAIModelProvider when llm_provider is 'openai'."""
        mock_provider = MagicMock(spec=ModelProvider)
        mock_cls = MagicMock(return_value=mock_provider)
        with patch.dict(
            "sys.modules",
            {
                "agentic_fraud_servicing.providers.openai_provider": MagicMock(
                    OpenAIModelProvider=mock_cls
                )
            },
        ):
            settings = self._make_settings("openai")
            result = get_model_provider(settings)
            assert result is mock_provider

    def test_returns_bedrock_provider_for_bedrock(self) -> None:
        """Factory returns BedrockModelProvider when llm_provider is 'bedrock'."""
        mock_provider = MagicMock(spec=ModelProvider)
        mock_cls = MagicMock(return_value=mock_provider)
        with patch.dict(
            "sys.modules",
            {
                "agentic_fraud_servicing.providers.bedrock_provider": MagicMock(
                    BedrockModelProvider=mock_cls
                )
            },
        ):
            settings = self._make_settings("bedrock")
            result = get_model_provider(settings)
            assert result is mock_provider

    def test_raises_value_error_for_unknown_provider(self) -> None:
        """Factory raises ValueError for unrecognised provider names."""
        settings = self._make_settings("bedrock")
        settings.llm_provider = "unknown_provider"
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_model_provider(settings)
