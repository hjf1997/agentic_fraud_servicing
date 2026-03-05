"""Tests for the OpenAI ModelProvider implementation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from agents import OpenAIChatCompletionsModel

from agentic_fraud_servicing.providers.base import ProviderError
from agentic_fraud_servicing.providers.openai_provider import (
    DEFAULT_MODEL,
    OpenAIModelProvider,
)


def _make_settings(api_key: str | None = "sk-test-key-123") -> MagicMock:
    """Create a mock Settings with the given API key."""
    settings = MagicMock()
    settings.llm_provider = "openai"
    settings.openai_api_key = api_key
    return settings


class TestOpenAIModelProviderInit:
    """Tests for OpenAIModelProvider construction."""

    @patch("agentic_fraud_servicing.providers.openai_provider.AsyncOpenAI")
    def test_constructs_with_valid_settings(self, mock_openai_cls: MagicMock) -> None:
        """Provider creates an AsyncOpenAI client from settings."""
        settings = _make_settings(api_key="sk-my-key")
        provider = OpenAIModelProvider(settings)

        mock_openai_cls.assert_called_once_with(api_key="sk-my-key")
        assert provider._client is mock_openai_cls.return_value

    def test_raises_provider_error_when_api_key_is_none(self) -> None:
        """Provider raises ProviderError if API key is None."""
        settings = _make_settings(api_key=None)
        with pytest.raises(ProviderError, match="API key is required"):
            OpenAIModelProvider(settings)

    def test_raises_provider_error_when_api_key_is_empty(self) -> None:
        """Provider raises ProviderError if API key is empty string."""
        settings = _make_settings(api_key="")
        with pytest.raises(ProviderError, match="API key is required"):
            OpenAIModelProvider(settings)

    def test_provider_error_includes_request_type(self) -> None:
        """ProviderError from missing key includes request_type context."""
        settings = _make_settings(api_key=None)
        with pytest.raises(ProviderError) as exc_info:
            OpenAIModelProvider(settings)
        assert exc_info.value.request_type == "init"


class TestOpenAIModelProviderGetModel:
    """Tests for OpenAIModelProvider.get_model()."""

    @patch("agentic_fraud_servicing.providers.openai_provider.AsyncOpenAI")
    def test_returns_chat_completions_model(self, mock_openai_cls: MagicMock) -> None:
        """get_model returns an OpenAIChatCompletionsModel instance."""
        provider = OpenAIModelProvider(_make_settings())
        model = provider.get_model("gpt-4o")
        assert isinstance(model, OpenAIChatCompletionsModel)

    @patch("agentic_fraud_servicing.providers.openai_provider.AsyncOpenAI")
    def test_uses_default_model_when_none(self, mock_openai_cls: MagicMock) -> None:
        """get_model with None uses the DEFAULT_MODEL constant."""
        provider = OpenAIModelProvider(_make_settings())
        model = provider.get_model(None)
        assert isinstance(model, OpenAIChatCompletionsModel)
        assert model.model == DEFAULT_MODEL

    @patch("agentic_fraud_servicing.providers.openai_provider.AsyncOpenAI")
    def test_uses_explicit_model_name(self, mock_openai_cls: MagicMock) -> None:
        """get_model with an explicit name uses that name."""
        provider = OpenAIModelProvider(_make_settings())
        model = provider.get_model("gpt-4o-mini")
        assert isinstance(model, OpenAIChatCompletionsModel)
        assert model.model == "gpt-4o-mini"

    @patch("agentic_fraud_servicing.providers.openai_provider.AsyncOpenAI")
    def test_client_created_once(self, mock_openai_cls: MagicMock) -> None:
        """Provider creates the AsyncOpenAI client exactly once."""
        provider = OpenAIModelProvider(_make_settings())
        provider.get_model("gpt-4o")
        provider.get_model("gpt-4o-mini")
        # AsyncOpenAI constructor called once during provider init, not per model
        mock_openai_cls.assert_called_once()
