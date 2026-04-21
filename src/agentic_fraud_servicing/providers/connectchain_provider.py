"""ConnectChain ModelProvider implementation.

Uses AMEX's ConnectChain enterprise framework to obtain EAS auth tokens and
Azure OpenAI endpoint configuration, then delegates to the Agents SDK model
classes via an AsyncAzureOpenAI client. Supports both the Chat Completions API
(OpenAIChatCompletionsModel) and the Responses API (OpenAIResponsesModel),
controlled by the AZURE_OPENAI_USE_RESPONSES_API env var.
"""

from __future__ import annotations

import time

from agents import OpenAIChatCompletionsModel
from agents.models.interface import Model, ModelProvider
from agents.models.openai_responses import OpenAIResponsesModel
from openai import AsyncAzureOpenAI

from agentic_fraud_servicing.config import Settings
from agentic_fraud_servicing.providers.base import ProviderError

# Default Azure OpenAI API version. Override via AZURE_OPENAI_API_VERSION env var.
_DEFAULT_API_VERSION = "2024-08-01-preview"


class ConnectChainModelProvider(ModelProvider):
    """ModelProvider that uses ConnectChain for auth and Azure OpenAI endpoint.

    Extracts the EAS token, Azure OpenAI base URL, deployment name, and API
    version from ConnectChain's configuration, then creates an AsyncAzureOpenAI
    client. Returns either OpenAIResponsesModel or OpenAIChatCompletionsModel
    depending on AZURE_OPENAI_USE_RESPONSES_API env var.

    Args:
        settings: Application settings containing connectchain_model_index.

    Raises:
        ProviderError: If ConnectChain is not installed or config is invalid.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.connectchain_model_index:
            raise ProviderError(
                "CONNECTCHAIN_MODEL_INDEX is required for the connectchain provider",
                request_type="init",
            )

        try:
            from connectchain.lcel.model import model as get_connectchain_model
            from connectchain.utils import get_token_from_env
        except ImportError as exc:
            raise ProviderError(
                "connectchain package is not installed. "
                "Install with: pip install connectchain>=0.1.0",
                request_type="init",
            ) from exc

        self._model_index = settings.connectchain_model_index
        self._raw_get_token = get_token_from_env
        self._cached_token: str | None = None
        self._token_expiry: float = 0

        try:
            lc_llm = get_connectchain_model(settings.connectchain_model_index)
            # Extract Azure OpenAI config from the LangChain LLM object
            azure_endpoint = getattr(lc_llm, "azure_endpoint", None) or getattr(
                lc_llm, "openai_api_base", None
            )
            deployment = (
                getattr(lc_llm, "deployment_name", None)
                or getattr(lc_llm, "engine", None)
                or getattr(lc_llm, "model_name", None)
            )
            api_version = (
                getattr(lc_llm, "openai_api_version", None)
                or settings.azure_openai_api_version
                or _DEFAULT_API_VERSION
            )
        except Exception as exc:
            raise ProviderError(
                f"ConnectChain initialization failed: {exc}",
                request_type="init",
            ) from exc

        if not azure_endpoint:
            raise ProviderError(
                "Could not extract Azure OpenAI endpoint from ConnectChain model. "
                "Check connectchain.config.yml.",
                request_type="init",
            )

        self._client = AsyncAzureOpenAI(
            azure_ad_token_provider=self._get_cached_token,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        self._default_model = deployment or "gpt-4o"
        self._use_responses_api = settings.azure_openai_use_responses_api

    @property
    def client(self) -> AsyncAzureOpenAI:
        """Return the underlying AsyncAzureOpenAI client for direct API calls.

        Used by the logprob-based hypothesis scorer which bypasses the
        Agents SDK to make raw completions calls with logprobs enabled.
        """
        return self._client

    @property
    def default_model(self) -> str:
        """Return the Azure deployment name for direct API calls."""
        return self._default_model

    def _get_cached_token(self) -> str:
        """Return a cached EAS token, refreshing via ConnectChain when expired.

        Caches the token for 5 minutes to avoid a network round-trip on every
        LLM call. On refresh failure, returns the stale token if available.
        """
        now = time.monotonic()
        if self._cached_token is not None and now < self._token_expiry:
            return self._cached_token
        try:
            self._cached_token = self._raw_get_token(self._model_index)
            self._token_expiry = now + 300  # 5 minutes
        except Exception:
            if self._cached_token is not None:
                return self._cached_token
            raise
        return self._cached_token

    def get_model(self, model_name: str | None) -> Model:
        """Return an Agents SDK model using Azure OpenAI credentials.

        Returns OpenAIResponsesModel when AZURE_OPENAI_USE_RESPONSES_API=true,
        otherwise OpenAIChatCompletionsModel (default).

        Args:
            model_name: Azure deployment name. If None, uses the deployment
                from ConnectChain config.

        Returns:
            An Agents SDK Model wrapping the ConnectChain-authenticated
            AsyncAzureOpenAI client.
        """
        if model_name is None:
            model_name = self._default_model
        if self._use_responses_api:
            return OpenAIResponsesModel(model=model_name, openai_client=self._client)
        return OpenAIChatCompletionsModel(model=model_name, openai_client=self._client)
