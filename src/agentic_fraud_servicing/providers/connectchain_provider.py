"""ConnectChain ModelProvider implementation.

Uses AMEX's ConnectChain enterprise framework to obtain EAS auth tokens and
Azure OpenAI endpoint configuration, then delegates to the standard
OpenAIChatCompletionsModel from the Agents SDK. This keeps all agent code
unchanged — ConnectChain only handles authentication and endpoint resolution.
"""

from __future__ import annotations

from agents import OpenAIChatCompletionsModel
from agents.models.interface import Model, ModelProvider
from openai import AsyncOpenAI

from agentic_fraud_servicing.config import Settings
from agentic_fraud_servicing.providers.base import ProviderError


class ConnectChainModelProvider(ModelProvider):
    """ModelProvider that uses ConnectChain for auth and endpoint resolution.

    Extracts the EAS token and Azure OpenAI base URL from ConnectChain's
    configuration, then creates a standard AsyncOpenAI client. All models
    returned are OpenAIChatCompletionsModel instances — identical to the
    OpenAI provider.

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

        try:
            token = get_token_from_env()
            lc_llm = get_connectchain_model(settings.connectchain_model_index)
            base_url = getattr(lc_llm, "openai_api_base", None)
            model_name = getattr(lc_llm, "model_name", None)
        except Exception as exc:
            raise ProviderError(
                f"ConnectChain initialization failed: {exc}",
                request_type="init",
            ) from exc

        if not base_url:
            raise ProviderError(
                "Could not extract openai_api_base from ConnectChain model. "
                "Check connectchain.config.yml.",
                request_type="init",
            )

        self._client = AsyncOpenAI(api_key=token, base_url=base_url)
        self._default_model = model_name or "gpt-4o"

    def get_model(self, model_name: str | None) -> Model:
        """Return an OpenAIChatCompletionsModel using ConnectChain credentials.

        Args:
            model_name: Model identifier. If None, uses the model name from
                ConnectChain config.

        Returns:
            An OpenAIChatCompletionsModel wrapping the ConnectChain-authenticated
            AsyncOpenAI client.
        """
        if model_name is None:
            model_name = self._default_model
        return OpenAIChatCompletionsModel(model=model_name, openai_client=self._client)
