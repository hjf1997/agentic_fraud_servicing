"""OpenAI ModelProvider implementation.

Wraps the OpenAI Agents SDK's built-in OpenAIChatCompletionsModel with an
AsyncOpenAI client configured from application settings. This is the simplest
provider since the SDK already handles the OpenAI Chat Completions format.
"""

from __future__ import annotations

from agents import OpenAIChatCompletionsModel
from agents.models.interface import Model, ModelProvider
from openai import AsyncOpenAI

from agentic_fraud_servicing.config import Settings
from agentic_fraud_servicing.providers.base import ProviderError

DEFAULT_MODEL = "gpt-4o"


class OpenAIModelProvider(ModelProvider):
    """ModelProvider that creates OpenAI Chat Completions models.

    Uses an AsyncOpenAI client initialised from the application settings.
    The client is reused across all models created by this provider.

    Args:
        settings: Application settings containing the OpenAI API key.

    Raises:
        ProviderError: If the OpenAI API key is not configured.
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ProviderError(
                "OpenAI API key is required when using the openai provider",
                request_type="init",
            )
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    def get_model(self, model_name: str | None) -> Model:
        """Return an OpenAIChatCompletionsModel for the given model name.

        Args:
            model_name: OpenAI model identifier (e.g. 'gpt-4o'). If None,
                defaults to 'gpt-4o'.

        Returns:
            An OpenAIChatCompletionsModel wrapping the shared AsyncOpenAI client.
        """
        if model_name is None:
            model_name = DEFAULT_MODEL
        return OpenAIChatCompletionsModel(model=model_name, openai_client=self._client)
