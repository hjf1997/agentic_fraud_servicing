"""Provider abstraction layer base module.

Defines the ProviderError exception, the get_model_provider() factory function,
and re-exports core SDK types so downstream code imports from providers.base
rather than directly from the agents SDK.
"""

from __future__ import annotations

from agents.model_settings import ModelSettings
from agents.models.interface import Model, ModelProvider, ModelResponse

from agentic_fraud_servicing.config import Settings

__all__ = [
    "Model",
    "ModelProvider",
    "ModelResponse",
    "ModelSettings",
    "ProviderError",
    "get_model_provider",
]


class ProviderError(RuntimeError):
    """Exception raised by LLM provider operations.

    Wraps provider-specific errors with context about the model and request type
    to aid debugging and structured error handling.

    Args:
        message: Human-readable error description.
        model_id: Identifier of the model that was being called, if known.
        request_type: Type of request that failed (e.g. 'get_response', 'stream').
    """

    def __init__(
        self,
        message: str,
        model_id: str | None = None,
        request_type: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.request_type = request_type
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        parts: list[str] = []
        if self.model_id is not None:
            parts.append(f"model_id={self.model_id}")
        if self.request_type is not None:
            parts.append(f"request_type={self.request_type}")
        if parts:
            return f"{base} [{', '.join(parts)}]"
        return base


def get_model_provider(settings: Settings) -> ModelProvider:
    """Create and return the appropriate ModelProvider based on config.

    Uses lazy imports to avoid pulling in boto3 or openai at module level.
    The provider modules are only imported when actually needed.

    Args:
        settings: Application settings containing llm_provider and credentials.

    Returns:
        A ModelProvider instance for the configured LLM backend.

    Raises:
        ValueError: If settings.llm_provider is not a recognised provider name.
    """
    if settings.llm_provider == "openai":
        from agentic_fraud_servicing.providers.openai_provider import (
            OpenAIModelProvider,
        )

        return OpenAIModelProvider(settings)

    if settings.llm_provider == "bedrock":
        from agentic_fraud_servicing.providers.bedrock_provider import (
            BedrockModelProvider,
        )

        return BedrockModelProvider(settings)

    if settings.llm_provider == "connectchain":
        from agentic_fraud_servicing.providers.connectchain_provider import (
            ConnectChainModelProvider,
        )

        return ConnectChainModelProvider(settings)

    raise ValueError(f"Unknown LLM provider: {settings.llm_provider!r}")
