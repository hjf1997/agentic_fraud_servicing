"""LLM provider abstraction layer.

Re-exports the public API for convenient access:
    from agentic_fraud_servicing.providers import get_model_provider, ProviderError
"""

from agentic_fraud_servicing.providers.base import (
    Model,
    ModelProvider,
    ModelResponse,
    ModelSettings,
    ProviderError,
    get_model_provider,
)
from agentic_fraud_servicing.providers.bedrock_provider import (
    BedrockModel,
    BedrockModelProvider,
)
from agentic_fraud_servicing.providers.openai_provider import OpenAIModelProvider

__all__ = [
    "BedrockModel",
    "BedrockModelProvider",
    "Model",
    "ModelProvider",
    "ModelResponse",
    "ModelSettings",
    "OpenAIModelProvider",
    "ProviderError",
    "get_model_provider",
]
