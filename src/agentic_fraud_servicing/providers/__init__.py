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
from agentic_fraud_servicing.providers.connectchain_provider import (
    ConnectChainModelProvider,
)

__all__ = [
    "ConnectChainModelProvider",
    "Model",
    "ModelProvider",
    "ModelResponse",
    "ModelSettings",
    "ProviderError",
    "get_model_provider",
]
