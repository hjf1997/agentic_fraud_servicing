"""Application configuration loaded from environment variables.

This is a leaf dependency — imported by all other modules, imports nothing
from the project. Uses python-dotenv to load .env file at import time.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

VALID_LLM_PROVIDERS = ("openai", "bedrock", "connectchain")


class Settings:
    """Application settings populated from environment variables.

    Attributes:
        llm_provider: LLM backend to use ('openai' or 'bedrock').
        aws_profile: AWS credentials profile for Bedrock access.
        aws_region: AWS region for the Bedrock endpoint.
        aws_bedrock_model_id: Bedrock model identifier.
        openai_api_key: OpenAI API key (required when llm_provider is 'openai').
    """

    def __init__(self) -> None:
        self.llm_provider: str = os.environ.get("LLM_PROVIDER", "bedrock")
        self.aws_profile: str = os.environ.get("AWS_PROFILE", "default")
        self.aws_region: str = os.environ.get("AWS_REGION", "us-east-1")
        self.aws_bedrock_model_id: str = os.environ.get(
            "AWS_BEDROCK_MODEL_ID",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        )
        self.openai_api_key: str | None = os.environ.get("OPENAI_API_KEY")
        self.connectchain_model_index: str | None = os.environ.get(
            "CONNECTCHAIN_MODEL_INDEX"
        )
        self.azure_openai_api_version: str | None = os.environ.get(
            "AZURE_OPENAI_API_VERSION"
        )

        self._validate()

    def _validate(self) -> None:
        """Validate settings after loading."""
        if self.llm_provider not in VALID_LLM_PROVIDERS:
            raise ValueError(
                f"LLM_PROVIDER must be one of {VALID_LLM_PROVIDERS}, got '{self.llm_provider}'"
            )

        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER is 'openai'")

        if self.llm_provider == "connectchain" and not self.connectchain_model_index:
            raise ValueError(
                "CONNECTCHAIN_MODEL_INDEX is required when LLM_PROVIDER is 'connectchain'"
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance.

    The settings are created once on first call and reused thereafter.
    To force re-creation (e.g., in tests), call get_settings.cache_clear().
    """
    return Settings()
