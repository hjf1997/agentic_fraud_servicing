"""Unit tests for the config module."""

from agentic_fraud_servicing.config import Settings, get_settings


def test_default_settings(monkeypatch):
    """get_settings() returns correct defaults when no env vars are set."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_BEDROCK_MODEL_ID", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()

    settings = get_settings()

    assert isinstance(settings, Settings)
    assert settings.llm_provider == "bedrock"
    assert settings.aws_profile == "default"
    assert settings.aws_region == "us-east-1"
    assert settings.aws_bedrock_model_id == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    assert settings.openai_api_key is None

    get_settings.cache_clear()


def test_env_override_llm_provider(monkeypatch):
    """Environment variables override default settings."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_BEDROCK_MODEL_ID", raising=False)

    settings = Settings()

    assert settings.llm_provider == "openai"
    assert settings.openai_api_key == "sk-test-key"


def test_env_override_aws_fields(monkeypatch):
    """AWS-related env vars are picked up correctly."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AWS_PROFILE", "prod-profile")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_BEDROCK_MODEL_ID", "custom-model-id")

    settings = Settings()

    assert settings.aws_profile == "prod-profile"
    assert settings.aws_region == "eu-west-1"
    assert settings.aws_bedrock_model_id == "custom-model-id"


def test_invalid_llm_provider_raises(monkeypatch):
    """Invalid LLM_PROVIDER raises ValueError."""
    monkeypatch.setenv("LLM_PROVIDER", "invalid")

    try:
        Settings()
        raise AssertionError("Expected ValueError")
    except ValueError as exc:
        assert "LLM_PROVIDER must be one of" in str(exc)
        assert "'invalid'" in str(exc)


def test_openai_without_api_key_raises(monkeypatch):
    """LLM_PROVIDER='openai' without OPENAI_API_KEY raises ValueError."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    try:
        Settings()
        raise AssertionError("Expected ValueError")
    except ValueError as exc:
        assert "OPENAI_API_KEY is required" in str(exc)


def test_openai_with_api_key_succeeds(monkeypatch):
    """LLM_PROVIDER='openai' with OPENAI_API_KEY set creates settings successfully."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-valid-key")

    settings = Settings()

    assert settings.llm_provider == "openai"
    assert settings.openai_api_key == "sk-valid-key"


def test_get_settings_caches(monkeypatch):
    """get_settings() returns the same cached instance on repeated calls."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()

    first = get_settings()
    second = get_settings()

    assert first is second

    get_settings.cache_clear()
