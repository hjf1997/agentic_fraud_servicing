"""Unit tests for the config module."""

from agentic_fraud_servicing.config import Settings, get_settings


def test_default_settings(monkeypatch):
    """get_settings() returns correct defaults when env vars are set."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("CONNECTCHAIN_MODEL_INDEX", "0")
    get_settings.cache_clear()

    settings = get_settings()

    assert isinstance(settings, Settings)
    assert settings.llm_provider == "connectchain"
    assert settings.connectchain_model_index == "0"

    get_settings.cache_clear()


def test_env_override_connectchain(monkeypatch):
    """ConnectChain environment variables are picked up correctly."""
    monkeypatch.setenv("LLM_PROVIDER", "connectchain")
    monkeypatch.setenv("CONNECTCHAIN_MODEL_INDEX", "2")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-12-01")
    monkeypatch.setenv("AZURE_OPENAI_USE_RESPONSES_API", "true")

    settings = Settings()

    assert settings.llm_provider == "connectchain"
    assert settings.connectchain_model_index == "2"
    assert settings.azure_openai_api_version == "2024-12-01"
    assert settings.azure_openai_use_responses_api is True


def test_invalid_llm_provider_raises(monkeypatch):
    """Invalid LLM_PROVIDER raises ValueError."""
    monkeypatch.setenv("LLM_PROVIDER", "invalid")
    monkeypatch.setenv("CONNECTCHAIN_MODEL_INDEX", "0")

    try:
        Settings()
        raise AssertionError("Expected ValueError")
    except ValueError as exc:
        assert "LLM_PROVIDER must be one of" in str(exc)
        assert "'invalid'" in str(exc)


def test_connectchain_without_model_index_raises(monkeypatch):
    """LLM_PROVIDER='connectchain' without CONNECTCHAIN_MODEL_INDEX raises ValueError."""
    monkeypatch.setenv("LLM_PROVIDER", "connectchain")
    monkeypatch.delenv("CONNECTCHAIN_MODEL_INDEX", raising=False)

    try:
        Settings()
        raise AssertionError("Expected ValueError")
    except ValueError as exc:
        assert "CONNECTCHAIN_MODEL_INDEX is required" in str(exc)


def test_connectchain_with_model_index_succeeds(monkeypatch):
    """LLM_PROVIDER='connectchain' with CONNECTCHAIN_MODEL_INDEX set creates settings."""
    monkeypatch.setenv("LLM_PROVIDER", "connectchain")
    monkeypatch.setenv("CONNECTCHAIN_MODEL_INDEX", "1")

    settings = Settings()

    assert settings.llm_provider == "connectchain"
    assert settings.connectchain_model_index == "1"


def test_get_settings_caches(monkeypatch):
    """get_settings() returns the same cached instance on repeated calls."""
    monkeypatch.setenv("LLM_PROVIDER", "connectchain")
    monkeypatch.setenv("CONNECTCHAIN_MODEL_INDEX", "0")
    get_settings.cache_clear()

    first = get_settings()
    second = get_settings()

    assert first is second

    get_settings.cache_clear()
