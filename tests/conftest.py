"""Shared test fixtures and configuration for pytest.

Provides reusable fixtures for both unit and integration tests:
mock_model_provider, sample_transcript_events, sample_case, gateway_factory.
"""

from unittest.mock import MagicMock

import pytest

from agentic_fraud_servicing.models.case import Case
from agentic_fraud_servicing.models.enums import AllegationType, CaseStatus, SpeakerType
from agentic_fraud_servicing.models.transcript import TranscriptEvent
from agentic_fraud_servicing.providers.base import ModelProvider
from agentic_fraud_servicing.ui.helpers import create_gateway


@pytest.fixture(scope="session")
def mock_model_provider() -> MagicMock:
    """Return a MagicMock with ModelProvider spec for mocking LLM calls."""
    return MagicMock(spec=ModelProvider)


@pytest.fixture()
def sample_transcript_events() -> list[TranscriptEvent]:
    """Return a minimal fraud call transcript with 4 events.

    Covers CCP greeting, CARDMEMBER claim, SYSTEM auth status, CCP follow-up.
    """
    return [
        TranscriptEvent(
            call_id="call-test-001",
            event_id="evt-001",
            timestamp_ms=1000,
            speaker=SpeakerType.CCP,
            text="Thank you for calling American Express. How can I help you today?",
        ),
        TranscriptEvent(
            call_id="call-test-001",
            event_id="evt-002",
            timestamp_ms=5000,
            speaker=SpeakerType.CARDMEMBER,
            text="I see a charge for $499.99 at Electronics Store that I did not make.",
        ),
        TranscriptEvent(
            call_id="call-test-001",
            event_id="evt-003",
            timestamp_ms=6000,
            speaker=SpeakerType.SYSTEM,
            text="Authentication status: KBA passed. Device recognized.",
        ),
        TranscriptEvent(
            call_id="call-test-001",
            event_id="evt-004",
            timestamp_ms=10000,
            speaker=SpeakerType.CCP,
            text="I can see the charge. When did you first notice this transaction?",
        ),
    ]


@pytest.fixture()
def sample_case() -> Case:
    """Return a Case instance with realistic fields for testing."""
    return Case(
        case_id="case-test-001",
        call_id="call-test-001",
        customer_id="cust-12345",
        account_id="acct-67890",
        allegation_type=AllegationType.FRAUD,
        status=CaseStatus.OPEN,
    )


@pytest.fixture()
def gateway_factory():
    """Factory fixture that creates a ToolGateway with real SQLite stores.

    Returns a callable: ``make_gateway(tmp_path)`` -> ToolGateway.
    Each call creates fresh stores under the given directory.
    """

    def _make_gateway(tmp_path):
        return create_gateway(tmp_path)

    return _make_gateway
