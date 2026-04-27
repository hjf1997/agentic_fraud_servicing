"""Tests for the Gradio web demo interface."""

from unittest.mock import AsyncMock, MagicMock, patch

import gradio as gr
import pytest

from agentic_fraud_servicing.ui.gradio_app import (
    create_app,
    process_transcript,
)


class TestCreateApp:
    """Tests for create_app factory function."""

    def test_returns_blocks_instance(self):
        """create_app() returns a gr.Blocks."""
        app = create_app()
        assert isinstance(app, gr.Blocks)


class TestProcessTranscript:
    """Tests for the process_transcript callback."""

    @pytest.fixture
    def _mock_infra(self):
        """Mock gateway, provider, and orchestrator for copilot tests."""
        suggestion = MagicMock()
        suggestion.model_dump.return_value = {
            "call_id": "c1",
            "running_summary": "test",
        }

        orchestrator = AsyncMock()
        orchestrator.process_event.return_value = suggestion

        with (
            patch("agentic_fraud_servicing.ui.gradio_app.create_gateway") as mock_gw,
            patch("agentic_fraud_servicing.ui.gradio_app.create_provider") as mock_prov,
            patch("agentic_fraud_servicing.ui.gradio_app.CopilotOrchestrator") as mock_orch_cls,
        ):
            mock_gw.return_value = MagicMock()
            mock_prov.return_value = MagicMock()
            mock_orch_cls.return_value = orchestrator
            yield {
                "gateway": mock_gw,
                "provider": mock_prov,
                "orchestrator_cls": mock_orch_cls,
                "orchestrator": orchestrator,
                "suggestion": suggestion,
            }

    @pytest.mark.usefixtures("_mock_infra")
    async def test_valid_json(self, _mock_infra):
        """Valid transcript JSON returns list of suggestion dicts."""
        transcript = (
            '[{"call_id":"c1","event_id":"e1","timestamp_ms":0,"speaker":"CCP","text":"hello"}]'
        )
        result = await process_transcript(transcript, "data/test")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["call_id"] == "c1"
        _mock_infra["orchestrator"].process_event.assert_called_once()

    async def test_invalid_json_returns_error(self):
        """Invalid JSON returns an error dict in a list."""
        result = await process_transcript("not valid json", "data/test")

        assert isinstance(result, list)
        assert len(result) == 1
        assert "error" in result[0]

    @pytest.mark.usefixtures("_mock_infra")
    async def test_redacts_pii(self, _mock_infra):
        """PAN in transcript is redacted before reaching the orchestrator."""
        # AMEX PAN embedded in transcript text
        transcript = (
            '[{"call_id":"c1","event_id":"e1",'
            '"timestamp_ms":0,"speaker":"CCP",'
            '"text":"card 371449635398431"}]'
        )
        await process_transcript(transcript, "data/test")

        # The orchestrator receives a TranscriptEvent with redacted text
        call_args = _mock_infra["orchestrator"].process_event.call_args
        event = call_args[0][0]
        assert "371449635398431" not in event.text
        assert "[PAN_REDACTED]" in event.text
