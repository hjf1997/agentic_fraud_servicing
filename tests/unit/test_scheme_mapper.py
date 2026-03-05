"""Tests for the scheme mapper specialist agent module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_fraud_servicing.investigator.scheme_mapper import (
    SCHEME_MAPPER_INSTRUCTIONS,
    SchemeMappingResult,
    run_scheme_mapping,
    scheme_mapper_agent,
)


class TestSchemeMappingResult:
    """Tests for the SchemeMappingResult Pydantic model."""

    def test_defaults(self):
        """SchemeMappingResult with all defaults has correct empty values."""
        result = SchemeMappingResult()
        assert result.reason_codes == []
        assert result.primary_reason_code == ""
        assert result.primary_network == ""
        assert result.documentation_gaps == []
        assert result.analysis_summary == ""

    def test_all_fields(self):
        """SchemeMappingResult accepts all fields with correct types."""
        result = SchemeMappingResult(
            reason_codes=[
                {
                    "network": "AMEX",
                    "code": "FR2",
                    "description": "Fraud Full Recourse",
                    "match_confidence": 0.9,
                },
                {
                    "network": "VISA",
                    "code": "10.4",
                    "description": "Other Fraud",
                    "match_confidence": 0.7,
                },
            ],
            primary_reason_code="FR2",
            primary_network="AMEX",
            documentation_gaps=["police report", "signed affidavit"],
            analysis_summary="Unauthorized transaction maps to AMEX FR2.",
        )
        assert len(result.reason_codes) == 2
        assert result.primary_reason_code == "FR2"
        assert result.primary_network == "AMEX"
        assert len(result.documentation_gaps) == 2
        assert "police report" in result.documentation_gaps

    def test_round_trip_json(self):
        """SchemeMappingResult survives JSON round-trip serialization."""
        original = SchemeMappingResult(
            reason_codes=[
                {
                    "network": "MC",
                    "code": "4837",
                    "description": "No Cardholder Authorization",
                    "match_confidence": 0.85,
                }
            ],
            primary_reason_code="4837",
            primary_network="MC",
            documentation_gaps=["device logs"],
            analysis_summary="Maps to Mastercard 4837.",
        )
        json_str = original.model_dump_json()
        restored = SchemeMappingResult.model_validate_json(json_str)
        assert restored == original

    def test_reason_codes_dict_structure(self):
        """Each reason code entry is a dict with expected keys."""
        result = SchemeMappingResult(
            reason_codes=[
                {
                    "network": "VISA",
                    "code": "13.1",
                    "description": "Merchandise/Services Not Received",
                    "match_confidence": 0.8,
                }
            ],
        )
        entry = result.reason_codes[0]
        assert entry["network"] == "VISA"
        assert entry["code"] == "13.1"
        assert entry["match_confidence"] == 0.8


class TestSchemeMapperAgent:
    """Tests for the scheme_mapper_agent Agent instance."""

    def test_agent_name(self):
        """Agent has the correct name."""
        assert scheme_mapper_agent.name == "scheme_mapper"

    def test_agent_output_type(self):
        """Agent has SchemeMappingResult as output_type."""
        assert scheme_mapper_agent.output_type is SchemeMappingResult

    def test_instructions_reference_amex(self):
        """Instructions reference AMEX reason codes."""
        assert "AMEX" in SCHEME_MAPPER_INSTRUCTIONS
        assert "C08" in SCHEME_MAPPER_INSTRUCTIONS
        assert "FR2" in SCHEME_MAPPER_INSTRUCTIONS

    def test_instructions_reference_visa(self):
        """Instructions reference Visa reason codes."""
        assert "Visa" in SCHEME_MAPPER_INSTRUCTIONS
        assert "10.4" in SCHEME_MAPPER_INSTRUCTIONS
        assert "13.1" in SCHEME_MAPPER_INSTRUCTIONS

    def test_instructions_reference_mastercard(self):
        """Instructions reference Mastercard reason codes."""
        assert "Mastercard" in SCHEME_MAPPER_INSTRUCTIONS
        assert "4837" in SCHEME_MAPPER_INSTRUCTIONS
        assert "4853" in SCHEME_MAPPER_INSTRUCTIONS


class TestRunSchemeMapping:
    """Tests for the run_scheme_mapping async function."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock ModelProvider."""
        return MagicMock()

    @pytest.fixture
    def sample_fraud_result(self):
        """Create a sample SchemeMappingResult for fraud scenario."""
        return SchemeMappingResult(
            reason_codes=[
                {
                    "network": "AMEX",
                    "code": "FR2",
                    "description": "Fraud Full Recourse",
                    "match_confidence": 0.92,
                }
            ],
            primary_reason_code="FR2",
            primary_network="AMEX",
            documentation_gaps=["police report", "signed affidavit"],
            analysis_summary="Unauthorized transaction maps to AMEX FR2.",
        )

    async def test_returns_result(self, mock_provider, sample_fraud_result):
        """run_scheme_mapping returns SchemeMappingResult from Runner.run."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = sample_fraud_result

        with patch(
            "agentic_fraud_servicing.investigator.scheme_mapper.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ):
            result = await run_scheme_mapping(
                "Card used at unknown merchant",
                "fraud",
                ["I didn't make this purchase"],
                "Transaction of $500 at Store X",
                mock_provider,
            )

        assert isinstance(result, SchemeMappingResult)
        assert result.primary_reason_code == "FR2"
        assert result.primary_network == "AMEX"

    async def test_passes_model_provider(self, mock_provider):
        """run_scheme_mapping passes model_provider in RunConfig."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = SchemeMappingResult()

        with patch(
            "agentic_fraud_servicing.investigator.scheme_mapper.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_scheme_mapping("summary", "fraud", [], "evidence", mock_provider)

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["run_config"].model_provider is mock_provider

    async def test_includes_claims_in_message(self, mock_provider):
        """run_scheme_mapping includes claims in the user message."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = SchemeMappingResult()

        with patch(
            "agentic_fraud_servicing.investigator.scheme_mapper.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_scheme_mapping(
                "Case summary",
                "dispute",
                ["charged twice", "item never arrived"],
                "Two transactions found",
                mock_provider,
            )

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "charged twice" in user_input
        assert "item never arrived" in user_input

    async def test_includes_allegation_type(self, mock_provider):
        """run_scheme_mapping includes allegation type in the user message."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = SchemeMappingResult()

        with patch(
            "agentic_fraud_servicing.investigator.scheme_mapper.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_scheme_mapping("summary", "scam", ["tricked"], "evidence", mock_provider)

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "scam" in user_input

    async def test_handles_empty_claims(self, mock_provider):
        """run_scheme_mapping handles empty claims list gracefully."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = SchemeMappingResult()

        with patch(
            "agentic_fraud_servicing.investigator.scheme_mapper.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_scheme_mapping("summary", "fraud", [], "evidence", mock_provider)

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "(none)" in user_input

    async def test_wraps_exceptions(self, mock_provider):
        """run_scheme_mapping wraps SDK exceptions in RuntimeError."""
        with patch(
            "agentic_fraud_servicing.investigator.scheme_mapper.Runner.run",
            new_callable=AsyncMock,
            side_effect=ValueError("LLM call failed"),
        ):
            with pytest.raises(RuntimeError, match="Scheme mapper agent failed"):
                await run_scheme_mapping("summary", "fraud", ["claim"], "evidence", mock_provider)
