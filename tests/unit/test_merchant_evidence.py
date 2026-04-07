"""Tests for the merchant evidence specialist agent module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_fraud_servicing.investigator.merchant_evidence import (
    MERCHANT_INSTRUCTIONS,
    MerchantAnalysis,
    merchant_agent,
    run_merchant_analysis,
)


class TestMerchantAnalysis:
    """Tests for the MerchantAnalysis Pydantic model."""

    def test_defaults(self):
        """MerchantAnalysis with all defaults has correct empty values."""
        result = MerchantAnalysis()
        assert result.normalized_merchants == []
        assert result.conflicts == []
        assert result.consolidated_summary == ""
        assert result.merchant_risk_score == 0.0
        assert result.recommendations == []

    def test_all_fields(self):
        """MerchantAnalysis accepts all fields with correct types."""
        result = MerchantAnalysis(
            normalized_merchants=[
                {
                    "merchant_id": "M001",
                    "normalized_name": "Amazon Marketplace",
                    "category": "retail",
                    "dispute_history_count": 3,
                    "risk_indicators": ["high dispute count"],
                }
            ],
            conflicts=[
                {
                    "description": "Duplicate charge at Amazon",
                    "severity": "high",
                    "evidence_refs": ["txn_001", "txn_002"],
                }
            ],
            consolidated_summary="Single merchant with duplicate charge detected.",
            merchant_risk_score=0.75,
            recommendations=["Request merchant response on duplicate charge"],
        )
        assert len(result.normalized_merchants) == 1
        assert result.normalized_merchants[0]["normalized_name"] == "Amazon Marketplace"
        assert len(result.conflicts) == 1
        assert result.conflicts[0]["severity"] == "high"
        assert result.merchant_risk_score == 0.75
        assert len(result.recommendations) == 1

    def test_round_trip_json(self):
        """MerchantAnalysis survives JSON round-trip serialization."""
        original = MerchantAnalysis(
            normalized_merchants=[
                {
                    "merchant_id": "M002",
                    "normalized_name": "Square - Coffee Shop",
                    "category": "dining",
                    "dispute_history_count": 0,
                    "risk_indicators": [],
                }
            ],
            conflicts=[],
            consolidated_summary="No conflicts detected.",
            merchant_risk_score=0.1,
            recommendations=[],
        )
        json_str = original.model_dump_json()
        restored = MerchantAnalysis.model_validate_json(json_str)
        assert restored == original

    def test_conflicts_dict_structure(self):
        """Conflict entries have expected keys."""
        result = MerchantAnalysis(
            conflicts=[
                {
                    "description": "Amount mismatch",
                    "severity": "medium",
                    "evidence_refs": ["txn_010"],
                }
            ],
        )
        entry = result.conflicts[0]
        assert entry["description"] == "Amount mismatch"
        assert entry["severity"] == "medium"
        assert "txn_010" in entry["evidence_refs"]


class TestMerchantAgent:
    """Tests for the merchant_agent Agent instance."""

    def test_agent_name(self):
        """Agent has the correct name."""
        assert merchant_agent.name == "merchant_evidence"

    def test_agent_output_type(self):
        """Agent has MerchantAnalysis as output_type."""
        assert merchant_agent.output_type.output_type is MerchantAnalysis

    def test_instructions_cover_normalization(self):
        """Instructions reference merchant name normalization."""
        assert "Normalize merchant names" in MERCHANT_INSTRUCTIONS
        assert "AMZN*Marketplace" in MERCHANT_INSTRUCTIONS

    def test_instructions_cover_conflict_detection(self):
        """Instructions reference conflict detection."""
        assert "conflicts" in MERCHANT_INSTRUCTIONS.lower()
        assert "duplicate charge" in MERCHANT_INSTRUCTIONS.lower()

    def test_instructions_cover_risk_assessment(self):
        """Instructions reference risk assessment."""
        assert "risk score" in MERCHANT_INSTRUCTIONS.lower()
        assert "category" in MERCHANT_INSTRUCTIONS.lower()

    def test_instructions_contain_four_categories(self):
        """Instructions reference all 4 investigation categories."""
        assert "THIRD_PARTY_FRAUD" in MERCHANT_INSTRUCTIONS
        assert "FIRST_PARTY_FRAUD" in MERCHANT_INSTRUCTIONS
        assert "SCAM" in MERCHANT_INSTRUCTIONS
        assert "DISPUTE" in MERCHANT_INSTRUCTIONS

    def test_instructions_merchant_category_context(self):
        """Instructions contextualize merchant analysis for each category."""
        # Third-party fraud: merchant uninvolved
        assert "uninvolved" in MERCHANT_INSTRUCTIONS.lower()
        # First-party fraud: merchant is legitimate
        assert "legitimate" in MERCHANT_INSTRUCTIONS.lower()
        # Scam: merchant may be fake
        assert "fake" in MERCHANT_INSTRUCTIONS.lower()
        # Dispute: merchant performance
        assert "delivery proof" in MERCHANT_INSTRUCTIONS.lower()


class TestRunMerchantAnalysis:
    """Tests for the run_merchant_analysis async function."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock ModelProvider."""
        return MagicMock()

    @pytest.fixture
    def sample_result(self):
        """Create a sample MerchantAnalysis result."""
        return MerchantAnalysis(
            normalized_merchants=[
                {
                    "merchant_id": "M001",
                    "normalized_name": "Amazon Marketplace",
                    "category": "retail",
                    "dispute_history_count": 2,
                    "risk_indicators": ["online merchant"],
                }
            ],
            conflicts=[],
            consolidated_summary="Single merchant, no conflicts.",
            merchant_risk_score=0.3,
            recommendations=["Verify delivery status"],
        )

    async def test_returns_result(self, mock_provider, sample_result):
        """run_merchant_analysis returns MerchantAnalysis from Runner.run."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = sample_result

        with patch(
            "agentic_fraud_servicing.investigator.merchant_evidence.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ):
            result = await run_merchant_analysis(
                [{"merchant_id": "M001", "name": "AMZN*Marketplace"}],
                [{"amount": 99.99, "merchant": "AMZN*Marketplace"}],
                mock_provider,
            )

        assert isinstance(result, MerchantAnalysis)
        assert result.merchant_risk_score == 0.3

    async def test_passes_model_provider(self, mock_provider):
        """run_merchant_analysis passes model_provider in RunConfig."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = MerchantAnalysis()

        with patch(
            "agentic_fraud_servicing.investigator.merchant_evidence.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_merchant_analysis([], [], mock_provider)

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["run_config"].model_provider is mock_provider

    async def test_includes_merchant_evidence(self, mock_provider):
        """run_merchant_analysis includes merchant evidence in the user message."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = MerchantAnalysis()

        with patch(
            "agentic_fraud_servicing.investigator.merchant_evidence.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_merchant_analysis(
                [{"merchant_id": "M099", "name": "TestMerchant"}],
                [],
                mock_provider,
            )

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "M099" in user_input
        assert "TestMerchant" in user_input

    async def test_includes_transaction_evidence(self, mock_provider):
        """run_merchant_analysis includes transaction evidence in the user message."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = MerchantAnalysis()

        with patch(
            "agentic_fraud_servicing.investigator.merchant_evidence.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_merchant_analysis(
                [],
                [{"amount": 250.00, "merchant": "STORE_XYZ"}],
                mock_provider,
            )

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "250.0" in user_input
        assert "STORE_XYZ" in user_input

    async def test_handles_empty_evidence(self, mock_provider):
        """run_merchant_analysis handles empty evidence lists gracefully."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = MerchantAnalysis()

        with patch(
            "agentic_fraud_servicing.investigator.merchant_evidence.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            result = await run_merchant_analysis([], [], mock_provider)

        assert isinstance(result, MerchantAnalysis)
        # Empty lists should still be serialized as JSON arrays
        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "[]" in user_input

    async def test_wraps_exceptions(self, mock_provider):
        """run_merchant_analysis wraps SDK exceptions in RuntimeError."""
        with patch(
            "agentic_fraud_servicing.investigator.merchant_evidence.run_with_retry",
            new_callable=AsyncMock,
            side_effect=ValueError("LLM call failed"),
        ):
            with pytest.raises(RuntimeError, match="Merchant evidence agent failed"):
                await run_merchant_analysis(
                    [{"merchant_id": "M001"}],
                    [{"amount": 50}],
                    mock_provider,
                )
