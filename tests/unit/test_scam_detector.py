"""Tests for the scam detector specialist agent module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_fraud_servicing.investigator.scam_detector import (
    SCAM_DETECTOR_INSTRUCTIONS,
    ScamAnalysis,
    run_scam_detection,
    scam_detector_agent,
)


class TestScamAnalysis:
    """Tests for the ScamAnalysis Pydantic model."""

    def test_defaults(self):
        """ScamAnalysis with all defaults has correct empty values."""
        result = ScamAnalysis()
        assert result.scam_likelihood == 0.0
        assert result.manipulation_indicators == []
        assert result.contradictions == []
        assert result.matched_patterns == []
        assert result.analysis_summary == ""

    def test_all_fields(self):
        """ScamAnalysis accepts all fields with correct types."""
        result = ScamAnalysis(
            scam_likelihood=0.75,
            manipulation_indicators=["urgency tactics", "story inconsistencies"],
            contradictions=[
                {
                    "claim": "didn't make purchase",
                    "contradicting_evidence": "chip+PIN auth at local POS",
                    "severity": "high",
                }
            ],
            matched_patterns=["first-party fraud"],
            analysis_summary="High likelihood of first-party fraud.",
        )
        assert result.scam_likelihood == 0.75
        assert len(result.manipulation_indicators) == 2
        assert len(result.contradictions) == 1
        assert result.contradictions[0]["severity"] == "high"
        assert "first-party fraud" in result.matched_patterns

    def test_scam_likelihood_validation_high(self):
        """ScamAnalysis rejects scam_likelihood above 1.0."""
        with pytest.raises(ValueError):
            ScamAnalysis(scam_likelihood=1.5)

    def test_scam_likelihood_validation_low(self):
        """ScamAnalysis rejects scam_likelihood below 0.0."""
        with pytest.raises(ValueError):
            ScamAnalysis(scam_likelihood=-0.1)

    def test_round_trip_json(self):
        """ScamAnalysis survives JSON round-trip serialization."""
        original = ScamAnalysis(
            scam_likelihood=0.6,
            manipulation_indicators=["coached language"],
            contradictions=[
                {
                    "claim": "card stolen last week",
                    "contradicting_evidence": "card used at enrolled device today",
                    "severity": "medium",
                }
            ],
            matched_patterns=["phishing"],
            analysis_summary="Moderate scam concern due to contradictions.",
        )
        json_str = original.model_dump_json()
        restored = ScamAnalysis.model_validate_json(json_str)
        assert restored == original

    def test_contradictions_dict_structure(self):
        """Each contradiction entry is a dict with expected keys."""
        result = ScamAnalysis(
            contradictions=[
                {
                    "claim": "never received item",
                    "contradicting_evidence": "delivery signed by cardholder",
                    "severity": "high",
                }
            ],
        )
        entry = result.contradictions[0]
        assert entry["claim"] == "never received item"
        assert entry["contradicting_evidence"] == "delivery signed by cardholder"
        assert entry["severity"] == "high"


class TestScamDetectorAgent:
    """Tests for the scam_detector_agent Agent instance."""

    def test_agent_name(self):
        """Agent has the correct name."""
        assert scam_detector_agent.name == "scam_detector"

    def test_agent_output_type(self):
        """Agent has ScamAnalysis as output_type."""
        assert scam_detector_agent.output_type.output_type is ScamAnalysis

    def test_instructions_cover_contradictions(self):
        """Instructions cover contradiction detection between claims and evidence."""
        assert "ALLEGATION" in SCAM_DETECTOR_INSTRUCTIONS
        assert "FACT" in SCAM_DETECTOR_INSTRUCTIONS
        assert "contradict" in SCAM_DETECTOR_INSTRUCTIONS.lower()

    def test_instructions_cover_manipulation(self):
        """Instructions cover manipulation indicator detection."""
        assert "manipulation" in SCAM_DETECTOR_INSTRUCTIONS.lower()
        assert "urgency" in SCAM_DETECTOR_INSTRUCTIONS.lower()
        assert "coached" in SCAM_DETECTOR_INSTRUCTIONS.lower()

    def test_instructions_cover_scam_patterns(self):
        """Instructions cover known scam pattern matching."""
        assert "Authorized Push Payment" in SCAM_DETECTOR_INSTRUCTIONS
        assert "romance scam" in SCAM_DETECTOR_INSTRUCTIONS.lower()
        assert "phishing" in SCAM_DETECTOR_INSTRUCTIONS.lower()
        assert "tech support" in SCAM_DETECTOR_INSTRUCTIONS.lower()

    def test_instructions_contain_four_categories(self):
        """Instructions include INVESTIGATION_CATEGORIES_REFERENCE content."""
        assert "THIRD_PARTY_FRAUD" in SCAM_DETECTOR_INSTRUCTIONS
        assert "FIRST_PARTY_FRAUD" in SCAM_DETECTOR_INSTRUCTIONS
        assert "SCAM" in SCAM_DETECTOR_INSTRUCTIONS
        assert "DISPUTE" in SCAM_DETECTOR_INSTRUCTIONS

    def test_instructions_distinguish_scam_vs_first_party_fraud(self):
        """Instructions explicitly distinguish SCAM from FIRST_PARTY_FRAUD."""
        lower = SCAM_DETECTOR_INSTRUCTIONS.lower()
        # External manipulator is the key distinguishing factor
        assert "external manipulator" in lower
        # SCAM = contradictions WITH external manipulator
        assert "with evidence of an external manipulator" in lower
        # FIRST_PARTY_FRAUD = contradictions WITHOUT external manipulator
        # (may be line-wrapped in the prompt text)
        normalized = " ".join(lower.split())
        assert "without evidence of an external manipulator" in normalized

    def test_instructions_document_first_party_fraud_signals(self):
        """Instructions document specific first-party fraud detection signals."""
        lower = SCAM_DETECTOR_INSTRUCTIONS.lower()
        assert "chip+pin contradiction" in lower
        assert "delivery proof contradiction" in lower
        assert "merchant familiarity" in lower
        assert "story shifts" in lower
        assert "first_party_fraud" in SCAM_DETECTOR_INSTRUCTIONS

    def test_instructions_two_axis_assessment(self):
        """Instructions describe the two-axis assessment framework."""
        lower = SCAM_DETECTOR_INSTRUCTIONS.lower()
        assert "contradiction level" in lower
        assert "external manipulator present" in lower


class TestRunScamDetection:
    """Tests for the run_scam_detection async function."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock ModelProvider."""
        return MagicMock()

    @pytest.fixture
    def sample_scam_result(self):
        """Create a sample ScamAnalysis with planted contradictions."""
        return ScamAnalysis(
            scam_likelihood=0.82,
            manipulation_indicators=["urgency tactics", "evasion of auth questions"],
            contradictions=[
                {
                    "claim": "I didn't make this purchase",
                    "contradicting_evidence": "chip+PIN auth at local POS with enrolled device",
                    "severity": "high",
                },
                {
                    "claim": "card was stolen last week",
                    "contradicting_evidence": "card used for online purchase yesterday",
                    "severity": "medium",
                },
            ],
            matched_patterns=["first-party fraud"],
            analysis_summary="Strong evidence of first-party fraud with multiple contradictions.",
        )

    async def test_returns_result(self, mock_provider, sample_scam_result):
        """run_scam_detection returns ScamAnalysis from Runner.run."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = sample_scam_result

        with patch(
            "agentic_fraud_servicing.investigator.scam_detector.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ):
            result = await run_scam_detection(
                [{"text": "I didn't make this purchase", "source_type": "ALLEGATION"}],
                [{"type": "AUTH_EVENT", "method": "chip_pin", "source_type": "FACT"}],
                "Caller claims unauthorized purchase but evidence shows chip+PIN.",
                mock_provider,
            )

        assert isinstance(result, ScamAnalysis)
        assert result.scam_likelihood == 0.82
        assert len(result.contradictions) == 2
        assert result.contradictions[0]["severity"] == "high"

    async def test_passes_model_provider(self, mock_provider):
        """run_scam_detection passes model_provider in RunConfig."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = ScamAnalysis()

        with patch(
            "agentic_fraud_servicing.investigator.scam_detector.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_scam_detection([], [], "summary", mock_provider)

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["run_config"].model_provider is mock_provider

    async def test_includes_allegations_in_message(self, mock_provider):
        """run_scam_detection includes ALLEGATION data in the user message."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = ScamAnalysis()

        with patch(
            "agentic_fraud_servicing.investigator.scam_detector.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_scam_detection(
                [{"text": "I was tricked into paying", "source_type": "ALLEGATION"}],
                [],
                "Caller describes manipulation",
                mock_provider,
            )

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "tricked into paying" in user_input
        assert "ALLEGATION" in user_input

    async def test_includes_facts_in_message(self, mock_provider):
        """run_scam_detection includes FACT evidence in the user message."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = ScamAnalysis()

        with patch(
            "agentic_fraud_servicing.investigator.scam_detector.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_scam_detection(
                [],
                [{"type": "TRANSACTION", "amount": 500, "source_type": "FACT"}],
                "Transaction evidence",
                mock_provider,
            )

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "FACT" in user_input
        assert "500" in user_input

    async def test_handles_empty_evidence(self, mock_provider):
        """run_scam_detection handles empty claims and facts gracefully."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = ScamAnalysis()

        with patch(
            "agentic_fraud_servicing.investigator.scam_detector.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_scam_detection([], [], "No evidence available", mock_provider)

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "[]" in user_input

    async def test_wraps_exceptions(self, mock_provider):
        """run_scam_detection wraps SDK exceptions in RuntimeError."""
        with patch(
            "agentic_fraud_servicing.investigator.scam_detector.run_with_retry",
            new_callable=AsyncMock,
            side_effect=ValueError("LLM call failed"),
        ):
            with pytest.raises(RuntimeError, match="Scam detector agent failed"):
                await run_scam_detection(
                    [{"text": "claim"}], [{"type": "fact"}], "summary", mock_provider
                )
