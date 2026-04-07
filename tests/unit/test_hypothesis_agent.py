"""Tests for the hypothesis scoring arbitrator and specialist panel."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_fraud_servicing.copilot.hypothesis_agent import (
    HYPOTHESIS_INSTRUCTIONS,
    HypothesisAssessment,
    hypothesis_agent,
    run_arbitrator,
)
from agentic_fraud_servicing.copilot.hypothesis_specialists import (
    SpecialistAssessment,
    run_specialists,
)

# ---------------------------------------------------------------------------
# SpecialistAssessment model tests
# ---------------------------------------------------------------------------


class TestSpecialistAssessment:
    """Tests for the SpecialistAssessment Pydantic model."""

    def test_defaults(self):
        """SpecialistAssessment with required fields has correct defaults."""
        assessment = SpecialistAssessment(category="DISPUTE")
        assert assessment.category == "DISPUTE"
        assert assessment.likelihood == 0.0
        assert assessment.reasoning == ""
        assert assessment.supporting_evidence == []
        assert assessment.contradicting_evidence == []
        assert assessment.policy_citations == []
        assert assessment.evidence_gaps == []
        assert assessment.eligibility == "eligible"

    def test_all_fields(self):
        """SpecialistAssessment accepts all fields."""
        assessment = SpecialistAssessment(
            category="SCAM",
            likelihood=0.7,
            reasoning="Strong social engineering indicators.",
            supporting_evidence=["Coached language in transcript"],
            contradicting_evidence=["No unusual payment method"],
            policy_citations=["Per fraud_case_checklist.md: 'scam documentation'"],
            evidence_gaps=["Communication trail with scammer"],
            eligibility="blocked",
        )
        assert assessment.likelihood == 0.7
        assert len(assessment.supporting_evidence) == 1
        assert len(assessment.policy_citations) == 1
        assert assessment.evidence_gaps == ["Communication trail with scammer"]
        assert assessment.eligibility == "blocked"

    def test_json_round_trip(self):
        """SpecialistAssessment survives JSON serialization."""
        original = SpecialistAssessment(
            category="THIRD_PARTY_FRAUD",
            likelihood=0.4,
            reasoning="Some unfamiliar device activity.",
            evidence_gaps=["Device fingerprint data"],
            eligibility="eligible",
        )
        json_str = original.model_dump_json()
        restored = SpecialistAssessment.model_validate_json(json_str)
        assert restored == original


# ---------------------------------------------------------------------------
# HypothesisAssessment model tests
# ---------------------------------------------------------------------------


class TestHypothesisAssessment:
    """Tests for the HypothesisAssessment Pydantic model."""

    def test_defaults(self):
        """Default scores are uniform 0.25 across all 4 categories."""
        assessment = HypothesisAssessment()
        assert assessment.scores == {
            "THIRD_PARTY_FRAUD": 0.25,
            "FIRST_PARTY_FRAUD": 0.25,
            "SCAM": 0.25,
            "DISPUTE": 0.25,
        }
        assert assessment.contradictions == []
        assert assessment.assessment_summary == ""
        assert assessment.specialist_assessments == {}

    def test_all_fields(self):
        """All fields can be set explicitly."""
        assessment = HypothesisAssessment(
            scores={
                "THIRD_PARTY_FRAUD": 0.1,
                "FIRST_PARTY_FRAUD": 0.6,
                "SCAM": 0.2,
                "DISPUTE": 0.1,
            },
            reasoning={
                "THIRD_PARTY_FRAUD": "Low — auth from enrolled device",
                "FIRST_PARTY_FRAUD": "High — chip+PIN contradicts claim",
                "SCAM": "Moderate — some urgency in language",
                "DISPUTE": "Low — no merchant issue",
            },
            contradictions=[
                "CM claims unauthorized but chip+PIN auth from enrolled device",
                "CM knew merchant name before being told",
            ],
            assessment_summary="Strong indicators of first-party fraud.",
        )
        assert assessment.scores["FIRST_PARTY_FRAUD"] == 0.6
        assert len(assessment.contradictions) == 2
        assert "first-party fraud" in assessment.assessment_summary

    def test_scores_dict_has_four_keys(self):
        """Default scores dict contains exactly the 4 investigation categories."""
        assessment = HypothesisAssessment()
        expected_keys = {
            "THIRD_PARTY_FRAUD",
            "FIRST_PARTY_FRAUD",
            "SCAM",
            "DISPUTE",
        }
        assert set(assessment.scores.keys()) == expected_keys

    def test_reasoning_dict_has_four_keys(self):
        """Default reasoning dict contains exactly the 4 investigation categories."""
        assessment = HypothesisAssessment()
        expected_keys = {
            "THIRD_PARTY_FRAUD",
            "FIRST_PARTY_FRAUD",
            "SCAM",
            "DISPUTE",
        }
        assert set(assessment.reasoning.keys()) == expected_keys

    def test_contradictions_is_list(self):
        """Contradictions field is a list of strings."""
        assessment = HypothesisAssessment(contradictions=["Claim A contradicts Evidence B"])
        assert isinstance(assessment.contradictions, list)
        assert assessment.contradictions[0] == "Claim A contradicts Evidence B"

    def test_json_round_trip(self):
        """HypothesisAssessment survives JSON serialization round-trip."""
        original = HypothesisAssessment(
            scores={
                "THIRD_PARTY_FRAUD": 0.1,
                "FIRST_PARTY_FRAUD": 0.7,
                "SCAM": 0.1,
                "DISPUTE": 0.1,
            },
            reasoning={
                "THIRD_PARTY_FRAUD": "Low",
                "FIRST_PARTY_FRAUD": "High",
                "SCAM": "Low",
                "DISPUTE": "Low",
            },
            contradictions=["chip+PIN from enrolled device"],
            assessment_summary="Likely first-party fraud.",
        )
        json_str = original.model_dump_json()
        restored = HypothesisAssessment.model_validate_json(json_str)
        assert restored.scores == original.scores
        assert restored.reasoning == original.reasoning
        assert restored.contradictions == original.contradictions
        assert restored.assessment_summary == original.assessment_summary

    def test_specialist_assessments_excluded_from_json(self):
        """specialist_assessments is excluded from JSON serialization."""
        assessment = HypothesisAssessment()
        assessment.specialist_assessments = {
            "DISPUTE": SpecialistAssessment(category="DISPUTE", likelihood=0.5),
        }
        dumped = assessment.model_dump()
        assert "specialist_assessments" not in dumped

    def test_specialist_assessments_stored(self):
        """specialist_assessments can be set and read programmatically."""
        assessment = HypothesisAssessment()
        sa = SpecialistAssessment(category="DISPUTE", likelihood=0.8)
        assessment.specialist_assessments = {"DISPUTE": sa}
        assert assessment.specialist_assessments["DISPUTE"].likelihood == 0.8


# ---------------------------------------------------------------------------
# Agent instance tests
# ---------------------------------------------------------------------------


class TestHypothesisAgent:
    """Tests for the hypothesis_agent (arbitrator) Agent instance."""

    def test_agent_name(self):
        """Agent has the correct name."""
        assert hypothesis_agent.name == "hypothesis"

    def test_agent_output_type(self):
        """Agent has HypothesisAssessment as output_type."""
        assert hypothesis_agent.output_type.output_type is HypothesisAssessment

    def test_instructions_contain_arbitrator_concepts(self):
        """Instructions reference arbitrator-specific concepts."""
        lower = HYPOTHESIS_INSTRUCTIONS.lower()
        assert "specialist" in lower
        assert "first_party_fraud" in lower or "first-party fraud" in lower
        assert "arbitrator" in lower
        assert "probability distribution" in lower
        assert "bayesian prior" in lower or "prior" in lower

    def test_instructions_contain_first_party_fraud_detection(self):
        """Instructions describe first-party fraud cross-cutting detection."""
        lower = HYPOTHESIS_INSTRUCTIONS.lower()
        assert "cross-cutting" in lower or "cross-specialist" in lower
        assert "external manipulator" in lower or "external deceiver" in lower
        assert "contradicting evidence" in lower


# ---------------------------------------------------------------------------
# run_specialists tests
# ---------------------------------------------------------------------------


class TestRunSpecialists:
    """Tests for the run_specialists parallel runner."""

    @pytest.fixture
    def mock_provider(self):
        return MagicMock()

    async def test_returns_three_assessments(self, mock_provider):
        """run_specialists returns dict with 3 category keys."""
        mock_result = MagicMock()
        mock_result.final_output = SpecialistAssessment(
            category="placeholder", likelihood=0.5, reasoning="Test."
        )

        with patch(
            "agentic_fraud_servicing.copilot.hypothesis_specialists.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            results = await run_specialists(
                "allegations", "evidence", "conversation", mock_provider
            )

        assert set(results.keys()) == {"DISPUTE", "SCAM", "THIRD_PARTY_FRAUD"}
        for assessment in results.values():
            assert isinstance(assessment, SpecialistAssessment)

    async def test_handles_specialist_failure(self, mock_provider):
        """Failing specialist returns default assessment with likelihood 0.0."""
        call_count = 0

        async def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("LLM timeout")
            mock_result = MagicMock()
            mock_result.final_output = SpecialistAssessment(
                category="ok", likelihood=0.5, reasoning="Fine."
            )
            return mock_result

        with patch(
            "agentic_fraud_servicing.copilot.hypothesis_specialists.run_with_retry",
            new_callable=AsyncMock,
            side_effect=_side_effect,
        ):
            results = await run_specialists(
                "allegations", "evidence", "conversation", mock_provider
            )

        # One failed, two succeeded
        likelihoods = [a.likelihood for a in results.values()]
        assert 0.0 in likelihoods  # the failed one
        assert sum(1 for v in likelihoods if v == 0.5) == 2

    async def test_passes_previous_assessments(self, mock_provider):
        """Previous assessments are included in specialist user messages."""
        captured_inputs: list[str] = []

        async def _capture(*args, **kwargs):
            captured_inputs.append(kwargs.get("input", args[1] if len(args) > 1 else ""))
            mock_result = MagicMock()
            mock_result.final_output = SpecialistAssessment(
                category="test", likelihood=0.3, reasoning="Ok."
            )
            return mock_result

        prev = {
            "DISPUTE": SpecialistAssessment(
                category="DISPUTE", likelihood=0.6, reasoning="Looks like dispute."
            ),
        }

        with patch(
            "agentic_fraud_servicing.copilot.hypothesis_specialists.run_with_retry",
            new_callable=AsyncMock,
            side_effect=_capture,
        ):
            await run_specialists(
                "allegations", "evidence", "conversation",
                mock_provider, previous_assessments=prev,
            )

        # The dispute specialist should see its previous assessment
        dispute_input = captured_inputs[0]  # DISPUTE is first in _SPECIALISTS
        assert "Your Previous Assessment" in dispute_input
        assert "0.60" in dispute_input


# ---------------------------------------------------------------------------
# run_arbitrator tests
# ---------------------------------------------------------------------------


class TestRunArbitrator:
    """Tests for the run_arbitrator async function."""

    @pytest.fixture
    def mock_provider(self):
        return MagicMock()

    @pytest.fixture
    def sample_assessment(self):
        return HypothesisAssessment(
            scores={
                "THIRD_PARTY_FRAUD": 0.1,
                "FIRST_PARTY_FRAUD": 0.6,
                "SCAM": 0.2,
                "DISPUTE": 0.1,
            },
            reasoning={
                "THIRD_PARTY_FRAUD": "Low — enrolled device used",
                "FIRST_PARTY_FRAUD": "High — chip+PIN contradiction",
                "SCAM": "Moderate — some urgency",
                "DISPUTE": "Low — no merchant issue",
            },
            contradictions=["CM claims unauthorized but chip+PIN auth"],
            assessment_summary="First-party fraud indicators present.",
        )

    @pytest.fixture
    def mock_specialist_outputs(self):
        return {
            "DISPUTE": SpecialistAssessment(
                category="DISPUTE", likelihood=0.1, reasoning="No merchant issue."
            ),
            "SCAM": SpecialistAssessment(
                category="SCAM", likelihood=0.2, reasoning="Some urgency."
            ),
            "THIRD_PARTY_FRAUD": SpecialistAssessment(
                category="THIRD_PARTY_FRAUD", likelihood=0.1, reasoning="Enrolled device."
            ),
        }

    @pytest.fixture
    def default_scores(self):
        return {
            "THIRD_PARTY_FRAUD": 0.25,
            "FIRST_PARTY_FRAUD": 0.25,
            "SCAM": 0.25,
            "DISPUTE": 0.25,
        }

    async def test_returns_hypothesis_assessment(
        self, mock_provider, sample_assessment, mock_specialist_outputs, default_scores
    ):
        """run_arbitrator returns HypothesisAssessment."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = sample_assessment

        with patch(
            "agentic_fraud_servicing.copilot.hypothesis_agent.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ):
            result = await run_arbitrator(
                specialist_assessments=mock_specialist_outputs,
                allegations_summary="UNRECOGNIZED_TRANSACTION: $499 charge",
                auth_summary="Impersonation risk: 0.2",
                current_scores=default_scores,
                model_provider=mock_provider,
            )

        assert isinstance(result, HypothesisAssessment)
        assert result.scores["FIRST_PARTY_FRAUD"] == 0.6

    async def test_includes_specialist_outputs_in_arbitrator_message(
        self, mock_provider, mock_specialist_outputs, default_scores
    ):
        """Arbitrator user message contains formatted specialist assessments."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = HypothesisAssessment()

        with patch(
            "agentic_fraud_servicing.copilot.hypothesis_agent.run_with_retry",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_arbitrator(
                specialist_assessments=mock_specialist_outputs,
                allegations_summary="claims",
                auth_summary="auth",
                current_scores=default_scores,
                model_provider=mock_provider,
            )

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "Specialist Assessments" in user_input
        assert "Dispute Specialist" in user_input
        assert "Scam Specialist" in user_input
        assert "Fraud Specialist (Third-Party)" in user_input
        assert "0.10" in user_input  # dispute likelihood
        assert "0.20" in user_input  # scam likelihood

    def test_does_not_import_run_specialists(self):
        """run_arbitrator module does not import run_specialists — decoupled."""
        import agentic_fraud_servicing.copilot.hypothesis_agent as mod

        assert not hasattr(mod, "run_specialists")

    async def test_wraps_exceptions(self, mock_provider, default_scores):
        """run_arbitrator wraps SDK exceptions in RuntimeError."""
        mock_specialist_outputs = {
            "DISPUTE": SpecialistAssessment(category="DISPUTE"),
            "SCAM": SpecialistAssessment(category="SCAM"),
            "THIRD_PARTY_FRAUD": SpecialistAssessment(category="THIRD_PARTY_FRAUD"),
        }

        with patch(
            "agentic_fraud_servicing.copilot.hypothesis_agent.run_with_retry",
            new_callable=AsyncMock,
            side_effect=ValueError("LLM call failed"),
        ):
            with pytest.raises(RuntimeError, match="Hypothesis agent failed"):
                await run_arbitrator(
                    specialist_assessments=mock_specialist_outputs,
                    allegations_summary="claims",
                    auth_summary="auth",
                    current_scores=default_scores,
                    model_provider=mock_provider,
                )
