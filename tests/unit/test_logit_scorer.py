"""Tests for the logprob-based hypothesis scorer."""

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agentic_fraud_servicing.copilot.hypothesis_specialists import SpecialistAssessment
from agentic_fraud_servicing.copilot.logit_scorer import (
    _FLOOR_PROB,
    _UNIFORM_SCORES,
    UTD_MAX_MASS,
    build_final_scores,
    build_logit_prompt,
    compute_entropy,
    compute_logprob_scores,
    derive_unable_to_determine,
    extract_category_probs,
)

# ---------------------------------------------------------------------------
# extract_category_probs tests
# ---------------------------------------------------------------------------


class TestExtractCategoryProbs:
    """Tests for logprob token extraction and normalization."""

    def _make_logprob(self, token: str, logprob: float):
        """Create a mock logprob entry with .token and .logprob attributes."""
        return SimpleNamespace(token=token, logprob=logprob)

    def test_all_four_tokens_present(self):
        """Extracts probabilities for A, B, C, D and normalizes to sum to 1.0."""
        top_logprobs = [
            self._make_logprob("A", math.log(0.5)),
            self._make_logprob("B", math.log(0.3)),
            self._make_logprob("C", math.log(0.1)),
            self._make_logprob("D", math.log(0.1)),
        ]
        probs = extract_category_probs(top_logprobs)
        assert set(probs.keys()) == {"THIRD_PARTY_FRAUD", "FIRST_PARTY_FRAUD", "SCAM", "DISPUTE"}
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_token_normalization_whitespace(self):
        """Handles tokens with leading/trailing whitespace."""
        top_logprobs = [
            self._make_logprob(" A", math.log(0.6)),
            self._make_logprob("B ", math.log(0.2)),
            self._make_logprob(" C ", math.log(0.1)),
            self._make_logprob("D", math.log(0.1)),
        ]
        probs = extract_category_probs(top_logprobs)
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        assert probs["THIRD_PARTY_FRAUD"] > probs["FIRST_PARTY_FRAUD"]

    def test_token_normalization_lowercase(self):
        """Handles lowercase tokens."""
        top_logprobs = [
            self._make_logprob("a", math.log(0.7)),
            self._make_logprob("b", math.log(0.1)),
            self._make_logprob("c", math.log(0.1)),
            self._make_logprob("d", math.log(0.1)),
        ]
        probs = extract_category_probs(top_logprobs)
        assert probs["THIRD_PARTY_FRAUD"] == pytest.approx(0.7, abs=1e-4)

    def test_missing_tokens_get_floor_probability(self):
        """Tokens not in top_logprobs receive floor probability."""
        top_logprobs = [
            self._make_logprob("A", math.log(0.9)),
            self._make_logprob("B", math.log(0.1)),
            # C and D missing
        ]
        probs = extract_category_probs(top_logprobs)
        assert probs["SCAM"] == pytest.approx(
            _FLOOR_PROB / (0.9 + 0.1 + 2 * _FLOOR_PROB), abs=1e-8
        )
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_irrelevant_tokens_ignored(self):
        """Non-ABCD tokens in top_logprobs are ignored."""
        top_logprobs = [
            self._make_logprob("A", math.log(0.5)),
            self._make_logprob("B", math.log(0.3)),
            self._make_logprob("C", math.log(0.1)),
            self._make_logprob("D", math.log(0.05)),
            self._make_logprob("E", math.log(0.03)),
            self._make_logprob("(", math.log(0.02)),
        ]
        probs = extract_category_probs(top_logprobs)
        assert "E" not in probs
        assert len(probs) == 4


# ---------------------------------------------------------------------------
# Entropy tests
# ---------------------------------------------------------------------------


class TestComputeEntropy:
    """Tests for Shannon entropy computation."""

    def test_uniform_distribution_max_entropy(self):
        """Uniform 4-way distribution has normalized entropy of 1.0."""
        probs = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        assert compute_entropy(probs) == pytest.approx(1.0, abs=1e-6)

    def test_peaked_distribution_low_entropy(self):
        """Peaked distribution has low entropy."""
        probs = {"A": 0.97, "B": 0.01, "C": 0.01, "D": 0.01}
        assert compute_entropy(probs) < 0.2

    def test_certain_distribution_zero_entropy(self):
        """Single-category distribution has entropy near 0.0."""
        probs = {"A": 1.0 - 3e-10, "B": 1e-10, "C": 1e-10, "D": 1e-10}
        assert compute_entropy(probs) < 0.01


# ---------------------------------------------------------------------------
# derive_unable_to_determine tests
# ---------------------------------------------------------------------------


class TestDeriveUnableToDetermine:
    """Tests for UTD derivation from entropy."""

    def test_peaked_distribution_low_utd(self):
        """Peaked distribution gives low UTD due to supralinear exponent."""
        probs = {"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}
        utd = derive_unable_to_determine(probs)
        assert utd < 0.2

    def test_uniform_distribution_capped_utd(self):
        """Uniform distribution gives UTD at the maximum cap."""
        probs = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        utd = derive_unable_to_determine(probs)
        # normalized_entropy=1.0, 1.0^1.5=1.0, min(1.0, 0.5)=0.5
        assert utd == pytest.approx(UTD_MAX_MASS, abs=1e-6)

    def test_utd_never_exceeds_max(self):
        """UTD never exceeds UTD_MAX_MASS regardless of entropy."""
        probs = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        utd = derive_unable_to_determine(probs)
        assert utd <= UTD_MAX_MASS


# ---------------------------------------------------------------------------
# build_final_scores tests
# ---------------------------------------------------------------------------


class TestBuildFinalScores:
    """Tests for combining 4-category probs with UTD into 5-key distribution."""

    def test_sums_to_one(self):
        """Final 5-key scores sum to 1.0."""
        category_probs = {
            "THIRD_PARTY_FRAUD": 0.5,
            "FIRST_PARTY_FRAUD": 0.3,
            "SCAM": 0.1,
            "DISPUTE": 0.1,
        }
        utd = 0.15
        scores = build_final_scores(category_probs, utd)
        assert abs(sum(scores.values()) - 1.0) < 1e-6

    def test_has_five_keys(self):
        """Final scores dict has exactly 5 investigation category keys."""
        category_probs = {
            "THIRD_PARTY_FRAUD": 0.25,
            "FIRST_PARTY_FRAUD": 0.25,
            "SCAM": 0.25,
            "DISPUTE": 0.25,
        }
        scores = build_final_scores(category_probs, 0.1)
        assert set(scores.keys()) == {
            "THIRD_PARTY_FRAUD",
            "FIRST_PARTY_FRAUD",
            "SCAM",
            "DISPUTE",
            "UNABLE_TO_DETERMINE",
        }

    def test_utd_value_matches(self):
        """UNABLE_TO_DETERMINE equals the provided UTD value."""
        category_probs = {
            "THIRD_PARTY_FRAUD": 0.5,
            "FIRST_PARTY_FRAUD": 0.3,
            "SCAM": 0.1,
            "DISPUTE": 0.1,
        }
        utd = 0.2
        scores = build_final_scores(category_probs, utd)
        assert scores["UNABLE_TO_DETERMINE"] == pytest.approx(utd)

    def test_zero_utd_preserves_original_probs(self):
        """With UTD=0, category scores equal original probabilities."""
        category_probs = {
            "THIRD_PARTY_FRAUD": 0.5,
            "FIRST_PARTY_FRAUD": 0.3,
            "SCAM": 0.1,
            "DISPUTE": 0.1,
        }
        scores = build_final_scores(category_probs, 0.0)
        for cat, prob in category_probs.items():
            assert scores[cat] == pytest.approx(prob)


# ---------------------------------------------------------------------------
# build_logit_prompt tests
# ---------------------------------------------------------------------------


class TestBuildLogitPrompt:
    """Tests for prompt construction."""

    def test_returns_system_and_user_messages(self):
        """build_logit_prompt returns two dicts with role and content keys."""
        assessments = {
            "DISPUTE": SpecialistAssessment(category="DISPUTE", reasoning="test"),
        }
        system_msg, user_msg = build_logit_prompt(assessments, "allegations", "auth")
        assert system_msg["role"] == "system"
        assert user_msg["role"] == "user"
        assert "A, B, C, or D" in user_msg["content"]

    def test_includes_specialist_evidence(self):
        """User message contains specialist evidence analysis."""
        assessments = {
            "DISPUTE": SpecialistAssessment(
                category="DISPUTE",
                reasoning="No merchant issue.",
                supporting_evidence=["Receipt confirms delivery"],
            ),
            "SCAM": SpecialistAssessment(
                category="SCAM",
                reasoning="Some urgency detected.",
            ),
        }
        _, user_msg = build_logit_prompt(assessments, "fraud allegations", "low risk")
        content = user_msg["content"]
        assert "Receipt confirms delivery" in content
        assert "Some urgency detected" in content
        assert "fraud allegations" in content


# ---------------------------------------------------------------------------
# compute_logprob_scores integration tests
# ---------------------------------------------------------------------------


class TestComputeLogprobScores:
    """Tests for the main scoring function with mocked OpenAI client."""

    @pytest.fixture
    def mock_client(self):
        return AsyncMock()

    @pytest.fixture
    def mock_specialist_assessments(self):
        return {
            "DISPUTE": SpecialistAssessment(category="DISPUTE", reasoning="test"),
            "SCAM": SpecialistAssessment(category="SCAM", reasoning="test"),
            "THIRD_PARTY_FRAUD": SpecialistAssessment(
                category="THIRD_PARTY_FRAUD", reasoning="test"
            ),
        }

    def _make_response(self, token_probs: dict[str, float]):
        """Create a mock OpenAI response with logprobs."""
        top_logprobs = [
            SimpleNamespace(token=tok, logprob=math.log(prob)) for tok, prob in token_probs.items()
        ]
        content_item = SimpleNamespace(top_logprobs=top_logprobs)
        logprobs = SimpleNamespace(content=[content_item])
        choice = SimpleNamespace(
            logprobs=logprobs,
            message=SimpleNamespace(content="A"),
        )
        return SimpleNamespace(choices=[choice])

    async def test_returns_five_key_dict(self, mock_client, mock_specialist_assessments):
        """compute_logprob_scores returns a 5-key dict summing to 1.0."""
        mock_client.chat.completions.create = AsyncMock(
            return_value=self._make_response({"A": 0.6, "B": 0.2, "C": 0.1, "D": 0.1})
        )

        with patch("agentic_fraud_servicing.copilot.logit_scorer._trace_logit_call"):
            scores = await compute_logprob_scores(
                client=mock_client,
                model="gpt-4.1",
                specialist_assessments=mock_specialist_assessments,
                allegations_summary="allegations",
                auth_summary="auth",
            )

        assert set(scores.keys()) == {
            "THIRD_PARTY_FRAUD",
            "FIRST_PARTY_FRAUD",
            "SCAM",
            "DISPUTE",
            "UNABLE_TO_DETERMINE",
        }
        assert abs(sum(scores.values()) - 1.0) < 1e-6

    async def test_peaked_distribution_high_top_score(
        self, mock_client, mock_specialist_assessments
    ):
        """Peaked logprobs produce a high top-category score."""
        mock_client.chat.completions.create = AsyncMock(
            return_value=self._make_response({"A": 0.95, "B": 0.03, "C": 0.01, "D": 0.01})
        )

        with patch("agentic_fraud_servicing.copilot.logit_scorer._trace_logit_call"):
            scores = await compute_logprob_scores(
                client=mock_client,
                model="gpt-4.1",
                specialist_assessments=mock_specialist_assessments,
                allegations_summary="allegations",
                auth_summary="auth",
            )

        assert scores["THIRD_PARTY_FRAUD"] > 0.7
        assert scores["UNABLE_TO_DETERMINE"] < 0.1

    async def test_api_failure_returns_uniform(self, mock_client, mock_specialist_assessments):
        """API failure returns uniform distribution as fallback."""
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("API timeout"))

        with patch("agentic_fraud_servicing.copilot.logit_scorer._trace_logit_error"):
            scores = await compute_logprob_scores(
                client=mock_client,
                model="gpt-4.1",
                specialist_assessments=mock_specialist_assessments,
                allegations_summary="allegations",
                auth_summary="auth",
            )

        assert scores == {
            cat: 0.2
            for cat in [
                "THIRD_PARTY_FRAUD",
                "FIRST_PARTY_FRAUD",
                "SCAM",
                "DISPUTE",
                "UNABLE_TO_DETERMINE",
            ]
        }

    async def test_no_logprobs_returns_uniform(self, mock_client, mock_specialist_assessments):
        """Response without logprobs returns uniform distribution."""
        choice = SimpleNamespace(
            logprobs=None,
            message=SimpleNamespace(content="A"),
        )
        response = SimpleNamespace(choices=[choice])
        mock_client.chat.completions.create = AsyncMock(return_value=response)

        scores = await compute_logprob_scores(
            client=mock_client,
            model="gpt-4.1",
            specialist_assessments=mock_specialist_assessments,
            allegations_summary="allegations",
            auth_summary="auth",
        )

        assert scores == _UNIFORM_SCORES
