"""Logprob-based hypothesis scorer using forced-choice classification.

Makes a single OpenAI API call with max_tokens=1 and logprobs=True to get
a probability distribution over 4 investigation categories. Derives
UNABLE_TO_DETERMINE from the Shannon entropy of the 4-category distribution
rather than asking the LLM to generate it as a class.

This produces more consistent and grounded scores than asking the LLM to
generate float values, because logprobs reflect the model's actual internal
confidence about its classification rather than a fabricated number.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from agentic_fraud_servicing.copilot.hypothesis_specialists import (
        SpecialistAssessment,
        SpecialistNoteUpdate,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category mapping
# ---------------------------------------------------------------------------

_TOKEN_TO_CATEGORY: dict[str, str] = {
    "A": "THIRD_PARTY_FRAUD",
    "B": "FIRST_PARTY_FRAUD",
    "C": "SCAM",
    "D": "DISPUTE",
}

# Token IDs for target letters, used in logit_bias to ensure all 4 appear
# in top_logprobs. Includes both bare ("A") and space-prefixed (" A")
# variants since chat completions may tokenize either way. Computed via
# tiktoken at import time; falls back to known cl100k_base IDs.
_FALLBACK_TOKEN_IDS: list[int] = [
    32, 33, 34, 35,  # A, B, C, D (bare)
    355, 418, 363, 415,  # " A", " B", " C", " D" (space-prefixed)
]


def _resolve_token_ids() -> list[int]:
    """Resolve token IDs for target letters using tiktoken if available.

    Returns IDs for both bare and space-prefixed variants of A/B/C/D.
    """
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model("gpt-4o")
        ids: list[int] = []
        for letter in _TOKEN_TO_CATEGORY:
            for variant in [letter, f" {letter}"]:
                encoded = enc.encode(variant)
                # " A" encodes to [token_for_space_A] as a single token
                # in cl100k_base / o200k_base.
                ids.append(encoded[-1])
        return ids
    except Exception:
        return list(_FALLBACK_TOKEN_IDS)


_TARGET_TOKEN_IDS: list[int] = _resolve_token_ids()

# Logit bias added to target tokens. Large enough to guarantee they appear
# in top_logprobs, but gets subtracted from observed logprobs to recover
# the original (unbiased) distribution.
_LOGIT_BIAS_BOOST: float = 10.0

_ALL_CATEGORIES = [
    "THIRD_PARTY_FRAUD",
    "FIRST_PARTY_FRAUD",
    "SCAM",
    "DISPUTE",
    "UNABLE_TO_DETERMINE",
]

_UNIFORM_SCORES: dict[str, float] = {cat: 0.20 for cat in _ALL_CATEGORIES}

# ---------------------------------------------------------------------------
# Entropy tuning knobs
# ---------------------------------------------------------------------------

# Exponent applied to normalized entropy before capping. >1 means low entropy
# yields very low UTD, high entropy yields substantial UTD (supralinear).
UTD_ENTROPY_EXPONENT: float = 1.5

# Maximum probability mass UNABLE_TO_DETERMINE can absorb. Prevents UTD from
# dominating even on perfectly ambiguous cases.
UTD_MAX_MASS: float = 0.5

# Floor probability for tokens missing from top_logprobs (very unlikely).
_FLOOR_PROB: float = 1e-6

# Minimum probability per category after normalization (Laplace-style
# smoothing). Prevents any category from being zeroed out when target
# tokens are missing from top_logprobs, which happens when Azure OpenAI
# silently ignores logit_bias or the model is extremely confident.
MIN_CATEGORY_PROB: float = 0.01

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a fraud investigation classifier for AMEX card disputes. You will
receive evidence analysis from three category specialists (Dispute, Scam,
Third-Party Fraud) plus accumulated allegations and authentication data.

## Classification Guidelines

- Base your judgment on system evidence and specialist findings, not on
  cardmember allegations alone. Allegations establish which hypotheses to
  investigate but are not evidence of what actually happened.
- Previously known information restated in a new turn is not new evidence.
  Only genuinely new findings should influence your classification.
- Contradictions between allegations and evidence are one signal among many.
  Weigh all specialist evidence proportionally — no single signal type
  should dominate your decision.
- Consider whether one specialist's contradicting evidence undermines
  another specialist's supporting evidence.
- If contradicting evidence exists across specialists without any evidence
  of an external manipulator, consider first-party fraud.

## Task

Based on ALL the evidence below, which single investigation category BEST
explains this case? Answer with exactly one letter.

A) Third-party fraud — unauthorized transaction by an external criminal
B) First-party fraud — cardmember is misrepresenting the transaction
C) Scam — cardmember was deceived into authorizing by an external fraudster
D) Dispute — authorized transaction with a merchant performance issue
"""


def _format_specialist_evidence(
    assessments: dict[str, SpecialistAssessment],
) -> str:
    """Format specialist outputs for the logit prompt (no likelihood scores)."""
    _LABELS = {
        "DISPUTE": "Dispute Specialist",
        "SCAM": "Scam Specialist",
        "THIRD_PARTY_FRAUD": "Fraud Specialist (Third-Party)",
    }
    parts: list[str] = []
    for category in ("DISPUTE", "SCAM", "THIRD_PARTY_FRAUD"):
        a = assessments.get(category)
        if a is None:
            parts.append(f"### {_LABELS[category]}\nNot available.")
            continue
        supporting = ", ".join(a.supporting_evidence) if a.supporting_evidence else "none"
        contradicting = ", ".join(a.contradicting_evidence) if a.contradicting_evidence else "none"
        gaps = ", ".join(a.evidence_gaps) if a.evidence_gaps else "none"
        parts.append(
            f"### {_LABELS[category]}\n"
            f"Eligibility: {a.eligibility}\n"
            f"Reasoning: {a.reasoning}\n"
            f"Supporting evidence: {supporting}\n"
            f"Contradicting evidence: {contradicting}\n"
            f"Evidence gaps: {gaps}"
        )
    return "\n\n".join(parts)


def _format_delta_changes(
    specialist_deltas: dict[str, SpecialistNoteUpdate],
) -> str:
    """Format specialist deltas into a brief changes summary for the logit prompt."""
    _LABELS = {
        "DISPUTE": "Dispute",
        "SCAM": "Scam",
        "THIRD_PARTY_FRAUD": "Fraud",
    }
    lines: list[str] = []
    for category in ("DISPUTE", "SCAM", "THIRD_PARTY_FRAUD"):
        delta = specialist_deltas.get(category)
        if delta is None:
            continue
        changes: list[str] = []
        for item in delta.add_supporting_evidence:
            changes.append(f"+supporting: {item}")
        for item in delta.remove_supporting_evidence:
            changes.append(f"-supporting: {item}")
        for item in delta.add_contradicting_evidence:
            changes.append(f"+contradicting: {item}")
        for item in delta.remove_contradicting_evidence:
            changes.append(f"-contradicting: {item}")
        for item in delta.add_evidence_gaps:
            changes.append(f"+gap: {item}")
        for item in delta.remove_evidence_gaps:
            changes.append(f"-gap: {item}")
        if changes:
            lines.append(f"{_LABELS[category]}: " + "; ".join(changes))
    return "\n".join(lines)


def build_logit_prompt(
    specialist_assessments: dict[str, SpecialistAssessment],
    allegations_summary: str,
    auth_summary: str,
    specialist_deltas: dict[str, SpecialistNoteUpdate] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Build the system and user messages for the logit classification call.

    Returns:
        (system_message, user_message) as dicts with 'role' and 'content' keys.
    """
    changes_section = ""
    if specialist_deltas:
        changes_text = _format_delta_changes(specialist_deltas)
        if changes_text:
            changes_section = f"\n\n## Changes This Turn\n{changes_text}"

    user_content = (
        f"## Specialist Evidence Analysis\n\n"
        f"{_format_specialist_evidence(specialist_assessments)}\n\n"
        f"## Authentication Assessment\n{auth_summary}\n\n"
        f"## Accumulated Allegations\n{allegations_summary}"
        f"{changes_section}\n\n"
        f"Answer with a single letter: A, B, C, or D."
    )
    return (
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    )


# ---------------------------------------------------------------------------
# Logprob extraction and scoring
# ---------------------------------------------------------------------------


def extract_category_probs(
    top_logprobs: list[Any],
    logit_bias_applied: float = 0.0,
) -> dict[str, float]:
    """Extract 4-category probabilities from the top_logprobs list.

    When logit_bias was applied to boost target tokens, subtracts the bias
    from observed logprobs to recover the model's original (unbiased)
    probability distribution.

    Args:
        top_logprobs: List of logprob objects from the OpenAI response,
            each with .token and .logprob attributes.
        logit_bias_applied: The logit bias that was added to target tokens
            in the API call. Subtracted from observed logprobs to recover
            original probabilities.

    Returns:
        Dict with 4 category keys mapped to normalized probabilities
        summing to 1.0.
    """
    # Build lookup: normalized token -> observed logprob
    token_logprobs: dict[str, float] = {}
    for entry in top_logprobs:
        normalized = entry.token.strip().upper()
        token_logprobs[normalized] = entry.logprob

    # Log diagnostics for missing target tokens
    target_tokens = set(_TOKEN_TO_CATEGORY.keys())
    found = target_tokens & set(token_logprobs.keys())
    missing = target_tokens - found
    if missing:
        raw_tokens = [(e.token, f"{e.logprob:.3f}") for e in top_logprobs]
        logger.warning(
            "Logit scorer: target tokens %s missing from top_logprobs despite "
            "logit_bias=%.0f. Raw tokens: %s",
            missing,
            logit_bias_applied,
            raw_tokens,
        )

    # Only subtract bias if it actually took effect. If all 4 targets
    # appear in top_logprobs, the bias worked. If some are missing, Azure
    # silently ignored logit_bias and subtracting it would artificially
    # deflate the found tokens' probabilities.
    bias_effective = not missing
    effective_bias = logit_bias_applied if bias_effective else 0.0

    # Extract probabilities, subtracting logit_bias from target tokens
    # to recover the model's original (unbiased) distribution.
    raw: dict[str, float] = {}
    for token, category in _TOKEN_TO_CATEGORY.items():
        if token in token_logprobs:
            original_logprob = token_logprobs[token] - effective_bias
            raw[category] = math.exp(original_logprob)
        else:
            raw[category] = _FLOOR_PROB

    # Normalize to sum to 1.0
    total = sum(raw.values())
    if total <= 0:
        return {cat: 0.25 for cat in _TOKEN_TO_CATEGORY.values()}
    probs = {cat: prob / total for cat, prob in raw.items()}

    # Apply Laplace-style smoothing: clamp each category to at least
    # MIN_CATEGORY_PROB, then renormalize. This prevents extreme 0/1
    # distributions when Azure OpenAI silently ignores logit_bias and
    # target tokens are missing from top_logprobs.
    needs_smoothing = any(p < MIN_CATEGORY_PROB for p in probs.values())
    if needs_smoothing:
        smoothed = {cat: max(p, MIN_CATEGORY_PROB) for cat, p in probs.items()}
        smooth_total = sum(smoothed.values())
        probs = {cat: p / smooth_total for cat, p in smoothed.items()}

    return probs


def compute_entropy(probs: dict[str, float]) -> float:
    """Compute normalized Shannon entropy for a 4-category distribution.

    Returns:
        Value between 0.0 (certain — one category dominates) and 1.0
        (maximally uncertain — uniform distribution).
    """
    h_max = math.log2(len(probs))  # log2(4) = 2.0
    if h_max == 0:
        return 0.0
    h = -sum(p * math.log2(p) for p in probs.values() if p > 0)
    return h / h_max


def derive_unable_to_determine(probs: dict[str, float]) -> float:
    """Derive UNABLE_TO_DETERMINE score from entropy of the 4-category distribution.

    Higher entropy (flatter distribution) → higher UTD. The relationship is
    supralinear (exponent > 1) so peaked distributions get very low UTD.

    Returns:
        UTD score between 0.0 and UTD_MAX_MASS.
    """
    normalized_entropy = compute_entropy(probs)
    return min(normalized_entropy**UTD_ENTROPY_EXPONENT, UTD_MAX_MASS)


def build_final_scores(category_probs: dict[str, float], utd: float) -> dict[str, float]:
    """Combine 4-category probs with UTD into a 5-key distribution summing to 1.0.

    Each real category is scaled down by (1 - utd) to make room for UTD.
    """
    scale = 1.0 - utd
    scores: dict[str, float] = {cat: prob * scale for cat, prob in category_probs.items()}
    scores["UNABLE_TO_DETERMINE"] = utd
    return scores


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------


async def compute_logprob_scores(
    client: AsyncOpenAI,
    model: str,
    specialist_assessments: dict[str, SpecialistAssessment],
    allegations_summary: str,
    auth_summary: str,
    specialist_deltas: dict[str, SpecialistNoteUpdate] | None = None,
) -> dict[str, float]:
    """Compute hypothesis scores via logprob-based forced-choice classification.

    Makes a single OpenAI API call asking the model to pick one of 4
    investigation categories. The logprob distribution over the answer
    tokens becomes the score. UNABLE_TO_DETERMINE is derived from entropy.

    Args:
        client: AsyncOpenAI client for direct API calls.
        model: OpenAI model identifier (e.g. 'gpt-4.1').
        specialist_assessments: Evidence analysis from 3 category specialists.
        allegations_summary: Formatted accumulated allegations.
        auth_summary: Formatted auth assessment.
        specialist_deltas: Raw specialist deltas from this turn. Included
            as a "Changes This Turn" section so the classifier can weigh
            new evidence more heavily.

    Returns:
        Dict with 5 keys (THIRD_PARTY_FRAUD, FIRST_PARTY_FRAUD, SCAM,
        DISPUTE, UNABLE_TO_DETERMINE) summing to 1.0.
    """
    system_msg, user_msg = build_logit_prompt(
        specialist_assessments, allegations_summary, auth_summary, specialist_deltas
    )

    start_time = time.monotonic()
    try:
        # logit_bias ensures all 4 target tokens (A/B/C/D) appear in
        # top_logprobs even when the model is very confident about one.
        # Without it, Azure OpenAI returns ~5 tokens and non-targets
        # ("Answer", "The", "**") displace the real alternatives.
        # Includes both bare and space-prefixed variants.
        logit_bias = {str(tid): int(_LOGIT_BIAS_BOOST) for tid in _TARGET_TOKEN_IDS}
        response = await client.chat.completions.create(
            model=model,
            messages=[system_msg, user_msg],
            max_tokens=1,
            logprobs=True,
            top_logprobs=10,
            temperature=0.0,
            logit_bias=logit_bias,
        )

        duration_ms = (time.monotonic() - start_time) * 1000

        # Extract logprobs from the first (only) content token
        choice = response.choices[0]
        if choice.logprobs and choice.logprobs.content:
            top_lps = choice.logprobs.content[0].top_logprobs
        else:
            logger.warning("Logit scorer: no logprobs in response, using uniform")
            return dict(_UNIFORM_SCORES)

        category_probs = extract_category_probs(top_lps, logit_bias_applied=_LOGIT_BIAS_BOOST)
        utd = derive_unable_to_determine(category_probs)
        scores = build_final_scores(category_probs, utd)

        # Manual LangFuse span for observability
        _trace_logit_call(
            model=model,
            system_msg=system_msg["content"],
            user_msg=user_msg["content"],
            completion_token=choice.message.content or "",
            scores=scores,
            category_probs=category_probs,
            entropy=compute_entropy(category_probs),
            duration_ms=duration_ms,
        )

        logger.debug(
            "Logit scorer: probs=%s, entropy=%.3f, utd=%.3f, scores=%s",
            {k: f"{v:.3f}" for k, v in category_probs.items()},
            compute_entropy(category_probs),
            utd,
            {k: f"{v:.3f}" for k, v in scores.items()},
        )

        return scores

    except Exception as exc:
        duration_ms = (time.monotonic() - start_time) * 1000
        logger.error("Logit scorer failed (%.0fms): %s", duration_ms, exc)
        _trace_logit_error(str(exc), duration_ms)
        return dict(_UNIFORM_SCORES)


# ---------------------------------------------------------------------------
# LangFuse observability
# ---------------------------------------------------------------------------


def _get_current_observation():
    """Return the current LangFuse observation (span/trace) if available.

    When the orchestrator wraps the pipeline in start_as_current_observation +
    propagate_attributes, this returns the active span so the logit scorer's
    generation appears nested under the copilot_turn trace — not orphaned at
    the top level.
    """
    try:
        from langfuse import get_client

        client = get_client()
        if client is None:
            return None
        # get_current_observation returns the span set by
        # start_as_current_observation in the orchestrator
        return client.get_current_observation()
    except Exception:
        return None


def _trace_logit_call(
    model: str,
    system_msg: str,
    user_msg: str,
    completion_token: str,
    scores: dict[str, float],
    category_probs: dict[str, float],
    entropy: float,
    duration_ms: float,
) -> None:
    """Record the logit scoring call as a LangFuse generation span.

    Creates the generation as a child of the current observation (typically
    the phase2 span) so it appears nested within the copilot_turn trace.
    Falls back to a top-level generation if no current observation exists.
    """
    try:
        from agentic_fraud_servicing.copilot.langfuse_tracing import get_langfuse

        lf = get_langfuse()
        if lf is None:
            return

        # Nest under current observation if available, otherwise top-level
        parent = _get_current_observation() or lf

        parent.generation(
            name="logit_scorer",
            model=model,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            output=completion_token,
            metadata={
                "category_probs": {k: round(v, 4) for k, v in category_probs.items()},
                "entropy": round(entropy, 4),
                "final_scores": {k: round(v, 4) for k, v in scores.items()},
                "max_tokens": 1,
                "logprobs": True,
                "top_logprobs": 10,
                "temperature": 0.0,
                "logit_bias_boost": _LOGIT_BIAS_BOOST,
            },
            level="DEFAULT",
            status_message=f"top={max(scores, key=scores.get)}, entropy={entropy:.3f}",
        )
    except Exception:
        pass


def _trace_logit_error(error_msg: str, duration_ms: float) -> None:
    """Record a failed logit scoring call in LangFuse."""
    try:
        from agentic_fraud_servicing.copilot.langfuse_tracing import get_langfuse

        lf = get_langfuse()
        if lf is None:
            return

        parent = _get_current_observation() or lf

        parent.generation(
            name="logit_scorer",
            level="ERROR",
            status_message=f"Failed ({duration_ms:.0f}ms): {error_msg[:500]}",
        )
    except Exception:
        pass
