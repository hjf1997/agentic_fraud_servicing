"""Risk flag timeliness evaluator — hybrid Python + LLM assessment.

Uses an LLM agent to semantically match expected risk flags against actually
raised flags, then uses Python to compute timing metrics: when each flag was
raised relative to when supporting evidence first appeared.
"""

from __future__ import annotations

from agents import Agent, AgentOutputSchema, ModelProvider
from agents.run_config import RunConfig

from agentic_fraud_servicing.providers.retry import run_with_retry
from pydantic import BaseModel, Field

from agentic_fraud_servicing.evaluation.models import (
    EvaluationRun,
    RiskFlagTimelinessResult,
)

# --- Output model for LLM flag matching ---


class FlagMatchResult(BaseModel):
    """Structured output from the flag matching agent.

    Attributes:
        matches: Each dict has expected_flag, raised_flag, raised_turn.
        unmatched: Expected flags that were never raised.
    """

    matches: list[dict] = Field(default_factory=list)
    unmatched: list[str] = Field(default_factory=list)


# --- Agent instance ---

_FLAG_MATCHING_INSTRUCTIONS = """\
You are an evaluation specialist matching expected risk flags against actually
raised risk flags from a copilot system.

You will receive:
1. A list of **expected risk flags** that should have been raised during the call.
2. A **timeline of raised flags** with the turn number when each flag first appeared.

## Matching Rules

- **Semantic matching**: An expected flag matches a raised flag if they describe
  the same risk concern, even if worded differently. For example, "chip+PIN
  contradiction" matches "Authentication method inconsistent with card possession
  claim".
- **Partial overlap**: If a raised flag covers part of an expected flag's concern,
  count it as a match.
- Each expected flag can match at most one raised flag.
- Each raised flag can match at most one expected flag.
- For each match, report the turn number when the raised flag first appeared.

## Output

- `matches`: List of dicts, each with:
  - `expected_flag`: The expected flag text.
  - `raised_flag`: The matching raised flag text.
  - `raised_turn`: The turn number when the raised flag first appeared.
- `unmatched`: List of expected flags that were NOT matched to any raised flag.
"""

_flag_matching_agent = Agent(
    name="risk_flag_matcher",
    instructions=_FLAG_MATCHING_INSTRUCTIONS,
    output_type=AgentOutputSchema(FlagMatchResult, strict_json_schema=False),
)


# --- Public function ---


async def evaluate_risk_flag_timeliness(
    run: EvaluationRun,
    model_provider: ModelProvider,
) -> RiskFlagTimelinessResult:
    """Evaluate when risk flags were raised relative to evidence availability.

    For each expected risk flag, determines: (1) whether it was raised via LLM
    matching, (2) at which turn it was raised, (3) at which turn supporting
    evidence first appeared, and (4) the delay in turns.

    Args:
        run: A completed EvaluationRun with turn_metrics and ground_truth.
        model_provider: LLM model provider for inference.

    Returns:
        RiskFlagTimelinessResult with per-flag timing and aggregate metrics.
    """
    expected_flags = run.ground_truth.get("expected_risk_flags", [])

    # Early return if no expected flags
    if not expected_flags:
        return RiskFlagTimelinessResult(
            per_flag_timing=[],
            average_delay_turns=0.0,
            flags_raised_count=0,
            flags_expected_count=0,
        )

    # Collect raised flags with the turn they first appeared
    raised_timeline = _collect_raised_flags(run)

    # No flags raised at all — all unmatched
    if not raised_timeline:
        return RiskFlagTimelinessResult(
            per_flag_timing=[],
            average_delay_turns=0.0,
            flags_raised_count=0,
            flags_expected_count=len(expected_flags),
        )

    # Use LLM to match expected flags against raised flags
    match_result = await _match_flags(expected_flags, raised_timeline, model_provider)

    # Compute per-flag timing
    per_flag_timing: list[dict] = []
    for m in match_result.matches:
        raised_turn = m.get("raised_turn", 0)
        flag_text = m.get("expected_flag", "")
        raised_flag_text = m.get("raised_flag", "")

        # Find earliest turn where evidence related to this flag appeared
        evidence_turn = _find_evidence_available_turn(run, flag_text, raised_flag_text)
        delay = raised_turn - evidence_turn if evidence_turn is not None else 0

        per_flag_timing.append(
            {
                "flag_text": flag_text,
                "raised_turn": raised_turn,
                "evidence_available_turn": (
                    evidence_turn if evidence_turn is not None else raised_turn
                ),
                "delay_turns": delay,
            }
        )

    # Compute average delay
    delays = [t["delay_turns"] for t in per_flag_timing]
    average_delay = sum(delays) / len(delays) if delays else 0.0

    return RiskFlagTimelinessResult(
        per_flag_timing=per_flag_timing,
        average_delay_turns=average_delay,
        flags_raised_count=len(per_flag_timing),
        flags_expected_count=len(expected_flags),
    )


def _collect_raised_flags(run: EvaluationRun) -> dict[str, int]:
    """Collect all risk flags and the turn number when each first appeared.

    Returns:
        Dict mapping flag text to the earliest turn_number it appeared.
    """
    flag_first_turn: dict[str, int] = {}
    for turn in run.turn_metrics:
        suggestion = turn.copilot_suggestion
        if suggestion is None:
            continue
        flags = suggestion.get("risk_flags", [])
        for flag in flags:
            if isinstance(flag, str) and flag not in flag_first_turn:
                flag_first_turn[flag] = turn.turn_number
    return flag_first_turn


def _find_evidence_available_turn(
    run: EvaluationRun,
    expected_flag: str,
    raised_flag: str,
) -> int | None:
    """Find the first turn where evidence related to a flag appeared.

    Scans retrieved_facts and running_summary in copilot_suggestions for
    key terms from the flag text. Returns the turn_number or None if no
    evidence-related content is found before the flag was raised.
    """
    # Extract key terms from both flag texts for matching
    terms = _extract_key_terms(expected_flag) | _extract_key_terms(raised_flag)
    if not terms:
        return None

    for turn in run.turn_metrics:
        suggestion = turn.copilot_suggestion
        if suggestion is None:
            continue

        # Check retrieved_facts
        facts = suggestion.get("retrieved_facts", [])
        facts_text = " ".join(str(f) for f in facts).lower()

        # Check running_summary
        summary = suggestion.get("running_summary", "")
        combined = f"{facts_text} {summary}".lower()

        if any(term in combined for term in terms):
            return turn.turn_number

    return None


def _extract_key_terms(text: str) -> set[str]:
    """Extract meaningful terms from a flag text for evidence matching.

    Filters out common stop words and returns lowercase terms of 4+ chars.
    """
    stop_words = {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "been",
        "have",
        "has",
        "had",
        "was",
        "were",
        "are",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "not",
        "but",
        "also",
        "very",
        "just",
        "about",
        "into",
    }
    words = text.lower().split()
    return {
        w.strip(".,;:!?()[]{}\"'") for w in words if len(w) >= 4 and w.lower() not in stop_words
    }


async def _match_flags(
    expected_flags: list[str],
    raised_timeline: dict[str, int],
    model_provider: ModelProvider,
) -> FlagMatchResult:
    """Use LLM agent to match expected flags against raised flags.

    Returns:
        FlagMatchResult with matches and unmatched lists. Falls back to
        all-unmatched on LLM failure.
    """
    expected_text = "\n".join(f"- {f}" for f in expected_flags)
    raised_text = "\n".join(
        f"- Turn {turn}: {flag}"
        for flag, turn in sorted(raised_timeline.items(), key=lambda x: x[1])
    )
    user_msg = (
        f"## Expected Risk Flags\n{expected_text}\n\n## Timeline of Raised Flags\n{raised_text}"
    )

    try:
        result = await run_with_retry(
            _flag_matching_agent,
            input=user_msg,
            run_config=RunConfig(model_provider=model_provider),
        )
        return result.final_output
    except Exception as exc:
        # Graceful degradation: all expected flags unmatched
        from agentic_fraud_servicing.copilot.langfuse_tracing import extract_http_error

        status_code, error_body = extract_http_error(exc)
        detail = f"HTTP {status_code}: {error_body[:200]}" if status_code else str(exc)
        print(f"[risk_flag_evaluator] LLM matching failed ({detail})", file=__import__("sys").stderr)
        return FlagMatchResult(matches=[], unmatched=list(expected_flags))
