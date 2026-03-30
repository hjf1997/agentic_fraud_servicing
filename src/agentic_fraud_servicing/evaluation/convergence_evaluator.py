"""Convergence evaluator — pure Python analysis of hypothesis score convergence.

Determines the first turn where the correct investigation category becomes
and stays the highest-scoring hypothesis for all subsequent turns.
"""

from __future__ import annotations

from agentic_fraud_servicing.evaluation.models import ConvergenceResult, EvaluationRun

# The four hypothesis score keys in standard order (used for tie-breaking)
_CATEGORIES = ["DISPUTE", "FIRST_PARTY_FRAUD", "SCAM", "THIRD_PARTY_FRAUD"]


def _highest_category(scores: dict[str, float]) -> str | None:
    """Return the category with the highest score, breaking ties alphabetically.

    Args:
        scores: Mapping of category name to score value.

    Returns:
        Category name with the highest score, or None if scores is empty.
    """
    if not scores:
        return None
    max_score = max(scores.values())
    # Among tied categories, pick first alphabetically
    candidates = sorted(k for k, v in scores.items() if v == max_score)
    return candidates[0] if candidates else None


def evaluate_convergence(run: EvaluationRun) -> ConvergenceResult:
    """Analyze hypothesis convergence speed from an evaluation run.

    Finds the first turn where the ground-truth category becomes the highest
    hypothesis score and remains highest for ALL subsequent turns.

    Args:
        run: A completed EvaluationRun with turn_metrics containing hypothesis_scores.

    Returns:
        ConvergenceResult with convergence turn, ratio, and per-turn score history.
    """
    correct_category = run.ground_truth.get("outcome_test", "")

    # No ground truth or empty turns → cannot assess convergence
    if not correct_category or not run.turn_metrics:
        return ConvergenceResult(
            convergence_turn=None,
            total_turns=len(run.turn_metrics),
            convergence_ratio=None,
            correct_category=correct_category,
            turn_scores=[],
        )

    # Only include assessed turns (those with a copilot_suggestion) so that
    # CCP/SYSTEM turns with empty hypothesis_scores don't break convergence detection.
    assessed = [tm for tm in run.turn_metrics if tm.copilot_suggestion is not None]

    if not assessed:
        return ConvergenceResult(
            convergence_turn=None,
            total_turns=len(run.turn_metrics),
            convergence_ratio=None,
            correct_category=correct_category,
            turn_scores=[],
        )

    # Build per-turn score records from assessed turns only
    turn_scores: list[dict] = []
    for tm in assessed:
        record: dict = {"turn_number": tm.turn_number}
        for cat in _CATEGORIES:
            record[cat] = tm.hypothesis_scores.get(cat, 0.0)
        turn_scores.append(record)

    # Find convergence point: first turn where correct_category is highest
    # AND it stays highest for all subsequent turns.
    # Scan from the end to find the last turn where it is NOT highest,
    # then convergence_turn is the turn right after that.
    total = len(turn_scores)
    convergence_turn: int | None = None

    # Check if correct category is highest at each turn
    is_highest = []
    for record in turn_scores:
        scores = {cat: record[cat] for cat in _CATEGORIES}
        highest = _highest_category(scores)
        is_highest.append(highest == correct_category)

    # Find the first turn where is_highest is True and stays True for all remaining
    for i in range(total):
        if all(is_highest[j] for j in range(i, total)):
            convergence_turn = turn_scores[i]["turn_number"]
            break

    convergence_ratio: float | None = None
    if convergence_turn is not None:
        convergence_ratio = convergence_turn / total if total > 0 else None

    return ConvergenceResult(
        convergence_turn=convergence_turn,
        total_turns=total,
        convergence_ratio=convergence_ratio,
        correct_category=correct_category,
        turn_scores=turn_scores,
    )
