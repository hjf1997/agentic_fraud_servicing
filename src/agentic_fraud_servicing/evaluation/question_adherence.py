"""Probing question lifecycle evaluator — pure-Python extraction from copilot state.

Extracts the probing question list from the last copilot suggestion in the
evaluation run and summarises lifecycle statistics (answered, invalidated,
skipped, pending). No LLM call required — the copilot already tracks question
status during the live run.
"""

from __future__ import annotations

from agentic_fraud_servicing.evaluation.models import (
    EvaluationRun,
    QuestionAdherenceResult,
)


def evaluate_question_adherence(run: EvaluationRun) -> QuestionAdherenceResult:
    """Extract probing question lifecycle stats from the last copilot suggestion.

    Scans turn_metrics in reverse for the last non-None copilot_suggestion that
    contains a ``probing_questions`` list, then tallies statuses.

    Args:
        run: A completed EvaluationRun with turn_metrics.

    Returns:
        QuestionAdherenceResult with lifecycle counts and question list.
    """
    # Find the last copilot suggestion with probing_questions
    probing_questions: list[dict] = []
    information_sufficient = False

    for turn in reversed(run.turn_metrics):
        suggestion = turn.copilot_suggestion
        if suggestion is None:
            continue
        pqs = suggestion.get("probing_questions", [])
        if pqs:
            probing_questions = pqs
            information_sufficient = suggestion.get("information_sufficient", False)
            break

    # Tally statuses
    total = len(probing_questions)
    answered = sum(1 for pq in probing_questions if pq.get("status") == "answered")
    invalidated = sum(1 for pq in probing_questions if pq.get("status") == "invalidated")
    skipped = sum(1 for pq in probing_questions if pq.get("status") == "skipped")
    pending = sum(1 for pq in probing_questions if pq.get("status") == "pending")

    adherence_rate = answered / total if total > 0 else 0.0

    return QuestionAdherenceResult(
        probing_questions=probing_questions,
        total_questions=total,
        answered=answered,
        invalidated=invalidated,
        skipped=skipped,
        pending=pending,
        information_sufficient=information_sufficient,
        overall_adherence_rate=adherence_rate,
    )
