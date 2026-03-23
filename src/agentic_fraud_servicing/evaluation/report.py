"""Report aggregator that orchestrates all 8 evaluators and produces an EvaluationReport.

Runs pure-Python evaluators (latency, convergence, evidence utilization) unconditionally.
LLM-powered evaluators (prediction, question adherence, allegation quality, risk flag
timeliness, decision explanation) run only when a model_provider is supplied. Each evaluator
call is wrapped in try/except for graceful degradation — failures produce None for that
dimension.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from agentic_fraud_servicing.evaluation.allegation_quality import (
    evaluate_allegation_quality,
)
from agentic_fraud_servicing.evaluation.convergence_evaluator import evaluate_convergence
from agentic_fraud_servicing.evaluation.decision_explainer import (
    evaluate_decision_explanation,
)
from agentic_fraud_servicing.evaluation.evidence_utilization import (
    evaluate_evidence_utilization,
)
from agentic_fraud_servicing.evaluation.latency_evaluator import evaluate_latency
from agentic_fraud_servicing.evaluation.models import EvaluationReport, EvaluationRun
from agentic_fraud_servicing.evaluation.prediction_evaluator import evaluate_prediction
from agentic_fraud_servicing.evaluation.question_adherence import (
    evaluate_question_adherence,
)
from agentic_fraud_servicing.evaluation.risk_flag_evaluator import (
    evaluate_risk_flag_timeliness,
)

if TYPE_CHECKING:
    from agentic_fraud_servicing.providers.base import ModelProvider

# Dimension weights for overall score calculation.
_WEIGHTS: dict[str, float] = {
    "latency": 0.10,
    "prediction": 0.20,
    "question_adherence": 0.10,
    "allegation_quality": 0.15,
    "evidence_utilization": 0.10,
    "convergence": 0.15,
    "risk_flag_timeliness": 0.10,
    "decision_explanation": 0.10,
}


def _extract_dimension_score(dimension: str, result: object) -> float | None:
    """Extract a 0-1 score from an evaluator result for the given dimension."""
    if result is None:
        return None

    if dimension == "latency":
        return result.compliance_rate  # type: ignore[union-attr]
    if dimension == "prediction":
        return 1.0 if result.match else 0.0  # type: ignore[union-attr]
    if dimension == "question_adherence":
        return result.overall_adherence_rate  # type: ignore[union-attr]
    if dimension == "allegation_quality":
        return result.f1_score  # type: ignore[union-attr]
    if dimension == "evidence_utilization":
        return (result.retrieval_coverage + result.reasoning_coverage) / 2  # type: ignore[union-attr]
    if dimension == "convergence":
        if result.convergence_ratio is not None:  # type: ignore[union-attr]
            return 1.0 - result.convergence_ratio  # type: ignore[union-attr]
        return 0.0
    if dimension == "risk_flag_timeliness":
        return result.flags_raised_count / max(result.flags_expected_count, 1)  # type: ignore[union-attr]
    if dimension == "decision_explanation":
        return 1.0 if result.reasoning_chain else 0.0  # type: ignore[union-attr]

    return None


def _compute_overall_score(dimension_scores: dict[str, float | None]) -> float:
    """Compute weighted average of available dimension scores, re-normalizing weights."""
    total_weight = 0.0
    weighted_sum = 0.0
    for dim, score in dimension_scores.items():
        if score is not None:
            weight = _WEIGHTS.get(dim, 0.0)
            weighted_sum += weight * score
            total_weight += weight

    if total_weight == 0.0:
        return 0.0
    return weighted_sum / total_weight


async def generate_report(
    run: EvaluationRun,
    model_provider: ModelProvider | None = None,
) -> EvaluationReport:
    """Run all evaluators and aggregate results into an EvaluationReport.

    Args:
        run: The completed EvaluationRun with per-turn metrics.
        model_provider: Optional LLM provider. When None, only pure-Python
            evaluators run (latency, convergence, evidence utilization).

    Returns:
        EvaluationReport with all available sub-reports and overall_score.
    """
    results: dict[str, object] = {
        "latency": None,
        "prediction": None,
        "question_adherence": None,
        "allegation_quality": None,
        "evidence_utilization": None,
        "convergence": None,
        "risk_flag_timeliness": None,
        "decision_explanation": None,
    }

    # --- Pure-Python evaluators (always run) ---
    try:
        results["latency"] = evaluate_latency(run)
    except Exception as exc:
        print(f"[report] latency evaluator failed: {exc}", file=sys.stderr)

    try:
        results["convergence"] = evaluate_convergence(run)
    except Exception as exc:
        print(f"[report] convergence evaluator failed: {exc}", file=sys.stderr)

    try:
        results["evidence_utilization"] = evaluate_evidence_utilization(run)
    except Exception as exc:
        print(f"[report] evidence_utilization evaluator failed: {exc}", file=sys.stderr)

    # --- LLM-powered evaluators (only when model_provider is available) ---
    if model_provider is not None:
        try:
            results["prediction"] = await evaluate_prediction(run, model_provider)
        except Exception as exc:
            print(f"[report] prediction evaluator failed: {exc}", file=sys.stderr)

        try:
            results["question_adherence"] = await evaluate_question_adherence(run, model_provider)
        except Exception as exc:
            print(f"[report] question_adherence evaluator failed: {exc}", file=sys.stderr)

        try:
            results["allegation_quality"] = await evaluate_allegation_quality(run, model_provider)
        except Exception as exc:
            print(f"[report] allegation_quality evaluator failed: {exc}", file=sys.stderr)

        try:
            results["risk_flag_timeliness"] = await evaluate_risk_flag_timeliness(
                run, model_provider
            )
        except Exception as exc:
            print(
                f"[report] risk_flag_timeliness evaluator failed: {exc}",
                file=sys.stderr,
            )

        try:
            results["decision_explanation"] = await evaluate_decision_explanation(
                run, model_provider
            )
        except Exception as exc:
            print(
                f"[report] decision_explanation evaluator failed: {exc}",
                file=sys.stderr,
            )

    # --- Compute overall score ---
    dimension_scores = {dim: _extract_dimension_score(dim, results[dim]) for dim in _WEIGHTS}
    overall_score = _compute_overall_score(dimension_scores)

    return EvaluationReport(
        scenario_name=run.scenario_name,
        overall_score=overall_score,
        latency=results["latency"],
        prediction=results["prediction"],
        question_adherence=results["question_adherence"],
        allegation_quality=results["allegation_quality"],
        evidence_utilization=results["evidence_utilization"],
        convergence=results["convergence"],
        risk_flag_timeliness=results["risk_flag_timeliness"],
        decision_explanation=results["decision_explanation"],
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def save_report(report: EvaluationReport, output_dir: str) -> str:
    """Write evaluation report JSON to output_dir/evaluation_report.json.

    Args:
        report: The EvaluationReport to persist.
        output_dir: Directory path for the output file.

    Returns:
        Absolute path of the written file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / "evaluation_report.json"
    file_path.write_text(report.model_dump_json(indent=2))
    return str(file_path)
