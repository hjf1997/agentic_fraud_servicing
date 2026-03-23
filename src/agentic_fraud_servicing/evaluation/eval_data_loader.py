"""Evaluation data loader — reads evaluation results from JSON files.

Pure data-loading module with no Gradio dependency. Each function handles
missing files gracefully by returning None or empty lists. Follows the same
pattern as ui/dashboard_data.py but reads JSON files instead of SQLite stores.
"""

from __future__ import annotations

import os
from pathlib import Path

from agentic_fraud_servicing.evaluation.models import EvaluationReport, EvaluationRun
from agentic_fraud_servicing.evaluation.report import extract_dimension_score


def discover_eval_scenarios(base_dir: str = "data/evaluations") -> list[str]:
    """Scan base_dir for subdirectories containing an evaluation_report.json.

    Args:
        base_dir: Root directory to scan for evaluation scenario subdirectories.

    Returns:
        Sorted list of scenario names (directory names). Empty list if base_dir
        doesn't exist or contains no valid evaluation scenarios.
    """
    if not os.path.isdir(base_dir):
        return []

    scenarios = []
    for entry in os.listdir(base_dir):
        scenario_dir = os.path.join(base_dir, entry)
        if os.path.isdir(scenario_dir) and os.path.isfile(
            os.path.join(scenario_dir, "evaluation_report.json")
        ):
            scenarios.append(entry)

    return sorted(scenarios)


def load_evaluation_run(scenario_dir: str) -> EvaluationRun | None:
    """Load an EvaluationRun from evaluation_run.json in the scenario directory.

    Args:
        scenario_dir: Path to the scenario directory containing evaluation_run.json.

    Returns:
        Deserialized EvaluationRun, or None if file is missing or invalid.
    """
    file_path = Path(scenario_dir) / "evaluation_run.json"
    if not file_path.is_file():
        return None

    try:
        content = file_path.read_text()
        return EvaluationRun.model_validate_json(content)
    except Exception:
        return None


def load_evaluation_report(scenario_dir: str) -> EvaluationReport | None:
    """Load an EvaluationReport from evaluation_report.json in the scenario directory.

    Args:
        scenario_dir: Path to the scenario directory containing evaluation_report.json.

    Returns:
        Deserialized EvaluationReport, or None if file is missing or invalid.
    """
    file_path = Path(scenario_dir) / "evaluation_report.json"
    if not file_path.is_file():
        return None

    try:
        content = file_path.read_text()
        return EvaluationReport.model_validate_json(content)
    except Exception:
        return None


def load_transcript_for_eval(scenario_dir: str) -> list[dict]:
    """Extract transcript turn data from the EvaluationRun in a scenario directory.

    Reads turn_metrics from the EvaluationRun and produces a list of dicts
    suitable for the dashboard transcript replay section.

    Args:
        scenario_dir: Path to the scenario directory containing evaluation_run.json.

    Returns:
        List of dicts with keys: turn_number, speaker, text, latency_ms,
        hypothesis_scores. Empty list if data is missing or invalid.
    """
    run = load_evaluation_run(scenario_dir)
    if run is None:
        return []

    turns = []
    for metric in run.turn_metrics:
        turns.append(
            {
                "turn_number": metric.turn_number,
                "speaker": metric.speaker,
                "text": metric.text,
                "latency_ms": metric.latency_ms,
                "hypothesis_scores": metric.hypothesis_scores,
            }
        )

    return turns


def extract_dimension_scores(report: EvaluationReport) -> dict[str, float | None]:
    """Extract a 0-1 score for each of the 8 evaluation dimensions.

    Uses the same scoring logic as report.py's extract_dimension_score to
    ensure consistency between the report aggregator and the dashboard display.

    Args:
        report: An EvaluationReport with optional sub-reports per dimension.

    Returns:
        Dict mapping dimension names to scores (0-1 float or None if the
        dimension was not evaluated). Keys: latency, prediction,
        question_adherence, allegation_quality, evidence_utilization,
        convergence, risk_flag_timeliness, decision_explanation.
    """
    dimension_results = {
        "latency": report.latency,
        "prediction": report.prediction,
        "question_adherence": report.question_adherence,
        "allegation_quality": report.allegation_quality,
        "evidence_utilization": report.evidence_utilization,
        "convergence": report.convergence,
        "risk_flag_timeliness": report.risk_flag_timeliness,
        "decision_explanation": report.decision_explanation,
    }

    return {dim: extract_dimension_score(dim, result) for dim, result in dimension_results.items()}
