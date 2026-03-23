"""Enterprise evaluation framework for copilot quality assessment."""

from agentic_fraud_servicing.evaluation.allegation_quality import (
    evaluate_allegation_quality,
)
from agentic_fraud_servicing.evaluation.convergence_evaluator import (
    evaluate_convergence,
)
from agentic_fraud_servicing.evaluation.decision_explainer import (
    evaluate_decision_explanation,
)

# Data loader functions for evaluation dashboard
from agentic_fraud_servicing.evaluation.eval_data_loader import (
    discover_eval_scenarios,
    extract_dimension_scores,
    load_evaluation_report,
    load_evaluation_run,
    load_transcript_for_eval,
)
from agentic_fraud_servicing.evaluation.evidence_utilization import (
    evaluate_evidence_utilization,
)
from agentic_fraud_servicing.evaluation.latency_evaluator import evaluate_latency
from agentic_fraud_servicing.evaluation.models import (
    AllegationQualityResult,
    ConvergenceResult,
    DecisionExplanation,
    EvaluationConfig,
    EvaluationReport,
    EvaluationRun,
    EvidenceUtilizationResult,
    LatencyReport,
    PredictionResult,
    QuestionAdherenceResult,
    RiskFlagTimelinessResult,
    TurnMetric,
)
from agentic_fraud_servicing.evaluation.prediction_evaluator import (
    evaluate_prediction,
)
from agentic_fraud_servicing.evaluation.question_adherence import (
    evaluate_question_adherence,
)
from agentic_fraud_servicing.evaluation.report import (
    extract_dimension_score,
    generate_report,
    save_report,
)
from agentic_fraud_servicing.evaluation.risk_flag_evaluator import (
    evaluate_risk_flag_timeliness,
)

__all__ = [
    # Models
    "AllegationQualityResult",
    "ConvergenceResult",
    "DecisionExplanation",
    "EvaluationConfig",
    "EvaluationReport",
    "EvaluationRun",
    "EvidenceUtilizationResult",
    "LatencyReport",
    "PredictionResult",
    "QuestionAdherenceResult",
    "RiskFlagTimelinessResult",
    "TurnMetric",
    # Evaluator functions
    "evaluate_allegation_quality",
    "evaluate_convergence",
    "evaluate_decision_explanation",
    "evaluate_evidence_utilization",
    "evaluate_latency",
    "evaluate_prediction",
    "evaluate_question_adherence",
    "evaluate_risk_flag_timeliness",
    # Report aggregator
    "extract_dimension_score",
    "generate_report",
    "save_report",
    # Data loader functions
    "discover_eval_scenarios",
    "extract_dimension_scores",
    "load_evaluation_report",
    "load_evaluation_run",
    "load_transcript_for_eval",
]
