"""Evaluation data models for the enterprise evaluation framework.

Defines core models for capturing copilot performance during transcript replay:
- EvaluationConfig: run configuration and ground truth
- TurnMetric: per-turn copilot output and latency
- EvaluationRun: aggregated metrics across all turns

Evaluator result models (output types for L1-18 evaluators):
- LatencyReport: per-turn latency analysis with percentiles
- PredictionResult: outcome prediction accuracy
- QuestionAdherenceResult: CCP question incorporation
- AllegationQualityResult: triage extraction precision/recall/F1
- EvidenceUtilizationResult: evidence retrieval and reasoning coverage
- ConvergenceResult: hypothesis convergence speed
- RiskFlagTimelinessResult: risk flag timing vs evidence availability
- DecisionExplanation: decision reasoning chain
- NoteAlignmentResult: copilot output vs CCP notes alignment
- EvaluationReport: full aggregated report across all dimensions
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvaluationConfig(BaseModel):
    """Configuration for a single evaluation run."""

    scenario_name: str
    ground_truth: dict[str, Any]
    transcript_path: str
    evaluator_flags: dict[str, bool] = Field(default_factory=dict)


class TurnMetric(BaseModel):
    """Per-turn data captured during copilot transcript replay."""

    turn_number: int
    speaker: str
    text: str
    latency_ms: float
    copilot_suggestion: dict | None = None
    hypothesis_scores: dict[str, float] = Field(default_factory=dict)
    allegations_extracted: list[dict] = Field(default_factory=list)


class EvaluationRun(BaseModel):
    """Aggregates all per-turn metrics from a single evaluation run."""

    scenario_name: str
    ground_truth: dict[str, Any]
    turn_metrics: list[TurnMetric]
    total_turns: int
    total_latency_ms: float
    start_time: str
    end_time: str
    copilot_final_state: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluator result models
# ---------------------------------------------------------------------------


class LatencyReport(BaseModel):
    """Per-turn latency analysis with percentile distribution."""

    per_turn_latency_ms: list[float]
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    compliance_target_ms: float = 1500.0
    compliance_rate: float
    flagged_turns: list[int] = Field(default_factory=list)


class PredictionResult(BaseModel):
    """Outcome prediction accuracy against ground truth."""

    predicted_category: str
    ground_truth_category: str
    match: bool
    confidence_delta: float
    reasoning: str = ""


class QuestionAdherenceResult(BaseModel):
    """CCP question incorporation measurement."""

    per_turn_scores: list[dict] = Field(default_factory=list)
    overall_adherence_rate: float
    turns_with_suggestions: int
    turns_with_adherence: int


class AllegationQualityResult(BaseModel):
    """Triage extraction quality: precision, recall, and F1."""

    precision: float
    recall: float
    f1_score: float
    ground_truth_allegations: list[str] = Field(default_factory=list)
    extracted_allegations: list[str] = Field(default_factory=list)
    matched: list[str] = Field(default_factory=list)
    missed: list[str] = Field(default_factory=list)
    false_positives: list[str] = Field(default_factory=list)


class EvidenceUtilizationResult(BaseModel):
    """Evidence retrieval and reasoning coverage rates."""

    total_evidence_nodes: int
    retrieved_nodes: int
    referenced_in_reasoning: int
    retrieval_coverage: float
    reasoning_coverage: float
    missed_evidence: list[dict] = Field(default_factory=list)


class ConvergenceResult(BaseModel):
    """Hypothesis convergence speed measurement."""

    convergence_turn: int | None
    total_turns: int
    convergence_ratio: float | None
    correct_category: str
    turn_scores: list[dict] = Field(default_factory=list)


class RiskFlagTimelinessResult(BaseModel):
    """Risk flag timing relative to evidence availability."""

    per_flag_timing: list[dict] = Field(default_factory=list)
    average_delay_turns: float
    flags_raised_count: int
    flags_expected_count: int


class DecisionExplanation(BaseModel):
    """Decision reasoning chain and quality assessment."""

    reasoning_chain: str = ""
    influential_evidence: list[dict] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)
    overall_quality_notes: str = ""


class NoteAlignmentResult(BaseModel):
    """Copilot output vs CCP notes alignment scores."""

    facts_coverage_score: float
    allegation_alignment_score: float
    category_action_score: float
    overall_score: float
    explanation: str = ""


class EvaluationReport(BaseModel):
    """Full aggregated evaluation report across all dimensions."""

    scenario_name: str
    overall_score: float
    latency: LatencyReport | None = None
    prediction: PredictionResult | None = None
    question_adherence: QuestionAdherenceResult | None = None
    allegation_quality: AllegationQualityResult | None = None
    evidence_utilization: EvidenceUtilizationResult | None = None
    convergence: ConvergenceResult | None = None
    risk_flag_timeliness: RiskFlagTimelinessResult | None = None
    decision_explanation: DecisionExplanation | None = None
    note_alignment: NoteAlignmentResult | None = None
    generated_at: str = ""
