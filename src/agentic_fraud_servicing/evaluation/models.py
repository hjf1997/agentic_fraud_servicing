"""Evaluation data models for the enterprise evaluation framework.

Defines core models for capturing copilot performance during transcript replay:
- EvaluationConfig: run configuration and ground truth
- TurnMetric: per-turn copilot output and latency
- EvaluationRun: aggregated metrics across all turns
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
