"""Question adherence evaluator — LLM-powered CCP question incorporation check.

Pairs copilot-suggested questions with the next CCP utterance and uses an LLM
agent to assess whether the CCP semantically incorporated the suggestions. The
CCP may rephrase, paraphrase, or ask a variant of the suggested question.
"""

from __future__ import annotations

from agents import Agent, AgentOutputSchema, ModelProvider
from agents.run_config import RunConfig

from agentic_fraud_servicing.providers.retry import run_with_retry
from pydantic import BaseModel

from agentic_fraud_servicing.evaluation.models import (
    EvaluationRun,
    QuestionAdherenceResult,
    TurnMetric,
)

# --- Output model for LLM adherence scoring ---


class AdherenceScore(BaseModel):
    """Structured output from the adherence scoring agent.

    Attributes:
        score: 0.0 = ignored, 0.5 = partially used, 1.0 = fully incorporated.
        explanation: Brief rationale for the score.
    """

    score: float = 0.0
    explanation: str = ""


# --- Agent instance ---

_ADHERENCE_INSTRUCTIONS = """\
You are an evaluation specialist assessing whether a Contact Center Professional
(CCP) incorporated suggested questions into their response.

You will receive:
1. A list of suggested questions the copilot recommended.
2. The CCP's actual utterance that followed.

## Scoring

- **1.0 (Fully incorporated)**: The CCP asked one or more of the suggested
  questions, even if rephrased, paraphrased, or combined with other questions.
- **0.5 (Partially used)**: The CCP's utterance touches on the topic of the
  suggestions but does not directly ask the question. Or the CCP addressed only
  a minor aspect of the suggestions.
- **0.0 (Ignored)**: The CCP's utterance is unrelated to any of the suggested
  questions, or is purely procedural with no informational overlap.

## Rules

1. Focus on semantic similarity, not exact wording.
2. The CCP may ask the question in a more natural, conversational way.
3. If the CCP asks about the same topic but with different framing, score 0.5.
4. Provide a brief 1-2 sentence explanation for your score.
"""

_adherence_agent = Agent(
    name="question_adherence_scorer",
    instructions=_ADHERENCE_INSTRUCTIONS,
    output_type=AgentOutputSchema(AdherenceScore, strict_json_schema=False),
)


# --- Public function ---


async def evaluate_question_adherence(
    run: EvaluationRun,
    model_provider: ModelProvider,
) -> QuestionAdherenceResult:
    """Evaluate whether the CCP incorporated copilot-suggested questions.

    For each turn with non-empty suggested_questions in the copilot_suggestion,
    finds the next CCP turn and uses an LLM to score semantic adherence.

    Args:
        run: A completed EvaluationRun with turn_metrics.
        model_provider: LLM model provider for inference.

    Returns:
        QuestionAdherenceResult with per-turn scores and overall rate.
    """
    per_turn_scores: list[dict] = []
    turns_with_suggestions = 0
    turns_with_adherence = 0
    metrics = run.turn_metrics

    for idx, turn in enumerate(metrics):
        # Skip turns without copilot suggestions or empty question lists
        suggestion = turn.copilot_suggestion
        if suggestion is None:
            continue
        questions = suggestion.get("suggested_questions", [])
        if not questions:
            continue

        turns_with_suggestions += 1

        # Find the next CCP turn after this one
        ccp_turn = _find_next_ccp_turn(metrics, idx)
        if ccp_turn is None:
            # No CCP follow-up — skip without penalty
            continue

        # Score adherence via LLM
        score, explanation = await _score_adherence(questions, ccp_turn.text, model_provider)

        per_turn_scores.append(
            {
                "turn_number": turn.turn_number,
                "suggested_questions": questions,
                "ccp_response_turn": ccp_turn.turn_number,
                "ccp_text": ccp_turn.text,
                "adherence_score": score,
                "explanation": explanation,
            }
        )

        if score >= 0.5:
            turns_with_adherence += 1

    overall_rate = (
        turns_with_adherence / turns_with_suggestions if turns_with_suggestions > 0 else 0.0
    )

    return QuestionAdherenceResult(
        per_turn_scores=per_turn_scores,
        overall_adherence_rate=overall_rate,
        turns_with_suggestions=turns_with_suggestions,
        turns_with_adherence=turns_with_adherence,
    )


def _find_next_ccp_turn(metrics: list[TurnMetric], start_idx: int) -> TurnMetric | None:
    """Scan forward from start_idx+1 for the next CCP speaker turn.

    Returns the TurnMetric or None if no CCP turn follows.
    """
    for i in range(start_idx + 1, len(metrics)):
        if metrics[i].speaker == "CCP":
            return metrics[i]
    return None


async def _score_adherence(
    questions: list[str],
    ccp_text: str,
    model_provider: ModelProvider,
) -> tuple[float, str]:
    """Use LLM agent to score how well the CCP incorporated suggested questions.

    Returns:
        Tuple of (score, explanation). Falls back to (0.0, error_msg) on failure.
    """
    questions_text = "\n".join(f"- {q}" for q in questions)
    user_msg = f"## Suggested Questions\n{questions_text}\n\n## CCP Utterance\n{ccp_text}"

    try:
        result = await run_with_retry(
            _adherence_agent,
            input=user_msg,
            run_config=RunConfig(model_provider=model_provider),
        )
        output: AdherenceScore = result.final_output
        return output.score, output.explanation
    except Exception as exc:
        from agentic_fraud_servicing.copilot.langfuse_tracing import extract_http_error

        status_code, error_body = extract_http_error(exc)
        detail = f"HTTP {status_code}: {error_body[:200]}" if status_code else str(exc)
        return 0.0, f"LLM scoring failed ({detail})"
