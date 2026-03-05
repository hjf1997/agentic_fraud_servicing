"""Question planner specialist agent for next-best question suggestion.

Analyzes current case state (missing fields, hypothesis scores, evidence collected)
and suggests 1-3 targeted questions for the CCP to ask the cardmember. Uses OpenAI
Agents SDK with structured output via QuestionPlan.
"""

from agents import Agent, ModelProvider, Runner
from agents.run_config import RunConfig
from pydantic import BaseModel, Field

# System prompt for the question planner agent
QUESTION_INSTRUCTIONS = """\
You are a question planning specialist for AMEX card dispute servicing. Your role
is to suggest the next-best questions for the Contact Center Professional (CCP) to
ask the cardmember during a live call.

You receive:
- A summary of the current case state
- A list of missing fields that still need to be gathered
- Hypothesis scores indicating how likely the case is fraud, dispute, or scam

Your tasks:
1. **Prioritize missing fields**: Identify the single most important missing field
   that would most reduce uncertainty about the case category and resolution.
2. **Suggest 1-3 questions**: Craft open-ended questions that target the priority
   field and any secondary gaps. Questions should be natural and conversational.
3. **Provide rationale**: For each question, briefly explain what information it
   aims to elicit and why it matters for the case.
4. **Assess confidence**: Rate how confident you are that these questions will
   elicit useful information from the cardmember (0.0-1.0).

Rules:
- NEVER ask the customer to reveal their full card number (PAN) or CVV/CVC.
- Keep questions open-ended — avoid yes/no questions when possible.
- Avoid leading questions that suggest the answer.
- Ask about the highest-value missing field first.
- If hypothesis scores are close (e.g., fraud 0.4 vs scam 0.35), ask questions
  that help disambiguate between the two leading categories.
- If key transaction details are missing, prioritize those.
- If authentication or identity details are missing, ask about those.

Respond with structured output only. Be precise and actionable.
"""


class QuestionPlan(BaseModel):
    """Structured output from the question planner agent.

    Attributes:
        questions: 1-3 suggested next-best questions for the CCP to ask.
        rationale: Brief explanation for each question (parallel list).
        priority_field: The most important missing field targeted by the questions.
        confidence: Confidence these questions will elicit useful info (0.0-1.0).
    """

    questions: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    priority_field: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


# Agent instance with structured output
question_agent = Agent(
    name="question_planner",
    instructions=QUESTION_INSTRUCTIONS,
    output_type=QuestionPlan,
)


async def run_question_planner(
    case_summary: str,
    missing_fields: list[str],
    hypothesis_scores: dict[str, float],
    model_provider: ModelProvider,
) -> QuestionPlan:
    """Run the question planner agent to suggest next-best questions.

    Args:
        case_summary: Summary of the current case state.
        missing_fields: List of fields that still need to be gathered.
        hypothesis_scores: Dict mapping category names to confidence scores.
        model_provider: LLM model provider for inference.

    Returns:
        QuestionPlan with suggested questions and rationale.

    Raises:
        RuntimeError: If the agent SDK call fails.
    """
    # Build user message with case context
    parts = [f"Case summary:\n{case_summary}"]

    if missing_fields:
        fields_str = ", ".join(missing_fields)
        parts.append(f"\nMissing fields: {fields_str}")
    else:
        parts.append("\nNo missing fields identified.")

    if hypothesis_scores:
        scores_str = ", ".join(f"{cat}: {score:.2f}" for cat, score in hypothesis_scores.items())
        parts.append(f"\nHypothesis scores: {scores_str}")
    else:
        parts.append("\nNo hypothesis scores available yet.")

    user_msg = "\n".join(parts)

    try:
        result = await Runner.run(
            question_agent,
            input=user_msg,
            run_config=RunConfig(model_provider=model_provider),
        )
        return result.final_output
    except Exception as exc:
        raise RuntimeError(f"Question planner agent failed: {exc}") from exc
