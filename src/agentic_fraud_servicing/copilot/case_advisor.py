"""Case Advisor agent — question planner consuming specialist assessments.

Plans targeted questions based on specialist-identified evidence gaps and
eligibility status. Does NOT load policies — specialists handle policy-
grounded reasoning and eligibility assessment. The case advisor translates
their findings into CCP-friendly questions and determines when enough
information has been gathered.

Also provides a lightweight Question Validator agent that checks whether
previously suggested probing questions have been answered or invalidated
by new conversation turns.

Provides Pydantic output models (CaseTypeAssessment, CaseAdvisory,
QuestionUpdate, QuestionValidationResult), the Case Advisor and Question
Validator Agent instances, and their async runner functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from agents import Agent, AgentOutputSchema, ModelProvider
from agents.run_config import RunConfig
from pydantic import BaseModel, Field

from agentic_fraud_servicing.copilot.hypothesis_specialists import (
    SpecialistAssessment,
)
from agentic_fraud_servicing.models.case import ProbingQuestion
from agentic_fraud_servicing.providers.retry import run_with_retry

# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class CaseTypeAssessment(BaseModel):
    """Assessment for a single case type (fraud or dispute)."""

    case_type: Literal["fraud", "dispute"]
    """Case opening type: exactly 'fraud' or 'dispute'."""

    eligibility: Literal["eligible", "blocked", "incomplete"]
    """Eligibility status."""

    met_criteria: list[str] = Field(default_factory=list)
    """Criteria that are satisfied."""

    unmet_criteria: list[str] = Field(default_factory=list)
    """Criteria not yet satisfied."""

    blockers: list[str] = Field(default_factory=list)
    """Active blocking rules with explanations."""

    policy_citations: list[str] = Field(default_factory=list)
    """Specific policy text cited for each determination."""


class CaseAdvisory(BaseModel):
    """Full output from the Case Advisor agent."""

    assessments: list[CaseTypeAssessment] = Field(default_factory=list)
    """One per case type evaluated. Populated programmatically from specialist
    eligibility, not produced by the LLM."""

    general_warnings: list[str] = Field(default_factory=list)
    """Cross-cutting warnings (e.g., escalation triggers)."""

    questions: list[str] = Field(default_factory=list)
    """0-3 NEW suggested next-best questions for the CCP to ask.

    Empty when information_sufficient is True. These are new questions
    only — not previously suggested ones.
    """

    question_targets: list[str] = Field(default_factory=list)
    """Parallel list with questions — which investigation category each
    question helps discriminate (e.g. 'THIRD_PARTY_FRAUD', 'SCAM')."""

    rationale: list[str] = Field(default_factory=list)
    """Brief explanation for each question (parallel list with questions)."""

    priority_field: str = ""
    """The most impactful missing information item, or empty when sufficient."""

    information_sufficient: bool = False
    """True when all required information has been gathered. The CCP can
    proceed to case opening."""

    summary: str = ""
    """2-4 sentence summary of the eligibility landscape and next steps."""


# ---------------------------------------------------------------------------
# Question Validator output models
# ---------------------------------------------------------------------------


class QuestionUpdate(BaseModel):
    """Status update for a single probing question."""

    question_text: str
    """The original question text (used to match back to the ProbingQuestion)."""

    new_status: Literal["pending", "answered", "invalidated"]
    """Updated status: pending (still unanswered), answered (CM addressed the
    topic), or invalidated (no longer relevant)."""

    reason: str = ""
    """Brief explanation for the status change (empty when still pending)."""


class QuestionValidationResult(BaseModel):
    """Output from the Question Validator agent."""

    updates: list[QuestionUpdate] = Field(default_factory=list)
    """One update per pending question that was evaluated."""


# ---------------------------------------------------------------------------
# Policy document loader (kept for backward compatibility)
# ---------------------------------------------------------------------------


def _find_project_root() -> Path:
    """Walk up from this file until we find pyproject.toml."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent


def load_policies(policies_dir: str | Path | None = None) -> str:
    """Load all .md policy files and concatenate with separators.

    .. deprecated::
        Policy loading is now handled by individual specialists via
        ``hypothesis_specialists.load_specialist_policies()``. This function
        is retained for backward compatibility but is no longer called at
        module import time.

    Args:
        policies_dir: Directory containing .md policy files.
            Defaults to ``docs/policies/`` relative to the project root.

    Returns:
        Concatenated policy text with ``--- filename.md ---`` separators.
        Returns an empty string if the directory is missing or has no .md files.
    """
    if policies_dir is None:
        policies_dir = _find_project_root() / "docs" / "policies"
    else:
        policies_dir = Path(policies_dir)

    if not policies_dir.is_dir():
        return ""

    md_files = sorted(policies_dir.rglob("*.md"))
    if not md_files:
        return ""

    sections: list[str] = []
    for md_file in md_files:
        relative = md_file.relative_to(policies_dir)
        sections.append(f"--- {relative} ---")
        sections.append(md_file.read_text(encoding="utf-8").strip())

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# System prompt — no policies, consumes specialist outputs
# ---------------------------------------------------------------------------

CASE_ADVISOR_INSTRUCTIONS = """\
You are a Case Advisor for AMEX card dispute servicing. Your role is to help
the Contact Center Professional (CCP) by suggesting targeted questions to fill
information gaps identified by category specialists.

This is ADVISORY only — the CCP makes the final decision.

## Your Input

You receive:
1. **Specialist Assessments** — Three independent evaluations (Dispute, Scam,
   Third-Party Fraud), each with: likelihood score, eligibility status
   (eligible/blocked), evidence gaps, policy-grounded reasoning, and policy
   citations. Specialists have already evaluated the evidence against their
   category's policies.
2. **Hypothesis Scores** — Current 5-category probability distribution
   (THIRD_PARTY_FRAUD, FIRST_PARTY_FRAUD, SCAM, DISPUTE, UNABLE_TO_DETERMINE).
3. **Recent Conversation** — Recent transcript turns for context.
4. **Current Question List** — All previously suggested probing questions with
   their lifecycle status: `pending` (not yet answered), `answered` (CM
   addressed the topic), or `invalidated` (no longer relevant). Use this to
   avoid generating duplicate questions and to assess probing progress.

## Stopping Condition — information_sufficient

Check the specialist eligibility statuses, evidence gaps, AND the question
list. Set `information_sufficient = true` and return an empty questions list
if:

- All questions in the list are answered or invalidated AND you determine no
  new questions are needed (probing is complete), OR
- The leading hypothesis specialist has eligibility `eligible` AND has no
  critical online evidence gaps remaining, OR
- All specialists have eligibility `blocked` for reasons that cannot be
  resolved by gathering more information from the cardmember.

If UNABLE_TO_DETERMINE is dominant but no online evidence gaps remain, flag
this in the summary ("evidence gaps remain but cannot be resolved through
probing") rather than generating more questions.

Note: Some evidence gaps are marked `[offline]` by specialists — these can
only be collected after case opening (e.g., merchant records, device forensics,
payment platform data). Offline gaps should NOT prevent `information_sufficient`
from being set to true. Focus only on gaps that are resolvable during the live
call (what the CCP can ask the cardmember).

## Question Generation Rules

1. **Prioritize by impact**: Identify the most critical evidence gap that can
   be resolved during this call — set this as `priority_field`. Focus on gaps
   from the specialist whose category aligns with the leading hypothesis.
   Ignore `[offline]` gaps when choosing questions (those require post-case
   investigation).

2. **Generate 0-3 NEW questions only**: Do NOT repeat any question already in
   the question list (whether pending, answered, or invalidated). Each question
   must target a distinct evidence gap not already covered. Questions must be
   concise (one sentence), natural, and suitable for a live phone call.

3. **Specify target category**: For each question, identify which investigation
   category it helps discriminate in `question_targets` (parallel list). This
   links questions to the hypothesis they aim to resolve.

4. **Provide rationale**: For each question, briefly explain what information
   it aims to elicit and which specialist's evidence gap it addresses.

5. **Disambiguation**: If hypothesis scores are close between two categories
   (e.g., THIRD_PARTY_FRAUD 0.4 vs SCAM 0.35), ask questions that help
   distinguish between them.

6. **High UNABLE_TO_DETERMINE**: When UNABLE_TO_DETERMINE is the leading
   score, the system lacks enough evidence to distinguish between categories.
   Prioritize questions that help discriminate: ask about transaction
   authorization (distinguishes third-party fraud from others), whether
   anyone else had access to the card (distinguishes first-party fraud),
   external contacts or manipulation (distinguishes scam), and merchant
   interaction details (distinguishes dispute).

7. **First-party fraud probing**: If the FIRST_PARTY_FRAUD hypothesis is
   elevated (≥ 0.3), suggest questions that probe for contradictions between
   the cardmember's claims and verifiable facts (e.g., transaction details,
   delivery, device usage) without being accusatory.

8. **NEVER ask the customer to reveal their full card number (PAN) or
   CVV/CVC.** These are already on file and asking violates PCI-DSS.

9. **Prefer open-ended questions** — avoid yes/no when possible.

## Warnings

Flag any cross-cutting concerns in `general_warnings`:
- Contradictions detected across multiple specialists
- Escalation triggers (high-value transactions, vulnerable cardholders)
- First-party fraud indicators when FIRST_PARTY_FRAUD score is elevated

## Summary

Provide a 2-4 sentence summary covering:
- Which case types are available or blocked (based on specialist eligibility).
- What the CCP should focus on gathering next (if anything).
- Whether the case is ready to proceed.

## Output Format

Return structured output with:
- general_warnings: list of cross-cutting warnings
- questions: 0-3 NEW suggested questions (empty list when information_sufficient)
- question_targets: parallel list — target investigation category per question
- rationale: parallel list with questions
- priority_field: most impactful evidence gap, or "" when sufficient
- information_sufficient: true when ready, false when more info needed
- summary: 2-4 sentence eligibility and next-steps summary

NOTE: Do NOT include assessments in your output — those are populated
separately from specialist data.
"""


# ---------------------------------------------------------------------------
# Agent instance
# ---------------------------------------------------------------------------

case_advisor = Agent(
    name="case_advisor",
    instructions=CASE_ADVISOR_INSTRUCTIONS,
    output_type=AgentOutputSchema(CaseAdvisory, strict_json_schema=False),
)


# ---------------------------------------------------------------------------
# Question Validator prompt and agent
# ---------------------------------------------------------------------------

QUESTION_VALIDATOR_INSTRUCTIONS = """\
You are a Question Validator for a fraud servicing copilot. Your job is to
check whether previously suggested probing questions have been answered or
are no longer relevant, based on new conversation turns.

## Your Input

You receive:
1. **Pending Questions** — Previously suggested questions that have not yet
   been answered. Each has a text and the investigation category it targets.
2. **New Conversation Turns** — Transcript turns since the questions were last
   evaluated.
3. **Current Hypothesis Scores** — 5-category probability distribution.

## Your Task

For EACH pending question, determine its new status:

- **answered** — The cardmember addressed the topic in the new conversation
  turns. The answer does not need to be exact — if the CM provided information
  that substantially covers what the question was seeking, mark it answered.
  Provide a brief reason (e.g., "CM confirmed they did not authorize the txn").

- **invalidated** — The question is no longer relevant because:
  - The hypothesis category it targets has collapsed to near-zero (< 0.05)
  - Evidence from other sources already resolved the underlying gap
  - The conversation has moved past the point where asking makes sense
  Provide a brief reason (e.g., "SCAM hypothesis collapsed to 0.02").

- **pending** — The question has NOT been answered and is still relevant.
  Leave reason empty.

## Rules

- **Immutability**: You only receive questions with status `pending`. Questions
  that were previously marked `answered` or `invalidated` are NOT provided —
  their status is final and cannot be changed. You can only transition a
  question FROM `pending` TO `answered` or `invalidated`, never the reverse.
- Be conservative: only mark as answered if the CM clearly addressed the topic
- A partial answer still counts — mark as answered if the core information was provided
- Do NOT invent new questions — only evaluate the ones provided

## Output Format

Return structured output with:
- updates: one entry per pending question with question_text, new_status, reason
"""

question_validator = Agent(
    name="question_validator",
    instructions=QUESTION_VALIDATOR_INSTRUCTIONS,
    output_type=AgentOutputSchema(QuestionValidationResult, strict_json_schema=False),
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_specialists_for_advisor(
    assessments: dict[str, SpecialistAssessment],
) -> str:
    """Format specialist outputs into the case advisor's user message."""
    _LABELS = {
        "DISPUTE": "Dispute Specialist",
        "SCAM": "Scam Specialist",
        "THIRD_PARTY_FRAUD": "Fraud Specialist (Third-Party)",
    }
    parts: list[str] = []
    for category in ("DISPUTE", "SCAM", "THIRD_PARTY_FRAUD"):
        a = assessments.get(category)
        if a is None:
            parts.append(f"### {_LABELS[category]}\nNot available.")
            continue
        gaps = ", ".join(a.evidence_gaps) if a.evidence_gaps else "none identified"
        citations = (
            "\n".join(f"  - {c}" for c in a.policy_citations) if a.policy_citations else "  none"
        )
        parts.append(
            f"### {_LABELS[category]}\n"
            f"Likelihood: {a.likelihood:.2f}\n"
            f"Eligibility: {a.eligibility}\n"
            f"Reasoning: {a.reasoning}\n"
            f"Evidence gaps: {gaps}\n"
            f"Policy citations:\n{citations}"
        )
    return "\n\n".join(parts)


def _map_specialists_to_case_types(
    assessments: dict[str, SpecialistAssessment],
) -> list[CaseTypeAssessment]:
    """Map specialist eligibility to CaseTypeAssessment for dashboard compat.

    Only DISPUTE and THIRD_PARTY_FRAUD map to case opening types (dispute
    and fraud respectively). SCAM does not have a separate case type — scam
    cases are opened under the fraud case type.
    """
    result: list[CaseTypeAssessment] = []

    dispute_spec = assessments.get("DISPUTE")
    if dispute_spec:
        # Map specialist "eligible"/"blocked" to CaseTypeAssessment
        # If blocked, the reasoning explains why
        blockers = [dispute_spec.reasoning] if dispute_spec.eligibility == "blocked" else []
        result.append(
            CaseTypeAssessment(
                case_type="dispute",
                eligibility=dispute_spec.eligibility,
                met_criteria=list(dispute_spec.supporting_evidence),
                unmet_criteria=list(dispute_spec.evidence_gaps),
                blockers=blockers,
                policy_citations=list(dispute_spec.policy_citations),
            )
        )

    fraud_spec = assessments.get("THIRD_PARTY_FRAUD")
    if fraud_spec:
        blockers = [fraud_spec.reasoning] if fraud_spec.eligibility == "blocked" else []
        result.append(
            CaseTypeAssessment(
                case_type="fraud",
                eligibility=fraud_spec.eligibility,
                met_criteria=list(fraud_spec.supporting_evidence),
                unmet_criteria=list(fraud_spec.evidence_gaps),
                blockers=blockers,
                policy_citations=list(fraud_spec.policy_citations),
            )
        )

    return result


# ---------------------------------------------------------------------------
# Runner wrapper
# ---------------------------------------------------------------------------


def _format_question_list_for_advisor(
    probing_questions: list[ProbingQuestion],
) -> str:
    """Format the full probing question list for the case advisor's user message."""
    if not probing_questions:
        return "No questions suggested yet."
    lines: list[str] = []
    for pq in probing_questions:
        target = f" [{pq.target_category}]" if pq.target_category else ""
        reason = f" — {pq.reason}" if pq.reason else ""
        lines.append(f"- [{pq.status}]{target} (turn {pq.turn_suggested}) {pq.text}{reason}")
    return "\n".join(lines)


async def run_case_advisor(
    specialist_assessments: dict[str, SpecialistAssessment],
    hypothesis_scores: dict[str, float],
    conversation_window: list[tuple[str, str]],
    model_provider: ModelProvider,
    probing_questions: list[ProbingQuestion] | None = None,
) -> CaseAdvisory:
    """Run the Case Advisor agent to suggest questions.

    Takes specialist assessments as primary input — the case advisor does
    not load policies. Eligibility assessments are mapped programmatically
    from specialist outputs after the LLM returns.

    Args:
        specialist_assessments: Specialist outputs keyed by category
            (DISPUTE, SCAM, THIRD_PARTY_FRAUD).
        hypothesis_scores: Current 5-category hypothesis score distribution.
        conversation_window: Recent (speaker, text) turns from the assessment-
            based conversation window.
        model_provider: LLM model provider for inference.
        probing_questions: Full probing question list with lifecycle statuses.

    Returns:
        CaseAdvisory with questions, stopping signal, and eligibility
        assessments mapped from specialist outputs.

    Raises:
        RuntimeError: If the agent SDK call fails.
    """
    scores_text = ", ".join(f"{k}: {v:.2f}" for k, v in hypothesis_scores.items())

    parts = [
        f"## Specialist Assessments\n\n{_format_specialists_for_advisor(specialist_assessments)}",
        f"## Current Hypothesis Scores\n{scores_text}",
    ]

    # Conversation window
    if conversation_window:
        turn_lines = [f"{speaker}: {text}" for speaker, text in conversation_window]
        parts.append("## Recent Conversation\n" + "\n".join(turn_lines))

    # Full question list with statuses
    if probing_questions:
        parts.append(
            "## Current Question List\n" + _format_question_list_for_advisor(probing_questions)
        )
    else:
        parts.append("## Current Question List\nNo questions suggested yet.")

    user_msg = "\n\n".join(parts)

    try:
        result = await run_with_retry(
            case_advisor,
            input=user_msg,
            run_config=RunConfig(model_provider=model_provider),
        )
        advisory: CaseAdvisory = result.final_output
        # Populate eligibility assessments from specialist data (not LLM)
        advisory.assessments = _map_specialists_to_case_types(specialist_assessments)
        return advisory
    except Exception as exc:
        raise RuntimeError(f"Case advisor agent failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Question Validator runner
# ---------------------------------------------------------------------------


async def validate_pending_questions(
    pending_questions: list[ProbingQuestion],
    new_turns: list[tuple[str, str]],
    hypothesis_scores: dict[str, float],
    model_provider: ModelProvider,
) -> QuestionValidationResult:
    """Validate pending probing questions against new conversation turns.

    Checks whether each pending question has been answered by the cardmember,
    invalidated by hypothesis changes, or remains pending.

    Args:
        pending_questions: Questions with status="pending" to evaluate.
        new_turns: New (speaker, text) turns since last validation.
        hypothesis_scores: Current 5-category hypothesis score distribution.
        model_provider: LLM model provider for inference.

    Returns:
        QuestionValidationResult with status updates for each question.

    Raises:
        RuntimeError: If the validator agent call fails.
    """
    # Format pending questions
    q_lines: list[str] = []
    for pq in pending_questions:
        target = f" [targets: {pq.target_category}]" if pq.target_category else ""
        q_lines.append(f"- {pq.text}{target}")
    questions_text = "\n".join(q_lines)

    # Format new conversation turns
    turn_lines = [f"{speaker}: {text}" for speaker, text in new_turns]
    turns_text = "\n".join(turn_lines) if turn_lines else "No new turns."

    # Format hypothesis scores
    scores_text = ", ".join(f"{k}: {v:.2f}" for k, v in hypothesis_scores.items())

    user_msg = (
        f"## Pending Questions\n{questions_text}\n\n"
        f"## New Conversation Turns\n{turns_text}\n\n"
        f"## Current Hypothesis Scores\n{scores_text}"
    )

    try:
        result = await run_with_retry(
            question_validator,
            input=user_msg,
            run_config=RunConfig(model_provider=model_provider),
        )
        return result.final_output
    except Exception as exc:
        raise RuntimeError(f"Question validator agent failed: {exc}") from exc
