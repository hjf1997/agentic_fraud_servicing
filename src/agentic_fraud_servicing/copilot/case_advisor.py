"""Case Advisor agent — policy-aware eligibility assessment and question planning.

Merges the former Case Advisor and Question Planner into a single agent that:
- Evaluates case eligibility per type (eligible/blocked/incomplete)
- Suggests 0-3 next-best questions when information is insufficient
- Signals information_sufficient=True when all required info is gathered

Provides Pydantic output models (CaseTypeAssessment, CaseAdvisory), a
policy document loader (load_policies), the Case Advisor Agent instance,
and the run_case_advisor async runner function. Policy documents are loaded
at module import time and embedded in the agent's system prompt.
"""

from __future__ import annotations

from pathlib import Path

from agents import Agent, AgentOutputSchema, ModelProvider, Runner
from agents.run_config import RunConfig
from pydantic import BaseModel, Field

from agentic_fraud_servicing.models.enums import INVESTIGATION_CATEGORIES_REFERENCE

# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class CaseTypeAssessment(BaseModel):
    """Assessment for a single case type (fraud or dispute)."""

    case_type: str
    """'fraud' or 'dispute'."""

    eligibility: str
    """'eligible', 'blocked', or 'incomplete'."""

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
    """One per case type evaluated."""

    general_warnings: list[str] = Field(default_factory=list)
    """Cross-cutting warnings from general guidelines."""

    questions: list[str] = Field(default_factory=list)
    """0-3 suggested next-best questions for the CCP to ask.

    Empty when information_sufficient is True.
    """

    rationale: list[str] = Field(default_factory=list)
    """Brief explanation for each question (parallel list with questions)."""

    priority_field: str = ""
    """The most impactful missing information item, or empty when sufficient."""

    information_sufficient: bool = False
    """True when all required information has been gathered per the policy
    checklists. The CCP can proceed to case opening."""

    summary: str = ""
    """2-4 sentence summary of the eligibility landscape and next steps."""


# ---------------------------------------------------------------------------
# Policy document loader
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

    md_files = sorted(policies_dir.glob("*.md"))
    if not md_files:
        return ""

    sections: list[str] = []
    for md_file in md_files:
        sections.append(f"--- {md_file.name} ---")
        sections.append(md_file.read_text(encoding="utf-8").strip())

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# System prompt — policies loaded once at module import time
# ---------------------------------------------------------------------------

_POLICY_TEXT = load_policies()

CASE_ADVISOR_INSTRUCTIONS = f"""\
You are a Case Advisor specialist for AMEX card dispute servicing. Your role is
to help the Contact Center Professional (CCP) by assessing case eligibility AND
suggesting targeted questions to fill information gaps — all in a single step.

This is ADVISORY only — the CCP makes the final decision.

## Investigation Categories

{INVESTIGATION_CATEGORIES_REFERENCE}

## Policy Documents

The following policy documents define the criteria and rules for opening each
case type. Read them carefully and cite specific passages in your determinations.

{_POLICY_TEXT}

## Part 1 — Eligibility Assessment

Given the current case state (allegations, evidence, hypothesis scores, and
conversation so far), evaluate eligibility for each case type:

### For each case type (fraud, dispute):

1. **Determine eligibility status**:
   - `eligible` — All required criteria from the policy checklist are met and
     no blocking rules apply. The CCP can proceed to open this case type.
   - `blocked` — An active blocking rule prevents opening this case type.
     Cite the specific blocking rule from the policy document.
   - `incomplete` — Some required criteria are not yet satisfied but no
     blocking rules apply. The case could become eligible once more information
     is gathered.

2. **List met criteria** — Which policy requirements are already satisfied,
   with brief evidence references.

3. **List unmet criteria** — Which policy requirements are NOT yet satisfied.
   Be specific about what information is missing.

4. **List blockers** — Any active blocking rules that prevent opening this
   case type. Cite the exact policy text.

5. **Cite policy text** — For every determination, reference the specific
   policy document and passage. Use the format:
   "Per [filename]: '[quoted policy text]'"

### General warnings

Review the general guidelines document and flag any cross-cutting warnings
that apply to the current case (e.g., escalation triggers, priority rules,
case type conflicts, documentation requirements).

## Part 2 — Question Planning

Based on the eligibility assessment above, determine whether more information
is needed and suggest targeted questions.

### Stopping condition — information_sufficient

Check the unmet_criteria across ALL case type assessments against what has
already been gathered. If ALL of the following are true, set
`information_sufficient = true` and return an empty questions list:

- The leading hypothesis case type has status `eligible` (all criteria met,
  no blockers), OR
- All case types are `blocked` for reasons that cannot be resolved by
  gathering more information from the cardmember (e.g., regulatory blocks,
  account-level restrictions).

Otherwise, set `information_sufficient = false` and suggest questions.

### Question generation rules (when information is insufficient)

1. **Prioritize by impact**: Identify the single most important unmet criterion
   from the policy checklists — set this as `priority_field`. Target the
   question(s) at resolving it first.

2. **Suggest 1-3 questions**: Craft open-ended questions that target the
   priority field and any secondary gaps. Questions should be natural and
   conversational — suitable for a live phone call.

3. **Provide rationale**: For each question, briefly explain what information
   it aims to elicit and which policy criterion it addresses.

4. **Deduplication**: If recently suggested questions are provided, do NOT
   repeat them. Ask about the same topic from a different angle if needed.

5. **Disambiguation**: If hypothesis scores are close between two categories
   (e.g., THIRD_PARTY_FRAUD 0.4 vs SCAM 0.35), ask questions that help
   distinguish between them.

6. **First-party fraud probing**: If the FIRST_PARTY_FRAUD hypothesis is
   elevated, suggest questions that probe for contradictions between the
   cardmember's claims and verifiable facts (e.g., transaction details,
   delivery, device usage) without being accusatory.

7. **NEVER ask the customer to reveal their full card number (PAN) or
   CVV/CVC.** These are already on file and asking violates PCI-DSS.

8. **Prefer open-ended questions** — avoid yes/no when possible.

## Part 3 — Summary

Provide a 2-4 sentence summary covering:
- Which case types are available, blocked, or incomplete.
- What the CCP should focus on gathering next (if anything).
- Whether the case is ready to proceed (if information_sufficient is true).

## Output Format

Return structured output with:
- assessments: list of CaseTypeAssessment (one per case type: fraud, dispute)
- general_warnings: list of applicable cross-cutting warnings
- questions: 0-3 suggested questions (empty list when information_sufficient)
- rationale: parallel list with questions
- priority_field: most impactful missing field, or "" when sufficient
- information_sufficient: true when ready, false when more info needed
- summary: 2-4 sentence eligibility and next-steps summary
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
# Runner wrapper
# ---------------------------------------------------------------------------


async def run_case_advisor(
    allegations_summary: str,
    evidence_summary: str,
    hypothesis_scores: dict[str, float],
    conversation_window: list[tuple[str, str]],
    model_provider: ModelProvider,
    missing_fields: list[str] | None = None,
    recent_questions: list[str] | None = None,
) -> CaseAdvisory:
    """Run the Case Advisor agent to assess eligibility and suggest questions.

    Args:
        allegations_summary: Formatted allegations with types and entities.
        evidence_summary: Retrieved evidence text (transactions, auth events).
        hypothesis_scores: Current 4-category hypothesis score distribution.
        conversation_window: Recent (speaker, text) turns from the assessment-
            based conversation window (consistent with other agents).
        model_provider: LLM model provider for inference.
        missing_fields: Triage-extracted fields still missing (supplementary
            input — the agent also derives gaps from policy checklists).
        recent_questions: Previously suggested questions to avoid repeating.

    Returns:
        CaseAdvisory with eligibility assessments, questions, and stopping signal.

    Raises:
        RuntimeError: If the agent SDK call fails.
    """
    scores_text = ", ".join(f"{k}: {v:.2f}" for k, v in hypothesis_scores.items())

    parts = [
        f"## Allegations\n{allegations_summary}",
        f"## Evidence\n{evidence_summary}",
        f"## Current Hypothesis Scores\n{scores_text}",
    ]

    # Conversation window
    if conversation_window:
        turn_lines = [f"{speaker}: {text}" for speaker, text in conversation_window]
        parts.append("## Recent Conversation\n" + "\n".join(turn_lines))

    # Supplementary missing fields from triage entity tracking
    if missing_fields:
        fields_str = ", ".join(missing_fields)
        parts.append(f"## Missing Fields (from triage)\n{fields_str}")

    # Deduplication: recently suggested questions
    if recent_questions:
        q_lines = [f"- {q}" for q in recent_questions]
        parts.append(
            "## Recently Suggested Questions (do NOT repeat)\n" + "\n".join(q_lines)
        )

    user_msg = "\n\n".join(parts)

    try:
        result = await Runner.run(
            case_advisor,
            input=user_msg,
            run_config=RunConfig(model_provider=model_provider),
        )
        return result.final_output
    except Exception as exc:
        raise RuntimeError(f"Case advisor agent failed: {exc}") from exc
