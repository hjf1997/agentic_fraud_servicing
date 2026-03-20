"""Case Advisor agent — policy-aware case opening eligibility assessment.

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
    """Assessment for a single case type (fraud, dispute, or scam)."""

    case_type: str
    """'fraud', 'dispute', or 'scam'."""

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

    next_info_needed: list[str] = Field(default_factory=list)
    """What information the CCP should gather next."""

    summary: str = ""
    """2-4 sentence summary of the eligibility landscape."""


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
to help the Contact Center Professional (CCP) determine which type of case to
open based on the current case state and AMEX policy documents.

This is ADVISORY only — the CCP makes the final decision on which case to open.

## Investigation Categories

{INVESTIGATION_CATEGORIES_REFERENCE}

## Policy Documents

The following policy documents define the criteria and rules for opening each
case type. Read them carefully and cite specific passages in your determinations.

{_POLICY_TEXT}

## Your Task

Given the current case state (allegations, evidence, hypothesis scores, and
conversation so far), evaluate eligibility for each case type:

### For each case type (fraud, dispute, scam):

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

### Next information needed

Identify what specific information the CCP should gather next to resolve any
`incomplete` statuses. Prioritize by impact — what single piece of information
would change the most eligibility statuses.

### Summary

Provide a 2-4 sentence summary of the overall eligibility landscape. Which
case types are available, which are blocked, and what should the CCP focus on
gathering next.

## Output Format

Return structured output with:
- assessments: list of CaseTypeAssessment (one per case type: fraud, dispute, scam)
- general_warnings: list of applicable cross-cutting warnings
- next_info_needed: list of information items the CCP should gather
- summary: 2-4 sentence eligibility summary
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
    conversation_summary: str,
    model_provider: ModelProvider,
) -> CaseAdvisory:
    """Run the Case Advisor agent to assess case opening eligibility.

    Args:
        allegations_summary: Formatted allegations with types and entities.
        evidence_summary: Retrieved evidence text (transactions, auth events).
        hypothesis_scores: Current 4-category hypothesis score distribution.
        conversation_summary: Running summary of the call so far.
        model_provider: LLM model provider for inference.

    Returns:
        CaseAdvisory with per-type eligibility assessments and warnings.

    Raises:
        RuntimeError: If the agent SDK call fails.
    """
    scores_text = ", ".join(f"{k}: {v:.2f}" for k, v in hypothesis_scores.items())

    user_msg = (
        f"## Allegations\n{allegations_summary}\n\n"
        f"## Evidence\n{evidence_summary}\n\n"
        f"## Current Hypothesis Scores\n{scores_text}\n\n"
        f"## Conversation Summary\n{conversation_summary}"
    )

    try:
        result = await Runner.run(
            case_advisor,
            input=user_msg,
            run_config=RunConfig(model_provider=model_provider),
        )
        return result.final_output
    except Exception as exc:
        raise RuntimeError(f"Case advisor agent failed: {exc}") from exc
