"""Case Advisor agent — policy-aware case opening eligibility assessment.

Provides Pydantic output models (CaseTypeAssessment, CaseAdvisory) and a
policy document loader (load_policies) that reads markdown files from
docs/policies/ and concatenates them for injection into the agent prompt.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

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
