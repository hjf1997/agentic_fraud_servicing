"""Category specialist agents for hypothesis scoring.

Three specialists — Dispute, Scam, and Fraud — evaluate evidence through their
category's policy lens in parallel. Each produces a SpecialistAssessment with a
likelihood score, policy-grounded reasoning, and evidence citations. The
hypothesis agent (arbitrator) synthesizes these into the final 4-category
distribution.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from agents import Agent, AgentOutputSchema, ModelProvider, Runner
from agents.run_config import RunConfig
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


class SpecialistAssessment(BaseModel):
    """Output from a single category specialist.

    Attributes:
        category: The investigation category this specialist evaluates.
        likelihood: How well this category explains the evidence (0.0-1.0).
        reasoning: Policy-grounded explanation (2-4 sentences).
        supporting_evidence: Evidence items supporting this category.
        contradicting_evidence: Evidence items contradicting this category.
        policy_citations: Specific policy passages cited.
        evidence_gaps: Information still needed to complete the evaluation.
        eligibility: Whether this case type can be opened. Default is
            ``eligible`` — blocked only when a specific policy rule or
            evidence directly prevents case opening.
    """

    category: str
    likelihood: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    policy_citations: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    eligibility: Literal["eligible", "blocked"] = "eligible"


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------


def _find_project_root() -> Path:
    """Walk up from this file until we find pyproject.toml."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent


def load_specialist_policies(
    specialist: str, policies_dir: str | Path | None = None
) -> str:
    """Load all .md policy files for a specialist category.

    Each specialist has its own subdirectory under the policies root
    (e.g., ``docs/policies/dispute/``, ``docs/policies/scam/``).
    All ``.md`` files in that subdirectory are concatenated with section
    headers derived from the filename.

    Args:
        specialist: Subdirectory name (``dispute``, ``scam``, or ``fraud``).
        policies_dir: Root policies directory.
            Defaults to ``docs/policies/`` relative to the project root.

    Returns:
        Concatenated policy text, or empty string if the directory is missing.
    """
    if policies_dir is None:
        policies_dir = _find_project_root() / "docs" / "policies"
    else:
        policies_dir = Path(policies_dir)

    specialist_dir = policies_dir / specialist
    if not specialist_dir.is_dir():
        return ""

    parts: list[str] = []
    for filepath in sorted(specialist_dir.glob("*.md")):
        content = filepath.read_text(encoding="utf-8").strip()
        if content:
            parts.append(content)
    return "\n\n---\n\n".join(parts)


# Load policies at module import time — each specialist reads its own subfolder
_DISPUTE_POLICIES = load_specialist_policies("dispute")
_SCAM_POLICIES = load_specialist_policies("scam")
_FRAUD_POLICIES = load_specialist_policies("fraud")


# ---------------------------------------------------------------------------
# Category definitions (extracted from INVESTIGATION_CATEGORIES_REFERENCE)
# ---------------------------------------------------------------------------

_DISPUTE_DEFINITION = """\
A transaction authorized by the cardmember where the complaint is about the
merchant's performance, billing, or service — not about fraud or deception.
- Authorization: YES — the CM made or authorized the transaction.
- Fraud actor: None — no fraud involved.
- CM role: Legitimate complainant — the CM has a valid grievance with the merchant.
- Evidence focus: Merchant records, delivery proof, refund policy, service level
  agreements, billing statements, prior communication with merchant, product/service
  description vs. what was delivered.
- Investigation question: "Did the merchant fail to deliver what was promised,
  or is there a legitimate billing error?"
"""

_SCAM_DEFINITION = """\
A transaction authorized by the cardmember, but the authorization was obtained
through deception or manipulation by an external fraudster. The CM willingly
made the payment but was tricked into doing so.
- Authorization: YES — the CM authorized the transaction, but under false pretenses.
- Fraud actor: External scammer (romance scammer, investment fraudster, tech
  support impersonator, phishing operator).
- CM role: Victim of manipulation — the CM was deceived into authorizing.
- Evidence focus: Narrative consistency and social-engineering patterns,
  communication trail with the scammer, urgency or pressure tactics in the
  transcript, coached language suggesting third-party influence, payment
  patterns typical of scam (wire, gift cards, crypto).
- Investigation question: "Did the cardmember authorize the payment because
  they were deceived by an external party?"
"""

_FRAUD_DEFINITION = """\
A transaction made without the cardmember's knowledge or permission by an
external criminal who gained unauthorized access to the account or card
credentials.
- Authorization: NO — the CM did not authorize the transaction.
- Fraud actor: External criminal (identity thief, card skimmer, data breach
  exploiter).
- CM role: Victim — the CM had no involvement in the transaction.
- Evidence focus: Authentication logs showing unusual device/IP/location,
  card-not-present indicators, chip vs. swipe discrepancies, device fingerprint
  mismatches, rapid sequential transactions, geographic impossibility.
- Investigation question: "Did the cardmember actually authorize the transaction?"
"""


# ---------------------------------------------------------------------------
# Specialist system prompts
# ---------------------------------------------------------------------------

DISPUTE_SPECIALIST_INSTRUCTIONS = f"""\
You are a Dispute Specialist evaluating whether a cardmember's complaint is a
legitimate merchant dispute — an authorized transaction where the issue is
about the merchant's performance, billing, or service.

## Your Category

{_DISPUTE_DEFINITION}

## Policy Documents

{_DISPUTE_POLICIES}

## Instructions

Assume the other investigation categories (fraud, scam) do not apply. Focus
solely on evaluating how well "merchant dispute" explains the evidence.

1. **Evaluate the allegations**: Do the CM's claims describe a merchant
   performance or billing issue (goods not received, defective, duplicate
   charge, service not rendered, recurring after cancellation)?
2. **Check evidence alignment**: Does retrieved evidence support or contradict
   a dispute narrative? Look for delivery proofs, refund records, merchant
   communications, billing patterns.
3. **Apply policy criteria**: Check the dispute case checklist — are the
   eligibility requirements met? Are any blocking rules triggered?
4. **Score likelihood**: 0.0 = evidence clearly rules out dispute,
   1.0 = evidence strongly confirms this is a merchant dispute.
5. **Cite policies**: Reference specific policy passages for your determination.
6. **Assess eligibility**: Based on the policy checklist and your own judgment:
   - `eligible` (default) — no blocking rules triggered, case can proceed
   - `blocked` — a specific blocking rule applies or evidence directly
     contradicts the dispute allegation. Explain why in your reasoning
     and cite the relevant policy.
7. **Identify evidence gaps**: List specific information still needed to
   complete the evaluation (e.g., merchant contact confirmation, delivery
   proof, cancellation date).

If you have a previous assessment, explain what changed since then.

Respond with structured output only.
"""

SCAM_SPECIALIST_INSTRUCTIONS = f"""\
You are a Scam Specialist evaluating whether a cardmember was deceived by an
external fraudster into authorizing a transaction they would not otherwise
have made.

## Your Category

{_SCAM_DEFINITION}

## Policy Documents

{_SCAM_POLICIES}

## Social Engineering Patterns

Look for these indicators in the conversation and evidence:

- **Romance scam**: Relationship built online, escalating financial requests,
  emotional manipulation, payments to individuals not businesses
- **Investment scam**: Promises of high returns, urgency to invest, unfamiliar
  platforms, crypto or wire transfers to unknown entities
- **Tech support scam**: Unsolicited contact about "problems," remote access
  granted, payments for fake services or software
- **Impersonation scam**: Caller claims to be from bank/IRS/law enforcement,
  creates urgency, demands immediate payment
- **Purchase scam**: Fake online store, social media marketplace fraud,
  payment outside platform protections

Behavioral signals in the transcript:
- Coached language (scripted responses, unnatural phrasing)
- Urgency or pressure from an external party
- CM mentions being told to say specific things
- Payment to unfamiliar recipient or unusual payment method
- CM seems confused about what they purchased or why

## Instructions

Assume the other investigation categories (dispute, third-party fraud) do not
apply. Focus solely on evaluating how well "scam" explains the evidence.

1. **Look for the external manipulator**: Scam requires an identifiable
   external party who deceived the CM. Without evidence of deception by a
   third party, this is not a scam.
2. **Check the authorization**: Did the CM authorize the transaction? If
   there is no evidence of CM authorization, this is more likely third-party
   fraud, not scam.
3. **Evaluate social engineering indicators**: Are there patterns in the
   conversation or evidence suggesting manipulation?
4. **Score likelihood**: 0.0 = no evidence of external deception,
   1.0 = strong evidence the CM was manipulated into authorizing.
5. **Cite policies**: Reference the fraud checklist and any applicable
   general guidelines.
6. **Assess eligibility**: Based on the policy checklist and your own judgment:
   - `eligible` (default) — no blocking rules triggered, case can proceed
   - `blocked` — a specific blocking rule applies or evidence directly
     contradicts the scam allegation. Explain why in your reasoning
     and cite the relevant policy.
7. **Identify evidence gaps**: List specific information still needed to
   complete the evaluation (e.g., communication trail with scammer,
   payment method details, timeline of external contact).

If you have a previous assessment, explain what changed since then.

Respond with structured output only.
"""

FRAUD_SPECIALIST_INSTRUCTIONS = f"""\
You are a Third-Party Fraud Specialist evaluating whether a transaction was
made without the cardmember's knowledge or permission by an external criminal.

## Your Category

{_FRAUD_DEFINITION}

## Policy Documents

{_FRAUD_POLICIES}

## Instructions

Assume the other investigation categories (dispute, scam) do not apply. Focus
solely on evaluating how well "third-party fraud" explains the evidence.

1. **Check authorization evidence**: Was the transaction authenticated via
   the CM's enrolled device, chip+PIN, or known credentials? Strong
   authentication from the CM's own device weakens the unauthorized claim.
2. **Look for external compromise**: Are there signs of account takeover,
   unfamiliar devices, unusual locations, card-not-present fraud, or
   credential theft?
3. **Evaluate the CM's claim**: Does the CM claim they didn't make or
   authorize the transaction? Is this claim consistent with the evidence?
4. **Apply policy criteria**: Check the fraud case checklist — are eligibility
   requirements met? Are any blocking rules triggered?
5. **Score likelihood**: 0.0 = evidence clearly shows CM authorized the
   transaction, 1.0 = strong evidence of unauthorized third-party access.
6. **Cite policies**: Reference specific policy passages for your determination.
7. **Assess eligibility**: Based on the policy checklist and your own judgment:
   - `eligible` (default) — no blocking rules triggered, case can proceed
   - `blocked` — a specific blocking rule applies or evidence directly
     contradicts the fraud allegation (e.g., proven CM authorization).
     Explain why in your reasoning and cite the relevant policy.
8. **Identify evidence gaps**: List specific information still needed to
   complete the evaluation (e.g., device fingerprint data, IP/location
   logs, card-present vs. card-not-present status).

If you have a previous assessment, explain what changed since then.

Respond with structured output only.
"""


# ---------------------------------------------------------------------------
# Agent instances
# ---------------------------------------------------------------------------

dispute_specialist = Agent(
    name="dispute_specialist",
    instructions=DISPUTE_SPECIALIST_INSTRUCTIONS,
    output_type=AgentOutputSchema(SpecialistAssessment, strict_json_schema=False),
)

scam_specialist = Agent(
    name="scam_specialist",
    instructions=SCAM_SPECIALIST_INSTRUCTIONS,
    output_type=AgentOutputSchema(SpecialistAssessment, strict_json_schema=False),
)

fraud_specialist = Agent(
    name="fraud_specialist",
    instructions=FRAUD_SPECIALIST_INSTRUCTIONS,
    output_type=AgentOutputSchema(SpecialistAssessment, strict_json_schema=False),
)

_SPECIALISTS = {
    "DISPUTE": dispute_specialist,
    "SCAM": scam_specialist,
    "THIRD_PARTY_FRAUD": fraud_specialist,
}


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------


def _format_specialist_input(
    allegations_summary: str,
    evidence_summary: str,
    conversation_summary: str,
    previous: SpecialistAssessment | None = None,
) -> str:
    """Build the user message for a specialist agent.

    All specialists receive the same shared context. If a previous assessment
    exists for this specialist, it is appended so the specialist can reason
    incrementally.
    """
    parts = [
        f"## Accumulated Allegations\n{allegations_summary}",
        f"## Retrieved Evidence\n{evidence_summary}",
        f"## Conversation Summary\n{conversation_summary}",
    ]

    if previous is not None:
        parts.append(
            f"## Your Previous Assessment\n"
            f"Likelihood: {previous.likelihood:.2f}\n"
            f"Reasoning: {previous.reasoning}\n"
            f"Supporting evidence: {', '.join(previous.supporting_evidence) or 'none'}\n"
            f"Contradicting evidence: {', '.join(previous.contradicting_evidence) or 'none'}"
        )

    return "\n\n".join(parts)


def _default_assessment(category: str, error_msg: str) -> SpecialistAssessment:
    """Create a fallback assessment when a specialist fails."""
    return SpecialistAssessment(
        category=category,
        likelihood=0.0,
        reasoning=f"Specialist unavailable: {error_msg}",
    )


async def _run_single_specialist(
    category: str,
    agent: Agent,
    user_msg: str,
    model_provider: ModelProvider,
) -> SpecialistAssessment:
    """Run a single specialist with error handling."""
    try:
        result = await Runner.run(
            agent,
            input=user_msg,
            run_config=RunConfig(model_provider=model_provider),
        )
        assessment = result.final_output
        # Ensure the category field is set correctly
        assessment.category = category
        return assessment
    except Exception as exc:
        return _default_assessment(category, str(exc))


async def run_specialists(
    allegations_summary: str,
    evidence_summary: str,
    conversation_summary: str,
    model_provider: ModelProvider,
    previous_assessments: dict[str, SpecialistAssessment] | None = None,
) -> dict[str, SpecialistAssessment]:
    """Run all three category specialists in parallel.

    Args:
        allegations_summary: Formatted allegations with types and entities.
        evidence_summary: Retrieved evidence text (transactions, auth events).
        conversation_summary: Running summary of the call so far.
        model_provider: LLM model provider for inference.
        previous_assessments: Previous specialist outputs keyed by category,
            passed to each specialist for incremental reasoning.

    Returns:
        Dict keyed by category name with SpecialistAssessment values.
    """
    if previous_assessments is None:
        previous_assessments = {}

    # Build per-specialist user messages (shared context + own previous output)
    tasks = []
    categories = []
    for category, agent in _SPECIALISTS.items():
        prev = previous_assessments.get(category)
        user_msg = _format_specialist_input(
            allegations_summary, evidence_summary, conversation_summary, prev
        )
        tasks.append(
            _run_single_specialist(category, agent, user_msg, model_provider)
        )
        categories.append(category)

    results = await asyncio.gather(*tasks)
    return dict(zip(categories, results))
