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

from agents import Agent, AgentOutputSchema, ModelProvider
from agents.run_config import RunConfig
from pydantic import BaseModel, Field

from agentic_fraud_servicing.providers.retry import run_with_retry

# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


class SpecialistAssessment(BaseModel):
    """Output from a single category specialist.

    Likelihood and eligibility serve different decision points:
    - ``likelihood`` is grounded in currently available evidence only.
      It answers: "how well does this category explain what we see now?"
    - ``eligibility`` is forward-looking. It answers: "should we open a
      case to investigate further?" Opening enables offline evidence
      collection (merchant records, device forensics, etc.).
    Low likelihood does NOT imply blocked — only contradicting evidence
    or a hard policy rule should trigger blocked.

    Attributes:
        category: The investigation category this specialist evaluates.
        likelihood: How well current evidence supports this category (0.0-1.0).
        reasoning: Policy-grounded explanation (2-4 sentences). When eligibility
            is ``blocked``, must start with "BLOCKED:" and the specific reason.
        supporting_evidence: Evidence items supporting this category.
        contradicting_evidence: Evidence items contradicting this category.
        policy_citations: Specific policy passages cited.
        evidence_gaps: Information still needed. Items suffixed with ``[offline]``
            are only obtainable after case opening.
        eligibility: Whether a case should be opened under this category.
            Default is ``eligible`` — blocked only when evidence actively
            contradicts the hypothesis or a policy rule prevents case opening.
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

_EVIDENCE_AVAILABILITY = """\
## Evidence Availability — Live Call vs. Offline Investigation

**Available during live call** (used for likelihood scoring):
- AMEX transaction records (amounts, merchants, dates, auth methods, channels)
- Authentication logs (device fingerprints, IP addresses, login history)
- Customer profile (account history, contact info, recent account changes)
- Cardmember's verbal statements from the live transcript

**Available only after case opening (offline)** — tag these as `[offline]`
in your evidence_gaps:
- Merchant records (delivery confirmation, refund history, service agreements)
- Merchant communication logs (dispute correspondence, return authorizations)
- Card network dispute data (chargeback responses, representment evidence)
- Detailed device forensics (full device fingerprint analysis, malware scans)
- IP/geolocation deep analysis (VPN detection, proxy analysis, travel patterns)
- Third-party payment platform records (PayPal, Venmo, crypto exchange data)
- Communication trails with alleged scammers (emails, texts, chat logs)
- Law enforcement reports or fraud affidavits
- Credit bureau alerts or identity theft reports

When assessing eligibility, consider whether offline evidence could resolve
the case. When listing evidence_gaps, suffix offline-only items with `[offline]`.
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

{_EVIDENCE_AVAILABILITY}

## Likelihood vs. Eligibility — Two Different Questions

These serve different decision points and must be assessed independently:

- **Likelihood** answers: "Based on currently available evidence, how well does
  merchant dispute explain this case?" Score strictly on what you can see now.
- **Eligibility** answers: "Should we open a dispute case so we can investigate
  further?" This is forward-looking. Opening a case enables offline evidence
  collection (merchant records, delivery confirmation, refund history, service
  agreements) that is impossible to obtain during a live call.

Low likelihood does NOT mean blocked. A case with likelihood 0.2 can still be
`eligible` if key evidence (e.g., merchant response, delivery proof) is
unavailable during the call but obtainable offline. Only block when evidence
actively contradicts the hypothesis or a hard policy rule prevents case opening.

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
4. **Score likelihood** (current evidence only): 0.0 = evidence clearly rules
   out dispute, 1.0 = evidence strongly confirms this is a merchant dispute.
   Score based only on what is available now — do not speculate about what
   offline evidence might show.
5. **Cite policies**: Reference specific policy passages for your determination.
6. **Assess eligibility** (forward-looking): Based on policy rules and whether
   opening a case would allow collecting evidence to resolve the claim:
   - `eligible` (default) — no blocking rules triggered. Use this even when
     likelihood is low, if crucial evidence (merchant records, delivery proof,
     service agreements) is only obtainable through offline investigation after
     case opening.
   - `blocked` — evidence actively contradicts the dispute allegation, OR a
     specific policy rule prevents case opening. Cite the relevant policy.
   **When eligibility is `blocked`, your `reasoning` field MUST start with
   "BLOCKED:" followed by the specific reason (e.g., "BLOCKED: CM confirmed
   receiving goods and acknowledges the charge — policy II ").**
7. **Identify evidence gaps**: List specific information still needed. Flag
   which gaps are obtainable only offline (e.g., "merchant delivery records
   [offline]", "refund policy documentation [offline]").

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

{_EVIDENCE_AVAILABILITY}

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

## Likelihood vs. Eligibility — Two Different Questions

These serve different decision points and must be assessed independently:

- **Likelihood** answers: "Based on currently available evidence, how well does
  scam explain this case?" Score strictly on what you can see now.
- **Eligibility** answers: "Should we open a scam investigation so we can
  investigate further?" This is forward-looking. Opening a case enables offline
  evidence collection (communication trails with the scammer, payment platform
  records, device/IP forensics) that is impossible to obtain during a live call.

Low likelihood does NOT mean blocked. A case with likelihood 0.2 can still be
`eligible` if key evidence (e.g., communication records with the alleged
scammer) is unavailable during the call but obtainable offline. Only block when
evidence actively contradicts the hypothesis or a hard policy rule prevents it.

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
4. **Score likelihood** (current evidence only): 0.0 = no evidence of external
   deception, 1.0 = strong evidence the CM was manipulated into authorizing.
   Score based only on what is available now — do not speculate about what
   offline evidence might show.
5. **Cite policies**: Reference the fraud checklist and any applicable
   general guidelines.
6. **Assess eligibility** (forward-looking): Based on policy rules and whether
   opening a case would allow collecting evidence to resolve the claim:
   - `eligible` (default) — no blocking rules triggered. Use this even when
     likelihood is low, if crucial evidence (scammer communication trail,
     payment platform records, device forensics) is only obtainable through
     offline investigation after case opening.
   - `blocked` — evidence actively contradicts the scam allegation (e.g., CM
     clearly initiated the transaction without any external influence), OR a
     specific policy rule prevents it. Cite the relevant policy.
   **When eligibility is `blocked`, your `reasoning` field MUST start with
   "BLOCKED:" followed by the specific reason (e.g., "BLOCKED: CM describes
   a voluntary purchase with no external influence — fraud_case_checklist §4.1").**
7. **Identify evidence gaps**: List specific information still needed. Flag
   which gaps are obtainable only offline (e.g., "communication trail with
   alleged scammer [offline]", "payment platform transaction records [offline]").

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

{_EVIDENCE_AVAILABILITY}

## Likelihood vs. Eligibility — Two Different Questions

These serve different decision points and must be assessed independently:

- **Likelihood** answers: "Based on currently available evidence, how well does
  third-party fraud explain this case?" Score strictly on what you can see now.
- **Eligibility** answers: "Should we open a fraud case so we can investigate
  further?" This is forward-looking. Opening a case enables offline evidence
  collection (detailed device forensics, IP/geolocation analysis, merchant-side
  authorization records, card network dispute data) that is impossible to obtain
  during a live call.

Low likelihood does NOT mean blocked. A case with likelihood 0.2 can still be
`eligible` if key evidence (e.g., detailed device fingerprint analysis, merchant
authorization records) is unavailable during the call but obtainable offline.
Only block when evidence actively contradicts the hypothesis or a hard policy
rule prevents case opening.

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
5. **Score likelihood** (current evidence only): 0.0 = evidence clearly shows
   CM authorized the transaction, 1.0 = strong evidence of unauthorized
   third-party access. Score based only on what is available now — do not
   speculate about what offline evidence might show.
6. **Cite policies**: Reference specific policy passages for your determination.
7. **Assess eligibility** (forward-looking): Based on policy rules and whether
   opening a case would allow collecting evidence to resolve the claim:
   - `eligible` (default) — no blocking rules triggered. Use this even when
     likelihood is low, if crucial evidence (device forensics, IP/geolocation
     logs, merchant authorization records) is only obtainable through offline
     investigation after case opening.
   - `blocked` — evidence actively contradicts the fraud allegation (e.g.,
     chip+PIN auth from CM's enrolled device with no signs of compromise), OR
     a specific policy rule prevents case opening. Cite the relevant policy.
   **When eligibility is `blocked`, your `reasoning` field MUST start with
   "BLOCKED:" followed by the specific reason (e.g., "BLOCKED: Chip+PIN auth
   from enrolled device with matching behavioral patterns confirms CM
   authorization — fraud_case_checklist §2.1").**
8. **Identify evidence gaps**: List specific information still needed. Flag
   which gaps are obtainable only offline (e.g., "detailed device fingerprint
   analysis [offline]", "merchant-side authorization records [offline]").

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
        result = await run_with_retry(
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
