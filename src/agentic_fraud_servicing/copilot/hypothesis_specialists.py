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
    """Output from a single category specialist (evidence analyst).

    The specialist classifies evidence as supporting, contradicting, or
    missing, and assesses case-opening eligibility. It does NOT produce a
    likelihood score — scoring is handled by the logprob-based logit scorer
    at the arbitrator level.

    Attributes:
        category: The investigation category this specialist evaluates.
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
    reasoning: str = ""
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    policy_citations: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    eligibility: Literal["eligible", "blocked"] = "eligible"


class SpecialistNoteUpdate(BaseModel):
    """Incremental update to a specialist's working notes.

    On subsequent turns, the specialist outputs only what changed.
    The host merges this into the previous SpecialistAssessment
    deterministically via merge_specialist_notes().

    Regenerated fields (reasoning, policy_citations, eligibility) are
    provided in full each turn. Evidence lists are updated via explicit
    add/remove operations — items only disappear when explicitly removed.
    """

    category: str

    # Regenerated each turn (full replacement)
    reasoning: str = ""
    policy_citations: list[str] = Field(default_factory=list)
    eligibility: Literal["eligible", "blocked"] = "eligible"

    # Incremental updates to evidence lists
    add_supporting_evidence: list[str] = Field(default_factory=list)
    remove_supporting_evidence: list[str] = Field(default_factory=list)
    add_contradicting_evidence: list[str] = Field(default_factory=list)
    remove_contradicting_evidence: list[str] = Field(default_factory=list)
    add_evidence_gaps: list[str] = Field(default_factory=list)
    remove_evidence_gaps: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------


def _remove_by_substring(items: list[str], removals: list[str]) -> list[str]:
    """Remove items containing any of the removal substrings (case-insensitive).

    An item is removed if any removal phrase appears as a substring within it.
    Unmatched removal phrases are silently ignored.
    """
    if not removals:
        return list(items)
    result = []
    for item in items:
        item_lower = item.lower()
        if not any(r.lower() in item_lower for r in removals):
            result.append(item)
    return result


def _add_deduped(existing: list[str], additions: list[str]) -> list[str]:
    """Add items not already present (substring dedup, case-insensitive).

    An addition is skipped if any existing item contains it or it contains
    any existing item.
    """
    result = list(existing)
    for add in additions:
        add_lower = add.lower()
        already = any(add_lower in ex.lower() or ex.lower() in add_lower for ex in result)
        if not already:
            result.append(add)
    return result


def merge_specialist_notes(
    previous: SpecialistAssessment,
    update: SpecialistNoteUpdate,
) -> SpecialistAssessment:
    """Merge an incremental update into a previous specialist assessment.

    Regenerated fields (reasoning, policy_citations, eligibility) are replaced
    wholesale. Evidence lists are patched: removals applied first, then additions.
    """
    supporting = _remove_by_substring(
        previous.supporting_evidence, update.remove_supporting_evidence
    )
    supporting = _add_deduped(supporting, update.add_supporting_evidence)

    contradicting = _remove_by_substring(
        previous.contradicting_evidence, update.remove_contradicting_evidence
    )
    contradicting = _add_deduped(contradicting, update.add_contradicting_evidence)

    gaps = _remove_by_substring(previous.evidence_gaps, update.remove_evidence_gaps)
    gaps = _add_deduped(gaps, update.add_evidence_gaps)

    return SpecialistAssessment(
        category=previous.category,
        reasoning=update.reasoning,
        supporting_evidence=supporting,
        contradicting_evidence=contradicting,
        policy_citations=update.policy_citations,
        evidence_gaps=gaps,
        eligibility=update.eligibility,
    )


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


def load_specialist_policies(specialist: str, policies_dir: str | Path | None = None) -> str:
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

_WORKING_NOTES_MODE = """\
## Working Notes Update Mode

Your output format depends on whether you have existing working notes.

**First assessment (no "Your Working Notes" section in input):**
Output a full SpecialistAssessment with all fields populated from scratch.

**Subsequent assessments (working notes shown in input):**
Output a SpecialistNoteUpdate with only what changed. The host will merge
your update into the previous state — you do NOT need to repeat unchanged
evidence items.

### Fields you regenerate in full each turn:
- `reasoning` — rewrite based on current evidence and what changed
- `policy_citations` — provide the full list of relevant citations
- `eligibility` — reassess based on current state

### Fields you update incrementally (add/remove only):
- `add_supporting_evidence` — genuinely NEW supporting items not already in
  your working notes. Do NOT re-add items already listed.
- `remove_supporting_evidence` — items to remove (use a substring that
  uniquely identifies the item). Remove items that are wrong, superseded,
  or no longer supported by evidence.
- `add_contradicting_evidence` / `remove_contradicting_evidence` — same rules
- `add_evidence_gaps` / `remove_evidence_gaps` — same rules. Remove gaps
  that have been filled by newly retrieved evidence.

### Rules:
1. **Items persist unless explicitly removed.** If an evidence item from your
   working notes is still valid, do nothing — it stays automatically.
2. **Do not re-add existing items.** Check your working notes before adding.
   If an item is already there (even worded slightly differently), skip it.
3. **If nothing changed in a list, leave both add and remove empty.**
4. **Removal uses substring matching.** You only need enough of the item text
   to uniquely identify it (e.g., "chip+PIN" to remove an item about
   chip+PIN authentication).
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

## Eligibility Assessment

Eligibility answers: "Should we open a dispute case so we can investigate
further?" This is forward-looking. Opening a case enables offline evidence
collection (merchant records, delivery confirmation, refund history, service
agreements) that is impossible to obtain during a live call.

Only block when evidence actively contradicts the hypothesis or a hard policy
rule prevents case opening.

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
4. **Cite policies**: Reference specific policy passages for your determination.
5. **Assess eligibility** (forward-looking): Based on policy rules and whether
   opening a case would allow collecting evidence to resolve the claim:
   - `eligible` (default) — no blocking rules triggered. Use this even when
     crucial evidence (merchant records, delivery proof, service agreements)
     is only obtainable through offline investigation after case opening.
   - `blocked` — evidence actively contradicts the dispute allegation, OR a
     specific policy rule prevents case opening. Cite the relevant policy.
   **When eligibility is `blocked`, your `reasoning` field MUST start with
   "BLOCKED:" followed by the specific reason and the relevant policy
   section header.**
6. **Identify evidence gaps**: List specific information still needed. Flag
   which gaps are obtainable only offline (e.g., "merchant delivery records
   [offline]", "refund policy documentation [offline]").

{_WORKING_NOTES_MODE}

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

## Instructions

Assume the other investigation categories (dispute, third-party fraud) do not
apply. Focus solely on evaluating how well "scam" explains the evidence.

Note: Scam does not have a separate case type — AMEX does not bear the cost
when the CM authorized the transaction. Your role is purely evidence analysis
for hypothesis scoring, not case opening eligibility. Always leave
`eligibility` as `"eligible"` (the default).

1. **Look for the external manipulator**: Scam requires an identifiable
   external party who deceived the CM. Without evidence of deception by a
   third party, this is not a scam.
2. **Check the authorization**: Did the CM authorize the transaction? If
   there is no evidence of CM authorization, this is more likely third-party
   fraud, not scam.
3. **Evaluate social engineering indicators**: Are there patterns in the
   conversation or evidence suggesting manipulation?
4. **Cite policies**: Reference the fraud checklist and any applicable
   general guidelines.
5. **Identify evidence gaps**: List specific information still needed. Flag
   which gaps are obtainable only offline (e.g., "communication trail with
   alleged scammer [offline]", "payment platform transaction records [offline]").

{_WORKING_NOTES_MODE}

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

## Eligibility Assessment

Eligibility answers: "Should we open a fraud case so we can investigate
further?" This is forward-looking. Opening a case enables offline evidence
collection (detailed device forensics, IP/geolocation analysis, merchant-side
authorization records, card network dispute data) that is impossible to obtain
during a live call.

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
5. **Cite policies**: Reference specific policy passages for your determination.
6. **Assess eligibility** (forward-looking): Based on policy rules and whether
   opening a case would allow collecting evidence to resolve the claim:
   - `eligible` (default) — no blocking rules triggered. Use this even when
     crucial evidence (device forensics, IP/geolocation logs, merchant
     authorization records) is only obtainable through offline investigation
     after case opening.
   - `blocked` — evidence actively contradicts the fraud allegation (e.g.,
     chip+PIN auth from CM's enrolled device with no signs of compromise), OR
     a specific policy rule prevents case opening. Cite the relevant policy.
   **When eligibility is `blocked`, your `reasoning` field MUST start with
   "BLOCKED:" followed by the specific reason and the relevant policy
   section header.**
7. **Identify evidence gaps**: List specific information still needed. Flag
   which gaps are obtainable only offline (e.g., "detailed device fingerprint
   analysis [offline]", "merchant-side authorization records [offline]").

{_WORKING_NOTES_MODE}

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

_SPECIALIST_INSTRUCTIONS = {
    "DISPUTE": DISPUTE_SPECIALIST_INSTRUCTIONS,
    "SCAM": SCAM_SPECIALIST_INSTRUCTIONS,
    "THIRD_PARTY_FRAUD": FRAUD_SPECIALIST_INSTRUCTIONS,
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

    All specialists receive the same shared context. When previous working
    notes exist, they are shown with full evidence lists so the LLM can
    reason about what to add or remove.
    """
    parts = [
        f"## Accumulated Allegations\n{allegations_summary}",
        f"## Retrieved Evidence\n{evidence_summary}",
        f"## Conversation Summary\n{conversation_summary}",
    ]

    if previous is not None:
        supporting = "\n".join(f"  - {e}" for e in previous.supporting_evidence) or "  (none)"
        contradicting = (
            "\n".join(f"  - {e}" for e in previous.contradicting_evidence) or "  (none)"
        )
        gaps = "\n".join(f"  - {e}" for e in previous.evidence_gaps) or "  (none)"
        citations = "\n".join(f"  - {c}" for c in previous.policy_citations) or "  (none)"
        parts.append(
            f"## Your Working Notes (current state — output updates only)\n"
            f"Eligibility: {previous.eligibility}\n"
            f"Reasoning: {previous.reasoning}\n\n"
            f"Supporting evidence:\n{supporting}\n\n"
            f"Contradicting evidence:\n{contradicting}\n\n"
            f"Evidence gaps:\n{gaps}\n\n"
            f"Policy citations:\n{citations}"
        )

    return "\n\n".join(parts)


def _default_assessment(category: str, error_msg: str) -> SpecialistAssessment:
    """Create a fallback assessment when a specialist fails."""
    return SpecialistAssessment(
        category=category,
        reasoning=f"Specialist unavailable: {error_msg}",
    )


async def _run_single_specialist(
    category: str,
    instructions: str,
    user_msg: str,
    model_provider: ModelProvider,
    previous: SpecialistAssessment | None = None,
) -> tuple[SpecialistAssessment, SpecialistNoteUpdate | None]:
    """Run a single specialist with error handling.

    On the first turn (no previous), the agent outputs a full
    SpecialistAssessment. On subsequent turns, it outputs a
    SpecialistNoteUpdate which is merged into the previous state.

    Returns:
        Tuple of (merged assessment, raw delta). Delta is None on the
        first turn or when the specialist fails.
    """
    is_update = previous is not None
    output_type = SpecialistNoteUpdate if is_update else SpecialistAssessment

    agent = Agent(
        name=f"{category.lower()}_specialist",
        instructions=instructions,
        output_type=AgentOutputSchema(output_type, strict_json_schema=False),
    )

    try:
        result = await run_with_retry(
            agent,
            input=user_msg,
            run_config=RunConfig(model_provider=model_provider),
        )
        output = result.final_output

        if is_update:
            output.category = category
            merged = merge_specialist_notes(previous, output)
            return merged, output
        else:
            output.category = category
            return output, None
    except Exception as exc:
        if previous is not None:
            return previous, None
        return _default_assessment(category, str(exc)), None


async def run_specialists(
    allegations_summary: str,
    evidence_summary: str,
    conversation_summary: str,
    model_provider: ModelProvider,
    previous_assessments: dict[str, SpecialistAssessment] | None = None,
) -> tuple[dict[str, SpecialistAssessment], dict[str, SpecialistNoteUpdate]]:
    """Run all three category specialists in parallel.

    Args:
        allegations_summary: Formatted allegations with types and entities.
        evidence_summary: Retrieved evidence text (transactions, auth events).
        conversation_summary: Running summary of the call so far.
        model_provider: LLM model provider for inference.
        previous_assessments: Previous specialist outputs keyed by category,
            passed to each specialist for incremental reasoning.

    Returns:
        Tuple of (merged assessments, raw deltas). Deltas dict is empty on
        the first turn or for specialists that failed.
    """
    if previous_assessments is None:
        previous_assessments = {}

    tasks = []
    categories = []
    for category, instructions in _SPECIALIST_INSTRUCTIONS.items():
        prev = previous_assessments.get(category)
        user_msg = _format_specialist_input(
            allegations_summary, evidence_summary, conversation_summary, prev
        )
        tasks.append(
            _run_single_specialist(category, instructions, user_msg, model_provider, prev)
        )
        categories.append(category)

    results = await asyncio.gather(*tasks)

    assessments = {}
    deltas = {}
    for category, (assessment, delta) in zip(categories, results):
        assessments[category] = assessment
        if delta is not None:
            deltas[category] = delta

    return assessments, deltas
