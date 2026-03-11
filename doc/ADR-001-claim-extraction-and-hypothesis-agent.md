# ADR-001: Granular Claim Extraction and Dedicated Hypothesis Agent

**Status**: Accepted
**Date**: 2026-03-11
**Context**: Copilot triage agent refactoring (L1-14)

---

## Decision

Refactor the copilot's `process_event()` pipeline to:

1. **Separate claim extraction from hypothesis scoring** into two dedicated agents.
2. **Adopt granular ClaimType taxonomy** (17 types: 9 fraud + 8 dispute) with structured entity extraction.
3. **Add a dedicated Hypothesis Agent** that scores the 4 investigation categories using all accumulated context.

---

## Context and Problem

### Problem 1: Triage agent does too much

The original triage agent (`copilot/triage_agent.py`) performed two cognitively
distinct tasks simultaneously:

- **Claim extraction** — "What did the cardmember say?" (factual, per-utterance)
- **Allegation classification** — "What category is this?" (analytical, cross-turn)

These are fundamentally different reasoning tasks. Claim extraction requires
precise attention to the current conversation segment. Category scoring requires
holistic reasoning across all accumulated evidence (claims, auth events,
retrieved facts, contradictions).

### Problem 2: Crude hypothesis scoring formula

The orchestrator used a hardcoded weighted moving average to map the triage's
3-value `AllegationType` to 4 hypothesis scores:

```python
# Old approach: formulaic scoring
scores[detected_key] = scores[detected_key] * 0.4 + confidence * 0.6
# Category shift? Just add 0.15 to first-party fraud
if triage_result.category_shift_detected:
    scores["FIRST_PARTY_FRAUD"] += 0.15
```

This formula:
- Ignores auth assessment results (impersonation risk, device enrollment)
- Ignores retrieved evidence (chip+PIN transactions, delivery proofs)
- Cannot reason about contradiction patterns across multiple claims
- Uses a magic-number heuristic (+0.15) instead of actual reasoning

### Problem 3: Claims lack structured entities

The original `TriageResult.claims` was `list[str]` — plain text strings with no
structured data. This meant:

- No machine-readable entities (amounts, merchants, dates)
- No claim type classification (was it a transaction dispute? location claim?)
- No way to systematically compare claims against evidence facts
- No way to track individual claims through the investigation pipeline

---

## Solution

### New Copilot Pipeline

```
process_event(event):
  1. Triage Agent     -> list[ClaimExtraction]  (granular types + entities)
  2. Auth Agent       -> AuthAssessment         (impersonation risk)
  3. Question Planner -> QuestionPlan           (next-best questions)
  4. Retrieval Agent  -> RetrievalResult        (facts from gateway)
  5. Hypothesis Agent -> HypothesisAssessment   (4 scores + reasoning)
```

### Granular ClaimType Taxonomy

Combined from the fraud and dispute claim type strategies (see references below).

**Fraud Tier 1 (9 types)** — covers 100% of unauthorized-transaction calls:

| ClaimType | Natural Language Match | % of Fraud Calls |
|---|---|---|
| TRANSACTION_DISPUTE | "I didn't make this charge" | 50% |
| CARD_NOT_PRESENT_FRAUD | "I never bought anything online" | 22% |
| ACCOUNT_TAKEOVER | "Someone accessed my account" | 11% |
| LOCATION_CLAIM | "I was in New York, not California" | 6% |
| LOST_STOLEN_CARD | "My card was stolen" | 4% |
| MERCHANT_FRAUD | "This merchant is a scam" | 3% |
| SPENDING_PATTERN | "These charges are unusual for me" | 2% |
| IDENTITY_VERIFICATION | "Verify I am who I say I am" | 1% |
| CARD_POSSESSION | "I have my card right here" | 1% |

**Dispute Tier 1 (8 types)** — covers 95% of authorized-but-problematic calls:

| ClaimType | Natural Language Match | % of Dispute Calls |
|---|---|---|
| GOODS_NOT_RECEIVED | "Package never arrived" | 30% |
| DUPLICATE_CHARGE | "I was charged twice" | 20% |
| RETURN_NOT_CREDITED | "Returned item but no refund" | 15% |
| INCORRECT_AMOUNT | "I was charged $200 but receipt shows $100" | 12% |
| GOODS_NOT_AS_DESCRIBED | "Item is completely different from listing" | 10% |
| RECURRING_AFTER_CANCEL | "Subscription charged after I cancelled" | 8% |
| SERVICES_NOT_RENDERED | "Paid for service that wasn't completed" | 3% |
| DEFECTIVE_MERCHANDISE | "Item arrived broken/damaged" | 2% |

### Structured Entity Extraction

Each claim now carries structured entities — machine-readable name-value pairs
that make claims directly comparable against evidence facts:

```python
class ClaimExtraction(BaseModel):
    claim_type: ClaimType
    claim_description: str      # What the CM claimed (not conclusions)
    entities: dict[str, Any]    # {"amount": 500.00, "merchant": "XYZ Store", "date": "2025-01-15"}
    confidence: float           # 0.0-1.0
    context: str | None = None  # Additional conversation context
```

Entity extraction is critical because:
- `{"amount": 500.00, "merchant": "XYZ Store"}` from a TRANSACTION_DISPUTE
  claim can be directly matched against a Transaction evidence node
- Contradictions become computable: claim says "never heard of TechVault"
  but Device evidence shows enrolled device used at that merchant

### Dedicated Hypothesis Agent

Instead of a formula, a dedicated LLM agent receives ALL accumulated context
and reasons about the 4 hypothesis scores:

**Input to Hypothesis Agent:**
- All claims extracted so far (with types and entities)
- Auth assessment (impersonation risk, step-up recommendation)
- Retrieved evidence (transactions, auth events, customer profile)
- Current hypothesis scores (from previous turns)
- Conversation history summary

**Output (HypothesisAssessment):**
- 4 scores: THIRD_PARTY_FRAUD, FIRST_PARTY_FRAUD, SCAM, DISPUTE
- Per-score reasoning (why this score?)
- Key contradictions detected (claims vs evidence)
- Overall assessment summary

This enables proper first-party fraud detection: the agent can see that
chip+PIN auth from an enrolled device contradicts the CM's "unauthorized
charge" claim, and reason that FIRST_PARTY_FRAUD should be elevated —
instead of relying on `+0.15` heuristics.

---

## Design Decisions

### No claim validation (for now)

The `agentic_ai` project has a `validate_claims()` method that filters by
confidence threshold and requires entities on verifiable claims. We deliberately
skip this for now — the LLM prompt is designed to produce high-quality claims
with entities, and adding validation adds complexity without proven benefit at
this stage. Claim validation can be added as a future enhancement once we have
production data showing quality issues.

### Prompt design follows agentic_ai patterns

The triage prompt follows the proven patterns from the `agentic_ai` project's
`claim_extraction_prompt()`:

- **Inference, not extraction**: The LLM interprets what the CM means, not just
  quotes their words
- **Describe claims, not conclusions**: "CM disputes $500 charge" not "Pattern
  indicates account takeover"
- **Entities are required**: Every claim must have specific details to be
  actionable
- **3-8 claims per turn**: Comprehensive but not redundant

### AllegationType remains for CCP-facing output

`AllegationType` (FRAUD, DISPUTE, SCAM) still exists for CCP-facing summaries.
It represents what the CM claims. `InvestigationCategory` (THIRD_PARTY_FRAUD,
FIRST_PARTY_FRAUD, SCAM, DISPUTE) represents what the system concludes. The
triage agent now outputs `ClaimExtraction` objects (with `ClaimType`), and the
hypothesis agent outputs `InvestigationCategory` scores.

### Latency trade-off

Adding the hypothesis agent means one additional LLM call per event. However:
- The old formula-based scoring was fast but inaccurate
- Accurate hypothesis scoring drives CCP suggested questions and risk flags
- The hypothesis agent has strong signal (all claims + evidence) so it should
  produce high-value output per call

---

## References

- Fraud claim types: `agentic_ai/docs/strategy/FRAUD_CLAIM_TYPES_STRATEGY.md`
- Dispute claim types: `agentic_ai/docs/strategy/DISPUTE_CLAIM_TYPES_STRATEGY.md`
- Claim extraction prompt patterns: `agentic_ai/src/llm/prompts.py`
- Claim extraction implementation: `agentic_ai/src/agent/claim_extractor.py`
- ClaimExtraction model: `agentic_ai/src/core/models.py`
