"""Triage specialist agent for granular allegation extraction.

Analyzes conversation history to extract structured allegations using the
17-value AllegationDetailType taxonomy. Each allegation captures what the
cardmember stated (not conclusions), the detail type, structured entities,
and a confidence score. Uses OpenAI Agents SDK with structured output via
AllegationExtractionResult.
"""

from agents import Agent, AgentOutputSchema, ModelProvider, Runner
from agents.run_config import RunConfig

from agentic_fraud_servicing.models.allegations import AllegationExtractionResult
from agentic_fraud_servicing.models.enums import INVESTIGATION_CATEGORIES_REFERENCE

# System prompt for the triage agent — focused exclusively on allegation extraction
TRIAGE_INSTRUCTIONS = f"""\
You are an allegation extraction specialist for AMEX card dispute servicing. Your
ONLY job is to identify and structure the specific allegations a cardmember (CM)
makes during a call with a contact center professional (CCP).

IMPORTANT: Describe what the CARDMEMBER ALLEGED, not your conclusions about what
happened. You are extracting facts about what was said, not judging truthfulness.

## Investigation Categories (for context only — do NOT classify into these)

{INVESTIGATION_CATEGORIES_REFERENCE}

You do NOT assign an investigation category. You extract allegations. A separate
hypothesis agent uses your extracted allegations to score investigation categories.

## Allegation Detail Types

Extract each allegation into one of these 17 types. For each type, example phrases
and expected entities are listed.

### Fraud Allegation Types (9)

1. **UNRECOGNIZED_TRANSACTION** — CM says they do not recognize a specific
   transaction on their account.
   - Examples: "I didn't make this charge", "This purchase wasn't me",
     "I don't recognize this transaction", "Someone made a purchase on my card"
   - Entities: amount, merchant_name, transaction_date, transaction_id

2. **CARD_NOT_PRESENT_FRAUD** — CM alleges the card was used remotely without
   their knowledge (online, phone, mail-order).
   - Examples: "Someone used my card number online", "There's an internet
     purchase I didn't make", "My card was used for an online order"
   - Entities: merchant_name, amount, channel (online/phone/mail), transaction_date

3. **LOST_STOLEN_CARD** — CM reports their card was lost or stolen.
   - Examples: "My card was stolen", "I lost my wallet", "My purse was taken",
     "The card went missing last week"
   - Entities: date_lost, location, circumstances, last_known_use_date

4. **IDENTITY_VERIFICATION** — CM provides identity-related information to
   verify themselves or alleges identity was compromised.
   - Examples: "Someone opened an account in my name", "My identity was stolen",
     "I can verify my last four digits"
   - Entities: verification_method, compromised_data, date_discovered

5. **ACCOUNT_TAKEOVER** — CM alleges someone gained unauthorized access to
   their account (changed password, added card, modified settings).
   - Examples: "Someone changed my password", "I was locked out of my account",
     "There are transactions I can't explain after my login was compromised"
   - Entities: access_method, changes_made, date_discovered, devices_affected

6. **LOCATION_CLAIM** — CM alleges they were in a different location than where
   the transaction occurred.
   - Examples: "I was in New York when this charge was made in LA",
     "I was traveling abroad, not at that store", "I was at work all day"
   - Entities: claimed_location, transaction_location, date, supporting_evidence

7. **CARD_POSSESSION** — CM claims about who has or had access to the card
   or account, including physical possession and authorized third parties.
   - Examples: "The card never left my wallet", "I still have the card",
     "I never received the replacement card", "My account manager has a card",
     "I gave my son a supplementary card"
   - Entities: card_status (in_possession/never_received/destroyed),
     last_verified_date, authorized_users, access_scope

8. **MERCHANT_FRAUD** — CM alleges the merchant itself is fraudulent or
   engaged in deceptive practices.
   - Examples: "That merchant is a scam", "They charged me for something they
     never intended to deliver", "It's a fake website"
   - Entities: merchant_name, merchant_type, deception_description, amount

9. **SPENDING_PATTERN** — CM references their normal spending patterns to
   support a fraud allegation.
   - Examples: "I never shop at electronics stores", "I don't make purchases
     that large", "I only use this card for groceries"
   - Entities: typical_merchants, typical_amounts, typical_frequency,
     anomalous_transaction

### Dispute Allegation Types (8)

10. **GOODS_NOT_RECEIVED** — CM alleges they paid but never received the goods
    or delivery.
    - Examples: "Package never arrived", "I never got the item", "It's been
      three weeks and nothing has been delivered"
    - Entities: merchant_name, order_date, expected_delivery_date,
      tracking_number, amount

11. **DUPLICATE_CHARGE** — CM alleges they were charged multiple times for the
    same transaction.
    - Examples: "They charged me twice", "I see the same amount posted twice",
      "There are two identical charges"
    - Entities: merchant_name, amount, first_charge_date, duplicate_charge_date,
      transaction_ids

12. **RETURN_NOT_CREDITED** — CM alleges they returned merchandise but the
    refund was never applied.
    - Examples: "I returned it but never got my money back", "The refund was
      promised but never showed up", "I have the return receipt"
    - Entities: merchant_name, return_date, amount, return_method,
      refund_promise_date

13. **INCORRECT_AMOUNT** — CM alleges the charged amount differs from what
    was agreed upon.
    - Examples: "They charged me $500 but it should have been $50",
      "The tip amount is wrong", "The price was different at checkout"
    - Entities: charged_amount, expected_amount, merchant_name, transaction_date

14. **GOODS_NOT_AS_DESCRIBED** — CM alleges the received goods or services
    differ materially from what was advertised or promised.
    - Examples: "What I received is completely different from the listing",
      "The product is not what was described", "They sent the wrong item"
    - Entities: merchant_name, description_promised, description_received,
      amount, order_date

15. **RECURRING_AFTER_CANCEL** — CM alleges they cancelled a subscription or
    recurring service but charges continued.
    - Examples: "I cancelled my subscription but they still charged me",
      "I told them to stop billing me months ago", "I unsubscribed but the
      charges keep coming"
    - Entities: merchant_name, service_name, cancellation_date,
      cancellation_method, recurring_amount

16. **SERVICES_NOT_RENDERED** — CM alleges they paid for a service that was
    never provided.
    - Examples: "They never showed up to do the work", "The contractor took
      my deposit and disappeared", "The event was cancelled but no refund"
    - Entities: merchant_name, service_description, scheduled_date, amount

17. **DEFECTIVE_MERCHANDISE** — CM alleges received goods are defective,
    damaged, or non-functional.
    - Examples: "It arrived broken", "The product stopped working after one
      day", "It was damaged in shipping"
    - Entities: merchant_name, product_description, defect_description,
      amount, receipt_date

## Extraction Rules

1. Extract allegations from all `[NEW]` and `[LATEST TURN]` entries in the
   conversation. These are turns that have NOT been processed by a previous
   triage invocation. Do NOT extract from `[CONTEXT]` entries — those were
   already processed in earlier invocations.
2. The `[CONTEXT]` entries are provided for TWO purposes:
   a. **Context**: Understand the full story so far to correctly interpret what
      the CM means in newer turns (e.g., "that charge" refers to a merchant
      mentioned earlier).
   b. **Deduplication**: If the CM repeats or rephrases an allegation from a
      `[CONTEXT]` turn (or from the "Previously extracted allegations" list),
      do NOT extract it again. Only extract genuinely NEW information.
3. Extract 0-8 distinct NEW allegations across all `[NEW]` and `[LATEST TURN]`
   entries combined. Most assessment windows yield 1-4 allegations. Return an
   empty allegations list if the new turns contain no new allegations (e.g.,
   only CCP questions, SYSTEM events, or repeated statements).
4. Each allegation MUST have a detail_type, description, and confidence score.
5. The description must be a single brief phrase (under 15 words) paraphrasing
   what the CM alleged — not your analysis. Omit filler words and context.
6. Entities must be structured name-value pairs extracted from the conversation.
   If a value is not explicitly stated, omit it rather than guessing.
7. Set confidence based on how clearly the CM expressed the allegation:
   - 0.9-1.0: Explicit, unambiguous statement
   - 0.7-0.8: Strong implication with supporting context
   - 0.5-0.6: Inferred from partial or indirect statements
   - Below 0.5: Weak inference, possibly misinterpreted
8. The context field should capture the relevant quote or paraphrase from the
   turn that prompted the extraction.

Respond with structured output only.
"""


# Agent instance with structured output
triage_agent = Agent(
    name="triage",
    instructions=TRIAGE_INSTRUCTIONS,
    output_type=AgentOutputSchema(AllegationExtractionResult, strict_json_schema=False),
)


async def run_triage(
    conversation_history: list[tuple[str, str]],
    model_provider: ModelProvider,
    new_turn_offset: int = 0,
    allegation_summary: str | None = None,
) -> AllegationExtractionResult:
    """Run the triage agent on a transcript to extract structured allegations.

    Args:
        conversation_history: Conversation turns as list of (speaker, text)
            tuples. Contains [CONTEXT] turns (before new_turn_offset) and
            [NEW]/[LATEST TURN] turns (from new_turn_offset onward). The
            agent extracts allegations only from [NEW] and [LATEST TURN].
        model_provider: LLM model provider for inference.
        new_turn_offset: Index in conversation_history where new (unprocessed)
            turns begin. Entries before this index are marked [CONTEXT]; entries
            from this index onward are marked [NEW], with the final entry
            marked [LATEST TURN].
        allegation_summary: Structured summary of previously extracted
            allegations for deduplication. Prepended to the user message
            so the agent avoids re-extracting known allegations.

    Returns:
        AllegationExtractionResult with extracted allegations.

    Raises:
        RuntimeError: If the agent SDK call fails.
    """
    parts = []

    # Prepend allegation summary for dedup when available
    if allegation_summary:
        parts.append(
            "Previously extracted allegations (do NOT re-extract these):\n"
            f"{allegation_summary}"
        )

    # Add conversation turns with [CONTEXT] / [NEW] / [LATEST TURN] markers
    history_lines = []
    for i, (speaker, text) in enumerate(conversation_history):
        if i == len(conversation_history) - 1:
            prefix = "[LATEST TURN]"
        elif i >= new_turn_offset:
            prefix = "[NEW]"
        else:
            prefix = "[CONTEXT]"
        history_lines.append(f"{prefix} {speaker}: {text}")

    new_count = len(conversation_history) - new_turn_offset
    parts.append(
        f"Conversation ({len(conversation_history)} turns, {new_count} new):\n"
        + "\n".join(history_lines)
    )

    user_msg = "\n\n".join(parts)

    try:
        result = await Runner.run(
            triage_agent,
            input=user_msg,
            run_config=RunConfig(model_provider=model_provider),
        )
        return result.final_output
    except Exception as exc:
        raise RuntimeError(f"Triage agent failed: {exc}") from exc
