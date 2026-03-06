"""Merchant evidence specialist agent for normalization and conflict detection.

Normalizes merchant names, identifies conflicts (duplicate charges, amount
mismatches), consolidates related merchant evidence, and assesses merchant-level
risk. Uses OpenAI Agents SDK with structured output via MerchantAnalysis.
"""

import json

from agents import Agent, AgentOutputSchema, ModelProvider, Runner
from agents.run_config import RunConfig
from pydantic import BaseModel, Field

# System prompt for the merchant evidence agent
MERCHANT_INSTRUCTIONS = """\
You are a merchant evidence specialist for card dispute servicing. Your role is to
normalize merchant data, detect conflicts, and assess merchant-level risk.

Your tasks:

1. **Normalize merchant names**: Clean up raw merchant descriptors into readable
   names. Common patterns:
   - 'AMZN*Marketplace' -> 'Amazon Marketplace'
   - 'SQ *COFFEE SHOP' -> 'Square - Coffee Shop'
   - 'PAYPAL *SELLER' -> 'PayPal - Seller'
   - 'TST* Restaurant Name' -> 'Toast - Restaurant Name'
   - Remove trailing reference numbers and extra whitespace.

2. **Identify merchant category and assess category-level risk**: Classify each
   merchant into a category (retail, dining, travel, digital goods, gambling,
   cryptocurrency, etc.). Higher-risk categories include:
   - Gambling / cryptocurrency exchanges (high risk)
   - Digital goods / online services (medium-high risk)
   - International merchants in high-fraud regions (medium-high risk)
   - Recurring subscription services (medium risk)
   - In-person retail with chip+PIN (low risk)

3. **Check for conflicts**: Identify discrepancies in merchant evidence:
   - Same merchant with different transaction amounts on same date (duplicate charge)
   - Merchant address mismatch across transactions
   - Multiple refunds exceeding original purchase amount
   - Transaction at a merchant after reported closure date
   - Merchant category code mismatch with actual goods/services

4. **Consolidate related merchant evidence**: Group transactions and evidence
   related to the same merchant:
   - Aggregate total spend at each merchant
   - Identify transaction patterns (frequency, amounts, timing)
   - Note any prior disputes involving the same merchant

5. **Assess merchant dispute history impact**: Consider the merchant's dispute
   history when assessing risk:
   - High dispute count suggests merchant-side issues
   - Pattern of similar disputes indicates systemic problems
   - Merchant cooperation history affects resolution likelihood

6. **Compute merchant risk score**: Produce a risk score from 0.0 (minimal risk)
   to 1.0 (maximum risk) based on: dispute history, category risk, detected
   conflicts, transaction pattern anomalies, and merchant cooperation record.

Respond with structured output only. Be precise about conflict descriptions and
evidence references.
"""


class MerchantAnalysis(BaseModel):
    """Structured output from the merchant evidence agent.

    Attributes:
        normalized_merchants: Merchant entries with normalized names and risk info.
        conflicts: Detected conflicts with severity and evidence references.
        consolidated_summary: Summary of merchant evidence findings.
        merchant_risk_score: Overall merchant risk score from 0.0 to 1.0.
        recommendations: Recommended follow-up actions for merchant evidence.
    """

    normalized_merchants: list[dict] = Field(default_factory=list)
    conflicts: list[dict] = Field(default_factory=list)
    consolidated_summary: str = ""
    merchant_risk_score: float = 0.0
    recommendations: list[str] = Field(default_factory=list)


# Agent instance with structured output
merchant_agent = Agent(
    name="merchant_evidence",
    instructions=MERCHANT_INSTRUCTIONS,
    output_type=AgentOutputSchema(MerchantAnalysis, strict_json_schema=False),
)


async def run_merchant_analysis(
    merchant_evidence: list[dict],
    transaction_evidence: list[dict],
    model_provider: ModelProvider,
) -> MerchantAnalysis:
    """Run the merchant evidence agent to normalize and analyze merchant data.

    Args:
        merchant_evidence: Merchant-type evidence node dicts from the evidence store.
        transaction_evidence: Transaction-type evidence node dicts with amounts,
            dates, and merchant references.
        model_provider: LLM model provider for inference.

    Returns:
        MerchantAnalysis with normalized merchants, conflicts, and risk assessment.

    Raises:
        RuntimeError: If the agent SDK call fails.
    """
    # Serialize evidence to JSON for the user message
    merchant_json = json.dumps(merchant_evidence, indent=2, default=str)
    transaction_json = json.dumps(transaction_evidence, indent=2, default=str)

    user_msg = (
        f"Merchant Evidence Nodes:\n{merchant_json}\n\n"
        f"Related Transaction Evidence:\n{transaction_json}"
    )

    try:
        result = await Runner.run(
            merchant_agent,
            input=user_msg,
            run_config=RunConfig(model_provider=model_provider),
        )
        return result.final_output
    except Exception as exc:
        raise RuntimeError(f"Merchant evidence agent failed: {exc}") from exc
