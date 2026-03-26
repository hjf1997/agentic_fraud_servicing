"""Extract structured Transaction objects from a text description using ConnectChain.

Standalone utility that takes a free-text description of transactions (e.g. from
a case HTML page) and returns a list of domain Transaction models via LLM
structured output.

Usage:
    from scripts.extract_transactions import extract_transactions

    transactions = extract_transactions(
        description="Customer disputes a $487.50 online purchase at TechVault ...",
        case_id="case-eval-techvault-001",
    )

Requires ConnectChain environment variables (CONFIG_PATH, WORKDIR, etc.) and
CONNECTCHAIN_MODEL_INDEX to be set.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from agentic_fraud_servicing.models.enums import (
    AuthMethod,
    EvidenceSourceType,
    TransactionChannel,
    TransactionOutcome,
)
from agentic_fraud_servicing.models.evidence import Transaction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schema for LLM structured output
# ---------------------------------------------------------------------------


class ExtractedTransaction(BaseModel):
    """Schema the LLM fills in for each transaction found in the text."""

    amount: float
    currency: str = "USD"
    merchant_name: str
    merchant_id: str | None = None
    transaction_date: datetime
    auth_method: AuthMethod | None = None
    channel: TransactionChannel | None = None
    outcome: TransactionOutcome | None = None


class ExtractedTransactions(BaseModel):
    """Wrapper so the LLM returns a list."""

    transactions: list[ExtractedTransaction] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# System prompt for extraction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a data extraction assistant for American Express fraud investigations.

Given a free-text description of one or more financial transactions, extract
every transaction into the structured schema provided. Follow these rules:

1. Extract ONLY what is explicitly stated or clearly implied in the text.
   Do NOT invent amounts, dates, merchants, or other details.
2. If a field cannot be determined from the text, leave it as null.
3. Date/time values must be ISO-8601 format (e.g. "2026-03-15T14:30:00Z").
   If only a date is given, use midnight UTC.
4. Auth method mapping:
   - "chip and PIN", "chip+PIN", "EMV" -> CHIP
   - "swipe", "magnetic stripe" -> SWIPE
   - "tap", "contactless", "NFC" -> CONTACTLESS
   - "online", "e-commerce", "card not present", "CNP" -> CNP
   - "manual entry", "key entered" -> MANUAL
5. Channel mapping:
   - "point of sale", "in-store", "retail" -> POS
   - "online", "web", "e-commerce" -> ONLINE
   - "ATM" -> ATM
   - "phone order", "MOTO" -> PHONE
6. Outcome mapping:
   - "approved", "authorized", "settled", "posted" -> APPROVED
   - "denied", "declined", "rejected" -> DENIED
   - "chargeback", "reversed", "charged back", "credit issued" -> CHARGEBACK
   - "pending", "processing", "in progress", "held" -> PENDING
"""

# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


def _call_with_retry(fn, max_attempts=5, base_delay=2.0, max_delay=60.0):
    """Call *fn* with exponential backoff on transient errors."""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            error_str = str(e).lower()
            retryable = any(
                k in error_str
                for k in ["timeout", "rate limit", "503", "502", "500", "429"]
            )
            if not retryable:
                raise
            last_error = e
            if attempt < max_attempts:
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                delay += random.uniform(0, delay * 0.1)
                logger.warning(
                    "Attempt %d/%d failed (%s), retrying in %.1fs",
                    attempt,
                    max_attempts,
                    e,
                    delay,
                )
                time.sleep(delay)
    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def _get_structured_llm(model_index: str | None = None):
    """Return a ConnectChain LLM bound to the ExtractedTransactions schema."""
    from connectchain.lcel.model import model as get_model

    index = model_index or _resolve_model_index()
    llm = get_model(index)
    return llm.with_structured_output(ExtractedTransactions, method="function_calling")


def _resolve_model_index() -> str:
    """Read CONNECTCHAIN_MODEL_INDEX from environment."""
    import os

    idx = os.environ.get("CONNECTCHAIN_MODEL_INDEX")
    if not idx:
        raise RuntimeError(
            "CONNECTCHAIN_MODEL_INDEX environment variable is required"
        )
    return idx


def extract_transactions(
    description: str,
    case_id: str,
    *,
    scenario_prefix: str = "eval",
    model_index: str | None = None,
) -> list[Transaction]:
    """Extract Transaction evidence nodes from a free-text description.

    Args:
        description: Free-text description of transactions (e.g. from HTML).
        case_id: Case ID to assign to each Transaction node.
        scenario_prefix: Prefix for node IDs (default "eval").
        model_index: ConnectChain model index override.

    Returns:
        List of Transaction domain models ready to be seeded into the evidence store.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    structured_llm = _get_structured_llm(model_index)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=description),
    ]

    def _invoke():
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(structured_llm.ainvoke(messages))

    result: ExtractedTransactions = _call_with_retry(_invoke)

    now = datetime.now(tz=timezone.utc)
    transactions = []
    for i, ext in enumerate(result.transactions, start=1):
        txn = Transaction(
            node_id=f"txn-{scenario_prefix}-{i:03d}",
            case_id=case_id,
            source_type=EvidenceSourceType.FACT,
            created_at=now,
            amount=ext.amount,
            currency=ext.currency,
            merchant_name=ext.merchant_name,
            merchant_id=ext.merchant_id,
            transaction_date=ext.transaction_date,
            auth_method=ext.auth_method,
            channel=ext.channel,
            outcome=ext.outcome,
        )
        transactions.append(txn)

    logger.info("Extracted %d transaction(s) from description", len(transactions))
    return transactions


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Extract structured transactions from a text description."
    )
    parser.add_argument(
        "description",
        nargs="?",
        help="Transaction description text. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "--case-id",
        default="case-eval-001",
        help="Case ID for extracted transactions (default: case-eval-001)",
    )
    parser.add_argument(
        "--prefix",
        default="eval",
        help="Scenario prefix for node IDs (default: eval)",
    )
    parser.add_argument(
        "--model-index",
        default=None,
        help="ConnectChain model index override",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    text = args.description or sys.stdin.read()
    if not text.strip():
        parser.error("No description provided")

    txns = extract_transactions(
        text,
        case_id=args.case_id,
        scenario_prefix=args.prefix,
        model_index=args.model_index,
    )

    output = [json.loads(t.model_dump_json()) for t in txns]
    print(json.dumps(output, indent=2, default=str))
