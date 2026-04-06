"""Pure-Python transaction summarizer for LLM consumption.

Transforms raw transaction dicts into compact, readable text summaries
that preserve investigation-relevant detail while minimizing token usage.
Uses a tiered approach: small sets get full individual detail, large sets
get per-merchant aggregates with representative samples.
"""

from collections import Counter, defaultdict
from datetime import datetime


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def summarize_transactions(
    transactions: list[dict],
    *,
    group_threshold: int = 30,
) -> tuple[list[dict], str]:
    """Summarize transactions into disputed dicts and a formatted text summary.

    Args:
        transactions: Raw transaction dicts from the evidence store (after
            field stripping). Each must have at least ``is_disputed`` and
            ``amount``; other fields are used when present.
        group_threshold: When disputed transaction count exceeds this value,
            switch from individual detail to per-merchant grouped format.

    Returns:
        Tuple of (disputed_dicts, summary_text):
        - disputed_dicts: The original disputed transaction dicts (unmodified)
          for structured use by the retrieval agent.
        - summary_text: Human-readable text covering both disputed and
          undisputed transactions.
    """
    disputed = [t for t in transactions if t.get("is_disputed")]
    undisputed = [t for t in transactions if not t.get("is_disputed")]

    disputed_merchants = {t.get("merchant_name", "") for t in disputed}

    parts: list[str] = []

    # Disputed section
    if disputed:
        if len(disputed) <= group_threshold:
            parts.append(_format_disputed_individual(disputed))
        else:
            parts.append(_format_disputed_grouped(disputed))
    else:
        parts.append("== Disputed Transactions (0) ==\nNo disputed transactions found.")

    # Undisputed section
    if undisputed:
        parts.append(_format_undisputed(undisputed, disputed_merchants))

    return disputed, "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Disputed formatting — individual detail (≤ threshold)
# ---------------------------------------------------------------------------


def _format_disputed_individual(disputed: list[dict]) -> str:
    """Format disputed transactions with full individual detail, grouped by date.

    Used when the disputed count is within the group_threshold.
    """
    total_amount = sum(t.get("amount", 0) for t in disputed)
    header = f"== Disputed Transactions ({len(disputed)} total, ${total_amount:,.2f}) =="

    # Sort by transaction_date descending (most recent first)
    sorted_txns = sorted(disputed, key=_txn_date_key, reverse=True)

    # Group by date string
    by_date: dict[str, list[dict]] = defaultdict(list)
    for t in sorted_txns:
        date_str = _extract_date_str(t)
        by_date[date_str].append(t)

    lines = [header]
    for date_str, txns in by_date.items():
        lines.append(date_str + ":")
        for t in txns:
            lines.append("- " + _format_single_txn(t))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Disputed formatting — grouped by merchant (> threshold)
# ---------------------------------------------------------------------------


def _format_disputed_grouped(disputed: list[dict]) -> str:
    """Format disputed transactions as per-merchant aggregates with samples.

    Used when the disputed count exceeds the group_threshold. Each merchant
    gets aggregate stats plus a sample of transactions spread across the
    date range (earliest, middle, latest).
    """
    total_amount = sum(t.get("amount", 0) for t in disputed)
    header = f"== Disputed Transactions ({len(disputed)} total, ${total_amount:,.2f}) =="

    # Group by merchant
    by_merchant: dict[str, list[dict]] = defaultdict(list)
    for t in disputed:
        merchant = t.get("merchant_name", "Unknown merchant")
        by_merchant[merchant].append(t)

    lines = [header]
    for merchant, txns in by_merchant.items():
        amounts = [t.get("amount", 0) for t in txns]
        merchant_total = sum(amounts)
        dates = [_txn_date_key(t) for t in txns]
        min_date = _extract_date_str(min(txns, key=_txn_date_key))
        max_date = _extract_date_str(max(txns, key=_txn_date_key))

        lines.append(
            f"{merchant} ({len(txns)} txns, ${merchant_total:,.2f}, "
            f"{min_date} to {max_date}):"
        )

        # Auth method distribution
        auth_counts = Counter(t.get("auth_method", "Unknown") for t in txns)
        auth_parts = [f"{method} ({count})" for method, count in auth_counts.most_common()]
        lines.append(f"  Auth: {', '.join(auth_parts)}")

        # Outcome distribution
        outcome_counts = Counter(t.get("outcome", "Unknown") for t in txns)
        outcome_parts = [f"{out} ({count})" for out, count in outcome_counts.most_common()]
        lines.append(f"  Outcome: {', '.join(outcome_parts)}")

        # Amount range
        lines.append(
            f"  Amounts: ${min(amounts):,.2f} - ${max(amounts):,.2f}, "
            f"avg ${merchant_total / len(txns):,.2f}"
        )

        # Sample transactions: pick ~3 dates spread across the range
        lines.append("  Sample transactions:")
        samples = _pick_date_samples(txns)
        for date_str, sample_txns in samples:
            amount_strs = [f"${t.get('amount', 0):,.2f}" for t in sample_txns]
            lines.append(f"  - {date_str}: {', '.join(amount_strs)}")

    return "\n".join(lines)


def _pick_date_samples(
    txns: list[dict], num_dates: int = 3
) -> list[tuple[str, list[dict]]]:
    """Pick representative date groups spread across the date range.

    Selects earliest, middle, and latest dates to show amount patterns
    over time.
    """
    # Group by date
    by_date: dict[str, list[dict]] = defaultdict(list)
    for t in txns:
        date_str = _extract_date_str(t)
        by_date[date_str].append(t)

    sorted_dates = sorted(by_date.keys())
    if len(sorted_dates) <= num_dates:
        return [(d, by_date[d]) for d in sorted_dates]

    # Pick earliest, middle, latest
    indices = [0, len(sorted_dates) // 2, len(sorted_dates) - 1]
    # Deduplicate indices (possible if very few dates)
    seen: set[int] = set()
    unique_indices = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    return [(sorted_dates[i], by_date[sorted_dates[i]]) for i in unique_indices]


# ---------------------------------------------------------------------------
# Undisputed formatting — always aggregated
# ---------------------------------------------------------------------------


def _format_undisputed(
    undisputed: list[dict], disputed_merchants: set[str]
) -> str:
    """Format undisputed transactions as aggregates.

    Shows overall stats and per-merchant breakdown only for merchants
    that also appear in disputed transactions (the overlap the LLM
    needs for pattern comparison).
    """
    total_amount = sum(t.get("amount", 0) for t in undisputed)
    dates = [_txn_date_key(t) for t in undisputed]
    min_date = _extract_date_str(min(undisputed, key=_txn_date_key))
    max_date = _extract_date_str(max(undisputed, key=_txn_date_key))

    header = (
        f"== Undisputed Transactions ({len(undisputed)} total, "
        f"${total_amount:,.2f}, {min_date} to {max_date}) =="
    )

    # Find overlapping merchants
    by_merchant: dict[str, list[dict]] = defaultdict(list)
    for t in undisputed:
        merchant = t.get("merchant_name", "")
        if merchant in disputed_merchants:
            by_merchant[merchant].append(t)

    lines = [header]
    if by_merchant:
        lines.append("Merchants also appearing in disputed transactions:")
        for merchant, txns in by_merchant.items():
            m_total = sum(t.get("amount", 0) for t in txns)
            m_min_date = _extract_date_str(min(txns, key=_txn_date_key))
            m_max_date = _extract_date_str(max(txns, key=_txn_date_key))
            lines.append(
                f"- {merchant}: {len(txns)} txns, ${m_total:,.2f}, "
                f"{m_min_date} to {m_max_date}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _txn_date_key(t: dict) -> str:
    """Extract a sortable date string from a transaction dict.

    Handles both ISO datetime strings and date-only strings. Returns
    the raw string (ISO-sortable) or a fallback for missing/unparseable
    values.
    """
    raw = t.get("transaction_date", "")
    if not raw:
        return "0000-00-00"
    if isinstance(raw, datetime):
        return raw.strftime("%Y-%m-%d")
    # ISO strings like "2024-04-21T00:00:00+00:00" sort correctly as-is,
    # but truncate to date for grouping consistency
    return str(raw)[:10]


def _extract_date_str(t: dict | str) -> str:
    """Extract a display-friendly date string (YYYY-MM-DD)."""
    if isinstance(t, str):
        return t[:10]
    return _txn_date_key(t)


def _format_single_txn(t: dict) -> str:
    """Format a single transaction as a concise one-liner."""
    amount = t.get("amount", 0)
    merchant = t.get("merchant_name", "Unknown merchant")
    outcome = t.get("outcome", "")
    auth = t.get("auth_method", "")
    channel = t.get("channel", "")

    parts = [f"${amount:,.2f} at {merchant}"]
    if outcome:
        parts.append(str(outcome))
    if auth:
        parts.append(str(auth))
    if channel:
        parts.append(str(channel))

    return ", ".join(parts)
