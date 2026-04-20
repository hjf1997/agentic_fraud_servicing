"""Firewall-aware content redactor for LLM prompts.

Detects and replaces sensitive data patterns with reversible placeholders
before sending agent prompts to the upstream LLM. Acts as a proactive
guardrail against enterprise firewall/DLP policy blocks that can reject
entire prompts when sensitive patterns are detected.

Adapted from the OpenCode-ConnectChain Gateway redactor. Patterns that are
too aggressive for fraud investigation context (where dollar amounts, dates,
and transaction details are essential for reasoning) are disabled.

A single Redactor instance should be used per orchestrator session so the
same original value always maps to the same placeholder, keeping the
conversation coherent across turns.
"""

import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------
# Order: most specific first -> broadest last.
# When matches overlap, the earlier (more specific) pattern wins.
#
# Each entry: (category_name, compiled_regex)
# The category name is used in placeholders: [REDACTED_{category}_{n}]
#
# DISABLED patterns are commented out with rationale — they strip information
# that agents need for fraud investigation reasoning.

_PATTERNS: list[tuple[str, re.Pattern]] = [
    # === PII — specific formats ===
    # SSN  (123-45-6789)
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Credit card — AMEX (3xxx xxxxxx xxxxx, 15 digits) then generic 16-digit
    (
        "CREDIT_CARD",
        re.compile(
            r"\b(?:"
            r"3[47]\d{2}[\s-]?\d{6}[\s-]?\d{5}"  # AMEX
            r"|(?:\d{4}[\s-]?){3}\d{4}"  # Visa / MC / Discover
            r")\b"
        ),
    ),
    # Phone — US formats:  +1 (212) 555-0123 / 212-555-0123 / 2125550123 etc.
    (
        "PHONE",
        re.compile(
            r"(?<!\d)"  # no digit before
            r"(?:\+?1[\s.-]?)?"  # optional country code
            r"\(?\d{3}\)?[\s.-]?"  # area code
            r"\d{3}[\s.-]?\d{4}"
            r"(?!\d)"  # no digit after
        ),
    ),
    # Email
    ("EMAIL", re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")),
    # === Financial ===
    # DISABLED: Dollar amounts are critical for fraud reasoning (transaction
    # amounts, dispute values, refund calculations).
    # ("DOLLAR_AMOUNT", re.compile(r"\$[\d,]+(?:\.\d{1,2})?")),
    # DISABLED: Percentages appear in hypothesis scores and risk assessments.
    # ("PERCENTAGE", re.compile(r"\b\d+(?:\.\d+)?%")),
    # === Date / Time ===
    # DISABLED: Dates are essential for fraud investigation — transaction dates,
    # incident timelines, card loss dates, delivery dates.
    # ("DATE", re.compile(...)),
    # DISABLED: Times appear in auth logs and transaction records.
    # ("TIME", re.compile(...)),
    # === Demographic — keyword lists ===
    # Gender
    (
        "GENDER",
        re.compile(
            r"\b(?:male|female|non-binary|nonbinary|transgender|cisgender"
            r"|genderqueer|genderfluid|agender|bigender|intersex)\b",
            re.IGNORECASE,
        ),
    ),
    # Gender expression / pronouns
    (
        "GENDER_EXPRESSION",
        re.compile(
            r"\b(?:he/him|she/her|they/them|ze/zir|xe/xem)\b",
            re.IGNORECASE,
        ),
    ),
    # Religion
    (
        "RELIGION",
        re.compile(
            r"\b(?:Christian(?:ity)?|Catholic(?:ism)?|Protestant(?:ism)?"
            r"|Baptist|Methodist|Lutheran|Presbyterian|Evangelical"
            r"|Orthodox|Muslim|Islam(?:ic)?|Jewish|Judaism"
            r"|Hindu(?:ism)?|Buddhist|Buddhism|Sikh(?:ism)?"
            r"|Mormon|Latter.?day|Jehovah|Scientolog\w*"
            r"|Baha.?i|Jain(?:ism)?|Shinto(?:ism)?"
            r"|Taoist|Taoism|Confucian(?:ism)?|Zoroastrian(?:ism)?"
            r"|Pagan(?:ism)?|Wiccan|Atheist|Agnostic)\b",
            re.IGNORECASE,
        ),
    ),
    # Nationality (major)
    (
        "NATIONALITY",
        re.compile(
            r"\b(?:American|British|Canadian|Mexican|Chinese|Japanese|Korean"
            r"|Indian|Pakistani|Bangladeshi|Filipino|Vietnamese|Thai"
            r"|Indonesian|Malaysian|Brazilian|Colombian|Argentin\w+"
            r"|Peruvian|Chilean|French|German|Italian|Spanish|Portuguese"
            r"|Dutch|Belgian|Swedish|Norwegian|Danish|Finnish|Polish"
            r"|Czech|Romanian|Russian|Ukrainian|Turkish|Iranian|Iraqi"
            r"|Saudi|Egyptian|Nigerian|Kenyan|South\s?African|Ethiopian"
            r"|Ghanaian|Australian|New\s?Zealand\w*)\b",
            re.IGNORECASE,
        ),
    ),
    # Medical terms
    (
        "MEDICAL",
        re.compile(
            r"\b(?:diagnosis|prognosis|prescription|medication|surgery"
            r"|therapy|treatment|symptom|disorder|disease|allergy|allergic"
            r"|disability|handicap|impairment|chronic|acute|terminal"
            r"|mental\s?health|depression|anxiety|PTSD|bipolar"
            r"|schizophren\w*|diabetes|cancer|HIV|AIDS|hepatitis"
            r"|epilepsy|asthma|hypertension|cholesterol|insulin"
            r"|chemotherapy|radiation|rehabilitation|prosthetic"
            r"|wheelchair|hearing\s?aid)\b",
            re.IGNORECASE,
        ),
    ),
    # Medical accommodation
    (
        "MEDICAL_ACCOMMODATION",
        re.compile(
            r"\b(?:accommodation|ADA|FMLA|medical\s?leave|sick\s?leave"
            r"|reasonable\s?accommodation|modified\s?duty|light\s?duty"
            r"|ergonomic|assistive\s?technology|service\s?animal)\b",
            re.IGNORECASE,
        ),
    ),
    # Security-question responses (best-effort pattern matching)
    (
        "SECURITY_QUESTION",
        re.compile(
            r"(?:mother'?s?\s+maiden\s+name|first\s+pet|elementary\s+school"
            r"|favorite\s+(?:color|movie|book|food|teacher|sport)"
            r"|city\s+(?:born|grew\s+up)|street\s+grew\s+up)"
            r"[\s:]+\S+",
            re.IGNORECASE,
        ),
    ),
    # Person name — title-prefixed (full NER would require spaCy / similar)
    (
        "PERSON_NAME",
        re.compile(
            r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Sr|Jr)"
            r"\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"
        ),
    ),
    # Address — US street address patterns.
    # Cap word repetitions at {1,4} to prevent backtracking on long text.
    (
        "ADDRESS",
        re.compile(
            r"\b\d+\s+[A-Z][a-zA-Z]+(?:\s+[A-Z]?[a-zA-Z]+){0,4}\s+"
            r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr"
            r"|Lane|Ln|Way|Court|Ct|Place|Pl|Circle|Cir|Trail|Trl"
            r"|Parkway|Pkwy|Highway|Hwy)\b\.?",
            re.IGNORECASE,
        ),
    ),
    # DISABLED: Zip codes are needed for location claims in fraud investigation
    # (CM alleges they were in a different location).
    # ("ZIP_CODE", re.compile(r"\b\d{5}(?:-\d{4})?\b")),
    # === Broad patterns (aggressive — ordered last) ===
    # DISABLED: OTP pattern matches ANY 3-digit number — far too aggressive.
    # Catches CVV references, partial amounts, counts, turn numbers.
    # ("OTP", re.compile(r"\b\d{3}\b")),
    # DISABLED: Catches card last-4 digits, transaction IDs, amounts, and
    # reference numbers that agents need for fraud investigation reasoning.
    # The CREDIT_CARD pattern already handles full card numbers.
    # ("LONG_NUMBER", re.compile(r"\b\d{4,}\b")),
]

# Subset of patterns safe for structured data (dicts/JSON). Excludes patterns
# that target numeric values (LONG_NUMBER) which would break JSON structure
# when applied to bare ints/floats, and demographic patterns that are too
# aggressive for evidence data fields.
_SAFE_PATTERNS: list[tuple[str, re.Pattern]] = [
    # PII — high-risk patterns that DLP firewalls commonly block
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    (
        "CREDIT_CARD",
        re.compile(
            r"\b(?:"
            r"3[47]\d{2}[\s-]?\d{6}[\s-]?\d{5}"
            r"|(?:\d{4}[\s-]?){3}\d{4}"
            r")\b"
        ),
    ),
    (
        "PHONE",
        re.compile(
            r"(?<!\d)"
            r"(?:\+?1[\s.-]?)?"
            r"\(?\d{3}\)?[\s.-]?"
            r"\d{3}[\s.-]?\d{4}"
            r"(?!\d)"
        ),
    ),
    ("EMAIL", re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")),
    (
        "PERSON_NAME",
        re.compile(
            r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Sr|Jr)"
            r"\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"
        ),
    ),
    (
        "ADDRESS",
        re.compile(
            r"\b\d+\s+[A-Z][a-zA-Z]+(?:\s+[A-Z]?[a-zA-Z]+){0,4}\s+"
            r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr"
            r"|Lane|Ln|Way|Court|Ct|Place|Pl|Circle|Cir|Trail|Trl"
            r"|Parkway|Pkwy|Highway|Hwy)\b\.?",
            re.IGNORECASE,
        ),
    ),
    (
        "SECURITY_QUESTION",
        re.compile(
            r"(?:mother'?s?\s+maiden\s+name|first\s+pet|elementary\s+school"
            r"|favorite\s+(?:color|movie|book|food|teacher|sport)"
            r"|city\s+(?:born|grew\s+up)|street\s+grew\s+up)"
            r"[\s:]+\S+",
            re.IGNORECASE,
        ),
    ),
    # DISABLED for safe mode — too aggressive for structured evidence data:
    # ("LONG_NUMBER", ...) — breaks bare numeric JSON values
    # ("GENDER", ...) — irrelevant for transaction/auth data
    # ("GENDER_EXPRESSION", ...) — irrelevant for transaction/auth data
    # ("RELIGION", ...) — irrelevant for transaction/auth data
    # ("NATIONALITY", ...) — may appear in legitimate merchant/location data
    # ("MEDICAL", ...) — irrelevant for transaction/auth data
    # ("MEDICAL_ACCOMMODATION", ...) — irrelevant for transaction/auth data
]


# ---------------------------------------------------------------------------
# Redactor class
# ---------------------------------------------------------------------------


class FirewallRedactor:
    """Detects and replaces sensitive data with reversible, numbered placeholders.

    A single instance should be used for the entire orchestrator session so the
    same original value always maps to the same placeholder, keeping the
    conversation coherent across turns.
    """

    def __init__(self) -> None:
        self._vault: dict[str, str] = {}  # original -> placeholder
        self._reverse: dict[str, str] = {}  # placeholder -> original
        self._counters: dict[str, int] = defaultdict(int)

    def _get_placeholder(self, category: str, original: str) -> str:
        """Return a consistent placeholder for *original*, creating one if needed."""
        if original in self._vault:
            return self._vault[original]
        self._counters[category] += 1
        placeholder = f"[REDACTED_{category}_{self._counters[category]}]"
        self._vault[original] = placeholder
        self._reverse[placeholder] = original
        return placeholder

    # Maximum text length to process. Beyond this, skip redaction to avoid
    # regex backtracking on very large inputs (e.g., system events with
    # embedded full context). 10 KB covers any realistic single utterance.
    _MAX_TEXT_LEN = 10_000

    def redact_text(self, text: str) -> str:
        """Replace all sensitive patterns in *text* with placeholders.

        Uses the full pattern list including LONG_NUMBER and demographic
        patterns. For structured data (dicts), use redact_dict() instead.
        """
        if not text:
            return text
        if len(text) > self._MAX_TEXT_LEN:
            logger.warning(
                "Firewall redactor skipping text of length %d (exceeds %d)",
                len(text),
                self._MAX_TEXT_LEN,
            )
            return text
        return self._redact_with_patterns(text, _PATTERNS)

    def redact_dict(self, data: dict | list) -> dict | list:
        """Redact sensitive patterns in string values of a dict/list structure.

        Walks the structure recursively, applying the safe pattern subset to
        string values only. Dict keys, numeric values, booleans, and None are
        left untouched to preserve JSON-serializable structure.

        Args:
            data: A dict or list (typically from evidence store query results).

        Returns:
            A new dict/list with string values redacted. Original is not mutated.
        """
        return self._walk(data)

    def _walk(self, node: object) -> object:
        """Recursively walk a data structure, redacting string values."""
        if isinstance(node, dict):
            return {k: self._walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [self._walk(item) for item in node]
        if isinstance(node, str):
            return self._redact_with_patterns(node, _SAFE_PATTERNS)
        # int, float, bool, None — pass through unchanged
        return node

    def _redact_with_patterns(self, text: str, patterns: list[tuple[str, re.Pattern]]) -> str:
        """Apply a specific pattern list to a text string."""
        if not text or len(text) > self._MAX_TEXT_LEN:
            return text

        matches: list[tuple[int, int, str, str]] = []
        for category, pattern in patterns:
            for m in pattern.finditer(text):
                matches.append((m.start(), m.end(), category, m.group()))

        if not matches:
            return text

        matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

        filtered: list[tuple[int, int, str, str]] = []
        last_end = 0
        for start, end, category, value in matches:
            if start >= last_end:
                filtered.append((start, end, category, value))
                last_end = end

        result = text
        for start, end, category, value in reversed(filtered):
            placeholder = self._get_placeholder(category, value)
            result = result[:start] + placeholder + result[end:]

        return result
