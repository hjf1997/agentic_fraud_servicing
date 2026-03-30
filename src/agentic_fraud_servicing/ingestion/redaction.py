"""PCI-compliant PII detection and redaction for transcript text.

Provides functions to detect and replace sensitive data (PAN, CVV, SSN, DOB,
addresses, phone numbers, emails) with typed placeholders before text reaches
any LLM or storage. All redaction uses regex pattern matching — no external
dependencies.
"""

import re

from agentic_fraud_servicing.models.transcript import RedactionInfo


class RedactionError(ValueError):
    """Raised when the redaction pipeline encounters malformed input."""


# --- PAN (Primary Account Number) patterns ---

# Match 15-digit AMEX (3[47]) or 16-digit Visa/MC with optional spaces/dashes.
# AMEX: 3[47]XX XXXXXX XXXXX or continuous
# Visa: 4XXX ... (16 digits)
# MC: 5[1-5]XX ... or 2[2-7]XX ... (16 digits)
_SEP = r"[\s\-]?"  # optional space or dash between digit groups

_PAN_AMEX = r"3[47]\d{2}" + _SEP + r"\d{6}" + _SEP + r"\d{5}"
_PAN_VISA = r"4\d{3}" + (_SEP + r"\d{4}") * 3
_PAN_MC = r"(?:5[1-5]\d{2}|2[2-7]\d{2})" + (_SEP + r"\d{4}") * 3
_PAN_RE = re.compile(r"(?<!\d)(?:" + _PAN_AMEX + "|" + _PAN_VISA + "|" + _PAN_MC + r")(?!\d)")

# --- CVV patterns (keyword-gated) ---

_CVV_RE = re.compile(r"(?i)(?:cvv2?|cvc|security\s+code)\s*:?\s*(\d{3,4})")

# --- SSN patterns ---

# XXX-XX-XXXX, XXX XX XXXX, or XXXXXXXXX (9 continuous digits)
# Exclude invalid: first 3 digits = 000, first digit = 9
_SSN_RE = re.compile(
    r"(?<!\d)"
    r"(?!000)(?!9\d{2})"  # not 000-xx-xxxx, not 9xx-xx-xxxx
    r"([0-8]\d{2})"
    r"([-\s])"
    r"(?!00)\d{2}"
    r"\2"  # same separator
    r"(?!0000)\d{4}"
    r"(?!\d)"
)

# Continuous 9-digit SSN (no separator)
_SSN_CONTINUOUS_RE = re.compile(
    r"(?<!\d)"
    r"(?!000)(?!9\d{2})"
    r"[0-8]\d{2}"
    r"(?!00)\d{2}"
    r"(?!0000)\d{4}"
    r"(?!\d)"
)

# --- DOB patterns (keyword-gated) ---

_MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December|"
    r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
)
# Date formats: MM/DD/YYYY, MM-DD-YYYY, Month DD YYYY, DD Month YYYY
_DATE_PATTERN = (
    r"(?:"
    r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}"  # MM/DD/YYYY or MM-DD-YYYY
    r"|" + _MONTHS + r"\.?\s+\d{1,2},?\s+\d{2,4}"  # Month DD, YYYY
    r"|\d{1,2}\s+" + _MONTHS + r"\.?\s+\d{2,4}"  # DD Month YYYY
    r")"
)
_DOB_RE = re.compile(r"(?i)(?:date\s+of\s+birth|DOB|born|birthday)\s*:?\s*" + _DATE_PATTERN)

# --- Address patterns ---

_STREET_TYPES = (
    r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Drive|Dr|"
    r"Lane|Ln|Road|Rd|Court|Ct|Place|Pl|Way|Circle|Cir)"
)
_ADDRESS_RE = re.compile(r"(?i)\d{1,6}\s+(?:[A-Za-z]+\s+){1,4}" + _STREET_TYPES + r"\.?")

# --- Phone number patterns ---

# US formats: (214) 449-5199, 214-449-5199, 214.449.5199, 2144495199,
# +1 (214) 449-5199, +1-214-449-5199, 1-800-555-0199
_PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:\+?1[\s.\-]?)?"  # optional country code +1
    r"(?:\(\d{3}\)|\d{3})"  # area code: (214) or 214
    r"[\s.\-]?"  # separator
    r"\d{3}"  # exchange
    r"[\s.\-]?"  # separator
    r"\d{4}"  # subscriber
    r"(?!\d)"
)

# --- Email patterns ---

# Standard email: user@domain.tld, also handles "dot" / "at" obfuscation
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)
# Obfuscated email: "user at domain dot com"
_EMAIL_OBFUSCATED_RE = re.compile(
    r"(?i)[a-zA-Z0-9._%+\-]+\s+(?:at|@)\s+[a-zA-Z0-9.\-]+\s+(?:dot|\.)\s+[a-zA-Z]{2,}"
)


def redact_pan(text: str) -> str:
    """Replace credit/debit card numbers with [PAN_REDACTED]."""
    return _PAN_RE.sub("[PAN_REDACTED]", text)


def redact_cvv(text: str) -> str:
    """Replace CVV/CVC codes following keywords with [CVV_REDACTED].

    Only redacts digits that appear after CVV/CVC/security code keywords
    to avoid false positives on random 3-digit numbers.
    """

    def _replace_cvv(match: re.Match) -> str:
        # Replace only the digit group, keep the keyword prefix
        full = match.group(0)
        digits = match.group(1)
        return full[: -len(digits)] + "[CVV_REDACTED]"

    return _CVV_RE.sub(_replace_cvv, text)


def redact_ssn(text: str) -> str:
    """Replace Social Security Numbers with [SSN_REDACTED].

    Handles XXX-XX-XXXX, XXX XX XXXX, and XXXXXXXXX formats.
    Excludes invalid prefixes (000, 9xx).
    """
    result = _SSN_RE.sub("[SSN_REDACTED]", text)
    result = _SSN_CONTINUOUS_RE.sub("[SSN_REDACTED]", result)
    return result


def redact_dob(text: str) -> str:
    """Replace date-of-birth patterns following keywords with [DOB_REDACTED].

    Only redacts dates that appear after DOB/born/birthday/date of birth
    keywords to avoid false positives on arbitrary dates.
    """
    return _DOB_RE.sub("[DOB_REDACTED]", text)


def redact_address(text: str) -> str:
    """Replace US street address patterns with [ADDRESS_REDACTED]."""
    return _ADDRESS_RE.sub("[ADDRESS_REDACTED]", text)


def redact_phone(text: str) -> str:
    """Replace US phone number patterns with [PHONE_REDACTED]."""
    return _PHONE_RE.sub("[PHONE_REDACTED]", text)


def redact_email(text: str) -> str:
    """Replace email addresses with [EMAIL_REDACTED].

    Handles both standard (user@domain.com) and obfuscated
    (user at domain dot com) formats.
    """
    result = _EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    result = _EMAIL_OBFUSCATED_RE.sub("[EMAIL_REDACTED]", result)
    return result


def redact_all(text: str) -> tuple[str, RedactionInfo]:
    """Apply all redaction functions in sequence and return redaction metadata.

    Args:
        text: The raw transcript text to redact.

    Returns:
        A tuple of (redacted_text, RedactionInfo) with flags set for each
        PII type that was detected and redacted.

    Raises:
        RedactionError: If text is not a string.
    """
    if not isinstance(text, str):
        raise RedactionError(f"Expected string input, got {type(text).__name__}")

    contains_pan = False
    contains_cvv = False
    pii_types: list[str] = []

    # Apply redactions in order: PAN, CVV, SSN, DOB, address, phone, email
    redacted = redact_pan(text)
    if redacted != text:
        contains_pan = True
    text = redacted

    redacted = redact_cvv(text)
    if redacted != text:
        contains_cvv = True
    text = redacted

    redacted = redact_ssn(text)
    if redacted != text:
        pii_types.append("SSN")
    text = redacted

    redacted = redact_dob(text)
    if redacted != text:
        pii_types.append("DOB")
    text = redacted

    redacted = redact_address(text)
    if redacted != text:
        pii_types.append("ADDRESS")
    text = redacted

    redacted = redact_phone(text)
    if redacted != text:
        pii_types.append("PHONE")
    text = redacted

    redacted = redact_email(text)
    if redacted != text:
        pii_types.append("EMAIL")

    info = RedactionInfo(
        contains_pan=contains_pan,
        contains_cvv=contains_cvv,
        pii_types=pii_types,
    )
    return redacted, info
