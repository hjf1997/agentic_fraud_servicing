"""Unit tests for the PCI-compliant PII redaction module."""

import pytest

from agentic_fraud_servicing.ingestion.redaction import (
    RedactionError,
    redact_address,
    redact_all,
    redact_cvv,
    redact_dob,
    redact_pan,
    redact_ssn,
)
from agentic_fraud_servicing.models.transcript import RedactionInfo


class TestRedactPan:
    """Tests for credit/debit card number redaction."""

    def test_visa_16_digit(self):
        assert redact_pan("card 4111111111111111 here") == "card [PAN_REDACTED] here"

    def test_mastercard_16_digit(self):
        assert redact_pan("mc 5500000000000004 end") == "mc [PAN_REDACTED] end"

    def test_amex_15_digit(self):
        assert redact_pan("amex 378282246310005 end") == "amex [PAN_REDACTED] end"

    def test_pan_with_spaces(self):
        assert redact_pan("card 3742 454554 00126") == "card [PAN_REDACTED]"

    def test_pan_with_dashes(self):
        assert redact_pan("card 4111-1111-1111-1111 end") == "card [PAN_REDACTED] end"

    def test_multiple_pans(self):
        text = "cards 4111111111111111 and 378282246310005"
        result = redact_pan(text)
        assert result.count("[PAN_REDACTED]") == 2
        assert "4111" not in result
        assert "3782" not in result

    def test_no_pan_unchanged(self):
        text = "no card numbers here, just 12345"
        assert redact_pan(text) == text

    def test_partial_card_number_not_redacted(self):
        """Partial card numbers (too few digits) should not be redacted."""
        text = "ref number 411111111111 is twelve digits"
        assert redact_pan(text) == text

    def test_mastercard_2xxx_range(self):
        assert redact_pan("mc 2221000000000009 end") == "mc [PAN_REDACTED] end"


class TestRedactCvv:
    """Tests for CVV/CVC code redaction."""

    def test_cvv_pattern(self):
        assert "CVV [CVV_REDACTED]" in redact_cvv("CVV 123")

    def test_cvc_colon_pattern(self):
        result = redact_cvv("CVC: 1234")
        assert "[CVV_REDACTED]" in result
        assert "1234" not in result

    def test_security_code_pattern(self):
        result = redact_cvv("security code 456")
        assert "[CVV_REDACTED]" in result
        assert "456" not in result

    def test_case_insensitive(self):
        result = redact_cvv("cvv 789")
        assert "[CVV_REDACTED]" in result
        assert "789" not in result

    def test_no_false_positive_without_keyword(self):
        """Random 3-digit numbers without keywords should NOT be redacted."""
        text = "the amount was 123 dollars and 456 cents"
        assert redact_cvv(text) == text


class TestRedactSsn:
    """Tests for Social Security Number redaction."""

    def test_ssn_with_dashes(self):
        assert redact_ssn("ssn 123-45-6789") == "ssn [SSN_REDACTED]"

    def test_ssn_with_spaces(self):
        assert redact_ssn("ssn 123 45 6789") == "ssn [SSN_REDACTED]"

    def test_ssn_continuous(self):
        assert redact_ssn("ssn 123456789") == "ssn [SSN_REDACTED]"

    def test_invalid_ssn_000_not_matched(self):
        """SSNs starting with 000 are invalid and should not match."""
        text = "number 000-12-3456 here"
        assert redact_ssn(text) == text

    def test_ssn_starting_with_9_not_matched(self):
        """SSNs starting with 9xx are reserved for ITIN."""
        text = "number 900-12-3456 here"
        assert redact_ssn(text) == text


class TestRedactDob:
    """Tests for date-of-birth redaction."""

    def test_dob_slash_format(self):
        assert redact_dob("DOB: 01/15/1990") == "[DOB_REDACTED]"

    def test_date_of_birth_dash_format(self):
        assert redact_dob("date of birth 03-22-1985") == "[DOB_REDACTED]"

    def test_born_month_name(self):
        assert redact_dob("born January 5, 1992") == "[DOB_REDACTED]"

    def test_birthday_format(self):
        assert redact_dob("birthday: 12/25/2000") == "[DOB_REDACTED]"

    def test_random_date_without_keyword_not_redacted(self):
        """Dates without DOB/born/birthday keywords should NOT be redacted."""
        text = "the transaction was on 01/15/1990"
        assert redact_dob(text) == text


class TestRedactAddress:
    """Tests for US street address redaction."""

    def test_main_street(self):
        assert redact_address("lives at 123 Main Street") == "lives at [ADDRESS_REDACTED]"

    def test_oak_ave(self):
        assert redact_address("456 Oak Ave is nearby") == "[ADDRESS_REDACTED] is nearby"

    def test_elm_boulevard(self):
        result = redact_address("789 Elm Boulevard, Apt 4")
        assert "[ADDRESS_REDACTED]" in result

    def test_drive(self):
        assert "[ADDRESS_REDACTED]" in redact_address("10 Sunset Dr")

    def test_lane(self):
        assert "[ADDRESS_REDACTED]" in redact_address("22 Willow Ln")

    def test_road(self):
        assert "[ADDRESS_REDACTED]" in redact_address("55 River Rd")

    def test_court(self):
        assert "[ADDRESS_REDACTED]" in redact_address("8 Cherry Ct")

    def test_place(self):
        assert "[ADDRESS_REDACTED]" in redact_address("3 Park Pl")

    def test_way(self):
        assert "[ADDRESS_REDACTED]" in redact_address("99 Kings Way")

    def test_circle(self):
        assert "[ADDRESS_REDACTED]" in redact_address("7 Maple Cir")


class TestRedactAll:
    """Tests for the combined redact_all function."""

    def test_multiple_pii_types(self):
        text = "Card 4111111111111111, SSN 123-45-6789, DOB: 01/15/1990"
        redacted, info = redact_all(text)
        assert "[PAN_REDACTED]" in redacted
        assert "[SSN_REDACTED]" in redacted
        assert "[DOB_REDACTED]" in redacted
        assert info.contains_pan is True
        assert "SSN" in info.pii_types
        assert "DOB" in info.pii_types

    def test_no_pii_unchanged(self):
        text = "Hello, how can I help you today?"
        redacted, info = redact_all(text)
        assert redacted == text
        assert info.contains_pan is False
        assert info.contains_cvv is False
        assert info.pii_types == []

    def test_non_string_raises_redaction_error(self):
        with pytest.raises(RedactionError, match="Expected string"):
            redact_all(12345)

    def test_none_raises_redaction_error(self):
        with pytest.raises(RedactionError, match="Expected string"):
            redact_all(None)

    def test_pan_in_natural_sentence(self):
        """PAN embedded in a natural sentence should still be redacted."""
        text = "My card number is 4111111111111111 and I need help."
        redacted, info = redact_all(text)
        assert "4111" not in redacted
        assert "[PAN_REDACTED]" in redacted
        assert info.contains_pan is True

    def test_returns_redaction_info_type(self):
        _, info = redact_all("some text")
        assert isinstance(info, RedactionInfo)

    def test_cvv_and_address(self):
        text = "CVV 123, address 456 Oak Ave"
        redacted, info = redact_all(text)
        assert "[CVV_REDACTED]" in redacted
        assert "[ADDRESS_REDACTED]" in redacted
        assert info.contains_cvv is True
        assert "ADDRESS" in info.pii_types
