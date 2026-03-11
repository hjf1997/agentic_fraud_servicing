"""Tests for ClaimType enum and ClaimExtraction models."""

import pytest
from pydantic import ValidationError

from agentic_fraud_servicing.models.claims import ClaimExtraction, ClaimExtractionResult
from agentic_fraud_servicing.models.enums import ClaimType

# -- ClaimType enum -----------------------------------------------------------


class TestClaimType:
    """Tests for the ClaimType enum."""

    FRAUD_TYPES = [
        "TRANSACTION_DISPUTE",
        "CARD_NOT_PRESENT_FRAUD",
        "LOST_STOLEN_CARD",
        "IDENTITY_VERIFICATION",
        "ACCOUNT_TAKEOVER",
        "LOCATION_CLAIM",
        "CARD_POSSESSION",
        "MERCHANT_FRAUD",
        "SPENDING_PATTERN",
    ]

    DISPUTE_TYPES = [
        "GOODS_NOT_RECEIVED",
        "DUPLICATE_CHARGE",
        "RETURN_NOT_CREDITED",
        "INCORRECT_AMOUNT",
        "GOODS_NOT_AS_DESCRIBED",
        "RECURRING_AFTER_CANCEL",
        "SERVICES_NOT_RENDERED",
        "DEFECTIVE_MERCHANDISE",
    ]

    def test_has_exactly_17_members(self):
        assert len(ClaimType) == 17

    def test_has_9_fraud_types(self):
        for name in self.FRAUD_TYPES:
            assert hasattr(ClaimType, name), f"Missing fraud type: {name}"

    def test_has_8_dispute_types(self):
        for name in self.DISPUTE_TYPES:
            assert hasattr(ClaimType, name), f"Missing dispute type: {name}"

    def test_str_mixin(self):
        assert ClaimType.TRANSACTION_DISPUTE == "TRANSACTION_DISPUTE"
        assert ClaimType.GOODS_NOT_RECEIVED == "GOODS_NOT_RECEIVED"

    def test_string_serialization(self):
        ct = ClaimType.LOST_STOLEN_CARD
        assert ct.value == "LOST_STOLEN_CARD"
        assert isinstance(ct.value, str)


# -- ClaimExtraction model ----------------------------------------------------


class TestClaimExtraction:
    """Tests for the ClaimExtraction Pydantic model."""

    def test_required_fields(self):
        claim = ClaimExtraction(
            claim_type=ClaimType.TRANSACTION_DISPUTE,
            claim_description="I didn't make this charge",
            confidence=0.85,
        )
        assert claim.claim_type == ClaimType.TRANSACTION_DISPUTE
        assert claim.claim_description == "I didn't make this charge"
        assert claim.confidence == 0.85
        assert claim.entities == {}
        assert claim.context is None

    def test_all_fields(self):
        claim = ClaimExtraction(
            claim_type=ClaimType.GOODS_NOT_RECEIVED,
            claim_description="Package never arrived",
            entities={
                "merchant_name": "Amazon",
                "order_date": "2026-02-15",
                "amount": 149.99,
            },
            confidence=0.92,
            context="CM called about missing delivery",
        )
        assert claim.claim_type == ClaimType.GOODS_NOT_RECEIVED
        assert claim.entities["merchant_name"] == "Amazon"
        assert claim.entities["amount"] == 149.99
        assert claim.context == "CM called about missing delivery"

    def test_confidence_validation_too_high(self):
        with pytest.raises(ValidationError):
            ClaimExtraction(
                claim_type=ClaimType.LOST_STOLEN_CARD,
                claim_description="Card was stolen",
                confidence=1.5,
            )

    def test_confidence_validation_too_low(self):
        with pytest.raises(ValidationError):
            ClaimExtraction(
                claim_type=ClaimType.LOST_STOLEN_CARD,
                claim_description="Card was stolen",
                confidence=-0.1,
            )

    def test_json_round_trip(self):
        claim = ClaimExtraction(
            claim_type=ClaimType.DUPLICATE_CHARGE,
            claim_description="I was charged twice",
            entities={"amount": 50.0, "merchant_name": "Walmart"},
            confidence=0.9,
            context="Second charge appeared same day",
        )
        json_str = claim.model_dump_json()
        restored = ClaimExtraction.model_validate_json(json_str)
        assert restored == claim
        assert restored.claim_type == ClaimType.DUPLICATE_CHARGE
        assert restored.entities["amount"] == 50.0

    def test_entities_dict_accepts_any_values(self):
        claim = ClaimExtraction(
            claim_type=ClaimType.INCORRECT_AMOUNT,
            claim_description="Charged wrong amount",
            entities={
                "expected_amount": 25.00,
                "actual_amount": 250.00,
                "merchant_name": "Store",
                "is_recurring": False,
                "tags": ["billing", "overcharge"],
            },
            confidence=0.8,
        )
        assert claim.entities["is_recurring"] is False
        assert claim.entities["tags"] == ["billing", "overcharge"]


# -- ClaimExtractionResult model ----------------------------------------------


class TestClaimExtractionResult:
    """Tests for the ClaimExtractionResult Pydantic model."""

    def test_defaults_empty_list(self):
        result = ClaimExtractionResult()
        assert result.claims == []
        assert len(result.claims) == 0

    def test_with_claims_list(self):
        claims = [
            ClaimExtraction(
                claim_type=ClaimType.TRANSACTION_DISPUTE,
                claim_description="Didn't make this charge",
                confidence=0.85,
            ),
            ClaimExtraction(
                claim_type=ClaimType.CARD_POSSESSION,
                claim_description="I have my card with me",
                entities={"location": "home"},
                confidence=0.95,
            ),
        ]
        result = ClaimExtractionResult(claims=claims)
        assert len(result.claims) == 2
        assert result.claims[0].claim_type == ClaimType.TRANSACTION_DISPUTE
        assert result.claims[1].claim_type == ClaimType.CARD_POSSESSION

    def test_json_round_trip(self):
        result = ClaimExtractionResult(
            claims=[
                ClaimExtraction(
                    claim_type=ClaimType.MERCHANT_FRAUD,
                    claim_description="Merchant is fraudulent",
                    entities={"merchant_name": "FakeShop"},
                    confidence=0.7,
                ),
            ]
        )
        json_str = result.model_dump_json()
        restored = ClaimExtractionResult.model_validate_json(json_str)
        assert len(restored.claims) == 1
        assert restored.claims[0].claim_type == ClaimType.MERCHANT_FRAUD
