"""Tests for AllegationDetailType enum and AllegationExtraction models."""

import pytest
from agentic_fraud_servicing.models.allegations import (
    AllegationExtraction,
    AllegationExtractionResult,
)
from pydantic import ValidationError

from agentic_fraud_servicing.models.enums import AllegationDetailType

# -- AllegationDetailType enum ------------------------------------------------


class TestAllegationDetailType:
    """Tests for the AllegationDetailType enum."""

    TRANSACTION_TYPES = [
        "UNRECOGNIZED_TRANSACTION",
        "CARD_NOT_PRESENT_FRAUD",
        "DUPLICATE_CHARGE",
        "INCORRECT_AMOUNT",
        "RECURRING_AFTER_CANCEL",
    ]

    IDENTITY_ACCESS_TYPES = [
        "ACCOUNT_TAKEOVER",
        "IDENTITY_VERIFICATION",
        "CARD_POSSESSION",
        "LOST_STOLEN_CARD",
    ]

    EVIDENCE_CONTEXT_TYPES = [
        "LOCATION_CLAIM",
        "SPENDING_PATTERN",
    ]

    MERCHANT_SERVICE_TYPES = [
        "GOODS_NOT_RECEIVED",
        "GOODS_NOT_AS_DESCRIBED",
        "SERVICES_NOT_RENDERED",
        "RETURN_NOT_CREDITED",
        "DEFECTIVE_MERCHANDISE",
        "MERCHANT_FRAUD",
    ]

    SOCIAL_ENGINEERING_TYPES = [
        "UNSOLICITED_CONTACT",
        "AUTHORITY_IMPERSONATION",
        "PRESSURE_OR_THREAT",
        "DIRECTED_PAYMENT",
        "DECEPTIVE_OFFER",
    ]

    def test_has_exactly_22_members(self):
        assert len(AllegationDetailType) == 22

    def test_has_transaction_types(self):
        for name in self.TRANSACTION_TYPES:
            assert hasattr(AllegationDetailType, name), f"Missing type: {name}"

    def test_has_identity_access_types(self):
        for name in self.IDENTITY_ACCESS_TYPES:
            assert hasattr(AllegationDetailType, name), f"Missing type: {name}"

    def test_has_evidence_context_types(self):
        for name in self.EVIDENCE_CONTEXT_TYPES:
            assert hasattr(AllegationDetailType, name), f"Missing type: {name}"

    def test_has_merchant_service_types(self):
        for name in self.MERCHANT_SERVICE_TYPES:
            assert hasattr(AllegationDetailType, name), f"Missing type: {name}"

    def test_has_social_engineering_types(self):
        for name in self.SOCIAL_ENGINEERING_TYPES:
            assert hasattr(AllegationDetailType, name), f"Missing type: {name}"

    def test_str_mixin(self):
        assert AllegationDetailType.UNRECOGNIZED_TRANSACTION == "UNRECOGNIZED_TRANSACTION"
        assert AllegationDetailType.GOODS_NOT_RECEIVED == "GOODS_NOT_RECEIVED"

    def test_string_serialization(self):
        ct = AllegationDetailType.LOST_STOLEN_CARD
        assert ct.value == "LOST_STOLEN_CARD"
        assert isinstance(ct.value, str)


# -- AllegationExtraction model -----------------------------------------------


class TestAllegationExtraction:
    """Tests for the AllegationExtraction Pydantic model."""

    def test_required_fields(self):
        allegation = AllegationExtraction(
            detail_type=AllegationDetailType.UNRECOGNIZED_TRANSACTION,
            description="I didn't make this charge",
            confidence=0.85,
        )
        assert allegation.detail_type == AllegationDetailType.UNRECOGNIZED_TRANSACTION
        assert allegation.description == "I didn't make this charge"
        assert allegation.confidence == 0.85
        assert allegation.entities == {}
        assert allegation.context is None

    def test_all_fields(self):
        allegation = AllegationExtraction(
            detail_type=AllegationDetailType.GOODS_NOT_RECEIVED,
            description="Package never arrived",
            entities={
                "merchant_name": "Amazon",
                "order_date": "2026-02-15",
                "amount": 149.99,
            },
            confidence=0.92,
            context="CM called about missing delivery",
        )
        assert allegation.detail_type == AllegationDetailType.GOODS_NOT_RECEIVED
        assert allegation.entities["merchant_name"] == "Amazon"
        assert allegation.entities["amount"] == 149.99
        assert allegation.context == "CM called about missing delivery"

    def test_confidence_validation_too_high(self):
        with pytest.raises(ValidationError):
            AllegationExtraction(
                detail_type=AllegationDetailType.LOST_STOLEN_CARD,
                description="Card was stolen",
                confidence=1.5,
            )

    def test_confidence_validation_too_low(self):
        with pytest.raises(ValidationError):
            AllegationExtraction(
                detail_type=AllegationDetailType.LOST_STOLEN_CARD,
                description="Card was stolen",
                confidence=-0.1,
            )

    def test_json_round_trip(self):
        allegation = AllegationExtraction(
            detail_type=AllegationDetailType.DUPLICATE_CHARGE,
            description="I was charged twice",
            entities={"amount": 50.0, "merchant_name": "Walmart"},
            confidence=0.9,
            context="Second charge appeared same day",
        )
        json_str = allegation.model_dump_json()
        restored = AllegationExtraction.model_validate_json(json_str)
        assert restored == allegation
        assert restored.detail_type == AllegationDetailType.DUPLICATE_CHARGE
        assert restored.entities["amount"] == 50.0

    def test_entities_dict_accepts_any_values(self):
        allegation = AllegationExtraction(
            detail_type=AllegationDetailType.INCORRECT_AMOUNT,
            description="Charged wrong amount",
            entities={
                "expected_amount": 25.00,
                "actual_amount": 250.00,
                "merchant_name": "Store",
                "is_recurring": False,
                "tags": ["billing", "overcharge"],
            },
            confidence=0.8,
        )
        assert allegation.entities["is_recurring"] is False
        assert allegation.entities["tags"] == ["billing", "overcharge"]


# -- AllegationExtractionResult model -----------------------------------------


class TestAllegationExtractionResult:
    """Tests for the AllegationExtractionResult Pydantic model."""

    def test_defaults_empty_list(self):
        result = AllegationExtractionResult()
        assert result.allegations == []
        assert len(result.allegations) == 0

    def test_with_allegations_list(self):
        allegations = [
            AllegationExtraction(
                detail_type=AllegationDetailType.UNRECOGNIZED_TRANSACTION,
                description="Didn't make this charge",
                confidence=0.85,
            ),
            AllegationExtraction(
                detail_type=AllegationDetailType.CARD_POSSESSION,
                description="I have my card with me",
                entities={"location": "home"},
                confidence=0.95,
            ),
        ]
        result = AllegationExtractionResult(allegations=allegations)
        assert len(result.allegations) == 2
        assert result.allegations[0].detail_type == AllegationDetailType.UNRECOGNIZED_TRANSACTION
        assert result.allegations[1].detail_type == AllegationDetailType.CARD_POSSESSION

    def test_json_round_trip(self):
        result = AllegationExtractionResult(
            allegations=[
                AllegationExtraction(
                    detail_type=AllegationDetailType.MERCHANT_FRAUD,
                    description="Merchant is fraudulent",
                    entities={"merchant_name": "FakeShop"},
                    confidence=0.7,
                ),
            ]
        )
        json_str = result.model_dump_json()
        restored = AllegationExtractionResult.model_validate_json(json_str)
        assert len(restored.allegations) == 1
        assert restored.allegations[0].detail_type == AllegationDetailType.MERCHANT_FRAUD
