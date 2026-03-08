"""Tests for the case writer specialist agent module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_fraud_servicing.investigator.case_writer import (
    CASE_WRITER_INSTRUCTIONS,
    CasePack,
    case_writer_agent,
    run_case_writer,
)


class TestCasePack:
    """Tests for the CasePack Pydantic model."""

    def test_defaults(self):
        """CasePack with all defaults has correct empty values."""
        pack = CasePack()
        assert pack.case_summary == ""
        assert pack.timeline == []
        assert pack.evidence_list == []
        assert pack.decision_recommendation == {}
        assert pack.investigation_notes == []

    def test_all_fields(self):
        """CasePack accepts all fields with correct types."""
        pack = CasePack(
            case_summary="Investigation found unauthorized transactions.",
            timeline=[
                {
                    "timestamp": "2026-03-01T10:00:00",
                    "event_type": "transaction",
                    "description": "Purchase at merchant XYZ",
                    "source": "FACT",
                }
            ],
            evidence_list=[
                {
                    "node_id": "n-001",
                    "node_type": "TRANSACTION",
                    "source_type": "FACT",
                    "summary": "Transaction of $500 at XYZ",
                }
            ],
            decision_recommendation={
                "category": "fraud",
                "confidence": 0.85,
                "top_factors": [
                    {"factor": "chip+PIN mismatch", "evidence_ref": "n-001", "weight": 0.9}
                ],
                "uncertainties": ["merchant response pending"],
                "suggested_actions": ["issue provisional credit"],
                "required_approvals": [],
            },
            investigation_notes=["Merchant has prior dispute history."],
        )
        assert "unauthorized" in pack.case_summary
        assert len(pack.timeline) == 1
        assert len(pack.evidence_list) == 1
        assert pack.decision_recommendation["category"] == "fraud"
        assert len(pack.investigation_notes) == 1

    def test_round_trip_json(self):
        """CasePack survives JSON round-trip serialization."""
        original = CasePack(
            case_summary="Test summary.",
            timeline=[
                {
                    "timestamp": "2026-01-01",
                    "event_type": "call",
                    "description": "Inbound call",
                    "source": "FACT",
                }
            ],
            evidence_list=[
                {
                    "node_id": "e-1",
                    "node_type": "CLAIM",
                    "source_type": "ALLEGATION",
                    "summary": "Claim text",
                }
            ],
            decision_recommendation={"category": "dispute", "confidence": 0.7},
            investigation_notes=["Note 1"],
        )
        json_str = original.model_dump_json()
        restored = CasePack.model_validate_json(json_str)
        assert restored == original

    def test_decision_recommendation_structure(self):
        """decision_recommendation contains all 6 expected sub-fields."""
        pack = CasePack(
            decision_recommendation={
                "category": "scam",
                "confidence": 0.65,
                "top_factors": [{"factor": "contradiction", "evidence_ref": "n-1", "weight": 0.8}],
                "uncertainties": ["missing merchant data"],
                "suggested_actions": ["escalate to fraud team"],
                "required_approvals": ["supervisor_review", "compliance_review"],
            },
        )
        rec = pack.decision_recommendation
        assert "category" in rec
        assert "confidence" in rec
        assert "top_factors" in rec
        assert "uncertainties" in rec
        assert "suggested_actions" in rec
        assert "required_approvals" in rec

    def test_timeline_has_source(self):
        """Timeline entries include a source field (FACT or ALLEGATION)."""
        pack = CasePack(
            timeline=[
                {
                    "timestamp": "2026-03-01",
                    "event_type": "transaction",
                    "description": "Buy",
                    "source": "FACT",
                },
                {
                    "timestamp": "2026-03-02",
                    "event_type": "claim",
                    "description": "Disputed",
                    "source": "ALLEGATION",
                },
            ],
        )
        for entry in pack.timeline:
            assert entry["source"] in ("FACT", "ALLEGATION")

    def test_evidence_list_has_source_type(self):
        """Evidence list entries include source_type (FACT or ALLEGATION)."""
        pack = CasePack(
            evidence_list=[
                {
                    "node_id": "n-1",
                    "node_type": "TRANSACTION",
                    "source_type": "FACT",
                    "summary": "Tx",
                },
                {
                    "node_id": "n-2",
                    "node_type": "CLAIM",
                    "source_type": "ALLEGATION",
                    "summary": "Claim",
                },
            ],
        )
        for entry in pack.evidence_list:
            assert entry["source_type"] in ("FACT", "ALLEGATION")


class TestCaseWriterAgent:
    """Tests for the case_writer_agent Agent instance."""

    def test_agent_name(self):
        """Agent has the correct name."""
        assert case_writer_agent.name == "case_writer"

    def test_agent_output_type(self):
        """Agent has CasePack as output_type."""
        assert case_writer_agent.output_type.output_type is CasePack

    def test_instructions_cover_narrative(self):
        """Instructions cover case narrative generation."""
        assert "narrative" in CASE_WRITER_INSTRUCTIONS.lower()
        assert "Case Summary" in CASE_WRITER_INSTRUCTIONS

    def test_instructions_cover_approvals(self):
        """Instructions cover required approvals logic."""
        assert "supervisor_review" in CASE_WRITER_INSTRUCTIONS
        assert "compliance_review" in CASE_WRITER_INSTRUCTIONS
        assert "senior_analyst_review" in CASE_WRITER_INSTRUCTIONS

    def test_instructions_cover_timeline(self):
        """Instructions cover chronological timeline building."""
        assert "timeline" in CASE_WRITER_INSTRUCTIONS.lower()
        assert "chronological" in CASE_WRITER_INSTRUCTIONS.lower()

    def test_instructions_contain_four_categories(self):
        """Instructions reference all four investigation categories."""
        assert "THIRD_PARTY_FRAUD" in CASE_WRITER_INSTRUCTIONS
        assert "FIRST_PARTY_FRAUD" in CASE_WRITER_INSTRUCTIONS
        assert "SCAM" in CASE_WRITER_INSTRUCTIONS
        assert "DISPUTE" in CASE_WRITER_INSTRUCTIONS

    def test_instructions_first_party_fraud_elevated_review(self):
        """Instructions require elevated review for FIRST_PARTY_FRAUD."""
        assert "FIRST_PARTY_FRAUD" in CASE_WRITER_INSTRUCTIONS
        assert "compliance_review" in CASE_WRITER_INSTRUCTIONS
        assert "supervisor_review" in CASE_WRITER_INSTRUCTIONS
        # Verify FIRST_PARTY_FRAUD specifically requires both
        idx = CASE_WRITER_INSTRUCTIONS.index("FIRST_PARTY_FRAUD: always")
        assert idx > 0

    def test_instructions_investigation_category_not_allegation(self):
        """Instructions use InvestigationCategory values for decision category."""
        assert "InvestigationCategory" in CASE_WRITER_INSTRUCTIONS
        assert "NOT AllegationType" in CASE_WRITER_INSTRUCTIONS


class TestRunCaseWriter:
    """Tests for the run_case_writer async function."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock ModelProvider."""
        return MagicMock()

    @pytest.fixture
    def sample_case_pack(self):
        """Create a sample CasePack for mock returns."""
        return CasePack(
            case_summary="Investigation complete. Fraud confirmed.",
            timeline=[
                {
                    "timestamp": "2026-03-01T10:00:00",
                    "event_type": "transaction",
                    "description": "Purchase",
                    "source": "FACT",
                },
            ],
            evidence_list=[
                {
                    "node_id": "n-001",
                    "node_type": "TRANSACTION",
                    "source_type": "FACT",
                    "summary": "$500 at XYZ",
                },
            ],
            decision_recommendation={
                "category": "fraud",
                "confidence": 0.9,
                "top_factors": [
                    {"factor": "chip mismatch", "evidence_ref": "n-001", "weight": 0.9}
                ],
                "uncertainties": [],
                "suggested_actions": ["issue credit"],
                "required_approvals": [],
            },
            investigation_notes=["No additional concerns."],
        )

    async def test_returns_case_pack(self, mock_provider, sample_case_pack):
        """run_case_writer returns CasePack from Runner.run."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = sample_case_pack

        with patch(
            "agentic_fraud_servicing.investigator.case_writer.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ):
            result = await run_case_writer(
                '{"case_id": "C-001"}',
                '{"primary_reason_code": "C08"}',
                '{"merchant_risk_score": 0.3}',
                '{"scam_likelihood": 0.1}',
                mock_provider,
            )

        assert isinstance(result, CasePack)
        assert "Fraud confirmed" in result.case_summary
        assert result.decision_recommendation["category"] == "fraud"

    async def test_passes_model_provider(self, mock_provider):
        """run_case_writer passes model_provider in RunConfig."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = CasePack()

        with patch(
            "agentic_fraud_servicing.investigator.case_writer.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_case_writer("{}", "{}", "{}", "{}", mock_provider)

        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["run_config"].model_provider is mock_provider

    async def test_includes_case_data_in_message(self, mock_provider):
        """run_case_writer includes case_data in the user message."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = CasePack()

        with patch(
            "agentic_fraud_servicing.investigator.case_writer.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_case_writer(
                '{"case_id": "C-999", "amount": 7500}',
                '{"reason_code": "FR2"}',
                '{"conflicts": []}',
                '{"scam_likelihood": 0.0}',
                mock_provider,
            )

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "C-999" in user_input
        assert "FR2" in user_input

    async def test_includes_all_specialist_results(self, mock_provider):
        """run_case_writer includes scheme, merchant, and scam results in message."""
        mock_run_result = MagicMock()
        mock_run_result.final_output = CasePack()

        with patch(
            "agentic_fraud_servicing.investigator.case_writer.Runner.run",
            new_callable=AsyncMock,
            return_value=mock_run_result,
        ) as mock_run:
            await run_case_writer(
                "case data here",
                "scheme mapping here",
                "merchant analysis here",
                "scam detection here",
                mock_provider,
            )

        call_args = mock_run.call_args
        user_input = call_args.kwargs.get("input") or call_args.args[1]
        assert "case data here" in user_input
        assert "scheme mapping here" in user_input
        assert "merchant analysis here" in user_input
        assert "scam detection here" in user_input

    async def test_wraps_exceptions(self, mock_provider):
        """run_case_writer wraps SDK exceptions in RuntimeError."""
        with patch(
            "agentic_fraud_servicing.investigator.case_writer.Runner.run",
            new_callable=AsyncMock,
            side_effect=ValueError("LLM call failed"),
        ):
            with pytest.raises(RuntimeError, match="Case writer agent failed"):
                await run_case_writer("{}", "{}", "{}", "{}", mock_provider)
