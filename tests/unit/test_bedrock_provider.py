"""Tests for the AWS Bedrock ModelProvider implementation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from agents.models.interface import ModelResponse
from agents.usage import Usage

from agentic_fraud_servicing.providers.base import ProviderError
from agentic_fraud_servicing.providers.bedrock_provider import (
    BedrockModel,
    BedrockModelProvider,
    _convert_bedrock_response_to_output,
    _convert_input_to_bedrock_messages,
    _convert_tools_to_bedrock,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(
    profile: str = "default",
    region: str = "us-east-1",
    model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
) -> MagicMock:
    """Create a mock Settings object with AWS fields."""
    settings = MagicMock()
    settings.llm_provider = "bedrock"
    settings.aws_profile = profile
    settings.aws_region = region
    settings.aws_bedrock_model_id = model_id
    return settings


def _make_bedrock_text_response(text: str = "Hello!") -> dict:
    """Create a mock Bedrock converse() response with a text block."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text}],
            }
        },
        "usage": {"inputTokens": 10, "outputTokens": 5},
    }


def _make_bedrock_tool_response(
    tool_use_id: str = "call_123",
    name: str = "get_transaction",
    input_args: dict | None = None,
) -> dict:
    """Create a mock Bedrock converse() response with a toolUse block."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": tool_use_id,
                            "name": name,
                            "input": input_args or {"txn_id": "T001"},
                        }
                    }
                ],
            }
        },
        "usage": {"inputTokens": 20, "outputTokens": 15},
    }


def _make_function_tool(
    name: str = "get_transaction",
    description: str = "Look up a transaction",
    schema: dict | None = None,
) -> MagicMock:
    """Create a mock FunctionTool with the expected attributes."""
    from agents.tool import FunctionTool

    tool = MagicMock(spec=FunctionTool)
    tool.name = name
    tool.description = description
    tool.params_json_schema = schema or {
        "type": "object",
        "properties": {"txn_id": {"type": "string"}},
    }
    return tool


# Shared dummy kwargs for get_response calls (the non-essential parameters)
_DUMMY_KWARGS = {
    "model_settings": MagicMock(),
    "tools": [],
    "output_schema": None,
    "handoffs": [],
    "tracing": MagicMock(),
    "previous_response_id": None,
    "conversation_id": None,
    "prompt": None,
}


# ---------------------------------------------------------------------------
# BedrockModel tests
# ---------------------------------------------------------------------------


class TestBedrockModelGetResponse:
    """Tests for BedrockModel.get_response()."""

    @pytest.mark.asyncio
    async def test_simple_text_response(self) -> None:
        """get_response converts a simple text input and returns a ModelResponse."""
        mock_client = MagicMock()
        mock_client.converse.return_value = _make_bedrock_text_response("Hi there")

        model = BedrockModel(model_id="test-model", boto3_client=mock_client)
        response = await model.get_response(
            system_instructions="Be helpful.",
            input="Hello",
            **_DUMMY_KWARGS,
        )

        assert isinstance(response, ModelResponse)
        assert response.response_id is None
        # Should have one output message with text
        assert len(response.output) == 1
        msg = response.output[0]
        assert msg.type == "message"
        assert msg.content[0].text == "Hi there"

    @pytest.mark.asyncio
    async def test_tool_call_response(self) -> None:
        """get_response converts toolUse blocks to ResponseFunctionToolCall items."""
        mock_client = MagicMock()
        mock_client.converse.return_value = _make_bedrock_tool_response(
            tool_use_id="call_abc",
            name="lookup",
            input_args={"id": "123"},
        )

        model = BedrockModel(model_id="test-model", boto3_client=mock_client)
        response = await model.get_response(
            system_instructions=None,
            input="Find transaction 123",
            **_DUMMY_KWARGS,
        )

        # Should have one function_call output
        assert len(response.output) == 1
        tool_call = response.output[0]
        assert tool_call.type == "function_call"
        assert tool_call.call_id == "call_abc"
        assert tool_call.name == "lookup"
        assert json.loads(tool_call.arguments) == {"id": "123"}

    @pytest.mark.asyncio
    async def test_wraps_boto3_errors_in_provider_error(self) -> None:
        """get_response wraps boto3 exceptions in ProviderError with context."""
        mock_client = MagicMock()
        mock_client.converse.side_effect = RuntimeError("Throttled")

        model = BedrockModel(model_id="claude-test", boto3_client=mock_client)
        with pytest.raises(ProviderError, match="Bedrock converse\\(\\) failed") as exc_info:
            await model.get_response(
                system_instructions=None,
                input="Hi",
                **_DUMMY_KWARGS,
            )

        assert exc_info.value.model_id == "claude-test"
        assert exc_info.value.request_type == "get_response"

    @pytest.mark.asyncio
    async def test_usage_extracted_from_response(self) -> None:
        """get_response extracts token usage from the Bedrock response."""
        mock_client = MagicMock()
        mock_client.converse.return_value = _make_bedrock_text_response()

        model = BedrockModel(model_id="test-model", boto3_client=mock_client)
        response = await model.get_response(
            system_instructions=None,
            input="Hi",
            **_DUMMY_KWARGS,
        )

        assert isinstance(response.usage, Usage)
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5
        assert response.usage.total_tokens == 15
        assert response.usage.requests == 1

    @pytest.mark.asyncio
    async def test_converse_receives_correct_model_id(self) -> None:
        """converse() is called with the correct modelId parameter."""
        mock_client = MagicMock()
        mock_client.converse.return_value = _make_bedrock_text_response()

        model = BedrockModel(model_id="my-special-model", boto3_client=mock_client)
        await model.get_response(
            system_instructions=None,
            input="Hi",
            **_DUMMY_KWARGS,
        )

        call_kwargs = mock_client.converse.call_args
        assert call_kwargs.kwargs["modelId"] == "my-special-model"

    @pytest.mark.asyncio
    async def test_converse_receives_correct_message_format(self) -> None:
        """converse() receives properly formatted Bedrock messages."""
        mock_client = MagicMock()
        mock_client.converse.return_value = _make_bedrock_text_response()

        model = BedrockModel(model_id="test-model", boto3_client=mock_client)
        await model.get_response(
            system_instructions="Be brief.",
            input="What is 2+2?",
            **_DUMMY_KWARGS,
        )

        call_kwargs = mock_client.converse.call_args.kwargs
        # System instructions
        assert call_kwargs["system"] == [{"text": "Be brief."}]
        # Messages should have user message
        assert call_kwargs["messages"] == [{"role": "user", "content": [{"text": "What is 2+2?"}]}]

    @pytest.mark.asyncio
    async def test_converse_with_tool_config(self) -> None:
        """converse() receives toolConfig when tools are provided."""
        mock_client = MagicMock()
        mock_client.converse.return_value = _make_bedrock_text_response()

        tool = _make_function_tool(name="my_tool", description="Does things")
        model = BedrockModel(model_id="test-model", boto3_client=mock_client)
        await model.get_response(
            system_instructions=None,
            input="Call my_tool",
            tools=[tool],
            model_settings=MagicMock(),
            output_schema=None,
            handoffs=[],
            tracing=MagicMock(),
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )

        call_kwargs = mock_client.converse.call_args.kwargs
        assert "toolConfig" in call_kwargs
        tool_specs = call_kwargs["toolConfig"]["tools"]
        assert len(tool_specs) == 1
        assert tool_specs[0]["toolSpec"]["name"] == "my_tool"


class TestBedrockModelStreamResponse:
    """Tests for BedrockModel.stream_response()."""

    def test_raises_not_implemented(self) -> None:
        """stream_response raises NotImplementedError."""
        model = BedrockModel(model_id="test", boto3_client=MagicMock())
        with pytest.raises(NotImplementedError, match="Bedrock streaming"):
            model.stream_response(
                system_instructions=None,
                input="Hi",
                **_DUMMY_KWARGS,
            )


# ---------------------------------------------------------------------------
# BedrockModelProvider tests
# ---------------------------------------------------------------------------


class TestBedrockModelProvider:
    """Tests for BedrockModelProvider."""

    @patch("agentic_fraud_servicing.providers.bedrock_provider.boto3")
    def test_get_model_returns_bedrock_model(self, mock_boto3: MagicMock) -> None:
        """get_model returns a BedrockModel instance."""
        provider = BedrockModelProvider(_make_settings())
        model = provider.get_model("some-model-id")

        assert isinstance(model, BedrockModel)
        assert model._model_id == "some-model-id"

    @patch("agentic_fraud_servicing.providers.bedrock_provider.boto3")
    def test_uses_default_model_id_when_none(self, mock_boto3: MagicMock) -> None:
        """get_model with None uses settings.aws_bedrock_model_id."""
        settings = _make_settings(model_id="default-claude-model")
        provider = BedrockModelProvider(settings)
        model = provider.get_model(None)

        assert model._model_id == "default-claude-model"

    @patch("agentic_fraud_servicing.providers.bedrock_provider.boto3")
    def test_creates_session_with_correct_profile_and_region(self, mock_boto3: MagicMock) -> None:
        """Provider creates boto3 Session with profile and region from settings."""
        settings = _make_settings(profile="my-profile", region="eu-west-1")
        BedrockModelProvider(settings)

        mock_boto3.Session.assert_called_once_with(
            profile_name="my-profile",
            region_name="eu-west-1",
        )
        mock_boto3.Session.return_value.client.assert_called_once_with("bedrock-runtime")


# ---------------------------------------------------------------------------
# Conversion helper tests
# ---------------------------------------------------------------------------


class TestConvertInputToBedrockMessages:
    """Tests for _convert_input_to_bedrock_messages()."""

    def test_string_input(self) -> None:
        """Plain string becomes a single user message."""
        result = _convert_input_to_bedrock_messages("Hello")
        assert result == [{"role": "user", "content": [{"text": "Hello"}]}]

    def test_function_call_and_output(self) -> None:
        """Function call + output items produce assistant toolUse + user toolResult."""
        items = [
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "search",
                "arguments": '{"q": "test"}',
            },
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": "Found 3 results",
            },
        ]
        result = _convert_input_to_bedrock_messages(items)

        # First message: assistant with toolUse
        assert result[0]["role"] == "assistant"
        tool_use = result[0]["content"][0]["toolUse"]
        assert tool_use["toolUseId"] == "c1"
        assert tool_use["name"] == "search"
        assert tool_use["input"] == {"q": "test"}

        # Second message: user with toolResult
        assert result[1]["role"] == "user"
        tool_result = result[1]["content"][0]["toolResult"]
        assert tool_result["toolUseId"] == "c1"
        assert tool_result["content"] == [{"text": "Found 3 results"}]


class TestConvertToolsToBedrock:
    """Tests for _convert_tools_to_bedrock()."""

    def test_produces_correct_tool_spec_format(self) -> None:
        """FunctionTool is converted to Bedrock toolSpec format."""
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        tool = _make_function_tool(name="calculate", description="Do math", schema=schema)

        result = _convert_tools_to_bedrock([tool])

        assert result is not None
        specs = result["tools"]
        assert len(specs) == 1
        spec = specs[0]["toolSpec"]
        assert spec["name"] == "calculate"
        assert spec["description"] == "Do math"
        assert spec["inputSchema"]["json"] == schema

    def test_returns_none_for_empty_tools(self) -> None:
        """Empty tool list returns None."""
        assert _convert_tools_to_bedrock([]) is None


class TestConvertBedrockResponseToOutput:
    """Tests for _convert_bedrock_response_to_output()."""

    def test_text_response(self) -> None:
        """Text content block produces ResponseOutputMessage."""
        response = _make_bedrock_text_response("world")
        items = _convert_bedrock_response_to_output(response)

        assert len(items) == 1
        assert items[0].type == "message"
        assert items[0].content[0].text == "world"

    def test_tool_use_response(self) -> None:
        """toolUse content block produces ResponseFunctionToolCall."""
        response = _make_bedrock_tool_response(tool_use_id="t1", name="fn", input_args={"a": 1})
        items = _convert_bedrock_response_to_output(response)

        assert len(items) == 1
        assert items[0].type == "function_call"
        assert items[0].call_id == "t1"
        assert items[0].name == "fn"
        assert json.loads(items[0].arguments) == {"a": 1}
