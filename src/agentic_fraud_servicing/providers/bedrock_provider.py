"""AWS Bedrock ModelProvider implementation.

Wraps AWS Bedrock's converse() API to conform to the OpenAI Agents SDK Model
interface. Translates between the SDK's OpenAI Responses format and Bedrock's
native message format, enabling Claude models on Bedrock as a drop-in backend
for the agent orchestration layer.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import TYPE_CHECKING, Any

import boto3
from agents.models.interface import Model, ModelProvider, ModelResponse
from agents.tool import FunctionTool
from agents.usage import Usage

from agentic_fraud_servicing.config import Settings
from agentic_fraud_servicing.providers.base import ProviderError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from agents.agent_output import AgentOutputSchemaBase
    from agents.handoffs import Handoff
    from agents.model_settings import ModelSettings
    from agents.models.interface import ModelTracing
    from agents.tool import Tool
    from openai.types.responses import ResponsePromptParam
    from openai.types.responses.response_input_item_param import TResponseInputItem
    from openai.types.responses.response_stream_event import TResponseStreamEvent


# ---------------------------------------------------------------------------
# Private conversion helpers
# ---------------------------------------------------------------------------


def _convert_input_to_bedrock_messages(
    input_items: str | list[TResponseInputItem],
) -> list[dict[str, Any]]:
    """Convert SDK input items (OpenAI Responses format) to Bedrock messages.

    Handles three main item types that appear during agent tool-call loops:
    1. Text messages (user/assistant) from EasyInputMessage, InputMessage, or
       ResponseOutputMessage formats.
    2. Function tool calls from the assistant (ResponseFunctionToolCallParam).
    3. Function tool call outputs from the framework (FunctionCallOutput).

    Args:
        input_items: Either a plain string (treated as a single user message)
            or a list of SDK input item dicts.

    Returns:
        A list of Bedrock message dicts with 'role' and 'content' keys.
    """
    if isinstance(input_items, str):
        return [{"role": "user", "content": [{"text": input_items}]}]

    messages: list[dict[str, Any]] = []

    # Accumulate assistant tool_use blocks so they can be merged into one message
    pending_tool_uses: list[dict[str, Any]] = []
    pending_assistant_text: str | None = None
    # Accumulate tool results so consecutive results merge into one user message
    pending_tool_results: list[dict[str, Any]] = []

    def _flush_assistant() -> None:
        nonlocal pending_tool_uses, pending_assistant_text
        if pending_tool_uses or pending_assistant_text is not None:
            content: list[dict[str, Any]] = []
            if pending_assistant_text is not None:
                content.append({"text": pending_assistant_text})
            content.extend(pending_tool_uses)
            messages.append({"role": "assistant", "content": content})
            pending_tool_uses = []
            pending_assistant_text = None

    def _flush_tool_results() -> None:
        nonlocal pending_tool_results
        if pending_tool_results:
            messages.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results = []

    for item in input_items:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")

        # --- EasyInputMessage / InputMessage ---
        # Also handle items with 'role' but no 'type' (SDK shorthand format)
        if item_type == "message" or (item_type is None and "role" in item):
            role = item.get("role", "user")
            raw_content = item.get("content", "")

            # Extract text from content (may be str or list of content parts)
            if isinstance(raw_content, str):
                text = raw_content
            elif isinstance(raw_content, list):
                parts = []
                for part in raw_content:
                    if isinstance(part, dict) and part.get("type") in (
                        "input_text",
                        "output_text",
                        "text",
                    ):
                        parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        parts.append(part)
                text = "\n".join(parts)
            else:
                text = str(raw_content)

            if role == "assistant":
                _flush_tool_results()
                _flush_assistant()
                pending_assistant_text = text
            else:
                # user / system / developer -> treat as user for Bedrock
                _flush_tool_results()
                _flush_assistant()
                messages.append({"role": "user", "content": [{"text": text}]})

        # --- ResponseFunctionToolCallParam (assistant requesting a tool call) ---
        elif item_type == "function_call":
            _flush_tool_results()
            call_id = item.get("call_id", "")
            name = item.get("name", "")
            arguments = item.get("arguments", "{}")
            try:
                args_dict = json.loads(arguments) if arguments else {}
            except (json.JSONDecodeError, TypeError):
                args_dict = {}
            pending_tool_uses.append(
                {
                    "toolUse": {
                        "toolUseId": call_id,
                        "name": name,
                        "input": args_dict,
                    }
                }
            )

        # --- FunctionCallOutput (tool result being sent back) ---
        elif item_type == "function_call_output":
            # Flush any pending assistant content first (tool results follow)
            _flush_assistant()
            call_id = item.get("call_id", "")
            output = item.get("output", "")
            if isinstance(output, list):
                text_parts = []
                for part in output:
                    if isinstance(part, dict):
                        text_parts.append(part.get("text", str(part)))
                    else:
                        text_parts.append(str(part))
                output = "\n".join(text_parts)
            # Accumulate tool results — they'll be merged into one user message
            pending_tool_results.append(
                {
                    "toolResult": {
                        "toolUseId": call_id,
                        "content": [{"text": str(output)}],
                    }
                }
            )

    # Flush any remaining content
    _flush_tool_results()
    _flush_assistant()
    return messages


def _convert_tools_to_bedrock(tools: list[Tool]) -> dict[str, Any] | None:
    """Convert SDK Tool objects to Bedrock toolConfig format.

    Only FunctionTool instances are supported — other tool types (file search,
    web search, etc.) are silently skipped since Bedrock doesn't support them.

    Args:
        tools: List of SDK Tool objects.

    Returns:
        A Bedrock toolConfig dict, or None if no convertible tools exist.
    """
    if not tools:
        return None

    tool_specs: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, FunctionTool):
            tool_specs.append(
                {
                    "toolSpec": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "inputSchema": {"json": tool.params_json_schema},
                    }
                }
            )

    if not tool_specs:
        return None
    return {"tools": tool_specs}


def _convert_bedrock_response_to_output(
    bedrock_response: dict[str, Any],
) -> list[Any]:
    """Convert a Bedrock converse() response to SDK output items.

    Handles two output content block types:
    - text blocks -> ResponseOutputMessage with ResponseOutputText content
    - toolUse blocks -> ResponseFunctionToolCall items

    Args:
        bedrock_response: The raw dict returned by boto3 converse().

    Returns:
        A list of SDK output items (ResponseOutputMessage and/or
        ResponseFunctionToolCall instances).
    """
    from openai.types.responses import (
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseOutputText,
    )

    output_items: list[Any] = []
    bedrock_output = bedrock_response.get("output", {})
    content_blocks = bedrock_output.get("message", {}).get("content", [])

    text_parts: list[ResponseOutputText] = []
    for block in content_blocks:
        if "text" in block:
            text_parts.append(
                ResponseOutputText(
                    type="output_text",
                    text=block["text"],
                    annotations=[],
                )
            )
        elif "toolUse" in block:
            tool_use = block["toolUse"]
            output_items.append(
                ResponseFunctionToolCall(
                    type="function_call",
                    call_id=tool_use.get("toolUseId", str(uuid.uuid4())),
                    name=tool_use.get("name", ""),
                    arguments=json.dumps(tool_use.get("input", {})),
                )
            )

    # Bundle all text parts into a single output message
    if text_parts:
        output_items.insert(
            0,
            ResponseOutputMessage(
                id=str(uuid.uuid4()),
                type="message",
                role="assistant",
                status="completed",
                content=text_parts,
            ),
        )

    return output_items


def _extract_json_from_output(output_items: list[Any]) -> list[Any]:
    """Extract raw JSON from model output when structured output was requested.

    Claude on Bedrock may wrap JSON in markdown code fences or include
    preamble text. This strips everything except the JSON object/array.
    """
    import re

    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    for i, item in enumerate(output_items):
        if not isinstance(item, ResponseOutputMessage):
            continue
        for j, content in enumerate(item.content):
            if not isinstance(content, ResponseOutputText):
                continue
            text = content.text.strip()
            # Strip markdown JSON fences: ```json ... ``` or ``` ... ```
            fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
            if fence_match:
                text = fence_match.group(1).strip()
            # If text still doesn't look like JSON, try to find { or [
            if not text.startswith(("{", "[")):
                json_start = re.search(r"[{\[]", text)
                if json_start:
                    # Find the matching JSON by taking from first { or [
                    text = text[json_start.start() :]
            item.content[j] = ResponseOutputText(type="output_text", text=text, annotations=[])
    return output_items


# ---------------------------------------------------------------------------
# BedrockModel — implements the SDK's Model interface
# ---------------------------------------------------------------------------


class BedrockModel(Model):
    """Model implementation that delegates to AWS Bedrock converse() API.

    Translates between the OpenAI Agents SDK's Responses format and Bedrock's
    native message format. Uses asyncio.to_thread() to call the synchronous
    boto3 client without blocking the event loop.

    Args:
        model_id: Bedrock model identifier (e.g.
            'us.anthropic.claude-sonnet-4-5-20250929-v1:0').
        boto3_client: A boto3 bedrock-runtime client instance.
    """

    def __init__(self, model_id: str, boto3_client: Any) -> None:
        self._model_id = model_id
        self._client = boto3_client

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> ModelResponse:
        """Send a request to Bedrock converse() and return an SDK ModelResponse.

        Converts SDK input items to Bedrock message format, calls converse()
        in a thread (boto3 is synchronous), then converts the response back
        to SDK output items.
        """
        # Build Bedrock request kwargs
        kwargs: dict[str, Any] = {"modelId": self._model_id}

        # System prompt — append JSON output instruction when structured output
        # is requested, since Bedrock/Claude doesn't natively enforce JSON mode
        # like OpenAI does.
        sys_text = system_instructions or ""
        if output_schema is not None and hasattr(output_schema, "_output_schema"):
            schema_json = json.dumps(output_schema._output_schema, indent=2)
            sys_text += (
                "\n\nIMPORTANT: You MUST respond with ONLY a valid JSON object "
                "matching this schema — no markdown, no explanation, no code "
                f"fences, just raw JSON:\n{schema_json}"
            )
        if sys_text:
            kwargs["system"] = [{"text": sys_text}]

        # Messages
        messages = _convert_input_to_bedrock_messages(input)
        kwargs["messages"] = messages

        # Tool configuration
        tool_config = _convert_tools_to_bedrock(tools)
        if tool_config:
            kwargs["toolConfig"] = tool_config

        # Call Bedrock (synchronous boto3 in a thread)
        try:
            bedrock_response = await asyncio.to_thread(self._client.converse, **kwargs)
        except Exception as exc:
            raise ProviderError(
                f"Bedrock converse() failed: {exc}",
                model_id=self._model_id,
                request_type="get_response",
            ) from exc

        # Convert response — extract JSON from markdown fences if needed
        output_items = _convert_bedrock_response_to_output(bedrock_response)
        if output_schema is not None:
            output_items = _extract_json_from_output(output_items)

        # Extract usage
        usage_data = bedrock_response.get("usage", {})
        input_tokens = usage_data.get("inputTokens", 0)
        output_tokens = usage_data.get("outputTokens", 0)

        return ModelResponse(
            output=output_items,
            usage=Usage(
                requests=1,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            response_id=None,
        )

    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        """Streaming is not yet implemented for the Bedrock provider."""
        raise NotImplementedError("Bedrock streaming not yet implemented")


# ---------------------------------------------------------------------------
# BedrockModelProvider — creates BedrockModel instances from settings
# ---------------------------------------------------------------------------


class BedrockModelProvider(ModelProvider):
    """ModelProvider that creates BedrockModel instances via boto3.

    Initialises a bedrock-runtime client using AWS profile and region from
    the application settings. The client is reused across all models.

    Args:
        settings: Application settings with AWS credentials and model config.
    """

    def __init__(self, settings: Settings) -> None:
        # Only pass profile_name if the profile actually exists in AWS config.
        # When credentials come from environment variables (e.g. bearer tokens,
        # IAM roles), specifying a non-existent profile causes a failure.
        session_kwargs: dict[str, str] = {"region_name": settings.aws_region}
        if settings.aws_profile:
            try:
                test_session = boto3.Session(profile_name=settings.aws_profile)
                test_session.get_credentials()
                session_kwargs["profile_name"] = settings.aws_profile
            except Exception:
                pass  # Fall back to env-based credentials
        session = boto3.Session(**session_kwargs)
        self._client = session.client("bedrock-runtime")
        self._default_model_id = settings.aws_bedrock_model_id

    def get_model(self, model_name: str | None) -> Model:
        """Return a BedrockModel for the given model identifier.

        Args:
            model_name: Bedrock model ID. If None, uses the default from
                settings (aws_bedrock_model_id).

        Returns:
            A BedrockModel wrapping the shared boto3 client.
        """
        if model_name is None:
            model_name = self._default_model_id
        return BedrockModel(model_id=model_name, boto3_client=self._client)
