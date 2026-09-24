"""Strict parsing of the pinned models' documented native tool-call formats."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal


class ToolCallParseError(ValueError):
    """The model attempted a tool call that could not be decoded safely."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


_QWEN_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_QWEN_FUNCTION = re.compile(
    r"<function\s*=\s*([^>]+)>\s*(.*?)\s*</function>", re.DOTALL
)
_QWEN_PARAMETER = re.compile(
    r"<parameter\s*=\s*([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL
)
_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _json_object(value: str, *, context: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ToolCallParseError(f"invalid JSON {context}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ToolCallParseError(f"{context} must be a JSON object")
    return decoded


def _validate_call(name: Any, arguments: Any) -> ToolCall:
    if not isinstance(name, str) or _TOOL_NAME.fullmatch(name) is None:
        raise ToolCallParseError(
            "tool name must match ^[a-zA-Z0-9_-]{1,64}$"
        )
    if not isinstance(arguments, dict):
        raise ToolCallParseError("tool arguments must be an object")
    if not all(isinstance(key, str) for key in arguments):
        raise ToolCallParseError("tool argument names must be strings")
    return ToolCall(name=name, arguments=arguments)


def parse_qwen3_tool_calls(text: str) -> list[ToolCall]:
    """Parse Qwen3 XML calls, accepting the older JSON envelope as documented.

    An ordinary assistant response with no tool marker is a valid empty call
    list. An attempted but malformed marker raises instead of being treated as
    abstention.
    """

    blocks = _QWEN_BLOCK.findall(text)
    if not blocks:
        if "<tool_call" in text or "</tool_call" in text:
            raise ToolCallParseError("unterminated Qwen tool-call envelope")
        return []

    calls: list[ToolCall] = []
    for block in blocks:
        stripped = block.strip()
        if stripped.startswith("{"):
            payload = _json_object(stripped, context="Qwen tool call")
            arguments = payload.get("arguments", {})
            if isinstance(arguments, str):
                arguments = _json_object(arguments, context="Qwen arguments")
            calls.append(_validate_call(payload.get("name"), arguments))
            continue

        function = _QWEN_FUNCTION.fullmatch(stripped)
        if function is None:
            raise ToolCallParseError("invalid Qwen function envelope")
        name, raw_parameters = function.groups()
        arguments: dict[str, Any] = {}
        matches = list(_QWEN_PARAMETER.finditer(raw_parameters))
        residue = _QWEN_PARAMETER.sub("", raw_parameters).strip()
        if residue:
            raise ToolCallParseError("unparsed text inside Qwen function call")
        for match in matches:
            key = match.group(1).strip()
            if not key or key in arguments:
                raise ToolCallParseError("empty or duplicate Qwen parameter")
            # The official Qwen3 parser preserves XML parameter bodies as
            # strings. Match that behavior instead of guessing Python types.
            arguments[key] = match.group(2).strip("\n")
        calls.append(_validate_call(name.strip(), arguments))
    return calls


def parse_mistral_tool_calls(text: str) -> list[ToolCall]:
    """Parse Mistral v11+ calls and the official pre-v11 JSON-array form."""

    marker = "[TOOL_CALLS]"
    if marker not in text:
        return []
    chunks = text.split(marker)[1:]
    if not chunks or any(not chunk.strip() for chunk in chunks):
        raise ToolCallParseError("empty Mistral tool-call envelope")

    # Pre-v11 Mistral emits one marker followed by a JSON list.
    if len(chunks) == 1 and chunks[0].lstrip().startswith("["):
        try:
            payload, _ = json.JSONDecoder().raw_decode(chunks[0].lstrip())
        except json.JSONDecodeError as exc:
            raise ToolCallParseError(f"invalid Mistral tool-call array: {exc}") from exc
        if not isinstance(payload, list):
            raise ToolCallParseError("Mistral tool-call envelope must be an array")
        calls: list[ToolCall] = []
        for entry in payload:
            if not isinstance(entry, dict):
                raise ToolCallParseError(
                    "Mistral tool-call entry must be an object"
                )
            calls.append(
                _validate_call(entry.get("name"), entry.get("arguments", {}))
            )
        return calls

    calls: list[ToolCall] = []
    for chunk in chunks:
        body = chunk.strip()
        if "[ARGS]" in body:
            name, raw_arguments = body.split("[ARGS]", 1)
        else:
            brace = body.find("{")
            if brace <= 0:
                raise ToolCallParseError("Mistral call lacks an argument object")
            name, raw_arguments = body[:brace], body[brace:]
        raw_arguments = raw_arguments.lstrip()
        try:
            arguments, end = json.JSONDecoder().raw_decode(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ToolCallParseError(f"invalid Mistral arguments: {exc}") from exc
        if raw_arguments[end:].strip():
            raise ToolCallParseError("trailing text inside Mistral tool call")
        calls.append(_validate_call(name.strip(), arguments))
    return calls


def parse_tool_calls(
    text: str, family: Literal["qwen3", "mistral"]
) -> list[ToolCall]:
    if family == "qwen3":
        return parse_qwen3_tool_calls(text)
    return parse_mistral_tool_calls(text)


def bfcl_execution_strings(calls: list[ToolCall]) -> list[str]:
    """Convert decoded calls to the exact Python-call form BFCL executes."""

    return [
        f"{call.name}({','.join(f'{key}={value!r}' for key, value in call.arguments.items())})"
        for call in calls
    ]


_OPENAPI_TYPES = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "byte": "integer",
    "short": "integer",
    "long": "integer",
    "double": "number",
    "char": "string",
    "ArrayList": "array",
    "Array": "array",
    "HashMap": "object",
    "Hashtable": "object",
    "Queue": "array",
    "Stack": "array",
    "Bigint": "integer",
}


def _normalize_schema(node: Any) -> Any:
    if isinstance(node, list):
        return [_normalize_schema(value) for value in node]
    if not isinstance(node, dict):
        return node
    normalized = {key: _normalize_schema(value) for key, value in node.items()}
    if "type" in normalized:
        normalized["type"] = _OPENAPI_TYPES.get(str(normalized["type"]), "string")
        if node.get("type") == "float":
            normalized["format"] = "float"
    return normalized


def openai_tools(function_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the official BFCL Gorilla-to-OpenAPI schema conversion."""

    tools: list[dict[str, Any]] = []
    for source in deepcopy(function_documents):
        source["parameters"] = _normalize_schema(source["parameters"])
        source["parameters"]["type"] = "object"
        tools.append({"type": "function", "function": source})
    return tools
