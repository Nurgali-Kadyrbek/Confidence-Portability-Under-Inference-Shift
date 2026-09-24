"""Pinned BFCL V3 multi-turn preparation, execution, and scoring adapter.

The adapter intentionally mirrors the pinned official ``BaseHandler`` loop:
tool calls execute against the official stateful simulator, their observations
are returned to the model, and a turn ends when the model emits no decodable
call. It does not reimplement benchmark state or scoring semantics.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from cpis.agent.tool_calls import (
    ToolCall,
    ToolCallParseError,
    bfcl_execution_strings,
    openai_tools,
    parse_tool_calls,
)


ADDITIONAL_FUNCTION_MESSAGE = (
    "I have updated some more functions you can choose from. What about now?"
)
FUNCTION_DOCUMENT_FILES = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}


def function_docs_bundle_sha256(function_docs_root: Path) -> str:
    """Hash the exact official function-document set used by BFCL episodes."""

    lines: list[str] = []
    for filename in sorted(FUNCTION_DOCUMENT_FILES.values()):
        path = function_docs_root / filename
        lines.append(f"{filename}\0{hashlib.sha256(path.read_bytes()).hexdigest()}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PreparedCase:
    source_case: dict[str, Any]
    initial_function_documents: tuple[dict[str, Any], ...]
    held_out_documents: dict[int, tuple[dict[str, Any], ...]]


@dataclass(frozen=True)
class AgentResponse:
    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class EpisodeStep:
    turn_index: int
    step_index: int
    sampling_seed: int
    messages_sha256: str
    tools_sha256: str
    raw_text: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    parse_status: Literal["valid_calls", "valid_no_call", "invalid"]
    parse_error: str | None
    calls: tuple[ToolCall, ...]
    execution_results: tuple[str, ...]


@dataclass(frozen=True)
class EpisodeResult:
    item_id: str
    force_terminated: bool
    invalid_tool_outputs: int
    turns_decoded: tuple[tuple[tuple[str, ...], ...], ...]
    steps: tuple[EpisodeStep, ...]
    final_messages: tuple[dict[str, Any], ...]


GenerateAgentResponse = Callable[
    [list[dict[str, Any]], list[dict[str, Any]], int], AgentResponse
]
ExecuteCalls = Callable[
    [list[str], dict[str, Any], list[str], str, str, bool, bool],
    tuple[list[str], dict[str, Any]],
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def prepare_case(source_case: dict[str, Any], function_docs_root: Path) -> PreparedCase:
    """Attach official class documents and apply BFCL missing-function holdouts."""

    case = deepcopy(source_case)
    documents: list[dict[str, Any]] = []
    for class_name in case["involved_classes"]:
        try:
            filename = FUNCTION_DOCUMENT_FILES[class_name]
        except KeyError as exc:
            raise ValueError(f"unsupported BFCL simulator class {class_name!r}") from exc
        documents.extend(_read_jsonl(function_docs_root / filename))

    by_name = {document["name"]: document for document in documents}
    if len(by_name) != len(documents):
        raise ValueError("BFCL function documents contain duplicate names")
    held_out: dict[int, tuple[dict[str, Any], ...]] = {}
    held_out_names: set[str] = set()
    for raw_turn, names in case.get("missed_function", {}).items():
        turn = int(raw_turn)
        try:
            selected = tuple(deepcopy(by_name[name]) for name in names)
        except KeyError as exc:
            raise ValueError(f"held-out BFCL function is absent: {exc.args[0]}") from exc
        held_out[turn] = selected
        held_out_names.update(names)
    initial = tuple(
        deepcopy(document)
        for document in documents
        if document["name"] not in held_out_names
    )
    return PreparedCase(
        source_case=case,
        initial_function_documents=initial,
        held_out_documents=held_out,
    )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _call_id(observation_id: str, turn: int, step: int, index: int) -> str:
    # Mistral v13+ requires a nine-character tool call ID.
    identity = f"{observation_id}\0{turn}\0{step}\0{index}".encode("utf-8")
    return hashlib.sha256(identity).hexdigest()[:9]


def _content_before_calls(text: str, family: Literal["qwen3", "mistral"]) -> str:
    marker = "<tool_call>" if family == "qwen3" else "[TOOL_CALLS]"
    return text.split(marker, 1)[0].strip()


def _assistant_message(
    raw_text: str,
    calls: list[ToolCall],
    family: Literal["qwen3", "mistral"],
    observation_id: str,
    turn: int,
    step: int,
) -> tuple[dict[str, Any], list[str]]:
    if not calls:
        return {"role": "assistant", "content": raw_text}, []
    call_ids = [
        _call_id(observation_id, turn, step, index)
        for index in range(len(calls))
    ]
    return (
        {
            "role": "assistant",
            "content": _content_before_calls(raw_text, family),
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for call_id, call in zip(call_ids, calls, strict=True)
            ],
        },
        call_ids,
    )


def _official_execute() -> ExecuteCalls:
    try:
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
            execute_multi_turn_func_call,
        )
    except ImportError as exc:
        raise RuntimeError(
            "the pinned BFCL checkout must be on PYTHONPATH for agent execution"
        ) from exc
    return execute_multi_turn_func_call


def run_episode(
    *,
    prepared: PreparedCase,
    observation_id: str,
    pairing_id: str | None = None,
    parser_family: Literal["qwen3", "mistral"],
    generate: GenerateAgentResponse,
    sampling_seed: Callable[[int, int], int],
    max_step_limit: int = 20,
    execute_calls: ExecuteCalls | None = None,
) -> EpisodeResult:
    """Run one BFCL episode using native tools and the official simulator.

    The official pinned loop increments its counter after executing a call and
    force-terminates when ``count > MAXIMUM_STEP_LIMIT``. This adapter retains
    that exact boundary behavior for reproducibility.
    """

    if max_step_limit != 20:
        raise ValueError("the pinned BFCL V3 protocol requires MAXIMUM_STEP_LIMIT=20")
    execute = _official_execute() if execute_calls is None else execute_calls
    case = prepared.source_case
    available_documents = list(deepcopy(prepared.initial_function_documents))
    messages: list[dict[str, Any]] = []
    steps: list[EpisodeStep] = []
    turns_decoded: list[tuple[tuple[str, ...], ...]] = []
    invalid_outputs = 0
    force_terminated = False
    category = case["id"].rsplit("_", 1)[0]
    long_context = "long_context" in category or "composite" in category
    tool_call_identity = pairing_id or observation_id

    # Initialize the official state namespace even if the first response has no call.
    execute(
        [],
        case["initial_config"],
        case["involved_classes"],
        observation_id,
        case["id"],
        long_context,
        False,
    )

    for turn_index, source_messages in enumerate(case["question"]):
        current_messages = deepcopy(source_messages)
        if turn_index in prepared.held_out_documents:
            if current_messages:
                raise ValueError("BFCL missing-function reveal turn must be empty")
            available_documents.extend(
                deepcopy(prepared.held_out_documents[turn_index])
            )
            current_messages = [
                {"role": "user", "content": ADDITIONAL_FUNCTION_MESSAGE}
            ]
        messages.extend(current_messages)
        turn_decoded: list[tuple[str, ...]] = []
        count = 0
        while True:
            tools = openai_tools(available_documents)
            seed = sampling_seed(turn_index, count)
            prompt_messages_sha256 = _canonical_sha256(messages)
            response = generate(deepcopy(messages), deepcopy(tools), seed)
            try:
                calls = parse_tool_calls(response.text, parser_family)
                parse_status: Literal[
                    "valid_calls", "valid_no_call", "invalid"
                ] = "valid_calls" if calls else "valid_no_call"
                parse_error = None
            except ToolCallParseError as exc:
                calls = []
                parse_status = "invalid"
                parse_error = str(exc)
                invalid_outputs += 1

            assistant_message, call_ids = _assistant_message(
                response.text,
                calls,
                parser_family,
                tool_call_identity,
                turn_index,
                count,
            )
            messages.append(assistant_message)
            execution_results: list[str] = []
            if calls:
                decoded = bfcl_execution_strings(calls)
                turn_decoded.append(tuple(decoded))
                execution_results, _ = execute(
                    decoded,
                    case["initial_config"],
                    case["involved_classes"],
                    observation_id,
                    case["id"],
                    long_context,
                    False,
                )
                if len(execution_results) != len(call_ids):
                    raise RuntimeError("BFCL simulator returned the wrong result count")
                for call, call_id, result in zip(
                    calls, call_ids, execution_results, strict=True
                ):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": call.name,
                            "content": result,
                        }
                    )

            steps.append(
                EpisodeStep(
                    turn_index=turn_index,
                    step_index=count,
                    sampling_seed=seed,
                    messages_sha256=prompt_messages_sha256,
                    tools_sha256=_canonical_sha256(tools),
                    raw_text=response.text,
                    finish_reason=response.finish_reason,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    parse_status=parse_status,
                    parse_error=parse_error,
                    calls=tuple(calls),
                    execution_results=tuple(execution_results),
                )
            )
            if parse_status != "valid_calls":
                break
            count += 1
            if count > max_step_limit:
                force_terminated = True
                break
        turns_decoded.append(tuple(turn_decoded))
        if force_terminated:
            break

    return EpisodeResult(
        item_id=case["id"],
        force_terminated=force_terminated,
        invalid_tool_outputs=invalid_outputs,
        turns_decoded=tuple(turns_decoded),
        steps=tuple(steps),
        final_messages=tuple(messages),
    )


def score_episode(
    *,
    episode: EpisodeResult,
    prepared: PreparedCase,
    evaluation_namespace: str,
) -> dict[str, Any]:
    """Score a saved decoded trajectory with the pinned official BFCL checker."""

    if episode.force_terminated:
        return {
            "valid": False,
            "error_type": "multi_turn:force_terminated",
            "error_message": "episode exceeded the pinned BFCL step limit",
        }
    if episode.invalid_tool_outputs:
        return {
            "valid": False,
            "error_type": "cpis:invalid_tool_output",
            "error_message": "at least one attempted tool call was malformed",
        }
    try:
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
            multi_turn_checker,
        )
        from bfcl_eval.utils import make_json_serializable
    except ImportError as exc:
        raise RuntimeError(
            "the pinned BFCL checkout must be on PYTHONPATH for scoring"
        ) from exc

    case = prepared.source_case
    ground_truth = case.get("ground_truth")
    if ground_truth is None:
        raise ValueError("prepared BFCL source case lacks attached ground_truth")
    decoded = [[list(step) for step in turn] for turn in episode.turns_decoded]
    if len(decoded) != len(ground_truth):
        raise ValueError("saved BFCL trajectory does not cover every benchmark turn")
    result = multi_turn_checker(
        decoded,
        ground_truth,
        case,
        case["id"].rsplit("_", 1)[0],
        evaluation_namespace,
    )
    # The official checker includes live simulator objects in some failure
    # diagnostics (for example GorillaFileSystem Directory instances).  Its
    # own result writer applies this exact normalizer before JSON encoding.
    # Preserve the official score while making the derived diagnostic payload
    # immutable and portable.
    return make_json_serializable(result)
