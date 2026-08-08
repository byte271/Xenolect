"""Bounded local synthesis of response-parser primitives from paid observations.

The discovery pass is intentionally small and structural.  It does not add a
format name or ask the endpoint another question.  It examines strict JSON
objects already present in one response, infers field mappings from the offered
tool names and the challenge arguments, derives adjacent literal framing when
present, and validates every candidate through the production parser.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from xenolect.abi.events import AssistantToolCall, ToolCall, ToolCallBatch
from xenolect.driver.ir import (
    Driver,
    FramedJsonToolCallsParser,
    JsonObjectToolCallsParser,
    NativeToolsRequest,
    ProtocolProgram,
    ResponsePrimitive,
    ResultField,
    TextFrame,
    ToolCallFields,
    ToolResultMessage,
)
from xenolect.driver.parse import iter_json_objects, parse_model_response_full, strict_json_loads

MAX_DISCOVERY_CONTENT_CHARS = 65_536
MAX_DISCOVERY_OBJECTS = 16
MAX_DISCOVERY_FIELDS = 32
MAX_DISCOVERY_FIELD_PAIRS = 256
MAX_FRAME_TOKEN_CHARS = 128


@dataclass(frozen=True)
class ResponseParserDiscovery:
    """Result of one deterministic response-parser synthesis attempt."""

    parser: ResponsePrimitive | None = None
    calls: tuple[ToolCall, ...] = ()
    candidates_validated: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.parser is not None and not self.error and bool(self.calls)


def _message_content(raw: dict[str, Any]) -> str | None:
    message: Any = None
    if "choices" in raw:
        choices = raw.get("choices") or []
        if choices:
            message = choices[0].get("message")
    elif isinstance(raw.get("message"), dict):
        message = raw["message"]
    elif "role" in raw:
        message = raw
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def _arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return strict_json_loads(value)
        except json.JSONDecodeError:
            return object()
    return value


def _tool_objects(
    objects: list[tuple[int, int, Any]],
    *,
    offered_tool_names: set[str],
    expected_arguments: dict[str, Any],
) -> tuple[str, str, list[tuple[int, int, dict[str, Any]]]] | str:
    """Infer one unambiguous name/arguments field pair and its objects."""
    pair_matches: dict[
        tuple[str, str], dict[str, list[tuple[int, int, dict[str, Any]]]]
    ] = {}
    field_pairs_seen = 0
    for start, end, value in objects:
        if not isinstance(value, dict):
            continue
        if len(value) > MAX_DISCOVERY_FIELDS:
            return "response JSON object exceeds the bounded field limit"
        name_fields = sorted(
            key
            for key, item in value.items()
            if isinstance(key, str)
            and isinstance(item, str)
            and item in offered_tool_names
            and item in expected_arguments
        )
        for name_field in name_fields:
            tool_name = value[name_field]
            for arguments_field in sorted(str(key) for key in value if key != name_field):
                field_pairs_seen += 1
                if field_pairs_seen > MAX_DISCOVERY_FIELD_PAIRS:
                    return "response field inference exceeds the bounded candidate limit"
                if _arguments(value.get(arguments_field)) != expected_arguments[tool_name]:
                    continue
                pair_matches.setdefault((name_field, arguments_field), {}).setdefault(
                    tool_name, []
                ).append((start, end, value))

    expected_names = set(expected_arguments)
    complete = [
        (pair, by_name)
        for pair, by_name in pair_matches.items()
        if set(by_name) == expected_names
        and all(len(matches) == 1 for matches in by_name.values())
    ]
    if not complete:
        if any(set(by_name) == expected_names for by_name in pair_matches.values()):
            return "ambiguous repeated tool-call objects; refusing to guess"
        return "no strict JSON field mapping matches the observed tool-call challenge"
    if len(complete) != 1:
        return "ambiguous response field mapping; refusing to guess"
    (name_field, arguments_field), by_name = complete[0]
    selected = sorted(
        (matches[0] for matches in by_name.values()), key=lambda item: item[0]
    )
    return name_field, arguments_field, selected


def _call_id_field(
    selected: list[tuple[int, int, dict[str, Any]]],
    *,
    name_field: str,
    arguments_field: str,
) -> str | None | object:
    common = set(selected[0][2])
    for _, _, obj in selected[1:]:
        common &= set(obj)
    candidates: list[str] = []
    for key in sorted(str(k) for k in common - {name_field, arguments_field}):
        values = [obj.get(key) for _, _, obj in selected]
        if (
            all(isinstance(value, str) and bool(value.strip()) for value in values)
            and len(set(values)) == len(values)
        ):
            candidates.append(key)
    if len(candidates) > 1:
        return _AMBIGUOUS
    return candidates[0] if candidates else None


_AMBIGUOUS = object()


def _adjacent_token_before(content: str, start: int) -> tuple[str, bool] | None:
    before = content[:start]
    stripped = before.rstrip()
    gap = len(before) != len(stripped)
    tag = re.search(rf"(<[^<>\r\n]{{1,{MAX_FRAME_TOKEN_CHARS - 2}}}>)$", stripped)
    if tag is not None:
        return tag.group(1), gap
    match = re.search(rf"([^\s{{}}]{{1,{MAX_FRAME_TOKEN_CHARS}}})$", stripped)
    if match is None:
        return None
    token = match.group(1)
    if not any(not (ch.isalnum() or ch in "_-.") for ch in token):
        return None
    return token, gap


def _adjacent_token_after(content: str, end: int) -> str | None:
    after = content[end:]
    stripped = after.lstrip()
    tag = re.match(rf"(<[^<>\r\n]{{1,{MAX_FRAME_TOKEN_CHARS - 2}}}>)", stripped)
    if tag is not None:
        return tag.group(1)
    match = re.match(rf"([^\s{{}}]{{1,{MAX_FRAME_TOKEN_CHARS}}})", stripped)
    if match is None:
        return None
    token = match.group(1)
    if not any(not (ch.isalnum() or ch in "_-.") for ch in token):
        return None
    return token


def _candidate_parsers(
    content: str,
    selected: list[tuple[int, int, dict[str, Any]]],
    fields: ToolCallFields,
) -> list[ResponsePrimitive]:
    candidates: list[ResponsePrimitive] = []
    multiple = len(selected) > 1
    first_start, first_end, _ = selected[0]
    if (
        len(selected) == 1
        and not content[:first_start].strip()
        and not content[first_end:].strip()
    ):
        candidates.append(
            FramedJsonToolCallsParser(
                frame=TextFrame(),
                fields=fields,
                multiple=False,
                whole_content=True,
            )
        )

    prefixes = [_adjacent_token_before(content, start) for start, _, _ in selected]
    if all(item is not None for item in prefixes):
        prefix_values = {item[0] for item in prefixes if item is not None}
        whitespace_values = {item[1] for item in prefixes if item is not None}
        if len(prefix_values) == 1 and len(whitespace_values) == 1:
            prefix = next(iter(prefix_values))
            suffixes = [_adjacent_token_after(content, end) for _, end, _ in selected]
            suffix_values = {suffix for suffix in suffixes if suffix is not None}
            suffix: str | None
            if all(item is None for item in suffixes):
                suffix = ""
            elif len(suffix_values) == 1 and all(suffixes):
                suffix = next(iter(suffix_values))
            else:
                suffix = None
            if suffix is not None:
                candidates.append(
                    FramedJsonToolCallsParser(
                        frame=TextFrame(
                            prefix=prefix,
                            suffix=suffix,
                            whitespace_after_prefix=next(iter(whitespace_values)),
                        ),
                        fields=fields,
                        multiple=multiple,
                        capture_surrounding_text=True,
                    )
                )

    candidates.append(
        JsonObjectToolCallsParser(
            fields=fields,
            multiple=multiple,
            capture_surrounding_text=True,
        )
    )
    # Model equality is semantic here; preserve stable first-seen ranking.
    unique: list[ResponsePrimitive] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = json.dumps(candidate.model_dump(mode="json"), sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _validation_driver(parser: ResponsePrimitive) -> Driver:
    return Driver(
        ir_version="0.2",
        protocol=ProtocolProgram(
            request=[NativeToolsRequest()],
            response=[parser],
            tool_result=ToolResultMessage(
                role="tool",
                segments=[ResultField(field="content")],
                attach_tool_call_id=True,
            ),
        ),
    )


def _parsed_calls(raw: dict[str, Any], parser: ResponsePrimitive) -> tuple[ToolCall, ...] | None:
    parsed = parse_model_response_full(raw, _validation_driver(parser))
    if parsed.errors:
        return None
    calls: list[ToolCall] = []
    for event in parsed.events:
        if isinstance(event, AssistantToolCall):
            calls.append(event.call)
        elif isinstance(event, ToolCallBatch):
            calls.extend(event.calls)
    return tuple(calls) if calls else None


def discover_response_parser(
    raw: dict[str, Any],
    *,
    offered_tool_names: set[str],
    expected_arguments: dict[str, Any],
) -> ResponseParserDiscovery:
    """Synthesize one parser primitive from a single already-paid response.

    Candidate extraction and validation are bounded by fixed content, object,
    field, and field-pair limits.  Any structural ambiguity returns an explicit
    failure instead of selecting a convenient interpretation.
    """
    content = _message_content(raw)
    if content is None or not content.strip():
        return ResponseParserDiscovery(error="response has no textual content to inspect")
    if len(content) > MAX_DISCOVERY_CONTENT_CHARS:
        return ResponseParserDiscovery(
            error="response content exceeds the bounded discovery size limit"
        )
    objects = iter_json_objects(content)
    if len(objects) > MAX_DISCOVERY_OBJECTS:
        return ResponseParserDiscovery(
            error="response contains too many JSON objects for bounded discovery"
        )
    if not objects:
        return ResponseParserDiscovery(error="response contains no strict JSON object")

    inferred = _tool_objects(
        objects,
        offered_tool_names=offered_tool_names,
        expected_arguments=expected_arguments,
    )
    if isinstance(inferred, str):
        return ResponseParserDiscovery(error=inferred)
    name_field, arguments_field, selected = inferred
    call_id_field = _call_id_field(
        selected,
        name_field=name_field,
        arguments_field=arguments_field,
    )
    if call_id_field is _AMBIGUOUS:
        return ResponseParserDiscovery(
            error="ambiguous response call-id field; refusing to guess"
        )
    assert call_id_field is None or isinstance(call_id_field, str)
    fields = ToolCallFields(
        name=name_field,
        arguments=arguments_field,
        call_id=call_id_field,
    )

    valid: list[tuple[ResponsePrimitive, tuple[ToolCall, ...]]] = []
    for parser in _candidate_parsers(content, selected, fields):
        calls = _parsed_calls(raw, parser)
        if calls is None:
            continue
        if [call.name for call in calls] != [obj[name_field] for _, _, obj in selected]:
            continue
        if any(call.arguments != expected_arguments.get(call.name) for call in calls):
            continue
        valid.append((parser, calls))

    if not valid:
        return ResponseParserDiscovery(
            error="inferred response parser candidates did not validate against the observation"
        )
    signatures = {
        json.dumps(
            [call.model_dump(mode="json") for call in calls],
            sort_keys=True,
            separators=(",", ":"),
        )
        for _, calls in valid
    }
    if len(signatures) != 1:
        return ResponseParserDiscovery(
            candidates_validated=len(valid),
            error="response parser candidates disagree on canonical tool calls",
        )
    # Candidate construction is ordered from most structurally constrained
    # (whole content / literal frame) to the generic embedded-object scanner.
    parser, calls = valid[0]
    return ResponseParserDiscovery(
        parser=parser,
        calls=calls,
        candidates_validated=len(valid),
    )
