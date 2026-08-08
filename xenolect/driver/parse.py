"""Response-side parsers: raw model output → normalized ABI tool events.

Soundness rules:
  - Strict JSON (no NaN/Infinity)
  - Malformed frames produce parse errors (not silently dropped)
  - No `_raw` success fallback for bad argument JSON
  - Balanced JSON scanner for nested/pretty payloads
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from xenolect.abi.events import AssistantText, AssistantToolCall, Event, ToolCall, ToolCallBatch
from xenolect.driver.ir import Driver, ParserKind


@dataclass
class ParseResult:
    events: list[Event] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    raw_message: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


def parse_model_response(raw: dict[str, Any], driver: Driver) -> list[Event]:
    """Backward-compatible: events only (errors discarded — prefer parse_model_response_full)."""
    return parse_model_response_full(raw, driver).events


def parse_model_response_full(raw: dict[str, Any], driver: Driver) -> ParseResult:
    message = _extract_message(raw)
    if message is None:
        return ParseResult(events=[AssistantText(content=str(raw))], raw_message=None)

    if driver.parser == ParserKind.NATIVE:
        return _parse_native(message)
    if driver.parser == ParserKind.TAGGED_JSON:
        return _parse_tagged_json(message)
    if driver.parser == ParserKind.XML_JSON:
        return _parse_xml_json(message)
    return ParseResult(
        events=[AssistantText(content=str(message.get("content") or ""))],
        raw_message=message,
    )


def strict_json_loads(s: str) -> Any:
    """JSON parse rejecting NaN/Infinity and non-standard constants."""

    def _reject_const(c: str) -> None:
        raise json.JSONDecodeError(f"non-standard JSON constant {c!r}", s, 0)

    return json.loads(s, parse_constant=_reject_const)


def extract_balanced_json(text: str, start: int) -> tuple[str | None, int, str | None]:
    """
    From text[start] which should be '{', extract a balanced JSON object.
    Returns (json_str, end_index, error).
    """
    if start >= len(text) or text[start] != "{":
        return None, start, "expected '{'"
    depth = 0
    in_str = False
    escape = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1], i + 1, None
        i += 1
    return None, start, "unbalanced JSON object"


def _extract_message(raw: dict[str, Any]) -> dict[str, Any] | None:
    if "choices" in raw:
        choices = raw.get("choices") or []
        if not choices:
            return None
        return choices[0].get("message") or {}
    if "message" in raw and isinstance(raw["message"], dict):
        return raw["message"]
    if "role" in raw:
        return raw
    return None


def _parse_args_strict(args_raw: Any) -> tuple[Any | None, str | None]:
    if isinstance(args_raw, (dict, list)):
        return args_raw, None
    if args_raw is None:
        return {}, None
    if isinstance(args_raw, str):
        if not args_raw.strip():
            return {}, None
        try:
            return strict_json_loads(args_raw), None
        except json.JSONDecodeError as exc:
            return None, f"malformed tool arguments JSON: {exc.msg}"
    return None, f"unsupported arguments type: {type(args_raw).__name__}"


def _parse_native(message: dict[str, Any]) -> ParseResult:
    tool_calls = message.get("tool_calls") or []
    content = message.get("content")
    errors: list[str] = []

    if tool_calls:
        calls: list[ToolCall] = []
        for i, tc in enumerate(tool_calls):
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments", "{}")
            args, err = _parse_args_strict(args_raw)
            if err:
                errors.append(f"tool_calls[{i}]: {err}")
                continue
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                errors.append(f"tool_calls[{i}]: missing or empty tool name")
                continue
            calls.append(
                ToolCall(
                    id=tc.get("id"),
                    name=name,
                    arguments=args if args is not None else {},
                )
            )
        if errors:
            return ParseResult(events=[], errors=errors, raw_message=message)
        if not calls:
            return ParseResult(
                events=[],
                errors=["native tool_calls present but none parseable"],
                raw_message=message,
            )
        if len(calls) == 1:
            return ParseResult(
                events=[AssistantToolCall(call=calls[0])], raw_message=message
            )
        return ParseResult(events=[ToolCallBatch(calls=calls)], raw_message=message)

    return ParseResult(
        events=[
            AssistantText(
                content=content if isinstance(content, str) else json.dumps(content)
            )
        ],
        raw_message=message,
    )


def _parse_object_to_call(obj: Any, *, context: str) -> tuple[ToolCall | None, str | None]:
    if not isinstance(obj, dict):
        return None, f"{context}: tool payload is not an object"
    if "name" not in obj:
        return None, f"{context}: missing name"
    name = obj.get("name")
    if not isinstance(name, str) or not name.strip():
        return None, f"{context}: missing or empty tool name"
    args = obj.get("arguments", {})
    if isinstance(args, str):
        parsed, err = _parse_args_strict(args)
        if err:
            return None, f"{context}: {err}"
        args = parsed
    if not isinstance(args, (dict, list)) and args is not None:
        return None, f"{context}: arguments must be object/array"
    # Preserve empty list arguments; only default when arguments is missing/None.
    if args is None:
        args = {}
    return (
        ToolCall(id=obj.get("id"), name=name, arguments=args),
        None,
    )


def _parse_tagged_json(message: dict[str, Any]) -> ParseResult:
    if message.get("tool_calls"):
        return _parse_native(message)
    content = message.get("content") or ""
    if not isinstance(content, str):
        content = json.dumps(content)

    errors: list[str] = []
    calls: list[ToolCall] = []
    # Find TOOL_CALL markers and extract balanced JSON
    idx = 0
    marker = "TOOL_CALL"
    found_marker = False
    while True:
        pos = content.find(marker, idx)
        if pos < 0:
            break
        found_marker = True
        j = pos + len(marker)
        while j < len(content) and content[j].isspace():
            j += 1
        if j >= len(content) or content[j] != "{":
            errors.append("TOOL_CALL marker without JSON object")
            idx = j
            continue
        blob, end, err = extract_balanced_json(content, j)
        if err or blob is None:
            errors.append(f"TOOL_CALL framing error: {err}")
            break
        try:
            obj = strict_json_loads(blob)
        except json.JSONDecodeError as exc:
            errors.append(f"TOOL_CALL payload JSON error: {exc.msg}")
            idx = end
            continue
        call, cerr = _parse_object_to_call(obj, context="TOOL_CALL")
        if cerr:
            errors.append(cerr)
        elif call:
            calls.append(call)
        idx = end

    if found_marker:
        if errors:
            return ParseResult(events=[], errors=errors, raw_message=message)
        if not calls:
            return ParseResult(
                events=[],
                errors=["TOOL_CALL markers produced no calls"],
                raw_message=message,
            )
        if len(calls) == 1:
            return ParseResult(
                events=[AssistantToolCall(call=calls[0])], raw_message=message
            )
        return ParseResult(events=[ToolCallBatch(calls=calls)], raw_message=message)

    # Whole-content JSON object — only when it is clearly a tool frame
    # (`arguments` present), not arbitrary prose JSON that happens to contain
    # a "name" key.
    stripped = content.strip()
    if stripped.startswith("{") and '"name"' in stripped and '"arguments"' in stripped:
        try:
            obj = strict_json_loads(stripped)
            if not isinstance(obj, dict) or "arguments" not in obj:
                return ParseResult(
                    events=[AssistantText(content=content)], raw_message=message
                )
            call, cerr = _parse_object_to_call(obj, context="content-json")
            if cerr:
                return ParseResult(events=[], errors=[cerr], raw_message=message)
            assert call is not None
            return ParseResult(
                events=[AssistantToolCall(call=call)], raw_message=message
            )
        except json.JSONDecodeError:
            return ParseResult(
                events=[AssistantText(content=content)],
                errors=[],  # plain text, not a tool frame
                raw_message=message,
            )

    return ParseResult(events=[AssistantText(content=content)], raw_message=message)


_XML_OPEN = re.compile(r"<tool_call\s*>", re.IGNORECASE)
_XML_CLOSE = re.compile(r"</tool_call\s*>", re.IGNORECASE)


def _parse_xml_json(message: dict[str, Any]) -> ParseResult:
    if message.get("tool_calls"):
        return _parse_native(message)
    content = message.get("content") or ""
    if not isinstance(content, str):
        content = json.dumps(content)

    errors: list[str] = []
    calls: list[ToolCall] = []
    idx = 0
    found = False
    while True:
        m = _XML_OPEN.search(content, idx)
        if not m:
            break
        found = True
        start_body = m.end()
        # skip whitespace
        j = start_body
        while j < len(content) and content[j].isspace():
            j += 1
        if j >= len(content) or content[j] != "{":
            # empty or non-json payload
            close = _XML_CLOSE.search(content, start_body)
            if close is None:
                errors.append("unclosed <tool_call> with empty/non-json payload")
                break
            payload = content[start_body : close.start()].strip()
            if not payload:
                errors.append("empty <tool_call> payload")
            else:
                errors.append("non-object <tool_call> payload")
            idx = close.end()
            continue
        blob, end, err = extract_balanced_json(content, j)
        if err or blob is None:
            errors.append(f"<tool_call> framing error: {err}")
            break
        try:
            obj = strict_json_loads(blob)
        except json.JSONDecodeError as exc:
            errors.append(f"<tool_call> payload JSON error: {exc.msg}")
            # still try to find closing tag
            close = _XML_CLOSE.search(content, end)
            idx = close.end() if close else end
            continue
        call, cerr = _parse_object_to_call(obj, context="<tool_call>")
        if cerr:
            errors.append(cerr)
        elif call:
            calls.append(call)
        close = _XML_CLOSE.search(content, end)
        if close is None:
            errors.append("unclosed <tool_call> after JSON payload")
            break
        # Only whitespace may appear between the JSON payload and the closing tag.
        trailing = content[end:close.start()]
        if trailing.strip():
            errors.append("unexpected content before </tool_call>")
        idx = close.end()

    if found:
        if errors:
            return ParseResult(events=[], errors=errors, raw_message=message)
        if not calls:
            return ParseResult(
                events=[],
                errors=["<tool_call> markers produced no calls"],
                raw_message=message,
            )
        if len(calls) == 1:
            return ParseResult(
                events=[AssistantToolCall(call=calls[0])], raw_message=message
            )
        return ParseResult(events=[ToolCallBatch(calls=calls)], raw_message=message)

    return ParseResult(events=[AssistantText(content=content)], raw_message=message)
