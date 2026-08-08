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
from xenolect.driver.ir import (
    Driver,
    FramedJsonToolCallsParser,
    NativeToolCallsParser,
    ToolCallFields,
    effective_protocol,
)


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

    matches: list[tuple[list[ToolCall], str | None]] = []
    errors: list[str] = []
    for primitive in effective_protocol(driver).response:
        if isinstance(primitive, NativeToolCallsParser):
            if not message.get("tool_calls"):
                continue
            parsed = _parse_native(message)
            if parsed.errors:
                errors.extend(parsed.errors)
            else:
                matches.append(
                    (_calls_from_events(parsed.events), _content_from_call_events(parsed.events))
                )
            continue
        if isinstance(primitive, FramedJsonToolCallsParser):
            parsed, matched, surrounding = _parse_framed_primitive(message, primitive)
            if parsed.errors:
                errors.extend(parsed.errors)
            elif matched:
                matches.append((_calls_from_events(parsed.events), surrounding))
            continue
        # Pydantic's discriminated union prevents this unless validation was
        # bypassed with model_copy/construct.
        errors.append(f"unsupported response primitive: {type(primitive).__name__}")

    if errors:
        return ParseResult(events=[], errors=errors, raw_message=message)
    if matches:
        signatures = {
            json.dumps(
                [call.model_dump(mode="json") for call in calls],
                sort_keys=True,
                separators=(",", ":"),
            )
            for calls, _ in matches
        }
        if len(signatures) != 1:
            return ParseResult(
                events=[],
                errors=[
                    "configured response primitives parsed different canonical tool calls; "
                    "refusing ambiguous output"
                ],
                raw_message=message,
            )
        calls = matches[0][0]
        surrounding = next((text for _, text in matches if text), None)
        return ParseResult(
            events=_events_for_calls(calls, content=surrounding),
            raw_message=message,
        )

    content = message.get("content")
    return ParseResult(
        events=[
            AssistantText(
                content=content if isinstance(content, str) else json.dumps(content)
            )
        ],
        raw_message=message,
    )


def _calls_from_events(events: list[Event]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for event in events:
        if isinstance(event, AssistantToolCall):
            calls.append(event.call)
        elif isinstance(event, ToolCallBatch):
            calls.extend(event.calls)
    return calls


def _content_from_call_events(events: list[Event]) -> str | None:
    for event in events:
        if isinstance(event, (AssistantToolCall, ToolCallBatch)) and event.content:
            return event.content
    return None


def _events_for_calls(calls: list[ToolCall], *, content: str | None = None) -> list[Event]:
    if len(calls) == 1:
        return [AssistantToolCall(call=calls[0], content=content)]
    return [ToolCallBatch(calls=calls, content=content)]


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
        assistant_content = content if isinstance(content, str) and content else None
        if len(calls) == 1:
            return ParseResult(
                events=[AssistantToolCall(call=calls[0], content=assistant_content)],
                raw_message=message,
            )
        return ParseResult(
            events=[ToolCallBatch(calls=calls, content=assistant_content)],
            raw_message=message,
        )

    return ParseResult(
        events=[
            AssistantText(
                content=content if isinstance(content, str) else json.dumps(content)
            )
        ],
        raw_message=message,
    )


def _parse_object_to_call(
    obj: Any,
    *,
    context: str,
    fields: ToolCallFields | None = None,
) -> tuple[ToolCall | None, str | None]:
    fields = fields or ToolCallFields()
    if not isinstance(obj, dict):
        return None, f"{context}: tool payload is not an object"
    if fields.name not in obj:
        return None, f"{context}: missing {fields.name}"
    name = obj.get(fields.name)
    if not isinstance(name, str) or not name.strip():
        return None, f"{context}: missing or empty tool name"
    args = obj.get(fields.arguments, {})
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
    call_id = obj.get(fields.call_id) if fields.call_id is not None else None
    if call_id is not None and not isinstance(call_id, str):
        return None, f"{context}: call id must be a string"
    return ToolCall(id=call_id, name=name, arguments=args), None


def _find_frame_token(
    text: str,
    token: str,
    start: int,
    *,
    case_sensitive: bool,
    flexible_whitespace: bool,
) -> tuple[int, int] | None:
    if not flexible_whitespace:
        if case_sensitive:
            position = text.find(token, start)
        else:
            match = re.search(re.escape(token), text[start:], flags=re.IGNORECASE)
            position = -1 if match is None else start + match.start()
        if position < 0:
            return None
        return position, position + len(token)

    parts = re.split(r"(\s+)", token)
    pattern = "".join(r"\s*" if part.isspace() else re.escape(part) for part in parts)
    flags = 0 if case_sensitive else re.IGNORECASE
    match = re.search(pattern, text[start:], flags=flags)
    if match is None:
        return None
    return start + match.start(), start + match.end()


def _parse_framed_primitive(
    message: dict[str, Any],
    primitive: FramedJsonToolCallsParser,
) -> tuple[ParseResult, bool, str | None]:
    """Return (result, whether this primitive matched, surrounding text)."""
    content = message.get("content") or ""
    if not isinstance(content, str):
        content = json.dumps(content)

    if primitive.whole_content:
        stripped = content.strip()
        if not stripped.startswith("{"):
            return ParseResult(raw_message=message), False, None
        try:
            obj = strict_json_loads(stripped)
        except json.JSONDecodeError:
            # A whole-content parser is an alternative, so arbitrary malformed
            # prose beginning with "{" is not automatically claimed as a frame.
            return ParseResult(raw_message=message), False, None
        if not isinstance(obj, dict) or primitive.fields.arguments not in obj:
            return ParseResult(raw_message=message), False, None
        call, error = _parse_object_to_call(
            obj,
            context="whole-content JSON tool call",
            fields=primitive.fields,
        )
        if error:
            return ParseResult(errors=[error], raw_message=message), True, None
        assert call is not None
        return ParseResult(events=_events_for_calls([call]), raw_message=message), True, None

    frame = primitive.frame
    idx = 0
    calls: list[ToolCall] = []
    consumed: list[tuple[int, int]] = []
    while True:
        prefix_match = _find_frame_token(
            content,
            frame.prefix,
            idx,
            case_sensitive=frame.case_sensitive,
            flexible_whitespace=frame.flexible_whitespace,
        )
        if prefix_match is None:
            break
        pos, body_start = prefix_match
        if calls and not primitive.multiple:
            return (
                ParseResult(
                    errors=["response primitive permits only one framed JSON tool call"],
                    raw_message=message,
                ),
                True,
                None,
            )
        json_start = body_start
        if frame.whitespace_after_prefix:
            while json_start < len(content) and content[json_start].isspace():
                json_start += 1
        if json_start >= len(content) or content[json_start] != "{":
            return (
                ParseResult(
                    errors=[
                        f"frame {frame.prefix!r} is not followed by a JSON object"
                    ],
                    raw_message=message,
                ),
                True,
                None,
            )
        blob, json_end, framing_error = extract_balanced_json(content, json_start)
        if framing_error or blob is None:
            return (
                ParseResult(
                    errors=[f"framed JSON tool call error: {framing_error}"],
                    raw_message=message,
                ),
                True,
                None,
            )
        try:
            obj = strict_json_loads(blob)
        except json.JSONDecodeError as exc:
            return (
                ParseResult(
                    errors=[f"framed tool payload JSON error: {exc.msg}"],
                    raw_message=message,
                ),
                True,
                None,
            )
        call, error = _parse_object_to_call(
            obj,
            context=f"frame {frame.prefix!r}",
            fields=primitive.fields,
        )
        if error:
            return ParseResult(errors=[error], raw_message=message), True, None
        assert call is not None

        frame_end = json_end
        if frame.suffix:
            suffix_match = _find_frame_token(
                content,
                frame.suffix,
                json_end,
                case_sensitive=frame.case_sensitive,
                flexible_whitespace=frame.flexible_whitespace,
            )
            if suffix_match is None:
                return (
                    ParseResult(
                        errors=[f"unclosed frame {frame.prefix!r}; missing {frame.suffix!r}"],
                        raw_message=message,
                    ),
                    True,
                    None,
                )
            suffix_pos, suffix_end = suffix_match
            if content[json_end:suffix_pos].strip():
                return (
                    ParseResult(
                        errors=[f"unexpected content before closing frame {frame.suffix!r}"],
                        raw_message=message,
                    ),
                    True,
                    None,
                )
            frame_end = suffix_end
        calls.append(call)
        consumed.append((pos, frame_end))
        idx = frame_end

    if not calls:
        return ParseResult(raw_message=message), False, None

    surrounding: str | None = None
    if primitive.capture_surrounding_text:
        pieces: list[str] = []
        cursor = 0
        for start, end in consumed:
            pieces.append(content[cursor:start])
            cursor = end
        pieces.append(content[cursor:])
        retained = "".join(pieces).strip()
        surrounding = retained or None
    return (
        ParseResult(events=_events_for_calls(calls, content=surrounding), raw_message=message),
        True,
        surrounding,
    )
