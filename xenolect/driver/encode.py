"""Request-side encoding: ABI tools/messages → model wire payload."""

from __future__ import annotations

import json
from typing import Any

from xenolect.abi.events import ToolDef, ToolResult
from xenolect.driver.ir import (
    Driver,
    JsonToolCatalogRequest,
    NativeToolsRequest,
    ResultField,
    ResultLiteral,
    TextFrame,
    ToolCallFields,
    effective_protocol,
)
from xenolect.driver.schema_ops import apply_schema_transforms


def transform_tool_def(tool: ToolDef, driver: Driver) -> ToolDef:
    params = apply_schema_transforms(tool.parameters, driver.schema_transforms)
    return ToolDef(name=tool.name, description=tool.description, parameters=params)


def tools_for_request(tools: list[ToolDef], driver: Driver) -> list[dict[str, Any]]:
    """OpenAI-style tools array after schema transforms."""
    out: list[dict[str, Any]] = []
    for t in tools:
        tt = transform_tool_def(t, driver)
        out.append(
            {
                "type": "function",
                "function": {
                    "name": tt.name,
                    "description": tt.description or "",
                    "parameters": tt.parameters,
                },
            }
        )
    return out


def build_tool_preamble_messages(
    tools: list[ToolDef], driver: Driver
) -> list[dict[str, Any]]:
    """Execute every textual tool-catalog request primitive."""
    transformed = [transform_tool_def(t, driver) for t in tools]
    messages: list[dict[str, Any]] = []
    for primitive in effective_protocol(driver).request:
        if not isinstance(primitive, JsonToolCatalogRequest):
            continue
        payload = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in transformed
        ]
        messages.append(
            {
                "role": primitive.role,
                "content": (
                    f"{primitive.instruction}\n{primitive.catalog_heading}\n"
                    + json.dumps(payload, indent=2)
                ),
            }
        )
    return messages


def build_system_tool_preamble(tools: list[ToolDef], driver: Driver) -> str | None:
    """Backward-compatible helper for the legacy single-system-preamble API."""
    messages = build_tool_preamble_messages(tools, driver)
    if not messages:
        return None
    if any(message["role"] != "system" for message in messages):
        raise ValueError(
            "protocol uses non-system tool catalogs; use build_tool_preamble_messages()"
        )
    return "\n\n".join(str(message["content"]) for message in messages)


def _frame_text(payload: str, frame: TextFrame) -> str:
    return f"{frame.prefix}{payload}{frame.suffix}"


def encode_textual_tool_call(
    *,
    name: str,
    arguments: Any,
    call_id: str | None,
    driver: Driver,
) -> str:
    """Encode one assistant-history call with the request program's text frame."""
    textual = [
        primitive
        for primitive in effective_protocol(driver).request
        if isinstance(primitive, JsonToolCatalogRequest)
    ]
    if len(textual) != 1:
        raise ValueError(
            "textual assistant call encoding requires exactly one JSON tool-catalog primitive"
        )
    primitive = textual[0]
    fields: ToolCallFields = primitive.fields
    obj: dict[str, Any] = {
        fields.name: name,
        fields.arguments: arguments,
    }
    if fields.call_id is not None and call_id is not None:
        obj[fields.call_id] = call_id
    payload = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    return _frame_text(payload, primitive.call_frame)


def encode_tool_result_message(
    result: ToolResult,
    driver: Driver,
) -> dict[str, Any]:
    """Encode a tool result into a chat message for the model."""
    content = result.content
    if not isinstance(content, str):
        content = json.dumps(content)

    program = effective_protocol(driver).tool_result
    values = {
        "call_id": result.call_id,
        "name": result.name,
        "content": content,
    }
    pieces: list[str] = []
    for segment in program.segments:
        if isinstance(segment, ResultLiteral):
            pieces.append(segment.text)
            continue
        assert isinstance(segment, ResultField)
        value = values[segment.field]
        if value is None and segment.omit_if_none:
            continue
        pieces.append(f"{segment.prefix}{'' if value is None else value}{segment.suffix}")

    msg: dict[str, Any] = {"role": program.role, "content": "".join(pieces)}
    if program.attach_tool_call_id:
        call_id = result.call_id
        if call_id is None:
            # Deterministic enough for a single conversation: name + content hash.
            import hashlib

            material = f"{result.name or ''}|{content}".encode()
            call_id = "anon_" + hashlib.sha256(material).hexdigest()[:12]
        msg["tool_call_id"] = call_id
    return msg


def should_send_native_tools(driver: Driver) -> bool:
    return any(
        isinstance(primitive, NativeToolsRequest)
        for primitive in effective_protocol(driver).request
    )


def uses_textual_tool_catalog(driver: Driver) -> bool:
    return any(
        isinstance(primitive, JsonToolCatalogRequest)
        for primitive in effective_protocol(driver).request
    )
