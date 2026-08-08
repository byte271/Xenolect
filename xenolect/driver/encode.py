"""Request-side encoding: ABI tools/messages → model wire payload."""

from __future__ import annotations

import json
from typing import Any

from xenolect.abi.events import ToolDef, ToolResult
from xenolect.driver.ir import Driver, ToolEncoding, ToolResultEncoding
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


def build_system_tool_preamble(tools: list[ToolDef], driver: Driver) -> str | None:
    """For non-native encodings, inject a textual tool description."""
    transformed = [transform_tool_def(t, driver) for t in tools]
    if driver.tool_encoding == ToolEncoding.NATIVE:
        return None
    if driver.tool_encoding == ToolEncoding.TAGGED_JSON:
        payload = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in transformed
        ]
        return (
            "You may call tools by emitting a line exactly like:\n"
            "TOOL_CALL {" + '"name"' + ': "<tool>", "arguments": {..}, "id": "<id>"}\n'
            "Available tools (JSON):\n"
            + json.dumps(payload, indent=2)
        )
    if driver.tool_encoding == ToolEncoding.XML_JSON:
        payload = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in transformed
        ]
        return (
            "You may call tools using:\n"
            "<tool_call>{\"name\": \"...\", \"arguments\": {}, \"id\": \"...\"}</tool_call>\n"
            "Available tools:\n"
            + json.dumps(payload, indent=2)
        )
    return None


def encode_tool_result_message(
    result: ToolResult,
    driver: Driver,
) -> dict[str, Any]:
    """Encode a tool result into a chat message for the model."""
    content = result.content
    if not isinstance(content, str):
        content = json.dumps(content)

    if driver.tool_result_encoding == ToolResultEncoding.TOOL_ROLE:
        # Identity/OpenAI-compatible baseline: role=tool + tool_call_id + content.
        # Do not send provider-optional `name` as part of the Identity wire contract.
        # Anonymous model calls (ABI-legal) still need a tool_call_id on many
        # OpenAI-compatible servers — synthesize a stable local id when missing.
        msg: dict[str, Any] = {
            "role": "tool",
            "content": content,
        }
        call_id = result.call_id
        if call_id is None:
            # Deterministic enough for a single conversation: name + content hash.
            import hashlib

            material = f"{result.name or ''}|{content}".encode()
            call_id = "anon_" + hashlib.sha256(material).hexdigest()[:12]
        msg["tool_call_id"] = call_id
        return msg

    # USER_MESSAGE encoding
    header = "TOOL_RESULT"
    if result.call_id is not None:
        header += f" id={result.call_id}"
    if result.name:
        header += f" name={result.name}"
    return {
        "role": "user",
        "content": f"{header}\n{content}",
    }


def should_send_native_tools(driver: Driver) -> bool:
    return driver.tool_encoding == ToolEncoding.NATIVE
