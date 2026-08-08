"""Normalized Tool ABI v0 event types."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

JSONValue = Any


class ToolDef(BaseModel):
    """Tool definition offered to the model."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ToolCall(BaseModel):
    """A single tool invocation in normalized form."""

    name: str
    arguments: JSONValue = Field(default_factory=dict)
    id: str | None = None


class UserMessage(BaseModel):
    type: Literal["user_message"] = "user_message"
    content: str
    tools: list[ToolDef] = Field(default_factory=list)


class AssistantText(BaseModel):
    type: Literal["assistant_text"] = "assistant_text"
    content: str


class AssistantToolCall(BaseModel):
    type: Literal["assistant_tool_call"] = "assistant_tool_call"
    call: ToolCall
    # Some model protocols put explanatory text beside a tool call.  Keeping it
    # on the same assistant event preserves the Tool ABI state transition.
    content: str | None = None


class ToolCallBatch(BaseModel):
    type: Literal["tool_call_batch"] = "tool_call_batch"
    calls: list[ToolCall]
    content: str | None = None

    def model_post_init(self, __context: Any) -> None:
        if not self.calls:
            raise ValueError("ToolCallBatch must contain at least one call")


class ToolResult(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    content: JSONValue
    call_id: str | None = None
    name: str | None = None


class ToolError(BaseModel):
    type: Literal["tool_error"] = "tool_error"
    error: str
    call_id: str | None = None
    name: str | None = None


Event = (
    UserMessage
    | AssistantText
    | AssistantToolCall
    | ToolCallBatch
    | ToolResult
    | ToolError
)


def event_type(event: Event) -> str:
    return event.type
