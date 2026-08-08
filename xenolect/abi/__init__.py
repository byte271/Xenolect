"""Tool ABI v0: normalized events, tool defs, and legal-trace checking."""

from xenolect.abi.events import (
    AssistantText,
    AssistantToolCall,
    Event,
    ToolCall,
    ToolCallBatch,
    ToolDef,
    ToolError,
    ToolResult,
    UserMessage,
)
from xenolect.abi.trace import (
    TraceState,
    TraceValidation,
    validate_completed_trace,
    validate_trace,
)

ABI_VERSION = "tool-abi-v0"

__all__ = [
    "ABI_VERSION",
    "AssistantText",
    "AssistantToolCall",
    "Event",
    "ToolCall",
    "ToolCallBatch",
    "ToolDef",
    "ToolError",
    "ToolResult",
    "TraceState",
    "TraceValidation",
    "UserMessage",
    "validate_completed_trace",
    "validate_trace",
]
