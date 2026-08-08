"""Legal-trace validation tests."""

from xenolect.abi.events import (
    AssistantText,
    AssistantToolCall,
    ToolCall,
    ToolDef,
    ToolResult,
    UserMessage,
)
from xenolect.abi.trace import TraceState, validate_trace


def test_simple_text_turn():
    events = [
        UserMessage(content="hi", tools=[]),
        AssistantText(content="hello"),
    ]
    v = validate_trace(events)
    assert v.legal
    assert v.final_state == TraceState.CHAT


def test_tool_call_cycle():
    tools = [ToolDef(name="get_weather", parameters={"type": "object", "properties": {}})]
    events = [
        UserMessage(content="weather?", tools=tools),
        AssistantToolCall(call=ToolCall(id="c1", name="get_weather", arguments={"city": "Paris"})),
        ToolResult(call_id="c1", name="get_weather", content={"temp": 20}),
        AssistantText(content="20C"),
    ]
    v = validate_trace(events)
    assert v.legal
    assert v.final_state == TraceState.CHAT


def test_empty_trace_illegal():
    v = validate_trace([])
    assert not v.legal


def test_tool_before_user_illegal():
    events = [AssistantText(content="nope")]
    v = validate_trace(events)
    assert not v.legal


def test_unknown_call_id():
    tools = [ToolDef(name="t", parameters={"type": "object", "properties": {}})]
    events = [
        UserMessage(content="x", tools=tools),
        AssistantToolCall(call=ToolCall(id="c1", name="t", arguments={})),
        ToolResult(call_id="wrong", name="t", content={}),
    ]
    v = validate_trace(events)
    assert not v.legal
