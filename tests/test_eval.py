"""Deterministic evaluator tests — interface vs semantic split."""

from xenolect.abi.events import (
    AssistantText,
    AssistantToolCall,
    ToolCall,
    ToolDef,
    ToolResult,
    UserMessage,
)
from xenolect.eval.evaluator import FailureCategory, ProbeExpectation, evaluate_trace


def _weather_tools():
    return [
        ToolDef(
            name="get_weather",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        ),
        ToolDef(
            name="calculator",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        ),
    ]


def test_schema_failure_is_interface():
    tools = _weather_tools()
    events = [
        UserMessage(content="w", tools=tools),
        AssistantToolCall(call=ToolCall(id="1", name="get_weather", arguments={})),
    ]
    r = evaluate_trace(
        events,
        tools=tools,
        expectation=ProbeExpectation(required_tool_names=["get_weather"]),
        kind="protocol",
    )
    assert not r.passed
    assert not r.interface_ok
    assert r.category == FailureCategory.SCHEMA_REPRESENTATION


def test_protocol_expectation_pass():
    tools = _weather_tools()
    events = [
        UserMessage(content='Call get_weather exactly once with city="Paris".', tools=tools),
        AssistantToolCall(
            call=ToolCall(id="1", name="get_weather", arguments={"city": "Paris"})
        ),
        ToolResult(call_id="1", content={"temp": 1}),
        AssistantText(content="done"),
    ]
    r = evaluate_trace(
        events,
        tools=tools,
        expectation=ProbeExpectation(
            required_tool_names=["get_weather"],
            argument_subsets=[("get_weather", {"city": "Paris"})],
            min_tool_cycles=1,
            require_final_text=True,
        ),
        kind="protocol",
    )
    assert r.passed
    assert r.interface_ok


def test_wrong_tool_selection_is_semantic_not_interface():
    """Wrong tool among offered set with schema-valid args → semantic only."""
    tools = _weather_tools()
    events = [
        UserMessage(content="What is the weather in Paris?", tools=tools),
        AssistantToolCall(
            call=ToolCall(id="1", name="calculator", arguments={"expression": "1+1"})
        ),
        ToolResult(call_id="1", content={"value": 2}),
        AssistantText(content="2"),
    ]
    r = evaluate_trace(
        events,
        tools=tools,
        expectation=ProbeExpectation(
            expected_primary_tool="get_weather",
            required_tool_names=["get_weather"],
            min_tool_cycles=1,
        ),
        kind="semantic",
    )
    assert not r.passed
    assert r.interface_ok is True  # protocol structure/schema OK
    assert r.semantic_ok is False
    assert r.category == FailureCategory.SEMANTIC_TOOL_SELECTION


def test_malformed_args_interface_even_if_right_tool():
    """Correct intended tool + schema-invalid args → interface failure."""
    tools = _weather_tools()
    events = [
        UserMessage(content='Call get_weather with city=["Paris"]', tools=tools),
        AssistantToolCall(
            call=ToolCall(id="1", name="get_weather", arguments={"city": ["Paris"]})
        ),
    ]
    r = evaluate_trace(
        events,
        tools=tools,
        expectation=ProbeExpectation(required_tool_names=["get_weather"]),
        kind="protocol",
    )
    assert not r.passed
    assert r.interface_ok is False
    assert r.category == FailureCategory.SCHEMA_REPRESENTATION


def test_protocol_missing_call_is_interface():
    tools = _weather_tools()
    events = [
        UserMessage(content='Call get_weather exactly once with city="Paris".', tools=tools),
        AssistantText(content="I refuse."),
    ]
    r = evaluate_trace(
        events,
        tools=tools,
        expectation=ProbeExpectation(
            required_tool_names=["get_weather"],
            require_any_tool_call=True,
        ),
        kind="protocol",
    )
    assert not r.interface_ok
    assert r.category in (
        FailureCategory.PROTOCOL_EXPECTATION,
        FailureCategory.RESPONSE_PARSING,
    )
