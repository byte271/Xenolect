"""True parallel ToolCallBatch protocol tests (Repair 2)."""

from __future__ import annotations

from typing import Any

from xenolect.abi.events import (
    AssistantText,
    ToolCall,
    ToolCallBatch,
    ToolDef,
    ToolError,
    ToolResult,
    UserMessage,
)
from xenolect.abi.trace import TraceState, validate_trace
from xenolect.driver.ir import identity_driver
from xenolect.driver.runtime import DriverRuntime


class CountingParallelMock:
    """Emits a 2-call batch, then final text only after both tool results exist."""

    def __init__(self) -> None:
        self.calls = 0
        self.tool_results_seen_at_resume: list[int] = []

    def chat_completions(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls += 1
        n_tool = sum(1 for m in messages if m.get("role") == "tool")
        self.tool_results_seen_at_resume.append(n_tool)

        if n_tool >= 2:
            return {
                "choices": [
                    {"message": {"role": "assistant", "content": "both done"}}
                ]
            }

        if n_tool == 1:
            # If runtime incorrectly resumes after one result, record and stall.
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "ERROR: premature resume with only one result",
                        }
                    }
                ]
            }

        # First turn: parallel batch
        import json

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "a1",
                                "type": "function",
                                "function": {
                                    "name": "alpha",
                                    "arguments": json.dumps({"x": 1}),
                                },
                            },
                            {
                                "id": "b1",
                                "type": "function",
                                "function": {
                                    "name": "beta",
                                    "arguments": json.dumps({"y": 2}),
                                },
                            },
                        ],
                    }
                }
            ]
        }


def _tools() -> list[ToolDef]:
    return [
        ToolDef(
            name="alpha",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        ),
        ToolDef(
            name="beta",
            parameters={
                "type": "object",
                "properties": {"y": {"type": "integer"}},
                "required": ["y"],
            },
        ),
    ]


def test_case_a_two_parallel_successful_no_premature_resume():
    mock = CountingParallelMock()
    rt = DriverRuntime(driver=identity_driver(), client=mock)
    user = UserMessage(content="run both", tools=_tools())
    trace = rt.run_probe_script(
        user,
        tool_executor={
            "alpha": lambda x: {"ok": x},
            "beta": lambda y: {"ok": y},
        },
    )
    # First model call (batch) + one resume after both results = 2
    assert mock.calls == 2
    assert mock.tool_results_seen_at_resume[-1] == 2
    # Never resumed with exactly one tool result
    assert 1 not in mock.tool_results_seen_at_resume
    assert any(isinstance(e, AssistantText) and e.content == "both done" for e in trace)


def test_case_b_results_in_call_order_legal():
    tools = _tools()
    events = [
        UserMessage(content="x", tools=tools),
        ToolCallBatch(
            calls=[
                ToolCall(id="a1", name="alpha", arguments={"x": 1}),
                ToolCall(id="b1", name="beta", arguments={"y": 2}),
            ]
        ),
        ToolResult(call_id="a1", name="alpha", content=1),
        ToolResult(call_id="b1", name="beta", content=2),
        AssistantText(content="ok"),
    ]
    v = validate_trace(events)
    assert v.legal
    assert v.final_state == TraceState.CHAT


def test_case_c_results_reordered_legal():
    tools = _tools()
    events = [
        UserMessage(content="x", tools=tools),
        ToolCallBatch(
            calls=[
                ToolCall(id="a1", name="alpha", arguments={"x": 1}),
                ToolCall(id="b1", name="beta", arguments={"y": 2}),
            ]
        ),
        ToolResult(call_id="b1", name="beta", content=2),
        ToolResult(call_id="a1", name="alpha", content=1),
        AssistantText(content="ok"),
    ]
    v = validate_trace(events)
    assert v.legal


def test_case_d_success_plus_error_legal():
    tools = _tools()
    events = [
        UserMessage(content="x", tools=tools),
        ToolCallBatch(
            calls=[
                ToolCall(id="a1", name="alpha", arguments={"x": 1}),
                ToolCall(id="b1", name="beta", arguments={"y": 2}),
            ]
        ),
        ToolResult(call_id="a1", name="alpha", content=1),
        ToolError(call_id="b1", name="beta", error="boom"),
        AssistantText(content="partial"),
    ]
    v = validate_trace(events)
    assert v.legal


def test_case_e_missing_result_not_ready_to_resume():
    from xenolect.driver.runtime import _calls_from

    rt = DriverRuntime(driver=identity_driver(), client=CountingParallelMock())
    ev = rt.handle_user(UserMessage(content="run both", tools=_tools()))
    batch = _calls_from(ev)
    assert len(batch) == 2
    assert not rt.ready_to_resume()
    rt.append_tool_result(ToolResult(call_id=batch[0].id, name=batch[0].name, content={}))
    assert not rt.ready_to_resume()  # still missing one
    out = [] if not rt.ready_to_resume() else rt.resume_model()
    assert out == []
    assert not rt.ready_to_resume()


def test_case_f_duplicate_result_id_illegal():
    tools = _tools()
    events = [
        UserMessage(content="x", tools=tools),
        ToolCallBatch(
            calls=[
                ToolCall(id="a1", name="alpha", arguments={"x": 1}),
                ToolCall(id="b1", name="beta", arguments={"y": 2}),
            ]
        ),
        ToolResult(call_id="a1", name="alpha", content=1),
        ToolResult(call_id="a1", name="alpha", content=1),  # duplicate
    ]
    v = validate_trace(events)
    assert not v.legal


def test_case_g_unknown_call_id_illegal():
    tools = _tools()
    events = [
        UserMessage(content="x", tools=tools),
        ToolCallBatch(
            calls=[
                ToolCall(id="a1", name="alpha", arguments={"x": 1}),
                ToolCall(id="b1", name="beta", arguments={"y": 2}),
            ]
        ),
        ToolResult(call_id="zzz", name="alpha", content=1),
    ]
    v = validate_trace(events)
    assert not v.legal


def test_handle_tool_result_defers_resume_until_batch_complete():
    from xenolect.driver.runtime import _calls_from

    mock = CountingParallelMock()
    rt = DriverRuntime(driver=identity_driver(), client=mock)
    events = rt.handle_user(UserMessage(content="run both", tools=_tools()))
    batch = _calls_from(events)
    assert len(batch) == 2
    # First result alone: no model resume
    out1 = rt.handle_tool_result(
        ToolResult(call_id=batch[0].id, name=batch[0].name, content={"a": 1})
    )
    assert out1 == []
    assert mock.calls == 1  # only initial user turn
    # Second result: resume once
    out2 = rt.handle_tool_result(
        ToolResult(call_id=batch[1].id, name=batch[1].name, content={"b": 2})
    )
    assert out2
    assert mock.calls == 2
    assert mock.tool_results_seen_at_resume[-1] == 2
