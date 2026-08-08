"""Release-level checks that the installed Driver is real protocol machinery.

These tests deliberately avoid claiming arbitrary protocol synthesis.  v0.1.0
compiles one Driver from a finite typed grammar and the proxy must then apply
that Driver to *client-supplied* tools and tool-result history.
"""

from xenolect.driver.ir import Driver, ParserKind, ToolEncoding, ToolResultEncoding, driver_grammar_size
from xenolect.proxy import translate_request, translate_response
from xenolect.xpt.planner import all_request_configs


def test_v010_driver_grammar_is_explicitly_finite_144_programs() -> None:
    # 3 request encodings * 8 transform subsets = 24 request configurations.
    assert len(all_request_configs()) == 24
    # Parser and result encoding are inferred later from stateful observations.
    assert 24 * len(ParserKind) * len(ToolResultEncoding) == 144
    assert driver_grammar_size() == 144


def test_installed_driver_rewrites_actual_client_tool_protocol() -> None:
    driver = Driver(
        tool_encoding=ToolEncoding.XML_JSON,
        parser=ParserKind.XML_JSON,
        tool_result_encoding=ToolResultEncoding.USER_MESSAGE,
    )
    client_body = {
        "model": "m",
        "messages": [{"role": "user", "content": "look it up"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "lookup a key",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                },
            }
        ],
    }

    messages, wire_tools, kwargs = translate_request(client_body, driver, "m")
    # The client supplied a real tool, but this Driver deliberately converts it
    # to model-visible XML/text rather than forwarding native OpenAI tools.
    assert wire_tools is None
    assert kwargs["model"] == "m"
    assert messages[0]["role"] == "system"
    assert "<tool_call>" in messages[0]["content"]
    assert '"name": "lookup"' in messages[0]["content"]

    upstream = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '<tool_call>{"name":"lookup","arguments":{"key":"x"},"id":"c1"}</tool_call>',
                }
            }
        ]
    }
    normalized = translate_response(upstream, driver, "m")
    call = normalized["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "lookup"
    assert call["id"] == "c1"

    followup = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "look it up"},
            normalized["choices"][0]["message"],
            {"role": "tool", "tool_call_id": "c1", "content": '{"value":42}'},
        ],
        "tools": client_body["tools"],
    }
    messages2, _, _ = translate_request(followup, driver, "m")
    # The assistant history and returned tool result are both rewritten according
    # to the compiled Driver, proving the artifact is used at runtime.
    assert any(
        m.get("role") == "assistant" and "<tool_call>" in str(m.get("content"))
        for m in messages2
    )
    assert any(
        m.get("role") == "user" and str(m.get("content", "")).startswith("TOOL_RESULT")
        for m in messages2
    )
