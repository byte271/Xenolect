"""Parser and encode tests."""

from xenolect.abi.events import AssistantToolCall, ToolDef, ToolResult
from xenolect.driver.encode import encode_tool_result_message, tools_for_request
from xenolect.driver.ir import Driver, ParserKind, ToolEncoding, ToolResultEncoding
from xenolect.driver.parse import parse_model_response


def test_parse_native_tool_call():
    raw = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Paris"}',
                            },
                        }
                    ],
                }
            }
        ]
    }
    events = parse_model_response(raw, Driver())
    assert len(events) == 1
    assert isinstance(events[0], AssistantToolCall)
    assert events[0].call.name == "get_weather"
    assert events[0].call.arguments["city"] == "Paris"


def test_parse_xml():
    payload = '{"name": "get_weather", "arguments": {"city": "Paris"}, "id": "x1"}'
    raw = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": f"<tool_call>{payload}</tool_call>",
                }
            }
        ]
    }
    driver = Driver(tool_encoding=ToolEncoding.XML_JSON, parser=ParserKind.XML_JSON)
    events = parse_model_response(raw, driver)
    assert isinstance(events[0], AssistantToolCall)
    assert events[0].call.id == "x1"


def test_tool_result_encodings():
    tr = ToolResult(call_id="c1", name="t", content={"a": 1})
    native = encode_tool_result_message(
        tr, Driver(tool_result_encoding=ToolResultEncoding.TOOL_ROLE)
    )
    assert native["role"] == "tool"
    assert native["tool_call_id"] == "c1"

    user = encode_tool_result_message(
        tr, Driver(tool_result_encoding=ToolResultEncoding.USER_MESSAGE)
    )
    assert user["role"] == "user"
    assert "TOOL_RESULT" in user["content"]


def test_tools_for_request_inline():
    tool = ToolDef(
        name="submit_item",
        parameters={
            "type": "object",
            "properties": {"item": {"$ref": "#/$defs/Item"}},
            "$defs": {"Item": {"type": "object", "properties": {"id": {"type": "integer"}}}},
        },
    )
    from xenolect.driver.ir import SchemaTransform

    driver = Driver(schema_transforms=[SchemaTransform.INLINE_REFS])
    wire = tools_for_request([tool], driver)
    params = wire[0]["function"]["parameters"]
    assert "$ref" not in str(params)
