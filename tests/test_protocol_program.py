"""Composable Driver IR v0.2 execution and compatibility tests."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from xenolect.abi.events import AssistantToolCall, ToolDef, ToolResult
from xenolect.driver.encode import (
    build_tool_preamble_messages,
    encode_textual_tool_call,
    encode_tool_result_message,
)
from xenolect.driver.ir import (
    Driver,
    FramedJsonToolCallsParser,
    JsonToolCatalogRequest,
    NativeToolCallsParser,
    ProtocolProgram,
    ResultField,
    ResultLiteral,
    TextFrame,
    ToolCallFields,
    ToolResultMessage,
    composed_driver,
)
from xenolect.driver.parse import parse_model_response_full
from xenolect.driver.serialize import driver_hash, load_driver, save_driver
from xenolect.proxy import translate_response
from xenolect.xpt.planner import RequestConfig


def _custom_driver(*, capture_text: bool = False) -> Driver:
    fields = ToolCallFields(name="tool", arguments="input", call_id="call")
    return Driver(
        ir_version="0.2",
        protocol=ProtocolProgram(
            request=[
                JsonToolCatalogRequest(
                    instruction="Emit a tool call as CALL[...]",
                    catalog_heading="TOOLS:",
                    call_frame=TextFrame(prefix="CALL[", suffix="]"),
                    fields=fields,
                )
            ],
            response=[
                NativeToolCallsParser(),
                FramedJsonToolCallsParser(
                    frame=TextFrame(prefix="CALL[", suffix="]"),
                    fields=fields,
                    capture_surrounding_text=capture_text,
                ),
            ],
            tool_result=ToolResultMessage(
                role="user",
                segments=[
                    ResultLiteral(text="RESULT["),
                    ResultField(field="call_id", prefix="call="),
                    ResultLiteral(text="]"),
                    ResultField(field="content", prefix="\n"),
                ],
            ),
        ),
    )


def test_v01_artifact_identity_does_not_gain_a_protocol_null(tmp_path) -> None:
    legacy = Driver()
    payload = legacy.canonical_dict()
    assert payload["ir_version"] == "0.1"
    assert "protocol" not in payload
    assert driver_hash(legacy) == "ee80c9b78784"

    path = tmp_path / "legacy.mdriver"
    save_driver(legacy, path)
    loaded = load_driver(path)
    assert driver_hash(loaded) == driver_hash(legacy)
    assert "protocol" not in json.loads(path.read_text(encoding="utf-8"))


def test_xpt_components_compose_into_v02_program() -> None:
    from xenolect.driver.ir import ParserKind, ToolEncoding, ToolResultEncoding

    driver = composed_driver(
        tool_encoding=ToolEncoding.XML_JSON,
        parser=ParserKind.XML_JSON,
        tool_result_encoding=ToolResultEncoding.USER_MESSAGE,
    )
    payload = driver.canonical_dict()
    assert payload["ir_version"] == "0.2"
    assert "tool_encoding" not in payload
    assert [op["op"] for op in payload["protocol"]["request"]] == [
        "json_tool_catalog"
    ]
    assert [op["op"] for op in payload["protocol"]["response"]] == [
        "native_tool_calls",
        "framed_json_tool_calls",
    ]

    candidate = RequestConfig("xml_json", ()).driver()
    assert candidate.ir_version == "0.2"
    assert candidate.protocol is not None


def test_whole_content_json_is_a_composable_response_primitive() -> None:
    driver = Driver(
        ir_version="0.2",
        protocol=ProtocolProgram(
            request=[
                JsonToolCatalogRequest(
                    instruction="Return one JSON tool object",
                    call_frame=TextFrame(),
                )
            ],
            response=[
                FramedJsonToolCallsParser(
                    frame=TextFrame(), multiple=False, whole_content=True
                )
            ],
            tool_result=ToolResultMessage(
                role="user", segments=[ResultField(field="content")]
            ),
        ),
    )
    raw = {
        "choices": [
            {
                "message": {
                    "content": '{"name":"weather","arguments":{"city":"NYC"}}'
                }
            }
        ]
    }
    parsed = parse_model_response_full(raw, driver)
    assert parsed.ok
    assert isinstance(parsed.events[0], AssistantToolCall)
    assert parsed.events[0].call.arguments == {"city": "NYC"}


def test_legacy_xml_parser_keeps_case_and_tag_whitespace_compatibility() -> None:
    from xenolect.driver.ir import ParserKind, ToolEncoding

    driver = Driver(tool_encoding=ToolEncoding.XML_JSON, parser=ParserKind.XML_JSON)
    raw = {
        "choices": [
            {
                "message": {
                    "content": (
                        '<TOOL_CALL   > {"name":"weather","arguments":{}} '
                        "</TOOL_CALL >"
                    )
                }
            }
        ]
    }
    parsed = parse_model_response_full(raw, driver)
    assert parsed.ok
    assert isinstance(parsed.events[0], AssistantToolCall)
    assert parsed.events[0].call.name == "weather"


def test_parameterized_frame_and_field_mapping_execute_end_to_end(tmp_path) -> None:
    driver = _custom_driver()
    tool = ToolDef(
        name="weather",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    preambles = build_tool_preamble_messages([tool], driver)
    assert preambles[0]["role"] == "system"
    assert "TOOLS:" in preambles[0]["content"]

    encoded = encode_textual_tool_call(
        name="weather",
        arguments={"city": "NYC"},
        call_id="c1",
        driver=driver,
    )
    assert encoded == 'CALL[{"tool":"weather","input":{"city":"NYC"},"call":"c1"}]'

    raw = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": encoded,
                }
            }
        ]
    }
    parsed = parse_model_response_full(raw, driver)
    assert parsed.ok
    assert isinstance(parsed.events[0], AssistantToolCall)
    assert parsed.events[0].call.model_dump() == {
        "name": "weather",
        "arguments": {"city": "NYC"},
        "id": "c1",
    }

    result = encode_tool_result_message(
        ToolResult(call_id="c1", name="weather", content={"temp": 70}),
        driver,
    )
    assert result == {"role": "user", "content": 'RESULT[call=c1]\n{"temp": 70}'}

    path = tmp_path / "custom.mdriver"
    save_driver(driver, path)
    assert load_driver(path).canonical_dict() == driver.canonical_dict()


def test_mixed_text_and_framed_call_is_preserved_in_openai_response() -> None:
    driver = _custom_driver(capture_text=True)
    raw = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        'I will check. CALL[{"tool":"weather",'
                        '"input":{"city":"NYC"},"call":"c1"}]'
                    ),
                }
            }
        ]
    }
    out = translate_response(raw, driver, "model")
    message = out["choices"][0]["message"]
    assert message["content"] == "I will check."
    assert message["tool_calls"][0]["id"] == "c1"
    assert out["choices"][0]["finish_reason"] == "tool_calls"


def test_mixed_text_and_native_call_is_preserved() -> None:
    raw = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I will check.",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "arguments": '{"city":"NYC"}',
                            },
                        }
                    ],
                }
            }
        ]
    }
    out = translate_response(raw, Driver(), "model")
    message = out["choices"][0]["message"]
    assert message["content"] == "I will check."
    assert message["tool_calls"][0]["id"] == "c1"


def test_conflicting_response_primitives_fail_closed() -> None:
    fields_a = ToolCallFields()
    fields_b = ToolCallFields(name="tool", arguments="input", call_id=None)
    driver = Driver(
        ir_version="0.2",
        protocol=ProtocolProgram(
            request=[
                JsonToolCatalogRequest(
                    instruction="x",
                    call_frame=TextFrame(prefix="CALL "),
                )
            ],
            response=[
                FramedJsonToolCallsParser(
                    frame=TextFrame(prefix="CALL "), fields=fields_a
                ),
                FramedJsonToolCallsParser(
                    frame=TextFrame(prefix="CALL "), fields=fields_b
                ),
            ],
            tool_result=ToolResultMessage(
                role="user", segments=[ResultField(field="content")]
            ),
        ),
    )
    raw = {
        "choices": [
            {
                "message": {
                    "content": (
                        'CALL {"name":"one","arguments":{},'
                        '"tool":"two","input":{}}'
                    )
                }
            }
        ]
    }
    parsed = parse_model_response_full(raw, driver)
    assert not parsed.ok
    assert "different canonical tool calls" in parsed.errors[0]


def test_unknown_or_incomplete_protocol_program_is_rejected_clearly() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ProtocolProgram.model_validate(
            {
                "request": [{"op": "native_tools", "pretend": True}],
                "response": [{"op": "native_tool_calls"}],
                "tool_result": {
                    "role": "tool",
                    "segments": [{"op": "field", "field": "content"}],
                },
            }
        )

    with pytest.raises(ValidationError, match="missing required actions"):
        ProtocolProgram(
            request=[JsonToolCatalogRequest(instruction="x", call_frame=TextFrame(prefix="X"))],
            response=[NativeToolCallsParser()],
            tool_result=ToolResultMessage(
                role="user", segments=[ResultField(field="content")]
            ),
            state=[],
        )
