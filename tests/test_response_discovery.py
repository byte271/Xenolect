"""Response-protocol synthesis from black-box observations."""

from __future__ import annotations

import json
import re
from typing import Any

from xenolect.driver.ir import (
    FramedJsonToolCallsParser,
    JsonObjectToolCallsParser,
    ParserKind,
)
from xenolect.driver.serialize import load_driver, save_driver
from xenolect.xpt.planner import load_compiled_program
from xenolect.xpt.response_discovery import discover_response_parser
from xenolect.xpt.runtime import CERTIFIED, xpt_compile
from xenolect.xpt.syndrome import evaluate_all_parsers


def _raw(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_discovers_whole_content_json_and_custom_fields() -> None:
    raw = _raw('{"verb":"lookup","payload":{"key":"x"},"ticket":"c-1"}')
    found = discover_response_parser(
        raw,
        offered_tool_names={"lookup"},
        expected_arguments={"lookup": {"key": "x"}},
    )
    assert found.ok
    assert isinstance(found.parser, FramedJsonToolCallsParser)
    assert found.parser.whole_content
    assert found.parser.fields.model_dump() == {
        "name": "verb",
        "arguments": "payload",
        "call_id": "ticket",
    }


def test_discovers_unframed_embedded_json_with_mixed_text_and_multiple_calls() -> None:
    raw = _raw(
        'First {"verb":"a","payload":{"x":1},"ticket":"c1"} then '
        '{"verb":"b","payload":{"y":2},"ticket":"c2"}.'
    )
    found = discover_response_parser(
        raw,
        offered_tool_names={"a", "b"},
        expected_arguments={"a": {"x": 1}, "b": {"y": 2}},
    )
    assert found.ok
    assert isinstance(found.parser, JsonObjectToolCallsParser)
    assert found.parser.multiple
    assert found.parser.capture_surrounding_text
    assert [call.name for call in found.calls] == ["a", "b"]


def test_discovers_arbitrary_non_xml_opening_and_closing_delimiters() -> None:
    raw = _raw(
        'Starting. [[invoke]]{"verb":"a","payload":{"x":1},"ticket":"c1"}'
        '[[/invoke]] Finished.'
    )
    found = discover_response_parser(
        raw,
        offered_tool_names={"a"},
        expected_arguments={"a": {"x": 1}},
    )
    assert found.ok
    assert isinstance(found.parser, FramedJsonToolCallsParser)
    assert found.parser.frame.prefix == "[[invoke]]"
    assert found.parser.frame.suffix == "[[/invoke]]"


def test_ambiguous_call_id_field_fails_closed() -> None:
    raw = _raw(
        '<x>{"verb":"a","payload":{"x":1},"ticket":"c1","trace":"t1"}</x>'
        '<x>{"verb":"b","payload":{"y":2},"ticket":"c2","trace":"t2"}</x>'
    )
    found = discover_response_parser(
        raw,
        offered_tool_names={"a", "b"},
        expected_arguments={"a": {"x": 1}, "b": {"y": 2}},
    )
    assert not found.ok
    assert found.parser is None
    assert found.error == "ambiguous response call-id field; refusing to guess"


def test_response_discovery_candidate_extraction_has_a_hard_object_bound() -> None:
    content = " ".join(
        json.dumps({"verb": "a", "payload": {"x": 1}, "ticket": f"c{index}"})
        for index in range(17)
    )
    found = discover_response_parser(
        _raw(content),
        offered_tool_names={"a"},
        expected_arguments={"a": {"x": 1}},
    )
    assert not found.ok
    assert found.error == "response contains too many JSON objects for bounded discovery"


class HoldoutProtocolEndpoint:
    """A protocol surface absent from the legacy 144-program grammar.

    XPT receives only this endpoint object.  It is not passed the delimiter,
    field names, parser, provider/model identity, or any candidate identifier.
    """

    def __init__(self) -> None:
        self.responses: list[dict[str, Any]] = []

    @staticmethod
    def _contents(messages: list[dict[str, Any]]) -> str:
        return "\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("content") is not None
        )

    @staticmethod
    def _framed(calls: list[tuple[str, dict[str, Any], str]], lead: str) -> str:
        frames = [
            '<invoke-block channel="tool">'
            + json.dumps(
                {"verb": name, "payload": arguments, "ticket": call_id},
                separators=(",", ":"),
            )
            + "</invoke-block>"
            for name, arguments, call_id in calls
        ]
        return lead + "\n" + "\n".join(frames) + "\nBatch queued."

    def chat_completions(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        text = self._contents(messages)
        ack = re.findall(r"ACK-[0-9A-F]+", text)
        if ack:
            content = ack[-1]
        else:
            alpha = re.findall(r"XPT_A_[0-9A-F]+", text)
            beta = re.findall(r"XPT_B_[0-9A-F]+", text)
            error = re.findall(r"E-[0-9A-F]+", text)
            if alpha and beta and error:
                content = self._framed(
                    [
                        (
                            "commit",
                            {"alpha": alpha[-1], "beta": beta[-1]},
                            "holdout-recovery-1",
                        ),
                        ("report", {"code": error[-1]}, "holdout-recovery-2"),
                    ],
                    "Recovering from returned values.",
                )
            else:
                matches = re.findall(
                    r"^  (record_(?:alpha|beta|gamma)) (\{.*\})$",
                    text,
                    flags=re.MULTILINE,
                )
                arguments = {name: json.loads(payload) for name, payload in matches}
                content = self._framed(
                    [
                        (name, arguments[name], f"holdout-initial-{index}")
                        for index, name in enumerate(
                            ("record_alpha", "record_beta", "record_gamma"), start=1
                        )
                    ],
                    "Preparing the requested calls.",
                )
        raw = _raw(content)
        self.responses.append(raw)
        return raw


def test_xpt_synthesizes_new_holdout_response_program_and_certifies_it(tmp_path) -> None:
    endpoint = HoldoutProtocolEndpoint()
    result = xpt_compile(endpoint, load_compiled_program(), seed=41)

    assert result.status == CERTIFIED, result.as_dict()
    assert result.driver is not None
    assert result.driver.ir_version == "0.2"
    assert result.driver.protocol is not None
    discovered = [
        parser
        for parser in result.driver.protocol.response
        if isinstance(parser, FramedJsonToolCallsParser)
        and parser.frame.prefix == '<invoke-block channel="tool">'
    ]
    assert len(discovered) == 1
    parser = discovered[0]
    assert parser.frame.suffix == "</invoke-block>"
    assert parser.fields.model_dump() == {
        "name": "verb",
        "arguments": "payload",
        "call_id": "ticket",
    }
    assert parser.multiple and parser.capture_surrounding_text

    # The paid holdout observation is not parseable by any legacy ParserKind.
    assert all(
        outcome.n_calls == 0
        for outcome in evaluate_all_parsers(endpoint.responses[0]).values()
    )
    assert len(ParserKind) == 3
    assert result.total_generations <= 12
    assert result.diagnosis_generations == 3
    assert result.certification_generations == 3
    assert any(
        "response discovery synthesized" in note for note in result.ledger.notes
    )
    artifact = tmp_path / "holdout.mdriver"
    save_driver(result.driver, artifact)
    assert load_driver(artifact).canonical_dict() == result.driver.canonical_dict()
