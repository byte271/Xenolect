"""Non-cooperative holdouts for active discriminating protocol synthesis."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from itertools import product
from typing import Any

from xenolect.driver.ir import (
    FramedJsonToolCallsParser,
    JsonObjectToolCallsParser,
    ResultLiteral,
    TemplatedJsonToolCatalogRequest,
    effective_protocol,
)
from xenolect.xpt.discrimination import ResultVersion, parse_protocol_rejection
from xenolect.xpt.planner import all_request_configs, load_compiled_program
from xenolect.xpt.runtime import CERTIFIED, UNSUPPORTED, xpt_compile


@dataclass(frozen=True)
class LatentProtocolConstraints:
    request_role: str
    catalog_depth: int
    schema_projection: str
    call_frame: str
    call_fields: str
    result_role: str
    result_association: str


def generated_latent_constraints(seed: int) -> LatentProtocolConstraints:
    """Generate identifiable structural constraints, never secret literals."""
    choices: list[LatentProtocolConstraints] = []
    for role, depth, projection, frame, fields, result in product(
        ("system", "user"),
        (1, 2),
        ("preserve", "inline"),
        ("embedded_json", "framed"),
        ("semantic", "compact"),
        (
            ResultVersion("tool", "attachment"),
            ResultVersion("tool", "embedded"),
            ResultVersion("user", "embedded"),
        ),
    ):
        request_distance = sum(
            (
                role != "system",
                depth != 1,
                projection != "preserve",
                frame != "embedded_json",
                fields != "semantic",
            )
        )
        result_distance = int((result.role, result.association) != ("tool", "attachment"))
        if request_distance + result_distance > 5:
            continue
        choices.append(
            LatentProtocolConstraints(
                role,
                depth,
                projection,
                frame,
                fields,
                result.role,
                result.association,
            )
        )
    rng = random.Random(seed)
    return choices[rng.randrange(len(choices))]


def _raw(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _rejection(error_type: str, parameter: str) -> dict[str, Any]:
    return {
        "error": {
            "type": error_type,
            "code": ("unsupported_parameter" if parameter == "tools" else "unsupported_value"),
            "param": parameter,
            "message": "The supplied protocol property was rejected.",
        }
    }


def _contains_key(value: Any, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(_contains_key(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, target) for item in value)
    return False


class NonCooperativeEndpoint:
    """Expose only normal calls/results and ordinary parameterized API errors."""

    def __init__(self, constraints: LatentProtocolConstraints) -> None:
        self.constraints = constraints
        self.requests: list[dict[str, Any]] = []
        self.responses: list[dict[str, Any]] = []

    @staticmethod
    def _text(messages: list[dict[str, Any]]) -> str:
        return "\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("content") is not None
        )

    def _catalog(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if tools is not None:
            return None, _rejection("invalid_request_error", "tools")
        candidates = [
            message
            for message in messages
            if str(message.get("content", "")).startswith("Use the JSON catalog below.")
        ]
        if len(candidates) != 1:
            return None, _rejection("invalid_request_error", "messages.role")
        message = candidates[0]
        if message.get("role") != self.constraints.request_role:
            return None, _rejection("invalid_request_error", "messages.role")
        content = str(message["content"])
        try:
            instruction, encoded = content.split("\nTool catalog:\n", 1)
            catalog: Any = json.loads(encoded)
        except (ValueError, json.JSONDecodeError):
            return None, _rejection("invalid_request_error", "catalog.container_depth")
        depth = 0
        while isinstance(catalog, dict) and len(catalog) == 1:
            catalog = next(iter(catalog.values()))
            depth += 1
        if depth != self.constraints.catalog_depth or not isinstance(catalog, list):
            return None, _rejection("invalid_request_error", "catalog.container_depth")
        if not catalog or not all(
            isinstance(item, dict)
            and len(item) == 3
            and sum(isinstance(value, dict) for value in item.values()) == 1
            and sum(isinstance(value, str) for value in item.values()) == 2
            for item in catalog
        ):
            return None, _rejection("invalid_request_error", "catalog.container_depth")
        has_refs = any(
            _contains_key(item, "$ref") or _contains_key(item, "$defs") for item in catalog
        )
        projection = "preserve" if has_refs else "inline"
        if projection != self.constraints.schema_projection:
            return None, _rejection("invalid_request_error", "tools.schema_projection")

        fields_match = re.search(r"keys are ([^,]+), ([^,]+), and ([^.]+)\.", instruction)
        if fields_match is None:
            return None, _rejection("invalid_request_error", "assistant.call_fields")
        fields = fields_match.groups()
        field_strategy = "compact" if all(len(field) == 1 for field in fields) else "semantic"
        if field_strategy != self.constraints.call_fields:
            return None, _rejection("invalid_request_error", "assistant.call_fields")
        framed = re.search(
            r"Surround each tool-call JSON object with (\S+) and (\S+)\.",
            instruction,
        )
        frame_strategy = "framed" if framed is not None else "embedded_json"
        if frame_strategy != self.constraints.call_frame:
            return None, _rejection("invalid_request_error", "assistant.call_frame")
        return {
            "items": catalog,
            "fields": fields,
            "frame": framed.groups() if framed is not None else ("", ""),
            "frame_strategy": frame_strategy,
        }, None

    def _result_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        last_assistant = max(
            (index for index, message in enumerate(messages) if message.get("role") == "assistant"),
            default=-1,
        )
        return messages[last_assistant + 1 :] if last_assistant >= 0 else []

    def _validate_results(self, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        results = self._result_messages(messages)
        if not results:
            return None
        if any(message.get("role") != self.constraints.result_role for message in results):
            return _rejection("invalid_tool_result", "messages.result_role")
        for message in results:
            attached = isinstance(message.get("tool_call_id"), str)
            association = "attachment" if attached else "embedded"
            if association != self.constraints.result_association:
                return _rejection("invalid_tool_result", "messages.tool_call_id")
            try:
                payload = json.loads(str(message.get("content", "")))
            except json.JSONDecodeError:
                return _rejection("invalid_tool_result", "messages.tool_call_id")
            if not isinstance(payload, dict) or len(payload) < 2:
                return _rejection("invalid_tool_result", "messages.tool_call_id")
            values = list(payload.values())
            if not any(
                isinstance(value, str)
                and value in {"record_alpha", "record_beta", "record_gamma", "commit", "report"}
                for value in values
            ) or not any(isinstance(value, dict) for value in values):
                return _rejection("invalid_tool_result", "messages.tool_call_id")
            if association == "embedded" and not any(
                isinstance(value, str) and value.startswith("active-") for value in values
            ):
                return _rejection("invalid_tool_result", "messages.tool_call_id")
        return None

    @staticmethod
    def _arguments(text: str) -> dict[str, dict[str, Any]]:
        matches = re.findall(
            r"^  (record_(?:alpha|beta|gamma)) (\{.*\})$",
            text,
            flags=re.MULTILINE,
        )
        return {name: json.loads(value) for name, value in matches}

    @staticmethod
    def _call_objects(catalog: dict[str, Any], calls: list[tuple[str, dict[str, Any], str]]) -> str:
        name_field, arguments_field, id_field = catalog["fields"]
        prefix, suffix = catalog["frame"]
        encoded = [
            prefix
            + json.dumps(
                {
                    name_field: name,
                    arguments_field: arguments,
                    id_field: call_id,
                },
                separators=(",", ":"),
            )
            + suffix
            for name, arguments, call_id in calls
        ]
        return "Working on the requested tools.\n" + "\n".join(encoded) + "\nDone."

    def chat_completions(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        self.requests.append({"messages": messages, "tools": tools})
        catalog, error = self._catalog(messages, tools)
        if error is not None:
            self.responses.append(error)
            return error
        assert catalog is not None
        result_error = self._validate_results(messages)
        if result_error is not None:
            self.responses.append(result_error)
            return result_error

        text = self._text(messages)
        ack_values = re.findall(r"ACK-[0-9A-F]+", text)
        assistant_messages = [message for message in messages if message.get("role") == "assistant"]
        if len(assistant_messages) >= 2 and ack_values:
            response = _raw(ack_values[-1])
            self.responses.append(response)
            return response

        alpha_tokens = re.findall(r"XPT_A_[0-9A-F]+", text)
        beta_tokens = re.findall(r"XPT_B_[0-9A-F]+", text)
        error_codes = re.findall(r"E-[0-9A-F]+", text)
        if assistant_messages and alpha_tokens and beta_tokens and error_codes:
            response = _raw(
                self._call_objects(
                    catalog,
                    [
                        (
                            "commit",
                            {"alpha": alpha_tokens[-1], "beta": beta_tokens[-1]},
                            "active-recovery-1",
                        ),
                        ("report", {"code": error_codes[-1]}, "active-recovery-2"),
                    ],
                )
            )
            self.responses.append(response)
            return response

        arguments = self._arguments(text)
        response = _raw(
            self._call_objects(
                catalog,
                [
                    (name, arguments[name], f"active-initial-{index}")
                    for index, name in enumerate(
                        ("record_alpha", "record_beta", "record_gamma"), start=1
                    )
                ],
            )
        )
        self.responses.append(response)
        return response


def _assert_unseen_program(result: Any) -> None:
    assert result.driver is not None and result.driver.protocol is not None
    request = result.driver.protocol.request
    assert len(request) == 1 and isinstance(request[0], TemplatedJsonToolCatalogRequest)
    assert any(
        isinstance(parser, (FramedJsonToolCallsParser, JsonObjectToolCallsParser))
        for parser in result.driver.protocol.response
    )
    renderer = result.driver.protocol.tool_result
    assert isinstance(renderer.segments[0], ResultLiteral)
    assert renderer.segments[0].text == "{"

    legacy_requests = {
        json.dumps(
            [item.model_dump(mode="json") for item in effective_protocol(config.driver()).request],
            sort_keys=True,
        )
        for config in all_request_configs()
    }
    assert (
        json.dumps([item.model_dump(mode="json") for item in request], sort_keys=True)
        not in legacy_requests
    )
    legacy_results = {
        json.dumps(
            effective_protocol(config.driver()).tool_result.model_dump(mode="json"),
            sort_keys=True,
        )
        for config in all_request_configs()
    }
    assert json.dumps(renderer.model_dump(mode="json"), sort_keys=True) not in legacy_results


def test_non_cooperative_multi_seed_holdout_sweep() -> None:
    drivers: set[str] = set()
    constraints_seen: set[LatentProtocolConstraints] = set()
    for seed in (2, 7, 13, 19, 29):
        constraints = generated_latent_constraints(seed)
        constraints_seen.add(constraints)
        endpoint = NonCooperativeEndpoint(constraints)
        result = xpt_compile(endpoint, load_compiled_program(), seed=101)
        assert result.status == CERTIFIED, result.as_dict()
        assert result.diagnosis_generations <= 9
        assert result.certification_generations == 3
        assert result.total_generations <= 12
        _assert_unseen_program(result)
        assert result.driver is not None
        drivers.add(json.dumps(result.driver.canonical_dict(), sort_keys=True))

        report = result.as_dict()["protocol_synthesis"]
        assert report["mode"] == "bounded_active_discriminating_synthesis"
        assert report["property_local_fault_localization_used"] is True
        assert report["property_local_rejections_observed"] > 0
        assert report["property_local_rejections_used"] > 0
        assert len(report["version_spaces"]) == 2
        assert {space["component"] for space in report["version_spaces"]} == {
            "request",
            "tool_result",
        }
        assert all(
            experiment.get("ranking_is_proof") is False for experiment in report["experiments"]
        )
        controlled = [
            experiment
            for experiment in report["experiments"]
            if experiment.get("kind") == "controlled_intervention"
        ]
        request_experiments = [
            experiment for experiment in controlled if experiment["component"] == "request"
        ]
        result_experiments = [
            experiment for experiment in controlled if experiment["component"] == "tool_result"
        ]
        assert all(len(experiment["interventions"]) <= 1 for experiment in request_experiments[1:])
        assert all(len(experiment["interventions"]) <= 2 for experiment in result_experiments)
        assert any(experiment["expected_information_gain"] > 0 for experiment in controlled)
        assert all(
            revision["changed_components"] == [revision["component"]]
            for revision in report["revisions"]
        )
        assert report["certification"]["certificate"]["complete"] is True
        assert all(
            delta["accepted_value_revealed"] is False for delta in report["behavioral_deltas"]
        )
        assert all(
            "xpt_counterexample" not in json.dumps(response)
            and "Structural example" not in json.dumps(response)
            for response in endpoint.responses
        )
        for response in endpoint.responses:
            if "error" in response:
                assert set(response["error"]) == {"type", "code", "param", "message"}
                assert "expected" not in response["error"]["message"].lower()

    assert len(constraints_seen) >= 4
    assert len(drivers) >= 4


def test_non_cooperative_synthesis_is_deterministic() -> None:
    constraints = generated_latent_constraints(13)
    first = xpt_compile(NonCooperativeEndpoint(constraints), load_compiled_program(), seed=101)
    second = xpt_compile(NonCooperativeEndpoint(constraints), load_compiled_program(), seed=101)
    assert first.status == second.status == CERTIFIED
    assert first.driver is not None and second.driver is not None
    assert first.driver.canonical_dict() == second.driver.canonical_dict()
    first_report = first.as_dict()["protocol_synthesis"]
    second_report = second.as_dict()["protocol_synthesis"]
    assert first_report["version_spaces"] == second_report["version_spaces"]
    assert first_report["behavioral_deltas"] == second_report["behavioral_deltas"]


class OrdinaryNegativeEndpoint(NonCooperativeEndpoint):
    """Return one stochastic-looking non-call after the initial native rejection."""

    def chat_completions(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if tools is not None:
            return super().chat_completions(messages, tools, **kwargs)
        self.requests.append({"messages": messages, "tools": tools})
        response = _raw("I did not produce a tool call on this sample.")
        self.responses.append(response)
        return response


def test_ordinary_negative_behavior_does_not_shrink_version_space() -> None:
    endpoint = OrdinaryNegativeEndpoint(generated_latent_constraints(2))
    result = xpt_compile(endpoint, load_compiled_program(), seed=101)
    assert result.status == UNSUPPORTED
    assert result.driver is None
    report = result.as_dict()["protocol_synthesis"]
    assert report["mode"] == "bounded_active_discriminating_synthesis"
    request_space = report["version_spaces"][0]
    # The deterministic native-tools API rejection removes one version. The
    # subsequent ordinary non-call removes none of the 32 textual versions.
    assert request_space["initial_size"] == 33
    assert len(request_space["survivor_fingerprints"]) == 32
    assert len(request_space["logical_eliminations"]) == 1
    assert any(
        row["kind"] == "negative_behavior" and row["strength"] == "heuristic"
        for row in report["evidence"]["rows"]
    )


def test_api_rejection_parser_does_not_accept_target_values_or_extra_fields() -> None:
    valid = _rejection("invalid_request_error", "messages.role")
    parsed = parse_protocol_rejection(valid)
    assert parsed is not None and parsed.parameter == "messages.role"
    invalid = json.loads(json.dumps(valid))
    invalid["error"]["expected"] = "user"
    assert parse_protocol_rejection(invalid) is None
    generic = json.loads(json.dumps(valid))
    generic["error"]["code"] = "invalid_value"
    assert parse_protocol_rejection(generic) is None
