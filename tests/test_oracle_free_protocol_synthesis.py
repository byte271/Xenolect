"""Oracle-free holdouts using only ordinary Chat Completions wire behavior."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import pytest

from xenolect.xpt.ablation import run_candidate_only_ablation, summarize_ablation
from xenolect.xpt.discrimination import (
    RequestVersion,
    ResultVersion,
    request_version_space,
    result_version_space,
)
from xenolect.xpt.planner import load_compiled_program
from xenolect.xpt.runtime import CERTIFIED, UNSUPPORTED, xpt_compile

GENERIC_ERROR = {
    "error": {"type": "invalid_request_error", "message": "invalid request"}
}


@dataclass(frozen=True)
class OracleFreeConstraints:
    request: RequestVersion
    result: ResultVersion


@dataclass(frozen=True)
class _Catalog:
    tools: tuple[dict[str, Any], ...]
    fields: tuple[str, str, str] | None
    frame: tuple[str, str]


def _contains_key(value: Any, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(_contains_key(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, target) for item in value)
    return False


def _const_value(value: Any) -> str | None:
    if isinstance(value, dict):
        if isinstance(value.get("const"), str):
            return value["const"]
        for item in value.values():
            found = _const_value(item)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _const_value(item)
            if found is not None:
                return found
    return None


def _required_field(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    required = value.get("required")
    if (
        isinstance(required, list)
        and len(required) == 1
        and isinstance(required[0], str)
    ):
        return required[0]
    return None


class OracleFreeEndpoint:
    """A deterministic endpoint whose validator consumes normal wire data only."""

    def __init__(self, constraints: OracleFreeConstraints) -> None:
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

    @staticmethod
    def _normalize_native(tools: list[dict[str, Any]]) -> tuple[dict[str, Any], ...] | None:
        normalized: list[dict[str, Any]] = []
        for item in tools:
            function = item.get("function") if isinstance(item, dict) else None
            if not isinstance(function, dict):
                return None
            if not isinstance(function.get("name"), str):
                return None
            normalized.append(
                {
                    "name": function["name"],
                    "description": function.get("description"),
                    "schema": function.get("parameters", {}),
                }
            )
        return tuple(normalized)

    @staticmethod
    def _textual_catalog(message: dict[str, Any]) -> tuple[RequestVersion, _Catalog] | None:
        content = message.get("content")
        if not isinstance(content, str) or not content.startswith("Use the JSON catalog below."):
            return None
        try:
            instruction, encoded = content.split("\nTool catalog:\n", 1)
            catalog: Any = json.loads(encoded)
        except (ValueError, json.JSONDecodeError):
            return None
        depth = 0
        while isinstance(catalog, dict) and len(catalog) == 1:
            catalog = next(iter(catalog.values()))
            depth += 1
        if depth not in {1, 2} or not isinstance(catalog, list) or not catalog:
            return None
        tools: list[dict[str, Any]] = []
        for item in catalog:
            if not isinstance(item, dict):
                return None
            schema_values = [value for value in item.values() if isinstance(value, dict)]
            text_values = [value for value in item.values() if isinstance(value, str)]
            if len(schema_values) != 1 or len(text_values) != 2:
                return None
            name = next((value for value in text_values if value), None)
            if name is None:
                return None
            description = next((value for value in text_values if value != name), None)
            tools.append({"name": name, "description": description, "schema": schema_values[0]})
        fields_match = re.search(r"keys are ([^,]+), ([^,]+), and ([^.]+)\.", instruction)
        if fields_match is None:
            return None
        fields = fields_match.groups()
        field_strategy = "compact" if all(len(field) == 1 for field in fields) else "semantic"
        framed = re.search(
            r"Surround each tool-call JSON object with (\S+) and (\S+)\.",
            instruction,
        )
        call_frame = "framed" if framed else "embedded_json"
        projection = (
            "preserve"
            if any(_contains_key(tool["schema"], "$ref") for tool in tools)
            else "inline"
        )
        version = RequestVersion(
            "textual",
            str(message.get("role")),
            depth,
            projection,
            call_frame,
            field_strategy,
        )
        return version, _Catalog(
            tuple(tools), tuple(fields), framed.groups() if framed else ("", "")
        )

    def _matching_catalog(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> _Catalog | None:
        target = self.constraints.request
        if target.mode == "native":
            normalized = self._normalize_native(tools or [])
            if not normalized or not any(
                _contains_key(tool["schema"], "$ref") for tool in normalized
            ):
                return None
            return _Catalog(normalized, None, ("", ""))
        matches = [
            catalog
            for message in messages
            if (parsed := self._textual_catalog(message)) is not None
            for version, catalog in (parsed,)
            if version == target
        ]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _decode_result(
        message: dict[str, Any], version: ResultVersion
    ) -> dict[str, Any] | None:
        if message.get("role") != version.role:
            return None
        attached = isinstance(message.get("tool_call_id"), str)
        if attached != (version.association == "attachment"):
            return None
        try:
            payload = json.loads(str(message.get("content", "")))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        if not isinstance(payload.get("tool"), str) or "body" not in payload:
            return None
        if version.association == "embedded" and not isinstance(
            payload.get("call_ref"), str
        ):
            return None
        if version.association == "attachment":
            payload = {**payload, "call_ref": message["tool_call_id"]}
        return payload

    def _consume_results(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
        last_assistant = max(
            (
                index
                for index, message in enumerate(messages)
                if message.get("role") == "assistant"
            ),
            default=-1,
        )
        suffix = messages[last_assistant + 1 :] if last_assistant >= 0 else []
        target_results = [
            decoded
            for message in suffix
            if (decoded := self._decode_result(message, self.constraints.result)) is not None
        ]
        any_result_shape = any(
            self._decode_result(message, version) is not None
            for message in suffix
            for version in (
                ResultVersion("tool", "attachment"),
                ResultVersion("tool", "embedded"),
                ResultVersion("user", "embedded"),
            )
        )
        return target_results, any_result_shape

    def _calls(
        self,
        catalog: _Catalog,
        calls: list[tuple[str, dict[str, Any], str]],
    ) -> dict[str, Any]:
        if self.constraints.request.mode == "native":
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(arguments),
                                    },
                                }
                                for name, arguments, call_id in calls
                            ],
                        }
                    }
                ]
            }
        assert catalog.fields is not None
        name_field, arguments_field, id_field = catalog.fields
        prefix, suffix = catalog.frame
        content = "ordinary assistant text\n" + "\n".join(
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
        )
        return {"choices": [{"message": {"role": "assistant", "content": content}}]}

    @staticmethod
    def _plain(content: str) -> dict[str, Any]:
        return {"choices": [{"message": {"role": "assistant", "content": content}}]}

    @staticmethod
    def _initial_arguments(text: str) -> dict[str, dict[str, Any]]:
        return {
            name: json.loads(value)
            for name, value in re.findall(
                r"^  (record_(?:alpha|beta|gamma)) (\{.*\})$",
                text,
                flags=re.MULTILINE,
            )
        }

    def _respond(
        self,
        messages: list[dict[str, Any]],
        catalog: _Catalog,
        consumed: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # This is ordinary instruction following over catalog schemas and consumed
        # result bodies. It has no access to XPT metadata or a reference Driver.
        if consumed:
            continuation_bodies = [
                item["body"]
                for item in consumed
                if isinstance(item.get("body"), dict)
                and isinstance(item["body"].get("code"), str)
                and isinstance(item["body"].get("next_call_id"), str)
            ]
            if len(continuation_bodies) == 1:
                body = continuation_bodies[0]
                return self._calls(
                    catalog,
                    [("report", {"code": body["code"]}, body["next_call_id"])],
                )
            if len(continuation_bodies) > 1:
                return self._plain("ambiguous result representations")

            material = json.dumps(consumed, sort_keys=True)
            acknowledgements = re.findall(r"ACK-[0-9A-F]+", material)
            if acknowledgements:
                return self._plain(acknowledgements[-1])
            alpha = re.findall(r"XPT_A_[0-9A-F]+", material)
            beta = re.findall(r"XPT_B_[0-9A-F]+", material)
            errors = re.findall(r"E-[0-9A-F]+", material)
            if alpha and beta and errors:
                return self._calls(
                    catalog,
                    [
                        (
                            "commit",
                            {"alpha": alpha[-1], "beta": beta[-1]},
                            "recovery-1",
                        ),
                        ("report", {"code": errors[-1]}, "recovery-2"),
                    ],
                )

        if len(catalog.tools) == 1:
            tool = catalog.tools[0]
            value = _const_value(tool["schema"])
            field = _required_field(tool["schema"])
            call_id_match = re.search(
                r"use call id (\S+)", str(tool.get("description", ""))
            )
            if value is not None and field is not None and call_id_match is not None:
                return self._calls(
                    catalog,
                    [
                        (
                            tool["name"],
                            {field: value},
                            call_id_match.group(1),
                        )
                    ],
                )

        arguments = self._initial_arguments(self._text(messages))
        if set(arguments) == {"record_alpha", "record_beta", "record_gamma"}:
            return self._calls(
                catalog,
                [
                    (name, arguments[name], f"initial-{index}")
                    for index, name in enumerate(
                        ("record_alpha", "record_beta", "record_gamma"), start=1
                    )
                ],
            )
        return self._plain("ordinary non-compliance")

    def chat_completions(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        self.requests.append({"messages": messages, "tools": tools})
        catalog = self._matching_catalog(messages, tools)
        if catalog is None:
            response = json.loads(json.dumps(GENERIC_ERROR))
        else:
            consumed, any_result_shape = self._consume_results(messages)
            if any_result_shape and not consumed:
                response = json.loads(json.dumps(GENERIC_ERROR))
            else:
                response = self._respond(messages, catalog, consumed)
        self.responses.append(response)
        return response


class NonIdentifiableEndpoint(OracleFreeEndpoint):
    """A normal endpoint policy that rejects requests with multiple catalogs."""

    def _matching_catalog(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> _Catalog | None:
        textual_count = sum(
            self._textual_catalog(message) is not None for message in messages
        )
        catalog_count = textual_count + int(bool(tools))
        if catalog_count != 1:
            return None
        return super()._matching_catalog(messages, tools)


def generated_coverage_matrix() -> tuple[OracleFreeConstraints, ...]:
    """Greedily cover every value and every feasible pairwise interaction."""
    universe = tuple(
        OracleFreeConstraints(request, result)
        for request in request_version_space()
        for result in result_version_space()
    )

    def features(item: OracleFreeConstraints) -> tuple[Any, ...]:
        request = item.request
        return (
            request.mode,
            request.role,
            request.catalog_depth,
            request.schema_projection,
            request.call_frame,
            request.call_fields,
            item.result.role,
            item.result.association,
        )

    def pairs(item: OracleFreeConstraints) -> set[tuple[int, Any, int, Any]]:
        values = features(item)
        return {
            (left, values[left], right, values[right])
            for left in range(len(values))
            for right in range(left + 1, len(values))
        }

    uncovered = set().union(*(pairs(item) for item in universe))
    selected: list[OracleFreeConstraints] = []
    remaining = list(universe)
    while uncovered:
        chosen = min(
            remaining,
            key=lambda item: (
                -len(pairs(item).intersection(uncovered)),
                item.request.complexity,
                item.result.complexity,
                item.request.fingerprint,
                item.result.fingerprint,
            ),
        )
        gain = pairs(chosen).intersection(uncovered)
        assert gain
        selected.append(chosen)
        uncovered.difference_update(gain)
        remaining.remove(chosen)

    worst_case = OracleFreeConstraints(
        request_version_space()[-1], result_version_space()[-1]
    )
    if worst_case not in selected:
        selected.append(worst_case)
    return tuple(selected)


CASES = generated_coverage_matrix()


@pytest.mark.parametrize("constraints", CASES)
def test_oracle_free_multi_case_holdout(constraints: OracleFreeConstraints) -> None:
    endpoint = OracleFreeEndpoint(constraints)
    result = xpt_compile(endpoint, load_compiled_program(), seed=211)
    assert result.status == CERTIFIED, result.as_dict()
    assert result.diagnosis_generations <= 9
    assert result.certification_generations == 3
    assert result.total_generations <= 12
    report = result.as_dict()["protocol_synthesis"]
    assert report["mode"] == "bounded_oracle_free_diagnostic_synthesis"
    assert report["failure_class"] is None
    assert report["property_local_fault_localization_used"] is False
    assert report["diagnostic_probe_is_production_driver"] is False
    assert report["identifiability"]["request"]["identifiable"] is True
    assert report["identifiability"]["tool_result"]["identifiable"] is True
    assert len(report["probe_plans"]) == 3
    assert all(plan["actual_observed_outcome"] for plan in report["probe_plans"])
    assert all(plan["elimination_reasons"] for plan in report["probe_plans"])
    assert all(plan["minimax_score_is_proof"] is False for plan in report["probe_plans"])
    assert all(plan["information_score_is_proof"] is False for plan in report["probe_plans"])
    assert all(plan["probability_prior_used"] is False for plan in report["probe_plans"])
    assert set(report["evidence"]["obligation_witnesses"][0]["generation_ids"]).isdisjoint(
        {
            generation.index
            for generation in result.ledger.generations
            if generation.diagnostic_probe is not None
        }
    )
    assert all(set(request) == {"messages", "tools"} for request in endpoint.requests)
    assert all(
        response == GENERIC_ERROR or "error" not in response
        for response in endpoint.responses
    )


def test_oracle_free_non_identifiable_endpoint_fails_closed() -> None:
    endpoint = NonIdentifiableEndpoint(CASES[1])
    result = xpt_compile(endpoint, load_compiled_program(), seed=211)
    assert result.status == UNSUPPORTED
    assert result.driver is None
    assert "observationally unidentifiable" in result.reason
    report = result.as_dict()["protocol_synthesis"]
    assert report["mode"] == "bounded_oracle_free_diagnostic_synthesis"
    assert report["failure_class"] == "observationally_unidentifiable"
    assert report["probe_plans"][0]["hypotheses_removed"] == []
    assert report["probe_plans"][0]["observed_kind"] == "generic_rejection"


def test_candidate_only_ablation_over_the_exact_same_endpoint_family() -> None:
    candidate_runs = [
        run_candidate_only_ablation(OracleFreeEndpoint(constraints), seed=211)
        for constraints in CASES
    ]
    diagnostic_runs = [
        xpt_compile(
            OracleFreeEndpoint(constraints), load_compiled_program(), seed=211
        )
        for constraints in CASES
    ]
    report = summarize_ablation(candidate_runs, diagnostic_runs)
    candidate = report["candidate_only"]
    diagnostic = report["diagnostic_probe_synthesis"]

    assert report["same_request_space"] == 33
    assert report["same_result_space"] == 3
    assert report["same_default_budget"] == {
        "total_generations": 12,
        "certification_reserve": 3,
    }
    assert diagnostic["certification_successes"] == len(CASES)
    assert candidate["certification_successes"] == 4
    assert candidate["diagnosis_generations"] == [3, 5, 9, 9, 9, 5, 7, 9, 9]
    assert diagnostic["diagnosis_generations"] == [7] * len(CASES)
    assert diagnostic["worst_case_generations"] == 7
    assert candidate["worst_case_generations"] == 9
    assert diagnostic["median_generations"] == 7
    assert candidate["median_generations"] == 9
    assert candidate["unresolved_or_ambiguous"] == 5
    assert diagnostic["unresolved_or_ambiguous"] == 0
