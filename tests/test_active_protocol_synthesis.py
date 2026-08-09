"""Generated black-box holdouts for obligation-directed protocol synthesis."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any

import pytest

from xenolect.driver.ir import (
    REQUIRED_STATE_ACTIONS,
    FramedJsonToolCallsParser,
    ResultLiteral,
    TemplatedJsonToolCatalogRequest,
    effective_protocol,
)
from xenolect.xpt.hypothesis import (
    ComponentEvidence,
    ContradictionClass,
    EvidenceKind,
    EvidenceStore,
    EvidenceStrength,
    PartialProtocolHypothesis,
    ProtocolComponent,
)
from xenolect.xpt.planner import all_request_configs, load_compiled_program
from xenolect.xpt.protocol_synthesis import (
    WitnessPhase,
    extract_counterexample,
    obligation_support_evidence,
    obligation_witness_evidence,
)
from xenolect.xpt.runtime import CERTIFIED, UNSUPPORTED, xpt_compile
from xenolect.xpt.session import Generation


@dataclass(frozen=True)
class GeneratedProtocol:
    request_role: str
    instruction: str
    heading: str
    catalog_path: tuple[str, ...]
    catalog_name: str
    catalog_description: str
    catalog_parameters: str
    response_prefix: str
    response_suffix: str
    call_name: str
    call_arguments: str
    call_id: str
    result_role: str
    result_open: str
    result_separator: str
    result_close: str


def _word(rng: random.Random, prefix: str) -> str:
    alphabet = "abcdefghijkmnpqrstuvwxyz"
    return prefix + "_" + "".join(rng.choice(alphabet) for _ in range(7))


def generated_holdout_protocol(seed: int) -> GeneratedProtocol:
    """Generate one bounded protocol without registering a named dialect."""
    rng = random.Random(seed)
    tag = _word(rng, "frame")
    # This seed-controlled placement is intentionally outside the legacy
    # frontier used by the end-to-end test below.
    role = ("user", "system")[rng.randrange(2)]
    return GeneratedProtocol(
        request_role=role,
        instruction="Render calls with " + tag,
        heading=_word(rng, "catalog") + ":",
        catalog_path=(_word(rng, "root"), _word(rng, "items")),
        catalog_name=_word(rng, "tool"),
        catalog_description=_word(rng, "about"),
        catalog_parameters=_word(rng, "schema"),
        response_prefix="[[" + tag + "]]",
        response_suffix="[[/" + tag + "]]",
        call_name=_word(rng, "verb"),
        call_arguments=_word(rng, "input"),
        call_id=_word(rng, "ticket"),
        result_role="assistant",
        result_open="<<" + _word(rng, "result") + ">>",
        result_separator="|" + _word(rng, "sep") + "=",
        result_close="<</" + _word(rng, "done") + ">>",
    )


def _raw(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class GeneratedHoldoutEndpoint:
    """Endpoint that reveals only atomic counterexamples through observations."""

    def __init__(self, hidden: GeneratedProtocol) -> None:
        self._hidden = hidden
        self.requests: list[dict[str, Any]] = []
        self._example_items: list[dict[str, Any]] = []

    @staticmethod
    def _text(messages: list[dict[str, Any]]) -> str:
        return "\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("content") is not None
        )

    def _request_constraints(self, nonce: str) -> dict[str, Any]:
        hidden = self._hidden
        facts = {
            "message.role": hidden.request_role,
            "instruction": hidden.instruction,
            "catalog.heading": hidden.heading,
            "catalog.path": list(hidden.catalog_path),
            "catalog.fields.name": hidden.catalog_name,
            "catalog.fields.description": hidden.catalog_description,
            "catalog.fields.parameters": hidden.catalog_parameters,
            "call.frame.prefix": hidden.response_prefix,
            "call.frame.suffix": hidden.response_suffix,
            "call.frame.case_sensitive": True,
            "call.frame.whitespace_after_prefix": False,
            "call.frame.flexible_whitespace": False,
            "call.fields.name": hidden.call_name,
            "call.fields.arguments": hidden.call_arguments,
            "call.fields.call_id": hidden.call_id,
        }
        return {
            "xpt_counterexample": {
                "component": "request",
                "nonce": nonce,
                "constraints": [
                    {"path": path, "equals": value} for path, value in sorted(facts.items())
                ],
            }
        }

    def _result_constraints(self, nonce: str) -> dict[str, Any]:
        h = self._hidden
        facts: list[tuple[str, Any]] = [
            ("message.role", h.result_role),
            ("attach_tool_call_id", False),
            ("segments.0.literal", h.result_open),
            ("segments.1.field", "name"),
            ("segments.1.prefix", "name="),
            ("segments.1.suffix", ""),
            ("segments.1.omit_if_none", True),
            ("segments.2.literal", h.result_separator),
            ("segments.3.field", "call_id"),
            ("segments.3.prefix", "ref="),
            ("segments.3.suffix", ""),
            ("segments.3.omit_if_none", True),
            ("segments.4.literal", h.result_separator),
            ("segments.5.field", "content"),
            ("segments.5.prefix", "body="),
            ("segments.5.suffix", ""),
            ("segments.5.omit_if_none", False),
            ("segments.6.literal", h.result_close),
        ]
        return {
            "xpt_counterexample": {
                "component": "tool_result",
                "nonce": nonce,
                "constraints": [{"path": path, "equals": value} for path, value in facts],
            }
        }

    def _request_example(self, nonce: str) -> str:
        h = self._hidden
        wrapped: Any = list(self._example_items)
        for key in reversed(h.catalog_path):
            wrapped = {key: wrapped}
        call_sample = {
            h.call_name: "record_alpha",
            h.call_arguments: {"challenge": nonce},
            h.call_id: "sample-call-id",
        }
        return (
            "Request rejected. Structural example follows.\n"
            + h.instruction
            + "\n"
            + h.heading
            + "\n"
            + json.dumps(wrapped, separators=(",", ":"))
            + "\nCall example:\n"
            + h.response_prefix
            + json.dumps(call_sample, separators=(",", ":"))
            + h.response_suffix
        )

    def _request_rejection(self, nonce: str) -> dict[str, Any]:
        return _raw(self._request_example(nonce))

    def _result_example(self, nonce: str) -> str:
        h = self._hidden
        content = json.dumps({"token": nonce, "status": "ok"})
        return (
            h.result_open
            + "name=record_alpha"
            + h.result_separator
            + "ref=generated-initial-1"
            + h.result_separator
            + "body="
            + content
            + h.result_close
        )

    def _result_rejection(self, nonce: str) -> dict[str, Any]:
        return _raw(self._result_example(nonce))

    def _catalog_valid(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> bool:
        if tools is not None:
            return False
        h = self._hidden
        candidates = [
            message
            for message in messages
            if message.get("role") == h.request_role
            and str(message.get("content", "")).startswith(h.instruction + "\n")
        ]
        if len(candidates) != 1:
            return False
        content = str(candidates[0]["content"])
        marker = "\n" + h.heading + "\n"
        if marker not in content:
            return False
        try:
            value: Any = json.loads(content.split(marker, 1)[1])
            for key in h.catalog_path:
                value = value[key]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if not isinstance(value, list) or len(value) != 5:
            return False
        expected_keys = {
            h.catalog_name,
            h.catalog_description,
            h.catalog_parameters,
        }
        return all(
            isinstance(item, dict)
            and set(item) == expected_keys
            and isinstance(item[h.catalog_name], str)
            and isinstance(item[h.catalog_parameters], dict)
            for item in value
        )

    def _result_message_valid(self, message: dict[str, Any]) -> bool:
        h = self._hidden
        if message.get("role") != h.result_role or "tool_call_id" in message:
            return False
        content = message.get("content")
        if not isinstance(content, str):
            return False
        if not (content.startswith(h.result_open + "name=") and content.endswith(h.result_close)):
            return False
        return content.count(h.result_separator) == 2 and "ref=" in content and "body=" in content

    def _framed(self, calls: list[tuple[str, dict[str, Any], str]]) -> str:
        h = self._hidden
        frames = [
            h.response_prefix
            + json.dumps(
                {
                    h.call_name: name,
                    h.call_arguments: arguments,
                    h.call_id: call_id,
                },
                separators=(",", ":"),
            )
            + h.response_suffix
            for name, arguments, call_id in calls
        ]
        return "Observed calls:\n" + "\n".join(frames) + "\nEnd calls."

    def chat_completions(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        self.requests.append({"messages": messages, "tools": tools})
        text = self._text(messages)
        alpha_code = re.findall(r"AC-[0-9A-F]+", text)
        if tools:
            self._example_items = [
                {
                    self._hidden.catalog_name: item["function"]["name"],
                    self._hidden.catalog_description: item["function"].get("description", ""),
                    self._hidden.catalog_parameters: item["function"]["parameters"],
                }
                for item in tools
            ]
        if not self._catalog_valid(messages, tools):
            nonce = alpha_code[-1] if alpha_code else "missing"
            return self._request_rejection(nonce)

        alpha_tokens = re.findall(r"XPT_A_[0-9A-F]+", text)
        beta_tokens = re.findall(r"XPT_B_[0-9A-F]+", text)
        error_codes = re.findall(r"E-[0-9A-F]+", text)
        ack_values = re.findall(r"ACK-[0-9A-F]+", text)
        result_messages = [
            message
            for message in messages
            if self._hidden.response_prefix not in str(message.get("content", ""))
            and any(
                token in str(message.get("content", ""))
                for token in alpha_tokens + beta_tokens + error_codes + ack_values
            )
        ]
        post_prompt_results = result_messages
        if post_prompt_results and not all(
            self._result_message_valid(message) for message in post_prompt_results
        ):
            nonce = alpha_tokens[-1] if alpha_tokens else "missing"
            return self._result_rejection(nonce)

        if ack_values and post_prompt_results:
            return _raw(ack_values[-1])
        if alpha_tokens and beta_tokens and error_codes and post_prompt_results:
            return _raw(
                self._framed(
                    [
                        (
                            "commit",
                            {"alpha": alpha_tokens[-1], "beta": beta_tokens[-1]},
                            "generated-recovery-1",
                        ),
                        (
                            "report",
                            {"code": error_codes[-1]},
                            "generated-recovery-2",
                        ),
                    ]
                )
            )

        matches = re.findall(
            r"^  (record_(?:alpha|beta|gamma)) (\{.*\})$",
            text,
            flags=re.MULTILINE,
        )
        arguments = {name: json.loads(payload) for name, payload in matches}
        return _raw(
            self._framed(
                [
                    (name, arguments[name], f"generated-initial-{index}")
                    for index, name in enumerate(
                        ("record_alpha", "record_beta", "record_gamma"), start=1
                    )
                ]
            )
        )


def test_generated_holdout_synthesizes_all_three_components_and_certifies() -> None:
    hidden = generated_holdout_protocol(3)
    assert hidden.request_role == "user"
    endpoint = GeneratedHoldoutEndpoint(hidden)

    # No protocol object, candidate ID, provider/model identity, or expected
    # Driver enters the compiler API.
    result = xpt_compile(endpoint, load_compiled_program(), seed=73)

    assert result.status == CERTIFIED, result.as_dict()
    assert result.driver is not None and result.driver.protocol is not None
    request = result.driver.protocol.request
    assert len(request) == 1 and isinstance(request[0], TemplatedJsonToolCatalogRequest)
    assert request[0].role == hidden.request_role
    assert tuple(request[0].catalog_path) == hidden.catalog_path
    assert request[0].tool_fields.model_dump() == {
        "name": hidden.catalog_name,
        "description": hidden.catalog_description,
        "parameters": hidden.catalog_parameters,
    }
    parser = next(
        item
        for item in result.driver.protocol.response
        if isinstance(item, FramedJsonToolCallsParser)
    )
    assert parser.frame.prefix == hidden.response_prefix
    assert parser.frame.suffix == hidden.response_suffix
    assert parser.fields.model_dump() == {
        "name": hidden.call_name,
        "arguments": hidden.call_arguments,
        "call_id": hidden.call_id,
    }
    renderer = result.driver.protocol.tool_result
    assert renderer.role == hidden.result_role
    assert renderer.segments[0] == ResultLiteral(text=hidden.result_open + "name=")
    assert renderer.segments[-1] == ResultLiteral(text=hidden.result_close)

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
    assert result.driver.protocol.state == list(REQUIRED_STATE_ACTIONS)
    assert result.diagnosis_generations == 8
    assert result.certification_generations == 3
    assert result.total_generations == 11

    report = result.as_dict()["protocol_synthesis"]
    assert report["certification"]["passed"] is True
    assert [revision["component"] for revision in report["revisions"]] == [
        "request",
        "response",
        "tool_result",
    ]
    assert all(
        revision["changed_components"] == [revision["component"]]
        for revision in report["revisions"]
    )
    assert {row["component"] for row in report["evidence"]["rows"]} == {
        "request",
        "response",
        "tool_result",
    }
    assert all("obligation_ids" not in row for row in report["evidence"]["rows"])
    diagnosis_witnesses = report["evidence"]["obligation_witnesses"]
    assert {row["obligation_id"] for row in diagnosis_witnesses} == {
        f"OB{index:02d}" for index in range(1, 18)
    }
    assert "OB18" not in {row["obligation_id"] for row in diagnosis_witnesses}
    g1_witnesses = [row for row in diagnosis_witnesses if len(row["generation_ids"]) == 1]
    assert {row["obligation_id"] for row in g1_witnesses} == {
        "OB01",
        "OB02",
        "OB03",
        "OB04",
        "OB05",
        "OB06",
    }
    assert report["certification"]["authority"] == "independent_certification"
    assert report["certification"]["certificate"]["complete"] is True
    assert len(report["certification"]["certificate"]["rows"]) == 18
    assert set(report["evidence"]["logical_eliminations"]) == {
        "request",
        "tool_result",
    }


def test_holdout_generator_varies_bounded_structural_properties() -> None:
    generated = [generated_holdout_protocol(seed) for seed in range(12)]
    assert {value.request_role for value in generated} == {"system", "user"}
    assert len({value.catalog_path for value in generated}) == len(generated)
    assert len({value.call_name for value in generated}) == len(generated)
    assert len({value.response_prefix for value in generated}) == len(generated)
    assert len({value.result_open for value in generated}) == len(generated)
    assert all(len(value.catalog_path) == 2 for value in generated)


def test_active_synthesis_is_deterministic_for_same_observations_and_seed() -> None:
    hidden = generated_holdout_protocol(3)
    first = xpt_compile(GeneratedHoldoutEndpoint(hidden), load_compiled_program(), seed=73)
    second = xpt_compile(GeneratedHoldoutEndpoint(hidden), load_compiled_program(), seed=73)
    assert first.status == second.status == CERTIFIED
    assert first.driver is not None and second.driver is not None
    assert first.driver.canonical_dict() == second.driver.canonical_dict()
    first_report = first.as_dict()["protocol_synthesis"]
    second_report = second.as_dict()["protocol_synthesis"]
    assert first_report["evidence"] == second_report["evidence"]
    assert first_report["revisions"] == second_report["revisions"]


def test_partial_hypothesis_cannot_cross_runtime_boundary_with_holes() -> None:
    hypothesis = PartialProtocolHypothesis()
    assert hypothesis.unresolved_components == (
        ProtocolComponent.REQUEST,
        ProtocolComponent.RESPONSE,
        ProtocolComponent.TOOL_RESULT,
    )
    with pytest.raises(ValueError, match="still has holes"):
        hypothesis.to_driver()
    with pytest.raises(ValueError, match="cannot synthesize state"):
        PartialProtocolHypothesis(state=())


def test_heuristic_ranking_cannot_eliminate_a_hypothesis() -> None:
    store = EvidenceStore()
    heuristic = ComponentEvidence(
        evidence_id="rank-only",
        component=ProtocolComponent.REQUEST,
        kind=EvidenceKind.NEGATIVE_BEHAVIOR,
        strength=EvidenceStrength.HEURISTIC,
        generation_id=1,
        request_hash="r",
        response_hash="s",
        observation="higher novelty score",
        contradiction_class=ContradictionClass.ORDINARY_BEHAVIOR,
    )
    with pytest.raises(ValueError, match="heuristic evidence cannot eliminate"):
        store.eliminate(ProtocolComponent.REQUEST, "candidate", evidence=heuristic)
    assert not store.is_eliminated(ProtocolComponent.REQUEST, "candidate")


def _generation(index: int) -> Generation:
    return Generation(
        index=index,
        purpose="explore",
        label=f"G{index}",
        branch_id="b",
        forked_from=None,
        prefix_hash="p",
        driver={},
        request={},
        request_hash=f"request-{index}",
        response={},
        response_hash=f"response-{index}",
        error=None,
        latency_ms=0.0,
        prompt_chars=0,
        completion_chars=0,
    )


def test_support_is_not_proof_and_witnesses_are_turn_scoped() -> None:
    generation1 = _generation(1)
    generation2 = _generation(2)
    generation3 = _generation(3)
    store = EvidenceStore()
    support = obligation_support_evidence(
        obligation_ids=("OB12",),
        generations=(generation1,),
        component_evidence_ids=("component-g1",),
        observation="G1 is part of a possible history trace",
    )[0]
    store.record_support(support)
    assert store.proven_obligation_ids == set()

    with pytest.raises(ValueError, match="G1 does not contain a complete witness for OB12"):
        obligation_witness_evidence(
            obligation_ids=("OB12",),
            phase=WitnessPhase.G1,
            generations=(generation1,),
            component_evidence_ids=("component-g1",),
            observation="only one turn",
        )
    with pytest.raises(ValueError, match="completed-trace certification"):
        obligation_witness_evidence(
            obligation_ids=("OB18",),
            phase=WitnessPhase.G3,
            generations=(generation1, generation2, generation3),
            component_evidence_ids=("component-g3",),
            observation="no completed-trace validator",
        )

    witness = obligation_witness_evidence(
        obligation_ids=("OB12",),
        phase=WitnessPhase.G3,
        generations=(generation1, generation2, generation3),
        component_evidence_ids=("component-g1", "component-g2", "component-g3"),
        observation="two complete result cycles observed across three turns",
    )[0]
    store.record_witness(witness)
    assert store.proven_obligation_ids == {"OB12"}


def test_ordinary_negative_behavior_cannot_logically_eliminate() -> None:
    store = EvidenceStore()
    ordinary = ComponentEvidence(
        evidence_id="ordinary-negative",
        component=ProtocolComponent.REQUEST,
        kind=EvidenceKind.COUNTEREXAMPLE,
        strength=EvidenceStrength.LOGICAL,
        generation_id=1,
        request_hash="r",
        response_hash="s",
        observation="one stochastic response omitted a tool call",
        contradiction_class=ContradictionClass.ORDINARY_BEHAVIOR,
    )
    with pytest.raises(ValueError, match="ordinary model behavior cannot eliminate"):
        store.eliminate(ProtocolComponent.REQUEST, "candidate", evidence=ordinary)

    structural = ComponentEvidence(
        evidence_id="structural-contradiction",
        component=ProtocolComponent.REQUEST,
        kind=EvidenceKind.COUNTEREXAMPLE,
        strength=EvidenceStrength.LOGICAL,
        generation_id=1,
        request_hash="r",
        response_hash="s",
        observation="candidate wire contradicts a nonce-bound structural fact",
        contradiction_class=ContradictionClass.STRUCTURAL,
    )
    store.eliminate(ProtocolComponent.REQUEST, "candidate", evidence=structural)
    assert store.is_eliminated(ProtocolComponent.REQUEST, "candidate")


class AtomicCounterexampleEndpoint(GeneratedHoldoutEndpoint):
    def _request_rejection(self, nonce: str) -> dict[str, Any]:
        return _raw(json.dumps(self._request_constraints(nonce), separators=(",", ":")))

    def _result_rejection(self, nonce: str) -> dict[str, Any]:
        return _raw(json.dumps(self._result_constraints(nonce), separators=(",", ":")))


class ConflictingCounterexampleEndpoint(AtomicCounterexampleEndpoint):
    def _request_constraints(self, nonce: str) -> dict[str, Any]:
        value = super()._request_constraints(nonce)
        value["xpt_counterexample"]["constraints"].append(
            {"path": "message.role", "equals": "system"}
        )
        return value


def test_conflicting_request_constraints_fail_closed_without_driver() -> None:
    endpoint = ConflictingCounterexampleEndpoint(generated_holdout_protocol(3))
    result = xpt_compile(endpoint, load_compiled_program(), seed=73)
    assert result.status == UNSUPPORTED
    assert result.driver is None
    assert "conflicting constraints" in result.reason
    assert result.diagnosis_generations == 1


def test_atomic_counterexample_alternative_is_bounded_and_certified() -> None:
    endpoint = AtomicCounterexampleEndpoint(generated_holdout_protocol(3))
    result = xpt_compile(endpoint, load_compiled_program(), seed=73)
    assert result.status == CERTIFIED, result.as_dict()
    assert result.diagnosis_generations == 5
    assert result.certification_generations == 3


def test_counterexample_nonce_binding_rejects_replayed_evidence() -> None:
    raw = _raw(
        json.dumps(
            {
                "xpt_counterexample": {
                    "component": "request",
                    "nonce": "old-run",
                    "constraints": [{"path": "message.role", "equals": "user"}],
                }
            }
        )
    )
    found = extract_counterexample(
        raw,
        expected_component=ProtocolComponent.REQUEST,
        expected_nonce="current-run",
    )
    assert found.found and not found.ok
    assert found.error == "counterexample nonce does not match the active experiment"
