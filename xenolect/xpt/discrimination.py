"""Bounded version spaces and controlled protocol interventions.

This module does not encode provider dialects.  It defines a small product of
v0.2 primitive properties that XPT can actively vary.  A structured API
rejection may identify the rejected *parameter* but never supplies its accepted
value.  The planner changes that property, retains unrelated evidence, and
selects the lowest-complexity survivor.  Information gain ranks experiments;
only deterministic wire/API rejection removes versions.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Literal

from xenolect.driver.ir import (
    FramedJsonToolCallsParser,
    JsonObjectToolCallsParser,
    NativeToolCallsParser,
    NativeToolsRequest,
    ResultField,
    ResultLiteral,
    SchemaTransform,
    TemplatedJsonToolCatalogRequest,
    TextFrame,
    ToolCallFields,
    ToolDefinitionFields,
    ToolResultMessage,
)
from xenolect.xpt.hypothesis import PartialProtocolHypothesis, ProtocolComponent

REQUEST_MODE = "tools"
REQUEST_ROLE = "messages.role"
CATALOG_CONTAINER = "catalog.container_depth"
SCHEMA_PROJECTION = "tools.schema_projection"
CALL_FRAME = "assistant.call_frame"
CALL_FIELDS = "assistant.call_fields"
RESULT_ROLE = "messages.result_role"
RESULT_ASSOCIATION = "messages.tool_call_id"

_REQUEST_PARAMS = {
    REQUEST_MODE,
    REQUEST_ROLE,
    CATALOG_CONTAINER,
    SCHEMA_PROJECTION,
    CALL_FRAME,
    CALL_FIELDS,
}
_RESULT_PARAMS = {RESULT_ROLE, RESULT_ASSOCIATION}
_REJECTION_TYPES = {"invalid_request_error", "invalid_tool_result"}
_REJECTION_CODES = {"unsupported_value", "unsupported_parameter"}


@dataclass(frozen=True)
class ProtocolRejection:
    """An ordinary deterministic API rejection that names no accepted value."""

    component: ProtocolComponent
    parameter: str
    error_type: str
    code: str
    message: str
    determinism_scope: Literal["property_value"] = "property_value"


def parse_protocol_rejection(raw: Any) -> ProtocolRejection | None:
    """Parse a strict API error establishing property-local determinism.

    ``unsupported_value`` and ``unsupported_parameter`` mean the tested value
    is rejected for that parameter across this endpoint.  A generic
    ``invalid_value`` is deliberately not enough to eliminate sibling versions.
    """
    if not isinstance(raw, dict) or set(raw) != {"error"}:
        return None
    error = raw.get("error")
    if not isinstance(error, dict):
        return None
    if set(error) - {"type", "code", "param", "message"}:
        return None
    error_type = error.get("type")
    code = error.get("code")
    parameter = error.get("param")
    message = error.get("message")
    if (
        error_type not in _REJECTION_TYPES
        or code not in _REJECTION_CODES
        or not isinstance(parameter, str)
        or not isinstance(message, str)
    ):
        return None
    if parameter in _REQUEST_PARAMS and error_type == "invalid_request_error":
        component = ProtocolComponent.REQUEST
    elif parameter in _RESULT_PARAMS and error_type == "invalid_tool_result":
        component = ProtocolComponent.TOOL_RESULT
    else:
        return None
    return ProtocolRejection(component, parameter, error_type, code, message)


@dataclass(frozen=True)
class RequestVersion:
    mode: Literal["native", "textual"]
    role: Literal["system", "user"] | None = None
    catalog_depth: Literal[1, 2] | None = None
    schema_projection: Literal["preserve", "inline"] | None = None
    call_frame: Literal["embedded_json", "framed"] | None = None
    call_fields: Literal["semantic", "compact"] | None = None

    def __post_init__(self) -> None:
        values = (
            self.role,
            self.catalog_depth,
            self.schema_projection,
            self.call_frame,
            self.call_fields,
        )
        if self.mode == "native" and any(value is not None for value in values):
            raise ValueError("native request versions cannot carry textual properties")
        if self.mode == "textual" and any(value is None for value in values):
            raise ValueError("textual request versions require every typed property")

    def value(self, parameter: str) -> Any:
        values = {
            REQUEST_MODE: self.mode,
            REQUEST_ROLE: self.role,
            CATALOG_CONTAINER: self.catalog_depth,
            SCHEMA_PROJECTION: self.schema_projection,
            CALL_FRAME: self.call_frame,
            CALL_FIELDS: self.call_fields,
        }
        if parameter not in values:
            raise ValueError(f"unsupported request intervention {parameter!r}")
        return values[parameter]

    @property
    def complexity(self) -> tuple[int, ...]:
        if self.mode == "native":
            return (0, 0, 0, 0, 0, 0)
        return (
            1,
            int(self.role == "user"),
            int(self.catalog_depth == 2),
            int(self.schema_projection == "inline"),
            int(self.call_frame == "framed"),
            int(self.call_fields == "compact"),
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "role": self.role,
            "catalog_depth": self.catalog_depth,
            "schema_projection": self.schema_projection,
            "call_frame": self.call_frame,
            "call_fields": self.call_fields,
        }


@dataclass(frozen=True)
class ResultVersion:
    role: Literal["tool", "user"]
    association: Literal["attachment", "embedded"]

    def __post_init__(self) -> None:
        if self.role == "user" and self.association == "attachment":
            raise ValueError("user result messages cannot attach tool_call_id")

    def value(self, parameter: str) -> Any:
        values = {RESULT_ROLE: self.role, RESULT_ASSOCIATION: self.association}
        if parameter not in values:
            raise ValueError(f"unsupported result intervention {parameter!r}")
        return values[parameter]

    @property
    def complexity(self) -> tuple[int, int]:
        return (int(self.role == "user"), int(self.association == "embedded"))

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {"role": self.role, "association": self.association}


ProtocolVersion = RequestVersion | ResultVersion


def request_version_space() -> tuple[RequestVersion, ...]:
    versions = [RequestVersion(mode="native")]
    versions.extend(
        RequestVersion("textual", role, depth, projection, frame, fields)
        for role, depth, projection, frame, fields in product(
            ("system", "user"),
            (1, 2),
            ("preserve", "inline"),
            ("embedded_json", "framed"),
            ("semantic", "compact"),
        )
    )
    return tuple(sorted(versions, key=lambda version: (version.complexity, version.fingerprint)))


def result_version_space() -> tuple[ResultVersion, ...]:
    return (
        ResultVersion("tool", "attachment"),
        ResultVersion("tool", "embedded"),
        ResultVersion("user", "embedded"),
    )


@dataclass(frozen=True)
class ControlledExperiment:
    component: ProtocolComponent
    candidate_fingerprint: str
    interventions: tuple[tuple[str, Any], ...]
    targeted_property: str | None
    survivors_before: int
    distinguishing_survivors: int
    ranking_property: str | None
    expected_information_gain: float
    expected_obligation_gain: int
    estimated_generations: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component.value,
            "candidate_fingerprint": self.candidate_fingerprint,
            "interventions": [
                {"property": name, "value": value} for name, value in self.interventions
            ],
            "targeted_property": self.targeted_property,
            "survivors_before": self.survivors_before,
            "distinguishing_survivors": self.distinguishing_survivors,
            "ranking_property": self.ranking_property,
            "expected_information_gain": self.expected_information_gain,
            "expected_obligation_gain": self.expected_obligation_gain,
            "estimated_generations": self.estimated_generations,
            "ranking_is_proof": False,
        }


@dataclass
class VersionSpace:
    component: ProtocolComponent
    versions: tuple[ProtocolVersion, ...]
    survivors: set[str] = field(init=False)
    elimination_log: list[dict[str, Any]] = field(default_factory=list)
    experiment_log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.survivors = {version.fingerprint for version in self.versions}

    def surviving_versions(self) -> list[ProtocolVersion]:
        return [version for version in self.versions if version.fingerprint in self.survivors]

    def eliminate_rejected_value(
        self,
        tested: ProtocolVersion,
        rejection: ProtocolRejection,
        *,
        evidence_id: str,
    ) -> int:
        """Eliminate only the value named by a deterministic API rejection."""
        if rejection.component != self.component:
            raise ValueError("rejection cannot refine an unrelated version space")
        value = tested.value(rejection.parameter)
        if value is None:
            raise ValueError("rejection parameter was not active in the tested version")
        removed = {
            version.fingerprint
            for version in self.surviving_versions()
            if version.value(rejection.parameter) == value
        }
        if not removed:
            raise ValueError("deterministic rejection did not eliminate a surviving value")
        self.survivors.difference_update(removed)
        self.elimination_log.append(
            {
                "parameter": rejection.parameter,
                "rejected_value": value,
                "evidence_id": evidence_id,
                "removed": sorted(removed),
                "survivors_after": len(self.survivors),
                "basis": "deterministic_wire_api_rejection",
            }
        )
        return len(removed)

    def choose(
        self,
        *,
        previous: ProtocolVersion | None,
        implicated_parameter: str | None,
        expected_obligation_gain: int,
    ) -> tuple[ProtocolVersion, ControlledExperiment]:
        versions = self.surviving_versions()
        if not versions:
            raise ValueError("protocol version space is empty")

        def distance(version: ProtocolVersion) -> int:
            if previous is None:
                return 0
            params = _params_for(self.component)
            return sum(version.value(param) != previous.value(param) for param in params)

        def information(version: ProtocolVersion) -> tuple[float, int, str | None]:
            ranked: list[tuple[float, int, str]] = []
            for parameter in _params_for(self.component):
                value = version.value(parameter)
                if value is None:
                    continue
                distinguish = sum(candidate.value(parameter) != value for candidate in versions)
                ranked.append((_binary_entropy(distinguish, len(versions)), distinguish, parameter))
            if not ranked:
                return 0.0, 0, None
            return max(ranked, key=lambda item: (item[0], item[1], item[2]))

        versions.sort(
            key=lambda version: (
                distance(version),
                -information(version)[0],
                -information(version)[1],
                version.complexity,
                version.fingerprint,
            )
        )
        selected = versions[0]
        params = _params_for(self.component)
        interventions = tuple(
            (parameter, selected.value(parameter))
            for parameter in params
            if previous is None or selected.value(parameter) != previous.value(parameter)
        )
        information_gain, distinguish, ranking_property = information(selected)
        total = len(versions)
        experiment = ControlledExperiment(
            component=self.component,
            candidate_fingerprint=selected.fingerprint,
            interventions=interventions,
            targeted_property=implicated_parameter,
            survivors_before=total,
            distinguishing_survivors=distinguish,
            ranking_property=ranking_property,
            expected_information_gain=information_gain,
            expected_obligation_gain=expected_obligation_gain,
        )
        self.experiment_log.append(experiment.as_dict())
        return selected, experiment

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component.value,
            "initial_size": len(self.versions),
            "survivors": [version.as_dict() for version in self.surviving_versions()],
            "survivor_fingerprints": sorted(self.survivors),
            "experiments": list(self.experiment_log),
            "logical_eliminations": list(self.elimination_log),
        }


def request_version_to_hypothesis(version: RequestVersion) -> PartialProtocolHypothesis:
    if version.mode == "native":
        return PartialProtocolHypothesis(
            request=(NativeToolsRequest(),),
            response=(NativeToolCallsParser(),),
        )
    assert version.role is not None
    assert version.catalog_depth is not None
    assert version.schema_projection is not None
    assert version.call_frame is not None
    assert version.call_fields is not None
    token = version.fingerprint[:8]
    fields = (
        ToolCallFields(name="function", arguments="input", call_id="call_ref")
        if version.call_fields == "semantic"
        else ToolCallFields(name="n", arguments="a", call_id="i")
    )
    if version.call_frame == "embedded_json":
        frame = TextFrame()
        parser = JsonObjectToolCallsParser(
            fields=fields,
            multiple=True,
            capture_surrounding_text=True,
        )
        frame_instruction = "Emit each tool call as an unframed JSON object."
    else:
        frame = TextFrame(prefix=f"<call-{token}>", suffix=f"</call-{token}>")
        parser = FramedJsonToolCallsParser(
            frame=frame,
            fields=fields,
            multiple=True,
            capture_surrounding_text=True,
        )
        frame_instruction = (
            f"Surround each tool-call JSON object with {frame.prefix} and {frame.suffix}."
        )
    instruction = (
        "Use the JSON catalog below. "
        + frame_instruction
        + f" The name, arguments, and call-id keys are {fields.name}, "
        + f"{fields.arguments}, and {fields.call_id}."
    )
    path = ["catalog"] if version.catalog_depth == 1 else ["protocol", "catalog"]
    request = TemplatedJsonToolCatalogRequest(
        role=version.role,
        instruction=instruction,
        catalog_heading="Tool catalog:",
        catalog_path=path,
        tool_fields=ToolDefinitionFields(
            name="tool", description="description", parameters="schema"
        ),
        call_frame=frame,
        fields=fields,
    )
    transforms = (SchemaTransform.INLINE_REFS,) if version.schema_projection == "inline" else ()
    return PartialProtocolHypothesis(
        request=(request,),
        response=(parser,),
        schema_transforms=transforms,
    )


def result_version_to_program(version: ResultVersion) -> ToolResultMessage:
    segments: list[ResultLiteral | ResultField] = [ResultLiteral(text="{")]
    if version.association == "embedded":
        segments.append(ResultField(field="call_id", prefix='"call_ref":"', suffix='",'))
    segments.extend(
        [
            ResultField(field="name", prefix='"tool":"', suffix='",'),
            ResultField(field="content", prefix='"body":', omit_if_none=False),
            ResultLiteral(text="}"),
        ]
    )
    return ToolResultMessage(
        role=version.role,
        segments=segments,
        attach_tool_call_id=version.association == "attachment",
    )


def _params_for(component: ProtocolComponent) -> tuple[str, ...]:
    if component == ProtocolComponent.REQUEST:
        return (
            REQUEST_MODE,
            REQUEST_ROLE,
            CATALOG_CONTAINER,
            SCHEMA_PROJECTION,
            CALL_FRAME,
            CALL_FIELDS,
        )
    if component == ProtocolComponent.TOOL_RESULT:
        return (RESULT_ROLE, RESULT_ASSOCIATION)
    raise ValueError(f"no discriminating version space for {component.value}")


def _binary_entropy(partition: int, total: int) -> float:
    if partition <= 0 or partition >= total:
        return 0.0
    probability = partition / total
    return round(
        -probability * math.log2(probability) - (1.0 - probability) * math.log2(1.0 - probability),
        6,
    )


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]
