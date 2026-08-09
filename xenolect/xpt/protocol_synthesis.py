"""Bounded extraction and composition of request/result protocol evidence.

Structural rejection examples are the preferred input: XPT maps a nested JSON
catalog and nonce-bound call sample into request parameters, or locates known
fresh result values inside an exact result-wire example.  A strict atomic
counterexample object is an alternative.  Neither form contains a Driver or a
named dialect.  Unknown, missing, duplicated, replayed, or conflicting evidence
fails closed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from xenolect.abi.events import ToolDef, ToolResult
from xenolect.driver.ir import (
    RequestPrimitive,
    ResultField,
    ResultLiteral,
    TemplatedJsonToolCatalogRequest,
    TextFrame,
    ToolCallFields,
    ToolDefinitionFields,
    ToolResultMessage,
)
from xenolect.driver.parse import iter_json_objects
from xenolect.xpt.hypothesis import (
    ComponentEvidence,
    EvidenceKind,
    EvidenceStrength,
    ProtocolComponent,
    attributed_obligations,
)
from xenolect.xpt.session import Generation

MAX_COUNTEREXAMPLE_CONTENT_CHARS = 65_536
MAX_COUNTEREXAMPLE_OBJECTS = 8
MAX_CONSTRAINTS = 32
MAX_CONSTRAINT_PATH_CHARS = 96
MAX_CONSTRAINT_VALUE_CHARS = 8_192
MAX_RESULT_SEGMENTS = 12


@dataclass(frozen=True)
class AtomicConstraint:
    path: str
    expected: Any


@dataclass(frozen=True)
class CounterexampleObservation:
    component: ProtocolComponent | None = None
    nonce: str | None = None
    constraints: tuple[AtomicConstraint, ...] = ()
    found: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.found and self.component is not None and not self.error

    def by_path(self) -> dict[str, Any]:
        return {constraint.path: constraint.expected for constraint in self.constraints}


@dataclass(frozen=True)
class RequestProgramDiscovery:
    """Request candidates inferred from one nonce-bound structural example."""

    candidates: tuple[tuple[RequestPrimitive, ...], ...] = ()
    observation: CounterexampleObservation | None = None
    found: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.candidates) and self.observation is not None and not self.error


@dataclass(frozen=True)
class ToolResultProgramDiscovery:
    """Result-renderer candidates inferred from one fresh-sentinel example."""

    candidates: tuple[ToolResultMessage, ...] = ()
    observation: CounterexampleObservation | None = None
    found: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.candidates) and self.observation is not None and not self.error


def _message_content(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    message: Any = None
    if "choices" in raw:
        choices = raw.get("choices") or []
        if choices:
            message = choices[0].get("message")
    elif isinstance(raw.get("message"), dict):
        message = raw["message"]
    elif "role" in raw:
        message = raw
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def _nested_lists(
    value: Any,
    *,
    path: tuple[str, ...] = (),
    depth: int = 0,
) -> list[tuple[tuple[str, ...], list[Any]]]:
    if isinstance(value, list):
        return [(path, value)]
    if not isinstance(value, dict) or depth >= 4 or len(value) > 32:
        return []
    found: list[tuple[tuple[str, ...], list[Any]]] = []
    for key in sorted(key for key in value if isinstance(key, str) and key):
        found.extend(_nested_lists(value[key], path=path + (key,), depth=depth + 1))
    return found


def _catalog_mapping(
    items: list[Any], tools: list[ToolDef]
) -> tuple[str, str, str] | None | object:
    if (
        len(items) != len(tools)
        or not items
        or not all(isinstance(item, dict) and len(item) <= 16 for item in items)
    ):
        return None
    objects = [item for item in items if isinstance(item, dict)]
    common = {key for key in objects[0] if isinstance(key, str)}
    for item in objects[1:]:
        common &= {key for key in item if isinstance(key, str)}
    offered = {tool.name for tool in tools}
    name_fields = [
        key
        for key in sorted(common)
        if all(isinstance(item.get(key), str) for item in objects)
        and {item[key] for item in objects} == offered
    ]
    if len(name_fields) != 1:
        return _STRUCTURAL_AMBIGUITY if name_fields else None
    name_field = name_fields[0]
    parameter_fields = [
        key
        for key in sorted(common - {name_field})
        if all(isinstance(item.get(key), dict) for item in objects)
    ]
    description_fields = [
        key
        for key in sorted(common - {name_field})
        if all(isinstance(item.get(key), str) or item.get(key) is None for item in objects)
    ]
    if len(parameter_fields) != 1 or len(description_fields) != 1:
        return _STRUCTURAL_AMBIGUITY
    return name_field, description_fields[0], parameter_fields[0]


_STRUCTURAL_AMBIGUITY = object()


def _adjacent_frame_before(content: str, start: int) -> tuple[str, bool] | None:
    before = content[:start]
    stripped = before.rstrip()
    gap = len(before) != len(stripped)
    tag = re.search(r"(<[^<>\r\n]{1,126}>)$", stripped)
    if tag is not None:
        return tag.group(1), gap
    token = re.search(r"([^\s{}]{1,128})$", stripped)
    if token is None or not any(
        not (character.isalnum() or character in "_-.") for character in token.group(1)
    ):
        return None
    return token.group(1), gap


def _adjacent_frame_after(content: str, end: int) -> str | None:
    stripped = content[end:].lstrip()
    tag = re.match(r"(<[^<>\r\n]{1,126}>)", stripped)
    if tag is not None:
        return tag.group(1)
    token = re.match(r"([^\s{}]{1,128})", stripped)
    if token is None or not any(
        not (character.isalnum() or character in "_-.") for character in token.group(1)
    ):
        return None
    return token.group(1)


def discover_request_program_from_example(
    raw: Any,
    *,
    tools: list[ToolDef],
    expected_nonce: str,
) -> RequestProgramDiscovery:
    """Infer a nested catalog and call syntax from a rejected-wire example.

    The response must contain exactly one nested list whose item fields map to
    the offered tool names, descriptions, and schema objects, plus exactly one
    framed call sample for an offered tool.  The sample arguments must contain
    the current challenge nonce, so a prior run cannot be replayed as evidence.
    Message role is not observable in response text; it remains a two-value
    typed choice tested in deterministic complexity order.
    """
    content = _message_content(raw)
    if content is None or not content.strip():
        return RequestProgramDiscovery(found=False)
    if len(content) > MAX_COUNTEREXAMPLE_CONTENT_CHARS:
        return RequestProgramDiscovery(
            found=True, error="request example exceeds the bounded size limit"
        )
    objects = iter_json_objects(content)
    if len(objects) > MAX_COUNTEREXAMPLE_OBJECTS:
        return RequestProgramDiscovery(
            found=True, error="request example has too many JSON objects"
        )
    catalogs: list[tuple[int, int, tuple[str, ...], tuple[str, str, str]]] = []
    saw_catalog_shape = False
    for start, end, value in objects:
        for path, items in _nested_lists(value):
            mapping = _catalog_mapping(items, tools)
            if mapping is _STRUCTURAL_AMBIGUITY:
                return RequestProgramDiscovery(
                    found=True, error="ambiguous catalog item field mapping"
                )
            if mapping is not None:
                saw_catalog_shape = True
                catalogs.append((start, end, path, mapping))
    if not catalogs:
        return RequestProgramDiscovery(found=saw_catalog_shape)
    if len(catalogs) != 1:
        return RequestProgramDiscovery(
            found=True, error="multiple matching catalog examples; refusing ambiguity"
        )
    catalog_start, catalog_end, catalog_path, mapping = catalogs[0]
    name_field, description_field, parameters_field = mapping

    offered = {tool.name for tool in tools}
    call_shapes: list[tuple[int, int, ToolCallFields, TextFrame]] = []
    for start, end, value in objects:
        if start == catalog_start and end == catalog_end:
            continue
        if not isinstance(value, dict) or len(value) > 16:
            continue
        possible_names = [
            key
            for key, item in value.items()
            if isinstance(key, str) and isinstance(item, str) and item in offered
        ]
        possible_arguments = [
            key
            for key, item in value.items()
            if isinstance(key, str)
            and isinstance(item, (dict, list))
            and expected_nonce in json.dumps(item, sort_keys=True)
        ]
        for call_name in sorted(possible_names):
            for call_arguments in sorted(possible_arguments):
                possible_ids = [
                    key
                    for key, item in value.items()
                    if key not in {call_name, call_arguments}
                    and isinstance(key, str)
                    and isinstance(item, str)
                    and bool(item)
                ]
                if len(possible_ids) != 1:
                    continue
                before = _adjacent_frame_before(content, start)
                suffix = _adjacent_frame_after(content, end)
                if before is None or suffix is None:
                    continue
                prefix, whitespace_after_prefix = before
                call_shapes.append(
                    (
                        start,
                        end,
                        ToolCallFields(
                            name=call_name,
                            arguments=call_arguments,
                            call_id=possible_ids[0],
                        ),
                        TextFrame(
                            prefix=prefix,
                            suffix=suffix,
                            whitespace_after_prefix=whitespace_after_prefix,
                        ),
                    )
                )
    if len(call_shapes) != 1:
        return RequestProgramDiscovery(
            found=True,
            error=(
                "request call example is missing or ambiguous"
                if not call_shapes
                else "multiple request call examples; refusing ambiguity"
            ),
        )
    _, _, call_fields, call_frame = call_shapes[0]
    preceding_lines = [
        line.strip() for line in content[:catalog_start].rstrip().splitlines() if line.strip()
    ]
    if len(preceding_lines) < 2:
        return RequestProgramDiscovery(
            found=True,
            error="request example must place instruction and heading before the catalog",
        )
    instruction, heading = preceding_lines[-2:]
    tool_fields = ToolDefinitionFields(
        name=name_field,
        description=description_field,
        parameters=parameters_field,
    )
    candidates: list[tuple[RequestPrimitive, ...]] = []
    for role in ("system", "user"):
        candidates.append(
            (
                TemplatedJsonToolCatalogRequest(
                    role=role,
                    instruction=instruction,
                    catalog_heading=heading,
                    catalog_path=list(catalog_path),
                    tool_fields=tool_fields,
                    call_frame=call_frame,
                    fields=call_fields,
                ),
            )
        )
    facts = {
        "instruction": instruction,
        "catalog.heading": heading,
        "catalog.path": list(catalog_path),
        "catalog.fields.name": name_field,
        "catalog.fields.description": description_field,
        "catalog.fields.parameters": parameters_field,
        "call.frame.prefix": call_frame.prefix,
        "call.frame.suffix": call_frame.suffix,
        "call.frame.case_sensitive": call_frame.case_sensitive,
        "call.frame.whitespace_after_prefix": call_frame.whitespace_after_prefix,
        "call.frame.flexible_whitespace": call_frame.flexible_whitespace,
        "call.fields.name": call_fields.name,
        "call.fields.arguments": call_fields.arguments,
        "call.fields.call_id": call_fields.call_id,
    }
    observation = CounterexampleObservation(
        component=ProtocolComponent.REQUEST,
        nonce=expected_nonce,
        constraints=tuple(
            AtomicConstraint(path=path, expected=value) for path, value in sorted(facts.items())
        ),
        found=True,
    )
    return RequestProgramDiscovery(
        candidates=tuple(candidates),
        observation=observation,
        found=True,
    )


def _content_value(result: ToolResult) -> str:
    return result.content if isinstance(result.content, str) else json.dumps(result.content)


def _result_constraints(program: ToolResultMessage) -> tuple[AtomicConstraint, ...]:
    constraints: list[AtomicConstraint] = []
    for index, segment in enumerate(program.segments):
        if isinstance(segment, ResultLiteral):
            constraints.append(AtomicConstraint(f"segments.{index}.literal", segment.text))
            continue
        constraints.extend(
            [
                AtomicConstraint(f"segments.{index}.field", segment.field),
                AtomicConstraint(f"segments.{index}.prefix", segment.prefix),
                AtomicConstraint(f"segments.{index}.suffix", segment.suffix),
                AtomicConstraint(f"segments.{index}.omit_if_none", segment.omit_if_none),
            ]
        )
    return tuple(constraints)


def discover_tool_result_program_from_example(
    raw: Any,
    *,
    result: ToolResult,
    expected_nonce: str,
) -> ToolResultProgramDiscovery:
    """Infer literal/field segments from one exact fresh-sentinel result sample."""
    content = _message_content(raw)
    if content is None or expected_nonce not in content:
        return ToolResultProgramDiscovery(found=False)
    if len(content) > MAX_COUNTEREXAMPLE_CONTENT_CHARS:
        return ToolResultProgramDiscovery(
            found=True, error="tool-result example exceeds the bounded size limit"
        )
    values: list[tuple[str, str]] = []
    if result.name is not None:
        values.append(("name", result.name))
    if result.call_id is not None:
        values.append(("call_id", result.call_id))
    values.append(("content", _content_value(result)))
    if any(value not in content for _, value in values):
        return ToolResultProgramDiscovery(found=False)
    located: list[tuple[int, int, str]] = []
    for field_name, value in values:
        start = content.find(value)
        if start < 0 or content.find(value, start + 1) >= 0:
            return ToolResultProgramDiscovery(
                found=True,
                error=f"tool-result example does not identify {field_name} uniquely",
            )
        located.append((start, start + len(value), field_name))
    located.sort()
    if any(located[index][1] > located[index + 1][0] for index in range(len(located) - 1)):
        return ToolResultProgramDiscovery(found=True, error="tool-result example fields overlap")
    segments: list[ResultLiteral | ResultField] = []
    cursor = 0
    for start, end, field_name in located:
        if start > cursor:
            segments.append(ResultLiteral(text=content[cursor:start]))
        segments.append(ResultField(field=field_name))
        cursor = end
    if cursor < len(content):
        segments.append(ResultLiteral(text=content[cursor:]))
    if len(segments) > MAX_RESULT_SEGMENTS:
        return ToolResultProgramDiscovery(
            found=True, error="inferred tool-result template has too many segments"
        )
    candidates = (
        ToolResultMessage(role="tool", segments=list(segments), attach_tool_call_id=True),
        ToolResultMessage(role="user", segments=list(segments)),
        ToolResultMessage(role="assistant", segments=list(segments)),
    )
    observation = CounterexampleObservation(
        component=ProtocolComponent.TOOL_RESULT,
        nonce=expected_nonce,
        constraints=_result_constraints(candidates[0]),
        found=True,
    )
    return ToolResultProgramDiscovery(
        candidates=candidates,
        observation=observation,
        found=True,
    )


def extract_counterexample(
    raw: Any,
    *,
    expected_component: ProtocolComponent,
    expected_nonce: str,
) -> CounterexampleObservation:
    """Extract one nonce-bound atomic constraint set from a paid response."""
    content = _message_content(raw)
    if content is None:
        return CounterexampleObservation(found=False)
    if len(content) > MAX_COUNTEREXAMPLE_CONTENT_CHARS:
        return CounterexampleObservation(
            found=True, error="counterexample content exceeds the bounded size limit"
        )
    objects = iter_json_objects(content)
    if len(objects) > MAX_COUNTEREXAMPLE_OBJECTS:
        return CounterexampleObservation(
            found=True, error="too many JSON objects in counterexample response"
        )
    marked = [
        value.get("xpt_counterexample")
        for _, _, value in objects
        if isinstance(value, dict) and "xpt_counterexample" in value
    ]
    if not marked:
        return CounterexampleObservation(found=False)
    if len(marked) != 1:
        return CounterexampleObservation(
            found=True, error="multiple protocol counterexamples; refusing ambiguity"
        )
    body = marked[0]
    if not isinstance(body, dict):
        return CounterexampleObservation(
            found=True, error="protocol counterexample body must be an object"
        )
    try:
        component = ProtocolComponent(body.get("component"))
    except (TypeError, ValueError):
        return CounterexampleObservation(
            found=True, error="counterexample has an unsupported component"
        )
    nonce = body.get("nonce")
    if component != expected_component:
        return CounterexampleObservation(
            component=component,
            found=True,
            error=(
                f"counterexample constrains {component.value}, expected {expected_component.value}"
            ),
        )
    if nonce != expected_nonce:
        return CounterexampleObservation(
            component=component,
            found=True,
            error="counterexample nonce does not match the active experiment",
        )
    raw_constraints = body.get("constraints")
    if not isinstance(raw_constraints, list) or not raw_constraints:
        return CounterexampleObservation(
            component=component,
            nonce=nonce if isinstance(nonce, str) else None,
            found=True,
            error="counterexample constraints must be a non-empty list",
        )
    if len(raw_constraints) > MAX_CONSTRAINTS:
        return CounterexampleObservation(
            component=component,
            nonce=nonce,
            found=True,
            error="counterexample exceeds the bounded constraint limit",
        )
    constraints: list[AtomicConstraint] = []
    seen: dict[str, str] = {}
    for index, item in enumerate(raw_constraints):
        if not isinstance(item, dict) or set(item) != {"path", "equals"}:
            return CounterexampleObservation(
                component=component,
                nonce=nonce,
                found=True,
                error=f"constraint {index} must contain only path and equals",
            )
        path = item.get("path")
        if not isinstance(path, str) or not path or len(path) > MAX_CONSTRAINT_PATH_CHARS:
            return CounterexampleObservation(
                component=component,
                nonce=nonce,
                found=True,
                error=f"constraint {index} has an invalid path",
            )
        try:
            encoded = json.dumps(item.get("equals"), sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            return CounterexampleObservation(
                component=component,
                nonce=nonce,
                found=True,
                error=f"constraint {path!r} has a non-JSON value",
            )
        if len(encoded) > MAX_CONSTRAINT_VALUE_CHARS:
            return CounterexampleObservation(
                component=component,
                nonce=nonce,
                found=True,
                error=f"constraint {path!r} exceeds the bounded value limit",
            )
        previous = seen.get(path)
        if previous is not None:
            if previous != encoded:
                return CounterexampleObservation(
                    component=component,
                    nonce=nonce,
                    found=True,
                    error=f"conflicting constraints for {path!r}",
                )
            continue
        seen[path] = encoded
        constraints.append(AtomicConstraint(path=path, expected=item.get("equals")))
    constraints.sort(key=lambda constraint: constraint.path)
    return CounterexampleObservation(
        component=component,
        nonce=nonce,
        constraints=tuple(constraints),
        found=True,
    )


_REQUEST_PATHS = {
    "message.role",
    "instruction",
    "catalog.heading",
    "catalog.path",
    "catalog.fields.name",
    "catalog.fields.description",
    "catalog.fields.parameters",
    "call.frame.prefix",
    "call.frame.suffix",
    "call.frame.case_sensitive",
    "call.frame.whitespace_after_prefix",
    "call.frame.flexible_whitespace",
    "call.fields.name",
    "call.fields.arguments",
    "call.fields.call_id",
}


def synthesize_request_program(
    observation: CounterexampleObservation,
) -> tuple[RequestPrimitive, ...]:
    """Compose one generic nested JSON catalog primitive from atomic facts."""
    if not observation.ok or observation.component != ProtocolComponent.REQUEST:
        raise ValueError(observation.error or "no valid request counterexample")
    values = observation.by_path()
    unknown = sorted(set(values) - _REQUEST_PATHS)
    missing = sorted(_REQUEST_PATHS - set(values))
    if unknown:
        raise ValueError("unsupported request constraint path(s): " + ", ".join(unknown))
    if missing:
        raise ValueError("incomplete request constraints; missing: " + ", ".join(missing))
    role = values["message.role"]
    if role not in {"system", "user"}:
        raise ValueError("request catalog role must be system or user")
    instruction = values["instruction"]
    heading = values["catalog.heading"]
    path = values["catalog.path"]
    if not isinstance(instruction, str) or not isinstance(heading, str):
        raise ValueError("request instruction and catalog heading must be strings")
    if (
        not isinstance(path, list)
        or len(path) > 4
        or any(not isinstance(key, str) or not key for key in path)
    ):
        raise ValueError("catalog path must contain at most four non-empty string keys")
    boolean_paths = (
        "call.frame.case_sensitive",
        "call.frame.whitespace_after_prefix",
        "call.frame.flexible_whitespace",
    )
    if any(not isinstance(values[key], bool) for key in boolean_paths):
        raise ValueError("request frame flags must be booleans")
    string_paths = (
        "catalog.fields.name",
        "catalog.fields.description",
        "catalog.fields.parameters",
        "call.frame.prefix",
        "call.frame.suffix",
        "call.fields.name",
        "call.fields.arguments",
    )
    if any(not isinstance(values[key], str) for key in string_paths):
        raise ValueError("request field names and frame literals must be strings")
    call_id = values["call.fields.call_id"]
    if call_id is not None and not isinstance(call_id, str):
        raise ValueError("request call-id field must be a string or null")
    return (
        TemplatedJsonToolCatalogRequest(
            role=role,
            instruction=instruction,
            catalog_heading=heading,
            catalog_path=list(path),
            tool_fields=ToolDefinitionFields(
                name=values["catalog.fields.name"],
                description=values["catalog.fields.description"],
                parameters=values["catalog.fields.parameters"],
            ),
            call_frame=TextFrame(
                prefix=values["call.frame.prefix"],
                suffix=values["call.frame.suffix"],
                case_sensitive=values["call.frame.case_sensitive"],
                whitespace_after_prefix=values["call.frame.whitespace_after_prefix"],
                flexible_whitespace=values["call.frame.flexible_whitespace"],
            ),
            fields=ToolCallFields(
                name=values["call.fields.name"],
                arguments=values["call.fields.arguments"],
                call_id=call_id,
            ),
        ),
    )


_SEGMENT_PATH = re.compile(
    r"^segments\.(?P<index>0|[1-9][0-9]*)\.(?P<attribute>literal|field|prefix|suffix|omit_if_none)$"
)
_RESULT_BASE_PATHS = {"message.role", "attach_tool_call_id"}


def synthesize_tool_result_program(
    observation: CounterexampleObservation,
) -> ToolResultMessage:
    """Compose a bounded literal/field result renderer from atomic facts."""
    if not observation.ok or observation.component != ProtocolComponent.TOOL_RESULT:
        raise ValueError(observation.error or "no valid tool-result counterexample")
    values = observation.by_path()
    if not _RESULT_BASE_PATHS.issubset(values):
        missing = sorted(_RESULT_BASE_PATHS - set(values))
        raise ValueError("incomplete tool-result constraints; missing: " + ", ".join(missing))
    role = values["message.role"]
    attach = values["attach_tool_call_id"]
    if role not in {"tool", "user", "assistant"}:
        raise ValueError("tool-result role is unsupported")
    if not isinstance(attach, bool):
        raise ValueError("attach_tool_call_id must be boolean")
    grouped: dict[int, dict[str, Any]] = {}
    for path, expected in values.items():
        if path in _RESULT_BASE_PATHS:
            continue
        match = _SEGMENT_PATH.fullmatch(path)
        if match is None:
            raise ValueError(f"unsupported tool-result constraint path: {path}")
        index = int(match.group("index"))
        if index >= MAX_RESULT_SEGMENTS:
            raise ValueError("tool-result segment index exceeds the bounded limit")
        grouped.setdefault(index, {})[match.group("attribute")] = expected
    if not grouped or sorted(grouped) != list(range(len(grouped))):
        raise ValueError("tool-result segment indices must be contiguous from zero")
    segments: list[ResultLiteral | ResultField] = []
    for index in range(len(grouped)):
        item = grouped[index]
        if "literal" in item:
            if set(item) != {"literal"} or not isinstance(item["literal"], str):
                raise ValueError(f"literal result segment {index} is malformed")
            segments.append(ResultLiteral(text=item["literal"]))
            continue
        required = {"field", "prefix", "suffix", "omit_if_none"}
        if set(item) != required:
            missing = sorted(required - set(item))
            extra = sorted(set(item) - required)
            raise ValueError(
                f"field result segment {index} is malformed; missing={missing}, extra={extra}"
            )
        if item["field"] not in {"call_id", "name", "content"}:
            raise ValueError(f"field result segment {index} names an unsupported value")
        if not isinstance(item["prefix"], str) or not isinstance(item["suffix"], str):
            raise ValueError(f"field result segment {index} affixes must be strings")
        if not isinstance(item["omit_if_none"], bool):
            raise ValueError(f"field result segment {index} omit flag must be boolean")
        segments.append(
            ResultField(
                field=item["field"],
                prefix=item["prefix"],
                suffix=item["suffix"],
                omit_if_none=item["omit_if_none"],
            )
        )
    return ToolResultMessage(
        role=role,
        segments=segments,
        attach_tool_call_id=attach,
    )


def evidence_from_counterexample(
    observation: CounterexampleObservation,
    generation: Generation,
) -> list[ComponentEvidence]:
    """Convert every atomic equality into a reusable auditable evidence row."""
    if not observation.ok or observation.component is None:
        return []
    obligations = attributed_obligations(observation.component)
    rows: list[ComponentEvidence] = []
    for constraint in observation.constraints:
        material = json.dumps(
            {
                "generation": generation.index,
                "component": observation.component.value,
                "path": constraint.path,
                "expected": constraint.expected,
                "request": generation.request_hash,
                "response": generation.response_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        evidence_id = "ev_" + hashlib.sha256(material).hexdigest()[:12]
        rows.append(
            ComponentEvidence(
                evidence_id=evidence_id,
                component=observation.component,
                kind=EvidenceKind.CONSTRAINT,
                strength=EvidenceStrength.LOGICAL,
                generation_id=generation.index,
                request_hash=generation.request_hash,
                response_hash=generation.response_hash,
                observation=(
                    f"paid observation constrains {constraint.path} == {constraint.expected!r}"
                ),
                constraint_path=constraint.path,
                expected=constraint.expected,
                obligation_ids=obligations,
            )
        )
    return rows


def proof_evidence(
    *,
    component: ProtocolComponent,
    generation: Generation,
    observation: str,
) -> ComponentEvidence:
    material = (
        f"{generation.index}|{component.value}|{generation.request_hash}|"
        f"{generation.response_hash}|{observation}"
    ).encode()
    return ComponentEvidence(
        evidence_id="ev_" + hashlib.sha256(material).hexdigest()[:12],
        component=component,
        kind=EvidenceKind.PROOF,
        strength=EvidenceStrength.LOGICAL,
        generation_id=generation.index,
        request_hash=generation.request_hash,
        response_hash=generation.response_hash,
        observation=observation,
        obligation_ids=attributed_obligations(component),
    )


def counterexample_evidence(
    *,
    component: ProtocolComponent,
    generation: Generation,
    observation: str,
) -> ComponentEvidence:
    material = (
        f"{generation.index}|{component.value}|{generation.request_hash}|"
        f"{generation.response_hash}|counterexample|{observation}"
    ).encode()
    return ComponentEvidence(
        evidence_id="ev_" + hashlib.sha256(material).hexdigest()[:12],
        component=component,
        kind=EvidenceKind.COUNTEREXAMPLE,
        strength=EvidenceStrength.LOGICAL,
        generation_id=generation.index,
        request_hash=generation.request_hash,
        response_hash=generation.response_hash,
        observation=observation,
        obligation_ids=attributed_obligations(component),
    )
