"""Typed partial protocol hypotheses and reusable component evidence.

This module is deliberately independent from the legacy request frontier.  A
partial hypothesis contains executable v0.2 primitives where they are known and
typed holes where black-box evidence has not resolved a component yet.  Holes
can never be serialized as a Driver: only a fully resolved hypothesis can cross
the production-runtime certification boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from xenolect.driver.ir import (
    REQUIRED_STATE_ACTIONS,
    Driver,
    ProtocolProgram,
    RequestPrimitive,
    ResponsePrimitive,
    SchemaTransform,
    StateAction,
    ToolResultMessage,
)
from xenolect.xpt.obligations import OBLIGATIONS


class ProtocolComponent(StrEnum):
    REQUEST = "request"
    RESPONSE = "response"
    TOOL_RESULT = "tool_result"
    SCHEMA = "schema"
    STATE = "state"


@dataclass(frozen=True)
class RequestProgramHole:
    """Unresolved request behavior with bounded typed properties."""

    unresolved: tuple[str, ...] = (
        "catalog_message_role",
        "catalog_wrapper_path",
        "tool_definition_fields",
        "assistant_call_frame",
        "assistant_call_fields",
    )
    component: ProtocolComponent = ProtocolComponent.REQUEST


@dataclass(frozen=True)
class ResponseProgramHole:
    """Unresolved response parsing behavior."""

    unresolved: tuple[str, ...] = ("parser_primitive", "parser_parameters")
    component: ProtocolComponent = ProtocolComponent.RESPONSE


@dataclass(frozen=True)
class ToolResultProgramHole:
    """Unresolved tool-result message rendering behavior."""

    unresolved: tuple[str, ...] = (
        "message_role",
        "content_segments",
        "tool_call_id_attachment",
    )
    component: ProtocolComponent = ProtocolComponent.TOOL_RESULT


RequestHypothesis = tuple[RequestPrimitive, ...] | RequestProgramHole
ResponseHypothesis = tuple[ResponsePrimitive, ...] | ResponseProgramHole
ToolResultHypothesis = ToolResultMessage | ToolResultProgramHole


def _canonical(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {key: _canonical(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, StrEnum):
        return value.value
    return value


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class PartialProtocolHypothesis:
    """A protocol program that may still contain typed component holes."""

    request: RequestHypothesis = field(default_factory=RequestProgramHole)
    response: ResponseHypothesis = field(default_factory=ResponseProgramHole)
    tool_result: ToolResultHypothesis = field(default_factory=ToolResultProgramHole)
    schema_transforms: tuple[SchemaTransform, ...] = ()
    # State synthesis is explicitly out of scope.  The only legal value is the
    # already-certified Tool ABI action sequence.
    state: tuple[StateAction, ...] = REQUIRED_STATE_ACTIONS

    def __post_init__(self) -> None:
        if self.state != REQUIRED_STATE_ACTIONS:
            raise ValueError("partial hypotheses cannot synthesize state actions")

    @property
    def unresolved_components(self) -> tuple[ProtocolComponent, ...]:
        holes: list[ProtocolComponent] = []
        if isinstance(self.request, RequestProgramHole):
            holes.append(ProtocolComponent.REQUEST)
        if isinstance(self.response, ResponseProgramHole):
            holes.append(ProtocolComponent.RESPONSE)
        if isinstance(self.tool_result, ToolResultProgramHole):
            holes.append(ProtocolComponent.TOOL_RESULT)
        return tuple(holes)

    @property
    def resolved(self) -> bool:
        return not self.unresolved_components

    def component_value(self, component: ProtocolComponent) -> Any:
        return {
            ProtocolComponent.REQUEST: self.request,
            ProtocolComponent.RESPONSE: self.response,
            ProtocolComponent.TOOL_RESULT: self.tool_result,
            ProtocolComponent.SCHEMA: self.schema_transforms,
            ProtocolComponent.STATE: self.state,
        }[component]

    def component_fingerprint(self, component: ProtocolComponent) -> str:
        return _fingerprint(self.component_value(component))

    def refine(
        self,
        component: ProtocolComponent,
        value: tuple[RequestPrimitive, ...] | tuple[ResponsePrimitive, ...] | ToolResultMessage,
    ) -> PartialProtocolHypothesis:
        if component == ProtocolComponent.REQUEST:
            if not isinstance(value, tuple) or not value:
                raise ValueError("request refinement must be a non-empty primitive tuple")
            return replace(self, request=value)
        if component == ProtocolComponent.RESPONSE:
            if not isinstance(value, tuple) or not value:
                raise ValueError("response refinement must be a non-empty primitive tuple")
            return replace(self, response=value)
        if component == ProtocolComponent.TOOL_RESULT:
            if not isinstance(value, ToolResultMessage):
                raise ValueError("tool-result refinement must be ToolResultMessage")
            return replace(self, tool_result=value)
        raise ValueError(f"component {component.value!r} is fixed in this milestone")

    def to_driver(self) -> Driver:
        if not self.resolved:
            names = ", ".join(component.value for component in self.unresolved_components)
            raise ValueError(f"partial protocol hypothesis still has holes: {names}")
        assert isinstance(self.request, tuple)
        assert isinstance(self.response, tuple)
        assert isinstance(self.tool_result, ToolResultMessage)
        return Driver(
            ir_version="0.2",
            schema_transforms=list(self.schema_transforms),
            protocol=ProtocolProgram(
                request=list(self.request),
                response=list(self.response),
                tool_result=self.tool_result,
                state=list(self.state),
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "request": _canonical(self.request),
            "response": _canonical(self.response),
            "tool_result": _canonical(self.tool_result),
            "schema_transforms": [value.value for value in self.schema_transforms],
            "state": [value.value for value in self.state],
            "unresolved_components": [value.value for value in self.unresolved_components],
        }


class EvidenceStrength(StrEnum):
    LOGICAL = "logical"
    HEURISTIC = "heuristic"


class EvidenceKind(StrEnum):
    OBSERVATION = "observation"
    CONSTRAINT = "constraint"
    COUNTEREXAMPLE = "counterexample"
    PROOF = "proof"


@dataclass(frozen=True)
class ComponentEvidence:
    """One reusable fact about an individual protocol component."""

    evidence_id: str
    component: ProtocolComponent
    kind: EvidenceKind
    strength: EvidenceStrength
    generation_id: int
    request_hash: str
    response_hash: str | None
    observation: str
    constraint_path: str | None = None
    expected: Any = None
    obligation_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "component": self.component.value,
            "kind": self.kind.value,
            "strength": self.strength.value,
            "generation_id": self.generation_id,
            "request_hash": self.request_hash,
            "response_hash": self.response_hash,
            "observation": self.observation,
            "constraint_path": self.constraint_path,
            "expected": self.expected,
            "obligation_ids": list(self.obligation_ids),
        }


def attributed_obligations(component: ProtocolComponent) -> tuple[str, ...]:
    aliases = {
        ProtocolComponent.REQUEST: {"tool_encoding", "schema_transforms"},
        ProtocolComponent.RESPONSE: {"parser"},
        ProtocolComponent.TOOL_RESULT: {"tool_result_encoding"},
        ProtocolComponent.SCHEMA: {"schema_transforms"},
        ProtocolComponent.STATE: {"state"},
    }[component]
    return tuple(
        obligation.id
        for obligation in OBLIGATIONS
        if aliases.intersection(obligation.driver_components)
    )


@dataclass
class EvidenceStore:
    """Reusable evidence with proof-only candidate elimination."""

    rows: list[ComponentEvidence] = field(default_factory=list)
    eliminated: dict[ProtocolComponent, set[str]] = field(default_factory=dict)

    def record(self, evidence: ComponentEvidence) -> None:
        if any(row.evidence_id == evidence.evidence_id for row in self.rows):
            return
        self.rows.append(evidence)

    def eliminate(
        self,
        component: ProtocolComponent,
        candidate_fingerprint: str,
        *,
        evidence: ComponentEvidence,
    ) -> None:
        """Eliminate only from a logical counterexample, never a ranking score."""
        if evidence.component != component:
            raise ValueError("component evidence cannot eliminate an unrelated component")
        if evidence.strength != EvidenceStrength.LOGICAL or evidence.kind not in {
            EvidenceKind.COUNTEREXAMPLE,
            EvidenceKind.CONSTRAINT,
        }:
            raise ValueError("heuristic evidence cannot eliminate a protocol hypothesis")
        self.record(evidence)
        self.eliminated.setdefault(component, set()).add(candidate_fingerprint)

    def is_eliminated(self, component: ProtocolComponent, candidate_fingerprint: str) -> bool:
        return candidate_fingerprint in self.eliminated.get(component, set())

    def for_component(self, component: ProtocolComponent) -> list[ComponentEvidence]:
        return [row for row in self.rows if row.component == component]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": [row.as_dict() for row in self.rows],
            "logical_eliminations": {
                component.value: sorted(values)
                for component, values in sorted(
                    self.eliminated.items(), key=lambda item: item[0].value
                )
            },
        }


@dataclass(frozen=True)
class PlannedExperiment:
    component: ProtocolComponent
    implicated_obligations: tuple[str, ...]
    expected_obligation_gain: int
    estimated_generations: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component.value,
            "implicated_obligations": list(self.implicated_obligations),
            "expected_obligation_gain": self.expected_obligation_gain,
            "estimated_generations": self.estimated_generations,
            "reason": self.reason,
            "ranking_is_proof": False,
        }


class ObligationDirectedPlanner:
    """Choose the next unresolved component without using rank as proof."""

    _DEPENDENCY_ORDER = {
        ProtocolComponent.REQUEST: 0,
        ProtocolComponent.RESPONSE: 1,
        ProtocolComponent.TOOL_RESULT: 2,
    }

    def choose(
        self,
        hypothesis: PartialProtocolHypothesis,
        evidence: EvidenceStore,
    ) -> PlannedExperiment | None:
        candidates = list(hypothesis.unresolved_components)
        if not candidates:
            return None

        def constrained(component: ProtocolComponent) -> bool:
            return any(
                row.strength == EvidenceStrength.LOGICAL
                and row.kind in {EvidenceKind.CONSTRAINT, EvidenceKind.COUNTEREXAMPLE}
                for row in evidence.for_component(component)
            )

        def remaining(component: ProtocolComponent) -> tuple[str, ...]:
            required = attributed_obligations(component)
            proven = {
                obligation_id
                for row in evidence.for_component(component)
                if row.kind == EvidenceKind.PROOF
                for obligation_id in row.obligation_ids
            }
            return tuple(value for value in required if value not in proven)

        # Dependency order is required for executable experiments.  Obligation
        # count is only a deterministic tie-break/ranking signal; it never
        # removes another candidate from the hypothesis space.
        candidates.sort(
            key=lambda component: (
                -int(constrained(component)),
                self._DEPENDENCY_ORDER[component],
                -len(remaining(component)),
                component.value,
            )
        )
        component = candidates[0]
        obligations = remaining(component)
        evidence_reason = (
            "logical counterexample constraints implicate this component"
            if constrained(component)
            else "this is the earliest unresolved executable component"
        )
        return PlannedExperiment(
            component=component,
            implicated_obligations=obligations,
            expected_obligation_gain=len(obligations),
            estimated_generations=1,
            reason=(
                f"{component.value}: {evidence_reason}; "
                f"the experiment targets {', '.join(obligations) or 'no mapped ABI row'}"
            ),
        )


@dataclass
class ProtocolSynthesisReport:
    """Auditable constraint and hypothesis-revision trace."""

    evidence: EvidenceStore = field(default_factory=EvidenceStore)
    revisions: list[dict[str, Any]] = field(default_factory=list)
    experiments: list[dict[str, Any]] = field(default_factory=list)
    final_hypothesis: PartialProtocolHypothesis | None = None
    failure: str | None = None
    certification: dict[str, Any] | None = None

    def record_experiment(self, experiment: PlannedExperiment) -> None:
        self.experiments.append(experiment.as_dict())

    def record_revision(
        self,
        before: PartialProtocolHypothesis,
        after: PartialProtocolHypothesis,
        *,
        component: ProtocolComponent,
        generation_id: int,
        evidence_ids: list[str],
        reason: str,
    ) -> None:
        changed = [
            candidate
            for candidate in ProtocolComponent
            if before.component_fingerprint(candidate) != after.component_fingerprint(candidate)
        ]
        if changed != [component]:
            raise ValueError(
                "hypothesis refinement changed unrelated components: "
                + ", ".join(value.value for value in changed)
            )
        self.revisions.append(
            {
                "generation_id": generation_id,
                "component": component.value,
                "changed_components": [value.value for value in changed],
                "before_fingerprint": before.component_fingerprint(component),
                "after_fingerprint": after.component_fingerprint(component),
                "evidence_ids": list(evidence_ids),
                "reason": reason,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": "bounded_obligation_directed_cegis",
            "claim": (
                "actively synthesize and independently certify a previously unseen "
                "request + response + tool-result protocol from black-box observations"
            ),
            "arbitrary_protocol_synthesis": False,
            "state_synthesis": False,
            "experiments": list(self.experiments),
            "evidence": self.evidence.as_dict(),
            "revisions": list(self.revisions),
            "final_hypothesis": (
                self.final_hypothesis.as_dict() if self.final_hypothesis is not None else None
            ),
            "certification": dict(self.certification) if self.certification else None,
            "failure": self.failure,
        }
