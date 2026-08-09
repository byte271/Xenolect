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
    STRUCTURAL_FACT = "structural_fact"
    COMPONENT_OBSERVATION = "component_observation"
    DIAGNOSTIC_WITNESS = "diagnostic_witness"
    NEGATIVE_BEHAVIOR = "negative_behavior"
    COUNTEREXAMPLE = "counterexample"


class ContradictionClass(StrEnum):
    """Why an observation can or cannot soundly eliminate a candidate."""

    STRUCTURAL = "deterministic_structural_contradiction"
    WIRE_API = "deterministic_wire_api_rejection"
    PARSER_SCHEMA = "parser_schema_contradiction"
    ORDINARY_BEHAVIOR = "ordinary_negative_model_behavior"


_SAFE_ELIMINATION_CLASSES = {
    ContradictionClass.STRUCTURAL,
    ContradictionClass.WIRE_API,
    ContradictionClass.PARSER_SCHEMA,
}


@dataclass(frozen=True)
class ComponentEvidence:
    """One reusable fact about a component, never an obligation proof."""

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
    contradiction_class: ContradictionClass | None = None
    determinism_assumption: str | None = None

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
            "contradiction_class": (
                self.contradiction_class.value if self.contradiction_class else None
            ),
            "determinism_assumption": self.determinism_assumption,
        }


@dataclass(frozen=True)
class ObligationSupport:
    """Evidence relevant to an obligation but insufficient to prove it."""

    support_id: str
    obligation_id: str
    generation_ids: tuple[int, ...]
    component_evidence_ids: tuple[str, ...]
    observation: str

    def __post_init__(self) -> None:
        if self.obligation_id not in {obligation.id for obligation in OBLIGATIONS}:
            raise ValueError(f"unknown obligation {self.obligation_id!r}")
        if not self.generation_ids:
            raise ValueError("obligation support requires at least one generation")

    def as_dict(self) -> dict[str, Any]:
        return {
            "support_id": self.support_id,
            "obligation_id": self.obligation_id,
            "status": "SUPPORTING",
            "generation_ids": list(self.generation_ids),
            "component_evidence_ids": list(self.component_evidence_ids),
            "observation": self.observation,
        }


@dataclass(frozen=True)
class ObligationWitness:
    """A complete diagnosis witness for exactly one obligation."""

    witness_id: str
    obligation_id: str
    generation_ids: tuple[int, ...]
    component_evidence_ids: tuple[str, ...]
    observation: str
    scope: str = "diagnosis"

    def __post_init__(self) -> None:
        if self.obligation_id not in {obligation.id for obligation in OBLIGATIONS}:
            raise ValueError(f"unknown obligation {self.obligation_id!r}")
        if not self.generation_ids:
            raise ValueError("complete obligation witness requires observed generations")
        if self.scope != "diagnosis":
            raise ValueError("independent certification has its own certificate boundary")

    def as_dict(self) -> dict[str, Any]:
        return {
            "witness_id": self.witness_id,
            "obligation_id": self.obligation_id,
            "status": "PROVEN",
            "scope": self.scope,
            "generation_ids": list(self.generation_ids),
            "component_evidence_ids": list(self.component_evidence_ids),
            "observation": self.observation,
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
    """Reusable facts, obligation support/witnesses, and SAFE eliminations."""

    rows: list[ComponentEvidence] = field(default_factory=list)
    obligation_support: list[ObligationSupport] = field(default_factory=list)
    obligation_witnesses: list[ObligationWitness] = field(default_factory=list)
    eliminated: dict[ProtocolComponent, set[str]] = field(default_factory=dict)

    def record(self, evidence: ComponentEvidence) -> None:
        if any(row.evidence_id == evidence.evidence_id for row in self.rows):
            return
        self.rows.append(evidence)

    def record_support(self, evidence: ObligationSupport) -> None:
        if any(row.support_id == evidence.support_id for row in self.obligation_support):
            return
        self.obligation_support.append(evidence)

    def record_witness(self, evidence: ObligationWitness) -> None:
        if any(row.witness_id == evidence.witness_id for row in self.obligation_witnesses):
            return
        self.obligation_witnesses.append(evidence)

    @property
    def proven_obligation_ids(self) -> set[str]:
        return {row.obligation_id for row in self.obligation_witnesses}

    def eliminate(
        self,
        component: ProtocolComponent,
        candidate_fingerprint: str,
        *,
        evidence: ComponentEvidence,
    ) -> None:
        """Eliminate only from a deterministic logical contradiction."""
        if evidence.component != component:
            raise ValueError("component evidence cannot eliminate an unrelated component")
        if evidence.strength != EvidenceStrength.LOGICAL:
            raise ValueError("heuristic evidence cannot eliminate a protocol hypothesis")
        if evidence.kind != EvidenceKind.COUNTEREXAMPLE:
            raise ValueError("component facts and positive observations cannot eliminate")
        if evidence.contradiction_class not in _SAFE_ELIMINATION_CLASSES:
            raise ValueError(
                "ordinary model behavior cannot eliminate without an explicit "
                "deterministic contradiction"
            )
        self.record(evidence)
        self.eliminated.setdefault(component, set()).add(candidate_fingerprint)

    def eliminate_by_diagnostic_partition(
        self,
        component: ProtocolComponent,
        candidate_fingerprints: set[str],
        *,
        evidence: ComponentEvidence,
    ) -> None:
        """Eliminate outcomes excluded by an exclusive positive witness.

        This is intentionally separate from contradiction elimination.  A
        nonce-bound structured call can prove which predicted probe partition
        produced the observation even though no candidate contradicted the
        endpoint.  The evidence remains component-level discrimination, never
        an ABI-obligation witness.
        """
        if evidence.component != component:
            raise ValueError("diagnostic evidence cannot refine an unrelated component")
        if evidence.strength != EvidenceStrength.LOGICAL:
            raise ValueError("heuristic diagnostic evidence cannot eliminate")
        if evidence.kind != EvidenceKind.DIAGNOSTIC_WITNESS:
            raise ValueError("partition elimination requires a diagnostic witness")
        self.record(evidence)
        self.eliminated.setdefault(component, set()).update(candidate_fingerprints)

    def is_eliminated(self, component: ProtocolComponent, candidate_fingerprint: str) -> bool:
        return candidate_fingerprint in self.eliminated.get(component, set())

    def for_component(self, component: ProtocolComponent) -> list[ComponentEvidence]:
        return [row for row in self.rows if row.component == component]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": [row.as_dict() for row in self.rows],
            "component_facts": [row.as_dict() for row in self.rows],
            "obligation_support": [row.as_dict() for row in self.obligation_support],
            "obligation_witnesses": [row.as_dict() for row in self.obligation_witnesses],
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
                and row.kind in {EvidenceKind.STRUCTURAL_FACT, EvidenceKind.COUNTEREXAMPLE}
                for row in evidence.for_component(component)
            )

        def remaining(component: ProtocolComponent) -> tuple[str, ...]:
            required = attributed_obligations(component)
            proven = evidence.proven_obligation_ids
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
    version_spaces: list[dict[str, Any]] = field(default_factory=list)
    behavioral_deltas: list[dict[str, Any]] = field(default_factory=list)
    discriminating: bool = False
    oracle_free: bool = False
    probe_plans: list[dict[str, Any]] = field(default_factory=list)
    identifiability: dict[str, Any] | None = None
    final_hypothesis: PartialProtocolHypothesis | None = None
    failure: str | None = None
    failure_class: str | None = None
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
        if self.oracle_free:
            mode = "bounded_oracle_free_diagnostic_synthesis"
            claim = (
                "XPT can synthesize nonce-bound diagnostic probes that distinguish a "
                "bounded protocol hypothesis space and produce an independently "
                "certified working request + response + tool-result Driver without "
                "receiving target values or property-local fault localization from "
                "the endpoint."
            )
        elif self.discriminating:
            mode = "bounded_active_discriminating_synthesis"
            claim = (
                "design discriminating black-box experiments and synthesize a certified "
                "working request + response + tool-result protocol without a target format"
            )
        else:
            mode = "bounded_obligation_directed_cegis"
            claim = (
                "actively synthesize and independently certify a previously unseen "
                "request + response + tool-result protocol from black-box observations"
            )
        return {
            "mode": mode,
            "claim": claim,
            "arbitrary_protocol_synthesis": False,
            "state_synthesis": False,
            "property_local_fault_localization_used": not self.oracle_free,
            "diagnostic_probe_is_production_driver": False,
            "experiments": list(self.experiments),
            "probe_plans": list(self.probe_plans),
            "identifiability": dict(self.identifiability) if self.identifiability else None,
            "version_spaces": list(self.version_spaces),
            "behavioral_deltas": list(self.behavioral_deltas),
            "evidence": self.evidence.as_dict(),
            "revisions": list(self.revisions),
            "final_hypothesis": (
                self.final_hypothesis.as_dict() if self.final_hypothesis is not None else None
            ),
            "certification": dict(self.certification) if self.certification else None,
            "failure": self.failure,
            "failure_class": self.failure_class,
        }
