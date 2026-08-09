"""Oracle-free diagnostic probes over the fixed v0.3 protocol version spaces.

Diagnostic probes are normal Chat Completions wires containing multiple
controlled alternatives.  They are not Drivers, never cross the registry or
runtime boundary, and never prove Tool ABI obligations.  Their sole logical
effect is to refine a finite hypothesis space when the endpoint emits an
exclusive nonce-bound structured witness predicted by one probe partition.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from xenolect.abi.events import AssistantToolCall, ToolCall, ToolCallBatch, ToolDef, ToolResult
from xenolect.driver.encode import (
    build_tool_preamble_messages,
    encode_tool_result_message,
    tools_for_request,
)
from xenolect.driver.ir import Driver
from xenolect.driver.parse import parse_model_response_full
from xenolect.xpt.discrimination import (
    ProtocolVersion,
    RequestVersion,
    ResultVersion,
    request_version_space,
    request_version_to_hypothesis,
    result_version_space,
    result_version_to_program,
)
from xenolect.xpt.hypothesis import (
    ComponentEvidence,
    EvidenceKind,
    EvidenceStrength,
    ProtocolComponent,
)

if TYPE_CHECKING:
    from xenolect.xpt.session import Generation

MAX_PROBE_OUTCOMES = 8
MAX_REQUEST_PROBES = 2
MAX_RESULT_PROBES = 1


class ProbeOutcomeKind(StrEnum):
    EXCLUSIVE_WITNESS = "exclusive_witness"
    GENERIC_REJECTION = "generic_rejection"
    AMBIGUOUS = "ambiguous"
    UNEXPECTED = "unexpected"


@dataclass(frozen=True)
class ProbeWitness:
    """Fresh on-wire values required in one parsed structured continuation."""

    witness_id: str
    tool_name: str
    arguments: dict[str, Any]
    call_id: str
    result_sentinel: str | None = None

    def matches(self, call: ToolCall) -> bool:
        return (
            call.name == self.tool_name
            and call.arguments == self.arguments
            and call.id == self.call_id
        )

    def tokens(self) -> tuple[str, ...]:
        # ``report`` is the public recovery operation used by result probes;
        # the branch-exclusive result sentinel and reply call ID are the fresh
        # canaries in that family. Request probes also mint a fresh tool name.
        values = [self.call_id]
        if self.result_sentinel is None:
            values.append(self.tool_name)
        values.extend(str(value) for value in self.arguments.values())
        if self.result_sentinel is not None:
            values.append(self.result_sentinel)
        return tuple(dict.fromkeys(values))

    def as_dict(self) -> dict[str, Any]:
        return {
            "witness_id": self.witness_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "call_id": self.call_id,
            "result_sentinel": self.result_sentinel,
        }


@dataclass(frozen=True)
class ProbeAlternative:
    """One exact production-version alternative embedded in a diagnostic wire."""

    alternative_id: str
    hypothesis_fingerprint: str
    outcome_key: str
    witness: ProbeWitness
    version: ProtocolVersion
    message_indexes: tuple[int, ...] = ()
    native_tool_names: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "alternative_id": self.alternative_id,
            "hypothesis_fingerprint": self.hypothesis_fingerprint,
            "outcome_key": self.outcome_key,
            "witness_id": self.witness.witness_id,
            "version": self.version.as_dict(),
            "message_indexes": list(self.message_indexes),
            "native_tool_names": list(self.native_tool_names),
        }


@dataclass(frozen=True)
class ProbePartition:
    outcome_key: str
    witness: ProbeWitness
    member_fingerprints: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome_key": self.outcome_key,
            "witness": self.witness.as_dict(),
            "members": list(self.member_fingerprints),
            "size": len(self.member_fingerprints),
        }


@dataclass
class ProbePlan:
    """Predicted outcome partition plus its auditable observed transition."""

    probe_id: str
    component: ProtocolComponent
    survivor_fingerprints_before: tuple[str, ...]
    partitions: tuple[ProbePartition, ...]
    worst_case_survivors: int
    information_score: float
    estimated_generations: int
    estimated_wire_size: int
    complexity: int
    fingerprint: str
    actual_observed_outcome: str | None = None
    observed_kind: ProbeOutcomeKind | None = None
    witness_evidence_ids: list[str] = field(default_factory=list)
    hypotheses_removed: list[str] = field(default_factory=list)
    hypotheses_remaining: list[str] = field(default_factory=list)
    elimination_reasons: list[dict[str, Any]] = field(default_factory=list)

    def partition(self, outcome_key: str) -> ProbePartition | None:
        return next(
            (partition for partition in self.partitions if partition.outcome_key == outcome_key),
            None,
        )

    def record_outcome(self, outcome: ProbeOutcome, *, evidence_id: str | None = None) -> None:
        self.observed_kind = outcome.kind
        self.actual_observed_outcome = outcome.outcome_key
        if evidence_id is not None:
            self.witness_evidence_ids.append(evidence_id)
        if outcome.kind != ProbeOutcomeKind.EXCLUSIVE_WITNESS or outcome.outcome_key is None:
            self.hypotheses_remaining = list(self.survivor_fingerprints_before)
            return
        partition = self.partition(outcome.outcome_key)
        if partition is None:
            self.observed_kind = ProbeOutcomeKind.UNEXPECTED
            self.hypotheses_remaining = list(self.survivor_fingerprints_before)
            return
        remaining = set(partition.member_fingerprints)
        removed = set(self.survivor_fingerprints_before) - remaining
        self.hypotheses_removed = sorted(removed)
        self.hypotheses_remaining = sorted(remaining)
        self.elimination_reasons = [
            {
                "hypothesis_fingerprint": fingerprint,
                "basis": "exclusive_nonce_bound_structured_witness",
                "observed_outcome": outcome.outcome_key,
                "reason": (
                    "the hypothesis predicted a different positive canary outcome; "
                    "the observed structured call exactly matched only this partition"
                ),
            }
            for fingerprint in sorted(removed)
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "component": self.component.value,
            "surviving_hypothesis_fingerprints_before": list(
                self.survivor_fingerprints_before
            ),
            "predicted_outcome_classes": [partition.as_dict() for partition in self.partitions],
            "worst_case_survivors": self.worst_case_survivors,
            "information_score": self.information_score,
            "information_score_basis": (
                "fraction_of_exact_hypothesis_pairs_separated"
            ),
            "probability_prior_used": False,
            "information_score_is_proof": False,
            "minimax_score_is_proof": False,
            "estimated_generations": self.estimated_generations,
            "estimated_wire_size": self.estimated_wire_size,
            "complexity": self.complexity,
            "stable_fingerprint": self.fingerprint,
            "actual_observed_outcome": self.actual_observed_outcome,
            "observed_kind": self.observed_kind.value if self.observed_kind else None,
            "witness_evidence_ids": list(self.witness_evidence_ids),
            "hypotheses_removed": list(self.hypotheses_removed),
            "hypotheses_remaining": list(self.hypotheses_remaining),
            "elimination_reasons": list(self.elimination_reasons),
        }


@dataclass(frozen=True)
class DiagnosticProbe:
    """A non-persistent normal-wire experiment, explicitly not a Driver."""

    probe_id: str
    component: ProtocolComponent
    nonce: str
    alternatives: tuple[ProbeAlternative, ...]
    partitions: tuple[ProbePartition, ...]
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.alternatives or not self.partitions:
            raise ValueError("diagnostic probe requires alternatives and predicted outcomes")
        if len(self.partitions) > MAX_PROBE_OUTCOMES:
            raise ValueError("diagnostic probe exceeds the bounded outcome count")
        members = [
            fingerprint
            for partition in self.partitions
            for fingerprint in partition.member_fingerprints
        ]
        alternatives = [alternative.hypothesis_fingerprint for alternative in self.alternatives]
        if sorted(members) != sorted(alternatives) or len(set(members)) != len(members):
            raise ValueError("probe partitions must cover every alternative exactly once")
        forbidden = {"XPT_PROBE", "diagnostic", "probe_id", "partition_id"}
        wire_keys = {
            str(key)
            for message in self.messages
            for key in message
        } | {str(key) for tool in self.tools for key in tool}
        if forbidden.intersection(wire_keys):
            raise ValueError("diagnostic metadata cannot enter the endpoint wire")

    def wire(self) -> dict[str, Any]:
        return {
            "messages": [dict(message) for message in self.messages],
            "tools": [dict(tool) for tool in self.tools] or None,
            "seed": None,
        }

    @property
    def wire_size(self) -> int:
        return len(json.dumps(self.wire(), sort_keys=True, separators=(",", ":")))

    def as_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "component": self.component.value,
            "nonce": self.nonce,
            "alternatives": [alternative.as_dict() for alternative in self.alternatives],
            "partitions": [partition.as_dict() for partition in self.partitions],
            "wire_size": self.wire_size,
            "production_driver": False,
            "registry_eligible": False,
            "abi_witness": False,
        }


@dataclass(frozen=True)
class ProbeOutcome:
    kind: ProbeOutcomeKind
    outcome_key: str | None = None
    witness_ids: tuple[str, ...] = ()
    parsed_calls: tuple[dict[str, Any], ...] = ()
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "outcome_key": self.outcome_key,
            "witness_ids": list(self.witness_ids),
            "parsed_calls": list(self.parsed_calls),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class IdentifiabilityReport:
    component: ProtocolComponent
    survivor_fingerprints: tuple[str, ...]
    separable_pairs: int
    total_pairs: int
    indistinguishable_pairs: tuple[tuple[str, str], ...]

    @property
    def identifiable(self) -> bool:
        return not self.indistinguishable_pairs

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component.value,
            "survivor_fingerprints": list(self.survivor_fingerprints),
            "separable_pairs": self.separable_pairs,
            "total_pairs": self.total_pairs,
            "indistinguishable_pairs": [list(pair) for pair in self.indistinguishable_pairs],
            "identifiable": self.identifiable,
            "equivalence_claimed": False,
        }


@dataclass(frozen=True)
class _PartitionCandidate:
    assignment: tuple[tuple[str, int], ...]
    worst_case: int
    information_score: float
    estimated_wire_size: int
    complexity: int
    fingerprint: str


def _hash(value: Any, length: int = 16) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:length]


def _partition_information(sizes: Iterable[int]) -> float:
    values = tuple(sizes)
    total = sum(values)
    if total <= 1:
        return 0.0
    total_pairs = total * (total - 1) // 2
    unresolved_pairs = sum(size * (size - 1) // 2 for size in values)
    # Count-based partition information: the fraction of exact hypothesis pairs
    # separated by the probe. No endpoint-outcome probability prior is invented.
    return round((total_pairs - unresolved_pairs) / total_pairs, 6)


def _universe(component: ProtocolComponent) -> tuple[ProtocolVersion, ...]:
    if component == ProtocolComponent.REQUEST:
        return request_version_space()
    if component == ProtocolComponent.TOOL_RESULT:
        return result_version_space()
    raise ValueError(f"no diagnostic probe family for {component.value}")


def _partition_candidates(
    component: ProtocolComponent,
    survivors: Sequence[ProtocolVersion],
) -> tuple[_PartitionCandidate, ...]:
    if len(survivors) < 2:
        return ()
    universe = _universe(component)
    global_index = {version.fingerprint: index for index, version in enumerate(universe)}
    survivor_fingerprints = {version.fingerprint for version in survivors}
    if len(survivor_fingerprints) != len(survivors):
        raise ValueError("only exact fingerprint identity may deduplicate versions")

    assignments: set[tuple[tuple[str, int], ...]] = set()
    if len(survivors) <= MAX_PROBE_OUTCOMES:
        assignments.add(
            tuple(
                (version.fingerprint, index)
                for index, version in enumerate(
                    sorted(survivors, key=lambda value: value.fingerprint)
                )
            )
        )
    max_bits = max(1, math.ceil(math.log2(len(universe))))
    # Base-8 digit partitions identify every current request version in two
    # adaptive probes while keeping each observable outcome count <= 8.
    for shift in range(0, max_bits, 3):
        assignments.add(
            tuple(
                sorted(
                    (
                        version.fingerprint,
                        (global_index[version.fingerprint] >> shift) & 0b111,
                    )
                    for version in survivors
                )
            )
        )
    # Binary bit partitions guarantee pairwise separability and make accidental
    # indistinguishable classes visible to the offline identifiability gate.
    for bit in range(max_bits):
        assignments.add(
            tuple(
                sorted(
                    (
                        version.fingerprint,
                        (global_index[version.fingerprint] >> bit) & 1,
                    )
                    for version in survivors
                )
            )
        )

    candidates: list[_PartitionCandidate] = []
    for assignment in assignments:
        groups: dict[int, list[str]] = {}
        for fingerprint, outcome in assignment:
            groups.setdefault(outcome, []).append(fingerprint)
        if len(groups) < 2 or len(groups) > MAX_PROBE_OUTCOMES:
            continue
        normalized = {
            old: new for new, old in enumerate(sorted(groups))
        }
        remapped = tuple((fingerprint, normalized[outcome]) for fingerprint, outcome in assignment)
        sizes = [len(groups[old]) for old in sorted(groups)]
        estimated_wire_size = sum(
            len(json.dumps(version.as_dict(), sort_keys=True)) + 256 for version in survivors
        )
        fingerprint = _hash(
            {
                "component": component.value,
                "assignment": remapped,
            }
        )
        candidates.append(
            _PartitionCandidate(
                assignment=remapped,
                worst_case=max(sizes),
                information_score=_partition_information(sizes),
                estimated_wire_size=estimated_wire_size,
                complexity=len(groups),
                fingerprint=fingerprint,
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.worst_case,
                -candidate.information_score,
                1,
                candidate.estimated_wire_size,
                candidate.complexity,
                candidate.fingerprint,
            ),
        )
    )


def check_identifiability(
    component: ProtocolComponent,
    survivors: Sequence[ProtocolVersion],
) -> IdentifiabilityReport:
    candidates = _partition_candidates(component, survivors)
    pairs: list[tuple[str, str]] = []
    fingerprints = sorted(version.fingerprint for version in survivors)
    for index, left in enumerate(fingerprints):
        for right in fingerprints[index + 1 :]:
            pairs.append((left, right))
    indistinguishable: list[tuple[str, str]] = []
    for left, right in pairs:
        separable = any(
            dict(candidate.assignment)[left] != dict(candidate.assignment)[right]
            for candidate in candidates
        )
        if not separable:
            indistinguishable.append((left, right))
    return IdentifiabilityReport(
        component=component,
        survivor_fingerprints=tuple(fingerprints),
        separable_pairs=len(pairs) - len(indistinguishable),
        total_pairs=len(pairs),
        indistinguishable_pairs=tuple(indistinguishable),
    )


def _mint_witnesses(
    *,
    component: ProtocolComponent,
    seed: int,
    sequence: int,
    outcome_count: int,
    collision_material: str,
) -> tuple[ProbeWitness, ...]:
    nonce = _hash({"seed": seed, "sequence": sequence, "component": component.value}, 20)
    witnesses: list[ProbeWitness] = []
    seen: set[str] = set()
    for outcome in range(outcome_count):
        token = _hash({"nonce": nonce, "outcome": outcome}, 12)
        if component == ProtocolComponent.REQUEST:
            witness = ProbeWitness(
                witness_id="pw_" + token,
                tool_name="diag_req_" + token,
                arguments={"probe_value": "arg_" + token},
                call_id="call_req_" + token,
            )
        else:
            sentinel = "result_" + token
            witness = ProbeWitness(
                witness_id="pw_" + token,
                tool_name="report",
                arguments={"code": sentinel},
                call_id="call_result_" + token,
                result_sentinel=sentinel,
            )
        for value in witness.tokens():
            if value in seen or value in collision_material:
                raise ValueError("diagnostic canary collision detected")
            seen.add(value)
        witnesses.append(witness)
    return tuple(witnesses)


def _resolved_driver(
    request_version: RequestVersion,
    result_version: ResultVersion | None = None,
) -> Driver:
    hypothesis = request_version_to_hypothesis(request_version)
    result = result_version_to_program(
        result_version or ResultVersion("tool", "attachment")
    )
    return hypothesis.refine(ProtocolComponent.TOOL_RESULT, result).to_driver()


def _make_plan(
    *,
    component: ProtocolComponent,
    survivors: Sequence[ProtocolVersion],
    seed: int,
    sequence: int,
    collision_material: str,
) -> tuple[ProbePlan, tuple[ProbeWitness, ...], dict[str, int]]:
    identifiability = check_identifiability(component, survivors)
    if not identifiability.identifiable:
        raise ValueError(
            "protocol version space is observationally unidentifiable under the "
            "available diagnostic probe family"
        )
    candidates = _partition_candidates(component, survivors)
    if not candidates:
        raise ValueError("no discriminating probe remains for the surviving hypotheses")
    selected = candidates[0]
    assignment = dict(selected.assignment)
    outcome_count = len(set(assignment.values()))
    witnesses = _mint_witnesses(
        component=component,
        seed=seed,
        sequence=sequence,
        outcome_count=outcome_count,
        collision_material=collision_material,
    )
    partitions = tuple(
        ProbePartition(
            outcome_key=f"outcome-{index}",
            witness=witnesses[index],
            member_fingerprints=tuple(
                sorted(
                    version.fingerprint
                    for version in survivors
                    if assignment[version.fingerprint] == index
                )
            ),
        )
        for index in range(outcome_count)
    )
    probe_id = f"diagnostic-{component.value}-{sequence}-{selected.fingerprint[:8]}"
    plan = ProbePlan(
        probe_id=probe_id,
        component=component,
        survivor_fingerprints_before=tuple(
            sorted(version.fingerprint for version in survivors)
        ),
        partitions=partitions,
        worst_case_survivors=selected.worst_case,
        information_score=selected.information_score,
        estimated_generations=1,
        estimated_wire_size=selected.estimated_wire_size,
        complexity=selected.complexity,
        fingerprint=selected.fingerprint,
    )
    return plan, witnesses, assignment


def build_request_probe(
    survivors: Sequence[RequestVersion],
    *,
    seed: int,
    sequence: int,
) -> tuple[DiagnosticProbe, ProbePlan]:
    plan, witnesses, assignment = _make_plan(
        component=ProtocolComponent.REQUEST,
        survivors=survivors,
        seed=seed,
        sequence=sequence,
        collision_material="record_alpha record_beta record_gamma commit report",
    )
    messages: list[dict[str, Any]] = []
    native_tools: list[dict[str, Any]] = []
    alternatives: list[ProbeAlternative] = []
    for version in sorted(survivors, key=lambda value: value.fingerprint):
        outcome_index = assignment[version.fingerprint]
        witness = witnesses[outcome_index]
        tool = ToolDef(
            name=witness.tool_name,
            description=(
                "Call this function exactly once and use call id " + witness.call_id
            ),
            parameters={
                "type": "object",
                "properties": {
                    "probe_value": {
                        "$ref": "#/$defs/ProbeValue",
                    }
                },
                "required": ["probe_value"],
                "$defs": {
                    "ProbeValue": {
                        "type": "string",
                        "const": witness.arguments["probe_value"],
                    }
                },
            },
        )
        driver = _resolved_driver(version)
        start = len(messages)
        if version.mode == "native":
            encoded_tools = tools_for_request([tool], driver)
            native_tools.extend(encoded_tools)
            indexes: tuple[int, ...] = ()
            tool_names = (witness.tool_name,)
        else:
            encoded_messages = build_tool_preamble_messages([tool], driver)
            messages.extend(encoded_messages)
            indexes = tuple(range(start, len(messages)))
            tool_names = ()
        alternatives.append(
            ProbeAlternative(
                alternative_id="alt_" + _hash(
                    {"probe": plan.probe_id, "version": version.fingerprint}, 12
                ),
                hypothesis_fingerprint=version.fingerprint,
                outcome_key=f"outcome-{outcome_index}",
                witness=witness,
                version=version,
                message_indexes=indexes,
                native_tool_names=tool_names,
            )
        )
    messages.append(
        {
            "role": "user",
            "content": (
                "Use exactly one compatible tool catalog. Call its only function once, "
                "copy the schema's required probe_value exactly, and use the call id "
                "stated in that function description. Return a structured tool call."
            ),
        }
    )
    probe = DiagnosticProbe(
        probe_id=plan.probe_id,
        component=ProtocolComponent.REQUEST,
        nonce=_hash({"seed": seed, "sequence": sequence, "component": "request"}, 20),
        alternatives=tuple(alternatives),
        partitions=plan.partitions,
        messages=tuple(messages),
        tools=tuple(native_tools),
    )
    plan.estimated_wire_size = probe.wire_size
    return probe, plan


def build_result_probe(
    survivors: Sequence[ResultVersion],
    *,
    request_version: RequestVersion,
    prefix_messages: Sequence[dict[str, Any]],
    tools: Sequence[ToolDef],
    call: ToolCall,
    seed: int,
    sequence: int,
) -> tuple[DiagnosticProbe, ProbePlan]:
    collision_material = json.dumps(prefix_messages, sort_keys=True, default=str)
    plan, witnesses, assignment = _make_plan(
        component=ProtocolComponent.TOOL_RESULT,
        survivors=survivors,
        seed=seed,
        sequence=sequence,
        collision_material=collision_material,
    )
    messages = [dict(message) for message in prefix_messages]
    alternatives: list[ProbeAlternative] = []
    for version in sorted(survivors, key=lambda value: value.fingerprint):
        outcome_index = assignment[version.fingerprint]
        witness = witnesses[outcome_index]
        driver = _resolved_driver(request_version, version)
        content = {
            "probe_result": witness.result_sentinel,
            "reply_call_id": witness.call_id,
        }
        message = encode_tool_result_message(
            ToolResult(call_id=call.id, name=call.name, content=content),
            driver,
        )
        message_index = len(messages)
        messages.append(message)
        alternatives.append(
            ProbeAlternative(
                alternative_id="alt_" + _hash(
                    {"probe": plan.probe_id, "version": version.fingerprint}, 12
                ),
                hypothesis_fingerprint=version.fingerprint,
                outcome_key=f"outcome-{outcome_index}",
                witness=witness,
                version=version,
                message_indexes=(message_index,),
            )
        )
    messages.append(
        {
            "role": "user",
            "content": (
                "Consume the one compatible result representation. Call report with "
                "code equal to its exact probe_result value and use its reply_call_id. "
                "Return a structured tool call; do not repeat the value as plain text."
            ),
        }
    )
    request_driver = _resolved_driver(request_version)
    wire_tools = (
        tools_for_request(list(tools), request_driver)
        if request_version.mode == "native"
        else []
    )
    probe = DiagnosticProbe(
        probe_id=plan.probe_id,
        component=ProtocolComponent.TOOL_RESULT,
        nonce=_hash({"seed": seed, "sequence": sequence, "component": "result"}, 20),
        alternatives=tuple(alternatives),
        partitions=plan.partitions,
        messages=tuple(messages),
        tools=tuple(wire_tools),
    )
    plan.estimated_wire_size = probe.wire_size
    return probe, plan


def candidate_drivers_for_probe(
    request_versions: Sequence[RequestVersion],
    *,
    result_version: ResultVersion | None = None,
) -> tuple[Driver, ...]:
    by_fingerprint: dict[str, Driver] = {}
    for version in request_versions:
        driver = _resolved_driver(version, result_version)
        fingerprint = _hash(driver.canonical_dict())
        by_fingerprint.setdefault(fingerprint, driver)
    return tuple(by_fingerprint[key] for key in sorted(by_fingerprint))


def _calls(raw: Any, driver: Driver) -> tuple[ToolCall, ...] | None:
    if not isinstance(raw, dict):
        return None
    parsed = parse_model_response_full(raw, driver)
    if parsed.errors:
        return None
    calls: list[ToolCall] = []
    for event in parsed.events:
        if isinstance(event, AssistantToolCall):
            calls.append(event.call)
        elif isinstance(event, ToolCallBatch):
            calls.extend(event.calls)
    return tuple(calls)


def is_generic_invalid_request(raw: Any) -> bool:
    if not isinstance(raw, dict) or set(raw) != {"error"}:
        return False
    error = raw.get("error")
    return (
        isinstance(error, dict)
        and set(error) == {"type", "message"}
        and error.get("type") in {"invalid_request_error", "invalid_tool_result"}
        and isinstance(error.get("message"), str)
    )


def observe_probe_response(
    probe: DiagnosticProbe,
    raw: Any,
    drivers: Sequence[Driver],
) -> ProbeOutcome:
    """Accept only one exact structured canary interpretation across parsers."""
    witnesses = {
        partition.outcome_key: partition.witness for partition in probe.partitions
    }
    interpretations: set[tuple[str, str, str]] = set()
    calls_by_interpretation: list[dict[str, Any]] = []
    saw_unmatched_structured_call = False
    for driver in drivers:
        parsed_calls = _calls(raw, driver)
        if not parsed_calls:
            continue
        if len(parsed_calls) != 1:
            return ProbeOutcome(
                ProbeOutcomeKind.AMBIGUOUS,
                parsed_calls=tuple(call.model_dump(mode="json") for call in parsed_calls),
                reason="a diagnostic response produced multiple structured calls",
            )
        call = parsed_calls[0]
        matches = [key for key, witness in witnesses.items() if witness.matches(call)]
        if len(matches) != 1:
            saw_unmatched_structured_call = True
            continue
        outcome_key = matches[0]
        canonical = json.dumps(call.model_dump(mode="json"), sort_keys=True)
        interpretations.add((outcome_key, witnesses[outcome_key].witness_id, canonical))
        calls_by_interpretation.append(call.model_dump(mode="json"))
    if saw_unmatched_structured_call:
        return ProbeOutcome(
            ProbeOutcomeKind.UNEXPECTED,
            parsed_calls=tuple(calls_by_interpretation),
            reason="structured output did not exactly match a predicted nonce-bound witness",
        )
    if len(interpretations) == 1:
        outcome_key, witness_id, _ = next(iter(interpretations))
        return ProbeOutcome(
            ProbeOutcomeKind.EXCLUSIVE_WITNESS,
            outcome_key=outcome_key,
            witness_ids=(witness_id,),
            parsed_calls=tuple(calls_by_interpretation[:1]),
            reason="all successful parser interpretations agree on one exclusive witness",
        )
    if len(interpretations) > 1:
        return ProbeOutcome(
            ProbeOutcomeKind.AMBIGUOUS,
            witness_ids=tuple(sorted(value[1] for value in interpretations)),
            parsed_calls=tuple(calls_by_interpretation),
            reason="parser interpretations disagree or multiple canaries were consumed",
        )
    if is_generic_invalid_request(raw):
        return ProbeOutcome(
            ProbeOutcomeKind.GENERIC_REJECTION,
            reason="generic rejection contains no exclusive positive partition witness",
        )
    return ProbeOutcome(
        ProbeOutcomeKind.UNEXPECTED,
        reason="response matches no valid predicted diagnostic outcome",
    )


def diagnostic_witness_evidence(
    *,
    component: ProtocolComponent,
    generation: Generation,
    plan: ProbePlan,
    outcome: ProbeOutcome,
) -> ComponentEvidence:
    if outcome.kind != ProbeOutcomeKind.EXCLUSIVE_WITNESS or outcome.outcome_key is None:
        raise ValueError("only an exclusive positive outcome creates diagnostic evidence")
    material = {
        "component": component.value,
        "generation": generation.index,
        "request": generation.request_hash,
        "response": generation.response_hash,
        "probe": plan.fingerprint,
        "outcome": outcome.outcome_key,
        "witnesses": outcome.witness_ids,
    }
    return ComponentEvidence(
        evidence_id="diag_" + _hash(material, 12),
        component=component,
        kind=EvidenceKind.DIAGNOSTIC_WITNESS,
        strength=EvidenceStrength.LOGICAL,
        generation_id=generation.index,
        request_hash=generation.request_hash,
        response_hash=generation.response_hash,
        observation=(
            "a parsed structured call exactly matched the fresh exclusive canary for "
            f"predicted {outcome.outcome_key}; diagnostic evidence does not prove ABI obligations"
        ),
        constraint_path="diagnostic_probe.predicted_outcome",
        expected=outcome.outcome_key,
    )
