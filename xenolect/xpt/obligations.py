"""Mandatory Tool ABI v0 obligations and the ABI coverage certificate.

Each entry is tied to a deterministic evaluator/runtime mechanism so a
certificate can be checked without an LLM judge.

An obligation is MANDATORY when a generic stateful driver cannot be called
Tool-ABI-v0 conformant without it. Anything that is a *semantic/intelligence*
property (which tool a model picks for an open-ended natural-language question)
is outside this compatibility certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ObligationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    UNPROVEN = "UNPROVEN"


@dataclass(frozen=True)
class Obligation:
    id: str
    title: str
    mandatory: bool
    # Which Driver IR component(s) the obligation can be blamed on.
    driver_components: tuple[str, ...]
    # Which existing evaluator/ABI mechanism decides it.
    decided_by: str


OBLIGATIONS: tuple[Obligation, ...] = (
    Obligation(
        "OB01",
        "tool call emission",
        True,
        ("tool_encoding", "parser"),
        "evaluator.require_any_tool_call / FailureCategory.RESPONSE_PARSING",
    ),
    Obligation(
        "OB02",
        "tool name resolution (offered set)",
        True,
        ("tool_encoding", "parser"),
        "evaluator: call.name in offered set -> TOOL_REPRESENTATION",
    ),
    Obligation(
        "OB03",
        "argument JSON structural validity",
        True,
        ("schema_transforms", "parser"),
        "eval.schema.validate_tool_arguments -> SCHEMA_REPRESENTATION",
    ),
    Obligation(
        "OB04",
        "argument value integrity",
        True,
        ("tool_encoding", "parser"),
        "evaluator.argument_subsets (exact declared values)",
    ),
    Obligation(
        "OB05",
        "nested / $ref schema support",
        True,
        ("schema_transforms",),
        "eval.schema.validate_tool_arguments on nested object args",
    ),
    Obligation(
        "OB06",
        "parallel tool-call batch",
        True,
        ("parser",),
        "evaluator.require_parallel_batch -> PARALLEL_CALL_PROTOCOL",
    ),
    Obligation(
        "OB07",
        "parallel call cardinality",
        True,
        ("parser",),
        "evaluator.exact_tool_call_count",
    ),
    Obligation(
        # call_id association is required when the assistant emitted ids.
        # Anonymous calls are conformant, a mixture is not — so the
        # obligation is all-or-none discipline, not unconditional presence.
        "OB08",
        "call ID discipline (all-or-none)",
        True,
        ("parser", "tool_result_encoding"),
        "evaluator.require_call_ids -> CALL_ID; abi.trace anonymous-call handling",
    ),
    Obligation(
        "OB09",
        "call ID uniqueness (when ids are emitted)",
        True,
        ("parser",),
        "abi.trace: duplicate call id is illegal",
    ),
    Obligation(
        "OB10",
        "tool result consumption",
        True,
        ("tool_result_encoding",),
        "evaluator.ResultDependency (dynamic sentinel)",
    ),
    Obligation(
        "OB11",
        "parallel result association",
        True,
        ("tool_result_encoding",),
        "evaluator.ResultDependency, one per parallel source call",
    ),
    Obligation(
        "OB12",
        "history preservation across cycles",
        True,
        ("tool_result_encoding", "tool_encoding"),
        "evaluator.min_tool_cycles -> HISTORY_REPRESENTATION",
    ),
    Obligation(
        "OB13",
        "ToolError consumption",
        True,
        ("tool_result_encoding",),
        "evaluator.ResultDependency over ToolError {'error': ...}",
    ),
    Obligation(
        "OB14",
        "error recovery transition",
        True,
        ("tool_result_encoding", "parser"),
        "evaluator.exact_tool_names after a ToolError",
    ),
    Obligation(
        "OB15",
        "no spurious tool call",
        True,
        ("tool_encoding",),
        "evaluator.forbid_tool_calls -> SEMANTIC_DECISION/PROTOCOL",
    ),
    Obligation(
        "OB16",
        "termination with final text",
        True,
        ("parser",),
        "evaluator.require_final_text + abi.trace CHAT terminal",
    ),
    Obligation(
        "OB17",
        "unambiguous parse",
        True,
        ("parser",),
        "xpt.syndrome: multi-parser consensus",
    ),
    Obligation(
        "OB18",
        "legal completed ABI trace",
        True,
        ("tool_encoding", "tool_result_encoding", "parser"),
        "abi.trace.validate_completed_trace",
    ),
)

OBLIGATION_INDEX: dict[str, Obligation] = {o.id: o for o in OBLIGATIONS}
MANDATORY_IDS: tuple[str, ...] = tuple(o.id for o in OBLIGATIONS if o.mandatory)


@dataclass
class ObligationEvidence:
    """One auditable coverage-certificate row."""

    obligation_id: str
    status: ObligationStatus
    generation_id: int | None = None
    request_hash: str | None = None
    response_hash: str | None = None
    observation: str = ""
    driver_components: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        ob = OBLIGATION_INDEX.get(self.obligation_id)
        return {
            "obligation_id": self.obligation_id,
            "title": ob.title if ob else "",
            "status": self.status.value,
            "generation_id": self.generation_id,
            "request_hash": self.request_hash,
            "response_hash": self.response_hash,
            "observation": self.observation,
            "driver_components": list(
                self.driver_components or (ob.driver_components if ob else ())
            ),
            "decided_by": ob.decided_by if ob else "",
            "detail": dict(self.detail),
        }


@dataclass
class CoverageCertificate:
    """Proof-carrying output. `complete` requires 100% mandatory coverage."""

    rows: list[ObligationEvidence] = field(default_factory=list)

    def record(self, evidence: ObligationEvidence) -> None:
        # Last write wins per obligation, but a FAILED row is sticky:
        # you cannot overwrite a failure with a later success in the same run.
        for i, row in enumerate(self.rows):
            if row.obligation_id == evidence.obligation_id:
                if row.status == ObligationStatus.FAILED:
                    return
                self.rows[i] = evidence
                return
        self.rows.append(evidence)

    def status_of(self, obligation_id: str) -> ObligationStatus:
        for row in self.rows:
            if row.obligation_id == obligation_id:
                return row.status
        return ObligationStatus.UNPROVEN

    @property
    def verified_ids(self) -> list[str]:
        return [r.obligation_id for r in self.rows if r.status == ObligationStatus.VERIFIED]

    @property
    def failed_ids(self) -> list[str]:
        return [r.obligation_id for r in self.rows if r.status == ObligationStatus.FAILED]

    @property
    def missing_ids(self) -> list[str]:
        seen = {r.obligation_id for r in self.rows if r.status == ObligationStatus.VERIFIED}
        return [i for i in MANDATORY_IDS if i not in seen]

    @property
    def mandatory_coverage(self) -> float:
        verified = sum(1 for i in MANDATORY_IDS if self.status_of(i) == ObligationStatus.VERIFIED)
        return verified / len(MANDATORY_IDS)

    @property
    def complete(self) -> bool:
        return not self.failed_ids and not self.missing_ids

    def as_dict(self) -> dict[str, Any]:
        return {
            "mandatory_coverage": self.mandatory_coverage,
            "complete": self.complete,
            "failed": self.failed_ids,
            "missing": self.missing_ids,
            "rows": [r.as_dict() for r in self.rows],
        }

    def render(self) -> str:
        lines = []
        for oid in MANDATORY_IDS:
            ob = OBLIGATION_INDEX[oid]
            row = next((r for r in self.rows if r.obligation_id == oid), None)
            status = row.status.value if row else "UNPROVEN"
            has_gen = row is not None and row.generation_id is not None
            witness = f"gen #{row.generation_id:02d}" if has_gen else "-"
            lines.append(f"{oid} {ob.title:<38} {status:<9} {witness}")
        lines.append("")
        lines.append(f"mandatory coverage: {self.mandatory_coverage * 100:.1f}%")
        return "\n".join(lines)
