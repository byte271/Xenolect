"""Deterministic evaluator: no LLM-as-judge for protocol correctness.

Core invariants:
  E1: max-cycle exhaustion cannot PASS
  E2: malformed tool frames cannot be silently discarded
  E3: exact call cardinality is enforced when declared
  E4: completed probe != legal prefix
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from xenolect.abi.events import (
    AssistantText,
    AssistantToolCall,
    Event,
    ToolCall,
    ToolCallBatch,
    ToolDef,
    ToolError,
    ToolResult,
    UserMessage,
)
from xenolect.abi.trace import validate_completed_trace, validate_trace
from xenolect.driver.termination import Termination
from xenolect.eval.schema import validate_tool_arguments
from xenolect.eval.termination import assess_event_g3_termination

ProbeKind = Literal["protocol", "semantic"]


@dataclass(frozen=True)
class ResultDependency:
    """Require a later tool argument to equal data from an earlier tool result.

    Paths are key sequences into JSON-like objects. The target call must occur
    after the matching source ToolResult/ToolError in the normalized trace.
    """

    source_tool: str
    source_path: tuple[str, ...]
    target_tool: str
    target_argument_path: tuple[str, ...]


class FailureCategory(str, Enum):
    NONE = "none"
    TRACE_ILLEGAL = "trace_illegal"
    RESPONSE_PARSING = "response_parsing"
    TOOL_REPRESENTATION = "tool_representation"
    SCHEMA_REPRESENTATION = "schema_representation"
    HISTORY_REPRESENTATION = "history_representation"
    TOOL_RESULT_ENCODING = "tool_result_encoding"
    CALL_ID = "call_id"
    PARALLEL_CALL_PROTOCOL = "parallel_call_protocol"
    SEMANTIC_TOOL_SELECTION = "semantic_tool_selection"
    SEMANTIC_ARGUMENT_VALUE = "semantic_argument_value"
    SEMANTIC_DECISION = "semantic_decision"
    PROTOCOL_EXPECTATION = "protocol_expectation"
    INCOMPLETE_TRACE = "incomplete_trace"
    MAX_CYCLES = "max_cycles"
    OTHER = "other"


@dataclass
class EvalResult:
    passed: bool
    interface_ok: bool
    semantic_ok: bool | None
    category: FailureCategory
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "interface_ok": self.interface_ok,
            "semantic_ok": self.semantic_ok,
            "category": self.category.value,
            "errors": list(self.errors),
            "details": dict(self.details),
        }


@dataclass
class ProbeExpectation:
    """Constraints on a probe execution.

    Protocol fields (interface when kind=protocol):
      exact_tool_names — exact ordered list of tool names (cardinality + sequence)
      exact_tool_call_count — total number of tool calls
      allowed_tool_names — if set, any call outside this set fails
      required_tool_names — at-least-once membership (weaker; prefer exact_*)
      require_completed — completed-trace semantics (default True for protocol)
      require_parallel_batch — first tool turn must be a single ToolCallBatch
      ...
    """

    required_tool_names: list[str] | None = None
    ordered_tool_names: bool = False
    exact_tool_names: list[str] | None = None
    exact_tool_call_count: int | None = None
    allowed_tool_names: list[str] | None = None
    argument_subsets: list[tuple[str, dict[str, Any]]] | None = None
    # Later-cycle argument checks: (tool_name, subset) against any call
    require_final_text: bool = False
    forbid_tool_calls: bool = False
    min_tool_cycles: int = 0
    require_any_tool_call: bool = False
    require_completed: bool = True
    require_parallel_batch: bool = False
    require_call_ids: bool = False
    expected_primary_tool: str | None = None
    # Optional final text exact match (protocol no-call scripts)
    expected_final_text: str | None = None
    # Protocol-semantic final witness. The sentinel must come from this tool's
    # result after being absent from all earlier trace material.
    expected_final_sentinel: str | None = None
    final_sentinel_source_tool: str | None = None
    # Optional instruction-following diagnostic; never an interface gate.
    diagnostic_expected_final_text: str | None = None
    result_dependencies: list[ResultDependency] | None = None


def evaluate_trace(
    events: list[Event],
    *,
    tools: list[ToolDef] | None = None,
    expectation: ProbeExpectation | None = None,
    kind: ProbeKind = "protocol",
    termination: Termination | None = None,
    parse_errors: list[str] | None = None,
) -> EvalResult:
    """Evaluate protocol/interface and optional semantic conformance."""
    tool_map = {t.name: t for t in (tools or [])}

    if not tool_map:
        for e in events:
            if isinstance(e, UserMessage) and e.tools:
                tool_map = {t.name: t for t in e.tools}
                break

    details: dict[str, Any] = {"kind": kind}
    if termination is not None:
        details["termination"] = termination.value

    # Parse errors always interface failure
    if parse_errors:
        return EvalResult(
            passed=False,
            interface_ok=False,
            semantic_ok=None,
            category=FailureCategory.RESPONSE_PARSING,
            errors=list(parse_errors),
            details=details,
        )

    # Termination failures
    if termination == Termination.MAX_CYCLES_EXHAUSTED:
        return EvalResult(
            passed=False,
            interface_ok=False,
            semantic_ok=None,
            category=FailureCategory.MAX_CYCLES,
            errors=["probe terminated: max_cycles_exhausted"],
            details=details,
        )
    if termination in (
        Termination.INFRASTRUCTURE_FAILED,
        Termination.CONFIGURATION_FAILED,
    ):
        return EvalResult(
            passed=False,
            interface_ok=False,
            semantic_ok=None,
            category=FailureCategory.OTHER,
            errors=[f"non-protocol termination: {termination.value}"],
            details={**details, "failure_domain": termination.value},
        )
    if termination == Termination.PARSE_ERROR:
        return EvalResult(
            passed=False,
            interface_ok=False,
            semantic_ok=None,
            category=FailureCategory.RESPONSE_PARSING,
            errors=[f"termination: {termination.value}"],
            details=details,
        )
    if termination == Termination.TOOL_PROTOCOL_ERROR:
        return EvalResult(
            passed=False,
            interface_ok=False,
            semantic_ok=None,
            category=FailureCategory.HISTORY_REPRESENTATION,
            errors=[f"termination: {termination.value}"],
            details=details,
        )
    if termination == Termination.MODEL_ERROR:
        return EvalResult(
            passed=False,
            interface_ok=False,
            semantic_ok=None,
            category=FailureCategory.OTHER,
            errors=[f"termination: {termination.value}"],
            details=details,
        )

    # Prefix legality
    tv = validate_trace(events)
    if not tv.legal:
        return EvalResult(
            passed=False,
            interface_ok=False,
            semantic_ok=None,
            category=FailureCategory.TRACE_ILLEGAL,
            errors=tv.errors,
            details={**details, "final_state": tv.final_state.value},
        )

    calls = _collect_calls(events)
    interface_errors: list[str] = []
    interface_category = FailureCategory.PROTOCOL_EXPECTATION

    # Schema/offered-set checks first (report structural issues even on prefixes)
    for call in calls:
        if call.name not in tool_map:
            return EvalResult(
                passed=False,
                interface_ok=False,
                semantic_ok=None,
                category=FailureCategory.TOOL_REPRESENTATION,
                errors=[f"tool {call.name!r} not in offered set"],
                details={**details, "call": call.model_dump()},
            )
        if isinstance(call.arguments, dict) and call.arguments.get("__parse_error__"):
            return EvalResult(
                passed=False,
                interface_ok=False,
                semantic_ok=None,
                category=FailureCategory.RESPONSE_PARSING,
                errors=[str(call.arguments.get("__parse_error__"))],
                details={**details, "call": call.model_dump()},
            )
        ok, msg = validate_tool_arguments(call.arguments, tool_map[call.name].parameters)
        if not ok:
            return EvalResult(
                passed=False,
                interface_ok=False,
                semantic_ok=None,
                category=FailureCategory.SCHEMA_REPRESENTATION,
                errors=[f"arguments for {call.name!r}: {msg}"],
                details={**details, "call": call.model_dump()},
            )

    require_completed = True
    if expectation is not None:
        require_completed = expectation.require_completed
    elif kind == "semantic":
        require_completed = False

    if require_completed:
        ctv = validate_completed_trace(events)
        if not ctv.legal:
            return EvalResult(
                passed=False,
                interface_ok=False,
                semantic_ok=None,
                category=FailureCategory.INCOMPLETE_TRACE,
                errors=ctv.errors,
                details={**details, "final_state": ctv.final_state.value},
            )

    if expectation is None:
        return EvalResult(
            passed=True,
            interface_ok=True,
            semantic_ok=None,
            category=FailureCategory.NONE,
            errors=[],
            details={**details, "call_count": len(calls)},
        )

    names = [c.name for c in calls]
    semantic_errors: list[str] = []
    semantic_ok: bool | None = None
    semantic_category = FailureCategory.SEMANTIC_TOOL_SELECTION

    def check_tool_names_and_args(*, as_interface: bool) -> None:
        nonlocal interface_errors, interface_category, semantic_errors, semantic_ok, semantic_category
        target = interface_errors if as_interface else semantic_errors

        if expectation.forbid_tool_calls and calls:
            target.append("tool calls forbidden but present")
            if not as_interface:
                semantic_ok = False
                semantic_category = FailureCategory.SEMANTIC_DECISION

        if expectation.require_any_tool_call and not calls:
            target.append("expected at least one tool call")
            if as_interface:
                interface_category = FailureCategory.RESPONSE_PARSING
            else:
                semantic_ok = False

        # Exact sequence / cardinality (strong protocol).
        # Parallel batches are order-free within a turn (ABI multiset); diagnosis
        # already scores set equality. When require_parallel_batch is set, match
        # the expected names as a multiset so cert does not reject reordered
        # correct batches.
        if expectation.exact_tool_names is not None:
            if expectation.require_parallel_batch:
                from collections import Counter

                if Counter(names) != Counter(expectation.exact_tool_names):
                    target.append(
                        f"exact tool multiset {names} != expected "
                        f"{expectation.exact_tool_names}"
                    )
                    if as_interface:
                        interface_category = FailureCategory.PROTOCOL_EXPECTATION
                    else:
                        semantic_ok = False
            elif names != expectation.exact_tool_names:
                target.append(
                    f"exact tool sequence {names} != expected {expectation.exact_tool_names}"
                )
                if as_interface:
                    interface_category = FailureCategory.PROTOCOL_EXPECTATION
                else:
                    semantic_ok = False

        if expectation.exact_tool_call_count is not None:
            if len(calls) != expectation.exact_tool_call_count:
                target.append(
                    f"tool call count {len(calls)} != exact {expectation.exact_tool_call_count}"
                )
                if as_interface:
                    interface_category = FailureCategory.PROTOCOL_EXPECTATION
                else:
                    semantic_ok = False

        if expectation.allowed_tool_names is not None:
            allowed = set(expectation.allowed_tool_names)
            extras = [n for n in names if n not in allowed]
            if extras:
                target.append(f"disallowed tool calls: {extras}")
                if as_interface:
                    interface_category = FailureCategory.PROTOCOL_EXPECTATION
                else:
                    semantic_ok = False

        if expectation.require_parallel_batch:
            if not _first_tool_turn_is_batch(events):
                target.append("expected first tool turn to be a parallel ToolCallBatch")
                if as_interface:
                    interface_category = FailureCategory.PARALLEL_CALL_PROTOCOL

        if expectation.require_call_ids:
            missing_ids = [c.name for c in calls if c.id is None]
            if missing_ids:
                target.append(f"tool calls missing required call IDs: {missing_ids}")
                if as_interface:
                    interface_category = FailureCategory.CALL_ID
                else:
                    semantic_ok = False

        if expectation.required_tool_names is not None and expectation.exact_tool_names is None:
            if expectation.ordered_tool_names:
                if names != expectation.required_tool_names:
                    target.append(
                        f"tool sequence {names} != expected {expectation.required_tool_names}"
                    )
                    if not as_interface:
                        semantic_ok = False
            else:
                missing = [n for n in expectation.required_tool_names if n not in names]
                if missing:
                    target.append(f"missing required tools: {missing}")
                    if not as_interface:
                        semantic_ok = False
                        semantic_category = FailureCategory.SEMANTIC_TOOL_SELECTION
                    elif not calls:
                        interface_category = FailureCategory.RESPONSE_PARSING

        if expectation.argument_subsets:
            for tool_name, subset in expectation.argument_subsets:
                matching = [c for c in calls if c.name == tool_name]
                if not matching:
                    target.append(f"no call to {tool_name!r} for argument check")
                    if not as_interface:
                        semantic_ok = False
                    continue
                if not any(_subset_match(c.arguments, subset) for c in matching):
                    target.append(
                        f"no call to {tool_name!r} matched argument subset {subset}"
                    )
                    if not as_interface:
                        semantic_ok = False
                        semantic_category = FailureCategory.SEMANTIC_ARGUMENT_VALUE

        if expectation.result_dependencies:
            dep_errors = _check_result_dependencies(events, expectation.result_dependencies)
            if dep_errors:
                target.extend(dep_errors)
                if as_interface:
                    interface_category = FailureCategory.TOOL_RESULT_ENCODING
                else:
                    semantic_ok = False
                    semantic_category = FailureCategory.SEMANTIC_ARGUMENT_VALUE

        final_text_required = (
            expectation.require_final_text
            or expectation.expected_final_text is not None
            or expectation.expected_final_sentinel is not None
            or expectation.diagnostic_expected_final_text is not None
        )
        if final_text_required:
            if not events or not isinstance(events[-1], AssistantText):
                if as_interface:
                    interface_errors.append("expected final AssistantText")
                    interface_category = FailureCategory.HISTORY_REPRESENTATION
                else:
                    semantic_errors.append("expected final AssistantText")
                    semantic_ok = False
            elif expectation.expected_final_text is not None:
                last = events[-1]
                assert isinstance(last, AssistantText)
                if last.content.strip() != expectation.expected_final_text.strip():
                    msg = (
                        f"final text {last.content!r} != expected "
                        f"{expectation.expected_final_text!r}"
                    )
                    if as_interface:
                        interface_errors.append(msg)
                    else:
                        semantic_errors.append(msg)
                        semantic_ok = False

        if expectation.expected_final_sentinel is not None:
            source_tool = expectation.final_sentinel_source_tool
            if not source_tool:
                raise ValueError(
                    "expected_final_sentinel requires final_sentinel_source_tool"
                )
            witness = assess_event_g3_termination(
                events,
                expected_sentinel=expectation.expected_final_sentinel,
                source_tool=source_tool,
                parse_errors=tuple(parse_errors or ()),
                normal_termination=termination in (None, Termination.COMPLETED),
            )
            details["final_termination_witness"] = witness.as_dict()
            if not witness.protocol_termination_verified:
                msg = (
                    "final protocol termination witness failed: "
                    + ", ".join(witness.failure_codes)
                )
                if as_interface:
                    interface_errors.append(msg)
                    interface_category = FailureCategory.HISTORY_REPRESENTATION
                else:
                    semantic_errors.append(msg)
                    semantic_ok = False

        if expectation.diagnostic_expected_final_text is not None:
            actual = (
                events[-1].content
                if events and isinstance(events[-1], AssistantText)
                else None
            )
            details["exact_response_format_followed"] = (
                actual is not None
                and actual.strip() == expectation.diagnostic_expected_final_text.strip()
            )

        if expectation.min_tool_cycles > 0:
            cycles = _count_tool_cycles(events)
            if cycles < expectation.min_tool_cycles:
                interface_errors.append(
                    f"tool cycles {cycles} < min {expectation.min_tool_cycles}"
                )
                interface_category = FailureCategory.HISTORY_REPRESENTATION

    if kind == "protocol":
        check_tool_names_and_args(as_interface=True)
        if expectation.expected_primary_tool is not None:
            semantic_ok = bool(names) and names[0] == expectation.expected_primary_tool
            if not semantic_ok:
                semantic_errors.append(
                    f"primary tool {names[0] if names else None!r} "
                    f"!= expected {expectation.expected_primary_tool!r}"
                )
    else:
        check_tool_names_and_args(as_interface=False)
        if expectation.expected_primary_tool is not None:
            if semantic_ok is None:
                semantic_ok = True
            if not (bool(names) and names[0] == expectation.expected_primary_tool):
                semantic_ok = False
                semantic_errors.append(
                    f"primary tool {names[0] if names else None!r} "
                    f"!= expected {expectation.expected_primary_tool!r}"
                )
                semantic_category = FailureCategory.SEMANTIC_TOOL_SELECTION
        elif semantic_errors:
            semantic_ok = False
        elif (
            expectation.required_tool_names is not None
            or expectation.exact_tool_names is not None
            or expectation.argument_subsets is not None
            or expectation.forbid_tool_calls
            or expectation.require_any_tool_call
            or expectation.expected_primary_tool is not None
            or expectation.result_dependencies is not None
        ):
            if semantic_ok is None and not semantic_errors:
                semantic_ok = True

        if expectation.min_tool_cycles > 0:
            cycles = _count_tool_cycles(events)
            if cycles < expectation.min_tool_cycles:
                interface_errors.append(
                    f"tool cycles {cycles} < min {expectation.min_tool_cycles}"
                )
                interface_category = FailureCategory.HISTORY_REPRESENTATION

    interface_ok = not interface_errors
    if semantic_errors and semantic_ok is None:
        semantic_ok = False

    details = {**details, "calls": [c.model_dump() for c in calls]}

    if not interface_ok:
        return EvalResult(
            passed=False,
            interface_ok=False,
            semantic_ok=semantic_ok,
            category=interface_category,
            errors=interface_errors + semantic_errors,
            details=details,
        )

    if semantic_ok is False:
        return EvalResult(
            passed=False,
            interface_ok=True,
            semantic_ok=False,
            category=semantic_category,
            errors=semantic_errors,
            details=details,
        )

    return EvalResult(
        passed=True,
        interface_ok=True,
        semantic_ok=semantic_ok,
        category=FailureCategory.NONE,
        errors=[],
        details=details,
    )


def _json_path_get(value: Any, path: tuple[str, ...]) -> tuple[bool, Any]:
    cur = value
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return False, None
        cur = cur[key]
    return True, cur


def _check_result_dependencies(
    events: list[Event], dependencies: list[ResultDependency]
) -> list[str]:
    errors: list[str] = []
    for dep in dependencies:
        source_value = None
        source_index = None
        for i, event in enumerate(events):
            if isinstance(event, ToolResult) and event.name == dep.source_tool:
                ok, value = _json_path_get(event.content, dep.source_path)
                if ok:
                    source_value = value
                    source_index = i
                    break
            if isinstance(event, ToolError) and event.name == dep.source_tool:
                # ToolError exposes its error string as a synthetic {error: ...} payload.
                ok, value = _json_path_get({"error": event.error}, dep.source_path)
                if ok:
                    source_value = value
                    source_index = i
                    break
        if source_index is None:
            errors.append(
                f"missing result dependency source {dep.source_tool!r} path {dep.source_path}"
            )
            continue

        matched = False
        for event in events[source_index + 1 :]:
            candidate_calls: list[ToolCall] = []
            if isinstance(event, AssistantToolCall):
                candidate_calls = [event.call]
            elif isinstance(event, ToolCallBatch):
                candidate_calls = list(event.calls)
            for call in candidate_calls:
                if call.name != dep.target_tool:
                    continue
                ok, target_value = _json_path_get(call.arguments, dep.target_argument_path)
                if ok and target_value == source_value:
                    matched = True
                    break
            if matched:
                break
        if not matched:
            errors.append(
                "tool-result dependency not preserved: "
                f"{dep.source_tool}{dep.source_path} -> "
                f"{dep.target_tool}{dep.target_argument_path}"
            )
    return errors


def _first_tool_turn_is_batch(events: list[Event]) -> bool:
    for e in events:
        if isinstance(e, ToolCallBatch):
            return True
        if isinstance(e, AssistantToolCall):
            return False
    return False


def _collect_calls(events: list[Event]) -> list[ToolCall]:
    out: list[ToolCall] = []
    for e in events:
        if isinstance(e, AssistantToolCall):
            out.append(e.call)
        elif isinstance(e, ToolCallBatch):
            out.extend(e.calls)
    return out


def _subset_match(arguments: Any, subset: dict[str, Any]) -> bool:
    if not isinstance(arguments, dict):
        return False
    for k, v in subset.items():
        if arguments.get(k) != v:
            return False
    return True


def _count_tool_cycles(events: list[Event]) -> int:
    cycles = 0
    i = 0
    while i < len(events):
        e = events[i]
        if isinstance(e, (AssistantToolCall, ToolCallBatch)):
            j = i + 1
            saw_result = False
            while j < len(events) and events[j].type in ("tool_result", "tool_error"):
                saw_result = True
                j += 1
            if saw_result:
                cycles += 1
            i = j
        else:
            i += 1
    return cycles
