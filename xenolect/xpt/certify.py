"""Independent certification of a synthesized driver.

Independence is structural, not just procedural:

  * it runs the **production** `DriverRuntime`, not the XPT session, so what is
    certified is the driver as the shipped runtime will execute it;
  * it scores with the **existing deterministic evaluator**
    (`xenolect.eval.evaluator.evaluate_trace`) and the existing ABI trace
    checker, not with XPT's own syndrome logic;
  * it uses a freshly minted instance — new challenge values, new sentinels, new
    error code, new ack, and a different surface form of the script than the one
    diagnosis used.

If any mandatory obligation is not VERIFIED on this run alone, there is no
certificate. Diagnosis evidence never contributes to coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xenolect.abi.events import (
    AssistantText,
    AssistantToolCall,
    Event,
    ToolCall,
    ToolCallBatch,
    ToolError,
    ToolResult,
    UserMessage,
)
from xenolect.abi.trace import validate_completed_trace
from xenolect.driver.ir import Driver
from xenolect.driver.runtime import DriverRuntime
from xenolect.driver.termination import Termination
from xenolect.eval.evaluator import ProbeExpectation, ResultDependency, evaluate_trace
from xenolect.eval.schema import validate_tool_arguments
from xenolect.eval.termination import assess_event_g3_termination
from xenolect.xpt.gauntlet import (
    BATCH_TOOLS,
    RECOVERY_TOOLS,
    GauntletInstance,
    gauntlet_tools,
    render_user_turn,
)
from xenolect.xpt.obligations import (
    CoverageCertificate,
    ObligationEvidence,
    ObligationStatus,
)
from xenolect.xpt.syndrome import build_syndrome, sha


@dataclass
class CertificationRun:
    certificate: CoverageCertificate
    generations: int
    events: list[Event] = field(default_factory=list)
    wire: list[dict[str, Any]] = field(default_factory=list)
    evaluator_result: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    termination: Termination = Termination.COMPLETED

    @property
    def passed(self) -> bool:
        return self.certificate.complete and self.evaluator_result.get("interface_ok") is True


def certification_expectation(inst: GauntletInstance) -> ProbeExpectation:
    """Expectation expressed purely in the existing evaluator's vocabulary."""
    batch = inst.expected_batch_arguments()
    recovery = inst.expected_recovery_arguments()
    return ProbeExpectation(
        exact_tool_names=list(BATCH_TOOLS) + list(RECOVERY_TOOLS),
        exact_tool_call_count=len(BATCH_TOOLS) + len(RECOVERY_TOOLS),
        allowed_tool_names=list(BATCH_TOOLS) + list(RECOVERY_TOOLS),
        argument_subsets=[(name, batch[name]) for name in BATCH_TOOLS]
        + [(name, recovery[name]) for name in RECOVERY_TOOLS],
        require_parallel_batch=True,
        min_tool_cycles=2,
        require_final_text=True,
        expected_final_sentinel=inst.ack_value,
        final_sentinel_source_tool="report",
        diagnostic_expected_final_text=inst.ack_value,
        result_dependencies=[
            ResultDependency(
                source_tool="record_alpha",
                source_path=("token",),
                target_tool="commit",
                target_argument_path=("alpha",),
            ),
            ResultDependency(
                source_tool="record_beta",
                source_path=("token",),
                target_tool="commit",
                target_argument_path=("beta",),
            ),
        ],
    )


def certification_executors(inst: GauntletInstance) -> dict[str, Any]:
    def _gamma(**_: Any) -> Any:
        raise RuntimeError(inst.gamma_error_text())

    return {
        "record_alpha": lambda **_: inst.result_for("record_alpha"),
        "record_beta": lambda **_: inst.result_for("record_beta"),
        "record_gamma": _gamma,
        "commit": lambda **_: inst.recovery_results()["commit"],
        "report": lambda **_: inst.recovery_results()["report"],
    }


# --------------------------------------------------------------------------
# trace inspection helpers
# --------------------------------------------------------------------------


def _turns(events: list[Event]) -> list[list[ToolCall]]:
    """Ordered list of assistant tool turns; each entry is that turn's calls."""
    out: list[list[ToolCall]] = []
    for e in events:
        if isinstance(e, ToolCallBatch):
            out.append(list(e.calls))
        elif isinstance(e, AssistantToolCall):
            out.append([e.call])
    return out


def _all_calls(events: list[Event]) -> list[ToolCall]:
    return [c for turn in _turns(events) for c in turn]


def _final_text(events: list[Event]) -> str | None:
    if events and isinstance(events[-1], AssistantText):
        return events[-1].content
    return None


def _tool_cycles(events: list[Event]) -> int:
    cycles = 0
    i = 0
    while i < len(events):
        if isinstance(events[i], (AssistantToolCall, ToolCallBatch)):
            j = i + 1
            saw = False
            while j < len(events) and isinstance(events[j], (ToolResult, ToolError)):
                saw = True
                j += 1
            if saw:
                cycles += 1
            i = j
        else:
            i += 1
    return cycles


# --------------------------------------------------------------------------
# certification
# --------------------------------------------------------------------------


def certify(
    driver: Driver,
    client: Any,
    inst: GauntletInstance,
    *,
    max_cycles: int = 6,
) -> CertificationRun:
    tools = gauntlet_tools()
    rt = DriverRuntime(driver=driver, client=client)
    user = UserMessage(content=render_user_turn(inst), tools=tools)
    run = rt.run_probe(
        user, tool_executor=certification_executors(inst), max_cycles=max_cycles
    )
    events = run.events
    wire = list(run.wire_interactions)
    cert = CoverageCertificate()

    result = evaluate_trace(
        events,
        tools=tools,
        expectation=certification_expectation(inst),
        kind="protocol",
        termination=run.termination,
        parse_errors=run.parse_errors or None,
    )

    gen_of = {i + 1: w for i, w in enumerate(wire)}

    def hashes(gen: int | None) -> tuple[str | None, str | None]:
        w = gen_of.get(gen) if gen else None
        if not w:
            return None, None
        return sha(w.get("request")), sha(w.get("response"))

    def record(
        oid: str, ok: bool, gen: int | None, observation: str, **detail: Any
    ) -> None:
        rq, rs = hashes(gen)
        cert.record(
            ObligationEvidence(
                obligation_id=oid,
                status=ObligationStatus.VERIFIED if ok else ObligationStatus.FAILED,
                generation_id=gen,
                request_hash=rq,
                response_hash=rs,
                observation=observation,
                detail=detail,
            )
        )

    turns = _turns(events)
    calls = _all_calls(events)
    names = [c.name for c in calls]
    tool_map = {t.name: t for t in tools}
    batch_args = inst.expected_batch_arguments()
    recovery_args = inst.expected_recovery_arguments()

    # ---- OB01 tool call emission -------------------------------------
    record("OB01", bool(calls), 1, f"{len(calls)} normalized tool call(s) in the trace")

    # ---- OB02 tool name resolution -----------------------------------
    unknown = sorted({n for n in names if n not in tool_map})
    record("OB02", not unknown and bool(names), 1, f"unknown names: {unknown or 'none'}")

    # ---- OB03 argument schema validity --------------------------------
    invalid = []
    for c in calls:
        schema = tool_map.get(c.name)
        if schema is None:
            continue
        ok, msg = validate_tool_arguments(c.arguments, schema.parameters)
        if not ok:
            invalid.append(f"{c.name}: {msg}")
    record("OB03", not invalid and bool(calls), 1, f"schema violations: {invalid or 'none'}")

    # ---- OB04 argument value integrity --------------------------------
    exact = {
        c.name: c.arguments == batch_args[c.name] for c in calls if c.name in batch_args
    }
    record(
        "OB04",
        bool(exact) and all(exact.values()) and set(exact) == set(BATCH_TOOLS),
        1,
        f"exact challenge-argument match: {exact}",
    )

    # ---- OB05 nested / $ref schema support ----------------------------
    alpha = next((c for c in calls if c.name == "record_alpha"), None)
    alpha_ok = alpha is not None and isinstance(alpha.arguments, dict) and isinstance(
        alpha.arguments.get("entry"), dict
    )
    record(
        "OB05",
        bool(alpha_ok) and validate_tool_arguments(
            alpha.arguments if alpha else None, tool_map["record_alpha"].parameters
        )[0],
        1,
        "record_alpha ($ref + nested object) invoked with a schema-valid nested argument"
        if alpha_ok
        else "record_alpha absent or not nested",
    )

    # ---- OB06 / OB07 parallel batch + cardinality ----------------------
    first_is_batch = _first_tool_turn_is_batch(events)
    record(
        "OB06",
        first_is_batch,
        1,
        f"first tool turn is a ToolCallBatch: {first_is_batch}",
    )
    card_ok = (
        len(turns) == 2
        and len(turns[0]) == len(BATCH_TOOLS)
        and len(turns[1]) == len(RECOVERY_TOOLS)
    )
    record(
        "OB07",
        card_ok,
        1,
        f"turn cardinalities {[len(t) for t in turns]} vs expected "
        f"{[len(BATCH_TOOLS), len(RECOVERY_TOOLS)]}",
    )

    # ---- OB08 / OB09 call id discipline --------------------------------
    ids = [c.id for c in calls]
    have = [i for i in ids if i is not None]
    # The ABI requires id association only when the assistant emitted ids
    # Both "all ids" and "no ids" are conformant; a mixture is not.
    discipline = len(have) in (0, len(ids)) and bool(ids)
    record(
        "OB08",
        discipline,
        1,
        f"{len(have)}/{len(ids)} calls carry ids (all-or-none is the ABI requirement)",
    )
    record("OB09", len(set(have)) == len(have), 1, f"ids unique: {sorted(have)}")

    # ---- OB10 / OB11 result consumption + association -------------------
    commit = next((c for c in calls if c.name == "commit"), None)
    consumed = commit is not None and isinstance(commit.arguments, dict) and (
        commit.arguments.get("alpha") == inst.alpha_token
        or commit.arguments.get("beta") == inst.beta_token
    )
    record(
        "OB10",
        bool(consumed),
        2,
        "commit carries at least one post-prompt sentinel minted after the user turn",
    )
    associated = commit is not None and commit.arguments == recovery_args["commit"]
    record(
        "OB11",
        bool(associated),
        2,
        "each parallel call's own sentinel reached its own argument slot "
        f"(observed {commit.arguments if commit else None})",
    )

    # ---- OB12 history preservation --------------------------------------
    cycles = _tool_cycles(events)
    record("OB12", cycles >= 2, 3, f"{cycles} completed tool-result cycles")

    # ---- OB13 / OB14 ToolError consumption + recovery --------------------
    report = next((c for c in calls if c.name == "report"), None)
    err_ok = report is not None and report.arguments == recovery_args["report"]
    record(
        "OB13",
        bool(err_ok),
        2,
        "report carries the fresh ToolError code "
        f"(observed {report.arguments if report else None})",
    )
    error_index = next(
        (i for i, e in enumerate(events) if isinstance(e, ToolError)), None
    )
    recovery_after_error = error_index is not None and any(
        isinstance(e, (ToolCallBatch, AssistantToolCall))
        and any(c.name in RECOVERY_TOOLS for c in (_calls_of(e)))
        for e in events[error_index + 1 :]
    )
    record(
        "OB14",
        bool(recovery_after_error) and len(turns) == 2 and len(turns[1]) == 2,
        2,
        "a valid parallel recovery turn follows the ToolError",
    )

    ambiguous = []
    for i, w in enumerate(wire):
        raw = w.get("response")
        if not isinstance(raw, dict):
            continue
        syn = build_syndrome(raw)
        if syn.consensus.value == "ambiguous":
            ambiguous.append(i + 1)

    # ---- OB15 / OB16 no-call + termination --------------------------------
    final = _final_text(events)
    record(
        "OB15",
        len(turns) == 2,
        3,
        f"no spurious tool turn after the recovery results ({len(turns)} tool turns total)",
    )
    termination_witness = assess_event_g3_termination(
        events,
        expected_sentinel=inst.ack_value,
        source_tool="report",
        parser_ambiguous=bool(ambiguous),
        parse_errors=tuple(run.parse_errors),
        normal_termination=run.termination == Termination.COMPLETED,
    )
    record(
        "OB16",
        termination_witness.protocol_termination_verified,
        3,
        (
            "final assistant turn contains one exclusive post-result ack sentinel "
            f"with normal no-call termination (observed {final!r})"
        ),
        termination_witness=termination_witness.as_dict(),
    )

    # ---- OB17 unambiguous parse -------------------------------------------
    record(
        "OB17",
        not ambiguous,
        1,
        f"multi-parser consensus on every response (ambiguous at: {ambiguous or 'none'})",
    )

    # ---- OB18 legal completed ABI trace -------------------------------------
    tv = validate_completed_trace(events)
    record("OB18", tv.legal, len(wire) or None, f"completed-trace check: {tv.errors or 'legal'}")

    return CertificationRun(
        certificate=cert,
        generations=len(wire),
        events=events,
        wire=wire,
        evaluator_result=result.as_dict(),
        errors=list(run.errors),
        termination=run.termination,
    )


def _calls_of(event: Event) -> list[ToolCall]:
    if isinstance(event, ToolCallBatch):
        return list(event.calls)
    if isinstance(event, AssistantToolCall):
        return [event.call]
    return []


def _first_tool_turn_is_batch(events: list[Event]) -> bool:
    """Same rule the production evaluator applies (evaluator._first_tool_turn_is_batch)."""
    for e in events:
        if isinstance(e, ToolCallBatch):
            return True
        if isinstance(e, AssistantToolCall):
            return False
    return False
