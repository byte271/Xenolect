"""XPT online runtime: diagnose an unknown endpoint, synthesize a driver, certify it.

Phase order and what each phase is allowed to cost:

    1  request-configuration diagnosis   walks the precompiled decision DAG
    2  parser resolution                 FREE (local, multi-parser, no generation)
    3  stateful trajectory G2/G3         continues the successful G1 branch
    4  tool-result-encoding resolution   counterfactual fork at the G1 state
    5  synthesis                         FREE
    6  independent certification         production runtime + production evaluator

The algorithm never sees a model name, a provider name, an endpoint type, a
candidate id or a reference answer. Its only input is observable values returned
by `chat_completions`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from xenolect.xpt.certify import certify
from xenolect.xpt.frontier import (
    CERTIFICATION_GENERATION_UPPER_BOUND,
    FrontierEvidence,
    WireFingerprint,
    fingerprint_request,
    g1_fingerprint,
    select_next_config,
)
from xenolect.xpt.gauntlet import (
    RECOVERY_TOOLS,
    gauntlet_tools,
    mint_instance,
)
from xenolect.xpt.obligations import CoverageCertificate
from xenolect.xpt.planner import (
    DiagnosticProgram,
    ProbeTemplate,
    RequestConfig,
    all_request_configs,
    annotate_arguments,
    observation_class,
    probe_payload,
    probe_succeeded,
)
from xenolect.xpt.session import (
    Branch,
    Budget,
    BudgetExhausted,
    ConfigurationFailed,
    DeadlineExceeded,
    Generation,
    InfrastructureFailed,
    Ledger,
    XptSession,
)
from xenolect.xpt.syndrome import ParseConsensus, Syndrome
from xenolect.driver.ir import Driver, ParserKind, SchemaTransform, ToolEncoding, ToolResultEncoding
from xenolect.driver.termination import Termination

CERTIFIED = "CERTIFIED"
UNSUPPORTED = "UNSUPPORTED"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
ENDPOINT_TOO_SLOW = "ENDPOINT_TOO_SLOW_FOR_CERTIFICATION"
INFRASTRUCTURE_FAILED = "INFRASTRUCTURE_FAILED"
CONFIGURATION_FAILED = "CONFIGURATION_FAILED"


@dataclass
class XptResult:
    status: str
    driver: Driver | None = None
    certificate: CoverageCertificate = field(default_factory=CoverageCertificate)
    ledger: Ledger = field(default_factory=Ledger)
    reason: str = ""
    diagnosis_generations: int = 0
    certification_generations: int = 0
    failed_obligations: list[str] = field(default_factory=list)
    wall_clock_s: float = 0.0
    left_compiled_dag: bool = False
    equivalent_parsers: list[str] = field(default_factory=list)
    #: (prompt_chars, completion_chars) per generation, diagnosis then certification.
    #: Retained for diagnostics because short and large generations have very
    #: different latency profiles.
    io_sizes: list[tuple[int, int]] = field(default_factory=list)

    @property
    def total_generations(self) -> int:
        return self.diagnosis_generations + self.certification_generations

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "driver": self.driver.canonical_dict() if self.driver else None,
            "reason": self.reason,
            "diagnosis_generations": self.diagnosis_generations,
            "certification_generations": self.certification_generations,
            "total_generations": self.total_generations,
            "failed_obligations": list(self.failed_obligations),
            "mandatory_coverage": self.certificate.mandatory_coverage,
            "wall_clock_s": self.wall_clock_s,
            "left_compiled_dag": self.left_compiled_dag,
            "equivalent_parsers": list(self.equivalent_parsers),
            "io_sizes": [list(x) for x in self.io_sizes],
        }


@dataclass
class _Trajectory:
    """A G1 branch that succeeded, together with what it proved."""

    branch: Branch
    config: RequestConfig
    syndrome: Syndrome
    frozen_prefix: str


def _driver_from(
    config: RequestConfig, parser: ParserKind, result_encoding: ToolResultEncoding
) -> Driver:
    return Driver(
        tool_encoding=ToolEncoding(config.tool_encoding),
        parser=parser,
        schema_transforms=[SchemaTransform(t) for t in config.transforms],
        tool_result_encoding=result_encoding,
    )


class XptCompiler:
    def __init__(
        self,
        client: Any,
        program: DiagnosticProgram,
        *,
        budget: Budget | None = None,
        seed: int = 1,
        clock: Callable[[], float] = time.perf_counter,
        started_at: float | None = None,
    ) -> None:
        self.client = client
        self.program = program
        selected_budget = budget or Budget()
        if selected_budget.certification_reserve < CERTIFICATION_GENERATION_UPPER_BOUND:
            raise ValueError(
                "certification_reserve is smaller than the frozen certification "
                f"upper bound ({selected_budget.certification_reserve} < "
                f"{CERTIFICATION_GENERATION_UPPER_BOUND})"
            )
        self.session = XptSession(
            client, budget=selected_budget, clock=clock, started_at=started_at
        )
        self.seed = seed
        self.diag_inst = mint_instance(seed=seed, salt="diagnose", surface_form="A")
        self.left_dag = False

    # ------------------------------------------------------------------
    # phase 1: request configuration
    # ------------------------------------------------------------------

    def _run_template(self, probe: ProbeTemplate) -> tuple[Syndrome, Branch, bool]:
        tools, content, expected = probe_payload(probe, self.diag_inst)
        branch = self.session.new_branch(probe.config.driver())
        branch.add_user(content, tools)
        syn, _ = branch.generate(
            purpose="explore",
            label=probe.id,
            reason=self._reason,
            offered_tool_names={t.name for t in tools},
        )
        annotate_arguments(syn, tools, expected)
        ok = probe_succeeded(syn, expected, batch=probe.kind == "gauntlet_turn1")
        return syn, branch, ok

    # ------------------------------------------------------------------
    # candidate configurations
    # ------------------------------------------------------------------

    def _frontier_evidence(self, tried: set[str]) -> FrontierEvidence:
        """Build replanner evidence from the live session + SAFE eliminations.

        ``tried`` is the set of *evaluated* RequestConfig keys (SAFE), not
        "every wire sibling we might never need to pay for again."
        """
        ev = FrontierEvidence(
            used_generations=self.session.ledger.generation_count,
            max_generations=self.session.budget.max_generations,
            certification_reserve=self.session.budget.certification_reserve,
            eliminated_hypothesis_ids=set(tried),
            tried_action_ids=set(tried),
        )
        # Paid wires (heuristic novelty + share accounting) from the ledger.
        for gen in self.session.ledger.generations:
            if gen.purpose != "explore":
                continue
            fp = fingerprint_request(gen.request)
            if fp.full_hash != gen.request_hash:
                fp = WireFingerprint(full_hash=gen.request_hash, features=fp.features)
            ev.record_observation(fp)
        for h in getattr(self, "_traj_failed_wires", set()):
            ev.trajectory_failed_wires.add(h)
        for h in getattr(self, "_g1_failed_wires", set()):
            ev.g1_failed_wires.add(h)
        for h, fp in getattr(self, "_wire_fingerprints", {}).items():
            ev.observed_fingerprints[h] = fp
        return ev

    def _wire_class_keys(self, cfg: RequestConfig) -> list[str]:
        """RequestConfig keys that emit the exact same G1 request as ``cfg``.

        SAFE justification (frozen grammar): RequestConfig only sets
        tool_encoding + schema_transforms; G1 wire is a pure function of those
        fields on the gauntlet tool set. Equal wire hash ⇒ equal G1 observation.
        Parser / tool_result_encoding are not part of RequestConfig (resolved
        later by free parse + counterfactual fork). Therefore wire-identical
        RequestConfigs form one diagnosis experiment class.
        """
        target = g1_fingerprint(cfg, seed=self.seed).full_hash
        return [
            c.key
            for c in all_request_configs()
            if g1_fingerprint(c, seed=self.seed).full_hash == target
        ]

    def _mark_g1_wire(self, cfg: RequestConfig, *, ok: bool) -> str:
        """Fingerprint the G1 request; on failure SAFE-eliminate the wire class."""
        fp = g1_fingerprint(cfg, seed=self.seed)
        if not hasattr(self, "_traj_failed_wires"):
            self._traj_failed_wires = set()
            self._g1_failed_wires = set()
            self._wire_fingerprints = {}
        self._wire_fingerprints[fp.full_hash] = fp
        if not ok:
            self._g1_failed_wires.add(fp.full_hash)
        return fp.full_hash

    def _mark_trajectory_failed(self, cfg: RequestConfig) -> None:
        if not hasattr(self, "_traj_failed_wires"):
            self._traj_failed_wires = set()
            self._g1_failed_wires = set()
            self._wire_fingerprints = {}
        fp = g1_fingerprint(cfg, seed=self.seed)
        self._wire_fingerprints[fp.full_hash] = fp
        self._traj_failed_wires.add(fp.full_hash)

    def _safe_eliminate_wire_class(self, cfg: RequestConfig, *, reason: str) -> None:
        """SAFE: eliminate every RequestConfig with the same exact G1 wire as ``cfg``."""
        if not hasattr(self, "_safe_eliminated"):
            self._safe_eliminated = set()
        keys = self._wire_class_keys(cfg)
        for key in keys:
            self._safe_eliminated.add(key)
        self.session.ledger.note(
            f"SAFE eliminate G1 wire class of {cfg.key} ({len(keys)} configs): {reason}"
        )

    def _candidate_configs(self):
        """Yield request configurations to try.

        While on the offline DAG: observation-directed walk as compiled.
        On unexpected observation, leaf confirmation failure, or after the
        caller exhausts a yielded trajectory: **budgeted open-world frontier
        replan** — never static complexity order of expensive experiments.
        """
        node = self.program.nodes[self.program.root]
        tried: set[str] = set()
        self._traj_failed_wires = set()
        self._g1_failed_wires = set()
        self._wire_fingerprints = {}
        self._safe_eliminated: set[str] = set()
        # True once we have left the DAG early (a success, or an observation the
        # DAG has no edge for). Reaching a leaf normally is NOT leaving early:
        # that is precisely when the leaf's inferred conclusion must be probed.
        exited_early = False

        while node.probe_id is not None:
            probe = self.program.probe_index[node.probe_id]
            self._reason = node.reason
            tried.add(probe.config.key)
            syn, branch, ok = self._run_template(probe)
            self._mark_g1_wire(probe.config, ok=ok)
            _, _, expected = probe_payload(probe, self.diag_inst)
            obs = observation_class(syn, expected)
            self.session.ledger.decide(
                phase="config",
                node=node.node_id,
                probe=probe.id,
                reason=node.reason,
                observation=obs,
                succeeded=ok,
                hypotheses_before=node.n_hypotheses,
            )
            if ok and probe.kind == "gauntlet_turn1":
                yield _Trajectory(branch, probe.config, syn, branch.freeze())
                exited_early = True
                break
            if not ok and probe.kind == "gauntlet_turn1":
                # SAFE: same G1 wire ⇒ same observation under the frozen grammar.
                for key in self._wire_class_keys(probe.config):
                    tried.add(key)
            child = node.children.get(obs)
            if child is None:
                self.left_dag = True
                exited_early = True
                self.session.ledger.note(
                    f"observation {obs!r} is outside the compiled hypothesis class; "
                    "falling back to budgeted open-world frontier replan"
                )
                break
            node = self.program.nodes[child]

        # A leaf conclusion the DAG inferred but never probed: confirm it now.
        if not exited_early and node.probe_id is None and node.conclusion and not node.unsupported:
            cfg = {c.key: c for c in all_request_configs()}[node.conclusion]
            if cfg.key not in tried:
                tried.add(cfg.key)
                self._reason = (
                    f"leaf conclusion {cfg.key} inferred from the hypothesis class; "
                    "probing to confirm"
                )
                probe = ProbeTemplate(f"G1@{cfg.key}", "gauntlet_turn1", cfg, True)
                syn, branch, ok = self._run_template(probe)
                self._mark_g1_wire(cfg, ok=ok)
                self.session.ledger.decide(
                    phase="config", probe=probe.id, reason=self._reason, succeeded=ok
                )
                if ok:
                    yield _Trajectory(branch, cfg, syn, branch.freeze())
                else:
                    self.left_dag = True
                    for key in self._wire_class_keys(cfg):
                        tried.add(key)

        # Open-world continuation: replan over remaining wire-distinct configs.
        # Complexity must not decide which expensive request is sent next.
        self.left_dag = self.left_dag or bool(tried)
        while True:
            # Resource exhaustion is not protocol evidence.  Propagate it to the
            # top-level compiler so it is reported as BUDGET_EXHAUSTED or
            # ENDPOINT_TOO_SLOW rather than UNSUPPORTED.
            self.session.check_can_explore()
            # Merge SAFE eliminations recorded by run() after a yielded trajectory
            # fails (trajectory / certification) — generator-local `tried` alone
            # cannot see those events.
            tried |= getattr(self, "_safe_eliminated", set())
            remaining = [c for c in all_request_configs() if c.key not in tried]
            if not remaining:
                return
            evidence = self._frontier_evidence(tried)
            picked = select_next_config(remaining, evidence, seed=self.seed)
            if picked is None:
                self.session.ledger.note(
                    "frontier replan: no budget-feasible wire-distinct continuation remains"
                )
                return
            cfg, selection = picked
            # Evaluate the representative only. Wire siblings stay live until a
            # SAFE post-evaluation rule eliminates the wire class (equal G1
            # request ⇒ equal observation under the frozen grammar).
            self._reason = selection.reason
            self.session.ledger.decide(
                phase="frontier-replan",
                probe=f"G1@{cfg.key}",
                reason=selection.reason,
                frontier_before=selection.frontier_size_before,
                frontier_after=selection.frontier_size_after_prune,
                novelty=selection.novelty,
                lower_bound=selection.lower_bound,
                wire_hash=selection.action.wire.full_hash,
                wire_members=list(selection.action.member_ids),
                safe_note=(
                    "wire sharing packages members for one expensive call; "
                    "hypothesis elimination is deferred until evaluation"
                ),
            )
            probe = ProbeTemplate(f"G1@{cfg.key}", "gauntlet_turn1", cfg, True)
            syn, branch, ok = self._run_template(probe)
            self._mark_g1_wire(cfg, ok=ok)
            self.session.ledger.decide(
                phase="config-continued",
                probe=probe.id,
                reason=self._reason,
                succeeded=ok,
            )
            if ok:
                # G1 success: only this config is "in flight". Siblings with the
                # same wire remain candidates only if this trajectory fails and
                # we would re-select — SAFE rule then eliminates the whole class
                # after a full trajectory/cert failure (continuation equivalence).
                tried.add(cfg.key)
                yield _Trajectory(branch, cfg, syn, branch.freeze())
            else:
                # SAFE: same G1 request would produce the same raw response;
                # eliminate the entire exact-wire RequestConfig class.
                for key in self._wire_class_keys(cfg):
                    tried.add(key)
                self.session.ledger.note(
                    f"SAFE eliminate wire class of {cfg.key} after G1 failure "
                    f"({len(self._wire_class_keys(cfg))} RequestConfigs share this G1 request)"
                )

    # ------------------------------------------------------------------
    # phase 2: parser, resolved across the WHOLE trajectory (free)
    # ------------------------------------------------------------------

    @staticmethod
    def _narrow_parsers(
        live: set[ParserKind], syn: Syndrome
    ) -> set[ParserKind]:
        """Intersect the surviving parser set with this turn's compatible set.

        Parsers form a capability lattice: `tagged_json` and `xml_json` each parse
        everything `native` parses (they delegate when `tool_calls` is present) and
        additionally handle their own text dialect. So agreement on ONE turn does
        not license committing to a parser — a later turn can emit a text frame that
        only the wider parsers survive. Carrying the compatible set forward and
        intersecting costs nothing and is strictly better informed than choosing
        from the first observation.
        """
        return live & set(syn.compatible_parsers)

    @staticmethod
    def _settle_parser(live: set[ParserKind]) -> ParserKind | None:
        """Least-capable survivor: minimal commitment among parsers that all work."""
        order = {ParserKind.NATIVE: 0, ParserKind.XML_JSON: 1, ParserKind.TAGGED_JSON: 2}
        return min(live, key=lambda p: order[p]) if live else None

    # ------------------------------------------------------------------
    # phase 3/4: stateful trajectory and the tool-result-encoding fork
    # ------------------------------------------------------------------

    def complete_trajectory(
        self, traj: _Trajectory, live: set[ParserKind]
    ) -> tuple[ToolResultEncoding, ParserKind] | None:
        """Counterfactual fork at the captured G1 state.

        Both candidate encodings extend a byte-identical prefix, so the expensive
        G1 generation is paid for once and only the divergent suffix is re-paid.
        The recovery turn is also where the parser set is narrowed.
        """
        inst = self.diag_inst
        expected = inst.expected_recovery_arguments()
        read_with = self._settle_parser(live)
        if read_with is None:
            return None
        for encoding in (ToolResultEncoding.TOOL_ROLE, ToolResultEncoding.USER_MESSAGE):
            driver = _driver_from(traj.config, read_with, encoding)
            fork = traj.branch.fork(
                driver,
                reason=f"counterfactual tool-result encoding {encoding.value}",
            )
            assert fork.freeze() == traj.frozen_prefix
            # Every parser in `live` agreed on this turn's canonical AST, so reading
            # the calls back through any of them is sound.
            for call in traj.syndrome.parser_outcomes[read_with].calls:
                if call.name == "record_gamma":
                    fork.add_tool_error(
                        call_id=call.id, name=call.name, error=inst.gamma_error_text()
                    )
                else:
                    fork.add_tool_result(
                        call_id=call.id, name=call.name, content=inst.result_for(call.name)
                    )
            self.session.check_can_explore()
            syn, _ = fork.generate(
                purpose="explore",
                label=f"G2@{encoding.value}",
                reason=(
                    "two tool-result encodings remain legal at this frozen state; "
                    "forking pays only for the divergent suffix"
                ),
                offered_tool_names={t.name for t in gauntlet_tools()},
            )
            # Fail-closed on multi-parser disagreement (same rule as G1).
            if syn.consensus == ParseConsensus.AMBIGUOUS:
                self.session.ledger.decide(
                    phase="result_encoding",
                    encoding=encoding.value,
                    observation="ambiguous_parse",
                    parsers_live=[],
                    succeeded=False,
                )
                continue
            narrowed = self._narrow_parsers(live, syn)
            settled = self._settle_parser(narrowed)
            ok = False
            outcome = None
            if settled is not None:
                outcome = syn.parser_outcomes[settled]
                names = {c.name for c in outcome.calls}
                values = {
                    c.name: c.arguments == expected.get(c.name) for c in outcome.calls
                }
                ok = (
                    outcome.ok
                    and names == set(RECOVERY_TOOLS)
                    and all(values.get(n) for n in RECOVERY_TOOLS)
                )
            self.session.ledger.decide(
                phase="result_encoding",
                encoding=encoding.value,
                observation=observation_class(syn, expected),
                parsers_live=[p.value for p in sorted(narrowed, key=lambda x: x.value)],
                succeeded=ok,
            )
            if not (ok and settled is not None and outcome is not None):
                continue

            # G3 — termination / no-call (OB15/OB16). Diagnosis previously
            # accepted after G2 alone, so a driver could pass the online
            # trajectory yet fail independent certification on the final turn.
            # Completing the stateful trajectory here is free of representation
            # priors and preserves fail-closed behaviour.
            for call in outcome.calls:
                if call.name in RECOVERY_TOOLS:
                    fork.add_tool_result(
                        call_id=call.id,
                        name=call.name,
                        content=inst.recovery_results().get(
                            call.name, {"status": "ok"}
                        ),
                    )
            self.session.check_can_explore()
            syn3, _ = fork.generate(
                purpose="explore",
                label=f"G3@{encoding.value}",
                reason=(
                    "completing the stateful trajectory: recovery results injected; "
                    "expect a no-call termination carrying the ack sentinel"
                ),
                offered_tool_names={t.name for t in gauntlet_tools()},
            )
            final_text = (syn3.content_text or "").strip()
            # G3 is a no-call turn: require that *no* parser produced tool calls
            # (not merely that accepted_parser is unset under AMBIGUOUS).
            any_parser_calls = any(
                o.n_calls > 0 for o in syn3.parser_outcomes.values()
            )
            g3_ok = (
                syn3.consensus != ParseConsensus.AMBIGUOUS
                and not any_parser_calls
                and not syn3.tool_call_emitted
                and final_text == inst.ack_value
            )
            self.session.ledger.decide(
                phase="termination",
                encoding=encoding.value,
                observation=(
                    "final_ack" if g3_ok
                    else (
                        "ambiguous_parse"
                        if syn3.consensus == ParseConsensus.AMBIGUOUS
                        else ("tool_calls" if any_parser_calls else "bad_or_missing_ack")
                    )
                ),
                succeeded=g3_ok,
                final_text=final_text[:80],
            )
            if g3_ok:
                traj.branch = fork
                return encoding, settled
        return None

    # ------------------------------------------------------------------
    # top level
    # ------------------------------------------------------------------

    def _account_cert_generations(
        self, n: int, *, driver: Driver, label: str
    ) -> None:
        """Record certification cost without permitting ledger budget overflow."""
        if n > CERTIFICATION_GENERATION_UPPER_BOUND:
            raise BudgetExhausted(
                "certification produced more generations than its hard upper bound "
                f"({n} > {CERTIFICATION_GENERATION_UPPER_BOUND})"
            )
        used = self.session.ledger.generation_count
        if used + n > self.session.budget.max_generations:
            raise BudgetExhausted(
                "certification accounting would exceed the hard generation budget "
                f"(used={used}, certification={n}, "
                f"max={self.session.budget.max_generations})"
            )
        for i in range(n):
            self.session.ledger.generations.append(
                Generation(
                    index=len(self.session.ledger.generations) + 1,
                    purpose="certify",
                    label=label,
                    branch_id="cert",
                    forked_from=None,
                    prefix_hash="",
                    driver=driver.canonical_dict(),
                    request={},
                    request_hash=f"cert:{label}:{i}",
                    response=None,
                    response_hash=None,
                    error=None,
                    latency_ms=0.0,
                    prompt_chars=0,
                    completion_chars=0,
                    selection_reason="independent certification",
                )
            )

    def _diag_gen_count(self) -> int:
        return sum(1 for g in self.session.ledger.generations if g.purpose != "certify")

    def run(self) -> XptResult:
        started = self.session.clock()
        result = XptResult(status=UNSUPPORTED, ledger=self.session.ledger)
        ambiguous = False
        attempts = 0
        last_failed_obligations: list[str] = []
        last_driver: Driver | None = None
        cert_gens_total = 0

        try:
            for traj in self._candidate_configs():
                attempts += 1
                live = set(traj.syndrome.compatible_parsers)
                if traj.syndrome.consensus == ParseConsensus.AMBIGUOUS:
                    ambiguous = True
                    self.session.ledger.note(
                        "ambiguous parse at the first turn; refusing to guess"
                    )
                    continue
                if not live:
                    continue
                result.equivalent_parsers = sorted(p.value for p in live)
                settled = self.complete_trajectory(traj, live)
                if settled is None:
                    self._mark_trajectory_failed(traj.config)
                    self._safe_eliminate_wire_class(
                        traj.config,
                        reason="stateful trajectory failed after G1",
                    )
                    continue

                encoding, parser = settled
                driver = _driver_from(traj.config, parser, encoding)
                self.session.ledger.decide(
                    phase="synthesis",
                    driver=driver.canonical_dict(),
                    configurations_attempted=attempts,
                    reason=(
                        "request configuration accepted only after the full stateful "
                        "trajectory (G1+G2+G3); parser is the least-capable survivor of "
                        "the compatible sets intersected across every observed turn; "
                        "tool-result encoding from the counterfactual fork"
                    ),
                )

                try:
                    self.session.check_can_certify(
                        required_generations=CERTIFICATION_GENERATION_UPPER_BOUND
                    )
                except DeadlineExceeded as exc:
                    result.status = ENDPOINT_TOO_SLOW
                    result.reason = str(exc)
                    result.driver = driver
                    result.diagnosis_generations = self._diag_gen_count()
                    result.certification_generations = cert_gens_total
                    result.wall_clock_s = self.session.clock() - started
                    result.left_compiled_dag = self.left_dag
                    return result
                except BudgetExhausted as exc:
                    result.status = BUDGET_EXHAUSTED
                    result.reason = str(exc)
                    result.driver = driver
                    result.diagnosis_generations = self._diag_gen_count()
                    result.certification_generations = cert_gens_total
                    result.wall_clock_s = self.session.clock() - started
                    result.left_compiled_dag = self.left_dag
                    return result

                cert_inst = mint_instance(
                    seed=self.seed + 977, salt="certify", surface_form="B"
                )
                # ``run_probe`` performs one initial generation plus at most
                # ``max_cycles`` resumptions.  Two tool cycles therefore hard-cap
                # the frozen G1->G2->G3 certification trajectory at 3 generations.
                run = certify(
                    driver,
                    self.client,
                    cert_inst,
                    max_cycles=CERTIFICATION_GENERATION_UPPER_BOUND - 1,
                )
                remaining_after_cert = self.session.remaining_s
                if remaining_after_cert is not None and remaining_after_cert <= 0:
                    raise DeadlineExceeded(
                        "compiler wall-clock deadline reached during certification"
                    )
                # Infra/config failures during cert must not SAFE-eliminate wire classes.
                if run.termination == Termination.INFRASTRUCTURE_FAILED:
                    raise InfrastructureFailed(
                        run.errors[0] if run.errors else "infrastructure failure during certification"
                    )
                if run.termination == Termination.CONFIGURATION_FAILED:
                    raise ConfigurationFailed(
                        run.errors[0] if run.errors else "configuration failure during certification"
                    )
                if run.generations > CERTIFICATION_GENERATION_UPPER_BOUND:
                    raise RuntimeError(
                        "certification exceeded its frozen generation upper bound "
                        f"({run.generations} > {CERTIFICATION_GENERATION_UPPER_BOUND})"
                    )
                cert_gens_total += run.generations
                self._account_cert_generations(
                    run.generations, driver=driver, label=f"cert@{traj.config.key}"
                )

                # Obligations + production evaluator must both pass (certify.passed).
                if run.passed:
                    result.status = CERTIFIED
                    result.reason = (
                        "all mandatory ABI obligations verified on a fresh instance"
                    )
                    result.driver = driver
                    result.certificate = run.certificate
                    result.certification_generations = cert_gens_total
                    result.diagnosis_generations = self._diag_gen_count()
                    result.wall_clock_s = self.session.clock() - started
                    result.io_sizes = [
                        (g.prompt_chars, g.completion_chars)
                        for g in self.session.ledger.generations
                        if g.purpose != "certify"
                    ] + [
                        (len(_json(w.get("request"))), len(_json(w.get("response"))))
                        for w in run.wire
                    ]
                    result.left_compiled_dag = self.left_dag
                    return result

                # Cert failed: SAFE-eliminate this wire class, replan.
                last_driver = driver
                last_failed_obligations = sorted(
                    set(run.certificate.failed_ids + run.certificate.missing_ids)
                )
                # When obligations passed but evaluate_trace interface failed,
                # surface evaluator errors so the result is not an empty-obligation
                # trajectory-budget message.
                if not last_failed_obligations and not run.passed:
                    eval_errors = list(run.evaluator_result.get("errors") or [])
                    if eval_errors:
                        last_failed_obligations = [
                            f"evaluator:{e}" for e in eval_errors[:8]
                        ]
                    elif run.evaluator_result.get("interface_ok") is False:
                        last_failed_obligations = ["evaluator:interface_ok=false"]
                result.certificate = run.certificate
                self._mark_trajectory_failed(traj.config)
                self._safe_eliminate_wire_class(
                    traj.config,
                    reason=(
                        f"independent certification failed: {last_failed_obligations}"
                    ),
                )
        except DeadlineExceeded as exc:
            result.status = ENDPOINT_TOO_SLOW
            result.reason = str(exc)
            result.diagnosis_generations = self._diag_gen_count()
            result.certification_generations = cert_gens_total
            result.driver = last_driver
            result.failed_obligations = last_failed_obligations
            result.wall_clock_s = self.session.clock() - started
            result.left_compiled_dag = self.left_dag
            return result
        except BudgetExhausted as exc:
            result.status = BUDGET_EXHAUSTED
            result.reason = str(exc)
            result.diagnosis_generations = self._diag_gen_count()
            result.certification_generations = cert_gens_total
            result.driver = last_driver
            result.failed_obligations = last_failed_obligations
            result.wall_clock_s = self.session.clock() - started
            result.left_compiled_dag = self.left_dag
            return result
        except InfrastructureFailed as exc:
            result.status = INFRASTRUCTURE_FAILED
            result.reason = str(exc)
            result.diagnosis_generations = self._diag_gen_count()
            result.certification_generations = cert_gens_total
            result.driver = last_driver
            result.failed_obligations = last_failed_obligations
            result.wall_clock_s = self.session.clock() - started
            result.left_compiled_dag = self.left_dag
            return result
        except ConfigurationFailed as exc:
            result.status = CONFIGURATION_FAILED
            result.reason = str(exc)
            result.diagnosis_generations = self._diag_gen_count()
            result.certification_generations = cert_gens_total
            result.driver = last_driver
            result.failed_obligations = last_failed_obligations
            result.wall_clock_s = self.session.clock() - started
            result.left_compiled_dag = self.left_dag
            return result

        result.diagnosis_generations = self._diag_gen_count()
        result.certification_generations = cert_gens_total
        result.driver = last_driver
        if last_failed_obligations:
            result.failed_obligations = last_failed_obligations
            result.reason = (
                f"{attempts} configuration(s) attempted; last independent certification "
                f"failed: {last_failed_obligations}"
            )
        elif ambiguous:
            result.reason = (
                "multiple parsers accepted the same response with different "
                "canonical ASTs; refusing to guess (ambiguous parse)"
            )
        elif attempts == 0:
            result.reason = (
                "no request configuration in the Driver Grammar produced a "
                "conformant first tool turn"
            )
        else:
            result.reason = (
                f"{attempts} configuration(s) passed the first tool turn, but none "
                "produced a conformant stateful trajectory within the online budget"
            )
        result.wall_clock_s = self.session.clock() - started
        result.left_compiled_dag = self.left_dag
        return result


def xpt_compile(
    client: Any,
    program: DiagnosticProgram,
    *,
    budget: Budget | None = None,
    seed: int = 1,
    clock: Callable[[], float] = time.perf_counter,
    started_at: float | None = None,
) -> XptResult:
    return XptCompiler(
        client,
        program,
        budget=budget,
        seed=seed,
        clock=clock,
        started_at=started_at,
    ).run()


def _json(value: Any) -> str:
    import json

    return json.dumps(value, default=str)
