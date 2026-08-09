"""XPT online runtime: diagnose an unknown endpoint, synthesize a driver, certify it.

The compatibility seed path still walks the precompiled v0.1 request frontier.
When a rejected wire exposes bounded nonce-bound structural evidence, the first
active path performs:

    1  partial request hypothesis        infer parameters locally; test only role
    2  response primitive synthesis      FREE over the successful paid G1 bytes
    3  partial result hypothesis         infer segments locally; test only placement
    4  unchanged G2/G3 validation        component-isolated counterexample refinement
    5  independent certification         production runtime + production evaluator

The active and legacy paths share the same hard generation/deadline accounting.

The stacked discriminating path does not require those examples.  A strict
ordinary API rejection may name a rejected protocol parameter without supplying
its accepted value. XPT then maintains an explicit primitive-property version
space, changes the smallest implicated property, compares behavioral deltas,
and validates the lowest-complexity working survivor. Generic negative model
behavior never eliminates a version.

The oracle-free diagnostic path removes even that property-local requirement.
A generic rejection activates bounded normal-wire probes containing controlled
catalog or result alternatives. Fresh structured canaries define predicted
outcome partitions; deterministic minimax planning chooses a probe, and only one
exclusive parsed witness can refine the version space. Diagnostic generations
never count as Tool ABI witnesses. The sole survivors must still pass a separate
clean G1/G2/G3 trajectory and independent production-runtime certification.

The algorithm never sees a model name, a provider name, an endpoint type, a
candidate id or a reference answer. Its only input is observable values returned
by `chat_completions`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from xenolect.abi.events import (
    AssistantText,
    AssistantToolCall,
    ToolCall,
    ToolCallBatch,
    ToolResult,
)
from xenolect.driver.ir import (
    Driver,
    NativeToolCallsParser,
    ParserKind,
    ProtocolProgram,
    ResponsePrimitive,
    ResultField,
    SchemaTransform,
    ToolEncoding,
    ToolResultEncoding,
    ToolResultMessage,
    composed_driver,
    effective_protocol,
)
from xenolect.driver.parse import parse_model_response_full
from xenolect.driver.termination import Termination
from xenolect.eval.termination import FinalTerminationWitness, assess_g3_termination
from xenolect.xpt.certify import certify
from xenolect.xpt.diagnostic_probe import (
    MAX_REQUEST_PROBES,
    build_request_probe,
    build_result_probe,
    candidate_drivers_for_probe,
    check_identifiability,
    diagnostic_witness_evidence,
    is_generic_invalid_request,
    observe_probe_response,
)
from xenolect.xpt.discrimination import (
    ControlledExperiment,
    ProtocolRejection,
    RequestVersion,
    ResultVersion,
    VersionSpace,
    parse_protocol_rejection,
    request_version_space,
    request_version_to_hypothesis,
    result_version_space,
    result_version_to_program,
)
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
    GauntletInstance,
    gauntlet_tools,
    mint_instance,
    render_user_turn,
)
from xenolect.xpt.hypothesis import (
    ContradictionClass,
    ObligationDirectedPlanner,
    PartialProtocolHypothesis,
    ProtocolComponent,
    ProtocolSynthesisReport,
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
from xenolect.xpt.protocol_synthesis import (
    WitnessPhase,
    component_observation_evidence,
    counterexample_evidence,
    discover_request_program_from_example,
    discover_tool_result_program_from_example,
    evidence_from_counterexample,
    extract_counterexample,
    negative_behavior_evidence,
    obligation_support_evidence,
    obligation_witness_evidence,
    synthesize_request_program,
    synthesize_tool_result_program,
)
from xenolect.xpt.response_discovery import discover_response_parser
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
from xenolect.xpt.syndrome import (
    ParseConsensus,
    Syndrome,
    apply_discovered_response,
    build_syndrome,
    sha,
)

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
    synthesis_report: ProtocolSynthesisReport | None = None
    failure_class: str | None = None

    @property
    def total_generations(self) -> int:
        return self.diagnosis_generations + self.certification_generations

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "driver": self.driver.canonical_dict() if self.driver else None,
            "reason": self.reason,
            "failure_class": self.resolved_failure_class,
            "diagnosis_generations": self.diagnosis_generations,
            "certification_generations": self.certification_generations,
            "total_generations": self.total_generations,
            "failed_obligations": list(self.failed_obligations),
            "mandatory_coverage": self.certificate.mandatory_coverage,
            "certificate": self.certificate.as_dict(),
            "wall_clock_s": self.wall_clock_s,
            "left_compiled_dag": self.left_compiled_dag,
            "equivalent_parsers": list(self.equivalent_parsers),
            "io_sizes": [list(x) for x in self.io_sizes],
            "evidence_summary": self.evidence_summary(),
            "ledger": self.ledger.as_dict(),
            "protocol_synthesis": (
                self.synthesis_report.as_dict() if self.synthesis_report is not None else None
            ),
        }

    @property
    def resolved_failure_class(self) -> str | None:
        if self.failure_class is not None:
            return self.failure_class
        if self.status == CERTIFIED:
            return None
        if self.status == BUDGET_EXHAUSTED:
            return "budget_exhaustion"
        if self.status == ENDPOINT_TOO_SLOW:
            return "endpoint_too_slow"
        if self.status == INFRASTRUCTURE_FAILED:
            return "infrastructure_failure"
        if self.status == CONFIGURATION_FAILED:
            return "endpoint_configuration_failure"
        if self.failed_obligations:
            return "independent_certification_failure"
        if self.synthesis_report is not None and self.synthesis_report.failure_class:
            return self.synthesis_report.failure_class
        return "unsupported_no_working_program"

    def evidence_summary(self) -> dict[str, Any]:
        termination = [
            decision
            for decision in self.ledger.decisions
            if "termination" in str(decision.get("phase", ""))
        ]
        g1_successes = sum(
            1
            for decision in self.ledger.decisions
            if decision.get("phase") in {"config", "config-continued"}
            and decision.get("succeeded") is True
        )
        g1_successes += sum(
            1
            for delta in (
                self.synthesis_report.behavioral_deltas
                if self.synthesis_report is not None
                else ()
            )
            if delta.get("outcome") == "exact_G1_batch"
        )
        g2_successes = sum(
            1
            for decision in self.ledger.decisions
            if decision.get("phase") == "result_encoding"
            and decision.get("succeeded") is True
        )
        g2_successes += sum(
            1
            for delta in (
                self.synthesis_report.behavioral_deltas
                if self.synthesis_report is not None
                else ()
            )
            if delta.get("outcome") == "exact_G2_recovery_batch"
        )
        return {
            "request_protocols_with_valid_g1": g1_successes,
            "result_trajectories_with_valid_g2": g2_successes,
            "protocol_termination_witnesses": sum(
                decision.get("protocol_termination_verified") is True
                for decision in termination
            ),
            "exact_response_format_style_deviations": sum(
                decision.get("protocol_termination_verified") is True
                and decision.get("exact_response_format_followed") is False
                for decision in termination
            ),
            "working_protocol_evidence_observed": g1_successes > 0 and g2_successes > 0,
            "independent_certification_attempted": self.certification_generations > 0,
            "failed_obligations": list(self.failed_obligations),
        }


@dataclass
class _Trajectory:
    """A G1 branch that succeeded, together with what it proved."""

    branch: Branch
    config: RequestConfig
    syndrome: Syndrome
    frozen_prefix: str
    active_hypothesis: PartialProtocolHypothesis | None = None
    discriminating: bool = False
    oracle_free: bool = False
    request_version: RequestVersion | None = None


def _driver_from(
    config: RequestConfig, parser: ParserKind, result_encoding: ToolResultEncoding
) -> Driver:
    return composed_driver(
        tool_encoding=ToolEncoding(config.tool_encoding),
        parser=parser,
        schema_transforms=[SchemaTransform(t) for t in config.transforms],
        tool_result_encoding=result_encoding,
    )


def _driver_from_discovered(
    config: RequestConfig,
    parser: ResponsePrimitive,
    result_encoding: ToolResultEncoding,
) -> Driver:
    """Keep the proven request/result program and replace only response parsing."""
    base = _driver_from(config, ParserKind.NATIVE, result_encoding)
    assert base.protocol is not None
    protocol = ProtocolProgram(
        request=list(base.protocol.request),
        response=[NativeToolCallsParser(), parser],
        tool_result=base.protocol.tool_result,
        state=list(base.protocol.state),
    )
    return Driver(
        ir_version="0.2",
        schema_transforms=list(base.schema_transforms),
        protocol=protocol,
    )


def _observe_with_driver(
    raw: Any, driver: Driver
) -> tuple[tuple[ToolCall, ...], bool, str, tuple[str, ...]]:
    """Evaluate one paid response through a synthesized Driver, locally."""
    if not isinstance(raw, dict):
        return (), False, "", ("response is not an object",)
    parsed = parse_model_response_full(raw, driver)
    calls: list[ToolCall] = []
    is_batch = False
    text_parts: list[str] = []
    for event in parsed.events:
        if isinstance(event, AssistantToolCall):
            calls.append(event.call)
            if event.content:
                text_parts.append(event.content)
        elif isinstance(event, ToolCallBatch):
            calls.extend(event.calls)
            is_batch = True
            if event.content:
                text_parts.append(event.content)
        elif isinstance(event, AssistantText):
            text_parts.append(event.content)
    return tuple(calls), is_batch, "".join(text_parts), tuple(parsed.errors)


def _diagnosis_g3_witness(
    inst: GauntletInstance,
    *,
    final_text: str,
    has_tool_calls: bool,
    parser_ambiguous: bool = False,
    parse_errors: tuple[str, ...] = (),
) -> FinalTerminationWitness:
    """Apply the same nonce/provenance rule to a paid diagnosis G3 turn."""

    unavailable = (
        render_user_turn(inst),
        inst.expected_batch_arguments(),
        inst.result_for("record_alpha"),
        inst.result_for("record_beta"),
        inst.gamma_error_text(),
        inst.recovery_results()["commit"],
    )
    return assess_g3_termination(
        final_text=final_text,
        expected_sentinel=inst.ack_value,
        source_payload=inst.recovery_results()["report"],
        unavailable_payloads=unavailable,
        has_tool_calls=has_tool_calls,
        parser_ambiguous=parser_ambiguous,
        parse_errors=parse_errors,
        normal_termination=True,
    )


def _termination_observation(witness: FinalTerminationWitness) -> str:
    if witness.protocol_termination_verified:
        return "protocol_termination_verified"
    if "parser_ambiguity" in witness.failure_codes or "parser_error" in witness.failure_codes:
        return "parser_or_schema_contradiction"
    if "spurious_tool_call" in witness.failure_codes:
        return "spurious_tool_call_observed"
    return "insufficient_nonce_bound_termination_evidence"


def _termination_evidence_class(witness: FinalTerminationWitness) -> str:
    if witness.protocol_termination_verified:
        return (
            "protocol_termination_verified"
            if witness.exact_response_format_followed
            else "model_style_instruction_deviation"
        )
    if "parser_ambiguity" in witness.failure_codes or "parser_error" in witness.failure_codes:
        return "parser_schema_contradiction"
    if (
        "spurious_tool_call" in witness.failure_codes
        or "noncompleted_turn" in witness.failure_codes
    ):
        return "termination_protocol_observation_failed"
    if "missing_or_nonunique_source" in witness.failure_codes:
        return "tool_result_protocol_evidence_insufficient"
    return "ordinary_negative_model_behavior"


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
        self.obligation_planner = ObligationDirectedPlanner()
        self.synthesis_report = ProtocolSynthesisReport()
        self._active_counterexample_seen = False
        self._active_discrimination_seen = False
        self._active_failure: str | None = None

    # ------------------------------------------------------------------
    # phase 1: request configuration
    # ------------------------------------------------------------------

    def _discover_paid_response(
        self,
        syn: Syndrome,
        generation: Generation,
        *,
        offered_names: set[str],
        expected: dict[str, dict[str, Any]],
        label: str,
    ) -> None:
        """Reuse one paid response for bounded local parser synthesis."""
        if syn.consensus != ParseConsensus.NONE or not isinstance(generation.response, dict):
            return
        discovery = discover_response_parser(
            generation.response,
            offered_tool_names=offered_names,
            expected_arguments=expected,
        )
        apply_discovered_response(
            syn,
            parser=discovery.parser,
            calls=discovery.calls,
            error=discovery.error,
            offered_tool_names=offered_names,
        )
        generation.syndrome = syn.as_dict()
        if discovery.ok:
            self.session.ledger.note(
                "response discovery synthesized and locally validated "
                f"{discovery.parser.op!r} from the paid {label} observation "
                f"({discovery.candidates_validated} agreeing candidate(s))"
            )

    def _run_template(self, probe: ProbeTemplate) -> tuple[Syndrome, Branch, bool]:
        tools, content, expected = probe_payload(probe, self.diag_inst)
        branch = self.session.new_branch(probe.config.driver())
        branch.add_user(content, tools)
        offered_names = {t.name for t in tools}
        syn, generation = branch.generate(
            purpose="explore",
            label=probe.id,
            reason=self._reason,
            offered_tool_names=offered_names,
        )
        self._discover_paid_response(
            syn,
            generation,
            offered_names=offered_names,
            expected=expected,
            label=probe.id,
        )
        annotate_arguments(syn, tools, expected)
        ok = probe_succeeded(syn, expected, batch=probe.kind == "gauntlet_turn1")
        return syn, branch, ok

    def _provisional_request_driver(self, hypothesis: PartialProtocolHypothesis) -> Driver:
        """Execute a resolved request while response/result remain typed holes."""
        if not isinstance(hypothesis.request, tuple):
            raise ValueError("request experiment requires a resolved request program")
        response = (
            list(hypothesis.response)
            if isinstance(hypothesis.response, tuple)
            else [NativeToolCallsParser()]
        )
        return Driver(
            ir_version="0.2",
            schema_transforms=list(hypothesis.schema_transforms),
            protocol=ProtocolProgram(
                request=list(hypothesis.request),
                response=response,
                tool_result=ToolResultMessage(
                    role="tool",
                    segments=[ResultField(field="content")],
                    attach_tool_call_id=True,
                ),
            ),
        )

    def _maybe_active_request_trajectory(
        self,
        probe: ProbeTemplate,
        branch: Branch,
        *,
        succeeded: bool,
    ) -> tuple[_Trajectory | None, bool]:
        """Refine a request hole from a nonce-bound example or counterexample.

        Structural examples are preferred: catalog nesting/fields and call
        framing/fields are inferred locally, while the unobservable message
        role remains a two-value typed choice.  The explicit atomic constraint
        object remains a strict alternative.  ``claimed`` means one of those
        shapes was recognized; malformed or ambiguous evidence then fails
        closed instead of being treated as ordinary model prose.
        """
        if succeeded or self._active_counterexample_seen:
            return None, False
        generation = branch.last_generation
        if generation is None:
            return None, False
        tools, content, expected = probe_payload(probe, self.diag_inst)
        structural = discover_request_program_from_example(
            generation.response,
            tools=tools,
            expected_nonce=self.diag_inst.alpha_code,
        )
        request_programs: tuple[tuple[Any, ...], ...]
        if structural.ok:
            assert structural.observation is not None
            observation = structural.observation
            request_programs = structural.candidates
        elif structural.found:
            self._active_counterexample_seen = True
            self.left_dag = True
            self._active_failure = structural.error or "invalid request example"
            self.synthesis_report.failure = self._active_failure
            return None, True
        else:
            observation = extract_counterexample(
                generation.response,
                expected_component=ProtocolComponent.REQUEST,
                expected_nonce=self.diag_inst.alpha_code,
            )
            if not observation.found:
                return None, False
            if observation.ok:
                try:
                    request_programs = (synthesize_request_program(observation),)
                except ValueError as exc:
                    self._active_counterexample_seen = True
                    self.left_dag = True
                    self._active_failure = str(exc)
                    self.synthesis_report.failure = self._active_failure
                    return None, True
            else:
                request_programs = ()
        self._active_counterexample_seen = True
        self.left_dag = True
        if not observation.ok:
            self._active_failure = observation.error or "invalid request counterexample"
            self.synthesis_report.failure = self._active_failure
            self.session.ledger.note(
                "active request synthesis rejected counterexample: " + self._active_failure
            )
            return None, True

        hypothesis = PartialProtocolHypothesis(
            schema_transforms=tuple(SchemaTransform(value) for value in probe.config.transforms)
        )
        constraint_rows = evidence_from_counterexample(observation, generation)
        for row in constraint_rows:
            self.synthesis_report.evidence.record(row)
        seed_request = tuple(effective_protocol(probe.config.driver()).request)
        seed_hypothesis = PartialProtocolHypothesis(
            request=seed_request,
            schema_transforms=hypothesis.schema_transforms,
        )
        rejected = counterexample_evidence(
            component=ProtocolComponent.REQUEST,
            generation=generation,
            observation=(
                "the paid G1 wire did not expose the catalog structure required by "
                "the endpoint's nonce-bound request constraints"
            ),
            contradiction_class=ContradictionClass.STRUCTURAL,
        )
        self.synthesis_report.evidence.eliminate(
            ProtocolComponent.REQUEST,
            seed_hypothesis.component_fingerprint(ProtocolComponent.REQUEST),
            evidence=rejected,
        )
        experiment = self.obligation_planner.choose(hypothesis, self.synthesis_report.evidence)
        if experiment is None or experiment.component != ProtocolComponent.REQUEST:
            self._active_failure = "obligation planner did not select the request hole"
            self.synthesis_report.failure = self._active_failure
            return None, True
        self.synthesis_report.record_experiment(experiment)
        self.session.ledger.decide(
            phase="active-synthesis-plan",
            **experiment.as_dict(),
        )
        offered_names = {tool.name for tool in tools}
        refined: PartialProtocolHypothesis | None = None
        active_branch: Branch | None = None
        syn: Syndrome | None = None
        paid: Generation | None = None
        for candidate_index, request_program in enumerate(request_programs, start=1):
            candidate = hypothesis.refine(ProtocolComponent.REQUEST, request_program)
            driver = self._provisional_request_driver(candidate)
            candidate_branch = self.session.new_branch(driver)
            candidate_branch.add_user(content, tools)
            candidate_syn, candidate_paid = candidate_branch.generate(
                purpose="explore",
                label=f"active-G1@request-{candidate_index}",
                reason=experiment.reason,
                offered_tool_names=offered_names,
            )
            self._discover_paid_response(
                candidate_syn,
                candidate_paid,
                offered_names=offered_names,
                expected=expected,
                label=f"active-G1@request-{candidate_index}",
            )
            annotate_arguments(candidate_syn, tools, expected)
            candidate_ok = probe_succeeded(candidate_syn, expected, batch=True)
            self.session.ledger.decide(
                phase="active-request-candidate",
                generation=candidate_paid.index,
                candidate=candidate.component_fingerprint(ProtocolComponent.REQUEST),
                candidate_index=candidate_index,
                observation=("exact_G1_batch" if candidate_ok else "request_obligations_unproven"),
                succeeded=candidate_ok,
            )
            if candidate_ok:
                refined = candidate
                active_branch = candidate_branch
                syn = candidate_syn
                paid = candidate_paid
                break
            candidate_rejected = negative_behavior_evidence(
                component=ProtocolComponent.REQUEST,
                generation=candidate_paid,
                observation=("this exact request candidate did not elicit a conformant G1 batch"),
            )
            self.synthesis_report.evidence.record(candidate_rejected)
        if refined is None or active_branch is None or syn is None or paid is None:
            self._active_failure = "bounded request candidates did not satisfy the G1 obligations"
            self.synthesis_report.failure = self._active_failure
            return None, True

        request_fact = component_observation_evidence(
            component=ProtocolComponent.REQUEST,
            generation=paid,
            observation=("synthesized request program elicited the exact three-call G1 batch"),
        )
        self.synthesis_report.evidence.record(request_fact)
        self.synthesis_report.record_revision(
            hypothesis,
            refined,
            component=ProtocolComponent.REQUEST,
            generation_id=paid.index,
            evidence_ids=[row.evidence_id for row in constraint_rows] + [request_fact.evidence_id],
            reason=(
                "composed structural parameters locally and selected the first "
                "lowest-complexity message placement proven by G1"
            ),
        )

        if syn.discovered_parser is not None:
            response_program: tuple[ResponsePrimitive, ...] = (
                NativeToolCallsParser(),
                syn.discovered_parser,
            )
        elif syn.accepted_parser is not None:
            response_program = tuple(
                effective_protocol(Driver(parser=syn.accepted_parser)).response
            )
        else:
            self._active_failure = "G1 calls have no unambiguous executable parser"
            self.synthesis_report.failure = self._active_failure
            return None, True
        response_refined = refined.refine(ProtocolComponent.RESPONSE, response_program)
        response_fact = component_observation_evidence(
            component=ProtocolComponent.RESPONSE,
            generation=paid,
            observation=(
                "response primitive parsed the exact G1 call names, arguments, batch, "
                "and call-ID discipline"
            ),
        )
        self.synthesis_report.evidence.record(response_fact)
        g1_evidence_ids = (request_fact.evidence_id, response_fact.evidence_id)
        for support in obligation_support_evidence(
            obligation_ids=("OB07", "OB08", "OB09", "OB12", "OB17", "OB18"),
            generations=(paid,),
            component_evidence_ids=g1_evidence_ids,
            observation=(
                "G1 is relevant to these obligations but does not contain their "
                "complete multi-turn witness"
            ),
        ):
            self.synthesis_report.evidence.record_support(support)
        g1_witnesses = obligation_witness_evidence(
            obligation_ids=("OB01", "OB02", "OB03", "OB04", "OB05", "OB06"),
            phase=WitnessPhase.G1,
            generations=(paid,),
            component_evidence_ids=g1_evidence_ids,
            observation=(
                "the exact G1 batch contains offered names, schema-valid exact "
                "arguments, nested schema use, and one parallel call turn"
            ),
        )
        for witness in g1_witnesses:
            self.synthesis_report.evidence.record_witness(witness)
        self.synthesis_report.record_revision(
            refined,
            response_refined,
            component=ProtocolComponent.RESPONSE,
            generation_id=paid.index,
            evidence_ids=[response_fact.evidence_id],
            reason=(
                "reused the paid G1 bytes to infer and validate one bounded response "
                "primitive without another endpoint call"
            ),
        )
        self.session.ledger.decide(
            phase="active-request-validated",
            generation=paid.index,
            request_fingerprint=response_refined.component_fingerprint(ProtocolComponent.REQUEST),
            response_fingerprint=response_refined.component_fingerprint(ProtocolComponent.RESPONSE),
            obligations=[witness.obligation_id for witness in g1_witnesses],
            succeeded=True,
        )
        return (
            _Trajectory(
                active_branch,
                probe.config,
                syn,
                active_branch.freeze(),
                active_hypothesis=response_refined,
            ),
            True,
        )

    def _record_controlled_experiment(self, experiment: ControlledExperiment) -> None:
        self.synthesis_report.experiments.append(
            {"kind": "controlled_intervention", **experiment.as_dict()}
        )
        self.session.ledger.decide(
            phase="active-discriminating-plan",
            kind="controlled_intervention",
            **experiment.as_dict(),
        )

    def _record_api_rejection(
        self,
        *,
        space: VersionSpace,
        tested: RequestVersion | ResultVersion,
        rejection: ProtocolRejection,
        generation: Generation,
    ) -> None:
        self.synthesis_report.property_local_rejections_observed += 1
        evidence = counterexample_evidence(
            component=rejection.component,
            generation=generation,
            observation=(
                f"ordinary API rejection named parameter {rejection.parameter!r} "
                "without revealing an accepted value"
            ),
            contradiction_class=ContradictionClass.WIRE_API,
            determinism_assumption=(
                "the endpoint returned an explicit unsupported-value/parameter "
                "envelope establishing property-local deterministic rejection"
            ),
        )
        self.synthesis_report.evidence.eliminate(
            rejection.component,
            tested.fingerprint,
            evidence=evidence,
        )
        removed = space.eliminate_rejected_value(
            tested, rejection, evidence_id=evidence.evidence_id
        )
        self.synthesis_report.property_local_rejections_used += 1
        self.synthesis_report.behavioral_deltas.append(
            {
                "generation_id": generation.index,
                "component": rejection.component.value,
                "candidate_fingerprint": tested.fingerprint,
                "outcome": "deterministic_api_rejection",
                "implicated_property": rejection.parameter,
                "accepted_value_revealed": False,
                "surviving_versions_removed": removed,
                "evidence_id": evidence.evidence_id,
            }
        )

    def _fail_oracle_free(
        self,
        reason: str,
        *,
        failure_class: str = "no_working_program_found",
    ) -> tuple[None, bool]:
        self._active_failure = reason
        self.synthesis_report.failure = reason
        self.synthesis_report.failure_class = failure_class
        self.session.ledger.note("oracle-free synthesis failed closed: " + reason)
        return None, True

    def _maybe_oracle_free_request_trajectory(
        self,
        probe: ProbeTemplate,
        branch: Branch,
        *,
        succeeded: bool,
    ) -> tuple[_Trajectory | None, bool]:
        """Identify a request version using only positive normal-wire witnesses.

        A generic rejection activates this path but conveys no property evidence.
        Each paid diagnostic request multiplexes the surviving exact production
        versions and accepts only a nonce-bound structured call predicted by one
        outcome partition.  Silence, prose, ambiguity and generic rejection do not
        eliminate any version.
        """
        if succeeded or self._active_discrimination_seen:
            return None, False
        initial_generation = branch.last_generation
        if initial_generation is None or not is_generic_invalid_request(
            initial_generation.response
        ):
            return None, False

        self._active_discrimination_seen = True
        self.left_dag = True
        self.synthesis_report.discriminating = True
        self.synthesis_report.oracle_free = True
        initial_negative = negative_behavior_evidence(
            component=ProtocolComponent.REQUEST,
            generation=initial_generation,
            observation=(
                "an exact production candidate received a generic rejection; this "
                "does not localize a property or eliminate sibling hypotheses"
            ),
        )
        self.synthesis_report.evidence.record(initial_negative)

        survivors = list(request_version_space())
        identifiability = check_identifiability(ProtocolComponent.REQUEST, survivors)
        self.synthesis_report.identifiability = {
            "request": identifiability.as_dict(),
        }
        if not identifiability.identifiable:
            return self._fail_oracle_free(
                "the current request protocol family is observationally "
                "unidentifiable under the available diagnostic probe family",
                failure_class="observationally_unidentifiable",
            )

        for sequence in range(1, MAX_REQUEST_PROBES + 1):
            if len(survivors) == 1:
                break
            diagnostic_probe, plan = build_request_probe(
                survivors,
                seed=self.seed,
                sequence=sequence,
            )
            diagnostic_branch = self.session.new_diagnostic_branch(diagnostic_probe)
            offered_names = {
                alternative.witness.tool_name
                for alternative in diagnostic_probe.alternatives
            }
            _, generation = diagnostic_branch.generate(
                purpose="explore",
                label=f"oracle-free-request-probe-{sequence}",
                reason=(
                    "minimax diagnostic probe over predicted positive outcome "
                    "partitions; information score is a ranking tie-break only"
                ),
                offered_tool_names=offered_names,
            )
            outcome = observe_probe_response(
                diagnostic_probe,
                generation.response,
                candidate_drivers_for_probe(survivors),
            )
            if outcome.outcome_key is None:
                plan.record_outcome(outcome)
                self.synthesis_report.probe_plans.append(plan.as_dict())
                return self._fail_oracle_free(
                    "the current request protocol family is observationally "
                    "unidentifiable under the available observations: " + outcome.reason,
                    failure_class="observationally_unidentifiable",
                )

            evidence = diagnostic_witness_evidence(
                component=ProtocolComponent.REQUEST,
                generation=generation,
                plan=plan,
                outcome=outcome,
            )
            plan.record_outcome(outcome, evidence_id=evidence.evidence_id)
            removed = set(plan.hypotheses_removed)
            self.synthesis_report.evidence.eliminate_by_diagnostic_partition(
                ProtocolComponent.REQUEST,
                removed,
                evidence=evidence,
            )
            self.synthesis_report.probe_plans.append(plan.as_dict())
            remaining = set(plan.hypotheses_remaining)
            survivors = [
                version for version in survivors if version.fingerprint in remaining
            ]
            self.session.ledger.decide(
                phase="oracle-free-request-partition",
                generation=generation.index,
                probe=plan.probe_id,
                observed_outcome=outcome.outcome_key,
                witness_ids=list(outcome.witness_ids),
                removed=sorted(removed),
                remaining=sorted(remaining),
                elimination_basis="exclusive_nonce_bound_structured_witness",
            )

        if len(survivors) != 1:
            return self._fail_oracle_free(
                "the bounded request diagnostic budget ended with multiple "
                "observationally unresolved hypotheses",
                failure_class="unresolved_under_bounded_diagnostic_budget",
            )
        selected = survivors[0]
        hypothesis = request_version_to_hypothesis(selected)
        driver = self._provisional_request_driver(hypothesis)
        tools, content, expected = probe_payload(probe, self.diag_inst)
        offered_names = {tool.name for tool in tools}
        clean_branch = self.session.new_branch(driver)
        clean_branch.add_user(content, tools)
        syndrome, generation = clean_branch.generate(
            purpose="explore",
            label=f"oracle-free-clean-G1@{selected.fingerprint[:8]}",
            reason=(
                "clean production G1 validation is separate from multiplexed "
                "diagnostic evidence"
            ),
            offered_tool_names=offered_names,
        )
        calls, is_batch, _, errors = _observe_with_driver(generation.response, driver)
        values = {call.name: call.arguments for call in calls}
        clean_ok = (
            not errors
            and is_batch
            and len(calls) == len(expected)
            and set(values) == set(expected)
            and all(values[name] == expected[name] for name in expected)
        )
        if not clean_ok:
            negative = negative_behavior_evidence(
                component=ProtocolComponent.REQUEST,
                generation=generation,
                observation=(
                    "the diagnostic survivor did not produce the complete clean G1 "
                    "witness; diagnostic success is not a production ABI witness"
                ),
            )
            self.synthesis_report.evidence.record(negative)
            return self._fail_oracle_free(
                "the diagnostic request survivor failed clean production G1 validation"
            )
        if not syndrome.accepted_calls:
            assert isinstance(hypothesis.response, tuple)
            apply_discovered_response(
                syndrome,
                parser=hypothesis.response[-1],
                calls=calls,
                offered_tool_names=offered_names,
            )
            generation.syndrome = syndrome.as_dict()
        if tuple(syndrome.accepted_calls) != tuple(calls):
            return self._fail_oracle_free(
                "clean G1 parser interpretations disagreed; refusing an ambiguous Driver",
                failure_class="ambiguous_observation",
            )
        annotate_arguments(syndrome, tools, expected)

        request_fact = component_observation_evidence(
            component=ProtocolComponent.REQUEST,
            generation=generation,
            observation="the diagnostic survivor elicited the exact clean production G1 batch",
        )
        response_fact = component_observation_evidence(
            component=ProtocolComponent.RESPONSE,
            generation=generation,
            observation=(
                "the survivor's response program parsed exact clean G1 names, "
                "arguments, batch structure and call IDs"
            ),
        )
        self.synthesis_report.evidence.record(request_fact)
        self.synthesis_report.evidence.record(response_fact)
        component_ids = (request_fact.evidence_id, response_fact.evidence_id)
        for witness in obligation_witness_evidence(
            obligation_ids=("OB01", "OB02", "OB03", "OB04", "OB05", "OB06"),
            phase=WitnessPhase.G1,
            generations=(generation,),
            component_evidence_ids=component_ids,
            observation=(
                "only the clean production G1 turn contains the complete call/schema "
                "witness for these obligations"
            ),
        ):
            self.synthesis_report.evidence.record_witness(witness)
        for support in obligation_support_evidence(
            obligation_ids=("OB07", "OB08", "OB09", "OB12", "OB17", "OB18"),
            generations=(generation,),
            component_evidence_ids=component_ids,
            observation="clean G1 supports but cannot complete these multi-turn obligations",
        ):
            self.synthesis_report.evidence.record_support(support)

        base = PartialProtocolHypothesis(schema_transforms=hypothesis.schema_transforms)
        request_only = base.refine(ProtocolComponent.REQUEST, hypothesis.request)
        self.synthesis_report.record_revision(
            base,
            request_only,
            component=ProtocolComponent.REQUEST,
            generation_id=generation.index,
            evidence_ids=[request_fact.evidence_id],
            reason="selected the sole positive-witness request partition survivor",
        )
        resolved = request_only.refine(ProtocolComponent.RESPONSE, hypothesis.response)
        self.synthesis_report.record_revision(
            request_only,
            resolved,
            component=ProtocolComponent.RESPONSE,
            generation_id=generation.index,
            evidence_ids=[response_fact.evidence_id],
            reason="validated the paired response program on a clean production G1 turn",
        )
        self.synthesis_report.version_spaces.append(
            {
                "component": ProtocolComponent.REQUEST.value,
                "initial_size": len(request_version_space()),
                "survivor_fingerprints": [selected.fingerprint],
                "survivors": [selected.as_dict()],
                "refinement_basis": "diagnostic_positive_witness_partitions",
            }
        )
        return (
            _Trajectory(
                clean_branch,
                probe.config,
                syndrome,
                clean_branch.freeze(),
                active_hypothesis=resolved,
                discriminating=True,
                oracle_free=True,
                request_version=selected,
            ),
            True,
        )

    def _maybe_discriminating_request_trajectory(
        self,
        probe: ProbeTemplate,
        branch: Branch,
        *,
        succeeded: bool,
    ) -> tuple[_Trajectory | None, bool]:
        """Actively refine a request version space from ordinary API behavior."""
        if succeeded or self._active_discrimination_seen:
            return None, False
        initial_generation = branch.last_generation
        if initial_generation is None:
            return None, False
        rejection = parse_protocol_rejection(initial_generation.response)
        if rejection is None or rejection.component != ProtocolComponent.REQUEST:
            return None, False

        self._active_discrimination_seen = True
        self.left_dag = True
        self.synthesis_report.discriminating = True
        space = VersionSpace(ProtocolComponent.REQUEST, request_version_space())
        native = next(
            version
            for version in space.surviving_versions()
            if isinstance(version, RequestVersion) and version.mode == "native"
        )
        self._record_api_rejection(
            space=space,
            tested=native,
            rejection=rejection,
            generation=initial_generation,
        )
        previous: RequestVersion = native
        implicated = rejection.parameter
        tools, content, expected = probe_payload(probe, self.diag_inst)
        offered_names = {tool.name for tool in tools}

        while space.survivors:
            selected_raw, experiment = space.choose(
                previous=previous,
                implicated_parameter=implicated,
                expected_obligation_gain=6,
            )
            assert isinstance(selected_raw, RequestVersion)
            selected = selected_raw
            self._record_controlled_experiment(experiment)
            hypothesis = request_version_to_hypothesis(selected)
            driver = self._provisional_request_driver(hypothesis)
            candidate_branch = self.session.new_branch(driver)
            candidate_branch.add_user(content, tools)
            self.session.check_can_explore()
            candidate_syn, generation = candidate_branch.generate(
                purpose="explore",
                label=f"discriminate-G1@{selected.fingerprint[:8]}",
                reason=(
                    "testing the lowest-complexity surviving request program; "
                    "only controlled property deltas distinguish it from the prior wire"
                ),
                offered_tool_names=offered_names,
            )
            self._discover_paid_response(
                candidate_syn,
                generation,
                offered_names=offered_names,
                expected=expected,
                label=f"discriminate-G1@{selected.fingerprint[:8]}",
            )
            annotate_arguments(candidate_syn, tools, expected)
            calls, is_batch, _, errors = _observe_with_driver(generation.response, driver)
            values = {call.name: call.arguments for call in calls}
            candidate_ok = (
                not errors
                and is_batch
                and len(calls) == len(expected)
                and set(values) == set(expected)
                and all(values[name] == expected[name] for name in expected)
            )
            if candidate_ok:
                request_fact = component_observation_evidence(
                    component=ProtocolComponent.REQUEST,
                    generation=generation,
                    observation=("the controlled request intervention elicited the exact G1 batch"),
                )
                response_fact = component_observation_evidence(
                    component=ProtocolComponent.RESPONSE,
                    generation=generation,
                    observation=(
                        "the invented response program parsed exact names, arguments, "
                        "parallel calls, and IDs from normal assistant output"
                    ),
                )
                self.synthesis_report.evidence.record(request_fact)
                self.synthesis_report.evidence.record(response_fact)
                g1_ids = (request_fact.evidence_id, response_fact.evidence_id)
                witnesses = obligation_witness_evidence(
                    obligation_ids=("OB01", "OB02", "OB03", "OB04", "OB05", "OB06"),
                    phase=WitnessPhase.G1,
                    generations=(generation,),
                    component_evidence_ids=g1_ids,
                    observation=(
                        "the accepted intervention contains the complete G1 witness "
                        "for the listed call/schema obligations"
                    ),
                )
                for witness in witnesses:
                    self.synthesis_report.evidence.record_witness(witness)
                for support in obligation_support_evidence(
                    obligation_ids=("OB07", "OB08", "OB09", "OB12", "OB17", "OB18"),
                    generations=(generation,),
                    component_evidence_ids=g1_ids,
                    observation="G1 contributes to but does not complete these obligations",
                ):
                    self.synthesis_report.evidence.record_support(support)
                base = PartialProtocolHypothesis(schema_transforms=hypothesis.schema_transforms)
                request_only = base.refine(ProtocolComponent.REQUEST, hypothesis.request)
                self.synthesis_report.record_revision(
                    base,
                    request_only,
                    component=ProtocolComponent.REQUEST,
                    generation_id=generation.index,
                    evidence_ids=[request_fact.evidence_id],
                    reason=(
                        "selected the lowest-complexity request version surviving "
                        "controlled API rejections"
                    ),
                )
                resolved = request_only.refine(ProtocolComponent.RESPONSE, hypothesis.response)
                self.synthesis_report.record_revision(
                    request_only,
                    resolved,
                    component=ProtocolComponent.RESPONSE,
                    generation_id=generation.index,
                    evidence_ids=[response_fact.evidence_id],
                    reason=(
                        "used the response frame and field map invented by the accepted "
                        "request intervention"
                    ),
                )
                self.synthesis_report.behavioral_deltas.append(
                    {
                        "generation_id": generation.index,
                        "component": "request",
                        "candidate_fingerprint": selected.fingerprint,
                        "outcome": "exact_G1_batch",
                        "implicated_property": None,
                        "accepted_value_revealed": False,
                        "surviving_versions_removed": 0,
                    }
                )
                self.synthesis_report.version_spaces.append(space.as_dict())
                return (
                    _Trajectory(
                        candidate_branch,
                        probe.config,
                        candidate_syn,
                        candidate_branch.freeze(),
                        active_hypothesis=resolved,
                        discriminating=True,
                        request_version=selected,
                    ),
                    True,
                )

            next_rejection = parse_protocol_rejection(generation.response)
            if next_rejection is None or next_rejection.component != ProtocolComponent.REQUEST:
                negative = negative_behavior_evidence(
                    component=ProtocolComponent.REQUEST,
                    generation=generation,
                    observation=(
                        "request intervention did not produce G1 and returned no "
                        "deterministic protocol rejection"
                    ),
                )
                self.synthesis_report.evidence.record(negative)
                self.synthesis_report.version_spaces.append(space.as_dict())
                self._active_failure = (
                    "discriminating request experiment produced only stochastic "
                    "negative behavior; no SAFE refinement is available"
                )
                self.synthesis_report.failure = self._active_failure
                return None, True
            self._record_api_rejection(
                space=space,
                tested=selected,
                rejection=next_rejection,
                generation=generation,
            )
            previous = selected
            implicated = next_rejection.parameter

        self.synthesis_report.version_spaces.append(space.as_dict())
        self._active_failure = "deterministic rejections exhausted the request version space"
        self.synthesis_report.failure = self._active_failure
        return None, True

    def _maybe_synthesis_request_trajectory(
        self,
        probe: ProbeTemplate,
        branch: Branch,
        *,
        succeeded: bool,
    ) -> tuple[_Trajectory | None, bool]:
        oracle_free, claimed = self._maybe_oracle_free_request_trajectory(
            probe, branch, succeeded=succeeded
        )
        if claimed:
            return oracle_free, True
        active, claimed = self._maybe_active_request_trajectory(probe, branch, succeeded=succeeded)
        if claimed:
            return active, True
        return self._maybe_discriminating_request_trajectory(probe, branch, succeeded=succeeded)

    # ------------------------------------------------------------------
    # candidate configurations
    # ------------------------------------------------------------------

    def _frontier_evidence(self, tried: set[str]) -> FrontierEvidence:
        """Build replanner evidence from paid observations and explicit trials.

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

        Equality permits observation reuse and experiment-cost ranking. It does
        not make a stochastic response repeatable, so it is not itself a SAFE
        hypothesis-elimination rule.
        """
        target = g1_fingerprint(cfg, seed=self.seed).full_hash
        return [
            c.key
            for c in all_request_configs()
            if g1_fingerprint(c, seed=self.seed).full_hash == target
        ]

    def _mark_g1_wire(self, cfg: RequestConfig, *, ok: bool) -> str:
        """Fingerprint G1 and retain failures as heuristic ranking evidence."""
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

    def _safe_eliminate_wire_class(
        self,
        cfg: RequestConfig,
        *,
        reason: str,
        contradiction_class: ContradictionClass,
    ) -> None:
        """Eliminate a wire class only after a deterministic contradiction."""
        if contradiction_class not in {
            ContradictionClass.STRUCTURAL,
            ContradictionClass.WIRE_API,
            ContradictionClass.PARSER_SCHEMA,
        }:
            raise ValueError("ordinary model behavior is not a SAFE wire-class elimination")
        if not hasattr(self, "_safe_eliminated"):
            self._safe_eliminated = set()
        keys = self._wire_class_keys(cfg)
        for key in keys:
            self._safe_eliminated.add(key)
        self.session.ledger.note(
            "SAFE deterministic elimination of G1 wire class "
            f"{cfg.key} ({len(keys)} configs): {reason}"
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
            active, claimed = self._maybe_synthesis_request_trajectory(probe, branch, succeeded=ok)
            if claimed:
                if active is not None:
                    yield active
                return
            if ok and probe.kind == "gauntlet_turn1":
                yield _Trajectory(branch, probe.config, syn, branch.freeze())
                exited_early = True
                break
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
                active, claimed = self._maybe_synthesis_request_trajectory(
                    probe, branch, succeeded=ok
                )
                if claimed:
                    if active is not None:
                        yield active
                    return
                if ok:
                    yield _Trajectory(branch, cfg, syn, branch.freeze())
                else:
                    self.left_dag = True

        # Open-world continuation: replan over remaining wire-distinct configs.
        # Complexity must not decide which expensive request is sent next.
        self.left_dag = self.left_dag or bool(tried)
        while True:
            # Resource exhaustion is not protocol evidence.  Propagate it to the
            # top-level compiler so it is reported as BUDGET_EXHAUSTED or
            # ENDPOINT_TOO_SLOW rather than UNSUPPORTED.
            self.session.check_can_explore()
            # Merge only explicit deterministic eliminations. Ordinary G1,
            # trajectory, and certification failures remain ranking evidence.
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
            # Evaluate one representative. Equal wires share cost features but
            # remain live unless deterministic protocol evidence rejects them.
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
            active, claimed = self._maybe_synthesis_request_trajectory(probe, branch, succeeded=ok)
            if claimed:
                if active is not None:
                    yield active
                return
            if ok:
                # G1 success: only this exact config is in flight.
                tried.add(cfg.key)
                yield _Trajectory(branch, cfg, syn, branch.freeze())
            else:
                tried.add(cfg.key)
                self.session.ledger.note(
                    f"negative G1 behavior for {cfg.key}; retained only as "
                    "heuristic evidence because the endpoint may be stochastic"
                )

    # ------------------------------------------------------------------
    # phase 2: parser, resolved across the WHOLE trajectory (free)
    # ------------------------------------------------------------------

    @staticmethod
    def _narrow_parsers(live: set[ParserKind], syn: Syndrome) -> set[ParserKind]:
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
                values = {c.name: c.arguments == expected.get(c.name) for c in outcome.calls}
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
                        content=inst.recovery_results().get(call.name, {"status": "ok"}),
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
            final_text = syn3.content_text or ""
            # G3 is a no-call turn: require that *no* parser produced tool calls
            # (not merely that accepted_parser is unset under AMBIGUOUS).
            any_parser_calls = any(o.n_calls > 0 for o in syn3.parser_outcomes.values())
            parser_errors = tuple(
                sorted(
                    {
                        error
                        for outcome in syn3.parser_outcomes.values()
                        for error in outcome.errors
                    }
                )
            )
            termination_witness = _diagnosis_g3_witness(
                inst,
                final_text=final_text,
                has_tool_calls=any_parser_calls or syn3.tool_call_emitted,
                parser_ambiguous=syn3.consensus == ParseConsensus.AMBIGUOUS,
                parse_errors=parser_errors,
            )
            g3_ok = termination_witness.protocol_termination_verified
            self.session.ledger.decide(
                phase="termination",
                encoding=encoding.value,
                observation=_termination_observation(termination_witness),
                evidence_class=_termination_evidence_class(termination_witness),
                succeeded=g3_ok,
                final_text=final_text[:80],
                **termination_witness.as_dict(),
            )
            if g3_ok:
                traj.branch = fork
                return encoding, settled
        return None

    def complete_discovered_trajectory(
        self, traj: _Trajectory, parser: ResponsePrimitive
    ) -> tuple[ToolResultEncoding, Driver] | None:
        """Validate a synthesized response primitive across G2 and G3.

        Parser parameters come only from G1.  Later paid outputs are validation
        observations: they may confirm the same program (or native calls, which
        the program composes alongside it), but they never mutate the candidate.
        """
        inst = self.diag_inst
        expected = inst.expected_recovery_arguments()
        for encoding in (ToolResultEncoding.TOOL_ROLE, ToolResultEncoding.USER_MESSAGE):
            driver = _driver_from_discovered(traj.config, parser, encoding)
            fork = traj.branch.fork(
                driver,
                reason=f"counterfactual tool-result encoding {encoding.value}",
            )
            assert fork.freeze() == traj.frozen_prefix
            for call in traj.syndrome.discovered_calls:
                if call.name == "record_gamma":
                    fork.add_tool_error(
                        call_id=call.id,
                        name=call.name,
                        error=inst.gamma_error_text(),
                    )
                else:
                    fork.add_tool_result(
                        call_id=call.id,
                        name=call.name,
                        content=inst.result_for(call.name),
                    )

            self.session.check_can_explore()
            _, generation = fork.generate(
                purpose="explore",
                label=f"G2@{encoding.value}",
                reason=(
                    "validating the synthesized response parser against a second "
                    "paid observation while resolving tool-result encoding"
                ),
                offered_tool_names={t.name for t in gauntlet_tools()},
            )
            calls, is_batch, _, errors = _observe_with_driver(generation.response, driver)
            names = {call.name for call in calls}
            values = {call.name: call.arguments == expected.get(call.name) for call in calls}
            g2_ok = (
                not errors
                and is_batch
                and names == set(RECOVERY_TOOLS)
                and len(calls) == len(RECOVERY_TOOLS)
                and all(values.get(name) for name in RECOVERY_TOOLS)
            )
            self.session.ledger.decide(
                phase="result_encoding",
                encoding=encoding.value,
                observation=(
                    "discovered_parser_validated" if g2_ok else "discovered_parser_rejected"
                ),
                parser=parser.model_dump(mode="json"),
                parse_errors=list(errors),
                succeeded=g2_ok,
            )
            if not g2_ok:
                continue

            for call in calls:
                fork.add_tool_result(
                    call_id=call.id,
                    name=call.name,
                    content=inst.recovery_results().get(call.name, {"status": "ok"}),
                )
            self.session.check_can_explore()
            _, generation3 = fork.generate(
                purpose="explore",
                label=f"G3@{encoding.value}",
                reason=("validating synthesized-parser no-call termination with the ack sentinel"),
                offered_tool_names={t.name for t in gauntlet_tools()},
            )
            final_calls, _, final_text, final_errors = _observe_with_driver(
                generation3.response, driver
            )
            termination_witness = _diagnosis_g3_witness(
                inst,
                final_text=final_text,
                has_tool_calls=bool(final_calls),
                parse_errors=tuple(final_errors),
            )
            g3_ok = termination_witness.protocol_termination_verified
            self.session.ledger.decide(
                phase="termination",
                encoding=encoding.value,
                observation=_termination_observation(termination_witness),
                evidence_class=_termination_evidence_class(termination_witness),
                parse_errors=list(final_errors),
                succeeded=g3_ok,
                final_text=final_text[:80],
                **termination_witness.as_dict(),
            )
            if g3_ok:
                traj.branch = fork
                return encoding, driver
        return None

    def _append_initial_diagnostic_results(
        self, branch: Branch, calls: tuple[ToolCall, ...]
    ) -> None:
        inst = self.diag_inst
        for call in calls:
            if call.name == "record_gamma":
                branch.add_tool_error(
                    call_id=call.id,
                    name=call.name,
                    error=inst.gamma_error_text(),
                )
            else:
                branch.add_tool_result(
                    call_id=call.id,
                    name=call.name,
                    content=inst.result_for(call.name),
                )

    def _active_g2_ok(
        self, generation: Generation, driver: Driver
    ) -> tuple[bool, tuple[ToolCall, ...], tuple[str, ...]]:
        expected = self.diag_inst.expected_recovery_arguments()
        calls, is_batch, _, errors = _observe_with_driver(generation.response, driver)
        names = {call.name for call in calls}
        values = {call.name: call.arguments == expected.get(call.name) for call in calls}
        ok = (
            not errors
            and is_batch
            and names == set(RECOVERY_TOOLS)
            and len(calls) == len(RECOVERY_TOOLS)
            and all(values.get(name) for name in RECOVERY_TOOLS)
        )
        return ok, calls, errors

    def _run_active_result_candidate(
        self,
        traj: _Trajectory,
        hypothesis: PartialProtocolHypothesis,
        *,
        label: str,
        reason: str,
    ) -> tuple[Driver, Branch, Generation, bool, tuple[ToolCall, ...]]:
        driver = hypothesis.to_driver()
        fork = traj.branch.fork(driver, reason=reason)
        assert fork.freeze() == traj.frozen_prefix
        self._append_initial_diagnostic_results(fork, traj.syndrome.accepted_calls)
        self.session.check_can_explore()
        _, generation = fork.generate(
            purpose="explore",
            label=label,
            reason=reason,
            offered_tool_names={tool.name for tool in gauntlet_tools()},
        )
        ok, calls, errors = self._active_g2_ok(generation, driver)
        self.session.ledger.decide(
            phase="active-tool-result-validation",
            generation=generation.index,
            candidate=hypothesis.component_fingerprint(ProtocolComponent.TOOL_RESULT),
            observation="exact_recovery_batch" if ok else "result_not_consumed",
            parse_errors=list(errors),
            succeeded=ok,
        )
        return driver, fork, generation, ok, calls

    def complete_oracle_free_trajectory(self, traj: _Trajectory) -> Driver | None:
        """Diagnose result consumption, then run a clean production G2/G3 trace."""
        hypothesis = traj.active_hypothesis
        generation1 = traj.branch.last_generation
        request_version = traj.request_version
        if hypothesis is None or generation1 is None or request_version is None:
            self._active_failure = "oracle-free trajectory lacks a clean G1 hypothesis"
            self.synthesis_report.failure = self._active_failure
            self.synthesis_report.failure_class = "no_working_program_found"
            return None

        survivors = list(result_version_space())
        identifiability = check_identifiability(ProtocolComponent.TOOL_RESULT, survivors)
        reports = dict(self.synthesis_report.identifiability or {})
        reports["tool_result"] = identifiability.as_dict()
        self.synthesis_report.identifiability = reports
        if not identifiability.identifiable:
            self._active_failure = (
                "the current tool-result protocol family is observationally "
                "unidentifiable under the available diagnostic probe family"
            )
            self.synthesis_report.failure = self._active_failure
            self.synthesis_report.failure_class = "observationally_unidentifiable"
            return None

        sample_call = next(iter(traj.syndrome.accepted_calls), None)
        if sample_call is None:
            self._active_failure = "clean G1 produced no call for result diagnostics"
            self.synthesis_report.failure = self._active_failure
            self.synthesis_report.failure_class = "no_working_program_found"
            return None
        diagnostic_probe, plan = build_result_probe(
            survivors,
            request_version=request_version,
            prefix_messages=traj.branch.model_messages,
            tools=gauntlet_tools(),
            call=sample_call,
            seed=self.seed,
            sequence=1,
        )
        diagnostic_branch = self.session.new_diagnostic_branch(diagnostic_probe)
        _, diagnostic_generation = diagnostic_branch.generate(
            purpose="explore",
            label="oracle-free-result-probe-1",
            reason=(
                "counterfactual result representations carry fresh sentinels; only "
                "an exact structured recovery call can select a predicted partition"
            ),
            offered_tool_names={"report"},
        )
        outcome = observe_probe_response(
            diagnostic_probe,
            diagnostic_generation.response,
            candidate_drivers_for_probe((request_version,)),
        )
        if outcome.outcome_key is None:
            plan.record_outcome(outcome)
            self.synthesis_report.probe_plans.append(plan.as_dict())
            self._active_failure = (
                "the current tool-result protocol family is observationally "
                "unidentifiable under the available observations: " + outcome.reason
            )
            self.synthesis_report.failure = self._active_failure
            self.synthesis_report.failure_class = "observationally_unidentifiable"
            return None

        evidence = diagnostic_witness_evidence(
            component=ProtocolComponent.TOOL_RESULT,
            generation=diagnostic_generation,
            plan=plan,
            outcome=outcome,
        )
        plan.record_outcome(outcome, evidence_id=evidence.evidence_id)
        removed = set(plan.hypotheses_removed)
        self.synthesis_report.evidence.eliminate_by_diagnostic_partition(
            ProtocolComponent.TOOL_RESULT,
            removed,
            evidence=evidence,
        )
        self.synthesis_report.probe_plans.append(plan.as_dict())
        remaining = set(plan.hypotheses_remaining)
        survivors = [
            version for version in survivors if version.fingerprint in remaining
        ]
        self.session.ledger.decide(
            phase="oracle-free-result-partition",
            generation=diagnostic_generation.index,
            probe=plan.probe_id,
            observed_outcome=outcome.outcome_key,
            witness_ids=list(outcome.witness_ids),
            removed=sorted(removed),
            remaining=sorted(remaining),
            elimination_basis="exclusive_nonce_bound_structured_result_witness",
        )
        if len(survivors) != 1:
            self._active_failure = (
                "the bounded result diagnostic probe left multiple unresolved hypotheses"
            )
            self.synthesis_report.failure = self._active_failure
            self.synthesis_report.failure_class = "unresolved_under_bounded_diagnostic_budget"
            return None

        selected_version = survivors[0]
        selected = hypothesis.refine(
            ProtocolComponent.TOOL_RESULT,
            result_version_to_program(selected_version),
        )
        self.synthesis_report.record_revision(
            hypothesis,
            selected,
            component=ProtocolComponent.TOOL_RESULT,
            generation_id=diagnostic_generation.index,
            evidence_ids=[evidence.evidence_id],
            reason=(
                "selected the sole result representation whose fresh sentinel "
                "produced the predicted structured recovery witness"
            ),
        )
        self.synthesis_report.version_spaces.append(
            {
                "component": ProtocolComponent.TOOL_RESULT.value,
                "initial_size": len(result_version_space()),
                "survivor_fingerprints": [selected_version.fingerprint],
                "survivors": [selected_version.as_dict()],
                "refinement_basis": "diagnostic_positive_witness_partitions",
            }
        )

        driver, fork, generation2, ok, calls = self._run_active_result_candidate(
            traj,
            selected,
            label=f"oracle-free-clean-G2@{selected_version.fingerprint[:8]}",
            reason=(
                "clean production G2 validation is separate from the multiplexed "
                "result diagnostic"
            ),
        )
        if not ok:
            negative = negative_behavior_evidence(
                component=ProtocolComponent.TOOL_RESULT,
                generation=generation2,
                observation=(
                    "the diagnostic survivor failed the complete clean G2 recovery witness"
                ),
            )
            self.synthesis_report.evidence.record(negative)
            self._active_failure = (
                "the diagnostic tool-result survivor failed clean production G2 validation"
            )
            self.synthesis_report.failure = self._active_failure
            self.synthesis_report.failure_class = "no_working_program_found"
            return None

        result_fact = component_observation_evidence(
            component=ProtocolComponent.TOOL_RESULT,
            generation=generation2,
            observation=(
                "the selected renderer carried all fresh result/error sentinels into "
                "the exact clean G2 recovery batch"
            ),
        )
        response_fact = component_observation_evidence(
            component=ProtocolComponent.RESPONSE,
            generation=generation2,
            observation="the unchanged response program parsed the exact clean G2 batch",
        )
        self.synthesis_report.evidence.record(result_fact)
        self.synthesis_report.evidence.record(response_fact)
        all_calls = tuple(traj.syndrome.accepted_calls) + tuple(calls)
        call_ids = [call.id for call in all_calls]
        present_ids = [call_id for call_id in call_ids if call_id is not None]
        g2_ids = ["OB07", "OB10", "OB11", "OB13", "OB14"]
        if call_ids and len(present_ids) in {0, len(call_ids)}:
            g2_ids.append("OB08")
        if len(set(present_ids)) == len(present_ids):
            g2_ids.append("OB09")
        for witness in obligation_witness_evidence(
            obligation_ids=tuple(g2_ids),
            phase=WitnessPhase.G2,
            generations=(generation1, generation2),
            component_evidence_ids=(result_fact.evidence_id, response_fact.evidence_id),
            observation=(
                "clean G1 plus clean G2 completely witnesses result association, "
                "error recovery, cardinality and observed ID discipline"
            ),
        ):
            self.synthesis_report.evidence.record_witness(witness)

        for call in calls:
            fork.add_tool_result(
                call_id=call.id,
                name=call.name,
                content=self.diag_inst.recovery_results().get(
                    call.name, {"status": "ok"}
                ),
            )
        self.session.check_can_explore()
        _, generation3 = fork.generate(
            purpose="explore",
            label="oracle-free-clean-G3@termination",
            reason="validate the unchanged production program on final no-call termination",
            offered_tool_names={tool.name for tool in gauntlet_tools()},
        )
        final_calls, _, final_text, final_errors = _observe_with_driver(
            generation3.response, driver
        )
        termination_witness = _diagnosis_g3_witness(
            self.diag_inst,
            final_text=final_text,
            has_tool_calls=bool(final_calls),
            parse_errors=tuple(final_errors),
        )
        g3_ok = termination_witness.protocol_termination_verified
        self.session.ledger.decide(
            phase="oracle-free-clean-termination",
            generation=generation3.index,
            observation=_termination_observation(termination_witness),
            evidence_class=_termination_evidence_class(termination_witness),
            parse_errors=list(final_errors),
            succeeded=g3_ok,
            **termination_witness.as_dict(),
        )
        if not g3_ok:
            negative = negative_behavior_evidence(
                component=ProtocolComponent.RESPONSE,
                generation=generation3,
                observation="clean G3 lacked a nonce-bound no-call termination witness",
            )
            self.synthesis_report.evidence.record(negative)
            self._active_failure = "oracle-free synthesized protocol failed clean G3"
            self.synthesis_report.failure = self._active_failure
            self.synthesis_report.failure_class = _termination_evidence_class(
                termination_witness
            )
            return None

        request_fact = component_observation_evidence(
            component=ProtocolComponent.REQUEST,
            generation=generation3,
            observation="the unchanged request program elicited no spurious G3 call",
        )
        termination_fact = component_observation_evidence(
            component=ProtocolComponent.RESPONSE,
            generation=generation3,
            observation="the response program parsed a nonce-bound final text and no call",
        )
        cycle_fact = component_observation_evidence(
            component=ProtocolComponent.TOOL_RESULT,
            generation=generation3,
            observation="the same result renderer completed the second result cycle",
        )
        for fact in (request_fact, termination_fact, cycle_fact):
            self.synthesis_report.evidence.record(fact)
        for witness in obligation_witness_evidence(
            obligation_ids=("OB12", "OB15", "OB16", "OB17"),
            phase=WitnessPhase.G3,
            generations=(generation1, generation2, generation3),
            component_evidence_ids=(
                request_fact.evidence_id,
                termination_fact.evidence_id,
                cycle_fact.evidence_id,
            ),
            observation=(
                "the complete clean three-turn diagnosis trace contains two result "
                "cycles and unambiguous nonce-bound no-call termination"
            ),
        ):
            self.synthesis_report.evidence.record_witness(witness)
        self.synthesis_report.final_hypothesis = selected
        self.synthesis_report.failure = None
        self.synthesis_report.failure_class = None
        traj.branch = fork
        traj.active_hypothesis = selected
        return driver

    def complete_discriminating_trajectory(self, traj: _Trajectory) -> Driver | None:
        """Resolve the result version space through controlled G2 interventions."""
        hypothesis = traj.active_hypothesis
        generation1 = traj.branch.last_generation
        if hypothesis is None or generation1 is None:
            return None
        planned = self.obligation_planner.choose(hypothesis, self.synthesis_report.evidence)
        if planned is None or planned.component != ProtocolComponent.TOOL_RESULT:
            self._active_failure = "obligation planner did not select the result hole"
            self.synthesis_report.failure = self._active_failure
            return None
        self.synthesis_report.record_experiment(planned)
        space = VersionSpace(ProtocolComponent.TOOL_RESULT, result_version_space())
        previous: ResultVersion | None = None
        implicated: str | None = None

        while space.survivors:
            selected_raw, experiment = space.choose(
                previous=previous,
                implicated_parameter=implicated,
                expected_obligation_gain=len(planned.implicated_obligations),
            )
            assert isinstance(selected_raw, ResultVersion)
            selected_version = selected_raw
            self._record_controlled_experiment(experiment)
            selected = hypothesis.refine(
                ProtocolComponent.TOOL_RESULT,
                result_version_to_program(selected_version),
            )
            driver, fork, generation2, ok, calls = self._run_active_result_candidate(
                traj,
                selected,
                label=f"discriminate-G2@{selected_version.fingerprint[:8]}",
                reason=(
                    "testing one lowest-complexity surviving result placement and "
                    "association intervention"
                ),
            )
            if not ok:
                rejection = parse_protocol_rejection(generation2.response)
                if rejection is None or rejection.component != ProtocolComponent.TOOL_RESULT:
                    negative = negative_behavior_evidence(
                        component=ProtocolComponent.TOOL_RESULT,
                        generation=generation2,
                        observation=(
                            "result intervention did not recover G2 and returned no "
                            "deterministic result rejection"
                        ),
                    )
                    self.synthesis_report.evidence.record(negative)
                    self.synthesis_report.version_spaces.append(space.as_dict())
                    self._active_failure = (
                        "discriminating result experiment produced only stochastic "
                        "negative behavior; no SAFE refinement is available"
                    )
                    self.synthesis_report.failure = self._active_failure
                    return None
                self._record_api_rejection(
                    space=space,
                    tested=selected_version,
                    rejection=rejection,
                    generation=generation2,
                )
                previous = selected_version
                implicated = rejection.parameter
                continue

            result_fact = component_observation_evidence(
                component=ProtocolComponent.TOOL_RESULT,
                generation=generation2,
                observation=(
                    "the controlled result intervention carried fresh result/error "
                    "sentinels into the exact G2 recovery batch"
                ),
            )
            response_fact = component_observation_evidence(
                component=ProtocolComponent.RESPONSE,
                generation=generation2,
                observation="the unchanged invented response parser parsed exact G2 calls",
            )
            self.synthesis_report.evidence.record(result_fact)
            self.synthesis_report.evidence.record(response_fact)
            self.synthesis_report.record_revision(
                hypothesis,
                selected,
                component=ProtocolComponent.TOOL_RESULT,
                generation_id=generation2.index,
                evidence_ids=[result_fact.evidence_id],
                reason=(
                    "selected the lowest-complexity result placement/association "
                    "surviving controlled API rejections"
                ),
            )
            all_calls = tuple(traj.syndrome.accepted_calls) + tuple(calls)
            call_ids = [call.id for call in all_calls]
            present_ids = [call_id for call_id in call_ids if call_id is not None]
            g2_ids = ["OB07", "OB10", "OB11", "OB13", "OB14"]
            if call_ids and len(present_ids) in {0, len(call_ids)}:
                g2_ids.append("OB08")
            if len(set(present_ids)) == len(present_ids):
                g2_ids.append("OB09")
            for witness in obligation_witness_evidence(
                obligation_ids=tuple(g2_ids),
                phase=WitnessPhase.G2,
                generations=(generation1, generation2),
                component_evidence_ids=(result_fact.evidence_id, response_fact.evidence_id),
                observation=(
                    "G1 plus accepted G2 fully witnesses result association, error "
                    "recovery, call cardinality, and observed ID discipline"
                ),
            ):
                self.synthesis_report.evidence.record_witness(witness)
            self.synthesis_report.behavioral_deltas.append(
                {
                    "generation_id": generation2.index,
                    "component": "tool_result",
                    "candidate_fingerprint": selected_version.fingerprint,
                    "outcome": "exact_G2_recovery_batch",
                    "implicated_property": None,
                    "accepted_value_revealed": False,
                    "surviving_versions_removed": 0,
                }
            )

            for call in calls:
                fork.add_tool_result(
                    call_id=call.id,
                    name=call.name,
                    content=self.diag_inst.recovery_results().get(call.name, {"status": "ok"}),
                )
            self.session.check_can_explore()
            _, generation3 = fork.generate(
                purpose="explore",
                label="discriminate-G3@termination",
                reason=(
                    "validating the unchanged surviving request/response/result "
                    "program on final no-call termination"
                ),
                offered_tool_names={tool.name for tool in gauntlet_tools()},
            )
            final_calls, _, final_text, final_errors = _observe_with_driver(
                generation3.response, driver
            )
            termination_witness = _diagnosis_g3_witness(
                self.diag_inst,
                final_text=final_text,
                has_tool_calls=bool(final_calls),
                parse_errors=tuple(final_errors),
            )
            g3_ok = termination_witness.protocol_termination_verified
            self.session.ledger.decide(
                phase="active-discriminating-termination",
                generation=generation3.index,
                observation=_termination_observation(termination_witness),
                evidence_class=_termination_evidence_class(termination_witness),
                parse_errors=list(final_errors),
                succeeded=g3_ok,
                **termination_witness.as_dict(),
            )
            if not g3_ok:
                negative = negative_behavior_evidence(
                    component=ProtocolComponent.RESPONSE,
                    generation=generation3,
                    observation="the final turn lacked a nonce-bound no-call termination witness",
                )
                self.synthesis_report.evidence.record(negative)
                self._active_failure = "synthesized discriminating protocol failed G3 termination"
                self.synthesis_report.failure = self._active_failure
                self.synthesis_report.failure_class = _termination_evidence_class(
                    termination_witness
                )
                self.synthesis_report.version_spaces.append(space.as_dict())
                return None

            request_fact = component_observation_evidence(
                component=ProtocolComponent.REQUEST,
                generation=generation3,
                observation="the unchanged request program elicited no spurious G3 call",
            )
            termination_fact = component_observation_evidence(
                component=ProtocolComponent.RESPONSE,
                generation=generation3,
                observation="the response parser emitted nonce-bound final text and no call",
            )
            cycle_fact = component_observation_evidence(
                component=ProtocolComponent.TOOL_RESULT,
                generation=generation3,
                observation="the selected renderer completed the second result cycle",
            )
            for fact in (request_fact, termination_fact, cycle_fact):
                self.synthesis_report.evidence.record(fact)
            for witness in obligation_witness_evidence(
                obligation_ids=("OB12", "OB15", "OB16", "OB17"),
                phase=WitnessPhase.G3,
                generations=(generation1, generation2, generation3),
                component_evidence_ids=(
                    request_fact.evidence_id,
                    termination_fact.evidence_id,
                    cycle_fact.evidence_id,
                ),
                observation=(
                    "the complete diagnosis trajectory contains two result cycles, "
                    "unambiguous nonce-bound no-call termination"
                ),
            ):
                self.synthesis_report.evidence.record_witness(witness)
            self.synthesis_report.version_spaces.append(space.as_dict())
            self.synthesis_report.final_hypothesis = selected
            self.synthesis_report.failure = None
            traj.branch = fork
            traj.active_hypothesis = selected
            return driver

        self.synthesis_report.version_spaces.append(space.as_dict())
        self._active_failure = "deterministic rejections exhausted result version space"
        self.synthesis_report.failure = self._active_failure
        return None

    def complete_active_trajectory(self, traj: _Trajectory) -> Driver | None:
        """Resolve only the result hole, then validate G2/G3 unchanged.

        One lowest-complexity executable baseline is tried.  If it fails, XPT
        recovers literal/field segments locally from a fresh-sentinel example
        (or strict atomic counterexample) and tests only the bounded message
        placement choices.  Only the tool-result component is revised; paid G1
        request/response evidence remains live.  There is no unbounded search.
        """
        hypothesis = traj.active_hypothesis
        if hypothesis is None:
            return None
        generation1 = traj.branch.last_generation
        if generation1 is None:
            self._active_failure = "active result synthesis has no G1 witness generation"
            self.synthesis_report.failure = self._active_failure
            return None
        experiment = self.obligation_planner.choose(hypothesis, self.synthesis_report.evidence)
        if experiment is None or experiment.component != ProtocolComponent.TOOL_RESULT:
            self._active_failure = "obligation planner did not select the result hole"
            self.synthesis_report.failure = self._active_failure
            return None
        self.synthesis_report.record_experiment(experiment)
        self.session.ledger.decide(
            phase="active-synthesis-plan",
            **experiment.as_dict(),
        )

        baseline = ToolResultMessage(
            role="tool",
            segments=[ResultField(field="content")],
            attach_tool_call_id=True,
        )
        baseline_hypothesis = hypothesis.refine(ProtocolComponent.TOOL_RESULT, baseline)
        driver, fork, generation, ok, calls = self._run_active_result_candidate(
            traj,
            baseline_hypothesis,
            label="active-G2@minimal-result",
            reason=(
                "testing the lowest-complexity result renderer for the obligations "
                + ", ".join(experiment.implicated_obligations)
            ),
        )

        selected = baseline_hypothesis
        if ok:
            result_fact = component_observation_evidence(
                component=ProtocolComponent.TOOL_RESULT,
                generation=generation,
                observation=(
                    "minimal result renderer carried all result/error sentinels into "
                    "the exact G2 recovery batch"
                ),
            )
            self.synthesis_report.evidence.record(result_fact)
            self.synthesis_report.record_revision(
                hypothesis,
                selected,
                component=ProtocolComponent.TOOL_RESULT,
                generation_id=generation.index,
                evidence_ids=[result_fact.evidence_id],
                reason="accepted the lowest-complexity result renderer proven by G2",
            )
        else:
            sample_call = next(
                (call for call in traj.syndrome.accepted_calls if call.name == "record_alpha"),
                None,
            )
            if sample_call is None:
                self._active_failure = "G1 has no record_alpha result witness"
                self.synthesis_report.failure = self._active_failure
                return None
            structural = discover_tool_result_program_from_example(
                generation.response,
                result=ToolResult(
                    call_id=sample_call.id,
                    name=sample_call.name,
                    content=self.diag_inst.result_for(sample_call.name),
                ),
                expected_nonce=self.diag_inst.alpha_token,
            )
            result_programs: tuple[ToolResultMessage, ...]
            if structural.ok:
                assert structural.observation is not None
                observation = structural.observation
                result_programs = structural.candidates
            elif structural.found:
                self._active_failure = structural.error or "invalid result example"
                self.synthesis_report.failure = self._active_failure
                return None
            else:
                observation = extract_counterexample(
                    generation.response,
                    expected_component=ProtocolComponent.TOOL_RESULT,
                    expected_nonce=self.diag_inst.alpha_token,
                )
                if observation.ok:
                    try:
                        result_programs = (synthesize_tool_result_program(observation),)
                    except ValueError as exc:
                        self._active_failure = str(exc)
                        self.synthesis_report.failure = self._active_failure
                        return None
                else:
                    result_programs = ()
            if not observation.found:
                self._active_failure = (
                    "result consumption failed without a fresh-sentinel result example "
                    "or reusable atomic constraints"
                )
                self.synthesis_report.failure = self._active_failure
                return None
            if not observation.ok:
                self._active_failure = observation.error or "invalid result counterexample"
                self.synthesis_report.failure = self._active_failure
                return None
            rejected = counterexample_evidence(
                component=ProtocolComponent.TOOL_RESULT,
                generation=generation,
                observation=(
                    "minimal renderer failed result consumption and the endpoint returned "
                    "nonce-bound tool-result constraints"
                ),
                contradiction_class=ContradictionClass.STRUCTURAL,
            )
            self.synthesis_report.evidence.eliminate(
                ProtocolComponent.TOOL_RESULT,
                baseline_hypothesis.component_fingerprint(ProtocolComponent.TOOL_RESULT),
                evidence=rejected,
            )
            constraint_rows = evidence_from_counterexample(observation, generation)
            for row in constraint_rows:
                self.synthesis_report.evidence.record(row)
            selected = hypothesis
            for candidate_index, result_program in enumerate(result_programs, start=1):
                candidate = hypothesis.refine(ProtocolComponent.TOOL_RESULT, result_program)
                driver, fork, generation, ok, calls = self._run_active_result_candidate(
                    traj,
                    candidate,
                    label=f"active-G2@result-{candidate_index}",
                    reason=(
                        "validating a bounded message-placement choice while "
                        "reusing all locally inferred result-template parameters"
                    ),
                )
                if ok:
                    selected = candidate
                    break
                candidate_rejected = negative_behavior_evidence(
                    component=ProtocolComponent.TOOL_RESULT,
                    generation=generation,
                    observation=(
                        "this exact result candidate did not carry the fresh sentinels "
                        "into the G2 recovery batch"
                    ),
                )
                self.synthesis_report.evidence.record(candidate_rejected)
            if selected is hypothesis or not ok:
                self._active_failure = "synthesized tool-result renderer did not satisfy G2"
                self.synthesis_report.failure = self._active_failure
                return None
            result_fact = component_observation_evidence(
                component=ProtocolComponent.TOOL_RESULT,
                generation=generation,
                observation=(
                    "synthesized renderer carried all result/error sentinels into the "
                    "exact G2 recovery batch"
                ),
            )
            self.synthesis_report.evidence.record(result_fact)
            self.synthesis_report.record_revision(
                hypothesis,
                selected,
                component=ProtocolComponent.TOOL_RESULT,
                generation_id=generation.index,
                evidence_ids=[row.evidence_id for row in constraint_rows]
                + [result_fact.evidence_id],
                reason=(
                    "refined only the result renderer; all template parameters came "
                    "from the paid fresh-sentinel example, and request/response "
                    "fingerprints remained unchanged"
                ),
            )

        response_fact = component_observation_evidence(
            component=ProtocolComponent.RESPONSE,
            generation=generation,
            observation="the unchanged response program parsed the exact G2 batch",
        )
        self.synthesis_report.evidence.record(response_fact)
        g2_evidence_ids = (result_fact.evidence_id, response_fact.evidence_id)
        all_calls = tuple(traj.syndrome.accepted_calls) + tuple(calls)
        call_ids = [call.id for call in all_calls]
        present_ids = [call_id for call_id in call_ids if call_id is not None]
        g2_witness_ids = ["OB07", "OB10", "OB11", "OB13", "OB14"]
        if call_ids and len(present_ids) in {0, len(call_ids)}:
            g2_witness_ids.append("OB08")
        if len(set(present_ids)) == len(present_ids):
            g2_witness_ids.append("OB09")
        g2_witnesses = obligation_witness_evidence(
            obligation_ids=tuple(g2_witness_ids),
            phase=WitnessPhase.G2,
            generations=(generation1, generation),
            component_evidence_ids=g2_evidence_ids,
            observation=(
                "G1 plus the exact G2 recovery batch completely witnesses call "
                "cardinality, ID discipline, result association, error consumption, "
                "and recovery for the listed obligations"
            ),
        )
        for witness in g2_witnesses:
            self.synthesis_report.evidence.record_witness(witness)
        for call in calls:
            fork.add_tool_result(
                call_id=call.id,
                name=call.name,
                content=self.diag_inst.recovery_results().get(call.name, {"status": "ok"}),
            )
        self.session.check_can_explore()
        _, generation3 = fork.generate(
            purpose="explore",
            label="active-G3@termination",
            reason=(
                "validating the unchanged synthesized request/response/result program "
                "on no-call termination"
            ),
            offered_tool_names={tool.name for tool in gauntlet_tools()},
        )
        final_calls, _, final_text, final_errors = _observe_with_driver(
            generation3.response, driver
        )
        termination_witness = _diagnosis_g3_witness(
            self.diag_inst,
            final_text=final_text,
            has_tool_calls=bool(final_calls),
            parse_errors=tuple(final_errors),
        )
        g3_ok = termination_witness.protocol_termination_verified
        self.session.ledger.decide(
            phase="active-termination",
            generation=generation3.index,
            observation=_termination_observation(termination_witness),
            evidence_class=_termination_evidence_class(termination_witness),
            parse_errors=list(final_errors),
            succeeded=g3_ok,
            **termination_witness.as_dict(),
        )
        if not g3_ok:
            self._active_failure = "synthesized protocol failed G3 termination"
            self.synthesis_report.failure = self._active_failure
            self.synthesis_report.failure_class = _termination_evidence_class(
                termination_witness
            )
            return None
        request_termination_fact = component_observation_evidence(
            component=ProtocolComponent.REQUEST,
            generation=generation3,
            observation="the unchanged request program elicited no spurious G3 tool call",
        )
        response_termination_fact = component_observation_evidence(
            component=ProtocolComponent.RESPONSE,
            generation=generation3,
            observation="the response program parsed nonce-bound final text with no tool calls",
        )
        result_termination_fact = component_observation_evidence(
            component=ProtocolComponent.TOOL_RESULT,
            generation=generation3,
            observation=(
                "the same result renderer carried recovery results and produced the "
                "nonce-bound no-call acknowledgement"
            ),
        )
        for fact in (
            request_termination_fact,
            response_termination_fact,
            result_termination_fact,
        ):
            self.synthesis_report.evidence.record(fact)
        g3_witnesses = obligation_witness_evidence(
            obligation_ids=("OB12", "OB15", "OB16", "OB17"),
            phase=WitnessPhase.G3,
            generations=(generation1, generation, generation3),
            component_evidence_ids=(
                request_termination_fact.evidence_id,
                response_termination_fact.evidence_id,
                result_termination_fact.evidence_id,
            ),
            observation=(
                "the full three-turn diagnosis trace completed two result cycles, "
                "had no spurious final call, terminated with a fresh sentinel, and parsed "
                "without ambiguity"
            ),
        )
        for witness in g3_witnesses:
            self.synthesis_report.evidence.record_witness(witness)
        self.synthesis_report.final_hypothesis = selected
        self.synthesis_report.failure = None
        traj.branch = fork
        traj.active_hypothesis = selected
        return driver

    # ------------------------------------------------------------------
    # top level
    # ------------------------------------------------------------------

    def _account_cert_generations(self, run: Any, *, driver: Driver, label: str) -> None:
        """Record the actual certification wire without permitting budget overflow."""
        n = run.generations
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
        for interaction in run.wire:
            request = interaction.get("request") or {}
            response = interaction.get("response")
            self.session.ledger.generations.append(
                Generation(
                    index=len(self.session.ledger.generations) + 1,
                    purpose="certify",
                    label=label,
                    branch_id="cert",
                    forked_from=None,
                    prefix_hash=sha(request.get("messages", [])),
                    driver=driver.canonical_dict(),
                    request=request,
                    request_hash=sha(request),
                    response=response,
                    response_hash=sha(response) if response is not None else None,
                    error=interaction.get("error"),
                    latency_ms=float(interaction.get("latency_ms", 0.0)),
                    prompt_chars=len(_json(request)),
                    completion_chars=len(_json(response)) if response is not None else 0,
                    syndrome=(
                        build_syndrome(response).as_dict()
                        if isinstance(response, dict)
                        else None
                    ),
                    selection_reason="independent certification",
                )
            )

    def _diag_gen_count(self) -> int:
        return sum(1 for g in self.session.ledger.generations if g.purpose != "certify")

    def run(self) -> XptResult:
        started = self.session.clock()
        result = XptResult(
            status=UNSUPPORTED,
            ledger=self.session.ledger,
            synthesis_report=self.synthesis_report,
        )
        ambiguous = False
        attempts = 0
        last_failed_obligations: list[str] = []
        last_driver: Driver | None = None
        cert_gens_total = 0

        try:
            for traj in self._candidate_configs():
                attempts += 1
                if traj.active_hypothesis is not None:
                    if traj.oracle_free:
                        driver = self.complete_oracle_free_trajectory(traj)
                    elif traj.discriminating:
                        driver = self.complete_discriminating_trajectory(traj)
                    else:
                        driver = self.complete_active_trajectory(traj)
                    result.equivalent_parsers = ["synthesized:v0.2-response-primitive"]
                    if traj.oracle_free:
                        synthesis_reason = (
                            "nonce-bound normal-wire probes minimized worst-case outcome "
                            "partitions, then the sole survivors passed a separate clean "
                            "G1/G2/G3 production trajectory"
                        )
                    elif traj.discriminating:
                        synthesis_reason = (
                            "a bounded version-space planner designed controlled request "
                            "and result interventions, compared ordinary API/behavioral "
                            "deltas, and validated the lowest-complexity survivor through G3"
                        )
                    else:
                        synthesis_reason = (
                            "a bounded obligation-directed loop refined typed request, "
                            "response, and tool-result holes from reusable black-box "
                            "constraints, then validated the unchanged program through G3"
                        )
                    if driver is None:
                        self.session.ledger.note(
                            "active protocol synthesis failed closed: "
                            + (self._active_failure or "unproven protocol component")
                        )
                        continue
                    self.session.ledger.decide(
                        phase="synthesis",
                        driver=driver.canonical_dict(),
                        configurations_attempted=attempts,
                        reason=synthesis_reason,
                    )
                else:
                    driver = None
                    synthesis_reason = ""
                live = set(traj.syndrome.compatible_parsers)
                if traj.syndrome.consensus == ParseConsensus.AMBIGUOUS:
                    ambiguous = True
                    self.session.ledger.note("ambiguous parse at the first turn; refusing to guess")
                    continue
                if traj.active_hypothesis is None:
                    discovered_parser = traj.syndrome.discovered_parser
                    if discovered_parser is not None:
                        result.equivalent_parsers = [f"discovered:{discovered_parser.op}"]
                        discovered = self.complete_discovered_trajectory(traj, discovered_parser)
                        if discovered is not None:
                            _, driver = discovered
                        synthesis_reason = (
                            "request configuration passed G1; a bounded local pass inferred "
                            "response parser parameters from that paid output; the unchanged "
                            "program then passed G2+G3 before independent certification"
                        )
                    else:
                        if not live:
                            continue
                        result.equivalent_parsers = sorted(p.value for p in live)
                        settled = self.complete_trajectory(traj, live)
                        if settled is not None:
                            encoding, parser = settled
                            driver = _driver_from(traj.config, parser, encoding)
                        synthesis_reason = (
                            "request configuration accepted only after the full stateful "
                            "trajectory (G1+G2+G3); parser is the least-capable survivor of "
                            "the compatible sets intersected across every observed turn; "
                            "tool-result encoding from the counterfactual fork"
                        )

                if driver is None:
                    if traj.active_hypothesis is None:
                        self._mark_trajectory_failed(traj.config)
                        self.session.ledger.note(
                            "stateful trajectory failure retained as negative behavioral "
                            f"evidence for {traj.config.key}; no SAFE elimination"
                        )
                    continue
                if traj.active_hypothesis is None:
                    self.session.ledger.decide(
                        phase="synthesis",
                        driver=driver.canonical_dict(),
                        configurations_attempted=attempts,
                        reason=synthesis_reason,
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

                cert_inst = mint_instance(seed=self.seed + 977, salt="certify", surface_form="B")
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
                        run.errors[0]
                        if run.errors
                        else "infrastructure failure during certification"
                    )
                if run.termination == Termination.CONFIGURATION_FAILED:
                    raise ConfigurationFailed(
                        run.errors[0]
                        if run.errors
                        else "configuration failure during certification"
                    )
                if run.generations > CERTIFICATION_GENERATION_UPPER_BOUND:
                    raise RuntimeError(
                        "certification exceeded its frozen generation upper bound "
                        f"({run.generations} > {CERTIFICATION_GENERATION_UPPER_BOUND})"
                    )
                cert_gens_total += run.generations
                self._account_cert_generations(
                    run, driver=driver, label=f"cert@{traj.config.key}"
                )

                # Obligations + production evaluator must both pass (certify.passed).
                if run.passed:
                    if traj.active_hypothesis is not None:
                        self.synthesis_report.certification = {
                            "passed": True,
                            "authority": "independent_certification",
                            "fresh_instance": True,
                            "production_runtime": True,
                            "generations": run.generations,
                            "mandatory_coverage": run.certificate.mandatory_coverage,
                            "protocol_termination_verified": (
                                run.certificate.status_of("OB16").value == "VERIFIED"
                            ),
                            "exact_response_format_followed": (
                                run.evaluator_result.get("details", {}).get(
                                    "exact_response_format_followed"
                                )
                            ),
                            "certificate": run.certificate.as_dict(),
                            "reason": (
                                "the final unchanged Driver survived all mandatory ABI "
                                "obligations on the independent production-runtime run"
                            ),
                        }
                    result.status = CERTIFIED
                    result.reason = "all mandatory ABI obligations verified on a fresh instance"
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
                        last_failed_obligations = [f"evaluator:{e}" for e in eval_errors[:8]]
                    elif run.evaluator_result.get("interface_ok") is False:
                        last_failed_obligations = ["evaluator:interface_ok=false"]
                result.certificate = run.certificate
                if traj.active_hypothesis is None:
                    self._mark_trajectory_failed(traj.config)
                    self.session.ledger.note(
                        "independent certification failure retained as negative behavioral "
                        f"evidence for {traj.config.key}: {last_failed_obligations}"
                    )
                else:
                    self.synthesis_report.failure = (
                        "independent certification rejected the synthesized program: "
                        + repr(last_failed_obligations)
                    )
                    self.synthesis_report.failure_class = (
                        "independent_certification_rejected"
                    )
                    self.synthesis_report.certification = {
                        "passed": False,
                        "authority": "independent_certification",
                        "fresh_instance": True,
                        "production_runtime": True,
                        "generations": run.generations,
                        "failed_obligations": list(last_failed_obligations),
                        "certificate": run.certificate.as_dict(),
                    }
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
        if self._active_failure:
            result.reason = "active protocol synthesis failed closed: " + self._active_failure
        elif last_failed_obligations:
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
