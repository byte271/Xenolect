"""Bounded candidate-only control for the diagnostic-probe ablation.

This is an evaluation control, not the default compiler. It evaluates the same
fixed 33 request and three result versions one complete candidate at a time,
never converts generic rejection or silence into property-local evidence, and
uses the same production runtime certification boundary as XPT.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from statistics import median
from typing import Any

from xenolect.abi.events import AssistantText, AssistantToolCall, ToolCall, ToolCallBatch
from xenolect.driver.parse import parse_model_response_full
from xenolect.xpt.certify import certify
from xenolect.xpt.discrimination import (
    request_version_space,
    request_version_to_hypothesis,
    result_version_space,
    result_version_to_program,
)
from xenolect.xpt.frontier import CERTIFICATION_GENERATION_UPPER_BOUND
from xenolect.xpt.gauntlet import RECOVERY_TOOLS, gauntlet_tools, mint_instance, render_user_turn
from xenolect.xpt.hypothesis import ProtocolComponent
from xenolect.xpt.session import Budget, BudgetExhausted, DeadlineExceeded, XptSession


@dataclass(frozen=True)
class CandidateOnlyRun:
    certification_success: bool
    diagnosis_generations: int
    certification_generations: int
    request_candidates_tested: int
    result_candidates_tested: int
    unresolved: bool
    reason: str

    @property
    def total_generations(self) -> int:
        return self.diagnosis_generations + self.certification_generations

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": "candidate_only",
            "request_space": 33,
            "result_space": 3,
            "certification_success": self.certification_success,
            "diagnosis_generations": self.diagnosis_generations,
            "certification_generations": self.certification_generations,
            "total_generations": self.total_generations,
            "request_candidates_tested": self.request_candidates_tested,
            "result_candidates_tested": self.result_candidates_tested,
            "unresolved": self.unresolved,
            "reason": self.reason,
        }


def _observe(raw: Any, driver: Any) -> tuple[tuple[ToolCall, ...], bool, str, tuple[str, ...]]:
    if not isinstance(raw, dict):
        return (), False, "", ("response is not an object",)
    parsed = parse_model_response_full(raw, driver)
    calls: list[ToolCall] = []
    batch = False
    text: list[str] = []
    for event in parsed.events:
        if isinstance(event, AssistantToolCall):
            calls.append(event.call)
            if event.content:
                text.append(event.content)
        elif isinstance(event, ToolCallBatch):
            calls.extend(event.calls)
            batch = True
            if event.content:
                text.append(event.content)
        elif isinstance(event, AssistantText):
            text.append(event.content)
    return tuple(calls), batch, "".join(text), tuple(parsed.errors)


def _g1_ok(
    raw: Any, driver: Any, expected: dict[str, dict[str, Any]]
) -> tuple[bool, tuple[ToolCall, ...]]:
    calls, batch, _, errors = _observe(raw, driver)
    values = {call.name: call.arguments for call in calls}
    return (
        not errors
        and batch
        and len(calls) == len(expected)
        and set(values) == set(expected)
        and all(values[name] == expected[name] for name in expected),
        calls,
    )


def _g2_ok(
    raw: Any, driver: Any, expected: dict[str, dict[str, Any]]
) -> tuple[bool, tuple[ToolCall, ...]]:
    calls, batch, _, errors = _observe(raw, driver)
    values = {call.name: call.arguments for call in calls}
    return (
        not errors
        and batch
        and len(calls) == len(RECOVERY_TOOLS)
        and set(values) == set(RECOVERY_TOOLS)
        and all(values[name] == expected[name] for name in RECOVERY_TOOLS),
        calls,
    )


def run_candidate_only_ablation(
    client: Any,
    *,
    seed: int,
    budget: Budget | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> CandidateOnlyRun:
    """Evaluate exact complete candidates until success or the shared budget ends."""
    selected_budget = budget or Budget()
    session = XptSession(client, budget=selected_budget, clock=clock)
    instance = mint_instance(seed=seed, salt="diagnose", surface_form="A")
    tools = gauntlet_tools()
    expected_g1 = instance.expected_batch_arguments()
    request_tests = 0
    result_tests = 0

    try:
        for request_version in request_version_space():
            request_tests += 1
            request_hypothesis = request_version_to_hypothesis(request_version)
            provisional = request_hypothesis.refine(
                ProtocolComponent.TOOL_RESULT,
                result_version_to_program(result_version_space()[0]),
            )
            request_driver = provisional.to_driver()
            branch = session.new_branch(request_driver)
            branch.add_user(render_user_turn(instance), tools)
            _, generation1 = branch.generate(
                purpose="explore",
                label=f"candidate-only-G1@{request_version.fingerprint[:8]}",
                reason="candidate-only ablation; no property-local elimination",
                offered_tool_names={tool.name for tool in tools},
            )
            ok1, initial_calls = _g1_ok(
                generation1.response, request_driver, expected_g1
            )
            if not ok1:
                continue

            for result_version in result_version_space():
                result_tests += 1
                candidate = request_hypothesis.refine(
                    ProtocolComponent.TOOL_RESULT,
                    result_version_to_program(result_version),
                )
                driver = candidate.to_driver()
                fork = branch.fork(
                    driver,
                    reason="candidate-only exact result representation",
                )
                for call in initial_calls:
                    if call.name == "record_gamma":
                        fork.add_tool_error(
                            call_id=call.id,
                            name=call.name,
                            error=instance.gamma_error_text(),
                        )
                    else:
                        fork.add_tool_result(
                            call_id=call.id,
                            name=call.name,
                            content=instance.result_for(call.name),
                        )
                _, generation2 = fork.generate(
                    purpose="explore",
                    label=f"candidate-only-G2@{result_version.fingerprint[:8]}",
                    reason="candidate-only exact result representation",
                    offered_tool_names={tool.name for tool in tools},
                )
                ok2, recovery_calls = _g2_ok(
                    generation2.response,
                    driver,
                    instance.expected_recovery_arguments(),
                )
                if not ok2:
                    continue
                for call in recovery_calls:
                    fork.add_tool_result(
                        call_id=call.id,
                        name=call.name,
                        content=instance.recovery_results().get(
                            call.name, {"status": "ok"}
                        ),
                    )
                _, generation3 = fork.generate(
                    purpose="explore",
                    label="candidate-only-G3@termination",
                    reason="candidate-only clean production termination",
                    offered_tool_names={tool.name for tool in tools},
                )
                final_calls, _, final_text, final_errors = _observe(
                    generation3.response, driver
                )
                if final_errors or final_calls or final_text.strip() != instance.ack_value:
                    continue

                session.check_can_certify(
                    required_generations=CERTIFICATION_GENERATION_UPPER_BOUND
                )
                cert_instance = mint_instance(
                    seed=seed + 977,
                    salt="certify",
                    surface_form="B",
                )
                certification = certify(
                    driver,
                    client,
                    cert_instance,
                    max_cycles=CERTIFICATION_GENERATION_UPPER_BOUND - 1,
                )
                remaining_after_cert = session.remaining_s
                if remaining_after_cert is not None and remaining_after_cert <= 0:
                    raise DeadlineExceeded(
                        "compiler wall-clock deadline reached during certification"
                    )
                return CandidateOnlyRun(
                    certification_success=certification.passed,
                    diagnosis_generations=session.ledger.generation_count,
                    certification_generations=certification.generations,
                    request_candidates_tested=request_tests,
                    result_candidates_tested=result_tests,
                    unresolved=not certification.passed,
                    reason=(
                        "independent certification passed"
                        if certification.passed
                        else "candidate passed diagnosis but failed certification"
                    ),
                )
    except (BudgetExhausted, DeadlineExceeded) as exc:
        return CandidateOnlyRun(
            certification_success=False,
            diagnosis_generations=session.ledger.generation_count,
            certification_generations=0,
            request_candidates_tested=request_tests,
            result_candidates_tested=result_tests,
            unresolved=True,
            reason=str(exc),
        )

    return CandidateOnlyRun(
        certification_success=False,
        diagnosis_generations=session.ledger.generation_count,
        certification_generations=0,
        request_candidates_tested=request_tests,
        result_candidates_tested=result_tests,
        unresolved=True,
        reason="no complete candidate produced a certified trajectory",
    )


def summarize_ablation(
    candidate_only: list[CandidateOnlyRun],
    diagnostic: list[Any],
) -> dict[str, Any]:
    """Produce the fixed metrics required by the v0.4 research gate."""

    def row(name: str, successes: list[bool], generations: list[int]) -> dict[str, Any]:
        return {
            "strategy": name,
            "cases": len(successes),
            "certification_successes": sum(successes),
            "diagnosis_generations": list(generations),
            "worst_case_generations": max(generations, default=0),
            "median_generations": median(generations) if generations else 0,
            "unresolved_or_ambiguous": len(successes) - sum(successes),
        }

    return {
        "candidate_only": row(
            "candidate_only",
            [run.certification_success for run in candidate_only],
            [run.diagnosis_generations for run in candidate_only],
        ),
        "diagnostic_probe_synthesis": row(
            "diagnostic_probe_synthesis",
            [getattr(run, "status", None) == "CERTIFIED" for run in diagnostic],
            [int(getattr(run, "diagnosis_generations", 0)) for run in diagnostic],
        ),
        "same_request_space": 33,
        "same_result_space": 3,
        "same_default_budget": {
            "total_generations": 12,
            "certification_reserve": 3,
        },
    }
