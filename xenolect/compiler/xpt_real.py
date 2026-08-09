"""Production bridge: real endpoint -> XPT -> verified .mdriver."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xenolect.driver.ir import DRIVER_GRAMMAR_VERSION, driver_grammar_size
from xenolect.driver.serialize import driver_hash, save_driver
from xenolect.endpoints.discovery import DiscoveredEndpoint, discover_openai_endpoint
from xenolect.endpoints.http import OpenAICompatClient
from xenolect.xpt.planner import load_compiled_program
from xenolect.xpt.runtime import (
    BUDGET_EXHAUSTED,
    CERTIFIED,
    CONFIGURATION_FAILED,
    ENDPOINT_TOO_SLOW,
    INFRASTRUCTURE_FAILED,
    XptResult,
    xpt_compile,
)
from xenolect.xpt.session import Budget


@dataclass
class RealCompileReport:
    base_url: str
    model: str
    status: str
    reason: str
    elapsed_s: float
    discovery_s: float
    xpt: XptResult
    output_driver: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "base_url": self.base_url,
            "model": self.model,
            "status": self.status,
            "reason": self.reason,
            "elapsed_s": self.elapsed_s,
            "discovery_s": self.discovery_s,
            "output_driver": self.output_driver,
            "compiler": {
                "mode": "bounded_active_discriminating_protocol_synthesis",
                "driver_grammar_version": DRIVER_GRAMMAR_VERSION,
                "driver_grammar_size": driver_grammar_size(),
                "legacy_seed_frontier_size": driver_grammar_size(),
                "online_frontier_size": driver_grammar_size(),
                "parameterized_protocol_ir": True,
                "bounded_response_parser_synthesis": True,
                "typed_partial_hypotheses": True,
                "reusable_component_evidence": True,
                "bounded_request_synthesis": True,
                "bounded_tool_result_synthesis": True,
                "structural_example_inference": True,
                "explicit_protocol_version_spaces": True,
                "controlled_protocol_interventions": True,
                "behavioral_delta_analysis": True,
                "property_local_api_rejections": True,
                "target_protocol_required": False,
                "provider_or_model_identity_used": False,
                "counterexample_constraints": "nonce_bound_atomic_equalities",
                "arbitrary_protocol_synthesis": False,
                "arbitrary_state_machine_synthesis": False,
            },
            "xpt": self.xpt.as_dict(),
        }
        if self.xpt.driver is not None:
            payload["driver_hash"] = driver_hash(self.xpt.driver)
        return payload


def compile_real_endpoint(
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    out: str | Path | None = "compiled.mdriver",
    report_path: str | Path | None = "compile.report.json",
    deadline_s: float = 300.0,
    max_generations: int = 12,
    seed: int = 1,
    request_timeout_s: float = 120.0,
) -> RealCompileReport:
    """Compile a real endpoint under one wall-clock budget.

    Discovery, diagnosis and certification share the same deadline.  A driver is
    written only after XPT returns ``CERTIFIED``.
    """
    if deadline_s <= 0:
        raise ValueError("deadline_s must be positive")
    if max_generations < 3:
        raise ValueError("max_generations must leave room for 3-step certification")

    started = time.perf_counter()
    if base_url and model and model != "unknown":
        # Explicit endpoint+model is already unambiguous; do not require /models.
        discovered = DiscoveredEndpoint(base_url.rstrip("/"), (model,))
    else:
        discovered = discover_openai_endpoint(
            base_url=base_url,
            api_key=api_key,
            timeout=min(1.5, max(0.1, deadline_s)),
        )
    discovery_s = time.perf_counter() - started
    remaining = deadline_s - discovery_s
    if remaining <= 0:
        dummy = XptResult(status=ENDPOINT_TOO_SLOW, reason="deadline exhausted during discovery")
        result = RealCompileReport(
            base_url=discovered.base_url,
            model=model or "unknown",
            status=ENDPOINT_TOO_SLOW,
            reason=dummy.reason,
            elapsed_s=time.perf_counter() - started,
            discovery_s=discovery_s,
            xpt=dummy,
        )
        if report_path is not None:
            report = Path(report_path)
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
        return result

    if model and model != "unknown":
        resolved_model = model
    elif len(discovered.models) == 1:
        resolved_model = discovered.models[0]
    elif not discovered.models:
        raise RuntimeError("endpoint advertised no model ids; pass --model explicitly")
    else:
        raise RuntimeError(
            "endpoint advertises multiple models; pass --model explicitly: "
            + ", ".join(discovered.models)
        )

    remaining_after_model = deadline_s - (time.perf_counter() - started)
    if remaining_after_model <= 0:
        dummy = XptResult(status=ENDPOINT_TOO_SLOW, reason="deadline exhausted during discovery")
        result = RealCompileReport(
            base_url=discovered.base_url,
            model=resolved_model,
            status=ENDPOINT_TOO_SLOW,
            reason=dummy.reason,
            elapsed_s=time.perf_counter() - started,
            discovery_s=discovery_s,
            xpt=dummy,
        )
        if report_path is not None:
            report = Path(report_path)
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
        return result

    # One wall-clock epoch for HTTP client + XPT session (absolute deadline_s from
    # compile start). Avoids dual remaining_s that mislabel client timeouts as protocol.
    client = OpenAICompatClient(
        base_url=discovered.base_url,
        api_key=api_key,
        model=resolved_model,
        timeout=min(request_timeout_s, remaining_after_model),
        temperature=0.0,
        max_retries=1,
        deadline_s=deadline_s,
        started_at=started,
    )

    program = load_compiled_program()
    try:
        xpt_result = xpt_compile(
            client,
            program,
            budget=Budget(
                max_generations=max_generations,
                certification_reserve=3,
                deadline_s=deadline_s,
            ),
            seed=seed,
            started_at=started,
        )
    finally:
        client.close()

    elapsed = time.perf_counter() - started
    # Do not clobber terminal non-protocol statuses (infra/config/budget) when the
    # wall clock also expired — those labels are more specific for install/ops.
    terminal_keep = {
        CERTIFIED,
        INFRASTRUCTURE_FAILED,
        CONFIGURATION_FAILED,
        BUDGET_EXHAUSTED,
    }
    if elapsed >= deadline_s and xpt_result.status not in terminal_keep:
        prev = xpt_result.status
        xpt_result.status = ENDPOINT_TOO_SLOW
        xpt_result.reason = (
            f"{deadline_s:g}-second compiler deadline reached before certification"
            + (f" (was {prev})" if prev else "")
        )

    output_driver: str | None = None
    if xpt_result.status == CERTIFIED and xpt_result.driver is not None and out is not None:
        output = Path(out)
        output.parent.mkdir(parents=True, exist_ok=True)
        save_driver(xpt_result.driver, output)
        output_driver = str(output)

    result = RealCompileReport(
        base_url=discovered.base_url,
        model=resolved_model,
        status=xpt_result.status,
        reason=xpt_result.reason,
        elapsed_s=elapsed,
        discovery_s=discovery_s,
        xpt=xpt_result,
        output_driver=output_driver,
    )
    if report_path is not None:
        report = Path(report_path)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    return result
