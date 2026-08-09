"""Product install orchestration for verified Xenolect drivers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xenolect.abi import ABI_VERSION
from xenolect.driver.ir import DRIVER_GRAMMAR_VERSION, driver_grammar_size
from xenolect.endpoints.discovery import DiscoveredEndpoint, inspect_openai_endpoint
from xenolect.storage.registry import DriverRegistry, InstalledDriver, RegistryError, export_driver

if TYPE_CHECKING:
    from xenolect.compiler.xpt_real import RealCompileReport

READY = "READY"


@dataclass(frozen=True)
class ResolvedTarget:
    base_url: str
    model: str
    model_fingerprint: str | None = None

    @classmethod
    def from_discovery(cls, endpoint: DiscoveredEndpoint, model: str) -> "ResolvedTarget":
        return cls(
            base_url=endpoint.base_url.rstrip("/"),
            model=model,
            model_fingerprint=endpoint.fingerprint_for(model),
        )


@dataclass
class InstallReport:
    base_url: str
    model: str
    status: str
    source: str
    reason: str
    elapsed_s: float
    generations: int
    installed: InstalledDriver | None = None
    compile: "RealCompileReport | None" = None
    report_path: Path | None = None
    exported_to: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "status": self.status,
            "source": self.source,
            "reason": self.reason,
            "elapsed_s": self.elapsed_s,
            "generations": self.generations,
            "driver_hash": self.installed.driver_hash if self.installed else None,
            "driver_path": str(self.installed.driver_path) if self.installed else None,
            "report_path": str(self.report_path) if self.report_path else None,
            "exported_to": str(self.exported_to) if self.exported_to else None,
            "compile": self.compile.as_dict() if self.compile else None,
        }


def resolve_explicit_target(
    *,
    base_url: str,
    model: str,
    api_key: str | None = None,
    timeout: float = 1.5,
) -> ResolvedTarget:
    """Resolve an explicit pair; /models fingerprinting is opportunistic only."""
    fingerprint: str | None = None
    try:
        endpoint = inspect_openai_endpoint(base_url=base_url, api_key=api_key, timeout=timeout)
        fingerprint = endpoint.fingerprint_for(model)
    except Exception:  # explicit endpoint/model remains usable even without /models
        pass
    return ResolvedTarget(base_url.rstrip("/"), model, fingerprint)


def install_target(
    target: ResolvedTarget,
    *,
    api_key: str | None = None,
    deadline_s: float = 300.0,
    max_generations: int = 12,
    seed: int = 1,
    request_timeout_s: float = 120.0,
    home: str | Path | None = None,
    force: bool = False,
    export_to: str | Path | None = None,
) -> InstallReport:
    """Reuse or compile one already-resolved endpoint/model target."""
    if deadline_s <= 0:
        raise ValueError("deadline_s must be positive")
    started = time.perf_counter()
    registry = DriverRegistry(home)

    if registry.is_banned(target.base_url, target.model):
        raise RegistryError(
            f"model {target.model!r} is banned in Xenolect; run `xenolect ban` to restore it"
        )

    if not force:
        try:
            cached = registry.lookup(
                target.base_url,
                target.model,
                ABI_VERSION,
                verify=True,
                expected_model_fingerprint=target.model_fingerprint,
            )
        except RegistryError:
            cached = None
        if (
            cached is not None
            and cached.model_fingerprint is not None
            and target.model_fingerprint is None
        ):
            cached = None
        if cached is not None:
            exported = export_driver(cached, export_to) if export_to else None
            return InstallReport(
                base_url=target.base_url,
                model=target.model,
                status=READY,
                source="cache",
                reason="verified driver already installed",
                elapsed_s=time.perf_counter() - started,
                generations=0,
                installed=cached,
                exported_to=exported,
            )

    # XPT is intentionally imported only after the cache path fails. A second
    # install of an already-verified model does not load the compiler at all.
    from xenolect.compiler.xpt_real import compile_real_endpoint

    compiled = compile_real_endpoint(
        base_url=target.base_url,
        model=target.model,
        api_key=api_key,
        out=None,
        report_path=None,
        deadline_s=deadline_s,
        max_generations=max_generations,
        seed=seed,
        request_timeout_s=request_timeout_s,
    )
    if compiled.status != "CERTIFIED" or compiled.xpt.driver is None:
        return InstallReport(
            base_url=target.base_url,
            model=target.model,
            status=compiled.status,
            source="compile",
            reason=compiled.reason,
            elapsed_s=time.perf_counter() - started,
            generations=compiled.xpt.total_generations,
            compile=compiled,
        )

    installed = registry.install(
        base_url=target.base_url,
        model=target.model,
        driver=compiled.xpt.driver,
        compiler=f"xpt-driver-grammar-v{DRIVER_GRAMMAR_VERSION}",
        generations=compiled.xpt.total_generations,
        compile_elapsed_s=compiled.elapsed_s,
        metadata={
            "seed": seed,
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
            "arbitrary_protocol_synthesis": False,
            "arbitrary_state_machine_synthesis": False,
        },
        model_fingerprint=target.model_fingerprint,
    )
    exported = export_driver(installed, export_to) if export_to else None
    result = InstallReport(
        base_url=target.base_url,
        model=target.model,
        status=READY,
        source="compiled",
        reason="driver certified and installed",
        elapsed_s=time.perf_counter() - started,
        generations=compiled.xpt.total_generations,
        installed=installed,
        compile=compiled,
        exported_to=exported,
    )
    result.report_path = registry.write_report(installed, result.as_dict())
    return result


def install_real_endpoint(
    *,
    base_url: str,
    model: str,
    api_key: str | None = None,
    deadline_s: float = 300.0,
    max_generations: int = 12,
    seed: int = 1,
    request_timeout_s: float = 120.0,
    home: str | Path | None = None,
    force: bool = False,
    export_to: str | Path | None = None,
) -> InstallReport:
    """Non-interactive explicit install API."""
    target = resolve_explicit_target(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=min(1.5, max(0.1, deadline_s)),
    )
    return install_target(
        target,
        api_key=api_key,
        deadline_s=deadline_s,
        max_generations=max_generations,
        seed=seed,
        request_timeout_s=request_timeout_s,
        home=home,
        force=force,
        export_to=export_to,
    )
