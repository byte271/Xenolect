from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import xenolect.cli.main as cli_main
import xenolect.compiler.xpt_real as xpt_real_mod
from xenolect.compiler.install import (
    READY,
    InstallReport,
    ResolvedTarget,
    install_target,
)
from xenolect.compiler.xpt_real import RealCompileReport
from xenolect.driver.ir import identity_driver
from xenolect.endpoints.discovery import DiscoveredEndpoint
from xenolect.runtime import DriverNotInstalledError, open_runtime
from xenolect.storage.registry import DriverRegistry, RegistryError
from xenolect.xpt.runtime import (
    BUDGET_EXHAUSTED,
    CERTIFIED,
    CONFIGURATION_FAILED,
    ENDPOINT_TOO_SLOW,
    INFRASTRUCTURE_FAILED,
    UNSUPPORTED,
    XptResult,
)
from xenolect.xpt.session import Generation, Ledger


def _compiled(base_url: str, model: str) -> RealCompileReport:
    xpt = XptResult(status=CERTIFIED, driver=identity_driver(), reason="ok")
    return RealCompileReport(
        base_url=base_url,
        model=model,
        status=CERTIFIED,
        reason="ok",
        elapsed_s=1.0,
        discovery_s=0.0,
        xpt=xpt,
    )


def test_install_then_cache_is_zero_generation(tmp_path: Path, monkeypatch) -> None:
    target = ResolvedTarget("http://127.0.0.1:11434/v1", "m", "fp1")
    calls = []

    def fake_compile(**kwargs):
        calls.append(kwargs)
        return _compiled(target.base_url, target.model)

    monkeypatch.setattr(xpt_real_mod, "compile_real_endpoint", fake_compile)
    first = install_target(target, home=tmp_path)
    second = install_target(target, home=tmp_path)

    assert first.status == READY and first.source == "compiled"
    assert second.status == READY and second.source == "cache"
    assert second.generations == 0
    assert len(calls) == 1
    assert first.report_path is not None and first.report_path.is_file()


def test_changed_model_fingerprint_recompiles(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_compile(**kwargs):
        calls.append(kwargs)
        return _compiled(kwargs["base_url"], kwargs["model"])

    monkeypatch.setattr(xpt_real_mod, "compile_real_endpoint", fake_compile)
    install_target(ResolvedTarget("http://127.0.0.1:11434/v1", "m", "fp1"), home=tmp_path)
    result = install_target(ResolvedTarget("http://127.0.0.1:11434/v1", "m", "fp2"), home=tmp_path)
    assert result.source == "compiled"
    assert len(calls) == 2


def test_runtime_requires_installed_driver(tmp_path: Path) -> None:
    with pytest.raises(DriverNotInstalledError):
        open_runtime(base_url="http://127.0.0.1:11434/v1", model="missing", home=tmp_path)


def test_stateful_runtime_default_matches_certified_execution_profile(tmp_path: Path) -> None:
    registry = DriverRegistry(tmp_path)
    base_url = "http://127.0.0.1:18473/v1"
    registry.install(base_url=base_url, model="m", driver=identity_driver())

    default_runtime = open_runtime(base_url=base_url, model="m", home=tmp_path)
    explicit_runtime = open_runtime(
        base_url=base_url,
        model="m",
        home=tmp_path,
        temperature=None,
    )
    try:
        assert default_runtime.client.generation_config() == {"temperature": 0.0}
        assert explicit_runtime.client.generation_config() == {}
    finally:
        default_runtime.client.close()
        explicit_runtime.client.close()


def test_registry_detects_driver_tamper(tmp_path: Path) -> None:
    registry = DriverRegistry(tmp_path)
    item = registry.install(base_url="http://x/v1", model="m", driver=identity_driver())
    item.driver_path.write_text('{"tool_encoding":"xml_json"}', encoding="utf-8")
    with pytest.raises(RegistryError):
        registry.lookup("http://x/v1", "m", verify=True)


def test_diagnostic_reports_are_bounded_per_binding(tmp_path):
    from xenolect.storage.registry import MAX_REPORTS_PER_BINDING

    reg = DriverRegistry(tmp_path)
    installed = reg.install(base_url="http://up/v1", model="m", driver=identity_driver())
    for i in range(MAX_REPORTS_PER_BINDING + 7):
        reg.write_report(installed, {"i": i})
    reports = list(reg.reports_dir.glob(f"{installed.binding_id}-*.json"))
    assert len(reports) == MAX_REPORTS_PER_BINDING


def test_registry_persists_certified_execution_profile(tmp_path: Path) -> None:
    reg = DriverRegistry(tmp_path)
    installed = reg.install(base_url="http://up/v1", model="m", driver=identity_driver())

    profile = installed.certified_execution_profile
    assert profile.request_defaults == {"temperature": 0.0}
    assert profile.as_dict()["all_sampling_settings_certified"] is False

    raw = json.loads(reg.path.read_text(encoding="utf-8"))
    assert raw["bindings"][installed.binding_id]["certified_execution_profile"] == (
        profile.as_dict()
    )


def test_compile_report_separates_supported_capability_from_used_path() -> None:
    report = _compiled("http://up/v1", "m").as_dict()

    assert report["compiler"]["mode"] == "bounded_obligation_directed_cegis"
    assert report["compiler"]["execution"]["oracle_free_probe_count"] == 0
    assert (
        report["compiler"]["supported_capabilities"]["oracle_free_diagnostic_probes"]
        is True
    )


def test_v04_registry_without_execution_profile_gets_defined_safe_default(
    tmp_path: Path,
) -> None:
    reg = DriverRegistry(tmp_path)
    installed = reg.install(base_url="http://up/v1", model="m", driver=identity_driver())
    raw = json.loads(reg.path.read_text(encoding="utf-8"))
    del raw["bindings"][installed.binding_id]["certified_execution_profile"]
    reg.path.write_text(json.dumps(raw), encoding="utf-8")

    migrated = reg.lookup("http://up/v1", "m")
    assert migrated is not None
    assert migrated.certified_execution_profile.provenance == "legacy_v0.4_registry"
    assert migrated.certified_execution_profile.request_defaults == {"temperature": 0.0}


def test_failed_compile_persists_redacted_report_with_bounded_ledger(
    tmp_path: Path,
    monkeypatch,
) -> None:
    secret = "test-secret-credential"
    generation = Generation(
        index=1,
        purpose="explore",
        label="G1@candidate",
        branch_id="b1",
        forked_from=None,
        prefix_hash="prefix",
        driver=identity_driver().canonical_dict(),
        request={"messages": [{"role": "user", "content": "probe"}]},
        request_hash="request-hash",
        response={
            "error": {
                "message": f"rejected {secret}",
                "url": "http://wire-user:wire-password@up/v1?token=wire-token",
            }
        },
        response_hash="response-hash",
        error=None,
        latency_ms=1.0,
        prompt_chars=5,
        completion_chars=5,
    )
    xpt = XptResult(
        status=BUDGET_EXHAUSTED,
        reason=f"budget ended after {secret}",
        diagnosis_generations=1,
        ledger=Ledger(generations=[generation]),
    )

    def fake_compile(**kwargs):
        return RealCompileReport(
            base_url=kwargs["base_url"],
            model=kwargs["model"],
            status=BUDGET_EXHAUSTED,
            reason=xpt.reason,
            elapsed_s=1.0,
            discovery_s=0.0,
            xpt=xpt,
        )

    monkeypatch.setattr(xpt_real_mod, "compile_real_endpoint", fake_compile)
    result = install_target(
        ResolvedTarget("http://user:password@up/v1?api_key=query-secret", "m"),
        api_key=secret,
        home=tmp_path,
    )

    assert result.status == BUDGET_EXHAUSTED
    assert result.installed is None
    assert result.report_path is not None and result.report_path.is_file()
    payload_text = result.report_path.read_text(encoding="utf-8")
    assert secret not in payload_text
    assert "password" not in payload_text
    assert "wire-token" not in payload_text
    payload = json.loads(payload_text)
    assert payload["status"] == BUDGET_EXHAUSTED
    assert payload["compile"]["xpt"]["failure_class"] == "budget_exhaustion"
    assert payload["compile"]["xpt"]["ledger"]["generation_count"] == 1
    assert payload["report_path"] == str(result.report_path)


def test_cli_displays_failure_report_path(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "reports" / "failure.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("{}", encoding="utf-8")
    endpoint = DiscoveredEndpoint("http://up/v1", ("m",))
    failed = InstallReport(
        base_url=endpoint.base_url,
        model="m",
        status=BUDGET_EXHAUSTED,
        source="compile",
        reason="safe exploration budget ended",
        elapsed_s=1.0,
        generations=9,
        report_path=report_path,
    )
    monkeypatch.setattr(cli_main, "_environment_check", lambda verbose: tmp_path)
    monkeypatch.setattr(
        cli_main,
        "_interactive_discovery",
        lambda **kwargs: cli_main.ModelChoice(endpoint=endpoint, model="m"),
    )
    import xenolect.compiler.install as install_module

    monkeypatch.setattr(install_module, "install_target", lambda *args, **kwargs: failed)

    result = CliRunner().invoke(cli_main.app, ["install"])
    assert result.exit_code == 3
    assert "Diagnostic report:" in result.stdout
    # Rich may wrap a long absolute path between any two filename characters.
    assert report_path.name in "".join(result.stdout.split())


@pytest.mark.parametrize(
    ("status", "failed_obligations", "failure_class"),
    [
        (ENDPOINT_TOO_SLOW, [], "endpoint_too_slow"),
        (INFRASTRUCTURE_FAILED, [], "infrastructure_failure"),
        (CONFIGURATION_FAILED, [], "endpoint_configuration_failure"),
        (UNSUPPORTED, [], "unsupported_no_working_program"),
        (UNSUPPORTED, ["OB16"], "independent_certification_failure"),
    ],
)
def test_every_terminal_compile_failure_status_is_persisted(
    status: str,
    failed_obligations: list[str],
    failure_class: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    xpt = XptResult(
        status=status,
        reason="terminal compile outcome",
        failed_obligations=failed_obligations,
    )

    def fake_compile(**kwargs):
        return RealCompileReport(
            base_url=kwargs["base_url"],
            model=kwargs["model"],
            status=status,
            reason=xpt.reason,
            elapsed_s=1.0,
            discovery_s=0.0,
            xpt=xpt,
        )

    monkeypatch.setattr(xpt_real_mod, "compile_real_endpoint", fake_compile)
    result = install_target(ResolvedTarget("http://127.0.0.1:18474/v1", "m"), home=tmp_path)

    assert result.report_path is not None and result.report_path.is_file()
    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert payload["status"] == status
    assert payload["compile"]["xpt"]["failure_class"] == failure_class
