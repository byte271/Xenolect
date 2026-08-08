from __future__ import annotations

from pathlib import Path

import pytest

import xenolect.compiler.xpt_real as xpt_real_mod
from xenolect.compiler.install import READY, ResolvedTarget, install_target
from xenolect.compiler.xpt_real import RealCompileReport
from xenolect.driver.ir import identity_driver
from xenolect.runtime import DriverNotInstalledError, open_runtime
from xenolect.storage.registry import DriverRegistry
from xenolect.xpt.runtime import CERTIFIED, XptResult


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


def test_registry_detects_driver_tamper(tmp_path: Path) -> None:
    registry = DriverRegistry(tmp_path)
    item = registry.install(base_url="http://x/v1", model="m", driver=identity_driver())
    item.driver_path.write_text('{"tool_encoding":"xml_json"}', encoding="utf-8")
    with pytest.raises(Exception):
        registry.lookup("http://x/v1", "m", verify=True)


def test_diagnostic_reports_are_bounded_per_binding(tmp_path):
    from xenolect.storage.registry import MAX_REPORTS_PER_BINDING

    reg = DriverRegistry(tmp_path)
    installed = reg.install(base_url="http://up/v1", model="m", driver=identity_driver())
    for i in range(MAX_REPORTS_PER_BINDING + 7):
        reg.write_report(installed, {"i": i})
    reports = list(reg.reports_dir.glob(f"{installed.binding_id}-*.json"))
    assert len(reports) == MAX_REPORTS_PER_BINDING
