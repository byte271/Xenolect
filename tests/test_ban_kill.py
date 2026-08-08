from __future__ import annotations

from pathlib import Path

import xenolect.service as service
from xenolect.driver.ir import identity_driver
from xenolect.proxy import RegistryProxyService
from xenolect.service import ServiceConfig, stop_background_service
from xenolect.storage.registry import DriverRegistry


def test_ban_hides_driver_and_unban_restores_without_reinstall(tmp_path: Path) -> None:
    registry = DriverRegistry(tmp_path)
    installed = registry.install(base_url="http://127.0.0.1:11434/v1", model="m", driver=identity_driver())

    assert [item.model for item in registry.list()] == ["m"]
    registry.ban(installed.base_url, installed.model)
    assert registry.is_banned(installed.base_url, installed.model) is True
    assert registry.list() == []
    assert [item.model for item in registry.list(include_banned=True)] == ["m"]
    assert installed.driver_path.is_file()

    assert registry.unban(installed.base_url, installed.model) is True
    restored = registry.lookup(installed.base_url, installed.model)
    assert restored is not None
    assert restored.driver_hash == installed.driver_hash


def test_proxy_refresh_drops_banned_model_without_restart(tmp_path: Path) -> None:
    registry = DriverRegistry(tmp_path)
    item = registry.install(base_url="http://127.0.0.1:11434/v1", model="m", driver=identity_driver())
    proxy = RegistryProxyService(registry=registry)

    assert [row["id"] for row in proxy.models()["data"]] == ["m"]
    registry.ban(item.base_url, item.model)
    assert proxy.models()["data"] == []
    registry.unban(item.base_url, item.model)
    assert [row["id"] for row in proxy.models()["data"]] == ["m"]


def test_stop_background_service_disables_autostart_and_uses_verified_health_pid(
    tmp_path: Path, monkeypatch
) -> None:
    config = ServiceConfig(port=8197)
    service.save_service_config(config, tmp_path)
    calls: list[int] = []
    running = {"value": True}

    monkeypatch.setattr(service, "unregister_autostart", lambda home=None: True)
    monkeypatch.setattr(
        service,
        "_health_payload",
        lambda value, timeout=0.4: {"service": "xenolect", "pid": 4321} if running["value"] else None,
    )

    def terminate(pid: int) -> None:
        calls.append(pid)
        running["value"] = False

    monkeypatch.setattr(service, "_terminate_pid", terminate)
    monkeypatch.setattr(service, "is_service_running", lambda value, timeout=0.4: running["value"])
    monkeypatch.setattr(service.time, "sleep", lambda value: None)

    state = stop_background_service(home=tmp_path)
    assert calls == [4321]
    assert state.running is False
    assert state.autostart_enabled is False


def test_stop_does_not_kill_when_configured_endpoint_is_not_xenolect(tmp_path: Path, monkeypatch) -> None:
    config = ServiceConfig(port=8198)
    service.save_service_config(config, tmp_path)
    called = {"terminate": False}

    monkeypatch.setattr(service, "unregister_autostart", lambda home=None: True)
    monkeypatch.setattr(service, "_health_payload", lambda value, timeout=0.4: None)
    monkeypatch.setattr(service, "_terminate_pid", lambda pid: called.__setitem__("terminate", True))

    state = stop_background_service(home=tmp_path)
    assert state.running is False
    assert called["terminate"] is False
