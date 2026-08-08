from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from xenolect import __version__
from xenolect.service import (
    ServiceConfig,
    ensure_background_service,
    load_service_config,
    save_service_config,
)


def test_service_config_round_trip(tmp_path: Path) -> None:
    config = ServiceConfig(host="127.0.0.1", port=8199)
    path = save_service_config(config, tmp_path)
    assert path.is_file()
    assert load_service_config(tmp_path) == config
    assert json.loads(path.read_text())["port"] == 8199


def test_ensure_service_reuses_running_config(tmp_path: Path, monkeypatch) -> None:
    config = ServiceConfig(port=8191)
    save_service_config(config, tmp_path)
    monkeypatch.setattr(
        "xenolect.service._health_payload",
        lambda value, timeout=0.4: {"service": "xenolect", "version": __version__, "pid": 10},
    )
    monkeypatch.setattr("xenolect.service.register_autostart", lambda home=None: True)

    state = ensure_background_service(home=tmp_path)
    assert state.running is True
    assert state.started is False
    assert state.autostart_enabled is True
    assert state.config == config
    assert state.service_version == __version__


def test_ensure_service_restarts_old_version(tmp_path: Path, monkeypatch) -> None:
    config = ServiceConfig(port=8192)
    save_service_config(config, tmp_path)
    health_calls = {"n": 0}
    stopped = {"n": 0}
    spawned = {"n": 0}

    def health(value, timeout=0.4):
        health_calls["n"] += 1
        if health_calls["n"] == 1:
            return {"service": "xenolect", "version": "0.0.9", "pid": 22}
        return {"service": "xenolect", "version": __version__, "pid": 23}

    monkeypatch.setattr("xenolect.service._health_payload", health)
    monkeypatch.setattr(
        "xenolect.service.stop_background_service",
        lambda **kwargs: stopped.__setitem__("n", stopped["n"] + 1),
    )
    monkeypatch.setattr("xenolect.service._port_available", lambda host, port: True)
    monkeypatch.setattr(
        "xenolect.service._spawn_background",
        lambda config_path, home: spawned.__setitem__("n", spawned["n"] + 1) or 23,
    )
    monkeypatch.setattr("xenolect.service.is_service_running", lambda value, timeout=0.4: True)
    monkeypatch.setattr("xenolect.service.register_autostart", lambda home=None: True)

    state = ensure_background_service(home=tmp_path)
    assert stopped["n"] == 1
    assert spawned["n"] == 1
    assert state.started is True
    assert state.service_version == __version__


def test_ensure_service_starts_once_and_waits_for_health(tmp_path: Path, monkeypatch) -> None:
    calls = {"spawn": 0, "health": 0}

    monkeypatch.setattr("xenolect.service._port_available", lambda host, port: True)

    def health(config, timeout=0.4):
        calls["health"] += 1
        return calls["health"] >= 2

    def spawn(config_path, home):
        calls["spawn"] += 1
        return 123

    monkeypatch.setattr("xenolect.service.is_service_running", health)
    monkeypatch.setattr("xenolect.service._spawn_background", spawn)
    monkeypatch.setattr("xenolect.service.register_autostart", lambda home=None: False)
    monkeypatch.setattr("xenolect.service.time.sleep", lambda value: None)

    state = ensure_background_service(home=tmp_path, enable_autostart=False)
    assert state.running is True
    assert state.started is True
    assert calls["spawn"] == 1


def test_register_autostart_writes_no_secret(tmp_path: Path, monkeypatch) -> None:
    import xenolect.service as service

    startup = tmp_path / "Startup" / "Xenolect.cmd"
    monkeypatch.setattr(service, "_platform_family", lambda: "windows")
    monkeypatch.setattr(service, "_windows_startup_path", lambda: startup)
    monkeypatch.setattr(service, "_pythonw_executable", lambda: r"C:\\Python\\pythonw.exe")
    save_service_config(ServiceConfig(port=8179), tmp_path)

    assert service.register_autostart(tmp_path) is True
    text = startup.read_text(encoding="utf-8")
    assert "pythonw.exe" in text
    assert "xenolect.service" in text
    assert "API" not in text.upper()
    assert str((tmp_path / "service.json").resolve()) in text


def test_macos_launchagent_registration(tmp_path: Path, monkeypatch) -> None:
    import plistlib
    import xenolect.service as service

    user_home = tmp_path / "user"
    xhome = tmp_path / "xenolect-home"
    monkeypatch.setattr(service, "_platform_family", lambda: "macos")
    monkeypatch.setattr(service, "_user_home", lambda: user_home)
    monkeypatch.setattr(service, "_pythonw_executable", lambda: "/usr/local/bin/python3")
    save_service_config(ServiceConfig(port=8179), xhome)

    assert service.register_autostart(xhome) is True
    plist_path = user_home / "Library" / "LaunchAgents" / "io.xenolect.service.plist"
    payload = plistlib.loads(plist_path.read_bytes())
    assert payload["RunAtLoad"] is True
    assert payload["Label"] == "io.xenolect.service"
    assert str((xhome / "service.json").resolve()) in payload["ProgramArguments"]
    assert "API" not in plist_path.read_text(encoding="utf-8").upper()
    assert service.is_autostart_registered(xhome) is True
    assert service.unregister_autostart(xhome) is True
    assert service.is_autostart_registered(xhome) is False


@pytest.mark.skipif(sys.platform == "win32", reason="Linux autostart path semantics require POSIX")
def test_linux_systemd_user_registration(tmp_path: Path, monkeypatch) -> None:
    import xenolect.service as service

    user_home = tmp_path / "user"
    xhome = tmp_path / "xenolect-home"
    monkeypatch.setattr(service, "_platform_family", lambda: "linux")
    monkeypatch.setattr(service, "_user_home", lambda: user_home)
    monkeypatch.setattr(service, "_linux_has_systemd", lambda: True)
    monkeypatch.setattr(service, "_pythonw_executable", lambda: "/usr/bin/python3")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    save_service_config(ServiceConfig(port=8179), xhome)

    assert service.register_autostart(xhome) is True
    unit = user_home / ".config" / "systemd" / "user" / "xenolect.service"
    wants = user_home / ".config" / "systemd" / "user" / "default.target.wants" / "xenolect.service"
    assert unit.is_file()
    assert wants.is_symlink()
    text = unit.read_text(encoding="utf-8")
    assert "xenolect.service" in text
    assert str((xhome / "service.json").resolve()) in text
    assert "API" not in text.upper()
    assert service.is_autostart_registered(xhome) is True
    service.unregister_autostart(xhome)
    assert service.is_autostart_registered(xhome) is False


@pytest.mark.skipif(sys.platform == "win32", reason="Linux autostart path semantics require POSIX")
def test_linux_desktop_autostart_fallback(tmp_path: Path, monkeypatch) -> None:
    import xenolect.service as service

    user_home = tmp_path / "user"
    xhome = tmp_path / "xenolect-home"
    monkeypatch.setattr(service, "_platform_family", lambda: "linux")
    monkeypatch.setattr(service, "_user_home", lambda: user_home)
    monkeypatch.setattr(service, "_linux_has_systemd", lambda: False)
    monkeypatch.setattr(service, "_pythonw_executable", lambda: "/opt/Python 3/bin/python3")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    save_service_config(ServiceConfig(port=8179), xhome)

    assert service.register_autostart(xhome) is True
    desktop = user_home / ".config" / "autostart" / "xenolect.desktop"
    text = desktop.read_text(encoding="utf-8")
    assert "[Desktop Entry]" in text
    assert '"/opt/Python 3/bin/python3"' in text
    assert str((xhome / "service.json").resolve()) in text
    assert service.is_autostart_registered(xhome) is True


def test_windows_startup_registration_is_platform_backend(tmp_path: Path, monkeypatch) -> None:
    import xenolect.service as service

    startup = tmp_path / "Startup" / "Xenolect.cmd"
    monkeypatch.setattr(service, "_platform_family", lambda: "windows")
    monkeypatch.setattr(service, "_windows_startup_path", lambda: startup)
    monkeypatch.setattr(service, "_pythonw_executable", lambda: r"C:\\Python\\pythonw.exe")
    save_service_config(ServiceConfig(port=8179), tmp_path)

    assert service.register_autostart(tmp_path) is True
    assert service.is_autostart_registered(tmp_path) is True
