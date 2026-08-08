"""Background Xenolect proxy service and lightweight OS integration.

The product path keeps one stable local OpenAI-compatible endpoint.  The
service reads the verified Driver registry dynamically, so installing another
model does not require spawning one daemon per model.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import plistlib
import shutil
import socket
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import uuid

from xenolect import __version__
from xenolect.storage.registry import xenolect_home

SERVICE_CONFIG_VERSION = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8179
MAX_SERVICE_LOG_BYTES = 1_000_000


class ServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ServiceConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT

    def __post_init__(self) -> None:
        if not _is_loopback_host(self.host):
            raise ValueError("Xenolect service must bind to a loopback address")
        if not (1 <= int(self.port) <= 65535):
            raise ValueError("Xenolect service port is invalid")

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1/health"


@dataclass(frozen=True)
class ServiceState:
    config: ServiceConfig
    running: bool
    started: bool = False
    autostart_enabled: bool = False
    service_version: str | None = None

    @property
    def base_url(self) -> str:
        return self.config.base_url


def service_config_path(home: str | Path | None = None) -> Path:
    return xenolect_home(home) / "service.json"


def service_pid_path(home: str | Path | None = None) -> Path:
    return xenolect_home(home) / "service.pid"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _is_loopback_host(host: str) -> bool:
    value = host.strip().lower()
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def save_service_config(config: ServiceConfig, home: str | Path | None = None) -> Path:
    path = service_config_path(home)
    payload = {
        "version": SERVICE_CONFIG_VERSION,
        "host": config.host,
        "port": config.port,
    }
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load_service_config(home: str | Path | None = None) -> ServiceConfig | None:
    path = service_config_path(home)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ServiceError(f"cannot read service configuration: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("version") != SERVICE_CONFIG_VERSION:
        raise ServiceError("unsupported service configuration")
    try:
        host = str(raw["host"])
        port = int(raw["port"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ServiceError("malformed service configuration") from exc
    try:
        return ServiceConfig(host=host, port=port)
    except ValueError as exc:
        raise ServiceError(f"invalid service configuration: {exc}") from exc


def _health_payload(config: ServiceConfig, timeout: float = 0.4) -> dict[str, Any] | None:
    request = urllib.request.Request(config.health_url, headers={"Accept": "application/json"})
    try:
        # Health checks are always loopback. Explicitly bypass environment proxies
        # so corporate/VPN proxy settings cannot make a local service look dead.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=timeout) as response:  # noqa: S310 - loopback URL
            if response.status != 200:
                return None
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("service") != "xenolect":
        return None
    return payload


def is_service_running(config: ServiceConfig, timeout: float = 0.4) -> bool:
    return _health_payload(config, timeout=timeout) is not None


def current_service_state(home: str | Path | None = None) -> ServiceState:
    config = load_service_config(home)
    if config is None:
        return ServiceState(ServiceConfig(), running=False)
    payload = _health_payload(config)
    version = payload.get("version") if isinstance(payload, dict) else None
    if not isinstance(version, str):
        version = None
    return ServiceState(
        config=config,
        running=payload is not None,
        autostart_enabled=is_autostart_registered(home),
        service_version=version,
    )


def _port_available(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def choose_service_port(host: str = DEFAULT_HOST, preferred: int = DEFAULT_PORT) -> int:
    for port in range(preferred, min(preferred + 32, 65536)):
        if _port_available(host, port):
            return port
    raise ServiceError("could not find a free local port for Xenolect")


def _pythonw_executable() -> str:
    exe = Path(sys.executable)
    if os.name == "nt":
        candidate = exe.with_name("pythonw.exe")
        if candidate.is_file():
            return str(candidate)
    return str(exe)


def _spawn_background(config_path: Path, home: Path) -> int:
    log_dir = home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "service.log"
    mode = "wb" if log_path.exists() and log_path.stat().st_size > MAX_SERVICE_LOG_BYTES else "ab"
    log = open(log_path, mode, buffering=0)  # noqa: SIM115 - child owns inherited handle
    command = [_pythonw_executable(), "-m", "xenolect.service", "--config", str(config_path)]
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": log,
        "stderr": log,
        "close_fds": True,
        "env": os.environ.copy(),
        "cwd": str(home),
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    else:
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(command, **kwargs)  # noqa: S603 - fixed executable/module
    finally:
        log.close()
    _atomic_write(service_pid_path(home), str(proc.pid) + "\n")
    return proc.pid


def _platform_family() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    return "other"


def autostart_label() -> str:
    """Human label for the current platform's login-start mechanism."""
    return {
        "windows": "Windows startup",
        "macos": "macOS login startup",
        "linux": "Linux login startup",
    }.get(_platform_family(), "Login startup")


def _user_home() -> Path:
    return Path.home()


def _windows_startup_path() -> Path | None:
    if _platform_family() != "windows":
        return None
    appdata = os.getenv("APPDATA")
    if not appdata:
        return None
    return (
        Path(appdata)
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "Startup"
        / "Xenolect.cmd"
    )


def _macos_launch_agent_path() -> Path | None:
    if _platform_family() != "macos":
        return None
    return _user_home() / "Library" / "LaunchAgents" / "io.xenolect.service.plist"


def _linux_systemd_paths() -> tuple[Path, Path] | None:
    if _platform_family() != "linux":
        return None
    config_home = Path(os.getenv("XDG_CONFIG_HOME", str(_user_home() / ".config")))
    unit = config_home / "systemd" / "user" / "xenolect.service"
    wants = config_home / "systemd" / "user" / "default.target.wants" / "xenolect.service"
    return unit, wants


def _linux_desktop_path() -> Path | None:
    if _platform_family() != "linux":
        return None
    config_home = Path(os.getenv("XDG_CONFIG_HOME", str(_user_home() / ".config")))
    return config_home / "autostart" / "xenolect.desktop"


def _linux_has_systemd() -> bool:
    # The unit can be enabled by a symlink without talking to the current user
    # manager, but only choose this backend on a system that is actually booted
    # with systemd.  Desktop autostart is the fallback elsewhere.
    return Path("/run/systemd/system").is_dir() and shutil.which("systemctl") is not None


def _systemd_quote(value: str) -> str:
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _desktop_quote(value: str) -> str:
    # freedesktop Exec= supports double-quoted arguments. Escape the characters
    # that are special inside those quotes; no shell is involved.
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"').replace('`', '\\`').replace('$', '\\$') + '"'


def _register_windows_autostart(home: str | Path | None) -> bool:
    startup = _windows_startup_path()
    if startup is None:
        return False
    config_path = service_config_path(home).resolve()
    pythonw = _pythonw_executable()
    try:
        startup.parent.mkdir(parents=True, exist_ok=True)
        # No secrets are written here. A protected endpoint must obtain its key
        # from the user's environment at runtime.
        text = (
            "@echo off\r\n"
            f'start "" /min "{pythonw}" -m xenolect.service --config "{config_path}"\r\n'
        )
        _atomic_write(startup, text)
    except OSError:
        return False
    return _is_windows_autostart_registered(home)


def _register_macos_autostart(home: str | Path | None) -> bool:
    startup = _macos_launch_agent_path()
    if startup is None:
        return False
    root = xenolect_home(home)
    config_path = service_config_path(root).resolve()
    log_path = (root / "logs" / "service.log").resolve()
    payload = {
        "Label": "io.xenolect.service",
        "ProgramArguments": [
            _pythonw_executable(),
            "-m",
            "xenolect.service",
            "--config",
            str(config_path),
        ],
        "RunAtLoad": True,
        "KeepAlive": False,
        "ProcessType": "Background",
        "StandardOutPath": str(log_path),
        "StandardErrorPath": str(log_path),
    }
    try:
        startup.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        raw = plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)
        tmp = startup.with_name(f".{startup.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_bytes(raw)
            os.replace(tmp, startup)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
    except OSError:
        return False
    return _is_macos_autostart_registered(home)


def _register_linux_autostart(home: str | Path | None) -> bool:
    root = xenolect_home(home)
    config_path = service_config_path(root).resolve()
    python = _pythonw_executable()

    paths = _linux_systemd_paths()
    desktop = _linux_desktop_path()
    if paths is None or desktop is None:
        return False
    unit, wants = paths

    if _linux_has_systemd():
        unit_text = (
            "[Unit]\n"
            "Description=Xenolect local compatibility service\n"
            "After=default.target\n\n"
            "[Service]\n"
            "Type=simple\n"
            f"ExecStart={_systemd_quote(python)} -m xenolect.service --config {_systemd_quote(str(config_path))}\n"
            "Restart=no\n\n"
            "[Install]\n"
            "WantedBy=default.target\n"
        )
        try:
            unit.parent.mkdir(parents=True, exist_ok=True)
            wants.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(unit, unit_text)
            try:
                desktop.unlink()
            except FileNotFoundError:
                pass
            if wants.exists() or wants.is_symlink():
                wants.unlink()
            # Relative link keeps the config tree relocatable within ~/.config.
            wants.symlink_to(Path("..") / unit.name)
        except OSError:
            return False
        return _is_linux_autostart_registered(home)

    desktop_text = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Xenolect\n"
        "NoDisplay=true\n"
        "X-GNOME-Autostart-enabled=true\n"
        f"Exec={_desktop_quote(python)} -m xenolect.service --config {_desktop_quote(str(config_path))}\n"
    )
    try:
        desktop.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(desktop, desktop_text)
        # Avoid a future duplicate if this config was previously enabled via
        # systemd and the host later changes session type.
        try:
            if wants.exists() or wants.is_symlink():
                wants.unlink()
        except OSError:
            pass
    except OSError:
        return False
    return _is_linux_autostart_registered(home)


def register_autostart(home: str | Path | None = None) -> bool:
    """Register best-effort per-user login startup on Windows, macOS, or Linux."""
    family = _platform_family()
    if family == "windows":
        return _register_windows_autostart(home)
    if family == "macos":
        return _register_macos_autostart(home)
    if family == "linux":
        return _register_linux_autostart(home)
    return False


def _unregister_path(path: Path | None, *, description: str) -> bool:
    if path is None:
        return False
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ServiceError(f"could not disable {description}: {exc}") from exc


def unregister_autostart(home: str | Path | None = None) -> bool:
    """Remove per-user login startup. Missing registration is already disabled."""
    family = _platform_family()
    if family == "windows":
        return _unregister_path(_windows_startup_path(), description="Windows startup")
    if family == "macos":
        return _unregister_path(_macos_launch_agent_path(), description="macOS login startup")
    if family == "linux":
        changed = False
        paths = _linux_systemd_paths()
        if paths is not None:
            unit, wants = paths
            changed = _unregister_path(wants, description="Linux login startup") or changed
            changed = _unregister_path(unit, description="Linux login startup") or changed
        changed = _unregister_path(_linux_desktop_path(), description="Linux login startup") or changed
        return changed
    return False


def _file_contains(path: Path | None, needle: str) -> bool:
    if path is None or not path.is_file():
        return False
    try:
        return needle in path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def _is_windows_autostart_registered(home: str | Path | None = None) -> bool:
    return _file_contains(_windows_startup_path(), str(service_config_path(home).resolve()))


def _is_macos_autostart_registered(home: str | Path | None = None) -> bool:
    startup = _macos_launch_agent_path()
    if startup is None or not startup.is_file():
        return False
    try:
        payload = plistlib.loads(startup.read_bytes())
    except (OSError, ValueError, plistlib.InvalidFileException):
        return False
    args = payload.get("ProgramArguments") if isinstance(payload, dict) else None
    return isinstance(args, list) and str(service_config_path(home).resolve()) in [str(x) for x in args]


def _is_linux_autostart_registered(home: str | Path | None = None) -> bool:
    paths = _linux_systemd_paths()
    if paths is not None:
        unit, wants = paths
        if (wants.exists() or wants.is_symlink()) and _file_contains(
            unit, str(service_config_path(home).resolve())
        ):
            return True
    return _file_contains(_linux_desktop_path(), str(service_config_path(home).resolve()))


def is_autostart_registered(home: str | Path | None = None) -> bool:
    family = _platform_family()
    if family == "windows":
        return _is_windows_autostart_registered(home)
    if family == "macos":
        return _is_macos_autostart_registered(home)
    if family == "linux":
        return _is_linux_autostart_registered(home)
    return False

def _terminate_pid(pid: int) -> None:
    if pid <= 0:
        raise ServiceError("invalid Xenolect service process id")
    if os.name == "nt":
        try:
            completed = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5.0,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ServiceError(f"could not stop Xenolect process {pid}: {exc}") from exc
        # 128/255-style failures are tolerated only if the process is already gone;
        # the caller verifies the health endpoint afterwards.
        _ = completed.returncode
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError as exc:
        raise ServiceError(f"could not stop Xenolect process {pid}: {exc}") from exc


def stop_background_service(
    *,
    home: str | Path | None = None,
    disable_autostart: bool = True,
) -> ServiceState:
    """Stop only a process proven to be the configured Xenolect service."""
    root = xenolect_home(home)
    config = load_service_config(root)
    if disable_autostart:
        try:
            unregister_autostart(root)
        except ServiceError:
            # Stopping the live service is more important than failing early on
            # an OS login-start registration problem. The returned state tells the
            # CLI whether startup is still enabled.
            pass
    auto_enabled = is_autostart_registered(root)
    if config is None:
        return ServiceState(ServiceConfig(), running=False, autostart_enabled=auto_enabled)

    payload = _health_payload(config, timeout=0.5)
    if payload is None:
        try:
            service_pid_path(root).unlink()
        except FileNotFoundError:
            pass
        return ServiceState(config, running=False, autostart_enabled=auto_enabled)

    pid_raw = payload.get("pid")
    pid: int | None = None
    if isinstance(pid_raw, int) and pid_raw > 0:
        pid = pid_raw
    else:
        # Backward-compatible fallback for an older Xenolect service that did
        # not publish its pid in /health. We only trust service.pid after the
        # configured endpoint has positively identified itself as Xenolect.
        try:
            pid = int(service_pid_path(root).read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = None
    if pid is None:
        raise ServiceError("Xenolect is running but its process id could not be verified")

    _terminate_pid(pid)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not is_service_running(config, timeout=0.2):
            try:
                service_pid_path(root).unlink()
            except FileNotFoundError:
                pass
            return ServiceState(config, running=False, autostart_enabled=auto_enabled)
        time.sleep(0.1)
    raise ServiceError("Xenolect did not stop cleanly")


def ensure_background_service(
    *,
    home: str | Path | None = None,
    preferred_port: int = DEFAULT_PORT,
    enable_autostart: bool = True,
) -> ServiceState:
    """Ensure the local Xenolect endpoint is running and return its address."""
    root = xenolect_home(home)
    root.mkdir(parents=True, exist_ok=True)

    existing = load_service_config(root)
    if existing is not None:
        payload = _health_payload(existing)
        if payload is not None:
            running_version = payload.get("version")
            if running_version == __version__:
                auto = register_autostart(root) if enable_autostart else is_autostart_registered(root)
                return ServiceState(
                    existing,
                    running=True,
                    started=False,
                    autostart_enabled=auto,
                    service_version=__version__,
                )
            # An older Xenolect process is real but must not survive an upgrade.
            stop_background_service(home=root, disable_autostart=False)

    # Adopt a compatible service already listening on the default address even
    # if it was started by an older Xenolect build before service.json existed.
    if existing is None:
        default_config = ServiceConfig()
        payload = _health_payload(default_config)
        if payload is not None:
            save_service_config(default_config, root)
            if payload.get("version") == __version__:
                auto = register_autostart(root) if enable_autostart else False
                return ServiceState(
                    default_config,
                    running=True,
                    started=False,
                    autostart_enabled=auto,
                    service_version=__version__,
                )
            stop_background_service(home=root, disable_autostart=False)
            existing = default_config

    host = existing.host if existing is not None else DEFAULT_HOST
    preferred = existing.port if existing is not None else preferred_port
    if not _port_available(host, preferred):
        preferred = choose_service_port(host, preferred_port)
    config = ServiceConfig(host=host, port=preferred)
    config_path = save_service_config(config, root)
    _spawn_background(config_path, root)

    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline:
        if is_service_running(config, timeout=0.25):
            auto = register_autostart(root) if enable_autostart else False
            return ServiceState(
                config,
                running=True,
                started=True,
                autostart_enabled=auto,
                service_version=__version__,
            )
        time.sleep(0.1)
    raise ServiceError(f"Xenolect could not start. See {root / 'logs' / 'service.log'}")


def _run(config_path: Path) -> None:
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        config = ServiceConfig(host=str(raw["host"]), port=int(raw["port"]))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"invalid Xenolect service config: {exc}") from exc

    home = config_path.parent
    from xenolect.proxy import RegistryProxyService, serve
    from xenolect.storage.registry import DriverRegistry

    registry = DriverRegistry(home)
    service = RegistryProxyService(registry=registry, api_key=os.getenv("XENOLECT_API_KEY"))
    serve(service, host=config.host, port=config.port)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    _run(args.config)


if __name__ == "__main__":
    main()
