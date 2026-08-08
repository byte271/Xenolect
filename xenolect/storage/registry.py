"""Persistent registry for verified Xenolect drivers.

The registry binds one exact observable endpoint/model/ABI tuple to a content-
addressed .mdriver artifact.  It never stores API keys or provider identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xenolect.abi import ABI_VERSION
from xenolect.driver.ir import Driver
from xenolect.driver.serialize import driver_hash, driver_to_json, load_driver

REGISTRY_VERSION = 1
MAX_REPORTS_PER_BINDING = 10


class RegistryError(RuntimeError):
    """Registry is malformed or an installed artifact fails integrity checks."""


def xenolect_home(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser()
    env = os.getenv("XENOLECT_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".xenolect"


def normalize_base_url(base_url: str) -> str:
    return base_url.strip().rstrip("/")


def binding_id(base_url: str, model: str, target_abi: str = ABI_VERSION) -> str:
    material = f"{normalize_base_url(base_url)}\0{model}\0{target_abi}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:20]


def ban_id(base_url: str, model: str) -> str:
    material = f"{normalize_base_url(base_url)}\0{model}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:20]


@dataclass(frozen=True)
class BannedModel:
    ban_id: str
    base_url: str
    model: str
    banned_at: str


@dataclass(frozen=True)
class InstalledDriver:
    binding_id: str
    base_url: str
    model: str
    target_abi: str
    driver_hash: str
    driver_path: Path
    installed_at: str
    compiler: str
    generations: int | None = None
    compile_elapsed_s: float | None = None
    model_fingerprint: str | None = None

    def load(self) -> Driver:
        driver = load_driver(self.driver_path)
        actual = driver_hash(driver)
        if actual != self.driver_hash:
            raise RegistryError(
                f"installed driver hash mismatch for {self.driver_path}: "
                f"registry={self.driver_hash} actual={actual}"
            )
        if driver.target_abi != self.target_abi:
            raise RegistryError(
                f"installed driver ABI mismatch: registry={self.target_abi} "
                f"driver={driver.target_abi}"
            )
        return driver


class DriverRegistry:
    def __init__(self, home: str | Path | None = None) -> None:
        self.home = xenolect_home(home)
        self.path = self.home / "registry.json"
        self.drivers_dir = self.home / "drivers"
        self.reports_dir = self.home / "reports"

    def _empty(self) -> dict[str, Any]:
        return {"version": REGISTRY_VERSION, "bindings": {}, "bans": {}}

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(f"cannot read registry {self.path}: {exc}") from exc
        if not isinstance(data, dict) or data.get("version") != REGISTRY_VERSION:
            raise RegistryError(f"unsupported or malformed registry: {self.path}")
        bindings = data.get("bindings")
        if not isinstance(bindings, dict):
            raise RegistryError(f"registry has no bindings object: {self.path}")
        # `bans` was added without bumping the file version so existing v6
        # registries remain readable. Missing means no banned models.
        bans = data.get("bans")
        if bans is None:
            data["bans"] = {}
        elif not isinstance(bans, dict):
            raise RegistryError(f"registry has malformed bans object: {self.path}")
        return data

    def _safe_driver_path(self, relative: str) -> Path:
        root = self.home.resolve()
        candidate = (self.home / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise RegistryError(f"registry driver path escapes XENOLECT_HOME: {relative}") from exc
        return candidate

    def _binding_from_raw(self, key: str, raw: Any, *, verify: bool) -> InstalledDriver:
        if not isinstance(raw, dict):
            raise RegistryError(f"malformed registry binding: {key}")
        try:
            path = self._safe_driver_path(str(raw["driver_file"]))
            item = InstalledDriver(
                binding_id=key,
                base_url=str(raw["base_url"]),
                model=str(raw["model"]),
                target_abi=str(raw["target_abi"]),
                driver_hash=str(raw["driver_hash"]),
                driver_path=path,
                installed_at=str(raw["installed_at"]),
                compiler=str(raw.get("compiler", "unknown")),
                generations=raw.get("generations"),
                compile_elapsed_s=raw.get("compile_elapsed_s"),
                model_fingerprint=raw.get("model_fingerprint"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RegistryError(f"malformed registry binding: {key}") from exc
        if not item.driver_path.is_file():
            raise RegistryError(f"installed driver is missing: {item.driver_path}")
        if verify:
            item.load()
        return item

    def is_banned(self, base_url: str, model: str) -> bool:
        return ban_id(base_url, model) in self._read()["bans"]

    def ban(self, base_url: str, model: str) -> BannedModel:
        data = self._read()
        key = ban_id(base_url, model)
        banned_at = datetime.now(timezone.utc).isoformat()
        data["bans"][key] = {
            "base_url": normalize_base_url(base_url),
            "model": model,
            "banned_at": banned_at,
        }
        self.home.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self.path, json.dumps(data, indent=2, sort_keys=True) + "\n")
        return BannedModel(key, normalize_base_url(base_url), model, banned_at)

    def unban(self, base_url: str, model: str) -> bool:
        data = self._read()
        key = ban_id(base_url, model)
        if key not in data["bans"]:
            return False
        del data["bans"][key]
        self.home.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self.path, json.dumps(data, indent=2, sort_keys=True) + "\n")
        return True

    def list_banned(self) -> list[BannedModel]:
        data = self._read()
        out: list[BannedModel] = []
        for key, raw in data["bans"].items():
            if not isinstance(raw, dict):
                raise RegistryError(f"malformed banned-model entry: {key}")
            try:
                out.append(
                    BannedModel(
                        ban_id=str(key),
                        base_url=str(raw["base_url"]),
                        model=str(raw["model"]),
                        banned_at=str(raw["banned_at"]),
                    )
                )
            except KeyError as exc:
                raise RegistryError(f"malformed banned-model entry: {key}") from exc
        return sorted(out, key=lambda item: (item.base_url, item.model))

    def lookup(
        self,
        base_url: str,
        model: str,
        target_abi: str = ABI_VERSION,
        *,
        verify: bool = True,
        expected_model_fingerprint: str | None = None,
        allow_banned: bool = False,
    ) -> InstalledDriver | None:
        data = self._read()
        if not allow_banned and ban_id(base_url, model) in data["bans"]:
            return None
        key = binding_id(base_url, model, target_abi)
        raw = data["bindings"].get(key)
        if raw is None:
            return None
        item = self._binding_from_raw(key, raw, verify=verify)
        if normalize_base_url(item.base_url) != normalize_base_url(base_url):
            raise RegistryError(f"registry binding URL mismatch: {key}")
        if item.model != model or item.target_abi != target_abi:
            raise RegistryError(f"registry binding identity mismatch: {key}")
        if expected_model_fingerprint is not None:
            if item.model_fingerprint != expected_model_fingerprint:
                return None
        return item

    def install(
        self,
        *,
        base_url: str,
        model: str,
        driver: Driver,
        compiler: str = "xpt",
        generations: int | None = None,
        compile_elapsed_s: float | None = None,
        metadata: dict[str, Any] | None = None,
        model_fingerprint: str | None = None,
    ) -> InstalledDriver:
        data = self._read()
        self.drivers_dir.mkdir(parents=True, exist_ok=True)
        digest = driver_hash(driver)
        driver_file = self.drivers_dir / f"{digest}.mdriver"
        payload = driver_to_json(driver)
        if driver_file.exists():
            try:
                existing = load_driver(driver_file)
                valid_existing = driver_hash(existing) == digest
            except Exception:  # noqa: BLE001
                valid_existing = False
            if not valid_existing:
                # The caller supplies a freshly certified in-memory driver, so replacing a
                # corrupt content-addressed file is safer than preserving bad cache state.
                _atomic_write_text(driver_file, payload)
        else:
            _atomic_write_text(driver_file, payload)

        key = binding_id(base_url, model, driver.target_abi)
        installed_at = datetime.now(timezone.utc).isoformat()
        rel = driver_file.relative_to(self.home).as_posix()
        entry: dict[str, Any] = {
            "base_url": normalize_base_url(base_url),
            "model": model,
            "target_abi": driver.target_abi,
            "driver_hash": digest,
            "driver_file": rel,
            "installed_at": installed_at,
            "compiler": compiler,
            "generations": generations,
            "compile_elapsed_s": compile_elapsed_s,
            "model_fingerprint": model_fingerprint,
        }
        if metadata:
            # Metadata is diagnostic only.  Never accept secrets through this API.
            entry["metadata"] = metadata
        data["bindings"][key] = entry
        self.home.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self.path, json.dumps(data, indent=2, sort_keys=True) + "\n")
        return self.lookup(base_url, model, driver.target_abi, verify=True)  # type: ignore[return-value]

    def remove(self, base_url: str, model: str, target_abi: str = ABI_VERSION) -> bool:
        data = self._read()
        key = binding_id(base_url, model, target_abi)
        if key not in data["bindings"]:
            return False
        del data["bindings"][key]
        self.home.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self.path, json.dumps(data, indent=2, sort_keys=True) + "\n")
        return True

    def list(
        self, *, include_banned: bool = False, skip_invalid: bool = False
    ) -> list[InstalledDriver]:
        data = self._read()
        banned_keys = set(data["bans"])
        out: list[InstalledDriver] = []
        for key, raw in data["bindings"].items():
            if not isinstance(raw, dict):
                raise RegistryError("malformed registry binding")
            try:
                base_url = str(raw["base_url"])
                model = str(raw["model"])
                target_abi = str(raw["target_abi"])
            except KeyError as exc:
                raise RegistryError("malformed registry binding") from exc
            if not include_banned and ban_id(base_url, model) in banned_keys:
                continue
            try:
                item = self._binding_from_raw(str(key), raw, verify=True)
                if item.binding_id != binding_id(base_url, model, target_abi):
                    raise RegistryError(f"registry binding key mismatch: {key}")
            except RegistryError:
                if skip_invalid:
                    continue
                raise
            out.append(item)
        return sorted(out, key=lambda item: (item.base_url, item.model))

    def write_report(self, binding: InstalledDriver, payload: dict[str, Any]) -> Path:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = self.reports_dir / f"{binding.binding_id}-{stamp}.json"
        _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

        # Reports are diagnostics, not product state. Keep them bounded so repeated
        # forced rebuilds cannot grow XENOLECT_HOME forever.
        reports = sorted(
            self.reports_dir.glob(f"{binding.binding_id}-*.json"),
            key=lambda item: item.name,
            reverse=True,
        )
        for old in reports[MAX_REPORTS_PER_BINDING:]:
            try:
                old.unlink()
            except OSError:
                pass
        return path


def export_driver(installed: InstalledDriver, destination: str | Path) -> Path:
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(installed.driver_path, dest)
    return dest


def _atomic_write_text(path: Path, text: str) -> None:
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
