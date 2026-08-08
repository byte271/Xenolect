"""Stable serialization and hashing of drivers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from xenolect.driver.ir import Driver


def driver_to_json(driver: Driver) -> str:
    return json.dumps(driver.canonical_dict(), sort_keys=True, indent=2) + "\n"


def driver_hash(driver: Driver) -> str:
    payload = json.dumps(driver.canonical_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def save_driver(driver: Driver, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(driver_to_json(driver), encoding="utf-8")
    return path


def load_driver(path: str | Path) -> Driver:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Driver.model_validate(data)
