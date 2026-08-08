"""Runtime entry points backed by the persistent verified-driver registry."""

from __future__ import annotations

from pathlib import Path

from xenolect.abi import ABI_VERSION
from xenolect.driver.runtime import DriverRuntime
from xenolect.endpoints.http import OpenAICompatClient
from xenolect.storage.registry import DriverRegistry, InstalledDriver


class DriverNotInstalledError(RuntimeError):
    pass


def resolve_installed_driver(
    *,
    base_url: str,
    model: str,
    home: str | Path | None = None,
) -> InstalledDriver:
    item = DriverRegistry(home).lookup(base_url, model, ABI_VERSION, verify=True)
    if item is None:
        raise DriverNotInstalledError(
            f"no verified driver installed for {base_url.rstrip('/')} model={model}; "
            "run `xenolect install` first"
        )
    return item


def open_runtime(
    *,
    base_url: str,
    model: str,
    api_key: str | None = None,
    home: str | Path | None = None,
    timeout: float = 120.0,
    temperature: float | None = 0.0,
    top_p: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
) -> DriverRuntime:
    """Create a stateful runtime using only a previously verified installed driver."""
    installed = resolve_installed_driver(base_url=base_url, model=model, home=home)
    driver = installed.load()
    client = OpenAICompatClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    )
    return DriverRuntime(driver=driver, client=client)
