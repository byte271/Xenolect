"""Fast, provider-neutral discovery of local OpenAI-compatible endpoints."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from xenolect.endpoints.http import OpenAICompatClient

DEFAULT_LOCAL_BASE_URLS = (
    "http://127.0.0.1:11434/v1",  # common local model server
    "http://127.0.0.1:1234/v1",
    "http://127.0.0.1:8080/v1",
    "http://127.0.0.1:8000/v1",
)


def normalize_endpoint(value: str) -> str:
    """Accept a port, host:port, or full URL and return an OpenAI base URL."""
    raw = value.strip().rstrip("/")
    if not raw:
        raise ValueError("endpoint cannot be empty")
    if raw.isdigit():
        return f"http://127.0.0.1:{raw}/v1"
    if "://" not in raw:
        raw = "http://" + raw
    try:
        parsed = urlsplit(raw)
        _ = parsed.port  # force invalid ports to raise here
    except ValueError as exc:
        raise ValueError(f"invalid server address: {value!r}") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("server address must use http:// or https:// and include a host")

    path = parsed.path.rstrip("/")
    for suffix in ("/chat/completions", "/models"):
        if path.endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
            break
    if not path:
        path = "/v1"
    elif not path.endswith("/v1"):
        path += "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def model_record_fingerprint(record: dict[str, object]) -> str:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class DiscoveredEndpoint:
    base_url: str
    models: tuple[str, ...]
    model_fingerprints: tuple[tuple[str, str], ...] = ()

    def fingerprint_for(self, model: str) -> str | None:
        return dict(self.model_fingerprints).get(model)


def candidate_base_urls(explicit: str | None = None) -> list[str]:
    values: list[str] = []
    if explicit:
        values.append(normalize_endpoint(explicit))
    for key in ("XENOLECT_BASE_URL", "OPENAI_BASE_URL"):
        value = os.getenv(key)
        if value:
            values.append(normalize_endpoint(value))
    # Environment hints are preferences, not replacements for local discovery.
    # A stale OPENAI_BASE_URL must not hide an otherwise healthy local server.
    values.extend(DEFAULT_LOCAL_BASE_URLS)
    return list(dict.fromkeys(v.rstrip("/") for v in values if v))


def _probe_endpoint(url: str, api_key: str | None, timeout: float) -> DiscoveredEndpoint:
    with OpenAICompatClient(
        base_url=url,
        api_key=api_key,
        timeout=timeout,
        max_retries=1,
    ) as client:
        records = client.list_model_records()
    # Never recursively select Xenolect's own local proxy as an upstream model
    # server.  Its /models records are intentionally marked owned_by=xenolect.
    if records and all(str(record.get("owned_by", "")).lower() == "xenolect" for record in records):
        raise RuntimeError("Xenolect's local proxy cannot be used as its own upstream")
    models = tuple(str(record["id"]) for record in records)
    fingerprints = tuple(
        (str(record["id"]), model_record_fingerprint(record)) for record in records
    )
    return DiscoveredEndpoint(url.rstrip("/"), models, fingerprints)


def inspect_openai_endpoint(
    *, base_url: str, api_key: str | None = None, timeout: float = 0.75
) -> DiscoveredEndpoint:
    return _probe_endpoint(normalize_endpoint(base_url), api_key, timeout)


def scan_openai_endpoints(
    *,
    base_urls: list[str] | tuple[str, ...] | None = None,
    api_key: str | None = None,
    timeout: float = 0.75,
) -> list[DiscoveredEndpoint]:
    """Probe local candidates concurrently and return every reachable endpoint."""
    urls = list(base_urls) if base_urls is not None else candidate_base_urls()
    urls = list(dict.fromkeys(normalize_endpoint(v) for v in urls))
    if not urls:
        return []
    found: list[DiscoveredEndpoint] = []
    workers = min(len(urls), 8)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="xenolect-discovery") as pool:
        futures = {pool.submit(_probe_endpoint, url, api_key, timeout): url for url in urls}
        for future in as_completed(futures):
            try:
                found.append(future.result())
            except Exception:  # one dead/incompatible port must not block discovery
                pass
    order = {url: i for i, url in enumerate(urls)}
    return sorted(found, key=lambda item: order.get(item.base_url, 10_000))


def discover_openai_endpoint(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 0.75,
) -> DiscoveredEndpoint:
    """Compatibility helper for non-interactive callers that require one endpoint."""
    found = scan_openai_endpoints(
        base_urls=[base_url] if base_url else candidate_base_urls(),
        api_key=api_key,
        timeout=timeout,
    )
    if not found:
        raise RuntimeError("no OpenAI-compatible endpoint found")
    if len(found) > 1:
        choices = ", ".join(item.base_url for item in found)
        raise RuntimeError(f"multiple OpenAI-compatible endpoints found ({choices})")
    return found[0]
