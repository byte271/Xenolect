"""OpenAI-compatible HTTP client with discovery, failure domains and hard deadlines."""

from __future__ import annotations

import ipaddress
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import httpx

from xenolect.endpoints.errors import ClientError, FailureDomain, classify_http_status




def _is_loopback_url(value: str) -> bool:
    try:
        host = urlsplit(value).hostname
    except ValueError:
        return False
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class OpenAICompatClient:
    """Minimal OpenAI-compatible client used by the compiler runtime.

    The client deliberately depends only on observable HTTP behaviour.  Model
    discovery uses ``GET /models`` and never branches on provider/model identity.
    ``deadline_s`` is an absolute budget relative to client construction; every
    HTTP attempt is capped to the remaining time so a single slow request cannot
    silently overrun the compiler wall-clock budget.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        model: str = "unknown",
        timeout: float = 120.0,
        *,
        temperature: float | None = 0.0,
        top_p: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        max_retries: int = 3,
        deadline_s: float | None = None,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.perf_counter,
        started_at: float | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_retries = max_retries
        self.deadline_s = deadline_s
        self._transport = transport
        self._clock = clock
        # Optional shared epoch with XptSession so client/session remaining_s agree.
        self._started_at = clock() if started_at is None else started_at
        self.last_attempts: list[dict[str, Any]] = []
        self.last_request_body: dict[str, Any] | None = None
        self.last_response_body: Any = None
        self.last_status_code: int | None = None
        self.interaction_log: list[dict[str, Any]] = []
        # Reuse one connection pool across generations/runtime requests.  Creating
        # a new httpx.Client for every turn adds avoidable TCP/TLS setup cost and
        # is especially wasteful during multi-turn tool use.
        self._client = httpx.Client(
            timeout=self.timeout,
            transport=self._transport,
            # Local model traffic must not be diverted by HTTP(S)_PROXY. Remote
            # endpoints retain normal environment-proxy behavior.
            trust_env=not _is_loopback_url(self.base_url),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenAICompatClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def elapsed_s(self) -> float:
        return self._clock() - self._started_at

    @property
    def remaining_s(self) -> float | None:
        if self.deadline_s is None:
            return None
        return self.deadline_s - self.elapsed_s

    def _effective_timeout(self) -> float:
        remaining = self.remaining_s
        if remaining is None:
            return self.timeout
        if remaining <= 0:
            raise ClientError(
                domain=FailureDomain.INFRASTRUCTURE,
                message="compiler wall-clock deadline exhausted before HTTP request",
                retryable=False,
            )
        return min(self.timeout, max(0.05, remaining))

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generation_config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        if self.temperature is not None:
            cfg["temperature"] = self.temperature
        if self.top_p is not None:
            cfg["top_p"] = self.top_p
        if self.max_tokens is not None:
            cfg["max_tokens"] = self.max_tokens
        if self.seed is not None:
            cfg["seed"] = self.seed
        return cfg

    def list_model_records(self) -> list[dict[str, Any]]:
        """Return generic records advertised by ``GET /models``.

        Xenolect treats the selected record only as observable endpoint metadata;
        it never branches on provider/model-family names.  A canonical hash of
        this record can invalidate a cached driver when the server exposes a
        changed model descriptor under the same id.
        """
        url = f"{self.base_url}/models"
        t0 = self._clock()
        try:
            resp = self._client.get(
                url,
                headers=self._headers(),
                timeout=self._effective_timeout(),
            )
        except httpx.TimeoutException as exc:
            raise ClientError(
                domain=FailureDomain.INFRASTRUCTURE,
                message=f"model discovery timeout: {exc}",
                retryable=False,
            ) from exc
        except httpx.HTTPError as exc:
            raise ClientError(
                domain=FailureDomain.INFRASTRUCTURE,
                message=f"model discovery failed: {exc}",
                retryable=False,
            ) from exc

        latency_ms = (self._clock() - t0) * 1000.0
        self.last_status_code = resp.status_code
        self.last_response_body = resp.text
        self.interaction_log.append(
            {
                "kind": "model_discovery",
                "method": "GET",
                "url": url,
                "status": resp.status_code,
                "latency_ms": latency_ms,
            }
        )
        if resp.status_code >= 400:
            domain, retryable = classify_http_status(resp.status_code)
            raise ClientError(
                domain=domain,
                message=f"GET /models returned HTTP {resp.status_code}: {resp.text[:200]}",
                status_code=resp.status_code,
                retryable=retryable,
            )
        try:
            payload = resp.json()
        except ValueError as exc:
            raise ClientError(
                domain=FailureDomain.CONFIGURATION,
                message="GET /models returned non-JSON content",
                status_code=resp.status_code,
                retryable=False,
            ) from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise ClientError(
                domain=FailureDomain.CONFIGURATION,
                message="GET /models response has no OpenAI-compatible data[] list",
                status_code=resp.status_code,
                retryable=False,
            )
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id or model_id in seen:
                continue
            seen.add(model_id)
            records.append(dict(item))
        return records

    def list_models(self) -> list[str]:
        """Return model IDs advertised by ``GET /models`` in server order."""
        return [str(item["id"]) for item in self.list_model_records()]

    def discover_model(self, requested: str | None = None) -> str:
        """Resolve and store the model id without using model-family knowledge."""
        if requested and requested != "unknown":
            self.model = requested
            return self.model
        models = self.list_models()
        if not models:
            raise ClientError(
                domain=FailureDomain.CONFIGURATION,
                message="endpoint advertised no model ids",
                retryable=False,
            )
        if len(models) != 1:
            raise ClientError(
                domain=FailureDomain.CONFIGURATION,
                message=(
                    "endpoint advertises multiple models; automatic selection would be ambiguous: "
                    + ", ".join(models)
                    + ". Pass --model explicitly."
                ),
                retryable=False,
                details={"models": models},
            )
        self.model = models[0]
        return self.model

    def chat_completions(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        clean_messages = [
            {k: v for k, v in m.items() if not str(k).startswith("_")} for m in messages
        ]
        body: dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": clean_messages,
        }
        if tools is not None:
            body["tools"] = tools

        temp = kwargs.get("temperature", self.temperature)
        if temp is not None:
            body["temperature"] = temp
        top_p = kwargs.get("top_p", self.top_p)
        if top_p is not None:
            body["top_p"] = top_p
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        seed = kwargs.get("seed", self.seed)
        if seed is not None:
            body["seed"] = seed
        if "tool_choice" in kwargs:
            body["tool_choice"] = kwargs["tool_choice"]

        extra_body = kwargs.get("extra_body")
        if extra_body is not None:
            if not isinstance(extra_body, dict):
                raise ClientError(
                    domain=FailureDomain.CONFIGURATION,
                    message="extra_body must be an object",
                    retryable=False,
                )
            for key, value in extra_body.items():
                if key not in {"model", "messages", "tools"}:
                    body[key] = value

        url = f"{self.base_url}/chat/completions"
        self.last_attempts = []
        self.last_request_body = body
        self.last_response_body = None
        self.last_status_code = None
        attempt = 0
        last_err: ClientError | None = None
        while attempt < max(1, self.max_retries):
            attempt += 1
            t0 = self._clock()
            try:
                resp = self._client.post(
                    url,
                    json=body,
                    headers=self._headers(),
                    timeout=self._effective_timeout(),
                )
                latency = (self._clock() - t0) * 1000
                selected_headers = {
                    k.lower(): v
                    for k, v in resp.headers.items()
                    if k.lower() in {"retry-after", "x-request-id", "cf-ray", "server", "date"}
                }
                self.last_status_code = resp.status_code
                self.last_response_body = resp.text
                self.last_attempts.append(
                    {
                        "attempt": attempt,
                        "status": resp.status_code,
                        "latency_ms": latency,
                        "response_headers": selected_headers,
                    }
                )
                if resp.status_code >= 400:
                    domain, retryable = classify_http_status(resp.status_code)
                    err = ClientError(
                        domain=domain,
                        message=f"HTTP {resp.status_code}: {resp.text[:200]}",
                        status_code=resp.status_code,
                        retryable=retryable,
                        details={
                            "request_body": body,
                            "response_text": resp.text,
                            "response_headers": selected_headers,
                            "attempt": attempt,
                        },
                    )
                    last_err = err
                    if retryable and attempt < self.max_retries:
                        continue
                    raise err
                try:
                    parsed = resp.json()
                except ValueError as exc:
                    raise ClientError(
                        domain=FailureDomain.PROTOCOL,
                        message="HTTP 200 response was not valid JSON",
                        status_code=resp.status_code,
                        retryable=False,
                        details={
                            "request_body": body,
                            "response_text": resp.text,
                            "response_headers": selected_headers,
                        },
                    ) from exc
                if not isinstance(parsed, dict):
                    raise ClientError(
                        domain=FailureDomain.PROTOCOL,
                        message="HTTP 200 response JSON was not an object",
                        status_code=resp.status_code,
                        retryable=False,
                        details={
                            "request_body": body,
                            "response_text": resp.text,
                            "response_headers": selected_headers,
                        },
                    )
                self.last_response_body = parsed
                self.interaction_log.append(
                    {
                        "kind": "chat_completion",
                        "model": body.get("model"),
                        "status": resp.status_code,
                        "latency_ms": latency,
                        "attempts": list(self.last_attempts),
                    }
                )
                return parsed
            except ClientError:
                raise
            except httpx.TimeoutException as exc:
                last_err = ClientError(
                    domain=FailureDomain.INFRASTRUCTURE,
                    message=f"timeout: {exc}",
                    retryable=True,
                )
                self.last_attempts.append({"attempt": attempt, "error": "timeout"})
                if attempt < self.max_retries and (self.remaining_s is None or self.remaining_s > 0):
                    continue
                raise last_err from exc
            except httpx.HTTPError as exc:
                last_err = ClientError(
                    domain=FailureDomain.INFRASTRUCTURE,
                    message=str(exc),
                    retryable=True,
                )
                self.last_attempts.append({"attempt": attempt, "error": str(exc)})
                if attempt < self.max_retries and (self.remaining_s is None or self.remaining_s > 0):
                    continue
                raise last_err from exc

        assert last_err is not None
        raise last_err

    def fingerprint(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
        }
