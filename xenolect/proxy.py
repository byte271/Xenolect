"""Local Chat Completions proxy backed by an installed Xenolect driver.

The proxy is deliberately a transport shell around the verified Driver IR:
client-visible Chat Completions requests/responses keep the OpenAI-style shape
while model-facing history/tools are translated according to the installed driver.

Streaming is *buffered compatibility streaming*: Xenolect must see a complete
model response before a textual tool frame can be parsed soundly, so the first
version buffers the upstream completion and then emits valid SSE chunks.  It
does not claim token-by-token latency.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterable
from urllib.parse import urlsplit

from xenolect import __version__
from xenolect.abi.events import AssistantText, AssistantToolCall, ToolCall, ToolCallBatch, ToolDef, ToolResult
from xenolect.driver.encode import (
    build_system_tool_preamble,
    encode_tool_result_message,
    should_send_native_tools,
    tools_for_request,
)
from xenolect.driver.ir import Driver, ToolEncoding
from xenolect.driver.parse import parse_model_response_full, strict_json_loads
from xenolect.endpoints.errors import ClientError
from xenolect.endpoints.http import OpenAICompatClient
from xenolect.storage.registry import DriverRegistry, InstalledDriver, RegistryError


class ProxyError(RuntimeError):
    """A client-visible proxy error with an HTTP status."""

    def __init__(self, message: str, *, status: int = 400, code: str = "xenolect_proxy_error") -> None:
        self.status = int(status)
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class ProxyTarget:
    installed: InstalledDriver
    api_key: str | None = None
    timeout: float = 120.0

    @property
    def base_url(self) -> str:
        return self.installed.base_url

    @property
    def model(self) -> str:
        return self.installed.model

    @property
    def driver(self) -> Driver:
        return self.installed.load()


_RESERVED_REQUEST_KEYS = {
    "model",
    "messages",
    "tools",
    "stream",
    "stream_options",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "seed",
    "tool_choice",
    "parallel_tool_calls",
    "n",
    "functions",
    "function_call",
}


def _tool_defs(raw_tools: Any) -> list[ToolDef]:
    if raw_tools is None:
        return []
    if not isinstance(raw_tools, list):
        raise ProxyError("tools must be a list")
    out: list[ToolDef] = []
    for i, item in enumerate(raw_tools):
        if not isinstance(item, dict) or item.get("type") != "function":
            raise ProxyError(f"tools[{i}] is not a function tool")
        fn = item.get("function")
        if not isinstance(fn, dict):
            raise ProxyError(f"tools[{i}].function must be an object")
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ProxyError(f"tools[{i}] has no valid function name")
        params = fn.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(params, dict):
            raise ProxyError(f"tools[{i}].function.parameters must be an object")
        out.append(
            ToolDef(
                name=name,
                description=fn.get("description") if isinstance(fn.get("description"), str) else None,
                parameters=params,
            )
        )
    return out


def _decode_arguments(raw: Any, *, context: str) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
    if raw is None or raw == "":
        return {}
    if not isinstance(raw, str):
        raise ProxyError(f"{context} arguments must be JSON text/object/array")
    try:
        return strict_json_loads(raw)
    except json.JSONDecodeError as exc:
        raise ProxyError(f"{context} has malformed arguments JSON: {exc.msg}") from exc


def _canonical_call(tc: Any, *, context: str) -> ToolCall:
    if not isinstance(tc, dict):
        raise ProxyError(f"{context} must be an object")
    fn = tc.get("function")
    if not isinstance(fn, dict):
        raise ProxyError(f"{context}.function must be an object")
    name = fn.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ProxyError(f"{context} has no valid function name")
    cid = tc.get("id")
    if cid is not None and not isinstance(cid, str):
        raise ProxyError(f"{context}.id must be a string")
    return ToolCall(
        id=cid,
        name=name,
        arguments=_decode_arguments(fn.get("arguments", "{}"), context=context),
    )


def _textual_call(call: ToolCall, encoding: ToolEncoding) -> str:
    obj: dict[str, Any] = {"name": call.name, "arguments": call.arguments}
    if call.id is not None:
        obj["id"] = call.id
    payload = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    if encoding == ToolEncoding.TAGGED_JSON:
        return f"TOOL_CALL {payload}"
    if encoding == ToolEncoding.XML_JSON:
        return f"<tool_call>{payload}</tool_call>"
    raise AssertionError("textual call requested for native encoding")


def translate_request(body: dict[str, Any], driver: Driver, upstream_model: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, dict[str, Any]]:
    """Translate one client OpenAI request to the model-facing wire.

    Returns ``(messages, tools, generation_kwargs)``.  The translation is
    stateless because an OpenAI request carries its complete conversation
    history; call-id/name associations are reconstructed while walking it.
    """
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ProxyError("messages must be a non-empty list")
    if body.get("n", 1) != 1:
        raise ProxyError(
            "Tool ABI v0 proxy supports exactly one completion choice (n=1)",
            code="unsupported_n",
        )
    if "functions" in body or "function_call" in body:
        raise ProxyError(
            "legacy functions/function_call bypass the installed Driver; use tools/tool_choice",
            code="legacy_function_call_unsupported",
        )

    tool_defs = _tool_defs(body.get("tools"))
    wire_tools = tools_for_request(tool_defs, driver) if (tool_defs and should_send_native_tools(driver)) else None

    if driver.tool_encoding != ToolEncoding.NATIVE:
        if body.get("tool_choice") not in (None, "auto"):
            raise ProxyError(
                "tool_choice other than auto is not representable by this installed textual-tool driver",
                code="unsupported_tool_choice",
            )
        if "parallel_tool_calls" in body:
            raise ProxyError(
                "parallel_tool_calls request policy is not representable by this textual-tool driver",
                code="unsupported_parallel_tool_policy",
            )

    call_names: dict[str, str] = {}
    messages: list[dict[str, Any]] = []
    for i, raw in enumerate(raw_messages):
        if not isinstance(raw, dict):
            raise ProxyError(f"messages[{i}] must be an object")
        role = raw.get("role")
        if role in {"system", "developer", "user"}:
            msg = {k: v for k, v in raw.items() if not str(k).startswith("_")}
            # Model selection belongs at request level, never inside history.
            msg.pop("model", None)
            messages.append(msg)
            continue

        if role == "assistant":
            calls_raw = raw.get("tool_calls") or []
            if calls_raw:
                if not isinstance(calls_raw, list):
                    raise ProxyError(f"messages[{i}].tool_calls must be a list")
                calls = [
                    _canonical_call(tc, context=f"messages[{i}].tool_calls[{j}]")
                    for j, tc in enumerate(calls_raw)
                ]
                for call in calls:
                    if call.id is not None:
                        call_names[call.id] = call.name
                if driver.tool_encoding == ToolEncoding.NATIVE:
                    msg = {k: v for k, v in raw.items() if not str(k).startswith("_")}
                    messages.append(msg)
                else:
                    pieces: list[str] = []
                    content = raw.get("content")
                    if isinstance(content, str) and content:
                        pieces.append(content)
                    pieces.extend(_textual_call(c, driver.tool_encoding) for c in calls)
                    messages.append({"role": "assistant", "content": "\n".join(pieces)})
            else:
                messages.append({k: v for k, v in raw.items() if not str(k).startswith("_")})
            continue

        if role == "tool":
            call_id = raw.get("tool_call_id")
            if call_id is not None and not isinstance(call_id, str):
                raise ProxyError(f"messages[{i}].tool_call_id must be a string")
            name = raw.get("name") if isinstance(raw.get("name"), str) else None
            if name is None and isinstance(call_id, str):
                name = call_names.get(call_id)
            result = ToolResult(call_id=call_id, name=name, content=raw.get("content", ""))
            messages.append(encode_tool_result_message(result, driver))
            continue

        raise ProxyError(f"messages[{i}] has unsupported role {role!r}")

    preamble = build_system_tool_preamble(tool_defs, driver) if tool_defs else None
    if preamble:
        messages.insert(0, {"role": "system", "content": preamble})

    kwargs: dict[str, Any] = {"model": upstream_model}
    for key in ("temperature", "top_p", "seed"):
        if key in body:
            kwargs[key] = body[key]
    if driver.tool_encoding == ToolEncoding.NATIVE and "tool_choice" in body:
        kwargs["tool_choice"] = body["tool_choice"]
    if driver.tool_encoding == ToolEncoding.NATIVE and "parallel_tool_calls" in body:
        kwargs.setdefault("extra_body", {})["parallel_tool_calls"] = body["parallel_tool_calls"]
    if "max_tokens" in body:
        kwargs["max_tokens"] = body["max_tokens"]
    elif "max_completion_tokens" in body:
        # Preserve the client's OpenAI field instead of silently changing its
        # semantics for upstreams that distinguish the two token limits.
        kwargs.setdefault("extra_body", {})["max_completion_tokens"] = body[
            "max_completion_tokens"
        ]

    extra = {k: v for k, v in body.items() if k not in _RESERVED_REQUEST_KEYS}
    if extra:
        kwargs.setdefault("extra_body", {}).update(extra)
    return messages, wire_tools, kwargs


def _stable_call_id(call: ToolCall, index: int) -> str:
    if call.id:
        return call.id
    material = json.dumps(
        {"name": call.name, "arguments": call.arguments, "index": index},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "call_xenolect_" + hashlib.sha256(material).hexdigest()[:16]


def _canonical_tool_call(call: ToolCall, index: int) -> dict[str, Any]:
    return {
        "id": _stable_call_id(call, index),
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(call.arguments, separators=(",", ":"), ensure_ascii=False),
        },
    }


def translate_response(raw: dict[str, Any], driver: Driver, client_model: str) -> dict[str, Any]:
    """Normalize a model response into OpenAI-compatible assistant output."""
    if not isinstance(raw, dict):
        raise ProxyError(
            "upstream response must be a JSON object",
            status=502,
            code="invalid_upstream_response",
        )
    parsed = parse_model_response_full(raw, driver)
    if parsed.errors:
        raise ProxyError(
            "installed driver could not parse upstream response: " + "; ".join(parsed.errors),
            status=502,
            code="driver_parse_error",
        )

    calls: list[ToolCall] = []
    text_parts: list[str] = []
    for event in parsed.events:
        if isinstance(event, AssistantText):
            text_parts.append(event.content)
        elif isinstance(event, AssistantToolCall):
            calls.append(event.call)
        elif isinstance(event, ToolCallBatch):
            calls.extend(event.calls)

    if calls and text_parts:
        # Current ABI parsers do not intentionally produce mixed events.  Fail
        # closed if a future parser does, rather than silently dropping content.
        raise ProxyError(
            "driver produced mixed assistant text and tool calls, unsupported by Tool ABI v0",
            status=502,
            code="mixed_assistant_output",
        )

    message: dict[str, Any] = {"role": "assistant", "content": None if calls else "".join(text_parts)}
    finish_reason = "stop"
    if calls:
        message["tool_calls"] = [_canonical_tool_call(c, i) for i, c in enumerate(calls)]
        finish_reason = "tool_calls"

    source_choice = {}
    choices = raw.get("choices") if isinstance(raw, dict) else None
    if isinstance(choices, list):
        if len(choices) > 1:
            raise ProxyError(
                "upstream returned multiple choices but Tool ABI v0 proxy is single-choice",
                status=502,
                code="multiple_upstream_choices",
            )
        if choices and isinstance(choices[0], dict):
            source_choice = choices[0]
    if not calls and isinstance(source_choice.get("finish_reason"), str):
        finish_reason = source_choice["finish_reason"]

    out: dict[str, Any] = {
        "id": raw.get("id") if isinstance(raw.get("id"), str) else "chatcmpl-xenolect-" + hashlib.sha256(repr(raw).encode()).hexdigest()[:16],
        "object": "chat.completion",
        "created": raw.get("created") if isinstance(raw.get("created"), int) else int(time.time()),
        "model": client_model,
        "choices": [
            {
                "index": int(source_choice.get("index", 0) or 0),
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
    }
    if "usage" in raw:
        out["usage"] = raw["usage"]
    if "system_fingerprint" in raw:
        out["system_fingerprint"] = raw["system_fingerprint"]
    return out


class ProxyService:
    """One installed driver bound to one upstream endpoint/model."""

    def __init__(self, target: ProxyTarget, *, client: OpenAICompatClient | None = None) -> None:
        self.target = target
        self.driver = target.driver
        self._client = client
        self._client_lock = threading.Lock()

    def _client_for_request(self) -> OpenAICompatClient:
        client = self._client
        if client is not None:
            return client
        with self._client_lock:
            if self._client is None:
                self._client = OpenAICompatClient(
                    base_url=self.target.base_url,
                    api_key=self.target.api_key,
                    model=self.target.model,
                    timeout=self.target.timeout,
                    temperature=None,
                    max_retries=1,
                )
            return self._client

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "xenolect",
            "version": __version__,
            "pid": os.getpid(),
            "models": [self.target.model],
        }

    def models(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.target.model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "xenolect",
                }
            ],
        }

    def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
        requested_model = body.get("model", self.target.model)
        if requested_model not in (None, self.target.model):
            raise ProxyError(
                f"this proxy is bound to model {self.target.model!r}, got {requested_model!r}",
                status=404,
                code="model_not_installed",
            )
        messages, tools, kwargs = translate_request(body, self.driver, self.target.model)
        try:
            raw = self._client_for_request().chat_completions(messages, tools=tools, **kwargs)
        except ClientError as exc:
            status = exc.status_code if exc.status_code and 400 <= exc.status_code <= 599 else 502
            raise ProxyError(str(exc), status=status, code=f"upstream_{exc.domain.value}") from exc
        return translate_response(raw, self.driver, self.target.model)

    def close(self) -> None:
        with self._client_lock:
            client = self._client
            self._client = None
        if client is not None:
            client.close()



class RegistryProxyService:
    """Dynamic proxy over all verified drivers in one registry.

    The registry file is refreshed only when its mtime changes.  This keeps the
    steady-state request path light while allowing `xenolect install` to add a
    model without restarting the background service.
    """

    def __init__(
        self,
        *,
        registry: DriverRegistry | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.registry = registry or DriverRegistry()
        self.api_key = api_key
        self.timeout = timeout
        self._registry_stamp: tuple[int, int, int] | None = None
        self._services: dict[str, ProxyService] = {}
        self._loaded = False
        # ThreadingHTTPServer may serve several clients while install/ban updates
        # the registry.  Refreshing must never close a client that another request
        # is actively using.
        self._lock = threading.RLock()
        self._active: dict[int, int] = {}
        self._retired: dict[int, ProxyService] = {}

    def _stamp(self) -> tuple[int, int, int] | None:
        try:
            stat = self.registry.path.stat()
            return (stat.st_mtime_ns, stat.st_size, getattr(stat, "st_ino", 0))
        except FileNotFoundError:
            return None

    def _retire_locked(self, service: ProxyService) -> None:
        sid = id(service)
        if self._active.get(sid, 0) > 0:
            self._retired[sid] = service
        else:
            service.close()

    def _refresh_locked(self) -> None:
        stamp = self._stamp()
        if stamp == self._registry_stamp and self._loaded:
            return
        try:
            installed = self.registry.list(skip_invalid=True)
        except RegistryError as exc:
            raise ProxyError(str(exc), status=500, code="registry_error") from exc

        # The most recently installed binding wins when the same model id exists
        # on multiple local endpoints.  The user selected that binding during
        # install, and installing it later is an explicit preference signal.
        by_model: dict[str, InstalledDriver] = {}
        for item in sorted(installed, key=lambda value: value.installed_at):
            by_model[item.model] = item

        services: dict[str, ProxyService] = {}
        for model, item in by_model.items():
            existing = self._services.get(model)
            if existing is not None and existing.target.installed == item:
                services[model] = existing
            else:
                if existing is not None:
                    self._retire_locked(existing)
                services[model] = ProxyService(
                    ProxyTarget(installed=item, api_key=self.api_key, timeout=self.timeout)
                )
        for model, existing in self._services.items():
            if model not in services:
                self._retire_locked(existing)
        self._services = services
        self._registry_stamp = stamp
        self._loaded = True

    def _refresh(self) -> None:
        with self._lock:
            self._refresh_locked()

    def _release(self, service: ProxyService) -> None:
        sid = id(service)
        with self._lock:
            count = self._active.get(sid, 0) - 1
            if count > 0:
                self._active[sid] = count
                return
            self._active.pop(sid, None)
            retired = self._retired.pop(sid, None)
            if retired is not None:
                retired.close()

    def health(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            models = sorted(self._services)
        return {
            "status": "ok",
            "service": "xenolect",
            "version": __version__,
            "pid": os.getpid(),
            "models": models,
        }

    def models(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            models = sorted(self._services)
        return {
            "object": "list",
            "data": [
                {"id": model, "object": "model", "created": 0, "owned_by": "xenolect"}
                for model in models
            ],
        }

    def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            requested = body.get("model")
            if requested is None:
                if len(self._services) == 1:
                    requested = next(iter(self._services))
                elif not self._services:
                    raise ProxyError(
                        "no model is installed; run `xenolect install`",
                        status=503,
                        code="no_model_installed",
                    )
                else:
                    raise ProxyError(
                        "multiple models are installed; include the model field",
                        code="model_required",
                    )
            if not isinstance(requested, str) or requested not in self._services:
                raise ProxyError(
                    f"model {requested!r} is not installed in Xenolect",
                    status=404,
                    code="model_not_installed",
                )
            service = self._services[requested]
            sid = id(service)
            self._active[sid] = self._active.get(sid, 0) + 1
        try:
            return service.chat_completions(body)
        finally:
            self._release(service)

    def close(self) -> None:
        with self._lock:
            services = {id(service): service for service in self._services.values()}
            services.update(self._retired)
            self._services.clear()
            self._retired.clear()
            self._loaded = False
            self._registry_stamp = None
        for service in services.values():
            service.close()

def _chunked_sse(response: dict[str, Any], *, include_usage: bool = False) -> Iterable[bytes]:
    choice = response["choices"][0]
    msg = choice["message"]
    base = {
        "id": response["id"],
        "object": "chat.completion.chunk",
        "created": response["created"],
        "model": response["model"],
    }

    first = {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
    yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n".encode("utf-8")

    if msg.get("tool_calls"):
        delta_calls = []
        for i, tc in enumerate(msg["tool_calls"]):
            delta_calls.append(
                {
                    "index": i,
                    "id": tc["id"],
                    "type": "function",
                    "function": tc["function"],
                }
            )
        second_delta = {"tool_calls": delta_calls}
    else:
        second_delta = {"content": msg.get("content") or ""}
    second = {**base, "choices": [{"index": 0, "delta": second_delta, "finish_reason": None}]}
    yield f"data: {json.dumps(second, ensure_ascii=False)}\n\n".encode("utf-8")

    final = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}]}
    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n".encode("utf-8")
    if include_usage and "usage" in response:
        usage = {**base, "choices": [], "usage": response["usage"]}
        yield f"data: {json.dumps(usage, ensure_ascii=False)}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"


def _loopback_origin(origin: str | None) -> str | None:
    if not origin:
        return None
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    host = parsed.hostname.lower()
    if host == "localhost":
        return origin
    try:
        if ipaddress.ip_address(host).is_loopback:
            return origin
    except ValueError:
        return None
    return None


def make_handler(service: ProxyService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = f"Xenolect/{__version__}"

        def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover - console plumbing
            if os.getenv("XENOLECT_DEBUG_PROXY") == "1":
                print(f"[xenolect] {self.address_string()} - {fmt % args}")

        def _common_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            allowed = _loopback_origin(self.headers.get("Origin"))
            if allowed is not None:
                self.send_header("Access-Control-Allow-Origin", allowed)
                self.send_header("Vary", "Origin")

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self._common_headers()
            self.end_headers()
            self.wfile.write(data)

        def _error(self, exc: ProxyError) -> None:
            self._json(
                exc.status,
                {
                    "error": {
                        "message": str(exc),
                        "type": "xenolect_error",
                        "code": exc.code,
                    }
                },
            )

        def do_OPTIONS(self) -> None:  # noqa: N802
            origin = self.headers.get("Origin")
            allowed = _loopback_origin(origin)
            if origin and allowed is None:
                self._json(
                    HTTPStatus.FORBIDDEN,
                    {"error": {"message": "cross-origin access is not allowed", "type": "xenolect_error"}},
                )
                return
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Access-Control-Allow-Headers", "authorization, content-type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self._common_headers()
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if self.headers.get("Origin") and _loopback_origin(self.headers.get("Origin")) is None:
                self._json(403, {"error": {"message": "cross-origin access is not allowed", "type": "xenolect_error"}})
                return
            path = urlsplit(self.path).path.rstrip("/")
            if path in {"/v1/models", "/models"}:
                self._json(200, service.models())
                return
            if path in {"/health", "/v1/health"}:
                self._json(200, service.health())
                return
            self._json(404, {"error": {"message": "not found", "type": "xenolect_error"}})

        def do_POST(self) -> None:  # noqa: N802
            if self.headers.get("Origin") and _loopback_origin(self.headers.get("Origin")) is None:
                self._json(403, {"error": {"message": "cross-origin access is not allowed", "type": "xenolect_error"}})
                return
            path = urlsplit(self.path).path.rstrip("/")
            if path not in {"/v1/chat/completions", "/chat/completions"}:
                self._json(404, {"error": {"message": "not found", "type": "xenolect_error"}})
                return
            try:
                content_type = self.headers.get_content_type()
                if content_type != "application/json" and not content_type.endswith("+json"):
                    raise ProxyError("Content-Type must be application/json", status=415)
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 16 * 1024 * 1024:
                    raise ProxyError("invalid request body size", status=413)
                try:
                    body = json.loads(self.rfile.read(size))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ProxyError("request body must be valid JSON") from exc
                if not isinstance(body, dict):
                    raise ProxyError("request JSON must be an object")
                response = service.chat_completions(body)
                if body.get("stream") is True:
                    stream_options = body.get("stream_options")
                    if stream_options is not None and not isinstance(stream_options, dict):
                        raise ProxyError("stream_options must be an object")
                    if isinstance(stream_options, dict):
                        unknown = set(stream_options) - {"include_usage"}
                        if unknown:
                            raise ProxyError(
                                "unsupported stream_options: " + ", ".join(sorted(unknown)),
                                code="unsupported_stream_options",
                            )
                    include_usage = bool(
                        isinstance(stream_options, dict) and stream_options.get("include_usage") is True
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self._common_headers()
                    self.end_headers()
                    for chunk in _chunked_sse(response, include_usage=include_usage):
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    self.close_connection = True
                else:
                    self._json(200, response)
            except ProxyError as exc:
                self._error(exc)
            except Exception as exc:  # noqa: BLE001
                self._error(ProxyError(str(exc), status=500, code="internal_error"))

    return Handler


def serve(service: ProxyService, *, host: str = "127.0.0.1", port: int = 8179) -> None:
    """Run the blocking local proxy server until interrupted."""
    server = ThreadingHTTPServer((host, port), make_handler(service))
    try:
        server.serve_forever()
    finally:
        server.server_close()
        close = getattr(service, "close", None)
        if callable(close):
            close()
