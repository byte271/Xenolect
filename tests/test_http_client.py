"""HTTP client integrity tests (fix-3 PASS 4)."""

import httpx
import pytest

from xenolect.endpoints.errors import ClientError, FailureDomain
from xenolect.endpoints.http import OpenAICompatClient


def _handler_factory(status: int, body: bytes = b"{}"):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body)

    return handler


def test_http_401_configuration():
    transport = httpx.MockTransport(_handler_factory(401, b'{"error":"no"}'))
    c = OpenAICompatClient("http://test/v1", transport=transport, max_retries=1)
    with pytest.raises(ClientError) as ei:
        c.chat_completions([{"role": "user", "content": "hi"}])
    assert ei.value.domain == FailureDomain.CONFIGURATION


def test_http_429_retries_then_fails():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, content=b'{"error":"rate"}')

    transport = httpx.MockTransport(handler)
    c = OpenAICompatClient("http://test/v1", transport=transport, max_retries=3)
    with pytest.raises(ClientError) as ei:
        c.chat_completions([{"role": "user", "content": "hi"}])
    assert ei.value.domain == FailureDomain.INFRASTRUCTURE
    assert calls["n"] == 3


def test_http_success_and_generation_params():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"total_tokens": 3},
            },
        )

    transport = httpx.MockTransport(handler)
    c = OpenAICompatClient(
        "http://test/v1",
        transport=transport,
        temperature=0.0,
        max_tokens=16,
        seed=7,
    )
    out = c.chat_completions([{"role": "user", "content": "hi"}])
    assert out["choices"][0]["message"]["content"] == "ok"
    assert seen["body"]["temperature"] == 0.0
    assert seen["body"]["max_tokens"] == 16
    assert seen["body"]["seed"] == 7


def test_classify_400_as_protocol():
    from xenolect.endpoints.errors import classify_http_status

    domain, retryable = classify_http_status(400)
    assert domain == FailureDomain.PROTOCOL
    assert retryable is False


def test_http_200_invalid_json_is_protocol_error_with_raw_evidence():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"not-json"))
    c = OpenAICompatClient("http://test/v1", transport=transport, max_retries=1)
    with pytest.raises(ClientError) as ei:
        c.chat_completions([{"role": "user", "content": "hi"}])
    assert ei.value.domain == FailureDomain.PROTOCOL
    assert ei.value.details and ei.value.details["response_text"] == "not-json"


def test_loopback_http_client_disables_environment_proxy(monkeypatch):
    import xenolect.endpoints.http as http_module

    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def close(self):
            pass

    monkeypatch.setattr(http_module.httpx, "Client", FakeClient)
    client = http_module.OpenAICompatClient("http://127.0.0.1:11434/v1")
    client.close()
    assert captured["trust_env"] is False


def test_remote_http_client_keeps_environment_proxy_support(monkeypatch):
    import xenolect.endpoints.http as http_module

    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def close(self):
            pass

    monkeypatch.setattr(http_module.httpx, "Client", FakeClient)
    client = http_module.OpenAICompatClient("https://models.example/v1")
    client.close()
    assert captured["trust_env"] is True
