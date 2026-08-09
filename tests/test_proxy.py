from __future__ import annotations

import json

import pytest

from xenolect.driver.ir import Driver, ParserKind, SchemaTransform, ToolEncoding, ToolResultEncoding, identity_driver
from xenolect.proxy import ProxyError, ProxyService, ProxyTarget, _chunked_sse, translate_request, translate_response
from xenolect.storage.registry import DriverRegistry


def _tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "weather lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]


def test_translate_native_preserves_openai_history_and_transforms_tools():
    d = Driver(schema_transforms=[SchemaTransform.FORCE_ADDITIONAL_PROPERTIES_FALSE])
    body = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "weather"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"city":"NYC"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": '{"temp":70}'},
        ],
        "tools": _tools(),
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    messages, tools, kwargs = translate_request(body, d, "m")
    assert messages[1]["tool_calls"][0]["id"] == "c1"
    assert messages[2] == {"role": "tool", "content": '{"temp":70}', "tool_call_id": "c1"}
    assert tools is not None
    assert tools[0]["function"]["parameters"]["additionalProperties"] is False
    assert kwargs["temperature"] == 0.2
    assert kwargs["extra_body"] == {"response_format": {"type": "json_object"}}


def test_translate_textual_reencodes_assistant_calls_and_tool_results():
    d = Driver(
        tool_encoding=ToolEncoding.XML_JSON,
        parser=ParserKind.XML_JSON,
        tool_result_encoding=ToolResultEncoding.USER_MESSAGE,
    )
    body = {
        "model": "m",
        "messages": [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "weather"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"city":"NYC"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": '{"temp":70}'},
        ],
        "tools": _tools(),
    }
    messages, tools, _ = translate_request(body, d, "m")
    assert tools is None
    assert messages[0]["role"] == "system"
    assert "Available tools" in messages[0]["content"]
    assistant = next(m for m in messages if m["role"] == "assistant")
    assert "<tool_call>" in assistant["content"]
    assert '"id":"c1"' in assistant["content"]
    result = messages[-1]
    assert result["role"] == "user"
    assert "TOOL_RESULT id=c1 name=weather" in result["content"]


def test_textual_driver_rejects_unrepresentable_tool_choice():
    d = Driver(
        tool_encoding=ToolEncoding.TAGGED_JSON,
        parser=ParserKind.TAGGED_JSON,
        tool_result_encoding=ToolResultEncoding.USER_MESSAGE,
    )
    with pytest.raises(ProxyError, match="tool_choice"):
        translate_request(
            {"messages": [{"role": "user", "content": "x"}], "tools": _tools(), "tool_choice": "none"},
            d,
            "m",
        )


def test_translate_xml_response_to_openai_tool_calls():
    d = Driver(
        tool_encoding=ToolEncoding.XML_JSON,
        parser=ParserKind.XML_JSON,
        tool_result_encoding=ToolResultEncoding.USER_MESSAGE,
    )
    raw = {
        "id": "upstream-id",
        "created": 10,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '<tool_call>{"name":"weather","arguments":{"city":"NYC"}}</tool_call>',
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    out = translate_response(raw, d, "m")
    assert out["model"] == "m"
    assert out["choices"][0]["finish_reason"] == "tool_calls"
    tc = out["choices"][0]["message"]["tool_calls"][0]
    assert tc["id"].startswith("call_xenolect_")
    assert tc["function"]["name"] == "weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "NYC"}
    assert out["usage"]["total_tokens"] == 3


def test_translate_plain_response_is_canonical_chat_completion():
    raw = {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}
    out = translate_response(raw, identity_driver(), "m")
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
    assert out["choices"][0]["finish_reason"] == "stop"


def test_buffered_sse_is_valid_and_finishes():
    raw = {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}
    out = translate_response(raw, identity_driver(), "m")
    chunks = list(_chunked_sse(out))
    assert chunks[-1] == b"data: [DONE]\n\n"
    assert any(b'"content": "hello"' in c for c in chunks)
    assert any(b'"finish_reason": "stop"' in c for c in chunks)


def test_buffered_sse_can_emit_usage_chunk():
    raw = {
        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    out = translate_response(raw, identity_driver(), "m")
    chunks = list(_chunked_sse(out, include_usage=True))
    assert any(b'"choices": []' in c and b'"total_tokens": 2' in c for c in chunks)


def test_proxy_service_uses_installed_driver_and_upstream(tmp_path):
    reg = DriverRegistry(tmp_path)
    installed = reg.install(base_url="http://up/v1", model="m", driver=identity_driver())

    class Client:
        def __init__(self):
            self.messages = None
            self.tools = None
            self.kwargs = None

        def chat_completions(self, messages, tools=None, **kwargs):
            self.messages = messages
            self.tools = tools
            self.kwargs = kwargs
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "abc",
                                    "type": "function",
                                    "function": {"name": "weather", "arguments": '{"city":"NYC"}'},
                                }
                            ],
                        }
                    }
                ]
            }

    client = Client()
    service = ProxyService(ProxyTarget(installed=installed), client=client)  # type: ignore[arg-type]
    out = service.chat_completions(
        {"model": "m", "messages": [{"role": "user", "content": "weather"}], "tools": _tools()}
    )
    assert client.tools is not None
    assert client.kwargs["model"] == "m"
    assert out["choices"][0]["message"]["tool_calls"][0]["id"] == "abc"


def test_proxy_applies_certified_sampling_default_only_when_app_omits_it(tmp_path):
    reg = DriverRegistry(tmp_path)
    installed = reg.install(base_url="http://up/v1", model="m", driver=identity_driver())

    class Client:
        def __init__(self):
            self.kwargs = None

        def chat_completions(self, messages, tools=None, **kwargs):
            self.kwargs = kwargs
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    client = Client()
    service = ProxyService(ProxyTarget(installed=installed), client=client)  # type: ignore[arg-type]
    base = {"model": "m", "messages": [{"role": "user", "content": "hello"}]}

    service.chat_completions(dict(base))
    assert client.kwargs["temperature"] == 0.0

    service.chat_completions({**base, "temperature": 0.7})
    assert client.kwargs["temperature"] == 0.7

    service.chat_completions({**base, "temperature": None})
    assert "temperature" in client.kwargs and client.kwargs["temperature"] is None


def test_proxy_rejects_wrong_model(tmp_path):
    reg = DriverRegistry(tmp_path)
    installed = reg.install(base_url="http://up/v1", model="m", driver=identity_driver())

    class Client:
        def chat_completions(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("must not call upstream")

    service = ProxyService(ProxyTarget(installed=installed), client=Client())  # type: ignore[arg-type]
    with pytest.raises(ProxyError, match="bound to model"):
        service.chat_completions({"model": "other", "messages": [{"role": "user", "content": "x"}]})


def test_textual_auto_tool_choice_is_not_sent_as_native_policy():
    d = Driver(
        tool_encoding=ToolEncoding.XML_JSON,
        parser=ParserKind.XML_JSON,
        tool_result_encoding=ToolResultEncoding.USER_MESSAGE,
    )
    _, _, kwargs = translate_request(
        {"messages": [{"role": "user", "content": "x"}], "tools": _tools(), "tool_choice": "auto"},
        d,
        "m",
    )
    assert "tool_choice" not in kwargs


def test_textual_no_tools_does_not_inject_empty_tool_preamble():
    d = Driver(
        tool_encoding=ToolEncoding.XML_JSON,
        parser=ParserKind.XML_JSON,
        tool_result_encoding=ToolResultEncoding.USER_MESSAGE,
    )
    messages, tools, _ = translate_request(
        {"messages": [{"role": "user", "content": "hello"}]}, d, "m"
    )
    assert tools is None
    assert messages == [{"role": "user", "content": "hello"}]


def test_proxy_rejects_multi_choice_and_legacy_functions():
    with pytest.raises(ProxyError, match="n=1"):
        translate_request({"messages": [{"role": "user", "content": "x"}], "n": 2}, identity_driver(), "m")
    with pytest.raises(ProxyError, match="legacy"):
        translate_request(
            {"messages": [{"role": "user", "content": "x"}], "functions": []},
            identity_driver(),
            "m",
        )


def test_text_finish_reason_is_preserved():
    raw = {
        "choices": [
            {"message": {"role": "assistant", "content": "partial"}, "finish_reason": "length"}
        ]
    }
    out = translate_response(raw, identity_driver(), "m")
    assert out["choices"][0]["finish_reason"] == "length"


def test_multiple_upstream_choices_fail_closed():
    raw = {
        "choices": [
            {"message": {"role": "assistant", "content": "a"}},
            {"message": {"role": "assistant", "content": "b"}},
        ]
    }
    with pytest.raises(ProxyError, match="multiple choices"):
        translate_response(raw, identity_driver(), "m")


def test_http_client_extra_body_forwards_non_protocol_fields_without_overrides():
    import httpx
    from xenolect.endpoints.http import OpenAICompatClient

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    client = OpenAICompatClient(
        "http://up/v1",
        model="real-model",
        transport=httpx.MockTransport(handler),
        max_retries=1,
    )
    client.chat_completions(
        [{"role": "user", "content": "hi"}],
        extra_body={
            "response_format": {"type": "json_object"},
            "model": "must-not-override",
            "messages": [{"role": "user", "content": "bad"}],
        },
    )
    assert captured["model"] == "real-model"
    assert captured["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["response_format"] == {"type": "json_object"}


def test_max_completion_tokens_is_not_silently_renamed():
    _, _, kwargs = translate_request(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 321,
        },
        identity_driver(),
        "m",
    )
    assert "max_tokens" not in kwargs
    assert kwargs["extra_body"]["max_completion_tokens"] == 321


def test_registry_proxy_lists_and_resolves_installed_models(tmp_path):
    from xenolect.proxy import RegistryProxyService

    reg = DriverRegistry(tmp_path)
    reg.install(base_url="http://up1/v1", model="m1", driver=identity_driver())
    reg.install(base_url="http://up2/v1", model="m2", driver=identity_driver())
    service = RegistryProxyService(registry=reg)

    assert [item["id"] for item in service.models()["data"]] == ["m1", "m2"]
    assert service.health()["service"] == "xenolect"
    with pytest.raises(ProxyError, match="multiple models"):
        service.chat_completions({"messages": [{"role": "user", "content": "hi"}]})
    with pytest.raises(ProxyError, match="not installed"):
        service.chat_completions(
            {"model": "missing", "messages": [{"role": "user", "content": "hi"}]}
        )


def test_http_proxy_restricts_cors_and_requires_json(tmp_path):
    import threading

    import httpx
    from http.server import ThreadingHTTPServer

    from xenolect.proxy import RegistryProxyService, make_handler

    reg = DriverRegistry(tmp_path)
    reg.install(base_url="http://up/v1", model="m", driver=identity_driver())
    service = RegistryProxyService(registry=reg)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        remote = httpx.options(
            base + "/v1/chat/completions",
            headers={"Origin": "https://example.com"},
        )
        assert remote.status_code == 403

        local = httpx.options(
            base + "/v1/chat/completions",
            headers={"Origin": "http://localhost:3000"},
        )
        assert local.status_code == 204
        assert local.headers["access-control-allow-origin"] == "http://localhost:3000"

        wrong_type = httpx.post(
            base + "/v1/chat/completions",
            content='{"model":"m","messages":[{"role":"user","content":"hi"}]}',
            headers={"Content-Type": "text/plain"},
        )
        assert wrong_type.status_code == 415

        models = httpx.get(base + "/v1/models?ignored=1")
        assert models.status_code == 200
        assert models.json()["data"][0]["id"] == "m"
    finally:
        server.shutdown()
        server.server_close()
        service.close()
        thread.join(timeout=2)


def test_registry_refresh_does_not_close_inflight_model(monkeypatch, tmp_path):
    import threading

    import xenolect.proxy as proxy_module
    from xenolect.proxy import RegistryProxyService

    reg = DriverRegistry(tmp_path)
    reg.install(base_url="http://up/v1", model="m", driver=identity_driver())

    started = threading.Event()
    release = threading.Event()
    instances = []

    class BlockingProxyService:
        def __init__(self, target):
            self.target = target
            self.closed = False
            instances.append(self)

        def chat_completions(self, body):
            started.set()
            assert release.wait(2)
            assert not self.closed
            return {"ok": True}

        def close(self):
            self.closed = True

    monkeypatch.setattr(proxy_module, "ProxyService", BlockingProxyService)
    service = RegistryProxyService(registry=reg)
    result = {}

    thread = threading.Thread(
        target=lambda: result.update(service.chat_completions({"model": "m", "messages": []})),
        daemon=True,
    )
    thread.start()
    assert started.wait(1)

    # A live ban changes the registry while the request is in flight.  Refresh
    # must remove the model for new requests without closing the client currently
    # serving the old request.
    reg.ban("http://up/v1", "m")
    assert service.models()["data"] == []
    assert instances[0].closed is False

    release.set()
    thread.join(timeout=2)
    assert result == {"ok": True}
    assert instances[0].closed is True
    service.close()


def test_corrupt_driver_is_isolated_from_other_models(tmp_path):
    from xenolect.proxy import RegistryProxyService

    reg = DriverRegistry(tmp_path)
    good = reg.install(base_url="http://good/v1", model="good", driver=identity_driver())
    bad_driver = Driver(
        tool_encoding=ToolEncoding.XML_JSON,
        parser=ParserKind.XML_JSON,
        tool_result_encoding=ToolResultEncoding.USER_MESSAGE,
    )
    bad = reg.install(base_url="http://bad/v1", model="bad", driver=bad_driver)
    assert bad.driver_path != good.driver_path
    bad.driver_path.write_text('{"tool_encoding":"native"}', encoding="utf-8")

    # Strict registry callers still see integrity failure.
    with pytest.raises(Exception):
        reg.list()

    # The live proxy fails closed only for the broken binding; a corrupt artifact
    # must not take every unrelated verified model offline.
    service = RegistryProxyService(registry=reg)
    assert [item["id"] for item in service.models()["data"]] == ["good"]
    assert good.driver_path.is_file()
    service.close()
