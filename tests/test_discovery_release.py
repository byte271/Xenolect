from __future__ import annotations

from xenolect.endpoints.discovery import DEFAULT_LOCAL_BASE_URLS, candidate_base_urls, normalize_endpoint


def test_environment_hint_does_not_hide_default_local_scan(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9999/v1")
    urls = candidate_base_urls()
    assert urls[0] == "http://127.0.0.1:9999/v1"
    for default in DEFAULT_LOCAL_BASE_URLS:
        assert default in urls


def test_endpoint_normalization_accepts_common_user_inputs() -> None:
    assert normalize_endpoint("11434") == "http://127.0.0.1:11434/v1"
    assert normalize_endpoint("localhost:11434") == "http://localhost:11434/v1"
    assert normalize_endpoint("http://localhost:11434/v1/models") == "http://localhost:11434/v1"
    assert normalize_endpoint("http://localhost:11434/v1/chat/completions") == "http://localhost:11434/v1"
