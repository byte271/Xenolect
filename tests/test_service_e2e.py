from __future__ import annotations

import json
import os
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from xenolect.driver.ir import identity_driver
from xenolect.service import ensure_background_service, stop_background_service
from xenolect.storage.registry import DriverRegistry


class _Upstream(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        payload = {
            "id": "chatcmpl-service-e2e",
            "object": "chat.completion",
            "created": 1,
            "model": body.get("model", "m"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "service-e2e-ok"},
                    "finish_reason": "stop",
                }
            ],
        }
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def test_real_background_process_models_chat_and_kill(tmp_path: Path, monkeypatch) -> None:
    # The child process must be able to import the source tree even when this test
    # is run without first installing the project. CI installs editable as well.
    repo_root = str(Path(__file__).resolve().parents[1])
    old_pythonpath = os.environ.get("PYTHONPATH")
    combined = repo_root if not old_pythonpath else os.pathsep.join([repo_root, old_pythonpath])
    monkeypatch.setenv("PYTHONPATH", combined)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    upstream_url = f"http://127.0.0.1:{upstream.server_port}/v1"

    xhome = tmp_path / "xhome"
    DriverRegistry(xhome).install(
        base_url=upstream_url,
        model="service-e2e",
        driver=identity_driver(),
    )

    state = ensure_background_service(
        home=xhome,
        preferred_port=19179,
        enable_autostart=False,
    )
    try:
        with urllib.request.urlopen(state.base_url + "/models", timeout=3) as response:  # noqa: S310 - loopback
            models = json.loads(response.read())
        assert [item["id"] for item in models["data"]] == ["service-e2e"]

        request = urllib.request.Request(
            state.base_url + "/chat/completions",
            data=json.dumps(
                {
                    "model": "service-e2e",
                    "messages": [{"role": "user", "content": "hello"}],
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 - loopback
            chat = json.loads(response.read())
        assert chat["choices"][0]["message"]["content"] == "service-e2e-ok"
    finally:
        stopped = stop_background_service(home=xhome, disable_autostart=True)
        upstream.shutdown()
        upstream.server_close()
    assert stopped.running is False
