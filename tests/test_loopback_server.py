from __future__ import annotations

import socket
from http.server import BaseHTTPRequestHandler

from xenolect.proxy import _LoopbackHTTPServer


def test_loopback_server_bind_never_performs_dns_lookup(monkeypatch) -> None:
    def fail_dns(*_args, **_kwargs):
        raise AssertionError("loopback bind must not perform DNS lookup")

    monkeypatch.setattr(socket, "getfqdn", fail_dns)
    server = _LoopbackHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    try:
        assert server.server_name == "127.0.0.1"
        assert server.server_port > 0
    finally:
        server.server_close()
