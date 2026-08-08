# Changelog

## 0.1.0

First alpha release of Xenolect's local model compatibility layer.

### Product

- One-command interactive `xenolect install` flow.
- Local model discovery and model selection.
- Persistent verified Driver registry with content-addressed artifacts.
- One background loopback proxy shared by all installed models.
- `install`, `status`, `ban`, `kill`, and `version` user commands.
- Per-user login startup on Windows, macOS, and Linux without administrator/root privileges where the OS session provides a standard user-start mechanism.

### Driver compiler

- Real black-box probing against the selected endpoint.
- v0.1.0 finite typed Driver grammar: 144 representable protocol programs.
- Stateful tool-result/history diagnosis and fresh certification before installation.
- Driver-backed runtime translation of client-supplied tools, assistant tool calls, and tool-result history.
- Fail-closed behavior when the required compatibility cannot be represented or certified.

### Stability and security

- Hard 12-generation and 300-second default setup budgets.
- Safe background-service upgrade/restart behavior.
- Loopback-only proxy binding and loopback proxy bypass for local health/upstream traffic.
- Restricted browser CORS, JSON request enforcement, bounded request bodies, logs, and reports.
- Connection reuse and thread-safe registry hot refresh.
- Driver integrity verification and isolation of invalid artifacts.

### Explicit limits

- The v0.1.0 compiler does not synthesize arbitrary state-machine Driver programs outside its 144-program grammar.
- Xenolect does not create application tools; it adapts the tools supplied by the client application.
- Chat Completions + function tool calling only; not a full OpenAI API replacement.
- `stream=true` is buffered SSE compatibility, not token-by-token upstream streaming.
