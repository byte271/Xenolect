# Changelog

## 0.2.0 - 2026-08-08

### Driver protocol IR

- Added a backward-compatible Driver IR v0.2 protocol program composed from typed request, response, tool-result, and state primitives.
- Replaced hardcoded tagged/XML execution branches with parameterized strict JSON framing and field mapping.
- Added multiple-parser agreement checks and optional preservation of assistant text beside tool calls.
- Added bounded, deterministic local discovery of strict whole-content and embedded JSON response parsers, including arbitrary literal frames, custom call fields, mixed assistant text, and multiple calls.
- Parser candidates are inferred from already-paid observations, revalidated across the stateful trajectory, and must still pass fresh production-runtime certification before emission.
- Added a holdout endpoint test proving synthesis of a v0.2 response program outside all three legacy parsers and the previous 144-program grammar.
- XPT retains the proven bounded request frontier, 12-generation budget, three-generation certification reserve, fresh certification, and 300-second default deadline.
- Existing v0.1 Driver serialization and content hashes remain stable; unknown or incomplete v0.2 programs fail closed.

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
- macOS immediate background startup uses the native POSIX spawn path to avoid fragile child-side fork setup.
- The loopback HTTP server bypasses hostname and reverse-DNS lookup during bind, preventing macOS mDNS stalls on `127.0.0.1`.
- Background readiness and model listing do not initialize upstream HTTP clients; connection pools are created lazily on the first real chat request.
- Diagnostic report filenames include a unique suffix so rapid writes cannot collide on platforms with coarse timestamp behavior.
- Loopback-only proxy binding and loopback proxy bypass for local health/upstream traffic.
- Restricted browser CORS, JSON request enforcement, bounded request bodies, logs, and reports.
- Connection reuse and thread-safe registry hot refresh.
- Driver integrity verification and isolation of invalid artifacts.

### Explicit limits

- The v0.1.0 compiler does not synthesize arbitrary state-machine Driver programs outside its 144-program grammar.
- Xenolect does not create application tools; it adapts the tools supplied by the client application.
- Chat Completions + function tool calling only; not a full OpenAI API replacement.
- `stream=true` is buffered SSE compatibility, not token-by-token upstream streaming.
