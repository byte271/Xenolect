# Changelog

## Unreleased

## 0.3.0 - 2026-08-09

### Active discriminating protocol synthesis

- Added explicit bounded version spaces for request and tool-result primitive properties, with controlled interventions and deterministic survivor accounting.
- Added property-local ordinary API rejection handling. Errors identify only the rejected parameter/value already present on the wire; they never disclose an accepted value or target Driver. Generic invalid-value errors and ordinary non-call behavior cannot shrink the version space.
- Added deterministic information-gain and obligation-gain experiment ranking. Ranking is recorded as non-proof, and each refinement preserves unrelated component evidence.
- Added synthesis of invented textual request programs, embedded or framed response programs, and structured result renderers across message placement, catalog depth, schema projection, field mapping, result placement, and result association choices.
- Added a non-cooperative generated five-seed holdout sweep. The endpoint returns only normal tool behavior or ordinary API errors, and all seeds synthesize distinct unseen programs and independently certify in 6–9 diagnosis generations plus the reserved three certification generations.

This stacked research milestone proves one narrow claim: XPT can design discriminating black-box experiments and synthesize a certified working request + response + tool-result protocol without being given the target protocol format. It does not claim arbitrary protocol or state-machine synthesis.

### Obligation-directed active protocol synthesis

- Added typed partial protocol hypotheses with unresolved request, response, and tool-result holes; unresolved hypotheses cannot become Driver artifacts.
- Added reusable per-component structural facts tied to generation and wire hashes, separate obligation-support rows, turn-scoped complete witnesses, and a distinct independent-certification boundary.
- Added deterministic obligation-directed experiment planning using the existing Tool ABI component attribution. Ranking signals never eliminate hypotheses.
- Added a bounded counterexample-guided loop that can infer a nested JSON catalog and framed call sample from a rejected wire, reuse the response-parser discovery pass, and recover a segmented tool-result renderer from a fresh-sentinel example.
- Added nonce binding, strict bounds, conflict detection, deterministic-contradiction-only elimination, and component-isolated refinement. Ordinary negative model behavior remains ranking evidence and cannot permanently eliminate a hypothesis. State actions remain fixed.
- Added a procedurally generated end-to-end holdout outside the legacy request, response, and result frontier; all literal parameters are inferred locally, bounded message-placement choices are tested, and the final program is discovered in eight diagnosis generations and independently certified in three.
- Expanded compiler reports with component constraints, hypothesis fingerprints/revisions, implicated obligations, and certification survival reasons.
- Preserved v0.1 JSON/hash identity, v0.2 compatibility, the 12-generation budget, three-generation certification reserve, and 300-second deadline.

This milestone proves one narrow claim: XPT can synthesize and independently certify one previously unseen request + response + tool-result program from black-box observations. It does not claim universal protocol or state-machine synthesis.

### Compatibility and verification

- Preserved v0.1 Driver JSON serialization and content-hash identity and retained v0.2 Driver compatibility.
- Kept the 300-second wall-clock limit, 12-generation budget, and three-generation independent-certification reserve.
- Verified 122 tests and all six Windows, macOS, and Linux jobs on Python 3.11 and 3.13 before release.

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
