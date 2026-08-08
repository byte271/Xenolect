# Xenolect v0.1.0

Xenolect is a local compatibility layer for model servers that expose an OpenAI-style Chat Completions API.

It probes a selected model through black-box interactions, compiles a verified **Driver** from the v0.1.0 typed compatibility grammar, stores that Driver locally, and runs one loopback endpoint that existing OpenAI-style apps can use.

> **v0.1.0 is an alpha release.** It is intentionally small and focused on Chat Completions + function tool calling. It is not a full implementation of every OpenAI API, and it does not synthesize arbitrary protocol programs yet.

## Supported platforms

Xenolect v0.1.0 supports:

- Windows 10/11
- macOS
- Linux
- Python 3.11 or newer

The same `py3-none-any` wheel is used on all three platforms.

Xenolect starts its local service immediately after setup. It also registers a best-effort per-user login start mechanism without administrator/root access:

- Windows: user Startup folder
- macOS: `~/Library/LaunchAgents/io.xenolect.service.plist`
- Linux: user systemd unit when systemd is available; desktop autostart fallback otherwise

Failure to register login startup does **not** invalidate a successfully prepared Driver or a currently running Xenolect service. `xenolect status --verbose` shows the actual state.

## What Xenolect really compiles

This distinction matters.

Xenolect v0.1.0 does not invent application tools such as search, weather, filesystem access, or shell commands. Your app supplies its normal function tools.

Xenolect compiles the **protocol Driver used to carry those tools through the selected model**. On every request it can transform the app's actual tool schemas, change how tools are presented to the model, parse model-emitted calls, translate tool-result history, and normalize the response back to OpenAI-style `tool_calls`.

The v0.1.0 Driver grammar is deliberately finite:

- 3 tool-request encodings: native, tagged JSON text, XML+JSON text
- 3 independent schema transforms, giving 8 transform subsets
- 3 response parsers
- 2 tool-result encodings

That is **144 representable Driver programs** in v0.1.0.

Xenolect uses black-box probing and a stateful certification trajectory to infer and verify one of those programs. It does not brute-force all 144 against the real model, but it also does not claim to generate arbitrary new state-machine logic outside this grammar. If the model needs behavior the grammar cannot express, setup fails rather than pretending compatibility.

Driver IR v0.2 adds an incremental composable foundation without widening the expensive online search budget. XPT still begins with the proven request configurations from the legacy 144-program frontier, then composes the observed request presentation, response parser, tool-result renderer, schema transforms, and required batch-state actions into a parameterized protocol program. The runtime executes those primitives rather than branching directly on three format names.

The parameterized primitives can represent native calls, strict whole-content JSON calls, custom tagged or XML-style JSON frames, custom object field names, multiple agreeing response parsers, segmented tool-result messages, and assistant text carried beside tool calls. Existing v0.1 `.mdriver` files retain their previous JSON shape and content hash.

XPT can now make one specific synthesis step beyond that fixed grammar: when the legacy response parsers produce no calls, it deterministically inspects the already-paid response for strict whole-content or embedded JSON tool objects, infers custom field names and adjacent literal delimiters, and validates the resulting response primitive across the remaining trajectory. Extraction is bounded; ambiguous field mappings, call IDs, or parser results fail closed. A Driver is still emitted only after the independent fresh-instance certification path passes.

This is not a claim of arbitrary protocol synthesis. Request and tool-result exploration remain bounded, arbitrary state transitions remain unsupported, and every compile stays inside the same 12-generation and 300-second defaults with three generations reserved for certification.

The diagnostic tools used during `xenolect install` are temporary conformance probes. They are not the tools your app later uses. At runtime, Xenolect transforms the real tools supplied by your app according to the installed Driver.

## Requirements

- Python 3.11 or newer.
- A running model server with `GET /v1/models` and `POST /v1/chat/completions`.

Common local servers are scanned automatically. If none is found, Xenolect asks for another port or server address.

## Install

### Windows

```powershell
py -m pip install .\xenolect-0.1.0-py3-none-any.whl
xenolect install
```

If Windows cannot find the launcher:

```powershell
py -m xenolect install
```

### macOS

```bash
python3 -m pip install ./xenolect-0.1.0-py3-none-any.whl
xenolect install
```

If `xenolect` is not on `PATH`:

```bash
python3 -m xenolect install
```

### Linux

```bash
python3 -m pip install ./xenolect-0.1.0-py3-none-any.whl
xenolect install
```

If `xenolect` is not on `PATH`:

```bash
python3 -m xenolect install
```

Some distributions mark the system Python as externally managed. In that case install the wheel into a Python environment you intend to keep, then run `xenolect install` from that environment. The background service uses the same Python interpreter that installed Xenolect.

### Install from source (developers)

```bash
python3 -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install .
xenolect install
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install .
xenolect install
```

Keep that environment in place if you use a source/venv installation, because login startup points to its Python interpreter.

## Normal setup

Run:

```text
xenolect install
```

Xenolect will:

1. check the local environment;
2. scan for model servers;
3. show the models it found;
4. ask you to choose one;
5. ask for another port/address if no usable model is found;
6. probe and compile a compatibility Driver from the v0.1.0 grammar;
7. verify it on a fresh stateful tool trajectory;
8. save the verified Driver;
9. start the local Xenolect service;
10. enable per-user login startup when the current platform supports it;
11. show the address and model name to enter in your app.

After setup, you do **not** need to run a separate server command.

Typical result:

```text
Qwen3 4B is ready

✓ Model connected
✓ Compatibility prepared
✓ Verified
✓ Xenolect is running
✓ <platform> login startup enabled

Address  http://127.0.0.1:8179/v1
Model    qwen3-4b-ctx4k:latest
API key  Any non-empty value (only if your app requires one)
```

The API-key field above is only a placeholder for clients that refuse an empty value. **Xenolect v0.1.0 does not use that value as local authentication.**

## Commands

### Set up or restore Xenolect

```text
xenolect install
```

A previously verified model can normally be reused without rebuilding compatibility.

### Check status

```text
xenolect status
```

For diagnostic details:

```text
xenolect status --verbose
```

### Ban or restore a model

```text
xenolect ban
```

Banning hides a model from the local Xenolect API without deleting its verified Driver. Running `xenolect ban` again can restore it without recompilation.

### Stop Xenolect

```text
xenolect kill
```

This stops the local service and attempts to disable the current platform's per-user login startup. Prepared Drivers are kept.

### Show version

```text
xenolect version
```

## API surface in v0.1.0

Xenolect exposes a local loopback service and supports:

- `GET /v1/models`
- `POST /v1/chat/completions`
- OpenAI-style `messages`
- function tools / `tool_calls`
- multi-turn tool-result history
- one completion choice (`n=1`)
- `stream=true` as **buffered SSE compatibility streaming**
- `stream_options.include_usage` when upstream usage data is available

Buffered streaming means Xenolect first receives the complete upstream completion, then emits valid SSE chunks. **v0.1.0 does not provide token-by-token upstream streaming latency.**

The following are **not** claimed by v0.1.0:

- arbitrary request, tool-result, or state-machine Driver synthesis; the v0.2 milestone currently adds bounded response-parser synthesis only
- creation of application tools or tool implementations
- `/v1/responses`
- embeddings
- audio or image APIs
- legacy `functions` / `function_call`
- multiple completion choices (`n>1`)
- complete behavioral compatibility for every optional OpenAI request field
- a full OpenAI API replacement

Fields outside the verified tool-calling path may be forwarded to the upstream server on a best-effort basis. Unsupported behavior fails explicitly rather than being silently invented.

## Local security model

The background service binds to a loopback address only. It is not intended to be exposed on a LAN or the public internet. Any process running as your local user can still connect to a loopback service; v0.1.0 does not provide local client authentication.

Browser cross-origin access is restricted to loopback origins. Chat requests require JSON content type. Xenolect does not store upstream API keys in its Driver registry.

Credentialed remote upstreams are not a first-class persistent setup in v0.1.0; the release is primarily intended for local model servers.

## Local data

Xenolect stores its local state under:

```text
~/.xenolect
```

This includes the model registry, verified Driver artifacts, service configuration, and bounded diagnostic reports/logs.

## Platform notes

### Windows

Login startup uses the current user's Startup folder. No administrator rights are required.

### macOS

Login startup uses a per-user LaunchAgent. No root privileges are required. If macOS removes or blocks the Python environment that installed Xenolect, reinstall Xenolect with a persistent Python installation and run `xenolect install` again.

### Linux

On systemd-based systems, Xenolect registers a user service for login startup. On non-systemd desktop sessions it writes a freedesktop autostart entry instead. No root privileges are required. Very minimal/headless non-systemd environments may not provide a standard per-user login-start facility; Xenolect still runs immediately after `install`, and `status --verbose` reports whether autostart was actually registered.

## Troubleshooting

**No model found**  
Start your model server, run `xenolect install` again, then enter its port or OpenAI-style base URL if automatic scanning does not find it.

**A model was banned**  
Run `xenolect ban` and select the banned model to restore it.

**Xenolect is not running**  
Run `xenolect install`. A cached model should start quickly without repeating compatibility preparation unless the observable model descriptor changed.

**Login startup is not enabled**  
The Driver and current service can still be valid. Run `xenolect status --verbose` to see the service state, then run `xenolect install` again after fixing permissions or the Python installation if needed.

**Need technical details**  
Use:

```text
xenolect install --verbose
xenolect status --verbose
```

## License

Apache License 2.0. See `LICENSE`.
