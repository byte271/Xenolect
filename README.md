# Xenolect v0.5.0

Xenolect is a local protocol compatibility layer for model servers that expose an
OpenAI-style Chat Completions API.

It probes a selected model, builds a bounded Driver, independently certifies that
Driver, saves it locally, and exposes a reusable loopback API for applications.

> Xenolect is an alpha release. It adapts tool-calling protocols; it does not make a
> weak model smarter and it is not a complete OpenAI API replacement.

## What Xenolect does

`xenolect install` performs one bounded setup run:

1. find an OpenAI-compatible model endpoint;
2. test request, response, and tool-result protocol behavior;
3. synthesize the lowest-complexity supported Driver that fits the evidence;
4. certify it on a fresh three-turn Tool ABI trace;
5. save the Driver only if all 18 mandatory obligations pass;
6. start a local API at `http://127.0.0.1:8179/v1`.

The default setup limit is 300 seconds and 12 model generations. Three generations
are always reserved for independent certification, so exploration can use at most
nine.

Unsupported or ambiguous behavior fails closed. Xenolect never invents a Driver just
to make setup appear successful.

## Supported systems

- Windows 10 or newer
- macOS 12 or newer
- Linux on a supported Python platform
- Python 3.11 or newer

One pure-Python wheel supports all three desktop platforms.

## Install

Download `xenolect-0.5.0-py3-none-any.whl` from the
[v0.5.0 release](https://github.com/byte271/Xenolect/releases/tag/v0.5.0).

Windows PowerShell:

```powershell
py -m pip install .\xenolect-0.5.0-py3-none-any.whl
xenolect install
```

macOS or Linux:

```bash
python3 -m pip install ./xenolect-0.5.0-py3-none-any.whl
xenolect install
```

For a source checkout:

```bash
python3 -m pip install .
xenolect install
```

Keep the Python installation available. Xenolect's login-start service uses the same
interpreter that installed the package.

## Use it from an application

Choose an OpenAI-compatible or custom OpenAI provider in the application and enter:

```text
Base URL  http://127.0.0.1:8179/v1
Model     the model selected during xenolect install
API key   any non-empty placeholder, only if the app requires one
```

The placeholder API key is not local authentication. Xenolect binds to loopback and
must not be exposed to a LAN or the public internet.

## Commands

```text
xenolect install          Set up or restore compatibility
xenolect status           Show service status
xenolect status --verbose Show diagnostic details
xenolect ban              Hide or restore a model
xenolect kill             Stop Xenolect and disable login startup
xenolect version          Show the installed version
```

A cached Driver is reused when the endpoint/model binding is unchanged and its
artifact still passes integrity checks.

## What certification means

A certificate proves that the unchanged Driver completed Xenolect's low-difficulty
G1/G2/G3 protocol trace through the production runtime under the recorded execution
profile. It covers structured calls, names, arguments, parallel calls, call IDs,
tool-result association and consumption, history, ToolError recovery, unambiguous
parsing, no spurious final call, nonce-bound termination, and a legal completed trace.

It does not prove that the model will choose the correct tool for every unseen task,
follow every prose-format instruction, or remain equally reliable under arbitrary
sampling settings.

v0.5 records `temperature=0.0` as the certified runtime default. An application's
explicit numeric temperature remains authoritative. Explicit `null` opts into the
endpoint default instead. Other omitted sampling and tool-policy fields remain at the
endpoint default and are not certified across every possible value.

G3 accepts a normal final assistant sentence containing one exact, fresh,
boundary-delimited result sentinel. Exact prose equality is recorded as an optional
instruction-following diagnostic, not as protocol compatibility. Stale, wrong,
premature, duplicated, conflicting, ambiguous, or tool-call-bearing final output
still fails.

See [Certification and failure semantics](docs/CERTIFICATION.md) for the complete
evidence rules and report fields.

## Failure reports

Every terminal compile outcome writes a redacted report under:

```text
~/.xenolect/reports
```

This includes success, budget or deadline exhaustion, infrastructure failure,
configuration failure, unsupported protocols, and independent-certification failure.
The CLI prints the saved path.

Reports include generation counts, the bounded wire ledger, obligation coverage,
failed obligations, evidence summaries, planner decisions, the actual synthesis path,
and the certified execution profile. API keys, authorization values, URL credentials,
and sensitive query parameters are redacted.

## API surface

v0.5 supports:

- `GET /v1/models`
- `POST /v1/chat/completions`
- OpenAI-style messages and function tools
- assistant `tool_calls` and multi-turn tool-result history
- one completion choice (`n=1`)
- buffered SSE compatibility for `stream=true`

Buffered SSE begins only after Xenolect receives the complete upstream response. It
is not token-by-token upstream streaming.

Not supported or claimed:

- universal or arbitrary protocol synthesis
- arbitrary state-machine or generated-code synthesis
- recovery of unobservable secret protocol literals
- automatic creation or execution of application tools
- `/v1/responses`, embeddings, audio, or image APIs
- legacy `functions` / `function_call`
- multiple completion choices
- compatibility under every optional sampling setting

Fields outside the certified tool-calling path may be forwarded on a best-effort
basis. Unrepresentable behavior fails explicitly.

## Runtime holdouts

Certification intentionally uses simple semantics. For an unseen runtime task, first
compare the raw upstream response with Xenolect's normalized response:

| Observation | Meaning |
|---|---|
| Raw upstream emitted no call | model behavior |
| Raw upstream emitted a valid call but Xenolect lost it | Xenolect runtime bug |
| Xenolect preserved the call but the tool or arguments were poor | model tool-use quality |

The manual Windows plan is in
[Real-model validation](docs/REAL_MODEL_VALIDATION.md). It is separate from CI and
must not be reported as passing unless it was actually run against the local models.

## Local data and security

Xenolect stores registry state, verified Drivers, service configuration, bounded
reports, and logs under `~/.xenolect`.

- the service listens on loopback only;
- browser cross-origin access is restricted to loopback origins;
- chat requests require JSON;
- request bodies, reports, and logs are bounded;
- upstream API keys are not stored in Driver artifacts or registry bindings.

Any process running as the same local user can connect to the loopback service. v0.5
does not provide local client authentication. Credentialed remote endpoints are not
the primary setup target.

## Platform startup

- Windows uses the current user's Startup folder.
- macOS uses a per-user LaunchAgent.
- Linux uses a systemd user service when available, otherwise a freedesktop autostart
  entry when the desktop supports it.

No administrator or root access is required. `xenolect status --verbose` reports when
the current session has no supported login-start mechanism.

## Development

```bash
python3 -m venv .venv
python3 -m pip install -e ".[dev]"
python3 -m pytest -q
python3 -m ruff check xenolect tests
python3 -m compileall -q xenolect
python3 -m build
```

Driver IR remains v0.2. v0.1 canonical serialization remains unchanged, including
the frozen reference hash `ee80c9b78784`.

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

Apache License 2.0. See [LICENSE](LICENSE).
