# Manual real-model validation

This plan is intentionally separate from deterministic tests and GitHub CI. Its
checks must not be reported as passing until they are run against the named local
models and the evidence is saved.

The purpose is to falsify the implementation, not to tune Xenolect for either
model. A failure should be investigated from fresh wire evidence; it is not a reason
to weaken certification.

## Preconditions

- Windows PowerShell with Python 3.11 or 3.13.
- The v0.5 research commit installed from a clean wheel or checked-out repository.
- Ollama serving its OpenAI-compatible endpoint at
  `http://127.0.0.1:11434/v1`.
- Both `qwen3-4b-ctx4k:latest` and `axiom-cyber:latest` already available locally.
- No other Xenolect service using port 8179.

Use a fresh Xenolect home for each target. This avoids cached Drivers and preserves
the reports and registry from each run without deleting user data.

## Target 1: Qwen3 4B

```powershell
$env:XENOLECT_HOME = Join-Path $env:TEMP ("xenolect-v050-qwen-" + [guid]::NewGuid().ToString("N"))
$targetModel = "qwen3-4b-ctx4k:latest"
$started = Get-Date
py -m xenolect install `
  --base-url "http://127.0.0.1:11434/v1" `
  --model $targetModel `
  --deadline 300 `
  --max-generations 12 `
  --force `
  --verbose
$elapsed = (Get-Date) - $started
py -m xenolect status
Write-Host "Elapsed seconds:" $elapsed.TotalSeconds
Write-Host "Artifacts:" $env:XENOLECT_HOME
```

## Target 2: Axiom Cyber 8B

Stop the first Xenolect service, then use a second fresh home.

```powershell
py -m xenolect kill
$env:XENOLECT_HOME = Join-Path $env:TEMP ("xenolect-v050-axiom-" + [guid]::NewGuid().ToString("N"))
$targetModel = "axiom-cyber:latest"
$started = Get-Date
py -m xenolect install `
  --base-url "http://127.0.0.1:11434/v1" `
  --model $targetModel `
  --deadline 300 `
  --max-generations 12 `
  --force `
  --verbose
$elapsed = (Get-Date) - $started
py -m xenolect status
Write-Host "Elapsed seconds:" $elapsed.TotalSeconds
Write-Host "Artifacts:" $env:XENOLECT_HOME
```

Do not reuse the old result as evidence. This must be a fresh compile with fresh
nonces. The acceptance gate for each target is:

- terminal status `CERTIFIED` / `READY`;
- no more than 300 seconds;
- no more than 12 paid generations;
- three independent certification generations;
- valid G1, G2, and nonce-bound G3 protocol witnesses;
- all 18 obligations certified;
- a Driver and report persisted under the fresh Xenolect home;
- no test-only or model-specific production logic.

If either target fails, preserve its full report and inspect the new evidence. Do not
retry until the failure has first been classified as protocol, model behavior,
infrastructure, configuration, deadline, budget, or certification evidence.

## Unseen runtime holdout and raw isolation

Run an application-representative tool request that was not part of G1/G2/G3. Send
the same OpenAI Chat Completions payload once to the raw endpoint and once through
Xenolect. Use a fresh tool name and fresh argument values. Save both unmodified JSON
responses before interpreting them.

The raw endpoint is:

```text
http://127.0.0.1:11434/v1/chat/completions
```

The Xenolect endpoint is:

```text
http://127.0.0.1:8179/v1/chat/completions
```

Use the same explicit sampling fields in both payloads. Test the certified default
first with `temperature: 0.0`; any higher-temperature run is a separate model
reliability experiment and is not covered by the deterministic protocol certificate.

Classify divergence only after comparing the raw responses:

| Observation | Classification |
|---|---|
| Raw upstream emitted no structured call | `MODEL_BEHAVIOR` |
| Raw upstream emitted a valid call but Xenolect lost or corrupted it | `XENOLECT_RUNTIME_BUG` |
| Xenolect preserved the call, but tool selection or arguments were unsuitable | `MODEL_TOOL_USE_QUALITY` |
| Both paths rejected or malformed the same protocol wire | investigate `PROTOCOL_INCOMPATIBILITY` |

Record the commit SHA, Python version, Windows version, model identifiers, endpoint
version, Xenolect report paths, elapsed times, generation counts, Driver summary, and
raw/Xenolect response files. Do not record API keys or authorization headers.

## Result statement

Only after the commands and holdout are run should a validation note say which gates
passed. A deterministic CI pass alone is not a real-model pass, and one successful
runtime holdout is not evidence of general model tool-use intelligence.
