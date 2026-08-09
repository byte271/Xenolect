# Certification and failure semantics

This page defines the v0.5 protocol-certificate boundary. The short version is:
Xenolect certifies a Driver under recorded request defaults; it does not certify a
model's general reasoning or instruction-following quality.

## Certification boundary

Diagnosis and certification are separate.

- Diagnosis may select and refine protocol hypotheses from paid observations.
- Diagnostic probes can distinguish hypotheses but cannot prove production Tool ABI
  obligations.
- Independent certification creates fresh nonces and executes the final unchanged
  Driver through the same runtime used after installation.
- The Driver is persisted only when all 18 mandatory obligations pass.

The fixed defaults are:

| Limit | Value |
|---|---:|
| Wall clock | 300 seconds |
| Total generations | 12 |
| Certification reserve | 3 |
| Maximum exploration generations | 9 |

Driver IR remains v0.2. v0.5 does not add provider rules,
grammar entries, arbitrary state transitions, or generated executable code.

## Evidence classes

Xenolect keeps these meanings separate:

| Evidence | What it can do |
|---|---|
| Component or structural fact | Support one protocol component |
| Ordinary negative model behavior | Inform ranking; cannot eliminate a hypothesis |
| Complete diagnosis witness | Support only the obligations witnessed by its full turn sequence |
| Deterministic structural, wire/API, or parser contradiction | Safely eliminate the contradicted hypothesis |
| Independent certification | Authoritatively decide all mandatory obligations |

A successful G1 call does not prove history preservation, recovery, termination, or
legal completion. A parser working on G1 does not prove later tool-result consumption.
One stochastic non-call response does not prove that a protocol is impossible.

## G3 and OB16

OB16 is nonce-bound final-text termination. It is not exact natural-language style.

A valid G3 witness requires:

1. normal runtime completion;
2. no parser disagreement or parse error;
3. no tool call after the final recovery results;
4. exactly one exact sentinel scalar in the controlled `report` result;
5. no occurrence of that sentinel in material available before the result;
6. exactly one boundary-delimited copy of the sentinel in final assistant text;
7. no conflicting member of the generated nonce family.

The sentinel is created after the prompt and becomes model-visible only through the
final tool result. This makes its valid appearance positive evidence of result
consumption. Random prose cannot predict it, and stale values from another run do not
match it.

These examples describe shape only; real tests use newly generated values.

| Final observation | Result |
|---|---|
| Exact fresh sentinel | pass |
| Harmless sentence containing the one fresh sentinel | pass; exact-style diagnostic is false |
| Stale, wrong, missing, duplicated, or substring-colliding sentinel | fail |
| Sentinel was visible before result injection | fail |
| Correct sentinel plus another tool call | fail |
| Ambiguous parser outcome or parse error | fail |

OB15 independently requires no spurious final tool turn. OB17 independently requires
unambiguous parsing. OB18 remains the legal-completed-trace boundary.

## Certified execution profile

The execution profile is registry and report metadata, not Driver serialization.

Compilation and independent certification use:

```text
temperature = 0.0
```

They omit `top_p`, `max_tokens`, `max_completion_tokens`, `seed`,
`parallel_tool_calls`, and `tool_choice`, leaving those fields at endpoint defaults.

At runtime:

- omitted temperature receives the certified `0.0` default;
- an explicit numeric temperature is preserved;
- explicit `null` suppresses the profile default and uses the endpoint default;
- other explicit application fields remain authoritative when representable.

Certification therefore means protocol compatibility at the recorded request
defaults, not reliability across every sampling policy. Existing v0.4 registry entries
load with a migration profile reflecting the v0.4 compiler's `temperature=0.0`
behavior. Malformed or unknown profile versions fail closed.

## Failure classes

Terminal results separate:

- protocol or parser incompatibility;
- ordinary model behavior or style deviation;
- infrastructure failure;
- endpoint configuration failure;
- budget exhaustion;
- endpoint deadline exhaustion;
- insufficient evidence or unsupported behavior;
- independent-certification failure.

A harmless final prose wrapper is recorded as an instruction-following style
deviation when the protocol witness is otherwise complete. It does not send the
planner into unrelated request or result protocol search.

Synthesis mode reports the path actually used by a compilation:

- `bounded_obligation_directed_cegis`;
- `bounded_active_discriminating_synthesis`;
- `bounded_oracle_free_diagnostic_synthesis`.

Supported compiler capabilities are reported separately from features used by one
run. Property-local counters increase only when a strict property-local API rejection
was observed and materially used.

## Reports

Every terminal install compilation writes a collision-safe JSON report under
`~/.xenolect/reports`. Reports contain:

- status, reason, failure class, model, and redacted endpoint identity;
- wall time and diagnosis/certification generation counts;
- the bounded request/response ledger;
- evidence and obligation summaries;
- failed obligations and planner decisions;
- actual synthesis mode and usage counters;
- independent certificate and execution profile.

API keys, authorization values, URL credentials, and sensitive query parameters are
redacted. The CLI displays the saved path for failed setup.

## Oracle-free boundary

Oracle-free probes use positive, nonce-bound structured witnesses. Silence, plain-text
canary repetition, generic rejection, unexpected output, parser disagreement, or
multiple competing witnesses cannot eliminate hypotheses.

If the endpoint exposes neither a discriminating positive behavior nor logically
usable deterministic evidence, Xenolect reports the hypothesis space as
observationally unidentifiable or unsupported and fails closed rather than guessing.

The narrow v0.5 claim is that Xenolect can distinguish protocol failure from harmless
response-style deviation more accurately while retaining its bounded, independently
certified protocol contract. It is not a claim of universal protocol synthesis or
general model reliability.
