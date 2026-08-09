# Changelog

## Unreleased

### Repository maintenance

- After successful CI on `main`, repository automation removes only `agent/*`
  branches whose current HEAD exactly matches the head SHA of a merged pull request
  and which have no open pull request. The default branch, tags, active work, and
  unmerged branch heads are excluded.

## 0.5.0 - 2026-08-09

### Reliability and certification semantics

- Redefined OB16 as nonce-bound final-text termination. One exact fresh sentinel may
  appear inside harmless prose; stale, wrong, premature, duplicated, conflicting,
  ambiguous, or tool-call-bearing final output still fails.
- Kept exact response-style compliance as a non-ABI diagnostic so harmless prose no
  longer sends the planner into unrelated protocol search.
- Added a registry-level certified execution profile without changing Driver IR v0.2.
  Compilation, certification, and omitted runtime temperature now agree on `0.0`;
  explicit application settings remain authoritative.
- Persisted redacted reports for every terminal install outcome, including bounded
  ledgers, evidence summaries, obligation coverage, failure class, actual synthesis
  path, and execution assumptions.
- Corrected property-local and synthesis-mode metrics to describe behavior actually
  observed and used by the run rather than compiler capability availability.
- Made release publication depend on successful CI for the exact `main` commit and
  reject pull-request or fork workflow runs.
- Preserved the 300-second deadline, 12-generation limit, three-generation
  certification reserve, 33 request versions, three result versions, v0.2 Driver
  compatibility, and v0.1 hash `ee80c9b78784`.

This release improves the distinction between protocol incompatibility and imperfect
model behavior. It does not add provider-specific rules, grammar expansion, arbitrary
protocol synthesis, generated code, or state-machine synthesis.

## 0.4.0 - 2026-08-09

### Oracle-free diagnostic synthesis

- Added a typed, non-persistent Diagnostic Probe IR with nonce-bound positive
  witnesses and auditable predicted outcome partitions.
- Added deterministic minimax request and result probe planning over the fixed 33 × 3
  production version space.
- Kept generic rejection, silence, prose, ambiguity, and unexpected output
  non-eliminating unless a valid exclusive witness exists.
- Added exhaustive offline separability tests, generated oracle-free holdouts, a
  non-identifiable endpoint, and candidate-only ablation.
- Preserved the three-generation clean production certification boundary and all
  v0.3 compatibility guarantees.

The bounded oracle-free family can synthesize and certify a working Driver without
target values or property-local fault localization. Endpoints without discriminating
positive behavior still fail closed.

## 0.3.0 - 2026-08-09

### Active, obligation-directed synthesis

- Added typed partial hypotheses for request, response, and tool-result components
  while keeping state actions fixed.
- Added reusable component facts, obligation support, complete turn witnesses, and a
  separate independent-certification boundary.
- Added bounded request and result version spaces with controlled interventions and
  deterministic information/obligation-gain ranking.
- Allowed only deterministic structural, wire/API, and parser/schema contradictions
  to eliminate hypotheses; ordinary model non-compliance remains heuristic evidence.
- Added generated non-cooperative holdouts requiring previously unseen request,
  response, and tool-result programs.
- Preserved Driver IR v0.2, the 12-generation budget, and the 300-second deadline.

## 0.2.0 - 2026-08-08

### Composable Driver protocol IR

- Added backward-compatible Driver IR v0.2 request, response, tool-result, and fixed
  state primitives.
- Replaced fixed textual parser branches with strict parameterized JSON framing and
  field mapping.
- Added bounded local discovery of whole-content and embedded JSON parsers, arbitrary
  frames, mixed text/calls, and multiple calls.
- Added a holdout proving discovery of a response program outside the original
  144-program grammar.
- Preserved v0.1 serialization and hashes.

## 0.1.0

### First alpha release

- Added one-command local model discovery, bounded black-box Driver compilation,
  independent Tool ABI certification, and a persistent verified registry.
- Added the loopback Chat Completions proxy and `install`, `status`, `ban`, `kill`, and
  `version` commands.
- Added per-user startup support for Windows, macOS, and Linux without administrator
  or root privileges where the desktop session provides a standard mechanism.
- Added the finite 144-program v0.1 grammar, stateful tool-result/history diagnosis,
  fail-closed unsupported behavior, integrity checks, bounded diagnostics, and local
  security controls.

The first release supported Chat Completions and function tools only. It did not
claim arbitrary state-machine synthesis or a complete OpenAI API replacement.
