"""Protocol Syndrome: everything cheaply derivable from ONE raw response.

Design rule: model generation is expensive, parsing and local
evaluation are free. So a single black-box response is decomposed into as many
deterministic observable features as the runtime/ABI semantics justify — never
into a single Boolean.

Parser choice is evaluated *locally* against the
same raw bytes, for every parser in the grammar, with no weakening of strictness.
Acceptance policy:

    exactly one parser yields tool calls          -> UNIQUE   (accept it)
    several parsers yield an identical canonical  -> AGREEING (safe equivalence)
    several parsers yield different canonical AST -> AMBIGUOUS (never guess)
    no parser yields tool calls                   -> TEXT / NONE
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from xenolect.abi.events import AssistantText, AssistantToolCall, ToolCall, ToolCallBatch
from xenolect.driver.ir import Driver, ParserKind
from xenolect.driver.parse import parse_model_response_full

ALL_PARSERS: tuple[ParserKind, ...] = tuple(ParserKind)


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha(value: Any) -> str:
    return hashlib.sha256(_canon(value).encode()).hexdigest()[:16]


class ParseConsensus(str, Enum):
    NONE = "none"          # no parser produced tool calls
    UNIQUE = "unique"      # exactly one parser produced tool calls
    AGREEING = "agreeing"  # several parsers agreed on the identical canonical AST
    AMBIGUOUS = "ambiguous"  # several parsers disagreed -> unsafe, do not choose


@dataclass(frozen=True)
class ParserOutcome:
    parser: ParserKind
    ok: bool                  # parsed without errors
    n_calls: int
    errors: tuple[str, ...]
    ast_hash: str | None      # canonical AST hash when calls were produced
    is_batch: bool
    calls: tuple[ToolCall, ...] = ()
    text: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "parser": self.parser.value,
            "ok": self.ok,
            "n_calls": self.n_calls,
            "errors": list(self.errors),
            "ast_hash": self.ast_hash,
            "is_batch": self.is_batch,
        }


def _canonical_ast(calls: list[ToolCall]) -> str:
    payload = [
        {"name": c.name, "arguments": c.arguments, "has_id": c.id is not None}
        for c in calls
    ]
    return sha(payload)


def evaluate_all_parsers(raw: dict[str, Any]) -> dict[ParserKind, ParserOutcome]:
    """Run every grammar parser over the same raw bytes. Zero endpoint cost."""
    out: dict[ParserKind, ParserOutcome] = {}
    for kind in ALL_PARSERS:
        driver = Driver(parser=kind)
        pr = parse_model_response_full(raw, driver)
        calls: list[ToolCall] = []
        is_batch = False
        text: str | None = None
        for ev in pr.events:
            if isinstance(ev, AssistantToolCall):
                calls.append(ev.call)
            elif isinstance(ev, ToolCallBatch):
                calls.extend(ev.calls)
                is_batch = True
            elif isinstance(ev, AssistantText):
                text = ev.content
        out[kind] = ParserOutcome(
            parser=kind,
            ok=not pr.errors,
            n_calls=len(calls),
            errors=tuple(pr.errors),
            ast_hash=_canonical_ast(calls) if calls else None,
            is_batch=is_batch,
            calls=tuple(calls),
            text=text,
        )
    return out


@dataclass
class Syndrome:
    """Deterministic local decomposition of one black-box response."""

    # transport
    transport_ok: bool = True
    transport_error: str | None = None

    # raw shape
    message_present: bool = False
    native_tool_calls_present: bool = False
    content_text: str = ""

    # multi-parser local evaluation
    parser_outcomes: dict[ParserKind, ParserOutcome] = field(default_factory=dict)
    consensus: ParseConsensus = ParseConsensus.NONE
    accepted_parser: ParserKind | None = None
    compatible_parsers: tuple[ParserKind, ...] = ()

    # normalized observations from the accepted parse
    tool_call_emitted: bool = False
    n_calls: int = 0
    parallel_batch_present: bool = False
    tool_names: tuple[str, ...] = ()
    unknown_tool_names: tuple[str, ...] = ()
    call_ids_present: bool = False
    call_ids_unique: bool = True

    # per-tool local checks (filled by the turn checker)
    args_schema_valid: dict[str, bool] = field(default_factory=dict)
    args_values_correct: dict[str, bool] = field(default_factory=dict)

    # surface hints that cost nothing to collect (never model-family specific)
    saw_xml_marker: bool = False
    saw_tagged_marker: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "transport_ok": self.transport_ok,
            "transport_error": self.transport_error,
            "message_present": self.message_present,
            "native_tool_calls_present": self.native_tool_calls_present,
            "content_text_len": len(self.content_text),
            "parsers": {k.value: v.as_dict() for k, v in self.parser_outcomes.items()},
            "consensus": self.consensus.value,
            "accepted_parser": self.accepted_parser.value if self.accepted_parser else None,
            "compatible_parsers": [p.value for p in self.compatible_parsers],
            "tool_call_emitted": self.tool_call_emitted,
            "n_calls": self.n_calls,
            "parallel_batch_present": self.parallel_batch_present,
            "tool_names": list(self.tool_names),
            "unknown_tool_names": list(self.unknown_tool_names),
            "call_ids_present": self.call_ids_present,
            "call_ids_unique": self.call_ids_unique,
            "args_schema_valid": dict(self.args_schema_valid),
            "args_values_correct": dict(self.args_values_correct),
            "saw_xml_marker": self.saw_xml_marker,
            "saw_tagged_marker": self.saw_tagged_marker,
        }

    def signature(self) -> str:
        """Compact, comparable feature vector used by the offline planner."""
        return _canon(
            {
                "t": self.transport_ok,
                "c": self.consensus.value,
                "p": sorted(p.value for p in self.compatible_parsers),
                "e": self.tool_call_emitted,
                "n": self.n_calls,
                "b": self.parallel_batch_present,
                "names": sorted(self.tool_names),
                "ids": self.call_ids_present,
                "uniq": self.call_ids_unique,
                "sv": dict(sorted(self.args_schema_valid.items())),
                "vv": dict(sorted(self.args_values_correct.items())),
            }
        )


def _resolve_consensus(
    outcomes: dict[ParserKind, ParserOutcome],
) -> tuple[ParseConsensus, ParserKind | None, tuple[ParserKind, ...]]:
    producing = [o for o in outcomes.values() if o.n_calls > 0 and o.ok]
    if not producing:
        return ParseConsensus.NONE, None, ()
    hashes = {o.ast_hash for o in producing}
    compatible = tuple(sorted((o.parser for o in producing), key=lambda p: p.value))
    if len(producing) == 1:
        return ParseConsensus.UNIQUE, producing[0].parser, compatible
    if len(hashes) == 1:
        # Safe equivalence: identical canonical AST from several parsers.
        # Prefer the lowest-complexity parser deterministically.
        order = {ParserKind.NATIVE: 0, ParserKind.XML_JSON: 1, ParserKind.TAGGED_JSON: 2}
        chosen = sorted(producing, key=lambda o: order[o.parser])[0].parser
        return ParseConsensus.AGREEING, chosen, compatible
    return ParseConsensus.AMBIGUOUS, None, compatible


def build_syndrome(
    raw: dict[str, Any] | None,
    *,
    transport_error: str | None = None,
    offered_tool_names: set[str] | None = None,
) -> Syndrome:
    """Decompose one raw endpoint response into the canonical syndrome."""
    if raw is None:
        return Syndrome(transport_ok=False, transport_error=transport_error or "no response")

    syn = Syndrome(transport_ok=True)
    message: dict[str, Any] | None = None
    if "choices" in raw:
        choices = raw.get("choices") or []
        message = (choices[0].get("message") if choices else None) or None
    elif isinstance(raw.get("message"), dict):
        message = raw["message"]
    elif "role" in raw:
        message = raw

    syn.message_present = message is not None
    if message is not None:
        syn.native_tool_calls_present = bool(message.get("tool_calls"))
        content = message.get("content")
        syn.content_text = content if isinstance(content, str) else ""

    syn.saw_xml_marker = "<tool_call" in syn.content_text
    syn.saw_tagged_marker = "TOOL_CALL" in syn.content_text

    syn.parser_outcomes = evaluate_all_parsers(raw)
    syn.consensus, syn.accepted_parser, syn.compatible_parsers = _resolve_consensus(
        syn.parser_outcomes
    )

    if syn.accepted_parser is not None:
        outcome = syn.parser_outcomes[syn.accepted_parser]
        syn.tool_call_emitted = outcome.n_calls > 0
        syn.n_calls = outcome.n_calls
        syn.parallel_batch_present = outcome.is_batch
        syn.tool_names = tuple(c.name for c in outcome.calls)
        ids = [c.id for c in outcome.calls]
        syn.call_ids_present = bool(ids) and all(i is not None for i in ids)
        syn.call_ids_unique = len(set(i for i in ids if i is not None)) == len(
            [i for i in ids if i is not None]
        )
        if offered_tool_names is not None:
            syn.unknown_tool_names = tuple(
                sorted({n for n in syn.tool_names if n not in offered_tool_names})
            )
    return syn
