"""Deterministic nonce-bound final-turn protocol evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from xenolect.abi.events import (
    AssistantText,
    AssistantToolCall,
    Event,
    ToolCallBatch,
    ToolResult,
)

_TOKEN_CHAR = r"\w-"
_NONCE_SUFFIX = re.compile(r"^(.*?)([0-9A-F]{4,64})$")


@dataclass(frozen=True)
class FinalTerminationWitness:
    """Auditable result of the final Tool ABI turn.

    ``exact_response_format_followed`` is deliberately diagnostic. Protocol
    certification is decided only by ``protocol_termination_verified``.
    """

    protocol_termination_verified: bool
    exact_response_format_followed: bool
    expected_sentinel: str
    final_sentinel_occurrences: int
    source_sentinel_occurrences: int
    premature_sentinel_occurrences: int
    observed_family_members: tuple[str, ...]
    failure_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_termination_verified": self.protocol_termination_verified,
            "exact_response_format_followed": self.exact_response_format_followed,
            "expected_sentinel": self.expected_sentinel,
            "final_sentinel_occurrences": self.final_sentinel_occurrences,
            "source_sentinel_occurrences": self.source_sentinel_occurrences,
            "premature_sentinel_occurrences": self.premature_sentinel_occurrences,
            "observed_family_members": list(self.observed_family_members),
            "failure_codes": list(self.failure_codes),
        }


def _bounded_token_pattern(token: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![{_TOKEN_CHAR}]){re.escape(token)}(?![{_TOKEN_CHAR}])"
    )


def _text_occurrences(value: Any, token: str) -> int:
    pattern = _bounded_token_pattern(token)

    def visit(item: Any) -> int:
        if isinstance(item, str):
            return len(pattern.findall(item))
        if isinstance(item, dict):
            return sum(visit(key) + visit(child) for key, child in item.items())
        if isinstance(item, (list, tuple)):
            return sum(visit(child) for child in item)
        return 0

    return visit(value)


def _exact_scalar_occurrences(value: Any, expected: str) -> int:
    if isinstance(value, str):
        return int(value == expected)
    if isinstance(value, dict):
        return sum(
            _exact_scalar_occurrences(key, expected)
            + _exact_scalar_occurrences(child, expected)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(_exact_scalar_occurrences(child, expected) for child in value)
    return 0


def _family_members(text: str, expected: str) -> tuple[str, ...]:
    match = _NONCE_SUFFIX.fullmatch(expected)
    if match is None or not match.group(1):
        return tuple(_bounded_token_pattern(expected).findall(text))
    prefix = match.group(1)
    pattern = re.compile(
        rf"(?<![{_TOKEN_CHAR}]){re.escape(prefix)}[A-Za-z0-9]+(?![{_TOKEN_CHAR}])"
    )
    return tuple(found.group(0) for found in pattern.finditer(text))


def assess_g3_termination(
    *,
    final_text: str,
    expected_sentinel: str,
    source_payload: Any,
    unavailable_payloads: tuple[Any, ...] = (),
    has_tool_calls: bool = False,
    parser_ambiguous: bool = False,
    parse_errors: tuple[str, ...] = (),
    normal_termination: bool = True,
) -> FinalTerminationWitness:
    """Decide G3 protocol termination from positive, nonce-bound evidence.

    The expected sentinel must occur as one exact scalar in the injected source
    payload, nowhere in material available before that source, and exactly once
    as a boundary-delimited token in the final assistant text. Any second member
    of the generated nonce family makes the observation ambiguous. Parser errors,
    parser disagreement, a further tool call, or a non-completed runtime turn all
    fail closed.
    """

    source_count = _text_occurrences(source_payload, expected_sentinel)
    source_exact_count = _exact_scalar_occurrences(source_payload, expected_sentinel)
    premature_count = sum(
        _text_occurrences(payload, expected_sentinel) for payload in unavailable_payloads
    )
    final_count = _text_occurrences(final_text, expected_sentinel)
    family_members = _family_members(final_text, expected_sentinel)
    failures: list[str] = []
    if not normal_termination:
        failures.append("noncompleted_turn")
    if parser_ambiguous:
        failures.append("parser_ambiguity")
    if parse_errors:
        failures.append("parser_error")
    if has_tool_calls:
        failures.append("spurious_tool_call")
    if source_count != 1 or source_exact_count != 1:
        failures.append("missing_or_nonunique_source")
    if premature_count:
        failures.append("sentinel_available_before_source")
    if final_count != 1:
        failures.append("missing_or_nonunique_final_sentinel")
    if family_members != (expected_sentinel,):
        failures.append("nonexclusive_sentinel_family")

    return FinalTerminationWitness(
        protocol_termination_verified=not failures,
        exact_response_format_followed=final_text.strip() == expected_sentinel,
        expected_sentinel=expected_sentinel,
        final_sentinel_occurrences=final_count,
        source_sentinel_occurrences=source_count,
        premature_sentinel_occurrences=premature_count,
        observed_family_members=family_members,
        failure_codes=tuple(failures),
    )


def assess_event_g3_termination(
    events: list[Event],
    *,
    expected_sentinel: str,
    source_tool: str,
    parser_ambiguous: bool = False,
    parse_errors: tuple[str, ...] = (),
    normal_termination: bool = True,
) -> FinalTerminationWitness:
    """Build a G3 witness from the actual normalized production trace."""

    source_indices = [
        index
        for index, event in enumerate(events)
        if isinstance(event, ToolResult) and event.name == source_tool
    ]
    first_source = source_indices[0] if source_indices else len(events)
    source_payload = [
        event.content
        for event in events
        if isinstance(event, ToolResult) and event.name == source_tool
    ]
    unavailable = tuple(
        event.model_dump(mode="json") for event in events[:first_source]
    )
    after_source = events[first_source + 1 :] if source_indices else ()
    has_tool_calls = any(
        isinstance(event, (AssistantToolCall, ToolCallBatch)) for event in after_source
    )
    final_text = events[-1].content if events and isinstance(events[-1], AssistantText) else ""
    return assess_g3_termination(
        final_text=final_text,
        expected_sentinel=expected_sentinel,
        source_payload=source_payload,
        unavailable_payloads=unavailable,
        has_tool_calls=has_tool_calls,
        parser_ambiguous=parser_ambiguous,
        parse_errors=parse_errors,
        normal_termination=normal_termination,
    )
