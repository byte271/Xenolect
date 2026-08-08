"""Legal-trace validation for Tool ABI v0 state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from xenolect.abi.events import (
    AssistantText,
    AssistantToolCall,
    Event,
    ToolCallBatch,
    ToolError,
    ToolResult,
    UserMessage,
)


class TraceState(str, Enum):
    CHAT = "CHAT"
    WAIT_MODEL = "WAIT_MODEL"
    WAIT_TOOL_RESULT = "WAIT_TOOL_RESULT"
    ILLEGAL = "ILLEGAL"


@dataclass
class TraceValidation:
    legal: bool
    final_state: TraceState
    errors: list[str] = field(default_factory=list)
    outstanding_call_ids: set[str] = field(default_factory=set)
    known_tool_names: set[str] = field(default_factory=set)


def validate_trace(events: Iterable[Event]) -> TraceValidation:
    """Validate a sequence of normalized ABI events against the Tool ABI v0 state machine."""
    state = TraceState.CHAT
    errors: list[str] = []
    outstanding: set[str] = set()
    known_tools: set[str] = set()
    # When model emits calls without ids, count outstanding anonymous calls.
    outstanding_anon = 0
    ids_required = False
    event_list = list(events)

    if not event_list:
        return TraceValidation(
            legal=False,
            final_state=TraceState.ILLEGAL,
            errors=["trace is empty"],
        )

    for i, event in enumerate(event_list):
        prefix = f"event[{i}] ({event.type})"

        if state == TraceState.CHAT:
            if isinstance(event, UserMessage):
                for t in event.tools:
                    known_tools.add(t.name)
                state = TraceState.WAIT_MODEL
            else:
                errors.append(f"{prefix}: expected UserMessage in CHAT")
                state = TraceState.ILLEGAL
                break

        elif state == TraceState.WAIT_MODEL:
            if isinstance(event, AssistantText):
                state = TraceState.CHAT
            elif isinstance(event, AssistantToolCall):
                _register_calls(
                    [event.call],
                    outstanding,
                    outstanding_anon_holder := _Anon(outstanding_anon),
                    known_tools,
                    errors,
                    prefix,
                )
                outstanding_anon = outstanding_anon_holder.value
                if event.call.id is not None:
                    ids_required = True
                state = TraceState.WAIT_TOOL_RESULT
            elif isinstance(event, ToolCallBatch):
                _register_calls(
                    event.calls,
                    outstanding,
                    outstanding_anon_holder := _Anon(outstanding_anon),
                    known_tools,
                    errors,
                    prefix,
                )
                outstanding_anon = outstanding_anon_holder.value
                if any(c.id is not None for c in event.calls):
                    ids_required = True
                state = TraceState.WAIT_TOOL_RESULT
            elif isinstance(event, UserMessage):
                # Allow continued user messages only after a completed text turn (CHAT).
                errors.append(f"{prefix}: UserMessage while WAIT_MODEL (model must respond first)")
                state = TraceState.ILLEGAL
                break
            else:
                errors.append(f"{prefix}: expected assistant text or tool call(s) in WAIT_MODEL")
                state = TraceState.ILLEGAL
                break

        elif state == TraceState.WAIT_TOOL_RESULT:
            if isinstance(event, (ToolResult, ToolError)):
                call_id = event.call_id
                if call_id is not None:
                    if call_id not in outstanding:
                        # Unknown or duplicate (already resolved) call id
                        errors.append(f"{prefix}: unknown or duplicate call_id {call_id!r}")
                    else:
                        outstanding.discard(call_id)
                else:
                    if outstanding_anon > 0:
                        outstanding_anon -= 1
                    elif outstanding and not ids_required:
                        # Drop one outstanding id arbitrarily when results omit ids.
                        outstanding.pop()
                    elif outstanding:
                        errors.append(f"{prefix}: missing call_id while ids were emitted")
                    else:
                        errors.append(f"{prefix}: unexpected tool result with no outstanding calls")

                if not outstanding and outstanding_anon == 0:
                    state = TraceState.WAIT_MODEL
            else:
                errors.append(f"{prefix}: expected ToolResult/ToolError in WAIT_TOOL_RESULT")
                state = TraceState.ILLEGAL
                break

        else:
            errors.append(f"{prefix}: already illegal")
            break

    if state == TraceState.WAIT_TOOL_RESULT and (outstanding or outstanding_anon > 0):
        # Incomplete but structure may still be prefix-legal for partial evaluation.
        pass

    legal = not errors and state != TraceState.ILLEGAL
    return TraceValidation(
        legal=legal,
        final_state=state,
        errors=errors,
        outstanding_call_ids=set(outstanding),
        known_tool_names=known_tools,
    )


def validate_completed_trace(events: Iterable[Event]) -> TraceValidation:
    """
    Prefix-legal is not enough: a completed probe must have no outstanding tool calls
    and must not end mid-protocol (WAIT_TOOL_RESULT).
    """
    tv = validate_trace(events)
    if not tv.legal:
        return tv
    errors = list(tv.errors)
    if tv.outstanding_call_ids:
        errors.append(
            f"completed trace has outstanding call ids: {sorted(tv.outstanding_call_ids)}"
        )
    if tv.final_state == TraceState.WAIT_TOOL_RESULT:
        errors.append("completed trace ends in WAIT_TOOL_RESULT")
    if tv.final_state == TraceState.WAIT_MODEL:
        # Legal if last assistant turn was text → CHAT; WAIT_MODEL means awaiting model
        # after results without a final model event — incomplete for probes.
        errors.append("completed trace ends awaiting model (WAIT_MODEL)")
    # CHAT is the normal terminal after AssistantText
    legal = not errors
    return TraceValidation(
        legal=legal,
        final_state=tv.final_state if legal else TraceState.ILLEGAL,
        errors=errors,
        outstanding_call_ids=tv.outstanding_call_ids,
        known_tool_names=tv.known_tool_names,
    )


class _Anon:
    def __init__(self, value: int) -> None:
        self.value = value


def _register_calls(
    calls: list,
    outstanding: set[str],
    outstanding_anon: _Anon,
    known_tools: set[str],
    errors: list[str],
    prefix: str,
) -> None:
    if not calls:
        errors.append(f"{prefix}: empty tool call list")
        return
    for call in calls:
        if known_tools and call.name not in known_tools:
            errors.append(f"{prefix}: unknown tool name {call.name!r}")
        if call.id is not None:
            if call.id in outstanding:
                errors.append(f"{prefix}: duplicate call id {call.id!r}")
            outstanding.add(call.id)
        else:
            outstanding_anon.value += 1
