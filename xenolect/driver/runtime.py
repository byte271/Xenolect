"""Stateful bidirectional driver runtime.

Parallel batch invariant (Tool ABI v0):
  When the assistant emits N tool calls in one turn, all N results/errors
  must be injected into model-facing history *before* the next model resume.
  Interleaving (result A → model → result B → model) is incorrect.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol

from xenolect.abi.events import (
    AssistantText,
    AssistantToolCall,
    Event,
    ToolCall,
    ToolCallBatch,
    ToolDef,
    ToolError,
    ToolResult,
    UserMessage,
)
from xenolect.driver.encode import (
    build_tool_preamble_messages,
    encode_tool_result_message,
    should_send_native_tools,
    tools_for_request,
)
from xenolect.driver.ir import Driver, StateAction, effective_protocol
from xenolect.driver.parse import parse_model_response_full
from xenolect.driver.termination import ProbeRunResult, Termination
from xenolect.endpoints.errors import ClientError, FailureDomain


def _termination_for_exception(exc: Exception) -> Termination:
    if isinstance(exc, ClientError):
        if exc.domain == FailureDomain.INFRASTRUCTURE:
            return Termination.INFRASTRUCTURE_FAILED
        if exc.domain == FailureDomain.CONFIGURATION:
            return Termination.CONFIGURATION_FAILED
    return Termination.MODEL_ERROR


class EndpointClient(Protocol):
    def chat_completions(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...


@dataclass
class DriverRuntime:
    """Executes a driver against an endpoint, maintaining chat history on the model side."""

    driver: Driver
    client: EndpointClient
    model_messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[ToolDef] = field(default_factory=list)
    outstanding: dict[str, str] = field(default_factory=dict)  # call_id -> name
    outstanding_anon: int = 0
    last_parse_errors: list[str] = field(default_factory=list)
    base_seed: int | None = None
    seed_context: str = ""
    request_index: int = 0
    wire_interactions: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.model_messages.clear()
        self.tools.clear()
        self.outstanding.clear()
        self.outstanding_anon = 0
        self.last_parse_errors.clear()
        self.request_index = 0
        self.wire_interactions.clear()

    def handle_user(self, message: UserMessage) -> list[Event]:
        """Send user message (and tools) to the model; return normalized model events."""
        if message.tools:
            self.tools = list(message.tools)

        preambles = build_tool_preamble_messages(self.tools, self.driver)
        if preambles and not any(
            m.get("_xenolect_preamble") for m in self.model_messages
        ):
            for index, preamble in reversed(list(enumerate(preambles))):
                self.model_messages.insert(
                    0,
                    {
                        **preamble,
                        "_xenolect_preamble": True,
                        "_xenolect_preamble_index": index,
                    },
                )

        self.model_messages.append({"role": "user", "content": message.content})
        return self._resume_model()

    def register_calls(self, calls: list[ToolCall]) -> None:
        """Record outstanding calls after an assistant tool turn."""
        self._require_state_action(StateAction.TRACK_OUTSTANDING_CALLS)
        for call in calls:
            if call.id is not None:
                self.outstanding[call.id] = call.name
            else:
                self.outstanding_anon += 1

    def ready_to_resume(self) -> bool:
        return not self.outstanding and self.outstanding_anon == 0

    def append_tool_result(self, result: ToolResult | ToolError) -> None:
        """Inject one tool result into history without resuming the model."""
        self._require_state_action(StateAction.APPEND_TOOL_RESULTS)
        if isinstance(result, ToolError):
            tr = ToolResult(
                call_id=result.call_id,
                name=result.name,
                content={"error": result.error},
            )
        else:
            tr = result

        msg = encode_tool_result_message(tr, self.driver)
        clean = {k: v for k, v in msg.items() if not str(k).startswith("_")}
        self.model_messages.append(clean)

        if tr.call_id is not None:
            self.outstanding.pop(tr.call_id, None)
        elif self.outstanding_anon > 0:
            self.outstanding_anon -= 1

    def resume_model(self) -> list[Event]:
        """Request the next model turn (only after outstanding calls are resolved)."""
        return self._resume_model()

    def handle_tool_result(self, result: ToolResult | ToolError) -> list[Event]:
        """
        Inject a single tool result and resume if (and only if) no calls remain outstanding.

        For true parallel batches, prefer append_tool_result for each result, then
        resume_model once — or handle_tool_results.
        """
        self.append_tool_result(result)
        self._require_state_action(StateAction.RESUME_WHEN_ALL_RESULTS)
        if self.ready_to_resume():
            return self.resume_model()
        return []

    def handle_tool_results(self, results: list[ToolResult | ToolError]) -> list[Event]:
        """Inject all results for the current outstanding batch, then resume once."""
        for r in results:
            self.append_tool_result(r)
        self._require_state_action(StateAction.RESUME_WHEN_ALL_RESULTS)
        if not self.ready_to_resume():
            # Incomplete batch: do not resume the model. Callers that treat an
            # empty event list as "no more tool calls" must check ready_to_resume
            # (run_probe does) so this is not silently COMPLETED.
            return []
        return self.resume_model()

    def _require_state_action(self, action: StateAction) -> None:
        """Fail clearly if a bypassed/foreign IR requests unsupported state logic."""
        if action not in effective_protocol(self.driver).state:
            raise RuntimeError(
                f"Driver state program does not provide required action {action.value!r}"
            )

    def run_probe_script(
        self,
        user: UserMessage,
        tool_executor: dict[str, Any] | None = None,
        max_cycles: int = 8,
    ) -> list[Event]:
        """Backward-compatible: return events only."""
        return self.run_probe(user, tool_executor=tool_executor, max_cycles=max_cycles).events

    def run_probe(
        self,
        user: UserMessage,
        tool_executor: dict[str, Any] | None = None,
        max_cycles: int = 8,
    ) -> ProbeRunResult:
        """
        Run a multi-turn probe with termination status.

        Parallel invariant: all results for a batch are applied before one model resume.
        """
        tool_executor = tool_executor or {}
        trace: list[Event] = [user]
        try:
            events = self.handle_user(user)
        except Exception as exc:  # noqa: BLE001
            return ProbeRunResult(
                events=trace,
                termination=_termination_for_exception(exc),
                errors=[str(exc)],
                wire_interactions=list(self.wire_interactions),
            )
        if self.last_parse_errors:
            return ProbeRunResult(
                events=trace,
                termination=Termination.PARSE_ERROR,
                parse_errors=list(self.last_parse_errors),
                errors=list(self.last_parse_errors),
                wire_interactions=list(self.wire_interactions),
            )
        trace.extend(events)

        cycles = 0
        while cycles < max_cycles:
            calls = _calls_from(events)
            if not calls:
                return ProbeRunResult(
                    events=trace,
                    termination=Termination.COMPLETED,
                    wire_interactions=list(self.wire_interactions),
                )

            batch_results: list[ToolResult | ToolError] = []
            for call in calls:
                batch_results.append(self._execute_call(call, tool_executor))

            for tr in batch_results:
                trace.append(tr)

            try:
                events = self.handle_tool_results(batch_results)
            except Exception as exc:  # noqa: BLE001
                return ProbeRunResult(
                    events=trace,
                    termination=_termination_for_exception(exc),
                    errors=[str(exc)],
                    wire_interactions=list(self.wire_interactions),
                )
            # Incomplete injection: no resume occurred (empty events) and calls
            # remain outstanding. A successful resume that emits new tool calls
            # also leaves ready_to_resume false — that path has non-empty events.
            if not events and not self.ready_to_resume():
                return ProbeRunResult(
                    events=trace,
                    termination=Termination.TOOL_PROTOCOL_ERROR,
                    errors=[
                        "tool results did not clear outstanding calls "
                        "(call_id / batch association mismatch)"
                    ],
                    wire_interactions=list(self.wire_interactions),
                )
            if self.last_parse_errors:
                return ProbeRunResult(
                    events=trace,
                    termination=Termination.PARSE_ERROR,
                    parse_errors=list(self.last_parse_errors),
                    errors=list(self.last_parse_errors),
                    wire_interactions=list(self.wire_interactions),
                )
            trace.extend(events)
            cycles += 1
            if not _calls_from(events):
                return ProbeRunResult(
                    events=trace,
                    termination=Termination.COMPLETED,
                    wire_interactions=list(self.wire_interactions),
                )

        # Still have tool calls or unfinished after max cycles
        return ProbeRunResult(
            events=trace,
            termination=Termination.MAX_CYCLES_EXHAUSTED,
            errors=[f"exceeded max_cycles={max_cycles}"],
            details={"cycles": cycles},
            wire_interactions=list(self.wire_interactions),
        )

    def _execute_call(
        self,
        call: ToolCall,
        tool_executor: dict[str, Any],
    ) -> ToolResult | ToolError:
        fn = tool_executor.get(call.name)
        if callable(fn):
            try:
                if isinstance(call.arguments, dict):
                    value = fn(**call.arguments)
                else:
                    # List/scalar arguments: do not coerce to {} (would drop [] etc.).
                    value = fn(call.arguments)
            except TypeError:
                try:
                    value = fn(call.arguments)
                except Exception as exc:  # noqa: BLE001
                    return ToolError(call_id=call.id, name=call.name, error=str(exc))
            except Exception as exc:  # noqa: BLE001
                return ToolError(call_id=call.id, name=call.name, error=str(exc))
        else:
            value = fn if fn is not None else {"ok": True, "echo": call.arguments}
        return ToolResult(call_id=call.id, name=call.name, content=value)

    def _derived_seed(self, turn_index: int) -> int | None:
        if self.base_seed is None:
            return None
        material = f"{self.base_seed}|{self.seed_context}|{turn_index}".encode()
        # Keep in signed 31-bit range for broad endpoint compatibility.
        return int(hashlib.sha256(material).hexdigest()[:8], 16) & 0x7FFFFFFF

    def _resume_model(self) -> list[Event]:
        wire_tools = None
        if should_send_native_tools(self.driver) and self.tools:
            wire_tools = tools_for_request(self.tools, self.driver)

        clean_messages = [
            {k: v for k, v in m.items() if not str(k).startswith("_")}
            for m in self.model_messages
        ]
        turn_index = self.request_index
        seed = self._derived_seed(turn_index)
        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = seed

        request_snapshot: dict[str, Any] = {
            "messages": clean_messages,
            "tools": wire_tools,
            "seed": seed,
        }
        if hasattr(self.client, "model"):
            request_snapshot["model"] = getattr(self.client, "model")
        if hasattr(self.client, "generation_config"):
            try:
                request_snapshot["generation_config"] = dict(self.client.generation_config())
                if seed is not None:
                    request_snapshot["generation_config"]["seed"] = seed
            except Exception:  # noqa: BLE001
                pass

        self.request_index += 1
        t0 = __import__("time").perf_counter()
        try:
            # Send cleaned messages so internal `_…` keys never reach the wire.
            raw = self.client.chat_completions(
                clean_messages, tools=wire_tools, **kwargs
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (__import__("time").perf_counter() - t0) * 1000
            error_response = getattr(self.client, "last_response_body", None)
            if isinstance(exc, ClientError) and exc.details:
                error_response = exc.details.get("response_text", error_response)
            interaction = {
                "turn_index": turn_index,
                "request": request_snapshot,
                "response": error_response,
                "error": str(exc),
                "latency_ms": latency_ms,
                "seed": seed,
            }
            attempts = getattr(self.client, "last_attempts", None)
            if attempts is not None:
                interaction["http_attempts"] = list(attempts)
            self.wire_interactions.append(interaction)
            raise

        latency_ms = (__import__("time").perf_counter() - t0) * 1000
        interaction = {
            "turn_index": turn_index,
            "request": request_snapshot,
            "response": raw,
            "error": None,
            "latency_ms": latency_ms,
            "seed": seed,
        }
        attempts = getattr(self.client, "last_attempts", None)
        if attempts is not None:
            interaction["http_attempts"] = list(attempts)
        self.wire_interactions.append(interaction)

        pr = parse_model_response_full(raw, self.driver)
        self.last_parse_errors = list(pr.errors)
        events = pr.events
        if not pr.errors:
            self._append_assistant_to_history(raw, events)
            new_calls = _calls_from(events)
            if new_calls:
                self.register_calls(new_calls)
        return events

    def _append_assistant_to_history(self, raw: dict[str, Any], events: list[Event]) -> None:
        """Keep model-facing history consistent with what the endpoint expects."""
        message = None
        if "choices" in raw:
            choices = raw.get("choices") or []
            if choices:
                message = choices[0].get("message")
        if message is None:
            for e in events:
                if isinstance(e, AssistantText):
                    self.model_messages.append({"role": "assistant", "content": e.content})
                elif isinstance(e, AssistantToolCall):
                    self.model_messages.append(
                        {
                            "role": "assistant",
                            "content": e.content,
                            "tool_calls": [_tc_wire(e.call)],
                        }
                    )
                elif isinstance(e, ToolCallBatch):
                    self.model_messages.append(
                        {
                            "role": "assistant",
                            "content": e.content,
                            "tool_calls": [_tc_wire(c) for c in e.calls],
                        }
                    )
            return

        hist = {k: v for k, v in message.items() if k in ("role", "content", "tool_calls", "name")}
        if "role" not in hist:
            hist["role"] = "assistant"
        self.model_messages.append(hist)


def _tc_wire(call: Any) -> dict[str, Any]:
    import json

    args = call.arguments
    if not isinstance(args, str):
        args = json.dumps(args)
    return {
        "id": call.id or "call_0",
        "type": "function",
        "function": {"name": call.name, "arguments": args},
    }


def _calls_from(events: list[Event]) -> list[ToolCall]:
    out: list[ToolCall] = []
    for e in events:
        if isinstance(e, AssistantToolCall):
            out.append(e.call)
        elif isinstance(e, ToolCallBatch):
            out.extend(e.calls)
    return out
