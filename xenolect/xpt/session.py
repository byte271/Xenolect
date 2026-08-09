"""XPT online runtime: one auditable, forkable conversation with an unknown endpoint.

Two things this module exists for:

1. **Exact reuse of production wire semantics.** Every request sent to the endpoint
   is produced by the same `xenolect.driver.encode` primitives that
   `DriverRuntime._resume_model` uses. This keeps compiler observations aligned
   with the runtime that will execute the installed compatibility profile.

2. **Counterfactual state forking**. A `Branch` can be forked
   at a frozen prefix so that two candidate *suffix* encodings are paid for
   separately while the expensive shared prefix is paid for once. Forking is only
   legal when the model-visible history is byte-identical at the fork point; the
   prefix hash is recorded and asserted on every fork.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from xenolect.abi.events import ToolDef, ToolResult
from xenolect.driver.encode import (
    build_tool_preamble_messages,
    encode_tool_result_message,
    should_send_native_tools,
    tools_for_request,
)
from xenolect.driver.ir import Driver
from xenolect.endpoints.errors import ClientError, FailureDomain
from xenolect.xpt.syndrome import Syndrome, build_syndrome, sha

if TYPE_CHECKING:
    from xenolect.xpt.diagnostic_probe import DiagnosticProbe


class BudgetExhausted(RuntimeError):
    """Raised when a generation would exceed the query budget or the deadline."""


class DeadlineExceeded(RuntimeError):
    """Raised when the wall-clock deadline cannot cover certification any more."""


class InfrastructureFailed(RuntimeError):
    """Raised when transport/infrastructure fails; not protocol evidence."""

    def __init__(self, message: str, *, domain: str = "infrastructure") -> None:
        self.domain = domain
        super().__init__(message)


class ConfigurationFailed(RuntimeError):
    """Raised when endpoint configuration fails; not protocol evidence."""


@dataclass
class Budget:
    """Hard online generation and wall-clock budgets."""

    max_generations: int = 12
    #: Generations that exploration may never consume; kept for certification.
    #: Equals the length of the ABI Gauntlet trajectory (3 turns).
    certification_reserve: int = 3
    #: Wall-clock deadline in seconds from session start (None = untimed internal run).
    deadline_s: float | None = None
    #: Fraction of remaining time that must still be free after an exploration call.
    safety_margin: float = 0.10


@dataclass
class Generation:
    """One expensive black-box interaction, fully recorded."""

    index: int
    purpose: str
    label: str
    branch_id: str
    forked_from: str | None
    prefix_hash: str
    driver: dict[str, Any] | None
    request: dict[str, Any]
    request_hash: str
    response: Any
    response_hash: str | None
    error: str | None
    latency_ms: float
    prompt_chars: int
    completion_chars: int
    syndrome: dict[str, Any] | None = None
    selection_reason: str = ""
    diagnostic_probe: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "purpose": self.purpose,
            "label": self.label,
            "branch_id": self.branch_id,
            "forked_from": self.forked_from,
            "prefix_hash": self.prefix_hash,
            "driver": self.driver,
            "request": self.request,
            "request_hash": self.request_hash,
            "response": self.response,
            "response_hash": self.response_hash,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "prompt_chars": self.prompt_chars,
            "completion_chars": self.completion_chars,
            "syndrome": self.syndrome,
            "selection_reason": self.selection_reason,
            "diagnostic_probe": self.diagnostic_probe,
        }


@dataclass
class Ledger:
    """Replayable trace of an XPT run."""

    experiment_id: str = ""
    generations: list[Generation] = field(default_factory=list)
    forks: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def generation_count(self) -> int:
        return len(self.generations)

    def note(self, text: str) -> None:
        self.notes.append(text)

    def decide(self, **payload: Any) -> None:
        self.decisions.append(dict(payload))

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "generation_count": self.generation_count,
            "generations": [g.as_dict() for g in self.generations],
            "forks": list(self.forks),
            "decisions": list(self.decisions),
            "notes": list(self.notes),
        }


def _clean(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in m.items() if not str(k).startswith("_")} for m in messages]


def prefix_hash(messages: list[dict[str, Any]]) -> str:
    return sha(_clean(messages))


class Branch:
    """A conversation state plus the driver configuration used to extend it."""

    _counter = 0

    def __init__(
        self,
        session: XptSession,
        driver: Driver | None,
        *,
        branch_id: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[ToolDef] | None = None,
        parent: str | None = None,
        diagnostic_probe: DiagnosticProbe | None = None,
    ) -> None:
        Branch._counter += 1
        self.session = session
        self.driver = driver
        self.branch_id = branch_id or f"b{Branch._counter:03d}"
        self.model_messages: list[dict[str, Any]] = messages if messages is not None else []
        self.tools: list[ToolDef] = list(tools or [])
        self.parent = parent
        self.diagnostic_probe = diagnostic_probe
        self.last_generation: Generation | None = None

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def freeze(self) -> str:
        return prefix_hash(self.model_messages)

    def fork(self, driver: Driver | None = None, *, reason: str = "") -> Branch:
        """Fork at the current frozen prefix. Prefix must be byte-identical."""
        if self.diagnostic_probe is not None or self.driver is None:
            raise ValueError("diagnostic probe branches cannot enter production history")
        parent_hash = self.freeze()
        child = Branch(
            self.session,
            driver or self.driver,
            messages=copy.deepcopy(self.model_messages),
            tools=list(self.tools),
            parent=self.branch_id,
        )
        assert child.freeze() == parent_hash, "fork must preserve the model-visible prefix"
        self.session.ledger.forks.append(
            {
                "parent": self.branch_id,
                "child": child.branch_id,
                "prefix_hash": parent_hash,
                "parent_driver": self.driver.canonical_dict(),
                "child_driver": child.driver.canonical_dict(),
                "reason": reason,
            }
        )
        return child

    # ------------------------------------------------------------------
    # message construction (mirrors DriverRuntime)
    # ------------------------------------------------------------------

    def add_user(self, content: str, tools: list[ToolDef] | None = None) -> None:
        if self.driver is None:
            raise ValueError("diagnostic probe branches cannot be extended as Drivers")
        if tools:
            self.tools = list(tools)
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
        self.model_messages.append({"role": "user", "content": content})

    def add_assistant_raw(self, raw: dict[str, Any]) -> None:
        """Mirror DriverRuntime._append_assistant_to_history for the `choices` shape."""
        message = None
        if "choices" in raw:
            choices = raw.get("choices") or []
            if choices:
                message = choices[0].get("message")
        if message is None:
            self.model_messages.append({"role": "assistant", "content": json.dumps(raw)})
            return
        hist = {
            k: v for k, v in message.items() if k in ("role", "content", "tool_calls", "name")
        }
        hist.setdefault("role", "assistant")
        self.model_messages.append(hist)

    def add_tool_result(
        self, *, call_id: str | None, name: str | None, content: Any
    ) -> None:
        if self.driver is None:
            raise ValueError("diagnostic probe branches cannot render production results")
        msg = encode_tool_result_message(
            ToolResult(call_id=call_id, name=name, content=content), self.driver
        )
        self.model_messages.append({k: v for k, v in msg.items() if not str(k).startswith("_")})

    def add_tool_error(self, *, call_id: str | None, name: str | None, error: str) -> None:
        # Same normalization the production runtime applies to ToolError.
        self.add_tool_result(call_id=call_id, name=name, content={"error": error})

    # ------------------------------------------------------------------
    # wire
    # ------------------------------------------------------------------

    def build_request(self) -> dict[str, Any]:
        if self.diagnostic_probe is not None:
            return self.diagnostic_probe.wire()
        if self.driver is None:
            raise ValueError("production branch requires a Driver")
        wire_tools = None
        if should_send_native_tools(self.driver) and self.tools:
            wire_tools = tools_for_request(self.tools, self.driver)
        return {"messages": _clean(self.model_messages), "tools": wire_tools, "seed": None}

    def generate(
        self,
        *,
        purpose: str,
        label: str,
        reason: str = "",
        offered_tool_names: set[str] | None = None,
    ) -> tuple[Syndrome, Generation]:
        return self.session.generate(
            self, purpose=purpose, label=label, reason=reason,
            offered_tool_names=offered_tool_names,
        )


class XptSession:
    """Owns the endpoint client, the budget, the deadline and the ledger."""

    def __init__(
        self,
        client: Any,
        *,
        budget: Budget | None = None,
        ledger: Ledger | None = None,
        clock: Callable[[], float] = time.perf_counter,
        started_at: float | None = None,
    ) -> None:
        self.client = client
        self.budget = budget or Budget()
        self.ledger = ledger or Ledger()
        self.clock = clock
        # Share an epoch with the HTTP client when compiling under one wall-clock budget.
        self.started_at = clock() if started_at is None else started_at
        self.latency_samples_ms: list[float] = []

    # ------------------------------------------------------------------
    # budget / deadline
    # ------------------------------------------------------------------

    @property
    def elapsed_s(self) -> float:
        return self.clock() - self.started_at

    @property
    def remaining_s(self) -> float | None:
        if self.budget.deadline_s is None:
            return None
        return self.budget.deadline_s - self.elapsed_s

    @property
    def observed_latency_ms(self) -> float:
        if not self.latency_samples_ms:
            return 0.0
        # Worst observed sample: budgeting must not be optimistic.
        return max(self.latency_samples_ms)

    def certification_affordable(self) -> bool:
        """Return whether the hard wall-clock deadline has not expired.

        A previous implementation extrapolated the slowest observed generation across
        the entire reserved certification run.  That is not a sound hard-budget
        rule for real endpoints: the first request can include model cold-start,
        so one slow sample can incorrectly abort an otherwise feasible compile.
        The HTTP client already caps every request to the remaining absolute
        deadline.  The compiler therefore uses actual elapsed time as the hard
        rule and keeps generation reserve accounting separate.
        """
        remaining = self.remaining_s
        return remaining is None or remaining > 0

    def check_can_explore(self) -> None:
        used = self.ledger.generation_count
        if used + self.budget.certification_reserve >= self.budget.max_generations:
            raise BudgetExhausted(
                f"exploration would consume the certification reserve "
                f"(used={used}, reserve={self.budget.certification_reserve}, "
                f"max={self.budget.max_generations})"
            )
        remaining = self.remaining_s
        if remaining is not None and remaining <= 0:
            raise DeadlineExceeded("compiler wall-clock deadline exhausted")

    def check_can_certify(self, *, required_generations: int | None = None) -> None:
        """Require enough hard query/time budget for the whole certification run.

        Certification is executed through the production DriverRuntime rather than
        ``XptSession.generate``.  Therefore the entire certification allowance must
        be reserved *before* that runtime is entered; post-hoc accounting is not a
        budget guard.
        """
        required = (
            self.budget.certification_reserve
            if required_generations is None
            else required_generations
        )
        if required < 0:
            raise ValueError("required_generations must be non-negative")
        used = self.ledger.generation_count
        if used + required > self.budget.max_generations:
            raise BudgetExhausted(
                "insufficient generation budget for certification "
                f"(used={used}, required={required}, max={self.budget.max_generations})"
            )
        remaining = self.remaining_s
        if remaining is not None and remaining <= 0:
            raise DeadlineExceeded("deadline passed before certification started")

    # ------------------------------------------------------------------
    # branches
    # ------------------------------------------------------------------

    def new_branch(self, driver: Driver) -> Branch:
        return Branch(self, driver)

    def new_diagnostic_branch(self, probe: DiagnosticProbe) -> Branch:
        wire = probe.wire()
        return Branch(
            self,
            None,
            messages=copy.deepcopy(wire["messages"]),
            diagnostic_probe=probe,
        )

    # ------------------------------------------------------------------
    # the one expensive operation
    # ------------------------------------------------------------------

    def generate(
        self,
        branch: Branch,
        *,
        purpose: str,
        label: str,
        reason: str = "",
        offered_tool_names: set[str] | None = None,
    ) -> tuple[Syndrome, Generation]:
        if purpose == "explore":
            self.check_can_explore()
        else:
            self.check_can_certify()

        request = branch.build_request()
        pre_hash = branch.freeze()
        t0 = self.clock()
        raw: Any = None
        err: str | None = None
        fatal: BaseException | None = None
        # Always send the cleaned wire (no internal `_…` bookkeeping keys).
        wire_messages = request["messages"]
        try:
            raw = self.client.chat_completions(wire_messages, tools=request["tools"])
        except ClientError as exc:
            err = f"{type(exc).__name__}: {exc}"
            # Infrastructure/configuration are not protocol observations: treating
            # them as G1 failures would SAFE-eliminate correct wire classes and
            # surface as UNSUPPORTED ("cannot speak this model").
            if exc.domain == FailureDomain.CONFIGURATION:
                fatal = ConfigurationFailed(str(exc))
            elif exc.domain == FailureDomain.INFRASTRUCTURE:
                msg = str(exc.message) if getattr(exc, "message", None) else str(exc)
                if "deadline" in msg.lower():
                    fatal = DeadlineExceeded(msg)
                else:
                    fatal = InfrastructureFailed(msg)
            # PROTOCOL-domain ClientError remains a local observation (raw=None).
        except Exception as exc:  # noqa: BLE001 — other client failures stay observations
            err = f"{type(exc).__name__}: {exc}"

        latency_ms = (self.clock() - t0) * 1000.0
        self.latency_samples_ms.append(latency_ms)

        syn = build_syndrome(
            raw if isinstance(raw, dict) else None,
            transport_error=err,
            offered_tool_names=offered_tool_names,
        )

        gen = Generation(
            index=len(self.ledger.generations) + 1,
            purpose=purpose,
            label=label,
            branch_id=branch.branch_id,
            forked_from=branch.parent,
            prefix_hash=pre_hash,
            driver=branch.driver.canonical_dict() if branch.driver is not None else None,
            request=request,
            request_hash=sha(request),
            response=raw,
            response_hash=sha(raw) if raw is not None else None,
            error=err,
            latency_ms=latency_ms,
            prompt_chars=len(json.dumps(request, default=str)),
            completion_chars=len(json.dumps(raw, default=str)) if raw is not None else 0,
            syndrome=syn.as_dict(),
            selection_reason=reason,
            diagnostic_probe=(
                branch.diagnostic_probe.as_dict()
                if branch.diagnostic_probe is not None
                else None
            ),
        )
        self.ledger.generations.append(gen)
        branch.last_generation = gen

        if fatal is not None:
            raise fatal

        # The request itself is allowed to use the remaining wall-clock budget,
        # but a response arriving after the absolute compiler deadline cannot be
        # treated as protocol evidence or silently converted into UNSUPPORTED.
        remaining = self.remaining_s
        if remaining is not None and remaining <= 0:
            raise DeadlineExceeded("compiler wall-clock deadline reached during generation")

        if isinstance(raw, dict):
            branch.add_assistant_raw(raw)
        return syn, gen
