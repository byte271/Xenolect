"""Runtime termination status for probe execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from xenolect.abi.events import Event


class Termination(str, Enum):
    COMPLETED = "completed"
    MAX_CYCLES_EXHAUSTED = "max_cycles_exhausted"
    MODEL_ERROR = "model_error"
    TOOL_PROTOCOL_ERROR = "tool_protocol_error"
    PARSE_ERROR = "parse_error"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"
    CONFIGURATION_FAILED = "configuration_failed"


@dataclass
class ProbeRunResult:
    events: list[Event]
    termination: Termination = Termination.COMPLETED
    errors: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    wire_interactions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return self.termination == Termination.COMPLETED
