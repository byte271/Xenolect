"""HTTP / trial failure domains."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class FailureDomain(str, Enum):
    PROTOCOL = "protocol"  # model/protocol observation
    INFRASTRUCTURE = "infrastructure"  # transport, 5xx, timeout, 429
    CONFIGURATION = "configuration"  # 401/403/404 endpoint config


@dataclass
class ClientError(Exception):
    domain: FailureDomain
    message: str
    status_code: int | None = None
    retryable: bool = False
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        return f"{self.domain.value}: {self.message}"


def classify_http_status(status: int) -> tuple[FailureDomain, bool]:
    """Return (domain, retryable)."""
    if status in (401, 403, 404):
        return FailureDomain.CONFIGURATION, False
    if status in (408, 429) or status >= 500:
        return FailureDomain.INFRASTRUCTURE, True
    if status in (400, 422):
        # May be protocol/request shape issues — treat as protocol observation
        return FailureDomain.PROTOCOL, False
    if status >= 400:
        return FailureDomain.INFRASTRUCTURE, status >= 500
    return FailureDomain.PROTOCOL, False
