"""Persisted execution assumptions for protocol-certified Drivers.

The profile is registry metadata, not Driver IR. It records the request fields
under which XPT observed and independently certified protocol compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CERTIFIED_EXECUTION_PROFILE_VERSION = 1
_PROFILE_SCOPE = "protocol_compatibility_at_recorded_request_defaults"
_OMITTED_DURING_CERTIFICATION = (
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "seed",
    "parallel_tool_calls",
    "tool_choice",
)


@dataclass(frozen=True)
class CertifiedExecutionProfile:
    """Conditions under which one Driver's protocol was certified.

    Runtime callers retain control of every explicitly supplied request field.
    Recorded defaults apply only when that field is absent from the application
    request, preventing the proxy from silently drifting away from certification.
    """

    profile_version: int = CERTIFIED_EXECUTION_PROFILE_VERSION
    certification_scope: str = _PROFILE_SCOPE
    request_defaults: dict[str, Any] = field(
        default_factory=lambda: {"temperature": 0.0}
    )
    endpoint_default_fields: tuple[str, ...] = _OMITTED_DURING_CERTIFICATION
    explicit_request_fields_override: bool = True
    provenance: str = "xpt_compiler"

    def __post_init__(self) -> None:
        if self.profile_version != CERTIFIED_EXECUTION_PROFILE_VERSION:
            raise ValueError(
                f"unsupported certified execution profile {self.profile_version!r}"
            )
        if self.certification_scope != _PROFILE_SCOPE:
            raise ValueError(f"unsupported certification scope {self.certification_scope!r}")
        if self.request_defaults != {"temperature": 0.0}:
            raise ValueError("v1 certified execution profile requires temperature=0.0")
        if tuple(self.endpoint_default_fields) != _OMITTED_DURING_CERTIFICATION:
            raise ValueError("v1 certified execution profile has unknown endpoint defaults")
        if self.explicit_request_fields_override is not True:
            raise ValueError("explicit application request fields must override certified defaults")
        if self.provenance not in {"xpt_compiler", "legacy_v0.4_registry"}:
            raise ValueError(f"unsupported execution-profile provenance {self.provenance!r}")

    @classmethod
    def legacy_v04(cls) -> CertifiedExecutionProfile:
        """Defined migration for entries written by the released v0.4 compiler."""

        return cls(provenance="legacy_v0.4_registry")

    @classmethod
    def from_dict(cls, payload: Any) -> CertifiedExecutionProfile:
        if not isinstance(payload, dict):
            raise ValueError("certified execution profile must be an object")
        expected = {
            "profile_version",
            "certification_scope",
            "request_defaults",
            "endpoint_default_fields",
            "explicit_request_fields_override",
            "provenance",
            "all_sampling_settings_certified",
        }
        if set(payload) != expected:
            raise ValueError("certified execution profile fields are malformed")
        defaults = payload.get("request_defaults")
        omitted = payload.get("endpoint_default_fields")
        if not isinstance(defaults, dict) or not isinstance(omitted, list):
            raise ValueError("certified execution profile values are malformed")
        if payload.get("all_sampling_settings_certified") is not False:
            raise ValueError("v1 profile cannot claim arbitrary sampling certification")
        return cls(
            profile_version=payload.get("profile_version"),
            certification_scope=payload.get("certification_scope"),
            request_defaults=dict(defaults),
            endpoint_default_fields=tuple(str(value) for value in omitted),
            explicit_request_fields_override=payload.get(
                "explicit_request_fields_override"
            ),
            provenance=payload.get("provenance"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_version": self.profile_version,
            "certification_scope": self.certification_scope,
            "request_defaults": dict(self.request_defaults),
            "endpoint_default_fields": list(self.endpoint_default_fields),
            "explicit_request_fields_override": self.explicit_request_fields_override,
            "provenance": self.provenance,
            "all_sampling_settings_certified": False,
        }

    def apply_defaults(
        self,
        kwargs: dict[str, Any],
        *,
        explicit_request_fields: set[str],
    ) -> None:
        """Apply certified defaults only for fields the application omitted."""

        for key, value in self.request_defaults.items():
            if key not in explicit_request_fields:
                kwargs.setdefault(key, value)


def compiler_execution_profile() -> CertifiedExecutionProfile:
    return CertifiedExecutionProfile()
