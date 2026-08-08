"""JSON Schema validation for tool arguments."""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


def validate_tool_arguments(
    arguments: Any,
    parameters: dict[str, Any],
) -> tuple[bool, str | None]:
    """Return (ok, error_message)."""
    try:
        Draft202012Validator(parameters).validate(arguments)
        return True, None
    except ValidationError as exc:
        return False, exc.message
    except Exception as exc:  # noqa: BLE001 — surface schema errors cleanly
        return False, str(exc)
