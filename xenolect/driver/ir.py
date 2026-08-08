"""Minimal typed Driver IR."""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable

from pydantic import BaseModel, Field, field_validator

from xenolect.abi import ABI_VERSION


class ToolEncoding(str, Enum):
    NATIVE = "native"
    TAGGED_JSON = "tagged_json"
    XML_JSON = "xml_json"


class ToolResultEncoding(str, Enum):
    TOOL_ROLE = "tool_role"
    USER_MESSAGE = "user_message"


class SchemaTransform(str, Enum):
    INLINE_REFS = "inline_refs"
    STRIP_TITLES = "strip_titles"
    FORCE_ADDITIONAL_PROPERTIES_FALSE = "force_additional_properties_false"


class ParserKind(str, Enum):
    NATIVE = "native"
    TAGGED_JSON = "tagged_json"
    XML_JSON = "xml_json"


DRIVER_GRAMMAR_VERSION = "0.1"


def driver_grammar_size() -> int:
    """Number of concrete programs representable by the v0.1 Driver grammar."""
    return (
        len(ToolEncoding)
        * (2 ** len(SchemaTransform))
        * len(ParserKind)
        * len(ToolResultEncoding)
    )


def canonical_schema_transforms(
    transforms: Iterable[SchemaTransform],
) -> list[SchemaTransform]:
    """De-duplicate and order a transform selection canonically.

    `schema_transforms` is declared as a list, so `[A, B]` and `[B, A]` would
    otherwise be different drivers with different `driver_hash` values despite
    being the same program. Canonical identity is required for cache keys and
    content-addressed artifacts.

    Collapsing the list to a canonical set is only sound because the v0
    transforms commute and are idempotent on JSON Schema documents. That is not
    assumed: `tests/test_transform_algebra.py` checks it over adversarial and
    generated schemas. If a future transform breaks either property, this
    normalisation must be revisited before the transform is added.
    """
    selected = set(transforms)
    return [t for t in SchemaTransform if t in selected]


class Driver(BaseModel):
    """Portable, deterministic driver program (IR)."""

    ir_version: str = "0.1"
    target_abi: str = ABI_VERSION
    tool_encoding: ToolEncoding = ToolEncoding.NATIVE
    tool_result_encoding: ToolResultEncoding = ToolResultEncoding.TOOL_ROLE
    schema_transforms: list[SchemaTransform] = Field(default_factory=list)
    parser: ParserKind = ParserKind.NATIVE

    @field_validator("schema_transforms")
    @classmethod
    def _canonicalize_transforms(
        cls, value: list[SchemaTransform]
    ) -> list[SchemaTransform]:
        return canonical_schema_transforms(value)

    def complexity(self) -> int:
        """Simple complexity score for minimality (lower is better)."""
        score = 0
        if self.tool_encoding != ToolEncoding.NATIVE:
            score += 2
        if self.tool_result_encoding != ToolResultEncoding.TOOL_ROLE:
            score += 2
        if self.parser != ParserKind.NATIVE:
            score += 1
        score += len(self.schema_transforms)
        # Prefer parser matching encoding.
        if self.parser.value != self.tool_encoding.value:
            score += 3
        return score

    def canonical_dict(self) -> dict[str, Any]:
        # Re-normalise defensively: `model_copy(update=...)` bypasses validators
        # in pydantic v2, so an out-of-order list can otherwise reach hashing.
        payload = self.model_dump(mode="json")
        payload["schema_transforms"] = [
            t.value for t in canonical_schema_transforms(self.schema_transforms)
        ]
        return payload


def identity_driver() -> Driver:
    """Nominal OpenAI-compatible behavior; no transforms."""
    return Driver(
        tool_encoding=ToolEncoding.NATIVE,
        tool_result_encoding=ToolResultEncoding.TOOL_ROLE,
        schema_transforms=[],
        parser=ParserKind.NATIVE,
    )


def with_encoding(driver: Driver, encoding: ToolEncoding) -> Driver:
    parser = ParserKind(encoding.value)
    return driver.model_copy(update={"tool_encoding": encoding, "parser": parser})


def with_tool_result(driver: Driver, encoding: ToolResultEncoding) -> Driver:
    return driver.model_copy(update={"tool_result_encoding": encoding})


def with_schema_transform(driver: Driver, transform: SchemaTransform) -> Driver:
    transforms = list(driver.schema_transforms)
    if transform not in transforms:
        transforms.append(transform)
    return driver.model_copy(update={"schema_transforms": transforms})
