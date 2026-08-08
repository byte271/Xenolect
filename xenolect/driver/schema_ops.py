"""Pure schema transforms applied before sending tools to a model."""

from __future__ import annotations

import copy
from typing import Any

from xenolect.driver.ir import SchemaTransform


def apply_schema_transforms(
    parameters: dict[str, Any],
    transforms: list[SchemaTransform],
) -> dict[str, Any]:
    schema = copy.deepcopy(parameters)
    for t in transforms:
        if t == SchemaTransform.INLINE_REFS:
            schema = inline_refs(schema)
        elif t == SchemaTransform.STRIP_TITLES:
            schema = strip_titles(schema)
        elif t == SchemaTransform.FORCE_ADDITIONAL_PROPERTIES_FALSE:
            schema = force_additional_properties_false(schema)
    return schema


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline local $ref against $defs / definitions.

    Two references cannot be expanded and are left in place:

      * **recursive** ones (``$defs.Node`` referring to ``#/$defs/Node``, the
        ordinary shape of a tree or linked-list type) — expanding them does not
        terminate;
      * **dangling** ones, whose target is not in the definition block.

    When any reference survives, the definition block is preserved so the result
    is still a resolvable schema. Previously it was dropped unconditionally,
    which turned a dangling reference into an invalid schema and made a recursive
    one raise `RecursionError` before it got that far.
    """
    defs_key = None
    for key in ("$defs", "definitions"):
        if isinstance(schema.get(key), dict):
            defs_key = key
            break
    defs = dict(schema.get(defs_key) or {})
    inlined, unresolved = _inline(schema, defs, _cyclic_names(defs))
    if unresolved and defs_key is not None:
        inlined = {**inlined, defs_key: copy.deepcopy(schema[defs_key])}
    return inlined


def _referenced_names(node: Any) -> set[str]:
    """Definition names referenced anywhere inside a subtree."""
    names: set[str] = set()
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            names.add(ref.rsplit("/", 1)[-1])
        for key, value in node.items():
            if key != "$ref":
                names |= _referenced_names(value)
    elif isinstance(node, list):
        for value in node:
            names |= _referenced_names(value)
    return names


def _cyclic_names(defs: dict[str, Any]) -> frozenset[str]:
    """Definition names that participate in a reference cycle.

    A cyclic definition has no finite inlined form, so it is left as a `$ref`.
    Detecting this up front (rather than stopping mid-descent) is what makes
    `inline_refs` idempotent: without it, each application unrolls the recursion
    one more level, and `[inline_refs, inline_refs]` would not equal
    `[inline_refs]`.
    """
    edges = {
        name: _referenced_names(body) & set(defs) for name, body in defs.items()
    }
    cyclic: set[str] = set()
    for start in edges:
        seen: set[str] = set()
        stack = list(edges[start])
        while stack:
            current = stack.pop()
            if current == start:
                cyclic.add(start)
                break
            if current in seen:
                continue
            seen.add(current)
            stack.extend(edges.get(current, ()))
    return frozenset(cyclic)


def _inline(
    node: Any, defs: dict[str, Any], cyclic: frozenset[str]
) -> tuple[Any, bool]:
    """Return (inlined node, whether any $ref could not be expanded)."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            # Support #/$defs/Name and #/definitions/Name
            name = ref.rsplit("/", 1)[-1]
            if name in defs and name not in cyclic:
                resolved = copy.deepcopy(defs[name])
                # Merge sibling keys (excluding $ref)
                siblings = {k: v for k, v in node.items() if k != "$ref"}
                if siblings and isinstance(resolved, dict):
                    # Prefer a prior `additionalProperties: false` over a sibling
                    # `true` so FORCE_ADDITIONAL_PROPERTIES_FALSE still commutes
                    # with INLINE_REFS when the definition body was already closed.
                    closed = resolved.get("additionalProperties") is False
                    resolved = {**resolved, **siblings}
                    if closed and (
                        resolved.get("type") == "object" or "properties" in resolved
                    ):
                        resolved["additionalProperties"] = False
                return _inline(resolved, defs, cyclic)
            # Recursive or dangling: no finite expansion exists.
            return node, True
        out: dict[str, Any] = {}
        unresolved = False
        for k, v in node.items():
            if k in ("$defs", "definitions"):
                continue
            out[k], sub_unresolved = _inline(v, defs, cyclic)
            unresolved = unresolved or sub_unresolved
        return out, unresolved
    if isinstance(node, list):
        items = [_inline(x, defs, cyclic) for x in node]
        return [x for x, _ in items], any(u for _, u in items)
    return node, False


# --------------------------------------------------------------------------
# Position-aware traversal.
#
# A JSON Schema document mixes three kinds of dictionary:
#
#   schema          keys are keywords          {"type": "object", "title": "T"}
#   map of schemas  keys are *names*           {"properties": {"title": {...}}}
#   data            keys are arbitrary         {"enum": [{"title": "x"}]}
#
# Treating all three the same corrupts ordinary tool schemas: a tool with an
# argument called `title` loses that argument to `strip_titles`, and a tool with
# an argument called `properties` gains a bogus `additionalProperties` *property*
# from `force_additional_properties_false`. Both are legal, unremarkable schemas.
# --------------------------------------------------------------------------

#: Keywords whose value is a map from *name* to schema.
SCHEMA_MAP_KEYWORDS = frozenset(
    {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
)

#: Keywords whose value is data, not schema — never rewrite inside these.
DATA_KEYWORDS = frozenset({"enum", "const", "default", "examples"})


def _map_schema(node: Any, fn, *, in_schema_map: bool = False) -> Any:
    """Apply `fn` to every dictionary that is genuinely a schema."""
    if isinstance(node, dict):
        if in_schema_map:
            # Keys here are property/definition names, not keywords.
            return {k: _map_schema(v, fn) for k, v in node.items()}
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k in DATA_KEYWORDS:
                out[k] = copy.deepcopy(v)
            else:
                out[k] = _map_schema(v, fn, in_schema_map=k in SCHEMA_MAP_KEYWORDS)
        return fn(out)
    if isinstance(node, list):
        return [_map_schema(x, fn) for x in node]
    return node


def strip_titles(node: Any) -> Any:
    """Remove the `title` *annotation keyword*.

    A property named `title` is a property, not an annotation, and is preserved.
    """
    return _map_schema(node, lambda schema: {k: v for k, v in schema.items() if k != "title"})


def force_additional_properties_false(node: Any) -> Any:
    """Close every object schema, without inventing properties.

    Only applied to dictionaries in schema position, so a tool argument named
    `properties` no longer causes `additionalProperties` to be injected into the
    property map itself.
    """

    def _close(schema: dict[str, Any]) -> dict[str, Any]:
        if schema.get("type") == "object" or "properties" in schema:
            # Force-close even when the author left additionalProperties: true.
            schema["additionalProperties"] = False
        return schema

    return _map_schema(node, _close)
