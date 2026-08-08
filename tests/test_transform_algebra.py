"""Proof obligation for canonicalising `Driver.schema_transforms`.

`Driver.schema_transforms` is declared as a list, but `driver_hash` treats the
list order as part of the driver's identity. Canonicalising that order is only
legitimate if the order is semantically irrelevant — otherwise canonicalisation
would silently change what a driver does.

This file discharges that obligation *before* the canonicalisation exists:

  * commutativity — every ordering of a transform subset produces byte-identical
    output on the same schema;
  * idempotence — repeating a transform changes nothing, so de-duplication is
    safe.

Both are checked against hand-picked adversarial schemas (`$ref` siblings, a
`$defs` entry literally named "properties", pre-existing `additionalProperties`,
titles on `$defs` members) and against Hypothesis-generated schemas.

If either property is ever falsified, `Driver` must stop canonicalising and the
driver identity guarantee is void.
"""

from __future__ import annotations

import json
from itertools import permutations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from xenolect.driver.ir import SchemaTransform
from xenolect.driver.schema_ops import apply_schema_transforms

ALL = list(SchemaTransform)


def _canon(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


ADVERSARIAL_SCHEMAS = [
    # plain object
    {"type": "object", "properties": {"a": {"type": "string"}}},
    # $ref with sibling keys, including one force_ap would otherwise add
    {
        "type": "object",
        "properties": {
            "x": {"$ref": "#/$defs/E", "title": "sibling", "additionalProperties": True}
        },
        "$defs": {
            "E": {
                "type": "object",
                "title": "E",
                "properties": {"n": {"type": "integer"}},
            }
        },
    },
    # a $defs entry literally named "properties"
    {
        "type": "object",
        "properties": {"y": {"$ref": "#/$defs/properties"}},
        "$defs": {"properties": {"type": "object", "properties": {"z": {}}}},
    },
    # nested arrays of objects, titles at several depths
    {
        "type": "object",
        "title": "Root",
        "properties": {
            "items": {
                "type": "array",
                "title": "Items",
                "items": {
                    "type": "object",
                    "title": "Item",
                    "properties": {"k": {"type": "string", "title": "K"}},
                },
            }
        },
    },
    # pre-existing additionalProperties in both polarities
    {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "open": {"type": "object", "additionalProperties": True, "properties": {}}
        },
    },
    # definitions/ spelling, ref to a ref
    {
        "type": "object",
        "properties": {"p": {"$ref": "#/definitions/A"}},
        "definitions": {
            "A": {"$ref": "#/definitions/B"},
            "B": {"type": "object", "properties": {"q": {"type": "number"}}},
        },
    },
    # dangling ref (no matching $defs entry)
    {"type": "object", "properties": {"d": {"$ref": "#/$defs/Missing"}}},
]


def _subsets():
    return [
        list(combo)
        for r in range(len(ALL) + 1)
        for combo in __import__("itertools").combinations(ALL, r)
    ]


def _assert_order_free(schema) -> None:
    for subset in _subsets():
        outputs = {
            _canon(apply_schema_transforms(schema, list(order)))
            for order in permutations(subset)
        }
        assert len(outputs) == 1, (
            f"schema_transforms order changes the result for {[t.value for t in subset]}: "
            f"{len(outputs)} distinct outputs on {_canon(schema)[:200]}"
        )


def _assert_idempotent(schema) -> None:
    for t in ALL:
        once = apply_schema_transforms(schema, [t])
        twice = apply_schema_transforms(schema, [t, t])
        assert _canon(once) == _canon(twice), f"{t.value} is not idempotent"
    everything = apply_schema_transforms(schema, ALL)
    doubled = apply_schema_transforms(schema, ALL + ALL)
    assert _canon(everything) == _canon(doubled)


def test_transforms_commute_on_adversarial_schemas():
    for schema in ADVERSARIAL_SCHEMAS:
        _assert_order_free(schema)


def test_transforms_are_idempotent_on_adversarial_schemas():
    for schema in ADVERSARIAL_SCHEMAS:
        _assert_idempotent(schema)


# --------------------------------------------------------------------------
# generated schemas
# --------------------------------------------------------------------------

_LEAF = st.sampled_from(
    [
        {"type": "string"},
        {"type": "integer"},
        {"type": "number"},
        {"type": "boolean"},
        {"type": "string", "title": "L"},
    ]
)


def _schema_tree(depth: int):
    if depth <= 0:
        return _LEAF
    inner = _schema_tree(depth - 1)
    obj = st.builds(
        lambda props, title, ap: {
            "type": "object",
            **({"title": title} if title else {}),
            **({"additionalProperties": ap} if ap is not None else {}),
            "properties": props,
        },
        st.dictionaries(
            st.sampled_from(["a", "b", "c", "properties", "title"]), inner, max_size=3
        ),
        st.one_of(st.none(), st.just("T")),
        st.one_of(st.none(), st.booleans()),
    )
    arr = st.builds(
        lambda items, title: {
            "type": "array",
            **({"title": title} if title else {}),
            "items": items,
        },
        inner,
        st.one_of(st.none(), st.just("A")),
    )
    ref = st.just({"$ref": "#/$defs/E"})
    ref_sib = st.just({"$ref": "#/$defs/E", "title": "S"})
    return st.one_of(_LEAF, obj, arr, ref, ref_sib)


_SCHEMA = st.builds(
    lambda body, defs: {
        "type": "object",
        "properties": {"root": body},
        "$defs": {"E": defs},
    },
    _schema_tree(3),
    _schema_tree(2),
)


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_SCHEMA)
def test_transforms_commute_on_generated_schemas(schema):
    _assert_order_free(schema)


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_SCHEMA)
def test_transforms_are_idempotent_on_generated_schemas(schema):
    _assert_idempotent(schema)
