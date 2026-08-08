"""Canonical Driver identity.

Driver identity is correctness-critical: it keys cached equivalence classes and
comparison, diagnostic leaves and artifact fingerprints. Two drivers that are the
same program must be the same driver.

The soundness of collapsing `schema_transforms` to a canonical set is proven
separately in `tests/test_transform_algebra.py` (commutativity + idempotence).
This file asserts the identity invariants that proof licenses.
"""

from __future__ import annotations

from itertools import combinations, permutations

from xenolect.driver.ir import (
    Driver,
    ParserKind,
    SchemaTransform,
    ToolEncoding,
    ToolResultEncoding,
    canonical_schema_transforms,
    with_schema_transform,
)
from xenolect.driver.serialize import driver_hash, driver_to_json, load_driver, save_driver

ALL = list(SchemaTransform)


def test_transform_order_does_not_change_driver_identity():
    for r in range(len(ALL) + 1):
        for subset in combinations(ALL, r):
            hashes = {
                driver_hash(Driver(schema_transforms=list(order)))
                for order in permutations(subset)
            }
            assert len(hashes) == 1, f"{[t.value for t in subset]} produced {hashes}"


def test_duplicate_transforms_do_not_change_driver_identity():
    base = Driver(schema_transforms=[SchemaTransform.INLINE_REFS, SchemaTransform.STRIP_TITLES])
    dupes = Driver(
        schema_transforms=[
            SchemaTransform.STRIP_TITLES,
            SchemaTransform.INLINE_REFS,
            SchemaTransform.INLINE_REFS,
            SchemaTransform.STRIP_TITLES,
        ]
    )
    assert driver_hash(base) == driver_hash(dupes)
    assert base.schema_transforms == dupes.schema_transforms


def test_canonical_order_is_the_ir_declaration_order():
    d = Driver(
        schema_transforms=[
            SchemaTransform.FORCE_ADDITIONAL_PROPERTIES_FALSE,
            SchemaTransform.INLINE_REFS,
        ]
    )
    assert d.schema_transforms == [
        SchemaTransform.INLINE_REFS,
        SchemaTransform.FORCE_ADDITIONAL_PROPERTIES_FALSE,
    ]
    assert canonical_schema_transforms(reversed(ALL)) == ALL


def test_model_copy_cannot_produce_a_non_canonical_hash():
    """`model_copy(update=...)` skips validators; hashing must still canonicalise."""
    canonical = Driver(
        schema_transforms=[SchemaTransform.INLINE_REFS, SchemaTransform.STRIP_TITLES]
    )
    sneaky = Driver().model_copy(
        update={
            "schema_transforms": [
                SchemaTransform.STRIP_TITLES,
                SchemaTransform.INLINE_REFS,
            ]
        }
    )
    assert driver_hash(sneaky) == driver_hash(canonical)
    assert driver_to_json(sneaky) == driver_to_json(canonical)


def test_round_trip_through_disk_preserves_identity(tmp_path):
    d = Driver(
        tool_encoding=ToolEncoding.XML_JSON,
        parser=ParserKind.NATIVE,
        tool_result_encoding=ToolResultEncoding.USER_MESSAGE,
        schema_transforms=[SchemaTransform.STRIP_TITLES, SchemaTransform.INLINE_REFS],
    )
    path = tmp_path / "d.mdriver"
    save_driver(d, path)
    assert driver_hash(load_driver(path)) == driver_hash(d)


def test_loading_a_legacy_non_canonical_artifact_canonicalises(tmp_path):
    path = tmp_path / "legacy.mdriver"
    path.write_text(
        '{"ir_version":"0.1","target_abi":"tool-abi-v0","tool_encoding":"native",'
        '"tool_result_encoding":"tool_role",'
        '"schema_transforms":["strip_titles","inline_refs","strip_titles"],'
        '"parser":"native"}',
        encoding="utf-8",
    )
    loaded = load_driver(path)
    assert loaded.schema_transforms == [
        SchemaTransform.INLINE_REFS,
        SchemaTransform.STRIP_TITLES,
    ]
    assert driver_hash(loaded) == driver_hash(
        Driver(schema_transforms=[SchemaTransform.INLINE_REFS, SchemaTransform.STRIP_TITLES])
    )


def test_with_schema_transform_helper_stays_canonical():
    d = Driver(schema_transforms=[SchemaTransform.FORCE_ADDITIONAL_PROPERTIES_FALSE])
    d = with_schema_transform(d, SchemaTransform.INLINE_REFS)
    assert driver_hash(d) == driver_hash(
        Driver(
            schema_transforms=[
                SchemaTransform.INLINE_REFS,
                SchemaTransform.FORCE_ADDITIONAL_PROPERTIES_FALSE,
            ]
        )
    )
