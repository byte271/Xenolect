"""Driver IR serialization and schema ops."""

from xenolect.driver.ir import (
    Driver,
    SchemaTransform,
    ToolEncoding,
    identity_driver,
)
from xenolect.driver.schema_ops import apply_schema_transforms, inline_refs
from xenolect.driver.serialize import driver_hash, load_driver, save_driver


def test_identity_hash_stable(tmp_path):
    d = identity_driver()
    h1 = driver_hash(d)
    path = tmp_path / "id.mdriver"
    save_driver(d, path)
    d2 = load_driver(path)
    assert driver_hash(d2) == h1


def test_inline_refs():
    schema = {
        "type": "object",
        "properties": {"item": {"$ref": "#/$defs/Item"}},
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
            }
        },
    }
    out = inline_refs(schema)
    assert "$ref" not in str(out)
    assert out["properties"]["item"]["properties"]["id"]["type"] == "integer"


def test_schema_transform_pipeline():
    schema = {
        "type": "object",
        "title": "T",
        "properties": {"x": {"type": "string", "title": "X"}},
    }
    out = apply_schema_transforms(schema, [SchemaTransform.STRIP_TITLES])
    assert "title" not in out
    assert "title" not in out["properties"]["x"]


def test_complexity_prefers_identity():
    ident = identity_driver()
    other = Driver(tool_encoding=ToolEncoding.XML_JSON)
    assert ident.complexity() < other.complexity()
