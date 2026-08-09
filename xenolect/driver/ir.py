"""Typed Driver IR.

``Driver`` keeps the frozen v0.1 fields readable so already-installed artifacts
retain their identity.  New v0.2 artifacts carry a composable ``protocol``
program.  The execution modules interpret the protocol primitives; the legacy
enums are translated to the same primitives at runtime.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum, StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


DRIVER_GRAMMAR_VERSION = "0.2"


class _IRModel(BaseModel):
    """Fail closed when an artifact contains a primitive we do not implement."""

    model_config = ConfigDict(extra="forbid")


class TextFrame(_IRModel):
    """Literal text surrounding one strict JSON tool-call object."""

    prefix: str = ""
    suffix: str = ""
    case_sensitive: bool = True
    whitespace_after_prefix: bool = False
    flexible_whitespace: bool = False


class ToolCallFields(_IRModel):
    """Map protocol object keys to the normalized ToolCall fields."""

    name: str = "name"
    arguments: str = "arguments"
    call_id: str | None = "id"

    @field_validator("name", "arguments")
    @classmethod
    def _required_key_is_not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("tool-call field names must not be empty")
        return value

    @model_validator(mode="after")
    def _keys_are_distinct(self) -> ToolCallFields:
        keys = [self.name, self.arguments]
        if self.call_id is not None:
            if not self.call_id:
                raise ValueError("call_id field must be non-empty or null")
            keys.append(self.call_id)
        if len(set(keys)) != len(keys):
            raise ValueError("tool-call field names must be distinct")
        return self


class NativeToolsRequest(_IRModel):
    op: Literal["native_tools"] = "native_tools"


class JsonToolCatalogRequest(_IRModel):
    """Inject transformed tool definitions as a JSON catalog in one message."""

    op: Literal["json_tool_catalog"] = "json_tool_catalog"
    # The current Chat Completions runtime has certified insertion semantics for
    # system catalogs only. Add other roles only with ordering/certification tests.
    role: Literal["system"] = "system"
    instruction: str
    catalog_heading: str = "Available tools (JSON):"
    call_frame: TextFrame
    fields: ToolCallFields = Field(default_factory=ToolCallFields)


class ToolDefinitionFields(_IRModel):
    """Map normalized tool-definition fields into a JSON catalog item."""

    name: str = "name"
    description: str = "description"
    parameters: str = "parameters"

    @model_validator(mode="after")
    def _keys_are_non_empty_and_distinct(self) -> ToolDefinitionFields:
        values = (self.name, self.description, self.parameters)
        if any(not value for value in values):
            raise ValueError("tool-definition field names must not be empty")
        if len(set(values)) != len(values):
            raise ValueError("tool-definition field names must be distinct")
        return self


class TemplatedJsonToolCatalogRequest(_IRModel):
    """Render a JSON catalog with parameterized placement and object shape.

    ``catalog_path`` describes nested object keys around the tool list.  For
    example ``["registry", "entries"]`` renders
    ``{"registry": {"entries": [...]}}``.  This is a structural primitive,
    not a named provider/model dialect.
    """

    op: Literal["templated_json_tool_catalog"] = "templated_json_tool_catalog"
    role: Literal["system", "user"]
    instruction: str
    catalog_heading: str = ""
    catalog_path: list[str] = Field(default_factory=list, max_length=4)
    tool_fields: ToolDefinitionFields = Field(default_factory=ToolDefinitionFields)
    call_frame: TextFrame
    fields: ToolCallFields = Field(default_factory=ToolCallFields)

    @field_validator("catalog_path")
    @classmethod
    def _catalog_path_keys_are_valid(cls, value: list[str]) -> list[str]:
        if any(not key for key in value):
            raise ValueError("catalog path keys must not be empty")
        return value


RequestPrimitive = Annotated[
    NativeToolsRequest | JsonToolCatalogRequest | TemplatedJsonToolCatalogRequest,
    Field(discriminator="op"),
]


class NativeToolCallsParser(_IRModel):
    op: Literal["native_tool_calls"] = "native_tool_calls"


class FramedJsonToolCallsParser(_IRModel):
    """Extract one or more strict JSON objects from a parameterized text frame."""

    op: Literal["framed_json_tool_calls"] = "framed_json_tool_calls"
    frame: TextFrame
    fields: ToolCallFields = Field(default_factory=ToolCallFields)
    multiple: bool = True
    whole_content: bool = False
    capture_surrounding_text: bool = False

    @model_validator(mode="after")
    def _whole_content_has_no_frame(self) -> FramedJsonToolCallsParser:
        if self.whole_content and (self.frame.prefix or self.frame.suffix):
            raise ValueError("whole-content JSON parser cannot also declare a text frame")
        if not self.whole_content and not self.frame.prefix:
            raise ValueError("framed JSON parser requires a non-empty prefix")
        return self


class JsonObjectToolCallsParser(_IRModel):
    """Extract strict tool-call objects embedded anywhere in assistant text.

    This is deliberately structural rather than a named response format.  Only
    JSON objects containing both configured tool-call fields are claimed; other
    JSON in the surrounding assistant text is preserved.
    """

    op: Literal["json_object_tool_calls"] = "json_object_tool_calls"
    fields: ToolCallFields = Field(default_factory=ToolCallFields)
    multiple: bool = True
    capture_surrounding_text: bool = False


ResponsePrimitive = Annotated[
    NativeToolCallsParser | FramedJsonToolCallsParser | JsonObjectToolCallsParser,
    Field(discriminator="op"),
]


class ResultLiteral(_IRModel):
    op: Literal["literal"] = "literal"
    text: str


class ResultField(_IRModel):
    op: Literal["field"] = "field"
    field: Literal["call_id", "name", "content"]
    prefix: str = ""
    suffix: str = ""
    omit_if_none: bool = True


ToolResultSegment = Annotated[ResultLiteral | ResultField, Field(discriminator="op")]


class ToolResultMessage(_IRModel):
    """Render a tool result by composing literal and field segments."""

    role: Literal["tool", "user", "assistant"]
    segments: list[ToolResultSegment]
    attach_tool_call_id: bool = False

    @model_validator(mode="after")
    def _has_content_and_valid_role_fields(self) -> ToolResultMessage:
        if not any(
            isinstance(segment, ResultField) and segment.field == "content"
            for segment in self.segments
        ):
            raise ValueError("tool-result program must render content")
        if self.attach_tool_call_id and self.role != "tool":
            raise ValueError("tool_call_id can only be attached to role=tool messages")
        return self


class StateAction(StrEnum):
    """Tool ABI state actions represented explicitly in a v0.2 program."""

    TRACK_OUTSTANDING_CALLS = "track_outstanding_calls"
    APPEND_TOOL_RESULTS = "append_tool_results"
    RESUME_WHEN_ALL_RESULTS = "resume_when_all_results"


REQUIRED_STATE_ACTIONS: tuple[StateAction, ...] = tuple(StateAction)


class ProtocolProgram(_IRModel):
    """Composable protocol compatibility program executed by a Driver."""

    request: list[RequestPrimitive]
    response: list[ResponsePrimitive]
    tool_result: ToolResultMessage
    state: list[StateAction] = Field(default_factory=lambda: list(REQUIRED_STATE_ACTIONS))

    @model_validator(mode="after")
    def _validate_program(self) -> ProtocolProgram:
        if not self.request:
            raise ValueError("protocol request pipeline must not be empty")
        if not self.response:
            raise ValueError("protocol response pipeline must not be empty")
        if len(self.state) != len(set(self.state)):
            raise ValueError("protocol state actions must not contain duplicates")
        missing = [action.value for action in REQUIRED_STATE_ACTIONS if action not in self.state]
        if missing:
            raise ValueError(
                "unsupported Tool ABI state program; missing required actions: "
                + ", ".join(missing)
            )
        self.state = list(REQUIRED_STATE_ACTIONS)
        return self


def driver_grammar_size() -> int:
    """Size of the bounded legacy frontier currently searched online.

    The v0.2 protocol IR is parameterized and therefore has no honest finite
    grammar size.  The legacy compatibility path still starts with the proven
    choices from the 144-program v0.1 frontier.  The active path instead carries
    typed request/response/result holes and may compose parameterized v0.2
    primitives from bounded black-box evidence.
    """
    return (
        len(ToolEncoding) * (2 ** len(SchemaTransform)) * len(ParserKind) * len(ToolResultEncoding)
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

    model_config = ConfigDict(extra="forbid")

    ir_version: str = "0.1"
    target_abi: str = ABI_VERSION
    tool_encoding: ToolEncoding = ToolEncoding.NATIVE
    tool_result_encoding: ToolResultEncoding = ToolResultEncoding.TOOL_ROLE
    schema_transforms: list[SchemaTransform] = Field(default_factory=list)
    parser: ParserKind = ParserKind.NATIVE
    protocol: ProtocolProgram | None = None

    @field_validator("schema_transforms")
    @classmethod
    def _canonicalize_transforms(cls, value: list[SchemaTransform]) -> list[SchemaTransform]:
        return canonical_schema_transforms(value)

    @model_validator(mode="after")
    def _version_matches_representation(self) -> Driver:
        if self.protocol is None and self.ir_version != "0.1":
            raise ValueError(f"Driver IR {self.ir_version!r} requires an explicit protocol program")
        if self.protocol is not None and self.ir_version != "0.2":
            raise ValueError("composable protocol programs require Driver IR '0.2'")
        return self

    def complexity(self) -> int:
        """Simple complexity score for minimality (lower is better)."""
        if self.protocol is not None:
            non_native_request = sum(
                isinstance(op, (JsonToolCatalogRequest, TemplatedJsonToolCatalogRequest))
                for op in self.protocol.request
            )
            non_native_parser = sum(
                isinstance(op, (FramedJsonToolCallsParser, JsonObjectToolCallsParser))
                for op in self.protocol.response
            )
            result_segments = max(0, len(self.protocol.tool_result.segments) - 1)
            return (
                non_native_request * 2
                + non_native_parser
                + result_segments
                + len(self.schema_transforms)
            )
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
        if self.protocol is not None:
            if self.ir_version != "0.2":
                raise ValueError("composable protocol programs require Driver IR '0.2'")
            return {
                "ir_version": "0.2",
                "target_abi": self.target_abi,
                "schema_transforms": [
                    t.value for t in canonical_schema_transforms(self.schema_transforms)
                ],
                "protocol": self.protocol.model_dump(mode="json"),
            }

        if self.ir_version != "0.1":
            raise ValueError(f"Driver IR {self.ir_version!r} requires an explicit protocol program")
        # Excluding the absent v0.2 field is correctness-critical: old .mdriver
        # files keep the same content hash after upgrading Xenolect.
        payload = self.model_dump(mode="json", exclude={"protocol"})
        payload["schema_transforms"] = [
            t.value for t in canonical_schema_transforms(self.schema_transforms)
        ]
        return payload


def _native_result_program() -> ToolResultMessage:
    return ToolResultMessage(
        role="tool",
        segments=[ResultField(field="content")],
        attach_tool_call_id=True,
    )


def _user_result_program() -> ToolResultMessage:
    return ToolResultMessage(
        role="user",
        segments=[
            ResultLiteral(text="TOOL_RESULT"),
            ResultField(field="call_id", prefix=" id="),
            ResultField(field="name", prefix=" name="),
            ResultLiteral(text="\n"),
            ResultField(field="content"),
        ],
    )


def legacy_protocol_program(
    tool_encoding: ToolEncoding,
    parser: ParserKind,
    result_encoding: ToolResultEncoding,
) -> ProtocolProgram:
    """Translate one frozen v0.1 tuple into executable v0.2 primitives."""
    fields = ToolCallFields()
    if tool_encoding == ToolEncoding.NATIVE:
        request: list[RequestPrimitive] = [NativeToolsRequest()]
    elif tool_encoding == ToolEncoding.TAGGED_JSON:
        request = [
            JsonToolCatalogRequest(
                instruction=(
                    "You may call tools by emitting a line exactly like:\n"
                    'TOOL_CALL {"name": "<tool>", "arguments": {..}, "id": "<id>"}'
                ),
                catalog_heading="Available tools (JSON):",
                call_frame=TextFrame(prefix="TOOL_CALL "),
                fields=fields,
            )
        ]
    elif tool_encoding == ToolEncoding.XML_JSON:
        request = [
            JsonToolCatalogRequest(
                instruction=(
                    "You may call tools using:\n"
                    '<tool_call>{"name": "...", "arguments": {}, "id": "..."}</tool_call>'
                ),
                catalog_heading="Available tools:",
                call_frame=TextFrame(
                    prefix="<tool_call>", suffix="</tool_call>", case_sensitive=False
                ),
                fields=fields,
            )
        ]
    else:  # pragma: no cover - Enum construction prevents this
        raise ValueError(f"unsupported legacy tool encoding: {tool_encoding!r}")

    response: list[ResponsePrimitive] = [NativeToolCallsParser()]
    if parser == ParserKind.TAGGED_JSON:
        response.extend(
            [
                FramedJsonToolCallsParser(
                    frame=TextFrame(prefix="TOOL_CALL", whitespace_after_prefix=True),
                    fields=fields,
                ),
                FramedJsonToolCallsParser(
                    frame=TextFrame(),
                    fields=fields,
                    multiple=False,
                    whole_content=True,
                ),
            ]
        )
    elif parser == ParserKind.XML_JSON:
        response.append(
            FramedJsonToolCallsParser(
                frame=TextFrame(
                    prefix="<tool_call >",
                    suffix="</tool_call >",
                    case_sensitive=False,
                    whitespace_after_prefix=True,
                    flexible_whitespace=True,
                ),
                fields=fields,
            )
        )
    elif parser != ParserKind.NATIVE:  # pragma: no cover
        raise ValueError(f"unsupported legacy parser: {parser!r}")

    return ProtocolProgram(
        request=request,
        response=response,
        tool_result=(
            _native_result_program()
            if result_encoding == ToolResultEncoding.TOOL_ROLE
            else _user_result_program()
        ),
    )


def effective_protocol(driver: Driver) -> ProtocolProgram:
    """Return the program to execute for either Driver IR generation."""
    if driver.protocol is not None:
        return driver.protocol
    return legacy_protocol_program(
        driver.tool_encoding,
        driver.parser,
        driver.tool_result_encoding,
    )


def composed_driver(
    *,
    tool_encoding: ToolEncoding,
    parser: ParserKind,
    tool_result_encoding: ToolResultEncoding,
    schema_transforms: Iterable[SchemaTransform] = (),
) -> Driver:
    """Compose observed legacy frontier choices into a v0.2 Driver artifact."""
    return Driver(
        ir_version="0.2",
        schema_transforms=list(schema_transforms),
        protocol=legacy_protocol_program(tool_encoding, parser, tool_result_encoding),
    )


def identity_driver() -> Driver:
    """Nominal OpenAI-compatible behavior; no transforms."""
    return Driver(
        tool_encoding=ToolEncoding.NATIVE,
        tool_result_encoding=ToolResultEncoding.TOOL_ROLE,
        schema_transforms=[],
        parser=ParserKind.NATIVE,
    )


def with_encoding(driver: Driver, encoding: ToolEncoding) -> Driver:
    if driver.protocol is not None:
        raise ValueError(
            "with_encoding() is a legacy Driver helper; edit the v0.2 request/response "
            "primitives explicitly"
        )
    parser = ParserKind(encoding.value)
    return driver.model_copy(update={"tool_encoding": encoding, "parser": parser})


def with_tool_result(driver: Driver, encoding: ToolResultEncoding) -> Driver:
    if driver.protocol is not None:
        raise ValueError(
            "with_tool_result() is a legacy Driver helper; edit the v0.2 tool-result "
            "primitive explicitly"
        )
    return driver.model_copy(update={"tool_result_encoding": encoding})


def with_schema_transform(driver: Driver, transform: SchemaTransform) -> Driver:
    transforms = list(driver.schema_transforms)
    if transform not in transforms:
        transforms.append(transform)
    return driver.model_copy(update={"schema_transforms": transforms})
