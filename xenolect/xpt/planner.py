"""Runtime representation of Xenolect's precompiled diagnostic program.

The product package ships only the compact request grammar, probe helpers and
serialized decision program required online.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

from xenolect.abi.events import ToolDef
from xenolect.driver.ir import (
    Driver,
    ParserKind,
    SchemaTransform,
    ToolEncoding,
    canonical_schema_transforms,
)
from xenolect.eval.schema import validate_tool_arguments
from xenolect.xpt.gauntlet import (
    GauntletInstance,
    gauntlet_tools,
    localization_probes,
    render_user_turn,
)
from xenolect.xpt.syndrome import Syndrome

COMPILED_DIR = Path(__file__).parent / "compiled"


def canonical_transforms(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        t.value for t in canonical_schema_transforms(SchemaTransform(v) for v in values)
    )


@dataclass(frozen=True)
class RequestConfig:
    tool_encoding: str
    transforms: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "transforms", canonical_transforms(self.transforms))

    @property
    def key(self) -> str:
        return f"{self.tool_encoding}[{'+'.join(self.transforms) or '-'}]"

    def driver(self, parser: str | None = None) -> Driver:
        return Driver(
            tool_encoding=ToolEncoding(self.tool_encoding),
            parser=ParserKind(parser or self.tool_encoding),
            schema_transforms=[SchemaTransform(t) for t in self.transforms],
        )


def request_config_from_key(key: str) -> RequestConfig:
    if "[" not in key or not key.endswith("]"):
        raise ValueError(f"invalid request config key: {key!r}")
    encoding, raw = key[:-1].split("[", 1)
    transforms = () if raw in ("", "-") else tuple(part for part in raw.split("+") if part)
    return RequestConfig(encoding, transforms)


@lru_cache(maxsize=1)
def all_request_configs() -> tuple[RequestConfig, ...]:
    flags = [t.value for t in SchemaTransform]
    subsets = [
        canonical_transforms(c)
        for r in range(len(flags) + 1)
        for c in combinations(flags, r)
    ]
    return tuple(RequestConfig(enc.value, sub) for enc in ToolEncoding for sub in subsets)


@dataclass(frozen=True)
class ProbeTemplate:
    id: str
    kind: str
    config: RequestConfig
    progresses_trajectory: bool


def probe_payload(
    probe: ProbeTemplate, inst: GauntletInstance
) -> tuple[list[ToolDef], str, dict[str, dict[str, Any]]]:
    if probe.kind == "gauntlet_turn1":
        return gauntlet_tools(), render_user_turn(inst), inst.expected_batch_arguments()
    by_id = {p.id: p for p in localization_probes(inst)}
    if probe.kind not in by_id:
        raise ValueError(f"unknown probe kind in compiled program: {probe.kind!r}")
    slp = by_id[probe.kind]
    return list(slp.tools), slp.user_content, {slp.expected_tool: slp.expected_arguments}


def observation_class(syn: Syndrome, expected: dict[str, dict[str, Any]]) -> str:
    if not syn.transport_ok:
        return "transport_error"
    if syn.consensus.value == "ambiguous":
        return "ambiguous_parse"
    if not syn.tool_call_emitted:
        hint = ""
        if syn.saw_xml_marker:
            hint = ":xml_marker"
        elif syn.saw_tagged_marker:
            hint = ":tagged_marker"
        return f"no_calls{hint}"
    called = sorted(set(syn.tool_names))
    correct = sorted(n for n in expected if syn.args_values_correct.get(n) is True)
    valid = sorted(n for n in expected if syn.args_schema_valid.get(n) is True)
    return f"calls:{','.join(called)}|valid:{','.join(valid)}|exact:{','.join(correct)}"


def probe_succeeded(
    syn: Syndrome, expected: dict[str, dict[str, Any]], *, batch: bool
) -> bool:
    if not syn.tool_call_emitted or syn.unknown_tool_names:
        return False
    if set(syn.tool_names) != set(expected):
        return False
    if batch and not syn.parallel_batch_present:
        return False
    if not all(syn.args_schema_valid.get(n) for n in expected):
        return False
    return all(syn.args_values_correct.get(n) for n in expected)


def annotate_arguments(
    syn: Syndrome, tools: list[ToolDef], expected: dict[str, dict[str, Any]]
) -> None:
    schema_by_name = {t.name: t.parameters for t in tools}
    if syn.accepted_parser is None:
        return
    for call in syn.parser_outcomes[syn.accepted_parser].calls:
        schema = schema_by_name.get(call.name)
        if schema is not None:
            ok, _ = validate_tool_arguments(call.arguments, schema)
            syn.args_schema_valid[call.name] = ok
        if call.name in expected:
            syn.args_values_correct[call.name] = call.arguments == expected[call.name]


@dataclass
class DecisionNode:
    node_id: str
    n_hypotheses: int = 0
    probe_id: str | None = None
    reason: str = ""
    children: dict[str, str] = field(default_factory=dict)
    conclusion: str | None = None
    unsupported: bool = False


@dataclass
class DiagnosticProgram:
    root: str
    nodes: dict[str, DecisionNode]
    probe_index: dict[str, ProbeTemplate]


def load_compiled_program(path: str | Path | None = None) -> DiagnosticProgram:
    source = Path(path) if path is not None else COMPILED_DIR / "diagnostic_program.json"
    data = json.loads(source.read_text(encoding="utf-8"))

    probe_index: dict[str, ProbeTemplate] = {}
    for probe_id, raw in data.get("probes", {}).items():
        if not isinstance(raw, dict):
            raise ValueError(f"compiled probe {probe_id!r} is malformed")
        probe_index[str(probe_id)] = ProbeTemplate(
            id=str(probe_id),
            kind=str(raw["kind"]),
            config=request_config_from_key(str(raw["config"])),
            progresses_trajectory=bool(raw.get("progresses_trajectory", False)),
        )

    nodes: dict[str, DecisionNode] = {}
    for node_id, raw in data["nodes"].items():
        count = int(raw.get("n_hypotheses", 0))
        nodes[str(node_id)] = DecisionNode(
            node_id=str(raw.get("node_id", node_id)),
            n_hypotheses=count,
            probe_id=raw.get("probe_id"),
            reason=str(raw.get("reason", "")),
            children={str(k): str(v) for k, v in raw.get("children", {}).items()},
            conclusion=raw.get("conclusion"),
            unsupported=bool(raw.get("unsupported", False)),
        )

    root = str(data["root"])
    if root not in nodes:
        raise ValueError(f"compiled diagnostic program root {root!r} is missing")
    for node in nodes.values():
        if node.probe_id is not None and node.probe_id not in probe_index:
            raise ValueError(f"compiled node {node.node_id} references unknown probe {node.probe_id}")
    return DiagnosticProgram(
        root=root,
        nodes=nodes,
        probe_index=probe_index,
    )
