"""The ABI Gauntlet: a compact stateful compatibility probe.

The whole
mandatory obligation set is discharged by a single **three**-generation stateful
trajectory:

    G1  structured multi-call challenge      OB01..OB09, OB17
    G2  hidden-result + ToolError challenge  OB10..OB14
    G3  no-call / termination challenge      OB15, OB16
    (whole trace)                            OB18

G2 deliberately merges the tool-result and the ToolError challenge: two of the
three parallel calls come back with fresh sentinels and the third comes back as a
ToolError carrying a fresh code, and the model must answer with a *parallel*
recovery batch. That is a strictly stronger ABI test than testing them in
sequence (it exercises mixed result/error batches), and it removes one expensive
generation from both diagnosis and certification.

Two secondary properties are engineered into G1 on purpose:

  * The offered tool set is a *schema differential*. `record_alpha` carries a
    local `$ref`, `record_gamma` carries `title` keys, `record_beta` carries
    neither. An endpoint that silently drops a tool whose schema it dislikes
    therefore localises the responsible schema feature inside ONE generation,
    at no extra endpoint cost.
  * The semantic difficulty is intentionally low. This checks interface
    behaviour, not general model intelligence.

Sentinels are minted AFTER the prompt is emitted, so no expected value is ever
visible in the user turn (existing dynamic-sentinel philosophy, reused).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from xenolect.abi.events import ToolDef

# --------------------------------------------------------------------------
# Tool schemas. The ABI-level schema is what the *application* declares; a
# driver's schema_transforms may rewrite it before it reaches the endpoint.
# --------------------------------------------------------------------------


def _alpha_schema() -> dict[str, Any]:
    """Nested object behind a local $ref. No `title` keys anywhere."""
    return {
        "type": "object",
        "properties": {"entry": {"$ref": "#/$defs/Entry"}},
        "required": ["entry"],
        "$defs": {
            "Entry": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "size": {"type": "integer"},
                },
                "required": ["code", "size"],
            }
        },
    }


def _beta_schema() -> dict[str, Any]:
    """Flat control: same shape as alpha, no $ref and no titles."""
    return {
        "type": "object",
        "properties": {
            "entry": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "size": {"type": "integer"},
                },
                "required": ["code", "size"],
            }
        },
        "required": ["entry"],
    }


def _gamma_schema() -> dict[str, Any]:
    """Array argument carrying `title` keys. No $ref."""
    return {
        "type": "object",
        "title": "RecordGammaArguments",
        "properties": {
            "tags": {
                "type": "array",
                "title": "Tags",
                "items": {"type": "string", "title": "Tag"},
            }
        },
        "required": ["tags"],
    }


def _delta_schema() -> dict[str, Any]:
    """Author-declared `additionalProperties: false`, no $ref, no titles.

    Diagnostic-only (see `localization_probes`). It is deliberately NOT part of
    the gauntlet's offered set: `force_additional_properties_false` is the one
    transform that *adds* a keyword, so a tool carrying it would make the whole
    request unacceptable to an endpoint that rejects the keyword — the gauntlet
    would then fail for a reason the Tool ABI never asked for.
    """
    return {
        "type": "object",
        "properties": {"note": {"type": "string"}},
        "required": ["note"],
        "additionalProperties": False,
    }


def _commit_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"alpha": {"type": "string"}, "beta": {"type": "string"}},
        "required": ["alpha", "beta"],
    }


def _report_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
    }


#: Which schema feature each differential tool isolates.
SCHEMA_DIFFERENTIAL: dict[str, str] = {
    "record_alpha": "ref",
    "record_gamma": "title",
    "record_beta": "none",
}


def gauntlet_tools() -> list[ToolDef]:
    return [
        ToolDef(
            name="record_alpha", description="Record an alpha entry", parameters=_alpha_schema()
        ),
        ToolDef(name="record_beta", description="Record a beta entry", parameters=_beta_schema()),
        ToolDef(name="record_gamma", description="Record gamma tags", parameters=_gamma_schema()),
        ToolDef(name="commit", description="Commit two tokens", parameters=_commit_schema()),
        ToolDef(name="report", description="Report an error code", parameters=_report_schema()),
    ]


BATCH_TOOLS: tuple[str, ...] = ("record_alpha", "record_beta", "record_gamma")
RECOVERY_TOOLS: tuple[str, ...] = ("commit", "report")


# --------------------------------------------------------------------------
# Challenge / sentinel material
# --------------------------------------------------------------------------


def _tok(material: str, prefix: str, length: int = 10) -> str:
    return prefix + hashlib.sha256(material.encode()).hexdigest()[:length].upper()


@dataclass(frozen=True)
class GauntletInstance:
    """One concrete, freshly randomised instantiation of the gauntlet."""

    salt: str
    surface_form: str          # which paraphrase of the script is used
    alpha_code: str            # challenge values -> visible in the prompt
    beta_code: str
    size: int
    gamma_tags: tuple[str, str]
    delta_note: str
    alpha_token: str           # sentinels -> minted after the prompt, never visible
    beta_token: str
    error_code: str
    ack_value: str

    # ---------------- challenge (prompt-visible) ----------------

    def expected_batch_arguments(self) -> dict[str, dict[str, Any]]:
        return {
            "record_alpha": {"entry": {"code": self.alpha_code, "size": self.size}},
            "record_beta": {"entry": {"code": self.beta_code, "size": self.size}},
            "record_gamma": {"tags": list(self.gamma_tags)},
        }

    def expected_recovery_arguments(self) -> dict[str, dict[str, Any]]:
        return {
            "commit": {"alpha": self.alpha_token, "beta": self.beta_token},
            "report": {"code": self.error_code},
        }

    # ---------------- sentinels (post-prompt) ----------------

    def result_for(self, tool_name: str) -> dict[str, Any]:
        return {
            "record_alpha": {"token": self.alpha_token, "status": "ok"},
            "record_beta": {"token": self.beta_token, "status": "ok"},
        }[tool_name]

    def gamma_error_text(self) -> str:
        # ToolError is carried as a string by the ABI; embed the fresh code in it.
        return json.dumps({"code": self.error_code, "message": "gamma rejected"})

    def recovery_results(self) -> dict[str, dict[str, Any]]:
        return {
            "commit": {"status": "committed"},
            "report": {"ack": self.ack_value, "status": "ok"},
        }


def mint_instance(seed: int, salt: str, surface_form: str = "A") -> GauntletInstance:
    """Deterministic (reproducible) but prompt-invisible sentinel material."""
    base = f"xpt|{seed}|{salt}"
    return GauntletInstance(
        salt=salt,
        surface_form=surface_form,
        alpha_code=_tok(base + "|alpha_code", "AC-", 6),
        beta_code=_tok(base + "|beta_code", "BC-", 6),
        size=3 + int(hashlib.sha256((base + "|size").encode()).hexdigest()[:4], 16) % 7,
        gamma_tags=(_tok(base + "|g1", "GT-", 5), _tok(base + "|g2", "GT-", 5)),
        delta_note=_tok(base + "|delta_note", "DN-", 6),
        alpha_token=_tok(base + "|alpha_token", "XPT_A_", 12),
        beta_token=_tok(base + "|beta_token", "XPT_B_", 12),
        error_code=_tok(base + "|err", "E-", 6),
        ack_value=_tok(base + "|ack", "ACK-", 8),
    )


# --------------------------------------------------------------------------
# Prompt rendering
# --------------------------------------------------------------------------

_HEADER_A = (
    "You are exercising a tool-calling interface. The task is deliberately trivial; "
    "only protocol behaviour is measured. Follow every step exactly.\n"
)
_HEADER_B = (
    "Protocol conformance exercise. Nothing here requires reasoning; the only thing "
    "being checked is that each step is carried out literally.\n"
)


def render_user_turn(inst: GauntletInstance) -> str:
    """Render the whole three-step script. Contains NO sentinel values."""
    args = inst.expected_batch_arguments()
    arg_lines = "".join(
        f"  {name} {json.dumps(args[name], sort_keys=True)}\n" for name in BATCH_TOOLS
    )
    if inst.surface_form == "A":
        header = _HEADER_A
        steps = (
            "STEP 1. In one single assistant turn, call record_alpha, record_beta and "
            "record_gamma in parallel, with exactly these arguments:\n"
            + arg_lines
            + "STEP 2. Some of those calls return a result carrying a \"token\" field and "
            "some come back as an error carrying a \"code\" field. In one single "
            "assistant turn, call commit and report in parallel: commit takes alpha and "
            "beta set to the tokens returned by record_alpha and record_beta, and report "
            "takes code set to the code from the error. Never invent a token or a code.\n"
            "STEP 3. When those two results arrive, reply with exactly the value of "
            "report's \"ack\" field as your message, and call no further tools.\n"
        )
    else:
        header = _HEADER_B
        steps = (
            "First turn: emit one assistant turn containing three parallel tool calls — "
            "record_alpha, record_beta and record_gamma — using precisely:\n"
            + arg_lines
            + "Next: those calls come back as a mixture of results and errors. A result "
            "carries \"token\"; an error carries \"code\". Answer with a single turn "
            "holding two parallel calls: commit, whose alpha and beta are the tokens "
            "that record_alpha and record_beta returned, and report, whose code is the "
            "code from the error. Do not make any of those values up.\n"
            "Finally: once both of those answer, your reply is exactly report's \"ack\" "
            "value, with no further tool calls.\n"
        )

    parallel = " | ".join(
        f"{name} {json.dumps(args[name], sort_keys=True)}" for name in BATCH_TOOLS
    )
    directive = (
        "XENOSCRIPT v1\n"
        f"S1 PARALLEL {parallel}\n"
        "S2 PARALLEL commit "
        '{"alpha":"$result.record_alpha.token","beta":"$result.record_beta.token"}'
        ' | report {"code":"$error.code"}\n'
        "S3 ON_RESULTS TEXT $result.report.ack\n"
    )
    return header + steps + directive


# --------------------------------------------------------------------------
# Cheap localisation probes, used only when the gauntlet's first turn fails in
# a schema-shaped way and the endpoint gave no differential signal.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalizationProbe:
    id: str
    feature: str                # ref | title | ap_false | plain
    tools: list[ToolDef] = field(default_factory=list)
    user_content: str = ""
    expected_tool: str = ""
    expected_arguments: dict[str, Any] = field(default_factory=dict)


def _mini_user(tool: str, args: dict[str, Any]) -> str:
    payload = json.dumps(args, sort_keys=True)
    return (
        f"Call {tool} exactly once with exactly these arguments: {payload}\n"
        "XENOSCRIPT v1\n"
        f"S1 CALL {tool} {payload}\n"
    )


def localization_probes(inst: GauntletInstance) -> list[LocalizationProbe]:
    """One minimal single-tool request per schema feature (cheap in tokens)."""
    plain_args = {"entry": {"code": inst.beta_code, "size": inst.size}}
    return [
        LocalizationProbe(
            id="slp_plain",
            feature="plain",
            tools=[ToolDef(name="probe_plain", description="plain", parameters=_beta_schema())],
            user_content=_mini_user("probe_plain", plain_args),
            expected_tool="probe_plain",
            expected_arguments=plain_args,
        ),
        LocalizationProbe(
            id="slp_ref",
            feature="ref",
            tools=[ToolDef(name="probe_ref", description="ref", parameters=_alpha_schema())],
            user_content=_mini_user("probe_ref", plain_args),
            expected_tool="probe_ref",
            expected_arguments=plain_args,
        ),
        LocalizationProbe(
            id="slp_title",
            feature="title",
            tools=[ToolDef(name="probe_title", description="title", parameters=_gamma_schema())],
            user_content=_mini_user("probe_title", {"tags": list(inst.gamma_tags)}),
            expected_tool="probe_title",
            expected_arguments={"tags": list(inst.gamma_tags)},
        ),
        LocalizationProbe(
            id="slp_ap",
            feature="ap_false",
            tools=[ToolDef(name="probe_ap", description="ap", parameters=_delta_schema())],
            user_content=_mini_user("probe_ap", {"note": inst.delta_note}),
            expected_tool="probe_ap",
            expected_arguments={"note": inst.delta_note},
        ),
    ]
