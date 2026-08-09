"""Fresh-sentinel G3 termination semantics and adversarial regressions."""

from __future__ import annotations

import json
import re
from typing import Any

from xenolect.driver.ir import identity_driver
from xenolect.eval.termination import assess_g3_termination
from xenolect.xpt.certify import certify
from xenolect.xpt.gauntlet import mint_instance
from xenolect.xpt.obligations import ObligationStatus
from xenolect.xpt.planner import load_compiled_program
from xenolect.xpt.runtime import CERTIFIED, xpt_compile

ACK = mint_instance(seed=401, salt="g3-unit").ack_value
STALE_ACK = mint_instance(seed=402, salt="g3-unit").ack_value
SOURCE = {"ack": ACK, "status": "ok"}


def _assess(
    text: str,
    *,
    source: Any = SOURCE,
    unavailable: tuple[Any, ...] = ("initial prompt",),
    has_tool_calls: bool = False,
    parser_ambiguous: bool = False,
    parse_errors: tuple[str, ...] = (),
    normal_termination: bool = True,
):
    return assess_g3_termination(
        final_text=text,
        expected_sentinel=ACK,
        source_payload=source,
        unavailable_payloads=unavailable,
        has_tool_calls=has_tool_calls,
        parser_ambiguous=parser_ambiguous,
        parse_errors=parse_errors,
        normal_termination=normal_termination,
    )


def test_g3_accepts_exact_or_explanatory_one_time_fresh_sentinel() -> None:
    exact = _assess(ACK)
    prose = _assess(f'The value of the report\'s "ack" field is {ACK}.')
    punctuation = _assess(f"({ACK})")

    assert exact.protocol_termination_verified
    assert exact.exact_response_format_followed
    assert prose.protocol_termination_verified
    assert not prose.exact_response_format_followed
    assert punctuation.protocol_termination_verified


def test_g3_rejects_stale_wrong_missing_and_substring_collisions() -> None:
    for text in (
        STALE_ACK,
        "No acknowledgement was returned.",
        f"prefix{ACK}",
        f"{ACK}suffix",
        f"é{ACK}",
        f"{ACK}界",
    ):
        witness = _assess(text)
        assert not witness.protocol_termination_verified


def test_g3_rejects_premature_or_unbound_sentinel() -> None:
    premature = _assess(ACK, unavailable=(f"The prompt already contained {ACK}.",))
    absent_from_source = _assess(ACK, source={"ack": STALE_ACK})
    duplicate_source = _assess(ACK, source={"ack": ACK, "echo": ACK})
    embedded_duplicate = _assess(
        ACK,
        source={"ack": ACK, "note": f"the source also mentions {ACK}"},
    )

    assert not premature.protocol_termination_verified
    assert "sentinel_available_before_source" in premature.failure_codes
    assert not absent_from_source.protocol_termination_verified
    assert not duplicate_source.protocol_termination_verified
    assert not embedded_duplicate.protocol_termination_verified


def test_g3_rejects_duplicate_or_conflicting_nonce_family_members() -> None:
    duplicate = _assess(f"{ACK}; repeated: {ACK}")
    conflict = _assess(f"{ACK}; stale: {STALE_ACK}")

    assert not duplicate.protocol_termination_verified
    assert not conflict.protocol_termination_verified
    assert "nonexclusive_sentinel_family" in duplicate.failure_codes
    assert "nonexclusive_sentinel_family" in conflict.failure_codes


def test_g3_rejects_spurious_calls_ambiguity_parse_errors_and_noncompletion() -> None:
    assert not _assess(ACK, has_tool_calls=True).protocol_termination_verified
    assert not _assess(ACK, parser_ambiguous=True).protocol_termination_verified
    assert not _assess(ACK, parse_errors=("ambiguous tool frame",)).protocol_termination_verified
    assert not _assess(ACK, normal_termination=False).protocol_termination_verified


def _native(calls: list[tuple[str, dict[str, Any], str]], content: str | None = None):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments, separators=(",", ":")),
                            },
                        }
                        for name, arguments, call_id in calls
                    ],
                }
            }
        ]
    }


class ExplanatoryTerminationEndpoint:
    """Ordinary native endpoint whose final ACK is wrapped in harmless prose."""

    def __init__(self, final_kind: str = "prose") -> None:
        self.final_kind = final_kind

    @staticmethod
    def _text(messages: list[dict[str, Any]]) -> str:
        return "\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("content") is not None
        )

    def chat_completions(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        text = self._text(messages)
        ack_values = re.findall(r"ACK-[0-9A-F]+", text)
        if ack_values:
            ack = ack_values[-1]
            if self.final_kind == "stale":
                content = STALE_ACK
                return {"choices": [{"message": {"role": "assistant", "content": content}}]}
            if self.final_kind == "spurious_call":
                return _native(
                    [("report", {"code": ack}, "spurious-final-call")],
                    content=ack,
                )
            if self.final_kind == "ambiguous_parse":
                xml = json.dumps(
                    {"name": "commit", "arguments": {}, "id": "ambiguous-xml"},
                    separators=(",", ":"),
                )
                tagged = json.dumps(
                    {"name": "report", "arguments": {}, "id": "ambiguous-tagged"},
                    separators=(",", ":"),
                )
                content = f"{ack}\n<tool_call>{xml}</tool_call>\nTOOL_CALL {tagged}"
                return {"choices": [{"message": {"role": "assistant", "content": content}}]}
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                "The value of the report's acknowledgement field is "
                                f"{ack}."
                            ),
                        }
                    }
                ]
            }

        alpha = re.findall(r"XPT_A_[0-9A-F]+", text)
        beta = re.findall(r"XPT_B_[0-9A-F]+", text)
        error = re.findall(r"E-[0-9A-F]+", text)
        if alpha and beta and error:
            return _native(
                [
                    (
                        "commit",
                        {"alpha": alpha[-1], "beta": beta[-1]},
                        "recovery-call-1",
                    ),
                    ("report", {"code": error[-1]}, "recovery-call-2"),
                ]
            )

        matches = re.findall(
            r"^  (record_(?:alpha|beta|gamma)) (\{.*\})$",
            text,
            flags=re.MULTILINE,
        )
        arguments = {name: json.loads(payload) for name, payload in matches}
        return _native(
            [
                (name, arguments[name], f"initial-call-{index}")
                for index, name in enumerate(
                    ("record_alpha", "record_beta", "record_gamma"), start=1
                )
            ]
        )


def test_xpt_certifies_protocol_termination_despite_harmless_prose() -> None:
    result = xpt_compile(
        ExplanatoryTerminationEndpoint(),
        load_compiled_program(),
        seed=307,
    )

    assert result.status == CERTIFIED, result.as_dict()
    assert result.diagnosis_generations == 3
    assert result.certification_generations == 3
    assert result.total_generations == 6
    synthesis = result.as_dict()["protocol_synthesis"]
    assert synthesis["mode"] == "bounded_obligation_directed_cegis"
    assert synthesis["property_local_fault_localization_used"] is False
    assert synthesis["property_local_rejections_observed"] == 0
    assert synthesis["property_local_rejections_used"] == 0
    termination_decisions = [
        decision
        for decision in result.ledger.decisions
        if "termination" in str(decision.get("phase", ""))
    ]
    assert termination_decisions
    assert termination_decisions[0]["protocol_termination_verified"] is True
    assert termination_decisions[0]["exact_response_format_followed"] is False
    assert termination_decisions[0]["evidence_class"] == "model_style_instruction_deviation"


def test_independent_certification_rejects_stale_final_sentinel() -> None:
    instance = mint_instance(seed=501, salt="cert-stale", surface_form="B")
    run = certify(
        identity_driver(),
        ExplanatoryTerminationEndpoint("stale"),
        instance,
        max_cycles=2,
    )

    assert not run.passed
    assert run.certificate.status_of("OB15") == ObligationStatus.VERIFIED
    assert run.certificate.status_of("OB16") == ObligationStatus.FAILED


def test_independent_certification_rejects_tool_call_beside_correct_ack() -> None:
    instance = mint_instance(seed=502, salt="cert-call", surface_form="B")
    run = certify(
        identity_driver(),
        ExplanatoryTerminationEndpoint("spurious_call"),
        instance,
        max_cycles=2,
    )

    assert not run.passed
    assert run.certificate.status_of("OB15") == ObligationStatus.FAILED
    assert run.certificate.status_of("OB16") == ObligationStatus.FAILED


def test_independent_certification_rejects_ambiguous_final_parse() -> None:
    instance = mint_instance(seed=503, salt="cert-ambiguous", surface_form="B")
    run = certify(
        identity_driver(),
        ExplanatoryTerminationEndpoint("ambiguous_parse"),
        instance,
        max_cycles=2,
    )

    assert not run.passed
    assert run.certificate.status_of("OB16") == ObligationStatus.FAILED
    assert run.certificate.status_of("OB17") == ObligationStatus.FAILED
