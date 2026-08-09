"""Offline correctness gates for oracle-free diagnostic probe planning."""

from __future__ import annotations

import json
from itertools import combinations
from typing import Any

import pytest

from xenolect.abi.events import ToolCall
from xenolect.xpt.diagnostic_probe import (
    MAX_PROBE_OUTCOMES,
    MAX_REQUEST_PROBES,
    MAX_RESULT_PROBES,
    ProbeOutcomeKind,
    build_request_probe,
    build_result_probe,
    candidate_drivers_for_probe,
    check_identifiability,
    observe_probe_response,
)
from xenolect.xpt.discrimination import (
    RequestVersion,
    request_version_space,
    result_version_space,
)
from xenolect.xpt.gauntlet import gauntlet_tools
from xenolect.xpt.hypothesis import (
    ComponentEvidence,
    EvidenceKind,
    EvidenceStore,
    EvidenceStrength,
    ProtocolComponent,
)
from xenolect.xpt.session import Budget, XptSession


def _response_for_alternative(alternative: Any) -> dict[str, Any]:
    witness = alternative.witness
    version = alternative.version
    if version.mode == "native":
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": witness.call_id,
                                "type": "function",
                                "function": {
                                    "name": witness.tool_name,
                                    "arguments": json.dumps(witness.arguments),
                                },
                            }
                        ],
                    }
                }
            ]
        }
    fields = (
        ("function", "input", "call_ref")
        if version.call_fields == "semantic"
        else ("n", "a", "i")
    )
    payload = json.dumps(
        {
            fields[0]: witness.tool_name,
            fields[1]: witness.arguments,
            fields[2]: witness.call_id,
        },
        separators=(",", ":"),
    )
    if version.call_frame == "framed":
        payload = f"<invoke>{payload}</invoke>"
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "ordinary prefix\n" + payload + "\nordinary suffix",
                }
            }
        ]
    }


def test_complete_version_spaces_are_fixed_and_pairwise_separable() -> None:
    requests = request_version_space()
    results = result_version_space()
    assert len(requests) == 33
    assert len(results) == 3

    request_report = check_identifiability(ProtocolComponent.REQUEST, requests)
    result_report = check_identifiability(ProtocolComponent.TOOL_RESULT, results)
    assert request_report.identifiable
    assert request_report.separable_pairs == len(tuple(combinations(requests, 2))) == 528
    assert result_report.identifiable
    assert result_report.separable_pairs == len(tuple(combinations(results, 2))) == 3


def test_request_probe_plan_is_bounded_deterministic_and_auditable() -> None:
    versions = request_version_space()
    first_probe, first_plan = build_request_probe(versions, seed=71, sequence=1)
    second_probe, second_plan = build_request_probe(versions, seed=71, sequence=1)

    assert first_probe == second_probe
    assert first_plan.as_dict() == second_plan.as_dict()
    assert len(first_probe.partitions) <= MAX_PROBE_OUTCOMES
    assert first_plan.worst_case_survivors <= 5
    assert first_plan.estimated_generations == 1
    assert first_plan.information_score > 0
    assert first_probe.as_dict()["production_driver"] is False
    assert first_probe.as_dict()["registry_eligible"] is False
    assert first_probe.as_dict()["abi_witness"] is False

    members = [
        member
        for partition in first_probe.partitions
        for member in partition.member_fingerprints
    ]
    assert sorted(members) == sorted(version.fingerprint for version in versions)
    assert len(members) == len(set(members))
    wire = json.dumps(first_probe.wire(), sort_keys=True)
    assert all(
        marker not in wire.lower()
        for marker in (
            "xpt_probe",
            "diagnostic",
            "diag_req_",
            "probe_id",
            "partition_id",
            "outcome-",
            "probe_value",
            "probe_result",
            "reply_call_id",
            "pw_",
        )
    )
    assert all(version.fingerprint not in wire for version in versions)
    assert all(version.fingerprint[:8] not in wire for version in versions)
    assert all(
        internal_name not in wire
        for internal_name in (
            "messages.role",
            "catalog.container_depth",
            "tools.schema_projection",
            "assistant.call_frame",
            "assistant.call_fields",
            "messages.result_role",
            "messages.tool_call_id",
        )
    )


def test_every_request_version_is_identified_by_the_two_probe_strategy() -> None:
    universe = request_version_space()
    for target in universe:
        survivors = list(universe)
        for sequence in range(1, MAX_REQUEST_PROBES + 1):
            if len(survivors) == 1:
                break
            probe, plan = build_request_probe(
                survivors, seed=89, sequence=sequence
            )
            alternative = next(item for item in probe.alternatives if item.version == target)
            outcome = observe_probe_response(
                probe,
                _response_for_alternative(alternative),
                candidate_drivers_for_probe(survivors),
            )
            assert outcome.kind == ProbeOutcomeKind.EXCLUSIVE_WITNESS
            plan.record_outcome(outcome, evidence_id=f"full-{sequence}")
            remaining = set(plan.hypotheses_remaining)
            survivors = [
                version for version in survivors if version.fingerprint in remaining
            ]
        assert survivors == [target]


def test_result_probe_exhaustively_selects_all_three_versions() -> None:
    request_version = RequestVersion("native")
    results = result_version_space()
    call = ToolCall(
        name="record_alpha", arguments={"entry": {"code": "x", "size": 3}}, id="c1"
    )
    probe, plan = build_result_probe(
        results,
        request_version=request_version,
        prefix_messages=[
            {"role": "user", "content": "ordinary"},
            {"role": "assistant", "content": None, "tool_calls": []},
        ],
        tools=gauntlet_tools(),
        call=call,
        seed=97,
        sequence=1,
    )
    assert plan.worst_case_survivors == 1
    assert len(probe.partitions) == 3
    tokens = [token for partition in probe.partitions for token in partition.witness.tokens()]
    assert len(tokens) == len(set(tokens))
    wire = json.dumps(probe.wire(), sort_keys=True)
    assert all(version.fingerprint not in wire for version in results)
    assert all(
        marker not in wire.lower()
        for marker in (
            "xpt_probe",
            "diagnostic",
            "diag_req_",
            "probe_id",
            "partition_id",
            "outcome-",
            "probe_value",
            "probe_result",
            "reply_call_id",
            "pw_",
        )
    )
    drivers = candidate_drivers_for_probe((request_version,))
    for alternative in probe.alternatives:
        witness = alternative.witness
        raw = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": witness.call_id,
                                "type": "function",
                                "function": {
                                    "name": witness.tool_name,
                                    "arguments": json.dumps(witness.arguments),
                                },
                            }
                        ],
                    }
                }
            ]
        }
        outcome = observe_probe_response(probe, raw, drivers)
        assert outcome.kind == ProbeOutcomeKind.EXCLUSIVE_WITNESS
        assert outcome.outcome_key == alternative.outcome_key

    with pytest.raises(ValueError, match="canary collision"):
        build_result_probe(
            results,
            request_version=request_version,
            prefix_messages=[
                {"role": "user", "content": probe.partitions[0].witness.result_sentinel}
            ],
            tools=gauntlet_tools(),
            call=call,
            seed=97,
            sequence=1,
        )


def test_default_budget_reserves_clean_trace_and_certification() -> None:
    initial_generic_candidate = 1
    diagnostic_turns = MAX_REQUEST_PROBES + MAX_RESULT_PROBES
    clean_diagnosis = 3
    diagnosis_upper_bound = initial_generic_candidate + diagnostic_turns + clean_diagnosis
    assert diagnosis_upper_bound == 7
    assert diagnosis_upper_bound <= 12 - 3


def test_diagnostic_evidence_refines_components_but_never_proves_obligations() -> None:
    store = EvidenceStore()
    evidence = ComponentEvidence(
        evidence_id="diag-exclusive",
        component=ProtocolComponent.REQUEST,
        kind=EvidenceKind.DIAGNOSTIC_WITNESS,
        strength=EvidenceStrength.LOGICAL,
        generation_id=2,
        request_hash="request",
        response_hash="response",
        observation="exclusive structured canary",
    )
    store.eliminate_by_diagnostic_partition(
        ProtocolComponent.REQUEST,
        {"candidate-a", "candidate-b"},
        evidence=evidence,
    )
    assert store.proven_obligation_ids == set()
    assert store.eliminated[ProtocolComponent.REQUEST] == {
        "candidate-a",
        "candidate-b",
    }
    with pytest.raises(ValueError, match="heuristic diagnostic evidence"):
        store.eliminate_by_diagnostic_partition(
            ProtocolComponent.REQUEST,
            {"candidate-c"},
            evidence=ComponentEvidence(
                evidence_id="ordinary-negative",
                component=ProtocolComponent.REQUEST,
                kind=EvidenceKind.NEGATIVE_BEHAVIOR,
                strength=EvidenceStrength.HEURISTIC,
                generation_id=3,
                request_hash="request-2",
                response_hash="response-2",
                observation="ordinary silence",
            ),
        )


@pytest.mark.parametrize("target_index", [0, 1, 7, 19, 32])
def test_positive_structured_witness_selects_exact_predicted_partition(
    target_index: int,
) -> None:
    versions = request_version_space()
    probe, plan = build_request_probe(versions, seed=73, sequence=1)
    alternative = next(
        item for item in probe.alternatives if item.version == versions[target_index]
    )
    outcome = observe_probe_response(
        probe,
        _response_for_alternative(alternative),
        candidate_drivers_for_probe(versions),
    )
    assert outcome.kind == ProbeOutcomeKind.EXCLUSIVE_WITNESS
    assert outcome.outcome_key == alternative.outcome_key
    plan.record_outcome(outcome, evidence_id="diag-test")
    assert versions[target_index].fingerprint in plan.hypotheses_remaining
    assert set(plan.hypotheses_remaining) == set(
        plan.partition(alternative.outcome_key).member_fingerprints  # type: ignore[union-attr]
    )
    assert plan.elimination_reasons
    assert all(
        row["basis"] == "exclusive_nonce_bound_structured_witness"
        for row in plan.elimination_reasons
    )


def test_plain_text_canary_and_generic_rejection_do_not_eliminate() -> None:
    versions = request_version_space()
    probe, plan = build_request_probe(versions, seed=79, sequence=1)
    witness = probe.partitions[0].witness
    prose = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": " ".join(witness.tokens()),
                }
            }
        ]
    }
    prose_outcome = observe_probe_response(
        probe, prose, candidate_drivers_for_probe(versions)
    )
    assert prose_outcome.kind != ProbeOutcomeKind.EXCLUSIVE_WITNESS
    plan.record_outcome(prose_outcome)
    assert plan.hypotheses_removed == []
    assert len(plan.hypotheses_remaining) == 33

    generic = {
        "error": {"type": "invalid_request_error", "message": "invalid request"}
    }
    rejection = observe_probe_response(
        probe, generic, candidate_drivers_for_probe(versions)
    )
    assert rejection.kind == ProbeOutcomeKind.GENERIC_REJECTION
    assert rejection.outcome_key is None


def test_multiple_positive_witnesses_are_ambiguous_and_remove_nothing() -> None:
    versions = request_version_space()
    probe, plan = build_request_probe(versions, seed=101, sequence=1)
    native = next(item for item in probe.alternatives if item.version.mode == "native")
    other_partition = next(
        item for item in probe.alternatives if item.outcome_key != native.outcome_key
    )
    raw = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item.witness.call_id,
                            "type": "function",
                            "function": {
                                "name": item.witness.tool_name,
                                "arguments": json.dumps(item.witness.arguments),
                            },
                        }
                        for item in (native, other_partition)
                    ],
                }
            }
        ]
    }
    outcome = observe_probe_response(
        probe, raw, candidate_drivers_for_probe(versions)
    )
    assert outcome.kind == ProbeOutcomeKind.AMBIGUOUS
    plan.record_outcome(outcome)
    assert plan.hypotheses_removed == []
    assert len(plan.hypotheses_remaining) == 33


class _ProbeClient:
    def chat_completions(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {"choices": [{"message": {"role": "assistant", "content": "silence"}}]}


def test_diagnostic_branch_never_has_or_becomes_a_driver() -> None:
    probe, _ = build_request_probe(request_version_space(), seed=83, sequence=1)
    session = XptSession(_ProbeClient(), budget=Budget())
    branch = session.new_diagnostic_branch(probe)
    _, generation = branch.generate(
        purpose="explore",
        label="offline-diagnostic",
        offered_tool_names={partition.witness.tool_name for partition in probe.partitions},
    )
    assert generation.driver is None
    assert generation.diagnostic_probe is not None
    assert generation.diagnostic_probe["production_driver"] is False
    with pytest.raises(ValueError, match="cannot enter production history"):
        branch.fork()
