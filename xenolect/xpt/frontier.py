"""Budgeted open-world frontier replanning — soundness-hardened.

Architecture (hard separation)
------------------------------
::

    candidate hypotheses
            |
            v
    SAFE REDUCTION
            |  only mathematically / logically justified removals
            v
    live hypotheses
            |
            v
    EXACT WIRE SHARING
            |  deduplicate expensive black-box work (not hypothesis death)
            v
    wire-distinct actions
            |
            v
    HEURISTIC RANKING
            |  novelty / obligation / cost — ordering only
            v
    next experiment

Critical invariants
-------------------
1. A heuristic (novelty, obligation_gain, estimated cost) may decide **what to
   try first**. It must never decide **what is impossible**.
2. Sharing an expensive experiment is **not** the same as eliminating a
   hypothesis. Permanent elimination requires a SAFE rule with a correctness
   argument.

Representation neutrality: no model / provider / encoding-name preferences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from xenolect.xpt.gauntlet import gauntlet_tools, mint_instance, render_user_turn
from xenolect.xpt.planner import RequestConfig, all_request_configs
from xenolect.xpt.session import XptSession
from xenolect.xpt.syndrome import sha

# ---------------------------------------------------------------------------
# Budget constants tied to the three-turn certification trajectory
# ---------------------------------------------------------------------------

#: Ideal successful diagnosis path for a fresh RequestConfig under the
#: certification trajectory: G1 + one successful tool-result encoding fork (G2) + G3 termination.
#: This is an *admissible lower bound* (true min cost is at least this; G2 may
#: cost 2 if the first encoding fails). See ``minimum_remaining_generation_cost``.
FRESH_DIAGNOSIS_LOWER_BOUND: int = 3

#: Independent certification re-runs the three-turn gauntlet on a fresh instance.
#: See ``certification_generation_upper_bound`` / session.Budget.certification_reserve.
CERTIFICATION_GENERATION_UPPER_BOUND: int = 3

DEFAULT_MAX_GENERATIONS: int = 12


# ---------------------------------------------------------------------------
# Pure frontier model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WireFingerprint:
    """Structural identity of a model-visible request.

    ``full_hash`` is exact wire identity (sharing key).
    ``features`` are coarse content-derived tokens used **only** for heuristic
    ordering distance — never for SAFE deletion, never as dialect preferences.
    """

    full_hash: str
    features: tuple[str, ...]


@dataclass(frozen=True)
class FrontierAction:
    """One candidate hypothesis (or a wire-shared experiment package)."""

    action_id: str
    wire: WireFingerprint
    member_ids: tuple[str, ...]
    #: Admissible lower bound on remaining *diagnosis* generations if this
    #: hypothesis is started next under ideal future observations.
    optimistic_diag_cost: int = FRESH_DIAGNOSIS_LOWER_BOUND
    progresses_trajectory: bool = True
    #: Heuristic only — never used for SAFE elimination.
    obligation_gain: int = 1


@dataclass
class FrontierEvidence:
    """Black-box evidence retained across online replans.

    SAFE elimination lives only in ``eliminated_hypothesis_ids``.
    Wire observations feed heuristic novelty and paid-wire accounting; they do
    not by themselves delete untried members of a wire class.
    """

    used_generations: int = 0
    max_generations: int = DEFAULT_MAX_GENERATIONS
    certification_reserve: int = CERTIFICATION_GENERATION_UPPER_BOUND
    #: Hypotheses that were fully evaluated and rejected (SAFE).
    eliminated_hypothesis_ids: set[str] = field(default_factory=set)
    #: Alias kept for API compatibility with older tests / runtime.
    tried_action_ids: set[str] = field(default_factory=set)
    #: Wires for which an expensive generation was paid (heuristic + share cache).
    paid_wire_hashes: set[str] = field(default_factory=set)
    tried_wire_hashes: set[str] = field(default_factory=set)  # alias of paid
    trajectory_failed_wires: set[str] = field(default_factory=set)
    g1_failed_wires: set[str] = field(default_factory=set)
    observed_fingerprints: dict[str, WireFingerprint] = field(default_factory=dict)

    def remaining_capacity(self) -> int:
        return self.max_generations - self.used_generations

    def explore_room(self) -> int:
        return max(
            0,
            self.max_generations - self.certification_reserve - self.used_generations,
        )

    def lower_bound(self, action: FrontierAction) -> int:
        return (
            self.used_generations
            + minimum_remaining_generation_cost(action)
            + self.certification_reserve
        )

    def budget_feasible(self, action: FrontierAction) -> bool:
        """SAFE: action is impossible if even the ideal path exceeds the hard wall."""
        return self.lower_bound(action) <= self.max_generations

    def reference_wires(self) -> set[str]:
        return set(self.paid_wire_hashes) | set(self.tried_wire_hashes) | set(
            self.trajectory_failed_wires
        )

    def eliminate(self, *hypothesis_ids: str) -> None:
        """SAFE: mark hypotheses as fully evaluated and rejected."""
        for hid in hypothesis_ids:
            self.eliminated_hypothesis_ids.add(hid)
            self.tried_action_ids.add(hid)

    def record_observation(
        self, wire: WireFingerprint, *, trajectory_failed: bool = False
    ) -> None:
        """Record that an expensive generation was paid for this wire (not elimination)."""
        self.paid_wire_hashes.add(wire.full_hash)
        self.tried_wire_hashes.add(wire.full_hash)
        self.observed_fingerprints[wire.full_hash] = wire
        if trajectory_failed:
            self.trajectory_failed_wires.add(wire.full_hash)


# ---------------------------------------------------------------------------
# SAFE layer — admissible costs
# ---------------------------------------------------------------------------


def minimum_remaining_generation_cost(
    action: FrontierAction,
    *,
    current_protocol_state: str = "fresh_g1",
    obligations_already_proven: frozenset[str] | None = None,
    certification_policy: str = "full_gauntlet",
) -> int:
    """Admissible lower bound on remaining *diagnosis* expensive generations.

    Lower-bound argument for the three-turn trajectory:
    - A successful fresh diagnosis must obtain G1, a successful G2 recovery
      turn, and a G3 no-call termination → at least 3 generations.
    - If the first tool-result encoding fails, G2 may cost 2; the *minimum*
      over ideal futures is still 3 (one encoding succeeds on first try).
    - This function must never return more than the true minimum remaining
      cost of any successful path (otherwise SAFE budget pruning is unsound).

    ``action.optimistic_diag_cost`` may override for abstract tests; production
    adapters set it to ``FRESH_DIAGNOSIS_LOWER_BOUND``.
    """
    del obligations_already_proven, certification_policy  # reserved for later states
    if current_protocol_state == "fresh_g1":
        return max(0, action.optimistic_diag_cost)
    if current_protocol_state == "post_g1_success":
        # Still need G2 + G3 under ideal path.
        return 2
    if current_protocol_state == "post_g2_success":
        return 1
    if current_protocol_state == "diagnosis_complete":
        return 0
    return max(0, action.optimistic_diag_cost)


def safe_reduce(
    candidates: Sequence[FrontierAction],
    evidence: FrontierEvidence,
) -> list[FrontierAction]:
    """Layer 1 — permanent eliminations with correctness arguments only.

    SAFE removals:
      (S1) hypothesis id already eliminated / fully evaluated
      (S2) budget-infeasible under admissible lower bound + cert reserve

    NOT safe (must not appear here):
      - novelty / obligation / progress dominance
      - "wire already paid" without evaluating the hypothesis
      - any encoding-name preference
    """
    live: list[FrontierAction] = []
    eliminated = evidence.eliminated_hypothesis_ids | evidence.tried_action_ids
    for a in candidates:
        if a.action_id in eliminated:
            continue
        # Members may still be live even if a collapsed package listed them.
        members_live = tuple(
            m for m in a.member_ids if m not in eliminated
        ) or ((a.action_id,) if a.action_id not in eliminated else ())
        if not members_live and a.action_id in eliminated:
            continue
        if a.action_id in eliminated:
            continue
        if not evidence.budget_feasible(a):
            continue
        live.append(a)
    return live



# ---------------------------------------------------------------------------
# Exact wire sharing (generation dedup, not hypothesis death)
# ---------------------------------------------------------------------------


def group_by_exact_wire(
    actions: Iterable[FrontierAction],
) -> list[FrontierAction]:
    """Package hypotheses that emit the same next request into one experiment.

    Semantics:
      - One expensive black-box call can be shared among members.
      - Members remain distinct hypotheses until SAFE-eliminated after evaluation.
      - ``member_ids`` lists every live config that would send this request.
    """
    buckets: dict[str, list[FrontierAction]] = {}
    for a in actions:
        buckets.setdefault(a.wire.full_hash, []).append(a)
    packages: list[FrontierAction] = []
    for _whash, group in sorted(buckets.items(), key=lambda kv: kv[0]):
        members: list[str] = []
        for g in group:
            members.extend(g.member_ids)
        rep = min(group, key=lambda g: (g.action_id, g.member_ids))
        packages.append(
            FrontierAction(
                action_id=rep.action_id,
                wire=rep.wire,
                member_ids=tuple(sorted(set(members))),
                optimistic_diag_cost=min(g.optimistic_diag_cost for g in group),
                progresses_trajectory=any(g.progresses_trajectory for g in group),
                obligation_gain=max(g.obligation_gain for g in group),
            )
        )
    return packages



# ---------------------------------------------------------------------------
# Heuristic ordering (never SAFE)
# ---------------------------------------------------------------------------


def wire_distance(a: WireFingerprint, b: WireFingerprint) -> int:
    """Higher ⇒ more different. Exact match → 0. Heuristic only."""
    if a.full_hash == b.full_hash:
        return 0
    sa, sb = set(a.features), set(b.features)
    if not sa and not sb:
        return 1
    inter = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return 1
    return max(1, 1000 - (1000 * inter) // union)


def novelty_vs_references(
    action: FrontierAction, references: Sequence[WireFingerprint]
) -> int:
    """Farthest-first structural scheduling score. Heuristic only."""
    if not references:
        return 1000
    return min(wire_distance(action.wire, r) for r in references)


def _reference_fingerprints(
    actions: Sequence[FrontierAction], evidence: FrontierEvidence
) -> list[WireFingerprint]:
    by_hash = {a.wire.full_hash: a.wire for a in actions}
    refs: list[WireFingerprint] = []
    for h in sorted(evidence.reference_wires()):
        if h in evidence.observed_fingerprints:
            refs.append(evidence.observed_fingerprints[h])
        elif h in by_hash:
            refs.append(by_hash[h])
        else:
            refs.append(WireFingerprint(full_hash=h, features=(f"seen:{h[:16]}",)))
    return refs


def rank_heuristic(
    actions: Sequence[FrontierAction],
    evidence: FrontierEvidence,
    *,
    use_novelty: bool = True,
    use_obligation: bool = True,
) -> list[FrontierAction]:
    """Order live wire packages. Does not drop any action."""
    if not actions:
        return []
    refs = _reference_fingerprints(actions, evidence)

    def sort_key(a: FrontierAction) -> tuple:
        nov = novelty_vs_references(a, refs) if use_novelty else 0
        # Prefer unpaid wires for the next expensive call (still ordering only:
        # paid wires with remaining members stay present for cache reuse paths).
        paid = 1 if a.wire.full_hash in evidence.reference_wires() else 0
        return (
            paid,  # unpaid first
            -(nov if use_novelty else 0),
            -(a.obligation_gain if use_obligation else 0),
            -(1 if a.progresses_trajectory else 0),
            a.optimistic_diag_cost,
            a.wire.full_hash,
            a.action_id,
        )

    return sorted(actions, key=sort_key)


@dataclass(frozen=True)
class Selection:
    action: FrontierAction
    reason: str
    frontier_size_before: int
    frontier_size_after_prune: int
    novelty: int
    lower_bound: int


def select_next_action(
    candidates: Sequence[FrontierAction],
    evidence: FrontierEvidence,
    *,
    use_collapse: bool = True,
    use_budget_bound: bool = True,
    use_novelty: bool = True,
    use_obligation: bool = True,
) -> Selection | None:
    """Select next experiment: SAFE reduce → wire share → heuristic rank."""
    n_before = len(candidates)
    # --- Layer 1: SAFE ---
    live = list(candidates)
    if use_budget_bound:
        live = safe_reduce(live, evidence)
    else:
        eliminated = evidence.eliminated_hypothesis_ids | evidence.tried_action_ids
        live = [a for a in live if a.action_id not in eliminated]
    if not live:
        return None

    # --- Layer 2a: exact wire sharing packages ---
    packages = group_by_exact_wire(live) if use_collapse else list(live)

    # Re-apply SAFE budget on packages (same bound).
    if use_budget_bound:
        packages = [a for a in packages if evidence.budget_feasible(a)]
    if not packages:
        return None


    # --- Layer 2b: heuristic rank ---
    ranked = rank_heuristic(
        packages, evidence, use_novelty=use_novelty, use_obligation=use_obligation
    )
    best = ranked[0]
    refs = _reference_fingerprints(packages, evidence)
    nov = novelty_vs_references(best, refs)
    lb = evidence.lower_bound(best)
    reason = (
        f"frontier replan (safe+heuristic): novelty={nov} "
        f"obligation={best.obligation_gain} diag_lb={minimum_remaining_generation_cost(best)} "
        f"wire={best.wire.full_hash[:12]} members={len(best.member_ids)} lower_bound={lb}"
    )
    return Selection(
        action=best,
        reason=reason,
        frontier_size_before=n_before,
        frontier_size_after_prune=len(packages),
        novelty=nov,
        lower_bound=lb,
    )



# ---------------------------------------------------------------------------
# RequestConfig adapter
# ---------------------------------------------------------------------------


class _NullClient:
    def chat_completions(self, messages, tools=None, **kwargs):  # pragma: no cover
        raise AssertionError("frontier fingerprinting must not hit an endpoint")


def fingerprint_request(request: dict[str, Any]) -> WireFingerprint:
    """Coarse structural fingerprint for *heuristic* novelty only."""
    full = sha(request)
    features: list[str] = []
    tools = request.get("tools")
    features.append("tools:yes" if tools else "tools:no")
    if tools:
        features.append(f"ntools:{len(tools)}")
        # Bucketed schema fingerprint — not used for SAFE deletion.
        features.append(f"th:{sha(tools)[:16]}")
    messages = request.get("messages") or []
    systems = [m for m in messages if m.get("role") == "system"]
    features.append("sys:yes" if systems else "sys:no")
    if systems:
        content = systems[0].get("content") or ""
        features.append(f"slen:{len(content) // 32}")
        features.append(f"sh:{sha(content)[:16]}")
        features.append(f"angle:{int('<' in content)}")
        features.append(
            f"brace_tag:{int('{' in content and 'tool' in content.lower())}"
        )
    users = [m for m in messages if m.get("role") == "user"]
    if users:
        features.append(f"uh:{sha(users[0].get('content') or '')[:12]}")
    return WireFingerprint(full_hash=full, features=tuple(features))


def g1_request_for_config(cfg: RequestConfig, *, seed: int = 1) -> dict[str, Any]:
    inst = mint_instance(seed=seed, salt="diagnose", surface_form="A")
    tools = gauntlet_tools()
    content = render_user_turn(inst)
    session = XptSession(_NullClient())
    branch = session.new_branch(cfg.driver())
    branch.add_user(content, tools)
    return branch.build_request()


def g1_fingerprint(cfg: RequestConfig, *, seed: int = 1) -> WireFingerprint:
    return fingerprint_request(g1_request_for_config(cfg, seed=seed))


def actions_from_request_configs(
    configs: Iterable[RequestConfig] | None = None,
    *,
    seed: int = 1,
    tried_keys: set[str] | None = None,
) -> list[FrontierAction]:
    tried_keys = tried_keys or set()
    actions: list[FrontierAction] = []
    for cfg in configs if configs is not None else all_request_configs():
        if cfg.key in tried_keys:
            continue
        fp = g1_fingerprint(cfg, seed=seed)
        actions.append(
            FrontierAction(
                action_id=cfg.key,
                wire=fp,
                member_ids=(cfg.key,),
                optimistic_diag_cost=FRESH_DIAGNOSIS_LOWER_BOUND,
                progresses_trajectory=True,
                obligation_gain=1,
            )
        )
    return actions


def select_next_config(
    remaining: Sequence[RequestConfig],
    evidence: FrontierEvidence,
    *,
    seed: int = 1,
    **select_kwargs: Any,
) -> tuple[RequestConfig, Selection] | None:
    by_key = {c.key: c for c in remaining}
    actions = actions_from_request_configs(remaining, seed=seed)
    sel = select_next_action(actions, evidence, **select_kwargs)
    if sel is None:
        return None
    # Representative for the shared experiment; other members stay live until SAFE-elim.
    key = sel.action.action_id
    if key not in by_key:
        key = min(sel.action.member_ids)
    cfg = by_key.get(key) or by_key.get(sel.action.action_id)
    if cfg is None:
        return None
    return cfg, sel
