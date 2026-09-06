"""Task pacing SSOT (v6.54.4): ONE urgency system for a task's time budget.

Absorbs the milestone CONTENT logic that lived inline in ``loop.py`` (deadline
50/25/10% TIME BUDGET notes and the v6.53.0 intrinsic no-deadline pacing) and
adds the acceptance-review budget layer: the finalization reserve, a budget
snapshot, and the improvement-pass gates driven by ``task_contract.budget_profile``
(``improvement_policy`` fixed | adaptive; the legacy ``until_deadline`` /
``stall_rounds_threshold`` aliases were removed in the 7.0 ABI window, Q10=A).

Design contract (owner-decided, sprint v6.55):
- Pacing notes fire only on milestone triggers, never per round (prompt-cache
  friendly), their wording is TASK-NEUTRAL, and note identification is by the
  checkpoint metadata — never a regex strip of transcript text.
- The gates are ADVISORY inputs to the host's review machinery; the model's own
  finalization stays P5 judgment. Forced-finalization escape hatches bypass the
  obligation gate unconditionally — a deadline never hangs on review passes.
- ``loop.py`` keeps only transport (message append + checkpoint emit); every
  threshold, text, and time computation lives here.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ouroboros.config import (
    get_acceptance_reserve_pct,
    get_acceptance_review_est_sec,
    get_finalization_grace_sec,
    get_pacing_interval_sec,
)
from ouroboros.contracts.task_contract import answer_protocol_active, normalize_budget_profile
from ouroboros.deadline_utils import parse_deadline_ts, utc_now
from ouroboros.review_cycles import (
    REASON_REVIEW_CYCLES_EXHAUSTED,
    emit_review_cycles_exhausted,
    get_acceptance_max_improvement_passes,
    review_max_cycles,
)


# The host never predicts how long a review takes (owner R52, 2026-09-03). A
# task has three host-owned rails: a deadline, a paid-cycle cap and a wallet —
# and the time rail is this ONE number, the minimum spendable window (remaining
# time above the finalization reserve) an acceptance panel needs in order to
# START. `OUROBOROS_ACCEPTANCE_REVIEW_EST_SEC` configures it and never lowers it
# below this floor; an improvement pass needs the same floor scaled by
# `_window_scale` (×2 under the adaptive policy, ×1 otherwise). Once launched a
# review is an ordinary operation, clamped to the owner deadline and the task
# ceiling with the per-send money fence, and a deadline-cut review is a typed
# degraded outcome. Panel durations are recorded as telemetry
# (`task_acceptance_review_timing`) and nothing decides on them.
_ACCEPTANCE_REVIEW_RESERVE_FLOOR_SEC = 200.0

# Proportional nanny-economics reminder thresholds (poltergeist phase B; owner
# decision 2=B: NO absolute round cap — reminders only, sized to the measured
# burn). A harness-dispatched child whose OWN metered rounds (or dollars) since
# its last delegated-run activity cross either threshold hears the reminder,
# re-armed per further threshold-width of rounds; a well-behaved custodian never
# does. Here and not in prompt text, so the prompt states measurements and the
# policy has one home (this module is the pacing-threshold SSOT).
NANNY_REMINDER_ROUNDS = 8
NANNY_REMINDER_USD = 2.0
# Early FIRST reminder for a harness-dispatched task that has made NO delegate
# verb call at all (owner-approved 2026-08-15): cheap children finish in 4-8
# rounds under the dollar threshold and never heard the reminder — live E2E
# showed 4 children with 0 delegate_start calls doing all research natively.
# The first firing comes at this round count regardless of dollars; after any
# delegate activity, and for every re-arm after the first firing, the ordinary
# dual-axis thresholds above apply unchanged. Same SSOT home as its siblings.
NANNY_FIRST_REMINDER_ROUNDS = 3
log = logging.getLogger(__name__)


def _protocol_marker_phrases(ctx: Any) -> bool:
    """v6.60.0: the `FINAL ANSWER:` marker PHRASES in pacing notes are protocol-gated
    (contract ``answer_protocol="final_answer_line"``); the milestones themselves
    fire regardless. One SSOT gate shared with the loop nudges/context instruction."""
    try:
        return answer_protocol_active(ctx)
    except Exception:
        return False


@dataclass(frozen=True)
class PacingNote:
    """One milestone note: the user-turn text + its checkpoint event payload."""

    text: str
    checkpoint: Dict[str, Any]


@dataclass(frozen=True)
class BudgetSnapshot:
    """The task's time-budget facts at one moment (all seconds).

    ``has_deadline=False`` disables the time axis entirely: gates then bound
    improvement passes by COUNT only (default 1 = the historical behavior)."""

    has_deadline: bool
    total_sec: float = 0.0
    elapsed_sec: float = 0.0
    remaining_sec: float = 0.0
    reserve_sec: float = 0.0

    @property
    def inside_reserve(self) -> bool:
        return self.has_deadline and self.remaining_sec <= self.reserve_sec

    @property
    def spendable_sec(self) -> float:
        """Time available ABOVE the finalization reserve."""
        if not self.has_deadline:
            return float("inf")
        return self.remaining_sec - self.reserve_sec


def _supplied_budget_profile(ctx: Any) -> Any:
    """The task's budget_profile exactly as SUPPLIED (task_contract, else the
    metadata copy); ``None`` when the task carries none."""
    contract = getattr(ctx, "task_contract", None)
    if not isinstance(contract, dict):
        meta = getattr(ctx, "task_metadata", {})
        contract = meta.get("task_contract") if isinstance(meta, dict) else None
    return contract.get("budget_profile") if isinstance(contract, dict) else None


def observe_budget_profile(ctx: Any) -> Dict[str, Any]:
    """The task's normalized budget_profile resolved SIDE-EFFECT FREE (R49): the
    reader the coordination poll (``delegate_supervision._time_fact``) uses."""
    return normalize_budget_profile(_supplied_budget_profile(ctx))


def resolve_budget_profile(ctx: Any) -> Dict[str, Any]:
    """The task's normalized budget_profile (from task_contract; absent ->
    defaults). The deprecated ``until_deadline`` / ``stall_rounds_threshold``
    aliases and their deprecation row are gone (7.0 ABI window), so this is
    the same side-effect-free read as ``observe_budget_profile``; both names
    stay so the observer contract remains explicit at its call sites."""
    return normalize_budget_profile(_supplied_budget_profile(ctx))


def _acceptance_floor_sec() -> float:
    """The time rail: the configured floor, never below 200 s."""
    return max(_ACCEPTANCE_REVIEW_RESERVE_FLOOR_SEC, float(get_acceptance_review_est_sec()))


def _window_scale(profile: Any) -> float:
    """The improvement window is 2× the floor under the adaptive policy."""
    return 2.0 if isinstance(profile, dict) and profile.get("improvement_policy") == "adaptive" else 1.0


def acceptance_timing_events_path(ctx: Any) -> pathlib.Path:
    """Return the canonical event stream shared across split task drives."""
    metadata = getattr(ctx, "task_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    root = pathlib.Path(str(
        metadata.get("budget_drive_root")
        or getattr(ctx, "budget_drive_root", "")
        or getattr(ctx, "drive_root")
    ))
    return root / "logs" / "events.jsonl"


def _reserve_sec(total_sec: float, profile: Dict[str, Any]) -> float:
    """Finalization reserve = max(grace window, reserve_pct × total)."""
    grace = float(get_finalization_grace_sec())
    pct = profile.get("reserve_finalization_pct")
    if pct is None:
        pct = get_acceptance_reserve_pct()
    return max(grace, total_sec * (float(pct) / 100.0))


def effective_finalization_reserve_sec(ctx: Any) -> float:
    """The EMIT window before a real deadline (v6.55.0): the plain finalization
    grace — the time needed to emit one best-effort answer / let a network tool
    return cleanly. Consumed by the local deadline-finalize trigger and the
    network-tool deadline clamp.

    This is deliberately the small GRACE window, NOT the percentage reserve
    (``BudgetSnapshot.reserve_sec`` = max(grace, pct×total)). The pct reserve is
    an acceptance-REVIEW gate concept (don't START a review you cannot finish);
    applying it to the finalize trigger amputated the tail of every long task —
    a 6h ProgramBench task force-finalized ~54 min early on the 15% profile
    (adversarial review r1). The finalize path fires just before the kill, so it
    needs only the emit window; the review gates keep the pct reserve via the
    snapshot. Restores the pre-sprint deadline-local behavior (monotonicity)."""
    return float(get_finalization_grace_sec())


def build_budget_snapshot(ctx: Any, *, profile: Optional[Dict[str, Any]] = None) -> BudgetSnapshot:
    """Snapshot the task's time budget from task_metadata deadline facts."""
    return _budget_snapshot(ctx, profile=profile, latch=True)


def observe_budget_snapshot(ctx: Any, *, profile: Dict[str, Any]) -> BudgetSnapshot:
    """The same snapshot WITHOUT the fallback-anchor latch (owner R49).

    A metadata-poor task (no ``created_at``/``started_at`` and no anchor latched
    yet) has no usable window facts a read-only observer could obtain without
    WRITING one, so it is reported as having no deadline axis — which its one
    caller, ``delegate_supervision._time_fact``, reports as ``not_set``.
    ``profile`` is required: an observer never falls back to the emitting resolver."""
    return _budget_snapshot(ctx, profile=profile, latch=False)


def _budget_snapshot(
    ctx: Any, *, profile: Optional[Dict[str, Any]], latch: bool,
) -> BudgetSnapshot:
    meta = getattr(ctx, "task_metadata", {})
    if not isinstance(meta, dict):
        return BudgetSnapshot(has_deadline=False)
    deadline = parse_deadline_ts(meta.get("deadline_at"))
    if deadline is None:
        return BudgetSnapshot(has_deadline=False)
    created = parse_deadline_ts(meta.get("created_at") or meta.get("started_at"))
    if created is None:
        created = getattr(ctx, "_time_budget_started_at", None)
        if created is None:
            if not latch:
                return BudgetSnapshot(has_deadline=False)
            # Latch the fallback anchor exactly like the note path does (fable-5
            # cumulative review F4): without the latch every metadata-poor
            # snapshot re-anchors total to "now" and the pct reserve silently
            # degrades toward the bare grace floor over the task's life.
            created = utc_now()
            try:
                ctx._time_budget_started_at = created
            except Exception:
                pass
    now = utc_now()
    total = max(1.0, (deadline - created).total_seconds())
    elapsed = max(0.0, (now - created).total_seconds())
    remaining = (deadline - now).total_seconds()
    reserve = _reserve_sec(total, profile if profile is not None else resolve_budget_profile(ctx))
    return BudgetSnapshot(
        has_deadline=True,
        total_sec=total,
        elapsed_sec=elapsed,
        remaining_sec=remaining,
        reserve_sec=reserve,
    )


def review_launch_allowed(snapshot: BudgetSnapshot) -> Tuple[bool, str]:
    """Gate 1: run an acceptance review only when it fits ABOVE the reserve.

    Historically a review could start two minutes before the deadline and kill
    the task; skipping inside the reserve is a strict improvement. No deadline →
    always allowed (the pass counter is the only axis). The host does not
    predict the panel's duration (owner R52): the configured floor
    (``_acceptance_floor_sec``) is the whole time rail, and a spendable window
    at or below it is refused ``review_skipped_deadline_reserve``. PURE — a
    read-only observer may ask; nothing is recorded here."""
    if not snapshot.has_deadline:
        return True, ""
    if snapshot.spendable_sec > _acceptance_floor_sec():
        return True, ""
    return False, "review_skipped_deadline_reserve"


def effective_max_improvement_passes(
    profile: Dict[str, Any], *,
    required_blocking: bool = False,
) -> Optional[int]:
    """The COUNT axis for improvement passes.

    An explicit task-local cap always binds (owner "Hurry up" overlays 0).
    Without one, the shared review-cycle cap binds under EVERY policy —
    Required+Blocking included (owner decisions D10/D20): passes = cycles - 1
    from ``OUROBOROS_REVIEW_MAX_CYCLES`` (``review_cycles.py``), ``None`` only
    when that setting is ``unlimited``. Deadline and global lifecycle rails
    apply on top. (The ``until_deadline`` alias that lifted the count axis
    outside Required+Blocking was removed in the 7.0 ABI window, Q10=A.)"""
    cap = profile.get("max_improvement_passes")
    # An explicit task-local cap is authoritative under every policy.
    if cap is not None:
        return max(0, int(cap))
    cap = get_acceptance_max_improvement_passes()
    return None if cap is None else max(0, int(cap))


from ouroboros.task_results import (  # noqa: E402,F401 - compatibility re-export
    effective_task_acceptance_review_cycles,
    project_task_acceptance_review_capacity,
)


def improvement_pass_allowed(
    snapshot: BudgetSnapshot,
    passes_done: int,
    profile: Dict[str, Any],
    *,
    required_blocking: bool = False,
    ctx: Any = None,
) -> Tuple[bool, str]:
    """Gate 2: one more improvement/obligation pass?

    The count cap (task-local or the shared review-cycle cap) and the
    deadline/reserve rail are independent; ``adaptive`` demands a comfortable
    window — twice the floor (``_window_scale``) — before spending another
    pass. The host predicts no duration (owner R52): the floor is the rail.
    Under Required+Blocking the SHARED cap (no task-local cap) exhausting is the
    typed ``review_cycles_exhausted`` reason (owner D10/D27) and — when ``ctx``
    is supplied — the typed escalation event; a task-local cap (owner hurry,
    budget_profile) keeps the generic ``improvement_passes_exhausted``."""
    cap = effective_max_improvement_passes(
        profile,
        required_blocking=required_blocking,
    )
    if cap is not None and passes_done >= cap:
        if required_blocking and profile.get("max_improvement_passes") is None:
            if ctx is not None:
                emit_review_cycles_exhausted(
                    getattr(ctx, "event_queue", None),
                    getattr(ctx, "budget_drive_root", "") or getattr(ctx, "drive_root", ""),
                    surface="task_acceptance", task_id=str(getattr(ctx, "task_id", "") or ""),
                    cycles_paid=passes_done + 1, cap=cap + 1, enforcement="blocking",
                )
            return False, REASON_REVIEW_CYCLES_EXHAUSTED
        return False, "improvement_passes_exhausted"
    if not snapshot.has_deadline:
        return True, ""
    if snapshot.spendable_sec > _acceptance_floor_sec() * _window_scale(profile):
        return True, ""
    return False, "improvement_window_inside_reserve"


# ---------------------------------------------------------------------------
# Milestone note content (moved from loop.py; loop keeps only transport).

_TIME_BUDGET_THRESHOLDS = ((0.50, "50%"), (0.25, "25%"), (0.10, "10%"))

# Cost axis (v6.56.0): thresholds are module constants like the time axis —
# deliberately NOT settings keys (owner decision; the per-task knob is the
# contract's budget_profile.cost_hard_stop_pct, not a global).
_COST_BUDGET_THRESHOLDS = ((0.50, "50%"), (0.25, "25%"), (0.10, "10%"))
_COST_WRAPUP_SPENT_FRACTION = 0.80
# The historical in-task hard stop: half the budget remaining at task start.
_DEFAULT_COST_HARD_STOP_PCT = 50

# Typed cost-ceiling states (v6.91): ``None`` is deliberately NOT overloaded to
# mean both "unlimited" and "exhausted" — a $0.50 bench root cap under a $3
# planning margin must soft-land, never run uncapped.
COST_CEILING_DISABLED = "disabled"
COST_CEILING_ACTIVE = "active"
COST_CEILING_EXHAUSTED_SOFT_LAND = "exhausted_soft_land"
COST_CEILING_UNKNOWN = "unknown"

# Planning margin subtracted from the root cap before the graceful in-task stop:
# an ABSOLUTE emit-window's worth of money (~2 forced-wrap-up call reservation
# bounds), NEVER a percentage of the cap (a pct reserve amputated ~54 min from a
# 6h task — v6.54.4 adversarial r1, see effective_finalization_reserve_sec) and
# NOT ledger-held — the ledger fence still binds at the full cap; the margin
# only pulls the graceful stop earlier so the wrap-up call fits under the fence.
_WRAPUP_CALL_RESERVATION_BOUND_USD = 1.50
COST_PLANNING_MARGIN_USD = max(1.0, 2.0 * _WRAPUP_CALL_RESERVATION_BOUND_USD)


# Deciding-spend basis vocabulary (v6.91). The tree-accounted number is the
# authority for every rooted task (with a root cap the ledger fence counts the
# TREE; without one the in-task ceiling still decides on the subtree); when it
# is momentarily unavailable the own-cost number still decides — but the
# substitution is DISCLOSED as a lower bound, never silent (BIBLE P1). Only a
# task with no root at all has no tree to read, so its own cost is complete.
SPEND_BASIS_TREE = "tree_accounted"
SPEND_BASIS_OWN_TREE_UNKNOWN = "own_fallback_tree_unknown"
SPEND_BASIS_OWN_NO_TREE_CAP = "own_only_no_tree_cap"


def resolve_deciding_spend(
    *,
    tree_cost_usd: Optional[float],
    task_cost_usd: Optional[float],
    root_cap_usd: Optional[float],
) -> Tuple[Optional[float], str]:
    """The spend that decides a cost surface, plus its DISCLOSED basis (SSOT).

    Shared by the loop's ceiling check and the milestone note so the stop and
    the nudge can never disagree about which number they are reading. Unknown
    spend stays None end-to-end — it is never coerced to $0."""
    from ouroboros.usage_accounting import current_usage_scope

    if tree_cost_usd is not None:
        return float(tree_cost_usd), SPEND_BASIS_TREE
    deciding = None if task_cost_usd is None else float(task_cost_usd)
    scope = current_usage_scope()
    if root_cap_usd is not None or (scope is not None and scope.root_task_id):
        return deciding, SPEND_BASIS_OWN_TREE_UNKNOWN
    return deciding, SPEND_BASIS_OWN_NO_TREE_CAP


@dataclass(frozen=True)
class CostCeiling:
    """Typed in-task cost-stop state, resolved ONCE at loop start.

    ``state``:
    - ``disabled``: no in-task stop — explicit ``cost_hard_stop_pct=0`` (bench
      contract, e.g. SWE-Pro) or no finite budget on either axis (e.g. GAIA);
      the whole cost axis stays silent.
    - ``active``: ``ceiling_usd`` is the root's strictly-positive original
      graceful-stop point, or a disclosed legacy local resolution when the
      original carrier is unavailable.
    - ``exhausted_soft_land``: the root cap leaves no room above the planning
      margin — the loop must enter its graceful best-effort wrap-up
      immediately; it must NEVER run uncapped.
    - ``unknown``: resolution inputs errored; the axis stays silent but the
      gap is represented, never filled in (BIBLE P1)."""

    state: str
    ceiling_usd: Optional[float] = None
    root_cap_usd: Optional[float] = None
    planning_margin_usd: Optional[float] = None
    basis: str = ""


def resolve_cost_ceiling(
    budget_remaining_start_usd: Optional[float],
    profile: Dict[str, Any],
    *,
    root_cap_usd: Optional[float] = None,
    non_root_member: bool = False,
    root_ceiling_usd: Optional[float] = None,
) -> CostCeiling:
    """The in-task cost stop, computed ONCE at loop start (typed; v6.91).

    The GLOBAL component keeps the historical semantics: ``cost_hard_stop_pct``
    None -> 50% of the global remaining at task start; 0 -> the whole in-task
    stop is disabled (bench contract). The ROOT component is the per-task tree
    cap (``OUROBOROS_PER_TASK_COST_USD`` -> ``UsageScope.root_limit_usd``, the
    SAME value the ledger fence enforces) minus the ABSOLUTE planning margin —
    deliberately NOT pct-scaled (pct applies to the global axis only; scaling
    the owner's chosen cap would silently halve it). The ceiling is
    min(available components); NEVER a computed $0 — a root cap at or below the
    margin resolves to ``exhausted_soft_land`` instead.

    An enabled non-root member keeps the propagated original root ceiling;
    a later global balance never re-mints that early threshold. Actual global
    and root dispatch fences still bind independently. Legacy missing carriers
    retain a disclosed local resolution, never a guessed original root fact.

    Stated plainly rather than implied: the ``room <= 0`` bail is the owner's
    "$0 ceiling" rule EXACTLY, no wider. A cap just ABOVE the margin therefore
    yields a real but tiny ceiling — ``root_cap_usd = COST_PLANNING_MARGIN_USD
    + 0.01`` gives ``ceiling_usd == 0.01``, which the first round's spend
    crosses — so a positive ``ceiling_usd`` is not by itself a promise of
    working room. Both numbers are disclosed on the carrier (``ceiling_usd``,
    ``root_cap_usd``, ``planning_margin_usd``) and printed in the stop text, so
    a reader sees the tiny ceiling instead of inferring a healthy one. Widening
    the bail into a minimum-room FLOOR would move caps the owner deliberately
    allows into immediate soft-land; that is an owner call, not a code one."""
    try:
        pct = profile.get("cost_hard_stop_pct")
        if pct is None:
            pct = _DEFAULT_COST_HARD_STOP_PCT
        pct = max(0, min(100, int(pct)))
        if pct == 0:
            return CostCeiling(
                state=COST_CEILING_DISABLED,
                root_cap_usd=(
                    float(root_cap_usd)
                    if root_cap_usd is not None and float(root_cap_usd) > 0
                    else None
                ),
                basis="cost_hard_stop_pct_zero",
            )
        components: list[float] = []
        basis_parts: list[str] = []
        inherited = (float(root_ceiling_usd) if non_root_member and root_ceiling_usd is not None
                     and float(root_ceiling_usd) > 0 else None)
        if inherited is None and budget_remaining_start_usd is not None and float(budget_remaining_start_usd) > 0:
            components.append(float(budget_remaining_start_usd) * (pct / 100.0))
            basis_parts.append("global_pct")
        margin: Optional[float] = None
        cap: Optional[float] = None
        if root_cap_usd is not None and float(root_cap_usd) > 0:
            cap = float(root_cap_usd)
            margin = COST_PLANNING_MARGIN_USD
            room = cap - margin
            if room <= 0:
                return CostCeiling(
                    state=COST_CEILING_EXHAUSTED_SOFT_LAND,
                    root_cap_usd=cap,
                    planning_margin_usd=margin,
                    basis="root_cap_at_or_below_planning_margin",
                )
            if inherited is None:
                components.append(room)
                basis_parts.append("root_cap_minus_margin")
            if non_root_member:
                basis_parts.append("non_root_member")
        if inherited is not None:
            components.append(inherited)
            basis_parts.append("root_resolved_ceiling")
        elif non_root_member:
            basis_parts.append("original_root_ceiling_unavailable")
        if not components:
            return CostCeiling(state=COST_CEILING_DISABLED, basis="no_finite_budget")
        return CostCeiling(
            state=COST_CEILING_ACTIVE,
            # Min of strictly-positive components, so never a computed $0 — but a
            # cap just above the margin makes this legitimately TINY (cap $3.01 ->
            # $0.01), which is a stop-after-the-first-spend ceiling, not headroom.
            # Documented in the docstring and pinned in test_budget_limits.
            ceiling_usd=min(components),
            root_cap_usd=cap,
            planning_margin_usd=margin,
            basis="min(" + ", ".join(basis_parts) + ")",
        )
    except Exception:
        log.warning("Cost ceiling resolution failed; axis stays silent", exc_info=True)
        return CostCeiling(state=COST_CEILING_UNKNOWN, basis="resolve_error")


def resolve_task_cost_ceiling(ctx: Any, budget_remaining_usd: Optional[float]) -> CostCeiling:
    """The typed in-task cost stop of ONE task, resolved once per task.

    The root cap comes from the bound usage scope -- the SAME
    ``OUROBOROS_PER_TASK_COST_USD``-derived value the ledger fence enforces
    (``agent.py`` wires it as ``UsageScope.root_limit_usd``), so the graceful
    stop and the fence can never disagree about the cap. The same scope says
    whether this task is the root of its tree or one of its members."""
    root_cap = None
    root_ceiling = None
    non_root_member = False
    try:
        from ouroboros.usage_accounting import current_usage_scope

        scope = current_usage_scope()
        if scope is not None:
            root_cap = getattr(scope, "root_limit_usd", None)
            root_ceiling = getattr(scope, "root_cost_ceiling_usd", None)
            non_root_member = bool(
                scope.root_task_id and scope.task_id and scope.root_task_id != scope.task_id
            )
    except Exception:
        log.debug("Usage scope unavailable for cost ceiling resolution", exc_info=True)
    return resolve_cost_ceiling(
        budget_remaining_usd,
        resolve_budget_profile(ctx),
        root_cap_usd=root_cap,
        non_root_member=non_root_member,
        root_ceiling_usd=root_ceiling,
    )


def cost_ceiling_disclosure(ceiling: CostCeiling) -> Dict[str, Any]:
    """The start-of-task shape of the ceiling the loop will actually decide on."""
    return {
        "state": ceiling.state,
        "ceiling_usd": ceiling.ceiling_usd,
        "root_cap_usd": ceiling.root_cap_usd,
        "planning_margin_usd": ceiling.planning_margin_usd,
        "basis": ceiling.basis,
        "rule": (
            "The graceful in-task cost stop of THIS task's whole tree, resolved once at task "
            "start: the root resolves min(configured share of global remaining, hard tree cap "
            "minus a planning margin); enabled descendants retain that original number. "
            "Legacy members without it disclose their local resolution. Crossing it asks for a "
            "best-effort final answer; the ledger fence at the full cap still binds "
            "independently. Budget checkpoints during the task report the live tree spend."
        ),
    }


def in_task_cost_ceiling_disclosure(ctx: Any, budget_remaining_usd: Optional[float]) -> Dict[str, Any]:
    """Resolve this task's ceiling once, stash it on ctx, and disclose that object.

    The loop reads the same stashed object, so the number the model is shown at
    task start and the number that later stops the task cannot differ."""
    ceiling = resolve_task_cost_ceiling(ctx, budget_remaining_usd)
    try:
        setattr(ctx, "_cost_ceiling", ceiling)
    except Exception:
        log.debug("Cost ceiling could not be stashed on the tool context", exc_info=True)
    return cost_ceiling_disclosure(ceiling)


def tree_spend_line(tree_info: Any, ceiling: Optional[CostCeiling] = None) -> str:
    """The one live tree-spend line the checkpoint and the pacing note share.

    Names the BINDING bound: the in-task ceiling when one is active (that is
    what stops the task first), with the hard tree cap the ledger fence
    enforces beside it. Empty string when tree accounting is unavailable --
    unknown is never rendered as $0."""
    if not isinstance(tree_info, dict) or tree_info.get("accounted_usd") is None:
        return ""
    raw_cap = tree_info.get("root_limit_usd")
    cap = float(raw_cap) if raw_cap is not None else None
    ceiling_usd = (
        ceiling.ceiling_usd
        if ceiling is not None and ceiling.state == COST_CEILING_ACTIVE
        else None
    )
    if ceiling_usd is not None:
        bound = f" of ${ceiling_usd:.2f} in-task cost ceiling"
        if cap is not None:
            bound += f" (${cap:.2f} hard tree cap)"
    else:
        bound = f" of ${cap:.2f} hard tree cap" if cap is not None else ""
    return (
        f"Task tree spend: ~${float(tree_info['accounted_usd']):.2f}{bound} "
        "(ledger-accounted incl. in-flight holds, subagents included)"
    )


def wrapup_unaffordable_text(deciding_usd: Optional[float], ceiling: CostCeiling, global_remaining_usd: Optional[float] = None) -> str:
    """The owner-facing reason a task ends without even one affordable wrap-up send."""
    cap = ceiling.root_cap_usd
    cap_text = f" of the ${cap:.2f} hard tree cap" if cap is not None else ""
    spent = f"Task tree spent ${deciding_usd:.3f}{cap_text}" if deciding_usd is not None else "Task-tree spend is unavailable"
    wallet = f"; global model budget remaining is ${global_remaining_usd:.3f}" if global_remaining_usd is not None else ""
    return (
        f"{spent}{wallet}; not even one wrap-up call can "
        "be reserved, so the host delivers the retained evidence without a model synthesis."
    )


def wrapup_last_fit_text(deciding_usd: Optional[float], ceiling: CostCeiling, global_remaining_usd: Optional[float] = None) -> str:
    """The owner-facing reason a task claims the last affordable wrap-up send."""
    cap = ceiling.root_cap_usd
    cap_text = f" of the ${cap:.2f} hard tree cap" if cap is not None else ""
    spent = f"Task tree spent ${deciding_usd:.3f}{cap_text}" if deciding_usd is not None else "Task-tree spend is unavailable"
    wallet = f"; global model budget remaining is ${global_remaining_usd:.3f}" if global_remaining_usd is not None else ""
    return (
        f"{spent}{wallet}; one wrap-up call is still "
        "admissible, but another similarly reserved work call would consume that room."
    )


def prospective_wrapup_attempt_request(
    *, llm: Any, messages: list[Dict[str, Any]], model: str,
    reasoning_effort: str, tools: Optional[list[Dict[str, Any]]] = None,
    allow_server_web_search: bool = False, prompt_tokens: int = 0,
) -> Any:
    """Build the conservative request facts from the prospective wire payload."""
    from ouroboros.llm import _attempt_request, _finalized_physical_candidate
    from ouroboros.loop_llm_call import MAIN_LOOP_MAX_TOKENS
    from ouroboros.request_wire_recovery import request_wire_call_scope
    from ouroboros.pricing import infer_provider_from_model
    from ouroboros.usage_accounting import AttemptRequest, _merge_scope

    if not callable(getattr(llm, "_resolve_remote_target", None)):
        return _merge_scope(AttemptRequest(
            model=model, provider=infer_provider_from_model(model),
            prompt_tokens_estimate=prompt_tokens,
            max_completion_tokens=MAIN_LOOP_MAX_TOKENS,
        ))[0]

    target = llm._resolve_remote_target(model)
    with request_wire_call_scope():
        candidate = llm._build_remote_candidate(
            target, messages, reasoning_effort, MAIN_LOOP_MAX_TOKENS, "auto", None, tools,
            skip_capability_fetch=True, allow_server_web_search=allow_server_web_search,
        )
        llm._normalize_payload_cache_ttl(target, candidate)
        candidate = _finalized_physical_candidate(
            target, candidate,
            "messages" if target.get("provider") == "anthropic" else "chat.completions",
        )
        llm._pop_thread_disclosure("_cache_breakpoint_tls")
    return _merge_scope(_attempt_request(target, candidate))[0]


def prepared_wrapup_candidate(
    ctx: Any, messages: list[Dict[str, Any]], *, allow_server_web_search: bool,
) -> Tuple[Any, list[Dict[str, Any]]]:
    """Prepare the exact first-send transcript and price that same payload."""
    from ouroboros.loop_llm_call import _prepare_main_messages

    send_messages = _prepare_main_messages(
        messages, model=ctx.active_model, llm=ctx.llm,
        accumulated_usage=ctx.accumulated_usage,
        drive_root=ctx.drive_root or pathlib.Path(ctx.drive_logs or ".").parent,
        task_id=ctx.task_id, event_queue=ctx.event_queue,
        use_local=ctx.active_use_local,
        task_attempt=ctx.accumulated_usage.get("_task_attempt"),
        deadline_ts=ctx.deadline_ts,
    )
    request = prospective_wrapup_attempt_request(
        llm=ctx.llm, messages=send_messages, model=ctx.active_model,
        reasoning_effort=ctx.active_effort, tools=ctx.tool_schemas,
        allow_server_web_search=allow_server_web_search,
        prompt_tokens=int(ctx.accumulated_usage.get("_context_prompt_estimate") or 0),
    )
    return request, send_messages


def wrapup_reservation_fits(
    *,
    model: str = "",
    prompt_tokens: int = 0,
    root_cap_usd: Optional[float],
    deciding_usd: Optional[float],
    global_remaining_usd: Optional[float] = None,
    reservation_count: int = 1,
    request: Any = None,
    llm: Any = None,
    messages: Optional[list[Dict[str, Any]]] = None,
    reasoning_effort: str = "medium",
    tools: Optional[list[Dict[str, Any]]] = None,
    use_local: bool = False,
    allow_server_web_search: bool = False,
) -> Optional[bool]:
    """Whether a wrap-up reservation fits every known root/global remainder.

    Borrows the ledger fence's OWN per-attempt reservation so the graceful stop
    and the fence can never disagree about what a wrap-up call costs: the same
    function, the same cache split, the same arithmetic. Returns None -- fail
    open, the axis stays silent -- when there is no bound task scope, no known
    remainder, or no known price for the route. ``reservation_count=2`` detects the
    last-fit window while one final call is still admissible.

    Deliberately does NOT read ``usage_projection``: the deciding spend is
    passed in by the caller alongside its fresh global remainder. Several
    candidate probes reuse those observations; the final atomic fence still
    arbitrates competing tasks. No money is reserved by this read."""
    try:
        from ouroboros.loop_llm_call import MAIN_LOOP_MAX_TOKENS
        from ouroboros.pricing import infer_provider_from_model
        from ouroboros.usage_accounting import AttemptRequest, _merge_scope, _reservation_cost, current_usage_scope

        scope = current_usage_scope()
        task_id = str(getattr(scope, "task_id", "") or "") if scope is not None else ""
        if not task_id or use_local:
            return None
        remaining = []
        if root_cap_usd is not None and deciding_usd is not None:
            remaining.append(float(root_cap_usd) - float(deciding_usd))
        if global_remaining_usd is not None:
            remaining.append(float(global_remaining_usd))
        if not remaining:
            return None
        if request is None and messages is not None and callable(getattr(llm, "_resolve_remote_target", None)):
            request = prospective_wrapup_attempt_request(
                llm=llm, messages=messages, model=model, reasoning_effort=reasoning_effort,
                tools=tools, allow_server_web_search=allow_server_web_search,
            )
        if request is None:
            if prompt_tokens <= 0:
                return None
            request = AttemptRequest(
                model=str(model or ""), provider=infer_provider_from_model(str(model or "")),
                prompt_tokens_estimate=int(prompt_tokens), max_completion_tokens=MAIN_LOOP_MAX_TOKENS,
                task_id=task_id,
            )
        request, _scope = _merge_scope(request)
        bound = _reservation_cost(request)
        if bound is None:
            return None
        return all(room > 1e-9 and float(bound) * max(1, int(reservation_count)) <= room + 1e-9
                   for room in remaining)
    except Exception:
        log.warning("Wrap-up affordability check failed; axis stays silent", exc_info=True)
        return None


def _cost_checkpoint(
    kind: str,
    *,
    deciding: float,
    task_cost: Optional[float],
    base: float,
    hard_stop: bool,
    spend_basis: str,
    **extra: Any,
) -> Dict[str, Any]:
    """The cost checkpoint payload, with ONE meaning per key name.

    ``task_cost_usd`` has meant THIS task's own accumulated cost since v6.56.0
    and still does — ``loop.py``'s ``_acceptance_loop_rails`` publishes the same
    name with the same meaning, and the rails line renders it as "$X spent this
    task". v6.91 briefly published the tree-accounted DECIDING number under that
    name, so the same key silently changed axis across a version boundary and
    every historical log reader would have been quietly re-pointed. The deciding
    number now rides its own honest name instead, and both are always present so
    no reader has to infer which axis a value came from (``spend_basis`` says
    which one the crossing used)."""
    return {
        "checkpoint_kind": kind,
        **extra,
        "deciding_spend_usd": round(float(deciding), 4),
        "task_cost_usd": round(float(task_cost), 4) if task_cost is not None else None,
        "base_usd": round(float(base), 4),
        "hard_stop": hard_stop,
        # Always present: a reader must be able to tell a tree number from an
        # own-cost stand-in without inferring it from a missing key.
        "spend_basis": spend_basis,
    }


def build_cost_budget_note(
    ctx: Any,
    *,
    start_remaining_usd: Optional[float],
    cost_ceiling_usd: Optional[float],
    task_cost: Optional[float],
    tree_cost_usd: Optional[float] = None,
    root_cap_usd: Optional[float] = None,
) -> Optional[PacingNote]:
    """Cost milestone note at 50/25/10% of the in-task cost budget remaining,
    plus a one-shot wrap-up note at ~80% spent. Fires only on crossings (never
    per round — prompt-cache friendly), latched on ctx like the time axis.

    The reference base is the hard-stop ceiling when one exists, else the
    budget remaining at task start (the informational base for
    ``cost_hard_stop_pct=0`` runs). ``start_remaining_usd`` None (no finite
    budget) keeps the axis silent. ADVISORY only — the hard stop itself lives
    in the loop's budget gate, not here (P5).

    ``tree_cost_usd`` (v6.91) is the root subtree's ledger-accounted spend
    (settled + reserved + unresolved holds, subagents included) — when known it
    is the DECIDING spend, because the ledger fence counts the tree, not this
    task's own calls (waves died at tree $84-94 while own showed $41-49).
    ``task_cost`` (own accumulated cost) stays the diagnostic line. Unknown tree
    spend falls back to own cost — never coerced to $0, and never SILENTLY
    substituted: under a ``root_cap_usd`` the fallback is a lower bound and the
    note says so (basis vocabulary in ``resolve_deciding_spend``)."""
    base = cost_ceiling_usd if cost_ceiling_usd is not None else start_remaining_usd
    deciding, spend_basis = resolve_deciding_spend(
        tree_cost_usd=tree_cost_usd, task_cost_usd=task_cost, root_cap_usd=root_cap_usd,
    )
    if base is None or base <= 0 or deciding is None:
        return None
    tree_basis = spend_basis == SPEND_BASIS_TREE
    spent_fraction = max(0.0, float(deciding)) / base
    fraction_remaining = max(0.0, 1.0 - spent_fraction)
    seen = getattr(ctx, "_cost_budget_milestones_seen", None)
    if not isinstance(seen, set):
        seen = set()
        ctx._cost_budget_milestones_seen = seen
    crossed = [(value, label) for value, label in _COST_BUDGET_THRESHOLDS if fraction_remaining <= value]
    unseen_crossed = [(value, label) for value, label in crossed if label not in seen]
    hard_stop = cost_ceiling_usd is not None
    base_kind = "in-task cost ceiling" if hard_stop else "start-of-task budget snapshot (no in-task cost stop)"
    if tree_basis:
        own_text = f"; own calls ~${task_cost:.2f}" if task_cost is not None else ""
        spent_line = (
            f"Spent this task tree: ~${deciding:.2f} "
            f"(ledger-accounted incl. in-flight holds, subagents included{own_text})"
        )
    elif spend_basis == SPEND_BASIS_OWN_TREE_UNKNOWN:
        spent_line = (
            f"Spent this task: ~${deciding:.2f} (OWN calls only — the tree-accounted "
            "total is unavailable right now, so subagent spend is NOT included; treat "
            "this as a lower bound against the tree cap)"
        )
    else:
        spent_line = f"Spent this task: ~${deciding:.2f}"
    if unseen_crossed:
        selected_label = unseen_crossed[-1][1]  # thresholds are coarse→fine
        for _value, label in crossed:
            seen.add(label)
        _late = spent_fraction >= _COST_WRAPUP_SPENT_FRACTION
        if _late:
            # The tightest milestones already carry the convergence call; a
            # separate wrap-up note right after would be pure duplication.
            ctx._cost_wrapup_seen = True
        # v6.74.4: a fast-spending workspace task can hit its FIRST cost note
        # already past the wrap-up fraction — that suppressed the wrap-up (the
        # only cost text with the tree sentence), so the late milestone itself
        # must carry it (commit triad r2, sol advisory).
        _tree_tail = (
            _TREE_FLUSH_SENTENCE if _late and _workspace_delivery(ctx) else ""
        )
        text = (
            f"[COST BUDGET — {selected_label} remaining crossed]\n"
            f"{spent_line} | Remaining: ~${max(0.0, base - deciding):.2f} "
            f"of ~${base:.2f} ({base_kind})\n"
            "Use this as planning context, not as a command to stop. Prefer the shortest path "
            "to a verifiable result; if a passing artifact or service already exists, prefer "
            "preserving and verifying it over speculative improvements." + _tree_tail
        )
        checkpoint = _cost_checkpoint(
            "cost_budget_milestone", deciding=deciding, task_cost=task_cost,
            base=base, hard_stop=hard_stop, spend_basis=spend_basis,
            milestone=selected_label,
        )
        return PacingNote(text=text, checkpoint=checkpoint)
    if spent_fraction >= _COST_WRAPUP_SPENT_FRACTION and not getattr(ctx, "_cost_wrapup_seen", False):
        ctx._cost_wrapup_seen = True
        # v6.60.0: the marker PHRASE is protocol-gated (the milestone itself is not).
        _marker_tail = (
            " If the task expects a short answer, record your current best "
            "with a `FINAL ANSWER:` line so it stays salvageable."
            if _protocol_marker_phrases(ctx) else ""
        )
        _tree_tail = _TREE_FLUSH_SENTENCE if _workspace_delivery(ctx) else ""
        if tree_basis:
            _tree_amount = f"tree-accounted ~${deciding:.2f} of ~${base:.2f}"
        elif spend_basis == SPEND_BASIS_OWN_TREE_UNKNOWN:
            _tree_amount = (
                f"~${deciding:.2f} of ~${base:.2f}, counting OWN calls only — the "
                "tree-accounted total is unavailable right now, so this is a lower bound"
            )
        else:
            _tree_amount = f"~${deciding:.2f} of ~${base:.2f}"
        text = (
            f"[COST BUDGET — wrap-up]\n"
            f"~{spent_fraction * 100:.0f}% of the {base_kind} is spent "
            f"({_tree_amount}).\n"
            "Start converging: prefer completing and verifying the current best path over "
            "opening new ones." + _tree_tail + _marker_tail
        )
        checkpoint = _cost_checkpoint(
            "cost_budget_wrapup", deciding=deciding, task_cost=task_cost,
            base=base, hard_stop=hard_stop, spend_basis=spend_basis,
        )
        return PacingNote(text=text, checkpoint=checkpoint)
    return None


# v6.74.4: one commit-neutral tree sentence shared by every pacing axis that
# can end a workspace task (time flush, cost wrap-up) — commit-neutral because
# it reaches acting self_worktree subagents, where git commits are blocked and
# a moved HEAD fails patch capture closed.
_TREE_FLUSH_SENTENCE = (
    " Your working tree ships as-is: leave it in a verified, building "
    "state — revert unverified edits rather than leaving them in the tree."
)


def _workspace_delivery(ctx: Any) -> bool:
    """True when the task's deliverable is a workspace tree. Prefers the
    canonical ``ToolContext.is_workspace_mode()`` authority (registry.py);
    falls back to the raw attribute for lightweight test contexts."""
    probe = getattr(ctx, "is_workspace_mode", None)
    if callable(probe):
        try:
            return bool(probe())
        except Exception:
            return False
    return bool(getattr(ctx, "workspace_root", None))


def acceptance_rails_line(
    budget_snapshot: Any,
    budget_profile: Dict[str, Any],
    passes_done: int,
    loop_rails: Optional[Dict[str, Any]],
    *,
    required_blocking: bool,
    workspace: bool = False,
) -> str:
    """Fail-soft wrapper: the rails line is advisory context; it must never
    take down the acceptance path (fable review r2 #3). Extracted from
    ``loop.py`` (S3 byte offset) — pacing display belongs to the pacing SSOT."""
    try:
        return _acceptance_rails_line_inner(
            budget_snapshot, budget_profile, passes_done, loop_rails,
            required_blocking=required_blocking, workspace=workspace,
        )
    except Exception:
        log.debug("acceptance rails line failed soft", exc_info=True)
        return ""


def _headroom_phrase(
    remaining_known_usd: Optional[float],
    cost_ceiling_usd: Optional[float],
    task_cost_usd: Optional[float],
) -> str:
    """Money headroom to the bound that actually binds first, and which one it is.

    The wallet remainder alone reads as more room than the task has: the
    in-task ceiling usually stops it earlier. Both are shown as one number so
    the mind plans against the real limit, with the binding bound named."""
    wallet = None if remaining_known_usd is None else float(remaining_known_usd)
    ceiling_room = None
    if cost_ceiling_usd is not None and task_cost_usd is not None:
        ceiling_room = max(0.0, float(cost_ceiling_usd) - float(task_cost_usd))
    if wallet is None and ceiling_room is None:
        return "budget left unknown"
    if ceiling_room is None:
        return f"${wallet:.2f} budget left (wallet binds)"
    if wallet is None or ceiling_room <= wallet:
        return f"${ceiling_room:.2f} budget left (in-task cost ceiling binds)"
    return f"${wallet:.2f} budget left (wallet binds)"


def _acceptance_rails_line_inner(
    budget_snapshot: Any,
    budget_profile: Dict[str, Any],
    passes_done: int,
    loop_rails: Optional[Dict[str, Any]],
    *,
    required_blocking: bool,
    workspace: bool = False,
) -> str:
    """One line naming every active termination source with its remaining
    headroom (v6.74.0 A1, owner Q6): money, time, rounds, review passes. Each
    rail comes from its real source — the usage ledger projection, the
    BudgetSnapshot, the loop's round counter, and the pacing pass cap — and an
    unavailable rail is omitted rather than guessed. For workspace deliveries
    the line also carries the tree directive (v6.74.4) — a delivery-state
    instruction, not a termination source. Fail-soft: never raises."""
    parts: list[str] = []
    rails = loop_rails if isinstance(loop_rails, dict) else {}
    try:
        money_bits: list[str] = []
        cost = rails.get("task_cost_usd")
        if cost is not None:
            money_bits.append(f"${float(cost):.2f} spent this task")
        try:
            from ouroboros.usage_accounting import current_usage_scope, usage_projection

            scope = current_usage_scope()
            if scope is not None and scope.root_task_id:
                projection = usage_projection(
                    scope.drive_root, global_limit_usd=scope.global_limit_usd,
                )
                root = (projection.get("by_root") or {}).get(scope.root_task_id) or {}
                remaining = projection.get("remaining_known_usd")
                money_bits.append(_headroom_phrase(remaining, rails.get("cost_ceiling_usd"), root.get("accounted_usd")))
        except Exception:
            log.debug("rails: budget projection unavailable", exc_info=True)
        if money_bits:
            parts.append("money: " + ", ".join(money_bits))
    except (TypeError, ValueError):
        pass
    try:
        if getattr(budget_snapshot, "has_deadline", False):
            parts.append(
                f"time: {max(0.0, budget_snapshot.remaining_sec) / 60:.0f} min left "
                f"({budget_snapshot.reserve_sec / 60:.0f} min finalization reserve)"
            )
    except (TypeError, ValueError):
        pass
    try:
        round_idx = rails.get("round_idx")
        max_rounds = rails.get("max_rounds")
        if round_idx is not None and max_rounds:
            parts.append(f"rounds: {int(round_idx)}/{int(max_rounds)}")
    except (TypeError, ValueError):
        pass
    try:
        cap = effective_max_improvement_passes(
            budget_profile,
            required_blocking=required_blocking,
        )
        if cap is None:
            # None comes only from the unlimited shared cap now (the
            # until_deadline alias path was removed in 7.0, Q10=A).
            why = "review cycles unlimited; " if review_max_cycles() is None else ""
            parts.append(
                f"review passes: {int(passes_done)} done, no local count cap "
                f"({why}deadline/budget rails bind)"
            )
        else:
            passes_part = f"review passes: {int(passes_done)}/{int(cap)}"
            # v6.74.4 freeze directive (count axis): the pass launched at
            # cap-1 is the last one improvement_pass_allowed will admit, so
            # say so. cap==0 never feeds a capsule back; skip the clause, and
            # passes_done >= cap (supersede-reset re-review) is not a launch.
            if 0 <= int(passes_done) < int(cap) and int(passes_done) + 1 >= int(cap):
                passes_part += " — FINAL improvement pass, no further passes will run"
            parts.append(passes_part)
    except (TypeError, ValueError):
        pass
    try:
        # v6.74.4: EVERY workspace improvement capsule carries the tree
        # directive, not just the provably-final one — a deadline/cost rail
        # can end the loop between capsules (commit triad r1, sol), and the
        # tree ships as-is on any forced end.
        if workspace:
            parts.append(
                "workspace delivery: the deliverable is your working tree as "
                "it stands when the task ends — keep it in a VERIFIED state "
                "(rebuild, verify, and commit if the task calls for a commit) "
                "and revert unverified edits rather than shipping them"
            )
    except (TypeError, ValueError):
        pass
    return "; ".join(parts)


def build_time_budget_note(
    ctx: Any,
    *,
    round_idx: int = 0,
    accumulated_usage: Optional[Dict[str, Any]] = None,
    tree_cost_provider: Optional[Any] = None,
) -> Optional[PacingNote]:
    """Deadline-aware milestone note at 50/25/10% remaining, never per-round.

    With no deadline_at (headless/benchmark runs), falls back to intrinsic
    self-pacing. Both are ADVISORY — the model judges when to finalize; neither
    is a deterministic stop gate (P5). Milestone state rides ctx attributes so a
    note fires at most once per threshold."""
    meta = getattr(ctx, "task_metadata", {})
    if not isinstance(meta, dict):
        return None
    created = parse_deadline_ts(meta.get("created_at") or meta.get("started_at"))
    if created is None:
        created = getattr(ctx, "_time_budget_started_at", None)
        if created is None:
            created = utc_now()
            ctx._time_budget_started_at = created
    now = utc_now()
    deadline = parse_deadline_ts(meta.get("deadline_at"))
    if deadline is None:
        return build_intrinsic_pacing_note(
            ctx, created=created, now=now, round_idx=round_idx, accumulated_usage=accumulated_usage,
            tree_cost_provider=tree_cost_provider,
        )
    total = max(1.0, (deadline - created).total_seconds())
    remaining = (deadline - now).total_seconds()
    fraction_remaining = 0.0 if remaining <= 0 else remaining / total
    seen = getattr(ctx, "_time_budget_milestones_seen", None)
    if not isinstance(seen, set):
        seen = set()
        ctx._time_budget_milestones_seen = seen
    # Fire the TIGHTEST crossed milestone, not the coarsest: a task starting
    # already past 50% remaining must announce the real urgency immediately.
    crossed = [(value, label) for value, label in _TIME_BUDGET_THRESHOLDS if fraction_remaining <= value]
    unseen_crossed = [(value, label) for value, label in crossed if label not in seen]
    if not unseen_crossed:
        return None
    selected_label = unseen_crossed[-1][1]  # thresholds are coarse→fine
    for _value, label in crossed:
        seen.add(label)
    elapsed = max(0.0, (now - created).total_seconds())
    remaining_clamped = max(0.0, remaining)
    deadline_text = deadline.isoformat().replace("+00:00", "Z")
    # M4 deadline-flush at the tightest milestone: prompt for a salvageable,
    # grounded deliverable before the hard cutoff. Prompt-only; forced
    # finalization is untouched.
    _marker_flush = (
        " If the task expects a short answer, ALSO end your response with a single line, "
        "exactly: FINAL ANSWER: <answer> — so a salvageable answer is captured before the cutoff."
        if _protocol_marker_phrases(ctx) else ""
    )
    # v6.74.4 (time axis of the freeze directive): for workspace deliverables
    # the tree ships as-is, so the flush must also protect the TREE state.
    _tree_flush = _TREE_FLUSH_SENTENCE if _workspace_delivery(ctx) else ""
    flush_clause = (
        " You are near the hard cutoff: WRITE your best current deliverable now "
        "(write_file/edit_text) and run ONE cheap verify_and_record on it, so a "
        "salvageable, grounded result is in place before the deadline."
        + _tree_flush + _marker_flush
        if selected_label == "10%" else ""
    )
    text = (
        f"[TIME BUDGET — {selected_label} remaining crossed]\n"
        f"Elapsed: ~{elapsed/60:.1f} min | Remaining: ~{remaining_clamped/60:.1f} min | "
        f"Deadline: {deadline_text}\n"
        "Use this as planning context, not as a command to stop. If a passing artifact "
        "or service already exists, prefer preserving and verifying it over speculative "
        "improvements. If not, focus on the shortest path to a verifiable result."
        + flush_clause
    )
    return PacingNote(text=text, checkpoint={
        "checkpoint_kind": "time_budget_milestone",
        "milestone": selected_label,
        "elapsed_sec": round(elapsed, 3),
        "remaining_sec": round(remaining_clamped, 3),
        "deadline_at": deadline_text,
    })


def build_intrinsic_pacing_note(
    ctx: Any,
    *,
    created,
    now,
    round_idx: int,
    accumulated_usage: Optional[Dict[str, Any]],
    tree_cost_provider: Optional[Any] = None,
) -> Optional[PacingNote]:
    """No deadline: surface the agent's OWN elapsed / rounds / cost periodically.

    ADVISORY only — awareness so the one mind can choose to wrap up; deliberately
    no deterministic time/round/cost stop (finalization stays P5 judgment).

    ``tree_cost_provider`` (v6.91): a zero-arg callable returning the root
    subtree's accounting snapshot (``{"accounted_usd", "root_limit_usd"}`` or
    None). Called ONLY when the note actually fires (a rare, already
    cache-breaking surface — never per round), so a fresh ledger read at most
    once per pacing interval keeps the number honest after long child waits.
    Unknown stays "unknown", never $0."""
    interval = get_pacing_interval_sec()
    if interval <= 0:
        return None
    elapsed = max(0.0, (now - created).total_seconds())
    bucket = int(elapsed // interval)
    if bucket <= 0:
        return None
    last_bucket = getattr(ctx, "_pacing_bucket_seen", 0)
    if bucket <= last_bucket:
        return None
    ctx._pacing_bucket_seen = bucket
    raw_cost = (accumulated_usage or {}).get("cost")
    cost = float(raw_cost) if raw_cost is not None else None
    cost_text = f"~${cost:.2f}" if cost is not None else "unknown"
    tree_line = ""
    tree_accounted: Optional[float] = None
    tree_cap: Optional[float] = None
    if callable(tree_cost_provider):
        try:
            tree_info = tree_cost_provider()
        except Exception:
            tree_info = None
        rendered = tree_spend_line(tree_info, getattr(ctx, "_cost_ceiling", None))
        if rendered:
            tree_accounted = float(tree_info["accounted_usd"])
            raw_cap = tree_info.get("root_limit_usd")
            tree_cap = float(raw_cap) if raw_cap is not None else None
            tree_line = f" | {rendered}"
    _marker_tail = (
        " If you have a current best short answer, record it with a `FINAL ANSWER:` line "
        "before continuing so it remains salvageable if later work stalls."
        if _protocol_marker_phrases(ctx) else ""
    )
    text = (
        f"[PACING — ~{elapsed/60:.0f} min elapsed]\n"
        f"Rounds so far: {round_idx} | Elapsed: ~{elapsed/60:.1f} min | Cost so far: {cost_text}"
        f"{tree_line}\n"
        "Planning context, not a command to stop. Periodically confirm you are still on the "
        "shortest path to a verifiable result; if a passing artifact or service already exists, "
        "prefer preserving and verifying it over speculative improvements." + _marker_tail
    )
    checkpoint = {
        "checkpoint_kind": "intrinsic_pacing",
        "elapsed_sec": round(elapsed, 3),
        "rounds": int(round_idx),
        "cost": round(cost, 4) if cost is not None else None,
    }
    if tree_accounted is not None:
        checkpoint["tree_accounted_usd"] = round(tree_accounted, 4)
        checkpoint["tree_cap_usd"] = round(tree_cap, 4) if tree_cap is not None else None
    ceiling = getattr(ctx, "_cost_ceiling", None)
    if isinstance(ceiling, CostCeiling):
        checkpoint["cost_ceiling"] = cost_ceiling_disclosure(ceiling)
    return PacingNote(text=text, checkpoint=checkpoint)
