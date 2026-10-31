"""Task pacing SSOT: threshold crossings, the finalization reserve and budget snapshots.

Owns the time/cost threshold notices, the finalization reserve and the budget snapshots
read from ``task_contract.budget_profile``.

Design contract:
- A threshold notice states the crossing once per threshold (owner 13 Oct: keep the
  one-time fact, drop the advice), rides the round about to be sent (never a paid
  round of its own), its wording is TASK-NEUTRAL and carries no strategy, and note
  identification is by the checkpoint metadata — never a regex strip of transcript text.
- Money crossings have two independent axes: the tree's known spend against its
  in-task ceiling or hard tree cap, and the shared wallet's current known headroom
  against what remained when this execution started (#1129) — so other tasks'
  spending moves the wallet crossing, and this task's spend is never presented as
  the wallet's.
- The model's own finalization stays P5 judgment.
- ``loop.py`` keeps only transport (message append + checkpoint emit); every
  threshold, text, and time computation lives here.
- Threshold notices carry ``resource_facts`` (``with_resource_facts``): per-tool
  call/error counts, producer-reported durations, own/tree/delegated ledger buckets and
  the unreserved global remainder. Only tool name, error flag and reported duration are
  read; no argv, stdout or sleep/poll classification selects behaviour. The every-round
  carrier is the facts line (``loop_messages``), not these notices.
"""

from __future__ import annotations

import logging
import json
import math
import pathlib
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

from ouroboros.config import get_finalization_grace_sec
from ouroboros.contracts.task_contract import normalize_budget_profile
from ouroboros.deadline_utils import parse_deadline_ts, utc_now


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PacingNote:
    """One threshold notice: the user-turn text + its checkpoint event payload."""

    text: str
    checkpoint: Dict[str, Any]


@dataclass(frozen=True)
class BudgetSnapshot:
    """Time-budget facts in seconds: the narrower explicit calendar deadline or execution
    budget. ``has_deadline=False`` (neither) means the task has no time axis."""

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
    """The task's normalized budget_profile, with defaults and no side effects.

    Coordination observation (``delegate_supervision._time_fact``, R49) and
    ordinary resolution share this reader. The retired ``until_deadline`` /
    ``stall_rounds_threshold`` aliases require no migration write or event.
    """
    return normalize_budget_profile(_supplied_budget_profile(ctx))


# Both public names retain their caller contracts over the same pure read.
resolve_budget_profile = observe_budget_profile


def acceptance_timing_events_path(ctx: Any) -> pathlib.Path:
    """Return the canonical event stream shared across split task drives.

    The root admission seal and the owner-source acknowledgement write their durable
    rows here; the name is the one the stream had when task acceptance owned it."""
    metadata = getattr(ctx, "task_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    root = pathlib.Path(str(
        metadata.get("budget_drive_root")
        or getattr(ctx, "budget_drive_root", "")
        or getattr(ctx, "drive_root")
    ))
    return root / "logs" / "events.jsonl"


def _reserve_sec(total_sec: float, profile: Dict[str, Any]) -> float:
    """Finalization reserve = the grace window, or max(grace, reserve_pct × total) when the
    task's budget profile states ``reserve_finalization_pct``."""
    grace = float(get_finalization_grace_sec())
    pct = profile.get("reserve_finalization_pct")
    if pct is None:
        return grace
    return max(grace, total_sec * (float(pct) / 100.0))


def effective_finalization_reserve_sec(ctx: Any) -> float:
    """The EMIT window before a real deadline (v6.55.0): the plain finalization grace to
    emit one best-effort answer or let a network tool return; read by the local
    deadline-finalize trigger and the network-tool clamp. NOT a profile's percentage
    reserve (``BudgetSnapshot.reserve_sec``): applying it here force-finalized a 6h task
    ~54 min early on the 15% profile (adversarial review r1)."""
    return float(get_finalization_grace_sec())


def build_budget_snapshot(ctx: Any, *, profile: Optional[Dict[str, Any]] = None) -> BudgetSnapshot:
    """Snapshot the task's time budget from its explicit calendar/execution facts."""
    return _budget_snapshot(ctx, profile=profile, latch=True)


def observe_budget_snapshot(ctx: Any, *, profile: Dict[str, Any]) -> BudgetSnapshot:
    """The same snapshot WITHOUT the fallback-anchor latch (owner R49): a metadata-poor
    task has no calendar axis a read-only observer could get without WRITING an anchor
    (``delegate_supervision._time_fact`` reports ``not_set``). ``profile`` is required:
    an observer never falls back to the emitting resolver."""
    return _budget_snapshot(ctx, profile=profile, latch=False)


def _budget_snapshot(
    ctx: Any, *, profile: Optional[Dict[str, Any]], latch: bool,
) -> BudgetSnapshot:
    meta = getattr(ctx, "task_metadata", {})
    if not isinstance(meta, dict):
        return BudgetSnapshot(has_deadline=False)
    from ouroboros.config import resolve_task_execution_timeout_sec
    from ouroboros.model_wait import explicit_execution_window_remaining

    execution = None  # #939: an explicit execution budget is a time axis on the shared clock
    left = explicit_execution_window_remaining(ctx)
    total = resolve_task_execution_timeout_sec(meta) if left is not None else None
    if total is not None and left is not None:
        execution = BudgetSnapshot(True, total, max(0.0, total - left), left, _reserve_sec(
            total, profile if profile is not None else resolve_budget_profile(ctx)))
    deadline = parse_deadline_ts(meta.get("deadline_at"))
    if deadline is None:
        return execution or BudgetSnapshot(has_deadline=False)
    created = parse_deadline_ts(meta.get("created_at") or meta.get("started_at"))
    if created is None:
        created = getattr(ctx, "_time_budget_started_at", None)
        if created is None:
            if not latch:
                return execution or BudgetSnapshot(has_deadline=False)
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
    calendar = BudgetSnapshot(has_deadline=True, total_sec=total, elapsed_sec=elapsed,
                              remaining_sec=remaining, reserve_sec=reserve)
    if execution is None:
        return calendar
    # The earlier cutoff owns remaining time; either bound's proportional reserve can
    # still end the spendable window first, so both reserves survive.
    nearest = execution if execution.remaining_sec < calendar.remaining_sec else calendar
    return replace(nearest, reserve_sec=nearest.remaining_sec - min(execution.spendable_sec, calendar.spendable_sec))


# ---------------------------------------------------------------------------
# Threshold notice content (moved from loop.py; loop keeps only transport).

_TIME_BUDGET_THRESHOLDS = ((0.50, "50%"), (0.25, "25%"), (0.10, "10%"))

# Cost axes (v6.56.0; the wallet axis #1129): thresholds are module constants like
# the time axis — deliberately NOT settings keys (owner decision; the per-task knob
# is the contract's budget_profile.cost_hard_stop_pct, not a global).
_COST_BUDGET_THRESHOLDS = ((0.50, "50%"), (0.25, "25%"), (0.10, "10%"))
_COST_WRAPUP_SPENT_FRACTION = 0.80
# Disclosed bases: no default in-task stop (owner 2026-10-07, #1128: no share of the
# wallet, no margin before a cap); and an inherited pre-upgrade number whose author is
# unreadable now, kept provisionally and never widened (a compatibility gap, named).
COST_BASIS_NO_DEFAULT_STOP = "no_default_cost_stop"
COST_BASIS_POLICY_UNVERIFIED = "legacy_policy_unverified"

# Typed cost-ceiling states (v6.91): ``None`` is deliberately NOT overloaded to
# mean both "unlimited" and "exhausted" — a $0.50 bench root cap under a $3
# planning margin must soft-land, never run uncapped.
COST_CEILING_DISABLED = "disabled"
COST_CEILING_ACTIVE = "active"
COST_CEILING_EXHAUSTED_SOFT_LAND = "exhausted_soft_land"
COST_CEILING_UNKNOWN = "unknown"

# Planning margin an EXPLICIT ``cost_hard_stop_pct`` profile subtracts from the
# root cap before its graceful in-task stop (authored experiment math, verbatim;
# ordinary tasks have none, owner 2026-10-07): an ABSOLUTE emit-window's worth of
# money (~2 forced-wrap-up call reservation bounds), NEVER a percentage of the cap
# (a pct reserve amputated ~54 min from a 6h task — v6.54.4 adversarial r1, see
# effective_finalization_reserve_sec) and NOT ledger-held: the fence binds at the full cap.
_WRAPUP_CALL_RESERVATION_BOUND_USD = 1.50
COST_PLANNING_MARGIN_USD = max(1.0, 2.0 * _WRAPUP_CALL_RESERVATION_BOUND_USD)


# Deciding-spend basis vocabulary (v6.91). The tree's KNOWN (settled) spend is
# the authority for every rooted task (#1487: with a root cap the ledger fence
# counts the TREE's known spend; without one the in-task ceiling still decides
# on the subtree); open holds are disclosed beside it, never counted. When it
# is momentarily unavailable the own-cost number still decides — but the
# substitution is DISCLOSED as a lower bound, never silent (BIBLE P1). Only a
# task with no root at all has no tree to read, so its own cost is complete.
SPEND_BASIS_TREE = "tree_known"
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
    - ``disabled``: no in-task stop — an ordinary task (no explicit
      ``cost_hard_stop_pct``, no producer allowance: owner 2026-10-07), an
      explicit ``cost_hard_stop_pct=0`` (bench contract, e.g. SWE-Pro), or no
      finite budget on either axis (e.g. GAIA); the ledger fences still bind.
    - ``active``: ``ceiling_usd`` is a real early stop: an explicit profile's
      authored point or a producer's allowance (a wake's daily remainder), also
      as a member's inherited number, or one whose root authority is unreadable
      now (``legacy_policy_unverified``).
    - ``exhausted_soft_land``: an explicit profile whose root cap leaves no room
      above its planning margin — the loop must enter its graceful best-effort
      wrap-up immediately; it must NEVER run uncapped.
    - ``unknown``: resolution inputs errored; the axis stays silent and only
      the hard money limits bind, the gap represented, never filled in (P1)."""

    state: str
    ceiling_usd: Optional[float] = None
    root_cap_usd: Optional[float] = None
    planning_margin_usd: Optional[float] = None
    basis: str = ""


def _positive(value: Any) -> Optional[float]:
    try:
        number = None if value is None or isinstance(value, bool) else float(value)
    except (TypeError, ValueError):
        return None
    return number if number is not None and math.isfinite(number) and number > 0 else None


def resolve_cost_ceiling(
    budget_remaining_start_usd: Optional[float],
    profile: Dict[str, Any],
    *,
    root_cap_usd: Optional[float] = None,
    non_root_member: bool = False,
    root_ceiling_usd: Optional[float] = None,
) -> CostCeiling:
    """The in-task cost stop, computed ONCE at loop start (typed; v6.91).

    Without an explicit ``cost_hard_stop_pct`` the host has no stop of its own
    (owner 2026-10-07, #1128): only a producer allowance stops early, at its
    actual value — a ROOT's ``root_ceiling_usd`` (a wake's daily remainder,
    ``producer_allowance``) or a member's inherited number (``root_resolved_ceiling``;
    the caller decides which carry that authority); otherwise ``disabled``. The
    ledger fences still bind every send at the full limits, on known spend.

    An EXPLICIT percentage keeps the authored math verbatim. The GLOBAL
    component is that share of the global remaining at task start; 0 disables
    the whole in-task stop (bench contract). The ROOT component is the per-task
    tree cap (``OUROBOROS_PER_TASK_COST_USD`` -> ``UsageScope.root_limit_usd``,
    the SAME value the ledger fence enforces) minus the ABSOLUTE planning margin
    — deliberately NOT pct-scaled. The ceiling is min(available components);
    NEVER a computed $0 — a root cap at or below the margin resolves to
    ``exhausted_soft_land`` instead. An enabled non-root member keeps the
    propagated original root ceiling; a later global balance never re-mints
    that early threshold. A ROOT's producer ceiling lands the same planning
    margin early under an explicit profile. Legacy missing carriers retain a
    disclosed local resolution, never a guessed original root fact.

    Stated plainly rather than implied: under an explicit profile the
    ``room <= 0`` bail is the owner's "$0 ceiling" rule EXACTLY, no wider. A cap
    just ABOVE the margin therefore yields a real but tiny ceiling —
    ``root_cap_usd = COST_PLANNING_MARGIN_USD + 0.01`` gives
    ``ceiling_usd == 0.01``, which the first round's spend crosses — so a
    positive ``ceiling_usd`` is not by itself a promise of working room. Both
    numbers are disclosed on the carrier (``ceiling_usd``, ``root_cap_usd``,
    ``planning_margin_usd``) and printed in the stop text."""
    try:
        pct = profile.get("cost_hard_stop_pct")
        if pct is None:
            cap = _positive(root_cap_usd)
            allowance = _positive(root_ceiling_usd)
            if allowance is None:
                return CostCeiling(state=COST_CEILING_DISABLED, root_cap_usd=cap,
                                   basis=COST_BASIS_NO_DEFAULT_STOP)
            return CostCeiling(state=COST_CEILING_ACTIVE, ceiling_usd=allowance, root_cap_usd=cap,
                               basis="root_resolved_ceiling" if non_root_member else "producer_allowance")
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
        if not non_root_member and root_ceiling_usd is not None and float(root_ceiling_usd) > 0:
            margin = COST_PLANNING_MARGIN_USD
            room = float(root_ceiling_usd) - margin
            if room <= 0:
                return CostCeiling(
                    state=COST_CEILING_EXHAUSTED_SOFT_LAND,
                    root_cap_usd=cap,
                    planning_margin_usd=margin,
                    basis="root_ceiling_at_or_below_planning_margin",
                )
            components.append(room)
            basis_parts.append("root_ceiling_minus_margin")
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


# Who authored a task's early cost stop: an explicit percentage profile, a
# producer allowance, nobody (the removed default), or unreadable.
COST_STOP_EXPLICIT, COST_STOP_PRODUCER, COST_STOP_NONE, COST_STOP_UNKNOWN = (
    "explicit", "producer", "none", "unknown")


def _root_cost_stop_authority(scope: Any) -> Tuple[str, Optional[float]]:
    """Who authored the number a member inherited, from the root's canonical row.

    ONE strict read, only for the ambiguous member (an inherited number, no
    explicit percentage of its own): the number cannot say whether the root took
    it from an explicit profile (returned: that percentage), a producer allowance
    (returned: its actual value) or the removed default (a child scheduled before
    the upgrade; a readable root with neither). A root row that cannot be read,
    or lacks its contract, is ``unknown``: the caller keeps the number, never
    guessing either way."""
    from ouroboros.contracts.task_contract import normalize_budget_profile
    from ouroboros.task_results import load_task_result

    try:
        row = load_task_result(pathlib.Path(scope.drive_root), scope.root_task_id, strict=True)
    except Exception:
        log.warning("Root %s cost-stop authority unreadable", getattr(scope, "root_task_id", ""), exc_info=True)
        return COST_STOP_UNKNOWN, None
    contract = row.get("task_contract") if isinstance(row, dict) else None
    if not isinstance(contract, dict):
        return COST_STOP_UNKNOWN, None
    pct = normalize_budget_profile(contract.get("budget_profile")).get("cost_hard_stop_pct")
    if pct is not None:
        return COST_STOP_EXPLICIT, float(pct)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    producer = _positive(row.get("root_cost_ceiling_usd")) or _positive(metadata.get("root_cost_ceiling_usd"))
    return (COST_STOP_PRODUCER, producer) if producer is not None else (COST_STOP_NONE, None)


def _cost_stop_inputs(ctx: Any) -> Dict[str, Any]:
    """The policy authority and resolver inputs of ONE task: the one reading a start
    (``resolve_task_cost_ceiling``), a cold restore (``restore_cost_ceiling``) and
    the Resume refresh share. Authority is never inferred from a number, a saved
    basis label, or a child's local ``None``."""
    root_cap = None
    root_ceiling = None
    non_root_member = False
    scope = None
    try:
        from ouroboros.usage_accounting import current_usage_scope

        scope = current_usage_scope()
        if scope is not None:
            # Who authored the stop is copied first: the group projection below can fail (a
            # store timeout), and that must leave the scope's carried cap, never erase the stop.
            root_ceiling = getattr(scope, "root_cost_ceiling_usd", None)
            non_root_member = bool(
                scope.root_task_id and scope.task_id and scope.root_task_id != scope.task_id
            )
            root_cap = getattr(scope, "root_limit_usd", None)
            if root_cap is None:  # a Continue's successor: the original root's carried group cap
                root_cap = getattr(scope, "billing_group_limit_usd", None)
            if scope.billing_group_id:
                from ouroboros.usage_admission import task_money_snapshot
                snapshot = task_money_snapshot(scope.drive_root, {"id": scope.task_id}, scope.root_task_id,
                                               root_limit=scope.root_limit_usd)
                if snapshot is not None:
                    root_cap = snapshot["root_limit_usd"]
    except Exception:
        log.debug("Usage scope or group cap projection unavailable for cost ceiling resolution", exc_info=True)
    profile = resolve_budget_profile(ctx)
    if profile.get("cost_hard_stop_pct") is not None:
        authority = COST_STOP_EXPLICIT
    elif _positive(root_ceiling) is None:
        authority, root_ceiling = COST_STOP_NONE, None
    elif not non_root_member:
        authority = COST_STOP_PRODUCER
    else:
        authority, value = _root_cost_stop_authority(scope)
        if authority == COST_STOP_EXPLICIT:
            # The root's authored percentage governs the member it handed its number to.
            profile = {**profile, "cost_hard_stop_pct": int(value)}
        elif authority != COST_STOP_UNKNOWN:
            root_ceiling = value
    return {"authority": authority, "profile": profile, "root_cap_usd": root_cap,
            "non_root_member": non_root_member, "root_ceiling_usd": root_ceiling}


def _resolve_from_inputs(inputs: Dict[str, Any], budget_remaining_usd: Optional[float]) -> CostCeiling:
    if inputs["authority"] == COST_STOP_UNKNOWN:
        # Neither removed as a default nor confirmed as authored: the number the
        # member already carries keeps binding, labelled, until its root is readable.
        return CostCeiling(state=COST_CEILING_ACTIVE, ceiling_usd=_positive(inputs["root_ceiling_usd"]),
                           root_cap_usd=_positive(inputs["root_cap_usd"]), basis=COST_BASIS_POLICY_UNVERIFIED)
    return resolve_cost_ceiling(
        budget_remaining_usd,
        inputs["profile"],
        root_cap_usd=inputs["root_cap_usd"],
        non_root_member=inputs["non_root_member"],
        root_ceiling_usd=inputs["root_ceiling_usd"],
    )


def resolve_task_cost_ceiling(ctx: Any, budget_remaining_usd: Optional[float]) -> CostCeiling:
    """The typed in-task cost stop of ONE task, resolved once per task.

    The root cap comes from the bound usage scope -- the SAME
    ``OUROBOROS_PER_TASK_COST_USD``-derived value the ledger fence enforces
    (``agent.py`` wires it as ``UsageScope.root_limit_usd``), so the graceful
    stop and the fence can never disagree about the cap. The same scope says
    whether this task is the root of its tree or one of its members, and which
    authority (``_cost_stop_inputs``) stands behind any early stop."""
    inputs = _cost_stop_inputs(ctx)
    _remember_cost_stop_policy(ctx, inputs)
    return _resolve_from_inputs(inputs, budget_remaining_usd)


def _remember_cost_stop_policy(ctx: Any, inputs: Dict[str, Any]) -> None:
    try:
        setattr(ctx, "_cost_stop_policy", dict(inputs))
    except Exception:
        log.debug("Cost-stop policy could not be stashed on the tool context", exc_info=True)


def cost_stop_policy(ctx: Any) -> Dict[str, Any]:
    """The authority and resolver inputs behind this task's ceiling: the read its
    latest start or restore made (read now only when neither did in this process)."""
    policy = getattr(ctx, "_cost_stop_policy", None)
    if not isinstance(policy, dict):
        policy = _cost_stop_inputs(ctx)
        _remember_cost_stop_policy(ctx, policy)
    return policy


def cost_stop_authority(ctx: Any) -> str:
    """The authority behind this task's ceiling (``cost_stop_policy``)."""
    return str(cost_stop_policy(ctx)["authority"])


def restore_cost_ceiling(ctx: Any, saved: Any) -> Optional[CostCeiling]:
    """The ceiling a cold continuation (budget pause, owner-wait restart) resumes under.

    Every restore re-reads the start's authority, never the saved number alone:
    an explicit percentage keeps its saved threshold (Q10 Resume moves it later);
    a producer allowance is re-read at its actual value (no old margin kept); a
    stop saved under the removed default is dropped, disclosed in the basis; an
    unreadable authority keeps the bound it had (``legacy_policy_unverified``),
    never re-armed or widened, until a later restore can read it. ``None`` only
    for an explicit policy with nothing saved: the loop resolves it as at any
    start. Callers assign the result."""
    previous = CostCeiling(**saved) if isinstance(saved, dict) else None
    inputs = _cost_stop_inputs(ctx)
    _remember_cost_stop_policy(ctx, inputs)
    if inputs["authority"] == COST_STOP_EXPLICIT:
        return previous
    if inputs["authority"] == COST_STOP_UNKNOWN and previous is not None:
        if previous.state not in {COST_CEILING_ACTIVE, COST_CEILING_EXHAUSTED_SOFT_LAND}:
            return previous
        basis = previous.basis if previous.basis.startswith(COST_BASIS_POLICY_UNVERIFIED) else (
            f"{COST_BASIS_POLICY_UNVERIFIED}({previous.basis})")
        return replace(previous, basis=basis)
    restored = _resolve_from_inputs(inputs, None)
    if (restored.state == COST_CEILING_DISABLED and previous is not None
            and previous.state in {COST_CEILING_ACTIVE, COST_CEILING_EXHAUSTED_SOFT_LAND}):
        removed = "soft_land" if previous.ceiling_usd is None else f"${float(previous.ceiling_usd):.2f}"
        restored = replace(restored, basis=f"{restored.basis}(saved default stop {removed} removed)")
    return restored


def cost_ceiling_disclosure(ceiling: CostCeiling) -> Dict[str, Any]:
    """The start-of-task shape of the ceiling the loop will actually decide on."""
    return {
        "state": ceiling.state,
        "ceiling_usd": ceiling.ceiling_usd,
        "root_cap_usd": ceiling.root_cap_usd,
        "planning_margin_usd": ceiling.planning_margin_usd,
        "basis": ceiling.basis,
        "allocation": "unreserved_shared_pool",
        "rule": (
            "The early in-task cost stop of THIS task's whole tree, if any. Ordinary tasks have none: new paid calls "
            "are refused only when known spend (confirmed and estimated, subagents included) reaches the hard tree "
            "cap, the whole-work group cap or the shared Total budget. An active stop is an explicit "
            "cost_hard_stop_pct profile's point (resolved once at task start, retained by descendants; spend over it "
            "stops) or a producer's allowance such as a wake's daily remainder (known spend reaching it stops). At "
            "any of these a task with an exact continuation pauses with its work saved; an actor without one ends "
            "budget_exhausted instead (only an explicit profile's stop may still buy it one final answer, fenced by "
            "the hard limits; at any other stop the host's notice ends it with no paid recap). "
            "Basis legacy_policy_unverified: a number inherited before this version whose author is unreadable now, "
            "kept provisionally, never widened, re-read at the next start or restore. Nothing here reserves money: "
            "concurrent tasks share the global pool, and calls in flight or charged late can take spend past any "
            "limit. Checkpoints report known spend, open holds beside it, and unreserved shared headroom."
        ),
    }


def in_task_cost_ceiling_disclosure(ctx: Any, budget_remaining_usd: Optional[float]) -> Dict[str, Any]:
    """Resolve this task's ceiling once, stash it on ctx, and disclose that object.

    The loop reads the same stashed object, so the number the model is shown at
    task start and the number that later stops the task cannot differ."""
    ceiling = getattr(ctx, "_cost_ceiling", None)
    if not isinstance(ceiling, CostCeiling):
        ceiling = resolve_task_cost_ceiling(ctx, budget_remaining_usd)
    try:
        setattr(ctx, "_cost_ceiling", ceiling)
    except Exception:
        log.debug("Cost ceiling could not be stashed on the tool context", exc_info=True)
    return cost_ceiling_disclosure(ceiling)


def tree_spend_line(tree_info: Any, ceiling: Optional[CostCeiling] = None) -> str:
    """The one live tree-spend phrase, rendered by the per-round facts line.

    Known (settled) spend decides; open holds ride beside it, never added in.
    Names the BINDING bound: the in-task ceiling when one is active (that is
    what stops the task first), with the hard tree cap the ledger fence
    enforces beside it. Empty string when tree accounting is unavailable --
    unknown is never rendered as $0."""
    if not isinstance(tree_info, dict) or tree_info.get("settled_usd") is None:
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
    known = float(tree_info["settled_usd"])
    holds = max(0.0, float(tree_info.get("accounted_usd") or known) - known)
    return (
        f"Task tree spend: ~${known:.2f}{bound} "
        f"(known, subagents included; +${holds:.2f} open holds not counted; ceiling is unreserved)"
    )


def _wrapup_stop_facts(
    deciding_usd: Optional[float], ceiling: CostCeiling, global_remaining_usd: Optional[float],
) -> Tuple[str, str]:
    """The stop's money facts with the BINDING bound first, and the owner's way out when there is one.

    The old sentence opened with the tree cap whatever had stopped the task, so an
    owner whose shared wallet ran dry at $125 of a $400 cap read it as a broken
    per-task cap. The bound that binds is the one with less room; the wallet is
    also the only one the owner can lift while the task is still running."""
    cap = ceiling.root_cap_usd
    room = None if cap is None or deciding_usd is None else float(cap) - float(deciding_usd)
    if deciding_usd is None:
        spent = "This task's tree spend is unavailable"
    elif cap is not None:
        spent = f"This task's tree spent ${deciding_usd:.2f} of its own ${cap:.2f} cap"
    else:
        spent = f"This task's tree spent ${deciding_usd:.2f}"
    if global_remaining_usd is not None and (room is None or float(global_remaining_usd) <= room):
        cleared = ", so the task cap is not what stopped it" if room is not None else ""
        return (
            f"The shared Total budget is nearly used up: ${global_remaining_usd:.2f} left across all "
            f"tasks. {spent}{cleared}.",
            " Raise Total budget in Settings for more room.",
        )
    wallet = (
        f"; the shared Total budget still has ${global_remaining_usd:.2f}"
        if global_remaining_usd is not None else ""
    )
    return f"{spent}{wallet}.", ""


def wrapup_unaffordable_text(deciding_usd: Optional[float], ceiling: CostCeiling, global_remaining_usd: Optional[float] = None) -> str:
    """The owner-facing reason a task ends without even one affordable wrap-up send."""
    facts, way_out = _wrapup_stop_facts(deciding_usd, ceiling, global_remaining_usd)
    return (
        f"{facts} Not even one wrap-up call can be reserved, so the host delivers the "
        f"retained evidence without a model synthesis.{way_out}"
    )


def wrapup_last_fit_text(deciding_usd: Optional[float], ceiling: CostCeiling, global_remaining_usd: Optional[float] = None) -> str:
    """The owner-facing reason a task claims the last affordable wrap-up send."""
    facts, way_out = _wrapup_stop_facts(deciding_usd, ceiling, global_remaining_usd)
    return (
        f"{facts} One wrap-up call still fits and another work call of this size would not, "
        f"so the task is finishing now with its best current answer.{way_out}"
    )


def main_loop_wire_options(
    model: str, *, allow_server_web_search: bool, bypass_response_cache: bool = False,
) -> Dict[str, Any]:
    """The payload-shaping options EVERY main-loop send declares — one owner for the send and for its pricing.

    The budget wrap-up is admitted against a candidate built BEFORE the send (below)
    that must equal it byte for byte. Each option the send grew on its own — cache
    affinity, then ``stream`` — made the two payloads differ, and the admitted
    answer was refused at dispatch. ``loop_llm_call.call_llm_with_retry`` spreads
    this dict into its send and the prospective builder below spreads the same
    one, so an option added here reaches both and one added elsewhere reaches one."""
    from ouroboros.llm_claudexor import cache_key_for_model
    from ouroboros.provider_models import provider_for_model

    remote = provider_for_model(model) != "claudexor"
    return {
        "stream": True,
        "cache_affinity": cache_key_for_model(model),
        "allow_server_web_search": bool(allow_server_web_search) and remote,
        "bypass_response_cache": bool(bypass_response_cache) and remote,
    }


def prospective_wrapup_attempt_request(
    *, llm: Any, messages: list[Dict[str, Any]], model: str,
    reasoning_effort: str, tools: Optional[list[Dict[str, Any]]] = None,
    allow_server_web_search: bool = False, prompt_tokens: int = 0,
    model_role: str = "main", model_account_override: Optional[str] = None,
    model_turn_state: Any = None,
    cache_affinity: str = "",
    processing_preference: Optional[str] = None,
) -> Any:
    """Build the conservative request facts from the prospective wire payload.

    ``model_turn_state`` is the caller's active-turn transport slot. A candidate
    that a forced send is admitted against must be priced from the SAME slot
    value the send will carry, or the two payloads differ by that field alone.
    Under a bound Main clock (``send_clock``) the copy carries a line of its own
    sample, so its price is advice: the send seals a fresh line and is admitted
    again at its own price (``loop_forced_finalization._forced_admission_predicate``)."""
    from ouroboros.llm import _attempt_request, _finalized_physical_candidate
    from ouroboros.loop_llm_call import MAIN_LOOP_MAX_TOKENS
    from ouroboros.request_wire_recovery import request_wire_call_scope
    from ouroboros.pricing import infer_provider_from_model
    from ouroboros.send_clock import stamp_clock_note
    from ouroboros.usage_accounting import AttemptRequest, _merge_scope
    from ouroboros.model_slots import resolve_processing_preference

    processing_preference = resolve_processing_preference(model_role, override=processing_preference)

    resolved = None
    if callable(getattr(llm, "_resolve_remote_target", None)):
        from ouroboros.llm_attempt import ConnectionUnavailable

        try:
            resolved = llm._resolve_remote_target(model)
        except ConnectionUnavailable:  # the send refuses it ($0) and walks the configured routes; the probe estimates
            resolved = None
    if resolved is None:
        return _merge_scope(AttemptRequest(
            model=model, provider=infer_provider_from_model(model),
            prompt_tokens_estimate=prompt_tokens,
            max_completion_tokens=MAIN_LOOP_MAX_TOKENS,
            processing_preference=processing_preference,
            force_unknown_reservation=bool(processing_preference),
        ))[0]

    target = {**resolved, "processing_preference": processing_preference}
    if target.get("provider") == "claudexor":
        from ouroboros.llm_claudexor import _request, prepare_processing_target

        target = prepare_processing_target(target)
        candidate = stamp_clock_note(_request(target, messages, tools, {"reasoning_effort": reasoning_effort,
            "model_role": model_role, "model_account_override": model_account_override,
            "model_turn_state": model_turn_state,
            "processing_preference": processing_preference,
            "cache_affinity": cache_affinity, "prospective": True}))
        return _merge_scope(replace(_attempt_request(target, candidate),
            force_unknown_reservation=True, max_completion_tokens=MAIN_LOOP_MAX_TOKENS))[0]
    with request_wire_call_scope():
        # The send's own options, from their one owner: a key the send carries and
        # this copy does not is a different payload, and the admitted answer is refused.
        candidate = llm._build_remote_candidate(
            target, messages, reasoning_effort, MAIN_LOOP_MAX_TOKENS, "auto", None, tools,
            skip_capability_fetch=True,
            **main_loop_wire_options(model, allow_server_web_search=allow_server_web_search),
        )
        llm._normalize_payload_cache_ttl(target, candidate)
        anthropic = target.get("provider") == "anthropic"
        candidate = _finalized_physical_candidate(
            target, stamp_clock_note(candidate, blocks=anthropic),
            "messages" if anthropic else "chat.completions",
        )
        llm._pop_thread_disclosure("_cache_breakpoint_tls")
    return _merge_scope(_attempt_request(target, candidate))[0]


def prepared_wrapup_candidate(
    ctx: Any, messages: list[Dict[str, Any]], *, allow_server_web_search: bool,
) -> Tuple[Any, list[Dict[str, Any]]]:
    """Prepare the exact first-send transcript and price that same payload.

    The forced send this candidate admits continues the loop's active transport
    turn, so the candidate is built from that same owner slot."""
    from ouroboros.llm_claudexor import cache_key_for_model
    from ouroboros.loop_llm_call import _prepare_main_messages
    from ouroboros.model_slots import task_model_binding, task_processing_preference
    from ouroboros.model_wait import current_model_wait
    from ouroboros.send_clock import MainSendClock, main_clock_policy
    from ouroboros.transcript_prefix import append_reasoning_retirement_fact, retire_main_reasoning

    owner_ctx = getattr(getattr(ctx, "tools", None), "_ctx", None)
    waiter = current_model_wait()
    role, account = task_model_binding({"model_role": getattr(ctx, "model_role", ""),
        "task_metadata": getattr(owner_ctx, "task_metadata", {})},
        context_fit_plan=getattr(owner_ctx, "context_fit_plan", None),
        overrides=waiter.overrides if waiter else None)

    retire_main_reasoning(ctx, messages)
    append_reasoning_retirement_fact(ctx.accumulated_usage, getattr(ctx, "round_idx", None), messages)
    # Named residual: reads the task's refused-image memory, never recovers a first refusal (vision_routing).
    send_messages = _prepare_main_messages(
        messages, model=ctx.active_model, llm=ctx.llm,
        accumulated_usage=ctx.accumulated_usage,
        drive_root=ctx.drive_root or pathlib.Path(ctx.drive_logs or ".").parent,
        task_id=ctx.task_id, event_queue=ctx.event_queue,
        use_local=ctx.active_use_local,
        task_attempt=ctx.accumulated_usage.get("_task_attempt"),
        deadline_ts=ctx.deadline_ts,
        model_role=role,
        model_account_override=account,
    )
    from ouroboros.loop_forced_finalization import _forced_physical_context
    from ouroboros.usage_accounting import bind_physical_attempt_context

    # ONE Main measurement of the final input sizes the priced copy here and the admitted send
    # (``loop_forced_finalization._call_forced_model_once`` reads it back): same bound context, same allowance.
    physical = _forced_physical_context(ctx, send_messages)
    if owner_ctx is not None:
        owner_ctx._forced_physical_context = physical
    # The forced send seals Main's clock line; its priced copy carries one too.
    with MainSendClock(main_clock_policy(getattr(owner_ctx, "task_metadata", {}),
                                         task_type=str(getattr(ctx, "task_type", "") or ""))).bound(), \
            bind_physical_attempt_context(physical):
        request = prospective_wrapup_attempt_request(
            llm=ctx.llm, messages=send_messages, model=ctx.active_model,
            reasoning_effort=ctx.active_effort, tools=ctx.tool_schemas,
            allow_server_web_search=allow_server_web_search,
            prompt_tokens=int(ctx.accumulated_usage.get("_context_prompt_estimate") or 0),
            model_role=role,
            model_account_override=account,
            model_turn_state=getattr(owner_ctx, "model_turn_state", None),
            # The admitted candidate must be the payload the send will produce: the
            # main loop declares the same install-scoped cache affinity, so this
            # prepared copy binds the same key as that dispatch does.
            cache_affinity="" if getattr(ctx, "active_use_local", False) else cache_key_for_model(ctx.active_model),
            processing_preference=task_processing_preference(
                {"task_metadata": getattr(owner_ctx, "task_metadata", {})}, model_role=role),
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
    """The tree-axis cost checkpoint payload, with ONE meaning per key name.

    ``task_cost_usd`` has meant THIS task's own accumulated cost since v6.56.0
    and still does. v6.91 briefly published the tree-accounted DECIDING number
    under that name, so the same key silently changed axis across a version
    boundary and every historical log reader would have been quietly re-pointed. The deciding
    number now rides its own honest name instead, and both are always present so
    no reader has to infer which axis a value came from (``spend_basis`` says
    which one the crossing used; ``budget_axis`` which bound it was measured on)."""
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


def _crossing(ctx: Any, seen_attr: str, fraction_remaining: float) -> Optional[str]:
    """Latch every crossed threshold and return the tightest NEW one (coarse→fine)."""
    seen = getattr(ctx, seen_attr, None)
    if not isinstance(seen, set):
        seen = set()
        setattr(ctx, seen_attr, seen)
    crossed = [label for value, label in _COST_BUDGET_THRESHOLDS if fraction_remaining <= value]
    unseen = [label for label in crossed if label not in seen]
    seen.update(crossed)
    return unseen[-1] if unseen else None


def build_cost_budget_note(
    ctx: Any,
    *,
    cost_ceiling_usd: Optional[float],
    task_cost: Optional[float],
    tree_cost_usd: Optional[float] = None,
    root_cap_usd: Optional[float] = None,
    whole_work_group: bool = False,
) -> Optional[PacingNote]:
    """The tree axis: one notice per crossing of 50/25/10% remaining, and once at ~80% spent.

    The bound is the active in-task cost ceiling, else the hard tree cap (both are
    stops of THIS tree); with neither the axis is silent — the shared wallet has its
    own axis (``build_wallet_budget_note``), so this task's spend is never measured
    against the wallet. Latched on ctx; the stop itself lives in the loop's budget
    gate, not here (P5). ``tree_cost_usd`` is the root subtree's KNOWN (settled)
    spend, subagents included — when known it is the DECIDING spend, because the
    ledger fence counts the tree (#1487: open holds are not spending). Unknown tree
    spend falls back to own cost, disclosed as a lower bound under a cap, never $0.
    ``whole_work_group``: a Continue's successor, whose spend and effective cap are the
    whole-work group's (the narrower of its root and group allowances)."""
    base = cost_ceiling_usd if cost_ceiling_usd is not None else root_cap_usd
    deciding, spend_basis = resolve_deciding_spend(
        tree_cost_usd=tree_cost_usd, task_cost_usd=task_cost, root_cap_usd=root_cap_usd,
    )
    if base is None or base <= 0 or deciding is None:
        return None
    cap_kind = "whole-work allowance (the narrower of root and group caps)" if whole_work_group else "hard tree cap"
    axis = ("in_task_cost_ceiling" if cost_ceiling_usd is not None
            else "whole_work_allowance" if whole_work_group else "hard_tree_cap")
    base_kind = "in-task cost ceiling" if cost_ceiling_usd is not None else cap_kind
    if cost_ceiling_usd is not None and root_cap_usd is not None:
        base_kind += f" (${float(root_cap_usd):.2f} {cap_kind})"
    spent_fraction = max(0.0, float(deciding)) / base
    if spend_basis == SPEND_BASIS_TREE:
        own_text = f"; own calls ~${task_cost:.2f}" if task_cost is not None else ""
        spent_line = (f"Spent this task tree: ~${deciding:.2f} "
                      f"(known spend, subagents included; open holds not counted{own_text})")
    elif spend_basis == SPEND_BASIS_OWN_TREE_UNKNOWN:
        spent_line = (f"Spent this task: ~${deciding:.2f} (OWN calls only — the tree's known total is "
                      "unavailable right now, so subagent spend is NOT included; a lower bound against the tree cap)")
    else:
        spent_line = f"Spent this task: ~${deciding:.2f}"
    late = spent_fraction >= _COST_WRAPUP_SPENT_FRACTION
    # A workspace task can meet its FIRST note already past ~80% spent (v6.74.4), so a
    # late notice carries the delivery fact itself.
    delivery = _WORKSPACE_DELIVERY_FACT if late and _workspace_delivery(ctx) else ""
    label = _crossing(ctx, "_cost_budget_milestones_seen", max(0.0, 1.0 - spent_fraction))
    checkpoint = dict(deciding=deciding, task_cost=task_cost, base=base, hard_stop=True,
                      spend_basis=spend_basis, budget_axis=axis)
    if label is not None:
        if late:
            ctx._cost_wrapup_seen = True  # the late crossing already states the ~80% fact
        text = (f"[COST BUDGET — {label} remaining crossed]\n"
                f"{spent_line} | Remaining: ~${max(0.0, base - deciding):.2f} of ~${base:.2f} {base_kind}, "
                "unreserved." + delivery)
        return PacingNote(text=text, checkpoint=_cost_checkpoint("cost_budget_milestone", milestone=label, **checkpoint))
    if late and not getattr(ctx, "_cost_wrapup_seen", False):
        ctx._cost_wrapup_seen = True
        if spend_basis == SPEND_BASIS_TREE:
            amount = f"tree known spend ~${deciding:.2f} of ~${base:.2f}"
        elif spend_basis == SPEND_BASIS_OWN_TREE_UNKNOWN:
            amount = (f"~${deciding:.2f} of ~${base:.2f}, counting OWN calls only — the tree's known total "
                      "is unavailable right now, so this is a lower bound")
        else:
            amount = f"~${deciding:.2f} of ~${base:.2f}"
        text = (f"[COST BUDGET — {_COST_WRAPUP_SPENT_FRACTION:.0%} spent crossed]\n"
                f"~{spent_fraction * 100:.0f}% of the {base_kind} is spent ({amount})." + delivery)
        return PacingNote(text=text, checkpoint=_cost_checkpoint("cost_budget_wrapup", **checkpoint))
    return None


def shared_wallet_observation() -> Dict[str, Any]:
    """One read of the shared wallet's known headroom, typed known / no limit / unknown.

    The same projection the wrap-up rail reads (``loop_budget._wrapup_global_remaining``),
    kept with its status instead of collapsing "no limit" and "unreadable" into one
    ``None``. Never a reservation: other tasks draw on the same pool."""
    from ouroboros.usage_accounting import current_usage_scope, usage_projection

    observed_at = utc_now().isoformat().replace("+00:00", "Z")
    scope = current_usage_scope()
    if scope is None or not scope.task_id:
        return {"status": "unknown", "reason": "no usage scope", "observed_at": observed_at}
    try:
        projection = usage_projection(scope.drive_root, global_limit_usd=scope.global_limit_usd, include_roots=False)
    except Exception:
        log.debug("Shared wallet observation unavailable", exc_info=True)
        return {"status": "unknown", "reason": "ledger unreadable", "observed_at": observed_at}
    if projection.get("integrity_degraded"):
        return {"status": "unknown", "reason": "ledger integrity degraded", "observed_at": observed_at}
    if projection.get("limit_usd") is None:
        return {"status": "no_global_limit", "observed_at": observed_at}
    return {"status": "known", "remaining_usd": float(projection.get("remaining_known_usd") or 0.0),
            "limit_usd": float(projection["limit_usd"]), "observed_at": observed_at}


def round_money_observation(usage: Dict[str, Any], round_idx: int, *, tree_reader: Any) -> Dict[str, Any]:
    """Share one wallet/tree reading between this round's notices and facts (#1129).

    A changed facts reprepare clears the cache after a wait or route/view change;
    an unchanged row reuses it. A failed tree read is ``None``.
    """
    cached = usage.get("_round_money_observation")
    if isinstance(cached, dict) and cached.get("round") == round_idx:
        return cached
    try:
        tree = tree_reader()
    except Exception:
        log.debug("Tree accounting unavailable for this round", exc_info=True)
        tree = None
    observation = {"round": round_idx, "wallet": shared_wallet_observation(),
                   "tree": _tree_observation(tree) if isinstance(tree, dict) else None}
    usage["_round_money_observation"] = observation
    return observation


def _tree_observation(tree: Dict[str, Any]) -> Dict[str, Any]:
    """The tree reading's facts the round renders: spend, holds, the effective cap, the
    snapshot age and, for a Continue's whole-work group (``usage_admission.task_money_snapshot``),
    its room, which axis binds and both axes' known spend and caps."""
    fields = ("settled_usd", "accounted_usd", "root_limit_usd", "remaining_known_usd", "age_sec",
              "integrity_degraded", "accounting_basis", "binding_axis")
    kept = {key: tree.get(key) for key in fields if key in tree}
    for axis in ("root_axis", "group_axis"):
        if isinstance(tree.get(axis), dict):
            kept[axis] = {key: tree[axis].get(key) for key in ("settled_usd", "limit_usd")}
    return kept


def build_wallet_budget_note(
    ctx: Any,
    *,
    baseline_usd: Optional[float],
    wallet: Optional[Dict[str, Any]],
    task_cost: Optional[float] = None,
) -> Optional[PacingNote]:
    """The shared-wallet axis (#1129): the wallet's CURRENT known headroom against what
    remained when this execution started, at the same 50/25/10% and ~80% thresholds.

    Every task's spending moves it, so a neighbour draining the pool crosses it while
    this task spends nothing; this task's own cost rides beside it as a diagnostic,
    never as the wallet's spend. Unknown or unlimited wallets and a non-positive
    baseline keep the axis silent; latched on ctx apart from the tree axis."""
    if baseline_usd is None or float(baseline_usd) <= 0 or not isinstance(wallet, dict):
        return None
    if wallet.get("status") != "known":
        return None
    baseline, remaining = float(baseline_usd), max(0.0, float(wallet["remaining_usd"]))
    fraction_remaining = min(1.0, remaining / baseline)
    own = f" This task's own recorded cost: ~${float(task_cost):.2f}." if task_cost is not None else ""
    facts = (f"Shared wallet: ~${remaining:.2f} known headroom left of the ~${baseline:.2f} that remained when "
             f"this execution started (limit ${float(wallet.get('limit_usd') or 0.0):.2f}; observed "
             f"{wallet.get('observed_at') or 'at an unknown time'}). It is unreserved and shared: other tasks' "
             "spending moves it too." + own)
    checkpoint = {"budget_axis": "shared_wallet", "wallet_remaining_known_usd": round(remaining, 4),
                  "wallet_baseline_usd": round(baseline, 4), "wallet_limit_usd": wallet.get("limit_usd"),
                  "observed_at": wallet.get("observed_at"), "hard_stop": True,
                  "task_cost_usd": round(float(task_cost), 4) if task_cost is not None else None}
    label = _crossing(ctx, "_wallet_budget_milestones_seen", fraction_remaining)
    late = 1.0 - fraction_remaining >= _COST_WRAPUP_SPENT_FRACTION
    if label is not None:
        if late:
            ctx._wallet_wrapup_seen = True
        return PacingNote(text=f"[COST BUDGET — shared wallet {label} remaining crossed]\n{facts}",
                          checkpoint={"checkpoint_kind": "cost_budget_milestone", "milestone": label, **checkpoint})
    if late and not getattr(ctx, "_wallet_wrapup_seen", False):
        ctx._wallet_wrapup_seen = True
        return PacingNote(text=f"[COST BUDGET — shared wallet {_COST_WRAPUP_SPENT_FRACTION:.0%} spent crossed]\n{facts}",
                          checkpoint={"checkpoint_kind": "cost_budget_wrapup", **checkpoint})
    return None


# v6.74.4: one commit-neutral delivery fact shared by every axis that can end a
# workspace task (time, cost) — commit-neutral because it reaches acting
# self_worktree subagents, where git commits are blocked and a moved HEAD fails
# patch capture closed.
_WORKSPACE_DELIVERY_FACT = " Workspace delivery: the working tree is delivered as it stands when the task ends."


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


def build_time_budget_note(ctx: Any) -> Optional[PacingNote]:
    """One notice per crossing of 50/25/10% remaining on the nearer explicit time bound.

    The bound is an explicit calendar ``deadline_at`` or API/CLI execution budget
    (``build_budget_snapshot``); without either there is no time axis and no
    notice (the facts line carries elapsed time every round). Each threshold is
    latched once on ctx; finalization stays the model's judgment and the host's
    own bound rails (P5)."""
    meta = getattr(ctx, "task_metadata", {})
    if not isinstance(meta, dict):
        return None
    snapshot = build_budget_snapshot(ctx)
    if not snapshot.has_deadline:
        return None
    total, remaining = max(1.0, snapshot.total_sec), snapshot.remaining_sec
    fraction_remaining = 0.0 if remaining <= 0 else remaining / total
    seen = getattr(ctx, "_time_budget_milestones_seen", None)
    if not isinstance(seen, set):
        seen = set()
        ctx._time_budget_milestones_seen = seen
    # The TIGHTEST crossed threshold, not the coarsest: a task starting already
    # past 50% remaining is told where it actually stands.
    crossed = [(value, label) for value, label in _TIME_BUDGET_THRESHOLDS if fraction_remaining <= value]
    unseen_crossed = [(value, label) for value, label in crossed if label not in seen]
    if not unseen_crossed:
        return None
    selected_label = unseen_crossed[-1][1]  # thresholds are coarse→fine
    for _value, label in crossed:
        seen.add(label)
    elapsed, remaining_clamped = snapshot.elapsed_sec, max(0.0, remaining)
    deadline = parse_deadline_ts(meta.get("deadline_at"))
    deadline_text = deadline.isoformat().replace("+00:00", "Z") if deadline is not None else ""
    from ouroboros.config import explicit_execution_timeout_sec
    time_axis = "; ".join(part for part in (f"Deadline: {deadline_text}" if deadline_text else "", (
        "Execution budget: queue and confirmed pauses excluded" if explicit_execution_timeout_sec(meta) else "")) if part)
    grace = effective_finalization_reserve_sec(ctx)
    text = (
        f"[TIME BUDGET — {selected_label} remaining crossed]\n"
        f"Elapsed: ~{elapsed/60:.1f} min | Remaining: ~{remaining_clamped/60:.1f} min | {time_axis}\n"
        f"Host finalization: one tool-less final answer is requested when ~{grace/60:.1f} min remain "
        "on the nearer explicit bound."
        + (_WORKSPACE_DELIVERY_FACT if selected_label == "10%" and _workspace_delivery(ctx) else "")
    )
    return PacingNote(text=text, checkpoint={
        "checkpoint_kind": "time_budget_milestone",
        "milestone": selected_label,
        "elapsed_sec": round(elapsed, 3),
        "remaining_sec": round(remaining_clamped, 3),
        "deadline_at": deadline_text,
    })


def record_tool_activity(ctx: Any, result: Dict[str, Any]) -> None:
    """Count delivered typed results once, without replaying tool history.

    The existing loop usage carrier also survives a native owner-wait handoff.
    Durations are producer-reported intervals, not a partition of elapsed time;
    absent measurements never become zero or a sleep/poll classification.
    """
    usage = getattr(ctx, "_accumulated_usage", None)
    if not isinstance(usage, dict):
        return
    activity = usage.setdefault("_pacing_tool_activity", {})
    name = str(result["fn_name"])
    row = activity.setdefault(name, {
        "calls": 0, "error_calls": 0, "duration_observations": 0,
        "reported_duration_ms": None,
    })
    row["calls"] += 1
    row["error_calls"] += int(bool(result.get("is_error")))
    from ouroboros.task_activity_facts import record_tool_count
    record_tool_count(ctx, activity)
    duration = (result.get("result_meta") or {}).get("duration_ms")
    if (isinstance(duration, (int, float)) and not isinstance(duration, bool)
            and math.isfinite(duration) and duration >= 0):
        row["duration_observations"] += 1
        row["reported_duration_ms"] = round((row["reported_duration_ms"] or 0) + duration, 3)


def with_resource_facts(note: PacingNote, ctx: Any, usage: Optional[Dict[str, Any]]) -> PacingNote:
    """Enrich an already-triggered threshold notice from existing observation owners.

    No new cadence, model call, admission decision or persistent store. Tool
    counters are incremental; money uses the ledger's existing cached readers
    only when a time/cost threshold notice fires. The two spend subsets
    overlap and unknown, estimated and open cash keep their separate fields.
    """
    from ouroboros.usage_accounting import current_usage_scope, usage_breakdown, usage_projection

    activity = (usage or {}).get("_pacing_tool_activity")
    tools: Dict[str, Any] = {"status": "unavailable"}
    if isinstance(activity, dict):
        names = sorted(activity, key=lambda name: (-activity[name]["calls"], name))
        shown, omitted = names[:10], names[10:]
        tools = {
            "status": "available", "source": "observed_tool_results",
            "coverage": "captured results only; earlier results without counters are not reconstructed",
            "calls": sum(row["calls"] for row in activity.values()),
            "by_tool": [{"tool": name, **activity[name]} for name in shown],
            "omitted_tools": len(omitted),
            "omitted_calls": sum(activity[name]["calls"] for name in omitted),
            "duration_basis": "reported intervals; may overlap; unmeasured calls are excluded",
            "source_task_id": str(getattr(ctx, "task_id", "") or ""),
            "source_path": str(pathlib.Path(ctx.drive_root) / "logs" / "tools.jsonl"),
        }
    spend: Dict[str, Any] = {"status": "unavailable", "source": "physical_attempt_ledger"}
    scope = current_usage_scope()
    if scope is not None and scope.root_task_id:
        fields = ("settled_usd", "confirmed_usd", "estimated_usd", "accounted_usd",
                  "reserved_usd", "unresolved_upper_bound_usd", "unknown_unmetered",
                  "cost_final", "integrity_degraded", "physical_calls", "subscription_sessions")
        def bucket(value: Any) -> Optional[Dict[str, Any]]:
            return {key: value.get(key) for key in fields} if isinstance(value, dict) else None
        try:
            breakdown = usage_breakdown(scope.drive_root, root_task_id=scope.root_task_id)
            global_budget = usage_projection(scope.drive_root, global_limit_usd=scope.global_limit_usd,
                                             include_roots=False)
            spend.update({
                "status": "available", "root_task_id": scope.root_task_id,
                "consistency": "separate ledger observations; concurrent work may change headroom",
                "own_task": bucket((breakdown.get("by_task") or {}).get(scope.task_id)),
                "tree": bucket(breakdown), "delegated_tree": bucket(breakdown.get("delegated")),
                "overlap": "own_task and delegated_tree are subsets of tree; own_task may include delegated sessions",
                "global": {
                    **(bucket(global_budget) or {}),
                    "limit_usd": global_budget.get("limit_usd"),
                    "remaining_known_usd": global_budget.get("remaining_known_usd"),
                    "allocation": "unreserved_shared_pool",
                    "limit_source": (scope.global_limit_source or "usage_scope") if scope.global_limit_usd is not None else "settings_budget_resolver",
                    "limit_revision": scope.global_limit_revision if scope.global_limit_usd is not None else None,
                },
            })
        except Exception:
            log.debug("Pacing spend facts unavailable", exc_info=True)
    facts = {"tools": tools, "spend": spend}
    return PacingNote(
        text=note.text + "\nObserved resource facts (known spend is settled_usd; accounted money adds open holds):\n"
             + json.dumps(facts, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        checkpoint={**note.checkpoint, "resource_facts": facts},
    )
