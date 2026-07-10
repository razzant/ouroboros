"""Task pacing SSOT (v6.54.4): ONE urgency system for a task's time budget.

Absorbs the milestone CONTENT logic that lived inline in ``loop.py`` (deadline
50/25/10% TIME BUDGET notes and the v6.53.0 intrinsic no-deadline pacing) and
adds the acceptance-review budget layer: the finalization reserve, a budget
snapshot, and the improvement-pass gates driven by ``task_contract.budget_profile``
(``improvement_policy`` fixed | adaptive | until_deadline).

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

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ouroboros.config import (
    get_acceptance_max_improvement_passes,
    get_acceptance_reserve_pct,
    get_acceptance_review_est_sec,
    get_finalization_grace_sec,
    get_pacing_interval_sec,
)
from ouroboros.contracts.task_contract import answer_protocol_active, normalize_budget_profile
from ouroboros.deadline_utils import parse_deadline_ts, utc_now


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


def resolve_budget_profile(ctx: Any) -> Dict[str, Any]:
    """The task's normalized budget_profile (from task_contract; absent -> defaults)."""
    contract = getattr(ctx, "task_contract", None)
    if not isinstance(contract, dict):
        meta = getattr(ctx, "task_metadata", {})
        contract = meta.get("task_contract") if isinstance(meta, dict) else None
    profile = contract.get("budget_profile") if isinstance(contract, dict) else None
    return normalize_budget_profile(profile)


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
    always allowed (the pass counter is the only axis)."""
    if not snapshot.has_deadline:
        return True, ""
    if snapshot.spendable_sec > float(get_acceptance_review_est_sec()):
        return True, ""
    return False, "review_skipped_deadline_reserve"


def effective_max_improvement_passes(profile: Dict[str, Any], *, has_deadline: bool = True) -> int:
    """The COUNT axis for improvement passes.

    ``until_deadline`` lifts the count axis ONLY when a deadline exists (the time
    gate is then the real bound); without a deadline the time axis is off, so the
    policy falls back to the configured count cap — otherwise a deadline-less
    task with until_deadline would loop near-unbounded (review round 2)."""
    if profile.get("improvement_policy") == "until_deadline" and has_deadline:
        return 10_000  # bounded by the time gate; a count backstop, not a knob
    cap = profile.get("max_improvement_passes")
    if cap is None:
        cap = get_acceptance_max_improvement_passes()
    return max(0, int(cap))


def improvement_pass_allowed(
    snapshot: BudgetSnapshot,
    passes_done: int,
    profile: Dict[str, Any],
) -> Tuple[bool, str]:
    """Gate 2: one more improvement/obligation pass?

    Bounded by TWO independent axes (count AND time), so an endless loop is
    structurally impossible: (passes < max) AND (remaining − reserve > est_review).
    ``adaptive`` stops early once the spendable window can no longer fit a review
    comfortably (2× the estimate)."""
    if passes_done >= effective_max_improvement_passes(profile, has_deadline=snapshot.has_deadline):
        return False, "improvement_passes_exhausted"
    if not snapshot.has_deadline:
        return True, ""
    est = float(get_acceptance_review_est_sec())
    needed = est * 2.0 if profile.get("improvement_policy") == "adaptive" else est
    if snapshot.spendable_sec > needed:
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


def resolve_cost_ceiling_usd(
    budget_remaining_start_usd: Optional[float],
    profile: Dict[str, Any],
) -> Optional[float]:
    """The in-task cost hard-stop ceiling in USD, computed ONCE at loop start.

    None start (no finite budget — e.g. GAIA runs) -> None: the whole cost axis
    stays silent. ``cost_hard_stop_pct`` None -> the historical 50%-of-remaining
    stop; 0 -> None (explicitly uncapped in-task: deadline/rounds/global gate
    remain the bounds) — NEVER a computed $0 ceiling, which would stop the task
    on its first micro-spend."""
    if budget_remaining_start_usd is None or budget_remaining_start_usd <= 0:
        return None
    pct = profile.get("cost_hard_stop_pct")
    if pct is None:
        pct = _DEFAULT_COST_HARD_STOP_PCT
    pct = max(0, min(100, int(pct)))
    if pct == 0:
        return None
    return budget_remaining_start_usd * (pct / 100.0)


def build_cost_budget_note(
    ctx: Any,
    *,
    start_remaining_usd: Optional[float],
    cost_ceiling_usd: Optional[float],
    task_cost: float,
) -> Optional[PacingNote]:
    """Cost milestone note at 50/25/10% of the in-task cost budget remaining,
    plus a one-shot wrap-up note at ~80% spent. Fires only on crossings (never
    per round — prompt-cache friendly), latched on ctx like the time axis.

    The reference base is the hard-stop ceiling when one exists, else the
    budget remaining at task start (the informational base for
    ``cost_hard_stop_pct=0`` runs). ``start_remaining_usd`` None (no finite
    budget) keeps the axis silent. ADVISORY only — the hard stop itself lives
    in the loop's budget gate, not here (P5)."""
    base = cost_ceiling_usd if cost_ceiling_usd is not None else start_remaining_usd
    if base is None or base <= 0:
        return None
    spent_fraction = max(0.0, float(task_cost)) / base
    fraction_remaining = max(0.0, 1.0 - spent_fraction)
    seen = getattr(ctx, "_cost_budget_milestones_seen", None)
    if not isinstance(seen, set):
        seen = set()
        ctx._cost_budget_milestones_seen = seen
    crossed = [(value, label) for value, label in _COST_BUDGET_THRESHOLDS if fraction_remaining <= value]
    unseen_crossed = [(value, label) for value, label in crossed if label not in seen]
    hard_stop = cost_ceiling_usd is not None
    base_kind = "in-task cost ceiling" if hard_stop else "start-of-task budget snapshot (no in-task cost stop)"
    if unseen_crossed:
        selected_label = unseen_crossed[-1][1]  # thresholds are coarse→fine
        for _value, label in crossed:
            seen.add(label)
        if spent_fraction >= _COST_WRAPUP_SPENT_FRACTION:
            # The tightest milestones already carry the convergence call; a
            # separate wrap-up note right after would be pure duplication.
            ctx._cost_wrapup_seen = True
        text = (
            f"[COST BUDGET — {selected_label} remaining crossed]\n"
            f"Spent this task: ~${task_cost:.2f} | Remaining: ~${max(0.0, base - task_cost):.2f} "
            f"of ~${base:.2f} ({base_kind})\n"
            "Use this as planning context, not as a command to stop. Prefer the shortest path "
            "to a verifiable result; if a passing artifact or service already exists, prefer "
            "preserving and verifying it over speculative improvements."
        )
        return PacingNote(text=text, checkpoint={
            "checkpoint_kind": "cost_budget_milestone",
            "milestone": selected_label,
            "task_cost_usd": round(float(task_cost), 4),
            "base_usd": round(float(base), 4),
            "hard_stop": hard_stop,
        })
    if spent_fraction >= _COST_WRAPUP_SPENT_FRACTION and not getattr(ctx, "_cost_wrapup_seen", False):
        ctx._cost_wrapup_seen = True
        # v6.60.0: the marker PHRASE is protocol-gated (the milestone itself is not).
        _marker_tail = (
            " If the task expects a short answer, record your current best "
            "with a `FINAL ANSWER:` line so it stays salvageable."
            if _protocol_marker_phrases(ctx) else ""
        )
        text = (
            f"[COST BUDGET — wrap-up]\n"
            f"~{spent_fraction * 100:.0f}% of the {base_kind} is spent "
            f"(~${task_cost:.2f} of ~${base:.2f}).\n"
            "Start converging: prefer completing and verifying the current best path over "
            "opening new ones." + _marker_tail
        )
        return PacingNote(text=text, checkpoint={
            "checkpoint_kind": "cost_budget_wrapup",
            "task_cost_usd": round(float(task_cost), 4),
            "base_usd": round(float(base), 4),
            "hard_stop": hard_stop,
        })
    return None


def build_time_budget_note(
    ctx: Any,
    *,
    round_idx: int = 0,
    accumulated_usage: Optional[Dict[str, Any]] = None,
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
    flush_clause = (
        " You are near the hard cutoff: WRITE your best current deliverable now "
        "(write_file/edit_text) and run ONE cheap verify_and_record on it, so a "
        "salvageable, grounded result is in place before the deadline." + _marker_flush
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
) -> Optional[PacingNote]:
    """No deadline: surface the agent's OWN elapsed / rounds / cost periodically.

    ADVISORY only — awareness so the one mind can choose to wrap up; deliberately
    no deterministic time/round/cost stop (finalization stays P5 judgment)."""
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
    cost = float((accumulated_usage or {}).get("cost") or 0.0)
    _marker_tail = (
        " If you have a current best short answer, record it with a `FINAL ANSWER:` line "
        "before continuing so it remains salvageable if later work stalls."
        if _protocol_marker_phrases(ctx) else ""
    )
    text = (
        f"[PACING — ~{elapsed/60:.0f} min elapsed]\n"
        f"Rounds so far: {round_idx} | Elapsed: ~{elapsed/60:.1f} min | Cost so far: ~${cost:.2f}\n"
        "Planning context, not a command to stop. Periodically confirm you are still on the "
        "shortest path to a verifiable result; if a passing artifact or service already exists, "
        "prefer preserving and verifying it over speculative improvements." + _marker_tail
    )
    return PacingNote(text=text, checkpoint={
        "checkpoint_kind": "intrinsic_pacing",
        "elapsed_sec": round(elapsed, 3),
        "rounds": int(round_idx),
        "cost": round(cost, 4),
    })
