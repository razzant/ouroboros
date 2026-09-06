"""Max-Review-Cycles semantics for skill review, plus the rebuttal ledger.

Owner decisions Q12/Q16/Q17/Q23 aligned every review gate on ONE meaning of
``OUROBOROS_REVIEW_MAX_CYCLES``: it bounds PAID review cycles per task, and
identical material is never re-reviewed for pay. For skill review that means:

* **Paid cycle** = one physical reviewer-panel dispatch, counted as MONEY:
  every dispatched wave counts whatever its terminal (a quorum-failed panel
  still spent reviewer money); only undispatched attempts stay outside the
  count. A chunked oversized skill (multiple packs) is ONE cycle per wave,
  not per pack. The dispatch fact is recorded WRITE-AHEAD: a durable dispatch
  marker (``skill_review_history``) lands immediately before the first
  transport call, shared by the lifecycle runner and direct callers; terminal
  history rows merge it, and an unmerged marker still counts (a crashed or
  swallowed wave spent the money). Counts are DERIVED from the existing
  append-only review history plus that marker (P7 — no counter file).
* **Ceiling key** (Q23): a task-driven review group (``task:<root>:<skill>``)
  shares the root task's ceiling across every skill that task reviews (a
  follow-up task is a fresh root); a manual group (``manual:<skill>``) has no
  session boundary, so its ceiling is scoped to the CURRENT content_hash —
  revised skill content always starts a fresh manual count.
* **Free replay** (Q17-A): an identical ``(group_id, content_hash,
  roster/contract fingerprint)`` with a recorded SUBSTANTIVE terminal verdict
  (clean/warnings/blockers — never pending/interrupted/timeout/failed/
  cancelled, which are infra facts) and no NEW rebuttal replays the recorded
  verdict at $0, quoting it. A rebuttal is identified by CONTENT: a hash new
  to the snapshot's rows buys exactly one paid rerun.
* **Exhaustion** = free typed refusal + ``emit_review_cycles_exhausted``
  (surface ``skill_review``) with honest LLM-first exits, never a silent grind.

This module also owns the accepted-rebuttal ledger (moved whole from
``skill_review.py`` at the module-size gate; ``skill_review`` re-exports the
historical names) — rebuttals are the currency the cycle semantics meter.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
from typing import Any, Dict, List, Optional

from ouroboros.review_cycles import emit_review_cycles_exhausted, review_max_cycles
from ouroboros.skill_review_history import iter_history_rows_bounded, load_history
from ouroboros.skill_review_status import (
    STATUS_BLOCKERS,
    STATUS_CLEAN,
    STATUS_PENDING,
    STATUS_WARNINGS,
    normalize_skill_review_status,
    review_status_grandfatherable,
)
from ouroboros.utils import atomic_write_json, utc_now_iso

log = logging.getLogger(__name__)

# Resolve repo root from this file for source and packaged builds.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# The ONLY statuses a free replay may quote: real reviewer verdicts. Everything
# else on a terminal row (pending/interrupted/timeout/failed/cancelled) is an
# infrastructure fact. The exact rule (F4): refusal/replay ELIGIBILITY
# (substantive verdicts only — infra terminals never replay, never lapse a
# streak, and never "spend" the rebuttal they carried) is distinct from MONEY
# accounting (every dispatched wave counts toward the ceiling via its ``paid``
# fact, whatever its terminal status). A dispatched infra terminal therefore
# consumes money but leaves the rebuttal fresh for the paid rerun it is owed.
SUBSTANTIVE_SKILL_REVIEW_STATUSES = frozenset(
    {STATUS_CLEAN, STATUS_WARNINGS, STATUS_BLOCKERS}
)


def _skill_prompt_contract_hash() -> str:
    """Identity of the SHIPPED skill-review prompt/aggregation contract: the
    prompt builder's source text (the template literals ARE the contract) plus
    the aggregation vocabulary skill_review_status pins. Any shipped change to
    either lapses free-replay authority — conservative in the paid direction.
    Fail-open "": an unreadable contract never matches, so nothing replays."""
    try:
        import inspect

        from ouroboros import skill_review as _skill_review
        from ouroboros import skill_review_status as _status

        parts = [
            inspect.getsource(_skill_review._build_review_prompt),
            json.dumps(sorted(_status.HARD_CRITICAL_ITEMS)),
            json.dumps(sorted(_status.SEVERITY_DRIVEN_ITEMS)),
            str(int(getattr(_status, "WARNINGS_CONVERGENCE_ROUNDS", 0) or 0)),
        ]
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    except Exception:
        log.debug("skill prompt-contract hash unavailable (fail-open)", exc_info=True)
        return ""


def skill_review_contract_fingerprint(
    models: List[str], *, required_items: Any = (), review_profile: str = "",
    delivery: Optional[Dict[str, Any]] = None,
) -> str:
    """Identity of the skill-review panel contract (Q17/Q22 pattern:
    plan_review's ``reviewer_config_fingerprint``): the configured roster, the
    required-item vocabulary, the shipped prompt/aggregation contract text
    (skill-4 parity with the commit gate) and the skill's RESOLVED review
    profile (official_hub aggregates blockers differently — a profile change
    must lapse replay). A changed fingerprint lapses free-replay authority —
    a fresh paid panel is due."""
    prompt_contract = _skill_prompt_contract_hash()
    if not prompt_contract:
        return ""  # unknown contract never matches (fail-open toward paying)
    # Preserve the exact historical bytes for a genuinely unchanged legacy
    # API/global-effort panel. Structured identity and explicit session routes
    # use the canonical per-row contract, sorted by stable owner slot id so a
    # reorder alone does not lapse replay.
    from ouroboros.config import resolve_effort

    identity: Dict[str, Any]
    if not delivery or delivery.get("legacy_skill_fingerprint"):
        identity = {
            "models": [str(model) for model in (models or [])],
            "effort": str(resolve_effort("review") or ""),
        }
    else:
        rows = [
            {
                "slot_id": str(slot_id or ""),
                "route": str(getattr(route, "value", route) or ""),
                "target": str(model or ""),
                "session_target": str(session_target or ""),
                "profile": str(profile or ""),
                "effort": str(effort or ""),
            }
            for model, route, effort, session_target, profile, slot_id in zip(
                delivery.get("models") or [], delivery.get("routes") or [],
                delivery.get("efforts") or [], delivery.get("session_targets") or [],
                delivery.get("session_profiles") or [], delivery.get("slot_ids") or [],
            )
        ]
        # Actor binding is delivery identity (native retrieval vs packet);
        # added only when present so unchanged rosters keep their exact bytes.
        actor_ids = [str(a or "") for a in (delivery.get("subagent_ids") or [])]
        if any(actor_ids):
            for row, actor in zip(rows, actor_ids):
                row["subagent_id"] = actor
        identity = {"reviewer_rows": sorted(rows, key=lambda row: row["slot_id"])}
        if any(row["route"] == "agent_session" for row in rows):
            from ouroboros.skill_review_passes import skill_review_session_contract_hash

            session_contract = skill_review_session_contract_hash()
            if not session_contract:
                return ""  # unknown contract never matches (fail-open toward paying)
            identity["session_prompt_contract"] = session_contract
    identity.update({
        "required_items": [str(item) for item in (required_items or ())],
        "prompt_contract": prompt_contract,
        "review_profile": str(review_profile or ""),
    })
    payload = json.dumps(identity, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _group_root_task_id(group_id: str) -> str:
    parts = str(group_id or "").split(":", 2)
    return parts[1] if len(parts) == 3 and parts[0] == "task" and parts[1] else ""


def _paid_row(row: Any) -> bool:
    # The ceiling counts MONEY (machine-5): every row whose panel physically
    # dispatched counts, whatever its terminal (a quorum-failed wave still
    # spent reviewer money). Rows without the paid fact — free replays,
    # pre-dispatch refusals, pre-upgrade history, interrupted jobs whose
    # dispatch is unknown — never count (fail-open).
    return isinstance(row, dict) and bool(row.get("paid"))


def _unmerged_markers_paid(
    drive_root: pathlib.Path,
    skill_name: str,
    rows: List[Dict[str, Any]],
    *,
    group_id: str = "",
    content_hash: str = "",
    root: str = "",
) -> int:
    """How many of this skill's write-ahead dispatch markers record a paid wave
    on the given ceiling key whose terminal row has NOT landed yet (crashed
    wave, or a direct-call infra outcome without a history row) — the money
    was spent before the first transport call and must count. Markers are
    per-wave files, so EVERY concurrent unmerged wave counts; a landed
    terminal row already carries the merged facts (its marker is cleared on
    merge; the wave-key check below also guards a failed clear)."""
    from ouroboros.skill_review_history import load_dispatch_markers

    landed_waves = {
        str(row.get("wave_id") or row.get("job_id") or "")
        for row in rows if _paid_row(row)
    }
    count = 0
    for marker in load_dispatch_markers(pathlib.Path(drive_root), skill_name):
        wave = str(marker.get("wave_id") or "")
        if not wave or not marker.get("paid") or wave in landed_waves:
            continue
        if root:
            if str(marker.get("root_task_id") or "") != root:
                continue
        elif (
            str(marker.get("group_id") or "") != str(group_id or "")
            or str(marker.get("content_hash") or "") != str(content_hash or "")
        ):
            continue
        count += 1
    return count


def count_paid_skill_review_cycles(
    drive_root: pathlib.Path, skill_name: str, group_id: str, *, content_hash: str = ""
) -> int:
    """Paid panel cycles already spent on the ceiling key, derived from the
    append-only review history (P7) plus this skill's unmerged write-ahead
    dispatch marker (a dispatched wave whose terminal row never landed still
    spent the money). Task-driven groups count across EVERY skill the root
    task reviewed (terminal rows carry ``root_task_id``) — the root task is a
    naturally bounded lifetime. The manual lane has no session boundary, so
    its ceiling is scoped to the CURRENT content_hash lineage
    (machine-1/skill-1): paid rows for the SAME snapshot only — a revised
    skill (new content_hash) always starts a fresh count, mirroring the
    commit gate's changed-diff reset and Q23-A's "manual = its own task".
    Marketplace-install rows normalize into the same manual lane and are
    scoped identically."""
    root = _group_root_task_id(group_id)
    if not root:
        rows = load_history(drive_root, skill_name, limit=0)
        paid = sum(
            1
            for row in rows
            if str(row.get("group_id") or "") == str(group_id or "")
            and _paid_row(row)
            and str(row.get("content_hash") or "") == str(content_hash or "")
        )
        return paid + _unmerged_markers_paid(
            drive_root, skill_name, rows,
            group_id=group_id, content_hash=content_hash,
        )
    count = 0
    skills_root = pathlib.Path(drive_root) / "state" / "skills"
    try:
        skill_dirs = sorted(path for path in skills_root.iterdir() if path.is_dir())
    except OSError:
        return 0
    for skill_dir in skill_dirs:
        rows: List[Dict[str, Any]] = []
        try:
            # Bounded like every other history reader (CPL4-C12): this is the
            # read that walks EVERY skill's log, so a whole-file scan here
            # multiplies by the number of installed skills. Same disclosed
            # residual as the rest of the family — a group whose newest
            # ordinal-bearing row aged past the window under-counts, it never
            # over-blocks.
            rows = list(iter_history_rows_bounded(drive_root, skill_dir.name))
        except OSError:
            rows = []
        count += sum(
            1
            for row in rows
            if str(row.get("root_task_id") or "") == root and _paid_row(row)
        )
        count += _unmerged_markers_paid(drive_root, skill_dir.name, rows, root=root)
    return count


def find_free_replay_row(
    drive_root: pathlib.Path,
    skill_name: str,
    *,
    group_id: str,
    content_hash: str,
    contract_fingerprint: str,
    rebuttal_sha256: str = "",
) -> Optional[Dict[str, Any]]:
    """The recorded verdict a free replay may quote, or ``None`` for a paid run.

    Replays require the identical triple ``(group_id, content_hash,
    roster/contract fingerprint)`` and a SUBSTANTIVE terminal verdict; rows
    without a recorded fingerprint (pre-upgrade) never replay, and a rebuttal
    hash new to this snapshot's rows buys exactly one paid rerun (a repeated
    hash replays free, quoting the previous outcome)."""
    if not content_hash or not contract_fingerprint:
        return None
    rows = [
        row
        for row in load_history(drive_root, skill_name, limit=0, group_id=group_id)
        if str(row.get("content_hash") or "") == content_hash
    ]
    if not rows:
        return None
    if rebuttal_sha256:
        # A rebuttal is "spent" only when a SUBSTANTIVE verdict answered it
        # (machine-6/skill-3): one recorded on an infra terminal (quorum
        # failure, timeout, interrupted) bought a wave that never delivered a
        # verdict, so it stays fresh and buys the paid rerun it is owed.
        # Spent-hash memory is scoped to the CURRENT contract fingerprint
        # (F4/Q22 parity with replay): a contract lapse clears the streak, so
        # a rebuttal answered under a retired roster/prompt contract is fresh
        # again for the contract that has never adjudicated it.
        seen_rebuttals = {
            str(row.get("rebuttal_sha256") or "")
            for row in rows
            if row.get("rebuttal_sha256")
            and str(row.get("review_contract_fingerprint") or "") == contract_fingerprint
            and normalize_skill_review_status(str(row.get("status") or ""))
            in SUBSTANTIVE_SKILL_REVIEW_STATUSES
        }
        if rebuttal_sha256 not in seen_rebuttals:
            return None  # a NEW rebuttal = one paid rerun
    for row in reversed(rows):
        status = normalize_skill_review_status(str(row.get("status") or ""))
        if status not in SUBSTANTIVE_SKILL_REVIEW_STATUSES:
            continue  # infra facts (pending/interrupted/timeout/…) neither replay nor lapse
        if str(row.get("review_contract_fingerprint") or "") != contract_fingerprint:
            return None  # roster/contract changed (or legacy row): a paid review is due
        return row
    return None


def _quote_replay_row(row: Dict[str, Any]) -> str:
    fails = [f for f in (row.get("fail_findings") or []) if isinstance(f, dict)]
    if not fails:
        return ""
    lines = ["", "Recorded FAIL findings:"]
    for finding in fails[:5]:
        lines.append(
            f"  - [{str(finding.get('severity') or '?').upper()}] "
            f"{finding.get('item') or '?'}: {finding.get('reason_excerpt') or ''}"
        )
    if len(fails) > 5:
        lines.append(f"  … and {len(fails) - 5} more in the skill's review history.")
    return "\n".join(lines)


def free_replay_outcome(
    skill: Any,
    *,
    drive_root: pathlib.Path,
    group_id: str,
    content_hash: str,
    contract_fingerprint: str,
    rebuttal_sha256: str = "",
) -> Optional[Any]:
    """A $0 ``SkillReviewOutcome`` replaying the recorded verdict (Q17-A), or
    ``None`` when a paid panel is due.

    A replay is allowed ONLY when the mutable persisted review state
    (review.json → ``skill.review``) already covers this exact
    (content_hash, status) — the $0 path must be effect-equivalent to the
    paid path, and the paid persist pipeline (state write, grants) already
    ran when that state landed. When the persisted state has diverged (e.g.
    content was edited away and reverted), the replay falls through to a
    PAID rerun that re-persists properly — never a live-lock on a stale
    verdict (skill-2)."""
    row = find_free_replay_row(
        drive_root,
        str(getattr(skill, "name", "") or ""),
        group_id=group_id,
        content_hash=content_hash,
        contract_fingerprint=contract_fingerprint,
        rebuttal_sha256=rebuttal_sha256,
    )
    if row is None:
        return None
    from ouroboros.skill_review import SkillReviewOutcome

    status = normalize_skill_review_status(str(row.get("status") or ""))
    review_state = getattr(skill, "review", None)
    if (
        review_state is None
        or str(getattr(review_state, "content_hash", "") or "") != content_hash
        or normalize_skill_review_status(str(getattr(review_state, "status", "") or ""))
        != status
    ):
        return None  # persisted state diverged from the history row: pay and re-persist
    findings: List[Dict[str, Any]] = list(getattr(review_state, "findings", []) or [])
    reviewer_models: List[str] = list(getattr(review_state, "reviewer_models", []) or [])
    return SkillReviewOutcome(
        skill_name=str(getattr(skill, "name", "") or ""),
        status=status,
        findings=findings,
        reviewer_models=reviewer_models,
        content_hash=content_hash,
        review_contract_fingerprint=contract_fingerprint,
        rebuttal_sha256=rebuttal_sha256,
        replayed_from_ts=str(row.get("ts") or ""),
        single_reviewer_no_diversity=bool(row.get("single_reviewer_no_diversity")),
        convergence_hint=(
            "FREE REPLAY (Max Review Cycles): this exact skill snapshot "
            f"(content_hash {content_hash[:12]}) already has a recorded "
            f"'{status}' verdict from {row.get('ts') or 'an earlier run'} under the "
            "same reviewer roster — no paid panel was dispatched and no cycle was "
            "spent. Change the skill content, or supply a NEW review_rebuttal to buy "
            "exactly one paid rerun." + _quote_replay_row(row)
        ),
    )


def skill_review_cycles_refusal(
    ctx: Any,
    skill_name: str,
    *,
    drive_root: pathlib.Path,
    group_id: str,
    models: List[str],
    content_hash: str,
    contract_fingerprint: str,
) -> Optional[Any]:
    """Free typed refusal when the ceiling key already spent
    ``review_max_cycles()`` paid panel cycles; ``None`` allows the dispatch.
    Fail-open on ledger errors — a cost guard, not a safety gate. DISCLOSED
    RESIDUAL (skill-5): read-at-gate-time with no reservation — the paid fact
    lands write-ahead at first physical dispatch (the dispatch marker), which
    narrows but does not close the window: concurrent dispatches sharing one
    ceiling key can still each read ``paid < cap`` and overshoot by the
    concurrency width."""
    cap = review_max_cycles()
    if cap is None:
        return None
    try:
        paid = count_paid_skill_review_cycles(
            drive_root, skill_name, group_id, content_hash=content_hash,
        )
    except Exception:
        log.debug("paid skill-review cycle count failed (fail-open)", exc_info=True)
        return None
    if paid < cap:
        return None
    root = _group_root_task_id(group_id)
    lane = (
        f"root task {root}"
        if root
        else f"manual lane {group_id} for snapshot {str(content_hash or '')[:12]}"
    )
    fresh_exit = (
        "leave the review to a fresh task with its own ceiling"
        if root
        else "revise the skill content — a new snapshot starts a fresh manual count"
    )
    message = (
        f"⚠️ REVIEW_CYCLES_EXHAUSTED: {lane} already spent {paid} of {cap} paid "
        "skill-review panel cycle(s) (OUROBOROS_REVIEW_MAX_CYCLES). Refusing to buy "
        "another panel; the skill stays honestly pending. Decide the honest exit "
        "yourself: finalize and disclose the unreviewed skill, ask the owner to raise "
        f"Max Review Cycles (3/5/unlimited are one settings change away), or {fresh_exit}. "
        "A rebuttal cannot buy past the ceiling — rebuttal cycles count toward it."
    )
    try:
        from ouroboros.config import get_review_enforcement

        enforcement = str(get_review_enforcement() or "")
    except Exception:
        enforcement = ""
    emit_review_cycles_exhausted(
        getattr(ctx, "event_queue", None),
        drive_root,
        surface="skill_review",
        task_id=str(getattr(ctx, "task_id", "") or ""),
        cycles_paid=int(paid),
        cap=int(cap),
        enforcement=enforcement,
        group_id=str(group_id or ""),
        skill=str(skill_name or ""),
        root_task_id=root,
    )
    from ouroboros.skill_review import SkillReviewOutcome

    return SkillReviewOutcome(
        skill_name=str(skill_name or ""),
        status=STATUS_PENDING,
        reviewer_models=list(models or []),
        content_hash=str(content_hash or ""),
        review_contract_fingerprint=str(contract_fingerprint or ""),
        error=message,
    )


def plugin_api_admission_refusal_outcome(
    ctx: Any,
    skill: Any,
    drive_root: pathlib.Path,
    *,
    content_hash: str,
    admission_error: str,
    persist: bool,
) -> Any:
    """ABI-1: typed $0 refusal at NEW-PASS issuance for an inadmissible extension.

    The predicate (``extension_new_pass_admission_error``) is common to every
    PASS-minting path and lives OUTSIDE the deterministic preflight. Clobber
    guard: when the SAME bytes already hold a live executable verdict (the
    grandfather), nothing is persisted — a repeat review must never destroy
    the hash-bound PASS the grandfather construction depends on.
    """
    from ouroboros.skill_loader import SkillReviewState, save_review_state
    from ouroboros.skill_review import SkillReviewOutcome, _append_skill_review_history

    findings = [{
        "item": "plugin_api_admission",
        "verdict": "FAIL",
        "severity": "critical",
        "reason": admission_error,
        "model": "plugin_api_admission",
    }]
    live = getattr(skill, "review", None)
    live_pass = bool(
        live is not None
        and not live.is_stale_for(content_hash)
        and review_status_grandfatherable(live.status)
    )
    outcome = SkillReviewOutcome(
        skill_name=skill.name,
        status=STATUS_PENDING,
        findings=findings,
        reviewer_models=["plugin_api_admission"],
        content_hash=content_hash,
        error=(
            "new review PASS refused (PluginAPI 2.0 admission): " + admission_error
            + (
                "; the existing hash-bound PASS for these bytes is preserved "
                "(grandfather) and nothing was persisted"
                if live_pass else ""
            )
        ),
    )
    if persist and not live_pass:
        review_state = SkillReviewState(
            status=outcome.status,
            content_hash=content_hash,
            findings=findings,
            reviewer_models=outcome.reviewer_models,
            timestamp=utc_now_iso(),
        )
        save_review_state(drive_root, skill.name, review_state)
        if not getattr(ctx, "_skill_review_lifecycle_guard", False):
            _append_skill_review_history(
                drive_root,
                skill.name,
                status=outcome.status,
                content_hash=content_hash,
                findings=findings,
            )
        skill.review = review_state
    return outcome


def install_skill_dispatch_stamp(
    ctx: Any,
    drive_root: pathlib.Path,
    skill_name: str,
    *,
    group_id: str,
    content_hash: str,
    contract_fp: str,
    rebuttal_sha: str,
) -> tuple[Any, Any]:
    """Install the write-ahead paid stamp for ONE skill-review wave (F3, the
    same seam as the commit gate's F2 stamp): each route executor invokes the
    captured ``ctx._review_paid_stamp`` at its physical point of no return,
    durably writing the ONE dispatch marker shared
    by lifecycle and direct callers (the lifecycle job id names the wave when
    present, so a timeout terminal can merge the marker back). Returns
    ``(stamp, previous_attr_value)`` — the caller restores the previous value
    after the wave."""
    import uuid

    from ouroboros.review_dispatch import ReviewPaidStamp
    from ouroboros.skill_review_history import write_dispatch_marker

    wave_id = (
        str(getattr(ctx, "_skill_review_lifecycle_job_id", "") or "") or uuid.uuid4().hex
    )

    def _write() -> None:
        write_dispatch_marker(
            drive_root,
            skill_name,
            wave_id=wave_id,
            group_id=group_id,
            content_hash=content_hash,
            root_task_id=_group_root_task_id(group_id),
            review_contract_fingerprint=contract_fp,
            rebuttal_sha256=rebuttal_sha,
        )

    stamp = ReviewPaidStamp(_write)
    stamp.wave_id = wave_id
    previous = getattr(ctx, "_review_paid_stamp", None)
    ctx._review_paid_stamp = stamp
    return stamp, previous


def review_wave_budget_block(
    ctx: Any,
    skill_name: str,
    file_packs: List[str],
    models: List[str],
) -> Optional[str]:
    """Return a human-readable refusal when the review wave cannot fit the
    remaining budget (global or root axis, the binding one named), else None.
    Read-only; emits one typed event."""
    from ouroboros.tools.review_helpers import review_wave_binding_fence, review_wave_budget_gate

    # Estimate the WHOLE wave: a chunked oversized skill runs one full
    # reviewer pass PER pack (run_skill_review_passes), and every pass re-sends
    # the stable governance/checklist/host-contract files the prompt builder
    # inlines — so both the payload chars and the governance chars multiply by
    # the pack count. A single-pack estimate would under-admit exactly the
    # multi-chunk waves most likely to die mid-review.
    governance_chars = 0
    for rel in (
        "docs/ARCHITECTURE.md", "docs/DEVELOPMENT.md", "BIBLE.md",
        "docs/CHECKLISTS.md", "docs/CREATING_SKILLS.md",
    ):
        try:
            governance_chars += int((_REPO_ROOT / rel).stat().st_size)
        except OSError:
            pass
    packs = max(1, len(file_packs))
    total_chars = sum(len(pack) + governance_chars for pack in file_packs)
    # One admission slot per PHYSICAL reviewer call (models x packs), each
    # sized at the average per-pack prompt: the input estimate sums to the
    # exact wave total while the per-call output reservation also multiplies
    # by the pack count (a models-only wave under-reserved chunked output).
    admission = review_wave_budget_gate(
        ctx, surface="skill_review", models=list(models) * packs,
        prompt_chars=total_chars // packs,
        extra={"skill_name": skill_name, "packs": packs},
    )
    if admission is None:
        return None
    fence, remedy = review_wave_binding_fence(admission)
    return (
        "review wave declined before dispatch: estimated reviewer-wave cost "
        f"~${admission.get('estimated_wave_usd')} exceeds the remaining budget "
        f"${admission.get('remaining_usd')} ({fence}). No reviewer was called; the skill "
        f"stays pending. Wait for in-flight attempts to settle, {remedy}, or re-run the review in a fresh task."
    )


# --- Accepted-rebuttal ledger (moved whole from skill_review.py) -------------


def accepted_rebuttals_path(drive_root: pathlib.Path, skill_name: str) -> pathlib.Path:
    """Path to persisted accepted rebuttals for one skill."""
    return drive_root / "state" / "skills" / skill_name / "accepted_rebuttals.json"


def load_accepted_rebuttals(drive_root: pathlib.Path, skill_name: str) -> List[Dict[str, Any]]:
    """Return persisted accepted rebuttals (empty list when none / unreadable)."""
    path = accepted_rebuttals_path(drive_root, skill_name)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    items = data.get("items")
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for entry in items:
        if isinstance(entry, dict):
            out.append(entry)
    return out


def persist_rebuttal_flips(
    drive_root: pathlib.Path,
    skill_name: str,
    *,
    history: List[Dict[str, Any]],
    findings: List[Dict[str, Any]],
    review_rebuttal: str,
    content_hash: str,
    responded_models: List[str],
) -> None:
    """Record rebuttals for items that flipped FAIL -> PASS on this attempt."""
    if not review_rebuttal or not history:
        return
    last_fail_items = fail_items_from_history_entry(history[-1])
    current_fail_items = {
        str(f.get("item") or "")
        for f in findings
        if isinstance(f, dict)
        and str(f.get("verdict") or "").upper() == "FAIL"
        and str(f.get("item") or "")
    }
    for item in sorted(last_fail_items - current_fail_items):
        record_accepted_rebuttal(
            drive_root,
            skill_name,
            item=item,
            rebuttal_text=review_rebuttal,
            content_hash=content_hash,
            passed_models=list(responded_models),
        )


def fail_items_from_history_entry(entry: Dict[str, Any]) -> set[str]:
    """Return FAIL item names from both v5.18 and legacy history entries."""
    out = {
        str(f.get("item") or "")
        for f in (entry.get("fail_findings") or [])
        if isinstance(f, dict) and str(f.get("item") or "")
    }
    if out:
        return out
    for signature in entry.get("failure_signature") or []:
        parts = str(signature or "").split(":")
        if len(parts) >= 2 and parts[1].upper() == "FAIL" and parts[0]:
            out.add(parts[0])
    return out


def record_accepted_rebuttal(
    drive_root: pathlib.Path,
    skill_name: str,
    *,
    item: str,
    rebuttal_text: str,
    content_hash: str,
    passed_models: Optional[List[str]] = None,
) -> None:
    """Persist (or refresh) an accepted rebuttal for ``item``."""
    path = accepted_rebuttals_path(drive_root, skill_name)
    existing = load_accepted_rebuttals(drive_root, skill_name)
    target: Optional[Dict[str, Any]] = None
    for entry in existing:
        if str(entry.get("item") or "") == item:
            target = entry
            break
    if target is None:
        target = {
            "item": item,
            "rebuttal_text": rebuttal_text,
            "accepted_at": utc_now_iso(),
            "content_hash_seen": [content_hash] if content_hash else [],
            "models_that_passed_after": list(passed_models or []),
        }
        existing.append(target)
    else:
        target["rebuttal_text"] = rebuttal_text
        target["accepted_at"] = utc_now_iso()
        seen = list(target.get("content_hash_seen") or [])
        if content_hash and content_hash not in seen:
            seen.append(content_hash)
        target["content_hash_seen"] = seen
        if passed_models:
            target["models_that_passed_after"] = list(passed_models)
    try:
        from ouroboros.contracts.schema_versions import with_schema_version
        from ouroboros.skill_loader import SKILL_OWNER_STATE_SCHEMA_VERSION

        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            path,
            with_schema_version({"items": existing}, SKILL_OWNER_STATE_SCHEMA_VERSION),
            trailing_newline=True,
        )
    except OSError:
        log.debug("accepted rebuttal write failed", exc_info=True)
