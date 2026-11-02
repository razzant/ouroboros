"""Usage accounting and budget-pause events reported by workers.

One owner for folding a worker's reported usage into the ledger and for the two
budget fences a paused root raises: the pause itself and the admission fence
that keeps its descendants out of the queue — including the per-row HOLD that
survives lifting that fence, and the explicit selection that releases one row
(#1196, owner Q9).
"""

from __future__ import annotations

import logging
import copy
import pathlib
import time
import uuid
from typing import Any, Dict, Optional
from ouroboros._usage_rows import REVIEW_ATTRIBUTION_KEYS
from ouroboros.contracts.record_contract import RECORD_CONTRACT_VERSION
from ouroboros.event_usage import event_token_counts
from ouroboros._usage_response import _normalized_input_token_usage, provider_cost_value
# Re-export the shared queue/money predicate and the historical hold key.
from ouroboros.budget_pause import BUDGET_HOLD_KEY, budget_fence_selected  # noqa: F401
from ouroboros.delegate_recovery import owner_restart_in_progress
from ouroboros.utils import append_jsonl, utc_now_iso
from ouroboros.task_results import STATUS_SCHEDULED, write_task_result
from ouroboros.work_stop import hold_fact as budget_hold_fact  # noqa: F401
from ouroboros.work_stop import (
    SELECTION_KEY, clear_refusal, continuation_refusal, declare_stop, reason_policy, record_policy,
    release_refusal, resume_selection, selection_record, stop_record,
)

log = logging.getLogger(__name__)


from supervisor.log_addressing import address_ctx_event as _address_ctx
from supervisor.log_addressing import address_task_event as _address_task_event


def _handle_llm_usage(evt: Dict[str, Any], ctx: Any) -> None:
    usage_raw = evt.get("usage")
    usage: Dict[str, Any] = usage_raw if isinstance(usage_raw, dict) else {}

    # Real-progress signal (activity model): a completed LLM round is genuine work,
    # not just process liveness. Stamp last_progress_at so the timeout enforcer keeps
    # an actively-working task alive (distinct from the 30s liveness heartbeat).
    _tid = str(evt.get("task_id") or "")
    _running = getattr(ctx, "RUNNING", None)
    if _tid and isinstance(_running, dict):
        _m = _running.get(_tid)
        # Mutate IN PLACE — _m is the same object RUNNING already holds. A write-back
        # (`_running[_tid] = _m`) would resurrect a task a cross-thread cancel popped
        # between the get and the write; mutating a popped dict is simply harmless.
        if isinstance(_m, dict):
            _m["last_progress_at"] = time.time()
            # Task-tree attribution: the durable llm_usage row declares
            # root/parent/delegation/lane fields. The worker's per-task endpoint
            # (``log_addressing.task_event_queue``) stamps them by value at the
            # producer, so a usage row handled after its task left RUNNING (the
            # telemetry lane is served after control) keeps its attribution;
            # the RUNNING record only fills a gap an older or partial emitter
            # left (emitter-supplied values win).
            _task = _m.get("task") if isinstance(_m.get("task"), dict) else {}
            for _field in (
                "root_task_id", "parent_task_id", "delegation_role",
                "task_group_id", "requested_model_lane", "effective_model_lane",
            ):
                if not evt.get(_field) and _task.get(_field):
                    evt[_field] = str(_task.get(_field))

    # Normalize usage across loop.py, web_search, review and delegated-run producers.
    # Each counter keeps its own knowledge state (#1578): a missing or malformed field
    # neither drops the event (dispatch_event would swallow it, losing the round) nor
    # claims the provider measured zero tokens.
    token_counts = event_token_counts(usage, evt)
    input_split = _normalized_input_token_usage(usage.get("input_token_usage"))
    prompt_cache_ttl = str(usage.get("prompt_cache_ttl") or evt.get("prompt_cache_ttl") or "")
    ledger_attempt_ids = [
        str(value)
        for value in (usage.get("ledger_attempt_ids") or evt.get("ledger_attempt_ids") or [])
        if value
    ]

    raw_cost = usage.get("cost")
    if raw_cost is None and not usage.get("cost_invalid"):
        raw_cost = evt.get("cost")
    resolved_cost = provider_cost_value(raw_cost)
    cost_known = resolved_cost is not None
    cost_invalid = not cost_known and bool(raw_cost is not None or usage.get("cost_invalid") or evt.get("cost_invalid"))

    # One compatibility-projection write per supervisor loop turn: the event only
    # marks the context dirty and the loop flushes it AFTER bridge intake
    # (``server_liveness.flush_budget_projection``), so N events in one drain cost
    # one ledger render and one STATE_LOCK acquisition, never N. The event row (and
    # its live frame) therefore says ``deferred``, not the outcome of that write.
    ctx.budget_projection_dirty = True
    projection_update_status = "deferred"

    # Server-side web-search citations ({url,title,content}, capped at 20 in
    # llm.py). Persisted so post-hoc audits (e.g. the GAIA leakage audit) can see
    # what the native web-search tool actually fetched — the search happens on the
    # provider side and never appears in tools.jsonl.
    web_search_sources = usage.get("web_search_sources")
    # Host-owned sealed-reasoning pin fact (issue #468): why same-model provider
    # failover was withheld on this call. Bounded {"sealed", "artifact"} dict.
    reasoning_pin = usage.get("reasoning_pin")
    # Provider wire projection / clamp of the requested reasoning effort
    # ({requested, applied, reason, model}); persisted so the owner can audit
    # what tier the provider actually received.
    effort_clamped = usage.get("reasoning_effort_clamped")

    usage_event = {
        "ts": evt.get("ts", utc_now_iso()),
        "type": "llm_usage",
        "record_contract_version": RECORD_CONTRACT_VERSION,
        "task_id": evt.get("task_id", ""),
        "root_task_id": evt.get("root_task_id", ""),
        "parent_task_id": evt.get("parent_task_id", ""),
        "delegation_role": evt.get("delegation_role", ""),
        "task_group_id": evt.get("task_group_id", ""),
        "requested_model_lane": evt.get("requested_model_lane", evt.get("model_lane", "")),
        "effective_model_lane": evt.get("effective_model_lane", ""),
        "category": evt.get("category", "other"),
        "model": evt.get("model", ""),
        "api_key_type": evt.get("api_key_type", ""),
        "model_category": evt.get("model_category", "other"),
        "provider": evt.get("provider") or usage.get("provider", ""),
        "source": evt.get("source", ""),
        **{key: evt[key] for key in ("llm_call_id", "execution_id", "round_id", "round", "session_id") if key in evt},
        **({"session_id": usage["session_id"]} if "session_id" not in evt and usage.get("session_id") else {}),
        **{key: str(evt[key]) for key in REVIEW_ATTRIBUTION_KEYS if evt.get(key)},  # #807: the review wave/slot
        "cost_estimated": bool(evt.get("cost_estimated") or usage.get("cost_estimated")),
        "cost": resolved_cost,
        "cost_known": cost_known,
        "cost_invalid": cost_invalid,
        "cost_final": cost_known and not bool(evt.get("cost_estimated") or usage.get("cost_estimated"))
                      and usage.get("cost_final", evt.get("cost_final", True)) is not False,
        **token_counts,
        **({"input_token_usage": input_split} if input_split is not None else {}),
        "prompt_cache_ttl": prompt_cache_ttl,
        "accounting_authority": "physical_attempt_ledger",
        "projection_update_status": projection_update_status,
        "ledger_attempt_ids": ledger_attempt_ids,
        **({"chat_id": evt["chat_id"]} if evt.get("chat_id") is not None else {}),
        **({"web_search_sources": web_search_sources} if isinstance(web_search_sources, list) and web_search_sources else {}),
        **({"reasoning_pin": reasoning_pin} if isinstance(reasoning_pin, dict) and reasoning_pin else {}),
        **({"reasoning_effort_clamped": effort_clamped} if isinstance(effort_clamped, dict) and effort_clamped else {}),
        **{key: usage[key] for key in ("effort", "request_wire", "effort_resolution", "claudexor")
           if isinstance(usage.get(key), dict)},
    }
    _address_ctx(ctx, usage_event)
    try:
        append_jsonl(ctx.DRIVE_ROOT / "logs" / "events.jsonl", usage_event)
    except Exception:
        log.warning("Failed to log llm_usage event to events.jsonl", exc_info=True)
    # ONE live frame (sink copy suppressed).
    try:
        ctx.bridge.push_log(usage_event)
    except Exception:
        log.debug("Failed to forward llm_usage to live logs", exc_info=True)


def _set_root_budget_pause_locked(root_task_id: str, pause: Dict[str, Any], *,
                                  prior_root: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Install the sole root-budget latch under the queue lock.

    Restore supplies the root row BEFORE parking changed its carriers; live
    callers still read the queue. Its selection must match the CURRENT latch.
    """
    from supervisor import queue as queue_mod

    root_task_id = str(root_task_id or "").strip()
    if not root_task_id:
        raise ValueError("root budget pause requires root_task_id")
    existing = queue_mod.BUDGET_ROOT_FENCES.get(root_task_id)
    root_rows = ([prior_root] if prior_root is not None else
                 list(queue_mod.PENDING) + [m.get("task", {}) for m in queue_mod.RUNNING.values()])
    resumed = any(str(t.get("id") or "") == root_task_id and budget_fence_selected(t, existing)
                  for t in root_rows)
    row = {
        "status": "paused",
        "scope": "root",
        "root_task_id": root_task_id,
        "fence_id": str(
            pause.get("fence_id")
            or (None if resumed else (existing or {}).get("fence_id"))
            or uuid.uuid4().hex
        ),
        "auto_resume": False,
        "paused_at": str(
            pause.get("paused_at")
            or (existing or {}).get("paused_at")
            or utc_now_iso()
        ),
    }
    # An owner Pause's latch keeps its cause while its members park under it:
    # it is never rewritten into a monetary latch (owner_pause.py).
    if (pause.get("reason") == "owner" or "owner_pause" in {
            pause.get("cause"),
            (existing or {}).get("cause") if not resumed or row["fence_id"] == (existing or {}).get("fence_id") else None}):
        row["cause"] = "owner_pause"
    queue_mod.BUDGET_ROOT_FENCES[root_task_id] = row
    if not existing or existing.get("fence_id") != row["fence_id"]:
        from supervisor.budget_resume import revoke_exact_budget_resume

        for member in queue_mod.PENDING:
            if str(member.get("root_task_id") or "") != root_task_id or member.get("id") == root_task_id:
                continue
            if isinstance(member.get("_budget_pause_resume"), dict):
                revoke_exact_budget_resume(member, "new_root_budget_fence")
            if isinstance(member.get(SELECTION_KEY), dict) and budget_hold_fact(member) is None:
                hold_budget_row(member, reason=HOLD_ROOT_FENCE_LIFTED,
                                extra={"root_task_id": root_task_id, "fence_id": row["fence_id"]})
    return row


def finish_resume_clock(meta: dict, paused_sec: float) -> None:
    """Apply a confirmed total to the matched interval, preserving Resume selection."""
    task = meta["task"]
    resume = task.get("_budget_pause_resume") or {}
    meta["budget_paused_sec"] = paused_sec
    meta.pop("sleep_parked_at", None)
    if resume.get("authority") == "explicit_resume":
        task["_budget_pause_resume"] = {key: value for key, value in resume.items()
                                       if key != "sleep_exclusion_since"}
    else:
        task.pop("_budget_pause_resume", None)


def _resume_observation(meta: dict, drive_root: Any) -> dict:
    from ouroboros.budget_pause import budget_pause_row

    task = meta["task"]
    if not (task.get("_budget_pause_resume") or {}).get("sleep_exclusion_since"):
        return {}  # No pending clock observation: ordinary recovery needs no storage read.
    return budget_pause_row(pathlib.Path(task.get("budget_drive_root") or drive_root), task["id"])


def _apply_consumed_resume(meta: Any, row: dict, evt: Optional[Dict[str, Any]] = None) -> bool:
    """Project the matching consumption; lineage permission belongs to Resume admission."""
    from ouroboros.budget_pause import STATE_RESUMED

    task = meta.get("task") if isinstance(meta, dict) else None
    resume = (task.get("_budget_pause_resume") if isinstance(task, dict) else None) or {}
    evt = evt or {"task_attempt": meta.get("attempt") if isinstance(meta, dict) else None,
                  **{key: resume.get(key) for key in ("pause_id", "grant_id")}}
    if (not isinstance(task, dict) or meta.get("attempt") != evt.get("task_attempt")
            or resume.get("pause_id") != evt.get("pause_id")
            or resume.get("grant_id") != evt.get("grant_id")
            or not resume.get("sleep_exclusion_since")
            or meta.get("sleep_parked_at") != resume["sleep_exclusion_since"]):
        return False
    grant = row.get("grant") or {}
    if (row.get("state") != STATE_RESUMED
            or row.get("pause_id") != evt.get("pause_id")
            or grant.get("grant_id") != evt.get("grant_id") or not grant.get("consumed_at")
            or grant.get("revoked_at")
            or row.get("source_ref") != resume.get("source_ref")
            or row.get("pause_generation") != resume.get("pause_generation")
            or grant.get("generation") != resume.get("grant_generation")
            or grant.get("granted_at_ts") != resume["sleep_exclusion_since"]
            or row.get("task_attempt") != evt.get("task_attempt")):
        return False
    finish_resume_clock(meta, float(row["paused_duration_sec"]))
    return True


def fold_consumed_resume(meta: Any, drive_root: Any, evt: Optional[Dict[str, Any]] = None) -> bool:
    """Recover a captured clock from storage; an unavailable observation changes nothing."""
    try:
        row = _resume_observation(meta, drive_root)
    except Exception:
        return False
    return _apply_consumed_resume(meta, row, evt)


def _handle_budget_pause(evt: Dict[str, Any], ctx: Any) -> None:
    """Move a paused task back to the same durable queue generation.

    Two shapes share this seam: the historical replay-safe ZERO-dispatch pause
    and the exact continuation (#1196), whose worker wrote a durable
    ``budget_pause`` row before unwinding. The exact shape is validated against
    THAT row, never against the event's prose.
    """
    if evt.get("phase") == "consumed":
        from ouroboros.store import acquisition_budget, StoreBusy
        from supervisor.queue import _queue_lock

        task_id = str(evt.get("task_id") or "")
        with _queue_lock:
            meta = ctx.RUNNING.get(task_id)
            if not isinstance(meta, dict):
                return
            captured = {"task": dict(meta.get("task") or {})}
        try:
            with acquisition_budget(0):
                observed = _resume_observation(captured, ctx.DRIVE_ROOT)
        except StoreBusy as exc:
            if exc.reason != "contention":
                log.warning("Resume clock store unavailable for %s: %s", task_id, exc)
            return  # The existing assignment tick reobserves the same consumption.
        except Exception:
            log.warning("Resume clock observation unavailable for %s", task_id, exc_info=True)
            return
        with _queue_lock:
            if ctx.RUNNING.get(task_id) is meta:
                _apply_consumed_resume(meta, observed, evt)
        return
    task_id = str(evt.get("task_id") or "")
    pause = evt.get("resource_limit") if isinstance(evt.get("resource_limit"), dict) else {}
    exact = bool(pause.get("exact_continuation")) and isinstance(pause.get("checkpoint"), dict)
    if exact:
        install_exact_budget_pause(ctx, task_id, pause["checkpoint"], evt=evt)
        return
    if (
        not task_id
        or not bool(pause.get("replay_safe"))
        or pause.get("physical_calls") != 0
    ):
        raise ValueError("budget pause requires a replay-safe zero-dispatch task")
    from supervisor.queue import _queue_lock

    prestart = bool(pause.get("before_task_start"))
    with _queue_lock:
        meta = ctx.RUNNING.get(task_id)
        task = meta.get("task") if isinstance(meta, dict) and isinstance(meta.get("task"), dict) else None
        direct = task is None and prestart and evt.get("_is_direct_chat")
        if direct and isinstance(evt.get("task"), dict) and evt["task"].get("id") == task_id:
            task = evt["task"]
        if task is None:
            raise RuntimeError(f"budget-paused task is not running: {task_id}")
        if prestart:
            from ouroboros.cancel_intents import active_intent
            if active_intent(task.get("budget_drive_root") or ctx.DRIVE_ROOT, task_id):
                return  # the accepted Stop owns settlement, never a later Pause event
            import math
            from ouroboros.owner_wait import governing_continuation
            spent = pause.get("execution_consumed_sec")
            attempt = int((meta or {}).get("attempt") or task.get("_attempt") or 1)
            if (attempt != 1 or pause.get("task_attempt") != attempt or evt.get("task_attempt") != attempt
                    or not pause.get("fence_id") or pause.get("cause") != "owner_pause"
                    or type(spent) not in (int, float) or not math.isfinite(spent) or spent < 0
                    or governing_continuation(task) or task.get("original_task_id") or task.get("timeout_retry_from")):
                raise ValueError("pre-start Pause requires this fresh attempt's no-body evidence")
        if str(pause.get("scope") or "") == "root":
            root_row = _set_root_budget_pause_locked(
                str(pause.get("root_task_id") or evt.get("root_task_id") or ""), pause)
            pause = {**pause, **root_row, "status": "paused_before_dispatch",
                     "replay_safe": True, "physical_calls": 0}
        ctx.RUNNING.pop(task_id, None)
        resumed_task = dict(task)
        resumed_task["_budget_pause"] = dict(pause)
        if prestart:
            resumed_task["_execution_handoff_sec"] = pause["execution_consumed_sec"]
        if direct:
            resumed_task.setdefault("priority", 1)
            resumed_task.setdefault("queued_at", utc_now_iso())
        if not any(str(item.get("id") or "") == task_id for item in ctx.PENDING):
            ctx.PENDING.append(resumed_task)
            ctx.sort_pending()
        worker_id = evt.get("worker_id")
        if worker_id in ctx.WORKERS and ctx.WORKERS[worker_id].busy_task_id == task_id:
            ctx.WORKERS[worker_id].busy_task_id = None
    from ouroboros.pause_notices import notice_fields
    import uuid
    episode_id = uuid.uuid4().hex
    try:
        write_task_result(
            ctx.DRIVE_ROOT,
            task_id,
            STATUS_SCHEDULED,
            _field_projector=lambda current, incoming: {
                **incoming, **notice_fields(current, ctx.DRIVE_ROOT, task_id, episode_id, "owner" if prestart else "budget")},
            reason_code="owner_pause" if prestart else "budget_exhausted",
            resource_limit=pause,
            result=("Task paused before its body started; explicit resume or cancel required." if prestart else
                    "Task paused before its first model dispatch; explicit resume or cancel required."),
        )
    except Exception:
        log.warning("Failed to persist budget pause for %s", task_id, exc_info=True)
    event = {
        "ts": evt.get("ts", utc_now_iso()),
        "type": "budget_scope_paused",
        "task_id": task_id,
        "task_type": evt.get("task_type") or task.get("type"),
        "owner_visible": True,
        "toast_once": f"{task_id}:budget-paused:{pause.get('scope') or 'global'}",
        **pause,
    }
    _address_task_event({task_id: meta} if isinstance(meta, dict) else None, ctx.DRIVE_ROOT, event)
    append_jsonl(ctx.DRIVE_ROOT / "logs" / "events.jsonl", event)
    ctx.persist_queue_snapshot(reason="budget_pause_before_dispatch")
    try:
        ctx.bridge.push_log(event)
    except Exception:
        log.warning("Failed to forward budget pause to Activity", exc_info=True)


def _owner_park_fence_id(result_root: pathlib.Path, root_task_id: str) -> Optional[str]:
    """The fence an owner Pause's parked root latch projects (#1730), or ``None``. Queue lock held.

    Its root's CURRENT closed durable fence: the one ``request_owner_pause`` latched,
    every Resume binds and lifts the latch on, and restore raises
    (``queue_snapshot._raise_parked_root_fences``). The fence a row parked under is
    history once a newer Pause supersedes it, possibly with its own latch still
    ``latch_pending``, so it is never projected. Released authority projects nothing
    (``""``: the generation rule). ``None``: the read failed (Q-held reads never wait),
    so the current fence is unknown and ``install_exact_budget_pause`` publishes none.
    """
    from ouroboros.owner_pause import fence_closed, read_fence
    from ouroboros.store import acquisition_budget

    try:
        with acquisition_budget(0):
            current = read_fence(result_root, root_task_id)
    except Exception:
        log.warning("Owner fence of %s is unreadable at its park", root_task_id, exc_info=True)
        return None
    return str(current.get("fence_id") or "") if fence_closed(current) else ""


class OwnerFenceUnknown(RuntimeError):
    """A deferring park met an unreadable current owner fence; nothing was changed."""


def install_exact_budget_pause(ctx: Any, task_id: str, checkpoint: Dict[str, Any], *,
                               evt: Dict[str, Any] | None = None,
                               source: str = "worker_event",
                               defer_unknown_owner_fence: bool = False) -> Dict[str, Any]:
    """Park the SAME task id under its exact continuation: RUNNING -> PENDING.

    Order: durable row must already say ``pausing`` for this pause_id/attempt
    (the worker wrote it before raising) -> queue transition -> snapshot ->
    row ``paused``. The WHOLE order — including the confirmation, the owner
    projection and the event — runs under the queue lock a Resume grant also
    holds, and the confirmation compare-and-sets the pause id AND its state, so
    a park that completes late can never overwrite a grant or republish a
    resumed task as paused (#1196, F1). A snapshot that cannot be persisted
    leaves the row at ``pausing``: the pause is real (the source exists) but not
    confirmed, and the next persisted snapshot or a restore completes it —
    never a fake ``paused``. Also the crash-during-pausing completion path (``source``
    names it), which is why nothing here reads the worker's event body — with
    ONE exception: a direct owner-chat turn was never in RUNNING, so its event
    carries the turn's own task record (``task`` beside ``_is_direct_chat``),
    and THAT record is what the SAME task id is parked under (#1196). The row
    check above still decides; the record only supplies the queue row.

    An owner Pause's root latch names its current durable fence (``_owner_park_fence_id``).
    When that read fails, the death recoveries (``defer_unknown_owner_fence``) raise
    ``OwnerFenceUnknown`` before any change and replay from their retained reaper job; a
    live or direct park, which nothing replays, keeps the standing latch and mints none:
    the durable fence still refuses every launch, as for an accepted ``latch_pending``.
    """
    from ouroboros.budget_pause import (
        STATE_PAUSED, STATE_PAUSING, budget_pause_row, exact_pause_marker, set_budget_pause,
    )
    from supervisor import queue as queue_mod
    from supervisor.queue import _queue_lock

    evt = evt or {}
    # The event-loop ctx and the reaper's pool module expose the same queue
    # state; the snapshot/sort seams fall back to the queue module's own.
    sort_pending = getattr(ctx, "sort_pending", None) or queue_mod.sort_pending
    persist_snapshot = getattr(ctx, "persist_queue_snapshot", None) or queue_mod.persist_queue_snapshot
    pause_id = str(checkpoint.get("pause_id") or "")
    if not task_id or not pause_id:
        raise ValueError("exact budget pause requires task_id and pause_id")
    with _queue_lock:
        meta = ctx.RUNNING.get(task_id)
        task = meta.get("task") if isinstance(meta, dict) and isinstance(meta.get("task"), dict) else None
        direct_turn = False
        if task is None and evt.get("_is_direct_chat") and isinstance(evt.get("task"), dict) \
                and str(evt["task"].get("id") or "") == task_id:
            # A direct turn ends its live actor on the way out (no second live
            # actor for this id); the queue row is minted from its own record,
            # already projected by ``pause_event`` (``parkable_direct_task``).
            task = dict(evt["task"])
            meta = {"task": task, "attempt": int(task.get("_attempt") or 1), "worker_id": None}
            direct_turn = True
        if task is None:
            raise RuntimeError(f"budget-paused task is not running: {task_id}")
        result_root = pathlib.Path(task.get("budget_drive_root") or ctx.DRIVE_ROOT)
        row = budget_pause_row(result_root, task_id)
        attempt = int(meta.get("attempt") or task.get("_attempt") or 1)
        if (row.get("pause_id") != pause_id or row.get("state") not in {STATE_PAUSING, STATE_PAUSED}
                or int(row.get("task_attempt") or 0) != attempt or not row.get("source_ref")):
            raise ValueError(f"exact budget pause {pause_id} has no live durable row for attempt {attempt}")
        if row.get("external_custody"):
            from ouroboros.sleep_custody import accepted_receipt
            if not accepted_receipt(task_id, row):
                raise ValueError("cold external custody has no exact body acceptance")
        marker = exact_pause_marker(row, default_root=str(task.get("root_task_id") or task_id))
        owner_fence = (_owner_park_fence_id(result_root, marker["root_task_id"])
                       if marker["scope"] == "root" and row.get("reason") == "owner" else "")
        if owner_fence is None and defer_unknown_owner_fence:
            raise OwnerFenceUnknown(f"current owner fence of {marker['root_task_id']} is unreadable")
        # Bind this exact park to its physical owner's fence: the parking pooled attempt/PID, or
        # the direct turn's own in-process fence (no pooled owner). On the in-memory fence it survives
        # marker reconstruction on revocation; a boot restores no fence row. No token enters cognition.
        fence = queue_mod.ACCEPTANCE_FENCES.get(marker["root_task_id"]) or {}
        worker = ctx.WORKERS.get(meta.get("worker_id"))
        owner = fence.get("pooled_owner")
        parked = {"direct_owner": {"task_attempt": attempt}} if direct_turn and "pooled_owner" not in fence else None
        if (not direct_turn and worker is not None and isinstance(owner, dict) and worker.busy_task_id == task_id
                and type(owner.get("pid")) is int and owner["pid"] == worker.proc.pid
                and type(owner.get("task_attempt")) is int and owner["task_attempt"] == attempt):
            parked = {"pooled_owner": dict(owner)}
        if parked and fence.get("status") == "active" and fence.get("task_id") == task_id:
            fence["paused_continuation"] = {**parked,
                **{key: row[key] for key in ("pause_id", "pause_generation", "source_ref")}}
        if marker["scope"] == "root" and owner_fence is None:
            standing = queue_mod.BUDGET_ROOT_FENCES.get(marker["root_task_id"])
            marker = {**marker, **({"fence_id": standing["fence_id"]} if standing else {})}
        elif marker["scope"] == "root":
            fence = _set_root_budget_pause_locked(
                marker["root_task_id"], {**marker, **({"fence_id": owner_fence} if owner_fence else {})})
            marker = {**marker, "fence_id": fence["fence_id"]}
        ctx.RUNNING.pop(task_id, None)
        paused_task = dict(task)
        paused_task["_budget_pause"] = marker
        if direct_turn:
            # The direct lane never enqueued this row, so it lacks the sequence,
            # priority and ``queued_at`` that ``enqueue_task`` would have given
            # it; the sort key and the census phase need them (queue lock held).
            counter = getattr(queue_mod, "QUEUE_SEQ_COUNTER_REF", None)
            if isinstance(counter, dict) and "_queue_seq" not in paused_task:
                counter["value"] = int(counter.get("value") or 0) + 1
                paused_task["_queue_seq"] = counter["value"]
            if "priority" not in paused_task:
                try:
                    paused_task["priority"] = queue_mod.coerce_queue_order(
                        None, queue_mod._task_priority(str(paused_task.get("type") or "task")))
                except Exception:
                    paused_task["priority"] = 1
            paused_task.setdefault("queued_at", utc_now_iso())
            paused_task["_is_direct_chat"] = True
        if not any(str(item.get("id") or "") == task_id for item in ctx.PENDING):
            ctx.PENDING.append(paused_task)
            sort_pending()
        worker_id = evt.get("worker_id") if evt else meta.get("worker_id")
        if worker_id in ctx.WORKERS and ctx.WORKERS[worker_id].busy_task_id == task_id:
            ctx.WORKERS[worker_id].busy_task_id = None
        # The confirmation, the status projection and the owner-visible event
        # stay under the SAME queue lock as the park. Resume holds that lock for
        # its whole grant, so a late park completion can no longer publish a
        # stale ``paused`` over a live grant; the state CAS below is the second
        # rail, for a writer this lock does not cover (#1196, F1).
        persisted = persist_snapshot(reason="budget_pause_exact_continuation")
        confirmed, superseded = False, ""
        from ouroboros.owner_pause import SETTLEMENT_EXTERNAL_RUNNING

        # An owner Pause whose sent delegated work still runs is saved but not
        # cleanly Paused: its row stays ``pausing`` (never a false Paused) until
        # a Resume's fresh observation finds that work settled.
        unsettled = str(row.get("settlement") or "") == SETTLEMENT_EXTERNAL_RUNNING
        if persisted and unsettled:
            log.info("Owner pause of %s parked with sent work still running; row stays 'pausing'", task_id)
        elif persisted:
            try:
                set_budget_pause(result_root, task_id, {**row, "state": STATE_PAUSED,
                                                       "paused_confirmed_at": time.time(),
                                                       "pause_source": source},
                                 expected_pause_id=pause_id,
                                 expected_state=(STATE_PAUSING, STATE_PAUSED))
                confirmed = True
            except Exception:
                log.warning("Exact budget pause row for %s was not confirmed 'paused'",
                            task_id, exc_info=True)
                # A write that failed transiently leaves the row at ``pausing`` for
                # THIS pause id — real, merely unconfirmed, and still this park's
                # to project. A row that names another pause, or is no longer
                # live, belongs to a newer writer and is never written over.
                try:
                    latest = budget_pause_row(result_root, task_id)
                except Exception:
                    superseded = "pause_record_unreadable"
                else:
                    latest_state = str(latest.get("state") or "")
                    if str(latest.get("pause_id") or "") != pause_id:
                        superseded = "pause_identity_changed"
                    elif latest_state not in {STATE_PAUSING, STATE_PAUSED}:
                        superseded = latest_state or "pause_record_missing"
        else:
            log.error("Exact budget pause for %s parked in memory but its snapshot was not persisted; "
                      "the durable row stays 'pausing' until a later snapshot confirms it", task_id)
        if not superseded:
            # A row this park no longer owns keeps whatever the newer writer
            # projected: a resumed task must never read as paused again. The
            # projection carries the ledger-derived cumulative cost planes (the
            # same ``reconstruct_task_cost`` projection every terminal write
            # takes), so the public detail of a paused task reads its rounds and
            # spend from the authority, never from a worker's pre-pause mirror.
            try:
                from supervisor.state import reconstruct_task_cost

                reason = str(row.get("reason") or "")
                cause = {"owner": "owner Pause", "authority": str(row.get("rail") or "").replace("_", " ")}.get(reason, "budget")
                cost_fields = reconstruct_task_cost(task_id, fields=True, drive_root=result_root)
                write_task_result(
                    result_root, task_id, STATUS_SCHEDULED, resource_limit=marker,
                    reason_code={"owner": "owner_paused", "authority": "authority_paused"}.get(reason, "budget_paused"),
                    result=(f"Task paused exactly at a completed boundary ({cause})"
                            + ("; delegated work it had already sent is still running" if unsettled else "")
                            + ". Cumulative spend, rounds and execution time are retained; an explicit owner "
                            "Resume continues the same task."),
                    # An unavailable projection is published too: it carries the
                    # explicit unknown state that must replace any stale amount
                    # write_task_result would otherwise keep merged in.
                    **cost_fields,
                )
            except Exception:
                log.warning("Failed to persist exact budget pause status for %s", task_id, exc_info=True)
        event = {
            "ts": (evt or {}).get("ts", utc_now_iso()),
            "type": "budget_scope_paused",
            "task_id": task_id,
            "task_type": (evt or {}).get("task_type") or task.get("type"),
            "owner_visible": True,
            "toast_once": f"{task_id}:budget-paused:{pause_id}",
            "pause_source": source,
            "park_state": STATE_PAUSED if confirmed else STATE_PAUSING,
            **({"pause_reason": str(row["reason"])} if row.get("reason") in {"owner", "authority"} else {}),
            **{key: value for key, value in marker.items() if key != "checkpoint"},
            "pause_id": pause_id,
            "external_runs": [
                {k: run.get(k) for k in ("run_id", "state", "stop_outcome")}
                for run in ((row.get("external_runs") or {}).get("runs") or []) if isinstance(run, dict)
            ],
        }
        if superseded:
            # Not an owner-visible pause: the durable row moved on (a grant, a
            # newer pause, an abandoned row). The anomaly is recorded as itself.
            event = {key: value for key, value in event.items() if key != "toast_once"}
            event.update(type="budget_pause_park_superseded", owner_visible=False,
                         park_state=superseded)
        _address_task_event({task_id: meta} if isinstance(meta, dict) else None, ctx.DRIVE_ROOT, event)
        append_jsonl(ctx.DRIVE_ROOT / "logs" / "events.jsonl", event)
        try:
            bridge = getattr(ctx, "bridge", None)
            if bridge is not None:
                bridge.push_log(event)
        except Exception:
            log.warning("Failed to forward exact budget pause to Activity", exc_info=True)
    if str(row.get("reason") or "") == "owner":
        from supervisor.owner_pause_control import refresh_owner_pause_tree

        refresh_owner_pause_tree(str(task.get("root_task_id") or task_id))
    return marker


def _handle_budget_resume_child(evt: Dict[str, Any], ctx: Any) -> None:
    """Owner Q9: a resumed root's model SELECTS one budget-paused child to continue.

    The requester must be the live parent/root of the target (its tree, not any
    tree): a root selects any stored descendant, an intermediate parent only its
    direct children. The grant itself goes through the ONE resume seam, so every typed
    refusal (money, cancel intent, deadline, lifetime, root still paused) is the
    same the owner would receive. The outcome is recorded as an event the
    requesting task can read back; nothing is auto-fanned-out.
    """
    from supervisor.queue import _queue_lock
    from supervisor.queue_transitions import resume_budget_paused_task

    task_id = str(evt.get("task_id") or "").strip()
    requester = str(evt.get("requested_by") or "").strip()
    with _queue_lock:
        target = next((row for row in ctx.PENDING if str(row.get("id") or "") == task_id), None)
        target = target or (getattr(ctx, "RUNNING", {}).get(task_id) or {}).get("task")
        lineage_ok = bool(
            target is not None and requester
            and requester in (str(target.get("parent_task_id") or ""), str(target.get("root_task_id") or ""))
        )
    if not lineage_ok:
        outcome: Dict[str, Any] = {"ok": False, "error": "not_a_budget_paused_descendant"}
    else:
        # A MODEL-issued selection: the grant additionally requires the live
        # owner-derived Resume grant of the target's own root (never lineage alone).
        from ouroboros.control_actor import mind_control
        with mind_control(requester, str(evt.get("reason") or "")):
            outcome = resume_budget_paused_task(task_id, selected_by=requester)
    event = {
        "ts": evt.get("ts", utc_now_iso()),
        "type": "budget_resume_child_outcome",
        "task_id": task_id,
        "requested_by": requester,
        "reason": str(evt.get("reason") or "")[:500],
        **{key: value for key, value in outcome.items() if key != "task_id"},
    }
    _address_ctx(ctx, event)
    append_jsonl(ctx.DRIVE_ROOT / "logs" / "events.jsonl", event)
    try:
        ctx.bridge.push_log(event)
    except Exception:
        log.debug("Failed to forward child resume outcome to Activity", exc_info=True)


def _handle_budget_root_fence(evt: Dict[str, Any], ctx: Any) -> None:
    """Latch one root after a refused dispatch; never reconcile its subtree."""
    task_id = str(evt.get("task_id") or "").strip()
    supplied = evt.get("resource_limit") if isinstance(evt.get("resource_limit"), dict) else {}
    root_task_id = str(supplied.get("root_task_id") or evt.get("root_task_id") or "").strip()
    if not task_id or not root_task_id or str(supplied.get("scope") or "") != "root":
        raise ValueError("root budget fence requires task_id, root_task_id, and root scope")

    from supervisor.queue import _queue_lock
    with _queue_lock:
        fence = _set_root_budget_pause_locked(root_task_id, supplied)
        ctx.persist_queue_snapshot(reason="budget_root_fenced")
    event = {
        "ts": evt.get("ts", utc_now_iso()),
        "type": "budget_scope_paused",
        "task_id": task_id,
        "task_type": evt.get("task_type"),
        "owner_visible": True,
        "toast_once": f"{root_task_id}:budget-paused:root",
        **fence,
    }
    _address_ctx(ctx, event)
    append_jsonl(ctx.DRIVE_ROOT / "logs" / "events.jsonl", event)
    try:
        ctx.bridge.push_log(event)
    except Exception:
        log.warning("Failed to forward root budget pause to Activity", exc_info=True)


# --- budget admission HOLDS (#1196) --------------------------------------------
#
# The root fence above keeps a paused root's descendants out of the queue. When
# that root is resumed the fence is LIFTED, which must not by itself make its
# zero-dispatch siblings assignable: each one takes the durable hold below until
# the model selects it through the same resume control (owner Q9). The same
# marker carries a row whose exact continuation could not be restored and one
# whose spent grant could not be revoked, so an un-dispatchable row is always a
# typed, visible fact instead of a dropped or silently runnable task.

# The typed hold reasons (one vocabulary for the queue row, the task result and
# the events log). A hold beside a retained ``_budget_pause`` marker is released
# by a successful exact grant (the grant re-validates the durable authority);
# a hold on a marker-less row is released only by an explicit selection.
HOLD_ROOT_FENCE_LIFTED = "root_fence_lifted_pending_selection"
# One MEMBER of a tree whose root latch is still up: the latch belongs to the
# root, so releasing this row must not lift it for every sibling (owner Q9).
# The selection recorded on the row names the fence it was granted against.
HOLD_ROOT_FENCE_MEMBER_SELECTION = "root_fence_member_pending_selection"
HOLD_RESTORE_REFUSED_PREFIX = "restore_refused:"
HOLD_ROOT_ACCEPTANCE_FENCED = "root_acceptance_fenced_at_restore"
HOLD_REVOCATION_UNWRITTEN = "resume_grant_revocation_unwritten"
HOLD_RECORD_UNREADABLE_AT_REVOKE = "pause_record_unreadable_at_revoke"
HOLD_STALE_GRANT_SUPERSEDED = "stale_resume_grant_superseded"
HOLD_MALFORMED_RESUME_IDENTITY = "malformed_resume_identity"
HOLD_RESTART_REVOCATION_UNWRITTEN = "restart_revocation_unwritten"
# A queue row still carrying a grant handoff whose grant the durable row says
# was CONSUMED: the task ran on. The row is stale, never re-armed as a pause
# and never dispatched; a restart fences it as the running work it names.
HOLD_GRANT_CONSUMED_STALE_ROW = "stale_queue_row_grant_consumed"
# A never-started row the owner's manual Restart caught (owner Batch4 7A): the
# same queued task keeps its id and waits for an explicit Resume instead of
# being cancelled (``supervisor/restart_retention.py``). Released exactly like a
# fence-lifted sibling: the owner's Resume, or a model selection under its
# root's live Resume grant.
HOLD_OWNER_RESTART = "owner_restart_hold"
# The owner's Continue admitted while the interrupted task's own writers were
# not proven settled (``supervisor/continuation_admission.py``).
HOLD_CONTINUATION_WRITER = "continuation_writer_unsettled"
# A model's cold sleep caught by a Panic: its readiness never wakes it; only an
# explicit Resume after the owner's next launch does (a Restart holds it the
# same way under ``owner_restart_hold``).
HOLD_PANIC = "panic_hold"
# Quit, crash, Panic or an unacknowledged restart: explicit same-ID Resume (restart_retention).
HOLD_SAVED_WORK = "saved_work_hold"
# An owner Pause seed whose attempt ended with no usable saved state (an inline actor that ended
# while pausing, or the retained timeout-retry consumer): pause intent, never an exact pause; the
# same id waits for explicit Resume (worker_health, #1729).
HOLD_UNSAVED_PAUSE = "unsaved_pause_intent"
# Malformed acceptance-fence evidence in the snapshot fails the restore closed
# for ordinary rows; a saved exact pause is retained under this hold instead.
HOLD_INVALID_ACCEPTANCE_FENCE_SNAPSHOT = HOLD_RESTORE_REFUSED_PREFIX + "invalid_acceptance_fence_snapshot"
HOLD_INVALID_BUDGET_FENCE_SNAPSHOT = HOLD_RESTORE_REFUSED_PREFIX + "invalid_budget_fence_snapshot"


def native_continuation_repair(task: Dict[str, Any]) -> bool:
    """Revalidate native wait custody by source kind, independent of its refusal text."""
    from supervisor.task_admission import governing_continuation

    return bool(continuation_refusal(task) and governing_continuation(task) == "owner_wait"
                and isinstance(task.get("_owner_wait_resume"), dict))


def budget_resume_dispatch_allowed(q: Any, task: Dict[str, Any]) -> Optional[bool]:
    """A child selection belongs to the CURRENT root grant and fence only.

    Both carriers are revalidated at dispatch: an exact grant handoff and a
    zero-dispatch selection alike name the root grant they were selected
    under, so a root that is pausing or paused again (with or without a new
    fence) admits neither on its old selection (#1196, owner Q9).
    ``None`` holds this dispatch on unreadable retry/root-grant authority;
    ``False`` retains stale-selection handling. Resolve the physical root once.
    """
    if continuation_refusal(task):
        return False
    handoff = task.get("_budget_pause_resume")
    selection = task.get(SELECTION_KEY)
    exact = isinstance(handoff, dict)
    carrier = handoff if exact else (selection if isinstance(selection, dict) else None)
    if carrier is None:
        return True
    root_id = str(task.get("root_task_id") or task.get("id") or "")
    fence_id = str((q.BUDGET_ROOT_FENCES.get(root_id) or {}).get("fence_id") or "")
    if fence_id != str(carrier.get("root_fence_id" if exact else "fence_id") or ""):
        return False
    if carrier.get("authority") == "sleep_readiness":
        from ouroboros.budget_pause import budget_pause_row
        from ouroboros.owner_pause import member_fence
        from types import SimpleNamespace

        result_root = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
        try:
            row = budget_pause_row(result_root, str(task.get("id") or ""))
            grant = row.get("grant") or {}
            return bool(row.get("reason") == "sleep" and row.get("sleep_ready")
                        and grant.get("authority") == "sleep_readiness"
                        and grant.get("grant_id") == carrier.get("grant_id")
                        and grant.get("sleep_id") == carrier.get("sleep_id")
                        and not grant.get("revoked_at") and not grant.get("consumed_at")
                        and budget_hold_fact(task) is None
                        and not member_fence(SimpleNamespace(task_id=task.get("id"),
                            root_task_id=root_id, budget_drive_root=result_root))
                        and not any(str(t.get("id") or "") == root_id and t.get("_budget_pause")
                                    for t in q.PENDING))
        except Exception:
            return False
    if carrier.get("selected_by") == "sleep_wake":
        return False
    if root_id == str(task.get("id") or ""):
        return True
    from ouroboros.cancel_intents import _validated_single_cancel_target

    result_root = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
    try:
        actor_id = _validated_single_cancel_target(result_root, root_id)
        owner_unbound = carrier.get('selected_by') == 'owner' and not carrier.get('root_grant_id')
        if owner_unbound and actor_id == str(task.get('id') or ''):
            return True
        state = _root_resume_state(q, root_id, result_root, actor_id=actor_id)
        live, durable = _root_resume_grants(state)
    except Exception:
        log.debug('Resume dispatch authority unreadable for %s', root_id, exc_info=True)
        return None

    def matches(grant):
        return bool(grant and carrier.get('root_grant_id') == grant['grant_id']
                    and int(carrier.get('root_resume_generation') or 0) == int(grant['generation']))

    from ouroboros.budget_pause import STATE_PAUSED, STATE_PAUSING
    from ouroboros.owner_pause import fence_closed

    owner_unbound = (
        carrier.get("selected_by") == "owner"
        and not carrier.get("root_grant_id")
    )
    current_root = state["task"]
    if not owner_unbound and (
        isinstance(current_root.get("_budget_pause"), dict)
        or budget_hold_fact(current_root) is not None
        or current_root.get("_owner_hold")
        or state["pause"].get("state") in {STATE_PAUSED, STATE_PAUSING}
        or fence_closed(state["owner_fence"])
    ):
        return False

    if 'root_resume_basis' in carrier:
        basis = carrier['root_resume_basis']
        if owner_unbound and fence_id and isinstance(basis, dict):
            # This owner choice borrowed no root grant. The root parking on
            # the same monetary latch does not supersede it; actor and owner
            # fence identities (and the latch check above) still bind it.
            basis = {**basis, 'budget': state['basis']['budget']}
        if basis != state['basis']:
            return False
        # A current queue selection still supersedes an older choice. With no
        # lender, unchanged durable identities preserve the choice already made.
        return not live or matches(live)

    # Legacy rows: accept matching evidence and the no-history case. Conflicting
    # history takes the existing selectable-hold path, never a timestamp guess.
    if live:
        return matches(live)
    if durable:
        return matches(durable)
    if owner_unbound:
        return True
    return not state['pause'] and not state['owner_fence'] and budget_hold_fact(state['task']) is None


def hold_budget_row(task: Dict[str, Any], *, reason: str, detail: str = "",
                    extra: Optional[Dict[str, Any]] = None,
                    result_root: Optional[pathlib.Path] = None, existing_result_only: bool = False) -> Dict[str, Any]:
    """Hold one queued row: typed, visible, never dropped and never cancelled.

    Keep the PENDING identity, remove stale resume grants and project the same
    reason to the result. Restore can require valid existing result authority.
    """
    from ouroboros.control_actor import control_actor

    hold = stop_record(str(reason), held_at=utc_now_iso(), detail=detail, by=control_actor(), binding=extra)
    declared = declare_stop(task, hold)
    if declared is not hold:
        return declared
    policy = reason_policy(str(reason))
    if result_root is not None:
        try:
            write_task_result(
                result_root, str(task.get("id") or ""), STATUS_SCHEDULED,
                reason_code=policy["ui_code"],
                resource_limit={"status": "budget_hold", "auto_resume": False, "exact_continuation": False,
                                "resume_policy": "explicit_selection_same_seam", **hold},
                # Restoring accepted work must not manufacture a scheduled
                # receipt over missing/corrupt authority. Keep the queue hold.
                _field_projector=(lambda current, incoming: incoming if current else None)
                    if existing_result_only else None,
                strict_existing_dict=existing_result_only,
            )
        except Exception:
            log.debug("Budget hold projection failed for %s", task.get("id"), exc_info=True)
    return hold


def hold_restored_budget_pause(task: Dict[str, Any], drive_root: Any, *, reason: str,
                               detail: str = "") -> Dict[str, Any]:
    """Restore-time hold for a paused row the restart could not clear for dispatch.

    A corrupt, missing or refused checkpoint source, or an acceptance fence over
    the root, does not drop or cancel the task: the row stays PENDING and
    un-dispatchable with a typed reason, and it KEEPS its ``_budget_pause``
    marker — the marker is the locator the owner's Resume validates against the
    durable authority. A later Resume re-reads that authority: a source that is
    readable again is granted (releasing this hold), one that is not refuses
    typed and leaves the saved pause where it is.
    """
    prior = task.get("_budget_pause") if isinstance(task.get("_budget_pause"), dict) else {}
    hold_budget_row(
        task, reason=reason,
        detail=detail or "the exact budget-pause continuation could not be cleared for dispatch after a restart",
        extra={"pause_id": str((prior.get("checkpoint") or {}).get("pause_id") or ""),
               "root_task_id": str(prior.get("root_task_id") or task.get("root_task_id") or "")},
        result_root=pathlib.Path(task.get("budget_drive_root") or drive_root),
    )
    return task


def hold_root_resume_descendants(q: Any, root_id: str, fence: dict, grant: dict, *, actor_id: str = "") -> tuple:
    """Replace lifted-fence eligibility with per-child holds; retain exact pauses.

    A fence-derived marker is not a checkpoint: convert it to a selection hold
    before removing its fence, or it would be stranded. Older exact child grants
    are revoked and selected again under this root grant. A sibling still held
    from an EARLIER Resume of this root (pauseA -> Resume -> pauseB -> Resume) is
    re-bound to the live grant in the same pass, so it stays selectable instead
    of refused forever as stale. Return ``(held, markers, rebound)``: the hold and
    marker changes and the prior re-bound holds, for the Resume transaction's
    snapshot rollback; revocations stay safe. ``actor_id``: the retry attempt
    resuming as the root (``owner_pause.root_actor``) is that root, never a member.
    """
    from supervisor.budget_resume import revoke_exact_budget_resume

    held, markers, rebound = [], {}, {}
    for member in q.PENDING:
        member_id = str(member.get("id") or "")
        if (not member_id or member_id in {root_id, actor_id}
                or str(member.get("root_task_id") or "") != root_id):
            continue
        if isinstance(member.get("_budget_pause_resume"), dict):
            revoke_exact_budget_resume(member, "root_resume_generation_changed")
        pause = member.get("_budget_pause")
        fence_derived = bool(isinstance(pause, dict) and not pause.get("exact_continuation")
                             and pause.get("scope") == "root" and pause.get("root_task_id") == root_id
                             and pause.get("fence_id") == fence.get("fence_id"))
        hold = member.get(BUDGET_HOLD_KEY) if isinstance(member.get(BUDGET_HOLD_KEY), dict) else None
        if (hold is not None and record_policy(hold)["cause"] == "awaiting_selection"
                and str(hold.get("root_grant_id") or "") != grant["grant_id"]):
            rebound[member_id] = dict(hold)
            member[BUDGET_HOLD_KEY] = {**hold, "root_grant_id": grant["grant_id"],
                                       "root_resume_generation": int(grant["generation"])}
            # The latch was lifted; carrying its old id would make the newly
            # selected child fail the dispatch fence comparison. Rollback above
            # retains the original hold, including that id.
            member[BUDGET_HOLD_KEY].pop("fence_id", None)
        # Existing choices survive this transaction; dispatch revalidates their old fence.
        if ((pause is not None and not fence_derived) or hold is not None
                or isinstance(member.get(SELECTION_KEY), dict)):
            continue
        if fence_derived:
            markers[member_id] = member.pop("_budget_pause")
        hold_budget_row(
            member, reason=HOLD_ROOT_FENCE_LIFTED,
            detail="root resumed; this zero-dispatch sibling awaits explicit selection",
            extra={"root_task_id": root_id, "root_grant_id": grant["grant_id"],
                   "root_resume_generation": grant["generation"],
                   **({"replaced_fence_pause": markers[member_id]}
                      if fence_derived else {})},
            result_root=pathlib.Path(member.get("budget_drive_root") or q.DRIVE_ROOT))
        held.append(member_id)
    return held, markers, rebound


def _held_selection_authority(q: Any, task: Dict[str, Any]) -> Dict[str, Any]:
    """Local selection identity under the queue lock; no transport or ledger I/O."""
    from types import SimpleNamespace
    from ouroboros.cancel_intents import cancel_pending
    from ouroboros.owner_pause import consumed_member_resume, member_fence, root_actor

    task_id = str(task.get("id") or "")
    root_id = str(task.get("root_task_id") or task_id)
    result_root = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
    predecessor = str(((task.get("metadata") or {}).get("continuation") or {}).get("predecessor_task_id") or "")
    roots = {root_id, predecessor} - {""}
    actor = root_actor(result_root, root_id, task_id, strict=True)
    root_state = _root_resume_state(q, root_id, result_root,
                                    actor_id=task_id if actor else None)
    return copy.deepcopy({
        "task": task, "root_actor": actor,
        "fences": {root: q.BUDGET_ROOT_FENCES.get(root) for root in roots},
        "pending": [row for row in q.PENDING if str(row.get("root_task_id") or row.get("id") or "") in roots],
        "running": {tid: meta.get("task") for tid, meta in q.RUNNING.items()
                    if str((meta.get("task") or {}).get("root_task_id") or tid) in roots},
        'root_grant': _root_resume_grants(root_state)[0] if not actor else {},
        'root_resume_basis': root_state['basis'],
        "owner_fence": member_fence(SimpleNamespace(task_id=task_id, root_task_id=root_id,
                                                     drive_root=result_root)),
        "consumed_resume": consumed_member_resume(result_root, root_id, task_id, attempt=int(
            task["_working_recovery"].get("from_attempt") or 0)) if task.get("_working_recovery") else {},
        "cancel_pending": any(cancel_pending(result_root, tid, strict=True) for tid in {task_id, root_id}),
        "owner_restart": owner_restart_in_progress(q.DRIVE_ROOT),
    })


def observe_held_budget_selection(q: Any, task_id: str) -> Dict[str, Any]:
    """Snapshot identity, then observe writer custody and money OFF the queue lock.

    The caller must not own the queue lock. Selection compares the snapshot
    after re-locating the exact row, so a newer hold/Stop/Pause or root grant
    invalidates this observation rather than inheriting its permission.
    """
    from supervisor.queue_transitions import pending_member_replay_safe
    from supervisor.continuation_admission import CONTINUE_PURPOSE, action_writers

    try:
        with q._queue_lock:
            task = next((row for row in q.PENDING if str(row.get("id") or "") == task_id), None)
            if task is None:
                return {"error": "task_not_pending"}
            authority = _held_selection_authority(q, task)
        candidate = authority["task"]
        predecessor = str(((candidate.get("metadata") or {}).get("continuation") or {}).get("predecessor_task_id") or "")
        blockers = action_writers(q, predecessor, purpose=CONTINUE_PURPOSE) if predecessor else []
        if native_continuation_repair(candidate):
            from ouroboros.owner_wait import restore_owner_wait_allowed

            safe, error = False, "owner_held"
            if not candidate.get("_owner_hold"):
                root = pathlib.Path(candidate.get("budget_drive_root") or q.DRIVE_ROOT)
                safe = restore_owner_wait_allowed(root, candidate, strict=True)
                error = "" if safe else "dispatch_outcome_unknown"
        else:
            safe, error = pending_member_replay_safe(q, candidate)
        return {"candidate": task, "authority": authority, "blockers": blockers,
                "safe": safe, "unsafe_error": error}
    except Exception:
        log.warning("Held selection observation unavailable for %s", task_id, exc_info=True)
        return {"error": "selection_observation_unavailable"}


def select_held_budget_row(q: Any, task: Dict[str, Any], hold: Dict[str, Any],
                           *, selected_by: str, observation: Dict[str, Any]) -> Dict[str, Any]:
    """Record an explicit selection on a held row (queue lock held).

    Release requires current saved-continuation or never-dispatched evidence.
    A model-issued selection additionally needs the live
    owner-derived Resume grant of its root: lifting a fence granted eligibility,
    never continuation.
    """
    task_id = str(task.get("id") or "")
    result_root = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
    root_task_id = str(hold.get("root_task_id") or task.get("root_task_id") or task_id)
    if observation.get("error"):
        return {"ok": False, "error": observation["error"]}
    try:
        current = _held_selection_authority(q, task)
    except Exception:
        return {"ok": False, "error": "selection_authority_unavailable"}
    if (task is not observation.get("candidate") or not any(row is task for row in q.PENDING)
            or current != observation.get("authority")
            or selection_record(task) != hold):
        return {"ok": False, "error": "selection_authority_changed"}
    from ouroboros.control_actor import control_actor, observed_control_stop
    if refusal := release_refusal(observed_control_stop(
            result_root, task, budget=q.BUDGET_ROOT_FENCES, owner=current["owner_fence"]), control_actor()):
        return refusal
    native_read_gap = native_continuation_repair(task)
    # This native repair cannot complete a shadow working carrier's root Resume.
    completing = {} if selected_by or native_read_gap else current["consumed_resume"]
    if current["cancel_pending"] or current["owner_fence"] and not completing or current["owner_restart"]:
        return {"ok": False, "error": "selection_owner_held"}
    continuation = ((task.get("metadata") or {}).get("continuation") or {}) if isinstance(
        task.get("metadata"), dict) else {}
    if continuation.get("predecessor_task_id"):
        # A Continue never releases over its predecessor's unsettled writers,
        # whichever hold (its own, or a later Restart's) is being released.
        blockers = observation["blockers"]
        if blockers:
            from supervisor.continuation_admission import writers_unsettled_refusal

            return writers_unsettled_refusal(blockers)
    # Use the identity already revalidated with this selection's snapshot.
    # A reciprocal retry is the root actor even though its physical id differs.
    actor = current["root_actor"]
    root_grant = current["root_grant"]
    if selected_by and not actor:
        if not root_grant:
            return {"ok": False, "error": "root_resume_grant_missing",
                    "root_task_id": root_task_id, "action": "resume_root_first"}
        if (hold.get("root_grant_id") and str(hold["root_grant_id"]) != root_grant["grant_id"]):
            return {"ok": False, "error": "root_resume_generation_stale",
                    "root_task_id": root_task_id, "action": "resume_root_first"}
    safe, unsafe_error = observation["safe"], observation["unsafe_error"]
    if not safe and record_policy(hold).get("needs_source") and "_working_recovery" not in task:
        # Its attempt saved nothing: the fact is a missing continuation, not an unknown dispatch (#1729).
        from ouroboros.budget_pause import RESTORE_REFUSAL_SOURCE_MISSING

        return {"ok": False, "error": RESTORE_REFUSAL_SOURCE_MISSING, "action": "cancel_or_new_run",
                "detail": "stopped while pausing, before anything was saved: there is nothing to continue"}
    if not safe:
        return {"ok": False, "error": unsafe_error, "action": "cancel_or_new_run"}
    selection = resume_selection(hold, by=control_actor(), selected_at=utc_now_iso(),
                                 selected_by=str(selected_by or "owner"))
    if not native_read_gap:
        selection['root_resume_basis'] = current['root_resume_basis']
        # A new owner choice does not inherit an obsolete upstream grant.
        selection.pop('root_grant_id', None)
        selection.pop('root_resume_generation', None)
        latch = q.BUDGET_ROOT_FENCES.get(root_task_id)
        selection["fence_id"] = str((latch or {}).get("fence_id") or "")
        if actor:
            selection.update(root_grant_id=uuid.uuid4().hex, root_resume_generation=1)
        elif root_grant:
            selection.update(root_grant_id=root_grant["grant_id"],
                             root_resume_generation=int(root_grant["generation"]))
    prior_points = {key: copy.deepcopy(task[key]) for key in (
        "_budget_pause", "_owner_wait_resume", "_working_recovery", SELECTION_KEY, BUDGET_HOLD_KEY) if key in task}
    task.pop("_budget_pause", None)

    rebound, invalidated, exact_rebound = [], [], []
    if not native_read_gap and actor and not q.BUDGET_ROOT_FENCES.get(root_task_id):
        from supervisor.budget_resume import revoke_exact_budget_resume

        for member in q.PENDING:
            if (member is task or member.get('id') == root_task_id
                    or member.get('root_task_id') != root_task_id):
                continue
            handoff = member.get('_budget_pause_resume')
            if isinstance(handoff, dict):
                if (handoff.get('authority') == 'explicit_resume'
                        and handoff.get('selected_by') == 'owner'
                        and not handoff.get('root_grant_id')
                        and not handoff.get('root_fence_id')
                        and not member.get('_budget_pause')
                        and budget_hold_fact(member) is None
                        and ('root_resume_basis' not in handoff
                             or handoff['root_resume_basis'] == current['root_resume_basis'])):
                    # Preserve the owner's existing choice, as for an unbound
                    # selection below. Only its queue dispatch binding
                    # changes; the exact single-use grant is not reissued.
                    exact_rebound.append((member, handoff))
                    continue
                # As at fenced root Resume, revocation remains safe even if
                # publication fails: an unused continuation is never replayed.
                revoke_exact_budget_resume(member, 'root_resume_generation_changed')
            if member.get('_budget_pause_resume'):
                continue
            held = budget_hold_fact(member)
            if held is not None:
                if record_policy(held)["cause"] == "awaiting_selection" and held.get('root_grant_id'):
                    rebound.append((member, BUDGET_HOLD_KEY, held))
                continue
            prior = member.get(SELECTION_KEY)
            if isinstance(prior, dict):
                if prior.get('selected_by') == 'owner' and not prior.get('root_grant_id') and not prior.get('fence_id'):
                    rebound.append((member, SELECTION_KEY, prior))
                else:
                    invalidated.append((member, prior))

    def withdraw(persist: bool = True) -> None:
        for key in ("_budget_pause", "_owner_wait_resume", "_working_recovery", SELECTION_KEY, BUDGET_HOLD_KEY):
            task.pop(key, None)
        task.update(copy.deepcopy(prior_points))
        for member, key, prior in rebound:
            member[key] = prior
        for member, prior in invalidated:
            member.pop(BUDGET_HOLD_KEY, None)
            member[SELECTION_KEY] = prior
        for member, prior in exact_rebound:
            member['_budget_pause_resume'] = prior
        if persist:
            q.persist_queue_snapshot(reason='budget_hold_selection_withdrawn')

    def publish() -> bool:
        clear_refusal(task)
        task.pop(BUDGET_HOLD_KEY, None)
        if not native_read_gap:  # A native repair rejoins its wait without creating a money or child grant.
            task[SELECTION_KEY] = selection
        for member, prior in exact_rebound:
            member['_budget_pause_resume'] = {
                **prior, 'root_grant_id': selection['root_grant_id'],
                'root_resume_generation': selection['root_resume_generation'],
                'root_resume_basis': current['root_resume_basis'],
            }
        for member, key, prior in rebound:
            member[key] = {
                **prior, 'root_grant_id': selection['root_grant_id'],
                'root_resume_generation': selection['root_resume_generation'],
                'root_resume_basis': current['root_resume_basis'],
            }
            member[key].pop('fence_id', None)
        for member, _prior in invalidated:
            hold_budget_row(
                member, reason=HOLD_ROOT_FENCE_LIFTED,
                detail='root_resume_generation_changed',
                extra={'root_task_id': root_task_id, 'root_grant_id': selection['root_grant_id'],
                       'root_resume_generation': selection['root_resume_generation']})
        return bool(q.persist_queue_snapshot(reason='budget_hold_selected'))

    from ouroboros.owner_pause import complete_consumed_member_resume
    refused = (complete_consumed_member_resume(result_root, root_task_id, task_id, completing, publish, withdraw)
               if completing else "" if publish() else "snapshot_not_persisted")
    if refused:
        withdraw(persist=False)
        return {"ok": False, "error": "snapshot_not_persisted" if refused == "snapshot_not_persisted"
                else "selection_owner_held", **({"detail": refused} if completing else {})}
    try:
        write_task_result(result_root, task_id, STATUS_SCHEDULED, reason_code="",
                          resource_limit={"status": "resumed", "auto_resume": False,
                                          "exact_continuation": False, **selection})
    except Exception:
        log.debug("Failed to project held-row selection for %s", task_id, exc_info=True)
    q.append_jsonl(
        q.DRIVE_ROOT / "logs" / "events.jsonl",
        {"ts": utc_now_iso(), "type": "budget_hold_selected", "task_id": task_id,
         "root_task_id": root_task_id, "selected_by": selection["selected_by"],
         "hold_reason": str(hold.get("reason") or "")},
    )
    return {"ok": True, "task_id": task_id, "root_task_id": root_task_id,
            "selection": "budget_hold_released", "same_generation": True}


def _root_resume_state(q: Any, root_task_id: str, result_root: pathlib.Path, *,
                       actor_id: Optional[str] = None) -> Dict[str, Any]:
    """Read the current physical actor and the two existing durable authorities.

    The basis contains control identities, not lifecycle progress or clocks.
    Read failures propagate; callers must not turn unknown into absent authority.
    """
    from ouroboros.budget_pause import budget_pause_row
    from ouroboros.cancel_intents import _validated_single_cancel_target
    from ouroboros.owner_pause import read_fence

    actor = (actor_id if actor_id is not None else
             _validated_single_cancel_target(result_root, root_task_id))
    rows = [meta['task'] for meta in q.RUNNING.values()
            if isinstance(meta, dict) and isinstance(meta.get('task'), dict)
            and str(meta['task'].get('id') or '') == actor]
    rows += [row for row in q.PENDING
             if isinstance(row, dict) and str(row.get('id') or '') == actor]
    from supervisor.workers import direct_chat_turn

    direct = direct_chat_turn(actor) if not rows else None
    current = next(iter(rows), direct or {})
    root_drive = next((row.get('budget_drive_root') for row in rows
                       if row.get('budget_drive_root')), result_root)
    pause = budget_pause_row(pathlib.Path(root_drive), actor)
    owner = read_fence(pathlib.Path(root_drive), root_task_id)
    return {'task': current, 'pause': pause, 'owner_fence': owner,
            'latch': q.BUDGET_ROOT_FENCES.get(root_task_id), 'basis': resume_basis(actor, pause, owner)}


def resume_basis(actor: str, pause: Dict[str, Any], owner: Dict[str, Any]) -> Dict[str, Any]:
    """The root's control identities a child selection binds to: its actor, money pause and owner fence
    generations and grants. ``work_stop`` reads the same basis into each held child's stop identity."""
    grant = pause.get('grant') or {}
    warm = owner.get('resume_grant') or {}
    return {
        'actor_id': actor,
        'budget': {
            'pause_id': str(pause.get('pause_id') or ''),
            'pause_generation': int(pause.get('pause_generation') or 0),
            'resume_generation': int(pause.get('resume_generation') or 0),
            'grant_id': str(grant.get('grant_id') or ''),
            'grant_generation': int(grant.get('generation') or 0),
            'revoked': bool(grant.get('revoked_at')),
        },
        'owner': {
            'fence_id': str(owner.get('fence_id') or ''),
            'generation': int(owner.get('generation') or 0),
            'grant_id': str(warm.get('grant_id') or ''),
            'grant_generation': int(warm.get('generation') or 0),
            'revoked': bool(warm.get('revoked_at')),
        },
    }


def _root_resume_grants(state: Dict[str, Any]) -> tuple:
    """Return (lendable grant, comparable durable grant).

    Consumed history can validate a prior selection, not lend on its own.
    Current direct or pooled actors retain warm lending; unconsumed exact
    grants are durable issuance authority. Cross-authority observations
    resolve exact/warm order without wall clocks.
    """
    from ouroboros.budget_pause import (
        STATE_PAUSED, STATE_PAUSING, STATE_RESUME_GRANTED, STATE_RESUMED,
    )
    from ouroboros.owner_pause import fence_closed

    current, row = state['task'], state['pause']
    owner, basis = state['owner_fence'], state['basis']
    selection = current.get(SELECTION_KEY) or {}
    grant = row.get('grant') or {}
    warm = owner.get('resume_grant') or {}
    if (budget_hold_fact(current) is not None or continuation_refusal(current) or current.get('_owner_hold')
            or fence_closed(owner) or row.get('state') in {STATE_PAUSED, STATE_PAUSING}):
        return {}, {}
    latch = state['latch']
    if latch:
        if budget_fence_selected(current, latch) and selection.get('root_grant_id'):
            return {'grant_id': str(selection['root_grant_id']),
                    'generation': int(selection['root_resume_generation']), 'pause_id': ''}, {}
        return {}, {}
    if any(current.get(key) for key in (
            '_budget_pause', '_continuation_prepared')):
        return {}, {}

    exact_basis = grant.get('root_resume_basis') or {}
    warm_basis = warm.get('root_resume_basis') or {}
    exact_ordered = bool(
        not owner
        or exact_basis.get('actor_id') == basis['actor_id']
        and exact_basis.get('owner') == basis['owner']
        or grant.get('owner_pause_fence_id')
        and grant.get('owner_pause_fence_id') == owner.get('fence_id')
        and not warm)
    warm_ordered = bool(
        not row
        or warm_basis.get('actor_id') == basis['actor_id']
        and warm_basis.get('budget') == basis['budget'])
    exact = ({'grant_id': str(grant['grant_id']),
              'generation': int(row.get('resume_generation') or 0),
              'pause_id': str(row['pause_id'])}
             if row.get('state') in {STATE_RESUME_GRANTED, STATE_RESUMED}
             and row.get('pause_id') and grant.get('grant_id')
             and not grant.get('revoked_at') and exact_ordered else {})
    warmed = ({'grant_id': str(warm['grant_id']),
               'generation': int(warm.get('generation') or 0), 'pause_id': ''}
              if warm.get('grant_id') and not warm.get('revoked_at')
              and warm_ordered else {})
    durable = {} if exact and warmed else exact or warmed
    handoff = current.get('_budget_pause_resume') or {}
    if exact and handoff.get('grant_id') == exact['grant_id']:
        return exact, durable
    if (selection and selection.get('selected_by') == 'owner' and not selection.get('fence_id')
            and not handoff):
        if selection.get('grant_id'):
            if (exact and row.get('state') == STATE_RESUMED and grant.get('consumed_at')
                    and grant.get('authority') == 'explicit_resume'
                    and selection['grant_id'] == grant.get('grant_id')
                    and selection.get('grant_generation') == grant.get('generation')):
                return exact, durable
        elif (selection.get('root_grant_id') and int(selection.get('root_resume_generation') or 0) > 0
              and (selection.get('root_resume_basis') == basis
                   or 'root_resume_basis' not in selection and not row and not owner)):
            return {'grant_id': str(selection['root_grant_id']),
                    'generation': int(selection['root_resume_generation']), 'pause_id': ''}, durable

    if warmed and (current or warm.get('tree_only')):
        # A terminal tree's explicit Resume lends only to individually selected children.
        return warmed, durable
    if exact and row.get('state') == STATE_RESUME_GRANTED and not grant.get('consumed_at'):
        return exact, durable
    return {}, durable


def live_root_resume_grant(q: Any, root_task_id: str, result_root: pathlib.Path, *,
                           actor_id: Optional[str] = None, strict: bool = False) -> Dict[str, Any]:
    """Current root authority for a NEW selection, not a carrier-survival test."""
    root_task_id = str(root_task_id or '').strip()
    if not root_task_id:
        return {}
    try:
        state = _root_resume_state(q, root_task_id, result_root, actor_id=actor_id)
        return _root_resume_grants(state)[0]
    except Exception:
        if strict:
            raise
        log.debug('Root resume grant unreadable for %s', root_task_id, exc_info=True)
        return {}
