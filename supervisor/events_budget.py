"""Usage accounting and budget-pause events reported by workers.

One owner for folding a worker's reported usage into the ledger and for the two
budget fences a paused root raises: the pause itself and the admission fence
that keeps its descendants out of the queue — including the per-row HOLD that
survives lifting that fence, and the explicit selection that releases one row
(#1196, owner Q9).
"""

from __future__ import annotations

import logging
import pathlib
import time
import uuid
from typing import Any, Dict, Optional
from ouroboros.utils import append_jsonl, utc_now_iso
from ouroboros.task_results import STATUS_SCHEDULED, write_task_result

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
            # root/parent/delegation/lane fields, but worker-side emitters do
            # not know the queue lineage. The supervisor DOES — fill the gaps
            # from the authoritative RUNNING record so per-tree cost rollups
            # over events.jsonl become possible (emitter-supplied values win).
            _task = _m.get("task") if isinstance(_m.get("task"), dict) else {}
            for _field in (
                "root_task_id", "parent_task_id", "delegation_role",
                "task_group_id", "requested_model_lane", "effective_model_lane",
            ):
                if not evt.get(_field) and _task.get(_field):
                    evt[_field] = str(_task.get(_field))

    # Normalize usage across loop.py, web_search, and delegated-run producers.
    # Tolerant coercion: one malformed token field must not raise and drop the
    # whole round from the budget ledger and events.jsonl (the exception would
    # be swallowed by dispatch_event and the cost silently lost).
    def _tolerant_int(*candidates: Any) -> int:
        for value in candidates:
            if value in (None, ""):
                continue
            try:
                return int(float(value))
            except (TypeError, ValueError):
                log.warning("llm_usage: non-numeric token field %r ignored", value)
        return 0

    prompt_tokens = _tolerant_int(
        usage.get("prompt_tokens"), usage.get("input_tokens"), evt.get("prompt_tokens")
    )
    completion_tokens = _tolerant_int(
        usage.get("completion_tokens"), usage.get("output_tokens"), evt.get("completion_tokens")
    )
    cached_tokens = _tolerant_int(usage.get("cached_tokens"), evt.get("cached_tokens"))
    cache_write_tokens = _tolerant_int(
        usage.get("cache_write_tokens"), evt.get("cache_write_tokens")
    )
    prompt_cache_ttl = str(
        usage.get("prompt_cache_ttl")
        or evt.get("prompt_cache_ttl")
        or ""
    )
    ledger_attempt_ids = [
        str(value)
        for value in (usage.get("ledger_attempt_ids") or evt.get("ledger_attempt_ids") or [])
        if value
    ]

    raw_cost = usage.get("cost")
    if raw_cost is None:
        raw_cost = evt.get("cost")
    cost_known = raw_cost not in (None, "")
    try:
        resolved_cost = float(raw_cost) if cost_known else None
    except (TypeError, ValueError):
        resolved_cost = None
        cost_known = False

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
        "provider": evt.get("provider", ""),
        "source": evt.get("source", ""),
        **{key: evt[key] for key in ("llm_call_id", "execution_id", "round_id", "round") if key in evt},
        "cost_estimated": bool(evt.get("cost_estimated", False)),
        "cost": resolved_cost,
        "cost_known": cost_known,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "prompt_cache_ttl": prompt_cache_ttl,
        "accounting_authority": "physical_attempt_ledger",
        "projection_update_status": projection_update_status,
        "ledger_attempt_ids": ledger_attempt_ids,
        **({"chat_id": evt["chat_id"]} if evt.get("chat_id") is not None else {}),
        **({"web_search_sources": web_search_sources} if isinstance(web_search_sources, list) and web_search_sources else {}),
        **({"reasoning_pin": reasoning_pin} if isinstance(reasoning_pin, dict) and reasoning_pin else {}),
        **({"reasoning_effort_clamped": effort_clamped} if isinstance(effort_clamped, dict) and effort_clamped else {}),
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


def _set_root_budget_pause_locked(root_task_id: str, pause: Dict[str, Any]) -> Dict[str, Any]:
    """Install the sole root-budget admission marker; caller holds queue lock."""
    from supervisor import queue as queue_mod

    root_task_id = str(root_task_id or "").strip()
    if not root_task_id:
        raise ValueError("root budget pause requires root_task_id")
    existing = queue_mod.BUDGET_ROOT_FENCES.get(root_task_id)
    root_rows = list(queue_mod.PENDING) + [m.get("task", {}) for m in queue_mod.RUNNING.values()]
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
    queue_mod.BUDGET_ROOT_FENCES[root_task_id] = row
    if not existing or existing.get("fence_id") != row["fence_id"]:
        from supervisor.budget_resume import revoke_exact_budget_resume

        for member in queue_mod.PENDING:
            if str(member.get("root_task_id") or "") != root_task_id or member.get("id") == root_task_id:
                continue
            if isinstance(member.get("_budget_pause_resume"), dict):
                revoke_exact_budget_resume(member, "new_root_budget_fence")
            hold = member.get(BUDGET_HOLD_KEY)
            if isinstance(hold, dict) and hold.get("selected"):
                hold_budget_row(member, reason=HOLD_ROOT_FENCE_LIFTED,
                                extra={"root_task_id": root_task_id, "fence_id": row["fence_id"]})
    return row


def _handle_budget_pause(evt: Dict[str, Any], ctx: Any) -> None:
    """Move a paused task back to the same durable queue generation.

    Two shapes share this seam: the historical replay-safe ZERO-dispatch pause
    and the exact continuation (#1196), whose worker wrote a durable
    ``budget_pause`` row before unwinding. The exact shape is validated against
    THAT row, never against the event's prose.
    """
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

    with _queue_lock:
        if str(pause.get("scope") or "") == "root":
            root_row = _set_root_budget_pause_locked(
                str(pause.get("root_task_id") or evt.get("root_task_id") or ""),
                pause,
            )
            pause = {
                **pause,
                **root_row,
                "status": "paused_before_dispatch",
                "replay_safe": True,
                "physical_calls": 0,
            }
        meta = ctx.RUNNING.pop(task_id, None)
        task = meta.get("task") if isinstance(meta, dict) and isinstance(meta.get("task"), dict) else None
        if task is None:
            raise RuntimeError(f"budget-paused task is not running: {task_id}")
        resumed_task = dict(task)
        resumed_task["_budget_pause"] = dict(pause)
        if not any(str(item.get("id") or "") == task_id for item in ctx.PENDING):
            ctx.PENDING.append(resumed_task)
            ctx.sort_pending()
        worker_id = evt.get("worker_id")
        if worker_id in ctx.WORKERS and ctx.WORKERS[worker_id].busy_task_id == task_id:
            ctx.WORKERS[worker_id].busy_task_id = None
    try:
        write_task_result(
            ctx.DRIVE_ROOT,
            task_id,
            STATUS_SCHEDULED,
            reason_code="budget_exhausted",
            resource_limit=pause,
            result="Task paused before its first model dispatch; explicit resume or cancel required.",
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


def install_exact_budget_pause(ctx: Any, task_id: str, checkpoint: Dict[str, Any], *,
                               evt: Dict[str, Any] | None = None,
                               source: str = "worker_event") -> Dict[str, Any]:
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
        marker = exact_pause_marker(row, default_root=str(task.get("root_task_id") or task_id))
        if marker["scope"] == "root":
            fence = _set_root_budget_pause_locked(marker["root_task_id"], marker)
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
        if persisted:
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

                cost_fields = reconstruct_task_cost(task_id, fields=True, drive_root=result_root)
                write_task_result(
                    result_root, task_id, STATUS_SCHEDULED,
                    reason_code="budget_paused", resource_limit=marker,
                    result=("Task paused exactly at a completed boundary (budget). Cumulative spend, rounds "
                            "and execution time are retained; an explicit owner Resume continues the same task."),
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
    return marker


def _handle_budget_resume_child(evt: Dict[str, Any], ctx: Any) -> None:
    """Owner Q9: a resumed root's model SELECTS one budget-paused child to continue.

    The requester must be the live parent/root of the target (its tree, not any
    tree); the grant itself goes through the ONE resume seam, so every typed
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
        lineage_ok = bool(
            target is not None and requester
            and requester in (str(target.get("parent_task_id") or ""), str(target.get("root_task_id") or ""))
        )
    if not lineage_ok:
        outcome: Dict[str, Any] = {"ok": False, "error": "not_a_budget_paused_descendant"}
    else:
        # A MODEL-issued selection: the grant additionally requires the live
        # owner-derived Resume grant of the target's own root (never lineage alone).
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

BUDGET_HOLD_KEY = "_budget_pause_hold"

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
# Malformed acceptance-fence evidence in the snapshot fails the restore closed
# for ordinary rows; a saved exact pause is retained under this hold instead.
HOLD_INVALID_ACCEPTANCE_FENCE_SNAPSHOT = HOLD_RESTORE_REFUSED_PREFIX + "invalid_acceptance_fence_snapshot"
HOLD_INVALID_BUDGET_FENCE_SNAPSHOT = HOLD_RESTORE_REFUSED_PREFIX + "invalid_budget_fence_snapshot"


def budget_hold_fact(task) -> Optional[Dict[str, Any]]:
    """The durable NON-dispatch hold on one queued row (#1196), or ``None``.

    Three shapes share it and none invents a checkpoint identity (no pause_id,
    no ``exact_continuation``): a zero-dispatch sibling whose paused root's
    admission fence was lifted by that root's Resume — lifting the fence must
    not make it assignable, the model selects it explicitly (owner Q9) — a row
    whose exact continuation could not be restored, and a row whose spent
    grant could not be revoked. ``selected`` is the only release.
    """
    hold = task.get(BUDGET_HOLD_KEY) if isinstance(task, dict) else None
    return hold if isinstance(hold, dict) and not hold.get("selected") else None


def budget_fence_selected(task: Any, fence: Any) -> bool:
    """Whether THIS row carries an explicit selection recorded against THIS fence.

    A root's admission latch keeps a whole tree out of the queue. The owner (or,
    under the root's live grant, the model) may select ONE member of that tree
    without lifting the latch for its siblings: the selection rides the row's own
    hold and names the fence generation it was granted against, so a later fence
    — a root that paused again — is never pre-released by an older selection
    (#1196, owner Q9).
    """
    hold = task.get(BUDGET_HOLD_KEY) if isinstance(task, dict) else None
    fence_id = str((fence or {}).get("fence_id") or "") if isinstance(fence, dict) else ""
    return bool(isinstance(hold, dict) and hold.get("selected") and fence_id
                and str(hold.get("fence_id") or "") == fence_id)


def budget_resume_dispatch_allowed(q: Any, task: Dict[str, Any]) -> bool:
    """A child selection belongs to the CURRENT root grant and fence only.

    Both carriers are revalidated at dispatch: an exact grant handoff and a
    zero-dispatch hold selection alike name the root grant they were selected
    under, so a root that is pausing or paused again (with or without a new
    fence) admits neither on its old selection (#1196, owner Q9).
    """
    handoff = task.get("_budget_pause_resume")
    hold = task.get(BUDGET_HOLD_KEY) if isinstance(task.get(BUDGET_HOLD_KEY), dict) else None
    exact = isinstance(handoff, dict)
    carrier = handoff if exact else (hold if hold is not None and hold.get("selected") else None)
    if carrier is None:
        return True
    root_id = str(task.get("root_task_id") or task.get("id") or "")
    fence_id = str((q.BUDGET_ROOT_FENCES.get(root_id) or {}).get("fence_id") or "")
    if fence_id != str(carrier.get("root_fence_id" if exact else "fence_id") or ""):
        return False
    if root_id == str(task.get("id") or ""):
        return True
    root_grant = live_root_resume_grant(q, root_id, pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT))
    return bool(carrier.get("selected_by") == "owner" and not carrier.get("root_grant_id") and not root_grant
                or root_grant and carrier.get("root_grant_id") == root_grant["grant_id"]
                and int(carrier.get("root_resume_generation") or 0) == int(root_grant["generation"]))


def hold_budget_row(task: Dict[str, Any], *, reason: str, detail: str = "",
                    extra: Optional[Dict[str, Any]] = None,
                    result_root: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """Hold one queued row: typed, visible, never dropped and never cancelled.

    The row stays PENDING with its own identity; any spent resume handoff is
    removed so no stale grant can dispatch, and the typed reason is projected
    onto the task result so the owner and the model read the same fact.
    """
    hold = {"reason": str(reason), "detail": str(detail or "")[:300], "held_at": utc_now_iso(),
            "selected": False, "dispatchable": False, **(extra or {})}
    task.pop("_budget_pause_resume", None)
    task[BUDGET_HOLD_KEY] = hold
    if result_root is not None:
        try:
            write_task_result(
                result_root, str(task.get("id") or ""), STATUS_SCHEDULED,
                reason_code="budget_paused",
                resource_limit={"status": "budget_hold", "auto_resume": False,
                                "exact_continuation": False,
                                "resume_policy": "explicit_selection_same_seam", **hold},
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


def hold_root_resume_descendants(q: Any, root_id: str, fence: dict, grant: dict) -> tuple:
    """Replace lifted-fence eligibility with per-child holds; retain exact pauses.

    A fence-derived marker is not a checkpoint: convert it to a selection hold
    before removing its fence, or it would be stranded. Older exact child grants
    are revoked and selected again under this root grant. A sibling still held
    from an EARLIER Resume of this root (pauseA -> Resume -> pauseB -> Resume) is
    re-bound to the live grant in the same pass, so it stays selectable instead
    of refused forever as stale. Return ``(held, markers, rebound)``: the hold and
    marker changes and the prior re-bound holds, for the Resume transaction's
    snapshot rollback; revocations stay safe.
    """
    from supervisor.budget_resume import revoke_exact_budget_resume

    held, markers, rebound = [], {}, {}
    for member in q.PENDING:
        member_id = str(member.get("id") or "")
        if not member_id or member_id == root_id or str(member.get("root_task_id") or "") != root_id:
            continue
        if isinstance(member.get("_budget_pause_resume"), dict):
            revoke_exact_budget_resume(member, "root_resume_generation_changed")
        pause = member.get("_budget_pause")
        fence_derived = bool(isinstance(pause, dict) and not pause.get("exact_continuation")
                             and pause.get("scope") == "root" and pause.get("root_task_id") == root_id
                             and pause.get("fence_id") == fence.get("fence_id"))
        hold = member.get(BUDGET_HOLD_KEY) if isinstance(member.get(BUDGET_HOLD_KEY), dict) else None
        if (hold is not None and not hold.get("selected") and str(hold.get("reason") or "") == HOLD_ROOT_FENCE_LIFTED
                and str(hold.get("root_grant_id") or "") != grant["grant_id"]):
            rebound[member_id] = dict(hold)
            member[BUDGET_HOLD_KEY] = {**hold, "root_grant_id": grant["grant_id"],
                                       "root_resume_generation": int(grant["generation"]),
                                       "rebound_at": utc_now_iso()}
        if (pause is not None and not fence_derived) or hold is not None:
            continue
        if fence_derived:
            markers[member_id] = member.pop("_budget_pause")
        hold_budget_row(
            member, reason=HOLD_ROOT_FENCE_LIFTED,
            detail="root resumed; this zero-dispatch sibling awaits explicit selection",
            extra={"root_task_id": root_id, "root_grant_id": grant["grant_id"],
                   "root_resume_generation": grant["generation"],
                   **({"replaced_fence_marker": True} if fence_derived else {})},
            result_root=pathlib.Path(member.get("budget_drive_root") or q.DRIVE_ROOT))
        held.append(member_id)
    return held, markers, rebound


def select_held_budget_row(q: Any, task: Dict[str, Any], hold: Dict[str, Any],
                           *, selected_by: str) -> Dict[str, Any]:
    """Record an explicit selection on a held row (queue lock held).

    The hold is released ONLY here, and only for a row that still proves it
    never dispatched. A model-issued selection additionally needs the live
    owner-derived Resume grant of its root: lifting a fence granted eligibility,
    never continuation.
    """
    task_id = str(task.get("id") or "")
    result_root = pathlib.Path(task.get("budget_drive_root") or q.DRIVE_ROOT)
    root_task_id = str(hold.get("root_task_id") or task.get("root_task_id") or task_id)
    if hold.get("reason") not in {HOLD_ROOT_FENCE_LIFTED, HOLD_ROOT_FENCE_MEMBER_SELECTION}:
        return {"ok": False, "error": str(hold.get("reason") or "budget_hold_unresolved")}
    root_grant = live_root_resume_grant(q, root_task_id, result_root) if task_id != root_task_id else {}
    if selected_by and task_id != root_task_id:
        if not root_grant:
            return {"ok": False, "error": "root_resume_grant_missing",
                    "root_task_id": root_task_id, "action": "resume_root_first"}
        if (hold.get("root_grant_id") and str(hold["root_grant_id"]) != root_grant["grant_id"]):
            return {"ok": False, "error": "root_resume_generation_stale",
                    "root_task_id": root_task_id, "action": "resume_root_first"}
    from supervisor.queue_transitions import pending_member_replay_safe

    safe, unsafe_error = pending_member_replay_safe(q, task)
    if not safe:
        return {"ok": False, "error": unsafe_error, "action": "cancel_or_new_run"}
    selection = {**hold, "selected": True, "selected_at": utc_now_iso(),
                 "selected_by": str(selected_by or "owner")}
    if task_id == root_task_id:
        selection.update(root_grant_id=uuid.uuid4().hex, root_resume_generation=1)
    elif root_grant:
        # The selection names the root grant it was made under, so dispatch can
        # recheck it exactly like an exact grant handoff (budget_resume_dispatch_allowed).
        selection.update(root_grant_id=root_grant["grant_id"],
                         root_resume_generation=int(root_grant["generation"]))
    prior_pause = task.pop("_budget_pause", None)
    task[BUDGET_HOLD_KEY] = selection
    if not q.persist_queue_snapshot(reason="budget_hold_selected"):
        task[BUDGET_HOLD_KEY] = dict(hold)
        if prior_pause is not None:
            task["_budget_pause"] = prior_pause
        return {"ok": False, "error": "snapshot_not_persisted"}
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


def live_root_resume_grant(q: Any, root_task_id: str, result_root: pathlib.Path) -> Dict[str, Any]:
    """The root's CURRENT owner-derived Resume grant, or ``{}`` (owner Q9).

    A descendant continues only under a root Resume that is still live: the
    root's durable pause row must carry a granted-or-consumed, unrevoked grant.
    A root that paused again after that grant opens a NEW generation, so a late
    selection carrying the old one finds nothing live and is refused.
    """
    from ouroboros.budget_pause import STATE_RESUME_GRANTED, STATE_RESUMED, budget_pause_row

    root_task_id = str(root_task_id or "").strip()
    if not root_task_id:
        return {}
    rows = [row for row in q.PENDING
            if isinstance(row, dict) and str(row.get("id") or "") == root_task_id]
    rows += [meta.get("task") for meta in q.RUNNING.values()
             if isinstance(meta, dict) and isinstance(meta.get("task"), dict)
             and str(meta["task"].get("id") or "") == root_task_id]
    fence = q.BUDGET_ROOT_FENCES.get(root_task_id)
    if fence:
        # Legacy zero-dispatch root Resume selects only the root. Its selection
        # grants child eligibility while the fence continues holding siblings.
        for task in rows:
            hold = task.get(BUDGET_HOLD_KEY) or {}
            if budget_fence_selected(task, fence) and hold.get("root_grant_id"):
                return {"grant_id": hold["root_grant_id"],
                        "generation": hold["root_resume_generation"], "pause_id": ""}
        return {}
    root_drive = next((str(row.get("budget_drive_root")) for row in rows
                       if isinstance(row, dict) and row.get("budget_drive_root")), "")
    try:
        row = budget_pause_row(pathlib.Path(root_drive or result_root), root_task_id)
    except Exception:
        log.debug("Root resume grant unreadable for %s", root_task_id, exc_info=True)
        return {}
    grant = row.get("grant") if isinstance(row.get("grant"), dict) else {}
    if (row.get("state") not in {STATE_RESUME_GRANTED, STATE_RESUMED}
            or not str(row.get("pause_id") or "").strip()
            or not str(grant.get("grant_id") or "").strip() or grant.get("revoked_at")):
        return {}
    return {"grant_id": str(grant["grant_id"]),
            "generation": int(row.get("resume_generation") or 0),
            "pause_id": str(row.get("pause_id") or "")}
