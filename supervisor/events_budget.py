"""Usage accounting and budget-pause events reported by workers.

One owner for folding a worker's reported usage into the ledger and for the two
budget fences a paused root raises: the pause itself and the admission fence
that keeps its descendants out of the queue.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict
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

    usage_for_budget = {
        **usage,
        "cost": resolved_cost,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "prompt_cache_ttl": prompt_cache_ttl,
    }
    projection_update_status = "available"
    try:
        ctx.update_budget_from_usage(usage_for_budget)
    except Exception:
        projection_update_status = "unavailable"
        log.error("Paid llm_usage retained but compatibility projection update failed", exc_info=True)

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
    row = {
        "status": "paused",
        "scope": "root",
        "root_task_id": root_task_id,
        "fence_id": str(
            pause.get("fence_id")
            or (existing or {}).get("fence_id")
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
    return row


def _handle_budget_pause(evt: Dict[str, Any], ctx: Any) -> None:
    """Move a zero-dispatch task back to the same durable queue generation."""
    task_id = str(evt.get("task_id") or "")
    pause = evt.get("resource_limit") if isinstance(evt.get("resource_limit"), dict) else {}
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
