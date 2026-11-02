"""What a running worker reports about itself, and what the host does with it.

Heartbeats and dispatch resolution keep the liveness rails honest, metrics and
forwarded log lines reach the owner's panel, the acceptance fence is
acknowledged back to the worker, a bounded external-wait lease spares the
idle rail alone, and a translation-memory write reaches the open windows.
"""

from __future__ import annotations

import logging
import pathlib
import time
from typing import Any, Dict
from ouroboros.utils import atomic_write_json, utc_now_iso
from ouroboros.outcomes import normalize_outcome_axes
from supervisor.events_chat_delivery import _bound_project_chat_id

log = logging.getLogger(__name__)


from supervisor.log_addressing import address_ctx_event as _address_ctx
from ouroboros.cost_projection import live_root_cost_projection


def _handle_task_heartbeat(evt: Dict[str, Any], ctx: Any) -> None:
    """Liveness stamp and live frame for one running task; a no-op once it left RUNNING.

    Mutates the RUNNING entry IN PLACE (the ``_handle_llm_usage`` rule): a
    write-back would resurrect a task a cross-thread cancel popped between the
    get and the write. The root cost figure is the stale-allowed projection,
    so a heartbeat never reads the current task-result store on the drain.
    """
    task_id = str(evt.get("task_id") or "")
    if task_id and task_id in ctx.RUNNING:
        meta = ctx.RUNNING.get(task_id)
        if not isinstance(meta, dict):
            return
        meta["last_heartbeat_at"] = time.time()
        phase = str(evt.get("phase") or "")
        if phase:
            meta["heartbeat_phase"] = phase
        task = meta.get("task") if isinstance(meta.get("task"), dict) else {}
        started_at = float(meta.get("started_at") or 0.0)
        runtime_sec = round(max(0.0, time.time() - started_at), 1) if started_at > 0 else None
        from supervisor.task_model_wait import quota_waited_seconds

        quota_wait_sec = quota_waited_seconds(meta, time.time())
        # Stamp the project thread so the live heartbeat routes to the project
        # panel (and not default-to-main); post-hoc bound tasks fall back to the
        # binding. Heartbeats themselves carry no chat_id from the worker. A
        # post-hoc bound task keeps its original (main) chat_id, so the binding
        # must take PRECEDENCE (same order as _handle_send_message/_handle_log_event).
        try:
            _hb_chat_id = _bound_project_chat_id(ctx, task_id, task.get("parent_task_id"), task.get("root_task_id")) or int(task.get("chat_id") or 0)
        except (TypeError, ValueError):
            _hb_chat_id = 0
        cost_fields = live_root_cost_projection(task_id, task, evt, ctx.DRIVE_ROOT)
        try:
            ctx.bridge.push_log({
                "ts": evt.get("ts", utc_now_iso()),
                "type": "task_heartbeat",
                "task_id": task_id,
                "task_type": task.get("type"),
                "chat_id": _hb_chat_id,
                "phase": phase or meta.get("heartbeat_phase") or "running",
                "runtime_sec": runtime_sec,
                "execution_sec": max(0.0, runtime_sec - quota_wait_sec) if runtime_sec is not None else None,
                "quota_wait_sec": quota_wait_sec,
                **({"model_waits": task["model_waits"]} if task.get("model_waits") else {}),
                "subagent_event": evt.get("subagent_event", ""),
                "subagent_task_id": evt.get("subagent_task_id", ""),
                "root_task_id": evt.get("root_task_id", ""),
                "parent_task_id": evt.get("parent_task_id", ""),
                "delegation_role": evt.get("delegation_role", ""),
                "subagent_role": evt.get("subagent_role", ""),
                **cost_fields,
            })
        except Exception:
            log.debug("Failed to forward task heartbeat to live logs", exc_info=True)


def _handle_task_dispatch_resolved(evt: Dict[str, Any], ctx: Any) -> None:
    """Merge a worker's dispatch-time resolution into the supervisor's RUNNING copy.

    ``agent.resolve_dispatch_axes`` runs INSIDE the worker process and stamps the
    worker's own clone of the task; ``assign_tasks`` stored a separate ``dict(task)``
    in RUNNING before dispatch, and ``persist_queue_snapshot`` serializes THAT copy.
    Without this merge a restart while a child was running restored the unresolved
    intent — `effective_model_lane`, `reasoning_effort`, the executor fields and
    `capability_delta` were lost, and a restore could re-derive different live facts
    (XG-2R.1, three reviewers converged). The merge is scoped to exactly
    ``SUBAGENT_RESOLUTION_FIELDS`` — a worker report can never overwrite scheduling
    intent or supervisor bookkeeping — runs under the queue lock, and persists the
    snapshot so the resolution is durable before anything else happens to the queue.
    """
    from ouroboros.subagents import SUBAGENT_RESOLUTION_FIELDS
    from supervisor.queue import _queue_lock

    task_id = str(evt.get("task_id") or "")
    resolution = evt.get("resolution") if isinstance(evt.get("resolution"), dict) else {}
    if not task_id or not resolution:
        return
    with _queue_lock:
        meta = ctx.RUNNING.get(task_id)
        task = meta.get("task") if isinstance(meta, dict) else None
        if not isinstance(task, dict):
            return
        for key in SUBAGENT_RESOLUTION_FIELDS:
            if key in resolution:
                task[key] = resolution[key]
    ctx.persist_queue_snapshot(reason="dispatch_resolved")


def _handle_task_focus_updated(evt: Dict[str, Any], ctx: Any) -> None:
    """Project a root-authored focus into the existing RUNNING snapshot."""
    from ouroboros.focus import compact_focus
    from supervisor.queue import _queue_lock

    task_id = str(evt.get("task_id") or "").strip()
    focus = compact_focus(evt.get("focus"))
    if not task_id or focus is None or str(focus.get("author_task_id") or "") != task_id:
        return
    changed = False
    with _queue_lock:
        meta = ctx.RUNNING.get(task_id)
        task = meta.get("task") if isinstance(meta, dict) else None
        if not isinstance(task, dict):
            return
        if str(task.get("parent_task_id") or "").strip() or str(task.get("delegation_role") or "") == "subagent":
            return
        # The event is advisory transport.  The canonical result remains the
        # lifecycle authority, so a focus queued just before completion cannot
        # resurrect a terminal task in the queue projection.
        drive_root = meta.get("budget_drive_root") if isinstance(meta, dict) else None
        drive_root = drive_root or task.get("budget_drive_root") or getattr(ctx, "DRIVE_ROOT", None)
        if drive_root:
            try:
                from ouroboros.task_results import STATUS_RUNNING, load_task_result

                durable = load_task_result(drive_root, task_id)
                if not isinstance(durable, dict) or str(durable.get("status") or "") != STATUS_RUNNING:
                    return
                durable_focus = compact_focus(durable.get("focus"))
                if durable_focus != focus:
                    return
            except Exception:
                return
        prior_task = compact_focus(task.get("focus"))
        if prior_task and str(prior_task.get("authored_at") or "") >= str(focus.get("authored_at") or ""):
            return
        task["focus"] = focus
        changed = True
    if changed:
        ctx.persist_queue_snapshot(reason="task_focus_updated")


def _handle_ui_i18n_updated(evt: Dict[str, Any], ctx: Any) -> None:
    """A turn wrote the translation memory (``pin_ui_translation``); this process holds the
    windows, so it tells them to read the memory again. The file is the truth: the frame
    carries only which revision now exists."""
    language, revision = evt.get("language"), evt.get("revision")
    if not isinstance(language, str) or not language.strip() or type(revision) is not int or revision < 0:
        return
    ctx.bridge.broadcast({"type": "ui_i18n_updated", "language": language.strip(), "revision": revision})


def _handle_update_progress_changed(evt: Dict[str, Any], ctx: Any) -> None:
    """A turn saved the update description (``record_update_description``); this process
    holds the windows, so it asks them to read the update status again. The record is the
    truth: the frame carries nothing else."""
    ctx.bridge.broadcast({"type": "update_progress_changed"})


def _handle_task_metrics(evt: Dict[str, Any], ctx: Any) -> None:
    payload = {
        "ts": str(evt.get("ts") or utc_now_iso()),
        "type": "task_metrics_event",
        "task_id": str(evt.get("task_id") or ""),
        "task_type": str(evt.get("task_type") or ""),
        "duration_sec": round(float(evt.get("duration_sec") or 0.0), 3),
        "tool_calls": None if evt.get("tool_calls") is None else int(evt["tool_calls"]),
        "tool_errors": None if evt.get("tool_errors") is None else int(evt["tool_errors"]),
        "outcome_axes": normalize_outcome_axes(evt),
        "reason_code": str(evt.get("reason_code") or ""),
    }
    for key in ("routing_tool_calls", "completion_tool_calls", "rounds"):
        if key in evt:
            payload[key] = None if evt[key] is None else int(evt[key])
    if "tool_call_counts" in evt:
        counts = evt["tool_call_counts"]
        payload["tool_call_counts"] = dict(counts) if isinstance(counts, dict) else None
    if evt.get("chat_id") is not None:
        payload["chat_id"] = evt["chat_id"]
    _address_ctx(ctx, payload)
    ctx.append_jsonl(ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl", payload)
    try:
        ctx.bridge.push_log(payload)
    except Exception:
        log.debug("Failed to forward task_metrics to live logs", exc_info=True)


def _handle_log_event(evt: Dict[str, Any], ctx: Any) -> None:
    """Address and forward one live frame; persist the one nested kind whose producer owns no row.

    A ``task_checkpoint`` arrives already durable: its producer appended the
    events.jsonl row (and the progress.jsonl row of a System checkpoint) before
    queueing the frame (``loop_messages._emit_checkpoint_event``, owner 1A(3)),
    with lineage and chat stamped by value, so this handler — which under the
    two-lane bus may run after the task left RUNNING — only addresses and pushes.
    """
    data = evt.get("data")
    if not isinstance(data, dict):
        return
    payload = {
        "ts": data.get("ts", utc_now_iso()),
        **data,
    }
    _address_ctx(ctx, payload)
    try:
        ctx.bridge.push_log(payload)
    except Exception:
        log.debug("Failed to forward live log event", exc_info=True)
    # task_start_settings_reload_failed is a durable owner disclosure (#285):
    # without persistence the fact evaporates on the next page load.
    if data.get("type") == "task_start_settings_reload_failed":
        try:
            ctx.append_jsonl(ctx.DRIVE_ROOT / "logs" / "events.jsonl", payload)
        except Exception:
            log.debug("Failed to persist %s event to events.jsonl", data.get("type"), exc_info=True)


def _handle_skill_lifecycle(evt: Dict[str, Any], ctx: Any) -> None:
    payload = dict(evt)
    payload.setdefault("ts", utc_now_iso())
    _address_ctx(ctx, payload)
    try:
        ctx.append_jsonl(ctx.DRIVE_ROOT / "logs" / "events.jsonl", payload)
    except Exception:
        log.debug("Failed to persist skill lifecycle event", exc_info=True)
    try:
        ctx.bridge.push_log(payload)
    except Exception:
        log.debug("Failed to forward skill lifecycle event to live logs", exc_info=True)
    try:
        from ouroboros.event_bus import SKILL_LIFECYCLE, publish_event

        publish_event(SKILL_LIFECYCLE, payload)
    except Exception:
        log.debug("Failed to publish skill lifecycle event", exc_info=True)


def _handle_acceptance_fence(evt: Dict[str, Any], ctx: Any) -> None:
    """Apply a worker's acceptance fence under the supervisor queue lock, then ack.

    The ack belongs to ONE request: ``<token>.<req>.json``. begin/inspect/end share
    the fence token, so a late answer must never be readable as another request's.
    """
    token, req = (str(evt.get(key) or "").strip().lower() for key in ("token", "req"))
    if any(not part or len(part) > 64 or any(ch not in "0123456789abcdef" for ch in part) for part in (token, req)):
        log.warning("Rejected malformed acceptance-fence token or request id")
        return
    try:
        from supervisor.queue import transition_acceptance_fence

        result = transition_acceptance_fence(
            action=str(evt.get("action") or ""),
            token=token,
            root_task_id=str(evt.get("root_task_id") or ""),
            task_id=str(evt.get("task_id") or ""),
            # A legacy event without it keeps the unattributed (pre-#1183) begin.
            pooled_owner=evt.get("pooled_owner") if isinstance(evt.get("pooled_owner"), dict) else None,
            outcome=str(evt.get("outcome") or ""),
            expected_generation=(
                int(evt["expected_generation"])
                if evt.get("expected_generation") is not None else None
            ),
        )
    except Exception as exc:
        log.warning("Acceptance-fence transition failed", exc_info=True)
        result = {"ok": False, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    ack_dir = pathlib.Path(ctx.DRIVE_ROOT) / "state" / "acceptance_fence_acks"
    ack_path = ack_dir / f"{token}.{req}.json"
    try:
        now = time.time()
        prior = sorted(ack_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        for index, path in enumerate(prior):
            if index >= 255 or now - path.stat().st_mtime > 3600.0:
                path.unlink(missing_ok=True)
    except Exception:
        log.warning("Could not compact stale acceptance-fence acknowledgements", exc_info=True)
    try:
        atomic_write_json(ack_path, {**result, "ts": utc_now_iso()}, trailing_newline=True)
    except Exception:
        # Loud: without the acknowledgement the worker fails closed rather than
        # reviewing against a possibly-racing subtree.
        log.error("Could not acknowledge acceptance-fence transition", exc_info=True)


def _handle_external_wait_lease(evt: Dict[str, Any], ctx: Any) -> None:
    """Typed idle-rail lease (poltergeist phase B, B3): a worker holding a bounded
    ``delegate_wait`` window over a live delegated run declares that its silence
    is a legitimate host-side hold, not idleness.

    The lease spares ONLY the idle rail (`_enforce_task_timeouts_locked` reads
    ``external_wait_lease_until`` into its ``progressing`` disjunction); the
    explicit deadline, the absolute ceiling, budget fences and cancel are
    untouched. The expiry is re-clamped here against the absolute ceiling so a
    malformed worker value can never mint an unbounded reprieve, and a release
    (``until_ts <= 0``) drops the lease immediately — but only when it NAMES the
    stored grant's ``lease_id`` (F5b lease identity). Mutate IN PLACE — see
    ``_handle_llm_usage``: a write-back would resurrect a task a cross-thread
    cancel popped between the get and the write.
    """
    task_id = str(evt.get("task_id") or "")
    _running = getattr(ctx, "RUNNING", None)
    if not task_id or not isinstance(_running, dict):
        return
    meta = _running.get(task_id)
    if not isinstance(meta, dict):
        return
    try:
        until = float(evt.get("until_ts") or 0.0)
    except (TypeError, ValueError):
        until = 0.0
    lease_id = str(evt.get("lease_id") or "")
    if until > 0:
        from ouroboros.delegate_progress import EXTERNAL_WAIT_LEASE_CEILING_SEC

        meta["external_wait_lease_until"] = min(
            until, time.time() + float(EXTERNAL_WAIT_LEASE_CEILING_SEC))
        meta["external_wait_lease_run_id"] = str(evt.get("run_id") or "")
        meta["external_wait_lease_id"] = lease_id
    else:
        # A release must NAME the grant it retires (F5b): an abandoned,
        # executor-killed wait thread's late release event would otherwise blank
        # the NEWER grant the task's next wait just made. A release without an
        # id (legacy emitter), or one matching the stored grant (or a stored
        # grant without an id), clears as before.
        stored = str(meta.get("external_wait_lease_id") or "")
        if lease_id and stored and stored != lease_id:
            return
        meta.pop("external_wait_lease_until", None)
        meta.pop("external_wait_lease_run_id", None)
        meta.pop("external_wait_lease_id", None)
