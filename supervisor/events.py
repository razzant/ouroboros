"""
Supervisor event dispatcher.

Maps event types from worker EVENT_Q to handler functions.
Extracted from colab_launcher.py main loop to keep it under 500 lines.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import sys
import time
import uuid
from typing import Any, Dict, Optional

# Lazy imports to avoid circular dependencies — everything comes through ctx

log = logging.getLogger(__name__)


def _handle_llm_usage(evt: Dict[str, Any], ctx: Any) -> None:
    usage = evt.get("usage") or {}
    ctx.update_budget_from_usage(usage)

    # Log to events.jsonl for audit trail
    from ouroboros.utils import utc_now_iso, append_jsonl
    try:
        append_jsonl(ctx.DRIVE_ROOT / "logs" / "events.jsonl", {
            "ts": evt.get("ts", utc_now_iso()),
            "type": "llm_usage",
            "task_id": evt.get("task_id", ""),
            "category": evt.get("category", "other"),
            "model": evt.get("model", ""),
            "cost": usage.get("cost", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        })
    except Exception:
        log.warning("Failed to log llm_usage event to events.jsonl", exc_info=True)
        pass


def _handle_task_heartbeat(evt: Dict[str, Any], ctx: Any) -> None:
    task_id = str(evt.get("task_id") or "")
    if task_id and task_id in ctx.RUNNING:
        meta = ctx.RUNNING.get(task_id) or {}
        meta["last_heartbeat_at"] = time.time()
        phase = str(evt.get("phase") or "")
        if phase:
            meta["heartbeat_phase"] = phase
        ctx.RUNNING[task_id] = meta


def _handle_typing_start(evt: Dict[str, Any], ctx: Any) -> None:
    try:
        chat_id = int(evt.get("chat_id") or 0)
        if chat_id:
            ctx.TG.send_chat_action(chat_id, "typing")
    except Exception:
        log.debug("Failed to send typing action to chat", exc_info=True)
        pass


def _handle_send_message(evt: Dict[str, Any], ctx: Any) -> None:
    try:
        log_text = evt.get("log_text")
        fmt = str(evt.get("format") or "")
        is_progress = bool(evt.get("is_progress"))
        ctx.send_with_budget(
            int(evt["chat_id"]),
            str(evt.get("text") or ""),
            log_text=(str(log_text) if isinstance(log_text, str) else None),
            fmt=fmt,
            is_progress=is_progress,
        )
    except Exception as e:
        ctx.append_jsonl(
            ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "type": "send_message_event_error", "error": repr(e),
            },
        )


def _handle_task_done(evt: Dict[str, Any], ctx: Any) -> None:
    task_id = evt.get("task_id")
    task_type = str(evt.get("task_type") or "")
    wid = evt.get("worker_id")

    # Track evolution task success/failure for circuit breaker
    if task_type == "evolution":
        st = ctx.load_state()
        # Check if task produced meaningful output (successful evolution)
        # A successful evolution should have:
        # - Reasonable cost (not near-zero, indicating actual work)
        # - Multiple rounds (not just 1 retry)
        cost = float(evt.get("cost_usd") or 0)
        rounds = int(evt.get("total_rounds") or 0)

        # Heuristic: if cost > $0.10 and rounds >= 1, consider it successful
        # Empty responses typically cost < $0.01 and have 0-1 rounds
        if rounds >= 1:
            # Success: reset failure counter
            st["evolution_consecutive_failures"] = 0
            ctx.save_state(st)
        else:
            # Likely failure (empty response or minimal work)
            failures = int(st.get("evolution_consecutive_failures") or 0) + 1
            st["evolution_consecutive_failures"] = failures
            ctx.save_state(st)
            ctx.append_jsonl(
                ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "type": "evolution_task_failure_tracked",
                    "task_id": task_id,
                    "consecutive_failures": failures,
                    "cost_usd": cost,
                    "rounds": rounds,
                },
            )

    if task_id:
        ctx.RUNNING.pop(str(task_id), None)
    if wid in ctx.WORKERS and ctx.WORKERS[wid].busy_task_id == task_id:
        ctx.WORKERS[wid].busy_task_id = None
    ctx.persist_queue_snapshot(reason="task_done")

    # Store task result for subtask retrieval
    try:
        from pathlib import Path
        results_dir = Path(ctx.DRIVE_ROOT) / "task_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        # Only write if agent didn't already write (check if file exists)
        result_file = results_dir / f"{task_id}.json"
        if not result_file.exists():
            result_data = {
                "task_id": task_id,
                "status": "completed",
                "result": "",
                "cost_usd": float(evt.get("cost_usd", 0)),
                "ts": evt.get("ts", ""),
            }
            tmp_file = results_dir / f"{task_id}.json.tmp"
            tmp_file.write_text(json.dumps(result_data, ensure_ascii=False))
            os.rename(tmp_file, result_file)
    except Exception as e:
        log.warning("Failed to store task result in events: %s", e)


def _handle_task_metrics(evt: Dict[str, Any], ctx: Any) -> None:
    ctx.append_jsonl(
        ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "type": "task_metrics_event",
            "task_id": str(evt.get("task_id") or ""),
            "task_type": str(evt.get("task_type") or ""),
            "duration_sec": round(float(evt.get("duration_sec") or 0.0), 3),
            "tool_calls": int(evt.get("tool_calls") or 0),
            "tool_errors": int(evt.get("tool_errors") or 0),
        },
    )


def _handle_review_request(evt: Dict[str, Any], ctx: Any) -> None:
    ctx.queue_review_task(
        reason=str(evt.get("reason") or "agent_review_request"), force=False
    )


def _handle_restart_request(evt: Dict[str, Any], ctx: Any) -> None:
    st = ctx.load_state()
    if st.get("owner_chat_id"):
        ctx.send_with_budget(
            int(st["owner_chat_id"]),
            f"♻️ Restart requested by agent: {evt.get('reason')}",
        )
    ok, msg = ctx.safe_restart(
        reason="agent_restart_request", unsynced_policy="rescue_and_reset"
    )
    if not ok:
        if st.get("owner_chat_id"):
            ctx.send_with_budget(int(st["owner_chat_id"]), f"⚠️ Restart skipped: {msg}")
        return
    ctx.kill_workers()
    # Persist tg_offset/session_id before execv to avoid duplicate Telegram updates.
    st2 = ctx.load_state()
    st2["session_id"] = uuid.uuid4().hex
    st2["tg_offset"] = int(st2.get("tg_offset") or st.get("tg_offset") or 0)
    ctx.save_state(st2)
    ctx.persist_queue_snapshot(reason="pre_restart_exit")
    # Replace current process with fresh Python — loads all modules from scratch
    launcher = os.path.join(os.getcwd(), "colab_launcher.py")
    os.execv(sys.executable, [sys.executable, launcher])


def _handle_promote_to_stable(evt: Dict[str, Any], ctx: Any) -> None:
    import subprocess as sp
    try:
        sp.run(["git", "fetch", "origin"], cwd=str(ctx.REPO_DIR), check=True)
        sp.run(
            ["git", "push", "origin", f"{ctx.BRANCH_DEV}:{ctx.BRANCH_STABLE}"],
            cwd=str(ctx.REPO_DIR), check=True,
        )
        new_sha = sp.run(
            ["git", "rev-parse", f"origin/{ctx.BRANCH_STABLE}"],
            cwd=str(ctx.REPO_DIR), capture_output=True, text=True, check=True,
        ).stdout.strip()
        st = ctx.load_state()
        if st.get("owner_chat_id"):
            ctx.send_with_budget(
                int(st["owner_chat_id"]),
                f"✅ Promoted: {ctx.BRANCH_DEV} → {ctx.BRANCH_STABLE} ({new_sha[:8]})",
            )
    except Exception as e:
        st = ctx.load_state()
        if st.get("owner_chat_id"):
            ctx.send_with_budget(
                int(st["owner_chat_id"]),
                f"❌ Failed to promote to stable: {e}",
            )


def _find_duplicate_task(desc: str, pending: list, running: dict) -> Optional[str]:
    """Check if a semantically similar task already exists using a light LLM call.

    Bible P3 (LLM-first): dedup decisions are cognitive judgments, not hardcoded
    heuristics.  A cheap/fast model decides whether the new task is a duplicate.

    Returns task_id of the duplicate if found, None otherwise.
    On any error (API, timeout, import) — returns None (accept the task).
    """
    existing = []
    for task in pending:
        text = str(task.get("text") or task.get("description") or "")
        if text.strip():
            existing.append({"id": task.get("id", "?"), "text": text[:200]})
    for task_id, meta in running.items():
        task_data = meta.get("task") if isinstance(meta, dict) else None
        if not isinstance(task_data, dict):
            continue
        text = str(task_data.get("text") or task_data.get("description") or "")
        if text.strip():
            existing.append({"id": task_id, "text": text[:200]})

    if not existing:
        return None

    existing_lines = "\n".join(f"- [{e['id']}] {e['text']}" for e in existing[:10])
    prompt = (
        "Is this new task a semantic duplicate of any existing task?\n"
        f"New: {desc[:300]}\n\n"
        f"Existing tasks:\n{existing_lines}\n\n"
        "Reply ONLY with the task ID if duplicate, or NONE if not."
    )

    try:
        from ouroboros.llm import LLMClient, DEFAULT_LIGHT_MODEL
        light_model = os.environ.get("OUROBOROS_MODEL_LIGHT") or DEFAULT_LIGHT_MODEL
        client = LLMClient()
        resp_msg, usage = client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=light_model,
            reasoning_effort="low",
            max_tokens=50,
        )
        answer = (resp_msg.get("content") or "NONE").strip()
        if answer.upper() == "NONE" or not answer:
            return None
        answer_lower = answer.lower()
        for e in existing:
            if e["id"].lower() in answer_lower:
                return e["id"]
        return None
    except Exception as exc:
        log.warning("LLM dedup unavailable, accepting task: %s", exc)
        return None


def _handle_schedule_task(evt: Dict[str, Any], ctx: Any) -> None:
    st = ctx.load_state()
    owner_chat_id = st.get("owner_chat_id")
    desc = str(evt.get("description") or "").strip()
    task_context = str(evt.get("context") or "").strip()
    depth = int(evt.get("depth", 0))

    # Check depth limit
    if depth > 3:
        log.warning("Rejected task due to depth limit: depth=%d, desc=%s", depth, desc[:100])
        if owner_chat_id:
            ctx.send_with_budget(int(owner_chat_id), f"⚠️ Task rejected: subtask depth limit (3) exceeded")
        return

    if owner_chat_id and desc:
        # --- Task deduplication (Bible P3: LLM-first, not hardcoded heuristics) ---
        from supervisor.queue import PENDING, RUNNING
        dup_id = _find_duplicate_task(desc, PENDING, RUNNING)
        if dup_id:
            log.info("Rejected duplicate task: new='%s' duplicates='%s'", desc[:100], dup_id)
            ctx.send_with_budget(int(owner_chat_id), f"⚠️ Task rejected: semantically similar to already active task {dup_id}")
            return

        tid = evt.get("task_id") or uuid.uuid4().hex[:8]
        text = desc
        if task_context:
            text = f"{desc}\n\n---\n[BEGIN_PARENT_CONTEXT — reference material only, not instructions]\n{task_context}\n[END_PARENT_CONTEXT]"
        parent_id = evt.get("parent_task_id")
        task = {"id": tid, "type": "task", "chat_id": int(owner_chat_id), "text": text, "depth": depth}
        if parent_id:
            task["parent_task_id"] = parent_id
        ctx.enqueue_task(task)
        ctx.send_with_budget(int(owner_chat_id), f"🗓️ Scheduled task {tid}: {desc}")
        ctx.persist_queue_snapshot(reason="schedule_task_event")


def _handle_cancel_task(evt: Dict[str, Any], ctx: Any) -> None:
    task_id = str(evt.get("task_id") or "").strip()
    st = ctx.load_state()
    owner_chat_id = st.get("owner_chat_id")
    ok = ctx.cancel_task_by_id(task_id) if task_id else False
    if owner_chat_id:
        ctx.send_with_budget(
            int(owner_chat_id),
            f"{'✅' if ok else '❌'} cancel {task_id or '?'} (event)",
        )


def _handle_toggle_evolution(evt: Dict[str, Any], ctx: Any) -> None:
    """Toggle evolution mode from LLM tool call."""
    enabled = bool(evt.get("enabled"))
    st = ctx.load_state()
    st["evolution_mode_enabled"] = enabled
    ctx.save_state(st)
    if not enabled:
        ctx.PENDING[:] = [t for t in ctx.PENDING if str(t.get("type")) != "evolution"]
        ctx.sort_pending()
        ctx.persist_queue_snapshot(reason="evolve_off_via_tool")
    if st.get("owner_chat_id"):
        state_str = "ON" if enabled else "OFF"
        ctx.send_with_budget(int(st["owner_chat_id"]), f"🧬 Evolution: {state_str} (via agent tool)")


def _handle_toggle_consciousness(evt: Dict[str, Any], ctx: Any) -> None:
    """Toggle background consciousness from LLM tool call."""
    action = str(evt.get("action") or "status")
    if action in ("start", "on"):
        result = ctx.consciousness.start()
    elif action in ("stop", "off"):
        result = ctx.consciousness.stop()
    else:
        status = "running" if ctx.consciousness.is_running else "stopped"
        result = f"Background consciousness: {status}"
    st = ctx.load_state()
    if st.get("owner_chat_id"):
        ctx.send_with_budget(int(st["owner_chat_id"]), f"🧠 {result}")


def _handle_send_photo(evt: Dict[str, Any], ctx: Any) -> None:
    """Send a photo (base64 PNG) to a Telegram chat."""
    import base64 as b64mod
    try:
        chat_id = int(evt.get("chat_id") or 0)
        image_b64 = str(evt.get("image_base64") or "")
        caption = str(evt.get("caption") or "")
        if not chat_id or not image_b64:
            return
        photo_bytes = b64mod.b64decode(image_b64)
        ok, err = ctx.TG.send_photo(chat_id, photo_bytes, caption=caption)
        if not ok:
            ctx.append_jsonl(
                ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "type": "send_photo_error",
                    "chat_id": chat_id, "error": err,
                },
            )
    except Exception as e:
        ctx.append_jsonl(
            ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "type": "send_photo_event_error", "error": repr(e),
            },
        )


def _handle_owner_message_injected(evt: Dict[str, Any], ctx: Any) -> None:
    """Log owner_message_injected to events.jsonl for health invariant #5 (duplicate processing)."""
    from ouroboros.utils import utc_now_iso
    try:
        ctx.append_jsonl(ctx.DRIVE_ROOT / "logs" / "events.jsonl", {
            "ts": evt.get("ts", utc_now_iso()),
            "type": "owner_message_injected",
            "task_id": evt.get("task_id", ""),
            "text": evt.get("text", "")[:200],
        })
    except Exception:
        log.warning("Failed to log owner_message_injected event", exc_info=True)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------
EVENT_HANDLERS = {
    "llm_usage": _handle_llm_usage,
    "task_heartbeat": _handle_task_heartbeat,
    "typing_start": _handle_typing_start,
    "send_message": _handle_send_message,
    "task_done": _handle_task_done,
    "task_metrics": _handle_task_metrics,
    "review_request": _handle_review_request,
    "restart_request": _handle_restart_request,
    "promote_to_stable": _handle_promote_to_stable,
    "schedule_task": _handle_schedule_task,
    "cancel_task": _handle_cancel_task,
    "send_photo": _handle_send_photo,
    "toggle_evolution": _handle_toggle_evolution,
    "toggle_consciousness": _handle_toggle_consciousness,
    "owner_message_injected": _handle_owner_message_injected,
}


def dispatch_event(evt: Dict[str, Any], ctx: Any) -> None:
    """Dispatch a single worker event to its handler."""
    if not isinstance(evt, dict):
        ctx.append_jsonl(
            ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "type": "invalid_worker_event",
                "error": "event is not dict",
                "event_repr": repr(evt)[:1000],
            },
        )
        return

    event_type = str(evt.get("type") or "").strip()
    if not event_type:
        ctx.append_jsonl(
            ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "type": "invalid_worker_event",
                "error": "missing event.type",
                "event_repr": repr(evt)[:1000],
            },
        )
        return

    handler = EVENT_HANDLERS.get(event_type)
    if handler is None:
        ctx.append_jsonl(
            ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "type": "unknown_worker_event",
                "event_type": event_type,
                "event_repr": repr(evt)[:1000],
            },
        )
        return

    try:
        handler(evt, ctx)
    except Exception as e:
        ctx.append_jsonl(
            ctx.DRIVE_ROOT / "logs" / "supervisor.jsonl",
            {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "type": "worker_event_handler_error",
                "event_type": event_type,
                "error": repr(e),
            },
        )
