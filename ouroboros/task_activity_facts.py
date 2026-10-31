"""Incremental card display facts of a task (the narration line, the tool count) in the task's own
card row (``store_task_cards``), never in the body and never activity authority.

The delivery seam records visible model narration; the loop publishes its existing tool-result
counter. Each is one row-only merge (count never down, newest line wins) that reads and writes no
body and moves no document signature; it waits at most the acquisition slice, so a worker loop or
the supervisor's delivery consumer never queues behind a long store hold for a display fact, and
a busy or refusing store keeps the previous card. The census reads the card from the hot row it
already reads. Old tasks without these facts keep an empty narration line.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def _record(root, task_id, facts):
    if not root or not task_id or not facts:
        return
    from ouroboros import store, store_task_cards
    from ouroboros._usage_wait import ACQUISITION_SLICE_SEC

    try:
        with store.acquisition_budget(ACQUISITION_SLICE_SEC), store.open_store(root, write=True) as txn:
            store_task_cards.merge(txn, str(task_id), facts)  # nothing for a task without a lifecycle row
    except store.StoreError:
        log.debug("Task card display facts of %s kept as they were: the store is busy or unavailable", task_id,
                  exc_info=True)
    except Exception:
        log.warning("Task card display facts of %s not recorded", task_id, exc_info=True)


def record_narration(root, row):
    """Only an appended, explicitly model-authored progress row supplies this line."""
    if row.get("narration") is not True:
        return
    text = str(row.get("content", row.get("text", "")) or "")
    # The worker's presentation prefix is removed by the room card as well.
    if text.startswith("💬"):
        text = text[1:].lstrip()
    if text.strip():
        _record(root, row.get("task_id"), {"last_line": text, "last_line_at": str(row.get("ts") or "")})


def record_tool_count(ctx, activity):
    """Publish the counter already accumulated by task_pacing, including restored usage."""
    _record(getattr(ctx, "budget_drive_root", None) or getattr(ctx, "drive_root", None),
            getattr(ctx, "task_id", ""), {"tool_calls": sum(row["calls"] for row in activity.values())})


def card_facts(task):
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    calls = task.get("tool_calls")
    return {"last_line": str(task.get("last_line") or ""),
            "last_line_at": str(task.get("last_line_at") or ""),
            "tool_calls": calls if type(calls) is int and calls >= 0 else 0,
            "detached": bool(task.get("detached_at") or metadata.get("detached_at"))}


def enrich_activity(activity, facts, projects):
    """Apply cached task facts after exact-work room binding has been resolved."""
    card = facts.get("card") or card_facts({})
    detached = bool(activity.get("detached") or card["detached"])
    activity.update(card, detached=detached)
    project = projects.get(str(activity.get("project_id") or "")) or {}
    activity["project_name"] = str(project.get("name") or activity.get("project_name") or "")
