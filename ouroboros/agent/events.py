"""
Ouroboros agent event emission.

Handles emitting end-of-task events to supervisor and storing task results.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import time
from typing import Any, Dict, List

from ouroboros.utils import utc_now_iso, append_jsonl, truncate_for_log

log = logging.getLogger(__name__)


def emit_task_results(
    pending_events: List[Dict[str, Any]],
    task: Dict[str, Any],
    text: str,
    usage: Dict[str, Any],
    llm_trace: Dict[str, Any],
    start_time: float,
    drive_logs: pathlib.Path,
    drive_root: pathlib.Path,
) -> None:
    """Emit all end-of-task events to supervisor."""
    # NOTE: per-round llm_usage events are already emitted in loop.py
    # (_emit_llm_usage_event). Do NOT emit an aggregate llm_usage here —
    # that would double-count in update_budget_from_usage.
    # Cost/token summaries are carried by task_metrics and task_done events.

    pending_events.append({
        "type": "send_message", "chat_id": task["chat_id"],
        "text": text or "\u200b", "log_text": text or "",
        "format": "markdown",
        "task_id": task.get("id"), "ts": utc_now_iso(),
    })

    duration_sec = round(time.time() - start_time, 3)
    n_tool_calls = len(llm_trace.get("tool_calls", []))
    n_tool_errors = sum(1 for tc in llm_trace.get("tool_calls", [])
                        if isinstance(tc, dict) and tc.get("is_error"))
    try:
        append_jsonl(drive_logs / "events.jsonl", {
            "ts": utc_now_iso(), "type": "task_eval", "ok": True,
            "task_id": task.get("id"), "task_type": task.get("type"),
            "duration_sec": duration_sec,
            "tool_calls": n_tool_calls,
            "tool_errors": n_tool_errors,
            "response_len": len(text),
        })
    except Exception:
        log.warning("Failed to log task eval event", exc_info=True)
        pass

    pending_events.append({
        "type": "task_metrics",
        "task_id": task.get("id"), "task_type": task.get("type"),
        "duration_sec": duration_sec,
        "tool_calls": n_tool_calls, "tool_errors": n_tool_errors,
        "cost_usd": round(float(usage.get("cost") or 0), 6),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_rounds": int(usage.get("rounds") or 0),
        "ts": utc_now_iso(),
    })

    pending_events.append({
        "type": "task_done",
        "task_id": task.get("id"),
        "task_type": task.get("type"),
        "cost_usd": round(float(usage.get("cost") or 0), 6),
        "total_rounds": int(usage.get("rounds") or 0),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "ts": utc_now_iso(),
    })
    append_jsonl(drive_logs / "events.jsonl", {
        "ts": utc_now_iso(),
        "type": "task_done",
        "task_id": task.get("id"),
        "task_type": task.get("type"),
        "cost_usd": round(float(usage.get("cost") or 0), 6),
        "total_rounds": int(usage.get("rounds") or 0),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    })

    # Store task result for parent task retrieval
    try:
        results_dir = pathlib.Path(drive_root) / "task_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        result_data = {
            "task_id": task.get("id"),
            "parent_task_id": task.get("parent_task_id"),
            "status": "completed",
            "result": text[:4000] if text else "",
            "cost_usd": round(float(usage.get("cost") or 0), 6),
            "total_rounds": int(usage.get("rounds") or 0),
            "ts": utc_now_iso(),
        }
        result_file = results_dir / f"{task.get('id')}.json"
        tmp_file = results_dir / f"{task.get('id')}.json.tmp"
        tmp_file.write_text(json.dumps(result_data, ensure_ascii=False, indent=2))
        os.rename(tmp_file, result_file)
    except Exception as e:
        log.warning("Failed to store task result: %s", e)
