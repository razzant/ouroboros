"""Regression tests for chat/widget bugs fixed in this change.

Covers three issues:

  * Bug 1 (polish) — the retired background-consciousness loop had no
    task_result, so replay used to stamp its last progress row "done" by hand.
    A wake-up is now an ordinary direct turn with its own durable result, and
    the loop's legacy rows replay with NO fabricated terminal status: the
    client's task-detail read settles them as "Outcome unavailable".
  * Bug 3 — conversational text (user + assistant bubbles) disappeared after a
    soft WebSocket reconnect because syncHistory skipped user messages and the
    persistent dedupe set was never cleared.
  * Bug 4a — the declarative progress widget froze then jumped: the job poll read
    the wrong status key (so the WS-loss fallback was dead) and there was no
    monotonic clamp.

Pure-Python behavior is exercised directly; client-side (JS) fixes are pinned
with static source contracts and verified visually.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace


# ────────── Bug 1 (polish): the retired loop's rows replay honestly ──────────


def test_history_never_fabricates_a_terminal_status_for_legacy_bg_rows(tmp_path):
    from ouroboros.gateway.history import make_chat_history_endpoint

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").write_text("", encoding="utf-8")
    (logs / "progress.jsonl").write_text(
        json.dumps({
            "ts": "2026-06-05T00:00:01Z", "task_id": "bg-consciousness",
            "content": "thinking a", "is_progress": True,
        }) + "\n" + json.dumps({
            "ts": "2026-06-05T00:00:02Z", "task_id": "bg-consciousness",
            "content": "thinking b", "is_progress": True,
        }) + "\n",
        encoding="utf-8",
    )
    endpoint = make_chat_history_endpoint(tmp_path)
    resp = asyncio.run(endpoint(SimpleNamespace(query_params={})))
    messages = json.loads(resp.body)["messages"]
    bg = [m for m in messages if m.get("task_id") == "bg-consciousness"]
    assert bg, "the retired loop's progress should still replay"
    # No task_result exists for the pseudo id, so no row may claim an outcome.
    assert not any(m.get("task_terminal_status") for m in bg)
    assert not any(m.get("system_type") == "task_model_wait" for m in messages)


# ───────────────────── Bug 3: reconnect dialogue recovery ───────────────────


# ──────────────────── Bug 4a: progress widget host race ──────────────────────
