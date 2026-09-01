"""Regression tests for chat/widget bugs fixed in this change.

Covers three issues:

  * Bug 1 (polish) — the background-consciousness live card lingered in a
    "thinking" phase forever because it has no task_result to mark it terminal.
    The global status is already guarded upstream (isBackgroundTaskId); this adds
    a structured end-of-cycle marker + history replay annotation so the card
    itself finalizes to "done".
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
import pathlib
from types import SimpleNamespace

REPO = pathlib.Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


# ───────────────── Bug 1 (polish): background card finalizes ─────────────────

def test_consciousness_emits_structured_idle_marker_not_text_matched():
    src = _read("ouroboros/consciousness.py")
    assert "_emit_cycle_idle" in src
    assert "consciousness_state" in src
    # The marker is structured, never a regex on log text (BIBLE P5).
    assert "Going back to sleep" not in src


def test_log_events_derives_bg_card_phase_from_marker():
    src = _read("web/modules/log_events.js")
    assert "consciousness_state" in src
    assert "bgTerminal" in src


def test_history_marks_bg_consciousness_terminal_on_replay(tmp_path):
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
    assert bg, "bg-consciousness progress should replay"
    latest = max(bg, key=lambda m: m["ts"])
    assert latest.get("task_terminal_status") == "done"
    # Earlier entries stay non-terminal so the card animates while a cycle runs.
    assert not bg[0].get("task_terminal_status")


def test_live_card_disclosure_is_explicit_user_owned_state():
    src = _read("web/modules/chat.js")
    assert "const explicitCardExpansion = new Map();" in src
    assert "explicitCardExpansion.set(record.groupId, nowExpanded);" in src
    assert "explicitCardExpansion.has(normalizedGroupId)" in src
    assert "explicitCardExpansion.get(normalizedGroupId)" in src
    assert "if (existing && !explicitCardExpansion.has(childId))" in src
    assert "setLiveCardExpanded(record, nestedSubagentsExpanded);" in src
    assert "stickyExpandedSlots" not in src


def test_live_card_timeline_only_follows_when_pinned():
    src = _read("web/modules/chat_render_batch.js")
    renderer = src[
        src.index("export function createLiveCardTimelineRenderer"):
        src.index('// "Load older" quota escalation ladder')
    ]
    assert renderer.count("const pinned =") == 2
    assert "const prevTop = el.scrollTop;" in renderer
    assert "el.scrollTop = pinned ? el.scrollHeight : prevTop;" in renderer
    assert "record.root.dataset.expanded === '1' && pinned" in renderer


# ───────────────────── Bug 3: reconnect feed rebuild ─────────────────────────

def test_reconnect_rebuilds_feed_and_clears_dedupe():
    src = _read("web/modules/chat.js")
    assert "const renderUser = includeUser || fromReconnect || offlineBootstrapPainted;" in src
    assert "seenMessageKeys.clear();" in src
    assert "messageKeyOrder.length = 0;" in src
    assert "querySelectorAll('.chat-bubble')" in src
    # User messages are restored on reconnect (not skipped).
    assert "if (!renderUser && msg.role === 'user') continue;" in src


# ──────────────────── Bug 4a: progress widget host race ──────────────────────

def test_widget_job_poll_merges_full_status_and_clamps():
    src = _read("web/modules/widgets.js")
    assert "clampMonotonicProgress" in src
    assert "progressValueKeys" in src
    assert "...data," in src  # full flat merge surfaces value_key (e.g. progress_pct)
    # The broken cherry-pick of the wrong key must be gone.
    assert "progress: data.progress," not in src
