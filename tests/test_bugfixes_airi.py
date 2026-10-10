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
import pathlib
from types import SimpleNamespace

REPO = pathlib.Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


# ────────── Bug 1 (polish): the retired loop's rows replay honestly ──────────


def test_log_events_has_no_background_loop_projector():
    src = _read("web/modules/log_events.js")
    assert "consciousness_state" not in src
    assert "bg-consciousness" not in src


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


def test_live_card_disclosure_is_explicit_user_owned_state():
    src = _read("web/modules/chat.js")
    assert "const explicitCardExpansion = new Map();" in src  # a room opens at its newest, with no carried disclosure
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
        src.index("export function createTimelineAnchors")
    ]
    assert renderer.count("const pinned =") == 2
    assert "const prevTop = el.scrollTop, newest = el.lastElementChild;" in renderer
    # Only a new newest line follows; a disclosure or late full output keeps its place.
    assert "el.scrollTop = pinned && el.lastElementChild !== newest ? el.scrollHeight : prevTop;" in renderer
    assert "record.root.dataset.expanded === '1' && pinned" in renderer


# ───────────────────── Bug 3: reconnect dialogue recovery ───────────────────

def test_reconnect_merges_user_rows_without_clearing_visible_history():
    src = _read("web/modules/chat.js")
    sync = src[src.index("async function syncHistory"):src.index("const readReceipt = createProjectReadReceipt(")]
    replay = src[src.index("function applyHistoryMessages"):src.index("async function syncHistory")]
    add = src[src.index("function addMessage"):src.index("function updateMessageAnnotation")]
    # Reconnect still fetches the canonical source and includes owner dialogue.
    assert "await fetchHistory(null)" in sync
    assert "apiClient.chatHistory({ chatId, cursor, ...options })" in src
    assert "historyPager.acceptRecent(data)" in sync
    assert "if (result.status !== 'applied') applyHistoryMessages(messages, { fromReconnect });" in sync
    assert "includeUser = true, archived = false" in replay
    assert "if (!includeUser && msg.role === 'user') continue;" in replay
    # Keyed reconciliation replaces the former clear-and-rebuild requirement:
    # physical rows dedupe and an offline/local echo is adopted in place.
    assert "opts.historyId ? `history:${opts.historyId}` : legacyKey" in add
    assert "if (messageKey && seenMessageKeys.has(messageKey))" in add
    assert "node.dataset.clientMessageId === clientMessageId" in add
    assert "stampHistoryNode(prior, opts.historyId, opts.historyPosition);" in add
    assert "seenMessageKeys.clear();" not in sync + replay
    assert "messageKeyOrder.length = 0;" not in sync + replay


# ──────────────────── Bug 4a: progress widget host race ──────────────────────

def test_widget_job_poll_merges_full_status_and_clamps():
    src = _read("web/modules/widgets.js")
    assert "clampMonotonicProgress" in src
    assert "progressValueKeys" in src
    assert "...data," in src  # full flat merge surfaces value_key (e.g. progress_pct)
    # The broken cherry-pick of the wrong key must be gone.
    assert "progress: data.progress," not in src
