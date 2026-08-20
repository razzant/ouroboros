"""Regression tests for subagent-reliability fixes (PR-D).

  * #18 — child artifact rebasing no longer fails silently: an unreachable source
    is flagged (copy_status=failed) instead of keeping a broken parent path.
  * #1/#2/#20 — on reload/reconnect the chat replay reconstructs subagent
    lineage + terminal state from durable history, so finished child cards
    finalize instead of sticking on "working".
"""

from __future__ import annotations

import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


# ───────────────────────── #18: artifact rebase failure ─────────────────────

def test_artifact_rebase_flags_missing_source(tmp_path):
    from ouroboros.headless import _copy_child_artifacts_to_parent

    parent = tmp_path / "parent"
    child = tmp_path / "child"
    parent.mkdir(parents=True)
    child.mkdir(parents=True)
    real = child / "out.txt"
    real.write_text("hello", encoding="utf-8")

    artifacts = [
        {"path": str(real), "name": "out.txt"},
        {"path": "does_not_exist.txt", "name": "ghost"},
    ]
    out = _copy_child_artifacts_to_parent(parent, "task1234", child, artifacts)

    # A real child-drive file is rebased into the parent store (path changes).
    real_item = next(a for a in out if a["name"] == "out.txt")
    assert real_item.get("copy_status") != "failed"
    assert real_item["path"] != str(real)

    # An unreachable source is FLAGGED, not silently kept with a broken path.
    ghost = next(a for a in out if a["name"] == "ghost")
    assert ghost.get("copy_status") == "failed"
    assert ghost.get("copy_error")


# ──────────────── #1/#2/#20: subagent lineage rebuilt on replay ──────────────

def test_replay_clears_and_rebuilds_subagent_lineage():
    # The replay pre-pass moved to chat_history_sync.js (wave D).
    src = _read("web/modules/chat_history_sync.js")
    # Cleared on rebuild so stale cross-session lineage cannot persist.
    assert "subagentChildParents.clear();" in src
    assert "subagentTerminalChildren.clear();" in src
    # Pre-pass reconstructs lineage + terminal set from durable history.
    assert "if (String(msg.delegation_role || '').toLowerCase() !== 'subagent') continue;" in src
    assert "setSubagentParent(childId, { parentId, role:" in src
    # A child is locked terminal from EITHER a terminal subagent event OR the
    # server task_terminal_status, so it cannot be revived by parent heartbeats.
    assert "if (msg.task_terminal_status || ['completed', 'completed_warn', 'failed', 'cancelled', 'rejected'].includes(ev)) {" in src
    assert "subagentTerminalChildren.add(childId);" in src


def test_progress_dedup_uses_full_array_not_last_item():
    """applyLiveCardState must dedup a progress line against the WHOLE card, not
    just the last item — otherwise a background syncHistory re-feeds historical
    progress and the 'Notes' count grows without bound (BUGREPORT-panic-working-notes).
    """
    # applyLiveCardState moved to chat_live_cards.js (wave D).
    src = _read("web/modules/chat_live_cards.js")
    # full-array dedup
    assert "const existingIdx = record.items.findIndex((it) => it.dedupeKey === syntheticKey);" in src
    # the old last-item-only check is gone
    assert "record.items[lastIdx].dedupeKey === syntheticKey ? lastIdx : -1" not in src
    # a historical re-feed (found, not the last item) is skipped, not re-appended
    assert "timelineUpdate = 'duplicate-skip';" in src
