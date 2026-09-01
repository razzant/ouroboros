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
    src = _read("web/modules/chat.js")
    # Cleared on rebuild so stale cross-session lineage cannot persist.
    assert "subagentChildParents.clear();" in src
    assert "subagentTerminalChildren.clear();" in src
    # One helper learns lineage from both replay rows and live final frames.
    assert "function learnSubagentLineage(msg)" in src
    assert "for (const msg of messages) learnSubagentLineage(msg);" in src
    fanout = src[src.index("onWs('chat'"):src.index("onWs('message_annotation'")]
    assert fanout.index("learnSubagentLineage(msg);") < fanout.index("registerEphemeralDecisionFrame(msg);")
    assert "forceTaskCard(childId, rawTs);" in src
    # A child is locked terminal from EITHER a terminal subagent event OR a
    # genuinely-settled server task_terminal_status; interrupted stays retryable.
    assert "const replayTerminal = msg.task_terminal_status" in src
    assert "? taskDoneIsTerminal({ ...msg, status: String(msg.task_terminal_status) })" in src
    assert "if (replayTerminal || ['completed', 'completed_warn', 'failed', 'cancelled', 'rejected'].includes(event)) {" in src
    assert "subagentTerminalChildren.add(childId);" in src


def test_progress_dedup_uses_full_array_not_last_item():
    """applyLiveCardState must dedup a progress line against the WHOLE card, not
    just the last item — otherwise a background syncHistory re-feeds historical
    progress and the 'Notes' count grows without bound (BUGREPORT-panic-working-notes).
    """
    src = _read("web/modules/chat.js")
    # full-array dedup
    assert "const existingIdx = record.items.findIndex((it) => it.dedupeKey === syntheticKey);" in src
    # the old last-item-only check is gone
    assert "record.items[lastIdx].dedupeKey === syntheticKey ? lastIdx : -1" not in src
    # a historical re-feed (found, not the last item) is skipped, not re-appended
    assert "timelineUpdate = 'duplicate-skip';" in src
