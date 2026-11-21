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

def test_replay_learns_subagent_lineage_before_merging_card_rows():
    src = _read("web/modules/chat.js")
    # One helper learns lineage from both replay rows and live final frames.
    assert "function learnSubagentLineage(msg)" in src
    lineage_pass = "for (const msg of messages) {\n                    learnSubagentLineage(msg);"
    assert lineage_pass in src
    history = src[src.index("function applyHistoryMessages"):src.index("async function syncHistory")]
    assert history.index(lineage_pass) < history.index("admitCardMetadata(msg)")
    # A page may contain a child whose parent is already represented elsewhere.
    # Keep those bindings during replay; only final instance disposal clears them.
    disposal = src[src.index("destroy() {"):]
    cleared = disposal[disposal.index("for (const held of ["):disposal.index("]) held.clear();")]
    for collection in ("subagentChildParents", "subagentTerminalChildren"):
        assert f"{collection}.clear()" not in history and "held.clear()" not in history
        assert f"{collection}," in cleared  # destroy's one release of the buffered collections
    fanout = src[src.index("onWs('chat'"):src.index("onWs('message_annotation'")]
    # Live lineage must be known before progress or a final can resolve a
    # child card; ephemeral registration now lives in the early reference seam.
    assert fanout.index("learnSubagentLineage(msg);") < fanout.index("updateLiveCardFromProgressMessage(msg,")
    assert fanout.index("learnSubagentLineage(msg);") < fanout.index("routeSubagentFinalMessageToCard(explicitTaskId, msg)")
    # A lineage-known child is minted as its parent's nested card by whichever
    # path reaches it first (#636) — no sticky force writer: the parent's anchor
    # follows the child's frame (when this room holds the parent's own card, A5)
    # and the predicate always admits a child block.
    assert "if (rawTs) reanchorTaskCard(liveCardRecords.get(parentId), rawTs);" in src
    assert "const record = getSubagentCardRecord(childId, parentId, role, rawTs);" in src
    assert "if (!record || record.isSubagent) return true;" in src
    # A child is locked terminal from EITHER a terminal subagent event OR a
    # genuinely-settled server task_terminal_status; interrupted stays retryable.
    assert "const replayTerminal = msg.task_terminal_status" in src
    assert "const replayTerminal = msg.task_terminal_status && taskDoneIsTerminal(msg);" in src
    assert "if (replayTerminal || ['completed', 'completed_warn', 'failed', 'cancelled', 'rejected'].includes(event)) {" in src
    assert "subagentTerminalChildren.add(childId);" in src
