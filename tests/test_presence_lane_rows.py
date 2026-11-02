"""A Presence turn's conversation-lane row is its place's conversation, exactly as the in-server author is (#2604).

Every Host Presence turn is a RUNNING conversation-lane row from its admission (``presence_carrier.dispatch_presence_turn``
-> ``conversation_lane.admit_turn``), published in the queue snapshot while its author runs. The roster, the mind's
live works, the Main router and the peer receipt read that snapshot; #2604's own tests run the author inline with no
lane row. Here a REAL author (``tests.test_presence_project_activity._live_turn``) runs while the lane admits and
dispatches its row (review N2): an unmoved turn is never a roster row, is the live work "busy with its own
conversation", is not steerable by the Main router and answers a peer with the Presence receipt; a moved one is
listed once, as the place's turn (``inline_presence``), never as an ordinary pooled root.
"""

from __future__ import annotations

from tests.test_presence_project_activity import PROJECT, _live_turn, _move, _routing_ctx, root as root


def _admit_lane_row(task):
    """The lane's own admission and dispatch of this turn's row, as ``dispatch_presence_turn`` leaves it."""
    from ouroboros.queue_observation import load_queue_document
    from supervisor import conversation_lane, queue, workers
    from tests.conversation_carrier_support import install_lane_spare

    spare = install_lane_spare(workers, 90)
    seed = {"id": task["id"], "type": "presence", "source": "presence", "chat_id": task["chat_id"],
            "text": str(task.get("text") or ""), "_is_direct_chat": True, "_presence_turn": True,
            "metadata": dict(task.get("metadata") or {})}
    with queue._queue_lock, workers._repo_writer_gate_lock:
        meta = conversation_lane.admit_turn(seed, chat_id=int(seed["chat_id"]), client_message_id="presence-event",
                                            quiet=True, reply_expected=False)
    assert meta["lane"] == "conversation" and meta["worker_id"] == spare.wid  # a real, dispatched lane row
    published = {row["id"]: row for row in load_queue_document(workers.DRIVE_ROOT)["running"]}
    assert published[task["id"]]["task"]["_execution_owner"]["kind"] == "presence"  # what the readers see


def test_a_presence_turns_lane_row_is_its_places_conversation_in_every_mind_view(root):
    from ouroboros.live_works import live_works_present
    from ouroboros.peer_roster import independent_roots, render_roster_note
    from ouroboros.server_routing_context import addressable_roots
    from ouroboros.tools.core import _forward_to_worker
    from ouroboros.tools.registry import ToolContext

    seen = {}

    def scenario(author, task):
        _admit_lane_row(task)
        seen["listed"] = [row["task_id"] for row in independent_roots(root)["roots"]]
        seen["addressable"] = [row for row in addressable_roots(_routing_ctx(root)).rows() if row["task_id"] == task["id"]]
        present = live_works_present(root)
        seen["works"] = [work["task_id"] for work in present["works"]]
        seen["other"] = [row for row in present.get("other_live_work") or [] if row["work_id"] == task["id"]]
        peer = ToolContext(repo_dir=root, drive_root=root, budget_drive_root=root, task_id="peer-root", task_metadata={})
        seen["receipt"] = _forward_to_worker(peer, task["id"], "peer context")
        _move(root, task["id"], PROJECT)
        roster = independent_roots(root)
        seen["moved"] = [row for row in roster["roots"] if row["task_id"] == task["id"]]
        seen["note"] = render_roster_note(roster)
        seen["moved_addressable"] = [row for row in addressable_roots(_routing_ctx(root)).rows()
                                     if row["task_id"] == task["id"]]
        present = live_works_present(root)
        seen["moved_works"] = ([work["task_id"] for work in present["works"]],
                               [row["work_id"] for row in present.get("other_live_work") or []])

    task_id = _live_turn(root, scenario)
    # Unmoved: no roster row, not the Main router's to steer, the place's live turn, the Presence receipt.
    assert task_id not in seen["listed"] and seen["addressable"] == [] and task_id not in seen["works"]
    [row] = seen["other"]
    assert row["kind"] == "inline_presence" and row["root_status"] == "running"
    assert "not that its model read it" in seen["receipt"] and '"kind": "presence"' in seen["receipt"]
    assert "it reads it at its next checkpoint" not in seen["receipt"]
    # Moved: one listed Project work, the place's turn, never the lane row as an ordinary root.
    [moved] = seen["moved"]
    assert moved["inline_presence"] is True and moved["project_id"] == PROJECT
    assert f"{task_id} · " in seen["note"] and "live place conversation turn" in seen["note"]
    assert all(row.get("inline_presence") for row in seen["moved_addressable"])  # a local author's own door only
    assert task_id in seen["moved_works"][0] and task_id not in seen["moved_works"][1]
