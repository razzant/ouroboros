"""#708: the room an evolution cleanup's row reaches, and the later writers its short fact survives.

The row takes the order the reaper's incident notice takes (C4, ``notification_chat_route``): the
task's Project room, else the room its own record names (chat 0, the hidden partition, is a room),
else the owner's chat for a task that names none. An unreadable record is not an absent one: the
task's own live queue row names its room, and without one the row is not owed (never Main). Each scenario
runs the real lifecycle over a real temporary Git repository, the real outbox and send handler, and
reads each room's history: the history endpoint for Main (1) and the rooms a page opens, and for
the hidden partition, which no page requests (the endpoint reads ``chat_id=0`` as Main), the same
assembly the endpoint runs. The later writers are the real ones that rewrite a settled
evolution result: the terminal projection, the terminal cost refresh and the post-task checkpoint.
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

from tests import test_evolution_cleanup_disclosure as disclosure
from tests.test_evolution_cleanup_disclosure import CHAT, TASK, _task_done, dirty, rows, stash_oids

cycle = disclosure.cycle
HIDDEN = 0  # the hidden partition, ``contracts/chat_id_policy.HIDDEN_CHAT_ID``
MAIN = 1  # the owner chat 7 of the fixture is no Project room, so Main shows its rows too


def hidden_rows(scene):
    """The cleanup rows of the hidden partition, through the endpoint's own assembly."""
    from ouroboros.gateway.history import _assemble_history_response

    body = _assemble_history_response(scene.data, HIDDEN, 50, 50, None, None)
    return [row for row in json.loads(body)["messages"] if str(row.get("card_row_id") or "").startswith("evolution-cleanup:")]


def live_rooms(scene):
    """The rooms the bridge broadcast this row to, live."""
    return [frame["chat_id"] for frame in scene.frames if str(frame.get("card_row_id") or "").startswith("evolution-cleanup:")]


def test_a_task_in_the_hidden_room_keeps_its_row_there_and_main_gets_none(cycle):
    dirty(cycle)

    cycle.end(chat_id=HIDDEN)
    cycle.deliver()

    assert live_rooms(cycle) == [HIDDEN]
    [row] = hidden_rows(cycle)
    assert row["task_id"] == TASK and stash_oids(cycle)[0] in row["text"]
    assert rows(cycle, CHAT) == [] and rows(cycle, MAIN) == []


def test_a_project_task_row_reaches_its_project_room(cycle):
    from ouroboros.projects_registry import bind_task_to_project, project_chat_for_task

    bind_task_to_project(cycle.data, TASK, "evolution-room", origin={"absent": "system"})
    project_chat = project_chat_for_task(cycle.data, TASK)
    assert project_chat not in (HIDDEN, MAIN, CHAT)
    dirty(cycle)

    cycle.end()
    cycle.deliver()

    assert live_rooms(cycle) == [project_chat]
    [row] = rows(cycle, project_chat)
    assert row["task_id"] == TASK and stash_oids(cycle)[0] in row["text"]
    assert rows(cycle, CHAT) == [] and rows(cycle, MAIN) == [] and hidden_rows(cycle) == []


def test_a_task_without_a_record_reaches_the_owner_chat_alone(cycle):
    """A cycle stopped before any result existed names no room and has no card: the owner's chat carries it."""
    from supervisor import evolution_cleanup

    dirty(cycle)
    tx = {"transaction_id": "orphan-tx", "base_head": cycle.base, "task_id": "never-recorded"}
    evolution_cleanup._cleanup_worktree_after_cycle(tx, "never-recorded")
    assert tx["cleanup_status"] == "stashed_dirty"
    evolution_cleanup.publish_cleanup(tx, "never-recorded", cycle.data)
    cycle.deliver()

    assert live_rooms(cycle) == [CHAT]
    [row] = rows(cycle)
    assert row["task_id"] == "never-recorded" and tx["cleanup_stash_oids"][0] in row["text"]
    assert len(rows(cycle, MAIN)) == 1 and hidden_rows(cycle) == []


def _unreadable_record(monkeypatch):
    from ouroboros import task_results

    readable = task_results.read_document

    def unreadable(root, task_id, **kwargs):
        if task_id == TASK:
            raise OSError("task store unreadable")
        return readable(root, task_id, **kwargs)

    monkeypatch.setattr(task_results, "read_document", unreadable)
    return lambda: monkeypatch.setattr(task_results, "read_document", readable)


def test_an_unreadable_record_without_a_live_row_owes_no_guessed_row(cycle, monkeypatch, caplog):
    """Unknown is not absent: no guess at Main. The campaign keeps what moved and the log names it; a later
    publication of the same transaction (here the terminal's duplicate) owes the row once the record reads."""
    dirty(cycle)
    restore = _unreadable_record(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="supervisor.evolution_cleanup"):
        first = cycle.end(chat_id=HIDDEN)
    cycle.deliver()

    oids = stash_oids(cycle)
    assert first["transaction"]["cleanup_stash_oids"] == oids != []
    assert live_rooms(cycle) == [] and rows(cycle, CHAT) == [] and rows(cycle, MAIN) == [] and hidden_rows(cycle) == []
    [warning] = [record.getMessage() for record in caplog.records if "is not owed" in record.getMessage()]
    assert oids[0] in warning and "waits" not in warning, "the log names what moved and promises no replay"
    assert cycle.campaign()["history"][-1]["transaction"]["cleanup_stash_oids"] == oids

    restore()
    replay = cycle.end(chat_id=HIDDEN)
    cycle.deliver()

    assert replay["replay"] is True and live_rooms(cycle) == [HIDDEN]
    [row] = hidden_rows(cycle)
    assert oids[0] in row["text"] and rows(cycle, CHAT) == [] and rows(cycle, MAIN) == []


def test_an_unreadable_record_takes_the_room_its_live_row_names(cycle, monkeypatch):
    """The task's own queue row (the reaper's room source) still names its room while it settles: the row is
    owed there at once. A send that cannot read the record yet fails, and the outbox's own replay delivers the
    owed row once it reads; a hidden-partition task stays in the hidden partition."""
    from supervisor import queue
    from supervisor import terminal_delivery as td

    dirty(cycle)
    queue.RUNNING[TASK] = {"task": {"id": TASK, "type": "evolution", "chat_id": HIDDEN}}
    restore = _unreadable_record(monkeypatch)

    cycle.end(chat_id=HIDDEN)
    [owed] = [row for row in td.pending_deliveries(cycle.data) if row.get("system_type") == "evolution_cleanup"]
    assert owed["chat_id"] == HIDDEN and stash_oids(cycle)[0] in owed["text"]
    queue.RUNNING.pop(TASK)  # the dispatcher pops the settling row before the queued send is handled
    cycle.deliver()
    assert live_rooms(cycle) == [] and len(td.pending_deliveries(cycle.data)) == 1, "still owed, not lost"

    restore()
    monkeypatch.setattr(td, "_REPLAY_MIN_AGE_SEC", 0)
    td.replay_pending_deliveries(cycle.data, event_queue=cycle.bus)
    cycle.deliver()

    assert live_rooms(cycle) == [HIDDEN] and td.pending_deliveries(cycle.data) == []
    [row] = hidden_rows(cycle)
    assert stash_oids(cycle)[0] in row["text"] and rows(cycle, CHAT) == [] and rows(cycle, MAIN) == []


def test_the_short_fact_survives_the_real_later_writers_of_its_result(cycle):
    """After the evolution terminal records the fact, the result is rewritten by the terminal projection's
    readiness and receipt, the terminal cost refresh (here with a real settled usage row, so it really
    writes) and the post-task checkpoint. Each writes its own fields, so the short line stays."""
    from ouroboros import usage_accounting as ua
    from ouroboros.post_task_checkpoint import set_root_post_task_checkpoint
    from ouroboros.project_dialogue import _completion_verdict, append_terminal_task_projection
    from ouroboros.task_results import load_task_result
    from supervisor.events_task_done import _refresh_terminal_task_cost

    dirty(cycle)
    event = _task_done(cycle, None)
    fact = event["outcome_axes"]["execution"]["evolution_cleanup"]
    assert fact["stash_oids"] == stash_oids(cycle) != []

    assert append_terminal_task_projection(cycle.data, TASK, {"id": TASK, "chat_id": CHAT},
                                           load_task_result(cycle.data, TASK), event)
    projected = load_task_result(cycle.data, TASK)
    assert projected.get("canonical_terminal_projection") or projected.get("canonical_terminal_projection_ready")

    reservation = ua.reserve_attempt(ua.AttemptRequest(
        model="openai/gpt-5.2", provider="openai", reservation_usd=0.4, drive_root=cycle.data,
        task_id=TASK, root_task_id=TASK, global_limit_usd=10, root_limit_usd=10))
    ua.mark_dispatched(reservation)
    ua.settle_attempt(reservation, {"prompt_tokens": 10, "completion_tokens": 5}, cost_usd=0.4, cost_final=True)
    _refresh_terminal_task_cost(cycle.data, TASK)
    events = (cycle.data / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert any(json.loads(line).get("type") == "task_cost_finalized" for line in events), "the refresh really wrote"
    assert load_task_result(cycle.data, TASK)["accounted_upper_bound_usd"] != projected.get("accounted_upper_bound_usd")

    stored = set_root_post_task_checkpoint(SimpleNamespace(drive_root=cycle.data),
                                           {"id": TASK, "root_task_id": TASK, "budget_drive_root": str(cycle.data)},
                                           "completed")
    assert stored["root_phase_checkpoint"]["post_task_synthesis"] == "completed"

    final = load_task_result(cycle.data, TASK)
    assert final["outcome_axes"]["execution"]["evolution_cleanup"] == fact
    assert fact["line"] in _completion_verdict(final, {})
