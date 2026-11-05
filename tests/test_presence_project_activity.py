"""A live inline Presence author is seen, addressed and stopped through the doors it really has.

The cases run a REAL ``run_presence_turn`` whose author is a real ``OuroborosAgent`` (its own admission lock
and ``_bind_turn``; only the model loop is replaced, so nothing calls a provider), moved by the owner's
``move_task_to_project`` transaction while it runs, or observed inside its agent factory before its first row. A
real direct actor, registered and moved the same way, is the positive control of each door: the harness reaches
it, so a refusal of the author is the door's own answer. Faults are injected only where a real one cannot be held.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue

PROJECT, OTHER = "activity-room", "activity-other"
REPO = pathlib.Path(__file__).resolve().parents[1]


class _Bridge:
    """The supervisor bridge the move and relay publication call: records nothing, sends nothing."""

    def __getattr__(self, _name):
        return lambda *_a, **_k: None


@pytest.fixture
def root(tmp_path, monkeypatch):
    from ouroboros import delegate_custody
    from supervisor import message_bus

    _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "_BRIDGE", _Bridge())
    delegate_custody._CUSTODY.clear()
    yield tmp_path
    delegate_custody._CUSTODY.clear()


def _move(root, task_id, project_id):
    from ouroboros.gateway.projects import _owner_task_origin
    from ouroboros.project_placement import move_task_to_project, placement_basis
    from supervisor import workers

    outcome = move_task_to_project(root, task_id, project_id, name=project_id, initiated_by="owner",
                                   origin=_owner_task_origin(root, task_id),
                                   basis=placement_basis(root, task_id, project_id),
                                   running=workers.RUNNING, pending=workers.PENDING)
    return int(outcome["chat_id"])


def _live_turn(root, scenario):
    """Run one real inline turn; ``scenario(author, task)`` runs while it is live. Returns its task id."""
    from ouroboros.agent import Env, OuroborosAgent
    from ouroboros.presence_runner import run_presence_turn
    from tests.test_presence_runner import _admission, _event, _terminal

    seen = {}

    class Author(OuroborosAgent):
        def handle_task(self, task):
            self._bind_turn(task)
            seen["task_id"] = task["id"]
            try:
                scenario(self, task)
            finally:
                _terminal(root, task, "")
                self._release_turn()
            return [{"type": "presence_result", "outcome": "silent", "text": "", "work_ref": ""}]

    def factory(**kwargs):
        return Author(Env(repo_dir=pathlib.Path(kwargs["repo_dir"]), drive_root=pathlib.Path(kwargs["drive_root"])))

    run_presence_turn(admission=_admission(), event=_event(), repo_dir=root, drive_root=root, agent_factory=factory)
    return seen["task_id"]


class _DirectActor:
    """A real direct actor: registered, bound to its turn, its RUNNING row on disk."""

    def __init__(self, root, task_id="direct-control"):
        from ouroboros.agent import make_agent
        from ouroboros.task_results import write_task_result
        from ouroboros.task_status import execution_owner_record
        from supervisor.active_activity import get_direct_activity_registry

        self.task_id = task_id
        self.agent = make_agent(repo_dir=str(root), drive_root=str(root), event_queue=None)
        task = {"id": task_id, "type": "task", "chat_id": 1, "text": "direct work", "_is_direct_chat": True,
                "metadata": {}}
        self.agent._bind_turn(task)
        write_task_result(root, task_id, "running", chat_id=1, _is_direct_chat=True, source="web",
                          execution_owner=execution_owner_record(root, task, "direct"), result="Task is running.")
        get_direct_activity_registry().register(task_id, 1, kind="direct_chat", phase="working", actor=self.agent)

    def close(self):
        from supervisor.active_activity import get_direct_activity_registry

        get_direct_activity_registry().unregister(self.task_id)
        self.agent._release_turn()


def _state(root):
    """The census and work projection exactly as ``gateway/state._state_snapshot`` builds them."""
    from ouroboros.gateway import state as gs
    from ouroboros.gateway.work_projection import works_snapshot
    from ouroboros.projects_registry import all_task_project_bindings

    bindings = all_task_project_bindings(root, strict=True)
    availability, owners, facts = {"complete": True}, [], {}
    census = gs._chat_activities_snapshot_safe(root, bindings, availability=availability,
                                               direct_turns=gs._direct_turns_snapshot_safe(availability=availability),
                                               decision_owners=owners, work_facts=facts)
    works = works_snapshot(root, owners, facts, complete=availability["complete"], bindings=bindings)
    return census, works, availability["complete"]


def _row(census, task_id):
    return next((row for row in census if row.get("activity_id") == task_id), None)


def _work(works, task_id):
    return next((work for work in works["works"] if (work.get("episode") or {}).get("task_id") == task_id), None)


def _actions(work):
    return sorted({control["action"] for control in work["controls"]})


def _routing_ctx(root):
    from ouroboros.utils import append_jsonl
    from supervisor import queue as q
    from supervisor import workers

    return SimpleNamespace(DRIVE_ROOT=root, RUNNING=workers.RUNNING, PENDING=workers.PENDING, bridge=None,
                           persist_queue_snapshot=q.persist_queue_snapshot, BUDGET_ROOT_FENCES=q.BUDGET_ROOT_FENCES,
                           append_jsonl=append_jsonl, send_with_budget=lambda *_a, **_k: None)


def _mailbox(root, task_id):
    from ouroboros.owner_mailbox import _mailbox_path

    path = _mailbox_path(root, task_id)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return [json.loads(line) for line in lines if line.strip()]


def _tell(root, target, label, **extra):
    """One ``tell_task`` relay through the supervisor's own door; returns the receipt it wrote."""
    from supervisor.steering import _handle_steer_task

    _handle_steer_task({"target_task_id": target, "message": f"note {label}", "quotes": [], "relay_id": label,
                        "routing_token": label, "chat_id": 1, "client_message_id": f"agent-receipt-{label}",
                        "issuer": {"kind": "task", "task_id": "teller-task"}, **extra}, _routing_ctx(root))
    path = root / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    receipt = [row for row in rows if row.get("type") == "task_message_routed" and row.get("target_task_id") == target
               and row.get("routing_token") == label]
    return {key: receipt[-1].get(key) for key in ("status", "reason")} if receipt else None


def test_census_lists_the_author_in_its_room_with_stop_only_and_no_queue_row(root):
    from ouroboros.presence_runner import presence_chat_id
    from supervisor import queue as q
    from tests.test_presence_runner import _event

    seen = {}

    def scenario(author, task):
        control = _DirectActor(root)
        try:
            seen["unmoved"] = _state(root)
            seen["queue"] = (task["id"] in q.RUNNING, any(row.get("id") == task["id"] for row in q.PENDING))
            q.persist_queue_snapshot(reason="activity")
            snapshot = json.loads((root / "state" / "queue_snapshot.json").read_text(encoding="utf-8"))
            seen["snapshot_ids"] = {str(row.get("id")) for key in ("running", "pending")
                                    for row in (snapshot.get("payload", snapshot).get(key) or [])}
            seen["chat"] = _move(root, task["id"], PROJECT)
            _move(root, control.task_id, OTHER)
            seen["moved"] = _state(root)
            # The restart handover is untouched: no direct-roots fragment is written any more (retired), and the
            # direct turns name the control only.
            from supervisor.workers import direct_chat_turns
            seen["handover"] = (["fragment"] if (root / "state" / "direct_roots.json").exists() else [],
                                [turn["id"] for turn in direct_chat_turns()])
        finally:
            control.close()

    task_id = _live_turn(root, scenario)
    census, works, complete = seen["unmoved"]
    row = _row(census, task_id)
    assert complete is True
    assert row["kind"] == "presence" and row["phase"] == "working" and row["cancelable"] is True
    assert row["chat_id"] == presence_chat_id(_event().conversation_key) and row["project_id"] == ""
    assert _actions(_work(works, task_id)) == ["stop"]
    assert seen["queue"] == (False, False) and task_id not in seen["snapshot_ids"]  # state A: no fake queue row
    census, works, _complete = seen["moved"]
    row, work = _row(census, task_id), _work(works, task_id)
    assert (row["chat_id"], row["project_id"]) == (seen["chat"], PROJECT)
    # Moved to a Project as text, the author takes that work's Pause beside Stop; never Hurry or Wrap up.
    assert work["room"] == seen["chat"] and _actions(work) == ["pause", "stop"] and row.get("presence_pause") is True
    # The positive control, read in the same body: the filter is the author's, never a phase's.
    assert {"pause", "stop"} <= set(_actions(_work(works, "direct-control")))
    # The restart handover is untouched by the census: the control stays a live direct turn; the direct-roots
    # fragment names nothing on this tree (every live focus is a RUNNING row of the queue snapshot; a
    # registry-only control has no lane row), the retired fragment's empty residual.
    assert seen["handover"] == ([], ["direct-control"])
    # Main's own room gains nothing: the author's work is its conversation's, then its Project's.
    assert all(str(work.get("room")) != "1" for work in works["works"] if work["work_id"] == task_id)
    # Once the turn ended, the census drops the author and nothing else stands for it.
    census, works, _complete = _state(root)
    assert _row(census, task_id) is None and _work(works, task_id) is None


def test_a_tail_only_work_and_the_mind_roster_follow_the_binding_not_the_stored_room(root):
    from ouroboros import delegate_custody as custody
    from ouroboros.peer_roster import independent_roots, render_roster_note, unlisted_live_work

    seen = {}

    def scenario(author, task):
        seen["chat"] = _move(root, task["id"], PROJECT)
        roster = independent_roots(root)
        seen["listed"] = [row for row in roster["roots"] if row["task_id"] == task["id"]]
        seen["note"] = render_roster_note(roster)
        custody.record_started(root, custody.RunCustody(run_id="run-tail", task_id=task["id"],
                                                        root_task_id=task["id"], route_id="codex", model="m",
                                                        ledger_root=str(root)))
        custody._CUSTODY.clear()

    task_id = _live_turn(root, scenario)
    listed = seen["listed"]
    assert len(listed) == 1 and listed[0]["inline_presence"] is True
    assert (listed[0]["chat_id"], listed[0]["project_id"]) == (seen["chat"], PROJECT)
    assert f"{task_id} · " in seen["note"] and "live place conversation turn" in seen["note"]
    # The author ended; its delegated run still runs: a work of tails alone, in its Project.
    _census, works, _complete = _state(root)
    work = next(work for work in works["works"] if work["work_id"] == task_id)
    assert work["room"] == seen["chat"] and work["project_id"] == PROJECT
    assert [tail for tail in work.get("tails") or [] if tail]
    other = [row for row in unlisted_live_work(root, set()) if row["work_id"] == task_id]
    assert other and other[0]["room"] == {"chat_id": seen["chat"], "project_id": PROJECT}


def test_answer_upkeep_of_a_moved_turn_runs_in_its_project_not_its_stored_room(root, monkeypatch):
    from ouroboros import post_task_checkpoint as post_task
    from ouroboros.task_results import write_task_result

    moved, kept = _DirectActor(root), _DirectActor(root, "direct-kept")
    chat = _move(root, moved.task_id, PROJECT)
    for actor in (moved, kept):  # both answered; only their answer upkeep still runs, under no census row
        actor.close()
        write_task_result(root, actor.task_id, "completed", result="answer")
    monkeypatch.setattr(post_task, "POST_TASK_SYNTHESIS_INFLIGHT", {})
    for actor in (moved, kept):
        post_task.POST_TASK_SYNTHESIS_INFLIGHT[(str(root.resolve()), actor.task_id)] = None
    census, works, _complete = _state(root)
    assert _row(census, moved.task_id) is None and _row(census, kept.task_id) is None
    rooms = {work["work_id"]: (work["room"], work["project_id"]) for work in works["works"]}
    # The stored row still names Main (chat 1): the move rewrote no source chat, the binding places it.
    assert rooms[moved.task_id] == (chat, PROJECT)
    assert rooms[kept.task_id] == (1, "")  # the unmoved control keeps its own room
    assert str(chat) in works["rooms"]


def test_a_peer_message_to_a_moved_author_keeps_the_presence_receipt(root):
    from ouroboros.tools.core import _forward_to_worker
    from ouroboros.tools.registry import ToolContext

    seen = {}

    def scenario(author, task):
        peer = ToolContext(repo_dir=root, drive_root=root, budget_drive_root=root, task_id="peer-root",
                           task_metadata={})
        seen["unmoved"] = _forward_to_worker(peer, task["id"], "peer context before the move")
        _move(root, task["id"], PROJECT)
        seen["moved"] = _forward_to_worker(peer, task["id"], "peer context after the move")
        seen["mail"] = [(row.get("provenance"), row.get("source_task_id")) for row in _mailbox(root, task["id"])]

    _live_turn(root, scenario)
    for receipt in (seen["unmoved"], seen["moved"]):
        assert "written to its mailbox" in receipt and "independent_task, never owner text" in receipt
        assert "not that its model read it" in receipt and '"kind": "presence"' in receipt
        assert "it reads it at its next checkpoint" not in receipt
    assert seen["mail"] == [("independent_task", "peer-root")] * 2


def test_an_unmoved_author_is_known_to_every_mind_but_no_roster_row_and_reads_no_room(root):
    from ouroboros.live_works import compact_other_live_work, live_works_present
    from ouroboros.peer_roster import independent_roots
    from ouroboros.presence_runner import presence_chat_id
    from ouroboros.server_routing_context import addressable_roots
    from tests.test_presence_runner import _event

    seen = {}

    def scenario(author, task):
        seen["listed"] = [row["task_id"] for row in independent_roots(root)["roots"]]
        seen["addressable"] = [row["task_id"] for row in addressable_roots(_routing_ctx(root)).rows()]
        present = live_works_present(root)
        seen["works"] = [work["task_id"] for work in present["works"]]
        seen["other"] = [row for row in present.get("other_live_work") or [] if row["work_id"] == task["id"]]
        seen["compact"] = compact_other_live_work(present.get("other_live_work"))
        seen["own_view"] = [row["work_id"] for row in live_works_present(root, exclude=task["id"]).get("other_live_work") or []]
        _move(root, task["id"], PROJECT)
        present = live_works_present(root)
        seen["moved"] = ([work["task_id"] for work in present["works"]],
                         [row["work_id"] for row in present.get("other_live_work") or []])

    task_id = _live_turn(root, scenario)
    assert task_id not in seen["listed"] and task_id not in seen["addressable"] and task_id not in seen["works"]
    [row] = seen["other"]
    assert row["kind"] == "inline_presence" and row["root_status"] == "running" and "direct_chat" not in row
    assert row["room"] == {"chat_id": presence_chat_id(_event().conversation_key), "project_id": ""}
    assert row in seen["compact"]["unabridged"]  # never folded into the counted tails
    assert task_id not in seen["own_view"]  # every other mind knows it; its own context never lists it as other work
    assert task_id in seen["moved"][0] and task_id not in seen["moved"][1]  # moved: a listed Project work
    assert task_id not in [row["work_id"] for row in live_works_present(root).get("other_live_work") or []]


def test_owner_input_in_the_project_reaches_its_one_accepting_author(root, monkeypatch):
    import queue as stdlib_queue

    from ouroboros import conversation_lease
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.presence_runner import presence_execution_witness
    from ouroboros.server_owner_routing import _route_project_chat_to_running_task
    from ouroboros.server_routing_context import addressable_roots

    seen = {}

    def route(chat, label):
        return _route_project_chat_to_running_task(_routing_ctx(root), chat, f"owner words {label}",
                                                   client_message_id=f"owner-{label}")

    def scenario(author, task):
        tid = task["id"]
        chat = seen["chat"] = _move(root, tid, PROJECT)
        rows = addressable_roots(_routing_ctx(root)).rows(chat)
        seen["reader"] = [(row["task_id"], row.get("inline_presence"), row.get("room_reader")) for row in rows]
        seen["single"] = route(chat, "single")
        control = _DirectActor(root)  # a second reader of the same room: the agent decides, nothing delivers
        try:
            _move(root, control.task_id, PROJECT)
            seen["two"] = route(chat, "two")
        finally:
            control.close()
        with author._owner_message_admission_lock:
            author._accepting_owner_messages = False  # closing: still live, takes no mail
        seen["closed"] = (route(chat, "closed"), presence_execution_witness(tid, drive_root=root))
        with author._owner_message_admission_lock:
            author._accepting_owner_messages = True
        real, actor = conversation_lease.live_presence_roots, conversation_lease.presence_author_agent
        monkeypatch.setattr(conversation_lease, "live_presence_roots", lambda drive_root: {
            "roots": [{"task_id": tid, "local": False}], "complete": True})  # as another process reads it
        # Read from another process, as every carried author is (#2654): no actor here, only its durable intake,
        # which the delivery re-checks under the mail lock. Closed (its final drain), nothing lands; open, the words do.
        monkeypatch.setattr(conversation_lease, "presence_author_agent", lambda drive_root, author: None)
        conversation_lease.close_author_intake(root, tid)
        seen["remote_closed"] = route(chat, "remote-closed")
        conversation_lease.open_author_intake(root, tid, task["metadata"]["presence"]["event"]["conversation_key"])
        seen["remote"] = route(chat, "remote")
        monkeypatch.setattr(conversation_lease, "live_presence_roots", real)
        monkeypatch.setattr(conversation_lease, "presence_author_agent", actor)
        seen["mail"] = [(row.get("kind"), row.get("client_message_id"), row.get("text")) for row in _mailbox(root, tid)]
        # The author's own loop drain reads it as the owner's words, not as a peer's context.
        owner_ctx = SimpleNamespace(task_id=tid, task_attempt=1, drive_root=root, budget_drive_root=root,
                                    current_chat_id=chat, task_metadata=task["metadata"], _owner_directives=[],
                                    event_queue=None)
        _drain_incoming_messages([], stdlib_queue.Queue(), root, tid, None, set(), owner_ctx=owner_ctx)
        seen["directives"] = [row.get("content") for row in owner_ctx._owner_directives]

    task_id = _live_turn(root, scenario)
    assert seen["reader"] == [(task_id, True, True)]
    assert seen["single"] == task_id
    assert seen["two"] == "" and seen["closed"] == ("", True) and seen["remote_closed"] == ""
    assert seen["remote"] == task_id
    assert seen["mail"] == [("owner_text", "owner-single", "owner words single"),
                            ("owner_text", "owner-remote", "owner words remote")]
    assert seen["directives"] == ["owner words single", "owner words remote"]


def test_tell_task_reaches_any_local_accepting_author_and_refuses_the_rest_truthfully(root, monkeypatch):
    from ouroboros import task_results
    from ouroboros.conversation_lease import live_author
    from ouroboros.project_dialogue import routing_refusal_cause
    from tests.test_presence_runner import _event

    seen = {}

    def scenario(author, task):
        tid = task["id"]
        control = _DirectActor(root)
        try:
            seen["control"] = _tell(root, control.task_id, "control")
        finally:
            control.close()
        # place_guard's advice names this author from the conversation it holds (place_guard.py:134-137).
        seen["guard_names"] = live_author(_event().conversation_key, drive_root=root) == tid
        seen["unmoved"] = _tell(root, tid, "unmoved")
        seen["same_binding"] = _tell(root, tid, "same-binding", presence_binding_id="1" * 32)
        seen["other_binding"] = _tell(root, tid, "other-binding", presence_binding_id="2" * 32)
        with author._owner_message_admission_lock:
            author._accepting_owner_messages = False  # closing: its turn still runs
        seen["closed"] = _tell(root, tid, "closed")
        with author._owner_message_admission_lock:
            author._accepting_owner_messages, author._busy = True, False  # out of its loop, turn still live
        seen["out_of_loop"] = _tell(root, tid, "out-of-loop")
        with author._owner_message_admission_lock:
            author._busy = True
        real = task_results.load_task_result

        def unreadable(drive_root, task_id, *args, strict=False, **kwargs):
            if task_id == tid and strict:
                raise ValueError("this probe cannot read the row")
            return real(drive_root, task_id, *args, strict=strict, **kwargs)

        monkeypatch.setattr(task_results, "load_task_result", unreadable)
        seen["unreadable"] = _tell(root, tid, "unreadable")
        monkeypatch.setattr(task_results, "load_task_result", real)
        _move(root, tid, PROJECT)
        seen["moved"] = _tell(root, tid, "moved")
        seen["mail"] = [row["msg_id"] for row in _mailbox(root, tid) if row.get("kind") == "relay_entry"]

    task_id = _live_turn(root, scenario)
    written = {"status": "written", "reason": ""}
    assert seen["control"] == written and seen["guard_names"] is True
    assert seen["unmoved"] == written and seen["moved"] == written and seen["same_binding"] == written
    assert seen["other_binding"] == {"status": "refused", "reason": "presence_work_not_related"}
    # A live turn is never called finished or gone: each refusal names what is true.
    assert seen["closed"] == seen["out_of_loop"] == {"status": "refused", "reason": "presence_author_closed"}
    assert seen["unreadable"] == {"status": "refused", "reason": "presence_author_unreadable"}
    assert seen["mail"] == [f"relay:{label}:{task_id}" for label in ("unmoved", "same-binding", "moved")]
    # The control: once the turn ended, it is no longer running, as the owner reads it.
    assert _tell(root, task_id, "ended") == {"status": "refused", "reason": "target_unknown"}
    assert routing_refusal_cause("steer_task", "needs_manual_target", "presence_author_closed") == (
        "Not delivered: that conversation turn is not taking messages now")


_ELSEWHERE = r'''
import json, sys
from pathlib import Path
import pytest
root, tid = Path(sys.argv[1]), sys.argv[2]
from tests._budget_pause_exact_helpers import _install_queue
_install_queue(root, pytest.MonkeyPatch())
import ouroboros.config
ouroboros.config.DATA_DIR = root
from supervisor import message_bus
from tests.test_presence_project_activity import _Bridge, _tell
message_bus._BRIDGE = _Bridge()
from ouroboros.conversation_lease import live_presence_roots
from ouroboros.live_works import live_works_present
other = live_works_present(root).get("other_live_work") or []
print(json.dumps({"census": live_presence_roots(root), "tell": _tell(root, tid, "elsewhere"),
                  "other": [row.get("kind") for row in other if row.get("work_id") == tid]}))
'''


def test_a_turn_in_another_process_is_known_there_and_refused_as_elsewhere_not_ended(root):
    seen = {}

    def scenario(author, task):
        env = {**os.environ, "OUROBOROS_DATA_DIR": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
        child = subprocess.run([sys.executable, "-c", _ELSEWHERE, str(root), task["id"]], cwd=str(REPO), env=env,
                               text=True, capture_output=True, timeout=90)
        assert child.returncode == 0, child.stdout + child.stderr
        seen.update(json.loads(child.stdout.strip().splitlines()[-1]))
        seen["mail"] = [row for row in _mailbox(root, task["id"]) if row.get("kind") == "relay_entry"]

    task_id = _live_turn(root, scenario)
    # The other process holds no endpoint for the author: it observes its live stack and says so.
    assert seen["census"] == {"roots": [{"task_id": task_id, "local": False}], "complete": True}
    assert seen["tell"] == {"status": "refused", "reason": "presence_author_elsewhere"}
    assert seen["other"] == ["inline_presence"] and seen["mail"] == []


def test_update_focus_belongs_to_a_moved_author_only(root):
    from ouroboros.tools.project_journal import _update_focus
    from ouroboros.utils import append_jsonl

    seen = {}
    source = {"reader": "journal_read", "project_id": PROJECT}

    def ctx(author, task, events):
        return SimpleNamespace(task_id=task["id"], drive_root=root, is_direct_chat=True,
                               task_metadata=task["metadata"], task_contract=task.get("task_contract") or {},
                               owner_message_admission_lock=author._owner_message_admission_lock,
                               event_queue=SimpleNamespace(put_nowait=events.append))

    def scenario(author, task):
        events = []
        seen["unmoved"] = _update_focus(ctx(author, task, events), "Drafting the reply", source)
        _move(root, task["id"], PROJECT)
        append_jsonl(root / "projects" / PROJECT / "journal.jsonl", {"kind": "note", "text": "reply facts"})
        seen["moved"] = _update_focus(ctx(author, task, events), "Drafting the reply", source)
        seen["events"] = [event["type"] for event in events]
        census, works, _complete = _state(root)
        seen["work_focus"] = (_work(works, task["id"]).get("focus") or {}).get("text")

    task_id = _live_turn(root, scenario)
    assert "TOOL_FORBIDDEN" in seen["unmoved"]
    assert seen["moved"].startswith(f"OK: focus[{task_id}]") and seen["events"] == ["task_focus_updated"]
    assert seen["work_focus"] == "Drafting the reply"


def _pause_doors(root, task_id, label):
    """The owner's HTTP door and the mind's ``pause_task`` door, for one id: their answers, as each caller reads them."""
    import asyncio

    from starlette.requests import Request

    from ouroboros.gateway.task_pause import api_task_pause
    from supervisor.events_mind_controls import _apply

    body = json.dumps({"request_id": f"pause-{label}", "actor": "owner"}).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": f"/api/tasks/{task_id}/pause", "headers": [],
                       "path_params": {"task_id": task_id},
                       "app": SimpleNamespace(state=SimpleNamespace(drive_root=root))}, receive)
    http = asyncio.run(api_task_pause(request))
    return ((http.status_code, json.loads(http.body).get("reason_code")),
            _apply({"type": "mind_task_control", "action": "pause_task", "task_id": task_id,
                    "actor_task_id": "main-mind-actor", "action_nonce": f"nonce-{label}",
                    "reason": "the owner asked to pause that conversation"}))


def test_pause_of_a_live_inline_turn_refuses_typed_at_every_door_and_changes_nothing(root, monkeypatch):
    from ouroboros import task_results
    from ouroboros.cancel_intents import cancel_pending
    from ouroboros.owner_pause import read_fence
    from ouroboros.presence_runner import presence_execution_witness
    from ouroboros.task_results import load_task_result
    from supervisor import queue as q
    from supervisor.owner_pause_control import request_owner_pause

    seen = {}

    def child(task_id):
        return {"id": "child-helper", "type": "task", "chat_id": 1, "root_task_id": task_id, "parent_task_id": task_id,
                "delegation_role": "subagent", "text": "child", "_attempt": 1, "queued_at": time.time()}

    def scenario(author, task):
        tid = task["id"]
        q.PENDING.append(child(tid))
        before = load_task_result(root, tid, strict=True)
        seen["witness"] = presence_execution_witness(tid, drive_root=root)
        seen["live"] = request_owner_pause(tid, request_id="pause-live")
        seen["http"], seen["mind"] = _pause_doors(root, tid, "live")

        def row_unreadable(drive_root, task_id, *args, strict=False, **kwargs):
            if task_id == tid and strict:
                raise ValueError("this probe cannot read the row")
            return load_task_result(drive_root, task_id, *args, strict=strict, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(task_results, "load_task_result", row_unreadable)
            seen["row_unreadable"] = request_owner_pause(tid, request_id="pause-unreadable")
        seen["unchanged"] = load_task_result(root, tid, strict=True) == before
        seen["fence"], seen["latch"] = read_fence(root, tid), tid in q.BUDGET_ROOT_FENCES
        seen["stop"] = cancel_pending(root, tid, strict=True)

    task_id = _live_turn(root, scenario)
    assert seen["witness"] is True
    # A turn every reader lists as live is never told "not live": each door says it has no Pause yet.
    assert seen["live"] == seen["mind"] == {"ok": False, "error": "presence_pause_unsupported"}
    assert seen["http"] == (409, "presence_pause_unsupported")
    assert seen["row_unreadable"] == {"ok": False, "error": "presence_pause_unsupported"}  # its stack still witnesses it
    # The refusal wrote nothing: no fence, no latch, no Stop, the row byte-for-byte what it was.
    assert not seen["fence"] and seen["latch"] is False and seen["stop"] is False and seen["unchanged"] is True
    # The control: once the author ended, the same tree IS an answered root's late work, and Pause takes it.
    ended = request_owner_pause(task_id, request_id="pause-ended")
    assert ended["ok"] is True and ended["root_task_id"] == task_id


def test_pause_says_unknown_when_nothing_can_witness_the_stack_and_not_live_only_when_it_is_gone(root, monkeypatch):
    from ouroboros import conversation_lease
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import write_task_result

    # A retained unsettled row whose stack this process does not hold: only the census can answer for it.
    write_task_result(root, "retained-turn", "running", chat_id=1, _is_direct_chat=True, source="presence",
                      execution_owner={"kind": "presence"}, result="Task is running.")

    def unreadable(*, drive_root=None):
        raise OSError("this probe cannot read the census")

    with monkeypatch.context() as patch:
        patch.setattr(conversation_lease, "live_authors", unreadable)
        unknown = _pause_doors(root, "retained-turn", "unknown")
    assert unknown == ((503, "presence_liveness_unknown"), {"ok": False, "error": "presence_liveness_unknown"})
    # The same census readable and empty: truly no such live work. Neither read wrote a fence.
    assert _pause_doors(root, "retained-turn", "gone") == ((404, "task_not_live"), {"ok": False, "error": "task_not_live"})
    assert not read_fence(root, "retained-turn")
    assert _pause_doors(root, "no-such-task", "nobody")[0] == (404, "task_not_live")


def test_the_author_census_observes_without_reclaiming_and_names_what_it_cannot_read(tmp_path):
    from ouroboros.conversation_lease import live_authors, live_presence_roots
    from ouroboros.platform_layer import file_lock_exclusive

    gate = tmp_path / "state" / "presence_turn_gate"
    gate.mkdir(parents=True)
    ended = gate / "author-ended.live"
    ended.write_text(json.dumps({"task_id": "ended-turn"}), encoding="utf-8")  # its stack ended: nobody holds it

    def names():
        return sorted(path.name for path in gate.iterdir())

    # Without the census lock no witness is read safely, and reading creates none.
    assert live_presence_roots(tmp_path) == {"roots": [], "complete": False}
    assert names() == ["author-ended.live"]
    (gate / "authors-census.lock").touch()
    held = os.open(gate / "author-held.live", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        file_lock_exclusive(held)
        os.write(held, b"{not json")  # a held witness this read cannot name
        before = names()
        assert live_presence_roots(tmp_path) == {"roots": [], "complete": False}
        assert names() == before  # nothing reclaimed, nothing created
        os.ftruncate(held, 0)
        os.lseek(held, 0, os.SEEK_SET)
        os.write(held, json.dumps({"task_id": "held-turn"}).encode())
        assert live_presence_roots(tmp_path) == {"roots": [{"task_id": "held-turn", "local": False}], "complete": True}
        assert names() == before
        # The same owner's cleanup branch still reclaims the ended witness, and only it.
        assert live_authors(drive_root=tmp_path) == ["held-turn"]
        assert not ended.exists() and (gate / "author-held.live").exists()
    finally:
        os.close(held)


def _starting_turn(root, scenario, *, running=None):
    """One REAL ``run_presence_turn`` observed inside its agent factory: the witness is published, no row is written.
    ``scenario(task_id)`` runs there; ``running(task)``, if given, once its RUNNING row stands. Returns its task id."""
    from ouroboros.conversation_lease import live_presence_roots
    from ouroboros.presence_runner import run_presence_turn
    from tests.test_presence_runner import _admission, _event, _terminal

    seen = {}

    class Agent:
        def handle_task(self, task):
            if running is not None:
                running(task)
            _terminal(root, task, "")
            return [{"type": "presence_result", "outcome": "silent", "text": "", "work_ref": ""}]

    def factory(**_kwargs):
        [fact] = live_presence_roots(root)["roots"]
        seen["task_id"] = fact["task_id"]
        scenario(fact["task_id"])
        return Agent()

    result = run_presence_turn(admission=_admission(), event=_event(), repo_dir=root, drive_root=root, agent_factory=factory)
    assert result.task_id == seen["task_id"]
    return seen["task_id"]


def _api_state(root):
    """The owner's whole ``GET /api/state`` body, through its real handler."""
    import asyncio

    from starlette.requests import Request

    from ouroboros.gateway.state import api_state

    request = Request({"type": "http", "method": "GET", "path": "/api/state", "query_string": b"", "headers": [],
                       "app": SimpleNamespace(state=SimpleNamespace(drive_root=root, repo_dir=REPO))})
    return json.loads(asyncio.run(api_state(request)).body)


def test_a_starting_turn_is_preparing_in_its_room_with_no_door_never_an_empty_complete_census(root):
    from ouroboros.gateway.tasks import _record_stop
    from ouroboros.live_works import live_works_present
    from ouroboros.presence_runner import presence_chat_id
    from ouroboros.task_results import load_task_result
    from supervisor.owner_pause_control import request_owner_pause
    from tests.test_presence_runner import _event

    seen = {}

    def starting(tid):
        seen["stored"] = load_task_result(root, tid, strict=True)
        seen["state"] = _state(root)
        seen["body"] = _api_state(root)
        seen["present"] = live_works_present(root)
        seen["tell"] = _tell(root, tid, "starting")
        seen["mail"] = _mailbox(root, tid)
        seen["pause"] = request_owner_pause(tid, request_id="pause-starting")
        seen["stop"] = _record_stop(tid, cascade=True, stop_action_id="stop-starting")
        seen["after"] = load_task_result(root, tid, strict=True)

    def running(task):
        seen["running"] = _state(root)

    task_id = _starting_turn(root, starting, running=running)
    # Observed, nothing written for it: no canonical row was invented, by the read or by the refused doors.
    assert seen["stored"] is None and seen["after"] is None and seen["mail"] == []
    census, works, complete = seen["state"]
    row, work = _row(census, task_id), _work(works, task_id)
    assert complete is True and works["works_complete"] is True
    assert row["kind"] == "presence" and row["phase"] == "preparing" and row["cancelable"] is False
    assert row["chat_id"] == presence_chat_id(_event().conversation_key) and row["project_id"] == ""
    assert work["activity"] == "working" and work["room"] == row["chat_id"]
    # The door really refuses that id (``_record_stop``), so the work offers nothing it would refuse.
    assert work["controls"] == [] and seen["stop"] is None
    # The owner's whole body says the same: no consumer of this body meets a row it cannot read.
    body = seen["body"]
    assert body["active_chat_activities_complete"] is True and body["works_complete"] is True
    assert _row(body["active_chat_activities"], task_id)["phase"] == "preparing" and body["rooms"][str(row["chat_id"])]["activity"] == "working"
    assert task_id not in {turn.get("activity_id") for turn in body["active_direct_turns"]}
    [mind] = [turn for turn in seen["present"]["other_live_work"] if turn["work_id"] == task_id]
    assert mind["kind"] == "inline_presence" and mind["root_status"] == "preparing" and "unavailable" not in mind
    assert mind["room"] == {"chat_id": row["chat_id"], "project_id": ""}
    # Never told it has ended: it takes no mail yet, and it has no Pause.
    assert seen["tell"] == {"status": "refused", "reason": "presence_author_closed"}
    assert seen["pause"] == {"ok": False, "error": "presence_pause_unsupported"}
    # The positive control: its first durable write opens the one door it has.
    census, works, complete = seen["running"]
    assert complete is True and _row(census, task_id)["phase"] == "working" and _row(census, task_id)["cancelable"] is True
    assert _actions(_work(works, task_id)) == ["stop"]
    # Ended: nothing stands for it, and the census is whole again.
    census, works, complete = _state(root)
    assert complete is True and _row(census, task_id) is None and _work(works, task_id) is None


def test_a_recorded_stop_is_said_beside_working_on_the_census_and_the_work_until_the_turn_ends(root):
    from ouroboros.cancel_intents import cancel_pending
    from ouroboros.gateway.tasks import _record_stop

    seen = {}

    def scenario(author, task):
        tid = task["id"]
        seen["before"] = _state(root)
        seen["intent"] = _record_stop(tid, cascade=True, stop_action_id="stop-recorded")
        seen["pending"] = cancel_pending(root, tid)
        seen["after"] = _state(root)

    task_id = _live_turn(root, scenario)
    assert seen["pending"] is True and seen["intent"]
    _census, works, _complete = seen["before"]
    assert _work(works, task_id)["because"] == []  # the control: nothing is said before the Stop is recorded
    census, works, complete = seen["after"]
    row, work = _row(census, task_id), _work(works, task_id)
    # The census row says Working with its one door, as before; the work says the recorded Stop beside it.
    assert complete is True and (row["phase"], row["cancelable"]) == ("working", True)
    # It still works (the model round it holds is its own), and the only door it has still answers a harder press.
    assert work["activity"] == "working" and _actions(work) == ["stop"]
    assert [entry["code"] for entry in work["because"]] == ["stop_requested"]
    assert work["because"][0]["text"] == "Stop requested: it ends at its next step"
    # Once the turn ended, nothing stands for it, Stop or not.
    census, works, _complete = _state(root)
    assert _row(census, task_id) is None and _work(works, task_id) is None


def test_a_settled_row_under_a_still_held_witness_is_ended_not_preparing(root):
    """What separates "no row yet" from "already over": the row's own settled status, read through every reader."""
    from ouroboros.conversation_lease import live_presence_roots
    from ouroboros.live_works import live_works_present
    from ouroboros.peer_roster import independent_roots
    from ouroboros.presence_runner import presence_turn_is_live
    from ouroboros.server_routing_context import addressable_roots
    from ouroboros.task_results import write_task_result

    seen = {}

    def running(task):
        tid = task["id"]
        _move(root, tid, PROJECT)
        write_task_result(root, tid, "completed", metadata=task["metadata"], result="done")
        seen["witnessed"] = [fact["task_id"] for fact in live_presence_roots(root)["roots"]]
        seen["live"] = presence_turn_is_live(tid, drive_root=root)
        seen["state"] = _state(root)
        seen["listed"] = [row["task_id"] for row in independent_roots(root)["roots"]]
        seen["other"] = [row for row in live_works_present(root).get("other_live_work") or [] if row["work_id"] == tid]
        seen["addressable"] = [row["task_id"] for row in addressable_roots(_routing_ctx(root)).rows()]
        seen["tell"] = _tell(root, tid, "settled")

    task_id = _starting_turn(root, lambda _tid: None, running=running)
    assert seen["witnessed"] == [task_id] and seen["live"] is True  # the stack is still there, and the row says it ended
    census, works, complete = seen["state"]
    assert complete is True and _row(census, task_id) is None and _work(works, task_id) is None
    assert task_id not in seen["listed"] and task_id not in seen["addressable"] and seen["other"] == []
    assert seen["tell"] == {"status": "refused", "reason": "target_unknown"}


_STARTING_ELSEWHERE = r'''
import json, sys
from pathlib import Path
import pytest
root, tid = Path(sys.argv[1]), sys.argv[2]
from tests._budget_pause_exact_helpers import _install_queue
_install_queue(root, pytest.MonkeyPatch())
import ouroboros.config
ouroboros.config.DATA_DIR = root
from supervisor import message_bus
from tests.test_presence_project_activity import _Bridge, _row, _state, _tell, _work
message_bus._BRIDGE = _Bridge()
from ouroboros.live_works import live_works_present
from supervisor.owner_pause_control import request_owner_pause
census, works, complete = _state(root)
other = live_works_present(root).get("other_live_work") or []
print(json.dumps({"row": _row(census, tid), "work": _work(works, tid), "complete": complete,
                  "works_complete": works["works_complete"], "tell": _tell(root, tid, "elsewhere"),
                  "pause": request_owner_pause(tid, request_id="pause-elsewhere"),
                  "other": [row for row in other if row.get("work_id") == tid]}))
'''


def test_a_starting_turn_in_another_process_is_a_gap_there_never_absent_and_never_ended(root):
    seen = {}

    def starting(tid):
        env = {**os.environ, "OUROBOROS_DATA_DIR": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
        child = subprocess.run([sys.executable, "-c", _STARTING_ELSEWHERE, str(root), tid], cwd=str(REPO), env=env,
                               text=True, capture_output=True, timeout=90)
        assert child.returncode == 0, child.stdout + child.stderr
        seen.update(json.loads(child.stdout.strip().splitlines()[-1]))

    _starting_turn(root, starting)
    # That process knows the stack and not its room: no row it could place, and the body says it is not whole.
    assert seen["row"] is None and seen["work"] is None
    assert seen["complete"] is False and seen["works_complete"] is False
    assert seen["tell"] == {"status": "refused", "reason": "presence_author_elsewhere"}
    assert seen["pause"] == {"ok": False, "error": "presence_pause_unsupported"}
    [mind] = seen["other"]
    assert mind["root_status"] == "preparing" and mind["room"]["chat_id"] is None


def test_the_mind_view_takes_one_reading_of_the_live_turns_and_discloses_its_gaps(root, monkeypatch):
    from ouroboros import conversation_lease, task_results
    from ouroboros.live_works import live_works_present
    from tests.test_live_works_present import _fixture

    _fixture(root)
    seen = {}
    real_census, real_load = conversation_lease.live_presence_roots, task_results.load_task_result

    def scenario(author, task):
        tid = task["id"]
        calls = []

        def counted(drive_root):
            calls.append(drive_root)
            return real_census(drive_root)

        with monkeypatch.context() as patch:
            patch.setattr(conversation_lease, "live_presence_roots", counted)
            seen["healthy"] = live_works_present(root)
        seen["census_reads"] = len(calls)

        def row_unreadable(drive_root, task_id, *args, strict=False, **kwargs):
            if task_id == tid and strict:
                raise ValueError("this probe cannot read the row")
            return real_load(drive_root, task_id, *args, strict=strict, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(task_results, "load_task_result", row_unreadable)
            seen["row_gap"] = live_works_present(root)

        def census_unreadable(_root):
            raise OSError("this probe cannot read the census")

        with monkeypatch.context() as patch:
            patch.setattr(conversation_lease, "_observe_live_stack_authors", census_unreadable)
            seen["census_gap"] = live_works_present(root)
            seen["census_gap_state"] = _state(root)

    task_id = _live_turn(root, scenario)
    assert seen["census_reads"] == 1  # the roster's reading serves the listed and the unlisted turns alike
    assert seen["healthy"]["incomplete"] is False  # the control: only each gap below turns it
    assert [row["root_status"] for row in seen["healthy"]["other_live_work"] if row["work_id"] == task_id] == ["running"]
    [gap] = [row for row in seen["row_gap"]["other_live_work"] if row["work_id"] == task_id]
    assert seen["row_gap"]["incomplete"] is True
    assert gap["root_status"] == "unknown" and gap["unavailable"] == {"task_result": "ValueError"}
    assert seen["census_gap"]["incomplete"] is True
    # The owner's body says the same, and keeps this process's author with its one door.
    census, works, complete = seen["census_gap_state"]
    assert complete is False and works["works_complete"] is False and _actions(_work(works, task_id)) == ["stop"]
    # This process's own author survives a census it cannot read: known local facts are never withdrawn.
    assert [row["work_id"] for row in seen["census_gap"]["other_live_work"] if row["work_id"] == task_id] == [task_id]


def test_routing_keeps_a_local_reader_past_a_foreign_gap_and_makes_its_own_gap_the_snapshots(root, monkeypatch):
    from ouroboros import task_results
    from ouroboros.conversation_lease import live_presence_roots
    from ouroboros.platform_layer import file_lock_exclusive
    from ouroboros.server_owner_routing import _route_project_chat_to_running_task
    from ouroboros.server_routing_context import addressable_roots

    seen = {}
    real_load = task_results.load_task_result

    def scenario(author, task):
        tid = task["id"]
        chat = _move(root, tid, PROJECT)
        gate = root / "state" / "presence_turn_gate"
        foreign = os.open(gate / "author-foreign.live", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            file_lock_exclusive(foreign)  # another process's stack this read cannot name
            os.write(foreign, b"{not json")
            seen["census_complete"] = live_presence_roots(root)["complete"]
            seen["delivered"] = _route_project_chat_to_running_task(_routing_ctx(root), chat, "owner words past the gap",
                                                                    client_message_id="owner-gap")
        finally:
            os.close(foreign)
            (gate / "author-foreign.live").unlink(missing_ok=True)

        def row_unreadable(drive_root, task_id, *args, strict=False, **kwargs):
            if task_id == tid and strict:
                raise ValueError("this probe cannot read the row")
            return real_load(drive_root, task_id, *args, strict=strict, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(task_results, "load_task_result", row_unreadable)
            snapshot = addressable_roots(_routing_ctx(root))
            seen["own_gap"] = (snapshot.gap, snapshot.rows(chat))
        seen["mail"] = [row.get("client_message_id") for row in _mailbox(root, tid)]

    task_id = _live_turn(root, scenario)
    assert seen["census_complete"] is False and seen["delivered"] == task_id and seen["mail"] == ["owner-gap"]
    # An unreadable row of the reader itself: "unknown who reads this room", never "nobody reads it".
    assert seen["own_gap"][0] and seen["own_gap"][1] == []


def test_a_refused_witness_delete_leaves_a_leftover_no_reader_believes_and_other_failures_stay_unknown(tmp_path, monkeypatch):
    import errno

    from ouroboros.conversation_lease import PresenceTurnGate, live_authors, live_presence_roots
    from ouroboros.presence_runner import presence_execution_witness

    gate_dir = tmp_path / "state" / "presence_turn_gate"
    gate = PresenceTurnGate(state_root=tmp_path / "state")
    lease = gate.acquire("conversation-a")
    lease.bind_author("turn-a")
    witness = next(gate_dir.glob("author-*.live"))
    real_unlink = pathlib.Path.unlink

    def refused(error):
        def unlink(self, missing_ok=False):
            if self.name.startswith("author-"):
                raise error
            return real_unlink(self, missing_ok=missing_ok)
        return unlink

    # The author's release while a passive reader held the witness open, as Windows answers it.
    with monkeypatch.context() as patch:
        patch.setattr(pathlib.Path, "unlink", refused(PermissionError(errno.EACCES, "sharing violation")))
        lease.release()
    assert witness.exists()
    assert live_presence_roots(tmp_path) == {"roots": [], "complete": True}  # unlocked: it names no live stack
    # The Stop census reclaims it, and a reclaim refused the same way leaves a whole, honest census.
    with monkeypatch.context() as patch:
        patch.setattr(pathlib.Path, "unlink", refused(PermissionError(errno.EACCES, "sharing violation")))
        assert live_authors(drive_root=tmp_path) == [] and witness.exists()
    # Any other failure is no sharing race: the census that met it stays unknown, never "not live".
    with monkeypatch.context() as patch:
        patch.setattr(pathlib.Path, "unlink", refused(OSError(errno.EIO, "disk failure")))
        assert presence_execution_witness("turn-a", drive_root=tmp_path) is None
    assert live_authors(drive_root=tmp_path) == [] and not witness.exists()
