"""#968: one owner-routing receipt states one set of facts on its durable row, its live
frame and its history replay, whichever producer wrote it.

The direct server path (``_record_routing_receipt``), the supervisor's token-bound
receipt (``_emit_routing_receipt``), the relay's read acknowledgement and the picker's
closing row all run here for real, over one data root: the row through
``append_chat_annotation``, the frame through ``LocalChatBridge.send_routing_ack``, the
replay through the history endpoint. A registry change right after a row is written
is the causal control: a producer that reads its facts a second time to send them
shows the owner a frame its own row (and so every reload) contradicts.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import re
from types import SimpleNamespace

import pytest

# What the browser paints from a receipt (routing_receipt.js) and the history replay must
# keep: listed here independently of the production list, so the test pins the contract.
SHOWN = ("action", "target", "target_label", "status", "routing_token", "cause", "options",
         "attachment_manifest", "reasoning_effort", "project_id", "project_chat_id", "project_name",
         "target_task_ids", "target_titles")
DESTINATION = ("project_id", "project_chat_id", "project_name", "target_task_ids", "target_titles")
TITLE = "Fix duplicate posts in the RSS import"


def _shown(row):
    return {key: row[key] for key in SHOWN if key in row}


def _room(root, *, task_id="work-1", title=TITLE):
    """A Project room ``Blog`` with one titled running work bound to it."""
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.task_results import write_task_result

    project = create_project(root, "blog", name="Blog")
    bind_task_to_project(root, task_id, "blog", project["chat_id"], origin={"absent": "system"})
    write_task_result(root, task_id, "running", chat_id=project["chat_id"], title=title, result="working")
    return project


def _bridge(root, monkeypatch):
    from supervisor import message_bus

    monkeypatch.setattr(message_bus, "DATA_DIR", root)
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_k: None)
    bridge = message_bus.LocalChatBridge({})
    frames: list = []
    bridge._broadcast_fn = frames.append
    return bridge, frames


def _replay(root, client_message_id, chat_id=1):
    """The owner message as the real history endpoint replays it after a reload."""
    from ouroboros.gateway.history import make_chat_history_endpoint

    response = asyncio.run(make_chat_history_endpoint(root)(SimpleNamespace(query_params={"chat_id": str(chat_id)})))
    row = next(row for row in json.loads(response.body)["messages"] if row.get("client_message_id") == client_message_id)
    return row.get("chat_annotation") or {}


def _owner_message(root, client_message_id, chat_id=1):
    from supervisor import message_bus

    message_bus.log_chat("in", chat_id, 7, "Keep the old feed address and redirect /feed.xml.",
                         source="web", client_message_id=client_message_id, drive_root=root)


def _rename_after_write(monkeypatch, root, new_name="Blog (renamed)"):
    """Rename the room right after the receipt's row is written: the drift window."""
    from ouroboros import project_dialogue
    from ouroboros.projects_registry import update_project

    real = project_dialogue.append_chat_annotation

    def append_then_rename(*args, **kwargs):
        written = real(*args, **kwargs)
        update_project(root, "blog", name=new_name)
        return written

    monkeypatch.setattr(project_dialogue, "append_chat_annotation", append_then_rename)


def _late_reply(root, bridge, client_message_id="cm-late"):
    from ouroboros.project_dialogue import routing_target_label
    from ouroboros.server_owner_routing import _record_routing_receipt

    _record_routing_receipt(
        bridge, SimpleNamespace(DRIVE_ROOT=root), chat_id=1, client_message_id=client_message_id,
        action="mailbox_delivery", target="work-1",
        target_label=routing_target_label(root, "mailbox_delivery", "work-1"), status="delivered")


def _annotation_frames(frames):
    return [frame for frame in frames if frame.get("type") == "message_annotation"]


def assignment(chars=6417, lines=39):
    """A work order shaped like the reported one (msg-1793137452751-1 → 8a5774f9): a heading, then the whole
    assignment, `chars` characters on `lines` lines. Synthetic words; only the heading is the real one."""
    rows = ["# Community watch — one six-hourly GitHub pass", ""]
    rows += [f"{n}. Read the current thread and its source before acting; keep exact coverage gaps." for n in range(lines - 2)]
    while len("\n".join(rows)) < chars:
        rows[-1] += " Retain the exact ids, links and dates."
    text = "\n".join(rows)[:chars]
    assert (len(text), text.count("\n") + 1) == (chars, lines)
    return text


def test_a_work_is_named_as_its_card_names_it_never_by_its_text_or_id(tmp_path):
    """The receipt names a work by its display name (suggested name, then title). A nameless work, here one whose
    text is its whole assignment, gets no name: the line says `the work` instead of quoting the assignment, and a
    nameless hidden teller's note card says `Another work`."""
    from ouroboros.owner_relay import relay_sender_facts, relay_target_facts
    from ouroboros.task_results import write_task_result

    named = {"suggested_name": "Community watch — one six-hourly GitHub pass", "title": "Old title", "text": assignment()}
    assert relay_target_facts(tmp_path, "w-1", named)["target_titles"] == ["Community watch — one six-hourly GitHub pass"]
    assert relay_target_facts(tmp_path, "w-2", {"title": TITLE, "text": assignment()})["target_titles"] == [TITLE]
    assert relay_target_facts(tmp_path, "w-3", {"text": assignment(), "chat_id": 1})["target_titles"] == []
    assert relay_target_facts(tmp_path, "w-4")["target_titles"] == [], "no record: no name, never the id"
    write_task_result(tmp_path, "teller", "running", chat_id=0, description=assignment())
    assert relay_sender_facts(tmp_path, "teller")["name"] == ""


def test_words_passed_on_to_a_nameless_work_say_the_work_live_after_reload_and_once_read(tmp_path, monkeypatch):
    """The reported case at its source: the work had no name and its text was its whole assignment. The receipt no
    longer copies that text into its row and frame, and the work's read acknowledgement no longer names it by id."""
    from ouroboros.owner_relay import acknowledge_relay
    from ouroboros.project_dialogue import chat_annotation_receipt
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.server_owner_routing import _record_routing_receipt
    from ouroboros.task_results import write_task_result

    project = create_project(tmp_path, "community-watch")
    bind_task_to_project(tmp_path, "8a5774f9", "community-watch", project["chat_id"], origin={"absent": "system"})
    write_task_result(tmp_path, "8a5774f9", "running", chat_id=project["chat_id"], description=assignment(), result="working")
    bridge, frames = _bridge(tmp_path, monkeypatch)
    _owner_message(tmp_path, "msg-1")
    _record_routing_receipt(bridge, SimpleNamespace(DRIVE_ROOT=tmp_path), chat_id=1, client_message_id="msg-1",
                            action="mailbox_delivery", target="8a5774f9", status="delivered",
                            task={"id": "8a5774f9", "chat_id": project["chat_id"], "text": assignment()})

    [frame] = _annotation_frames(frames)
    assert (frame["project_name"], frame["target_titles"]) == ("community-watch", [])
    assert _shown(_replay(tmp_path, "msg-1")) == _shown(frame)
    relay = {"relay_id": "relay-1", "quotes": [{"source": {"chat_id": 1, "client_message_id": "msg-1"}}]}
    acknowledge_relay(tmp_path, "8a5774f9", {"relay": relay})
    assert chat_annotation_receipt(tmp_path, "msg-1", "relay-1")["target_titles"] == []


def test_a_late_reply_names_the_work_and_its_project_live_and_after_reload(tmp_path, monkeypatch):
    """Owner choice A: the late answer from Main saved for a work in Blog says so, the same
    words live and after a reload ("Passed on to «…» in [Blog ↗]" in the browser)."""
    project = _room(tmp_path)
    bridge, frames = _bridge(tmp_path, monkeypatch)
    _owner_message(tmp_path, "cm-late")
    _late_reply(tmp_path, bridge)

    [frame] = _annotation_frames(frames)
    assert {key: frame.get(key) for key in DESTINATION} == {
        "project_id": "blog", "project_chat_id": project["chat_id"], "project_name": "Blog",
        "target_task_ids": ["work-1"], "target_titles": [TITLE]}
    assert (frame["action"], frame["status"]) == ("mailbox_delivery", "delivered")
    assert _shown(_replay(tmp_path, "cm-late")) == _shown(frame)


def test_a_late_answer_from_main_names_its_work_and_room_through_the_real_intake(tmp_path, monkeypatch):
    """The approved scene end to end: the owner's late answer in Main reaches the work that an
    earlier late answer to the same asker started, running in Blog (server intake →
    ``_route_owner_message`` → durable row → live frame → history replay)."""
    import threading

    import server
    from supervisor import message_bus
    from tests.test_quiz_late_answer import _late_answered
    from tests.test_v6730_origin_invariant import _ImmediateThread

    project = _room(tmp_path)
    bridge = _late_answered(tmp_path, monkeypatch, 1)
    frames: list = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "publish_event", lambda *_a, **_k: None)
    work = {"id": "work-1", "chat_id": 1, "root_task_id": "work-1", "delegation_role": "root", "title": TITLE,
            "drive_root": str(tmp_path), "metadata": {"late_answer": {"task_id": "task-1", "quiz_id": "q0"}}}
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path, PENDING=[], RUNNING={"work-1": {"task": work}},
        load_state=lambda: {"owner_id": 1, "owner_chat_id": 1, "session_id": "s-1"},
        update_state=lambda fn: fn({"owner_id": 1, "owner_chat_id": 1}),
        consciousness=SimpleNamespace(inject_observation=lambda _t: None, pause=lambda: None, resume=lambda: None),
        get_chat_agent=lambda: SimpleNamespace(_busy=False),
        handle_chat_direct=lambda *a, **k: pytest.fail("the late answer belongs to the work its asker's thread runs in"),
        send_with_budget=lambda *a, **k: None,
    )
    with monkeypatch.context() as immediate:  # only the intake: the history read below needs real threads
        immediate.setattr(threading, "Thread", _ImmediateThread)
        server._process_bridge_updates(bridge, 0, ctx)

    [frame] = _annotation_frames(frames)
    assert (frame["action"], frame["status"], frame["target"]) == ("mailbox_delivery", "delivered", "work-1")
    assert {key: frame.get(key) for key in DESTINATION} == {
        "project_id": "blog", "project_chat_id": project["chat_id"], "project_name": "Blog",
        "target_task_ids": ["work-1"], "target_titles": [TITLE]}
    assert _shown(_replay(tmp_path, frame["client_message_id"])) == _shown(frame)


def test_the_direct_receipt_sends_the_facts_its_row_was_written_with(tmp_path, monkeypatch):
    _room(tmp_path)
    bridge, frames = _bridge(tmp_path, monkeypatch)
    _owner_message(tmp_path, "cm-late")
    _rename_after_write(monkeypatch, tmp_path)
    _late_reply(tmp_path, bridge)

    [frame] = _annotation_frames(frames)
    assert frame.get("project_name") == "Blog"
    assert _shown(_replay(tmp_path, "cm-late")) == _shown(frame)


def test_a_relay_receipt_replays_the_work_and_room_its_live_frame_named(tmp_path, monkeypatch):
    """The supervisor's relay receipt already names the work and its room live; the history
    replay used to drop the names, so a reload read "Passed on to the work in [Project ↗]"."""
    from supervisor.events_project_routing import _emit_routing_receipt

    _room(tmp_path)
    bridge, frames = _bridge(tmp_path, monkeypatch)
    _owner_message(tmp_path, "cm-relay")
    task = {"id": "work-1", "title": TITLE, "chat_id": 1}
    receipt = _emit_routing_receipt(
        SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=bridge),
        {"client_message_id": "cm-relay", "routing_token": "relay-1", "relay_id": "relay-1", "chat_id": 1,
         "relay_target": task},
        action="steer_task", target="work-1", target_label="Blog › " + TITLE, status="delivered")

    assert receipt["persisted"] is True
    [frame] = _annotation_frames(frames)
    assert (frame["project_name"], frame["target_titles"]) == ("Blog", [TITLE])
    assert _shown(_replay(tmp_path, "cm-relay")) == _shown(frame)


@pytest.mark.parametrize("change", ["rename", "delete"])
def test_a_supervisor_receipt_sends_the_facts_its_row_was_written_with(tmp_path, monkeypatch, change):
    """Persist then publish read the destination twice; a room renamed or deleted in between
    gave the live frame facts the durable row (and so every reload) does not have."""
    from ouroboros import project_dialogue
    from ouroboros.project_dialogue import latest_chat_annotations
    from ouroboros.projects_registry import begin_project_deletion
    from supervisor.events_project_routing import _emit_routing_receipt

    _room(tmp_path)
    bridge, frames = _bridge(tmp_path, monkeypatch)
    if change == "rename":
        _rename_after_write(monkeypatch, tmp_path)
        evt = {"client_message_id": "cm-1", "routing_token": "relay-1", "relay_id": "relay-1", "chat_id": 1,
               "relay_target": {"id": "work-1", "title": TITLE}}
        action, status = "steer_task", "delivered"
    else:
        real = project_dialogue.append_chat_annotation
        monkeypatch.setattr(project_dialogue, "append_chat_annotation",
                            lambda *a, **k: (real(*a, **k), begin_project_deletion(tmp_path, "blog"))[0])
        evt = {"client_message_id": "cm-1", "routing_token": "promote-1", "chat_id": 1}
        action, status = "promote_chat_to_task", "scheduled"
    _emit_routing_receipt(SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=bridge), evt,
                          action=action, target="work-1", target_label="Blog › " + TITLE, status=status)

    [frame] = _annotation_frames(frames)
    row = latest_chat_annotations(tmp_path)["cm-1"]
    assert row["project_id"] == "blog"
    assert _shown(frame) == _shown(row)


def test_the_relay_read_ack_publishes_the_row_it_saved(tmp_path, monkeypatch):
    """The work's read acknowledgement is saved in the worker and published by the supervisor:
    the live frame is that saved row, not a second read of the work's title and room."""
    from ouroboros.owner_relay import acknowledge_relay, publish_relay_ack
    from ouroboros.project_dialogue import chat_annotation_receipt
    from ouroboros.task_results import write_task_result

    _room(tmp_path)
    bridge, frames = _bridge(tmp_path, monkeypatch)
    relay = {"relay_id": "relay-1", "quotes": [{"source": {"chat_id": 1, "client_message_id": "cm-q"}}]}
    acknowledge_relay(tmp_path, "work-1", {"relay": relay})
    saved = chat_annotation_receipt(tmp_path, "cm-q", "relay-1")
    write_task_result(tmp_path, "work-1", "running", title="Renamed while the ack travelled")
    publish_relay_ack(tmp_path, "work-1", relay, bridge)

    [frame] = _annotation_frames(frames)
    assert (saved["status"], saved["target_titles"]) == ("read", [TITLE])
    assert _shown(frame) == _shown(saved)


def test_a_relay_read_ack_whose_row_was_not_saved_still_reaches_the_owner(tmp_path, monkeypatch):
    """Failure policy kept: the mailbox ACK is the authority, so a read row the sidecar could
    not take still updates the owner's line live, as before."""
    from ouroboros import project_dialogue
    from ouroboros.owner_relay import acknowledge_relay, publish_relay_ack

    _room(tmp_path)
    bridge, frames = _bridge(tmp_path, monkeypatch)
    monkeypatch.setattr(project_dialogue, "append_chat_annotation", lambda *_a, **_k: False)
    relay = {"relay_id": "relay-1", "quotes": [{"source": {"chat_id": 1, "client_message_id": "cm-q"}}]}
    acknowledge_relay(tmp_path, "work-1", {"relay": relay})
    publish_relay_ack(tmp_path, "work-1", relay, bridge)

    [frame] = _annotation_frames(frames)
    assert (frame["status"], frame["routing_token"], frame["target_titles"]) == ("read", "relay-1", [TITLE])


@pytest.mark.parametrize("failure", ["raises", "refused"])
def test_a_late_reply_whose_row_failed_is_still_shown_live(tmp_path, monkeypatch, failure):
    """Failure policy kept: the delivery's authority is the mailbox it already wrote, so a lost
    annotation never turns it into an undelivered or forever-Sending message."""
    from ouroboros import project_dialogue

    _room(tmp_path)
    bridge, frames = _bridge(tmp_path, monkeypatch)

    def failed(*_a, **_k):
        if failure == "raises":
            raise OSError("synthetic annotation failure")
        return False

    monkeypatch.setattr(project_dialogue, "append_chat_annotation", failed)
    _late_reply(tmp_path, bridge)

    [frame] = _annotation_frames(frames)
    assert (frame["status"], frame["project_id"], frame["target_titles"]) == ("delivered", "blog", [TITLE])


def test_a_supervisor_receipt_whose_row_failed_publishes_nothing(tmp_path, monkeypatch):
    """Failure policy kept: the token-bound receipt withholds its frame and reports the
    act unconfirmed when its durable row was not written."""
    from ouroboros import project_dialogue
    from supervisor.events_project_routing import _emit_routing_receipt

    _room(tmp_path)
    bridge, frames = _bridge(tmp_path, monkeypatch)
    monkeypatch.setattr(project_dialogue, "append_chat_annotation", lambda *_a, **_k: False)
    receipt = _emit_routing_receipt(
        SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=bridge),
        {"client_message_id": "cm-1", "routing_token": "relay-1", "relay_id": "relay-1", "chat_id": 1},
        action="steer_task", target="work-1", status="delivered")

    assert _annotation_frames(frames) == []
    assert (receipt["status"], receipt["reason"], receipt["persisted"]) == (
        "unconfirmed", "routing_annotation_persist_failed", False)


def _unreadable(*_args, **_kwargs):
    raise PermissionError("synthetic read failure")


def test_a_supervisor_relay_whose_work_cannot_be_read_fails_as_before(tmp_path, monkeypatch):
    """Failure policy kept: a relay's receipt reads its work before any row is written, and that
    read failing fails the act (no row, no frame), as it did before the fields were shared."""
    from ouroboros import owner_relay
    from ouroboros.project_dialogue import latest_chat_annotations
    from supervisor.events_project_routing import _emit_routing_receipt

    _room(tmp_path)
    bridge, frames = _bridge(tmp_path, monkeypatch)
    monkeypatch.setattr(owner_relay, "relay_target_facts", _unreadable)
    with pytest.raises(PermissionError, match="synthetic read failure"):
        _emit_routing_receipt(
            SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=bridge),
            {"client_message_id": "cm-1", "routing_token": "relay-1", "relay_id": "relay-1", "chat_id": 1},
            action="steer_task", target="work-1", status="delivered")

    assert "cm-1" not in latest_chat_annotations(tmp_path)
    assert _annotation_frames(frames) == []


@pytest.mark.parametrize("producer", ["direct_late_reply", "picker_steer"])
def test_a_landed_receipt_whose_work_cannot_be_read_keeps_its_project_address(tmp_path, monkeypatch, producer):
    """The work's name is added enrichment for an act that already landed (words saved into the
    work's mailbox, a picker steer): an unreadable work leaves the receipt its plain Project
    address, never no receipt."""
    from ouroboros import owner_relay
    from ouroboros.project_dialogue import latest_chat_annotations
    from supervisor.events_project_routing import _emit_routing_receipt

    project = _room(tmp_path)
    bridge, frames = _bridge(tmp_path, monkeypatch)
    monkeypatch.setattr(owner_relay, "relay_target_facts", _unreadable)
    if producer == "direct_late_reply":
        _late_reply(tmp_path, bridge, client_message_id="cm-1")
    else:
        _emit_routing_receipt(SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=bridge),
                              {"client_message_id": "cm-1", "routing_token": "pick-1", "chat_id": 1},
                              action="steer_task", target="work-1", target_label="Blog › " + TITLE, status="delivered")

    [frame] = _annotation_frames(frames)
    row = latest_chat_annotations(tmp_path)["cm-1"]
    assert (row["status"], row["project_id"], row["project_chat_id"]) == ("delivered", "blog", project["chat_id"])
    assert "target_titles" not in row
    assert _shown(frame) == _shown(row)


@pytest.mark.parametrize("lifecycle", ["deleting", "tombstoned"])
@pytest.mark.parametrize("producer", ["direct_late_reply", "picker_steer"])
def test_words_reaching_a_work_whose_project_is_closing_offer_no_room_to_open(tmp_path, monkeypatch, producer, lifecycle):
    """Only an ACTIVE Project is a room the receipt can open (the address rule these producers always had); the
    work's own name stays. Live and after a reload alike: no `Open project` reference into a closing room."""
    from ouroboros.project_dialogue import latest_chat_annotations
    from ouroboros.projects_registry import begin_project_deletion, complete_project_deletion
    from supervisor.events_project_routing import _emit_routing_receipt

    _room(tmp_path)
    begin_project_deletion(tmp_path, "blog")
    if lifecycle == "tombstoned":
        complete_project_deletion(tmp_path, "blog")
    bridge, frames = _bridge(tmp_path, monkeypatch)
    _owner_message(tmp_path, "cm-1")
    if producer == "direct_late_reply":
        _late_reply(tmp_path, bridge, client_message_id="cm-1")
    else:
        _emit_routing_receipt(SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=bridge),
                              {"client_message_id": "cm-1", "routing_token": "pick-1", "chat_id": 1},
                              action="steer_task", target="work-1", target_label="Blog › " + TITLE, status="delivered")

    [frame] = _annotation_frames(frames)
    row = latest_chat_annotations(tmp_path)["cm-1"]
    for stated in (frame, row):
        assert {"project_id", "project_chat_id"}.isdisjoint(stated), stated
        assert (stated["status"], stated["target_task_ids"], stated["target_titles"]) == ("delivered", ["work-1"], [TITLE])
    assert _shown(_replay(tmp_path, "cm-1")) == _shown(frame) == _shown(row)


def test_a_relay_read_ack_with_an_unreadable_sidecar_still_reaches_the_owner(tmp_path, monkeypatch):
    """The saved-row read is new: when the sidecar cannot be read, the ack goes out from the
    work's facts, as it always did."""
    from ouroboros import project_dialogue
    from ouroboros.owner_relay import publish_relay_ack

    _room(tmp_path)
    bridge, frames = _bridge(tmp_path, monkeypatch)
    monkeypatch.setattr(project_dialogue, "chat_annotation_receipt", _unreadable)
    publish_relay_ack(tmp_path, "work-1", {"relay_id": "relay-1",
                                           "quotes": [{"source": {"chat_id": 1, "client_message_id": "cm-q"}}]}, bridge)

    [frame] = _annotation_frames(frames)
    assert (frame["status"], frame["routing_token"], frame["target_titles"]) == ("read", "relay-1", [TITLE])


def _bound_live_work(tmp_path, workers):
    from ouroboros.projects_registry import bind_task_to_project, get_project
    from ouroboros.task_results import write_task_result

    workers.PENDING.append({"id": "t-live", "type": "task", "chat_id": 7, "root_task_id": "t-live", "title": "Fix CI"})
    write_task_result(tmp_path, "t-live", "scheduled", root_task_id="t-live", title="Fix CI")
    bind_task_to_project(tmp_path, "t-live", "p1", get_project(tmp_path, "p1")["chat_id"], origin={"absent": "system"})


@pytest.mark.parametrize("option", [0, 1], ids=["steer", "new_task"])
def test_a_picker_click_replays_the_room_its_live_receipt_named(tmp_path, monkeypatch, option):
    """The picker's closing row is what a reload paints: it keeps where the dispatched act's
    own receipt said the words landed, so the reference does not vanish on reload."""
    from ouroboros.gateway.history import _user_annotation
    from ouroboros.project_dialogue import latest_chat_annotations
    from tests.test_routing_decision import _manual_picker

    client, click, _card, _queue, workers, frames = _manual_picker(tmp_path, monkeypatch)
    _bound_live_work(tmp_path, workers)
    click["option_index"] = option
    response = client.post("/api/decisions", json=click)
    assert response.status_code == 200, response.text

    live = frames[-1]
    replay = _user_annotation("user", "cm-1", latest_chat_annotations(tmp_path))
    assert live["project_id"] == "p1"
    assert {key: replay.get(key) for key in DESTINATION} == {key: live.get(key) for key in DESTINATION}


def test_a_picker_closing_row_is_written_when_its_dispatch_receipt_cannot_be_read(tmp_path, monkeypatch):
    """The dispatch-receipt read is new: when it fails, the closing row the replay path reads
    back is still written, without the destination, as it always was."""
    from ouroboros import project_dialogue
    from ouroboros.project_dialogue import latest_chat_annotations
    from tests.test_routing_decision import _manual_picker

    client, click, card, _queue, _workers, _frames = _manual_picker(tmp_path, monkeypatch)
    monkeypatch.setattr(project_dialogue, "chat_annotation_receipt", _unreadable)
    click["option_index"] = 1
    response = client.post("/api/decisions", json=click)
    assert response.status_code == 200, response.text

    closing = latest_chat_annotations(tmp_path)["cm-1"]
    assert (closing["routing_token"], closing["status"], closing["detail"]) == (
        card["routing_token"], "scheduled", f"request:{click['request_id']}")
    assert "project_id" not in closing


def test_a_scheduled_promote_publishes_the_facts_its_row_was_written_with(tmp_path, monkeypatch):
    """A promote's receipt is written before its admission record and sent after it; the
    room deleted in between must not leave the live frame and the row disagreeing."""
    from ouroboros import project_handoff
    from ouroboros.gateway.routing_decision import _derived_identity
    from ouroboros.project_dialogue import chat_annotation_receipt
    from ouroboros.projects_registry import begin_project_deletion
    from tests.test_routing_decision import _manual_picker

    client, click, card, _queue, _workers, frames = _manual_picker(tmp_path, monkeypatch)
    real = project_handoff.enqueue_project_handoff
    monkeypatch.setattr(project_handoff, "enqueue_project_handoff",
                        lambda root, tid: (real(root, tid), begin_project_deletion(tmp_path, "p1"))[0])
    click["option_index"] = 1
    response = client.post("/api/decisions", json=click)
    assert response.status_code == 200, response.text

    dispatch_token, _task_id = _derived_identity("cm-1", card["routing_token"], 1)
    row = chat_annotation_receipt(tmp_path, "cm-1", dispatch_token)
    live = frames[-1]
    assert (row["status"], row["project_id"]) == ("scheduled", "p1")
    assert _shown(live) == _shown(row)


def test_every_receipt_field_is_one_store_bus_contract_and_replay_field():
    """The one list the readers keep is exactly what the store takes, the bus sends and the
    frozen frame contract (and its JS twin) declares."""
    from pathlib import Path

    from ouroboros.gateway.contracts import MessageAnnotationOutbound
    from ouroboros.gateway.history import _user_annotation
    from ouroboros.project_dialogue import ROUTING_RECEIPT_FIELDS, append_chat_annotation
    from supervisor.message_bus import LocalChatBridge

    assert set(ROUTING_RECEIPT_FIELDS) == set(SHOWN)
    for writer in (append_chat_annotation, LocalChatBridge.send_routing_ack):
        assert set(SHOWN) <= set(inspect.signature(writer).parameters), writer
    assert set(SHOWN) <= set(MessageAnnotationOutbound.__annotations__)
    twin = (Path(__file__).resolve().parents[1] / "web" / "modules" / "api_types.js").read_text(encoding="utf-8")
    block = twin.split("@typedef {Object} MessageAnnotationOutbound", 1)[1].split("*/", 1)[0]
    assert [key for key in SHOWN if not re.search(rf"@property \{{[^}}]+\}} {key}\b", block)] == []
    row = {key: f"<{key}>" for key in (*SHOWN, "detail", "reason", "ts", "type", "client_message_id")}
    assert set(_user_annotation("user", "cm", {"cm": row})) == {*SHOWN, "detail"}
