"""Every lens that shows a card's recorded winner shows the files its Telegram Reply carried (owner's choice, 18 Nov):
the live card, its Project pointer in Main, the history replay, the activity census and the delivery
acknowledgement all name the same files as ``chat_uploads.answer_files`` states them (name, type, size, digest or why
one never arrived), never a stored id, a path or a link."""

from __future__ import annotations

import json
from types import SimpleNamespace

from ouroboros import chat_uploads
from ouroboros.gateway.history import _assemble_history_response
from ouroboros.owner_quiz import quiz_states, record_answered, record_asked
from ouroboros.projects_registry import create_project
from ouroboros.task_results import STATUS_RUNNING, write_task_result

_PNG = b"\x89PNG\r\n\x1a\n" + b"pixels" * 4


def _shown(ref):
    return {key: ref[key] for key in ("name", "kind", "mime", "size", "sha256")}


def _never_a_handle(value):
    """No stored id, stat witness or download address anywhere in what a lens states."""
    text = json.dumps(value, sort_keys=True)
    assert "upload" not in text and "mtime_ns" not in text and "/api/files" not in text, text


def _answered_project_question(tmp_path, monkeypatch, *, comment="this one", extra=()):
    """A Project's question the owner answered by a Telegram Reply with a file, as the host recorded it."""
    from supervisor import message_bus, state

    project = create_project(tmp_path, "question-project", name="Question Project")
    write_task_result(tmp_path, "task-1", STATUS_RUNNING, project_id=project["id"], chat_id=project["chat_id"])
    ref = chat_uploads.store_upload(_PNG, "photo.png", data_dir=tmp_path)[1]
    fact = chat_uploads.delivery_ref({"name": "clip.mp4", "unavailable": "download_failed"})
    record_asked(tmp_path, "task-1", quiz_id="q1", chat_id=project["chat_id"], question="Which screen?",
                 options=["A", "B"], assumption="A meanwhile")
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    state.init(tmp_path, 100)
    frames, events = [], []
    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(message_bus, "publish_event", lambda topic, payload: events.append((topic, payload)))
    refs = [ref, fact] if extra else [ref]
    assert record_answered(tmp_path, "task-1", quiz_id="q1", option_index=None, request_id="tg:9:m:42:7",
                           comment=comment, attachments=refs)["ok"]
    return project, bridge, frames, events, refs


def test_the_card_published_after_its_answer_and_its_project_pointer_carry_the_winners_files(tmp_path, monkeypatch):
    """A queued ask can reach the screen after the answer: the card it renders is the canonical, answered one."""
    project, bridge, frames, events, refs = _answered_project_question(tmp_path, monkeypatch, extra=True)

    assert bridge.send_quiz(project["chat_id"], "q1", "Which screen?", [{"label": "A"}, {"label": "B"}],
                            assumption="A meanwhile", task_id="task-1", state="answered")[0]

    card, pointer = frames
    assert card["type"] == "quiz" and pointer["system_type"] == "project_question_pointer"
    shown = [_shown(refs[0]), {key: refs[1][key] for key in ("name", "kind", "mime", "unavailable")}]
    assert card["comment"] == "this one" and card["attachments"] == shown
    assert pointer["comment"] == "this one" and pointer["attachments"] == shown
    _never_a_handle([card, pointer])
    (topic, event), = [item for item in events if item[0] == "chat.quiz"]
    assert event["attachments"] == shown, "a transport that renders the card again states the same files"
    _never_a_handle(event)
    stored = json.loads(_assemble_history_response(tmp_path, project["chat_id"], 10, 0))["messages"][0]["quiz"]
    assert stored["attachments"] == shown
    _never_a_handle(stored)


def test_history_replays_the_winners_files_on_the_room_card_and_the_main_pointer(tmp_path, monkeypatch):
    project, bridge, _frames, _events, refs = _answered_project_question(tmp_path, monkeypatch)
    assert bridge.send_quiz(project["chat_id"], "q1", "Which screen?", [{"label": "A"}, {"label": "B"}],
                            assumption="A meanwhile", task_id="task-1", state="open")[0]
    # The block is answered since: the replay merges the canonical record, never the row as it was first stored.
    assert quiz_states(tmp_path, "task-1")["q1"]["state"] == "answered"

    room = json.loads(_assemble_history_response(tmp_path, project["chat_id"], 10, 0))["messages"]
    main = json.loads(_assemble_history_response(tmp_path, 1, 10, 0))["messages"]

    assert room[0]["quiz"]["attachments"] == [_shown(refs[0])]
    assert main[0]["system_type"] == "project_question_pointer" and main[0]["attachments"] == [_shown(refs[0])]
    assert main[0]["comment"] == "this one" and main[0]["quiz_state"] == "answered"
    _never_a_handle([room[0]["quiz"], main[0]])


def test_the_activity_census_pointer_names_the_winners_files(tmp_path, monkeypatch):
    from ouroboros.gateway import state as gs
    from ouroboros.owner_wait import set_owner_wait
    from supervisor import queue

    project = create_project(tmp_path, "wait-room", name="Wait room")
    write_task_result(tmp_path, "t1", STATUS_RUNNING, project_id=project["id"],
                      root_phase_checkpoint={"post_task_synthesis": "running"})
    record_asked(tmp_path, "t1", quiz_id="q1", question="?", options=["a", "b"], wait_for_answer=True)
    set_owner_wait(tmp_path, "t1", {"quiz_id": "q1", "wait_id": "w1", "state": "waiting"})
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "RUNNING", {"t1": {"task": {"id": "t1", "project_id": project["id"],
                                                          "chat_id": project["chat_id"]}}})
    gs._FINALIZING_MEMO.clear()
    ref = chat_uploads.store_upload(_PNG, "photo.png", data_dir=tmp_path)[1]
    assert record_answered(tmp_path, "t1", quiz_id="q1", option_index=None, request_id="r", comment="the photo one",
                           attachments=[ref])["ok"]

    pointer = gs._chat_activities_snapshot_safe(tmp_path, direct_turns=[])[0]["required_question"]

    assert pointer["quiz_state"] == "answered" and pointer["comment"] == "the photo one"
    assert pointer["attachments"] == [_shown(ref)]
    _never_a_handle(pointer)


def test_the_delivery_acknowledgement_restates_the_whole_winner_including_its_files(tmp_path, monkeypatch):
    """The asker's read of the answer re-publishes the winner to the card: a card rebuilt from that fact keeps it."""
    import supervisor.message_bus as mb
    from supervisor.events_runtime_controls import _handle_owner_message_injected

    ref = chat_uploads.store_upload(_PNG, "photo.png", data_dir=tmp_path)[1]
    record_asked(tmp_path, "task-1", quiz_id="q1", question="Which screen?", options=["A", "B"], assumption="A")
    assert record_answered(tmp_path, "task-1", quiz_id="q1", option_index=None, request_id="r1", comment="",
                           attachments=[ref])["ok"]
    frames = []
    monkeypatch.setattr(mb, "get_bridge", lambda: SimpleNamespace(
        send_quiz_state=lambda quiz_id, task_id, state, **fields: frames.append({"state": state, **fields})))

    _handle_owner_message_injected({"quiz_ack": {"task_id": "task-1", "quiz_id": "q1"}}, SimpleNamespace(DRIVE_ROOT=tmp_path))

    (frame,) = frames
    assert frame["state"] == "answered" and frame["attachments"] == [_shown(ref)] and "comment" not in frame
    _never_a_handle(frame)
