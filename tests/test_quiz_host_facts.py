"""Host facts on the owner quiz card (4A) and the Project name on the ``chat.quiz`` event.

The host writes one sentence under the question from facts only it knows — the asking
task, how that task's run started (``dialogue_provenance.run_origin``) and when the
owner last wrote in the card's chat — never from the question text; an unrecorded fact
says unknown. The sentence rides the durable block, the live frame, the host event, the
chat row, history replay and the Main pointer; the Project name rides the event only.
"""

from __future__ import annotations

import datetime
import json

from ouroboros.owner_quiz import quiz_states, record_asked
from ouroboros.task_results import STATUS_RUNNING, write_task_result
from tests.test_quiz_answer import _escalate, _tool_ctx

def _stamp(minutes_ago: float) -> str:
    # Built at CALL time, never at import: the helper measures against the real clock when the
    # card is asked, and collection under xdist can run tens of seconds before this test body.
    return (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=minutes_ago)).isoformat()


def _shown(iso: str) -> str:
    return datetime.datetime.fromisoformat(iso).astimezone(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _chat_rows(tmp_path, *rows):
    path = tmp_path / "logs" / "chat.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _ask(ctx):
    out = _escalate(ctx, question="Started by your message? Ship it?", options=["Ship", "Hold"],
                    assumption="Hold meanwhile")
    assert out.startswith("OK: quiz "), out
    event = next(evt for evt in ctx.pending_events if evt.get("type") == "send_quiz")
    return event, quiz_states(ctx.drive_root, ctx.task_id)[event["quiz_id"]]


def test_record_asked_stores_host_facts_only_when_present(tmp_path):
    with_facts = record_asked(tmp_path, "t-1", quiz_id="q-1", question="Q?", options=["a", "b"],
                              assumption="a", host_facts="Asked by task t-1, origin unknown.")
    assert with_facts["host_facts"] == "Asked by task t-1, origin unknown."
    without = record_asked(tmp_path, "t-1", quiz_id="q-2", question="Q?", options=["a", "b"], assumption="a")
    assert "host_facts" not in without
    stored = quiz_states(tmp_path, "t-1")
    assert stored["q-1"]["host_facts"] == "Asked by task t-1, origin unknown."
    assert "host_facts" not in stored["q-2"]


def test_owner_started_root_names_its_message_and_the_last_owner_message_in_this_chat(tmp_path):
    started, last = _stamp(90.3), _stamp(47.3)
    ctx = _tool_ctx(tmp_path, chat_id=1)
    ctx.task_metadata["origin_message_ref"] = {
        "chat_id": 1, "client_message_id": "m-1", "ts": started, "text_sha256": "0" * 64}
    _chat_rows(
        tmp_path,
        {"ts": started, "direction": "in", "chat_id": 1, "text": "please ship"},
        {"ts": last, "direction": "in", "chat_id": 1, "text": "and quickly"},
        # Neither the assistant's own row nor another chat's owner message is "your last message here".
        {"ts": _stamp(10), "direction": "out", "chat_id": 1, "text": "working"},
        {"ts": _stamp(5), "direction": "in", "chat_id": 7, "text": "other room"},
    )
    event, block = _ask(ctx)
    prefix = (f"Asked by task root-1, started by your message of {_shown(started)}; "
              f"your last message in this chat: {_shown(last)} (")
    assert block["host_facts"].startswith(prefix)
    # 47 minutes and 18 seconds before the ask; a slow machine may cross into the 48th.
    assert block["host_facts"][len(prefix):] in ("47 minutes before this question).", "48 minutes before this question).")
    assert event["host_facts"] == block["host_facts"]


def test_scheduled_follow_up_names_the_task_it_follows(tmp_path):
    write_task_result(tmp_path, "root-1", STATUS_RUNNING,
                      metadata={"source": "task_followup", "origin_task_id": "prev-7", "schedule_id": "followup-prev-7"})
    event, block = _ask(_tool_ctx(tmp_path, chat_id=1))
    assert block["host_facts"] == ("Asked by task root-1, started as a scheduled follow-up of task prev-7; "
                                   "your last message in this chat: unknown.")
    assert event["host_facts"] == block["host_facts"]


def test_consciousness_origin_is_named(tmp_path):
    ctx = _tool_ctx(tmp_path, chat_id=1)
    ctx.task_metadata["initiator"] = "consciousness"
    _chat_rows(tmp_path, {"ts": _stamp(3.2), "direction": "in", "chat_id": 1, "text": "hi"})
    _event, block = _ask(ctx)
    assert block["host_facts"].startswith("Asked by task root-1, started by background consciousness; ")
    assert block["host_facts"].endswith(("(3 minutes before this question).", "(4 minutes before this question)."))


def test_unknown_origin_and_no_owner_message_are_said_as_unknown(tmp_path):
    # An unrelated source marker is shown as recorded, never mapped to a guess.
    _event, block = _ask(_tool_ctx(tmp_path, chat_id=1))
    assert block["host_facts"] == "Asked by task root-1, origin unknown; your last message in this chat: unknown."
    write_task_result(tmp_path, "root-2", STATUS_RUNNING, metadata={"source": "mystery_lane"})
    _event, block = _ask(_tool_ctx(tmp_path, task_id="root-2", chat_id=1))
    assert block["host_facts"] == ("Asked by task root-2, origin unknown (recorded source: mystery_lane); "
                                   "your last message in this chat: unknown.")


def _bridge(tmp_path, monkeypatch):
    from supervisor import message_bus, state

    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    state.init(tmp_path, 100)
    frames, events = [], []
    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "publish_event", lambda topic, data: events.append((topic, data)))
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    return bridge, frames, events


_OPTIONS = [{"label": "Local"}, {"label": "Shared"}]


def test_send_quiz_carries_host_facts_and_names_the_project_on_the_event_only(tmp_path, monkeypatch):
    from ouroboros.event_bus import CHAT_QUIZ
    from ouroboros.gateway.history import _assemble_history_response
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "facts-project", name="Facts Project")
    record_asked(tmp_path, "task-1", quiz_id="q-1", question="Which storage?", options=["Local", "Shared"],
                 assumption="Local", chat_id=project["chat_id"], host_facts="Asked by task task-1, origin unknown.")
    bridge, frames, events = _bridge(tmp_path, monkeypatch)
    ok, _ = bridge.send_quiz(project["chat_id"], quiz_id="q-1", question="Which storage?", options=_OPTIONS,
                             assumption="Local", task_id="task-1", host_facts="Asked by task task-1, origin unknown.")
    assert ok
    quiz_frame, pointer = frames
    assert quiz_frame["type"] == "quiz" and quiz_frame["host_facts"] == "Asked by task task-1, origin unknown."
    assert "project_name" not in quiz_frame  # the browser wire names the Project through its own pointer
    assert pointer["system_type"] == "project_question_pointer"
    assert pointer["host_facts"] == "Asked by task task-1, origin unknown."
    [(topic, event)] = events
    assert topic == CHAT_QUIZ
    assert event["host_facts"] == "Asked by task task-1, origin unknown."
    assert event["project_name"] == "Facts Project"
    stored = [json.loads(line) for line in (tmp_path / "logs/chat.jsonl").read_text(encoding="utf-8").splitlines()]
    assert stored[-1]["quiz"]["host_facts"] == "Asked by task task-1, origin unknown."
    room = json.loads(_assemble_history_response(tmp_path, project["chat_id"], 10, 0))["messages"]
    assert room[0]["msg_type"] == "quiz" and room[0]["quiz"]["host_facts"] == "Asked by task task-1, origin unknown."
    main = json.loads(_assemble_history_response(tmp_path, 1, 10, 0))["messages"]
    assert main[0]["system_type"] == "project_question_pointer"
    assert main[0]["host_facts"] == "Asked by task task-1, origin unknown."


def test_send_quiz_without_host_facts_or_project_adds_neither(tmp_path, monkeypatch):
    bridge, frames, events = _bridge(tmp_path, monkeypatch)
    assert bridge.send_quiz(1, quiz_id="q-2", question="Which storage?", options=_OPTIONS,
                            assumption="Local", task_id="task-2")[0]
    [quiz_frame] = frames
    [(_topic, event)] = events
    assert "host_facts" not in quiz_frame and "host_facts" not in event and "project_name" not in event
    stored = json.loads((tmp_path / "logs/chat.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert "host_facts" not in stored["quiz"]


def test_history_replays_the_block_sentence_for_a_row_logged_without_it(tmp_path, monkeypatch):
    from ouroboros.gateway.history import _assemble_history_response

    record_asked(tmp_path, "task-3", quiz_id="q-3", question="Which storage?", options=["Local", "Shared"],
                 assumption="Local", chat_id=1, host_facts="Asked by task task-3, origin unknown.")
    record_asked(tmp_path, "task-4", quiz_id="q-4", question="Which storage?", options=["Local", "Shared"],
                 assumption="Local", chat_id=1)
    bridge, _frames, _events = _bridge(tmp_path, monkeypatch)
    for task_id, quiz_id in (("task-3", "q-3"), ("task-4", "q-4")):
        assert bridge.send_quiz(1, quiz_id=quiz_id, question="Which storage?", options=_OPTIONS,
                                assumption="Local", task_id=task_id)[0]
    rows = {row["quiz"]["quiz_id"]: row["quiz"]
            for row in json.loads(_assemble_history_response(tmp_path, 1, 10, 0))["messages"]
            if row.get("msg_type") == "quiz"}
    assert rows["q-3"]["host_facts"] == "Asked by task task-3, origin unknown."
    assert "host_facts" not in rows["q-4"]


def test_the_supervisor_event_handler_forwards_host_facts_to_send_quiz():
    from types import SimpleNamespace

    from supervisor.events_chat_delivery import _handle_send_quiz

    sent = []

    class Bridge:
        def send_quiz(self, chat_id, **kwargs):
            sent.append((chat_id, kwargs))
            return True, "ok"

    ctx = SimpleNamespace(bridge=Bridge())
    evt = {"type": "send_quiz", "chat_id": 1, "quiz_id": "q-5", "question": "Q?", "options": _OPTIONS,
           "assumption": "a", "task_id": "t-5", "host_facts": "Asked by task t-5, origin unknown."}
    _handle_send_quiz(evt, ctx)
    _handle_send_quiz({**evt, "host_facts": ""}, ctx)
    assert sent[0][1]["host_facts"] == "Asked by task t-5, origin unknown."
    assert sent[1][1]["host_facts"] == ""
