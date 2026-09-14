"""Required Project questions share their durable ask with every display lens."""
import json
import os

import pytest

from ouroboros.gateway.history import _assemble_history_response
from ouroboros.owner_quiz import quiz_states, record_asked, record_answered, reconcile_terminal
from ouroboros.owner_wait import set_owner_wait
from ouroboros.projects_registry import create_project
from ouroboros.task_results import write_task_result, STATUS_RUNNING
from tests.test_quiz_answer import _tool_ctx, _escalate, _decision_app, _post


def test_real_escalation_bridge_history_detail_and_answer(tmp_path, monkeypatch):
    from supervisor import message_bus, state
    from ouroboros.gateway import tasks
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    project = create_project(tmp_path, "question-project", name="Question Project")
    write_task_result(tmp_path, "task-1", STATUS_RUNNING, project_id=project["id"], chat_id=project["chat_id"])
    ctx = _tool_ctx(tmp_path, task_id="task-1", chat_id=project["chat_id"])
    ctx.owner_wait_callback = lambda *_: None
    result = _escalate(ctx, question="Which storage?", wait_for_answer=True,
                       options=[{"label": "Local", "detail": "Single-machine files"},
                                {"label": "Shared", "detail": "Multiple writers over network"}])
    assert result.startswith("OK:")
    event = ctx.pending_events[0]
    qid = event["quiz_id"]
    assert quiz_states(tmp_path, "task-1")[qid]["option_details"] == [
        "Single-machine files", "Multiple writers over network"]
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    state.init(tmp_path, 100)
    frames = []
    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    assert bridge.send_quiz(project["chat_id"], **{key: event[key] for key in (
        "quiz_id", "question", "options", "stake", "assumption", "state", "task_id", "wait_for_answer")})[0]
    assert [row["type"] for row in frames] == ["quiz", "chat"]
    pointer = frames[1]
    assert pointer["system_type"] == "project_question_pointer" and pointer["chat_id"] == 1
    assert not {"options", "question", "comment", "answered_index"} & pointer.keys()
    stored = (tmp_path / "logs/chat.jsonl").read_text().splitlines()
    assert len(stored) == 1 and json.loads(stored[0])["chat_id"] == project["chat_id"]
    main = json.loads(_assemble_history_response(tmp_path, 1, 10, 0))["messages"]
    assert len(main) == 1 and main[0]["quiz_id"] == qid
    assert not {"task_phase", "outcome_axes", "cost_final", "cancelable"} & main[0].keys()
    room = json.loads(_assemble_history_response(tmp_path, project["chat_id"], 10, 0))["messages"]
    assert room[0]["msg_type"] == "quiz" and room[0]["quiz"]["options"][1]["detail"] == "Multiple writers over network"
    monkeypatch.setattr(tasks, "request_drive_root", lambda request: tmp_path)
    app = Starlette(routes=[Route('/api/tasks/{task_id}', tasks.api_task_get)])
    detail = TestClient(app).get('/api/tasks/task-1').json()
    assert detail["task_id"] == "task-1" and detail["project_id"] == project["id"]
    assert detail["owner_quiz"][qid]["option_details"] == quiz_states(tmp_path, "task-1")[qid]["option_details"]
    answered = _post(_decision_app(tmp_path, monkeypatch, live_task={"id": "task-1", "chat_id": project["chat_id"]}),
                     {"request_id": "answer-1", "decision_id": f"quiz:task-1:{qid}", "option_index": 1})
    assert answered.status_code == 200, answered.text
    main = json.loads(_assemble_history_response(tmp_path, 1, 10, 0))["messages"]
    assert main[0]["quiz_state"] == "answered"
    assert reconcile_terminal(tmp_path, "task-1") == []
    assert quiz_states(tmp_path, "task-1")[qid]["option_details"][1] == "Multiple writers over network"


def test_pointer_quota_dedup_and_optional_filter(tmp_path):
    from ouroboros.utils import append_jsonl

    project = create_project(tmp_path, "many-questions", name="Many Questions")
    for i in range(8):
        qid = f"q{i}"
        record_asked(tmp_path, "t1", quiz_id=qid, question="?", options=["a", "b"], wait_for_answer=True)
        row = {"type": "quiz", "direction": "out", "chat_id": project["chat_id"], "task_id": "t1",
               "ts": f"2026-09-09T00:00:0{i}Z", "text": "?", "quiz": {"quiz_id": qid,
               "options": ["a", "b"], "wait_for_answer": True}}
        append_jsonl(tmp_path / "logs/chat.jsonl", row)
        append_jsonl(tmp_path / "logs/chat.jsonl", row)
    append_jsonl(tmp_path / "logs/chat.jsonl", {**row, "quiz": {**row["quiz"], "quiz_id": "optional", "wait_for_answer": False}})
    messages = json.loads(_assemble_history_response(tmp_path, 1, 3, 0))["messages"]
    assert len(messages) == 3
    assert len({(row["task_id"], row["quiz_id"]) for row in messages}) == 3
    assert all(row["system_type"] == "project_question_pointer" for row in messages)


@pytest.mark.parametrize("same_timestamp", [False, True])
def test_activity_question_uses_same_memo_and_preserves_wait_semantics(tmp_path, monkeypatch, same_timestamp):
    from ouroboros.gateway import state as gs
    from ouroboros import utils
    from supervisor import queue

    project = create_project(tmp_path, "waiting-project", name="Waiting Project")
    write_task_result(tmp_path, "t1", STATUS_RUNNING, project_id=project["id"],
                      root_phase_checkpoint={"post_task_synthesis": "running"})
    record_asked(tmp_path, "t1", quiz_id="q1", question="?", options=["a", "b"], wait_for_answer=True)
    set_owner_wait(tmp_path, "t1", {"quiz_id": "q1", "wait_id": "w1", "state": "waiting"})
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "RUNNING", {"t1": {"task": {"id": "t1", "project_id": project["id"], "chat_id": project["chat_id"]}}})
    gs._FINALIZING_MEMO.clear()
    reads = []
    real = utils.read_json_dict
    monkeypatch.setattr(utils, "read_json_dict", lambda path: (reads.append(str(path)), real(path))[1])
    rows = gs._chat_activities_snapshot_safe(tmp_path, direct_turns=[])
    assert rows[0]["required_question"]["quiz_state"] == "open"
    assert reads.count(str(tmp_path / "task_results/t1.json")) == 1
    gs._chat_activities_snapshot_safe(tmp_path, direct_turns=[])
    assert reads.count(str(tmp_path / "task_results/t1.json")) == 1
    path = tmp_path / "task_results/t1.json"
    before = path.stat()
    set_owner_wait(tmp_path, "t1", {"quiz_id": "q1", "wait_id": "w1", "state": "resumed"}, "w1")
    if same_timestamp:
        # Atomic replacement within one filesystem timestamp tick can retain
        # both mtime and size: waiting/resumed have the same serialized length.
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = path.stat()
        assert after.st_ino != before.st_ino
        assert (after.st_mtime_ns, after.st_size) == (before.st_mtime_ns, before.st_size)
    previous_reads = reads.count(str(path))
    rows = gs._chat_activities_snapshot_safe(tmp_path, direct_turns=[])
    assert rows[0]["required_question"]["owner_wait_state"] == "resumed"
    assert rows[0]["required_question"]["text"] == "Question in Waiting Project"
    assert reads.count(str(path)) == previous_reads + 1
    gs._chat_activities_snapshot_safe(tmp_path, direct_turns=[])
    assert reads.count(str(path)) == previous_reads + 1
    assert quiz_states(tmp_path, "t1")["q1"]["state"] == "open"


def test_details_are_immutable_optional_and_length_checked(tmp_path):
    record_asked(tmp_path, "t", quiz_id="new", question="?", options=["a", "b"], option_details=["A", "B"])
    record_asked(tmp_path, "t", quiz_id="new", question="?", options=["a", "b"], option_details=["changed", "changed"])
    record_answered(tmp_path, "t", quiz_id="new", option_index=1, request_id="r")
    record_asked(tmp_path, "t", quiz_id="old", question="?", options=["a", "b"])
    reconcile_terminal(tmp_path, "t")
    assert quiz_states(tmp_path, "t")["new"]["option_details"] == ["A", "B"]
    assert "option_details" not in quiz_states(tmp_path, "t")["old"]
    with pytest.raises(ValueError):
        record_asked(tmp_path, "t", quiz_id="bad", question="?", options=["a", "b"], option_details=["A"])
