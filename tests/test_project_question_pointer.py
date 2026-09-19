"""Project questions share their durable ask with every display lens."""
import ast
import pathlib
import re
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
    result = _escalate(ctx, question="Which storage?", wait_for_answer=True, stake="Where every later write lands.",
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
    # The live pointer frame is complete for display: the browser paints it without a detail read.
    assert pointer["question"] == "Which storage?" and pointer["options"] == ["Local", "Shared"]
    assert pointer["wait_for_answer"] is True and pointer["content"] == "Waiting for your answer in Question Project"
    # Main mirrors the Project's own form: the option details and the stake ride along too.
    details = ["Single-machine files", "Multiple writers over network"]
    assert pointer["option_details"] == details and pointer["stake"] == "Where every later write lands."
    assert not {"comment", "answered_index"} & pointer.keys()
    stored = (tmp_path / "logs/chat.jsonl").read_text().splitlines()
    assert len(stored) == 1 and json.loads(stored[0])["chat_id"] == project["chat_id"]
    main = json.loads(_assemble_history_response(tmp_path, 1, 10, 0))["messages"]
    assert len(main) == 1 and main[0]["quiz_id"] == qid
    assert main[0]["option_details"] == details and main[0]["stake"] == "Where every later write lands."
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
    assert main[0]["quiz_state"] == "answered" and main[0]["answered_index"] == 1
    assert main[0]["question"] == "Which storage?" and main[0]["options"] == ["Local", "Shared"]
    assert main[0]["text"] == "You answered in Question Project"
    assert reconcile_terminal(tmp_path, "task-1") == []
    assert quiz_states(tmp_path, "task-1")[qid]["option_details"][1] == "Multiple writers over network"


def test_pointer_quota_and_dedup(tmp_path):
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


def test_an_optional_question_projects_into_main_with_its_assumption_and_never_reads_as_resumed(tmp_path, monkeypatch):
    """Every Project question is one Main row, whether or not the task waits on it. The optional
    one carries the assumption the task continues under and the asker's recommendation; a newer
    REQUIRED wait of the same task must not repaint it as a wait that ended, because it never
    waited (only a question the task waited on can have been resumed)."""
    from supervisor import message_bus, state

    project = create_project(tmp_path, "mixed-project", name="Mixed Project")
    write_task_result(tmp_path, "t1", STATUS_RUNNING, project_id=project["id"], chat_id=project["chat_id"])
    ctx = _tool_ctx(tmp_path, task_id="t1", chat_id=project["chat_id"])
    ctx.owner_wait_callback = lambda *_: None
    assert _escalate(ctx, question="Which format?", assumption="WebP meanwhile",
                     options=[{"label": "PNG"}, {"label": "WebP", "recommended": True}]).startswith("OK:")
    optional = ctx.pending_events[0]
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    state.init(tmp_path, 100)
    frames = []
    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    assert bridge.send_quiz(project["chat_id"], **{key: optional[key] for key in (
        "quiz_id", "question", "options", "stake", "assumption", "state", "task_id")})[0]
    assert [row["type"] for row in frames] == ["quiz", "chat"]
    live = frames[1]
    assert live["system_type"] == "project_question_pointer" and live["chat_id"] == 1
    assert live["assumption"] == "WebP meanwhile" and live["recommended_index"] == 1
    assert live["options"] == ["PNG", "WebP"] and "wait_for_answer" not in live
    assert live["content"] == "Unanswered · an answer is still accepted in Mixed Project"

    # The same task now waits on a newer required question.
    assert _escalate(ctx, question="Publish?", wait_for_answer=True,
                     options=[{"label": "Yes", "recommended": True}, {"label": "No"}]).startswith("OK:")
    required = ctx.pending_events[1]
    assert bridge.send_quiz(project["chat_id"], **{key: required[key] for key in (
        "quiz_id", "question", "options", "stake", "assumption", "state", "task_id", "wait_for_answer")})[0]
    set_owner_wait(tmp_path, "t1", {"quiz_id": required["quiz_id"], "wait_id": "w1", "state": "waiting"})
    main = {row["quiz_id"]: row for row in json.loads(_assemble_history_response(tmp_path, 1, 10, 0))["messages"]}
    assert set(main) == {optional["quiz_id"], required["quiz_id"]}
    calm, waiting = main[optional["quiz_id"]], main[required["quiz_id"]]
    assert "owner_wait_state" not in calm and "wait_for_answer" not in calm
    assert calm["assumption"] == "WebP meanwhile" and calm["recommended_index"] == 1
    assert calm["text"] == "Unanswered · an answer is still accepted in Mixed Project"
    assert waiting["owner_wait_state"] == "waiting" and waiting["recommended_index"] == 0
    assert waiting["text"] == "Waiting for your answer in Mixed Project"
    # The Project room's own card of the optional question reads the same facts.
    room = {row["quiz"]["quiz_id"]: row["quiz"] for row in json.loads(
        _assemble_history_response(tmp_path, project["chat_id"], 10, 0))["messages"] if row.get("msg_type") == "quiz"}
    assert "owner_wait_state" not in room[optional["quiz_id"]]


@pytest.mark.parametrize("same_timestamp", [False, True])
def test_activity_question_uses_same_memo_and_preserves_wait_semantics(tmp_path, monkeypatch, same_timestamp):
    from ouroboros.gateway import state as gs
    from ouroboros import utils
    from supervisor import queue

    project = create_project(tmp_path, "waiting-project", name="Waiting Project")
    write_task_result(tmp_path, "t1", STATUS_RUNNING, project_id=project["id"],
                      root_phase_checkpoint={"post_task_synthesis": "running"})
    record_asked(tmp_path, "t1", quiz_id="q1", question="?", options=["a", "b"], wait_for_answer=True,
                 recommended_index=0, option_details=["Alpha", ""], stake="The rest of the run")
    set_owner_wait(tmp_path, "t1", {"quiz_id": "q1", "wait_id": "w1", "state": "waiting"})
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "RUNNING", {"t1": {"task": {"id": "t1", "project_id": project["id"], "chat_id": project["chat_id"]}}})
    gs._FINALIZING_MEMO.clear()
    reads = []
    real = utils.read_json_dict
    monkeypatch.setattr(utils, "read_json_dict", lambda path: (reads.append(str(path)), real(path))[1])
    rows = gs._chat_activities_snapshot_safe(tmp_path, direct_turns=[])
    assert rows[0]["required_question"]["quiz_state"] == "open"
    assert rows[0]["required_question"]["text"] == "Waiting for your answer in Waiting Project"
    # The census pointer is as complete as the history row: the browser never paints a blank over it.
    assert rows[0]["required_question"]["question"] == "?" and rows[0]["required_question"]["options"] == ["a", "b"]
    assert rows[0]["required_question"]["option_details"] == ["Alpha", ""]
    assert rows[0]["required_question"]["stake"] == "The rest of the run"
    # A card painted from the census alone still badges the recommended option (index zero included).
    assert rows[0]["required_question"]["recommended_index"] == 0
    # Folding an older Main card needs the named question's OWN asked_at, so the census
    # pointer carries it as `ts`. Losing that stamp does not fail loudly — it silently
    # stops every fold (web/modules/chat_decision.js::appendActivityQuestion), so the
    # cross-boundary contract is pinned on the producer side too.
    assert rows[0]["required_question"]["ts"] == quiz_states(tmp_path, "t1")["q1"]["asked_at"]
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
    assert rows[0]["required_question"]["text"] == "Unanswered · the task continued; an answer is still accepted in Waiting Project"
    assert reads.count(str(path)) == previous_reads + 1
    gs._chat_activities_snapshot_safe(tmp_path, direct_turns=[])
    assert reads.count(str(path)) == previous_reads + 1
    assert quiz_states(tmp_path, "t1")["q1"]["state"] == "open"
    # A wait that ended on its OWN bound resumed without an answer, so the
    # question is still wanted: no new wait state, just the additive reason.
    set_owner_wait(tmp_path, "t1", {"quiz_id": "q1", "wait_id": "w1", "state": "resumed",
                                    "resume_reason": "timeout"}, "w1")
    gs._FINALIZING_MEMO.clear()
    rows = gs._chat_activities_snapshot_safe(tmp_path, direct_turns=[])
    pointer = rows[0]["required_question"]
    assert pointer["text"] == "Unanswered · the task continued; an answer is still accepted in Waiting Project"
    assert pointer["owner_wait_resume_reason"] == "timeout"


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


def test_question_presentation_shared_fixture():
    """Both sides of the parity fixture: Python emits exactly the fixture row for each case, and the
    browser (web/tests/question_presentation.test.js) reads the fixture status out of that row."""
    from pathlib import Path
    from ouroboros.project_dialogue import project_question_pointer

    cases = json.loads((Path(__file__).resolve().parents[1] /
                        'web/tests/fixtures/question_presentation_parity.json').read_text(encoding="utf-8"))
    keys = ("quiz_state", "owner_wait_state", "owner_wait_resume_reason", "wait_for_answer",
            "wait_ended_at", "answered_index", "comment", "source_status")
    for case in cases:
        block = {"quiz_id": "q", **case["block"]} if case["block"] is not None else None
        wait = {"quiz_id": "q", **case["owner_wait"]} if case["owner_wait"] is not None else None
        pointer = project_question_pointer(
            {"task_id": "task", "text": "Which?", "quiz": {"quiz_id": "q", "wait_for_answer": True, "options": ["a", "b"]}},
            block, {"id": "p", "chat_id": 12, "name": "Project"}, wait)
        assert pointer["text"] == case["status"] + " in Project", case["case"]
        assert {key: pointer[key] for key in keys if key in pointer} == case["row"], case["case"]
        assert pointer["question"] == "Which?" and pointer["options"] == ["a", "b"]
        # A legacy label-only ask retained no details; Main says so exactly as the Project card does.
        assert "option_details" not in pointer and "stake" not in pointer
    narrow = project_question_pointer({"task_id": "task", "quiz_id": "q", "wait_for_answer": True},
                                      {"quiz_id": "q", "state": "open"}, {"id": "p", "chat_id": 12, "name": "Project"}, None)
    assert not {"question", "options", "option_details", "stake"} & narrow.keys(), \
        "unknown display fields are omitted, never blanked"


def test_project_room_quiz_row_carries_the_wait_facts(tmp_path):
    """A wait the owner resumed by ordinary input leaves no frame behind: the replayed card
    reads the task's wait record like the Main pointer does."""
    from ouroboros.utils import append_jsonl

    project = create_project(tmp_path, "resumed-project", name="Resumed Project")
    write_task_result(tmp_path, "t1", STATUS_RUNNING, project_id=project["id"], chat_id=project["chat_id"])
    record_asked(tmp_path, "t1", quiz_id="q1", question="Which?", options=["a", "b"], wait_for_answer=True)
    append_jsonl(tmp_path / "logs/chat.jsonl", {"type": "quiz", "direction": "out", "chat_id": project["chat_id"],
                 "task_id": "t1", "ts": "2026-09-16T00:00:00Z", "text": "Which?",
                 "quiz": {"quiz_id": "q1", "options": ["a", "b"], "wait_for_answer": True}})
    room = json.loads(_assemble_history_response(tmp_path, project["chat_id"], 10, 0))["messages"]
    assert "owner_wait_state" not in room[0]["quiz"]
    set_owner_wait(tmp_path, "t1", {"quiz_id": "q1", "wait_id": "w1", "state": "waiting"})
    room = json.loads(_assemble_history_response(tmp_path, project["chat_id"], 10, 0))["messages"]
    assert room[0]["quiz"]["owner_wait_state"] == "waiting"
    set_owner_wait(tmp_path, "t1", {"quiz_id": "q1", "wait_id": "w1", "state": "resumed"})
    room = json.loads(_assemble_history_response(tmp_path, project["chat_id"], 10, 0))["messages"]
    assert room[0]["quiz"]["owner_wait_state"] == "resumed" and room[0]["quiz"]["wait_for_answer"] is True
    main = json.loads(_assemble_history_response(tmp_path, 1, 10, 0))["messages"]
    assert main[0]["text"] == "Unanswered · the task continued; an answer is still accepted in Resumed Project"
    set_owner_wait(tmp_path, "t1", {"quiz_id": "q2", "wait_id": "w2", "state": "waiting"})
    room = json.loads(_assemble_history_response(tmp_path, project["chat_id"], 10, 0))["messages"]
    assert room[0]["quiz"]["owner_wait_state"] == "resumed"


from tests.test_contracts import REPO_ROOT, _dict_type_discriminator, _dict_literal_keys


def _string_tuple(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Tuple) and node.elts and all(
            isinstance(elt, ast.Constant) and isinstance(elt.value, str) for elt in node.elts):
        return tuple(elt.value for elt in node.elts)
    return None


def _function_node(source_path: pathlib.Path, name: str) -> ast.AST:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    found = [node for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    assert len(found) == 1, f"expected one {name} in {source_path.name}"
    return found[0]


def test_project_question_pointer_display_fields_share_one_contract():
    """Main mirrors a Project question as the Project's own form, so each pointer producer and
    both ABI mirrors carry the same display fields, ``option_details`` and ``stake`` included.

    The live frame (``message_bus.send_quiz``) copies them in a key loop and the census
    (``state._task_activity_facts``) in a key tuple, neither of which the envelope literal scan
    above can see; the browser keeps what it merges in ``chat_decision.js`` MIRROR_FIELDS."""
    from typing import get_args, get_origin, get_type_hints

    from ouroboros.gateway.contracts import ChatOutbound
    from ouroboros.project_dialogue import project_question_pointer

    hints = get_type_hints(ChatOutbound, include_extras=True)
    assert "option_details" in hints and "stake" in hints
    # Both optional: a legacy label-only ask retained no details, and most asks name no stake.
    for field in ("option_details", "stake"):
        assert str(hints[field]).startswith(("typing.NotRequired[", "typing_extensions.NotRequired[")), hints[field]
    details = get_args(hints["option_details"])[0]
    assert get_origin(details) is list and get_args(details) == (str,)
    assert get_args(hints["stake"]) == (str,)
    api_types = (REPO_ROOT / "web" / "modules" / "api_types.js").read_text(encoding="utf-8")
    typedef = re.search(r"@typedef \{Object\} ChatOutbound\b(?P<body>.*?)\n \*/", api_types, re.S)
    assert typedef, "api_types.js missing ChatOutbound"
    assert re.search(r"@property \{string\[\]=\} option_details\b", typedef.group("body"))
    assert re.search(r"@property \{string=\} stake\b", typedef.group("body"))

    mirror = re.search(r"const MIRROR_FIELDS = \[([^\]]*)\]",
                       (REPO_ROOT / "web" / "modules" / "chat_decision.js").read_text(encoding="utf-8"))
    assert mirror, "chat_decision.js MIRROR_FIELDS moved"
    browser_fields = set(re.findall(r"'([a-z_]+)'", mirror.group(1)))
    assert {"option_details", "stake"} <= browser_fields
    assert browser_fields <= set(hints), sorted(browser_fields - set(hints))

    # History: the producer itself emits every field the browser merges.
    block = {"quiz_id": "q", "state": "open", "question": "Which?", "options": ["a", "b"],
             "option_details": ["A detail", ""], "stake": "What rides on it", "assumption": "a",
             "recommended_index": 0}
    pointer = project_question_pointer({"task_id": "t", "quiz_id": "q"}, block,
                                       {"id": "p", "chat_id": 12, "name": "Project"}, None)
    assert browser_fields <= set(pointer), sorted(browser_fields - set(pointer))
    assert pointer["option_details"] == ["A detail", ""] and pointer["stake"] == "What rides on it"

    # Live delivery: the frame literal plus its copied key loop, all declared in ChatOutbound.
    send_quiz = _function_node(REPO_ROOT / "supervisor" / "message_bus.py", "send_quiz")
    frames = [node for node in ast.walk(send_quiz) if isinstance(node, ast.Dict)
              and _dict_type_discriminator(node) == "chat"]
    assert len(frames) == 1, "send_quiz builds one pointer frame"
    literal, unknown = _dict_literal_keys(frames[0])
    assert not unknown
    copied = [_string_tuple(node.iter) for node in ast.walk(send_quiz)
              if isinstance(node, ast.For) and _string_tuple(node.iter) and "recommended_index" in _string_tuple(node.iter)]
    assert len(copied) == 1, "send_quiz copies the pointer's display fields in one key loop"
    live = literal | set(copied[0])
    assert {"option_details", "stake"} <= set(copied[0])
    assert browser_fields <= live, sorted(browser_fields - live)
    assert live <= set(hints), sorted(live - set(hints))

    # Census: the quiz facts the stat-keyed memo keeps for required_question.
    facts = _function_node(REPO_ROOT / "ouroboros" / "gateway" / "state.py", "_task_activity_facts")
    census = [_string_tuple(gen.iter) for node in ast.walk(facts) if isinstance(node, ast.DictComp)
              for gen in node.generators if _string_tuple(gen.iter) and "question" in _string_tuple(gen.iter)]
    assert len(census) == 1, "_task_activity_facts keeps the quiz display fields in one key tuple"
    quiz_sourced = browser_fields - {"project_name"}
    assert quiz_sourced <= set(census[0]), sorted(quiz_sourced - set(census[0]))
