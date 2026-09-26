"""Accepted quiz answers remain exact room evidence after lifecycle/mailbox GC."""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from ouroboros.gateway.task_decision import answer_decision
from ouroboros.memory import Memory
from ouroboros.owner_mailbox import cleanup_task_mailbox, drain_owner_entries
from ouroboros.owner_quiz import quiz_states, record_asked
from supervisor import message_bus, queue, state


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    # Explicitly bind process-global roots before any production helper runs.
    for module, names in (
        (state, ("DRIVE_ROOT", "STATE_PATH", "STATE_LAST_GOOD_PATH", "STATE_LOCK_PATH")),
        (queue, ("DRIVE_ROOT", "QUEUE_SNAPSHOT_PATH")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))
    state.init(tmp_path)
    queue.init(tmp_path)
    assert state.DRIVE_ROOT == queue.DRIVE_ROOT == tmp_path
    assert state.STATE_PATH == tmp_path / "state" / "state.json"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    bridge = message_bus.LocalChatBridge()
    frames = []
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    task = {"id": "task-quiz", "chat_id": 1, "drive_root": str(tmp_path)}
    monkeypatch.setattr(queue, "RUNNING", {task["id"]: {"task": task}})
    monkeypatch.setattr(queue, "PENDING", [])
    return SimpleNamespace(root=tmp_path, bridge=bridge, frames=frames, task=task)


def _ask(runtime, quiz_id, *, chat_id=1):
    block = record_asked(
        runtime.root, runtime.task["id"], quiz_id=quiz_id, chat_id=chat_id,
        question=f"Choose for {quiz_id}?", options=["First", "Second"],
        option_details=["First benefit and cost", "Second benefit and cost"],
        recommended_index=1, stake="Delivery time", assumption="Research meanwhile",
    )
    ok, detail = runtime.bridge.send_quiz(
        chat_id, quiz_id, block["question"],
        [{"label": "First", "detail": block["option_details"][0]},
         {"label": "Second", "detail": block["option_details"][1], "recommended": True}],
        stake=block["stake"], assumption=block["assumption"], task_id=runtime.task["id"],
    )
    assert ok, detail


def _answer(runtime, quiz_id, *, request_id, index=0, comment=""):
    body = {"request_id": request_id,
            "decision_id": f"quiz:{runtime.task['id']}:{quiz_id}", "comment": comment}
    if index is not None:
        body["option_index"] = index
    return asyncio.run(answer_decision(runtime.root, body))


def _facts(runtime):
    rows, coverage = Memory(runtime.root).read_chat_generations()
    assert coverage["snapshot_stable"] and not coverage["gaps"]
    return [row for row in rows if row.get("type") == "quiz_answer"]


def test_answers_survive_eighteen_quizzes_mailbox_gc_and_rotation(runtime):
    for index in range(18):
        quiz_id = f"q{index:02}"
        _ask(runtime, quiz_id)
        selected = 0 if index % 2 == 0 else None
        assert _answer(runtime, quiz_id, request_id=f"answer-{index}", index=selected,
                       comment=f"  Verbatim choice {index}\nsecond line  ")[0] == 200
        if index == 8:
            state.rotate_jsonl_log_if_needed(runtime.root, "chat.jsonl", "chat", max_bytes=1)
    assert len(quiz_states(runtime.root, runtime.task["id"])) == 16
    assert "q00" not in quiz_states(runtime.root, runtime.task["id"])
    cleanup_task_mailbox(runtime.root, runtime.task["id"])
    state.rotate_jsonl_log_if_needed(runtime.root, "chat.jsonl", "chat", max_bytes=1)
    assert not drain_owner_entries(runtime.root, runtime.task["id"], include_acknowledged=True)
    facts = _facts(runtime)
    assert len(facts) == 18
    for index, row in enumerate(facts):
        quiz = row["quiz"]
        assert row["source"] == "owner_quiz_answer"
        assert row["client_message_id"] == f"quiz_answer:task-quiz:q{index:02}"
        assert row["ts"] == quiz["answered_at"] and quiz["asked_at"] <= quiz["answered_at"]
        assert quiz["question"] == f"Choose for q{index:02}?"
        assert quiz["options"] == ["First", "Second"]
        assert quiz["option_details"] == ["First benefit and cost", "Second benefit and cost"]
        assert quiz["recommended_index"] == 1
        assert quiz["comment"] == f"  Verbatim choice {index}\nsecond line  "
        assert quiz["request_id"] == f"answer-{index}"
        if index % 2:
            assert "answered_index" not in quiz and (
                "answered in their own words without choosing an offered option" in row["text"])
            assert "rejected" not in row["text"]
        else:
            assert quiz["answered_index"] == 0 and "chose option 1: First" in row["text"]
    assert len([frame for frame in runtime.frames if frame.get("type") == "quiz"]) == 18
    assert len([frame for frame in runtime.frames if frame.get("type") == "quiz_state"]) == 18
    assert not [frame for frame in runtime.frames if frame.get("type") == "chat"]


def test_a_late_answer_keeps_its_evidence_row_and_also_enters_dialogue(runtime, monkeypatch):
    """В17a=A sibling: a card answered after its task finished keeps the SAME
    durable quiz_answer evidence row, and because no mailbox will be drained the
    answer additionally becomes the owner's own message in the card's chat."""
    from ouroboros.owner_quiz import reconcile_terminal

    _ask(runtime, "late")
    monkeypatch.setattr(queue, "RUNNING", {})  # the author is gone
    assert reconcile_terminal(runtime.root, runtime.task["id"]) == ["late"]
    status, body = _answer(runtime, "late", request_id="late-1", index=1,
                           comment="  After the fact  ")
    assert status == 200, body
    assert body["answered_after_terminal"] is True and body["forwarded"] is True

    [fact] = [row for row in _facts(runtime) if row["quiz"]["quiz_id"] == "late"]
    assert fact["client_message_id"] == "quiz_answer:task-quiz:late"
    assert fact["source"] == "owner_quiz_answer"
    assert fact["quiz"]["answered_after_terminal"] is True
    assert fact["quiz"]["comment"] == "  After the fact  "

    rows, coverage = Memory(runtime.root).read_chat_generations()
    assert coverage["snapshot_stable"]
    inbound = [row for row in rows
               if row.get("client_message_id") == "quiz_late_answer:task-quiz:late"]
    assert len(inbound) == 1 and inbound[0]["direction"] == "in"
    assert inbound[0]["chat_id"] == 1 and inbound[0]["source"] == "web"
    # The owner's row is the owner's words (the ingress strips edge whitespace
    # of every owner row), never the host frame; the frame is for the model.
    assert inbound[0]["text"] == "After the fact"
    assert "[Owner quiz answer]" not in inbound[0]["text"]
    chats = [frame for frame in runtime.frames if frame.get("type") == "chat"]
    assert [frame["role"] for frame in chats] == ["user"]
    assert chats[0]["content"] == "After the fact"

    # Reload: history replays the late answer as an ordinary user row carrying
    # the owner's words; no path re-injects the frame into the bubble.
    from ouroboros.gateway.history import make_chat_history_endpoint

    response = asyncio.run(make_chat_history_endpoint(runtime.root)(
        SimpleNamespace(query_params={"n_human": "100", "thread": "1"})))
    messages = json.loads(response.body)["messages"]
    replayed = [row for row in messages
                if row.get("client_message_id") == "quiz_late_answer:task-quiz:late"]
    assert len(replayed) == 1 and replayed[0]["role"] == "user"
    assert replayed[0]["text"] == "After the fact"
    assert not [row for row in messages if row.get("role") == "user"
                and "[Owner quiz answer]" in str(row.get("text") or "")]


@pytest.mark.parametrize("initial_index", [0, None])
def test_same_request_retry_after_rotation_preserves_winning_source(runtime, initial_index):
    _ask(runtime, "retry")
    assert _answer(runtime, "retry", request_id="one", index=initial_index,
                   comment="  Winning words  ")[0] == 200
    state.rotate_jsonl_log_if_needed(runtime.root, "chat.jsonl", "chat", max_bytes=1)
    before = _facts(runtime)
    status, result = _answer(runtime, "retry", request_id="one", index=1, comment="Changed retry")
    assert status == 200 and result["duplicate"] is True
    assert _facts(runtime) == before
    assert len(before) == 1 and before[0]["quiz"]["comment"] == "  Winning words  "
    assert before[0]["quiz"].get("answered_index") == initial_index


@pytest.mark.parametrize("force_overlap", [False, True])
def test_competing_answers_preserve_only_the_recorded_winner(runtime, monkeypatch, force_overlap):
    from ouroboros.dialogue_evidence import read_room_source
    from ouroboros.gateway.history import make_chat_history_endpoint

    _ask(runtime, "race")
    if force_overlap:
        original = message_bus.log_chat
        barrier = threading.Barrier(2, timeout=10)

        def append_after_both_history_checks(*args, **kwargs):
            if kwargs.get("record_type") == "quiz_answer":
                barrier.wait()
            return original(*args, **kwargs)

        monkeypatch.setattr(message_bus, "log_chat", append_after_both_history_checks)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda index: _answer(
            runtime, "race", request_id=f"request-{index}", index=index,
            comment=f"Owner choice {index}",
        ), [0, 1]))
    assert sorted(status for status, _ in outcomes) == [200, 409]
    winner = quiz_states(runtime.root, runtime.task["id"])["race"]
    # Concurrent healing may append the same source twice; the dialogue reader
    # owns identity deduplication, while every physical row must keep the winner.
    facts = _facts(runtime)
    assert facts and all(fact["quiz"] == winner for fact in facts)
    assert {fact["client_message_id"] for fact in facts} == {"quiz_answer:task-quiz:race"}
    if force_overlap:
        assert len(facts) == 2
    source = read_room_source(runtime.root, 1, task_id=runtime.task["id"])
    [fact] = [row for row in source["rows"] if row.get("type") == "quiz_answer"]
    assert fact["quiz"] == winner
    assert fact["quiz"]["comment"] == f"Owner choice {winner['answered_index']}"
    response = asyncio.run(make_chat_history_endpoint(runtime.root)(
        SimpleNamespace(query_params={"n_human": "100", "thread": "1"})))
    messages = json.loads(response.body)["messages"]
    [card] = [row for row in messages if row.get("msg_type") == "quiz"]
    assert card["quiz"]["answered_index"] == winner["answered_index"]
    assert card["quiz"]["comment"] == winner["comment"]
    assert not [row for row in messages if row.get("system_type") == "quiz_answer"]
    assert len([frame for frame in runtime.frames if frame.get("type") == "quiz_state"]) == 1
    [delivery] = drain_owner_entries(runtime.root, runtime.task["id"], include_acknowledged=True)
    assert delivery["text"] == fact["text"]


def test_retry_heals_history_write_failure_from_the_winning_block(runtime, monkeypatch):
    _ask(runtime, "heal")
    real = message_bus.log_chat

    def fail_answer(*args, **kwargs):
        if kwargs.get("record_type") == "quiz_answer":
            raise OSError("simulated history write failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(message_bus, "log_chat", fail_answer)
    status, result = _answer(runtime, "heal", request_id="same", index=0, comment="Accepted")
    assert status == 503 and result["reason_code"] == "quiz_history_write_failed"
    assert quiz_states(runtime.root, runtime.task["id"])["heal"]["answered_index"] == 0
    assert not _facts(runtime)
    monkeypatch.setattr(message_bus, "log_chat", real)
    status, result = _answer(runtime, "heal", request_id="same", index=1, comment="Retry payload")
    assert status == 200 and result["duplicate"] is True
    [fact] = _facts(runtime)
    assert fact["quiz"]["answered_index"] == 0 and fact["quiz"]["comment"] == "Accepted"
    [mailbox] = drain_owner_entries(runtime.root, runtime.task["id"], include_acknowledged=True)
    assert mailbox["text"] == fact["text"]


def test_answer_history_uses_canonical_root_and_project_binding(runtime):
    from ouroboros.projects_registry import bind_task_to_project, create_project

    project = create_project(runtime.root, "quiz-room", name="Quiz room")
    bind_task_to_project(runtime.root, runtime.task["id"], project["id"], origin={"absent": "system"})
    child = runtime.root / "child-drive"
    runtime.task["drive_root"] = str(child)
    _ask(runtime, "project", chat_id=project["chat_id"])
    assert _answer(runtime, "project", request_id="project-answer", comment="Keep the room")[0] == 200
    [fact] = _facts(runtime)
    assert fact["chat_id"] == project["chat_id"]
    assert not (child / "logs" / "chat.jsonl").exists()
    [mailbox] = drain_owner_entries(child, runtime.task["id"], include_acknowledged=True)
    assert mailbox["text"] == fact["text"]


def test_history_recovers_evicted_answer_without_an_extra_bubble(runtime):
    import json
    from ouroboros.gateway.history import make_chat_history_endpoint

    for index in range(18):
        qid = f"history-{index}"
        _ask(runtime, qid)
        assert _answer(runtime, qid, request_id=qid, index=1, comment=f"choice-{index}")[0] == 200
    cleanup_task_mailbox(runtime.root, runtime.task["id"])
    response = asyncio.run(make_chat_history_endpoint(runtime.root)(
        SimpleNamespace(query_params={"n_human": "100", "thread": "1"})))
    messages = json.loads(response.body)["messages"]
    cards = [row for row in messages if row.get("msg_type") == "quiz"]
    assert len(cards) == 18
    assert not [row for row in messages if row.get("system_type") == "quiz_answer"]
    first = next(row["quiz"] for row in cards if row["quiz"]["quiz_id"] == "history-0")
    assert first["state"] == "answered" and first["answered_index"] == 1
    assert first["comment"] == "choice-0" and first["options"][1]["recommended"] is True


def test_recent_room_and_history_share_parent_root_and_origin_membership(runtime):
    from ouroboros.gateway.history import _make_thread_filter
    from ouroboros.project_dialogue import project_recent_dialogue
    from ouroboros.projects_registry import bind_task_to_project, create_project

    project = create_project(runtime.root, "membership", name="Membership")
    bind_task_to_project(runtime.root, "parent", project["id"], origin={"absent": "system"})
    for field in ("parent_task_id", "root_task_id"):
        message_bus.log_chat("out", 1, 0, field, task_id="child-" + field,
                             message_meta={field: "parent"}, drive_root=runtime.root)
    rows, coverage, _origins = project_recent_dialogue(Memory(runtime.root), project["chat_id"], 20)
    assert {row["text"] for row in rows} == {"parent_task_id", "root_task_id"}
    project_filter = _make_thread_filter(project["chat_id"], {project["chat_id"]}, [], {"parent": project["chat_id"]})
    main_filter = _make_thread_filter(1, {project["chat_id"]}, [], {"parent": project["chat_id"]})
    assert all(project_filter(row["chat_id"], row) and not main_filter(row["chat_id"], row) for row in rows)


def test_new_request_heals_committed_answer_before_projection_eviction(runtime):
    from ouroboros.owner_quiz import record_answered

    _ask(runtime, 'crash')
    # Exact crash boundary: projection committed, canonical history not written.
    assert record_answered(runtime.root, runtime.task['id'], quiz_id='crash', option_index=0,
                           request_id='winner', comment='Original owner answer')['ok']
    status, reply = _answer(runtime, 'crash', request_id='new-browser-request', index=1, comment='Losing payload')
    assert status == 409 and reply['answered_index'] == 0
    for i in range(16):
        _ask(runtime, f'following-{i}')
        assert _answer(runtime, f'following-{i}', request_id=f'following-{i}')[0] == 200
    cleanup_task_mailbox(runtime.root, runtime.task['id'])
    state.rotate_jsonl_log_if_needed(runtime.root, 'chat.jsonl', 'chat', max_bytes=1)
    assert 'crash' not in quiz_states(runtime.root, runtime.task['id'])
    [fact] = [row for row in _facts(runtime) if row['quiz']['quiz_id'] == 'crash']
    assert fact['quiz']['comment'] == 'Original owner answer'


def test_competing_request_history_failure_is_retryable_without_changing_winner(runtime, monkeypatch):
    from ouroboros.owner_quiz import record_answered

    _ask(runtime, 'recover-new-id')
    record_answered(runtime.root, runtime.task['id'], quiz_id='recover-new-id', option_index=0,
                    request_id='winner', comment='Winning choice')
    real = message_bus.log_chat
    def fail(*args, **kwargs):
        if kwargs.get('record_type') == 'quiz_answer':
            raise OSError('unavailable history')
        return real(*args, **kwargs)
    monkeypatch.setattr(message_bus, 'log_chat', fail)
    status, response = _answer(runtime, 'recover-new-id', request_id='loser', index=1)
    assert status == 503 and response['reason_code'] == 'quiz_history_write_failed'
    monkeypatch.setattr(message_bus, 'log_chat', real)
    status, response = _answer(runtime, 'recover-new-id', request_id='another-retry', index=1)
    assert status == 409 and response['answered_index'] == 0
    [fact] = _facts(runtime)
    assert fact['quiz']['comment'] == 'Winning choice'


@pytest.mark.parametrize('new_request_id', [False, True])
def test_history_failure_retry_delivers_winner_to_task(runtime, monkeypatch, new_request_id):
    _ask(runtime, 'redelivery')
    original = message_bus.log_chat
    def fail(*args, **kwargs):
        if kwargs.get('record_type') == 'quiz_answer':
            raise OSError('history unavailable')
        return original(*args, **kwargs)
    monkeypatch.setattr(message_bus, 'log_chat', fail)
    status, result = _answer(runtime, 'redelivery', request_id='winner', index=0, comment='Winning words')
    assert status == 503 and result['reason_code'] == 'quiz_history_write_failed'
    assert not drain_owner_entries(runtime.root, runtime.task['id'], include_acknowledged=True)
    monkeypatch.setattr(message_bus, 'log_chat', original)
    status, result = _answer(runtime, 'redelivery', request_id='new-id' if new_request_id else 'winner',
                             index=1, comment='Losing retry payload')
    assert status == (409 if new_request_id else 200)
    [message] = drain_owner_entries(runtime.root, runtime.task['id'], include_acknowledged=True)
    assert 'Winning words' in message['text'] and 'Losing retry payload' not in message['text']
    assert _answer(runtime, 'redelivery', request_id='another-competitor', index=1)[0] == 409
    assert len(drain_owner_entries(runtime.root, runtime.task['id'], include_acknowledged=True)) == 1


def test_new_request_delivery_failure_keeps_retryable_winning_answer(runtime, monkeypatch):
    from ouroboros.owner_quiz import record_answered
    from ouroboros import owner_mailbox

    _ask(runtime, 'mailbox-heal')
    record_answered(runtime.root, runtime.task['id'], quiz_id='mailbox-heal', option_index=0,
                    request_id='winner', comment='Only this answer')
    original = owner_mailbox.write_owner_message
    monkeypatch.setattr(owner_mailbox, 'write_owner_message', lambda *a, **kw: False)
    status, result = _answer(runtime, 'mailbox-heal', request_id='new-id', index=1)
    assert status == 503 and result['reason_code'] == 'mailbox_write_failed'
    monkeypatch.setattr(owner_mailbox, 'write_owner_message', original)
    assert _answer(runtime, 'mailbox-heal', request_id='another-id', index=1)[0] == 409
    [message] = drain_owner_entries(runtime.root, runtime.task['id'], include_acknowledged=True)
    assert 'Only this answer' in message['text']
