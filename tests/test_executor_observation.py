"""Live executor facts survive the real progress/delivery/history seams."""

import asyncio
import json
import queue
from copy import deepcopy
from functools import partial
from types import SimpleNamespace

import pytest

from ouroboros import delegate_progress as progress
from ouroboros.agent import OuroborosAgent
from ouroboros.delegate_custody import RunCustody
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.subagent_messages import executor_observation_meta, subagent_message_meta
from supervisor import events_chat_delivery, message_bus


def _detail(*, seq=5, attempt="a01", harness="cursor", phase="harness.event"):
    return {
        "lastSeq": seq,
        "summary": {"runId": "run-1", "state": "running", "effectiveAccess": "readonly",
                    "route": {"observedModel": "old-final-model", "harnessId": "old-harness"}},
        "timeline": [{"type": phase, "title": "working", "harnessId": harness, "attemptId": attempt}],
    }


def _entry(**changes):
    return RunCustody(**{
        "task_id": "child", "run_id": "run-1", "route_id": "cursor",
        "model": "requested-model", "access": "readonly", **changes,
    })


def _observation(**changes):
    return {
        "task_id": "child", "task_attempt": "0", "run_id": "run-1", "attempt_id": "a01",
        "harness_id": "cursor", "phase": "harness.event", "revision": 5,
        "model": "requested-model", "model_source": "requested", **changes,
    }


def _project(detail=None, entry=None, **ctx_fields):
    detail = _detail() if detail is None else detail
    ctx = SimpleNamespace(task_id="child", task_attempt=0, **ctx_fields)
    advance = progress.WindowObservations().record(detail, detail["lastSeq"], 0)
    return progress.executor_observation(ctx, "run-1", advance, detail, entry or _entry())


def test_typed_live_actor_does_not_borrow_the_final_route_model():
    detail = _detail()
    original = deepcopy(detail)
    assert _project(detail) == _observation()
    assert detail == original
    later = _detail(seq=8, attempt="a02", harness="claude", phase="reviewer.started")
    expected = _observation(revision=8, attempt_id="a02", harness_id="claude", phase="reviewer.started")
    expected.pop("model")
    expected.pop("model_source")
    assert _project(later) == expected


def test_actor_comes_from_raw_typed_fields_even_outside_the_display_tail():
    detail = _detail()
    detail["timeline"][0]["attemptId"] = "a" * 350
    detail["timeline"] += [{"type": "run.event", "title": "Claude with a new model"}] * 15
    assert _project(detail)["attempt_id"] == "a" * 350
    assert _project(detail)["harness_id"] == "cursor"
    detail["timeline"] = [{"type": "run.event", "title": "[cursor/a02] model-x is working"}]
    assert _project(detail) == {}


@pytest.mark.parametrize("change", [
    {"task_id": "other-task"}, {"run_id": "other-run"},
])
def test_foreign_custody_does_not_mint_an_observation(change):
    assert _project(entry=_entry(**change)) == {}


def test_snapshot_run_and_revision_must_match_the_emitted_advance():
    detail = _detail()
    advance = progress.WindowObservations().record(detail, 5, 0)
    ctx = SimpleNamespace(task_id="child", task_attempt=0)
    detail["summary"]["runId"] = "other-run"
    assert progress.executor_observation(ctx, "run-1", advance, detail, _entry()) == {}
    detail["summary"]["runId"] = "run-1"
    detail["lastSeq"] = 8
    assert progress.executor_observation(ctx, "run-1", advance, detail, _entry()) == {}


@pytest.mark.parametrize("changes", [
    {"task_id": "parent"}, {"task_attempt": "1"}, {"run_id": ""},
    {"attempt_id": ""}, {"harness_id": None}, {"revision": True}, {"revision": -1},
])
def test_delivery_rejects_cross_task_attempt_or_incomplete_identity(changes):
    assert executor_observation_meta(_observation(**changes), task_id="child", task_attempt=0) == {}


def test_observation_projection_preserves_zero_and_unknown_without_mutating_input():
    value = _observation(extra="not a field")
    assert executor_observation_meta(value, task_id="child", task_attempt=0) == _observation()
    assert value["extra"] == "not a field"
    unknown = _observation(task_attempt="")
    assert executor_observation_meta(unknown, task_id="child") == unknown
    assert executor_observation_meta(unknown, task_id="child", task_attempt=0) == {}
    value["model_source"] = "guessed"
    projected = executor_observation_meta(value, task_id="child")
    assert "model" not in projected and "model_source" not in projected


def test_plain_emit_stays_positional_and_metadata_errors_are_not_retried():
    detail = _detail()
    advance = progress.WindowObservations().record(detail, 5, 0)
    plain = []
    progress.emit(SimpleNamespace(emit_progress_fn=plain.append), "run-1", advance)
    assert plain == [progress.live_line("run-1", advance)]
    calls = []

    def broken(text, **metadata):
        calls.append((text, metadata))
        raise TypeError("an internal callback bug")

    ctx = SimpleNamespace(task_id="child", task_attempt=0, emit_progress_fn=broken)
    progress.emit(ctx, "run-1", advance, detail=detail, entry=_entry())
    assert len(calls) == 1
    assert calls[0][1] == {"executor_observation": _observation()}


def _agent(tool_ctx):
    events = queue.Queue()
    agent = SimpleNamespace(
        _last_progress_ts=None, _event_queue=events, _current_chat_id=1,
        _current_task_id="child", tools=SimpleNamespace(_ctx=tool_ctx),
        _subagent_progress_meta=lambda event: subagent_message_meta({
            "delegation_role": "subagent", "root_task_id": "root", "parent_task_id": "root",
            "subagent_role": "critic", "model": "coordinator-model", "executor_route": "cursor",
        }, task_id="child", event=event),
    )
    tool_ctx.emit_progress_fn = partial(OuroborosAgent._emit_progress, agent)
    return agent, events


def test_a_later_plain_note_does_not_inherit_the_previous_executor_observation():
    agent, events = _agent(SimpleNamespace(task_attempt=0))
    OuroborosAgent._emit_progress(agent, "run progress", executor_observation=_observation())
    OuroborosAgent._emit_progress(agent, "coordinator resumes")
    assert events.get_nowait()["progress_meta"]["executor_observation"] == _observation()
    assert "executor_observation" not in events.get_nowait()["progress_meta"]


@pytest.mark.parametrize("changes", [{"task_id": "old-task"}, {"task_attempt": "1"}])
def test_agent_cannot_relabel_a_late_callback_as_its_new_task_or_attempt(changes):
    agent, events = _agent(SimpleNamespace(task_attempt=0))
    OuroborosAgent._emit_progress(agent, "late progress", executor_observation=_observation(**changes))
    event = events.get_nowait()
    assert "executor_observation" not in event["progress_meta"]
    assert event["progress_meta"]["model"] == "coordinator-model"


def test_wait_to_live_chat_to_history_keeps_each_own_actor_without_terminal_claim(tmp_path, monkeypatch):
    from ouroboros.gateways import claudexor
    from ouroboros.tools import delegate
    from tests._delegated_transport_shared import _nanny_ctx

    ctx = _nanny_ctx(tmp_path, task_id="child")
    ctx.task_attempt = 0
    _agent_obj, events = _agent(ctx)
    entry = _entry()
    monkeypatch.setitem(delegate._CUSTODY, "run-1", entry)
    clock = SimpleNamespace(now=0.0)

    class Gateway:
        reads = 0
        closed = False

        def handshake(self, **kwargs):
            return {"compatible": True, "protocolMajor": 3}

        def get_run(self, rid, **kwargs):
            assert rid == "run-1"
            self.reads += 1
            return _detail(seq=self.reads, attempt=f"a0{self.reads}")

        def close(self):
            self.closed = True

    gateway = Gateway()
    monkeypatch.setattr(claudexor, "ClaudexorGateway", lambda: gateway)
    with monkeypatch.context() as timing:
        timing.setattr(delegate.time, "monotonic", lambda: clock.now)
        timing.setattr(delegate.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds))
        result = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1, since_seq=0))
    assert result["status"] == "progress"
    assert gateway.reads == 2 and gateway.closed
    queued = [events.get_nowait() for _ in range(events.qsize())]
    frames = [row for row in queued if row["type"] == "send_message"]
    assert len(frames) == 2
    observations = [_observation(attempt_id=f"a0{seq}", revision=seq) for seq in (1, 2)]
    assert [frame["progress_meta"]["executor_observation"] for frame in frames] == observations

    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "logs" / "chat.jsonl").touch()
    live = []
    bridge = message_bus.LocalChatBridge()
    bridge._broadcast_fn = live.append
    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {"owner_id": 1})
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "publish_event", lambda *_: None)
    monkeypatch.setattr(events_chat_delivery, "_bound_project_chat_id", lambda *_: 0)
    task = {"id": "child", "_attempt": 0, "delegation_role": "subagent",
            "root_task_id": "root", "parent_task_id": "root", "model": "coordinator-model"}
    delivery = SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING={"child": {"task": task}},
        send_with_budget=message_bus.send_with_budget,
        append_jsonl=lambda *_: pytest.fail("delivery raised"),
    )
    for frame in frames:
        events_chat_delivery._handle_send_message(frame, delivery)
    stored = [json.loads(line) for line in (tmp_path / "logs" / "progress.jsonl").read_text().splitlines()]
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"limit": "10"})))
    replay = [row for row in json.loads(response.body)["messages"] if row.get("is_progress")]
    for rows in (live, stored, replay):
        assert [row["executor_observation"] for row in rows] == observations
        assert all(row["model"] == "coordinator-model" for row in rows)
        assert all("execution_evidence" not in row and "actual_substrate" not in row for row in rows)
    assert [row["ts"] for row in live] == [row["ts"] for row in replay]


def test_supervisor_drops_stale_attempt_metadata_without_dropping_the_note(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(events_chat_delivery, "_bound_project_chat_id", lambda *_: 0)
    delivery = SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING={"child": {"task": {"id": "child", "_attempt": 2}}},
        send_with_budget=lambda *args, **kwargs: sent.append((args, kwargs)),
        append_jsonl=lambda *_: pytest.fail("delivery raised"),
    )
    events_chat_delivery._handle_send_message({
        "chat_id": 1, "task_id": "child", "text": "late progress", "is_progress": True,
        "progress_meta": {"executor_observation": _observation(), "model": "coordinator"},
    }, delivery)
    assert sent[0][0] == (1, "late progress")
    assert "executor_observation" not in sent[0][1]["progress_meta"]


def test_history_does_not_move_a_stored_observation_to_another_task(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "chat.jsonl").touch()
    (logs / "progress.jsonl").write_text(json.dumps({
        "ts": "2026-09-09T00:00:00Z", "task_id": "parent", "content": "note",
        "executor_observation": _observation(),
    }) + "\n")
    response = asyncio.run(make_chat_history_endpoint(tmp_path)(SimpleNamespace(query_params={"limit": "10"})))
    row, = json.loads(response.body)["messages"]
    assert row["task_id"] == "parent" and row["text"] == "note"
    assert "executor_observation" not in row


def test_meta_stamps_the_frame_but_never_overrides_subagent_lineage():
    """`meta` merges into `progress_meta` verbatim (the reasoning stamp), yet the
    child's real lineage still wins over any lineage key smuggled through it, and a
    later plain note does not inherit the stamp."""
    agent, events = _agent(SimpleNamespace(task_attempt=0))
    OuroborosAgent._emit_progress(
        agent, "weigh the options",
        meta={"reasoning": True, "root_task_id": "forged", "delegation_role": "root"},
    )
    OuroborosAgent._emit_progress(agent, "plain note")
    stamped = events.get_nowait()["progress_meta"]
    assert stamped["reasoning"] is True
    assert stamped["root_task_id"] == "root" and stamped["delegation_role"] == "subagent"
    assert "reasoning" not in events.get_nowait()["progress_meta"]
