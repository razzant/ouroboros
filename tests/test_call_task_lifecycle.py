"""An admitted call keeps ordinary task custody before and after its live run."""
import json
import queue
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture
def direct_call(monkeypatch, tmp_path):
    from ouroboros import call_protocol as cp
    from ouroboros.call_peers import close_hub, open_hub
    from ouroboros.live_conversation import LiveConversationTurn
    from supervisor import worker_chat_lane, workers

    turn = object.__new__(LiveConversationTurn)  # no provider/configuration reads
    turn.spec = cp.CallSpec("synthetic-call", 1, "synthetic", "synthetic")
    turn.model = "synthetic::voice"
    turn.request = {"kind": "live_conversation", "call_id": turn.spec.call_id}
    turn._lock, turn._closed = threading.Lock(), False
    hub = open_hub(turn.spec)
    hub.join(cp.PeerInfo("owner", cp.PEER_OWNER, True, True), 24000)
    other = open_hub(cp.CallSpec("unrelated-call", 2, "synthetic", "synthetic"))
    notices, logs = [], []
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "repo_writer_admission_closed", lambda: "")
    monkeypatch.setattr(workers, "get_event_q", queue.Queue)
    monkeypatch.setattr(workers, "send_with_budget", lambda *args, **kwargs: notices.append((args, kwargs)))
    monkeypatch.setattr(worker_chat_lane, "append_jsonl", lambda path, row: logs.append(row))
    try:
        yield SimpleNamespace(turn=turn, hub=hub, other=other, notices=notices, logs=logs)
    finally:
        close_hub(turn.spec.call_id, "fixture_cleanup")
        close_hub(other.spec.call_id, "fixture_cleanup")


@pytest.mark.parametrize("case", ["store_busy", "unexpected", "agent_creation", "already_closed", "text", "success"])
def test_pre_run_failure_closes_only_its_admitted_call(direct_call, monkeypatch, case):
    from ouroboros.call_peers import close_hub, get_hub
    from ouroboros.store import StoreBusy
    from supervisor import worker_chat_lane, workers

    world = direct_call
    def run(task=None):
        if case == "success":
            return []
        if case in {"store_busy", "agent_creation"}:
            raise StoreBusy("synthetic billing write contention")
        raise RuntimeError("synthetic pre-run failure")
    agent = SimpleNamespace(handle_task=run)
    if case == "agent_creation":
        monkeypatch.setattr(workers, "_get_chat_agent", run)
        agent = None
    if case == "already_closed":
        close_hub(world.turn.spec.call_id, "finished", detail="The live turn already ended.")
    # Metadata alone never authorizes this error path to close another live call.
    admitted = worker_chat_lane._admit_chat_task(agent, 1, "Opening", task_metadata={
        "execution_transport": world.turn.request}, realtime_turn=None if case == "text" else world.turn)
    if case == "agent_creation":
        assert admitted is None
    else:
        assert admitted is not None
        assert worker_chat_lane._execute_chat_task(admitted) is (case == "success")
    assert not world.other.closed
    if case in {"text", "success"}:
        assert not world.hub.closed and get_hub(world.turn.spec.call_id) is world.hub
    else:
        assert world.hub.closed and get_hub(world.turn.spec.call_id) is None
        if admitted is not None:
            worker_chat_lane._report_direct_chat_error(admitted, RuntimeError("repeated report"))
        frames = []
        while (frame := world.hub.next_outgoing("owner")) is not None:
            if isinstance(frame, str):
                frames.append(json.loads(frame))
        [closed] = [frame for frame in frames if frame.get("state") == "closed"]
        expected = ("finished" if case == "already_closed" else
                    "store_busy" if case in {"store_busy", "agent_creation"} else "call_start_failed")
        assert closed["reason"] == expected and closed["detail"]
        assert "synthetic" not in closed["detail"]
    assert bool(world.logs) is (case != "success")


@pytest.mark.parametrize('failure', ['', 'project_source_unknown', 'store_busy'])
def test_pre_live_failure_closes_before_busy_result_recovery(direct_call, monkeypatch, failure):
    from ouroboros import budget_pause
    from ouroboros.call_peers import get_hub
    from ouroboros.store import StoreBusy
    from supervisor import direct_pause, worker_chat_lane

    world, retained, recovery_closed = direct_call, [], []
    def run(task):
        if failure == 'store_busy':
            raise StoreBusy('synthetic store contention')
        if failure:
            raise ValueError(failure)
        return []
    admitted = worker_chat_lane._admit_chat_task(SimpleNamespace(handle_task=run), 1, 'Opening',
        task_metadata={'execution_transport': world.turn.request}, realtime_turn=world.turn)
    assert admitted is not None
    def busy(*args):
        recovery_closed.append(world.hub.closed)
        raise StoreBusy('synthetic result store contention')
    monkeypatch.setattr(budget_pause, 'budget_pause_row', busy)
    monkeypatch.setattr(direct_pause, 'retain_direct_hold', lambda task, **kw: retained.append((task, kw)))
    assert worker_chat_lane._execute_chat_task(admitted)
    if failure:
        assert recovery_closed == [True], 'call closure waited for result recovery'
        assert retained == [(admitted['task'], {'unsaved': False})]
        assert world.hub.closed and get_hub(world.turn.spec.call_id) is None
        assert world.hub._state['reason'] == ('store_busy' if failure == 'store_busy' else 'call_start_failed')
        assert world.hub._state['detail']
    else:
        assert not recovery_closed and not retained
        assert not world.hub.closed and get_hub(world.turn.spec.call_id) is world.hub
    assert not world.other.closed
