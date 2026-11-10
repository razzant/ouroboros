"""An admitted call keeps ordinary task custody before and after its live run.

A live call is the one owner turn that keeps a thread of the server process (``executor=inline_call``,
owner decision 3A): its admission writes the RUNNING lane row and its body (the same worker task body)
runs on this thread; a text turn is admitted to the lane and carried by a worker."""
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
    from supervisor import message_bus, worker_chat_lane, workers
    from tests.conversation_carrier_support import inline_synchronously, isolate_lane_pool

    turn = object.__new__(LiveConversationTurn)  # no provider/configuration reads
    turn.spec = cp.CallSpec("synthetic-call", 1, "synthetic", "synthetic")
    turn.model = "synthetic::voice"
    turn.request = {"kind": "live_conversation", "call_id": turn.spec.call_id}
    turn._lock, turn._closed = threading.Lock(), False
    hub = open_hub(turn.spec)
    hub.join(cp.PeerInfo("owner", cp.PEER_OWNER, True, True), 24000)
    other = open_hub(cp.CallSpec("unrelated-call", 2, "synthetic", "synthetic"))
    notices, logs = [], []
    bus = queue.Queue()
    [spare] = isolate_lane_pool(monkeypatch, tmp_path, event_q=bus)  # one lane carrier for a text turn
    inline_synchronously(monkeypatch)  # a live call's body runs on this thread
    monkeypatch.setattr(workers, "repo_writer_admission_closed", lambda: "")
    monkeypatch.setattr(workers, "send_with_budget", lambda *args, **kwargs: notices.append((args, kwargs)))
    monkeypatch.setattr(worker_chat_lane, "append_jsonl", lambda path, row: logs.append(row))
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(send_chat_action=lambda *a, **k: None))
    try:
        yield SimpleNamespace(turn=turn, hub=hub, other=other, notices=notices, logs=logs, bus=bus, spare=spare,
                              root=tmp_path)
    finally:
        close_hub(turn.spec.call_id, "fixture_cleanup")
        close_hub(other.spec.call_id, "fixture_cleanup")


def _agent(root, body):
    """A stand-in actor with the env the carrier's exception terminal writes through."""
    return SimpleNamespace(handle_task=body, env=SimpleNamespace(drive_root=root, drive_path=lambda name: root / name))


@pytest.mark.parametrize("case", ["store_busy", "agent_creation", "success"])
def test_pre_run_failure_closes_only_its_admitted_call(direct_call, monkeypatch, case):
    from ouroboros.call_peers import close_hub, get_hub
    from ouroboros.store import StoreBusy
    from ouroboros.task_results import load_task_result
    from supervisor import worker_chat_lane, workers
    from tests.conversation_carrier_support import carry, lane_rows

    world, seen = direct_call, []

    def run(task=None):
        if case == "agent_creation":
            raise StoreBusy("synthetic billing write contention")
        seen.append(task["id"])
        if case == "success":
            return []
        if case == "store_busy":
            raise StoreBusy("synthetic billing write contention")
        raise RuntimeError("synthetic pre-run failure")

    agent = _agent(world.root, run)
    monkeypatch.setattr(workers, "_get_chat_agent", run if case == "agent_creation" else (lambda: agent))
    if case == "already_closed":
        close_hub(world.turn.spec.call_id, "finished", detail="The live turn already ended.")
    # Metadata alone never authorizes this error path to close another live call.
    worker_chat_lane.handle_chat_direct(1, "Opening", task_metadata={"execution_transport": world.turn.request},
                                        realtime_turn=None if case == "text" else world.turn)
    if case == "text":
        # No live turn: an ordinary lane admission; its carrier's failure is the turn's own terminal.
        with pytest.raises(RuntimeError, match="synthetic pre-run failure"):
            carry(workers, world.spare, agent)
    rows = lane_rows(workers)
    if case == "agent_creation":
        # The actor could not be constructed: the call is refused, and nothing of it is left to run.
        assert seen == [] and rows == {}, rows
    else:
        assert seen == list(rows)
    assert not world.other.closed
    recorded = list(world.logs)  # before the repeated report below adds its own row
    if case in {"text", "success"}:
        assert not world.hub.closed and get_hub(world.turn.spec.call_id) is world.hub
    else:
        assert world.hub.closed and get_hub(world.turn.spec.call_id) is None
        admitted = {"task": {"id": seen[0] if seen else "refused"}, "chat_id": 1, "client_msg_id": "",
                    "task_metadata": None, "realtime_turn": world.turn, "agent": None}
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
    # A failure is recorded exactly once: the refused admission's error row, or the carrier's own terminal.
    if case == "agent_creation":
        assert [row["type"] for row in recorded] == ["direct_chat_error"]
    elif case == "success":
        assert not recorded and (load_task_result(world.root, seen[0]) or {}).get("status") != "failed"
    else:
        assert not recorded and load_task_result(world.root, seen[0])["status"] == "failed"


@pytest.mark.parametrize('failure', ['', 'project_source_unknown', 'store_busy'])
def test_pre_live_failure_closes_before_busy_result_recovery(direct_call, monkeypatch, failure):
    from ouroboros import budget_pause
    from ouroboros.call_peers import get_hub
    from ouroboros.store import StoreBusy
    from ouroboros.task_results import load_task_result
    from supervisor import worker_chat_lane, workers
    from tests.conversation_carrier_support import lane_rows

    world, recovery_closed = direct_call, []

    def run(task):
        if failure == 'store_busy':
            raise StoreBusy('synthetic store contention')
        if failure:
            raise ValueError(failure)
        return []

    monkeypatch.setattr(workers, "_get_chat_agent", lambda: _agent(world.root, run))

    def busy(*args, **kwargs):
        recovery_closed.append(world.hub.closed)
        raise StoreBusy('synthetic result store contention')

    monkeypatch.setattr(budget_pause, 'budget_pause_row', busy)
    worker_chat_lane.handle_chat_direct(1, 'Opening', task_metadata={'execution_transport': world.turn.request},
                                        realtime_turn=world.turn)
    [task_id] = list(lane_rows(workers))
    if failure:
        assert recovery_closed == [True], 'call closure waited for result recovery'
        # Unreadable pause evidence is never a fabricated terminal: the custody that decides keeps the row.
        assert (load_task_result(world.root, task_id) or {}).get("status") != "failed"
        assert task_id in workers.RUNNING
        assert world.hub.closed and get_hub(world.turn.spec.call_id) is None
        assert world.hub._state['reason'] == ('store_busy' if failure == 'store_busy' else 'call_start_failed')
        assert world.hub._state['detail']
    else:
        assert not recovery_closed
        assert not world.hub.closed and get_hub(world.turn.spec.call_id) is world.hub
    assert not world.other.closed
