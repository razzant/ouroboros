"""Acceptance laws of the conversation lane that no single module owns (E1, E6, E9).

E1 — placement: no server-side module starts cognition (``handle_task`` / ``make_agent``) outside
the two named carriers of the worker body; the admission doors construct no agent.
E6 — the owner's wait is flat: an owner turn's admission does the same work with 0 and 5 other
lane turns live (counted, not timed).
E9 — Panic kills a lane carrier by measured identity and its row is settled; a project fence does
not; a lane turn handed off at a planned restart returns as the same id's next attempt with its
saved source, never a replay; a lane row's Stop is the pooled custody (no inline arming).
"""

import pathlib
import re
import time
from types import SimpleNamespace

import pytest

from ouroboros import store_timing
from ouroboros.task_carrier import LANE_CONVERSATION
from supervisor import conversation_lane, worker_chat_lane, workers
from tests.test_owner_wait_pool import InputQueue, Process, pool as pool  # noqa: F401 -- fixture
from tests.worker_pool_support import affirm_ready


REPO = pathlib.Path(__file__).resolve().parents[1]
# The processes that may START cognition: the worker body and its inline twin, the Presence body
# and its carrier factory, and the fresh native actor the inline executors borrow.
COGNITION_STARTERS = {
    "supervisor/worker_process.py", "supervisor/inline_carrier.py",
    "ouroboros/presence_runner.py", "ouroboros/presence_carrier.py",
    "supervisor/workers.py",  # _get_chat_agent: the inline executors' actor, called only from inline_carrier
}


def _server_side_modules():
    roots = [REPO / "supervisor", REPO / "ouroboros" / "gateway"]
    files = [path for root in roots for path in sorted(root.glob("*.py"))]
    files += sorted((REPO / "ouroboros").glob("server*.py")) + sorted((REPO / "ouroboros").glob("consciousness*.py"))
    files.append(REPO / "server.py")
    return files


def test_e1_no_server_side_module_starts_cognition_outside_the_named_carriers():
    offenders = []
    pattern = re.compile(r"(\.handle_task\(|\bmake_agent\()")
    for path in _server_side_modules():
        rel = path.relative_to(REPO).as_posix()  # the carriers are spelled with "/" on every OS
        if rel in COGNITION_STARTERS:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or "def make_agent" in line or "def handle_task" in line:
                continue
            if pattern.search(line) and "import" not in line:
                offenders.append(f"{rel}:{number}: {stripped[:100]}")
    assert offenders == [], "cognition started in a server-side module:\n" + "\n".join(offenders)
    # The inline executors' actor is borrowed only by the inline carrier.
    callers = [p.relative_to(REPO).as_posix() for p in _server_side_modules()
               if "_get_chat_agent()" in p.read_text(encoding="utf-8")
               and p.relative_to(REPO).as_posix() != "supervisor/workers.py"]
    assert callers == ["supervisor/inline_carrier.py"], callers


@pytest.fixture
def lane(pool, monkeypatch):
    from supervisor import message_bus

    monkeypatch.setattr(message_bus, "_BRIDGE", SimpleNamespace(send_chat_action=lambda *a, **k: None,
                                                                 broadcast=lambda m: None, push_log=lambda *a, **k: None))
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(workers, "_get_chat_agent", lambda: pytest.fail("the server constructed an agent"))
    monkeypatch.setattr("ouroboros.agent.make_agent", lambda **k: pytest.fail("the server constructed an agent"))
    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", id(workers.WORKERS))
    return pool


def _spare(wid):
    slot = workers.Worker(wid, Process(6000 + wid), InputQueue(), lane=LANE_CONVERSATION)
    affirm_ready(slot)
    workers.WORKERS[wid] = slot
    return slot


def _admit(lane_spare, text):
    with store_timing.metered() as meter:
        worker_chat_lane.handle_chat_direct(1, text, None, task_metadata={"client_message_id": f"cm-{text}"})
    snapshot = store_timing.snapshot(meter)
    return snapshot


def test_e6_admission_work_is_equal_with_zero_and_five_other_lane_turns_live(lane):
    spare = _spare(10)
    alone = _admit(spare, "alone")
    spare.busy_task_id = None
    spare.in_q.get_nowait()
    for i in range(5):  # five other owner turns live on their carriers
        other = _spare(20 + i)
        other.busy_task_id = f"turn-{i}"
        workers.RUNNING[f"turn-{i}"] = {"task": {"id": f"turn-{i}", "chat_id": 1, "_is_direct_chat": True},
                                        "worker_id": other.wid, "lane": LANE_CONVERSATION, "attempt": 1,
                                        "started_at": time.time(), "last_heartbeat_at": time.time()}
    crowded = _admit(spare, "crowded")
    counted = {key: value for key, value in alone.items() if isinstance(value, (int, float)) and key not in {"wall_sec", "ms"}}
    assert counted == {key: value for key, value in crowded.items() if key in counted}, (alone, crowded)
    assert len(workers.PENDING) == 0


def test_e9_panic_kills_a_lane_carrier_by_identity_and_settles_its_row(lane, monkeypatch):
    monkeypatch.setattr(workers, "_reconcile_confirmed_dead_review_owner", lambda _pid: None)
    monkeypatch.setattr(workers, "_audit_delegate_terminal_custody", lambda *a, **k: None)
    monkeypatch.setattr(workers, "reconstruct_task_cost", lambda *a, **k: {"cost_accounting_status": "unavailable",
                                                                             "cost_final": False})
    monkeypatch.setattr(workers, "append_jsonl", lambda *a, **k: True)
    killed = []
    from supervisor import worker_pool_lifecycle as lifecycle

    def kill(pid, **kwargs):
        killed.append(pid)
        for worker in workers.WORKERS.values():
            if worker.proc.pid == pid:
                worker.proc.alive = False
                worker.proc.exitcode = -9

    monkeypatch.setattr(lifecycle, "kill_worker_tree", kill)
    monkeypatch.setattr(workers, "kill_worker_tree", kill)
    spare = _spare(10)
    worker_chat_lane.handle_chat_direct(1, "hello", None, task_metadata={"client_message_id": "cm-p"})
    turn = spare.busy_task_id
    assert turn and spare.proc.pid in {w.proc.pid for w in workers.WORKERS.values()}
    assert workers.kill_workers(stop_source="panic")  # Panic: every slot, by measured identity
    assert spare.proc.pid in killed and 42 in killed  # the lane carrier AND the project worker
    assert turn not in workers.RUNNING and not workers.WORKERS
    from ouroboros.task_results import load_task_result
    assert load_task_result(lane.root, turn)["status"] in {"failed", "cancelled", "interrupted"}


def test_e9_a_lane_turn_survives_a_planned_restart_as_the_same_ids_next_attempt(lane, monkeypatch):
    monkeypatch.setattr(workers, "_reconcile_confirmed_dead_review_owner", lambda _pid: None)
    monkeypatch.setattr(workers, "append_jsonl", lambda *a, **k: True)
    from supervisor import worker_pool_lifecycle as lifecycle

    def kill(pid, **kwargs):
        for worker in workers.WORKERS.values():
            if worker.proc.pid == pid:
                worker.proc.alive = False
                worker.proc.exitcode = 0

    monkeypatch.setattr(lifecycle, "kill_worker_tree", kill)
    monkeypatch.setattr(workers, "kill_worker_tree", kill)
    spare = _spare(10)
    worker_chat_lane.handle_chat_direct(1, "hello", None, task_metadata={"client_message_id": "cm-r"})
    turn = spare.busy_task_id
    workers.RUNNING.pop("owner")  # the pooled fixture row is not this law's subject
    assert workers.kill_workers(preserve_pending=True, preserve_running_task_ids={turn},
                                terminal_status="interrupted", retain_saved_work=True, stop_source="owner_restart")
    successors = [row for row in workers.PENDING if row.get("id") == turn]
    assert len(successors) == 1, workers.PENDING
    successor = successors[0]
    assert successor["_attempt"] == 2 and successor["_is_direct_chat"]
    assert turn not in workers.RUNNING


def test_e9_a_lane_rows_stop_is_the_pooled_custody_not_the_inline_arming(lane, monkeypatch):
    armed = []
    monkeypatch.setattr(workers, "arm_direct_chat_turn", lambda *a, **k: armed.append(a) or None)
    spare = _spare(10)
    worker_chat_lane.handle_chat_direct(1, "hello", None, task_metadata={"client_message_id": "cm-s"})
    turn = spare.busy_task_id
    assert workers.direct_chat_turn(turn) is None  # no in-process actor owns a lane row
    from supervisor import cancel_publication, queue
    assert cancel_publication.finish_in_process_turn(queue, turn, None, {}, intent=None) is None
    assert armed == []
