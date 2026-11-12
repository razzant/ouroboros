"""Q spans memory only: no file or store I/O while the queue lock is owned, with a shrinking named inventory.

The structural twin of tests/test_executor_inventory.py for the queue lock (ARCHITECTURE §10 invariant 24). An
audit hook (``sys.addaudithook``: ``open``, ``os.rename``, ``os.remove``, ``os.mkdir``, ``sqlite3.connect``) sees every
such call in the process, and the store's transaction seams (``read_document``, ``patch_document``, ...: SQLite I/O
after a thread's first connection raises no audit event) are wrapped as ``store.read``/``store.write``; while a test
here arms them, a call made by a thread that owns Q is recorded against the site that took Q (``MeasuredRLock._site``:
the file and function, never the line) and the event. The drives below
run the real admissions, the owner wait's park and grant, the Continue, the API task and the Presence turn, and
the tick, with the tick publisher armed as in a live server (marks are marks, barriers publish). What is still
I/O under Q is named in ``ALLOWED`` with the reason it is still there, in both directions: a site seen that no row
names fails by name (a new barrier or store write under Q), and a row no drive sees any more fails too (shrink
the inventory). Import I/O (the body's own modules) and procfs reads (kernel facts) are not the class and are not counted.
"""

from __future__ import annotations

import os
import pathlib
import sys
import threading
from typing import Dict, Set, Tuple

import pytest

from supervisor import queue, queue_publication
from supervisor.loop_load import code_place
from tests._main_followup import main as main
from tests.test_conversation_lane_presence import lane as lane  # noqa: F401 -- fixture
from tests.test_owner_wait_pool import pool as pool  # noqa: F401 -- fixture

Site = Tuple[str, str]  # (the Q acquisition's "path in function", the audited event)

# ---- the inventory: what still touches the disk or the store while Q is owned, and why ------------------------------

ALLOWED: Dict[Site, str] = {
    ("supervisor/continuation_admission.py in admit_continuation", "store.write"):
        "the successor's receipt (write_task_result) is written under the second hold with the row revalidated still "
        "queued, so a replay under Q never meets a row without it and a Stop's record is never written over",
    ("ouroboros/gateway/tasks.py in _enqueue_api_task_durably", "store.read"):
        "the API admission's receipt merges over the record as the window left it (an assignment's running, a Stop's "
        "terminal): the read of that record and the readback of the exact receipt, under the second hold",
    ("ouroboros/gateway/tasks.py in _enqueue_api_task_durably", "store.write"):
        "the API admission's receipt itself, under the same hold (the launch and the Stop write under Q: one state)",
    ("supervisor/queue.py in enqueue_task", "open"):
        "the admission's pending-placement read (projects_registry._load_bindings) inside enqueue_task's own hold: "
        "one transaction with the placement it decides; a residual named in the journal, not a barrier",
    ("ouroboros/gateway/tasks.py in _enqueue_api_task_durably", "open"):
        "the same pending-placement read, under the API admission's transaction (its hold covers the enqueue)",
    ("supervisor/continuation_admission.py in admit_continuation", "open"):
        "the same pending-placement read, under the Continue's admission transaction",
    ("supervisor/continuation_admission.py in _replay", "store.read"):
        "the recorded admission of the same nonce is read and decided under Q (the successor's row and its record are "
        "one state for the replay); its barrier runs outside Q; a residual named in the journal",
    ("supervisor/task_admission.py in reserve_task_admission", "store.read"):
        "the reservation reads the task's stored record under its own hold (task_admission.py: not this item's file; "
        "the hunk that moves the read before the hold is in the journal)",
}

_EVENTS = frozenset({"open", "os.rename", "os.remove", "os.mkdir", "sqlite3.connect"})
#: The store's transaction seams: SQLite reads and writes after a thread's first connection raise no audit event, so
#: a read or write transaction entered while Q is owned is recorded as its own event (``store.read``/``store.write``).
_STORE_SEAMS = (("ouroboros.task_results", "read_document", "store.read"),
                ("ouroboros.task_results", "list_documents", "store.read"),
                ("ouroboros.task_store", "documents_by_ids", "store.read"),
                ("ouroboros.task_results", "patch_document", "store.write"))
_CODE_SUFFIXES = (".py", ".pyc", ".pyi", ".so", ".dylib", ".pth", ".zip")
_INTERPRETER = tuple(pathlib.Path(p).resolve() for p in {sys.prefix, sys.base_prefix, sys.exec_prefix})
_STATE = {"armed": False, "seen": set(), "samples": {}, "hook": False}
_LOCK = threading.Lock()


def _is_code(path: str) -> bool:
    """A module's own import, or a procfs read (a kernel fact, memory-backed, never the disk): not the class."""
    text = str(path)
    if text.endswith(_CODE_SUFFIXES) or text.startswith("/proc/"):
        return True
    try:
        resolved = pathlib.Path(text).resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    return any(prefix == resolved or prefix in resolved.parents for prefix in _INTERPRETER)


def _record(event: str, sample: str) -> None:
    """Under Q (the caller checked): the event against the function that took Q; the first sighting keeps its path
    and its stack, so a drift names what opened what from where."""
    import traceback

    site = queue._queue_lock._site
    key = (code_place((site[0], site[2]) if site else None), event)
    with _LOCK:
        if key not in _STATE["seen"]:
            _STATE["seen"].add(key)
            _STATE["samples"][key] = (sample, "".join(traceback.format_stack()[-12:-2]))


def _hook(event: str, args: tuple) -> None:
    if not _STATE["armed"] or event not in _EVENTS:
        return
    if not queue._queue_lock._is_owned():
        return
    if event == "open" and args and _is_code(os.fsdecode(args[0]) if not isinstance(args[0], int) else ""):
        return
    _record(event, str(args[0]) if args else "")


@pytest.fixture
def store_seams(monkeypatch):
    """The store's read and write transactions entered under Q, recorded as events the audit hook cannot see."""
    import importlib

    for module_name, attr, event in _STORE_SEAMS:
        module = importlib.import_module(module_name)
        real = getattr(module, attr)

        def wrapped(*args, _real=real, _event=event, **kwargs):
            if _STATE["armed"] and queue._queue_lock._is_owned():
                _record(_event, f"{_real.__module__}.{_real.__name__}")
            return _real(*args, **kwargs)

        monkeypatch.setattr(module, attr, wrapped)


@pytest.fixture
def audited():
    """Arm the (process-lifetime) audit hook for one test; its findings, keyed as the inventory is."""
    if not _STATE["hook"]:
        sys.addaudithook(_hook)
        _STATE["hook"] = True
    _STATE["seen"] = set()
    _STATE["samples"] = {}
    _STATE["armed"] = True
    try:
        yield _STATE["seen"]
    finally:
        _STATE["armed"] = False


def _tick_armed():
    """The live server's shape: marks answer None (the tick publishes them), barriers publish at once."""
    queue_publication.tick_publisher_started()


# ---- the drives ------------------------------------------------------------------------------------------------------

def _drive_owner_wait(pool) -> None:
    from supervisor import worker_owner_wait, workers

    worker_owner_wait.handle_owner_wait(pool.event, workers)
    assert pool.original.in_q.get_nowait()["phase"] == "parked"
    worker_owner_wait.handle_owner_wait({**pool.event, "phase": "resume"}, workers)
    worker_owner_wait.grant_warm_returns()
    assert pool.original.in_q.get_nowait()["phase"] == "resume_granted"


def _drive_owner_turn(lane) -> None:
    from supervisor import worker_chat_lane, workers

    worker_chat_lane.handle_chat_direct(1, "hello", None, task_metadata={"client_message_id": "cm-audit"})
    meta = next(m for m in workers.RUNNING.values() if m.get("lane") == "conversation")
    assert meta["worker_id"] == lane.spare.wid, "the carrier took the turn"


def _drive_presence(lane, monkeypatch) -> None:
    from ouroboros import presence_admission, presence_runner
    from ouroboros.presence_carrier import dispatch_presence_turn
    from ouroboros.presence_runner import PresenceTurnExecutions
    from supervisor import workers
    from tests.test_conversation_lane_presence import _payload
    from tests.test_presence_runner import _admission, _event

    admission, event = _admission(), _event()
    monkeypatch.setattr(presence_admission, "admit_presence_turn", lambda **kw: admission)
    turn_id = presence_runner.presence_turn_task_id(admission.binding_id, event.source_event_id)
    registry = PresenceTurnExecutions()
    _execution, started = registry.start_or_dispatch(
        turn_id, identity="id-audit", reserve=lambda: True, release=lambda: None,
        dispatch=lambda execution: dispatch_presence_turn(
            execution, drive_root=lane.root, skill_name="telegram-bot", admission=admission, event=event,
            event_payload=_payload(event), staged_files=[], gate_wait=True))
    assert started and turn_id in workers.RUNNING


def _drive_continue(tmp_path, monkeypatch) -> None:
    from supervisor.continuation_admission import admit_continuation
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_owner_continue import NONCE, _interrupted

    _install_queue(tmp_path, monkeypatch)
    _tick_armed()
    _interrupted(tmp_path)
    answer = admit_continuation("pred-1", action_nonce=NONCE)
    assert answer["ok"] is True, answer


def _own_queue(data, monkeypatch) -> None:
    """A fresh live queue on ``data`` for one drive, its store created by one write outside Q: a live store exists
    before any admission, so the one-time costs of a fresh root (the kernel-lock probe, the import rename, this
    thread's first connection) are not the drive's."""
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result
    from supervisor import workers

    (data / "state").mkdir(parents=True)
    for module in (queue, workers):
        monkeypatch.setattr(module, "DRIVE_ROOT", data)
        monkeypatch.setattr(module, "PENDING", [])
        monkeypatch.setattr(module, "RUNNING", {})
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", data / "state" / "queue_snapshot.json")
    monkeypatch.setattr(queue, "ADMISSION_RESERVATIONS", {})
    queue_publication.reset()
    _tick_armed()
    assert write_task_result(data, "warm", STATUS_SCHEDULED, description="warm")["status"] == STATUS_SCHEDULED


def _drive_restart_pending_admission(tmp_path, monkeypatch) -> None:
    """An admission while a restart is pending (``enqueue_task``'s own ``durable`` barrier, t2 §1.2 queue.py:519)."""
    from ouroboros import server_restart

    data = tmp_path / "data"
    _own_queue(data, monkeypatch)
    monkeypatch.setattr(server_restart, "agent_restart_pending", lambda: True)
    admitted = queue.enqueue_task({"id": "restart-audit", "type": "task", "chat_id": 0, "text": "audit", "metadata": {},
                                   "description": "audit"})
    assert not admitted.get("_admission_blocked"), admitted
    assert (data / "state" / "queue_snapshot.json").exists(), "durable before the reply"


def _drive_api_task(tmp_path, monkeypatch) -> None:
    from ouroboros.gateway.tasks import _enqueue_api_task_durably

    data = tmp_path / "data"
    _own_queue(data, monkeypatch)
    assert queue.reserve_task_admission("api-audit", "token-1", drive_root=data)["status"] == "reserved"
    task = {"id": "api-audit", "type": "task", "chat_id": 0, "text": "audit", "_admission_token": "token-1",
            "metadata": {}, "description": "audit"}
    admitted = _enqueue_api_task_durably(task, drive_root=data, task_id="api-audit", admission_token="token-1",
                                         result_fields={"description": "audit", "chat_id": 0, "metadata": {},
                                                        "api_admission": {"token": "token-1", "status": "accepted"},
                                                        "result": "accepted"})
    assert not admitted.get("_admission_blocked") and not admitted.get("_admission_uncertain"), admitted


def _drive_tick() -> None:
    assert queue_publication.publish_queue_changes() in (True, None)


# ---- the laws --------------------------------------------------------------------------------------------------------

def _unnamed_or_gone(seen: Set[Site]) -> Dict[str, list]:
    return {key: rows for key, rows in (("unnamed", sorted(seen - set(ALLOWED))), ("gone", sorted(set(ALLOWED) - seen)))
            if rows}


def test_no_io_under_q_beyond_the_named_inventory(lane, tmp_path, monkeypatch, audited, store_seams):
    """Every drive, one process: what touched the disk or the store while Q was owned is exactly the inventory.
    The controller identity is read once per process (``review_operation.controller_identity``: a ``ps`` on macOS,
    procfs on Linux) and here before the drives, as a live server has read it long before its first admission."""
    from ouroboros.review_operation import controller_identity

    controller_identity()
    _tick_armed()
    _drive_owner_wait(lane)
    _drive_owner_turn(lane)
    _drive_presence(lane, monkeypatch)
    _drive_tick()
    with monkeypatch.context() as scoped:
        _drive_continue(tmp_path / "continue", scoped)
    with monkeypatch.context() as scoped:
        _drive_api_task(tmp_path / "api", scoped)
    with monkeypatch.context() as scoped:
        _drive_restart_pending_admission(tmp_path / "restart", scoped)
    drift = _unnamed_or_gone(set(audited))
    for key in drift.get("unnamed", []):
        sample, stack = _STATE["samples"].get(key, ("", ""))
        print(f"I/O under Q: {key} {sample}\n{stack}")  # the failure names the path and the stack too
    assert not drift, drift


def test_a_main_follow_up_reaches_its_owner_turn_with_no_io_under_q(main, audited, store_seams):
    """Main's follow-up through the real intake: Q spans the row judged and its fences only; the mailbox reads, the
    mail lock's wait and the append run outside it, so the drive adds no site to the inventory."""
    from tests._main_followup import _delivered, _say, _turns

    _tick_armed()
    _say(main.bridge, main.ctx, "Send it to our chat", "cm-audit-first")
    (turn,) = _turns()
    _STATE["seen"].clear()  # the first turn's admission is _drive_owner_turn's; this drive is the follow-up
    _say(main.bridge, main.ctx, "This chat", "cm-audit-second")
    assert [ack["target"] for ack in _delivered(main.bridge, "cm-audit-second")] == [turn["task"]["id"]]
    unnamed = sorted(set(audited) - set(ALLOWED))
    for key in unnamed:
        sample, stack = _STATE["samples"].get(key, ("", ""))
        print(f"I/O under Q: {key} {sample}\n{stack}")
    assert not unnamed, unnamed


def test_the_hook_sees_io_under_q_and_names_the_site(audited):
    """Both directions of the guard itself: a write under Q is recorded against the function that took Q; the same
    write outside Q is not; a module's own import is not the class."""
    target = pathlib.Path(os.environ["OUROBOROS_DATA_DIR"]) / "audit-probe.txt"
    target.parent.mkdir(parents=True, exist_ok=True)

    def writes_under_q():
        with queue._queue_lock:
            target.write_text("under q", encoding="utf-8")
            os.rename(target, target.with_suffix(".moved"))
            import json  # noqa: F401 -- an import under Q is not the class
    writes_under_q()
    target.with_suffix(".moved").write_text("outside q", encoding="utf-8")
    seen = set(audited)
    assert ("tests/test_queue_lock_io_inventory.py in writes_under_q", "open") in seen
    assert ("tests/test_queue_lock_io_inventory.py in writes_under_q", "os.rename") in seen
    assert not any(site.startswith("tests/test_queue_lock_io_inventory.py in test_") for site, _event in seen)
    assert _unnamed_or_gone({("x in y", "open")}) == {"unnamed": [("x in y", "open")], "gone": sorted(ALLOWED)}


def test_a_store_transaction_under_q_is_seen_through_its_seam(audited, store_seams):
    """SQLite I/O raises no audit event once a thread is connected: the seam records the transaction instead."""
    from ouroboros.task_results import STATUS_SCHEDULED, load_task_result, write_task_result

    root = pathlib.Path(os.environ["OUROBOROS_DATA_DIR"]) / "seam-probe"
    write_task_result(root, "outside", STATUS_SCHEDULED, description="outside q")  # the store exists, this thread connected

    def writes_under_q():
        with queue._queue_lock:
            write_task_result(root, "under", STATUS_SCHEDULED, description="under q")
            assert load_task_result(root, "under")["description"] == "under q"
    writes_under_q()
    seen = set(audited)
    assert ("tests/test_queue_lock_io_inventory.py in writes_under_q", "store.write") in seen
    assert ("tests/test_queue_lock_io_inventory.py in writes_under_q", "store.read") in seen
    assert not any(site.startswith("tests/test_queue_lock_io_inventory.py in test_") for site, _event in seen)


def test_every_row_names_a_reason():
    events = _EVENTS | {event for _module, _attr, event in _STORE_SEAMS}
    for site, why in ALLOWED.items():
        assert site[1] in events and " in " in site[0] and why, site
