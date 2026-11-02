"""A shutdown door records why its generation began stopping before any wait (ARCHITECTURE §9).

The first explicit door cause replaces ``state/shutdown_entry.json`` with one lock-free atomic
write, after the grace deadline starts and before the ledger stamp; a cause-less or later door
keeps it, and a failed write never stops the door. The next start reports the previous
generation's record and claims the file, so an older cause never explains a later death before
any door. The record is an entry, never an exit or Resume authority; the late ``server_shutdown``
row still follows terminal custody.
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import pathlib
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def data(tmp_path, monkeypatch):
    """This installation's data root, as the server configures it."""
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    return root


def _generation(monkeypatch, name):
    """A new server generation: its own process generation and its own one stop."""
    from ouroboros import delegate_recovery, owned_shutdown

    monkeypatch.setattr(delegate_recovery, "_PROCESS_GENERATION", name)
    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())


def _entry(root):
    from ouroboros import owned_shutdown

    return json.loads(owned_shutdown.shutdown_entry_path(root).read_text(encoding="utf-8"))


def _refuse_entry_writes(monkeypatch):
    from ouroboros import owned_shutdown

    real, attempts = owned_shutdown.atomic_write_json, []

    def write(path, payload, **kwargs):
        if pathlib.Path(path).name == owned_shutdown.SHUTDOWN_ENTRY_FILENAME:
            attempts.append(dict(payload))
            raise OSError(28, "No space left on device")
        return real(path, payload, **kwargs)

    monkeypatch.setattr(owned_shutdown, "atomic_write_json", write)
    return attempts


def test_the_first_explicit_cause_is_recorded_once_and_later_doors_keep_it(data, monkeypatch):
    from ouroboros import owned_shutdown

    _generation(monkeypatch, "gen-a")
    assert owned_shutdown.claim_shutdown_entry(data) == {"state": "absent"}
    owned_shutdown.begin_owned_stop(data)  # a cause-less first door starts the grace...
    deadline = owned_shutdown._GENERATION_STOP.deadline
    assert deadline is not None and "cause" not in _entry(data)
    owned_shutdown.begin_owned_stop(data, cause="owner_restart")  # ...and never suppresses the first cause
    first = owned_shutdown.shutdown_entry_path(data).read_bytes()
    entry = json.loads(first)
    assert entry["generation"] == "gen-a" and entry["supervisor_pid"] == os.getpid() and entry["schema_version"] == 1
    assert entry["cause"] == "owner_restart" and entry["restart_exit"] is True
    assert datetime.fromisoformat(entry["started_at"]) <= datetime.fromisoformat(entry["entered_at"])
    for later in ("restart_requested", "launcher_quit", ""):  # the teardown and emergency doors that follow
        owned_shutdown.begin_owned_stop(data, cause=later)
    assert owned_shutdown.shutdown_entry_path(data).read_bytes() == first
    assert owned_shutdown._GENERATION_STOP.deadline == deadline  # recording a cause never moves the grace


@pytest.mark.parametrize("cause,restart_exit", [("launcher_quit", False), ("launcher_gone", False),
                                                ("external_signal", False), ("restart_requested", True)])
def test_a_generic_door_names_its_cause_and_restart_exit(data, monkeypatch, cause, restart_exit):
    from ouroboros import owned_shutdown

    _generation(monkeypatch, "gen-a")
    owned_shutdown.begin_owned_stop(data, cause=cause)
    entry = _entry(data)
    assert (entry["cause"], entry["restart_exit"]) == (cause, restart_exit)
    assert "started_at" not in entry  # a door before the boot claim still records its cause


def test_the_server_cause_rule_is_shared_by_the_doors_and_the_late_row(monkeypatch):
    import server
    from ouroboros import server_process

    teardown = inspect.getsource(server.lifespan).split("\n    finally:\n", 1)[1]
    assert '"cause": server_stop_cause(restart_requested)' in teardown  # the late row: the custody-time flag
    assert "cause=server_stop_cause()" in teardown  # the door: the live flag at entry
    restart = server_process.threading.Event()
    monkeypatch.setattr(server_process, "_restart_requested", restart)
    monkeypatch.setattr(server_process, "_stop_source", "launcher_quit")
    assert server_process.server_stop_cause() == "launcher_quit"
    assert server_process.server_stop_cause(True) == "restart_requested"
    restart.set()
    assert server_process.server_stop_cause() == "restart_requested"
    assert server_process.server_stop_cause(False) == "launcher_quit"  # the flag the custody write used


def test_the_entry_precedes_the_ledger_stamp_and_takes_no_lock(data, monkeypatch):
    """Order inside the door: the deadline, the entry (no lock primitive at all), then the stamp
    that may wait for the ledger lock. A held supervisor.jsonl lock delays nothing here."""
    from ouroboros import owned_shutdown, platform_layer
    from ouroboros.utils import jsonl_append_lock_path

    _generation(monkeypatch, "gen-a")
    events = []
    real_write, real_stamp = owned_shutdown.atomic_write_json, owned_shutdown._stamp
    real_acquire, release = platform_layer.acquire_exclusive_file_lock, platform_layer.release_exclusive_file_lock

    def write(path, payload, **kwargs):
        if pathlib.Path(path).name == owned_shutdown.SHUTDOWN_ENTRY_FILENAME:
            events.append(("entry", owned_shutdown._GENERATION_STOP.deadline is not None))
        return real_write(path, payload, **kwargs)

    def stamp(root, field, select, **kwargs):
        events.append(("stamp", field))
        return real_stamp(root, field, select, **kwargs)

    def acquire(path, **kwargs):
        events.append(("lock", pathlib.Path(path).name))
        return real_acquire(path, **kwargs)

    journal_lock = jsonl_append_lock_path(data / "logs" / "supervisor.jsonl")
    journal_lock.parent.mkdir(parents=True)
    fd = real_acquire(journal_lock, timeout_sec=1.0)
    assert fd is not None
    monkeypatch.setattr(owned_shutdown, "atomic_write_json", write)
    monkeypatch.setattr(owned_shutdown, "_stamp", stamp)
    monkeypatch.setattr(platform_layer, "acquire_exclusive_file_lock", acquire)
    try:
        started = time.monotonic()
        owned_shutdown.begin_owned_stop(data, cause="launcher_quit")
        elapsed = time.monotonic() - started
    finally:
        release(journal_lock, fd)
    assert events[:2] == [("entry", True), ("stamp", "stop_requested_at")], events
    assert ("lock", journal_lock.name) not in events and elapsed < 1.0
    assert _entry(data)["cause"] == "launcher_quit"


def test_a_failed_entry_write_never_stops_the_door(data, monkeypatch, caplog):
    from ouroboros import owned_shutdown

    _generation(monkeypatch, "gen-a")
    attempts = _refuse_entry_writes(monkeypatch)
    stamped = []
    monkeypatch.setattr(owned_shutdown, "_stamp", lambda root, field, select, **_kw: stamped.append(field) or [])
    with caplog.at_level(logging.WARNING, logger="ouroboros.owned_shutdown"):
        owned_shutdown.begin_owned_stop(data, cause="launcher_quit")
    assert stamped == ["stop_requested_at"]  # the stop requests are still stamped
    assert not owned_shutdown.shutdown_entry_path(data).exists()
    assert "Shutdown entry (launcher_quit) not written" in caplog.text
    owned_shutdown.begin_owned_stop(data, cause="restart_requested")  # a later door neither retries nor replaces
    assert [attempt["cause"] for attempt in attempts] == ["launcher_quit"] and stamped == ["stop_requested_at"] * 2


@pytest.mark.parametrize("refused", [False, True])
def test_the_emergency_door_records_the_restart_cause_and_still_stops_workers(data, monkeypatch, refused):
    import server

    _generation(monkeypatch, "gen-a")
    if refused:
        _refuse_entry_writes(monkeypatch)
    worker_calls = []
    monkeypatch.setattr(server, "DATA_DIR", data)
    monkeypatch.setattr(server, "stop_owned_work", lambda _root: {"state": "completed"})
    monkeypatch.setattr(server, "_stop_owned_daemon_for_new_pin", lambda: None)
    monkeypatch.setattr(server.body_adoption, "arm", lambda *_a, **_k: None)
    monkeypatch.setattr("supervisor.workers.kill_workers", lambda **kw: worker_calls.append(kw))
    monkeypatch.setattr("supervisor.workers.last_worker_exit_census", lambda: {"unconfirmed": []})
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda: None)
    server._restart_requested.set()
    try:
        server._emergency_process_cleanup()
    finally:
        server._restart_requested.clear()
    assert len(worker_calls) == 1 and worker_calls[0]["stop_source"] == "server_shutdown"
    if refused:
        from ouroboros import owned_shutdown

        assert not owned_shutdown.shutdown_entry_path(data).exists()
    else:
        assert (_entry(data)["cause"], _entry(data)["restart_exit"]) == ("restart_requested", True)


def test_the_next_start_reports_the_previous_entry_and_claims_the_file(data, monkeypatch, caplog):
    from ouroboros import owned_shutdown

    _generation(monkeypatch, "gen-a")
    owned_shutdown.begin_owned_stop(data, cause="launcher_quit")
    _generation(monkeypatch, "gen-b")  # the launcher's kill: the next start is a new process
    with caplog.at_level(logging.INFO, logger="ouroboros.owned_shutdown"):
        assert owned_shutdown.finish_unconfirmed_stops(data) == {"retried": 0, "confirmed": 0, "unconfirmed": 0}
    claimed = _entry(data)
    assert claimed["generation"] == "gen-b" and claimed["started_at"] and "cause" not in claimed
    assert "gen-a" in caplog.text and "with cause launcher_quit" in caplog.text and "not a confirmed exit" in caplog.text
    assert owned_shutdown.claim_shutdown_entry(data) == {"state": "current"}  # claimed once per generation


@pytest.mark.parametrize("stored,state", [
    (None, "absent"),
    ("{torn", "unreadable"),
    ("[]", "unreadable"),
    (json.dumps({"schema_version": 1, "cause": "launcher_quit"}), "unreadable"),
    (json.dumps({"schema_version": 2, "generation": "gen-a", "cause": "launcher_quit"}), "unreadable"),
    (json.dumps({"schema_version": 1, "generation": "gen-a", "supervisor_pid": 7, "started_at": "t0"}), "no_entry"),
    (json.dumps({"schema_version": 1, "generation": "gen-a", "supervisor_pid": 7, "started_at": "t0",
                 "entered_at": "t1", "cause": "launcher_gone", "restart_exit": False}), "entered"),
])
def test_the_reader_names_each_previous_record_and_always_claims(data, monkeypatch, stored, state):
    from ouroboros import owned_shutdown

    path = owned_shutdown.shutdown_entry_path(data)
    if stored is not None:
        path.parent.mkdir(parents=True)
        path.write_text(stored, encoding="utf-8")
    _generation(monkeypatch, "gen-b")
    report = owned_shutdown.claim_shutdown_entry(data)
    assert report["state"] == state
    if state in ("entered", "no_entry"):
        assert report["previous"] == json.loads(stored)
    claimed = _entry(data)
    assert claimed["generation"] == "gen-b" and "cause" not in claimed


def test_an_older_cause_never_explains_a_later_death_before_any_door(data, monkeypatch):
    from ouroboros import owned_shutdown

    _generation(monkeypatch, "gen-a")
    owned_shutdown.claim_shutdown_entry(data)
    owned_shutdown.begin_owned_stop(data, cause="launcher_quit")
    _generation(monkeypatch, "gen-b")
    assert owned_shutdown.claim_shutdown_entry(data)["previous"]["cause"] == "launcher_quit"
    # gen-b dies (OOM, a kill) before any door: nothing more is written.
    _generation(monkeypatch, "gen-c")
    report = owned_shutdown.claim_shutdown_entry(data)
    assert report["state"] == "no_entry"
    assert report["previous"]["generation"] == "gen-b" and "cause" not in report["previous"]


def test_this_generation_never_reads_its_own_record_as_the_previous_one(data, monkeypatch):
    from ouroboros import owned_shutdown

    _generation(monkeypatch, "gen-a")
    owned_shutdown.begin_owned_stop(data, cause="external_signal")  # a door before the boot step
    before = owned_shutdown.shutdown_entry_path(data).read_bytes()
    assert owned_shutdown.claim_shutdown_entry(data) == {"state": "current"}
    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())  # same process, fresh stop
    assert owned_shutdown.claim_shutdown_entry(data) == {"state": "current"}
    assert owned_shutdown.shutdown_entry_path(data).read_bytes() == before


def test_an_unclaimable_file_does_not_keep_an_older_cause(data, monkeypatch):
    from ouroboros import owned_shutdown

    _generation(monkeypatch, "gen-a")
    owned_shutdown.begin_owned_stop(data, cause="launcher_quit")
    _generation(monkeypatch, "gen-b")
    with monkeypatch.context() as patch:
        _refuse_entry_writes(patch)
        report = owned_shutdown.claim_shutdown_entry(data)
    assert report["state"] == "entered" and report["previous"]["generation"] == "gen-a"
    assert not owned_shutdown.shutdown_entry_path(data).exists()  # gen-b's later death stays uncaused
    _generation(monkeypatch, "gen-c")
    assert owned_shutdown.claim_shutdown_entry(data) == {"state": "absent"}


def test_a_failing_claim_never_skips_the_custody_retry(data, monkeypatch):
    from ouroboros import owned_shutdown

    _generation(monkeypatch, "gen-a")
    monkeypatch.setattr(owned_shutdown, "_generation_record", lambda: (_ for _ in ()).throw(ImportError("x")))
    retried = []
    monkeypatch.setattr(owned_shutdown, "_stop_records", lambda root, pending, deadline: retried.append(pending) or [])
    owned_shutdown._update(data, lambda document: document["records"].update(
        {"r1": {"record_id": "r1", "kind": "foreground", "stop_requested_at": "t"}}) or True)
    assert owned_shutdown.finish_unconfirmed_stops(data)["retried"] == 1 and len(retried) == 1


def test_a_planned_restart_keeps_its_own_record_before_its_teardown_entry():
    """The reader contract's other half: a planned restart stops its workers before its teardown door,
    so its restart transaction, not the entry, records that window (behaviour: test_delegate_recovery's
    door test reads the prepared transaction before any process teardown). Its teardown door then writes
    ``restart_requested``; no earlier grace or entry is started for a diagnostic."""
    import server
    from ouroboros import server_restart

    planned = inspect.getsource(server_restart._perform_supervisor_restart)
    assert planned.index("prepare_restart_returns(") < planned.index("ctx.kill_workers(")
    assert "begin_owned_stop" not in planned and "shutdown_entry" not in planned
    assert "begin_owned_stop(lifespan_drive_root, cause=server_stop_cause())" in inspect.getsource(server.lifespan)


def test_panic_never_enters_an_ordinary_door():
    from ouroboros import server_control

    source = inspect.getsource(server_control)
    assert "begin_owned_stop" not in source and "shutdown_entry" not in source


_GENERATION_CHILD = r'''
import json, os, sys
from ouroboros import delegate_recovery, owned_shutdown
root, mode = sys.argv[1], sys.argv[2]
report = owned_shutdown.claim_shutdown_entry(owned_shutdown.installation_root(root))  # the boot step's claim
print(json.dumps({"generation": delegate_recovery._PROCESS_GENERATION, "pid": os.getpid(), "report": report}), flush=True)
if mode == "door":
    owned_shutdown.begin_owned_stop(root, cause="launcher_quit")
    print("entered", flush=True)
sys.stdin.read()  # terminal custody and the late row would follow; the parent kills this process first
'''


def _generation_process(root, mode):
    env = {**os.environ, "OUROBOROS_DATA_DIR": str(root), "PYTHONPATH": str(REPO)}
    return subprocess.Popen([sys.executable, "-c", _GENERATION_CHILD, str(root), mode], cwd=REPO, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _lines(proc, count, timeout=60.0):
    out: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [out.put(proc.stdout.readline()) for _ in range(count)], daemon=True).start()
    lines = []
    for _ in range(count):
        try:
            line = out.get(timeout=timeout)
        except queue.Empty:
            line = ""
        if not line:
            proc.kill()
            pytest.fail(f"generation child stopped early: {proc.communicate(timeout=10)}")
        lines.append(line)
    return lines


def _kill(proc):
    """SIGKILL our own unreaped child (its PID cannot be reused before ``wait``)."""
    assert proc.poll() is None
    proc.kill()
    proc.wait(timeout=10)
    proc.stdin.close()
    proc.stdout.close()
    proc.stderr.close()


def test_a_kill_after_the_entry_leaves_the_cause_for_the_next_generation(data):
    """Three real processes: A enters its shutdown and is killed before any late row; B reads A's
    cause and dies before any door; C reads that B recorded no entry, never A's older cause."""
    from ouroboros import owned_shutdown

    a = _generation_process(data, "door")
    first, entered = _lines(a, 2)
    assert entered.strip() == "entered" and json.loads(first)["report"] == {"state": "absent"}
    _kill(a)
    entry = _entry(data)
    a_self = json.loads(first)  # the child's own identity (a Windows venv launcher's PID is another process)
    assert (entry["generation"], entry["supervisor_pid"]) == (a_self["generation"], a_self["pid"])
    assert (entry["cause"], entry["restart_exit"]) == ("launcher_quit", False)
    journal = data / "logs" / "supervisor.jsonl"
    assert not journal.exists() or "server_shutdown" not in journal.read_text(encoding="utf-8")

    b = _generation_process(data, "claim")
    [line] = _lines(b, 1)
    read_by_b = json.loads(line)
    assert read_by_b["report"]["state"] == "entered"
    assert read_by_b["report"]["previous"]["generation"] == entry["generation"]
    _kill(b)  # B dies before any door

    c = _generation_process(data, "claim")
    [line] = _lines(c, 1)
    _kill(c)
    report = json.loads(line)["report"]
    assert report["state"] == "no_entry" and report["previous"]["generation"] == read_by_b["generation"]
    assert "cause" not in report["previous"]
    assert owned_shutdown.shutdown_entry_path(data).is_file()
