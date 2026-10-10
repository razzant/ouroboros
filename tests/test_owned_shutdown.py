"""The installation's ownership set and its one bounded stop (ARCHITECTURE §9).

Both custody funnels write ``state/owned_processes.json``; the exit stop reads it (never a
walk), stamps before waiting, confirms through liveness and leaves the unconfirmed rest
stamped for the next start, which retries them before admission under the same bound.
"""
import inspect
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.serial


def _sleeper():
    from ouroboros.platform_layer import subprocess_new_group_kwargs
    from ouroboros import workspace_executor as executor

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                            **subprocess_new_group_kwargs())
    for _ in range(200):  # the registered command hash must already be readable
        if executor._process_command_sha256(proc.pid):
            break
        time.sleep(0.02)
    return proc


def _reap(*procs):
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=10)
        except Exception:
            pass


def _by_id(root):
    from ouroboros import owned_shutdown

    return {entry["record_id"]: entry for entry in owned_shutdown.owned_records(root)}


def _docker_service(root):
    from ouroboros import workspace_executor as executor

    return executor._register_process(root, {
        "record_type": "service", "service_id": "task:svc", "task_id": "task", "name": "svc",
        "executor_type": "docker_exec", "executor_id": "docker", "container_name": "bench",
        "backend_pid": "4242", "host_pid": 0})


def _budget(monkeypatch, seconds):
    from ouroboros import owned_shutdown

    monkeypatch.setattr(owned_shutdown, "_stop_budget_sec", lambda: seconds)


def test_both_funnels_write_the_set(tmp_path):
    from ouroboros import owned_shutdown
    from ouroboros import workspace_executor as executor
    from ouroboros.process_custody import ledger_path, spawn_supervised

    data, drive = tmp_path / "data", tmp_path / "data" / "task_drives" / "t1"
    drive.mkdir(parents=True)
    foreground = _sleeper()
    service = None
    try:
        path = executor._register_process(drive, {"record_type": "foreground", "executor_type": "local",
                                                  "executor_id": "host", "host_pid": foreground.pid})
        service = spawn_supervised([sys.executable, "-c", "import time; time.sleep(60)"], drive_root=drive,
                                   purpose="service:demo", scope="task", owner_task_id="t1",
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        docker = _docker_service(data)
        records = _by_id(data)
        assert owned_shutdown.owned_processes_path(data).is_file()
        assert not (drive / "state" / owned_shutdown.OWNED_PROCESSES_FILENAME).exists()
        local = records[path.stem]
        assert (local["kind"], local["host_pid"], local["record_path"]) == ("foreground", foreground.pid, str(path))
        assert local["drive_root"] == str(drive.resolve()) and local["stop_requested_at"] is None
        supervised = records[f"pid-{service.pid}"]
        assert (supervised["kind"], supervised["scope"], supervised["owner_task"]) == ("supervised", "task", "t1")
        assert supervised["record_path"] == str(ledger_path(drive.resolve())) and supervised["birth"]
        backend = records[docker.stem]
        assert (backend["kind"], backend["host_pid"]) == ("service", 0)
        assert (backend["container_name"], backend["backend_pid"]) == ("bench", "4242")
        # Ledger rows are still written exactly as before: the set is an addition.
        rows = [json.loads(line) for line in ledger_path(drive).read_text(encoding="utf-8").splitlines()]
        assert [row["pid"] for row in rows] == [service.pid]
        executor._forget_process(path)
        assert path.stem not in _by_id(data) and not path.exists()
    finally:
        _reap(foreground, *([service] if service else []))


def test_a_record_under_a_task_drive_is_stopped_at_exit(tmp_path, monkeypatch):
    from ouroboros import owned_shutdown
    from ouroboros import workspace_executor as executor
    from ouroboros.process_custody import spawn_supervised

    _budget(monkeypatch, 5.0)
    data, drive = tmp_path / "data", tmp_path / "data" / "task_drives" / "t1"
    drive.mkdir(parents=True)
    foreground = _sleeper()
    service = spawn_supervised([sys.executable, "-c", "import time; time.sleep(60)"], drive_root=drive,
                               purpose="service:demo", scope="session", owner_task_id="t1",
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    try:
        executor._register_process(drive, {"record_type": "foreground", "executor_type": "local",
                                           "executor_id": "host", "host_pid": foreground.pid})
        monkeypatch.setattr(os, "walk", lambda *_a, **_k: pytest.fail("the exit never walks data/state"))
        outcome = owned_shutdown.stop_owned_work(data)
        assert outcome["state"] == "completed" and outcome["targets"] == outcome["confirmed"] == 2
        for proc in (foreground, service):
            proc.wait(timeout=10)
        assert _by_id(data) == {}
        assert not list((drive / "state" / "workspace_executor_processes").glob("*.json"))
    finally:
        _reap(foreground, service)


def test_machinery_daemons_and_companions_are_left_to_their_owners(tmp_path, monkeypatch):
    from ouroboros import owned_shutdown
    from ouroboros.process_custody import record_process

    _budget(monkeypatch, 2.0)
    data = tmp_path / "data"
    procs = [_sleeper() for _ in range(3)]
    try:
        for proc, purpose, scope in zip(procs, ("worker:0", "claudexor_daemon", "companion:skill:bot"),
                                        ("session", "daemon", "daemon")):
            record_process(data, pid=proc.pid, cmd="fixture", purpose=purpose, scope=scope)
        outcome = owned_shutdown.stop_owned_work(data)
        assert outcome["targets"] == 0
        assert all(proc.poll() is None for proc in procs)
        kinds = sorted(entry["kind"] for entry in _by_id(data).values())
        assert kinds == ["companion", "supervised", "supervised"]
        assert all(entry["stop_requested_at"] is None for entry in _by_id(data).values())
    finally:
        _reap(*procs)


def test_two_concurrent_stop_callers_run_one_operation(tmp_path, monkeypatch):
    from ouroboros import owned_shutdown

    runs, entered, release = [], threading.Event(), threading.Event()

    def run(root, deadline):
        runs.append(root)
        entered.set()
        release.wait(10)
        return {"state": "completed", "targets": 0, "confirmed": 0, "unconfirmed": [], "elapsed_sec": 0.0}

    monkeypatch.setattr(owned_shutdown, "_run_stop", run)
    results = []
    first = threading.Thread(target=lambda: results.append(owned_shutdown.stop_owned_work(tmp_path)))
    first.start()
    assert entered.wait(10)
    second = threading.Thread(target=lambda: results.append(owned_shutdown.stop_owned_work(tmp_path)))
    second.start()
    time.sleep(0.2)
    assert second.is_alive(), "the second caller joins the running stop instead of returning or starting one"
    release.set()
    first.join(10)
    second.join(10)
    assert len(runs) == 1 and len(results) == 2 and results[0] == results[1]
    assert owned_shutdown.stop_owned_work(tmp_path) == results[0] and len(runs) == 1


def test_a_joiner_waits_only_until_the_shared_deadline(tmp_path, monkeypatch):
    from ouroboros import owned_shutdown

    _budget(monkeypatch, 0.3)
    release = threading.Event()
    monkeypatch.setattr(owned_shutdown, "_run_stop", lambda root, deadline: release.wait(10) and {})
    starter = threading.Thread(target=lambda: owned_shutdown.stop_owned_work(tmp_path))
    starter.start()
    time.sleep(0.05)
    began = time.monotonic()
    assert owned_shutdown.stop_owned_work(tmp_path) == {"state": "in_progress", "joined": True}
    assert time.monotonic() - began < 2.0
    release.set()
    starter.join(10)


def test_an_unconfirmable_stop_stays_stamped_and_returns_at_the_deadline(tmp_path, monkeypatch):
    from ouroboros import owned_shutdown
    from ouroboros import workspace_executor as executor

    _budget(monkeypatch, 0.5)
    data = tmp_path / "data"
    path = _docker_service(data)
    calls, answer = [], threading.Event()

    def stuck_backend(record, *, wait=True):
        calls.append(record["backend_pid"])
        answer.wait(10)  # a Docker CLI that does not answer inside the grace
        return False

    monkeypatch.setattr(executor, "_kill_docker_record", stuck_backend)
    began = time.monotonic()
    try:
        outcome = owned_shutdown.stop_owned_work(data)
        assert time.monotonic() - began < 2.0
    finally:
        answer.set()
    assert outcome["state"] == "unconfirmed" and outcome["unconfirmed"] == [path.stem]
    entry = _by_id(data)[path.stem]
    assert entry["stop_requested_at"] and entry["unconfirmed_since"] and calls == ["4242"]
    assert path.exists()
    rows = (data / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(rows[-1])["type"] == "owned_stop_unconfirmed"


def test_targets_are_stamped_before_any_wait(tmp_path, monkeypatch):
    """A launcher kill inside the stop still leaves the pending stop recorded."""
    from ouroboros import owned_shutdown
    from ouroboros import workspace_executor as executor

    _budget(monkeypatch, 2.0)
    data = tmp_path / "data"
    path = _docker_service(data)
    seen = []

    def observe(record, *, wait=True):
        seen.append(_by_id(data)[path.stem]["stop_requested_at"])
        return False

    monkeypatch.setattr(executor, "_kill_docker_record", observe)
    owned_shutdown.stop_owned_work(data)
    assert seen and seen[0]


@pytest.mark.parametrize("backend_confirms", [True, False])
def test_the_next_start_finishes_a_stamped_stop(tmp_path, monkeypatch, backend_confirms):
    """Both directions: a confirmed stop leaves the set; a still-live one stays recorded and counted."""
    from ouroboros import owned_shutdown
    from ouroboros import workspace_executor as executor

    data = tmp_path / "data"
    path = _docker_service(data)
    assert owned_shutdown.import_inherited_records(data) == 1
    monkeypatch.setattr(executor, "_kill_docker_record", lambda record, *, wait=True: False)
    _budget(monkeypatch, 0.3)
    owned_shutdown.stop_owned_work(data)  # the previous generation could not confirm it
    stamped = _by_id(data)[path.stem]
    assert stamped["unconfirmed_since"]

    _budget(monkeypatch, 1.0)
    attempts = []
    monkeypatch.setattr(executor, "_kill_docker_record",
                        lambda record, *, wait=True: attempts.append(record["id"]) or backend_confirms)
    began = time.monotonic()
    counts = owned_shutdown.finish_unconfirmed_stops(data)
    assert time.monotonic() - began < 3.0 and attempts == [path.stem]
    assert "imported" not in counts and counts["retried"] == 1
    if backend_confirms:
        assert counts["confirmed"] == 1 and counts["unconfirmed"] == 0
        assert path.stem not in _by_id(data) and not path.exists()
    else:
        assert counts["confirmed"] == 0 and counts["unconfirmed"] == 1
        kept = _by_id(data)[path.stem]
        assert kept["unconfirmed_since"] == stamped["unconfirmed_since"] and path.exists()
    row = json.loads((data / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["type"] == "owned_stops_finished" and row["retried"] == 1


def test_a_crash_mid_stop_is_finished_at_the_next_start(tmp_path, monkeypatch):
    """Killed after the stamp, before any unconfirmed_since: the next start still owns the stop."""
    from ouroboros import owned_shutdown
    from ouroboros import workspace_executor as executor

    _budget(monkeypatch, 5.0)
    data = tmp_path / "data"
    proc = _sleeper()
    try:
        path = executor._register_process(data, {"record_type": "foreground", "executor_type": "local",
                                                  "executor_id": "host", "host_pid": proc.pid})
        assert owned_shutdown.import_inherited_records(data) == 1
        owned_shutdown._stamp(data, "stop_requested_at", owned_shutdown._is_stop_target)
        assert _by_id(data)[path.stem]["unconfirmed_since"] is None and proc.poll() is None
        counts = owned_shutdown.finish_unconfirmed_stops(data)
        assert counts["retried"] == counts["confirmed"] == 1
        proc.wait(timeout=10)
        assert _by_id(data) == {} and not path.exists()
    finally:
        _reap(proc)



def test_a_target_recorded_after_the_stop_began_is_finished_by_the_next_start(tmp_path, monkeypatch):
    """The owner-Restart door stops before the server exits: a process recorded in that window is born
    pending (stamped), never silently missed; a late non-target is never stamped."""
    from ouroboros import owned_shutdown
    from ouroboros import workspace_executor as executor
    from ouroboros.process_custody import record_process

    _budget(monkeypatch, 5.0)
    data = tmp_path / "data"
    assert owned_shutdown.stop_owned_work(data)["targets"] == 0  # this generation's stop is over
    late, machinery = _sleeper(), _sleeper()
    try:
        path = executor._register_process(data, {"record_type": "foreground", "executor_type": "local",
                                                  "executor_id": "host", "host_pid": late.pid})
        record_process(data, pid=machinery.pid, cmd="fixture", purpose="worker:0", scope="session")
        records = _by_id(data)
        assert records[path.stem]["stop_requested_at"]
        assert records[f"pid-{machinery.pid}"]["stop_requested_at"] is None
        assert owned_shutdown.stop_owned_work(data)["targets"] == 0  # joined: no second stop starts
        assert late.poll() is None
        counts = owned_shutdown.finish_unconfirmed_stops(data)  # the next start
        assert counts["retried"] == counts["confirmed"] == 1
        late.wait(timeout=10)
        assert machinery.poll() is None and path.stem not in _by_id(data)
    finally:
        _reap(late, machinery)

def test_a_live_process_under_an_unproven_identity_is_never_signalled(tmp_path, monkeypatch):
    from ouroboros import owned_shutdown
    from ouroboros import workspace_executor as executor

    _budget(monkeypatch, 0.5)
    data = tmp_path / "data"
    proc = _sleeper()
    try:
        path = executor._register_process(data, {"record_type": "foreground", "executor_type": "local",
                                                  "executor_id": "host", "host_pid": proc.pid})
        record = json.loads(path.read_text(encoding="utf-8"))
        record["host_command_sha256"] = "0" * 64  # the pid now names some other command
        path.write_text(json.dumps(record), encoding="utf-8")
        monkeypatch.setattr(executor, "_kill_host_pid", lambda _pid: pytest.fail("unproven identity signalled"))
        outcome = owned_shutdown.stop_owned_work(data)
        assert outcome["unconfirmed"] == [path.stem] and proc.poll() is None
    finally:
        _reap(proc)


def test_a_ledgered_process_whose_identity_cannot_be_measured_is_never_signalled(tmp_path, monkeypatch):
    """Retention may keep a row whose start time is unmeasurable (a fresh Windows service); the stop's signal
    needs the explicit stop's measured identity, so such a row stays recorded and its pid is never signalled.
    The measured case is stopped (test_a_record_under_a_task_drive_is_stopped_at_exit)."""
    from ouroboros import owned_shutdown
    from ouroboros import platform_layer
    from ouroboros.process_custody import record_process

    _budget(monkeypatch, 0.5)
    data = tmp_path / "data"
    proc = _sleeper()
    try:
        record_process(data, pid=proc.pid, cmd="fixture", purpose="service:demo", scope="task",
                       owner_task_id="t1", reap_process_group=False)
        record_id = f"pid-{proc.pid}"

        def unmeasured(document):
            fingerprint = document["records"][record_id]["ledger_entry"]["fingerprint"]
            fingerprint.pop("start_time_boot", None)
            fingerprint["start_time"] = ""
            return True

        assert owned_shutdown._update(owned_shutdown.installation_root(data), unmeasured)
        signals = []  # the stop runs on its own threads: record calls, never raise inside them
        for name in ("kill_pid_tree", "kill_process_group_id"):
            monkeypatch.setattr(platform_layer, name, lambda *a, _name=name, **_k: signals.append((_name, a)))
        outcome = owned_shutdown.stop_owned_work(data)
        assert signals == [] and outcome["unconfirmed"] == [record_id] and proc.poll() is None
        assert _by_id(data)[record_id]["unconfirmed_since"]
    finally:
        _reap(proc)


def test_the_shutdown_entry_records_every_pending_stop_before_any_wait(tmp_path, monkeypatch):
    """A door stamps at entry; if the launcher kills the server before the stop itself runs, the
    next start still finds the stop requested and finishes it."""
    from ouroboros import owned_shutdown
    from ouroboros import workspace_executor as executor

    _budget(monkeypatch, 5.0)
    data = tmp_path / "data"
    proc = _sleeper()
    try:
        path = executor._register_process(data, {"record_type": "foreground", "executor_type": "local",
                                                  "executor_id": "host", "host_pid": proc.pid})
        owned_shutdown.begin_owned_stop(data)
        assert _by_id(data)[path.stem]["stop_requested_at"]
        # The launcher's kill: this generation's stop never ran; the next start is a new process.
        monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())
        counts = owned_shutdown.finish_unconfirmed_stops(data)
        assert counts["retried"] == counts["confirmed"] == 1
        proc.wait(timeout=10)
        assert path.stem not in _by_id(data)
    finally:
        _reap(proc)


def test_the_grace_starts_at_the_first_door_and_no_lock_is_awaited_after_it(tmp_path, monkeypatch):
    """The stop's deadline is the one the first door started; past it the stop takes no fresh lock wait
    (a held custody lock costs one attempt, not two 2 s waits) and leaves the target stamped."""
    from ouroboros import owned_shutdown
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.process_custody import ledger_path
    from ouroboros.utils import jsonl_append_lock_path
    from ouroboros import workspace_executor as executor

    _budget(monkeypatch, 1.0)
    data = tmp_path / "data"
    proc = _sleeper()
    lock_path = jsonl_append_lock_path(ledger_path(data))
    fd = None
    try:
        path = executor._register_process(data, {"record_type": "foreground", "executor_type": "local",
                                                  "executor_id": "host", "host_pid": proc.pid})
        owned_shutdown.begin_owned_stop(data)  # the grace starts at this door
        time.sleep(1.1)  # other owners used the whole grace before the stop runs
        fd = acquire_exclusive_file_lock(lock_path, timeout_sec=1.0)
        assert fd is not None
        started = time.monotonic()
        outcome = owned_shutdown.stop_owned_work(data)
        # A deadline restarted at the stop would wait ~1 s on the held lock; an expired one waits for nothing.
        assert time.monotonic() - started < 0.5
        assert outcome["state"] == "unconfirmed" and path.stem in outcome["unconfirmed"]
        assert _by_id(data)[path.stem]["stop_requested_at"]  # the next start retries it
    finally:
        if fd is not None:
            release_exclusive_file_lock(lock_path, fd)
        _reap(proc)


def test_a_registration_under_a_held_custody_lock_is_never_lost(tmp_path, monkeypatch):
    """Another writer holds the custody lock while a launch registers: the entry lands as a lock-free
    pending file, every read sees it, the next update folds it in, and the exit stops the process."""
    from ouroboros import owned_shutdown
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.process_custody import ledger_path
    from ouroboros.utils import jsonl_append_lock_path
    from ouroboros import workspace_executor as executor

    _budget(monkeypatch, 5.0)
    monkeypatch.setattr(owned_shutdown, "_update", lambda root, change, *, timeout_sec=2.0, _real=owned_shutdown._update:
                        _real(root, change, timeout_sec=min(timeout_sec, 0.2)))
    data = tmp_path / "data"
    proc = _sleeper()
    lock_path = jsonl_append_lock_path(ledger_path(data))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = acquire_exclusive_file_lock(lock_path, timeout_sec=1.0)
    try:
        assert fd is not None
        path = executor._register_process(data, {"record_type": "foreground", "executor_type": "local",
                                                  "executor_id": "host", "host_pid": proc.pid})
        pending = list((data / "state" / owned_shutdown.PENDING_DIRNAME).glob("*.json"))
        assert len(pending) == 1 and path.stem in _by_id(data)  # named while the lock is held
        release_exclusive_file_lock(lock_path, fd)
        fd = None
        outcome = owned_shutdown.stop_owned_work(data)
        assert outcome["state"] == "completed" and outcome["confirmed"] == 1
        proc.wait(timeout=10)
        assert path.stem not in _by_id(data)
        assert not list((data / "state" / owned_shutdown.PENDING_DIRNAME).glob("*.json"))  # folded, then forgotten
    finally:
        if fd is not None:
            release_exclusive_file_lock(lock_path, fd)
        _reap(proc)


def test_a_forget_of_a_pending_registration_does_not_resurrect_it(tmp_path):
    from ouroboros import owned_shutdown

    data = tmp_path / "data"
    entry = {"record_id": "gone", "kind": "foreground", "host_pid": 0, "birth": "b", "drive_root": str(data),
             "record_path": str(data / "missing.json"), "stop_requested_at": None, "unconfirmed_since": None}
    pending_dir = data / "state" / owned_shutdown.PENDING_DIRNAME
    pending_dir.mkdir(parents=True)
    (pending_dir / "gone.x.json").write_text(json.dumps(entry), encoding="utf-8")
    assert "gone" in _by_id(data)
    assert owned_shutdown._forget(data, ["gone"])
    assert "gone" not in _by_id(data) and not list(pending_dir.glob("*.json"))


def test_a_contended_registration_during_the_grace_is_retried_by_the_next_start(tmp_path, monkeypatch):
    """A launch registers after the shutdown door opened, while the custody lock is held: its pending file
    is born stamped, so the next start (a new process with no deadline) still stops it."""
    from ouroboros import owned_shutdown
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.process_custody import ledger_path
    from ouroboros.utils import jsonl_append_lock_path
    from ouroboros import workspace_executor as executor

    _budget(monkeypatch, 5.0)
    monkeypatch.setattr(owned_shutdown, "_update", lambda root, change, *, timeout_sec=2.0, _real=owned_shutdown._update:
                        _real(root, change, timeout_sec=min(timeout_sec, 0.2)))
    data = tmp_path / "data"
    proc = _sleeper()
    lock_path = jsonl_append_lock_path(ledger_path(data))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    owned_shutdown.begin_owned_stop(data)  # the door is open: this generation is shutting down
    fd = acquire_exclusive_file_lock(lock_path, timeout_sec=1.0)
    try:
        assert fd is not None
        path = executor._register_process(data, {"record_type": "foreground", "executor_type": "local",
                                                  "executor_id": "host", "host_pid": proc.pid})
        [pending] = (data / "state" / owned_shutdown.PENDING_DIRNAME).glob("*.json")
        assert json.loads(pending.read_text(encoding="utf-8"))["stop_requested_at"]
        release_exclusive_file_lock(lock_path, fd)
        fd = None
        monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())  # the launcher's kill
        counts = owned_shutdown.finish_unconfirmed_stops(data)
        assert counts["retried"] == counts["confirmed"] == 1
        proc.wait(timeout=10)
        assert path.stem not in _by_id(data)
    finally:
        if fd is not None:
            release_exclusive_file_lock(lock_path, fd)
        _reap(proc)


def test_a_later_door_after_the_grace_takes_one_lock_attempt(tmp_path, monkeypatch):
    from ouroboros import owned_shutdown
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.process_custody import ledger_path
    from ouroboros.utils import jsonl_append_lock_path

    _budget(monkeypatch, 0.2)
    data = tmp_path / "data"
    lock_path = jsonl_append_lock_path(ledger_path(data))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    owned_shutdown.begin_owned_stop(data)  # the first door
    time.sleep(0.3)  # the grace is over
    fd = acquire_exclusive_file_lock(lock_path, timeout_sec=1.0)
    try:
        started = time.monotonic()
        owned_shutdown.begin_owned_stop(data)  # a later door (owner Restart, then the teardown)
        assert time.monotonic() - started < 0.5
    finally:
        release_exclusive_file_lock(lock_path, fd)


def test_a_contended_first_door_still_records_its_stop_requests(tmp_path, monkeypatch):
    """The first door's stamp cannot take the custody lock: the stop requests land as pending files, so
    a launcher kill before the stop runs still leaves the next start a stop to finish."""
    from ouroboros import owned_shutdown
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.process_custody import ledger_path
    from ouroboros.utils import jsonl_append_lock_path
    from ouroboros import workspace_executor as executor

    _budget(monkeypatch, 5.0)
    monkeypatch.setattr(owned_shutdown, "_update", lambda root, change, *, timeout_sec=2.0, _real=owned_shutdown._update:
                        _real(root, change, timeout_sec=min(timeout_sec, 0.2)))
    data = tmp_path / "data"
    proc = _sleeper()
    lock_path = jsonl_append_lock_path(ledger_path(data))
    fd = None
    try:
        path = executor._register_process(data, {"record_type": "foreground", "executor_type": "local",
                                                  "executor_id": "host", "host_pid": proc.pid})
        fd = acquire_exclusive_file_lock(lock_path, timeout_sec=1.0)
        owned_shutdown.begin_owned_stop(data)
        assert _by_id(data)[path.stem]["stop_requested_at"]
        release_exclusive_file_lock(lock_path, fd)
        fd = None
        monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())  # the launcher's kill
        counts = owned_shutdown.finish_unconfirmed_stops(data)
        assert counts["retried"] == counts["confirmed"] == 1
        proc.wait(timeout=10)
    finally:
        if fd is not None:
            release_exclusive_file_lock(lock_path, fd)
        _reap(proc)


def test_a_pending_registration_is_stamped_durably_by_a_contended_first_door(tmp_path, monkeypatch):
    """Registered while the custody lock was held (a pending file), then the first door is contended too:
    the stamp a read gives that pending entry is persisted, so the next start still finishes the stop."""
    from ouroboros import owned_shutdown
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.process_custody import ledger_path
    from ouroboros.utils import jsonl_append_lock_path
    from ouroboros import workspace_executor as executor

    _budget(monkeypatch, 5.0)
    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())
    monkeypatch.setattr(owned_shutdown, "_update", lambda root, change, *, timeout_sec=2.0, _real=owned_shutdown._update:
                        _real(root, change, timeout_sec=min(timeout_sec, 0.2)))
    data = tmp_path / "data"
    proc = _sleeper()
    lock_path = jsonl_append_lock_path(ledger_path(data))
    fd = acquire_exclusive_file_lock(lock_path, timeout_sec=1.0)
    try:
        path = executor._register_process(data, {"record_type": "foreground", "executor_type": "local",
                                                  "executor_id": "host", "host_pid": proc.pid})
        assert list((data / "state" / owned_shutdown.PENDING_DIRNAME).glob("*.json"))
        owned_shutdown.begin_owned_stop(data)
        release_exclusive_file_lock(lock_path, fd)
        fd = None
        monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())  # the launcher's kill
        assert _by_id(data)[path.stem]["stop_requested_at"]
        counts = owned_shutdown.finish_unconfirmed_stops(data)
        assert counts["retried"] == counts["confirmed"] == 1
        proc.wait(timeout=10)
    finally:
        if fd is not None:
            release_exclusive_file_lock(lock_path, fd)
        _reap(proc)


def test_a_reader_racing_a_fold_still_names_the_pending_registration(tmp_path, monkeypatch):
    """Another process folds a pending file into the document and deletes it while a lock-free reader
    runs: reading pending files before the document keeps the entry visible either way."""
    from ouroboros import owned_shutdown
    import ouroboros.utils as utils

    data = tmp_path / "data"
    entry = {"record_id": "x", "kind": "foreground", "host_pid": 0, "birth": "b", "drive_root": str(data),
             "record_path": str(data / "x.json"), "stop_requested_at": "t", "unconfirmed_since": None}
    assert owned_shutdown._update(data, lambda document: True)  # an existing, empty document
    assert owned_shutdown.owned_processes_path(data).is_file()
    assert owned_shutdown._write_pending(data, entry)
    real_read, folding = utils.read_text_across_replace, []

    def read_then_fold(path, *args, **kwargs):
        old = real_read(path, *args, **kwargs)
        if not folding and pathlib.Path(path).name == owned_shutdown.OWNED_PROCESSES_FILENAME:
            folding.append(1)
            assert owned_shutdown._update(data, lambda document: False)  # the other process folds and deletes
        return old  # this reader saw the document as it was before the fold

    monkeypatch.setattr(utils, "read_text_across_replace", read_then_fold)
    assert "x" in {e["record_id"] for e in owned_shutdown.owned_records(data)}
    assert folding and not list((data / "state" / owned_shutdown.PENDING_DIRNAME).glob("*.json"))


def test_every_shutdown_door_records_its_stop_requests_before_any_wait():
    """Wiring: owner Restart, the lifespan teardown and the emergency exit each call begin_owned_stop
    before their first wait (deleting any one call fails here)."""
    import inspect

    import server
    from ouroboros import server_restart

    restart = inspect.getsource(server_restart._stop_owned_work)
    assert restart.index("begin_owned_stop(") < restart.index("_owned_live_task_ids(") < restart.index("kill_workers(")
    teardown = inspect.getsource(server.lifespan)
    stop_flag = teardown.index("_supervisor_stop.set()  # first")
    assert stop_flag < teardown.index("begin_owned_stop(", stop_flag) < teardown.index("supervisor_thread.join(", stop_flag)
    emergency = inspect.getsource(server._emergency_process_cleanup)
    assert emergency.index("begin_owned_stop(") < emergency.index("kill_workers(")


def test_the_ledger_compaction_forgets_dropped_rows(tmp_path):
    from ouroboros import owned_shutdown
    from ouroboros.process_custody import _read_ledger_records, _rewrite_ledger, record_process

    data = tmp_path / "data"
    live, gone = _sleeper(), _sleeper()
    try:
        for proc in (live, gone):
            record_process(data, pid=proc.pid, cmd="fixture", purpose="worker:0", scope="session")
        _reap(gone)
        _ok, entries, previous = _read_ledger_records(data, strict=False)
        _rewrite_ledger(data, [entry for entry in entries if entry["pid"] == live.pid], previous=previous)
        assert set(_by_id(data)) == {f"pid-{live.pid}"}
        assert owned_shutdown.owned_records(data)[0]["ledger_entry"]["purpose"] == "worker:0"
    finally:
        _reap(live)


def test_an_unchanged_re_record_does_not_rewrite_the_set(tmp_path):
    from ouroboros import owned_shutdown
    from ouroboros.process_custody import record_process

    data = tmp_path / "data"
    proc = _sleeper()
    try:
        record_process(data, pid=proc.pid, cmd="fixture", purpose="worker:0", scope="session")
        stamp = owned_shutdown.owned_processes_path(data).stat().st_mtime_ns
        time.sleep(0.05)
        record_process(data, pid=proc.pid, cmd="fixture", purpose="worker:0", scope="session")
        assert owned_shutdown.owned_processes_path(data).stat().st_mtime_ns == stamp
    finally:
        _reap(proc)


def test_child_drives_map_to_their_installation(tmp_path, monkeypatch):
    from ouroboros import owned_shutdown

    root = tmp_path / "isolated"
    nested = root / "state" / "headless_tasks" / "t1" / "data" / "task_drives" / "t2"
    assert owned_shutdown.installation_root(nested) == root.resolve()
    assert owned_shutdown.installation_root(root / "elsewhere") == (root / "elsewhere").resolve()
    configured = tmp_path / "configured"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(configured))
    assert owned_shutdown.installation_root(configured / "state" / "any" / "nested" / "drive") == configured.resolve()


def test_kill_workers_joins_outside_the_queue_lock(tmp_path, monkeypatch):
    from tests._budget_pause_exact_helpers import _install_queue

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    observed = []

    def join(timeout=None):
        holder = []

        def probe():  # another thread: the queue RLock is reentrant for the joining one
            holder.append(queue._queue_lock.acquire(timeout=1))
            if holder[0]:
                queue._queue_lock.release()

        prober = threading.Thread(target=probe)
        prober.start()
        prober.join(5)
        observed.append((bool(holder and holder[0]), workers.WORKERS[0].reaping))

    proc = SimpleNamespace(pid=4242, is_alive=lambda: False, join=join, terminate=lambda: None)
    workers.WORKERS[0] = SimpleNamespace(wid=0, proc=proc, busy_task_id=None, reaping=False)
    monkeypatch.setattr(workers, "kill_worker_tree", lambda pid, **_k: None)
    monkeypatch.setattr(workers, "_kill_survivors", lambda: None)
    workers.kill_workers(force=True, archive_service_logs=False, reconcile_delegate_custody=False)
    assert observed == [(True, True)], "joins run outside _queue_lock while the slot is held as reaping"
    assert workers.WORKERS == {}


def test_every_door_joins_the_one_stop_and_the_boot_retry_precedes_launches():
    import server
    from ouroboros import server_restart

    lifespan = inspect.getsource(server.lifespan)
    boot = lifespan.index("finish_unconfirmed_stops(lifespan_drive_root)")
    assert boot < lifespan.index("start_inherited_import(lifespan_drive_root)")
    for later in ("auto_start_local_model", "warm_owned_daemon()", "create_host_service_app(",
                  "_reload_extensions(lifespan_drive_root", "_start_supervisor_if_needed(settings)"):
        assert boot < lifespan.index(later), later
    teardown = lifespan.split("\n    finally:\n", 1)[1]
    assert teardown.index("kill_workers(") < teardown.index("stop_owned_work(lifespan_drive_root)")
    emergency = inspect.getsource(server._emergency_process_cleanup)
    assert emergency.index("kill_workers(") < emergency.index("stop_owned_work(DATA_DIR)")
    owner = inspect.getsource(server_restart._stop_owned_work)
    assert owner.index('_stop_owned_daemon("Owner restart")') < owner.index("stop_owned_work(DATA_DIR)")
    assert "_stop_owned_local_processes" not in inspect.getsource(server)
    assert not hasattr(__import__("ouroboros.workspace_executor").workspace_executor, "_iter_process_records")


def test_set_file_is_one_document_under_the_canonical_state(tmp_path):
    from ouroboros import owned_shutdown

    data = tmp_path / "data"
    _docker_service(data)
    document = json.loads(owned_shutdown.owned_processes_path(data).read_text(encoding="utf-8"))
    assert document["schema_version"] == 1 and isinstance(document["records"], dict)
    assert pathlib.Path(owned_shutdown.owned_processes_path(data)).parent == data / "state"
