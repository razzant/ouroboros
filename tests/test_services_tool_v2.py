import hashlib
import json
import os
import pathlib
import shlex
import stat
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros.tools.registry import ToolRegistry
from ouroboros.tools.services import archive_task_service_logs, prune_service_logs


def _wait_for_service_log(drive, task_id, name, predicate, timeout_sec=15.0):
    """Wait until the service's live log satisfies ``predicate``.

    The start_service readiness contract without a stdout marker is "process
    alive == ready" — it never waits for output. These suites assert on log
    CONTENT, so they must wait for the observable condition themselves: on a
    slow CI runner the child's first write can land well after start_service
    returns (a latent race exposed when the guard path got faster), and the
    assertions below are about redaction/finalization, never about timing.
    """
    log_path = drive / "services" / task_id / name
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            if predicate(log_path.read_bytes() if log_path.exists() else b""):
                return
        except OSError:
            pass
        time.sleep(0.1)
    raise AssertionError(
        f"service log {log_path} did not satisfy the wait predicate within {timeout_sec}s"
    )


def _force_advanced_runtime(monkeypatch):
    from ouroboros import config as cfg

    cfg.reset_runtime_mode_baseline_for_tests()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.delenv(cfg.BOOT_RUNTIME_MODE_ENV_KEY, raising=False)


def _force_light_runtime(monkeypatch):
    from ouroboros import config as cfg

    cfg.reset_runtime_mode_baseline_for_tests()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.delenv(cfg.BOOT_RUNTIME_MODE_ENV_KEY, raising=False)


def test_task_scoped_service_lifecycle(tmp_path, monkeypatch):
    _force_advanced_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-1"

    start = registry.execute("start_service", {
        "name": "demo",
        "cmd": [
            sys.executable,
            "-c",
            "import time; print('READY', flush=True); time.sleep(60)",
        ],
        "readiness": {"log_contains": "READY", "timeout_sec": 3},
    })
    start_payload = json.loads(start)
    assert start_payload["state"] == "running"
    assert start_payload["ready"] is True
    assert start_payload["pid"] > 0

    logs = json.loads(registry.execute("service_logs", {"name": "demo", "tail": 200}))
    assert "READY" in logs["tail"]
    assert logs["full_log_ref"]["sha256"]

    stopped = json.loads(registry.execute("stop_service", {"name": "demo"}))
    assert stopped["state"] == "exited"
    assert stopped["log_finalization"]["deleted_live_log"] is True
    assert not (drive / "services" / "task-1" / "demo.log").exists()
    assert registry.execute("service_status", {"name": "demo"}).startswith("⚠️ SERVICE_NOT_FOUND")


def test_a_start_whose_names_earlier_log_stays_in_place_writes_a_file_of_its_own(tmp_path, monkeypatch):
    """An earlier launch's log under this name (its stop kept it, or its worker died) is never appended to: the new
    launch writes the file its own launch id names, and the earlier bytes stay as they were for retention."""
    _force_advanced_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo, drive = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-1"
    earlier = drive / "services" / "task-1" / "demo.log"
    earlier.parent.mkdir(parents=True)
    earlier.write_bytes(b"the earlier launch's own lines\n")

    started = json.loads(registry.execute("start_service", {
        "name": "demo", "cmd": [sys.executable, "-c", "import time; print('READY', flush=True); time.sleep(60)"],
        "readiness": {"log_contains": "READY", "timeout_sec": 3}}))
    try:
        own = pathlib.Path(started["log_path"]).resolve()
        assert started["ready"] is True and own.parent == earlier.parent.resolve() and own != earlier.resolve(), started
        assert own.name.startswith("demo.") and own.name.endswith(".log") and b"READY" in own.read_bytes()
    finally:
        registry.execute("stop_service", {"name": "demo"})
    assert earlier.read_bytes() == b"the earlier launch's own lines\n"


def test_service_logs_redact_secret_assignments(tmp_path, monkeypatch):
    _force_advanced_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-1"

    start = registry.execute("start_service", {
        "name": "secretlog",
        "cmd": [
            sys.executable,
            "-c",
            "print('OPENAI_API_KEY=thisisaverylongsecretvalue123456', flush=True)",
        ],
        "readiness": {"timeout_sec": 1},
    })
    assert json.loads(start)["state"] in {"running", "exited"}
    _wait_for_service_log(drive, "task-1", "secretlog.log", lambda b: b"OPENAI_API_KEY=" in b)
    logs = json.loads(registry.execute("service_logs", {"name": "secretlog", "tail": 500}))
    registry.execute("stop_service", {"name": "secretlog"})

    assert "thisisaverylongsecretvalue" not in logs["tail"]
    assert "***REDACTED***" in logs["tail"]


def test_service_logs_tail_is_capped(tmp_path, monkeypatch):
    _force_advanced_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-1"

    start = registry.execute("start_service", {
        "name": "bigtail",
        "cmd": [sys.executable, "-c", "print('x' * 120000, flush=True)"],
        "readiness": {"timeout_sec": 1},
    })
    assert json.loads(start)["state"] in {"running", "exited"}
    logs = json.loads(registry.execute("service_logs", {"name": "bigtail", "tail": 1_000_000}))
    registry.execute("stop_service", {"name": "bigtail"})

    assert len(logs["tail"]) <= 80_000


def test_service_readiness_marker_before_large_suffix_latches_until_exit(tmp_path):
    from ouroboros.tools.services import ServiceRecord, _refresh_ready

    class RunningProcess:
        pid = 4321

        @staticmethod
        def poll():
            return None

    log_path = tmp_path / "service.log"
    log_path.write_bytes(b"READY\n" + (b"x" * 25_000))
    record = ServiceRecord(
        name="burst",
        service_id="task:burst",
        task_id="task",
        cmd=["service"],
        cwd=str(tmp_path),
        log_path=log_path,
        proc=RunningProcess(),
        readiness={"log_contains": "READY"},
    )

    assert _refresh_ready(record) is True
    log_path.write_bytes(b"y" * 30_000)
    assert _refresh_ready(record) is True

    record.proc = SimpleNamespace(poll=lambda: 0, pid=4321)
    assert _refresh_ready(record) is False
    assert record.ready is False


def test_service_readiness_scan_resets_when_log_rotates(tmp_path):
    from ouroboros.tools.services import ServiceRecord, _refresh_ready

    process = SimpleNamespace(poll=lambda: None, pid=4321)
    log_path = tmp_path / "service.log"
    log_path.write_bytes(b"x" * 30_000)
    record = ServiceRecord(
        name="rotating",
        service_id="task:rotating",
        task_id="task",
        cmd=["service"],
        cwd=str(tmp_path),
        log_path=log_path,
        proc=process,
        readiness={"log_contains": "READY"},
    )
    assert _refresh_ready(record) is False

    replacement = tmp_path / "replacement.log"
    replacement.write_bytes(b"READY\n" + (b"y" * 30_000))
    replacement.replace(log_path)

    assert _refresh_ready(record) is True


def test_service_readiness_refresh_serializes_incremental_scans(tmp_path, monkeypatch):
    from ouroboros.tools import services as services_mod

    record = services_mod.ServiceRecord(
        name="concurrent",
        service_id="task:concurrent",
        task_id="task",
        cmd=["service"],
        cwd=str(tmp_path),
        log_path=tmp_path / "service.log",
        proc=SimpleNamespace(poll=lambda: None, pid=4321),
        readiness={"log_contains": "READY"},
    )
    first_scan_entered = threading.Event()
    release_first_scan = threading.Event()
    counter_lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_scan(_record, _marker):
        nonlocal active, max_active
        with counter_lock:
            active += 1
            max_active = max(max_active, active)
            first_scan_entered.set()
        release_first_scan.wait(timeout=2)
        with counter_lock:
            active -= 1
        return False

    monkeypatch.setattr(services_mod, "_readiness_marker_observed", fake_scan)
    first = threading.Thread(target=services_mod._refresh_ready, args=(record,))
    second = threading.Thread(target=services_mod._refresh_ready, args=(record,))
    first.start()
    assert first_scan_entered.wait(timeout=1)
    second.start()
    time.sleep(0.05)
    release_first_scan.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert max_active == 1
    assert not first.is_alive()
    assert not second.is_alive()


def test_host_status_race_never_reports_terminal_service_ready(tmp_path):
    from ouroboros.tools.services import ServiceRecord, _status_payload

    class RacingProcess:
        pid = 9911
        calls = 0

        def poll(self):
            self.calls += 1
            return None if self.calls == 1 else 0

    log_path = tmp_path / "race.log"
    log_path.write_text("READY\n", encoding="utf-8")
    proc = RacingProcess()
    record = ServiceRecord(
        name="race",
        service_id="task:race",
        task_id="task",
        cmd=["service"],
        cwd=str(tmp_path),
        log_path=log_path,
        proc=proc,
        readiness={"log_contains": "READY"},
    )

    payload = _status_payload(record)

    assert payload["state"] == "exited"
    assert payload["ready"] is False
    assert payload["ready_observed_at"]


def test_executor_local_status_refreshes_late_readiness_after_timeout(tmp_path):
    import ouroboros.workspace_executor as workspace_executor

    log_path = tmp_path / "late.log"
    log_path.write_bytes(b"boot\n")
    record = SimpleNamespace(
        service_id="task:late",
        name="late",
        task_id="task",
        executor=SimpleNamespace(executor_id="local", kind="local", network="host"),
        backend_pid="1234",
        backend_cwd="/workspace",
        host_cwd=tmp_path,
        cwd_root="active_workspace",
        cwd_base=str(tmp_path),
        cwd_source="active_workspace",
        skill_name="",
        cmd=["service"],
        outputs=[],
        keep_alive=False,
        backend_log_path=str(log_path),
        readiness={"log_contains": "READY"},
        started_at=time.time(),
        ready=False,
        ready_observed_at="",
        local_proc=SimpleNamespace(poll=lambda: None),
        readiness_log_offset=0,
        readiness_log_carry=b"",
        readiness_log_identity=None,
    )
    assert workspace_executor._service_payload(record)["ready"] is False
    log_path.write_bytes(b"boot\nREADY\n")
    payload = workspace_executor._service_payload(record)
    assert payload["state"] == "running"
    assert payload["ready"] is True


def test_executor_payload_repolls_local_terminal_after_readiness_refresh(tmp_path):
    import ouroboros.workspace_executor as workspace_executor

    class RacingProcess:
        def __init__(self):
            self.calls = 0

        def poll(self):
            self.calls += 1
            return None if self.calls == 1 else 0

    log_path = tmp_path / "race.log"
    log_path.write_bytes(b"READY\n")
    record = SimpleNamespace(
        service_id="task:race",
        name="race",
        task_id="task",
        executor=SimpleNamespace(executor_id="local", kind="local", network="host"),
        backend_pid="1234",
        backend_cwd="/workspace",
        host_cwd=tmp_path,
        cwd_root="active_workspace",
        cwd_base=str(tmp_path),
        cwd_source="active_workspace",
        skill_name="",
        cmd=["service"],
        outputs=[],
        keep_alive=False,
        backend_log_path=str(log_path),
        readiness={"log_contains": "READY"},
        started_at=time.time(),
        ready=False,
        ready_observed_at="",
        local_proc=RacingProcess(),
        readiness_log_offset=0,
        readiness_log_carry=b"",
        readiness_log_identity=None,
    )

    payload = workspace_executor._service_payload(record)

    assert payload["state"] == "exited"
    assert payload["ready"] is False
    assert payload["ready_observed_at"]


def test_executor_payload_repolls_docker_terminal_after_readiness_refresh(monkeypatch, tmp_path):
    import ouroboros.workspace_executor as workspace_executor

    record = SimpleNamespace(
        service_id="task:docker-race",
        name="docker-race",
        task_id="task",
        executor=SimpleNamespace(executor_id="docker", kind="docker_exec", network="none", container_name="svc"),
        backend_pid="1234",
        backend_cwd="/workspace",
        host_cwd=tmp_path,
        cwd_root="active_workspace",
        cwd_base=str(tmp_path),
        cwd_source="active_workspace",
        skill_name="",
        cmd=["service"],
        outputs=[],
        keep_alive=False,
        backend_log_path="/tmp/service.log",
        readiness={"log_contains": "READY"},
        started_at=time.time(),
        ready=False,
        ready_observed_at="",
        readiness_log_offset=0,
        readiness_log_carry=b"",
        readiness_log_identity=(7, 42),
    )
    state_calls = 0

    def fake_run(cmd, **kwargs):
        nonlocal state_calls
        shell = str(cmd[-1])
        if "stat -c" in shell:
            return subprocess.CompletedProcess(cmd, 0, stdout=b"7:42:5\nREADY", stderr=b"")
        state_calls += 1
        result = "running\n" if state_calls == 1 else "exited\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=result, stderr="")

    monkeypatch.setattr(workspace_executor.subprocess, "run", fake_run)
    payload = workspace_executor._service_payload(record)

    assert payload["state"] == "exited"
    assert payload["ready"] is False
    assert state_calls == 2


def test_executor_docker_readiness_uses_incremental_cursor_and_carry(monkeypatch):
    import ouroboros.workspace_executor as workspace_executor

    record = SimpleNamespace(
        executor=SimpleNamespace(kind="docker_exec", container_name="svc-container"),
        backend_log_path="/tmp/service.log",
        readiness_log_offset=0,
        readiness_log_carry=b"",
        readiness_log_identity=(7, 42),
    )
    calls = []
    remote = [b"7:42:3\nabc", b"7:42:10\ndefREADY\n"]

    def fake_run(cmd, **kwargs):
        calls.append(cmd[-1])
        item = remote.pop(0)
        return subprocess.CompletedProcess(cmd, 0, stdout=item, stderr=b"")

    monkeypatch.setattr(workspace_executor.subprocess, "run", fake_run)
    assert workspace_executor._executor_readiness_marker_observed(record, "READY") is False
    assert workspace_executor._executor_readiness_marker_observed(record, "READY") is True
    assert all("grep" not in shell for shell in calls)
    assert "tail -c +1" in calls[0]
    assert "tail -c +4" in calls[1]


def test_executor_docker_readiness_bounds_each_remote_chunk_and_keeps_cursor(monkeypatch):
    import ouroboros.workspace_executor as workspace_executor

    record = SimpleNamespace(
        executor=SimpleNamespace(kind="docker_exec", container_name="svc-container"),
        backend_log_path="/tmp/service.log",
        readiness_log_offset=0,
        readiness_log_carry=b"",
        readiness_log_identity=(7, 42),
    )
    calls = []
    remote = [b"7:42:100000\n" + (b"x" * 65536), b"7:42:100000\n"]

    def fake_run(cmd, **kwargs):
        calls.append(cmd[-1])
        return subprocess.CompletedProcess(cmd, 0, stdout=remote.pop(0), stderr=b"")

    monkeypatch.setattr(workspace_executor.subprocess, "run", fake_run)
    assert workspace_executor._executor_readiness_marker_observed(record, "READY") is False
    assert record.readiness_log_offset == 65536
    assert workspace_executor._executor_readiness_marker_observed(record, "READY") is False
    assert record.readiness_log_offset == 65536
    assert "head -c 65536" in calls[0]
    assert "tail -c +65537" in calls[1]


def test_service_readiness_is_independent_of_bounded_log_tail(tmp_path, monkeypatch):
    _force_advanced_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-burst-ready"

    started = json.loads(registry.execute("start_service", {
        "name": "burst",
        "cmd": [
            sys.executable,
            "-c",
            "import os,time; os.write(1, b'READY\\n' + b'x' * 25000); time.sleep(30)",
        ],
        "readiness": {"log_contains": "READY", "timeout_sec": 3},
    }))
    try:
        status = json.loads(registry.execute("service_status", {"name": "burst"}))
        logs = json.loads(registry.execute("service_logs", {"name": "burst", "tail": 1000}))
    finally:
        registry.execute("stop_service", {"name": "burst"})

    assert started["ready"] is True
    assert status["ready"] is True
    assert started["ready_observed_at"]
    assert "READY" not in logs["tail"]
    assert "x" in logs["tail"]


def _private_copy(drive, event):
    """Where a leftover log's receipt says its exact bytes are kept: ``observability/service_logs/<sha256>.log``."""
    return drive / "observability" / "service_logs" / f"{event['private_log']['sha256']}.log"


def test_service_log_retention_prunes_stale_directories(tmp_path):
    drive = tmp_path / "data"
    stale = drive / "services" / "task-old"
    stale.mkdir(parents=True)
    log = stale / "demo.log"
    log.write_text("old", encoding="utf-8")
    now = 1_000_000.0
    old = now - 30 * 86400
    os.utime(stale, (old, old))
    os.utime(log, (old, old))

    report = prune_service_logs(drive, retention_days=14, now=now)

    assert report["archived_files"] == 1
    assert report["deleted_files"] == 1
    assert report["deleted_dirs"] == 1
    assert not stale.exists()
    events = [
        json.loads(line)
        for line in (drive / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    event = [item for item in events if item.get("type") == "service_log_pruned"][-1]
    assert event["task_id"] == "task-old"
    assert event["name"] == "demo"
    assert _private_copy(drive, event).read_bytes() == b"old"


def test_archive_task_service_logs_finalizes_forced_worker_leftovers(tmp_path):
    """A finished task's leftover log, which no stop can mask any more, leaves its folder only as its exact private
    copy: no history blob holds the bytes, and a child drive's copy lands in the canonical root that outlives it."""
    from ouroboros.observability import posix_private_modes_supported

    drive = tmp_path / "data"
    task_dir = drive / "services" / "task-forced"
    task_dir.mkdir(parents=True)
    log = task_dir / "devserver.log"
    payload = b"READY\n\xff\xfe not text\n"
    log.write_bytes(payload)

    report = archive_task_service_logs(drive, "task-forced")

    assert report["archived_files"] == 1
    assert report["deleted_files"] == 1
    assert report["deleted_dirs"] == 1
    assert not task_dir.exists()
    events = [
        json.loads(line)
        for line in (drive / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    event = [item for item in events if item.get("type") == "service_log_archived"][-1]
    assert event["task_id"] == "task-forced"
    assert event["name"] == "devserver"
    assert event["private_log"] == {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
    assert "full_log_ref" not in event and not (drive / "observability" / "blobs").exists()
    private = _private_copy(drive, event)
    assert private.read_bytes() == payload
    if posix_private_modes_supported():
        assert stat.S_IMODE(private.stat().st_mode) == 0o600
        assert stat.S_IMODE(private.parent.stat().st_mode) == 0o700

    child_drive = drive / "task_drives" / "task-child"
    child_task_dir = child_drive / "services" / "task-child"
    child_task_dir.mkdir(parents=True)
    (child_task_dir / "devserver.log").write_bytes(b"READY\n")

    child_report = archive_task_service_logs(
        drive,
        "task-child",
        {"child_drive_root": str(child_drive)},
    )

    assert child_report["archived_files"] == 1
    assert child_report["deleted_dirs"] == 1
    assert not child_task_dir.exists()
    events = [json.loads(line) for line in (drive / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    (child_event,) = [item for item in events
                      if item.get("type") == "service_log_archived" and item.get("task_id") == "task-child"]
    assert _private_copy(drive, child_event).read_bytes() == b"READY\n"
    assert not (child_drive / "observability" / "service_logs").exists()


@pytest.mark.parametrize("failure", ["copy_refused", "copy_differs"])
def test_a_leftover_log_whose_private_copy_is_not_confirmed_stays_where_it_is(tmp_path, monkeypatch, failure):
    """Only a copy that reads back as the source lets the live log go: a refused copy, or one whose bytes differ,
    leaves the log and its folder as they were, the failure reported and no receipt written."""
    from ouroboros import artifacts

    drive = tmp_path / "data"
    task_dir = drive / "services" / "task-held"
    task_dir.mkdir(parents=True)
    log = task_dir / "devserver.log"
    log.write_bytes(b"READY\n")
    real_copy = artifacts.copy_artifact_file

    def copy_artifact_file(source, destination, **kwargs):
        if failure == "copy_refused":
            raise OSError("no space left for the copy")
        measured = real_copy(source, destination, **kwargs)
        pathlib.Path(destination).write_bytes(b"READY\nnot the bytes it was copied from\n")
        return measured

    monkeypatch.setattr(artifacts, "copy_artifact_file", copy_artifact_file)

    report = archive_task_service_logs(drive, "task-held")

    assert (report["archived_files"], report["deleted_files"], report["deleted_dirs"]) == (0, 0, 0)
    assert report["retained_files"] == 1 and report["errors"], report
    assert log.read_bytes() == b"READY\n"
    events = drive / "logs" / "events.jsonl"
    assert not events.exists() or "service_log_archived" not in events.read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="kept-service custody is read by the POSIX launch census")
def test_archive_task_service_logs_keeps_a_kept_services_live_log_until_it_is_stopped(tmp_path, monkeypatch):
    """A terminal task's archive (worker death, cancel, reaper) must not unlink a log its kept service still writes;
    once that launch is stopped the same archive takes it, with nothing it wrote lost."""
    from ouroboros.launch_custody import kept_services, stop_kept_service

    _force_advanced_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo, drive = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-kept"
    script = "import sys, time\nprint('first', flush=True)\nwhile True:\n    time.sleep(0.05)\n    print('tick', flush=True)\n"
    started = json.loads(registry.execute("start_service", {
        "name": "writer", "cmd": [sys.executable, "-c", script], "keep_alive": True,
        "readiness": {"log_contains": "first", "timeout_sec": 20}}))
    log = pathlib.Path(started["log_path"])
    try:
        assert started["state"] == "running", started
        report = archive_task_service_logs(drive, "task-kept", {"id": "task-kept"})
        assert report["retained_files"] == 1 and report["archived_files"] == 0 and report["deleted_files"] == 0
        size = log.stat().st_size
        _wait_for_service_log(drive, "task-kept", log.name, lambda data: len(data) > size)
        assert log.read_text(encoding="utf-8").startswith("first\n"), "the same live file keeps growing"

        launch_id = kept_services(drive)["services"][0]["launch_id"]
        assert stop_kept_service(drive, launch_id, reason="test over", actor="owner:test")["outcome"] == "stopped"
        report = archive_task_service_logs(drive, "task-kept", {"id": "task-kept"})
        assert report["archived_files"] == 1 and report["deleted_files"] == 1 and not log.exists()
        events = [json.loads(line) for line in (drive / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        archived = [item for item in events if item.get("type") == "service_log_archived"][-1]
        assert archived["task_id"] == "task-kept" and _private_copy(drive, archived).read_bytes().startswith(b"first\n")
    finally:
        registry.execute("stop_service", {"name": "writer"})


def test_stop_service_retains_live_log_when_full_blob_omitted(tmp_path, monkeypatch):
    from ouroboros.tools import services as services_mod

    _force_advanced_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setattr(services_mod, "_MAX_SERVICE_LOG_BLOB_BYTES", 10)
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-oversize"

    registry.execute("start_service", {
        "name": "oversize",
        "cmd": [sys.executable, "-c", "print('x' * 100, flush=True)"],
        "readiness": {"timeout_sec": 1},
    })
    _wait_for_service_log(drive, "task-oversize", "oversize.log", lambda b: len(b) > 100)
    stopped = json.loads(registry.execute("stop_service", {"name": "oversize"}))

    finalization = stopped["log_finalization"]
    assert finalization["deleted_live_log"] is False
    assert finalization["retained_live_log_path"].endswith("oversize.log")
    assert (drive / "services" / "task-oversize" / "oversize.log").exists()


def test_service_log_finalization_checks_size_before_full_read(tmp_path, monkeypatch):
    from ouroboros.tools import services as services_mod

    drive = tmp_path / "data"
    log_path = drive / "services" / "task-big" / "big.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("x" * 100, encoding="utf-8")
    monkeypatch.setattr(services_mod, "_MAX_SERVICE_LOG_BLOB_BYTES", 10)
    original_read_text = pathlib.Path.read_text

    def guarded_read_text(self, *args, **kwargs):
        if self == log_path:
            raise AssertionError("oversized service log should not be fully read")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", guarded_read_text)

    result = services_mod._finalize_service_log_for_drive(drive, SimpleNamespace(log_path=log_path))

    assert result["deleted_live_log"] is False
    assert result["tail"]
    assert "full_log_omitted" in result
    assert log_path.exists()


def test_kill_all_services_records_shutdown_cleanup_event(tmp_path, monkeypatch):
    from ouroboros.tools.services import kill_all_services

    _force_advanced_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-shutdown"

    registry.execute("start_service", {
        "name": "shutdown",
        "cmd": [sys.executable, "-c", "import time; print('READY', flush=True); time.sleep(60)"],
        "readiness": {"log_contains": "READY", "timeout_sec": 3},
    })
    stopped = kill_all_services(drive)

    assert stopped
    events = [
        json.loads(line)
        for line in (drive / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    cleanup = [event for event in events if event.get("type") == "services_shutdown_cleanup"]
    assert cleanup
    service = cleanup[-1]["services"][0]
    assert service["name"] == "shutdown"
    assert service["task_id"] == "task-shutdown"
    assert service["log_finalization"]["full_log_ref"]["sha256"]
    assert "tail" not in service["log_finalization"]


def test_service_log_retention_uses_log_mtime_and_skips_symlinks(tmp_path):
    drive = tmp_path / "data"
    services_root = drive / "services"
    stale_dir = services_root / "task-old-dir"
    stale_dir.mkdir(parents=True)
    fresh_log = stale_dir / "fresh.log"
    fresh_log.write_text("fresh", encoding="utf-8")
    stale_log = stale_dir / "stale.log"
    stale_log.write_text("old", encoding="utf-8")
    other_file = stale_dir / "notes.txt"
    other_file.write_text("keep", encoding="utf-8")
    now = 1_000_000.0
    old = now - 30 * 86400
    os.utime(stale_dir, (old, old))
    os.utime(stale_log, (old, old))
    os.utime(fresh_log, (now, now))
    os.utime(other_file, (old, old))

    target = tmp_path / "outside"
    target.mkdir()
    (target / "evil.log").write_text("keep", encoding="utf-8")
    symlink_dir = services_root / "linked"
    try:
        symlink_dir.symlink_to(target, target_is_directory=True)
    except OSError:
        symlink_dir = None

    report = prune_service_logs(drive, retention_days=14, now=now)

    assert report["archived_files"] == 1
    assert report["deleted_files"] == 1
    assert stale_log.exists() is False
    assert fresh_log.exists()
    assert other_file.exists()
    assert stale_dir.exists()
    if symlink_dir is not None:
        assert symlink_dir.exists()
        assert (target / "evil.log").exists()


def test_service_log_retention_keeps_an_oversized_stale_log_privately_and_frees_its_folder(tmp_path, monkeypatch):
    """A log over the history blob cap no longer holds its task's folder: it leaves whole as its private copy."""
    from ouroboros.tools import services as services_mod

    drive = tmp_path / "data"
    stale = drive / "services" / "task-big"
    stale.mkdir(parents=True)
    log = stale / "big.log"
    log.write_text("x" * 100, encoding="utf-8")
    monkeypatch.setattr(services_mod, "_MAX_SERVICE_LOG_BLOB_BYTES", 10)
    now = 1_000_000.0
    old = now - 30 * 86400
    os.utime(log, (old, old))

    report = prune_service_logs(drive, retention_days=14, now=now)

    assert (report["archived_files"], report["deleted_files"], report["deleted_dirs"], report["retained_files"]) == (
        1, 1, 1, 0), report
    assert not stale.exists()
    events = [json.loads(line) for line in (drive / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    (event,) = [item for item in events if item.get("type") == "service_log_pruned"]
    assert _private_copy(drive, event).read_bytes() == b"x" * 100


def test_light_start_service_blocks_repo_default_but_allows_task_drive_cwd(tmp_path, monkeypatch):
    _force_light_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    invalidations = []
    monkeypatch.setattr(
        "ouroboros.tools.commit_gate._invalidate_advisory",
        lambda *args, **kwargs: invalidations.append({"args": args, "kwargs": kwargs}),
    )
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-light-service"

    blocked = registry.execute("start_service", {
        "name": "repo_default",
        "cmd": [sys.executable, "-c", "print('READY', flush=True)"],
        "readiness": {"timeout_sec": 1},
    })
    assert "LIGHT_MODE_BLOCKED" in blocked

    task_drive = drive / "task_drives" / "task-light-service"
    artifact_store = drive / "task_results" / "artifacts" / "task-light-service"
    assert not task_drive.exists()
    assert not artifact_store.exists()
    for name, cwd in (("task_drive_service", task_drive), ("artifact_store_service", artifact_store)):
        started = registry.execute("start_service", {
            "name": name,
            "cmd": [sys.executable, "-c", "import time; print('READY', flush=True); time.sleep(60)"],
            "cwd": str(cwd),
            "readiness": {"log_contains": "READY", "timeout_sec": 3},
        })

        payload = json.loads(started)
        assert payload["cwd"] == str(cwd)
        assert payload["state"] == "running"
        assert payload["ready"] is True
        assert cwd.is_dir()
        assert "LIGHT_MODE_BLOCKED" not in started
        registry.execute("stop_service", {"name": name})
    assert invalidations == []


def test_user_files_service_without_outputs_reports_audit_gap(tmp_path, monkeypatch):
    _force_light_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    home = tmp_path / "home"
    desktop = home / "Desktop"
    desktop.mkdir(parents=True)
    monkeypatch.setattr(pathlib.Path, "home", staticmethod(lambda: home))
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-user-service"

    started = registry.execute("start_service", {
        "name": "user_file_service",
        "cmd": [sys.executable, "-c", "import time; print('READY', flush=True); time.sleep(60)"],
        "cwd": str(desktop),
        "readiness": {"log_contains": "READY", "timeout_sec": 3},
    })
    start_payload = json.loads(started)
    assert start_payload["cwd_root"] == "user_files"

    stopped = json.loads(registry.execute("stop_service", {"name": "user_file_service"}))

    assert "ARTIFACT_AUDIT_GAP" in stopped["artifact_audit_gap"]
    # The same advice as the shell tools' notes: a fact and both ways to register, no order to rerun.
    from ouroboros.tools.shell_audit import REGISTER_OUTPUT_ADVICE

    assert stopped["artifact_audit_gap"].endswith(REGISTER_OUTPUT_ADVICE) and "rerun" not in stopped["artifact_audit_gap"]
    assert stopped["artifact_output_failed"] is False


@pytest.mark.parametrize("relative", [False, True])
def test_light_service_python_body_executes_once_with_bound_environment(tmp_path, monkeypatch, relative):
    from ouroboros.tools import services

    _force_light_runtime(monkeypatch)
    repo, drive, home = (tmp_path / name for name in ("repo", "data", "home"))
    for path in (repo, drive, home):
        path.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-service-runtime-data"
    upload = drive / "uploads" / "report.html"
    selected_path = "../../uploads/report.html" if relative else str(upload)
    original_env = services._service_env
    child_env = {**original_env(), "HOME": str(home), "USERPROFILE": str(home),
                 "OUROBOROS_DATA_DIR": str(drive), "OUROBOROS_SETTINGS_PATH": str(drive / "settings.json")}
    monkeypatch.setattr(services, "_service_env", lambda: dict(child_env))
    decisions = []

    def supervisor(name, args, *_a, **_kw):
        if name == "start_service":
            decisions.append((name, args["cmd"]))
        return True, ""

    monkeypatch.setattr("ouroboros.safety.check_safety", supervisor)
    body = ("import os, time\nfrom pathlib import Path\n"
            f"p = Path({selected_path!r})\np.parent.mkdir(parents=True, exist_ok=True)\n"
            "with p.open('a', newline='') as output: output.write('once\\n')\n"
            + "\n".join(f"assert os.environ[{key!r}] == {child_env[key]!r}" for key in
                        ("HOME", "USERPROFILE", "OUROBOROS_DATA_DIR", "OUROBOROS_SETTINGS_PATH"))
            + "\nprint('READY', flush=True)\ntime.sleep(60)\n")
    command = [sys.executable, "-c", body]
    try:
        started = registry.execute("start_service", {
            "name": "python_writer", "cmd": command, "cwd": str(registry._ctx.task_drive_root()),
            "readiness": {"log_contains": "READY", "timeout_sec": 3},
        })
        payload = json.loads(started)
        assert payload["ready"] and payload["pid"] > 0, started
        assert decisions == [("start_service", command)]
        assert upload.read_bytes() == b"once\n"
    finally:
        services._stop_service(registry._ctx, name="python_writer")
    assert registry.execute("service_status", {"name": "python_writer"}).startswith("⚠️ SERVICE_NOT_FOUND")


def test_light_start_service_blocks_explicit_runtime_data_upload_redirect(tmp_path, monkeypatch):
    _force_light_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-service-runtime-data"
    upload = drive / "uploads" / "env-report.html"

    result = registry.execute("start_service", {
        "name": "runtime_data_env_writer",
        "cmd": ["sh", "-c", f"mkdir -p {shlex.quote(str(upload.parent))} && echo bad > {shlex.quote(str(upload))}"],
        "cwd": str(registry._ctx.task_drive_root()),
        "readiness": {"timeout_sec": 1},
    })

    assert "BLOCKED" in result, result
    assert str(upload.parent) in result
    assert not upload.exists()
    assert registry.execute("service_status", {"name": "runtime_data_env_writer"}).startswith("⚠️ SERVICE_NOT_FOUND")


def test_service_outputs_register_artifacts_on_stop(tmp_path, monkeypatch):
    _force_light_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-service-output"
    task_drive = drive / "task_drives" / "task-service-output"
    task_drive.mkdir(parents=True)

    start = registry.execute("start_service", {
        "name": "artifact_service",
        "cmd": [
            sys.executable,
            "-c",
            "from pathlib import Path; Path('service.html').write_text('<h1>ok</h1>'); print('READY', flush=True)",
        ],
        "cwd": str(task_drive),
        "outputs": ["service.html"],
        "readiness": {"timeout_sec": 1},
    })
    assert "LIGHT_MODE_BLOCKED" not in start
    _wait_for_service_log(drive, "task-service-output", "artifact_service.log", lambda b: b"READY" in b)

    stopped = json.loads(registry.execute("stop_service", {"name": "artifact_service"}))

    assert "ARTIFACT_OUTPUTS" in stopped["artifact_outputs"]
    artifact_path = drive / "task_results" / "artifacts" / "task-service-output" / "service.html"
    assert artifact_path.read_text(encoding="utf-8") == "<h1>ok</h1>"


def test_service_output_left_unchanged_is_captured_at_stop_then_reused(tmp_path, monkeypatch):
    from ouroboros.tools.services import stop_task_services

    _force_light_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-service-unchanged"
    task_drive = drive / "task_drives" / "task-service-unchanged"
    task_drive.mkdir(parents=True)
    (task_drive / "report.md").write_text("# service report\n", encoding="utf-8")

    def start(name):
        started = registry.execute("start_service", {
            "name": name,
            "cmd": [sys.executable, "-c", "print('READY', flush=True)"],
            "cwd": str(task_drive),
            "outputs": ["report.md"],
            "readiness": {"timeout_sec": 1},
        })
        assert "BLOCKED" not in started, started
        _wait_for_service_log(drive, "task-service-unchanged", f"{name}.log", lambda b: b"READY" in b)

    start("verifier")
    stopped = registry.execute_result("stop_service", {"name": "verifier"})
    # Declared and left unchanged by the service, the existing report is still its deliverable.
    assert stopped.code == "OK" and stopped.meta.get("artifact_registered") is True, stopped.text
    assert "ARTIFACT_OUTPUTS" in json.loads(stopped.text)["artifact_outputs"]
    assert registry.execute("read_file", {"root": "artifact_store", "path": "report.md"}).count("# service report") == 1
    start("again")
    finalized = stop_task_services(registry._ctx)  # the ordinary task-end stop finds that capture current
    assert [item["lifecycle"] for item in finalized] == ["stopped"]
    assert finalized[0]["artifact_outputs"].startswith("ARTIFACT_OUTPUT_NOTE")
    assert "unchanged output (cosmetic): report.md" in finalized[0]["artifact_outputs"]


def test_stop_task_services_preserves_output_finalization_failure(tmp_path, monkeypatch):
    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.tools.services import stop_task_services

    _force_light_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-service-missing-output"
    task_drive = drive / "task_drives" / "task-service-missing-output"
    task_drive.mkdir(parents=True)

    registry.execute("start_service", {
        "name": "missing_output_service",
        "cmd": [sys.executable, "-c", "print('READY', flush=True)"],
        "cwd": str(task_drive),
        "outputs": ["missing.html"],
        "readiness": {"timeout_sec": 1},
    })

    stopped = stop_task_services(registry._ctx)
    outcome = derive_loop_outcome(
        "Done",
        {"rounds": 2},
        {"tool_calls": [], "verification_events": [{"kind": "services_stopped", "services": stopped}]},
    )

    assert stopped[0]["artifact_output_failed"] is True
    assert "ARTIFACT_OUTPUT_ERROR" in stopped[0]["artifact_outputs"]
    assert outcome["outcome_axes"]["execution"]["status"] == "degraded"
    assert outcome["failure"]["kind"] == "verification"


def test_task_teardown_maps_the_structured_stop_outcome_to_lifecycle(tmp_path, monkeypatch):
    """Teardown consumes the stop's typed outcome, never its rendered text: a payload whose
    own facts contain the old text-split hazard (``\\n\\n{``) still arrives whole."""
    from ouroboros.tools import services
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.services import (
        STOP_CONFIRMED,
        STOP_NOT_FOUND,
        STOP_UNCONFIRMED,
        ServiceStop,
        stop_task_services,
    )

    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path, task_id="task")
    monkeypatch.setattr(services, "_SERVICES", {"task:demo": SimpleNamespace(task_id="task", keep_alive=False)})
    monkeypatch.setattr(services, "executor_stop_task_services", lambda _ctx: [])  # the named stop is the subject
    answers = iter([
        ServiceStop({"service_id": "task:demo", "stop_error": "denied\n\n{"}, STOP_UNCONFIRMED),
        ServiceStop({"service_id": "task:demo", "artifact_output_failed": True}, STOP_CONFIRMED, True),
        ServiceStop(None, STOP_NOT_FOUND),
    ])
    monkeypatch.setattr(services, "_stop_service_launch", lambda _ctx, _name: next(answers))

    [pending] = stop_task_services(ctx)
    assert pending["lifecycle"] == "cleanup_pending" and pending["stop_error"] == "denied\n\n{"
    [stopped] = stop_task_services(ctx)
    assert stopped["lifecycle"] == "stopped" and stopped["artifact_output_failed"] is True
    assert stop_task_services(ctx) == [], "a launch another stop retired owns its own result"


def start_demo_report(registry, marker, *, missing=False):
    """CI helper: build one real synthetic service in the caller's task drive."""
    task_drive = registry._ctx.task_drive_root()
    task_drive.mkdir(parents=True, exist_ok=True)
    return json.loads(registry.execute("start_service", {
        "name": "demo",
        "cmd": [sys.executable, "-c", "from pathlib import Path; import sys, time; "
                "Path('report.txt').write_bytes(sys.argv[1].encode()); print('READY', flush=True); time.sleep(60)", marker],
        "cwd": str(task_drive),
        "outputs": ["report.txt", "not-created.txt"] if missing else ["report.txt"],
        "readiness": {"log_contains": "READY", "timeout_sec": 10},
    }))


def stop_demo_report(registry):
    """Use the ordinary Stop tool. The existing host emitter writes its real checkpoint."""
    result = registry.execute("stop_service", {"name": "demo"})
    return json.loads(result.partition("\n\n")[2] if result.startswith("⚠️ ") else result)["service_stop"]


def test_two_demo_launches_keep_exact_stop_results(tmp_path, monkeypatch):
    from starlette.routing import Route

    from ouroboros import artifacts
    from ouroboros.gateway import task_archive
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.gateway.logs import api_logs_tail
    from ouroboros.task_results import write_task_result
    from tests._task_file_http import task_file_client

    if not task_archive.CONFINED:
        pytest.skip("needs the platform's confined task file reader")
    _force_light_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo, drive, task_id = tmp_path / "repo", tmp_path / "data", "task-two-demo-runs"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = task_id
    registry._ctx.current_chat_id = 1
    registry._ctx._execution_trace = {}
    with task_file_client(drive) as client:
        client.app.router.routes.extend([
            Route("/api/chat/history", make_chat_history_endpoint(drive)),
            Route("/api/logs/{name}", api_logs_tail),
        ])
        first = start_demo_report(registry, "run-a")
        assert first["ready"], first
        a = stop_demo_report(registry)
        [a_output] = a["outputs"]
        [a_file] = a_output["files"]
        assert a["launch_id"] == first["launch_id"] and a_output["state"] == "saved"
        assert client.get(a_file["download_url"]).content == b"run-a"  # before any second launch
        second = start_demo_report(registry, "run-b", missing=True)
        assert second["ready"] and second["launch_id"] != a["launch_id"], second
        try:
            assert client.get(a_file["download_url"]).content == b"run-a"  # reading a receipt is not control
            assert json.loads(registry.execute("service_status", {"name": "demo"}))["state"] == "running"
        finally:
            b = stop_demo_report(registry)
        saved, failed = b["outputs"]
        [b_file] = saved["files"]
        assert b["launch_id"] == second["launch_id"] and saved["state"] == "saved"
        assert failed == {"name": "not-created.txt", "state": "failed", "files": []}
        assert a_file["name"] == b_file["name"] == "report.txt" and a_file["sha256"] != b_file["sha256"]
        for receipt, file, marker in ((a, a_file, "run-a"), (b, b_file, "run-b")):
            assert receipt["process"]["state"] == "stopped" and receipt["journal"]["state"] == "saved"
            assert marker in registry.execute("read_file", {"root": "artifact_store", "path": file["path"]})
        rows = artifacts.collect_task_artifact_records(drive, task_id)
        write_task_result(drive, task_id, "completed", result="done", artifacts=rows, artifact_status="ready")
        for file, marker in ((a_file, "run-a"), (b_file, "run-b")):
            got = client.get(file["download_url"])
            assert got.status_code == 200 and got.content == marker.encode()
            assert got.headers["x-ouroboros-artifact-sha256"] == file["sha256"]
        assert client.get(f"/api/tasks/{task_id}/artifacts/report.txt").content == b"run-b"

        for url, field in (("/api/logs/events?view=compact", "entries"), ("/api/chat/history", "messages")):
            restored = client.get(url)
            assert restored.status_code == 200, restored.text
            assert [row["service_stop"] for row in restored.json()[field]
                    if row.get("checkpoint_kind") == "service_stop"] == [a, b]
    assert registry._ctx._execution_trace["verification_events"] == [
        {"kind": "service_stop", "service_stop": a}, {"kind": "service_stop", "service_stop": b},
    ]


def _start_sleeper(registry, name, **extra_args):
    payload = json.loads(registry.execute("start_service", {
        "name": name,
        "cmd": [
            sys.executable,
            "-c",
            "import time; print('READY', flush=True); time.sleep(60)",
        ],
        "readiness": {"log_contains": "READY", "timeout_sec": 3},
        **extra_args,
    }))
    assert payload["state"] == "running"
    return payload


def test_keep_alive_service_survives_task_teardown(tmp_path, monkeypatch):
    from ouroboros.platform_layer import pid_is_alive
    from ouroboros.tools.services import kill_all_services, stop_task_services

    _force_advanced_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-keep"

    kept_payload = _start_sleeper(registry, "devserver", keep_alive=True)
    assert kept_payload["keep_alive"] is True
    _start_sleeper(registry, "scratch")

    finalized = stop_task_services(registry._ctx)
    by_name = {item["name"]: item for item in finalized}
    assert by_name["devserver"]["lifecycle"] == "kept"
    assert by_name["devserver"]["state"] == "running"
    assert by_name["scratch"]["lifecycle"] == "stopped"
    assert pid_is_alive(kept_payload["pid"]) is True

    # Custody ledger records the survivor as session-scoped.
    ledger = (drive / "state" / "process_ledger.jsonl").read_text(encoding="utf-8")
    entries = [json.loads(line) for line in ledger.splitlines() if line.strip()]
    kept_entries = [e for e in entries if e.get("pid") == kept_payload["pid"]]
    assert kept_entries and kept_entries[-1]["scope"] == "session"

    # Graceful shutdown leaves it running; panic-style cleanup kills it.
    assert kill_all_services(drive, wait=False, include_keep_alive=False) == []
    assert pid_is_alive(kept_payload["pid"]) is True
    killed = kill_all_services(drive, wait=True)
    assert any(item["name"] == "devserver" for item in killed)


def test_task_level_service_teardown_keep(tmp_path, monkeypatch):
    from ouroboros.tools.services import kill_all_services, stop_task_services

    _force_advanced_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-keep-all"
    registry._ctx.task_metadata = {"service_teardown": "keep"}

    payload = _start_sleeper(registry, "plain")
    assert payload["keep_alive"] is True  # task-level keep marks every service

    finalized = stop_task_services(registry._ctx)
    assert [item["lifecycle"] for item in finalized] == ["kept"]
    kill_all_services(drive, wait=True)


def test_default_teardown_still_stops_services(tmp_path, monkeypatch):
    from ouroboros.platform_layer import pid_is_alive
    from ouroboros.tools.services import stop_task_services

    _force_advanced_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-default"

    payload = _start_sleeper(registry, "ephemeral")
    assert payload["keep_alive"] is False

    finalized = stop_task_services(registry._ctx)
    assert [item["lifecycle"] for item in finalized] == ["stopped"]
    assert pid_is_alive(payload["pid"]) is False


def test_confirmed_stop_survives_log_capture_failure_without_leaking_or_repeating_stop(tmp_path, monkeypatch):
    import gzip
    from ouroboros.tools import services as services_mod
    from ouroboros.tools.registry import ToolContext

    repo, drive = tmp_path / "repo", tmp_path / "data"
    key, secret = "task-log-fault:demo", "synthetic-service-secret-701"
    cwd, log_path = drive / "task_drives" / "task-log-fault", drive / "services" / "task-log-fault" / "demo.log"
    repo.mkdir()
    cwd.mkdir(parents=True)
    log_path.parent.mkdir(parents=True)
    ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id="task-log-fault")
    stops: list[tuple[str, ...]] = []

    def fake_end_held(held, **_kwargs):
        item = held[0]
        stops.append(tuple(item.get("markers") or ()))
        item["proc"].returncode = 0
        return {item["key"]: {"birth": "b1", "known": [], "facts": {"exit_confirmed": True, "ended": [item["proc"].pid]}}}

    def seed(pid: int, marker: str):
        proc = SimpleNamespace(pid=pid, returncode=None)
        proc.poll = lambda: proc.returncode
        record = services_mod.ServiceRecord(
            name="demo", service_id=key, task_id=ctx.task_id, cmd=["svc"], cwd=str(cwd),
            cwd_root="task_drive", cwd_base=str(cwd), cwd_source="task_drive", log_path=log_path,
            proc=proc, outputs=["report.txt"], before_outputs={"report.txt": (False, 0, "")},
            secret_values=(secret,), scope_marker=marker,
        )
        monkeypatch.setitem(services_mod._SERVICES, key, record)
        return record

    monkeypatch.setattr("ouroboros.process_scope.end_held_processes", fake_end_held)
    initial_bytes = f"selected={secret}\nREADY launch-1\n".encode("utf-8")
    log_path.write_bytes(initial_bytes)
    (cwd / "report.txt").write_text("ok-1", encoding="utf-8")
    record_1 = seed(4101, "OURO_LAUNCH_SCOPE_1")
    real_stat, real_read_text = pathlib.Path.stat, pathlib.Path.read_text

    with monkeypatch.context() as fault:
        fault.setattr(pathlib.Path, "stat", lambda self, *a, **k: (
            (_ for _ in ()).throw(OSError(f"raw-exc-detail {secret}")) if self == log_path else real_stat(self, *a, **k)))
        [stopped] = services_mod.stop_task_services(ctx)
    dumped = json.dumps(stopped)
    assert stops == [("OURO_LAUNCH_SCOPE_1",)]
    assert stopped["lifecycle"] == "stopped" and stopped["process_cleanup"]["exit_confirmed"] is True
    assert stopped["log_finalization"] == {"deleted_live_log": False, "full_log_ref": {}, "tail": "",
                                           "errors": ["capture: OSError"], "retained_live_log_path": str(log_path)}
    assert secret not in dumped and "raw-exc-detail" not in dumped
    assert log_path.read_bytes() == initial_bytes
    assert (drive / "task_results" / "artifacts" / ctx.task_id / "report.txt").read_text(encoding="utf-8") == "ok-1"

    with monkeypatch.context() as read_fault:
        read_fault.setattr(pathlib.Path, "read_text", lambda self, *a, **k: (
            (_ for _ in ()).throw(PermissionError(f"raw-exc-detail {secret}")) if self == log_path else real_read_text(self, *a, **k)))
        mid_fail = services_mod._finalize_service_log_for_drive(drive, record_1)
    assert mid_fail == {"deleted_live_log": False, "full_log_ref": {}, "tail": "",
                        "errors": ["capture: PermissionError"], "retained_live_log_path": str(log_path)}
    assert log_path.read_bytes() == initial_bytes

    with monkeypatch.context() as store_fault:
        store_fault.setattr(services_mod, "write_blob", lambda *_a, **_k: (_ for _ in ()).throw(OSError(f"raw-exc-detail {secret}")))
        store_fail = services_mod._finalize_service_log_for_drive(drive, record_1)
    assert store_fail == {"deleted_live_log": False, "full_log_ref": {}, "tail": "selected=***\nREADY launch-1\n",
                          "errors": ["capture: OSError"], "retained_live_log_path": str(log_path)}
    assert log_path.read_bytes() == initial_bytes

    absent = services_mod._finalize_service_log_for_drive(drive, record_1, log_path=log_path.with_name("absent.log"))
    empty_path = log_path.with_name("empty.log")
    empty_path.write_bytes(b"")
    empty = services_mod._finalize_service_log_for_drive(drive, record_1, log_path=empty_path)
    assert absent == {"deleted_live_log": True, "full_log_ref": {}, "tail": "", "errors": []}
    assert empty["deleted_live_log"] is True and empty["full_log_ref"]["size"] == 0 and empty["full_log_ref"]["sha256"]

    assert services_mod._service_status(ctx, "demo") == "⚠️ SERVICE_NOT_FOUND: demo"
    assert services_mod._service_logs(ctx, "demo") == "⚠️ SERVICE_NOT_FOUND: demo"
    assert services_mod._stop_service(ctx, "demo") == "⚠️ SERVICE_NOT_FOUND: demo"
    assert services_mod.stop_task_services(ctx) == [] and stops == [("OURO_LAUNCH_SCOPE_1",)]

    log_path.write_text(f"selected={secret}\nREADY launch-2\n", encoding="utf-8")
    (cwd / "report.txt").write_text("ok-2", encoding="utf-8")
    seed(4102, "OURO_LAUNCH_SCOPE_2")
    assert json.loads(services_mod._service_status(ctx, "demo"))["state"] == "running"
    live_logs = services_mod._service_logs(ctx, "demo")
    assert "READY launch-2" in live_logs and secret not in live_logs and "selected=***" in live_logs

    stopped_2 = json.loads(services_mod._stop_service(ctx, "demo"))
    blob_2 = gzip.decompress(pathlib.Path(stopped_2["log_finalization"]["full_log_ref"]["path"]).read_bytes()).decode("utf-8")
    assert stops == [("OURO_LAUNCH_SCOPE_1",), ("OURO_LAUNCH_SCOPE_2",)]
    assert stopped_2["log_finalization"]["deleted_live_log"] is True and not log_path.exists()
    assert not stopped_2["log_finalization"]["errors"] and "selected=***" in stopped_2["log_finalization"]["tail"]
    assert secret not in json.dumps(stopped_2) and secret not in blob_2 and "READY launch-2" in blob_2


@pytest.mark.parametrize("stop_kind", ["named", "teardown"])
def test_byte_named_declared_output_keeps_a_readable_version(tmp_path, monkeypatch, stop_kind):
    """Both stop paths keep the downloadable version and carry byte names through final evidence."""
    import errno

    from ouroboros import loop_budget, loop_delivery, loop_forced_finalization
    from ouroboros.tools import services as services_mod
    from ouroboros.tools.registry import ToolContext

    if os.name != "posix":
        pytest.skip("POSIX byte filename")

    repo, drive = tmp_path / "repo", tmp_path / "data"
    task_id, key = "task-byte-name", "task-byte-name:demo"
    cwd, log_path = drive / "task_drives" / task_id, drive / "services" / task_id / "demo.log"
    repo.mkdir()
    cwd.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    raw_name = b"report-\xff.t\xfext"
    byte_name = raw_name.decode("utf-8", "surrogateescape")
    try:
        (cwd / byte_name).write_bytes(b"byte-named run")
    except OSError as exc:
        if exc.errno in {errno.EILSEQ, errno.EINVAL}:
            pytest.skip("fixture filesystem does not create non-UTF-8 names")
        raise
    (cwd / "report.txt").write_bytes(b"ordinary run")
    log_path.write_bytes(b"READY launch-1\n")
    ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id=task_id)
    trace = {"verification_events": []}
    ctx._execution_trace = trace

    def fake_end_held(held, **_kwargs):
        item = held[0]
        item["proc"].returncode = 0
        return {item["key"]: {"birth": "b1", "known": [], "facts": {"exit_confirmed": True, "ended": [item["proc"].pid]}}}

    proc = SimpleNamespace(pid=4311, returncode=None)
    proc.poll = lambda: proc.returncode
    record = services_mod.ServiceRecord(
        name="demo", service_id=key, task_id=task_id, cmd=["svc"], cwd=str(cwd),
        cwd_root="task_drive", cwd_base=str(cwd), cwd_source="task_drive", log_path=log_path,
        proc=proc, outputs=["report.txt", byte_name],
        before_outputs={"report.txt": (False, 0, ""), byte_name: (False, 0, "")},
        secret_values=(), scope_marker="OURO_LAUNCH_SCOPE_" + "0" * 32,
    )
    monkeypatch.setitem(services_mod._SERVICES, key, record)
    monkeypatch.setattr("ouroboros.process_scope.end_held_processes", fake_end_held)

    if stop_kind == "named":
        stopped = json.loads(services_mod._stop_service(ctx, "demo").encode("utf-8"))
    else:
        turn = SimpleNamespace(tools=SimpleNamespace(_ctx=ctx), drive_root=drive, task_id=task_id,
                               drive_logs=ctx.drive_logs(), event_queue=None, llm_trace=trace)
        assert loop_budget._finalize_task_services(turn)
        event = trace["verification_events"][-1]
        assert event["kind"] == "services_stopped", event
        [stopped] = event["services"]
        assert stopped["lifecycle"] == "stopped"
    assert stopped["state"] == "exited"
    assert stopped["outputs"] == ["report.txt", byte_name]
    assert byte_name in stopped["artifact_outputs"]

    facts = stopped["service_stop"]
    assert facts["launch_id"] == stopped["launch_id"] == "launch-" + "0" * 32
    assert facts["process"] == {"state": "stopped", "returncode": 0}
    store = drive / "task_results" / "artifacts" / task_id
    expected = ((b"report.txt", "report.txt", ".txt", b"ordinary run"),
                (raw_name, r'"report-\udcff.t\udcfext"', ".t_xt", b"byte-named run"))
    for row, (original, shown, suffix, content) in zip(facts["outputs"], expected, strict=True):
        assert row["state"] == "saved" and row["name"] == shown
        [file] = row["files"]
        name_hash = hashlib.sha256(original).hexdigest()[:16]
        digest = hashlib.sha256(content).hexdigest()
        version = f"service-output-{name_hash}-{digest}{suffix}"
        assert file["name"] == shown and file["path"] == version and file["sha256"] == digest
        assert file["download_url"] == f"/api/tasks/{task_id}/artifacts/{version}"
        assert (store / version).read_bytes() == content
        assert (store / os.fsdecode(original)).read_bytes() == content

    assert loop_delivery.observed_delivery_evidence(ctx, trace)
    author = SimpleNamespace(messages=[], forced_service_evidence_fingerprint="", drive_root=drive, task_id=task_id)
    loop_forced_finalization._disclose_service_finalization(author, trace)
    assert author.forced_service_evidence_fingerprint
    [message] = author.messages
    [row] = json.loads(message["content"].encode("utf-8").splitlines()[-1])
    [service] = row["services"]
    assert service["service_stop"] == facts
    if stop_kind == "teardown":  # The full teardown row also carries the original audit text.
        assert service["artifact_outputs"] == stopped["artifact_outputs"]
