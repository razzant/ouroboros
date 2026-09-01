import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

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
    assert event["full_log_ref"]["sha256"]


def test_archive_task_service_logs_finalizes_forced_worker_leftovers(tmp_path):
    drive = tmp_path / "data"
    task_dir = drive / "services" / "task-forced"
    task_dir.mkdir(parents=True)
    log = task_dir / "devserver.log"
    log.write_text("READY\n", encoding="utf-8")

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
    assert event["full_log_ref"]["sha256"]

    child_drive = tmp_path / "child-data"
    child_task_dir = child_drive / "services" / "task-child"
    child_task_dir.mkdir(parents=True)
    (child_task_dir / "devserver.log").write_text("READY\n", encoding="utf-8")

    child_report = archive_task_service_logs(
        drive,
        "task-child",
        {"child_drive_root": str(child_drive)},
    )

    assert child_report["archived_files"] == 1
    assert child_report["deleted_dirs"] == 1
    assert not child_task_dir.exists()


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


def test_service_log_retention_retains_oversized_stale_logs(tmp_path, monkeypatch):
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

    assert report["deleted_files"] == 0
    assert report["archived_files"] == 0
    assert report["retained_files"] == 1
    assert log.exists()


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
    assert stopped["artifact_output_failed"] is False


def test_light_start_service_blocks_runtime_data_upload_write(tmp_path, monkeypatch):
    _force_light_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-service-runtime-data"
    upload = drive / "uploads" / "report.html"
    task_drive = registry._ctx.task_drive_root()

    result = registry.execute("start_service", {
        "name": "runtime_data_writer",
        "cmd": [
            sys.executable,
            "-c",
            (
                "from pathlib import Path\n"
                f"p = Path({str(upload)!r})\n"
                "p.parent.mkdir(parents=True, exist_ok=True)\n"
                "p.write_text('bad')\n"
            ),
        ],
        "cwd": str(task_drive),
        "readiness": {"timeout_sec": 1},
    })

    assert "LIGHT_MODE_BLOCKED" in result
    assert "runtime_data" in result
    assert not upload.exists()
    assert registry.execute("service_status", {"name": "runtime_data_writer"}).startswith("⚠️ SERVICE_NOT_FOUND")


def test_light_start_service_blocks_relative_runtime_data_upload_write(tmp_path, monkeypatch):
    _force_light_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-service-runtime-data"
    upload = drive / "uploads" / "relative-report.html"
    task_drive = registry._ctx.task_drive_root()

    result = registry.execute("start_service", {
        "name": "runtime_data_relative_writer",
        "cmd": [
            sys.executable,
            "-c",
            (
                "from pathlib import Path\n"
                "p = Path('../../uploads/relative-report.html')\n"
                "p.parent.mkdir(parents=True, exist_ok=True)\n"
                "p.write_text('bad')\n"
            ),
        ],
        "cwd": str(task_drive),
        "readiness": {"timeout_sec": 1},
    })

    assert "LIGHT_MODE_BLOCKED" in result
    assert "runtime_data" in result
    assert not upload.exists()
    assert registry.execute("service_status", {"name": "runtime_data_relative_writer"}).startswith("⚠️ SERVICE_NOT_FOUND")


def test_light_start_service_blocks_env_runtime_data_upload_write(tmp_path, monkeypatch):
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
        "cmd": ["sh", "-c", "mkdir -p \"$OUROBOROS_DATA_DIR/uploads\" && echo bad > \"$OUROBOROS_DATA_DIR/uploads/env-report.html\""],
        "cwd": str(registry._ctx.task_drive_root()),
        "readiness": {"timeout_sec": 1},
    })

    assert "LIGHT_MODE_BLOCKED" in result
    assert "runtime_data" in result
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
