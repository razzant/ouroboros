import json
import os
import pathlib
import shlex
import subprocess
import sys
import threading
import time
from contextlib import contextmanager

import pytest
from types import SimpleNamespace

from ouroboros.launch_custody import service_log_launch_id
from ouroboros.tools.registry import ToolRegistry
from ouroboros.tools.services import archive_task_service_logs, prune_service_logs
from tests._launch_stop_shared import _force_advanced_runtime, _force_light_runtime, _launch_log, _wait_for_service_log


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
    assert not pathlib.Path(start_payload["log_path"]).exists()
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
    _wait_for_service_log(json.loads(start)["log_path"], lambda b: b"OPENAI_API_KEY=" in b)
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
    log = _launch_log(stale, "demo", b"old")
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


def test_ordinary_stop_uses_its_launch_values_for_both_diagnostics(tmp_path):
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.observability import read_blob_ref
    from ouroboros.tools import services

    path = _launch_log(tmp_path / "services/task", "paper", b"sample public label\n", needs_value_masking=True)
    record = SimpleNamespace(log_path=path, secret_values=("public label",))
    captured = services._finalize_service_log_for_drive(tmp_path, record)
    assert captured["deleted_live_log"] and not path.exists()
    [event] = map(json.loads, (tmp_path / "logs/events.jsonl").read_bytes().splitlines())
    assert captured["source_scope"] == "full"
    assert read_actor_source_bytes(tmp_path, "task", captured["source_ref"]) == b"sample ***\n"
    assert read_blob_ref(tmp_path, event["full_log_ref"], expected_kind="txt") == "sample ***\n"


@pytest.fixture
def retired_service_drive(tmp_path, monkeypatch):
    from ouroboros import body_candidate, config, task_custody
    from ouroboros.launch_custody import open_service_log
    from ouroboros.process_census import LAUNCH_SCOPE_PREFIX, mint_marker
    from ouroboros.task_results import write_task_result

    data = tmp_path.resolve() / "data"
    task = "paper-task"
    drive = data / "task_drives" / task
    drive.mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(body_candidate, "live_lineage_processes", lambda *_: [])
    marker = mint_marker(LAUNCH_SCOPE_PREFIX)
    path, handle = open_service_log(drive, task, "paper", marker, needs_value_masking=True)
    with handle:
        handle.write(b"field notes\n")
    write_task_result(data, "parent", "running", root_task_id="parent")
    write_task_result(data, task, "completed", result="Paper is ready.", parent_task_id="parent",
                      root_task_id="parent", child_drive_root=str(drive))
    return SimpleNamespace(data=data, drive=drive, task=task, path=path, marker=marker,
                           settle=lambda **kw: task_custody.settle_child_drive(
                               data, task, drive, live=lambda _: False, **kw))


@pytest.mark.parametrize("failure", ["size", "write"])
def test_only_the_ended_service_log_survives_drive_release(retired_service_drive, monkeypatch, failure):
    from ouroboros import task_custody
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.launch_custody import open_service_log_source, service_log_launch_id
    from ouroboros.outcomes import public_task_result
    from ouroboros.tool_access_paths import is_private_service_log_path
    from ouroboros.tools import services

    w = retired_service_drive
    (w.drive / "scratch.txt").write_text("working note", encoding="utf-8")
    store = w.data / "private_service_logs" / w.task
    assert w.path.parent == store and is_private_service_log_path(w.path, w.drive)
    assert not is_private_service_log_path(w.data / "projects/paper" / store.name / w.path.name, w.data)
    if os.name == "posix":
        assert not (store.parent.stat().st_mode | store.stat().st_mode | w.path.stat().st_mode) & 0o077

    if failure == "size":
        monkeypatch.setattr(services, "_MAX_SERVICE_LOG_BLOB_BYTES", 5)
    else:
        def no_archive(*_args):
            raise OSError(f"sample archive write failure for {w.path}")
        monkeypatch.setattr(services, "_store_service_log", no_archive)
    prior_log = services._finalize_service_log_for_drive(w.drive, SimpleNamespace(log_path=w.path, secret_values=("field",)))
    assert prior_log["live_log"] == "retained" and not prior_log["deleted_live_log"]
    if failure == "size":
        assert prior_log["source_unavailable"] == "tail_masking_unverified" and "source_ref" not in prior_log
    else:
        assert prior_log["source_scope"] == "full" and prior_log["error_count"] == 1
        assert read_actor_source_bytes(w.data, w.task, prior_log["source_ref"], retained=False) == b"*** notes\n"
    prior = {"launch_id": service_log_launch_id(w.path.name), "log_path": str(w.path), "log_finalization": prior_log}
    shown = public_task_result({"services": [prior], "outcome_axes": {"execution": {"services": [prior]}}})
    assert shown["services"] == shown["outcome_axes"]["execution"]["services"]
    assert shown["services"][0]["log_finalization"]["live_log"] == "retained"
    assert shown["services"][0]["log_outcome"] == ("retained" if failure == "size" else "not_saved")
    assert str(w.path) not in json.dumps(shown) and "field notes" not in json.dumps(shown)

    assert w.settle()["status"] == "removed" and not w.drive.exists()
    assert task_custody.remove_custody_trash(w.data)["removed"] == 1
    assert list(store.iterdir()) == [w.path]
    with open_service_log_source(w.drive, w.path, closed=True)[0] as source:
        assert source.read() == b"field notes\n"


def test_stop_releases_original_only_after_a_canonical_launch_receipt(retired_service_drive, monkeypatch):
    from ouroboros import service_outcomes, task_custody
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.observability import read_blob_ref
    from ouroboros.tools import services

    w = retired_service_drive
    record = SimpleNamespace(log_path=w.path, secret_values=("field",))
    events = w.data / "logs" / "events.jsonl"
    with monkeypatch.context() as patch:
        def refuse(path, row, **kw):
            assert path == events and row["launch_id"] == service_log_launch_id(w.path.name)
            assert kw == {"ensure_record_boundary": True, "require_lock": True, "fsync": True}
            assert w.path.is_file()  # the only original stays while publishing the receipt
            return False

        patch.setattr(services, "append_jsonl", refuse)
        kept = services._finalize_service_log_for_drive(w.drive, record)
    assert kept["archive_outcome"] == "not_saved" and kept["source_scope"] == "full" and kept["error_count"] == 1
    assert read_actor_source_bytes(w.data, w.task, kept["source_ref"], retained=False) == b"*** notes\n"
    assert not kept["deleted_live_log"] and w.path.read_bytes() == b"field notes\n"

    done = services._finalize_service_log_for_drive(w.drive, record)
    assert done["deleted_live_log"] and not w.path.exists()
    [receipt] = [row for row in map(json.loads, events.read_bytes().splitlines())
                 if row.get("type") == "service_log_archived"]
    assert receipt["task_id"] == w.task and receipt["launch_id"] == service_log_launch_id(w.path.name)
    assert set(receipt) == {"ts", "type", "task_id", "launch_id", "name", "full_log_ref", "tail_chars"}
    shown = service_outcomes.display({"log_finalization": done})
    assert shown["log_finalization"] == done and done["source_scope"] == "full"
    assert done["source_ref"]["sha256"] == receipt["full_log_ref"]["sha256"]
    assert w.settle()["status"] == "removed" and not w.drive.exists()
    assert task_custody.remove_custody_trash(w.data)["removed"] == 1
    assert read_blob_ref(w.data, receipt["full_log_ref"], expected_kind="txt") == "*** notes\n"


@pytest.mark.parametrize("indexed", [True, False])
def test_kept_docker_writer_without_a_host_log_holds_only_its_drive(retired_service_drive, monkeypatch, indexed):
    from ouroboros import owned_shutdown, task_custody, workspace_executor
    from ouroboros.task_results import write_task_result

    w = retired_service_drive
    w.path.unlink()  # Docker log storage is backend-only; no host log or output declaration witnesses it.
    path = w.drive / "state" / workspace_executor._PROCESS_STATE_DIR / "service-note.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"id": path.stem, "owner": workspace_executor._PROCESS_RECORD_OWNER,
              "schema_version": workspace_executor._PROCESS_RECORD_SCHEMA_VERSION, "record_type": "service",
              "executor_type": "docker_exec", "task_id": w.task, "host_pid": 0,
              "container_name": "paper-preview", "backend_pid": "101", "keep_alive": True}
    path.write_text(json.dumps(record), encoding="utf-8")
    if indexed:
        assert owned_shutdown.record_executor_process(path, record)

    under_guard = False

    @contextmanager
    def guard():
        nonlocal under_guard
        under_guard = True
        try:
            yield True
        finally:
            under_guard = False

    def probe(container, pid):
        assert not under_guard and (container, pid) == ("paper-preview", "101")
        return state

    monkeypatch.setattr(workspace_executor, "_docker_pid_state", probe)
    for state in ("running", "unknown"):
        assert w.settle()["reason"] == "service_writer_unsettled" and w.drive.is_dir()
    other = w.drive.with_name("other")
    other.mkdir()
    write_task_result(w.data, "other", "completed", result="No background work.", child_drive_root=str(other))
    assert task_custody.settle_child_drive(w.data, "other", other, live=lambda _: False, guard=guard)["status"] == "removed"
    state = "exited"
    assert w.settle(guard=guard)["status"] == "removed" and not w.drive.exists()


def test_owner_keep_and_legacy_log_hold_precede_release(retired_service_drive):
    """Keep precedes everything a settlement would do to a legacy log: nothing is moved out of a kept drive."""
    from ouroboros.task_results import write_task_result

    w = retired_service_drive
    legacy = _launch_log(w.drive / "services" / w.task, "earlier", b"previous note", needs_value_masking=True)
    write_task_result(w.data, w.task, "completed", owner_keep={"basis": "owner"})
    assert w.settle()["reason"] == "owner_keep"
    assert legacy.read_bytes() == b"previous note" and w.path.is_file()
    assert [entry.name for entry in w.path.parent.iterdir()] == [w.path.name], "the private store got no copy"


def test_ordinary_no_outputs_checkpoint_survives_separate_drive_release(retired_service_drive, monkeypatch):
    from ouroboros import dialogue_store, service_finalization, service_outcomes, task_custody, utils, workspace_executor
    from ouroboros.task_results import write_task_result
    from ouroboros.tools import services
    from ouroboros.tools.kept_service_outputs import declaration_path
    from ouroboros.tools.registry import ToolContext
    from tests._launch_stop_shared import _exit_context

    w = retired_service_drive
    room = 19
    write_task_result(w.data, w.task, "completed", chat_id=room)
    tool_ctx = ToolContext(repo_dir=w.drive, drive_root=w.drive, task_id=w.task)
    ctx = _exit_context(w.drive, tool_ctx)
    record = services.ServiceRecord(name="paper", service_id=f"{w.task}:paper", task_id=w.task,
        cmd=["synthetic"], cwd=str(w.drive), log_path=w.path, proc=SimpleNamespace(pid=42),
        drive_root=str(w.drive), scope_marker=w.marker)
    launch = service_outcomes.launch_facts(record)["launch_id"]
    declaration = declaration_path(w.drive, w.task, launch, owed=False)
    monkeypatch.setattr(services, "_SERVICES", {record.service_id: record})
    monkeypatch.setattr(workspace_executor, "_SERVICES", {})
    monkeypatch.setattr(services, "_stop_host_record", lambda *_a, **_k: (
        {**service_outcomes.process_facts(record, "exited"), "process_cleanup": {"exit_confirmed": True}}, True))
    append, write = utils.append_jsonl, utils.atomic_write_json
    refused = []

    def miss_first_declaration(target, value, **kw):
        if target == declaration and not refused:
            refused.append(target)
            raise OSError("first stop declaration not written")
        return write(target, value, **kw)

    with monkeypatch.context() as missing_logs:
        missing_logs.setattr(utils, "atomic_write_json", miss_first_declaration)
        missing_logs.setattr(utils, "append_jsonl", lambda path, row, **kw: False if row.get("card_row_id")
                             else append(path, row, **kw))
        assert service_finalization.finalize(ctx, ctx.llm_trace)
        assert refused == [declaration] and declaration.is_file()  # finalization repaired the missing stop record
        assert w.settle()["reason"] == "service_log_uncustodied"
        assert w.drive.is_dir() and declaration.is_file()
    [room_row] = [item.row for item in dialogue_store.newest(w.data, "progress", limit=20)
                  if item.row.get("card_row_id")]
    assert room_row["chat_id"] == room and room_row["card_row_revision"] == 1
    monkeypatch.setattr(services, "_stop_host_record", lambda *_a, **_k: pytest.fail("called Stop again"))
    assert w.settle()["status"] == "removed" and not w.drive.exists()
    assert task_custody.remove_custody_trash(w.data)["removed"] == 1
    [logged] = [row for row in map(json.loads, (w.data / "logs/events.jsonl").read_text().splitlines())
                if row.get("card_row_id")]
    assert logged == room_row and "field notes" not in logged["text"] and str(w.path) not in logged["text"]


def test_a_legacy_log_is_kept_alone_in_the_private_store_and_its_drive_released(retired_service_drive, monkeypatch):
    """The owner's choice (18 Nov 2026): a closed log never holds a whole drive. The settlement hands each legacy
    drive-local log, alone, to the installation's private store and then releases the drive as any other. A copy
    that cannot be proven keeps its source and the drive; an unsettled writer keeps both as well."""
    from ouroboros import launch_custody, task_custody

    w = retired_service_drive
    store = w.path.parent
    legacy = _launch_log(w.drive / "services" / w.task, "earlier", b"previous note", needs_value_masking=True)
    unnamed = legacy.with_name("old.log")  # a log from before launches named their files
    unnamed.write_bytes(b"unnamed note")
    (w.drive / "scratch.txt").write_text("working note", encoding="utf-8")

    with monkeypatch.context() as unproven:
        unproven.setattr(launch_custody, "_keep_legacy_log", lambda *_args: False)
        assert w.settle()["reason"] == "service_log_uncustodied"
    with monkeypatch.context() as writing:
        moved = []
        writing.setattr(launch_custody, "drive_service_writers_unsettled", lambda *_args: True)
        writing.setattr(launch_custody, "_keep_legacy_log", lambda *args: moved.append(args) or True)
        assert w.settle()["reason"] == "service_log_uncustodied" and not moved, "an open log is nobody's to move"
    assert legacy.read_bytes() == b"previous note" and unnamed.read_bytes() == b"unnamed note"
    assert [entry.name for entry in store.iterdir()] == [w.path.name]

    assert w.settle()["status"] == "removed" and not w.drive.exists()
    assert task_custody.remove_custody_trash(w.data)["removed"] == 1
    kept = {entry.name: entry.read_bytes() for entry in store.iterdir()}
    assert kept[w.path.name] == b"field notes\n" and kept[legacy.name] == b"previous note"
    [renamed] = [name for name in kept if name.startswith("old@legacy-")]
    assert kept[renamed] == b"unnamed note" and len(kept) == 3
    assert service_log_launch_id(renamed) == "", "a log that named no launch is given none"
    if os.name == "posix":
        assert not any((store / name).stat().st_mode & 0o077 for name in kept)
    # Kept without a term: retention archives only a log whose launch opted in to standard masking.
    report = prune_service_logs(w.data, retention_days=0, now=time.time() + 60)
    assert report["archived_files"] == 0 and sorted(entry.name for entry in store.iterdir()) == sorted(kept)


@pytest.mark.skipif(os.name != "posix", reason="the planted copy needs the store's private mode bits")
@pytest.mark.parametrize("left", ["cut_short", "other_bytes"])
def test_a_repeated_legacy_transfer_redoes_a_cut_short_copy_and_never_touches_other_bytes(retired_service_drive, left):
    """An earlier transfer was cut short: its copy is a strict prefix of the source, so the repeat writes it whole
    and only then lets the source go. A name already holding other bytes is a conflict: neither file is touched and
    the drive stays."""
    from ouroboros.launch_custody import _keep_legacy_log

    w = retired_service_drive
    whole = b"previous note\n" * 4
    legacy = _launch_log(w.drive / "services" / w.task, "earlier", whole, needs_value_masking=True)
    copy = w.path.parent / legacy.name
    copy.write_bytes(whole[:len(whole) // 2] if left == "cut_short" else b"another launch's note\n")
    os.chmod(copy, 0o600)
    moved = _keep_legacy_log(w.drive, w.task, legacy.name)
    if left == "cut_short":
        assert moved and not legacy.exists() and copy.read_bytes() == whole
        assert not copy.stat().st_mode & 0o077
        assert w.settle()["status"] == "removed"
    else:
        assert not moved and legacy.read_bytes() == whole and copy.read_bytes() == b"another launch's note\n"
        assert w.settle()["reason"] == "service_log_uncustodied"


@pytest.mark.parametrize("close_at", ["copy", "directory_sync"])
def test_transfer_is_atomically_published_and_generation_close_keeps_the_original(retired_service_drive, monkeypatch, close_at):
    import shutil

    from ouroboros import utils

    w = retired_service_drive
    sample = b"synthetic legacy record\n"
    legacy = _launch_log(w.drive / "services" / w.task, "earlier", sample, needs_value_masking=True)
    retained = w.path.parent / legacy.name
    closed = False
    with monkeypatch.context() as race:
        if close_at == "copy":
            def copy(source, target, _length):
                nonlocal closed
                target.write(source.read(5))
                target.flush()
                assert legacy.exists() and not retained.exists(), "a reader must not see the partial final file"
                closed = True

            race.setattr(shutil, "copyfileobj", copy)
        else:
            fsync = utils.fsync_folder

            def synced(directory):
                nonlocal closed
                fsync(directory)
                if directory == retained.parent:
                    assert retained.read_bytes() == sample, "the only visible destination is complete"
                    closed = True

            race.setattr(utils, "fsync_folder", synced)
        assert w.settle(stop=lambda: closed)["reason"] == "generation_closed"
    assert legacy.read_bytes() == sample and retained.exists() is (close_at == "directory_sync")
    assert w.settle()["status"] == "removed" and retained.read_bytes() == sample


def test_private_open_does_not_replace_an_existing_launch(retired_service_drive):
    from ouroboros.launch_custody import open_service_log, open_service_log_source

    w = retired_service_drive
    with pytest.raises(FileExistsError):
        open_service_log(w.drive, w.task, "paper", w.marker, needs_value_masking=True)
    with open_service_log_source(w.drive, w.path, closed=True)[0] as source:
        assert source.read() == b"field notes\n"


def test_private_open_refuses_an_unverified_existing_store(tmp_path, monkeypatch):
    from ouroboros import config
    from ouroboros.launch_custody import open_service_log
    from ouroboros.process_census import LAUNCH_SCOPE_PREFIX, mint_marker

    data = tmp_path.resolve() / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setattr(config, "DATA_DIR", data)
    unverified = data / "private_service_logs"
    unverified.mkdir(parents=True)  # ordinary inheritance on Windows, not an explicit protected DACL
    if os.name == "posix":
        unverified.chmod(0o755)
    with pytest.raises(OSError):
        open_service_log(data, "paper-task", "paper", mint_marker(LAUNCH_SCOPE_PREFIX), needs_value_masking=True)
    assert not list(unverified.iterdir())


def test_deferred_masked_archive_uses_the_existing_private_file(retired_service_drive):
    from ouroboros.launch_custody import open_service_log
    from ouroboros.observability import read_blob_ref
    from ouroboros.process_census import LAUNCH_SCOPE_PREFIX, mint_marker

    w = retired_service_drive
    marker = mint_marker(LAUNCH_SCOPE_PREFIX)
    neighbor, handle = open_service_log(w.drive, w.task, "indexer", marker, needs_value_masking=False)
    with handle:
        handle.write(b"Index ready.\n")
    report = archive_task_service_logs(w.drive, w.task)
    assert report["archived_files"] == report["deleted_files"] == report["retained_files"] == 1
    events = (json.loads(line) for line in (w.data / "logs/events.jsonl").read_bytes().splitlines())
    archived, = (event for event in events if event.get("type") == "service_log_archived")
    assert read_blob_ref(w.data, archived["full_log_ref"], expected_kind="txt") == "Index ready.\n"
    assert not neighbor.exists() and w.path.is_file()


@pytest.mark.skipif(os.name != "nt", reason="native Windows data-writer sharing")
def test_windows_closed_read_waits_for_the_log_handle(retired_service_drive):
    from ouroboros.launch_custody import open_service_log, open_service_log_source
    from ouroboros.process_census import LAUNCH_SCOPE_PREFIX, mint_marker

    w = retired_service_drive
    marker = mint_marker(LAUNCH_SCOPE_PREFIX)
    path, handle = open_service_log(w.drive, w.task, "drawing", marker, needs_value_masking=False)
    with handle:
        handle.write(b"Drawing complete\n")
        with pytest.raises(OSError):
            open_service_log_source(w.drive, path, closed=True)
    with open_service_log_source(w.drive, path, closed=True)[0] as source:
        assert source.read() == b"Drawing complete\n"


@pytest.mark.parametrize("retire", ["terminal", "prune"])
def test_deferred_log_capture_requires_this_launch_to_allow_standard_masking(tmp_path, monkeypatch, retire):
    from ouroboros import body_candidate
    from ouroboros.observability import read_blob_ref
    from ouroboros.tools import services

    directory = tmp_path / "services" / "task"
    private = _launch_log(directory, "paper", b"ordinary sample", needs_value_masking=True)
    legacy = directory / "old.log"
    legacy.write_bytes(b"historical sample")
    first = _launch_log(directory, "one", b"first sample")
    second = _launch_log(directory, "two", b"second sample")
    projected, checked = [], []

    def standard_mask(text):
        projected.append(text)
        return SimpleNamespace(value="standard projection")

    monkeypatch.setattr(services, "redact_projection", standard_mask)
    monkeypatch.setattr(body_candidate, "live_lineage_processes", lambda _, task: checked.append(task) or [])
    monkeypatch.setattr(services, "service_execution_facts", lambda _: None)
    if retire == "terminal":
        report = archive_task_service_logs(tmp_path, "task")
    else:
        report = prune_service_logs(tmp_path, retention_days=0, now=time.time() + 60)

    assert report["archived_files"] == report["deleted_files"] == report["retained_files"] == 2
    assert sorted(projected) == ["first sample", "second sample"] and checked == ["task"]
    assert private.read_bytes() == b"ordinary sample" and legacy.read_bytes() == b"historical sample"
    assert not first.exists() and not second.exists()
    rows = [json.loads(line) for line in (tmp_path / "logs/events.jsonl").read_bytes().splitlines()]
    assert {row["launch_id"] for row in rows} == {service_log_launch_id(path.name) for path in (first, second)}
    assert all(read_blob_ref(tmp_path, row["full_log_ref"], expected_kind="txt") == "standard projection" for row in rows)


def test_retirement_keeps_the_launch_log_until_its_own_archive_receipt_exists(tmp_path, monkeypatch):
    from ouroboros import body_candidate
    from ouroboros.observability import read_blob_ref
    from ouroboros.tools import services

    directory = tmp_path / "services" / "task"
    first = _launch_log(directory, "demo", b"FIRST\n")
    second = _launch_log(directory, "demo", b"SECOND\n")
    events = tmp_path / "logs" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_bytes(b'{"unfinished"')
    monkeypatch.setattr(body_candidate, "live_lineage_processes", lambda *_: [])
    append = services.append_jsonl
    with monkeypatch.context() as patch:
        def record(path, row, **kwargs):
            assert kwargs == {"ensure_record_boundary": True, "require_lock": True, "fsync": True}
            return row["launch_id"] != service_log_launch_id(first.name) and append(path, row, **kwargs)
        patch.setattr(services, "append_jsonl", record)
        report = archive_task_service_logs(tmp_path, "task")
    assert report["archived_files"] == report["deleted_files"] == report["retained_files"] == 1 and report["errors"]
    assert first.read_bytes() == b"FIRST\n" and not second.exists()
    assert archive_task_service_logs(tmp_path, "task")["archived_files"] == 1
    original, *lines = events.read_bytes().splitlines()
    assert original == b'{"unfinished"'
    rows = [json.loads(line) for line in lines]
    assert [(row["name"], row["launch_id"], read_blob_ref(tmp_path, row["full_log_ref"], expected_kind="txt"))
            for row in rows if row.get("type") == "service_log_archived"] == [
                ("demo", service_log_launch_id(second.name), "SECOND\n"),
                ("demo", service_log_launch_id(first.name), "FIRST\n")]


def test_service_log_remains_when_the_event_cannot_be_synced(tmp_path, monkeypatch):
    from ouroboros import body_candidate, utils

    path = _launch_log(tmp_path / "services/task", "paper", b"sample output")
    monkeypatch.setattr(body_candidate, "live_lineage_processes", lambda *_: [])
    sync = utils.fsync_folder
    attempted = []

    def refuse_sync(directory):
        if directory == tmp_path / "logs":
            attempted.append(directory)
            raise OSError("sample sync failure")
        sync(directory)

    monkeypatch.setattr(utils, "fsync_folder", refuse_sync)
    report = archive_task_service_logs(tmp_path, "task")
    assert attempted == [tmp_path / "logs"]
    assert report["archived_files"] == report["deleted_files"] == 0
    assert report["retained_files"] == 1 and report["errors"]
    assert path.read_bytes() == b"sample output"


def test_archive_task_service_logs_finalizes_forced_worker_leftovers(tmp_path, monkeypatch):
    from ouroboros import body_candidate
    from ouroboros.observability import read_blob_ref

    monkeypatch.setattr(body_candidate, "live_lineage_processes", lambda *_: [])
    drive = tmp_path / "data"
    task_dir = drive / "services" / "task-forced"
    _launch_log(task_dir, "devserver", b"READY\n")

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

    child_drive = drive / "state" / "headless_tasks" / "task-child" / "data"
    child_task_dir = child_drive / "services" / "task-child"
    child_log = _launch_log(child_task_dir, "devserver", b"CHILD READY\n")
    elsewhere = tmp_path / "other-installation"
    other_log = _launch_log(elsewhere / "services" / "task-child", "devserver", b"OTHER READY\n")

    child_report = archive_task_service_logs(
        drive,
        "task-child",
        {"child_drive_root": str(child_drive), "drive_root": str(elsewhere)},
    )

    assert child_report["archived_files"] == 1
    assert child_report["deleted_dirs"] == 1
    assert not child_task_dir.exists()
    assert other_log.read_bytes() == b"OTHER READY\n" and not (elsewhere / "logs").exists()
    events = map(json.loads, (drive / "logs/events.jsonl").read_bytes().splitlines())
    [row] = [row for row in events if row.get("launch_id") == service_log_launch_id(child_log.name)]
    assert read_blob_ref(drive, row["full_log_ref"], expected_kind="txt") == "CHILD READY\n"


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
        _wait_for_service_log(log, lambda data: len(data) > size)
        assert log.read_text(encoding="utf-8").startswith("first\n"), "the same live file keeps growing"

        launch_id = kept_services(drive)["services"][0]["launch_id"]
        assert stop_kept_service(drive, launch_id, reason="test over", actor="owner:test")["outcome"] == "stopped"
        report = archive_task_service_logs(drive, "task-kept", {"id": "task-kept"})
        assert report["archived_files"] == 1 and report["deleted_files"] == 1 and not log.exists()
        events = [json.loads(line) for line in (drive / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        archived = [item for item in events if item.get("type") == "service_log_archived"][-1]
        assert archived["task_id"] == "task-kept" and archived["full_log_ref"]["sha256"]
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

    started = json.loads(registry.execute("start_service", {
        "name": "oversize",
        "cmd": [sys.executable, "-c", "print('x' * 100, flush=True)"],
        "readiness": {"timeout_sec": 1},
    }))
    log = pathlib.Path(started["log_path"])
    _wait_for_service_log(log, lambda b: len(b) > 100)
    stopped = json.loads(registry.execute("stop_service", {"name": "oversize"}))

    finalization = stopped["log_finalization"]
    assert finalization["deleted_live_log"] is False
    assert finalization["live_log"] == "retained" and str(log) not in json.dumps(stopped)
    assert finalization["source_unavailable"] == "tail_masking_unverified" and "source_ref" not in finalization
    assert log.exists()


def test_service_log_finalization_checks_size_before_full_read(tmp_path, monkeypatch):
    from ouroboros.tools import services as services_mod

    drive = tmp_path / "data"
    log_path = drive / "services" / "task-big" / "big.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("x" * 100, encoding="utf-8")
    monkeypatch.setattr(services_mod, "_MAX_SERVICE_LOG_BLOB_BYTES", 10)
    monkeypatch.setattr(services_mod, "_full_log_text", lambda *_: pytest.fail("oversized log read in full"))
    monkeypatch.setattr(services_mod, "_read_tail", lambda *_: pytest.fail("raw tail read at the stop boundary"))

    result = services_mod._finalize_service_log_for_drive(drive, SimpleNamespace(log_path=log_path))

    assert result["deleted_live_log"] is False
    assert result["archive_outcome"] == "retained" and result["source_unavailable"] == "tail_masking_unverified"
    assert "source_ref" not in result and "tail" not in result
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
    assert service["log_finalization"]["source_scope"] == "full"
    assert service["log_finalization"] == stopped[0]["log_finalization"] and "tail" not in service["log_finalization"]


def test_service_log_retention_uses_log_mtime_and_skips_symlinks(tmp_path):
    drive = tmp_path / "data"
    services_root = drive / "services"
    stale_dir = services_root / "task-old-dir"
    stale_dir.mkdir(parents=True)
    fresh_log = stale_dir / "fresh.log"
    fresh_log.write_text("fresh", encoding="utf-8")
    stale_log = _launch_log(stale_dir, "stale", b"old")
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
    log = _launch_log(stale, "big", b"x" * 100)
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
    _wait_for_service_log(json.loads(start)["log_path"], lambda b: b"READY" in b)

    stopped = json.loads(registry.execute("stop_service", {"name": "artifact_service"}))

    assert stopped["output_outcomes"][0]["status"] == "saved"
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
        _wait_for_service_log(json.loads(started)["log_path"], lambda b: b"READY" in b)

    start("verifier")
    stopped = registry.execute_result("stop_service", {"name": "verifier"})
    # Declared and left unchanged by the service, the existing report is still its deliverable.
    assert stopped.code == "OK" and stopped.meta.get("artifact_registered") is True, stopped.text
    assert json.loads(stopped.text)["output_outcomes"][0]["status"] == "saved"
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
