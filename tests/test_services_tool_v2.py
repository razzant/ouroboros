import json
import os
import pathlib
import sys
from types import SimpleNamespace

from ouroboros.tools.registry import ToolRegistry
from ouroboros.tools.services import archive_task_service_logs, prune_service_logs


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

    stopped_result = registry.execute_result("stop_service", {"name": "artifact_service"})
    stopped = json.loads(stopped_result.text)

    assert "ARTIFACT_OUTPUTS" in stopped["artifact_outputs"]
    assert (stopped_result.status, stopped_result.code) == ("ok", "OK")
    assert dict(stopped_result.meta) == {"artifact_registered": True}
    assert "exit_code" not in stopped_result.meta
    artifact_path = drive / "task_results" / "artifacts" / "task-service-output" / "service.html"
    assert artifact_path.read_text(encoding="utf-8") == "<h1>ok</h1>"


def test_stop_service_unchanged_output_is_not_registered(
    tmp_path, monkeypatch,
):
    _force_light_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-service-unchanged-output"
    task_drive = drive / "task_drives" / "task-service-unchanged-output"
    task_drive.mkdir(parents=True)
    (task_drive / "existing.txt").write_text("same", encoding="utf-8")

    registry.execute("start_service", {
        "name": "unchanged_output_service",
        "cmd": [sys.executable, "-c", "print('READY', flush=True)"],
        "cwd": str(task_drive),
        "outputs": ["existing.txt"],
        "readiness": {"timeout_sec": 1},
    })
    stopped_result = registry.execute_result(
        "stop_service", {"name": "unchanged_output_service"},
    )
    stopped = json.loads(stopped_result.text)

    assert (stopped_result.status, stopped_result.code) == ("ok", "OK")
    assert dict(stopped_result.meta) == {}
    assert "unchanged output (cosmetic)" in stopped["artifact_outputs"]


def test_stop_service_partial_output_failure_never_claims_registration(
    tmp_path, monkeypatch,
):
    _force_light_runtime(monkeypatch)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry._ctx.task_id = "task-service-partial-output"
    task_drive = drive / "task_drives" / "task-service-partial-output"
    task_drive.mkdir(parents=True)

    registry.execute("start_service", {
        "name": "partial_output_service",
        "cmd": [
            sys.executable,
            "-c",
            "from pathlib import Path; Path('present.txt').write_text('ok'); print('READY', flush=True)",
        ],
        "cwd": str(task_drive),
        "outputs": ["present.txt", "missing.txt"],
        "readiness": {"timeout_sec": 1},
    })
    stopped_result = registry.execute_result(
        "stop_service", {"name": "partial_output_service"},
    )

    assert (stopped_result.status, stopped_result.code) == (
        "error",
        "ARTIFACT_OUTPUT_ERROR",
    )
    assert dict(stopped_result.meta) == {}
    assert "registered output" in stopped_result.text
    assert "missing output" in stopped_result.text


def test_executor_stop_service_publishes_actual_registration_only(
    tmp_path, monkeypatch,
):
    from ouroboros.tools import services as services_mod
    from ouroboros.tools import shell as shell_mod
    from ouroboros.tools.tool_result import ToolResult

    ctx = SimpleNamespace(
        task_id="task-executor-stop",
        _active_builtin_tool_result=object(),
    )
    monkeypatch.setattr(
        services_mod,
        "executor_ref_from_ctx",
        lambda _ctx: {"type": "fixture"},
    )
    monkeypatch.setattr(
        services_mod,
        "_service_output_binding",
        lambda *_args, **_kwargs: None,
    )

    def stopped_payload(_ctx, _name):
        return {
            "state": "stopped",
            "outputs": ["report.txt"],
            "cwd_root": "active_workspace",
            "cwd_base": str(tmp_path),
            "cwd_source": "active_workspace",
            "skill_name": "",
            "host_cwd": str(tmp_path),
            "_before_outputs": {},
        }

    monkeypatch.setattr(services_mod, "executor_stop_service", stopped_payload)
    monkeypatch.setattr(
        shell_mod,
        "_register_process_outputs",
        lambda *_args, **_kwargs: (
            "\n\nARTIFACT_OUTPUTS:\n- registered output",
            False,
            True,
        ),
    )

    success_text = services_mod._stop_service(ctx, "fixture")
    success = ctx._active_builtin_tool_result
    assert isinstance(success, ToolResult)
    assert success.text == success_text
    assert (success.status, success.code) == ("ok", "OK")
    assert dict(success.meta) == {"artifact_registered": True}
    assert "exit_code" not in success.meta

    ctx._active_builtin_tool_result = object()
    monkeypatch.setattr(
        shell_mod,
        "_register_process_outputs",
        lambda *_args, **_kwargs: (
            "\n\n⚠️ ARTIFACT_OUTPUT_ERROR:\n- partial failure",
            True,
            True,
        ),
    )

    failed_text = services_mod._stop_service(ctx, "fixture")
    failed = ctx._active_builtin_tool_result
    assert isinstance(failed, ToolResult)
    assert failed.text == failed_text
    assert (failed.status, failed.code) == (
        "error",
        "ARTIFACT_OUTPUT_ERROR",
    )
    assert dict(failed.meta) == {}


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
