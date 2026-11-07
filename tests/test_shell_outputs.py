import pathlib
import sys

import pytest

from ouroboros.tools.registry import ToolContext, ToolRegistry


@pytest.fixture
def project(tmp_path, monkeypatch):
    from ouroboros import config as cfg

    cfg.reset_runtime_mode_baseline_for_tests()
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.delenv(cfg.BOOT_RUNTIME_MODE_ENV_KEY, raising=False)
    home, system, active, data, outside = (tmp_path / name for name in ("home", "system", "active", "data", "outside"))
    for path in (home, system, active / "site", data, outside / "deep" / "er"):
        path.mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(active / "Deliverables"))
    (active / "Deliverables").mkdir()
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    ctx = ToolContext(
        repo_dir=system,
        system_repo_dir=system,
        drive_root=data,
        workspace_root=active,
        workspace_mode="external",
        task_id="task-outputs-2916",
    )
    registry = ToolRegistry(repo_dir=system, drive_root=data)
    registry.set_context(ctx)
    return registry, active, outside


def test_unchanged_task_drive_file_first_declaration_registers_canonical_artifact(project):
    registry, _active, _outside = project
    task_drive = pathlib.Path(registry._ctx.drive_root) / "task_drives" / registry._ctx.task_id
    task_drive.mkdir(parents=True, exist_ok=True)
    report = task_drive / "report.md"
    report.write_text("# Deliverable Report\nAll tests passed.\n", encoding="utf-8")

    result = registry.execute(
        "run_command",
        {
            "cmd": [sys.executable, "-c", "print('verify complete')"],
            "cwd": str(task_drive),
            "outputs": ["report.md"],
        },
    )

    assert "exit_code=0" in result
    assert "ARTIFACT_OUTPUT_ERROR" not in result
    assert "registered output" in result
    assert "ARTIFACT_OUTPUTS:" in result

    artifact_file = pathlib.Path(registry._ctx.drive_root) / "task_results" / "artifacts" / registry._ctx.task_id / "report.md"
    assert artifact_file.is_file()
    assert artifact_file.read_text(encoding="utf-8") == "# Deliverable Report\nAll tests passed.\n"

    read_result = registry.execute(
        "read_file",
        {"root": "artifact_store", "path": "report.md"},
    )
    assert "# Deliverable Report" in read_result


def test_unchanged_task_drive_file_already_captured_is_cosmetic_without_duplication(project):
    """Case 2: An existing unchanged task-drive file already captured identically skips cosmetically."""
    registry, _active, _outside = project
    task_drive = pathlib.Path(registry._ctx.drive_root) / "task_drives" / registry._ctx.task_id
    task_drive.mkdir(parents=True, exist_ok=True)
    report = task_drive / "report.md"
    report.write_text("# Initial Deliverable\n", encoding="utf-8")

    first_result = registry.execute(
        "run_command",
        {
            "cmd": [sys.executable, "-c", "print('build report')"],
            "cwd": str(task_drive),
            "outputs": ["report.md"],
        },
    )
    assert "registered output" in first_result
    assert "ARTIFACT_OUTPUTS:" in first_result

    # Second run without modifying report.md: should be recognized as already captured and skipped cosmetically
    second_result = registry.execute(
        "run_command",
        {
            "cmd": [sys.executable, "-c", "print('re-verify report')"],
            "cwd": str(task_drive),
            "outputs": ["report.md"],
        },
    )
    assert "ARTIFACT_OUTPUT_ERROR" not in second_result
    assert "unchanged output (cosmetic): report.md" in second_result
    assert "ARTIFACT_OUTPUT_NOTE:" in second_result
    assert "ARTIFACT_OUTPUTS:" not in second_result


def test_edited_task_drive_file_with_older_capture_updates_canonical_artifact(project):
    """Case 3: A task-drive file edited before a no-op command must update the canonical capture."""
    registry, _active, _outside = project
    task_drive = pathlib.Path(registry._ctx.drive_root) / "task_drives" / registry._ctx.task_id
    task_drive.mkdir(parents=True, exist_ok=True)
    report = task_drive / "report.md"
    report.write_text("version 1 content", encoding="utf-8")

    first_result = registry.execute(
        "run_command",
        {
            "cmd": [sys.executable, "-c", "print('v1')"],
            "cwd": str(task_drive),
            "outputs": ["report.md"],
        },
    )
    assert "registered output" in first_result

    # Edit the file in task drive before running a command that does not touch it
    import time
    time.sleep(0.05)
    report.write_text("version 2 updated content", encoding="utf-8")
    time.sleep(0.05)

    second_result = registry.execute(
        "run_command",
        {
            "cmd": [sys.executable, "-c", "print('noop check')"],
            "cwd": str(task_drive),
            "outputs": ["report.md"],
        },
    )
    assert "registered output" in second_result
    assert "ARTIFACT_OUTPUTS:" in second_result

    read_result = registry.execute(
        "read_file",
        {"root": "artifact_store", "path": "report.md"},
    )
    assert "version 2 updated content" in read_result
