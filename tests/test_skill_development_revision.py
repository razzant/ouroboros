"""Ordinary managed Repair keeps its selected revision across real tool calls."""

from __future__ import annotations

import sys

import pytest

from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.skill_loader import compute_content_hash
from ouroboros.skill_repair_admission import load_repair_admission, record_repair_admission
from ouroboros.tool_access import active_tool_profile
from tests.test_d16_skill_payload_authority import _registry, _skill


@pytest.mark.parametrize(("legacy", "mode"), [(False, "advanced"), (True, "light"), (True, "pro")])
def test_repair_shell_then_file_write_keeps_observed_revision(tmp_path, monkeypatch, mode, legacy):
    registry, _repo, data = _registry(tmp_path, monkeypatch, mode=mode)
    payload = _skill(data, "demo", bucket="external")
    ctx = registry._ctx
    ctx.task_id = "repair-demo"
    ctx.task_constraint = TaskConstraint(
        mode="skill_repair" if legacy else "normal", skill_name="demo",
        payload_root="skills/external/demo", allow_enable=False,
    )
    initial = compute_content_hash(payload)
    record_repair_admission(data, "demo", task_id=ctx.task_id, base_content_hash=initial)
    assert active_tool_profile(ctx) == "self_modification"
    result = registry.execute("run_command", {
        "cwd": "skill_payload",
        "cmd": [sys.executable, "-c", "from pathlib import Path; Path('notes.txt').write_text('shell edit\\n')"],
    })
    assert (payload / "notes.txt").read_text() == "shell edit\n", result
    observed = load_repair_admission(data, "demo")
    assert observed["base_content_hash"] == initial
    assert observed["expected_content_hash"] == compute_content_hash(payload)
    assert observed["revision_attribution"] == "opaque_operation_unproven"
    edited = registry.execute("edit_text", {
        "root": "skill_payload", "path": "notes.txt", "old_str": "shell edit", "new_str": "file edit",
    })
    assert "Replaced" in edited, edited
    assert load_repair_admission(data, "demo")["expected_content_hash"] == compute_content_hash(payload)
    assert load_repair_admission(data, "demo")["revision_attribution"] == "file_tool"


@pytest.mark.parametrize("tool", ["write_file", "edit_text", "run_command"])
def test_known_drift_refuses_every_selected_writer(tmp_path, monkeypatch, tool):
    registry, _repo, data = _registry(tmp_path, monkeypatch, mode="advanced")
    payload = _skill(data, "demo", bucket="external")
    registry._ctx.task_id = "repair-demo"
    registry._ctx.task_constraint = TaskConstraint(skill_name="demo", payload_root="skills/external/demo")
    record_repair_admission(data, "demo", task_id="repair-demo", base_content_hash=compute_content_hash(payload))
    (payload / "notes.txt").write_text("foreign\n")
    args = {
        "write_file": {"root": "skill_payload", "path": "notes.txt", "content": "overwrite\n"},
        "edit_text": {"root": "skill_payload", "path": "notes.txt", "old_str": "foreign", "new_str": "overwrite"},
        "run_command": {"cwd": "skill_payload", "cmd": [sys.executable, "-c", "from pathlib import Path; Path('notes.txt').write_text('overwrite\\n')"]},
    }[tool]
    result = registry.execute(tool, args)
    assert "SKILL_REPAIR_STALE" in result, result
    assert (payload / "notes.txt").read_text() == "foreign\n"


def test_project_child_reads_canonical_admission_for_selected_write(tmp_path, monkeypatch):
    registry, _repo, data = _registry(tmp_path, monkeypatch, mode="advanced")
    payload = _skill(data, "demo", bucket="external")
    child = tmp_path / "child"
    child.mkdir()
    registry._ctx.drive_root = child
    registry._ctx.task_metadata = {"budget_drive_root": str(data)}
    registry._ctx.task_id = "repair-demo"
    registry._ctx.task_constraint = TaskConstraint(skill_name="demo", payload_root="skills/external/demo")
    record_repair_admission(data, "demo", task_id="repair-demo", base_content_hash=compute_content_hash(payload))
    result = registry.execute("write_file", {"root": "skill_payload", "path": "notes.txt", "content": "canonical\n"})
    assert result.startswith("OK:"), result
    assert (payload / "notes.txt").read_text() == "canonical\n"
    assert load_repair_admission(data, "demo")["expected_content_hash"] == compute_content_hash(payload)
    assert not (child / "state" / "skills" / "demo" / "repair_admission.json").exists()
