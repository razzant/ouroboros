"""Owner-control names remain ordinary data in a useful process operation."""
from __future__ import annotations

import pathlib
import sys

import pytest

from ouroboros.artifacts import registered_task_artifact
from ouroboros.tools.registry import ToolContext, ToolRegistry

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("example", [
    "ouroboros.config.save_settings({'OUROBOROS_RUNTIME_MODE': 'pro'})",
    "OUROBOROS_CONTEXT_MODE /api/owner/context-mode",
    "OUROBOROS_SAFETY_MODE settings.json /api/owner/safety-mode",
    "/api/owner/skills/demo/attest-review",
    "OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS settings.json",
    "OUROBOROS_POST_TASK_EVOLUTION state/skills/demo/review.json",
])
def test_owner_control_example_reaches_a_registered_result(tmp_path, monkeypatch, example):
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    drive.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id="control-example")
    reg = ToolRegistry(repo_dir=repo, drive_root=drive)
    reg.set_context(ctx)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    script = "import sys; from pathlib import Path; Path('example.txt').write_text(sys.argv[1], encoding='utf-8')"

    result = reg.execute("run_command", {
        "cmd": [sys.executable, "-c", script, example],
        "cwd": "task_drive", "outputs": ["example.txt"],
    })

    assert "ARTIFACT_OUTPUTS" in result, result
    record = registered_task_artifact(drive, ctx.task_id, "example.txt")
    assert record and pathlib.Path(record["path"]).read_text(encoding="utf-8") == example


def test_inplace_editor_is_not_a_pure_read_inspection():
    from ouroboros.tools.registry import _is_pure_read_inspection

    assert not _is_pure_read_inspection("yq -i '.mode = off' settings.json")
    assert not _is_pure_read_inspection("yq --inplace '.mode = low' settings.json")
    assert _is_pure_read_inspection("yq '.mode' settings.json")
    assert _is_pure_read_inspection("jq '.mode' settings.json")
