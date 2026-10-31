"""Real native writer handlers retain ordinary-folder outputs through copy-back."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess

import pytest

from ouroboros.artifacts import collect_task_artifact_records
from ouroboros.headless import copy_child_task_result, finalize_task_artifacts, task_artifacts_dir, retry_child_task_refs
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.task_status import load_effective_task_result
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests._store_shared import remove_store_tree


def _registry(tmp_path, monkeypatch, *, git=False, system_target=False):
    system, folder, child, parent = (tmp_path / name for name in ("system", "folder", "child", "parent"))
    for directory in (system, folder, child, parent):
        directory.mkdir()
    if git:
        subprocess.run(["git", "init", "-q", str(folder)], check=True)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "1")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    selected = system if system_target else folder
    constraint = {"mode": "acting_subagent", "surface": "external_workspace", "write_root": str(selected)}
    ctx = ToolContext(repo_dir=system, system_repo_dir=system, drive_root=child,
                      workspace_root=selected, workspace_mode="external", task_id="child1",
                      task_constraint=constraint,
                      task_metadata={"root_task_id": "parent1", "parent_task_id": "parent1",
                                     "budget_drive_root": str(parent)})
    registry = ToolRegistry(repo_dir=system, drive_root=child)
    registry.set_context(ctx)
    task = {"id": "child1", "workspace_root": str(selected), "workspace_mode": "external",
            "child_drive_root": str(child), "drive_root": str(child), "task_constraint": constraint,
            "parent_task_id": "parent1", "root_task_id": "parent1", "delegation_role": "subagent"}
    return registry, ctx, task, parent


def _finish(ctx, task, parent):
    # This is the worker's ordinary result collection, not explicit output registration.
    write_task_result(ctx.drive_root, ctx.task_id, "completed", result="Requested changes written",
                      workspace_root=str(ctx.workspace_root), task_constraint=task["task_constraint"],
                      parent_task_id="parent1", root_task_id="parent1", delegation_role="subagent",
                      artifacts=collect_task_artifact_records(ctx.drive_root, ctx.task_id))
    copied = copy_child_task_result(parent, task)
    copied = retry_child_task_refs(parent, ctx.drive_root, ctx.task_id)
    assert copied and copied["child_ref_promotion"]["status"] == "complete"
    finalize_task_artifacts(parent, task)
    remove_store_tree(ctx.drive_root)
    return load_effective_task_result(parent, ctx.task_id)


def test_all_native_writers_register_outputs_before_result_and_survive_reopen(tmp_path, monkeypatch):
    registry, ctx, task, parent = _registry(tmp_path, monkeypatch)
    folder = ctx.workspace_root
    for name in ("edited.txt", "patched.txt", "batched.txt"):
        (folder / name).write_text("before\n", encoding="utf-8")
    original_rglob = Path.rglob
    def no_folder_inventory(path, pattern, *args, **kwargs):
        if path == folder:
            pytest.fail("native post-write capture inventoried the owner folder")
        return original_rglob(path, pattern, *args, **kwargs)
    monkeypatch.setattr(Path, "rglob", no_folder_inventory)
    calls = [
        ("write_file", {"path": "written.txt", "content": "written\n"}),
        ("write_file", {"path": "written.txt", "content": "appended\n", "mode": "append"}),
        ("edit_text", {"path": "edited.txt", "old_str": "before", "new_str": "edited"}),
        ("apply_patch", {"patch": "*** Update File: patched.txt\n-before\n+patched\n"}),
        ("edit_batch", {"edits": [{"path": "batched.txt", "old_str": "before", "new_str": "batched"}]}),
    ]
    for tool, args in calls:
        result = str(registry.execute(tool, args))
        assert "Retained file outputs:" in result, result
        assert "emit a patch artifact" not in result
        assert "OUTPUT_CAPTURE_FAILED" not in result
    expected = {name: (folder / name).read_bytes() for name in
                ("written.txt", "edited.txt", "patched.txt", "batched.txt")}
    before = collect_task_artifact_records(ctx.drive_root, ctx.task_id)
    assert len(before) == 4
    assert (ctx.drive_root / "task_results" / "artifact_versions" / "child1" / "written.txt").is_dir()
    reopened = _finish(ctx, task, parent)
    assert reopened["artifact_status"] == "ready"
    manifest = json.loads((task_artifacts_dir(parent, ctx.task_id) / "workspace_patch.json").read_text())
    assert manifest["capture_kind"] == "directory_direct"
    assert len(manifest["registered_outputs"]) == 4
    for record in manifest["registered_outputs"]:
        source = Path(record["source_path"])
        assert source.parent == folder
        assert record["sha256"] == sha256(expected[source.name]).hexdigest()
        assert Path(record["path"]).read_bytes() == expected[source.name]
        assert Path(record["path"]).is_relative_to(parent)
    assert not (folder / ".git").exists()
    assert load_task_result(parent, ctx.task_id)["result"] == "Requested changes written"


@pytest.mark.parametrize("tool", ["write_file", "edit_text", "apply_patch", "edit_batch"])
def test_capture_failure_reports_already_written_file_without_retry(tmp_path, monkeypatch, tool):
    registry, ctx, task, parent = _registry(tmp_path, monkeypatch)
    target = ctx.workspace_root / "out.txt"
    target.write_bytes(b"before\n")
    calls = []
    def failed_copy(source, dest, **kwargs):
        calls.append(str(source))
        raise OSError("artifact destination unavailable")
    monkeypatch.setattr("ouroboros.artifacts.copy_artifact_file", failed_copy)
    args = {
        "write_file": {"path": "out.txt", "content": "after\n"},
        "edit_text": {"path": "out.txt", "old_str": "before", "new_str": "after"},
        "apply_patch": {"patch": "*** Update File: out.txt\n-before\n+after\n"},
        "edit_batch": {"edits": [{"path": "out.txt", "old_str": "before", "new_str": "after"}]},
    }[tool]
    result = str(registry.execute(tool, args))
    assert "OUTPUT_CAPTURE_FAILED" in result, result
    assert "writes remain applied" in result and "do not repeat" in result
    assert target.read_text(encoding="utf-8") == "after\n"
    assert calls == [str(target)]
    assert collect_task_artifact_records(ctx.drive_root, ctx.task_id) == []


@pytest.mark.parametrize("tool", ["write_file", "apply_patch", "edit_batch"])
def test_partial_batch_keeps_successful_file_artifact(tmp_path, monkeypatch, tool):
    registry, ctx, task, parent = _registry(tmp_path, monkeypatch)
    folder = ctx.workspace_root
    for name in ("first.txt", "second.txt"):
        (folder / name).write_text("before\n", encoding="utf-8")
    from ouroboros.tools import edit_ops, git
    writer_module = git if tool == "write_file" else edit_ops
    original = writer_module.write_text
    def write_once(path, content):
        if Path(path).name == "second.txt":
            raise OSError("second target unavailable")
        return original(path, content)
    monkeypatch.setattr(writer_module, "write_text", write_once)
    args = {
        "write_file": {"files": [{"path": name, "content": "after\n"} for name in ("first.txt", "second.txt")]},
        "apply_patch": {"patch": "*** Update File: first.txt\n-before\n+after\n*** Update File: second.txt\n-before\n+after\n"},
        "edit_batch": {"edits": [{"path": name, "old_str": "before", "new_str": "after"} for name in ("first.txt", "second.txt")]},
    }[tool]
    result = str(registry.execute(tool, args))
    assert "first.txt" in result and "Retained file outputs:" in result, result
    assert (folder / "first.txt").read_text() == "after\n"
    assert (folder / "second.txt").read_text() == "before\n"
    records = collect_task_artifact_records(ctx.drive_root, ctx.task_id)
    assert [Path(row["source_path"]).name for row in records] == ["first.txt"]
    assert records[0]["sha256"] == sha256(b"after\n").hexdigest()


@pytest.mark.parametrize("git,system_target", [(True, False), (False, True)])
def test_git_and_self_repo_writes_keep_existing_artifact_behavior(tmp_path, monkeypatch, git, system_target):
    registry, ctx, task, parent = _registry(tmp_path, monkeypatch, git=git, system_target=system_target)
    if system_target:
        ctx.task_constraint = None
        ctx.workspace_root = None
        ctx.workspace_mode = ""
    monkeypatch.setattr("ouroboros.artifacts.copy_file_to_task_artifacts",
                        lambda *a, **k: pytest.fail("Git or system write unexpectedly captured full file"))
    result = str(registry.execute("write_file", {"path": "unchanged-contract.txt", "content": "text\n"}))
    assert "Written 1 file" in result, result
    assert "Retained file outputs:" not in result
    assert collect_task_artifact_records(ctx.drive_root, ctx.task_id) == []


@pytest.mark.parametrize("mode", ["overwrite", "append"])
@pytest.mark.parametrize("failure", ["", "write", "capture"])
def test_batch_retains_each_success_with_one_capture_preamble(tmp_path, monkeypatch, mode, failure):
    registry, ctx, _, _ = _registry(tmp_path, monkeypatch)
    names = ("first.txt", "second.txt", "third.txt")
    for name in names:
        target = ctx.workspace_root / name
        if failure == "write" and name == "third.txt":
            target.mkdir()  # Both write and append fail after the first two writes.
        else:
            target.write_bytes(b"before\n")
    if failure == "capture":
        from ouroboros import artifacts
        original_copy = artifacts.copy_artifact_file

        def copy_except_second(source, destination, **kwargs):
            if Path(source).name == "second.txt":
                raise OSError("second artifact unavailable")
            return original_copy(source, destination, **kwargs)

        monkeypatch.setattr(artifacts, "copy_artifact_file", copy_except_second)
    result = str(registry.execute("write_file", {
        "files": [{"path": name, "content": "after\n"} for name in names], "mode": mode,
    }))
    assert result.count("no separate patch apply is needed") == 1, result
    expected = b"before\nafter\n" if mode == "append" else b"after\n"
    written = names[:2] if failure == "write" else names
    for name in written:
        assert (ctx.workspace_root / name).read_bytes() == expected
    retained = set(written) - ({"second.txt"} if failure == "capture" else set())
    records = collect_task_artifact_records(ctx.drive_root, ctx.task_id)
    assert {Path(row["source_path"]).name for row in records} == retained
    for row in records:
        assert Path(row["path"]).read_bytes() == expected
        assert row["sha256"] == sha256(expected).hexdigest()
        assert f"artifact_store:{row['name']}, {len(expected)} bytes, sha256={row['sha256']}" in result
    if failure == "write":
        assert "FILE_WRITE_ERROR" in result and "Successfully written before error" in result
    if failure == "capture":
        assert "OUTPUT_CAPTURE_FAILED: second.txt" in result
        assert "writes remain applied; do not repeat" in result
