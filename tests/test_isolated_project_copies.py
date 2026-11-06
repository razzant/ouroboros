"""Task-owned current-tree copies across source identity and both apply levels."""
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from ouroboros import subagent_worktrees as trees
from ouroboros.artifacts import task_artifact_dir_path
from ouroboros.task_results import write_task_result
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.subagent_integration import _integrate_subagent_patch
from ouroboros.workspace_copies import admitted_copy_metadata, is_system_copy
from ouroboros.workspace_patch_capture import write_workspace_patch_artifacts
from supervisor.events_subagent_admission import _resolve_subagent_constraint


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout


def repo(path, files):
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.name", "Fixture")
    git(path, "config", "user.email", "fixture@example.invalid")
    for name, content in files.items():
        (path / name).write_bytes(content)
    git(path, "add", "-A")
    git(path, "commit", "-qm", "fixture")
    return path


@pytest.fixture
def case(tmp_path, monkeypatch):
    system = repo(tmp_path / "system", {"system.txt": b"own body\n"})
    source = repo(tmp_path / "project", {"a.txt": b"base\n", "BIBLE.md": b"project docs\n"})
    data, copies = tmp_path / "data", tmp_path / "copies"
    data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(copies))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.delenv("OUROBOROS_BOOT_RUNTIME_MODE", raising=False)
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "")
    return system, source, data, copies


def admit(case, task_id, parent_id, source=None):
    system, project, data, _ = case
    source = source or project
    constraint, folder, mode, error = _resolve_subagent_constraint(
        SimpleNamespace(REPO_DIR=system, DRIVE_ROOT=data), tid=task_id,
        requested_constraint={"mode": "acting_subagent", "surface": "self_worktree"},
        workspace_root=str(source), workspace_mode="external", base_sha="", parent_task_id=parent_id)
    assert not error, error
    binding = admitted_copy_metadata(folder)
    task = {"id": task_id, "parent_task_id": parent_id,
            "workspace_root": folder, "workspace_mode": mode,
            "task_constraint": constraint, "workspace_copy": binding,
            "metadata": {"workspace_copy": binding}}
    return Path(folder), task


def capture(case, folder, task):
    data = case[2]
    artifacts = task_artifact_dir_path(data, task["id"])
    artifacts.mkdir(parents=True, exist_ok=True)
    _, manifest = write_workspace_patch_artifacts(folder, artifacts, task=task)
    write_task_result(data, task["id"], "completed", **{k: v for k, v in task.items() if k != "id"})
    return artifacts, manifest


def parent_context(case, task_id="parent", source=None, task=None):
    system, project, data, _ = case
    return ToolContext(repo_dir=system, drive_root=data, task_id=task_id,
                       workspace_root=source or project,
                       workspace_mode=task["workspace_mode"] if task else "external",
                       task_constraint=task["task_constraint"] if task else None,
                       task_metadata=task["metadata"] if task else {})


def test_current_staged_unstaged_new_and_binary_inputs_return_only_child_delta(case):
    _, source, data, _ = case
    (source / "a.txt").write_bytes(b"base\nparent staged\n")
    git(source, "add", "a.txt")
    (source / "a.txt").write_bytes(b"base\nparent staged\nparent unstaged\n")
    (source / "new.txt").write_bytes(b"new parent input\n")
    (source / "picture.bin").write_bytes(b"parent\x00image")
    head, index, status = git(source, "rev-parse", "HEAD"), (source / ".git/index").read_bytes(), git(source, "status", "--porcelain")
    folder, task = admit(case, "child", "parent")
    assert not (folder / "system.txt").exists()
    assert (folder / "a.txt").read_bytes() == (source / "a.txt").read_bytes()
    assert (folder / "new.txt").read_bytes() == b"new parent input\n"
    assert (folder / "picture.bin").read_bytes() == b"parent\x00image"
    assert git(source, "rev-parse", "HEAD") == head
    assert (source / ".git/index").read_bytes() == index
    assert git(source, "status", "--porcelain") == status
    assert trees.prune_execution_snapshots(set())["removed"] == []
    assert folder.exists()  # task-owned copy is never delegated-run garbage
    (folder / "a.txt").write_bytes((folder / "a.txt").read_bytes() + b"child\n")
    (folder / "picture.bin").write_bytes(b"child\x00image")
    artifacts, manifest = capture(case, folder, task)
    patch = (artifacts / "workspace.patch").read_text(encoding="utf-8")
    assert "+child" in patch and "+parent staged" not in patch and "+parent unstaged" not in patch
    assert "new.txt" not in manifest["untracked_included"]
    assert (source / "picture.bin").read_bytes() == b"parent\x00image"
    result = _integrate_subagent_patch(parent_context(case), task_id="child")
    assert "✅ Integrated" in result, result
    assert (source / "a.txt").read_bytes() == b"base\nparent staged\nparent unstaged\nchild\n"
    assert (source / "picture.bin").read_bytes() == b"child\x00image"
    assert (source / "new.txt").read_bytes() == b"new parent input\n"
    assert git(source, "rev-parse", "HEAD") == head
    assert git(source, "diff", "--cached", "--name-only").splitlines() == [b"a.txt", b"picture.bin"]


def test_nested_copy_sees_parent_work_and_bubbles_child_delta(case):
    _, source, _, _ = case
    child, child_task = admit(case, "child", "parent")
    (child / "a.txt").write_bytes(b"base\nchild\n")
    grand, grand_task = admit(case, "grand", "child", child)
    assert (grand / "a.txt").read_bytes() == b"base\nchild\n"
    (grand / "a.txt").write_bytes(b"base\nchild\ngrand\n")
    capture(case, grand, grand_task)
    child_ctx = parent_context(case, "child", child, child_task)
    out = _integrate_subagent_patch(child_ctx, task_id="grand")
    assert "✅ Integrated" in out, out
    assert (source / "a.txt").read_bytes() == b"base\n"
    capture(case, child, child_task)
    out = _integrate_subagent_patch(parent_context(case), task_id="child")
    assert "✅ Integrated" in out, out
    assert (source / "a.txt").read_bytes() == b"base\nchild\ngrand\n"
    assert trees.remove_worktree(task_id="child")
    assert trees.remove_worktree(task_id="grand")
    assert git(source, "branch", "--list", "subagent/*") == b""


@pytest.mark.parametrize("mode", ["light", "advanced", "cyber_pro"])
def test_foreign_copy_uses_foreign_policy_in_every_mode(case, monkeypatch, mode):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    folder, task = admit(case, "child", "parent")
    assert task["workspace_copy"]["source_is_system_repo"] is False
    assert not is_system_copy(parent_context(case, "child", folder, task))
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.process_interpreters import _surface_for
    from ouroboros.tool_access import build_resolved_resource_binding
    from ouroboros.contracts.task_constraint import normalize_task_constraint
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **kw: (True, ""))
    child_ctx = parent_context(case, "child", folder, task)
    registry = ToolRegistry(repo_dir=case[0], drive_root=case[2])
    registry.set_context(child_ctx)
    result = registry.execute("write_file", {"root": "active_workspace", "path": "BIBLE.md", "content": "ordinary foreign project file\n"})
    assert (folder / "BIBLE.md").read_bytes() == b"ordinary foreign project file\n", result
    if mode != "cyber_pro":  # Existing Cyber authority does not preempt the shell.
        blocked = registry.execute("run_command", {"cmd": "git commit -am forbidden"})
        assert "WORKSPACE_GIT_BLOCKED" in str(blocked), blocked
    binding = build_resolved_resource_binding(child_ctx, root="active_workspace", path=".", operation="shell")
    assert _surface_for(child_ctx, binding, normalize_task_constraint(task["task_constraint"])) == "external_workspace"
    capture(case, folder, task)
    out = _integrate_subagent_patch(parent_context(case), task_id="child")
    assert "✅ Integrated" in out, out


def test_source_drift_and_wrong_parent_keep_captured_copy(case):
    _, source, _, _ = case
    (source / "a.txt").write_bytes(b"parent uncommitted\n")
    folder, task = admit(case, "child", "parent")
    (folder / "a.txt").write_bytes(b"child\n")
    artifacts, _ = capture(case, folder, task)
    (source / "a.txt").write_bytes(b"concurrent parent\n")
    out = _integrate_subagent_patch(parent_context(case), task_id="child")
    assert "INTEGRATE_CONFLICT" in out and "changed since copy" in out
    assert (source / "a.txt").read_bytes() == b"concurrent parent\n"
    assert folder.exists() and (artifacts / "workspace.patch").exists()
    wrong = parent_context(case, source=case[0])
    assert "INTEGRATE_TARGET_FORBIDDEN" in _integrate_subagent_patch(wrong, task_id="child", target_root=str(case[0]))


def test_sibling_leaf_snapshot_and_actual_recursion_check(case):
    _, source, data, copies = case
    folder, task = admit(case, "child", "parent")
    leaf = trees.provision_execution_snapshot(target_root=folder, task_id="child", snapshot_id="leaf")
    assert Path(leaf.path).parent == folder.parent
    assert (Path(leaf.path) / "a.txt").read_bytes() == b"base\n"
    with pytest.raises(ValueError, match="overlaps"):
        trees.provision_execution_snapshot(target_root=folder, task_id="nested", snapshot_id="recursive",
                                            worktree_root=folder / "inside", data_dir=data)
    with pytest.raises(ValueError, match="overlaps"):
        trees.provision_worktree(repo_dir=folder, task_id="child", worktree_root=copies, data_dir=data)
    assert (folder / "a.txt").read_bytes() == b"base\n"
    assert (source / "a.txt").read_bytes() == b"base\n"
    assert trees.remove_worktree(task_id="child")
    assert Path(leaf.path).exists()  # shared task id does not select the leaf's row
    assert trees.find_execution_snapshot("leaf") is not None
    assert trees.remove_execution_snapshot("leaf")
    assert git(source, "for-each-ref", "--format=%(refname)", "refs/ouroboros/delegated/") == b""


def test_legacy_copy_keeps_own_body_identity(case):
    folder, task = admit(case, "child", "parent", case[0])
    ctx = parent_context(case, "child", folder, task)
    assert is_system_copy(ctx)
    ctx.task_metadata = {}
    assert is_system_copy(ctx)


def test_task_copy_refusal_never_initializes_plain_folder(case):
    plain = case[1].parent / "plain"
    plain.mkdir()
    with pytest.raises(ValueError, match="not a git working tree"):
        trees.provision_worktree(repo_dir=plain, task_id="plain")
    assert not (plain / ".git").exists()


def test_real_queued_terminal_and_copyback_preserve_source_binding(case, monkeypatch):
    from ouroboros.agent_task_pipeline import _store_task_result
    from ouroboros.headless import copy_child_task_result, finalize_task_artifacts
    from ouroboros.task_results import load_task_result
    from supervisor.task_dispatch import build_scheduled_task_payload

    folder, admitted = admit(case, "child", "parent")
    child_drive = case[2].parent / "child-drive"
    child_drive.mkdir()
    queued = build_scheduled_task_payload({
        "tid": "child", "parent_id": "parent", "desc": "Change the project", "text": "Change it",
        "expected_output": "patch", "role": "builder", "depth": 1,
        "root_task_id": "parent", "delegation_role": "subagent", "memory_mode": "empty",
        "drive_root": str(child_drive), "child_drive_root": str(child_drive),
        "budget_drive_root": str(case[2]), "workspace_root": str(folder),
        "workspace_mode": "self_worktree", "task_constraint": admitted["task_constraint"],
        "workspace_copy": admitted["workspace_copy"],
    })
    assert queued["metadata"]["workspace_copy"] == admitted["workspace_copy"]
    from ouroboros import agent as agent_module
    from ouroboros.agent import Env, OuroborosAgent
    from tests.test_available_subagents_runtime import _api_row, _settings, _snapshot
    import ouroboros.provider_models as provider_models
    queued["configured_subagent"] = _snapshot(_settings(_api_row()), "api-builder")
    monkeypatch.setattr(provider_models, "model_has_credentials", lambda _model: True)
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr(agent_module, "build_llm_messages", lambda **kw: ([], {}))
    agent = OuroborosAgent(Env(repo_dir=case[0], drive_root=child_drive))
    actual_ctx, _, _ = agent._prepare_task_context(queued)
    assert actual_ctx.task_metadata["workspace_copy"] == admitted["workspace_copy"]
    assert actual_ctx.active_repo_dir() == folder
    (folder / "a.txt").write_bytes(b"terminal chain\n")
    _store_task_result(SimpleNamespace(drive_root=child_drive), queued, "done", {}, {},
                       loop_outcome={"reason_code": "completed"})
    child_result = load_task_result(child_drive, "child")
    assert child_result["metadata"]["workspace_copy"] == admitted["workspace_copy"]
    copied = copy_child_task_result(case[2], queued)
    assert copied["metadata"]["workspace_copy"] == admitted["workspace_copy"]
    finalize_task_artifacts(case[2], queued)
    out = _integrate_subagent_patch(parent_context(case), task_id="child")
    assert "✅ Integrated" in out, out
    assert (case[1] / "a.txt").read_bytes() == b"terminal chain\n"


def test_apply_exception_is_unknown_and_keeps_copy(case, monkeypatch):
    import ouroboros.tools.subagent_integration as integration
    folder, task = admit(case, "child", "parent")
    (folder / "a.txt").write_bytes(b"changed\n")
    artifacts, _ = capture(case, folder, task)
    def broken(*a, **kw):
        raise OSError("lost filesystem answer")
    monkeypatch.setattr(integration, "_locked_apply", broken)
    out = integration._integrate_subagent_patch(parent_context(case), task_id="child")
    assert "INTEGRATE_APPLY_UNKNOWN" in out
    assert folder.exists() and (artifacts / "workspace.patch").exists()


def test_a_busy_target_is_not_started_and_the_same_result_integrates_later(case, monkeypatch):
    """06.11, both directions: a target another git step holds answers INTEGRATE_NOT_STARTED with no conflict verdict
    and keeps the captured result; once free, the same result integrates."""
    import ouroboros.tools.subagent_integration as integration
    folder, task = admit(case, "child", "parent")
    (folder / "a.txt").write_bytes(b"changed\n")
    artifacts, _ = capture(case, folder, task)
    real = integration._locked_apply
    monkeypatch.setattr(integration, "_locked_apply", lambda *a, **kw: {"lock_error": "TimeoutError: held by task other"})
    out = integration._integrate_subagent_patch(parent_context(case), task_id="child")
    assert "INTEGRATE_NOT_STARTED" in out and "the target is busy" in out and "CONFLICT" not in out, out
    assert folder.exists() and (artifacts / "workspace.patch").exists()
    monkeypatch.setattr(integration, "_locked_apply", real)
    out = integration._integrate_subagent_patch(parent_context(case), task_id="child")
    assert "✅ Integrated" in out, out
    assert (case[1] / "a.txt").read_bytes() == b"changed\n"


def ignore_fixture(source, *, dirty):
    """The source ignores .build/ and dist/ yet tracks .build/pinned.cfg; a dirty source also holds the
    parent's own staged and unstaged edit of a.txt, which sends its copy through the working-tree apply."""
    (source / ".gitignore").write_bytes(b".build/\ndist/\n")
    (source / ".build").mkdir()
    (source / ".build" / "pinned.cfg").write_bytes(b"base\n")
    git(source, "add", "-f", "--", ".gitignore", ".build/pinned.cfg")
    git(source, "commit", "-qm", "ignore fixture")
    if dirty:
        (source / "a.txt").write_bytes(b"base\nparent staged\n")
        git(source, "add", "a.txt")
        (source / "a.txt").write_bytes(b"base\nparent staged\nparent unstaged\n")


def index_state(source):
    """Entries, staged diff and what Git reads as modified: a restored entry must not look dirty."""
    return (git(source, "ls-files", "--stage", "-z"), git(source, "diff", "--cached", "--binary"),
            git(source, "diff-files", "--name-only", "-z"))


@pytest.mark.parametrize("dirty", [False, True])  # three-way apply from a clean source, or working-tree apply
@pytest.mark.parametrize("kind", ["tracked_in_ignored_folder", "new_ignored", "ordinary"])
def test_ignored_paths_are_delivered_and_staged_by_index_membership(case, dirty, kind):
    """#2446: a file the source already tracks stages whatever its ignore rules say; a new file they exclude
    is delivered into the tree but kept out of the index and named; the ordinary edit and deletion beside it
    land either way (a three-way apply stages that deletion itself), and the parent's own staged and unstaged
    work stays as it was."""
    import json

    _, source, data, _ = case
    ignore_fixture(source, dirty=dirty)
    child, task = admit(case, "child", "parent")
    (child / "README.txt").write_bytes(b"ordinary contribution\n")
    (child / "BIBLE.md").unlink()
    rel = {"tracked_in_ignored_folder": ".build/pinned.cfg", "new_ignored": "dist/new.bin"}.get(kind)
    if rel:
        (child / rel).parent.mkdir(parents=True, exist_ok=True)
        (child / rel).write_bytes(b"captured contribution\n")
        git(child, "add", "-f", "--", rel)
    _, manifest = capture(case, child, task)
    assert manifest["status"] == "ready_with_changes", manifest
    out = _integrate_subagent_patch(parent_context(case), task_id="child")
    assert "✅ Integrated" in out, out
    assert (source / "README.txt").read_bytes() == b"ordinary contribution\n"
    assert git(source, "show", ":README.txt") == b"ordinary contribution\n"
    assert not (source / "BIBLE.md").exists() and git(source, "ls-files", "--", "BIBLE.md") == b""
    if rel:
        assert (source / rel).read_bytes() == b"captured contribution\n"
    if kind == "tracked_in_ignored_folder":
        assert git(source, "show", ":.build/pinned.cfg") == b"captured contribution\n"
    verdict = json.loads((task_artifact_dir_path(data, "parent") / "subagent_patch_verdict_child.json").read_text())
    if kind == "new_ignored":
        assert git(source, "ls-files", "--", "dist/new.bin") == b""
        assert "3 file(s), 2 staged)" in out and "Delivered, not indexed" in out and "dist/new.bin" in out, out
        assert verdict["unindexed"] == ["dist/new.bin"]
        assert "Changes are staged" not in out and "the files named above are not staged at all" in out, out
    else:
        assert "file(s), staged)" in out and "Delivered, not indexed" not in out and "unindexed" not in verdict, out
        assert "Changes are staged but NOT committed" in out, out
    if dirty:  # the parent's own work: staged blob and unstaged bytes, untouched by the child's result
        assert git(source, "show", ":a.txt") == b"base\nparent staged\n"
        assert (source / "a.txt").read_bytes() == b"base\nparent staged\nparent unstaged\n"


def index_flags(source):
    """The flags Git keeps beside entries (lowercase tag: assume-unchanged, S: skip-worktree) and the
    intent-to-add entries porcelain status reports; read without writing the index."""
    return (git(source, "ls-files", "-v", "-z"),
            git(source, "--no-optional-locks", "status", "--porcelain=v2", "-z", "--untracked-files=no"))


@pytest.mark.parametrize("dirty, flag", [(False, ""), (True, ""), (True, "intent_to_add"),
                                         (False, "skip_worktree")])
def test_a_genuine_index_failure_reverses_the_tree_and_restores_only_its_own_entries(case, monkeypatch, dirty, flag):
    """A real Git index failure while staging (another Git process holds index.lock) still fails the apply,
    which is reversed; exactly the entries the attempt changed return, a three-way apply's own included,
    with the flags Git kept beside them: intent-to-add (a working-tree apply stages its content first),
    assume-unchanged and skip-worktree (staging clears them, a three-way apply rewrites those entries without
    them). The index, the parent's staged and unstaged work and the tree are as they were."""
    import json
    import os
    from ouroboros.tools import patch_staging as staging

    _, source, data, _ = case
    ignore_fixture(source, dirty=dirty)
    held = "ita.txt" if flag == "intent_to_add" else "BIBLE.md" if flag else ""
    if flag == "intent_to_add":
        (source / held).write_bytes(b"intended\n")
        git(source, "add", "-N", "--", held)
    elif flag:
        git(source, "update-index", f"--{flag.replace('_', '-')}", "--", held)
    if flag == "skip_worktree" and not dirty:
        (source / held).unlink()  # outside a sparse checkout the file is absent
    child, task = admit(case, "child", "parent")
    (child / "a.txt").write_bytes((child / "a.txt").read_bytes() + b"child\n")
    (child / "new.txt").write_bytes(b"child's new file\n")
    if held:
        (child / held).write_bytes(b"edited by the child\n")
    _, manifest = capture(case, child, task)
    assert not held or held in manifest["tracked_changed"] + manifest["untracked_included"], manifest
    os.utime(source / ".gitignore", (946684800, 946684800))  # unrelated, stale in the stat cache: never refreshed
    assert b".gitignore" in git(source, "diff-files", "--name-only").split()
    present = lambda: held and (source / held).exists()  # noqa: E731 (a sparse file stays absent)
    before = index_state(source), index_flags(source), (source / "a.txt").read_bytes(), present()
    real = staging._git_paths

    def locked(target, args, paths, prefix=b""):
        if args[1:2] == ["add"] and not {"-u", "-N", "--refresh"} & set(args):  # the new-file add meets a held lock
            lock = Path(target) / ".git" / "index.lock"
            lock.write_bytes(b"")
            try:
                return real(target, args, paths, prefix)
            finally:
                lock.unlink()
        return real(target, args, paths, prefix)

    monkeypatch.setattr(staging, "_git_paths", locked)
    out = _integrate_subagent_patch(parent_context(case), task_id="child")
    assert "INTEGRATE_CONFLICT" in out and "did not finish cleanly" in out and "index.lock" in out, out
    assert (index_state(source), index_flags(source), (source / "a.txt").read_bytes(), present()) == before
    assert not (source / "new.txt").exists()
    verdict = json.loads((task_artifact_dir_path(data, "parent") / "subagent_patch_verdict_child.json").read_text())
    assert (verdict["outcome"], verdict["applied"]) == ("conflict", False)
    monkeypatch.setattr(staging, "_git_paths", real)  # the lock is gone: the same captured result now lands
    assert "✅ Integrated" in _integrate_subagent_patch(parent_context(case), task_id="child")
    assert git(source, "show", ":new.txt") == b"child's new file\n"


@pytest.mark.parametrize("dirty, flag, absent, lands", [
    (True, "assume_unchanged", False, True),  # plain `add -u` skips this silently: unstaged under a "staged" receipt
    (False, "skip_worktree", False, True),  # a present file the index's stat data still matches lands the same way
    (True, "skip_worktree", True, False),  # a working-tree apply has no file to patch
])
def test_a_flagged_entry_really_stages_and_keeps_its_flag(case, dirty, flag, absent, lands):
    """A captured edit of an assume-unchanged or skip-worktree entry reaches the index and the entry keeps its
    flag, or nothing changes and the refusal says so; "staged" is never claimed over unstaged content."""
    _, source, _, _ = case
    ignore_fixture(source, dirty=dirty)
    git(source, "update-index", f"--{flag.replace('_', '-')}", "--", "BIBLE.md")
    if absent:
        (source / "BIBLE.md").unlink()
    tag = {"assume_unchanged": b"h", "skip_worktree": b"S"}[flag]
    child, task = admit(case, "child", "parent")
    (child / "BIBLE.md").write_bytes(b"edited by the child\n")
    (child / "README.txt").write_bytes(b"ordinary\n")
    capture(case, child, task)
    before = index_state(source)
    out = _integrate_subagent_patch(parent_context(case), task_id="child")
    assert git(source, "ls-files", "-v", "--", "BIBLE.md")[:1] == tag
    if not lands:
        assert "INTEGRATE_CONFLICT" in out and "✅" not in out, out
        assert index_state(source) == before and not (source / "README.txt").exists()
        assert (source / "BIBLE.md").exists() != absent
        return
    assert "✅ Integrated" in out and "2 file(s), staged)" in out and "Changes are staged but NOT committed" in out, out
    assert git(source, "show", ":BIBLE.md") == (source / "BIBLE.md").read_bytes() == b"edited by the child\n"
    assert git(source, "show", ":README.txt") == b"ordinary\n"
    if dirty:  # the parent's own work, on an entry the child did not touch
        assert git(source, "show", ":a.txt") == b"base\nparent staged\n"
        assert (source / "a.txt").read_bytes() == b"base\nparent staged\nparent unstaged\n"


def test_a_hidden_local_edit_stages_with_the_entry_the_child_edited(case):
    """A touched entry stages whole, as any touched path with unstaged edits does: the child saw and kept the
    local edit an assume-unchanged flag hid, and that receipt is true because the index holds exactly that."""
    _, source, _, _ = case
    git(source, "update-index", "--assume-unchanged", "--", "BIBLE.md")
    (source / "BIBLE.md").write_bytes(b"project docs\nhidden local edit\n")
    child, task = admit(case, "child", "parent")
    (child / "BIBLE.md").write_bytes((child / "BIBLE.md").read_bytes() + b"child line\n")
    capture(case, child, task)
    out = _integrate_subagent_patch(parent_context(case), task_id="child")
    assert "✅ Integrated" in out and "1 file(s), staged)" in out, out
    assert git(source, "show", ":BIBLE.md") == b"project docs\nhidden local edit\nchild line\n"
    assert git(source, "ls-files", "-v", "--", "BIBLE.md")[:1] == b"h"


@pytest.mark.parametrize("dirty", [False, True])
def test_the_ignore_rules_the_apply_leaves_decide_for_new_paths(case, dirty):
    """Ignore rules are read after the apply: a rule the result adds keeps its new file delivered but unindexed,
    and a rule the result removes lets its new file stage."""
    _, source, _, _ = case
    (source / ".gitignore").write_bytes(b"gone-rule/\n")
    git(source, "add", "--", ".gitignore")
    git(source, "commit", "-qm", "rules")
    if dirty:
        (source / "a.txt").write_bytes(b"base\nparent staged\n")
        git(source, "add", "--", "a.txt")
    child, task = admit(case, "child", "parent")
    (child / ".gitignore").write_bytes(b"added-rule/\n")
    for rel in ("added-rule/a.bin", "gone-rule/b.bin"):
        (child / rel).parent.mkdir()
        (child / rel).write_bytes(b"contribution\n")
    git(child, "add", "-f", "--", "added-rule/a.bin")
    capture(case, child, task)
    out = _integrate_subagent_patch(parent_context(case), task_id="child")
    assert "✅ Integrated" in out and "3 file(s), 2 staged)" in out and "added-rule/a.bin" in out, out
    assert git(source, "show", ":.gitignore") == b"added-rule/\n"
    assert git(source, "show", ":gone-rule/b.bin") == b"contribution\n"
    assert (source / "added-rule/a.bin").read_bytes() == b"contribution\n"
    assert git(source, "ls-files", "--", "added-rule") == b""


@pytest.mark.skipif(__import__("os").name == "nt", reason="NTFS forbids these names")
def test_odd_names_reach_git_literally(case):
    """Paths are literal and NUL-delimited: a glob-shaped name stages itself and never the file its pattern
    matches, whose unstaged edit stays unstaged; a leading colon or a newline stays part of the name, in the
    ignore check too."""
    import json

    _, source, data, _ = case
    (source / ".gitignore").write_bytes(b":ignored*\n")
    (source / "gb.txt").write_bytes(b"base\n")
    git(source, "add", "--", ".gitignore", "gb.txt")
    git(source, "commit", "-qm", "names")
    (source / "gb.txt").write_bytes(b"base\nparent's unstaged edit\n")
    child, task = admit(case, "child", "parent")
    staged = ["g[ab].txt", ":colon.txt", "new\nline.txt"]
    for name in (*staged, ":ignored.txt"):
        (child / name).write_bytes(b"contribution\n")
    git(child, "--literal-pathspecs", "add", "-f", "--", ":ignored.txt")
    capture(case, child, task)
    out = _integrate_subagent_patch(parent_context(case), task_id="child")
    assert "✅ Integrated" in out and "4 file(s), 3 staged)" in out, out
    listed = git(source, "ls-files", "-z").split(b"\0")
    assert all(name.encode() in listed for name in staged) and b":ignored.txt" not in listed
    assert (source / ":ignored.txt").read_bytes() == b"contribution\n"
    assert git(source, "show", ":gb.txt") == b"base\n"
    verdict = json.loads((task_artifact_dir_path(data, "parent") / "subagent_patch_verdict_child.json").read_text())
    assert verdict["unindexed"] == [":ignored.txt"]


@pytest.mark.parametrize("options,refused", [
    ({"directory_strategy": "copy", "scope_paths": ["."]}, True),
    ({"directory_strategy": "direct", "scope_paths": []}, False),
    ({}, False),
])
def test_schedule_git_geometry_fails_before_child_queue_and_keeps_default(case, monkeypatch, options, refused):
    from ouroboros.tools import control
    from ouroboros.tools.control_scheduling import _schedule_task
    from tests.test_available_subagents_runtime import _session_row, _settings

    settings = _settings(_session_row())
    monkeypatch.setattr(control, "load_settings", lambda: settings)
    ctx = parent_context(case)
    out = _schedule_task(ctx, subagent_id="session-builder", objective="Edit project",
                         expected_output="patch", memory_mode="empty", write_surface="self_worktree", **options)
    if refused:
        assert "Directory options apply to ordinary folders" in str(out), out
        assert ctx.pending_events == []
    else:
        assert "queued" in str(out), out
        assert any(evt.get("type") == "schedule_subagent" for evt in ctx.pending_events)


@pytest.mark.parametrize("own_body", [False, True])
def test_real_leaf_start_capture_child_apply_parent_apply(case, monkeypatch, own_body):
    import json
    from ouroboros import delegate_custody as custody
    from ouroboros.gateways import claudexor
    from ouroboros.tools import delegate
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch
    from tests.test_delegated_directory import DirectoryEngine
    from tests._delegated_transport_shared import _transport_snapshot
    from ouroboros import claudexor_daemon, subagent_runtime, subagents

    source = case[0] if own_body else case[1]
    source_name = "system.txt" if own_body else "a.txt"
    folder, task = admit(case, "child", "parent", source)
    ctx = parent_context(case, "child", folder, task)
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    engine = DirectoryEngine(folder)
    monkeypatch.setattr(claudexor, "ClaudexorGateway", lambda *a, **k: engine)
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: engine)
    token = subagent_runtime._EXACT_START_SELECTION.set({"snapshot": _transport_snapshot(subagents.get_subagent_harness())})
    custody._CUSTODY.clear()
    try:
        started = json.loads(delegate._delegate_start(ctx, "Edit the assigned file").text)
    finally:
        subagent_runtime._EXACT_START_SELECTION.reset(token)
    assert started["status"] == "started", started
    wire = engine.posts[0][0]
    execution = Path(wire["execution"]["workspaceRoot"])
    assert execution.parent == folder.parent and execution != folder
    assert wire["scope"]["root"] == str(folder)
    assert (execution / source_name).read_bytes() == (source / source_name).read_bytes()
    (execution / source_name).write_bytes(b"session contribution\n")
    entry = custody._CUSTODY[started["run_id"]]
    entry.settled = True
    assert custody.emit(case[2], custody.SETTLED, {"run_id": entry.run_id, "task_id": "child", "state": "succeeded"})
    capture_result = delegate._capture_terminal_patch(ctx, entry)
    assert capture_result["status"] == "ready_with_changes", capture_result
    assert (folder / source_name).read_bytes() != b"session contribution\n"
    assert (source / source_name).read_bytes() != b"session contribution\n"
    out = _integrate_delegated_patch(ctx, started["run_id"], "apply", "verified")
    assert "✅ Integrated" in out, out
    assert (folder / source_name).read_bytes() == b"session contribution\n"
    assert not execution.exists() and folder.exists()
    capture(case, folder, task)
    parent = parent_context(case, source=source)
    if own_body:
        parent.workspace_root = None
        parent.workspace_mode = ""
    out = _integrate_subagent_patch(parent, task_id="child")
    assert "✅ Integrated" in out, out
    assert (source / source_name).read_bytes() == b"session contribution\n"
    custody._CUSTODY.clear()
