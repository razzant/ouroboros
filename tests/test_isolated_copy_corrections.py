"""R1: deleted inputs, independent clean edits, and source-bound write authority."""
from pathlib import Path

import pytest

from ouroboros import subagent_worktrees as trees
from ouroboros.workspace_copies import same_directory, source_is_system_repo, copy_apply_refusal
from ouroboros.tools.subagent_integration import _integrate_subagent_patch
from tests.test_isolated_project_copies import case as copy_case, admit, capture, parent_context, git, repo


@pytest.fixture
def case(tmp_path, monkeypatch):
    return copy_case.__wrapped__(tmp_path, monkeypatch)


@pytest.mark.parametrize("owner", ["task", "run"])
@pytest.mark.parametrize("change", ["delete", "rename", "recreated"])
def test_staged_deleted_inputs_leave_current_snapshot_and_source_unchanged(case, owner, change):
    """The copy is the source's CURRENT tree: a staged delete or rename is in it, and a file
    recreated after ``git rm`` is an ordinary untracked input of the snapshot."""
    source = case[1]
    name = "a.txt"
    if change == "rename":
        git(source, "mv", name, "renamed.txt")
    else:
        git(source, "rm", name)
        if change == "recreated":
            (source / name).write_bytes(b"recreated after the staged delete\n")
    head, index = git(source, "rev-parse", "HEAD"), (source / ".git/index").read_bytes()
    status = git(source, "status", "--porcelain")
    args = {"task_id": "copy", "worktree_root": case[3], "data_dir": case[2]}
    handle = (trees.provision_worktree(repo_dir=source, **args) if owner == "task" else
              trees.provision_execution_snapshot(target_root=source, snapshot_id="run-copy", **args))
    output = Path(handle.path)
    if change == "recreated":
        assert (output / name).read_bytes() == b"recreated after the staged delete\n"
        assert git(output, "ls-files", name) == name.encode() + b"\n"
    else:
        assert not (output / name).exists()
        assert git(output, "ls-files", name) == b""
    if change == "rename":
        assert (output / "renamed.txt").read_bytes() == b"base\n"
    assert git(source, "rev-parse", "HEAD") == head
    assert (source / ".git/index").read_bytes() == index
    assert git(source, "status", "--porcelain") == status


def test_case_aliases_and_common_git_directory_are_physical_identity(case):
    system, external, _, _ = case
    alias = system.with_name(system.name.upper())
    if not alias.exists() or not alias.samefile(system):
        pytest.skip("filesystem does not support a samefile case alias")
    assert same_directory(alias, system)
    assert source_is_system_repo(alias, system)
    child, _ = admit(case, "body", "parent", system)
    assert source_is_system_repo(child, alias)
    assert not source_is_system_repo(external, alias)
    assert not same_directory(system.parent / "missing", system)


def seed_lines(source):
    (source / "a.txt").write_text("".join(f"line {i}\n" for i in range(30)), encoding="utf-8")
    git(source, "add", "a.txt"); git(source, "commit", "-qm", "lines")


def edit_line(folder, old, new):
    path = folder / "a.txt"
    path.write_text(path.read_text(encoding="utf-8").replace(old + "\n", new + "\n"), encoding="utf-8")


@pytest.mark.parametrize("commit_first", [False, True])
def test_clean_copies_synthesize_disjoint_hunks_on_same_file(case, commit_first):
    source = case[1]; seed_lines(source)
    a, ta = admit(case, "a", "parent"); b, tb = admit(case, "b", "parent")
    assert ta["workspace_copy"]["source_index_clean"]
    assert ta["workspace_copy"]["baseline_sha"] == ta["workspace_copy"]["target_head"]
    edit_line(a, "line 2", "A"); edit_line(b, "line 25", "B")
    capture(case, a, ta); capture(case, b, tb)
    ctx = parent_context(case)
    assert "✅ Integrated" in _integrate_subagent_patch(ctx, task_id="a")
    if commit_first:
        git(source, "commit", "-qm", "first contribution")
    out = _integrate_subagent_patch(ctx, task_id="b")
    assert "✅ Integrated" in out, out
    assert "A\n" in (source / "a.txt").read_text(encoding="utf-8")
    assert "B\n" in (source / "a.txt").read_text(encoding="utf-8")


def test_clean_copies_real_conflict_retains_both_results(case):
    seed_lines(case[1])
    a, ta = admit(case, "a", "parent"); b, tb = admit(case, "b", "parent")
    edit_line(a, "line 2", "A"); edit_line(b, "line 2", "B")
    capture(case, a, ta); artifacts, _ = capture(case, b, tb)
    ctx = parent_context(case)
    assert "✅ Integrated" in _integrate_subagent_patch(ctx, task_id="a")
    out = _integrate_subagent_patch(ctx, task_id="b")
    assert "INTEGRATE_CONFLICT" in out, out
    assert b.exists() and (artifacts / "workspace.patch").exists()
    assert git(case[1], "ls-files", "--unmerged")


def test_index_dirty_worktree_matches_head_keeps_current_tree_apply(case):
    source = case[1]
    (source / "a.txt").write_bytes(b"staged edit\n"); git(source, "add", "a.txt")
    (source / "a.txt").write_bytes(b"base\n")
    a, ta = admit(case, "a", "parent")
    assert ta["workspace_copy"]["baseline_sha"] == ta["workspace_copy"]["target_head"]
    assert ta["workspace_copy"]["source_index_clean"] is False
    (a / "a.txt").write_bytes(b"base\nchild\n"); capture(case, a, ta)
    out = _integrate_subagent_patch(parent_context(case), task_id="a")
    assert "✅ Integrated" in out, out
    assert (source / "a.txt").read_bytes() == b"base\nchild\n"


@pytest.mark.parametrize("parent_focus", ["main", "other", "room"])
@pytest.mark.parametrize("mode", ["light", "advanced", "pro", "cyber_pro"])
def test_parent_returns_named_source_under_existing_write_rights(case, monkeypatch, parent_focus, mode):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    # Production shape: Ouroboros data is under its own parent, not HOME/data
    # where the existing user-files policy correctly protects all of HOME.
    runtime = case[2].parent / "runtime-state"
    case[2].rename(runtime)
    case = (case[0], case[1], runtime, case[3])
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(runtime))
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(case[1].parent))
    child, task = admit(case, "child", "parent")
    (child / "BIBLE.md").write_bytes(b"foreign project docs\n"); capture(case, child, task)
    ctx = parent_context(case)
    if parent_focus in {"main", "room"}:
        ctx.workspace_root = None; ctx.workspace_mode = ""; ctx.is_direct_chat = True
        if parent_focus == "room":
            ctx.task_metadata = {"_project_room_dir": str(case[1])}
    else:
        ctx.workspace_root = repo(case[1].parent / "other", {"untouched": b"other"})
    out = _integrate_subagent_patch(ctx, task_id="child")
    assert "✅ Integrated" in out, out
    assert (case[1] / "BIBLE.md").read_bytes() == b"foreign project docs\n"


@pytest.mark.parametrize("restriction", ["readonly", "acting", "presence", "owner_store"])
def test_named_source_never_mints_parent_write_authority(case, monkeypatch, restriction):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.presence_authority import presence_ceiling_payload
    from tests.frozen_presence_authority import build_presence_capability_ceiling
    from tests.frozen_presence_capabilities import PresenceResourceTarget
    from tests.test_presence_authority import _resolution

    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(case[1].parent))
    child, task = admit(case, "child", "parent")
    path = ".ssh/id_rsa" if restriction == "owner_store" else "a.txt"
    (child / path).parent.mkdir(parents=True, exist_ok=True)
    (child / path).write_bytes(b"child output\n")
    if restriction == "owner_store":
        runtime = case[2].parent / "runtime-state"
        case[2].rename(runtime)
        case = (case[0], case[1], runtime, case[3])
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(runtime))
        ctx = parent_context(case); ctx.workspace_root = None; ctx.workspace_mode = ""
        # The folder is writable via user_files, but that does not authorize
        # its control files. Test the per-file rule after positive root binding.
        assert copy_apply_refusal(ctx, case[1], ["a.txt"]) == ""
        assert "VCS control directory" in copy_apply_refusal(ctx, case[1], [".git/config"])
        return
    capture(case, child, task)
    ctx = parent_context(case, source=case[0])
    if restriction in {"readonly", "acting"}:
        ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent" if restriction == "readonly" else "acting_subagent",
            surface="" if restriction == "readonly" else "external_workspace", write_root=str(case[0]))
    else:
        ceiling = build_presence_capability_ceiling(skill_name="fixture", skill_content_hash="c"*64,
            state_fingerprint="d"*64, resolution=_resolution(PresenceResourceTarget("active_workspace", ("write",), ".")))
        ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(ceiling)}
    out = _integrate_subagent_patch(ctx, task_id="child")
    assert "✅ Integrated" not in out and ("FORBIDDEN" in out or "PRESENCE_RESOURCE_BLOCKED" in out), out
    assert (case[1] / "a.txt").read_bytes() == b"base\n"


def test_actual_body_policy_survives_false_child_label(case, monkeypatch):
    child, task = admit(case, "child", "parent", case[0])
    task["workspace_copy"]["source_is_system_repo"] = False
    (child / "system.txt").write_bytes(b"body change\n"); capture(case, child, task)
    ctx = parent_context(case)  # parent is focused on the foreign project
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    out = _integrate_subagent_patch(ctx, task_id="child")
    assert "LIGHT_MODE_BLOCKED" in out, out
    assert (case[0] / "system.txt").read_bytes() == b"own body\n"


def test_apply_rechecks_presence_after_lock(case, monkeypatch):
    from ouroboros.tools import git as git_tools
    from ouroboros.presence_authority import presence_ceiling_payload
    from tests.frozen_presence_authority import build_presence_capability_ceiling
    from tests.test_presence_authority import _resolution
    child, task = admit(case, "child", "parent")
    (child / "a.txt").write_bytes(b"child\n"); capture(case, child, task)
    ctx = parent_context(case)
    lock = git_tools._acquire_git_lock
    def acquire(*args, **kwargs):
        held = lock(*args, **kwargs)
        ceiling = build_presence_capability_ceiling(skill_name="fixture", skill_content_hash="c"*64,
            state_fingerprint="d"*64, resolution=_resolution())
        ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(ceiling)}
        return held
    monkeypatch.setattr(git_tools, "_acquire_git_lock", acquire)
    out = _integrate_subagent_patch(ctx, task_id="child")
    assert "PRESENCE_RESOURCE_BLOCKED" in out and "✅ Integrated" not in out
    assert (case[1] / "a.txt").read_bytes() == b"base\n"


def test_real_schedule_supervisor_stamps_copy_metadata(case, monkeypatch):
    from ouroboros.tools.control_scheduling import _schedule_task
    from supervisor.events_schedule_task import _handle_schedule_task
    from ouroboros.task_results import load_task_result
    from tests.test_nested_rights_depth import _fake_ctx
    from tests.test_task_status_flow import _configure_test_subagent
    _configure_test_subagent(monkeypatch)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(case[1].parent))
    ctx = parent_context(case)
    (case[1] / "a.txt").write_bytes(b"uncommitted source\n")
    out = _schedule_task(ctx, subagent_id="api-scout", objective="Change source", expected_output="patch",
                         workspace_root=str(case[1]), write_surface="self_worktree")
    assert "queued" in out, out
    evt = ctx.pending_events[-1]
    enqueued = []
    supervisor = _fake_ctx(case[2], enqueued); supervisor.REPO_DIR = case[0]
    _handle_schedule_task(evt, supervisor)
    [task] = enqueued
    binding = task["metadata"]["workspace_copy"]
    assert binding["source_root"] == str(case[1])
    assert binding["execution_root"] == task["workspace_root"]
    assert binding == load_task_result(case[2], task["id"])["workspace_copy"]
    assert task["task_contract"]["workspace"]["root"] == task["workspace_root"]
    assert (Path(task["workspace_root"]) / "a.txt").read_bytes() == b"uncommitted source\n"
