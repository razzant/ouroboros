"""Retired action restrictions, bounded history, and configured comparisons."""
import pytest
from tests.test_vcs_read_slice import world as _world, git

world = _world


@pytest.mark.parametrize("retired,action,other", [("vcs_status", "status", "diff"), ("vcs_diff", "diff", "status")])
def test_retired_names_are_unknown_and_saved_disables_withhold_only_their_action(world, retired, action, other):
    registry, ctx, _, _project = world
    assert registry.get_schema_by_name(retired) is None
    assert "Unknown tool" in registry.execute(retired, {})
    ctx.task_contract = {"disabled_tools": [retired]}
    blocked = registry.execute_result("vcs_read", {"action": action})
    assert blocked.code == "RESOURCE_CONSTRAINT_BLOCKED"
    assert registry.execute_result("vcs_read", {"action": other}).status == "ok"
    ctx.task_contract = {"disabled_tools": ["vcs_read"]}
    assert registry.execute_result("vcs_read", {"action": other}).code == "RESOURCE_CONSTRAINT_BLOCKED"
    ctx.task_contract = {}
    assert registry.execute_result("vcs_read", {"action": action}).status == "ok"
    from tests.test_preflight_recovery_authority import _presence_contract
    ctx.task_contract = _presence_contract(retired)
    assert registry.get_schema_by_name("vcs_read") is None


@pytest.mark.parametrize("repository", [None, "."])
def test_status_diff_keep_clean_filter_comparison(world, repository):
    registry, _, _, project = world
    (project / ".gitattributes").write_text("note.txt filter=normalize\n")
    git(project, "config", "filter.normalize.clean", "sed s/WORKTREE/CANONICAL/g")
    (project / "note.txt").write_text("WORKTREE\n")
    git(project, "add", ".gitattributes", "note.txt")
    git(project, "commit", "-qm", "normalized")
    # Force Git to compare bytes, rather than trusting a cached stat.
    (project / "note.txt").write_text("WORKTREE\n")
    args = {} if repository is None else {"repository": repository}
    for action in ("status", "diff"):
        result = registry.execute_result("vcs_read", {"action": action, "path": "note.txt", **args})
        assert result.status == "ok", result.text
        assert result.text.startswith("VCS target:"), result.text
    (project / "note.txt").write_text("changed WORKTREE\n")
    difference = registry.execute_result("vcs_read", {"action": "diff", "path": "note.txt", **args})
    assert "+changed CANONICAL" in difference.text


def test_status_diff_paths_normalize_like_the_retired_tools(world):
    registry, _, _, project = world
    (project / "note.txt").write_text("changed\n")
    for action in ("status", "diff"):
        for path in (str(project / "note.txt"), f"{project.name}/note.txt", "*.txt", "./"):
            result = registry.execute_result("vcs_read", {"action": action, "path": path, "max_chars": "4000"})
            assert result.status == "ok" and "note.txt" in result.text.split("VCS target:")[0], (action, path, result.text)
        assert registry.execute_result("vcs_read", {"action": action, "path": "*.md"}).text.startswith("VCS target:")


def test_history_names_entries_as_given_while_status_and_diff_take_worktree_spellings(world):
    registry, _, _, project = world
    # A package that moved from <repository name>/ to src/<repository name>/ (a src-layout migration).
    (project / project.name).mkdir()
    (project / project.name / "cli.py").write_text("old layout\n")
    git(project, "add", ".")
    git(project, "commit", "-qm", "old layout")
    old = git(project, "rev-parse", "HEAD")
    (project / "src").mkdir()
    git(project, "mv", project.name, f"src/{project.name}")
    git(project, "commit", "-qm", "src layout")
    entry = f"{project.name}/cli.py"
    assert "old layout" in registry.execute_result("vcs_read", {"action": "log", "path": entry}).text
    assert registry.execute_result("vcs_read", {"action": "file", "path": entry, "ref": old}).text.startswith("old layout")
    (project / "note.txt").write_text("changed\n")
    status = registry.execute_result("vcs_read", {"action": "status", "path": f"{project.name}/note.txt"})
    assert "note.txt" in status.text.split("VCS target:")[0], status.text


def test_docker_backend_paths_map_to_the_workspace_for_every_action(world):
    registry, ctx, _, project = world
    (project / "note.txt").write_text("changed\n")
    assert registry.execute_result("vcs_read", {"action": "status", "path": "/app/note.txt"}).code == "GIT_ERROR"
    ctx.task_metadata = {"executor_ref": {"type": "docker_exec", "container_name": "probe",
                                          "workspace_host_path": str(project), "workspace_backend_path": "/app"}}
    registry.set_context(ctx)
    for action in ("status", "diff"):
        result = registry.execute_result("vcs_read", {"action": action, "path": "/app/note.txt"})
        assert "note.txt" in result.text.split("VCS target:")[0], (action, result.text)
    assert registry.execute_result("vcs_read", {"action": "file", "path": "/app/note.txt"}).text.startswith("first")


def test_a_root_refusal_names_vcs_reads_own_roots_and_the_readers_of_a_delegated_copy(world):
    """A refusal listed the profile's vcs roots, which vcs_read never takes, and nothing that reads a
    delegated run's private copy; the readers now ride the refusal, never a working read."""
    registry, _ctx, _, project = world
    refused = registry.execute_result("vcs_read", {"action": "status", "repository": str(project.parent / "dlg_copy")})
    assert refused.code == "GIT_ERROR" and "outside selected root" in refused.text, refused.text
    assert "root='active_workspace' or 'system_repo' only" in refused.text and "patch_read/manifest_read" in refused.text
    (project / "note.txt").write_text("changed\n")
    working = registry.execute_result("vcs_read", {"action": "status"})
    assert working.status == "ok" and "note.txt" in working.text and "patch_read" not in working.text


def test_protected_history_is_judged_on_the_entry_git_reads_not_a_symlinks_target(world):
    registry, ctx, _, project = world
    (project / "private").mkdir()
    (project / "src").mkdir()
    (project / "src" / "current.txt").write_text("PUBLIC BODY\n")
    (project / "private" / "legacy.txt").write_text("PROTECTED HISTORY\n")
    git(project, "add", ".")
    git(project, "commit", "-qm", "history")
    old = git(project, "rev-parse", "HEAD")
    (project / "private" / "legacy.txt").unlink()
    try:
        (project / "private" / "legacy.txt").symlink_to("../src/current.txt")
    except OSError:
        pytest.skip("host cannot create symlinks")
    git(project, "add", "-A")
    git(project, "commit", "-qm", "compatibility link")
    ctx.task_metadata = {"task_contract": {"resource_policy": {"protected_artifacts": [
        {"id": "private", "role": "black_box_reference", "paths": ["./private"], "allow": ["execute"]}]}}}
    for spelling in ("private/legacy.txt", str(project / "private" / "legacy.txt")):
        blocked = registry.execute_result("vcs_read", {"action": "file", "path": spelling, "ref": old})
        assert blocked.code == "RESOURCE_POLICY_BLOCKED" and "PROTECTED HISTORY" not in blocked.text, (spelling, blocked.text)
    allowed = registry.execute_result("vcs_read", {"action": "file", "path": "src/current.txt", "ref": old})
    assert allowed.status == "ok" and allowed.text.startswith("PUBLIC BODY")


def test_diff_pattern_that_can_reach_a_protected_file_is_blocked_like_a_bare_diff(world):
    registry, ctx, _, project = world
    (project / "secret.txt").write_text("PROTECTED-BYTES\n")
    git(project, "add", "secret.txt")
    git(project, "commit", "-qm", "secret")
    (project / "secret.txt").write_text("PROTECTED-BYTES-CHANGED\n")
    (project / "note.txt").write_text("note changed\n")
    ctx.task_metadata = {"task_contract": {"resource_policy": {"protected_artifacts": [
        {"id": "ref", "role": "black_box_reference", "paths": ["./secret.txt"], "allow": ["execute"]}]}}}
    for path in ("*.txt", "*", "secre?.txt"):
        result = registry.execute_result("vcs_read", {"action": "diff", "path": path})
        assert result.code == "RESOURCE_POLICY_BLOCKED" and "PROTECTED-BYTES" not in result.text, (path, result.text)
    named = registry.execute_result("vcs_read", {"action": "diff", "path": "note.txt"})
    assert named.status == "ok" and "+note changed" in named.text and "PROTECTED-BYTES" not in named.text


def test_history_capture_is_physically_bounded_and_discloses_narrowing(world, monkeypatch):
    from ouroboros.tools import vcs_read
    registry, _, _, project = world
    (project / "large.txt").write_text("x" * 4096)
    git(project, "add", "large.txt")
    git(project, "commit", "-qm", "large")
    monkeypatch.setattr(vcs_read, "_CAPTURE_BYTES", 256)
    captured = vcs_read._capture_read_git(project, ["show", "HEAD:large.txt"])
    assert len(captured.stdout) == 256 and captured.stdout_omitted == 3840
    result = registry.execute_result("vcs_read", {"action": "file", "path": "large.txt"})
    assert result.status == "ok" and "3840 bytes omitted" in result.text
    assert "max_count (log), ref, or path" in result.text


def test_history_timeout_reaps_child_and_closes_pipe_readers(tmp_path, monkeypatch):
    import subprocess
    import sys
    from ouroboros.tools import vcs_read

    popen = subprocess.Popen
    children = []
    def spawn(_argv, **kwargs):
        process = popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        children.append(process)
        wait = process.wait
        def expire(timeout=None):
            if timeout == vcs_read.GIT_READ_CAPTURE_TIMEOUT_SEC:
                raise subprocess.TimeoutExpired("git", timeout)
            return wait(timeout=timeout)
        process.wait = expire
        return process
    monkeypatch.setattr(vcs_read.subprocess, "Popen", spawn)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            vcs_read._capture_read_git(tmp_path, ["log"])
        assert len(children) == 1 and children[0].poll() is not None
        assert children[0].stdout.closed and children[0].stderr.closed
    finally:
        for process in children:
            if process.poll() is None:
                process.kill()
                process.wait()
