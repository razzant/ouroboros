"""Workspace tool context and run_shell routing/safety in headless tasks.

Split verbatim out of ``tests/test_headless_cli.py`` by theme. This module
owns where a workspace task may read, write and execute: project-file
routing, allowed shell cwds, redirect and symlink-escape guards, task-local
git, and the preflight inference of binaries from manifests.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

from ouroboros.tools.core_file_tools import _repo_read
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.workspace_preflight import _infer_tools_from_manifests


from tests._headless_cli_shared import (  # noqa: F401  (autouse fixture applies on import)
    _init_repo_with_file,
    _managed_worker_pool_available,
)

from tests._typed_guard_shared import _shell_guard_text



def test_workspace_context_routes_project_files_and_keeps_system_tools_reachable(tmp_path):
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    system_repo.mkdir()
    workspace.mkdir()
    data.mkdir()
    (system_repo / "README.md").write_text("system", encoding="utf-8")
    (workspace / "README.md").write_text("workspace", encoding="utf-8")
    (workspace / "BIBLE.md").write_text("external bible", encoding="utf-8")

    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
    )

    assert "workspace" in _repo_read(ctx, "README.md")
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)
    commit_result = registry.execute("commit_reviewed", {"commit_message": "nope"})
    assert "WORKSPACE_MODE_BLOCKED" not in commit_result
    assert registry.get_schema_by_name("commit_reviewed") is not None
    assert registry.get_schema_by_name("request_restart") is not None
    assert "Written" in registry.execute("write_file", {"path": "BIBLE.md", "content": "external edit"})
    assert (workspace / "BIBLE.md").read_text(encoding="utf-8") == "external edit"
    replaced = registry.execute(
        "edit_text",
        {"path": "README.md", "old_str": "workspace", "new_str": "workspace edited"},
    )
    assert "Replaced" in replaced
    assert (workspace / "README.md").read_text(encoding="utf-8") == "workspace edited"


def test_workspace_run_shell_cwd_allows_scratch_and_explicit_system(tmp_path, monkeypatch):
    """External-workspace tasks may run from host scratch (a sibling checkout, a
    /tmp tree) and explicitly select the system repo; generic runtime data stays
    off-limits and system-repo mutation remains independently governed."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    # Pin $HOME outside tmp_path so the host-scratch cwd allowance holds on Windows
    # CI too (where pytest's tmp dir lives UNDER home and the data-parent-under-home
    # protection would otherwise block the sibling scratch cwd). See the same fixture
    # in test_external_workspace_access.py.
    fake_home = tmp_path / "_home"
    fake_home.mkdir()
    monkeypatch.setattr(pathlib.Path, "home", lambda: fake_home)
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    data = tmp_path / "data"
    for path in (system_repo, workspace, outside, data):
        path.mkdir()
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    # Host scratch outside the declared workspace is now a legitimate cwd...
    scratch_cwd = registry.execute("run_command", {"cmd": ["pwd"], "cwd": str(outside)})
    assert "SHELL_CWD_BLOCKED" not in scratch_cwd
    # The approved root contract makes system_repo an explicit cwd; generic
    # runtime_data remains unavailable to process tools.
    runtime_repo_cwd = registry.execute("run_command", {"cmd": ["pwd"], "cwd": str(system_repo)})
    assert "SHELL_CWD_BLOCKED" not in runtime_repo_cwd
    assert f"cwd={system_repo.resolve()}" in runtime_repo_cwd
    runtime_data_cwd = registry.execute("run_command", {"cmd": ["pwd"], "cwd": str(data)})
    assert "SHELL_CWD_BLOCKED" in runtime_data_cwd
    # READ-ONLY git at a runtime target is ALLOWED (owner contract "read-only
    # everywhere"; the f14baf8f false-block class). Only MUTATING git is target-checked.
    git_read = _shell_guard_text(registry,
        {"cmd": ["git", "-C", str(system_repo), "status"]}, "advanced"
    )
    assert git_read is None, git_read
    git_escape = _shell_guard_text(registry,
        {"cmd": ["git", "-C", str(system_repo), "commit", "-m", "x"]}, "advanced"
    )
    assert git_escape and "WORKSPACE_GIT_BLOCKED" in git_escape
    git_chain = registry.execute("run_command", {"cmd": ["sh", "-c", "true && git --version; echo git binary OK"]})
    assert "WORKSPACE_GIT_BLOCKED" not in git_chain
    outside_write = registry.execute("run_command", {"cmd": ["touch", str(system_repo / "README.md")]})
    assert "WORKSPACE_SHELL_BLOCKED" in outside_write
    embedded_outside_write = registry.execute(
        "run_command",
        {"cmd": ["python", "-c", "open('/tmp/ouroboros-outside.txt','w').write('x')"]},
    )
    assert "WORKSPACE_SHELL_BLOCKED" in embedded_outside_write


def test_workspace_shell_safe_stdio_redirects_are_not_write_like(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    data = tmp_path / "data"
    for path in (system_repo, workspace, outside, data):
        path.mkdir()
    (outside / "visible.txt").write_text("ok\n", encoding="utf-8")
    ctx = ToolContext(repo_dir=system_repo, drive_root=data, workspace_root=workspace, workspace_mode="external")
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    stderr_sink = registry.execute("run_command", {"cmd": f"find {outside} -maxdepth 1 2>/dev/null"})
    fd_dup = registry.execute("run_command", {"cmd": f"ls {outside} 2>&1 | head -n 1"})
    fd_close = registry.execute("run_command", {"cmd": f"find {outside} -maxdepth 1 2>&-"})
    real_redirect = registry.execute("run_command", {"cmd": f"echo x > {outside / 'out.txt'}"})

    assert "WORKSPACE_SHELL_BLOCKED" not in stderr_sink, stderr_sink
    assert "WORKSPACE_SHELL_BLOCKED" not in fd_dup, fd_dup
    assert "WORKSPACE_SHELL_BLOCKED" not in fd_close, fd_close
    assert "WORKSPACE_SHELL_BLOCKED" in real_redirect


def test_workspace_shell_blocks_windows_absolute_redirects_before_shell_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    for path in (system_repo, workspace, data):
        path.mkdir()
    ctx = ToolContext(repo_dir=system_repo, drive_root=data, workspace_root=workspace, workspace_mode="external")
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    drive_redirect = registry.execute("run_command", {"cmd": r"echo x > C:\ouroboros-outside\out.txt"})
    unc_redirect = registry.execute("run_command", {"cmd": r"echo x > \\server\share\out.txt"})

    assert "WORKSPACE_SHELL_BLOCKED" in drive_redirect
    assert "SHELL_SYNTAX_UNSUPPORTED" not in drive_redirect
    assert "WORKSPACE_SHELL_BLOCKED" in unc_redirect
    assert "SHELL_SYNTAX_UNSUPPORTED" not in unc_redirect


def test_workspace_shell_keeps_symlinked_workspace_absolute_paths_allowed(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    system_repo = tmp_path / "system"
    real_workspace = tmp_path / "real_workspace"
    workspace_link = tmp_path / "workspace_link"
    data = tmp_path / "data"
    for path in (system_repo, real_workspace, data):
        path.mkdir()
    try:
        workspace_link.symlink_to(real_workspace, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink unavailable on this platform: {exc}")
    ctx = ToolContext(repo_dir=system_repo, drive_root=data, workspace_root=workspace_link, workspace_mode="external")
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    target = workspace_link / "inside.txt"
    result = registry.execute("run_command", {"cmd": [sys.executable, "-c", f"open({str(target)!r}, 'w').write('ok')"]})

    assert "WORKSPACE_SHELL_BLOCKED" not in result, result
    assert (real_workspace / "inside.txt").exists()


def test_workspace_shell_blocks_nested_symlink_escape_absolute_path(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    data = tmp_path / "data"
    for path in (system_repo, workspace, outside, data):
        path.mkdir()
    outlink = workspace / "outlink"
    outside_file = outside / "target.txt"
    outside_file.write_text("old\n", encoding="utf-8")
    filelink = workspace / "filelink"
    executable_name_link = workspace / "touch"
    try:
        outlink.symlink_to(outside, target_is_directory=True)
        filelink.symlink_to(outside_file)
        executable_name_link.symlink_to(outside_file)
    except OSError as exc:
        pytest.skip(f"symlink unavailable on this platform: {exc}")
    ctx = ToolContext(repo_dir=system_repo, drive_root=data, workspace_root=workspace, workspace_mode="external")
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    result = registry.execute("run_command", {"cmd": f"touch {outlink / 'escaped.txt'}"})
    relative_result = registry.execute("run_command", {"cmd": "touch outlink/escaped-relative.txt"})
    bare_result = registry.execute("run_command", {"cmd": "touch outlink"})
    executable_name_result = registry.execute("run_command", {"cmd": ["touch", "touch"]})
    redirect_result = registry.execute("run_command", {"cmd": "echo changed > filelink"})
    compact_redirect_result = registry.execute("run_command", {"cmd": "echo changed >filelink"})
    shell_inline_result = registry.execute("run_command", {"cmd": ["sh", "-c", "echo changed > filelink"]})
    shell_inline_touch_result = registry.execute("run_command", {"cmd": ["sh", "-c", "touch filelink"]})
    bash_redirect_result = registry.execute("run_command", {"cmd": ["bash", "-c", "echo changed &> filelink"]})
    compact_bash_redirect_result = registry.execute("run_command", {"cmd": ["bash", "-c", "echo changed &>filelink"]})
    tee_result = registry.execute("run_command", {"cmd": "printf changed | tee filelink"})
    shell_inline_tee_result = registry.execute("run_command", {"cmd": ["sh", "-c", "printf changed | tee filelink"]})
    python_inline_result = registry.execute(
        "run_command",
        {"cmd": [sys.executable, "-c", "open('filelink', 'w').write('changed')"]},
    )
    python_versioned_result = registry.execute(
        "run_command",
        {"cmd": ["python3.12", "-c", "open('filelink', 'w').write('changed')"]},
    )
    node_script_result = registry.execute(
        "run_script",
        {
            "interpreter": "node",
            "script": "require('fs').writeFileSync('filelink', 'changed')",
        },
    )

    assert "WORKSPACE_SHELL_BLOCKED" in result
    assert "WORKSPACE_SHELL_BLOCKED" in relative_result
    assert "WORKSPACE_SHELL_BLOCKED" in bare_result
    assert "WORKSPACE_SHELL_BLOCKED" in executable_name_result
    assert "WORKSPACE_SHELL_BLOCKED" in redirect_result
    assert "WORKSPACE_SHELL_BLOCKED" in compact_redirect_result
    assert "WORKSPACE_SHELL_BLOCKED" in shell_inline_result
    assert "WORKSPACE_SHELL_BLOCKED" in shell_inline_touch_result
    assert "WORKSPACE_SHELL_BLOCKED" in bash_redirect_result
    assert "WORKSPACE_SHELL_BLOCKED" in compact_bash_redirect_result
    assert "WORKSPACE_SHELL_BLOCKED" in tee_result
    assert "WORKSPACE_SHELL_BLOCKED" in shell_inline_tee_result
    assert "WORKSPACE_SHELL_BLOCKED" in python_inline_result
    assert "WORKSPACE_SHELL_BLOCKED" in python_versioned_result
    assert "WORKSPACE_SHELL_BLOCKED" in node_script_result
    assert not (outside / "escaped.txt").exists()
    assert not (outside / "escaped-relative.txt").exists()
    assert outside_file.read_text(encoding="utf-8") == "old\n"


def test_external_workspace_shell_allows_task_local_git(tmp_path, monkeypatch):
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    system_repo.mkdir()
    data.mkdir()
    _init_repo_with_file(workspace)
    ctx = ToolContext(repo_dir=system_repo, drive_root=data, workspace_root=workspace, workspace_mode="external")
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)
    monkeypatch.setenv("OUROBOROS_TEST_RUNTIME_REPO", str(system_repo))

    allowed = [
        ["git", "for-each-ref", "--format=%(refname)"],
        ["git", "rev-list", "--count", "HEAD"],
        ["git", "show-ref", "--heads"],
        ["git", "branch", "--show-current"],
        ["git", "branch", "--list"],
        ["git", "branch", "--list", "ma*"],
        ["git", "branch", "-av"],
        ["git", "tag", "-l"],
        ["git", "tag", "--list", "v*"],
        ["git", "branch", "new-branch"],
        ["git", "branch", "-v", "new-branch"],
        ["git", "branch", "--verbose", "new-branch"],
        ["git", "branch", "-d", "main"],
        ["git", "tag", "v1"],
        ["git", "tag", "-a", "v1", "-m", "x"],
        ["git", "commit", "--allow-empty", "-m", "task-local commit"],
        ["sh", "-c", "git --version; echo git binary OK"],
    ]

    for cmd in allowed:
        assert _shell_guard_text(registry, {"cmd": cmd}, "advanced") is None, cmd

    # READ-ONLY git reaches the runtime through EVERY retarget vector — that is the
    # owner contract ("read-only everywhere, including at a runtime target") and the
    # recorded false-block class f14baf8f. Before the Q4=A composition these four
    # were refused: the target-aware resolver let them through and the
    # external-workspace runtime-READ guard then blocked them as
    # WORKSPACE_SHELL_BLOCKED, naming the wrong reason.
    for cmd in (
        ["git", "-C", str(system_repo), "status"],
        ["git", "--git-dir", str(system_repo / ".git"), "status"],
        # as_posix(): a POSIX shell (sh -c) uses forward slashes; a Windows
        # backslash literal would be eaten as shell escapes during parsing.
        ["sh", "-c", f"cd {system_repo.as_posix()} && git status"],
        ["sh", "-c", "git -C $OUROBOROS_TEST_RUNTIME_REPO status"],
    ):
        result = _shell_guard_text(registry, {"cmd": cmd}, "advanced")
        assert result is None, (cmd, result)

    # ...while the MUTATING form of each vector stays blocked.
    for cmd in (
        ["git", "-C", str(system_repo), "commit", "-m", "x"],
        ["git", "--git-dir", str(system_repo / ".git"), "commit", "-m", "x"],
        ["sh", "-c", f"cd {system_repo.as_posix()} && git commit -m x"],
        ["sh", "-c", "git -C $OUROBOROS_TEST_RUNTIME_REPO commit -m x"],
    ):
        result = _shell_guard_text(registry, {"cmd": cmd}, "advanced")
        assert result and "WORKSPACE_GIT_BLOCKED" in result, (cmd, result)

    # The read-only exemption is ALL-or-NOTHING per segment: a compound that only
    # STARTS with git still meets the runtime/secret read guard in full.
    mixed = _shell_guard_text(registry,
        {"cmd": ["sh", "-c", f"git status && cat {(data / 'settings.json').as_posix()}"]},
        "advanced",
    )
    assert mixed and "WORKSPACE_SHELL_BLOCKED" in mixed, mixed


def test_workspace_shell_git_ls_remote_requires_network_contract(tmp_path):
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    system_repo.mkdir()
    data.mkdir()
    _init_repo_with_file(workspace)
    contract = {
        "allowed_resources": {"network": False},
        "resource_policy": {},
    }
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_contract=contract,
        task_metadata={"task_contract": contract},
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    for cmd in (
        ["git", "ls-remote", "origin"],
        ["git", "submodule", "update", "--init", "--recursive"],
    ):
        result = _shell_guard_text(registry, {"cmd": cmd}, "advanced")
        assert result and "RESOURCE_CONSTRAINT_BLOCKED" in result, (cmd, result)


def test_workspace_run_shell_allows_absolute_cwd_under_workspace_and_child_drive(tmp_path):
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    parent_data = tmp_path / "data"
    parent_task_dir = parent_data / "task_drives" / "task-workspace" / "scratch"
    child_drive = tmp_path / "child-data"
    child_dir = child_drive / "task_drives" / "task-workspace" / "scratch"
    child_control_dir = child_drive / "memory"
    for path in (system_repo, workspace, parent_data / "logs", parent_task_dir, child_dir, child_control_dir):
        path.mkdir(parents=True)
    ctx = ToolContext(
        repo_dir=system_repo,
        drive_root=parent_data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="task-workspace",
        task_metadata={"drive_root": str(child_drive), "budget_drive_root": str(parent_data)},
    )
    registry = ToolRegistry(repo_dir=system_repo, drive_root=parent_data)
    registry.set_context(ctx)

    def assert_python_cwd(path):
        output = registry.execute(
            "run_command",
            {"cmd": [sys.executable, "-c", "import os; print(os.getcwd())"], "cwd": str(path)},
        )
        assert "exit_code=0" in output
        # #447 H1: host notes (SAFETY_WARNING and siblings) trail the payload
        # now — take the first line after STDOUT, not the whole tail.
        cwd_output = output.rsplit("STDOUT:\n", 1)[-1].strip().splitlines()[0].strip()
        assert pathlib.Path(cwd_output).resolve() == path.resolve()

    assert_python_cwd(workspace)
    assert_python_cwd(child_dir)
    child_control = registry.execute("run_command", {"cmd": ["pwd"], "cwd": str(child_control_dir)})
    assert "SHELL_CWD_BLOCKED" in child_control
    blocked = registry.execute("run_command", {"cmd": ["pwd"], "cwd": str(parent_data / "logs")})
    assert "SHELL_CWD_BLOCKED" in blocked
    # Read-only git is allowed everywhere now; the escape check uses a MUTATING form.
    git_read = _shell_guard_text(registry,
        {"cmd": ["git", "-C", "../other-repo", "status"], "cwd": str(child_dir)},
        "advanced",
    )
    assert git_read is None, git_read
    git_escape = _shell_guard_text(registry,
        {"cmd": ["git", "-C", "..", "commit", "-m", "x"], "cwd": str(child_dir)},
        "advanced",
    )
    assert git_escape and "WORKSPACE_GIT_BLOCKED" in git_escape, git_escape
    protected_escape = _shell_guard_text(registry,
        {"cmd": ["touch", "../data/state/state.json"]},
        "pro",
    )
    assert "WORKSPACE_SHELL_BLOCKED" in protected_escape
    task_drive_write = registry.execute("run_command", {"cmd": ["touch", "output.txt"], "cwd": str(child_dir)})
    assert "WORKSPACE_SHELL_BLOCKED" not in task_drive_write
    assert (child_dir / "output.txt").is_file()
    parent_task_drive_write = registry.execute("run_command", {"cmd": ["touch", "output.txt"], "cwd": str(parent_task_dir)})
    assert "WORKSPACE_SHELL_BLOCKED" not in parent_task_drive_write
    assert (parent_task_dir / "output.txt").is_file()
    absolute_task_drive_file = parent_task_dir / "absolute-python.txt"
    absolute_task_drive_write = registry.execute(
        "run_command",
        {"cmd": [sys.executable, "-c", f"open({str(absolute_task_drive_file)!r}, 'w').write('ok')"]},
    )
    assert "WORKSPACE_SHELL_BLOCKED" not in absolute_task_drive_write
    assert absolute_task_drive_file.read_text(encoding="utf-8") == "ok"


def test_workspace_shell_allows_nested_relative_write_paths(tmp_path):
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    for path in (system_repo, workspace, data):
        path.mkdir(parents=True)
    ctx = ToolContext(repo_dir=system_repo, drive_root=data, workspace_root=workspace, workspace_mode="external")
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    assert _shell_guard_text(registry, {"cmd": ["touch", "subdir/file.txt"]}, "advanced") is None
    assert _shell_guard_text(registry, {"cmd": ["mkdir", "-p", "build/output"]}, "advanced") is None
    python_write = {"cmd": [sys.executable, "-c", "open('subdir/python.txt', 'w').write('ok')"]}
    assert _shell_guard_text(registry, python_write, "advanced") is None


def test_workspace_shell_sudo_and_pro_passthrough_policy(tmp_path):
    system_repo = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    for path in (system_repo, workspace, data):
        path.mkdir()
    ctx = ToolContext(repo_dir=system_repo, drive_root=data, workspace_root=workspace, workspace_mode="external")
    registry = ToolRegistry(repo_dir=system_repo, drive_root=data)
    registry.set_context(ctx)

    assert "SUDO_INTERACTIVE_BLOCKED" in _shell_guard_text(registry, {"cmd": ["sudo", "true"]}, "pro")
    assert "SUDO_INTERACTIVE_BLOCKED" in _shell_guard_text(registry, {"cmd": ["sh", "-c", "sudo true"]}, "pro")
    assert "SUDO_INTERACTIVE_BLOCKED" in _shell_guard_text(registry, {"cmd": ["sudo", "-S", "true"]}, "pro")
    assert "SUDO_INTERACTIVE_BLOCKED" in _shell_guard_text(registry, {"cmd": ["sudo", "-nS", "true"]}, "pro")
    assert "SUDO_INTERACTIVE_BLOCKED" in _shell_guard_text(registry, {"cmd": ["sudoedit", "/etc/hosts"]}, "pro")
    assert _shell_guard_text(registry, {"cmd": ["sudo", "-n", "python", "-S", "-c", "print(1)"]}, "pro") is None
    assert "SAFETY_VIOLATION" in _shell_guard_text(registry, {"cmd": ["sh", "-c", "gh repo create x"]}, "pro")
    assert "SAFETY_VIOLATION" in _shell_guard_text(registry, {"cmd": ["sh", "-c", "gh auth login"]}, "pro")
    # Line-continuation still spells ONE gh invocation and stays blocked...
    assert "SAFETY_VIOLATION" in _shell_guard_text(registry, {"cmd": ["sh", "-c", "gh \\\nrepo create x"]}, "pro")
    # ...while bare newlines run `gh`, `repo`, `create x` as three separate
    # commands that create nothing — that spelling was only ever a text
    # mention, not an invocation (#447 A7 argv-positional gh policy).
    assert _shell_guard_text(registry, {"cmd": ["sh", "-c", "gh\nrepo\ncreate x"]}, "pro") is None
    outside_write = {"cmd": ["python", "-c", "open('/tmp/ouroboros-pro.txt','w').write('x')"]}
    assert "WORKSPACE_SHELL_BLOCKED" in _shell_guard_text(registry, outside_write, "advanced")
    assert _shell_guard_text(registry, outside_write, "pro") is None


def test_workspace_preflight_infers_binaries_from_script_commands():
    tools = _infer_tools_from_manifests([
        {
            "type": "node",
            "scripts": ["test"],
            "script_commands": {"test": "vitest --run"},
        }
    ])
    assert "vitest" in tools
    assert "test" not in tools
    noisy = _infer_tools_from_manifests([
        {
            "type": "node",
            "scripts": ["build"],
            "script_commands": {"build": "NODE_ENV=production cd web && vite build"},
        }
    ])
    assert "NODE_ENV=production" not in noisy
    assert "cd" not in noisy
    assert "vite" in noisy
