"""Runtime-mode gating of run_shell: mutation detection and the light-mode tripwire.

Split verbatim out of ``tests/test_runtime_mode_core.py`` by theme. This module owns
the git mutations detected through env and shell wrappers, the default lane that allows
mutating git outside the runtime but never at it, the in-place and protected-path
writers refused up front, the read-only mentions that stay allowed, and the tripwire
that catches a repo write the pre-checks missed.
"""

from __future__ import annotations

import sys

import pytest

from ouroboros.tools.registry import ToolRegistry

from tests._runtime_mode_core_shared import _git_repo, _registry


# ----- run_shell mutation detection (light + advanced) -----


def test_light_mode_blocks_runshell_mutation(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path)
    result = reg.execute("run_command", {"cmd": "git commit -m 'x'"})
    assert "GIT_VIA_SHELL_BLOCKED" in result


@pytest.mark.parametrize("cmd", [
    ["env", "git", "commit", "-m", "x"],
    ["/usr/bin/env", "git", "commit", "-m", "x"],
    ["/usr/bin/env", "-S", "git commit -m x"],
])
def test_run_shell_blocks_env_wrapped_git_mutation(cmd, tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path)
    result = reg.execute("run_command", {"cmd": cmd})
    assert "GIT_VIA_SHELL_BLOCKED" in result


@pytest.mark.parametrize("cmd", [
    ["sh", "-c", "git commit -m x"],
    ["bash", "-c", "git add README.md && git commit -m x"],
])
def test_run_shell_blocks_shell_wrapped_git_mutation(cmd, tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    reg = _registry(tmp_path)
    result = reg.execute("run_command", {"cmd": cmd})
    assert "GIT_VIA_SHELL_BLOCKED" in result


def _outside_runtime_registry(tmp_path, monkeypatch):
    """Registry whose repo/data/user-files roots are DISJOINT, so an out-of-runtime
    git target is actually outside every protected root (repo_dir == drive_root ==
    tmp_path in _registry makes everything runtime-contained)."""
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo = tmp_path / "repo"; repo.mkdir()
    data = tmp_path / "data"; data.mkdir()
    home = tmp_path / "home"; (home / "proj").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    return ToolRegistry(repo_dir=repo, drive_root=data), repo, home


@pytest.mark.parametrize("runtime_mode", ["light", "advanced"])
def test_default_lane_allows_mutating_git_outside_runtime(runtime_mode, tmp_path, monkeypatch):
    """Q4=A sandbox unwind: the default (non-workspace) lane is TARGET-aware in
    every runtime mode — `git init` in a user tree is legitimate task work."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", runtime_mode)
    reg, _repo, home = _outside_runtime_registry(tmp_path, monkeypatch)
    result = reg.execute("run_command", {"cmd": ["git", "init"], "cwd": str(home / "proj")})
    assert "GIT_VIA_SHELL_BLOCKED" not in result
    assert "WORKSPACE_GIT_BLOCKED" not in result


def test_default_lane_blocks_mutating_git_targeting_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    reg, repo, home = _outside_runtime_registry(tmp_path, monkeypatch)
    typed = reg.execute_result(
        "run_command",
        {"cmd": ["git", "-C", str(repo), "commit", "-m", "x"], "cwd": str(home)},
    )
    result = typed.text
    assert typed.code == "GIT_VIA_SHELL_BLOCKED"
    assert "GIT_VIA_SHELL_BLOCKED" in result
    assert "commit_reviewed" in result


def test_default_lane_allows_readonly_git_at_runtime_cwd(tmp_path, monkeypatch):
    """Read-only git stays allowed even at the system-repo cwd (v4.5.1 line)."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    reg, _repo, _home = _outside_runtime_registry(tmp_path, monkeypatch)
    result = reg.execute("run_command", {"cmd": ["git", "status"]})
    assert "GIT_VIA_SHELL_BLOCKED" not in result


def test_default_lane_allows_minusC_retarget_from_default_cwd(tmp_path, monkeypatch):
    """The default lane's DEFAULT cwd is the system repo; `git -C <outside>` must
    be judged by its effective (-C) target, not the shell cwd, or the flip
    re-creates the false-block class it removes."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    reg, _repo, home = _outside_runtime_registry(tmp_path, monkeypatch)
    result = reg.execute(
        "run_command",
        {"cmd": ["git", "-C", str(home / "proj"), "init"]},  # no cwd -> repo default
    )
    assert "GIT_VIA_SHELL_BLOCKED" not in result
    assert "WORKSPACE_GIT_BLOCKED" not in result


def test_advanced_mode_blocks_runshell_protected_python_writer(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    reg = _registry(tmp_path)
    result = reg.execute(
        "run_command",
        {"cmd": "python -c \"from pathlib import Path; Path('BIBLE.md').write_text('x')\""},
    )
    assert "SAFETY_VIOLATION" in result
    assert "BIBLE.md" in result


def test_advanced_mode_blocks_runshell_protected_backslash_path(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    reg = _registry(tmp_path)
    result = reg.execute(
        "run_command",
        {"cmd": "python -c \"open('ouroboros\\\\contracts\\\\plugin_api.py','w').write('x')\""},
    )
    assert "SAFETY_VIOLATION" in result


def test_light_mode_allows_extension_tool_dispatch(tmp_path, monkeypatch):
    """v5.1.2 Frame A: ``light`` lets reviewed + enabled extension tools dispatch."""
    from ouroboros import extension_loader

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path)
    tool_name = extension_loader.extension_surface_name("testskill", "echo")
    with extension_loader._lock:
        extension_loader._tools[tool_name] = {
            "name": tool_name,
            "handler": lambda ctx, **kwargs: "extension-tool-ran",
            "description": "echo",
            "schema": {},
            "timeout_sec": 10,
            "skill": "testskill",
        }
    monkeypatch.setattr(extension_loader, "is_extension_live", lambda *_a, **_k: True)
    unloaded: list[str] = []
    monkeypatch.setattr(extension_loader, "unload_extension", unloaded.append)
    try:
        result = reg.execute(tool_name, {})
        assert "LIGHT_MODE_BLOCKED" not in result
        assert "extension-tool-ran" in result
        assert unloaded == []
    finally:
        with extension_loader._lock:
            extension_loader._tools.pop(tool_name, None)


@pytest.mark.parametrize("bad_cmd", [
    "sed -i 's/foo/bar/' docs/README.md",
    "perl -i -pe 's/foo/bar/' docs/README.md",
    "truncate -s 0 docs/README.md",
    "chmod 755 docs/README.md",
    "chown anton docs/README.md",
    "ln -s /tmp/x docs/link",
])
def test_light_mode_blocks_inplace_mutation_tools(bad_cmd, tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path)
    result = reg.execute("run_command", {"cmd": bad_cmd})
    assert "LIGHT_MODE_BLOCKED" in result, f"cmd={bad_cmd!r}: {result[:200]}"


@pytest.mark.parametrize(("tool_name", "args"), [
    ("fetch_pr_ref", {"pr_number": 1}),
    ("create_integration_branch", {"pr_number": 1}),
    ("cherry_pick_pr_commits", {"shas": ["deadbeef"]}),
    ("stage_adaptations", {}),
    ("stage_pr_merge", {"branch": "integration/test"}),
])
def test_light_mode_blocks_pr_integration_tools(tool_name, args, tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path)
    result = reg.execute(tool_name, args)
    assert "LIGHT_MODE_BLOCKED" in result


def test_light_mode_allows_readonly_runshell(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path)
    result = reg.execute("run_command", {"cmd": "git status"})
    assert "LIGHT_MODE_BLOCKED" not in result


@pytest.mark.parametrize("cmd", [
    "mkdir /tmp/ouroboros-light-mode-scratch",
    "touch /tmp/ouroboros-light-mode-scratch-file",
    "chmod +x /tmp/ouroboros-light-mode-scratch-file",
    "sed -i 's/foo/bar/' /tmp/ouroboros-light-mode-scratch-file",
    "chown nobody /tmp/ouroboros-light-mode-scratch-file",
    "cp README.md /tmp/ouroboros-light-mode-copy-out",
    "python3 -c \"open('/tmp/ouroboros-light-mode-scratch-file', 'r').read()\"",
])
def test_light_mode_allows_non_repo_shell_file_operations(cmd, tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path)
    result = reg.execute("run_command", {"cmd": cmd})
    assert "LIGHT_MODE_BLOCKED" not in result, result[:200]


def test_advanced_mode_blocks_python_os_remove_protected_path(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    reg = _registry(tmp_path)
    result = reg.execute("run_command", {"cmd": "python3 -c \"import os; os.remove('BIBLE.md')\""})
    assert "SAFETY_VIOLATION" in result


@pytest.mark.parametrize("cmd", [
    "sort -o BIBLE.md BIBLE.md",
    "uniq BIBLE.md BIBLE.md",
])
def test_run_shell_blocks_sort_uniq_protected_output_paths(cmd, tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    reg = _registry(tmp_path)
    result = reg.execute("run_command", {"cmd": cmd})
    assert "SAFETY_VIOLATION" in result
    assert "BIBLE.md" in result or "protected" in result.lower()


@pytest.mark.parametrize("cmd", ["cat BIBLE.md", "git diff BIBLE.md", "du BIBLE.md"])
def test_run_shell_allows_readonly_mentions_of_protected_paths(cmd, tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    reg = _registry(tmp_path)
    result = reg.execute("run_command", {"cmd": cmd})
    assert "SAFETY_VIOLATION" not in result


@pytest.mark.parametrize("cmd", [
    ["bash", "-c", "printf x > README.md"],
    ["sh", "-c", "touch README.md"],
])
def test_light_mode_blocks_simple_shell_c_repo_writer(cmd, tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path)
    result = reg.execute("run_command", {"cmd": cmd})
    assert "LIGHT_MODE_BLOCKED" in result


def test_light_mode_allows_shell_wrapper_non_repo_writer(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path)
    result = reg.execute("run_command", {"cmd": ["bash", "-c", "mkdir /tmp/ouroboros-light-wrapper"]})
    assert "LIGHT_MODE_BLOCKED" not in result, result[:200]


def test_light_mode_inline_writer_is_refused_upfront(tmp_path, monkeypatch):
    """H2 (owner decision 2026-08-03): the INVERTED interpreter write fence refuses
    an inline payload it cannot prove repo-safe BEFORE execution — python gets a
    real AST proof, and a proven write is refused with nothing executed. The old
    enumerate-and-detect fence ADMITTED this exact vector and left the post-hoc
    tripwire to report the already-done write; that contract deliberately no
    longer exists, and the file staying untouched is the point."""
    import ouroboros.safety as safety_mod

    repo = _git_repo(tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.setattr(safety_mod, "check_safety", lambda *a, **k: (True, ""))
    reg = ToolRegistry(repo_dir=repo, drive_root=tmp_path / "drive")

    result = reg.execute(
        "run_command",
        {"cmd": [sys.executable, "-c", "from pathlib import Path; Path('README.md').write_text('hacked\\n')"]},
    )

    assert "LIGHT_MODE_BLOCKED" in result, result[:300]
    assert "LIGHT_MODE_REPO_WRITE_BLOCKED" not in result  # refused upfront, not detected after
    assert (repo / "README.md").read_text(encoding="utf-8") != "hacked\n"


def test_light_mode_tripwire_catches_python_repo_writer(tmp_path, monkeypatch):
    """The tripwire is the DETECTION layer BEHIND the fence: a SCRIPT-file
    invocation hands the fence nothing inline (by design — the fence judges only
    payloads it can read), executes, and the post-hoc snapshot catches the repo
    mutation. Vector updated at the H2 synthesis: the old inline vector is now
    refused upfront (see test_light_mode_inline_writer_is_refused_upfront), so
    it can no longer reach the layer this test exists to cover."""
    import ouroboros.safety as safety_mod

    repo = _git_repo(tmp_path)
    payload = tmp_path / "writer.py"
    payload.write_text("from pathlib import Path\nPath('README.md').write_text('hacked\\n')\n")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.setattr(safety_mod, "check_safety", lambda *a, **k: (True, ""))
    reg = ToolRegistry(repo_dir=repo, drive_root=tmp_path / "drive")

    result = reg.execute("run_command", {"cmd": [sys.executable, str(payload)]})

    assert "LIGHT_MODE_REPO_WRITE_BLOCKED" in result, result[:300]
    assert "README.md" in result
    assert (repo / "README.md").read_text(encoding="utf-8") == "hacked\n"


def test_light_mode_tripwire_catches_untracked_repo_file(tmp_path, monkeypatch):
    import ouroboros.safety as safety_mod

    repo = _git_repo(tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.setattr(safety_mod, "check_safety", lambda *a, **k: (True, ""))
    reg = ToolRegistry(repo_dir=repo, drive_root=tmp_path / "drive")

    payload = tmp_path / "creator.py"
    payload.write_text("from pathlib import Path\nPath('new_tool.py').write_text('x\\n')\n")
    result = reg.execute("run_command", {"cmd": [sys.executable, str(payload)]})

    assert "LIGHT_MODE_REPO_WRITE_BLOCKED" in result, result[:300]
    assert "new_tool.py" in result
    assert (repo / "new_tool.py").read_text(encoding="utf-8") == "x\n"


def test_light_mode_workspace_artifact_does_not_trip_self_repo_snapshot(tmp_path, monkeypatch):
    import ouroboros.safety as safety_mod
    from ouroboros.tools.registry import ToolContext

    system_repo = _git_repo(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data = tmp_path / "drive"
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.setattr(safety_mod, "check_safety", lambda *a, **k: (True, ""))
    reg = ToolRegistry(repo_dir=system_repo, drive_root=data)
    reg.set_context(ToolContext(
        repo_dir=system_repo,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
    ))

    result = reg.execute(
        "run_command",
        {"cmd": ["python3", "-c", "from pathlib import Path; Path('build.out').write_text('ok\\n')"]},
    )

    assert "LIGHT_MODE_REPO_WRITE_BLOCKED" not in result, result[:300]
    assert "WORKSPACE_GIT_REF_CHANGED" not in result, result[:300]
    assert (workspace / "build.out").read_text(encoding="utf-8") == "ok\n"


def test_light_mode_tripwire_runs_after_failed_command(tmp_path, monkeypatch):
    import ouroboros.safety as safety_mod

    repo = _git_repo(tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.setattr(safety_mod, "check_safety", lambda *a, **k: (True, ""))
    reg = ToolRegistry(repo_dir=repo, drive_root=tmp_path / "drive")

    payload = tmp_path / "failing_writer.py"
    payload.write_text(
        "from pathlib import Path\nPath('README.md').write_text('bad\\n')\nraise SystemExit(2)\n")
    result = reg.execute("run_command", {"cmd": [sys.executable, str(payload)]})

    assert "LIGHT_MODE_REPO_WRITE_BLOCKED" in result, result[:300]
    assert "SHELL_EXIT_ERROR" in result


def test_advanced_mode_does_not_run_light_tripwire(tmp_path, monkeypatch):
    import ouroboros.safety as safety_mod

    repo = _git_repo(tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(safety_mod, "check_safety", lambda *a, **k: (True, ""))
    reg = ToolRegistry(repo_dir=repo, drive_root=tmp_path / "drive")

    result = reg.execute(
        "run_command",
        {"cmd": [sys.executable, "-c", "from pathlib import Path; Path('README.md').write_text('advanced\\n')"]},
    )

    assert "LIGHT_MODE_REPO_WRITE_BLOCKED" not in result, result[:300]
