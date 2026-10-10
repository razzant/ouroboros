"""Owner decision 5A (2026-09-21): a TOP-LEVEL principal's shell write is fenced by
explicit evidence only — a real output redirect or a utility with a certain
destination grammar (cp/mv/ln, dd of=, sed -i, tar -C, rsync). The word guess
(``touch``/``rm``/``mkdir``/``tee``/``sort``/``uniq``/``gzip`` + option skipping)
that refused ``rm -rf /tmp/x`` while ``mkdir -p /tmp/x`` and ``git clone … /tmp/x``
passed is gone for top-level tasks and stays for subordinate subagents, whose
write confinement is their contract.

Both directions, through the surviving guards (CHECKLISTS item 7):
top-level word-guessed writes outside every root → allowed; the same argv from an
acting or read-only subagent → refused, naming the reason and the writable roots;
protected repo paths and light mode → refused for everyone; explicit shell syntax
(a redirect) into a place no root admits → still refused with the real reason.
"""
from __future__ import annotations

import pathlib
import shlex

import pytest

from ouroboros import config, safety
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tools.registry import ToolContext, ToolRegistry

pytestmark = pytest.mark.serial


@pytest.fixture
def principal(tmp_path, monkeypatch):
    """A NON-external top-level principal (a direct owner turn) whose user_files root
    is a jailed home: OS scratch outside it is exactly the /tmp class of 5A."""
    home, system, data, scratch = [tmp_path / name for name in ("home", "system", "data", "scratch")]
    for path in (home / "project", system, data, scratch, system / "prompts", system / "ouroboros"):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "SETTINGS_PATH", data / "settings.json")
    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_kw: (True, ""))
    ctx = ToolContext(repo_dir=system, system_repo_dir=system, drive_root=data,
                      task_id="fence-5a", is_direct_chat=True)
    registry = ToolRegistry(repo_dir=system, drive_root=data)
    registry.set_context(ctx)
    return registry, ctx, home, system, data, scratch


def _acting(ctx: ToolContext, workspace: pathlib.Path) -> None:
    """Turn the context into an acting child confined to ``workspace``."""
    workspace.mkdir(exist_ok=True)
    ctx.workspace_root, ctx.workspace_mode = workspace, "external"
    ctx.task_constraint = TaskConstraint(mode="acting_subagent", surface="external_workspace", write_root=str(workspace))


def _mode(monkeypatch, name):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", name)
    monkeypatch.setattr(config, "get_runtime_mode", lambda: name)


def _word_commands(scratch: pathlib.Path) -> dict:
    victim = scratch / "victim"
    victim.mkdir(exist_ok=True)
    (victim / "f.txt").write_text("x", encoding="utf-8")
    return {
        "rm": (["rm", "-rf", str(victim)], victim, False),
        "mkdir": (["mkdir", str(scratch / "made")], scratch / "made", True),
        "touch": (["touch", str(scratch / "touched.txt")], scratch / "touched.txt", True),
    }


@pytest.mark.parametrize("utility", ["rm", "mkdir", "touch"])
@pytest.mark.parametrize("runtime", ["light", "advanced"])
def test_top_level_word_guessed_write_outside_every_root_runs(principal, monkeypatch, utility, runtime):
    registry, _ctx, home, _system, _data, scratch = principal
    _mode(monkeypatch, runtime)
    cmd, path, exists_after = _word_commands(scratch)[utility]
    result = registry.execute_result("run_command", {"cmd": cmd, "cwd": str(home / "project")})
    assert result.status == "ok", result.text
    assert path.exists() is exists_after


@pytest.mark.parametrize("utility", ["rm", "mkdir", "touch"])
def test_acting_subagent_keeps_the_word_fence_and_learns_where_it_may_write(principal, monkeypatch, utility):
    registry, ctx, home, _system, _data, scratch = principal
    _mode(monkeypatch, "advanced")
    workspace = home / "child-ws"
    _acting(ctx, workspace)
    cmd, path, exists_after = _word_commands(scratch)[utility]
    result = registry.execute_result("run_command", {"cmd": cmd})
    assert result.status == "blocked", result.text
    assert path.exists() is (not exists_after)  # nothing ran
    assert str(path) in result.text
    assert "outside every root this task may write" in result.text
    assert f"active_workspace={workspace.resolve()}" in result.text
    assert "task_drive=" not in result.text and "user_files=" not in result.text  # not writable for it
    # The same child still writes inside its own root: the confinement, not a blanket.
    assert registry.execute_result("run_command", {"cmd": ["touch", "inside.txt"]}).status == "ok"


@pytest.mark.parametrize("utility", ["rm", "mkdir", "touch"])
def test_read_only_subagent_has_no_shell_at_all(principal, monkeypatch, utility):
    registry, ctx, _home, _system, _data, scratch = principal
    _mode(monkeypatch, "advanced")
    ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent")
    cmd, path, exists_after = _word_commands(scratch)[utility]
    result = registry.execute_result("run_command", {"cmd": cmd})
    assert result.status == "blocked", result.text
    assert path.exists() is (not exists_after)
    assert "LOCAL_READONLY_SUBAGENT_BLOCKED" in result.text and "shell" in result.text


@pytest.mark.parametrize("rel", ["BIBLE.md", "prompts/SAFETY.md", "ouroboros/safety.py"])
def test_protected_paths_stay_refused_for_a_top_level_principal(principal, monkeypatch, rel):
    registry, _ctx, _home, system, _data, _scratch = principal
    _mode(monkeypatch, "advanced")
    target = system / rel
    result = registry.execute_result("run_command", {"cmd": ["touch", str(target)], "cwd": str(system)})
    assert result.status == "blocked", result.text
    assert not target.exists()
    assert "protected" in result.text.lower() and rel in result.text


@pytest.mark.parametrize("rel", ["BIBLE.md", "prompts/SAFETY.md"])
def test_protected_paths_stay_refused_for_an_acting_child(principal, monkeypatch, rel):
    registry, ctx, home, system, _data, _scratch = principal
    _mode(monkeypatch, "advanced")
    _acting(ctx, home / "ws")
    result = registry.execute_result("run_command", {"cmd": ["touch", str(system / rel)]})
    assert result.status == "blocked", result.text
    assert not (system / rel).exists()


def test_light_mode_still_keeps_the_repository_read_only_for_a_top_level_principal(principal, monkeypatch):
    registry, _ctx, home, system, _data, _scratch = principal
    _mode(monkeypatch, "light")
    target = system / "ordinary.py"
    result = registry.execute_result("run_command", {"cmd": ["touch", str(target)], "cwd": str(home / "project")})
    assert result.status == "blocked" and result.code == "LIGHT_MODE_BLOCKED", result.text
    assert not target.exists()
    assert "light" in result.text.lower()


def test_runtime_data_drive_word_guess_is_light_only_for_a_top_level_principal(principal, monkeypatch):
    """The runtime data drive: light mode still refuses a word-guessed write (the
    light contract keeps repo and runtime read-only); in advanced mode the bare
    word no longer fences a top-level task, while explicit shell syntax into the
    drive is still refused with the real reason and the writable roots."""
    registry, _ctx, home, _system, data, _scratch = principal
    (data / "logs").mkdir()
    target = data / "logs" / "planted.jsonl"
    _mode(monkeypatch, "light")
    result = registry.execute_result("run_command", {"cmd": ["touch", str(target)], "cwd": str(home / "project")})
    assert result.status == "blocked" and result.code == "LIGHT_MODE_BLOCKED", result.text
    assert not target.exists() and "runtime_mode=light" in result.text
    _mode(monkeypatch, "advanced")
    result = registry.execute_result(
        "run_command", {"cmd": ["sh", "-c", f"printf x > {shlex.quote(str(target))}"], "cwd": str(home / "project")})
    assert result.status == "blocked" and "outside every root this task may write" in result.text, result.text
    assert not target.exists() and "user_files declines it" in result.text
    result = registry.execute_result("run_command", {"cmd": ["touch", str(target)], "cwd": str(home / "project")})
    assert result.status == "ok" and target.exists(), result.text


def test_explicit_redirect_outside_every_root_is_still_fenced_and_names_the_writable_roots(principal, monkeypatch):
    """The surviving top-level fence is EVIDENCE-based: a real output redirect is
    explicit shell syntax, not a guess. Its refusal names the reason and every
    root this task may write (task_drive / artifact_store paths included)."""
    registry, _ctx, home, _system, data, scratch = principal
    _mode(monkeypatch, "advanced")
    target = scratch / "redirected.txt"
    result = registry.execute_result(
        "run_command", {"cmd": ["sh", "-c", f"printf x > {shlex.quote(str(target))}"], "cwd": str(home / "project")})
    assert result.status == "blocked", result.text
    assert not target.exists()
    assert "outside every root this task may write" in result.text
    assert f"task_drive={data.resolve() / 'task_drives' / 'fence-5a'}" in result.text
    assert "artifact_store=" in result.text and f"user_files={home.resolve()}" in result.text
    assert "outside the user_files home" in result.text  # the real reason user_files declined it
