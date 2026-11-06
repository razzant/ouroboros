"""Protected operands follow real file roles and independent shell redirections."""
from __future__ import annotations

import os
import pathlib
import shlex
import shutil

import pytest

from ouroboros import config, safety
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.tools.shell_guards import direct_shell_rows, direct_utility_target_rows



@pytest.fixture
def protected(tmp_path, monkeypatch):
    home, data, system, workspace = [tmp_path / name for name in ("home", "data", "system", "workspace")]
    for path in (home, data, system, workspace):
        path.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "SETTINGS_PATH", data / "settings.json")
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_k: (True, ""))
    reference = workspace / "reference"
    reference.write_text("#!/bin/sh\nprintf 'reference-output\\n'\n")
    reference.chmod(0o700)
    contract = {"resource_policy": {"protected_artifacts": [{
        "id": "fixture_reference", "role": "black_box_reference", "paths": [str(reference)], "allow": ["execute"],
    }]}}
    ctx = ToolContext(repo_dir=system, system_repo_dir=system, drive_root=data,
                      workspace_root=workspace, workspace_mode="external", task_contract=contract, task_id="operands")
    registry = ToolRegistry(repo_dir=system, drive_root=data)
    registry.set_context(ctx)
    (workspace / "driver.txt").write_text("reference NOT_A_PATH=reference\n")
    (workspace / "ordinary").mkdir()
    (workspace / "ordinary" / "driver.txt").write_text("reference\n")
    return registry, ctx, reference, workspace


@pytest.mark.parametrize("cmd", [
    ["grep", "--regexp=reference", "driver.txt"], ["grep", "-e", "reference", "--", "driver.txt"],
    ["awk", "-v", "v=reference", "{print v}", "driver.txt"],
    ["find", "ordinary", "-name", "reference"],
])
def test_search_patterns_and_programs_do_not_become_file_targets(protected, cmd):
    registry, _ctx, reference, _workspace = protected
    original = reference.read_bytes()
    if cmd[0] == "find":
        # Bind the PATH-selected utility instead of Windows' System32 FIND.
        executable = shutil.which("find")
        assert executable is not None
        cmd = [executable, *cmd[1:]]
    result = registry.execute_result("run_command", {"cmd": cmd})
    assert result.status == "ok", result.text
    assert reference.read_bytes() == original


@pytest.mark.parametrize("cmd", [
    ["cat", "reference"], ["grep", "-f", "reference", "driver.txt"], ["dd", "if=driver.txt", "of=reference"],
])
def test_actual_read_copy_hash_script_and_pattern_file_remain_protected(protected, cmd):
    registry, _ctx, reference, workspace = protected
    original = reference.read_bytes()
    result = registry.execute_result("run_command", {"cmd": cmd})
    assert result.status == "blocked" and "RESOURCE_POLICY_BLOCKED" in result.text, result.text
    assert reference.read_bytes() == original and not (workspace / "copy.txt").exists()


@pytest.mark.parametrize("body", ["./reference > ./reference", "wc -c < ./reference"])
def test_shell_read_and_write_redirects_cannot_hide_behind_execute_permission(protected, body):
    registry, _ctx, reference, _workspace = protected
    original = reference.read_bytes()
    result = registry.execute_result("run_command", {"cmd": ["sh", "-c", body]})
    assert result.status == "blocked" and "RESOURCE_POLICY_BLOCKED" in result.text, result.text
    assert reference.read_bytes() == original


def test_execute_output_capture_and_ordinary_input_remain_usable(protected):
    registry, _ctx, reference, workspace = protected
    original = reference.read_bytes()
    result = registry.execute_result("run_command", {"cmd": ["sh", "-c", "./reference > output.txt"], "outputs": ["output.txt"]})
    assert result.status == "ok", result.text
    assert reference.read_bytes() == original and (workspace / "output.txt").read_text() == "reference-output\n"
    result = registry.execute_result("run_command", {"cmd": ["sh", "-c", "wc -c < driver.txt"]})
    assert result.status == "ok" and str((workspace / "driver.txt").stat().st_size) in result.text, result.text
    result = registry.execute_result("run_command", {"cmd": ["dd", "if=driver.txt", "of=copy.txt"], "outputs": ["copy.txt"]})
    assert result.status == "ok" and (workspace / "copy.txt").read_bytes() == (workspace / "driver.txt").read_bytes(), result.text


def test_literal_argv_operators_globs_and_descriptors_keep_their_roles(protected, monkeypatch):
    registry, _ctx, reference, _workspace = protected
    # Native Windows parents otherwise trigger MSYS/Cygwin globbing before rm sees argv.
    for name in ("MSYS", "CYGWIN"):
        monkeypatch.setenv(name, f"{os.environ.get(name, '')} noglob".strip())
    original = reference.read_bytes()
    for cmd in (["echo", "<", "reference"], ["echo", ">", "reference"], ["rm", "-f", "ref*"]):
        result = registry.execute_result("run_command", {"cmd": cmd})
        assert result.status == "ok", result.text
    assert reference.read_bytes() == original
    rows = direct_shell_rows(["sh", "-c", "./reference 2>&1 <&0 > out"])
    assert rows == [(["./reference"], [], ["out"], True)]
    assert direct_utility_target_rows(["sh", "-c", "wc -c < reference"]) == [(["wc", "-c"], [], (), False)]
    assert direct_shell_rows(["sh", "-c", "wc -c < reference"])[0][1] == ["reference"]


def test_sequential_cwd_and_cyber_keep_actual_operation_identity(protected, monkeypatch):
    registry, _ctx, reference, workspace = protected
    original = reference.read_bytes()
    body = f"cd {shlex.quote(str(workspace / 'ordinary'))}; wc -c < ../reference"
    result = registry.execute_result("run_command", {"cmd": ["sh", "-c", body]})
    assert result.status == "blocked" and reference.read_bytes() == original, result.text
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    result = registry.execute_result("run_command", {"cmd": ["sh", "-c", body]})
    assert result.status == "ok" and str(len(original)) in result.text, result.text
    assert reference.read_bytes() == original


@pytest.mark.parametrize("command", ["cp", "mv", "ln"])
def test_directory_destination_only_targets_the_actual_child(protected, command):
    registry, _ctx, reference, workspace = protected
    original = reference.read_bytes()
    (workspace / "ordinary/input.txt").write_text("ordinary input\n")
    result = registry.execute_result("run_command", {
        "cmd": [command, "ordinary/input.txt", "."], "outputs": ["input.txt"],
    })
    assert result.status == "ok", result.text
    assert (workspace / "input.txt").read_text() == "ordinary input\n"
    assert reference.read_bytes() == original


@pytest.mark.parametrize("command", ["cp", "mv", "ln"])
def test_directory_destination_cannot_overwrite_the_protected_child(protected, command):
    registry, _ctx, reference, workspace = protected
    original = reference.read_bytes()
    (workspace / "ordinary/reference").write_text("ordinary source")
    result = registry.execute_result("run_command", {"cmd": [command, "ordinary/reference", "."]})
    assert result.status == "blocked" and "RESOURCE_POLICY_BLOCKED" in result.text, result.text
    assert reference.read_bytes() == original
