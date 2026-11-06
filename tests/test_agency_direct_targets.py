"""Certain utility writes use resource authority without interpreting program bodies."""
from __future__ import annotations

import os
import pathlib
import shlex
import shutil
import sys
import tarfile

import pytest

from ouroboros import config, safety
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.tools.shell_guards import direct_utility_target_rows



@pytest.fixture
def resources(tmp_path, monkeypatch):
    home, system, data = [tmp_path / name for name in ("home", "system", "data")]
    workspace = home / "project"
    for path in (home, system, data, workspace, home / "Desktop"):
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
                      workspace_root=workspace, workspace_mode="external", task_id="direct-targets")
    registry = ToolRegistry(repo_dir=system, drive_root=data)
    registry.set_context(ctx)
    return registry, ctx, home, system, data, workspace


def mode(monkeypatch, name):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", name)
    monkeypatch.setattr(config, "get_runtime_mode", lambda: name)


@pytest.mark.parametrize("utility", ["touch", "cp", "rm", "sort", "redirect"])
@pytest.mark.parametrize("destination", ["system", "data", "desktop"])
def test_light_direct_target_matches_resource_authority(resources, monkeypatch, utility, destination):
    registry, _ctx, home, system, data, workspace = resources
    mode(monkeypatch, "light")
    target = {"system": system, "data": data, "desktop": home / "Desktop"}[destination] / "output.txt"
    source = system / "input.txt"
    source.write_text("z\na\n")
    if utility == "rm":
        target.write_text("retained unless authorized")
    commands = {
        "touch": ["touch", str(target)], "cp": ["cp", str(source), str(target)],
        "rm": ["rm", str(target)], "sort": ["sort", "-o", str(target), str(source)],
        "redirect": ["sh", "-c", f"printf x > {shlex.quote(str(target))}"],
    }
    if utility == "sort":
        # Windows searches System32 before PATH for a bare executable name.
        executable = shutil.which("sort")
        assert executable is not None
        commands[utility][0] = executable
    result = registry.execute_result("run_command", {"cmd": commands[utility], "cwd": str(workspace),
        "outputs": [str(target)] if utility != "rm" else []})
    if destination == "desktop":
        assert result.status == "ok", result.text
        assert target.exists() is (utility != "rm")
    else:
        assert result.status == "blocked", result.text
        assert target.exists() is (utility == "rm")
        if utility == "rm":
            assert target.read_text() == "retained unless authorized"


@pytest.mark.parametrize("runtime", ["advanced", "pro", "cyber_pro"])
def test_protected_direct_target_uses_effective_runtime_mode(resources, monkeypatch, runtime):
    registry, _ctx, _home, system, _data, workspace = resources
    mode(monkeypatch, runtime)
    target = system / "BIBLE.md"
    result = registry.execute_result("run_command", {"cmd": ["touch", str(target)], "cwd": str(workspace)})
    assert target.exists() is (runtime != "advanced")
    assert result.status == ("blocked" if runtime == "advanced" else "ok"), result.text


@pytest.mark.parametrize("runtime", ["advanced", "cyber_pro"])
def test_acting_write_root_and_real_symlink_target(resources, monkeypatch, runtime):
    registry, ctx, home, _system, _data, workspace = resources
    mode(monkeypatch, runtime)
    ctx.task_constraint = TaskConstraint(mode="acting_subagent", surface="external_workspace", write_root=str(workspace))
    assert registry.execute_result("run_command", {"cmd": ["touch", "inside.txt"]}).status == "ok"
    outside = home / "Desktop" / "outside.txt"
    (workspace / "link").symlink_to(outside.parent, target_is_directory=True)
    result = registry.execute_result("run_command", {"cmd": ["touch", "link/outside.txt"], "outputs": [str(outside)]})
    assert outside.exists() is (runtime == "cyber_pro")
    assert result.status == ("ok" if runtime == "cyber_pro" else "blocked"), result.text


@pytest.mark.parametrize("cmd", [
    ["echo", ">", "/runtime/settings.json"], ["sed", "-f", "program.sed", "/runtime/file"],
    ["sh", "-c", "printf '%s' '>' /runtime/settings.json"],
])
def test_unknown_bodies_and_literal_operators_do_not_manufacture_targets(cmd):
    original = list(cmd)
    assert not any(row[1] for row in direct_utility_target_rows(cmd))
    assert cmd == original


def test_literal_operator_reaches_real_program_without_touching_named_path(resources, monkeypatch):
    registry, _ctx, _home, system, _data, workspace = resources
    mode(monkeypatch, "light")
    target = system / "untouched.txt"
    result = registry.execute_result("run_command", {"cmd": ["echo", ">", str(target)], "cwd": str(workspace)})
    assert result.status == "ok", result.text
    assert not target.exists()
    script = "import io; f=io.StringIO(); f.write('settings.json'); print(f.getvalue())"
    assert registry.execute_result("run_command", {"cmd": [sys.executable, "-c", script]}).status == "ok"


def test_explicit_shell_cwd_and_outer_redirect_are_checked(resources, monkeypatch):
    registry, _ctx, _home, system, _data, workspace = resources
    mode(monkeypatch, "light")
    result = registry.execute_result("run_command", {"cmd": ["sh", "-c", f"cd {shlex.quote(str(system))}; touch target.txt"]})
    assert result.status == "blocked" and not (system / "target.txt").exists(), result.text
    result = registry.execute_result("run_command", {"cmd": ["sh", "-c", f"sh -c ':' > {shlex.quote(str(system / 'outer.txt'))}"]})
    assert result.status == "blocked" and not (system / "outer.txt").exists(), result.text
    result = registry.execute_result("run_command", {"cmd": ["sh", "-c", f"cd {shlex.quote(str(workspace))}; touch target.txt"]})
    assert result.status == "ok" and (workspace / "target.txt").exists(), result.text


@pytest.mark.parametrize("utility", ["cp", "ln", "mv"])
def test_directory_destination_binds_the_actual_child(resources, monkeypatch, utility):
    registry, _ctx, _home, system, _data, workspace = resources
    mode(monkeypatch, "advanced")
    source = workspace / "BIBLE.md"
    source.write_text("exact candidate")
    result = registry.execute_result("run_command", {"cmd": [utility, str(source), str(system)]})
    assert result.status == "blocked", result.text
    assert source.read_text() == "exact candidate" and not (system / "BIBLE.md").exists()
    destination = workspace / "output"
    destination.mkdir()
    result = registry.execute_result("run_command", {"cmd": [utility, str(source), str(destination)]})
    assert result.status == "ok", result.text
    assert (destination / "BIBLE.md").read_text() == "exact candidate"


def test_cyber_keeps_explicit_readonly_task(resources, monkeypatch):
    registry, ctx, _home, _system, _data, workspace = resources
    mode(monkeypatch, "cyber_pro")
    ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent")
    result = registry.execute_result("run_command", {"cmd": ["touch", "not-created"]})
    assert result.status == "blocked" and not (workspace / "not-created").exists()


@pytest.mark.parametrize(("marker", "runtime"), [
    (".seed-origin", "light"),
    (".seed-origin", "cyber_pro"),
    (".clawhub.json", "advanced"),
    (".clawhub.json", "pro"),
])
def test_selected_payload_actual_marker_write_and_ordinary_file(resources, monkeypatch, runtime, marker):
    registry, _ctx, _home, _system, data, _workspace = resources
    from tests.test_skill_exec import _build_skill

    payload = _build_skill(data / "skills" / "external", "alpha")
    mode(monkeypatch, runtime)
    args = {"cwd": "skill_payload", "bucket": "external", "skill_name": "alpha"}
    ordinary = registry.execute_result("run_command", {**args, "cmd": ["touch", "notes.txt"]})
    assert ordinary.status == "ok" and (payload / "notes.txt").exists(), ordinary.text
    result = registry.execute_result("run_command", {**args, "cmd": ["touch", marker]})
    assert (payload / marker).exists() is (runtime == "cyber_pro")
    assert result.status == ("ok" if runtime == "cyber_pro" else "blocked"), result.text


@pytest.mark.parametrize("utility", ["sed", "uniq", "tar_archive_first", "tar_directory_first", "gzip", "rsync"])
@pytest.mark.parametrize("runtime,outside", [("advanced", False), ("advanced", True), ("cyber_pro", True)])
def test_explicit_utility_roles_preserve_source_and_destination(resources, monkeypatch, utility, runtime, outside):
    registry, ctx, home, system, _data, workspace = resources
    mode(monkeypatch, runtime)
    ctx.task_constraint = TaskConstraint(mode="acting_subagent", surface="external_workspace", write_root=str(workspace))
    source = system / "input.txt"
    source.write_text("a\na\nb\n")
    archive = system / "input.tar"
    with tarfile.open(archive, "w") as stream:
        stream.add(source, arcname="unpacked.txt")
    target = (home / "Desktop" if outside else workspace) / "output"
    target.mkdir()
    output = target / "value.txt"
    before = "replace this existing file\n" if utility == "rsync" else "a\na\nb\n"
    output.write_text(before)
    commands = {
        "sed": ["sed", "-i", *([""] if sys.platform == "darwin" else []), "s/a/z/", str(output)],
        "uniq": ["uniq", str(source), str(output)],
        "tar_archive_first": ["tar", "-xf", str(archive), "-C", str(target)],
        "tar_directory_first": ["tar", "-C", str(target), "-xf", str(archive)],
        "gzip": ["gzip", str(output)],
        # A Windows drive colon selects an rsync remote host; these operands are local.
        "rsync": ["rsync", pathlib.Path(os.path.relpath(source, workspace)).as_posix(),
                  pathlib.Path(os.path.relpath(output, workspace)).as_posix()],
    }
    declared = target if utility.startswith("tar_") else pathlib.Path(str(output) + ".gz") if utility == "gzip" else output
    result = registry.execute_result("run_command", {"cmd": commands[utility], "outputs": [str(declared)]})
    assert source.read_text() == "a\na\nb\n"
    if outside and runtime == "advanced":
        assert result.status == "blocked", result.text
        assert output.read_text() == before
        assert not (target / "unpacked.txt").exists() and not pathlib.Path(str(output) + ".gz").exists()
    else:
        assert result.status == "ok", result.text
        if utility == "sed":
            assert output.read_text() == "z\nz\nb\n"
        elif utility == "uniq":
            assert output.read_text() == "a\nb\n"
        elif utility.startswith("tar_"):
            assert (target / "unpacked.txt").read_bytes() == source.read_bytes()
        elif utility == "gzip":
            assert not output.exists() and declared.is_file()
        else:
            assert output.read_bytes() == source.read_bytes()
