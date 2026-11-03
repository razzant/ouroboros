"""Backend cwd aliases use the shared authority binding on every process door."""
from __future__ import annotations

import json
import pathlib

import pytest

from ouroboros import workspace_executor as executor
from ouroboros.tool_access import build_resolved_resource_binding as bind
from ouroboros.tools.registry import ToolContext, ToolRegistry

pytestmark = pytest.mark.serial


def _shared_absolute(path):
    """A container-rooted spelling that also names the host path on the current drive."""
    text = path.as_posix().removeprefix(path.drive)
    assert text.startswith("/") and pathlib.Path(text).resolve() == path.resolve()
    return text


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # rooted Windows paths inherit this fixture's drive
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    active, data, system = [tmp_path / name for name in ("active", "data", "system")]
    for path in (active / "sub", data, system):
        path.mkdir(parents=True)
    ctx = ToolContext(repo_dir=system, drive_root=data, task_id="cwd-test",
                      workspace_root=active, workspace_mode="external", executor_ref={
        "type": "docker_exec", "container_name": "fixture", "id": "fixture",
        "workspace_host_path": str(active), "workspace_backend_path": "/workspace/project"})
    return ctx, active


SURFACES = [
    ("run_command", {"cmd": ["python3", "-c", "print('bound')"]}),
    ("run_script", {"script": "print('bound')", "interpreter": "python3"}),
    ("start_service", {"name": "preview", "cmd": ["python3", "-c", "print('bound')"]}),
    ("verify_and_record", {"contract_kind": "explicit_command", "check": ["python3", "-c", "print('bound')"]}),
]


@pytest.mark.parametrize("surface,arguments", SURFACES)
def test_registry_routes_alias_through_guards_and_reads_back_marker(workspace, monkeypatch, surface, arguments):
    from ouroboros.tools import shell, services
    import ouroboros.safety as safety

    ctx, active = workspace
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    observed = []

    def check(_name, _args, **kwargs):
        binding = kwargs["resolved_binding"]
        if _name == "read_file":
            assert binding.target_path == (active / "sub" / "marker.txt").resolve()
            return True, ""
        assert binding.root == "active_workspace"
        assert binding.base_path == active.resolve()
        assert binding.target_path == (active / "sub").resolve()
        observed.append(binding)
        return True, ""

    def effect(_ctx, cmd, cwd, timeout_sec, **kwargs):
        assert observed and cwd == observed[-1].target_path
        assert executor.map_host_path(executor.executor_ref_from_ctx(ctx), cwd) == "/workspace/project/sub"
        if surface == "run_script":
            assert cmd[0] == "python3"  # never a host virtualenv interpreter
            staged = executor.map_backend_path(executor.executor_ref_from_ctx(ctx), cmd[1])
            assert staged.read_text() == "print('bound')"
        (cwd / "marker.txt").write_text("bound")
        return executor.ExecutorResult(0, "bound\n", operation_outcome="completed")

    def service(_ctx, **kwargs):
        effect(_ctx, kwargs["cmd"], kwargs["host_cwd"], 5)
        assert kwargs["cwd_root"] == observed[-1].root
        assert kwargs["cwd_source"] == observed[-1].source
        return {"state": "running", "service_id": "cwd-test:preview"}

    monkeypatch.setattr(safety, "check_safety", check)
    monkeypatch.setattr(shell, "executor_execute", effect)
    monkeypatch.setattr(executor, "execute", effect)
    monkeypatch.setattr(services, "executor_start_service", service)
    result = registry.execute(surface, {**arguments, "cwd": "/workspace/project/sub"})
    assert (active / "sub" / "marker.txt").is_file(), result
    assert len(observed) == 1
    readback = registry.execute("read_file", {"path": "/workspace/project/sub/marker.txt"})
    assert "bound" in readback



@pytest.mark.parametrize("surface,arguments", SURFACES)
def test_file_and_process_consumers_pick_one_target_for_a_colliding_spelling(workspace, monkeypatch, surface, arguments):
    """read_file and every process door resolve the same text to the same directory.

    The backend prefix is the workspace's parent, so ``<active>/sub`` is both a
    real host path inside active_workspace and the alias of ``<active>/active/sub``;
    ``<parent>/only`` exists only as an alias. Distinct bytes name the winner.
    """
    from ouroboros.tools import services

    ctx, active = workspace
    parent = active.parent
    ctx.executor_ref["workspace_backend_path"] = _shared_absolute(parent)
    host, shadow, alias = active / "sub", active / active.name / "sub", active / "only"
    for directory, marker in ((host, "host"), (shadow, "shadow"), (alias, "alias")):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "marker.txt").write_text(marker)
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    ran = []

    def effect(_ctx, cmd, cwd, timeout_sec, **kwargs):
        ran.append((cwd / "marker.txt").read_text())
        return executor.ExecutorResult(0, ran[-1] + "\n", operation_outcome="completed")

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setattr("ouroboros.tools.shell.executor_execute", effect)
    monkeypatch.setattr(executor, "execute", effect)
    monkeypatch.setattr(services, "executor_start_service", lambda _ctx, **kw: (
        effect(_ctx, kw["cmd"], kw["host_cwd"], 5) and {"state": "running", "service_id": "cwd-test:preview"}))
    for text, expected in ((_shared_absolute(host), "host"), (f"{_shared_absolute(parent)}/only", "alias")):
        read = registry.execute("read_file", {"path": f"{text}/marker.txt"})
        shown = {marker for marker in ("host", "shadow", "alias") if f"\t{marker}" in read}
        assert shown == {expected}, read
        registry.execute(surface, {**arguments, "cwd": text})
        assert ran[-1] == expected, (surface, text, ran)
    assert len(ran) == 2

@pytest.mark.parametrize("surface,arguments", SURFACES)
@pytest.mark.parametrize("cwd", ["/workspace/project/../escape", "/foreign/sub"])
def test_registry_refuses_matched_invalid_and_outside_alias(workspace, monkeypatch, surface, arguments, cwd):
    ctx, active = workspace
    ctx.executor_ref["path_mappings"] = [{"host_path": str(ctx.drive_root), "backend_path": "/foreign"}]
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: pytest.fail("invalid cwd reached safety"))
    result = registry.execute(surface, {**arguments, "cwd": cwd})
    assert "BLOCKED" in result, result
    assert "escapes executor mapping" in result or "outside active_workspace" in result


@pytest.mark.parametrize("surface,arguments", SURFACES)
def test_identity_mount_keeps_permitted_host_root_on_every_process_door(workspace, monkeypatch, surface, arguments):
    from ouroboros.tools import services

    ctx, active = workspace
    scratch = active.parent / "scratch"
    scratch.mkdir()
    ctx.executor_ref["path_mappings"] = [
        {"host_path": str(scratch), "backend_path": _shared_absolute(scratch)},
        {"host_path": str(ctx.drive_root / "task_drives" / ctx.task_id), "backend_path": "/task"},
    ]  # run_script also needs its existing task-drive staging mount.
    ctx.executor_ref["type"] = "local"
    assert bind(ctx, operation="shell", process_cwd=str(scratch)).root == "user_files"
    ctx.executor_ref["type"] = "docker_exec"
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    calls = []

    def safety(_name, _args, **kwargs):
        binding = kwargs["resolved_binding"]
        assert binding.root == "user_files"
        assert binding.target_path == scratch.resolve()
        return True, ""

    def effect(_ctx, cmd, cwd, timeout_sec, **kwargs):
        assert cwd == scratch.resolve()
        assert executor.map_host_path(executor.executor_ref_from_ctx(ctx), cwd) == _shared_absolute(scratch)
        calls.append(cwd)
        return executor.ExecutorResult(0, "bound\n", operation_outcome="completed")

    monkeypatch.setattr("ouroboros.safety.check_safety", safety)
    monkeypatch.setattr("ouroboros.tools.shell.executor_execute", effect)
    monkeypatch.setattr(executor, "execute", effect)
    monkeypatch.setattr(services, "executor_start_service", lambda _ctx, **kw: (
        effect(_ctx, kw["cmd"], kw["host_cwd"], 5) and {"state": "running", "service_id": "cwd-test:preview"}))
    result = registry.execute(surface, {**arguments, "cwd": _shared_absolute(scratch)})
    assert calls == [scratch.resolve()], result


def test_host_workspace_precedence_and_other_host_roots_do_not_bypass_alias(workspace):
    ctx, active = workspace
    ctx.executor_ref["workspace_backend_path"] = _shared_absolute(active.parent)
    physical = bind(ctx, operation="shell", process_cwd=_shared_absolute(active / "sub"))
    assert physical.target_path == (active / "sub").resolve()
    # An identity mount cannot grant process access to the runtime data root.
    ctx.executor_ref["path_mappings"] = [{"host_path": str(ctx.drive_root), "backend_path": _shared_absolute(ctx.drive_root)}]
    with pytest.raises(ValueError):
        bind(ctx, operation="shell", process_cwd=_shared_absolute(ctx.drive_root))
    # copy_apply_refusal uses operation=write: its physical copy target is unchanged.
    copy_target = active.parent / "physical-copy"
    copy_target.mkdir()
    ctx.executor_ref["path_mappings"].append({"host_path": str(ctx.drive_root), "backend_path": _shared_absolute(copy_target)})
    with pytest.raises(ValueError, match="outside active_workspace"):
        bind(ctx, operation="shell", process_cwd=_shared_absolute(copy_target))
    write = bind(ctx, operation="write", process_cwd=_shared_absolute(copy_target))
    assert write.target_path == copy_target.resolve()


def test_longest_prefix_boundary_and_no_match_are_distinct(workspace):
    ctx, active = workspace
    ctx.executor_ref["path_mappings"] = [{"host_path": str(ctx.drive_root), "backend_path": "/workspace/project/private"}]
    with pytest.raises(ValueError, match="outside active_workspace"):
        bind(ctx, operation="shell", process_cwd="/workspace/project/private/sub")
    with pytest.raises(ValueError, match="escapes executor mapping") as bad:
        executor.map_backend_path(executor.executor_ref_from_ctx(ctx), "/workspace/project/private/../sub")
    assert not isinstance(bad.value, executor.BackendPathNoMatch)
    for text in ("/workspace/projectish", "C:/host/scratch"):
        with pytest.raises(executor.BackendPathNoMatch):
            executor.map_backend_path(executor.executor_ref_from_ctx(ctx), text)
    scratch = active.parent / "scratch"
    scratch.mkdir()
    fallback = bind(ctx, operation="shell", process_cwd=str(scratch))
    assert (fallback.root, fallback.target_path) == ("user_files", scratch.resolve())
    ctx.executor_ref["type"] = "local"
    local = bind(ctx, operation="shell", process_cwd=str(scratch))
    assert local == fallback


def test_symlink_escape_is_not_reinterpreted_as_scratch(workspace):
    ctx, active = workspace
    try:
        (active / "link").symlink_to(ctx.drive_root, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="outside active_workspace"):
        bind(ctx, operation="shell", process_cwd="/workspace/project/link")


@pytest.mark.parametrize("cwd", ["", ".", "sub", "active_workspace/sub", "task_drive"])
def test_existing_cwd_spellings_keep_same_binding_as_local_executor(workspace, cwd):
    ctx, _active = workspace
    docker = bind(ctx, operation="shell", process_cwd=cwd)
    ctx.executor_ref["type"] = "local"
    assert bind(ctx, operation="shell", process_cwd=cwd) == docker


def test_alias_keeps_protected_artifact_guard(workspace, monkeypatch):
    ctx, active = workspace
    (active / "sub" / "secret.bin").write_text("private")
    ctx.task_contract = {"resource_policy": {"protected_artifacts": [
        {"id": "reference", "role": "black_box_reference", "paths": [str(active / "sub" / "secret.bin")]}]}}
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    monkeypatch.setattr("ouroboros.tools.shell.executor_execute", lambda *a, **k: pytest.fail("protected artifact reached executor"))
    result = registry.execute("run_command", {"cmd": ["cat", "secret.bin"], "cwd": "/workspace/project/sub"})
    assert "RESOURCE_POLICY_BLOCKED" in result


def test_readonly_child_cannot_gain_process_access_via_alias(workspace, monkeypatch):
    ctx, active = workspace
    ctx.task_constraint = {"mode": "local_readonly_subagent"}
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    monkeypatch.setattr("ouroboros.tools.shell.executor_execute", lambda *a, **k: pytest.fail("readonly process"))
    for cwd in (str(active / "sub"), "/workspace/project/sub"):
        result = registry.execute("run_command", {"cmd": ["true"], "cwd": cwd})
        assert "BLOCKED" in result, result
    # The same alias still names inherited readable content.
    (active / "sub" / "readable.txt").write_text("visible")
    assert "visible" in registry.execute("read_file", {"path": "/workspace/project/sub/readable.txt"})


def test_malformed_executor_does_not_change_host_cwd_binding(workspace):
    ctx, active = workspace
    expected = bind(ctx, operation="shell", process_cwd=str(active / "sub"))
    ctx.executor_ref = {"type": "docker_exec"}  # no usable mapping/container
    assert bind(ctx, operation="shell", process_cwd=str(active / "sub")) == expected


def test_matched_traversal_operand_keeps_container_identity_for_protected_guard(workspace, monkeypatch):
    ctx, active = workspace
    (active / "sub" / "secret.bin").write_text("private")
    (active / "scratch").mkdir()
    ctx.task_contract = {"resource_policy": {"protected_artifacts": [
        {"id": "reference", "role": "black_box_reference", "paths": [str(active / "sub" / "secret.bin")]}]}}
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    ran = []
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    monkeypatch.setattr("ouroboros.tools.shell.executor_execute", lambda _ctx, cmd, *a, **k: (
        ran.append(cmd) or executor.ExecutorResult(0, "", operation_outcome="completed")))
    # Inside the container this names /workspace/project, which holds the artifact;
    # a host reinterpretation of the unmappable spelling would see no directory.
    blocked = registry.execute("run_command", {"cmd": ["grep", "-r", "x", "/workspace/project/scratch/.."]})
    assert "RESOURCE_POLICY_BLOCKED" in blocked and not ran
    allowed = registry.execute("run_command", {"cmd": ["grep", "-r", "x", "/workspace/project/sub/../scratch"]})
    assert "RESOURCE_POLICY_BLOCKED" not in allowed and len(ran) == 1


def test_declared_output_refuses_matched_traversal_instead_of_host_file(workspace):
    from ouroboros.tools.shell_outputs import _resolve_declared_output

    ctx, active = workspace
    # The backend spelling is also a real host path, so a host reinterpretation
    # of the traversal would silently select active/out.txt; the container wrote
    # <backend>/active/out.txt, which is active/active/out.txt on the host.
    ctx.executor_ref["workspace_backend_path"] = _shared_absolute(active.parent)
    (active / "out.txt").write_text("decoy")
    source, reason = _resolve_declared_output(ctx, f"{_shared_absolute(active.parent)}/x/../active/out.txt", active)
    assert source is None and "not a usable backend path" in reason
    mapped, reason = _resolve_declared_output(ctx, f"{_shared_absolute(active.parent)}/out.txt", active)
    assert (mapped, reason) == ((active / "out.txt").resolve(), "")


def test_output_audit_ignores_matched_traversal_but_keeps_unmapped_host_output(tmp_path, monkeypatch):
    from ouroboros.tools.shell_audit import _mentioned_user_file_outputs_without_declaration

    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    desktop, work = home / "Desktop", tmp_path / "work"
    for path in (desktop, work, tmp_path / "repo", tmp_path / "data"):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data", task_id="audit", executor_ref={
        "type": "docker_exec", "container_name": "fixture",
        "workspace_host_path": str(work), "workspace_backend_path": f"{_shared_absolute(home)}/mount"})
    for name in ("decoy.txt", "real.txt"):
        (desktop / name).write_text("fresh")
    mentioned = _mentioned_user_file_outputs_without_declaration(ctx, [
        "touch", f"{_shared_absolute(home)}/mount/../Desktop/decoy.txt", f"{_shared_absolute(home)}/Desktop/real.txt",
    ], None, cwd=work)
    assert str((desktop / "real.txt").resolve()) in mentioned
    assert not any("decoy" in item for item in mentioned)


@pytest.mark.parametrize("surface", ["run_command", "start_service"])
def test_a_relative_output_resolved_outside_backend_mounts_is_refused_before_launch(workspace, monkeypatch, surface):
    """A physical container escape stays refused before either process door."""
    ctx, active = workspace
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    def probe(argv):
        if argv[0] == "inspect":
            return json.dumps({"id": "physical-id", "running": True, "network": "host",
                               "mounts": [{"Type": "bind", "Source": str(active),
                                           "Destination": "/workspace/project"}]})
        assert argv[-2:] == ["/workspace/project/sub/../../out.txt", "prospective"]
        return "/workspace/out.txt\0"
    monkeypatch.setattr("ouroboros.workspace_executor_paths._docker_output_probe", probe)
    launched = []
    monkeypatch.setattr("ouroboros.tools.shell.executor_execute", lambda *a, **k: launched.append(a) or executor.ExecutorResult(0, ""))
    monkeypatch.setattr("ouroboros.tools.services.executor_start_service", lambda *a, **k: launched.append(k) or {})
    args = {"cmd": ["python3", "-c", "print(1)"], "cwd": str(active / "sub"), "outputs": ["../../out.txt"]}
    if surface == "start_service":
        args["name"] = "preview"
    result = registry.execute_result(surface, args)
    assert result.code == "RESOURCE_POLICY_BLOCKED" and "no confirmed bind mount" in result.text, result.text
    assert launched == []
