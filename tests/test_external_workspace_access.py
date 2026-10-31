"""External-workspace focus and independent path guards.

Workspace mode chooses the default repo/cwd. It does not choose a weaker
top-level principal; host-scratch compatibility and credential/runtime path
guards remain independent of the shared operation matrix.
"""

from __future__ import annotations

import pathlib
import shlex

import pytest

# This suite executes real shell subprocesses through ``_run_shell``; keep it
# in the dedicated serial pytest lane so xdist workers cannot crash or race it.
pytestmark = pytest.mark.serial

from ouroboros.tool_access import (
    active_tool_profile,
    decide_tool_access,
    is_external_workspace,
    resolve_shell_cwd,
    user_files_path_block_reason,
)
from ouroboros.presence_authority import presence_ceiling_payload
from tests.frozen_presence_authority import build_presence_capability_ceiling
from tests.frozen_presence_capabilities import (
    PresenceProfileResolution,
    PresenceResourceTarget,
    PresenceSelection,
    PresenceToolTarget,
)
from tests.frozen_presence_runtime import ResolvedPresenceRuntime
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests._typed_guard_shared import _shell_guard_text




@pytest.fixture(autouse=True)
def _home_outside_tmp(tmp_path, monkeypatch):
    """These host-scratch tests assume the pytest tmp dir is OUTSIDE $HOME — true on
    Linux (/tmp) but FALSE on Windows CI (C:\\Users\\runneradmin\\AppData\\Local\\Temp),
    where tmp_path falls under home and the data-parent-under-home protection
    (tool_access.py) then blocks the sibling scratch. Pin $HOME to a controlled dir
    that never contains tmp_path so the "scratch outside home / non-runtime" premise
    holds on every platform (the guard reads pathlib.Path.home(), so test + code stay
    consistent)."""
    fake_home = tmp_path / "_home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setattr(pathlib.Path, "home", lambda: fake_home)


def _ctx(tmp_path: pathlib.Path, *, mode: str, child_drive: pathlib.Path | None = None) -> ToolContext:
    system = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    for p in (system, workspace, data):
        p.mkdir(exist_ok=True)
    meta: dict = {}
    if child_drive is not None:
        child_drive.mkdir(parents=True, exist_ok=True)
        meta["drive_root"] = str(child_drive)
    return ToolContext(
        repo_dir=system,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode=mode,
        task_id="task-ext",
        task_metadata=meta,
    )


def test_is_external_workspace_only_for_external_mode(tmp_path):
    assert is_external_workspace(_ctx(tmp_path, mode="external")) is True
    # A different (test-only) workspace value is workspace mode but NOT external.
    assert is_external_workspace(_ctx(tmp_path, mode="workspace")) is False
    assert is_external_workspace(_ctx(tmp_path, mode="")) is False


def test_external_profile_uses_shared_top_level_principal(tmp_path):
    ext = _ctx(tmp_path, mode="external")
    assert active_tool_profile(ext) == "external_workspace_task"
    for op in ("read", "list", "search", "write", "edit", "shell", "service"):
        assert decide_tool_access(profile="external_workspace_task", root="user_files", operation=op).allow
    assert not decide_tool_access(profile="external_workspace_task", root="user_files", operation="vcs").allow


def test_workspace_task_uses_same_top_level_principal(tmp_path):
    ws = _ctx(tmp_path, mode="workspace")
    assert active_tool_profile(ws) == "workspace_task"
    for op in ("read", "list", "search", "write", "edit", "shell", "service"):
        assert decide_tool_access(profile="workspace_task", root="user_files", operation=op).allow
    assert not decide_tool_access(profile="workspace_task", root="user_files", operation="vcs").allow


def test_subagent_inherits_active_external_workspace_when_metadata_missing(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import ouroboros.tools.control_scheduling as control

    system = tmp_path / "system"
    active = tmp_path / "app"
    system.mkdir()
    active.mkdir()
    ctx = SimpleNamespace()
    monkeypatch.setattr(control, "system_repo_dir_for", lambda _ctx: system)
    monkeypatch.setattr(control, "active_repo_dir_for", lambda _ctx: active)

    workspace_root, workspace_mode = control._inherited_workspace_from_active_repo(ctx, "", "")

    assert workspace_root == str(active)
    assert workspace_mode == "external"


def test_root_config_mutations_preserve_enumerated_credentials_without_dotdir_default_deny(tmp_path):
    """Root configuration access is broad; named stores/leaves keep their fences.

    Root reads remain location-authorized and preserve file content.
    Unlisted dotted directories are explicitly not a blanket credential fence.
    """
    home = tmp_path / "_home"
    ctx = _ctx(tmp_path, mode="workspace")  # non-external, the now-readable user_files profile
    # benign project dotdirs / dotfiles -> allowed even for mutation
    for rel in (".github/workflows/ci.yml", ".vscode/launch.json", ".gitignore", "proj/.github/x.yml"):
        assert user_files_path_block_reason(ctx, home / rel) == "", f"benign blocked: {rel}"
    # These unlisted stores retain the approved ordinary-config capability.
    for rel in (
        ".terraform.d/credentials.tfrc.json", ".cargo/credentials.toml", ".oci/config",
        ".pip/pip.conf", ".m2/settings.xml", ".mysql_history", ".kaggle/kaggle.json",
        ".bash_history", ".cache/huggingface/token.json", ".gitconfig",
    ):
        assert user_files_path_block_reason(ctx, home / rel) == "", rel
        assert user_files_path_block_reason(ctx, home / rel, operation="read") == "", rel
    for rel in (
        ".aws/credentials", ".ssh/id_rsa", ".gnupg/secring.gpg", ".git/config",
    ):
        assert user_files_path_block_reason(ctx, home / rel) != "", f"mutation gate lost: {rel}"
        for op in ("read", "list", "search"):
            assert user_files_path_block_reason(ctx, home / rel, operation=op) == "", (
                f"root read still denied: {rel}"
            )


def test_block_reason_allows_scratch_only_in_external_mode(tmp_path):
    scratch = tmp_path / "scratch" / "note.txt"  # outside $HOME, non-runtime
    assert user_files_path_block_reason(_ctx(tmp_path, mode="external"), scratch) == ""
    # Non-external: a path outside home is still rejected.
    assert "outside user home" in user_files_path_block_reason(_ctx(tmp_path, mode="workspace"), scratch)


def test_block_reason_protects_runtime_and_credentials_even_in_external(tmp_path):
    child = tmp_path / "child-data"
    ext = _ctx(tmp_path, mode="external", child_drive=child)
    # System repo and parent data drive stay protected.
    assert user_files_path_block_reason(ext, tmp_path / "system" / "BIBLE.md")
    assert user_files_path_block_reason(ext, tmp_path / "data" / "settings.json")
    # The CHILD data drive control plane stays protected (enumerated explicitly),
    # for READS too (location boundary, not a name shape).
    assert user_files_path_block_reason(ext, child / "memory" / "identity.md")
    assert user_files_path_block_reason(ext, child / "memory" / "identity.md", operation="read")
    # A credential-shaped NAME outside a credential location no longer refuses
    # mutation either: the fence is the location (~/.ssh, ~/.aws, ...) and the
    # exact credential leaves, never the suffix. Root reads stay location-only
    # and admitted content is delivered unchanged.
    assert user_files_path_block_reason(ext, tmp_path / "scratch" / "id_rsa.pem") == ""
    assert user_files_path_block_reason(ext, tmp_path / "scratch" / "id_rsa.pem", operation="read") == ""


def test_shell_cwd_scratch_scoped_not_filesystem_root(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    ext = _ctx(tmp_path, mode="external")
    work_dir, label, allowed = resolve_shell_cwd(ext, str(scratch))
    assert label == "user_files"
    assert work_dir.resolve() == scratch.resolve()
    # The returned allow-list (reused by the workspace write guard) must be scoped
    # to the chosen cwd, NEVER widened to the filesystem root.
    roots = {str(pathlib.Path(root).resolve()) for _lbl, root in allowed}
    assert str(pathlib.Path("/").resolve()) not in roots
    assert str(scratch.resolve()) in roots


def test_shell_cwd_data_is_rejected_but_system_is_explicit_in_external(tmp_path):
    ext = _ctx(tmp_path, mode="external")
    with pytest.raises(ValueError):
        resolve_shell_cwd(ext, str(tmp_path / "data"))  # parent data drive
    work_dir, label, _allowed = resolve_shell_cwd(ext, str(tmp_path / "system"))
    assert label == "system_repo"
    assert work_dir == (tmp_path / "system").resolve()


def test_external_workspace_shell_delivers_files_and_reports_undeclared_outputs(
    tmp_path, monkeypatch,
):
    """A normal external workspace may copy a generated file to Deliverables.

    The declared-output resolver and the post-execution audit preserve the
    real destination and disclose whether the generated file was registered.
    """
    system = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    deliverables = tmp_path / "Deliverables"
    home = tmp_path / "home"
    for path in (system, workspace, data, deliverables, home):
        path.mkdir()
    source = workspace / "dist" / "app.html"
    source.parent.mkdir()
    source.write_text("<html>ok</html>", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(deliverables))

    ctx = ToolContext(
        repo_dir=system,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="deliverable-test",
    )
    reg = ToolRegistry(repo_dir=system, drive_root=data)
    reg.set_context(ctx)
    destination = deliverables / "app.html"
    command = ["cp", str(source), str(destination)]
    assert _shell_guard_text(reg, {"cmd": command, "cwd": str(workspace)}, "advanced") is None

    from ouroboros.tools.shell import _resolve_declared_output
    from ouroboros.tools.shell import _run_shell

    resolved, reason = _resolve_declared_output(
        ctx, str(destination), workspace, cwd_root="active_workspace",
    )
    assert reason == ""
    assert resolved == destination.resolve()

    execution_result = _run_shell(
        ctx,
        command,
        cwd=str(workspace),
    )
    assert destination.exists()
    assert "ARTIFACT_OUTPUT_UNDECLARED" in execution_result

    directory_result = _run_shell(
        ctx,
        ["cp", str(source), str(deliverables)],
        cwd=str(workspace),
    )
    assert "ARTIFACT_OUTPUT_UNDECLARED" in directory_result

    env_result = _run_shell(
        ctx,
        ["env", "DELIVERABLE_TEST=1", "cp", str(source), str(deliverables / "env.html")],
        cwd=str(workspace),
    )
    assert "ARTIFACT_OUTPUT_UNDECLARED" in env_result

    wrapped_result = _run_shell(
        ctx,
        [
            "sh",
            "-c",
            "cp "
            f"{shlex.quote(str(source))} "
            f"{shlex.quote(str(deliverables / 'wrapped.html'))}",
        ],
        cwd=str(workspace),
    )
    assert "ARTIFACT_OUTPUT_UNDECLARED" in wrapped_result

    casefold_parent = pathlib.Path(str(deliverables).casefold())
    casefold_parent.mkdir(parents=True, exist_ok=True)
    casefold_destination = casefold_parent / "casefold.html"
    casefold_result = _run_shell(
        ctx,
        ["cp", str(source), str(casefold_destination)],
        cwd=str(workspace),
    )
    assert casefold_destination.exists()
    assert "ARTIFACT_OUTPUT_UNDECLARED" in casefold_result

    relative_result = _run_shell(
        ctx,
        ["sh", "-c", "echo relative > relative.html"],
        cwd=str(deliverables),
    )
    assert "ARTIFACT_OUTPUT_UNDECLARED" in relative_result

    arbitrary_home_target = home / "other.txt"
    allowed = _shell_guard_text(reg,
        {"cmd": ["cp", str(source), str(arbitrary_home_target)], "cwd": str(workspace)},
        "advanced",
    )
    assert allowed is None


def test_nested_deliverables_keep_declared_output_identity_and_custody(tmp_path, monkeypatch):
    from ouroboros.tools.shell import _resolve_declared_output, _run_shell

    system, workspace, data, home = (tmp_path / name for name in ("system", "workspace", "data", "home"))
    for path in (system, workspace, data, home):
        path.mkdir()
    deliverables = workspace / "Deliverables"
    deliverables.mkdir()
    outside = workspace / "other"
    outside.mkdir()
    source = workspace / "source.txt"
    source.write_text("ok", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(deliverables))
    ctx = ToolContext(repo_dir=system, drive_root=data, workspace_root=workspace,
                      workspace_mode="external", task_id="nested-deliverables-test")
    for relative in (".hidden/file", ".ssh/key"):
        resolved, reason = _resolve_declared_output(
            ctx, str(deliverables / relative), workspace, cwd_root="active_workspace",
        )
        assert resolved == (deliverables / relative).resolve() and reason == ""
    link = deliverables / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")
    resolved, reason = _resolve_declared_output(ctx, str(link / "declared.txt"), workspace, cwd_root="active_workspace")
    assert resolved is None and "escapes" in reason.lower()
    custody = _run_shell(ctx, ["cp", str(source), str(deliverables / "custody.txt")], cwd=str(workspace))
    assert "ARTIFACT_OUTPUT_UNDECLARED" in custody
    assert (deliverables / "custody.txt").read_text() == "ok"


def test_external_workspace_deliverables_guard_maps_executor_paths(tmp_path, monkeypatch):
    """Backend command paths must receive the same Deliverables admission and custody."""
    system = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    deliverables = tmp_path / "Deliverables"
    home = tmp_path / "home"
    for path in (system, workspace, data, deliverables, home):
        path.mkdir()
    source = workspace / "dist" / "app.html"
    source.parent.mkdir()
    source.write_text("ok", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(deliverables))
    executor_ref = {
        "type": "docker_exec",
        "id": "deliverables-guard",
        "container_name": "deliverables-guard",
        "network": "none",
        "path_mappings": [
            {"host_path": str(workspace), "backend_path": "/workspace"},
            {"host_path": str(deliverables), "backend_path": "/deliverables"},
        ],
    }
    ctx = ToolContext(
        repo_dir=system,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="deliverable-executor-test",
        executor_ref=executor_ref,
    )
    reg = ToolRegistry(repo_dir=system, drive_root=data)
    reg.set_context(ctx)

    for command in (
        ["cp", "/workspace/dist/app.html", "/deliverables/app.html"],
        ["sh", "-c", "cp /workspace/dist/app.html /deliverables/app2.html"],
    ):
        assert _shell_guard_text(reg,
            {"cmd": command, "cwd": str(workspace)}, "advanced",
        ) is None

    # Case-insensitive host semantics must agree with user_files_path_block_reason
    # for a new target whose spelling differs from the configured root.
    casefolded = pathlib.Path(str(deliverables).casefold()) / "casefold.html"
    assert _shell_guard_text(reg,
        {"cmd": ["cp", "/workspace/dist/app.html", str(casefolded)], "cwd": str(workspace)},
        "advanced",
    ) is None

    # Exercise the post-exec audit against a backend spelling as well. The
    # fake executor stands in for the host-owned docker backend and writes the
    # mapped host file, so this remains an end-to-end custody assertion without
    # requiring a live container in the unit lane.
    from shutil import copyfile
    from ouroboros.tools.shell import _run_shell
    from ouroboros.workspace_executor import ExecutorResult, executor_ref_from_ctx, map_backend_path

    def fake_execute(fake_ctx, fake_cmd, _cwd, _timeout_sec, env_overlay=None):
        destination = map_backend_path(
            executor_ref_from_ctx(fake_ctx),
            fake_cmd[-1],
        )
        copyfile(source, destination)
        return ExecutorResult(returncode=0, args=list(fake_cmd))

    monkeypatch.setattr("ouroboros.tools.shell.executor_execute", fake_execute)
    backend_result = _run_shell(
        ctx,
        ["cp", "/workspace/dist/app.html", "/deliverables/backend.html"],
        cwd=str(workspace),
    )
    assert "ARTIFACT_OUTPUT_UNDECLARED" in backend_result
    assert (deliverables / "backend.html").exists()


def test_executor_deliverables_root_symlink_keeps_target_policy(tmp_path, monkeypatch):
    system = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    home = tmp_path / "home"
    physical = tmp_path / "physical-deliverables"
    outside = tmp_path / "outside"
    for path in (system, workspace, data, home, physical, outside):
        path.mkdir()
    configured = tmp_path / "Deliverables"
    try:
        configured.symlink_to(physical, target_is_directory=True)
        (physical / "escape").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(configured))
    executor_ref = {
        "type": "docker_exec",
        "id": "deliverables-root-symlink",
        "container_name": "deliverables-root-symlink",
        "network": "none",
        "path_mappings": [
            {"host_path": str(workspace), "backend_path": "/workspace"},
            {"host_path": str(physical), "backend_path": "/deliverables"},
        ],
    }
    ctx = ToolContext(
        repo_dir=system,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="deliverables-root-symlink-test",
        executor_ref=executor_ref,
    )
    from ouroboros.tools.shell import _resolve_declared_output

    resolved, reason = _resolve_declared_output(ctx, "/deliverables/ordinary.txt", workspace, cwd_root="active_workspace")
    assert resolved == (physical / "ordinary.txt").resolve() and reason == ""
    resolved, reason = _resolve_declared_output(ctx, "/deliverables/.env", workspace, cwd_root="active_workspace")
    assert resolved is None and "credential" in reason.lower()
    resolved, reason = _resolve_declared_output(ctx, "/deliverables/escape/file", workspace, cwd_root="active_workspace")
    assert resolved is None and "escapes" in reason.lower()


def test_deliverables_keep_actual_runtime_sources_distinct_from_sibling_outputs(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    system = runtime / "repo"
    workspace = tmp_path / "workspace"
    data = runtime / "data"
    for path in (system, workspace, data):
        path.mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(tmp_path / "home"))
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(runtime))
    ctx = ToolContext(
        repo_dir=system,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="deliverable-broad-root-test",
    )
    from ouroboros.tools.shell import _resolve_declared_output

    resolved, reason = _resolve_declared_output(ctx, str(runtime / "sibling.txt"), workspace, cwd_root="active_workspace")
    assert resolved == (runtime / "sibling.txt").resolve() and reason == ""
    for source in (system / "BIBLE.md", data / "settings.json"):
        resolved, reason = _resolve_declared_output(ctx, str(source), workspace, cwd_root="active_workspace")
        assert resolved is None and reason, source


def test_malformed_deliverables_config_preserves_ordinary_home_authority_without_shell_crash(tmp_path, monkeypatch):
    system = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    for path in (system, workspace, data):
        path.mkdir()
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(tmp_path / "home"))
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", "~definitely_no_such_user_xyz/Deliverables")
    ctx = ToolContext(
        repo_dir=system,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="deliverable-malformed-config-test",
    )
    reg = ToolRegistry(repo_dir=system, drive_root=data)
    reg.set_context(ctx)
    allowed = _shell_guard_text(reg,
        {"cmd": ["touch", str(tmp_path / "home" / "out.html")], "cwd": str(workspace)},
        "advanced",
    )
    assert allowed is None

    # The optional Deliverables setting must not disable the ordinary
    # user_files-home custody nudge when it cannot be resolved.
    home = tmp_path / "home"
    home.mkdir()
    existing = home / "ordinary.txt"
    existing.write_text("old", encoding="utf-8")
    from ouroboros.tools.shell import _run_shell

    result = _run_shell(ctx, ["cp", str(existing), str(home / "new.txt")], cwd=str(workspace))
    assert "ARTIFACT_OUTPUT_UNDECLARED" in result


def test_deliverables_custody_audit_ignores_metadata_and_readonly_reads(tmp_path, monkeypatch):
    system = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    deliverables = tmp_path / "Deliverables"
    home = tmp_path / "home"
    for path in (system, workspace, data, deliverables, home):
        path.mkdir()
    existing = deliverables / "existing.txt"
    existing.write_text("old", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(deliverables))
    ctx = ToolContext(
        repo_dir=system,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="deliverable-read-audit-test",
    )
    from ouroboros.tools.shell import _run_shell

    for command in (
        ["chmod", "600", str(existing)],
        ["sed", "-n", "1p", str(existing)],
    ):
        result = _run_shell(ctx, command, cwd=str(workspace))
        assert "ARTIFACT_OUTPUT_UNDECLARED" not in result
    assert existing.read_text(encoding="utf-8") == "old"


def test_declared_deliverables_output_respects_presence_resource_ceiling(tmp_path, monkeypatch):
    system = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    deliverables = tmp_path / "Deliverables"
    home = tmp_path / "home"
    for path in (system, workspace, data, deliverables, home):
        path.mkdir()
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(deliverables))
    resolution = PresenceProfileResolution(
        active=(
            PresenceSelection("1" * 64, PresenceToolTarget("builtin", "run_command")),
            PresenceSelection("2" * 64, PresenceResourceTarget("active_workspace", ("shell",), ".")),
        ),
        missing_required=(),
        missing_optional=(),
        orphaned=(),
        runtime=ResolvedPresenceRuntime("main", 10, 10, False),
        profile_fingerprint="a" * 64,
        selection_fingerprint="b" * 64,
        required_selections_present=True,
    )
    ceiling = build_presence_capability_ceiling(
        skill_name="presence-declared-output-test",
        skill_content_hash="c" * 64,
        state_fingerprint="d" * 64,
        resolution=resolution,
    )
    ctx = ToolContext(
        repo_dir=system,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="deliverable-declared-presence-test",
        task_contract={"capability_ceiling": presence_ceiling_payload(ceiling)},
    )
    from ouroboros.tools.shell import _resolve_declared_output

    resolved, reason = _resolve_declared_output(
        ctx,
        str(deliverables / "declared.html"),
        workspace,
        cwd_root="active_workspace",
    )
    assert resolved is None
    assert "presence" in reason.lower()


def test_deliverables_carveout_respects_presence_resource_ceiling(tmp_path, monkeypatch):
    system = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    deliverables = tmp_path / "Deliverables"
    home = tmp_path / "home"
    for path in (system, workspace, data, deliverables, home):
        path.mkdir()
    source = workspace / "app.html"
    source.write_text("ok", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(deliverables))
    resolution = PresenceProfileResolution(
        active=(
            # The tool itself and its active-workspace shell are admitted, but
            # user_files:shell is intentionally absent from the ceiling.
            PresenceSelection(
                "1" * 64, PresenceToolTarget("builtin", "run_command"),
            ),
            PresenceSelection(
                "2" * 64, PresenceResourceTarget("active_workspace", ("shell",), "."),
            ),
        ),
        missing_required=(),
        missing_optional=(),
        orphaned=(),
        runtime=ResolvedPresenceRuntime("main", 10, 10, False),
        profile_fingerprint="a" * 64,
        selection_fingerprint="b" * 64,
        required_selections_present=True,
    )
    ceiling = build_presence_capability_ceiling(
        skill_name="presence-test",
        skill_content_hash="c" * 64,
        state_fingerprint="d" * 64,
        resolution=resolution,
    )
    ctx = ToolContext(
        repo_dir=system,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="deliverable-presence-test",
        task_contract={"capability_ceiling": presence_ceiling_payload(ceiling)},
    )
    reg = ToolRegistry(repo_dir=system, drive_root=data)
    reg.set_context(ctx)
    # The actual process cwd is the admitted workspace. A caller explicitly
    # selecting a user_files process root still needs that resource grant.
    blocked = reg.execute("run_command", {"cmd": ["true"], "cwd": str(deliverables)})
    assert "PRESENCE" in blocked and "BLOCKED" in blocked, blocked
    assert not (deliverables / "out.html").exists()


def test_deliverables_shell_presence_grant_preserves_declared_and_undeclared_custody(
    tmp_path, monkeypatch,
):
    system = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    deliverables = tmp_path / "Deliverables"
    home = tmp_path / "home"
    for path in (system, workspace, data, deliverables, home):
        path.mkdir()
    source = workspace / "app.html"
    source.write_text("ok", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(deliverables))
    resolution = PresenceProfileResolution(
        active=(
            PresenceSelection("1" * 64, PresenceToolTarget("builtin", "run_command")),
            PresenceSelection(
                "2" * 64,
                PresenceResourceTarget("active_workspace", ("shell",), "."),
            ),
            PresenceSelection(
                "3" * 64,
                PresenceResourceTarget("user_files", ("shell",), "."),
            ),
        ),
        missing_required=(),
        missing_optional=(),
        orphaned=(),
        runtime=ResolvedPresenceRuntime("main", 10, 10, False),
        profile_fingerprint="a" * 64,
        selection_fingerprint="b" * 64,
        required_selections_present=True,
    )
    ceiling = build_presence_capability_ceiling(
        skill_name="presence-deliverables-shell-test",
        skill_content_hash="c" * 64,
        state_fingerprint="d" * 64,
        resolution=resolution,
    )
    ctx = ToolContext(
        repo_dir=system,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="deliverable-presence-shell-test",
        task_contract={"capability_ceiling": presence_ceiling_payload(ceiling)},
    )
    reg = ToolRegistry(repo_dir=system, drive_root=data)
    reg.set_context(ctx)
    destination = deliverables / "out.html"
    command = ["cp", str(source), str(destination)]
    assert _shell_guard_text(reg,
        {"cmd": command, "cwd": str(workspace)}, "advanced",
    ) is None

    from ouroboros.tools.shell import _resolve_declared_output
    from ouroboros.artifacts import collect_task_artifact_records
    from hashlib import sha256

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_kw: (True, ""))
    result = reg.execute("run_command", {"cmd": command, "cwd": str(workspace)})
    assert destination.read_bytes() == source.read_bytes() == b"ok", result
    assert "ARTIFACT_OUTPUT_UNDECLARED" in result
    resolved, reason = _resolve_declared_output(
        ctx, str(destination), workspace, cwd_root="active_workspace",
    )
    assert reason == ""
    assert resolved == destination.resolve()
    declared = deliverables / "declared.html"
    result = reg.execute("run_command", {"cmd": ["cp", str(source), str(declared)],
        "cwd": str(workspace), "outputs": [str(declared)]})
    assert "exit_code=0" in result and "registered output" in result, result
    records = collect_task_artifact_records(data, ctx.task_id)
    record = next(row for row in records if row["name"] == declared.name)
    assert pathlib.Path(record["path"]).read_bytes() == declared.read_bytes() == source.read_bytes()
    assert record["sha256"] == sha256(source.read_bytes()).hexdigest()


def test_deliverables_presence_prefix_uses_logical_user_files_path(
    tmp_path, monkeypatch,
):
    """A physical Deliverables remap cannot turn ``report.html`` into ``.``."""
    system = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    deliverables = tmp_path / "Deliverables"
    home = tmp_path / "home"
    for path in (system, workspace, data, deliverables, home):
        path.mkdir()
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(deliverables))
    resolution = PresenceProfileResolution(
        active=(
            PresenceSelection("1" * 64, PresenceToolTarget("builtin", "run_command")),
            PresenceSelection(
                "2" * 64,
                PresenceResourceTarget("user_files", ("shell", "write"), "report.html"),
            ),
        ),
        missing_required=(),
        missing_optional=(),
        orphaned=(),
        runtime=ResolvedPresenceRuntime("main", 10, 10, False),
        profile_fingerprint="a" * 64,
        selection_fingerprint="b" * 64,
        required_selections_present=True,
    )
    ceiling = build_presence_capability_ceiling(
        skill_name="presence-deliverables-prefix-test",
        skill_content_hash="c" * 64,
        state_fingerprint="d" * 64,
        resolution=resolution,
    )
    ctx = ToolContext(
        repo_dir=system,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="deliverable-prefix-test",
        task_contract={"capability_ceiling": presence_ceiling_payload(ceiling)},
    )
    reg = ToolRegistry(repo_dir=system, drive_root=data)
    reg.set_context(ctx)
    from ouroboros.tools.shell import _resolve_declared_output

    resolved, reason = _resolve_declared_output(
        ctx,
        str(deliverables / "report.html"),
        workspace,
        cwd_root="active_workspace",
    )
    assert resolved is None and "presence" in reason.lower()
    blocked = reg.execute("run_command", {"cmd": ["true"], "cwd": str(deliverables)})
    assert "PRESENCE" in blocked and "BLOCKED" in blocked, blocked


def test_nested_default_deliverables_presence_prefix_stays_narrow(
    tmp_path, monkeypatch,
):
    """Default ~/Ouroboros/Deliverables keeps its full logical user_files prefix."""
    home = tmp_path / "home"
    system = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    deliverables = home / "Ouroboros" / "Deliverables"
    for path in (home, system, workspace, data, deliverables):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("OUROBOROS_USER_FILES_ROOT", raising=False)
    monkeypatch.delenv("OUROBOROS_DELIVERABLES_ROOT", raising=False)
    monkeypatch.setattr(pathlib.Path, "home", lambda: home)

    resolution = PresenceProfileResolution(
        active=(
            PresenceSelection("1" * 64, PresenceToolTarget("builtin", "run_command")),
            PresenceSelection(
                "2" * 64,
                PresenceResourceTarget(
                    "user_files", ("shell", "write"),
                    "Ouroboros/Deliverables/report.html",
                ),
            ),
        ),
        missing_required=(),
        missing_optional=(),
        orphaned=(),
        runtime=ResolvedPresenceRuntime("main", 10, 10, False),
        profile_fingerprint="a" * 64,
        selection_fingerprint="b" * 64,
        required_selections_present=True,
    )
    ceiling = build_presence_capability_ceiling(
        skill_name="nested-default-presence-test",
        skill_content_hash="c" * 64,
        state_fingerprint="d" * 64,
        resolution=resolution,
    )
    ctx = ToolContext(
        repo_dir=system,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="nested-default-presence-test",
        task_contract={"capability_ceiling": presence_ceiling_payload(ceiling)},
    )
    from ouroboros.tools.shell import _resolve_declared_output

    resolved, reason = _resolve_declared_output(
        ctx, str(deliverables / "report.html"), workspace,
        cwd_root="active_workspace",
    )
    assert reason == ""
    assert resolved == (deliverables / "report.html").resolve()


def test_external_deliverables_presence_uses_logical_name_not_physical_basename(
    tmp_path, monkeypatch,
):
    """An external configured container keeps the user_files Deliverables name."""
    home = tmp_path / "home"
    system = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    physical_output = tmp_path / "physical-output"
    for path in (home, system, workspace, data, physical_output):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    monkeypatch.setenv("OUROBOROS_DELIVERABLES_ROOT", str(physical_output))

    resolution = PresenceProfileResolution(
        active=(
            PresenceSelection("1" * 64, PresenceToolTarget("builtin", "run_command")),
            PresenceSelection(
                "2" * 64,
                PresenceResourceTarget(
                    "user_files", ("shell", "write"), "Deliverables/report.html",
                ),
            ),
        ),
        missing_required=(),
        missing_optional=(),
        orphaned=(),
        runtime=ResolvedPresenceRuntime("main", 10, 10, False),
        profile_fingerprint="a" * 64,
        selection_fingerprint="b" * 64,
        required_selections_present=True,
    )
    ceiling = build_presence_capability_ceiling(
        skill_name="external-logical-name-test",
        skill_content_hash="c" * 64,
        state_fingerprint="d" * 64,
        resolution=resolution,
    )
    ctx = ToolContext(
        repo_dir=system,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="external-logical-name-test",
        task_contract={"capability_ceiling": presence_ceiling_payload(ceiling)},
    )
    reg = ToolRegistry(repo_dir=system, drive_root=data)
    reg.set_context(ctx)
    from ouroboros.tools.shell import _resolve_declared_output

    resolved, reason = _resolve_declared_output(
        ctx, str(physical_output / "report.html"), workspace,
        cwd_root="active_workspace",
    )
    assert reason == ""
    assert resolved == (physical_output / "report.html").resolve()


def test_git_output_targets_remain_distinct_from_read_operands(tmp_path):
    """Git output flags identify an actual write; read operands remain reads."""
    system = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    for p in (system, workspace, data):
        p.mkdir()
    (data / "settings.json").write_text('{"OPENROUTER_API_KEY": "sk-secret"}', encoding="utf-8")
    reg = ToolRegistry(repo_dir=system, drive_root=data)
    reg.set_context(ToolContext(repo_dir=system, drive_root=data, workspace_root=workspace, workspace_mode="external"))

    def _check(cmd):
        return _shell_guard_text(reg, {"cmd": cmd, "cwd": str(workspace)}, "advanced") or ""

    # WRITE via the diff `--output` option — glued, split, and through `-C`.
    assert _check(["git", "log", f"--output={data / 'settings.json'}"])
    assert _check(["git", "diff", "--output", str(system / "BIBLE.md")])
    assert _check(["git", "-C", "/tmp", "show", f"--output={data / 'logs' / 'chat.jsonl'}"])
    # A read operand does not become a runtime mutation target.
    assert _check(["git", "diff", "--no-index", "/dev/null", str(data / "settings.json")]) == ""
    # The exemption itself must survive: read-only git AT a runtime target, and an
    # `--output` that lands in host scratch, both stay allowed.
    assert _check(["git", "-C", str(system), "status"]) == ""
    assert _check(["git", "--git-dir", str(system / ".git"), "log"]) == ""
    assert _check(["git", "log", "--output=/tmp/history.txt"]) == ""
    assert _check(["git", "diff", "--no-index", "/tmp/a", "/tmp/b"]) == ""
