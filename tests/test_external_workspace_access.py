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
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.presence_authority import build_presence_capability_ceiling, presence_ceiling_payload
from ouroboros.presence_capabilities import (
    PresenceProfileResolution,
    PresenceResourceTarget,
    PresenceSelection,
    PresenceToolTarget,
)
from ouroboros.presence_runtime import ResolvedPresenceRuntime
from ouroboros.tools.registry import ToolContext, ToolRegistry, _command_mentions_protected_root
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


def test_hidden_dotdir_default_deny_is_mutation_only(tmp_path):
    """capinv-447 / В23=A: the hidden/credential DEFAULT-DENY is now a MUTATION
    gate (and the fail-closed default for unknown-operation callers). Root
    READS of the owner's home are location-authorized only — the same paths
    are readable (secret bytes are masked at egress, not refused)."""
    home = tmp_path / "_home"
    ctx = _ctx(tmp_path, mode="workspace")  # non-external, the now-readable user_files profile
    # benign project dotdirs / dotfiles -> allowed even for mutation
    for rel in (".github/workflows/ci.yml", ".vscode/launch.json", ".gitignore", "proj/.github/x.yml"):
        assert user_files_path_block_reason(ctx, home / rel) == "", f"benign blocked: {rel}"
    # credential stores / unknown dotfiles: mutation (default op) stays BLOCKED,
    # root reads are allowed.
    for rel in (
        ".terraform.d/credentials.tfrc.json", ".cargo/credentials.toml", ".oci/config",
        ".pip/pip.conf", ".m2/settings.xml", ".bash_history", ".mysql_history", ".kaggle/kaggle.json",
        ".cache/huggingface/token.json", ".aws/credentials", ".ssh/id_rsa", ".gnupg/secring.gpg",
        ".git/config", ".gitconfig",
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
    # Credential-like names: mutation stays shape-denied; root reads are
    # location-only (capinv-447 / В23=A — bytes are masked at egress instead).
    assert user_files_path_block_reason(ext, tmp_path / "scratch" / "id_rsa.pem")
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


def test_external_shell_read_cannot_reach_runtime_or_secrets(tmp_path):
    """claudexor B1: even READ-only shell in external mode must not reach the
    Ouroboros runtime (system repo / data drive) or credential paths — raw shell
    must not bypass the user_files path guard."""
    system = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    for p in (system, workspace, data):
        p.mkdir()
    (data / "settings.json").write_text("{}", encoding="utf-8")
    reg = ToolRegistry(repo_dir=system, drive_root=data)
    reg.set_context(ToolContext(repo_dir=system, drive_root=data, workspace_root=workspace, workspace_mode="external"))

    # Runtime repo read -> blocked.
    assert "WORKSPACE_SHELL_BLOCKED" in (_shell_guard_text(reg, {"cmd": ["cat", str(system / "BIBLE.md")]}, "advanced") or "")
    # Data drive read -> blocked.
    assert "WORKSPACE_SHELL_BLOCKED" in (_shell_guard_text(reg, {"cmd": ["cat", str(data / "settings.json")]}, "advanced") or "")
    # Credential path read -> blocked (secret markers).
    assert "WORKSPACE_SHELL_BLOCKED" in (_shell_guard_text(reg, {"cmd": ["cat", str(pathlib.Path.home() / ".ssh" / "id_rsa")]}, "advanced") or "")
    # Embedded-string read of a secret -> blocked.
    assert "WORKSPACE_SHELL_BLOCKED" in (_shell_guard_text(reg, {"cmd": ["python", "-c", f"open({str(data / 'settings.json')!r})"]}, "advanced") or "")
    # A genuine host-scratch read -> allowed (None).
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "note.txt").write_text("hi", encoding="utf-8")
    assert _shell_guard_text(reg, {"cmd": ["cat", str(scratch / "note.txt")]}, "advanced") is None


def test_external_shell_write_protects_child_drive(tmp_path):
    """claudexor B2: the shell write guard's protected roots must include the
    task's CHILD data drive (not only system repo + parent/budget)."""
    system = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    child = tmp_path / "child-data"
    for p in (system, workspace, data, child):
        p.mkdir()
    reg = ToolRegistry(repo_dir=system, drive_root=data)
    reg.set_context(ToolContext(
        repo_dir=system, drive_root=data, workspace_root=workspace, workspace_mode="external",
        task_id="t", task_metadata={"drive_root": str(child)},
    ))
    # pro mode would otherwise pass an absolute outside-workspace write; the child
    # drive control path must still be blocked.
    out = _shell_guard_text(reg, {"cmd": ["touch", str(child / "memory" / "x")]}, "pro")
    assert "WORKSPACE_SHELL_BLOCKED" in (out or "")


def test_external_workspace_shell_can_write_configured_deliverable_only_at_top_level(
    tmp_path, monkeypatch,
):
    """A normal external workspace may copy a generated file to Deliverables.

    The registry guard, rather than only the ``cwd=user_files`` resolver, must
    admit the destination.  The same target remains unavailable to an acting
    child, and an arbitrary home target stays outside the carve-out.
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
    blocked = _shell_guard_text(reg,
        {"cmd": ["cp", str(source), str(arbitrary_home_target)], "cwd": str(workspace)},
        "advanced",
    )
    assert blocked and "WORKSPACE_SHELL_BLOCKED" in blocked

    child_ctx = ToolContext(
        repo_dir=system,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="deliverable-child-test",
        task_constraint=TaskConstraint(
            mode="acting_subagent",
            surface="external_workspace",
            write_root=str(workspace),
        ),
    )
    reg.set_context(child_ctx)
    child_blocked = _shell_guard_text(reg,
        {"cmd": command, "cwd": str(workspace)}, "advanced",
    )
    assert child_blocked and "WORKSPACE_SHELL_BLOCKED" in child_blocked

    outside = tmp_path / "outside"
    outside.mkdir()
    targets = [
        deliverables / ".env",
        deliverables / "token.pem",
        deliverables / ".ssh" / "key",
    ]
    try:
        (deliverables / "escape").symlink_to(outside, target_is_directory=True)
    except OSError:
        pass
    else:
        targets.append(deliverables / "escape" / "written.txt")
    for target in targets:
        blocked_target = _shell_guard_text(reg,
            {"cmd": ["touch", str(target)], "cwd": str(workspace)}, "advanced",
        )
        assert blocked_target and (
            "WORKSPACE_SHELL_BLOCKED" in blocked_target
            or "SUBAGENT_SECRET_READ_BLOCKED" in blocked_target
        ), target


def test_nested_deliverables_keeps_target_policy_before_workspace_root(
    tmp_path, monkeypatch,
):
    """A Deliverables root inside an allowed workspace does not inherit its bypasses."""
    system = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    home = tmp_path / "home"
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

    ctx = ToolContext(
        repo_dir=system,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="nested-deliverables-test",
    )
    reg = ToolRegistry(repo_dir=system, drive_root=data)
    reg.set_context(ctx)
    assert _shell_guard_text(reg,
        {"cmd": ["touch", str(deliverables / "ordinary.txt")], "cwd": str(workspace)},
        "advanced",
    ) is None
    for target in (
        deliverables / ".hidden" / "file",
        deliverables / ".ssh" / "key",
        deliverables / "token.pem",
    ):
        blocked = _shell_guard_text(reg,
            {"cmd": ["touch", str(target)], "cwd": str(workspace)}, "advanced",
        )
        assert blocked and "WORKSPACE_SHELL_BLOCKED" in blocked

    # cp/mv/ln directory syntax creates a child named after the source. The
    # directory operand alone must not inherit the broader workspace root.
    hidden_source = workspace / ".env"
    hidden_source.write_text("secret", encoding="utf-8")
    for command in (
        ["cp", str(hidden_source), str(deliverables)],
        ["cp", "-r", str(hidden_source), str(deliverables)],
        ["cp", "-t" + str(deliverables), str(hidden_source)],
        ["cp", "-Ssuffix", str(hidden_source), str(deliverables)],
        ["cp", "--parents", "supervisor/../README.md", str(deliverables)],
        ["cp", "--backup", str(hidden_source), str(deliverables)],
        ["cp", "--reflink", str(hidden_source), str(deliverables)],
        ["cp", "--sparse", str(hidden_source), str(deliverables)],
        ["cp", "--context", str(hidden_source), str(deliverables)],
        ["mv", str(hidden_source), str(deliverables)],
        # Regression pin: the destination is refused by the per-candidate
        # Deliverables decision either way, and the per-segment writer-target
        # view now also shows the cp segment itself to the direct-target check
        # (a leading `cd` used to hide it behind argv[0]).
        ["sh", "-c", f"cd . && cp {hidden_source} {deliverables / '.env'}"],
    ):
        blocked = _shell_guard_text(reg,
            {"cmd": command, "cwd": str(workspace)}, "advanced",
        )
        assert blocked and "WORKSPACE_SHELL_BLOCKED" in blocked

    # cp -s/--symbolic-link creates a symlink from an ordinary source. It is
    # distinct from cp's link-preservation flags, but its new payload must use
    # the same Deliverables boundary.
    for command in (
        ["cp", "-s", str(source), str(deliverables)],
        ["cp", "-as", str(source), str(deliverables)],
        ["cp", "--symbolic-link", str(source), str(deliverables)],
    ):
        blocked = _shell_guard_text(reg,
            {"cmd": command, "cwd": str(workspace)}, "advanced",
        )
        assert blocked and "WORKSPACE_SHELL_BLOCKED" in blocked

    link_source = workspace / "ordinary.txt"
    link_source.write_text("ordinary", encoding="utf-8")
    try:
        link_blocked = _shell_guard_text(reg,
            {"cmd": ["ln", "-s", str(link_source), str(deliverables)], "cwd": str(workspace)},
            "advanced",
        )
    except OSError:
        pytest.skip("symlinks unavailable on this platform")
    assert link_blocked and "WORKSPACE_SHELL_BLOCKED" in link_blocked

    ordinary_move = workspace / "move-me.txt"
    ordinary_move.write_text("move", encoding="utf-8")
    assert _shell_guard_text(reg,
        {"cmd": ["mv", str(ordinary_move), str(deliverables)], "cwd": str(workspace)},
        "advanced",
    ) is None
    moved_link = workspace / "moved-link"
    try:
        moved_link.symlink_to(link_source)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")
    moved_link_blocked = _shell_guard_text(reg,
        {"cmd": ["mv", str(moved_link), str(deliverables)], "cwd": str(workspace)},
        "advanced",
    )
    assert moved_link_blocked and "WORKSPACE_SHELL_BLOCKED" in moved_link_blocked

    # A regular source remains usable with link-preserving flags; only an
    # actual source symlink needs the payload-target check.
    assert _shell_guard_text(reg,
        {"cmd": ["cp", "-P", str(ordinary_move), str(deliverables)], "cwd": str(workspace)},
        "advanced",
    ) is None
    # The uppercase ``-S`` suffix option must not be mistaken for lowercase
    # ``cp -s`` symlink creation when its attached suffix contains an ``s``.
    assert _shell_guard_text(reg,
        {"cmd": ["cp", "-Ssuffix", str(ordinary_move), str(deliverables)], "cwd": str(workspace)},
        "advanced",
    ) is None
    # Attached suffix text that contains flag-like letters must not turn an
    # ordinary symlink-following copy into link-preservation mode.
    suffix_target = workspace / "suffix-target.txt"
    suffix_target.write_text("suffix", encoding="utf-8")
    suffix_link = workspace / "suffix-link"
    try:
        suffix_link.symlink_to(suffix_target)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")
    assert _shell_guard_text(reg,
        {"cmd": ["cp", "-Sbak", str(suffix_link), str(deliverables)], "cwd": str(workspace)},
        "advanced",
    ) is None
    relative_link = workspace / "relative-link"
    try:
        relative_link.symlink_to("../outside-target")
    except OSError:
        pytest.skip("symlinks unavailable on this platform")
    for command in (
        ["mv", str(relative_link), str(deliverables)],
        ["cp", "-P", str(relative_link), str(deliverables)],
        ["cp", "-d", str(relative_link), str(deliverables)],
        ["cp", "--preserve=links", str(relative_link), str(deliverables)],
    ):
        blocked = _shell_guard_text(reg,
            {"cmd": command, "cwd": str(workspace)}, "advanced",
        )
        assert blocked and "WORKSPACE_SHELL_BLOCKED" in blocked

    # The same payload check applies when the destination is a new explicit
    # pathname rather than an existing directory. Ordinary files keep this
    # path, while link-preserving cp/mv and ln -s cannot smuggle an escaping
    # symlink through the generic workspace root.
    for command in (
        ["mv", str(relative_link), str(deliverables / "move-explicit")],
        ["cp", "-P", str(relative_link), str(deliverables / "copy-p-explicit")],
        ["cp", "-d", str(relative_link), str(deliverables / "copy-d-explicit")],
        ["cp", "-a", str(relative_link), str(deliverables / "copy-a-explicit")],
        ["cp", "--preserve=links", str(relative_link), str(deliverables / "copy-links-explicit")],
        ["cp", "-s", str(source), str(deliverables / "copy-s-explicit")],
        ["cp", "--symbolic-link", str(source), str(deliverables / "copy-symbolic-explicit")],
        ["ln", "-s", "../outside-target", str(deliverables / "ln-explicit")],
        ["ln", "-s", "-r", "ordinary.txt", str(deliverables / "ln-relative-explicit")],
        ["ln", "-sr", "ordinary.txt", str(deliverables / "ln-relative-cluster")],
        ["ln", "-s", "--relative", "ordinary.txt", str(deliverables / "ln-relative-long")],
    ):
        blocked = _shell_guard_text(reg,
            {"cmd": command, "cwd": str(workspace)}, "advanced",
        )
        assert blocked and "WORKSPACE_SHELL_BLOCKED" in blocked
    assert _shell_guard_text(reg,
        {"cmd": ["cp", str(ordinary_move), str(deliverables / "copy-explicit.txt")], "cwd": str(workspace)},
        "advanced",
    ) is None
    assert _shell_guard_text(reg,
        {"cmd": ["mv", str(ordinary_move), str(deliverables / "move-explicit.txt")], "cwd": str(workspace)},
        "advanced",
    ) is None

    # Relative-link mode remains usable when the cwd-resolved source is inside
    # Deliverables; the fix must not become a blanket denial of ``ln -r``.
    inside = deliverables / "inside.txt"
    inside.write_text("inside", encoding="utf-8")
    assert _shell_guard_text(reg,
        {"cmd": ["ln", "-s", "-r", str(inside), str(deliverables / "inside-link")], "cwd": str(workspace)},
        "advanced",
    ) is None

    link = deliverables / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")
    blocked_link = _shell_guard_text(reg,
        {"cmd": ["touch", str(link / "escaped.txt")], "cwd": str(workspace)},
        "advanced",
    )
    assert blocked_link and "WORKSPACE_SHELL_BLOCKED" in blocked_link

    alias = workspace / "public"
    try:
        alias.symlink_to(deliverables, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")
    alias_into_deliverables = _shell_guard_text(reg,
        {"cmd": ["touch", str(alias / ".env")], "cwd": str(workspace)},
        "advanced",
    )
    assert alias_into_deliverables and "WORKSPACE_SHELL_BLOCKED" in alias_into_deliverables

    from ouroboros.tools.shell import _resolve_declared_output, _run_shell

    declared_hidden, hidden_reason = _resolve_declared_output(
        ctx,
        str(deliverables / ".hidden" / "file"),
        workspace,
        cwd_root="active_workspace",
    )
    assert declared_hidden is None and "hidden" in hidden_reason.lower()
    declared_link, link_reason = _resolve_declared_output(
        ctx,
        str(link / "declared.txt"),
        workspace,
        cwd_root="active_workspace",
    )
    assert declared_link is None and "escapes" in link_reason.lower()

    # A lexical hidden/credential name stays protected even when an existing
    # Deliverables entry is a symlink whose resolved target is ordinary.
    benign = deliverables / "benign.txt"
    benign.write_text("ordinary", encoding="utf-8")
    hidden_alias = deliverables / ".env"
    try:
        hidden_alias.symlink_to(benign)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")
    for command in (
        ["touch", str(hidden_alias)],
        ["cp", str(source), str(hidden_alias)],
    ):
        blocked = _shell_guard_text(reg,
            {"cmd": command, "cwd": str(workspace)}, "advanced",
        )
        assert blocked and "WORKSPACE_SHELL_BLOCKED" in blocked

    custody = _run_shell(
        ctx,
        ["cp", str(source), str(deliverables / "custody.txt")],
        cwd=str(workspace),
    )
    assert "ARTIFACT_OUTPUT_UNDECLARED" in custody

    child_ctx = ToolContext(
        repo_dir=system,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="nested-deliverables-child-test",
        task_constraint=TaskConstraint(
            mode="acting_subagent",
            surface="external_workspace",
            write_root=str(workspace),
        ),
    )
    child_registry = ToolRegistry(repo_dir=system, drive_root=data)
    child_registry.set_context(child_ctx)
    child_blocked = _shell_guard_text(child_registry,
        {"cmd": ["touch", str(deliverables / "child.txt")], "cwd": str(workspace)},
        "advanced",
    )
    assert child_blocked and "WORKSPACE_SHELL_BLOCKED" in child_blocked

    executor_ref = {
        "type": "docker_exec",
        "id": "nested-deliverables-executor",
        "container_name": "nested-deliverables-executor",
        "network": "none",
        "path_mappings": [
            {"host_path": str(workspace), "backend_path": "/workspace"},
            {"host_path": str(deliverables), "backend_path": "/deliverables"},
        ],
    }
    executor_ctx = ToolContext(
        repo_dir=system,
        drive_root=data,
        workspace_root=workspace,
        workspace_mode="external",
        task_id="nested-deliverables-executor-test",
        executor_ref=executor_ref,
    )
    executor_registry = ToolRegistry(repo_dir=system, drive_root=data)
    executor_registry.set_context(executor_ctx)
    assert _shell_guard_text(executor_registry,
        {"cmd": ["touch", "/deliverables/ordinary-backend.txt"], "cwd": str(workspace)},
        "advanced",
    ) is None
    backend_hidden = _shell_guard_text(executor_registry,
        {"cmd": ["touch", "/deliverables/.env"], "cwd": str(workspace)},
        "advanced",
    )
    assert backend_hidden and "WORKSPACE_SHELL_BLOCKED" in backend_hidden
    backend_link = _shell_guard_text(executor_registry,
        {"cmd": ["touch", "/deliverables/link/backend.txt"], "cwd": str(workspace)},
        "advanced",
    )
    assert backend_link and "WORKSPACE_SHELL_BLOCKED" in backend_link


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

    backend_hidden = _shell_guard_text(reg,
        {"cmd": ["touch", "/deliverables/.env"], "cwd": str(workspace)},
        "advanced",
    )
    assert backend_hidden and "WORKSPACE_SHELL_BLOCKED" in backend_hidden

    blocked = _shell_guard_text(reg,
        {"cmd": ["cp", "/workspace/dist/app.html", "/tmp/not-deliverables.html"], "cwd": str(workspace)},
        "advanced",
    )
    assert blocked and "WORKSPACE_SHELL_BLOCKED" in blocked


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
    reg = ToolRegistry(repo_dir=system, drive_root=data)
    reg.set_context(ctx)
    assert _shell_guard_text(reg,
        {"cmd": ["touch", "/deliverables/ordinary.txt"], "cwd": str(workspace)},
        "advanced",
    ) is None
    hidden = _shell_guard_text(reg,
        {"cmd": ["touch", "/deliverables/.env"], "cwd": str(workspace)},
        "advanced",
    )
    assert hidden and "WORKSPACE_SHELL_BLOCKED" in hidden
    escaped = _shell_guard_text(reg,
        {"cmd": ["touch", "/deliverables/escape/file"], "cwd": str(workspace)},
        "advanced",
    )
    assert escaped and "WORKSPACE_SHELL_BLOCKED" in escaped


def test_deliverables_carveout_rejects_a_root_containing_protected_drives(tmp_path, monkeypatch):
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
    reg = ToolRegistry(repo_dir=system, drive_root=data)
    reg.set_context(ctx)
    blocked = _shell_guard_text(reg,
        {"cmd": ["touch", str(runtime / "sibling.txt")], "cwd": str(workspace)},
        "advanced",
    )
    assert blocked and "WORKSPACE_SHELL_BLOCKED" in blocked


def test_malformed_deliverables_config_fails_closed_without_shell_crash(tmp_path, monkeypatch):
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
    blocked = _shell_guard_text(reg,
        {"cmd": ["touch", str(tmp_path / "home" / "out.html")], "cwd": str(workspace)},
        "advanced",
    )
    assert blocked and "WORKSPACE_SHELL_BLOCKED" in blocked

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
    blocked = _shell_guard_text(reg,
        {"cmd": ["cp", str(source), str(deliverables / "out.html")], "cwd": str(workspace)},
        "advanced",
    )
    assert blocked and "WORKSPACE_SHELL_BLOCKED" in blocked
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

    from ouroboros.tools.shell import _resolve_declared_output, _run_shell

    result = _run_shell(ctx, command, cwd=str(workspace))
    assert destination.exists()
    assert "ARTIFACT_OUTPUT_UNDECLARED" in result
    resolved, reason = _resolve_declared_output(
        ctx, str(destination), workspace, cwd_root="active_workspace",
    )
    assert reason == ""
    assert resolved == destination.resolve()


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
    blocked = _shell_guard_text(reg,
        {"cmd": ["touch", str(deliverables / "report.html")], "cwd": str(workspace)},
        "advanced",
    )
    assert blocked and "WORKSPACE_SHELL_BLOCKED" in blocked


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


def test_command_mentions_protected_root_is_boundary_aware():
    root = "/x/ouroboros/data"
    # Whole path or a child path → match (the real protected-path cases).
    assert _command_mentions_protected_root(f"touch {root}", root)
    assert _command_mentions_protected_root(f"touch {root}/state.json", root)
    assert _command_mentions_protected_root(f"cat '{root}/x' ", root)
    # A different sibling path that merely shares the string prefix → NOT a match.
    assert not _command_mentions_protected_root("touch /x/ouroboros/database/x", root)
    assert not _command_mentions_protected_root("touch /x/ouroboros/data-backup", root)
    assert not _command_mentions_protected_root("", root)
    assert not _command_mentions_protected_root("touch /other/path", root)


def test_external_shell_read_blocks_relative_and_symlink_traversal(tmp_path):
    """Round-2 review: the external read guard must resolve relative paths against
    the cwd and canonicalize symlinks — string matching alone is bypassable."""
    system = tmp_path / "system"
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    for p in (system, workspace, data):
        p.mkdir()
    (data / "settings.json").write_text("{}", encoding="utf-8")
    reg = ToolRegistry(repo_dir=system, drive_root=data)
    reg.set_context(ToolContext(repo_dir=system, drive_root=data, workspace_root=workspace, workspace_mode="external"))

    # Relative traversal from the workspace cwd into the sibling data drive.
    rel = _shell_guard_text(reg, {"cmd": ["cat", "../data/settings.json"], "cwd": str(workspace)}, "advanced")
    assert "WORKSPACE_SHELL_BLOCKED" in (rel or ""), rel

    # Intra-workspace symlink pointing at the data drive.
    try:
        (workspace / "evil").symlink_to(data, target_is_directory=True)
    except OSError:
        return  # platform without symlinks
    sym = _shell_guard_text(reg, {"cmd": ["cat", "evil/settings.json"], "cwd": str(workspace)}, "advanced")
    assert "WORKSPACE_SHELL_BLOCKED" in (sym or ""), sym
    # A legitimate relative read inside the workspace stays allowed.
    (workspace / "ok.txt").write_text("x", encoding="utf-8")
    assert _shell_guard_text(reg, {"cmd": ["cat", "ok.txt"], "cwd": str(workspace)}, "advanced") is None


def test_readonly_git_exemption_does_not_open_a_runtime_write_or_secret_read(tmp_path):
    """The runtime/secret READ guard exempts commands proven read-only git in every
    segment. Two git flag families are NOT read-only however read-only the
    subcommand looks, and both were measured against real git: `--output=<file>`
    (log/show/diff) TRUNCATES the file, and `--no-index` (diff/grep) prints ANY host
    file. Riding the exemption, they let an external-workspace task overwrite
    settings.json and read the credentials the guard exists to protect."""
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
    # READ of the credential file through `--no-index`.
    assert _check(["git", "diff", "--no-index", "/dev/null", str(data / "settings.json")])
    # The exemption itself must survive: read-only git AT a runtime target, and an
    # `--output` that lands in host scratch, both stay allowed.
    assert _check(["git", "-C", str(system), "status"]) == ""
    assert _check(["git", "--git-dir", str(system / ".git"), "log"]) == ""
    assert _check(["git", "log", "--output=/tmp/history.txt"]) == ""
    assert _check(["git", "diff", "--no-index", "/tmp/a", "/tmp/b"]) == ""
