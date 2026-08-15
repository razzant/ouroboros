"""External-workspace focus and independent path guards.

Workspace mode chooses the default repo/cwd. It does not choose a weaker
top-level principal; host-scratch compatibility and credential/runtime path
guards remain independent of the shared operation matrix.
"""

from __future__ import annotations

import pathlib

import pytest

from ouroboros.tool_access import (
    active_tool_profile,
    decide_tool_access,
    is_external_workspace,
    resolve_shell_cwd,
    user_files_path_block_reason,
)
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.tools.shell_guards import _command_mentions_protected_root


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

    import ouroboros.tools.control as control

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


def test_hidden_dotdir_default_deny_with_benign_allowlist(tmp_path):
    """v6.52.0 (P1): hidden (dot) components are DEFAULT-DENY — only a small benign project
    allowlist (.github/.vscode/.cache/...) is readable; every other in-home dotfile/dotdir
    (including credential stores a blocklist enumeration would miss) stays blocked."""
    home = tmp_path / "_home"
    ctx = _ctx(tmp_path, mode="workspace")  # non-external, the now-readable user_files profile
    # benign project dotdirs / dotfiles -> allowed (empty block reason)
    for rel in (".github/workflows/ci.yml", ".vscode/launch.json", ".gitignore", "proj/.github/x.yml"):
        assert user_files_path_block_reason(ctx, home / rel) == "", f"benign blocked: {rel}"
    # credential stores / unknown dotfiles -> BLOCKED (the exact leaks an enumeration missed).
    # `.cache` is intentionally NOT allowlisted (security-reviewer call: highest exposure, e.g.
    # credential caches, for the least benefit vs project-config dotdirs).
    for rel in (
        ".terraform.d/credentials.tfrc.json", ".cargo/credentials.toml", ".oci/config",
        ".pip/pip.conf", ".m2/settings.xml", ".bash_history", ".mysql_history", ".kaggle/kaggle.json",
        ".cache/huggingface/token.json", ".aws/credentials", ".ssh/id_rsa", ".gnupg/secring.gpg",
        ".git/config", ".gitconfig",
    ):
        assert user_files_path_block_reason(ctx, home / rel) != "", f"secret LEAKED: {rel}"


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
    # The CHILD data drive control plane stays protected (enumerated explicitly).
    assert user_files_path_block_reason(ext, child / "memory" / "identity.md")
    # Credential-like names stay protected wherever they live.
    assert user_files_path_block_reason(ext, tmp_path / "scratch" / "id_rsa.pem")


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
    assert "WORKSPACE_SHELL_BLOCKED" in (reg._run_shell_safety_check({"cmd": ["cat", str(system / "BIBLE.md")]}, "advanced") or "")
    # Data drive read -> blocked.
    assert "WORKSPACE_SHELL_BLOCKED" in (reg._run_shell_safety_check({"cmd": ["cat", str(data / "settings.json")]}, "advanced") or "")
    # Credential path read -> blocked (secret markers).
    assert "WORKSPACE_SHELL_BLOCKED" in (reg._run_shell_safety_check({"cmd": ["cat", str(pathlib.Path.home() / ".ssh" / "id_rsa")]}, "advanced") or "")
    # Embedded-string read of a secret -> blocked.
    assert "WORKSPACE_SHELL_BLOCKED" in (reg._run_shell_safety_check({"cmd": ["python", "-c", f"open({str(data / 'settings.json')!r})"]}, "advanced") or "")
    # A genuine host-scratch read -> allowed (None).
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "note.txt").write_text("hi", encoding="utf-8")
    assert reg._run_shell_safety_check({"cmd": ["cat", str(scratch / "note.txt")]}, "advanced") is None


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
    out = reg._run_shell_safety_check({"cmd": ["touch", str(child / "memory" / "x")]}, "pro")
    assert "WORKSPACE_SHELL_BLOCKED" in (out or "")


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
    rel = reg._run_shell_safety_check({"cmd": ["cat", "../data/settings.json"], "cwd": str(workspace)}, "advanced")
    assert "WORKSPACE_SHELL_BLOCKED" in (rel or ""), rel

    # Intra-workspace symlink pointing at the data drive.
    try:
        (workspace / "evil").symlink_to(data, target_is_directory=True)
    except OSError:
        return  # platform without symlinks
    sym = reg._run_shell_safety_check({"cmd": ["cat", "evil/settings.json"], "cwd": str(workspace)}, "advanced")
    assert "WORKSPACE_SHELL_BLOCKED" in (sym or ""), sym
    # A legitimate relative read inside the workspace stays allowed.
    (workspace / "ok.txt").write_text("x", encoding="utf-8")
    assert reg._run_shell_safety_check({"cmd": ["cat", "ok.txt"], "cwd": str(workspace)}, "advanced") is None


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
        return reg._run_shell_safety_check({"cmd": cmd, "cwd": str(workspace)}, "advanced") or ""

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
