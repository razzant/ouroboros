# tests/test_placement_root_matrix.py — the root-placement matrix (RWS v2 §3.1, Q2а).
#
# Only `active_workspace` is SSH-native; every other resource root is Home-native.
# The matrix has teeth only if the accessor seams REFUSE typed for the ssh case
# instead of falling back to a Home path — a fallback would silently aim a remote
# task at the live Ouroboros repo, which is the invariant "an SSH binding never
# degrades to system-repo execution".
#
# Accessor seams were only HALF of it. The matrix also has to hold at the ROUTING
# point, and it did not: `prepare_operation` asked whether the TOOL has a native
# counterpart and never whether THIS CALL's root lives on the target, so a remote
# `read_file(root='artifact_store')` was prepared and executed on the target — which
# does not model `root` at all and answered about its own worktree. The matrix was
# enforced everywhere except where the operation was assigned a host. The routing
# half is pinned here; where the operation actually WENT (a transport counter, not an
# inference) is pinned in `tests/test_registry_remote_dispatch.py`.
import pathlib

import pytest

from ouroboros.tool_access import (
    resolve_resource_path,
    resolve_shell_cwd,
    resource_root_path,
)
from ouroboros.tool_capabilities import (
    IMPLICIT_RESOURCE_ROOT,
    ROOT_AFFINITY_TOOL_NAMES,
    ROOT_LABELLED_TOOL_ARG,
    dispatch_resource_root,
    remote_native_operation_for_tool,
)
from ouroboros.tools.dispatch_prepare import prepare_operation
from ouroboros.tools.registry import (
    _normalize_dispatch_path_args,
    _root_relative_normalizer,
    active_repo_dir_for,
    system_repo_dir_for,
)
from ouroboros.workspace_ref import (
    HOME_NATIVE_ROOTS,
    SSH_NATIVE_ROOTS,
    LocalWorkspaceRef,
    RemoteWorkspacePathError,
    SEALED_WORKSPACE_REF_KEY,
    SshWorkspaceRef,
    normalize_remote_root_relative,
    root_is_target_native,
)
from tests.golden_traces import scenarios

_SSH_PAYLOAD = {
    "kind": "ssh",
    "connection_id": "conn-1",
    "remote_root": "/srv/work/app",
    "workspace_id": "ws-1",
}


@pytest.fixture
def local_ctx(tmp_path):
    registry, _roots = scenarios._workspace(tmp_path)
    return registry._ctx


@pytest.fixture
def ssh_ctx(tmp_path):
    registry, _roots = scenarios._workspace(tmp_path)
    ctx = registry._ctx
    ctx.task_metadata[SEALED_WORKSPACE_REF_KEY] = dict(_SSH_PAYLOAD)
    return ctx


# ── the matrix itself ────────────────────────────────────────────────────────
def test_only_active_workspace_is_ssh_native():
    assert SSH_NATIVE_ROOTS == {"active_workspace"}
    assert "active_workspace" not in HOME_NATIVE_ROOTS
    assert {"system_repo", "task_drive", "artifact_store", "user_files", "skill_payload"} <= HOME_NATIVE_ROOTS


def test_root_is_target_native_only_for_ssh_and_only_for_the_native_root():
    ssh = SshWorkspaceRef(connection_id="c", remote_root="/srv/app", workspace_id="w")
    local = LocalWorkspaceRef(local_root="/home/u/app")
    assert root_is_target_native(ssh, "active_workspace") is True
    for root in sorted(HOME_NATIVE_ROOTS):
        assert root_is_target_native(ssh, root) is False
    # Every root of a local (or docker-projected) placement is a Home path.
    for root in sorted(HOME_NATIVE_ROOTS | SSH_NATIVE_ROOTS):
        assert root_is_target_native(local, root) is False
        assert root_is_target_native(None, root) is False


# ── the matrix at the ROUTING point (which host runs the operation) ──────────
def test_dispatch_resource_root_reads_the_call_not_the_tool():
    """Routing needs a root per CALL: the same tool serves both groups."""
    for tool in sorted(ROOT_AFFINITY_TOOL_NAMES):
        assert ROOT_LABELLED_TOOL_ARG[tool] == "root"
        assert dispatch_resource_root(tool, {"root": "artifact_store"}) == "artifact_store"
        # The schema default has to be spelled at the routing point too: routing reads
        # the caller's RAW arguments, before any handler applies its own default.
        assert dispatch_resource_root(tool, {}) == IMPLICIT_RESOURCE_ROOT
        assert dispatch_resource_root(tool, {"root": ""}) == IMPLICIT_RESOURCE_ROOT
        assert dispatch_resource_root(tool, None) == IMPLICIT_RESOURCE_ROOT
    # A tool with no root label is about the active workspace by CONSTRUCTION — a
    # process cwd, the workspace's own git, a workspace-relative media path.
    for tool in ("run_command", "run_script", "vcs_status", "vcs_diff", "start_service",
                 "extract_video_frames", "tree_read"):
        assert tool not in ROOT_LABELLED_TOOL_ARG
        assert dispatch_resource_root(tool, {"root": "artifact_store"}) == IMPLICIT_RESOURCE_ROOT


@pytest.mark.parametrize("tool", sorted(ROOT_AFFINITY_TOOL_NAMES))
def test_only_an_ssh_native_root_is_routed_to_the_target(tool, ssh_ctx):
    """The routing answer for a remote task, root by root, tool by tool.

    `prepare_operation` needs no broker for this: the Home-native answer returns
    before any RPC is attempted, and the ssh-native answer keeps its `operation`
    while reporting the missing broker as `unavailable` — which is itself the proof
    that a Home-native root asked the target for NOTHING.
    """
    native = prepare_operation(ssh_ctx, tool, {"root": "active_workspace", "path": "x"})
    assert native.operation == remote_native_operation_for_tool(tool)
    assert native.operation, f"{tool} must have a native counterpart to make this a real test"
    assert "SSH_EXECUTOR_UNAVAILABLE" in native.unavailable
    for root in sorted(HOME_NATIVE_ROOTS):
        home = prepare_operation(ssh_ctx, tool, {"root": root, "path": "x"})
        assert home.operation == "", f"{tool}(root={root}) was routed to the target"
        assert home.native is None
        assert home.unavailable == "", "a Home-native root must not even try the wire"
        # The placement is still the task's — a per-tool convenience placement is
        # exactly what the sealed ref exists to prevent.
        assert home.placement == "ssh"


def test_a_rootless_tool_on_a_remote_task_still_routes(ssh_ctx):
    """Fixing the root matrix must not un-route what has no root to consult."""
    for tool in ("run_command", "vcs_status", "vcs_diff", "run_script", "start_service"):
        prepared = prepare_operation(ssh_ctx, tool, {})
        assert prepared.operation == remote_native_operation_for_tool(tool)
        assert "SSH_EXECUTOR_UNAVAILABLE" in prepared.unavailable


def test_a_local_placement_routes_nothing_regardless_of_root(local_ctx):
    for root in sorted(HOME_NATIVE_ROOTS | SSH_NATIVE_ROOTS):
        prepared = prepare_operation(local_ctx, "read_file", {"root": root, "path": "x"})
        assert prepared.native is None and not prepared.native_routed
        assert prepared.placement == "local"


# ── accessor seams refuse typed, never fall back ─────────────────────────────
def test_active_repo_dir_for_refuses_typed_instead_of_the_repo_fallback(ssh_ctx):
    with pytest.raises(RemoteWorkspacePathError) as excinfo:
        active_repo_dir_for(ssh_ctx)
    assert "target-native" in str(excinfo.value)


def test_system_repo_stays_home_native_under_ssh(ssh_ctx, local_ctx):
    assert system_repo_dir_for(ssh_ctx) == system_repo_dir_for(local_ctx)


def test_resource_root_path_refuses_active_workspace_but_serves_home_roots(ssh_ctx):
    with pytest.raises(RemoteWorkspacePathError):
        resource_root_path(ssh_ctx, "active_workspace")
    for root in ("system_repo", "runtime_data", "task_drive", "artifact_store", "user_files"):
        assert isinstance(resource_root_path(ssh_ctx, root), pathlib.Path)


def test_resolve_resource_path_refuses_active_workspace_before_any_resolve(ssh_ctx):
    with pytest.raises(RemoteWorkspacePathError):
        resolve_resource_path(ssh_ctx, root="active_workspace", path="../../etc/passwd")


def test_process_cwd_refuses_for_ssh_placement(ssh_ctx):
    """Every candidate root of the resolver is a Home path, so a remote process
    cwd cannot come from here — including an explicitly Home root label."""
    for cwd in ("", ".", "task_drive", "/srv/work/app"):
        with pytest.raises(RemoteWorkspacePathError):
            resolve_shell_cwd(ssh_ctx, cwd)


# ── placement-aware path-arg normalization (RWS-05) ──────────────────────────
def test_normalizer_uses_target_spellings_for_an_ssh_active_workspace(ssh_ctx):
    norm = _root_relative_normalizer(ssh_ctx, "active_workspace")
    assert norm("/srv/work/app/src/mod.py") == "src/mod.py"
    assert norm("/srv/work/app") == "."
    assert norm("src/mod.py") == "src/mod.py"
    # A path OUTSIDE the target root is left for the confinement check.
    assert norm("/etc/passwd") == "/etc/passwd"
    # A Home absolute path is not under the target root and is never stripped.
    assert norm(str(pathlib.Path(ssh_ctx.repo_dir) / "x.py")).startswith("/")


def test_normalizer_keeps_the_home_resolver_for_system_repo_under_ssh(ssh_ctx):
    norm = _root_relative_normalizer(ssh_ctx, "system_repo")
    system = system_repo_dir_for(ssh_ctx).resolve(strict=False)
    assert norm(str(system / "README.md")) == "README.md"


def test_dispatch_normalizes_an_ssh_workspace_path_arg_against_the_target_root(ssh_ctx):
    args = {"root": "active_workspace", "path": "/srv/work/app/src/mod.py"}
    assert _normalize_dispatch_path_args(ssh_ctx, "read_file", args) == ""
    assert args["path"] == "src/mod.py"


def test_dispatch_never_auto_routes_user_files_under_ssh(ssh_ctx):
    """user_files is Home-native and the workspace is remote, so a Home absolute
    path can never be 'under the active workspace' — no note, no reroute, and no
    Home/target spelling comparison."""
    home_path = str(pathlib.Path(ssh_ctx.repo_dir) / "hello.txt")
    args = {"root": "user_files", "path": home_path}
    assert _normalize_dispatch_path_args(ssh_ctx, "read_file", args) == ""
    assert args == {"root": "user_files", "path": home_path}


# ── local placement is untouched ─────────────────────────────────────────────
def test_local_dispatch_normalization_is_unchanged(local_ctx):
    workspace = active_repo_dir_for(local_ctx).resolve(strict=False)
    args = {"root": "active_workspace", "path": str(workspace / "app.py")}
    assert _normalize_dispatch_path_args(local_ctx, "read_file", args) == ""
    assert args["path"] == "app.py"


def test_local_auto_route_note_is_unchanged(local_ctx):
    workspace = active_repo_dir_for(local_ctx).resolve(strict=False)
    args = {"root": "user_files", "path": str(workspace / "app.py")}
    note = _normalize_dispatch_path_args(local_ctx, "read_file", args)
    assert note.startswith("⚠️ AUTO_ROUTED_TO_ACTIVE_WORKSPACE")
    assert args["root"] == "active_workspace"


# ── the target normalizer itself ─────────────────────────────────────────────
def test_remote_root_relative_is_pure_posix_and_fail_safe():
    root = "/srv/work/app"
    assert normalize_remote_root_relative(root, "/srv/work/app/a/b.py") == "a/b.py"
    assert normalize_remote_root_relative(root, "/srv/work/app/") == "."
    assert normalize_remote_root_relative(root, "/srv/work/appendix/x") == "/srv/work/appendix/x"
    assert normalize_remote_root_relative(root, ".") == "."
    assert normalize_remote_root_relative(root, "") == ""
    # Windows separators in a target spelling normalize to posix, never to a
    # Home pathlib form.
    assert normalize_remote_root_relative(root, "/srv/work/app\\a\\b.py") == "a/b.py"
    # Traversal collapses BEFORE the containment decision.
    assert normalize_remote_root_relative(root, "/srv/work/app/../etc") == "/srv/work/app/../etc"


def test_a_relative_remote_path_is_left_to_the_target():
    """Home does not strip a redundant root basename under remote placement.

    Deciding that needs the target fact "does the root contain a same-named
    subdirectory", and this runs BEFORE prepare — asking would be a probe issued
    before the operation exists. So the path travels as written and the TARGET
    resolves it in its own spelling space; an unstripped `app/x` merely resolves
    under the root, and a real escape is still refused by confinement. (A
    `root_subdir_exists=` parameter used to exist for a caller that might hold the
    fact; nothing ever did, so it went.)
    """
    root = "/srv/work/app"
    assert normalize_remote_root_relative(root, "app/x.py") == "app/x.py"
    assert normalize_remote_root_relative(root, "app") == "app"
    assert normalize_remote_root_relative(root, "x.py") == "x.py"


def test_malformed_remote_root_leaves_the_path_untouched():
    assert normalize_remote_root_relative("", "a/b") == "a/b"
    assert normalize_remote_root_relative("relative/root", "/abs/x") == "/abs/x"
