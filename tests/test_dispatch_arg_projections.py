# tests/test_dispatch_arg_projections.py — the three argument projections (RWS v2 §3.1 step 3).
#
# codex #2: process_shell_guard_args is casually treated as "the args" while it
# is really the GUARD projection — it adds __tool_name, drops verify_and_record's
# contract fields and rewrites run_script into an inline argv. These tests turn
# each of those differences into an ASSERTED fact and pin the invariants that
# must survive per pair: guard_args and execution_args agree on argv and cwd,
# handler_args round-trips the tool's public schema.
import dataclasses
import pathlib

import pytest

from ouroboros.tools.dispatch_args import (
    GUARD_PROJECTION_DIVERGENCES,
    ArgProjections,
    ExecutionArgs,
    project_dispatch_args,
)
from ouroboros.tools.registry import (
    ToolRegistry,
    _build_builtin_target_binding,
    _target_binding_operation,
)
from ouroboros.tools.tool_args import _entry_public_params
from tests.golden_traces import scenarios


@pytest.fixture
def ctx(tmp_path):
    registry, _roots = scenarios._normal(tmp_path)
    return registry._ctx


def _binding(ctx, tool, args):
    """The dispatch's ONE Home-side resolution, built the way `_dispatch` builds it.

    Two cases yield no binding, and both are the projection's UNBOUND lane rather
    than an error here: a remote placement has no Home path to bind, and a cwd the
    resolver refuses never produces one — `_dispatch` turns that refusal into a
    typed block message and never reaches the projection at all.
    """
    from ouroboros.workspace_ref import is_remote_workspace

    if is_remote_workspace(ctx) or _target_binding_operation(tool, args) is None:
        return None
    try:
        return _build_builtin_target_binding(ctx, tool, args)
    except Exception:
        return None


def _project(ctx, tool, args, runtime_mode="advanced"):
    return project_dispatch_args(
        ctx, tool, args, runtime_mode=runtime_mode, binding=_binding(ctx, tool, args),
    )


# ── guard_args vs execution_args ─────────────────────────────────────────────
def test_run_command_guard_and_execution_agree_on_argv_and_cwd(ctx):
    p = _project(ctx, "run_command", {"cmd": ["echo", "hi"], "cwd": ""})
    assert p.guard_argv == ("echo", "hi")
    assert p.execution_args.argv == ("echo", "hi")
    assert p.agrees_on_argv()
    assert p.agrees_on_cwd()
    assert p.execution_args.cwd_root == "active_workspace"


def test_verify_and_record_guard_and_execution_agree_on_the_normalized_check(ctx):
    """The check is normalized to argv ONCE, so the guard inspects exactly what
    runs — no `-lc`/`-c` drift between authorization and execution."""
    p = _project(ctx, "verify_and_record", {
        "contract_kind": "explicit_command",
        "check": '["cat", "hello.txt"]',
        "cwd": "",
    })
    assert p.execution_args.argv == ("cat", "hello.txt")
    assert p.agrees_on_argv()
    assert p.agrees_on_cwd()


def test_start_service_projects_the_service_cwd_root(ctx):
    p = _project(ctx, "start_service", {"name": "svc", "cmd": ["sleep", "1"], "cwd": "task_drive"})
    assert p.agrees_on_argv()
    assert p.execution_args.cwd_root == "task_drive"
    assert p.execution_args.cwd.endswith("task_drives/golden-task")


def test_run_script_argv_divergence_is_declared_not_accidental(ctx):
    """run_script executes the guarded BODY from a staged file, so its guard argv
    (inline `-c`) is not the launched argv. The surviving invariant is the pair
    (interpreter, body) — and the divergence is listed in the table."""
    p = _project(ctx, "run_script", {"interpreter": "python3", "script": "print(1)", "cwd": ""})
    assert p.guard_argv == ("python3", "-c", "print(1)")
    assert not p.agrees_on_argv()
    assert "interpreter_script_rewritten_to_inline_argv" in p.declared_divergences()
    assert p.agrees_on_inline_body()
    assert p.execution_args.argv == ("python3",)
    assert p.execution_args.inline_body == "print(1)"


def test_a_cwd_refusal_is_projected_not_raised(ctx, tmp_path):
    """The guard that owns a cwd refusal must still be the one to report it, so
    the projection captures it instead of raising at projection time."""
    p = _project(ctx, "run_command", {"cmd": ["echo", "x"], "cwd": str(tmp_path.parent / "outside")})
    assert not p.execution_args.resolved
    assert "cwd is outside allowed roots" in p.execution_args.cwd_error
    assert not p.agrees_on_cwd()


# ── the declared guard/handler divergences (codex #2) ────────────────────────
def test_guard_projection_adds_the_tool_name_marker(ctx):
    p = _project(ctx, "run_command", {"cmd": ["echo", "hi"]})
    assert p.guard_args["__tool_name"] == "run_command"
    assert "__tool_name" not in p.handler_args
    assert "adds___tool_name" in p.declared_divergences()


def test_guard_projection_drops_verify_contract_fields(ctx):
    handler = {"contract_kind": "explicit_command", "check": ["cat", "hello.txt"], "expected_match": "exit_zero"}
    p = _project(ctx, "verify_and_record", handler)
    assert set(p.handler_args) >= {"contract_kind", "expected_match"}
    assert "contract_kind" not in p.guard_args
    assert "expected_match" not in p.guard_args
    assert "drops_contract_fields" in p.declared_divergences()


def test_guard_projection_replaces_run_script_interpreter_and_script(ctx):
    p = _project(ctx, "run_script", {"interpreter": "python3", "script": "print(1)"})
    assert {"interpreter", "script"} <= set(p.handler_args)
    assert "interpreter" not in p.guard_args and "script" not in p.guard_args
    assert p.guard_args["cmd"] == ["python3", "-c", "print(1)"]


def test_every_shell_guarded_tool_has_a_declared_divergence_row():
    """A new process tool must declare how its guard projection differs — an
    undeclared difference is a desync, not a design choice."""
    from ouroboros.tools.shell_guards import SHELL_GUARDED_TOOLS

    assert SHELL_GUARDED_TOOLS <= set(GUARD_PROJECTION_DIVERGENCES)


# ── handler_args round-trips the public schema ───────────────────────────────
@pytest.mark.parametrize(
    "tool,args",
    [
        ("run_command", {"cmd": ["echo", "hi"], "cwd": ""}),
        ("run_script", {"interpreter": "python3", "script": "print(1)"}),
        ("start_service", {"name": "svc", "cmd": ["sleep", "1"]}),
        ("verify_and_record", {"contract_kind": "explicit_command", "check": ["cat", "hello.txt"]}),
    ],
)
def test_handler_args_round_trip_the_public_schema(ctx, tool, args):
    registry = ToolRegistry(repo_dir=pathlib.Path(ctx.repo_dir), drive_root=pathlib.Path(ctx.drive_root))
    registry.set_context(ctx)
    public = set(_entry_public_params(registry._entries[tool]))
    p = _project(ctx, tool, args)
    assert set(p.handler_args) <= public, sorted(set(p.handler_args) - public)
    assert dict(p.handler_args) == args


# ── immutability ─────────────────────────────────────────────────────────────
def test_all_three_projections_are_immutable(ctx):
    p = _project(ctx, "run_command", {"cmd": ["echo", "hi"]})
    with pytest.raises(TypeError):
        p.handler_args["cmd"] = ["rm", "-rf", "/"]
    with pytest.raises(TypeError):
        p.guard_args["cmd"] = ["rm", "-rf", "/"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.execution_args.argv = ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.tool = "other"


def test_projection_does_not_re_resolve_the_operation_cwd(ctx, monkeypatch):
    """Building the projection READS the dispatch's one resolution.

    The projection is the seam where this is easiest to get wrong: it needs a cwd,
    the resolver is one import away, and calling it looks harmless. It is not — a
    second resolution is a second authority over the same fact, and the two can
    disagree (a relative label re-joined against a different root is the v6.74.0 D1
    class). Handed the binding, the projection has nothing left to resolve; project
    twice and the count stays at the one call that BUILT the binding.
    """
    from ouroboros import tool_access

    args = {"cmd": ["echo", "hi"]}
    binding = _binding(ctx, "run_command", args)

    calls = []
    original = tool_access._select_process_target

    def counted(c, cwd, operation, **kwargs):
        calls.append((str(cwd or ""), str(operation)))
        return original(c, cwd, operation, **kwargs)

    monkeypatch.setattr(tool_access, "_select_process_target", counted)
    project_dispatch_args(ctx, "run_command", args, runtime_mode="advanced", binding=binding)
    project_dispatch_args(ctx, "run_command", args, runtime_mode="advanced", binding=binding)
    assert calls == []


def test_non_process_tool_projects_no_command_but_still_binds_its_path(ctx):
    """A root-labelled tool has no argv and no cwd — and its PATH is its command.

    While `paths`/`resource_root` were absent the execution hash of a `read_file` was
    a constant per tool, so the prepared token committed to nothing about which file
    the operation was for.
    """
    p = _project(ctx, "read_file", {"path": "hello.txt", "root": "active_workspace"})
    assert isinstance(p, ArgProjections)
    assert p.execution_args == ExecutionArgs(
        tool="read_file", resource_root="active_workspace", paths=("hello.txt",),
    )
    assert p.execution_args.argv == () and not p.execution_args.cwd
    assert dict(p.handler_args) == {"path": "hello.txt", "root": "active_workspace"}


def test_a_write_file_batch_binds_every_path_it_names(ctx):
    """`files=[…]` rows are paths too: a target rewriting one of them decides where
    the bytes land while Home's guards approved somewhere else."""
    p = _project(
        ctx,
        "write_file",
        {"root": "active_workspace", "files": [{"path": "a.txt", "content": "a"},
                                               {"path": "b.txt", "content": "b"}]},
    )
    assert p.execution_args.paths == ("a.txt", "b.txt")


def test_a_tool_with_no_root_label_projects_no_paths(ctx):
    """The projection is per SCHEMA, not per argument name: a cwd-bearing tool's
    `path`-shaped arguments are its own business."""
    p = _project(ctx, "run_command", {"cmd": ["true"], "cwd": ""}, runtime_mode="advanced")
    assert p.execution_args.paths == () and p.execution_args.resource_root == ""


def _ssh(ctx):
    from ouroboros.workspace_ref import SEALED_WORKSPACE_REF_KEY

    ctx.task_metadata[SEALED_WORKSPACE_REF_KEY] = {
        "kind": "ssh",
        "connection_id": "conn-1",
        "remote_root": "/srv/app",
        "workspace_id": "ws-1",
    }
    return ctx


def test_an_ssh_projection_carries_the_target_canonical_cwd(ctx):
    """The ssh execution args are REAL, not a placeholder.

    They used to be degenerate — `cwd=""` plus a `cwd_error`, which made
    `agrees_on_cwd()` False for every remote dispatch and the equality contract
    vacuous exactly where it had to hold. The cwd now comes from the operation's
    ONE prepare: the target's own canonical spelling, under the one SSH-native
    resource root, with no Home resolver consulted and no second placement read.
    """
    from ouroboros.execution_facts import facts_for_ref
    from ouroboros.workspace_ref import SshWorkspaceRef

    ref = SshWorkspaceRef(connection_id="conn-1", remote_root="/srv/app", workspace_id="ws-1")
    facts = facts_for_ref(ref, {"path_stats": {}, "git_toplevels": {}, "interpreters": {}})
    p = project_dispatch_args(
        _ssh(ctx), "run_command", {"cmd": ["echo", "hi"], "cwd": ""},
        runtime_mode="advanced", facts=facts, target_cwd="/srv/app/api",
    )
    assert p.execution_args.resolved and not p.execution_args.cwd_error
    assert p.execution_args.cwd == "/srv/app/api"
    assert p.execution_args.cwd_root == "active_workspace"
    assert p.execution_args.argv == ("echo", "hi")
    assert p.agrees_on_argv() and p.agrees_on_cwd()


def test_an_ssh_projection_without_a_prepared_cwd_refuses_typed(ctx):
    """No prepare bundle means no cwd — and the refusal is the resolver's typed
    one, projected rather than raised, never a Home spelling."""
    p = _project(_ssh(ctx), "run_command", {"cmd": ["echo", "hi"], "cwd": ""})
    assert not p.execution_args.resolved
    assert "RemoteWorkspacePathError" in p.execution_args.cwd_error
    assert "resolves on the target" in p.execution_args.cwd_error
    # The argv projection is placement-neutral and still built.
    assert p.execution_args.argv == ("echo", "hi")
