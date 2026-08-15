# tests/test_dispatch_prepare.py — the PREPARE phase and its dispatch (RWS v2 §3.1 step 2).
#
# The three-phase protocol earns its keep only if (a) the placement is resolved
# at exactly ONE site, (b) the prepared token binds the arguments that were
# actually authorized, and (c) an ssh placement stops at the prepare boundary
# with a TYPED refusal instead of being authorized over facts nobody has.
import ast
import dataclasses
import importlib
import inspect
import pathlib
import types

import pytest

from ouroboros.execution_facts import LOCAL_FACTS, RemoteExecutionFacts
from ouroboros.python_interpreter import resolve_process_python
from ouroboros.tools import dispatch_prepare
from ouroboros.tools.dispatch_args import ExecutionArgs
from ouroboros.tools.dispatch_prepare import (
    PreparedCall,
    bind_execution_args,
    prepare_operation,
)
from ouroboros.tools.registry import ToolRegistry
from ouroboros.workspace_executor import (
    ExecutorRef,
    SshExecutorUnavailableError,
    execute_prepared,
    prepare_native_operation,
)
from ouroboros.workspace_ref import SEALED_WORKSPACE_REF_KEY
from tests.golden_traces import scenarios

_SSH_PAYLOAD = {
    "kind": "ssh",
    "connection_id": "conn-1",
    "remote_root": "/srv/work/app",
    "workspace_id": "ws-1",
}


@pytest.fixture
def local(tmp_path):
    registry, _roots = scenarios._normal(tmp_path)
    return registry


@pytest.fixture
def remote(tmp_path):
    registry, _roots = scenarios._workspace(tmp_path)
    registry._ctx.task_metadata[SEALED_WORKSPACE_REF_KEY] = dict(_SSH_PAYLOAD)
    return registry


# ── prepare ──────────────────────────────────────────────────────────────────
def test_local_prepare_uses_the_home_fact_door_and_is_available(local):
    prepared = prepare_operation(local._ctx, "run_command")
    assert prepared.available
    assert prepared.placement == "local"
    assert prepared.facts is LOCAL_FACTS
    assert prepared.operation_id and not prepared.bound


def test_ssh_prepare_refuses_typed_through_the_executor_facade(remote):
    prepared = prepare_operation(remote._ctx, "run_command")
    assert not prepared.available
    assert "SSH_EXECUTOR_UNAVAILABLE" in prepared.unavailable
    assert "conn-1" in prepared.unavailable
    assert isinstance(prepared.facts, RemoteExecutionFacts)
    assert prepared.placement == "ssh"
    assert prepared.executor is not None and prepared.executor.kind == "ssh"


def test_ssh_dispatch_stops_at_the_prepare_boundary(remote):
    """No guard runs over facts nobody has, and nothing falls back to Home."""
    out = remote.execute("run_command", {"cmd": ["echo", "remote"]})
    assert out.startswith("⚠️ REMOTE_EXECUTION_UNAVAILABLE:")
    assert "SSH_EXECUTOR_UNAVAILABLE" in out


@pytest.mark.parametrize("executor", [
    None,
    ExecutorRef(kind="local", executor_id="x", network="host", mappings=()),
    ExecutorRef(kind="docker_exec", executor_id="c", network="none", mappings=(), container_name="c"),
])
def test_the_prepared_facade_seams_reject_non_ssh_placements(executor):
    """Local and docker prepare on Home and run through `execute`; routing them
    into the native seams would be a second execution path, not a fallback."""
    with pytest.raises(ValueError):
        prepare_native_operation(executor, "run_command")
    with pytest.raises(ValueError):
        execute_prepared(executor, None)


def test_the_prepared_facade_seams_fail_typed_when_no_broker_is_registered():
    """The ONE condition `SSH_EXECUTOR_UNAVAILABLE` survives synthesis for.

    A registered broker's own failures keep their own typed `RemoteWorkspaceError`
    codes — "the remote refused" and "this process can reach no remote at all"
    have to stay distinguishable. This pins the second: a process with no broker
    (a CLI, a worker before lifespan, a build without the transport) refuses
    loudly rather than silently running the command on the Home host.
    """
    ssh_ref = ExecutorRef(
        kind="ssh", executor_id="ws-1", network="host", mappings=(),
        connection_id="conn-1", remote_root="/srv/app",
    )
    with pytest.raises(SshExecutorUnavailableError):
        prepare_native_operation(ssh_ref, "run_command")
    bound = types.SimpleNamespace(prepared_token="op-1.deadbeef", execution_args={"cmd": ["true"]})
    with pytest.raises(SshExecutorUnavailableError):
        execute_prepared(ssh_ref, bound)
    # An UNBOUND prepared call is a programming error, not an absent transport.
    with pytest.raises(ValueError):
        execute_prepared(ssh_ref, None)


# ── token binding ────────────────────────────────────────────────────────────
def test_binding_commits_to_the_final_execution_args(local):
    prepared = prepare_operation(local._ctx, "run_command")
    bound = bind_execution_args(prepared, local._ctx, {"cmd": ["echo", "one"], "cwd": ""})
    assert bound.bound and bound.available
    assert bound.prepared_hash and bound.prepared_token.startswith(bound.operation_id + ".")
    assert bound.binds(bound.projections.execution_args)


def test_binding_rejects_argv_or_cwd_drift_after_authorization(local):
    prepared = prepare_operation(local._ctx, "run_command")
    bound = bind_execution_args(prepared, local._ctx, {"cmd": ["echo", "one"], "cwd": ""})
    authorized = bound.projections.execution_args
    assert not bound.binds(dataclasses.replace(authorized, argv=("echo", "TWO")))
    assert not bound.binds(dataclasses.replace(authorized, cwd="/elsewhere"))
    assert not bound.binds(dataclasses.replace(authorized, cwd_root="task_drive"))


def test_an_unbound_prepared_call_binds_nothing(local):
    prepared = prepare_operation(local._ctx, "run_command")
    assert not prepared.binds(ExecutionArgs(tool="run_command"))


def test_two_prepares_of_identical_args_get_distinct_tokens_but_one_hash(local):
    first = bind_execution_args(prepare_operation(local._ctx, "run_command"), local._ctx, {"cmd": ["echo", "x"]})
    second = bind_execution_args(prepare_operation(local._ctx, "run_command"), local._ctx, {"cmd": ["echo", "x"]})
    assert first.prepared_hash == second.prepared_hash  # same authorized command
    assert first.prepared_token != second.prepared_token  # distinct operations


# ── exactly one placement-resolution site ────────────────────────────────────
#
# The primitives that answer "which machine is this". Every one is a real symbol, and
# the assertion below says so: an allowlist that can silently name nothing is an
# allowlist that stops constraining the moment a function is renamed.
_PLACEMENT_READERS = {
    "workspace_ref_for": "ouroboros.workspace_ref",
    "facts_for_ref": "ouroboros.execution_facts",
    "prepare_operation": "ouroboros.tools.dispatch_prepare",
    "executor_ref_from_ctx": "ouroboros.workspace_executor",
    "is_remote_workspace": "ouroboros.workspace_ref",
    "is_external_workspace": "ouroboros.tool_access",
}

# Names that were REMOVED because they were second doors into the placement. Kept as
# tripwires: the scan below would see them if they came back, and this test fails if
# they exist at all. `facts_for(ctx)` is recorded as removed in `execution_facts`;
# `local_workspace_path_for` never existed on this branch.
_REMOVED_PLACEMENT_READERS = {
    "facts_for": "ouroboros.execution_facts",
    "local_workspace_path_for": "ouroboros.workspace_ref",
}


def _registry_bodies() -> dict[str, list[ast.AST]]:
    """Every function and method DEFINED in `tools/registry.py`, by name."""
    source = pathlib.Path(inspect.getsourcefile(ToolRegistry)).read_text(encoding="utf-8")
    bodies: dict[str, list[ast.AST]] = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bodies.setdefault(node.name, []).append(node)
    return bodies


def _reachable_placement_reads() -> dict[str, set[str]]:
    """Placement reads reachable from `ToolRegistry.execute`, by the body doing them.

    TRANSITIVE, which is the whole correction. The previous scanner read exactly two
    function bodies with no reach past them — so when the gate chain was extracted into
    `_placement_blind_gates` for the size ratchet, the pre-prepare reads inside it left
    the scanner's view in the same commit that introduced them, and the test kept
    passing while its docstring stopped being true. Any placement-dependent branch added
    to a helper either function calls was invisible.

    `getattr(ctx, "is_workspace_mode", ...)` is followed too: a reader named by a STRING
    is still a reader, and that spelling is how the pipeline actually reaches this one.
    """

    bodies = _registry_bodies()
    found: dict[str, set[str]] = {}
    seen: set[str] = set()
    stack = ["execute"]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        for node in bodies.get(name, []):
            for child in ast.walk(node):
                if isinstance(child, ast.Constant) and child.value in bodies:
                    stack.append(str(child.value))
                    continue
                if not isinstance(child, ast.Call):
                    continue
                func = child.func
                callee = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                if callee in _PLACEMENT_READERS or callee in _REMOVED_PLACEMENT_READERS:
                    found.setdefault(name, set()).add(callee)
                elif callee in bodies:
                    stack.append(callee)
    return found


# Every placement read reachable from `registry.execute`, and what each one is FOR.
# One of them routes. The rest are disclosed here because "exactly one resolution site"
# is a claim about ROUTING, and a claim nobody can enumerate is a claim nobody checks.
_DECLARED_PLACEMENT_READS = {
    # THE routing site. Everything downstream — the fact door, the executor projection,
    # the guards, the binding gate — is derived from this one read.
    "_dispatch": {"prepare_operation"},
    # SPELLING SPACE, not routing: which vocabulary a path argument is normalized in.
    # Runs before prepare and cannot change which machine executes.
    "_normalize_dispatch_path_args": {"workspace_ref_for"},
    "_root_relative_normalizer": {"workspace_ref_for"},
    # REFUSAL, not routing: an ssh placement has no Home path for `active_workspace`,
    # so these raise the typed seam error rather than degrading to `ctx.repo_dir`.
    "active_repo_dir_for": {"workspace_ref_for"},
    "active_repo_dir": {"workspace_ref_for"},
    "is_workspace_mode": {"workspace_ref_for"},
    # PROFILE, not routing: whether the task is external at all, which is placement-
    # blind in intent and answered identically on every host.
    "_shell_git_and_runtime_block": {"is_external_workspace"},
}


def test_every_declared_placement_reader_is_a_real_symbol():
    """An allowlist of names that match nothing constrains nothing."""

    for name, module in _PLACEMENT_READERS.items():
        assert hasattr(importlib.import_module(module), name), f"{module}.{name}"
    for name, module in _REMOVED_PLACEMENT_READERS.items():
        assert not hasattr(importlib.import_module(module), name), (
            f"{module}.{name} was removed as a second door into the placement"
        )


def test_the_dispatch_pipeline_has_exactly_one_placement_resolution_site():
    """§3.3: exactly ONE placement read in `registry.execute` decides where this runs.

    Everything downstream is derived from that single read, so a second, DISAGREEING
    placement cannot exist inside a dispatch. The other reads are enumerated above with
    what each is for; a new one has to be added there, which is the point — the failure
    message names the body that acquired it.
    """

    reads = _reachable_placement_reads()
    assert reads == _DECLARED_PLACEMENT_READS, {
        "undeclared": {k: sorted(v) for k, v in reads.items()
                       if v != _DECLARED_PLACEMENT_READS.get(k)},
        "gone": sorted(set(_DECLARED_PLACEMENT_READS) - set(reads)),
    }
    routing = {body for body, names in reads.items() if "prepare_operation" in names}
    assert routing == {"_dispatch"}, routing


def test_the_dispatch_pipeline_prepares_exactly_once_per_call(local, monkeypatch):
    calls = []
    original = dispatch_prepare.prepare_operation

    def counted(ctx, tool, args=None, **kwargs):
        calls.append(tool)
        return original(ctx, tool, args, **kwargs)

    monkeypatch.setattr("ouroboros.tools.registry.prepare_operation", counted)
    local.execute("read_file", {"path": "hello.txt", "root": "active_workspace"})
    assert calls == ["read_file"]


# ── remote python resolution comes from prepare, never from Home ─────────────
class _FakeRemoteFacts:
    placement = "ssh"

    def __init__(self, resolved="", boom=False):
        self._resolved = resolved
        self._boom = boom

    def interpreter_fact(self, requested):
        if self._boom:
            raise RuntimeError("SSH_FACTS_UNAVAILABLE")
        return types.SimpleNamespace(
            requested=requested, resolved=self._resolved, usable=bool(self._resolved)
        )


def _remote_ctx(tmp_path):
    registry, _roots = scenarios._workspace(tmp_path)
    return registry._ctx


def test_remote_python_resolution_uses_the_prepared_interpreter_fact(tmp_path):
    ctx = _remote_ctx(tmp_path)
    args, trace = resolve_process_python(
        ctx, "run_command", {"cmd": ["python3", "-V"]},
        runtime_mode="advanced",
        facts=_FakeRemoteFacts(resolved="/usr/bin/python3.12"),
    )
    assert args["cmd"] == ["/usr/bin/python3.12", "-V"]
    assert trace.reason == "remote_prepare_interpreter"
    assert trace.environment == "target_native"
    assert trace.surface == "ssh"
    assert trace.verified is True
    assert not trace.error_reason


def test_remote_python_resolution_never_falls_back_to_a_home_interpreter(tmp_path):
    ctx = _remote_ctx(tmp_path)
    args, trace = resolve_process_python(
        ctx, "run_command", {"cmd": ["python3", "-V"]},
        runtime_mode="advanced",
        facts=_FakeRemoteFacts(boom=True),
    )
    assert args["cmd"] == ["python3", "-V"]  # untouched
    assert trace.error_reason == "remote_prepare_unavailable"
    assert trace.verified is False


def test_remote_python_resolution_reports_an_unprovable_interpreter(tmp_path):
    ctx = _remote_ctx(tmp_path)
    _args, trace = resolve_process_python(
        ctx, "run_command", {"cmd": ["python3", "-V"]},
        runtime_mode="advanced",
        facts=_FakeRemoteFacts(resolved=""),
    )
    assert trace.error_reason == "remote_interpreter_unavailable"


def test_local_python_resolution_is_unchanged_by_the_facts_parameter(tmp_path):
    ctx = _remote_ctx(tmp_path)
    plain = resolve_process_python(ctx, "run_command", {"cmd": ["python3", "-V"]}, runtime_mode="advanced")
    with_local = resolve_process_python(
        ctx, "run_command", {"cmd": ["python3", "-V"]}, runtime_mode="advanced", facts=LOCAL_FACTS
    )
    assert plain[0] == with_local[0]
    assert plain[1] == with_local[1]


def test_prepared_call_is_immutable(local):
    prepared = prepare_operation(local._ctx, "run_command")
    assert isinstance(prepared, PreparedCall)
    with pytest.raises(dataclasses.FrozenInstanceError):
        prepared.unavailable = "nope"


def test_a_malformed_executor_ref_reads_as_non_coverage_and_is_recorded(local):
    """Today a malformed operator executor_ref silently means "run on the host"
    (`covers()` treats a mapping failure as non-coverage). Prepare preserves that
    exactly, and makes the malformed ref visible instead of swallowing it."""
    local._ctx.task_metadata["executor_ref"] = {"type": "docker_exec"}  # no mappings
    prepared = prepare_operation(local._ctx, "run_command")
    assert prepared.available          # not a refusal — unchanged behavior
    assert prepared.executor is None   # non-coverage: host execution
    assert prepared.executor_error.startswith("ValueError")
    # The dispatch still completes on the host, exactly as before.
    assert "host" in local.execute("run_command", {"cmd": ["echo", "host"]})


@pytest.mark.serial
def test_argument_drift_after_authorization_is_refused(local, monkeypatch):
    """The token's production consumer: nothing between AUTHORIZE and EXECUTE may
    rewrite argv or cwd. `check_safety` receives the LIVE args dict, so it is the
    realistic drift vector — and a rewrite there now stops the dispatch instead of
    launching a command no guard ever saw."""
    import ouroboros.safety as safety

    def rewriting_check(tool_name, arguments, **kwargs):
        arguments["cmd"] = ["echo", "SMUGGLED"]
        return True, None

    monkeypatch.setattr(safety, "check_safety", rewriting_check)
    out = local.execute("run_command", {"cmd": ["echo", "authorized"]})
    assert out.startswith("⚠️ PREPARED_CALL_BINDING_MISMATCH")
    assert "SMUGGLED" not in out


@pytest.mark.serial
def test_an_unmodified_dispatch_passes_the_binding_check(local, monkeypatch):
    import ouroboros.safety as safety

    monkeypatch.setattr(safety, "check_safety", lambda *a, **k: (True, None))
    out = local.execute("run_command", {"cmd": ["echo", "authorized"]})
    assert "authorized" in out
    assert "BINDING_MISMATCH" not in out
