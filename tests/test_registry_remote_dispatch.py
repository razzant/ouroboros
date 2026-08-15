# tests/test_registry_remote_dispatch.py — the SEAM: ToolRegistry.execute on an ssh placement.
#
# Every half of the remote path had tests. The JOIN did not, and that is where the
# feature was missing entirely: `workspace_executor.execute_prepared` had no
# production caller, so an ssh dispatch prepared on the target and then ran the
# LOCAL builtin handler — which either threw a typed placement refusal into the
# tool loop (`RemoteWorkspacePathError` out of the shell guard) or answered about
# the Ouroboros checkout instead of the project the task was pointed at. Unit tests
# on either side could not see it, because neither side was wrong.
#
# So this file drives the WHOLE dispatch, once per tool class, and asserts the four
# properties that together mean "the execute phase exists":
#
#   (a) the transport saw an `execute_prepared` — counted, not inferred;
#   (b) the call returned a normal tool result — not an exception, not a
#       Home-oriented error about `active_repo_dir`;
#   (c) exactly the authorized `execution_args` ran — the binding, checked on both
#       sides of the wire;
#   (d) ONE prepare per operation (§3.1 structural bound: no per-fact RPC).
#
# The transport is fake; the TARGET is not. `prepare`/`execute_prepared` delegate to
# the real `ouroboros.workspace_native` kernel against a real temp git worktree, so
# the results are the texts the real target produces and the assertions are about
# behavior rather than about a stub's return value. Only the wire is simulated.
#
# Serial: the native kernel spawns real processes (run_command/run_script/services)
# and the broker runs real I/O threads.
from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import time
import uuid
from typing import Any

import pytest

from ouroboros import workspace_native
from ouroboros.remote_workspace import RemoteSessionBroker, set_remote_workspace_service
from ouroboros.tool_capabilities import REMOTE_NATIVE_TOOL_OPERATION
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.workspace_native_contract import (
    MANDATORY_REMOTE_NATIVE_OPERATIONS,
    bundle_prepared_facts,
)
from ouroboros.workspace_ref import (
    HOME_NATIVE_ROOTS,
    SEALED_WORKSPACE_REF_KEY,
    SSH_NATIVE_ROOTS,
    SshWorkspaceRef,
)

pytestmark = pytest.mark.serial

_TASK_ID = "remote-dispatch-task"
_MANIFEST = {
    "schema_version": 1,
    "manifest_sha256": "a" * 64,
    "public_schema_sha256": "c" * 64,
    "native_operations": [{"name": name} for name in sorted(MANDATORY_REMOTE_NATIVE_OPERATIONS)],
    "native_kernel_modules": ["ouroboros.workspace_native"],
    "native_import_modules": ["ouroboros.workspace_native"],
    "native_import_edges": {},
}
_CONNECTION = {"id": "conn-1", "ssh_alias": "build"}


class _NativeTransport:
    """A wire, and nothing more: the real native kernel behind a fake session.

    Records every RPC by name so the structural bounds (§3.1: one prepare per
    operation, no per-fact chatter) are COUNTED, and keeps each prepared
    operation's `execution_args`/`native_facts` so a test can compare what was
    authorized against what actually ran.
    """

    def __init__(self, root, *, home_importer=None):
        self.root = root
        self.home_importer = home_importer
        self.calls: list[str] = []
        self.prepared: dict[tuple[str, str], dict[str, Any]] = {}
        self.executed: list[dict[str, Any]] = []

    # ── session plumbing ──
    def handshake(self):
        self.calls.append("handshake")
        return {
            "host_id": "host-1",
            "workspace_id": "ws-1",
            "canonical_root": self.root.as_posix(),
            "capability_hash": _MANIFEST["manifest_sha256"],
            "platform": {"system": "linux", "python": "3.11.5"},
        }

    def artifact_identity(self):
        return {}

    def reconcile(self):
        return []

    def renew_lease(self, _message):
        return None

    def cancel(self, _message):
        return True

    def task_lease(self, _task_id, forget=False):
        del forget
        return False

    def health(self):
        return {"status": "ready", "phase": "ready"}

    def panic(self):
        pass

    def close(self):
        self.calls.append("close")

    def fetch_blob(self, blob_id, max_bytes):
        del blob_id, max_bytes
        self.calls.append("fetch_blob")
        return b""

    # ── the two doors under test ──
    def prepare(self, message, blobs):
        tool = str(message["tool"])
        self.calls.append(f"prepare:{tool}")
        prepared = workspace_native.prepare_native_operation(
            self.root,
            tool,
            dict(message.get("args") or {}),
            task_id=str(message.get("task_id") or ""),
            **({"blobs": dict(blobs or {})} if tool == "execute_reviewed_payload" else {}),
        )
        bundle_prepared_facts(
            prepared.native_facts,
            root=self.root,
            run_git=lambda argv: subprocess.run(
                argv, capture_output=True, text=True, timeout=10, check=False,
            ),
        )
        key = (str(message["request_id"]), str(message["operation_id"]))
        row = {
            "request_id": key[0],
            "operation_id": key[1],
            "tool": tool,
            "prepared_token": f"token-{uuid.uuid4().hex[:12]}",
            "prepared_hash": hashlib.sha256(key[1].encode()).hexdigest(),
            "expires_at_ms": int(time.time() * 1000) + 60_000,
            "execution_args": dict(prepared.execution_args),
            "native_facts": dict(prepared.native_facts),
        }
        self.prepared[key] = row
        return dict(row)

    def execute_prepared(self, message):
        key = (str(message["request_id"]), str(message["operation_id"]))
        row = self.prepared[key]
        self.calls.append(f"execute_prepared:{row['tool']}")
        assert message["prepared_hash"] == row["prepared_hash"]
        assert message["prepared_token"] == row["prepared_token"]
        self.executed.append(dict(row))
        result = workspace_native.execute_native_operation(
            self.root,
            row["tool"],
            row["execution_args"],
            native_facts=row["native_facts"],
            task_id=str(message.get("task_id") or ""),
        )
        # The real wire is JSON, so the tuple-shaped envelope fields have to
        # become lists here exactly as execd's canonical encoding makes them.
        return json.loads(json.dumps(dataclasses.asdict(result.envelope)))

    def abort_prepared(self, message):
        self.calls.append("abort_prepared")
        self.prepared.pop((str(message["request_id"]), str(message["operation_id"])), None)
        return True


def _target_repo(base):
    """A real git worktree standing in for the remote project."""
    root = base / "target"
    root.mkdir(parents=True, exist_ok=True)
    (root / "hello.txt").write_text("hello target\n", encoding="utf-8")
    (root / "src").mkdir(exist_ok=True)
    (root / "src" / "mod.py").write_text("def target_fn():\n    return 7\n", encoding="utf-8")
    run = lambda *argv: subprocess.run(  # noqa: E731 — tiny local helper
        argv, cwd=str(root), capture_output=True, text=True, timeout=30, check=False,
    )
    run("git", "init", "-q", "-b", "main")
    run("git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    run("git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "base", "--no-gpg-sign")
    return root


@dataclasses.dataclass
class _Wired:
    registry: ToolRegistry
    transport: _NativeTransport
    root: Any

    def run(self, tool, args):
        before = len(self.transport.calls)
        result = self.registry.execute(tool, dict(args))
        return result, self.transport.calls[before:]


def wire_ssh_registry(tmp_path, monkeypatch):
    """The ssh-placement harness as a plain generator, so a SECOND test module can
    build the same wiring without importing (and shadowing) this file's fixture."""
    import ouroboros.safety as safety

    monkeypatch.setattr(safety, "check_safety", lambda *a, **k: (True, None))
    root = _target_repo(tmp_path)
    transports: list[_NativeTransport] = []

    def factory(_request, *, home_importer=None):
        transport = _NativeTransport(root, home_importer=home_importer)
        transports.append(transport)
        return transport

    broker = RemoteSessionBroker(
        tmp_path / "drive", "generation-1", _MANIFEST, transport_factory=factory
    )
    broker.start()
    set_remote_workspace_service(broker)
    try:
        broker.admit_workspace(
            _CONNECTION,
            remote_root=root.as_posix(),
            project_id="project-1",
            workspace_id="ws-1",
            task_id=_TASK_ID,
        )
        repo = tmp_path / "repo"
        repo.mkdir(parents=True, exist_ok=True)
        drive = tmp_path / "data"
        drive.mkdir(parents=True, exist_ok=True)
        ctx = ToolContext(
            repo_dir=repo,
            drive_root=drive,
            task_id=_TASK_ID,
            workspace_root=root.as_posix(),
            workspace_mode="external",
        )
        ctx.task_metadata[SEALED_WORKSPACE_REF_KEY] = SshWorkspaceRef(
            connection_id="conn-1", remote_root=root.as_posix(), workspace_id="ws-1",
        ).to_payload()
        registry = ToolRegistry(repo_dir=repo, drive_root=drive)
        registry.set_context(ctx)
        yield _Wired(registry, transports[0], root)
    finally:
        set_remote_workspace_service(None)
        broker.close(timeout_sec=5)


@pytest.fixture()
def wired(tmp_path, monkeypatch):
    """A registry on an ssh placement with a LIVE broker over the fake wire."""
    yield from wire_ssh_registry(tmp_path, monkeypatch)


def _assert_routed(result: str, calls: list[str], tool: str) -> None:
    """The four seam properties, minus the binding (asserted per case)."""
    operation = REMOTE_NATIVE_TOOL_OPERATION[tool]
    # (a) the execute phase actually ran, and (d) exactly one prepare preceded it.
    assert calls == [f"prepare:{operation}", f"execute_prepared:{operation}"], calls
    # (b) a normal tool result: not a Home-oriented refusal about a path the
    # placement does not have, and not a prepare-boundary stop.
    assert "active_repo_dir" not in result
    assert "Home-local" not in result
    assert "has no Home path" not in result
    for marker in (
        "REMOTE_EXECUTION_UNAVAILABLE",
        "REMOTE_EXECUTION_FAILED",
        "SSH_EXECUTOR_UNAVAILABLE",
        "SSH_FACTS_UNAVAILABLE",
        "PREPARED_CALL_BINDING_MISMATCH",
        "TOOL_ERROR",
        "TOOL_ARG_ERROR",
    ):
        assert marker not in result, result


# ── one case per tool class ──────────────────────────────────────────────────


def test_read_file_reads_the_target_not_home(wired):
    result, calls = wired.run("read_file", {"root": "active_workspace", "path": "hello.txt"})
    _assert_routed(result, calls, "read_file")
    assert "hello target" in result


def test_write_file_writes_on_the_target(wired):
    result, calls = wired.run(
        "write_file",
        {"root": "active_workspace", "path": "new.txt", "content": "written remotely\n"},
    )
    _assert_routed(result, calls, "write_file")
    assert (wired.root / "new.txt").read_text(encoding="utf-8") == "written remotely\n"


def test_edit_text_edits_on_the_target(wired):
    result, calls = wired.run(
        "edit_text",
        {
            "root": "active_workspace",
            "path": "hello.txt",
            "old_str": "hello target",
            "new_str": "edited target",
        },
    )
    _assert_routed(result, calls, "edit_text")
    assert "edited target" in (wired.root / "hello.txt").read_text(encoding="utf-8")


def test_list_files_lists_the_target(wired):
    result, calls = wired.run("list_files", {"root": "active_workspace", "path": "."})
    _assert_routed(result, calls, "list_files")
    assert "hello.txt" in result


def test_search_code_searches_the_target(wired):
    result, calls = wired.run(
        "search_code", {"root": "active_workspace", "query": "target_fn", "path": "."}
    )
    _assert_routed(result, calls, "search_code")
    assert "mod.py" in result


def test_query_code_queries_the_target(wired):
    result, calls = wired.run(
        "query_code", {"op": "symbols", "root": "active_workspace", "path": "src/mod.py"}
    )
    _assert_routed(result, calls, "query_code")
    assert "target_fn" in result


def test_vcs_status_and_diff_read_target_git(wired):
    (wired.root / "hello.txt").write_text("dirty\n", encoding="utf-8")
    status, calls = wired.run("vcs_status", {})
    _assert_routed(status, calls, "vcs_status")
    assert "hello.txt" in status
    diff, calls = wired.run("vcs_diff", {})
    _assert_routed(diff, calls, "vcs_diff")
    assert "hello.txt" in diff


def test_run_command_runs_on_the_target(wired):
    result, calls = wired.run("run_command", {"cmd": ["pwd"]})
    _assert_routed(result, calls, "run_command")
    assert wired.root.as_posix() in result
    assert "exit_code=0" in result


def test_run_script_runs_on_the_target(wired):
    result, calls = wired.run(
        "run_script",
        {"interpreter": "python3", "script": "import os; print('cwd=' + os.getcwd())"},
    )
    _assert_routed(result, calls, "run_script")
    assert f"cwd={wired.root.as_posix()}" in result or "cwd=" in result


def test_service_lifecycle_runs_on_the_target(wired):
    started, calls = wired.run(
        "start_service", {"name": "sleeper", "cmd": ["sh", "-c", "sleep 30"]}
    )
    _assert_routed(started, calls, "start_service")
    status, calls = wired.run("service_status", {"name": "sleeper"})
    _assert_routed(status, calls, "service_status")
    assert json.loads(status)["running"] is True
    stopped, calls = wired.run("stop_service", {"name": "sleeper"})
    _assert_routed(stopped, calls, "stop_service")


def test_verify_and_record_runs_on_the_target_and_records_on_home(wired):
    """The hybrid that is NOT routed WHOLE, and is not refused either.

    `verify_and_record` exists to write a durable Home receipt. Routing the whole tool
    would run the check on the target and record nothing, so the proof would vanish
    with the session; refusing it (which this build used to do) leaves a remote task
    with no way to verify anything. Both halves are wired instead: the check executes
    through the same prepared path the target takes for every other operation, and Home
    records the attested facts.

    What must NOT come back is the old `VERIFY_CWD_BLOCKED` — "the check cwd escapes
    allowed roots" — a lie that sent the model hunting a path problem it cannot have.
    The cwd was never out of bounds; it was on another host.
    """
    result, calls = wired.run(
        "verify_and_record",
        {
            "contract_kind": "explicit_command",
            "criterion_id": "c1",
            "criterion_source": "owner_request",
            "check": ["sh", "-c", "test -f hello.txt"],
            "expected": "exit_code=0",
        },
    )
    assert "VERIFY_CWD_BLOCKED" not in result
    assert "escapes allowed roots" not in result
    assert "REMOTE workspace" in result and "receipt was recorded on Home" in result
    # The check reached the target through the ordinary prepared path.
    assert calls == ["prepare:verify_remote_check", "execute_prepared:verify_remote_check"]


# ── the binding (c): exactly the authorized args run ─────────────────────────


def test_the_executed_args_are_exactly_the_prepared_ones(wired):
    """The cross-wire half of the binding.

    Home checks its own projection did not drift (`PreparedCall.binds`); this
    checks the other half — the argv/cwd that RAN are the ones the target
    canonicalized during prepare and Home authorized against. The cwd is the
    target's own absolute spelling, never a Home path, which is the whole reason
    the degenerate `cwd=''` projection had to go.
    """
    result, _calls = wired.run("run_command", {"cmd": ["pwd"], "cwd": "."})
    assert "exit_code=0" in result
    executed = wired.transport.executed[-1]
    assert executed["execution_args"]["cmd"] == ["pwd"]
    assert executed["execution_args"]["cwd"] == wired.root.as_posix()
    # …and the prepared row the target kept is the row it executed: one operation,
    # one set of args, no second authorization.
    assert wired.transport.prepared[
        (executed["request_id"], executed["operation_id"])
    ]["execution_args"] == executed["execution_args"]


def test_the_broker_refuses_args_that_are_not_the_authorized_ones(wired):
    """Replaying a prepared token with different args is refused, not executed."""
    from ouroboros.workspace_diagnostics import RemoteWorkspaceError
    from ouroboros.workspace_executor import execute_prepared, executor_ref_from_ctx, prepare_native_operation

    executor = executor_ref_from_ctx(wired.registry._ctx)
    prepared = prepare_native_operation(
        executor, "run_command", args={"cmd": ["pwd"]}, task_id=_TASK_ID
    )
    with pytest.raises(RemoteWorkspaceError) as excinfo:
        execute_prepared(
            executor, prepared, canonical_args={"cmd": ["whoami"], "cwd": wired.root.as_posix()},
            task_id=_TASK_ID,
        )
    assert excinfo.value.code == "prepared_arguments_mismatch"
    assert "execute_prepared:run_command" not in wired.transport.calls


def test_home_authorized_the_same_argv_and_cwd_the_target_prepared(wired):
    """The three projections agree, and they agree on TARGET facts.

    `agrees_on_cwd()` used to be False for every ssh dispatch because the
    projection could not resolve a cwd at all (`cwd=''` + `cwd_error`) — a
    degeneracy that made the equality contract vacuous exactly where it mattered.
    """
    from ouroboros.tools.dispatch_prepare import bind_execution_args, prepare_operation

    ctx = wired.registry._ctx
    prepared = prepare_operation(ctx, "run_command", {"cmd": ["pwd"]})
    assert prepared.available and prepared.native_routed
    bound = bind_execution_args(prepared, ctx, {"cmd": ["pwd"]}, runtime_mode="advanced")
    execution = bound.projections.execution_args
    assert execution.resolved and not execution.cwd_error
    assert execution.cwd == wired.root.as_posix()
    assert execution.cwd_root == "active_workspace"
    assert bound.projections.agrees_on_argv()
    assert bound.projections.agrees_on_cwd()
    assert bound.binds(execution)


# ── one prepare per operation (d), even when many facts are read ─────────────


def test_a_dispatch_makes_exactly_one_prepare_and_one_execute(wired):
    for tool, args in (
        ("read_file", {"root": "active_workspace", "path": "hello.txt"}),
        ("run_command", {"cmd": ["true"]}),
    ):
        _result, calls = wired.run(tool, args)
        operation = REMOTE_NATIVE_TOOL_OPERATION[tool]
        assert calls.count(f"prepare:{operation}") == 1
        assert calls.count(f"execute_prepared:{operation}") == 1
        assert "fetch_blob" not in calls


def test_a_home_local_tool_never_asks_the_target(wired):
    """A Home faculty on a remote task stays on Home.

    Routing is a declared table, not "everything on an ssh task goes remote": the
    target never declared `tree_read`, so sending it there would turn a working
    Home tool into a remote refusal.
    """
    result, calls = wired.run("tree_read", {})
    assert calls == []
    assert "REMOTE_EXECUTION_UNAVAILABLE" not in result


# ── negative: the two refusals that must stay distinguishable ────────────────


def test_without_a_broker_the_dispatch_refuses_with_the_typed_code(wired):
    """No broker in this process → `SSH_EXECUTOR_UNAVAILABLE`, and nothing runs."""
    set_remote_workspace_service(None)
    result = wired.registry.execute("read_file", {"root": "active_workspace", "path": "hello.txt"})
    assert result.startswith("⚠️ REMOTE_EXECUTION_UNAVAILABLE:")
    assert "SSH_EXECUTOR_UNAVAILABLE" in result
    assert "conn-1" in result
    assert not any(call.startswith("execute_prepared") for call in wired.transport.calls)


def test_a_target_refusal_reaches_the_model_with_its_own_code(wired, monkeypatch):
    """The remote's typed code is not flattened into a Home code.

    "The remote refused" and "this process can reach no remote" are different
    answers, and the owner has to be able to tell them apart — that distinction is
    the only thing `SSH_EXECUTOR_UNAVAILABLE` is allowed to mean.
    """
    from ouroboros.workspace_diagnostics import RemoteWorkspaceError

    def refusing_execute(_message):
        raise RemoteWorkspaceError(
            "remote_disk_full",
            "The target has no space left for this operation.",
            phase="execute",
            completion="not_started",
        )

    monkeypatch.setattr(wired.transport, "execute_prepared", refusing_execute)
    result = wired.registry.execute("write_file", {
        "root": "active_workspace", "path": "n.txt", "content": "x",
    })
    assert result.startswith("⚠️ REMOTE_EXECUTION_FAILED:")
    assert "remote_disk_full" in result
    assert "SSH_EXECUTOR_UNAVAILABLE" not in result
    assert "no space left" in result


def test_a_prepare_refusal_stops_before_any_execute(wired, monkeypatch):
    """A target that refuses PREPARE keeps its code and never reaches execute."""
    from ouroboros.workspace_diagnostics import RemoteWorkspaceError

    def refusing_prepare(_message, _blobs):
        raise RemoteWorkspaceError(
            "workspace_path_denied", "That path is outside the workspace.", phase="prepare",
        )

    monkeypatch.setattr(wired.transport, "prepare", refusing_prepare)
    result = wired.registry.execute("read_file", {"root": "active_workspace", "path": "hello.txt"})
    assert result.startswith("⚠️ REMOTE_EXECUTION_UNAVAILABLE:")
    assert "workspace_path_denied" in result
    assert not any(call.startswith("execute_prepared") for call in wired.transport.calls)


# ── the shell guard judges TARGET facts, and still judges ────────────────────


def test_a_write_outside_the_target_workspace_is_blocked_before_the_wire(wired):
    """Home policy still speaks first, in target spellings.

    The guard cannot resolve a Home `Path` here — there is none — so it reasons
    over the prepare facts. It must still REFUSE, or the containment rule would
    only exist on the target side.
    """
    result = wired.registry.execute("run_command", {"cmd": ["sh", "-c", "rm -rf /etc/passwd"]})
    assert result.startswith("⚠️ WORKSPACE_SHELL_BLOCKED")
    assert not any(call.startswith("execute_prepared") for call in wired.transport.calls)


def test_the_placement_blind_shell_policies_still_run_on_a_target(wired):
    result = wired.registry.execute("run_command", {"cmd": ["sudo", "apt-get", "install", "x"]})
    assert result.startswith("⚠️ SUDO_INTERACTIVE_BLOCKED")
    assert not any(call.startswith("execute_prepared") for call in wired.transport.calls)


def test_the_network_resource_contract_binds_on_the_target(wired):
    """`allowed_resources.network=false` must not become satisfiable by running
    the fetch on another host."""
    wired.registry._ctx.task_metadata["task_contract"] = {
        "allowed_resources": {"network": False}
    }
    result = wired.registry.execute("run_command", {"cmd": ["git", "fetch", "origin"]})
    assert result.startswith("⚠️ RESOURCE_CONSTRAINT_BLOCKED")
    assert "git fetch" in result
    assert not any(call.startswith("execute_prepared") for call in wired.transport.calls)


# ── exhaustiveness: no native operation is unreachable, none is invented ─────


def test_every_routed_tool_names_a_declared_native_operation():
    unknown = sorted(
        set(REMOTE_NATIVE_TOOL_OPERATION.values()) - MANDATORY_REMOTE_NATIVE_OPERATIONS
    )
    assert unknown == [], unknown


def test_the_unrouted_native_operations_are_named_not_forgotten():
    """Every native operation no tool reaches is listed WITH a reason.

    These are the target's internal halves of hybrid tools — a reviewed-payload
    execution, a snapshot export, a patch apply, a path classification — plus
    `verify_remote_check`, whose Home half owns the receipt and therefore cannot be
    replaced by the target (see `tool_capabilities`). Pinning the list means adding
    a native operation forces a decision about whether a tool now reaches it,
    instead of leaving a remote capability Home can never call.
    """
    unrouted = sorted(MANDATORY_REMOTE_NATIVE_OPERATIONS - set(REMOTE_NATIVE_TOOL_OPERATION.values()))
    assert unrouted == [
        "classify_ambiguous_workspace_path",
        "execute_reviewed_payload",
        "guarded_patch_apply",
        "snapshot_manifest_and_blob_export",
        "verify_remote_check",
    ], unrouted
    assert "verify_and_record" not in REMOTE_NATIVE_TOOL_OPERATION
    assert "extract_video_frames" in REMOTE_NATIVE_TOOL_OPERATION


# ── (h) the argv Home authorized IS the argv the target executes ──────────────


def test_a_disclosed_autocorrect_is_re_authorized_and_disclosed(wired):
    """The target rewrites `grep 'a\\|b'` into `grep -E 'a|b'` and says so.

    Before the reconciliation, Home authorized its own spelling and the target executed
    its own: the prepared token bound Home's hash and the target revalidated against ITS
    args, so both sides agreed with themselves and nothing compared the two. The rewrite
    is now adopted BEFORE the guard pipeline, so the guards authorize the command that
    actually runs, and the model is told it happened.
    """
    result, calls = wired.run("run_command", {"cmd": ["grep", "-c", "a\\|b", "hello.txt"]})

    assert "REMOTE_ARGV_AUTOCORRECTED" in result
    assert "execute_prepared:run_command" in calls
    executed = wired.transport.prepared[
        (wired.transport.executed[-1]["request_id"], wired.transport.executed[-1]["operation_id"])
    ]
    # What ran is the corrected form, and it is what the guards saw.
    assert executed["execution_args"]["cmd"] == ["grep", "-E", "-c", "a|b", "hello.txt"]


def test_an_undisclosed_substitution_is_refused_and_nothing_runs(wired, monkeypatch):
    """A target that rewrites a command without saying so gets no execution.

    "Authorized one thing, ran another" is the exact failure the three-phase protocol
    exists to make impossible, so an argv difference with no disclosure is a refusal
    rather than a quiet acceptance.
    """
    from ouroboros import workspace_native

    real = workspace_native.prepare_native_operation

    def substituting(root, tool, args, **kwargs):
        prepared = real(root, tool, args, **kwargs)
        if tool == "run_command":
            prepared.execution_args["cmd"] = ["rm", "-rf", "hello.txt"]
            prepared.native_facts.pop("autocorrect_note", None)
        return prepared

    monkeypatch.setattr(workspace_native, "prepare_native_operation", substituting)

    result, calls = wired.run("run_command", {"cmd": ["true"]})

    assert result.startswith("⚠️ REMOTE_ARGV_SUBSTITUTED")
    assert "rm" in result and "true" in result
    assert not any(call.startswith("execute_prepared") for call in calls)
    assert (wired.root / "hello.txt").exists()
    # …and the operation the target prepared under the substituted argv is WITHDRAWN,
    # not left holding a token that binds a command Home refused.
    assert calls.count("abort_prepared") == 1, calls


def test_an_agreeing_argv_adds_no_note(wired):
    result, calls = wired.run("run_command", {"cmd": ["true"]})
    assert "REMOTE_ARGV_AUTOCORRECTED" not in result
    assert "REMOTE_ARGV_SUBSTITUTED" not in result
    assert "execute_prepared:run_command" in calls


def test_reconciliation_is_inert_for_a_home_local_tool_and_a_local_placement():
    """Only a natively routed remote operation has a target argv to reconcile."""
    from ouroboros.tools.dispatch_prepare import reconcile_target_argv

    class _Prepared:
        tool = "run_command"
        native = None
        native_routed = False

    args = {"cmd": ["true"]}
    assert reconcile_target_argv(_Prepared(), args) == ("", "")
    assert args == {"cmd": ["true"]}


# ── the ROOT MATRIX at the routing point (ratified Q2а) ──────────────────────
#
# `_assert_routed` above answers "did the operation reach the target". These answer
# the question that was never asked: SHOULD it have. Only `active_workspace` lives on
# the target; every other resource root is Home's. The dispatch consulted the routing
# TABLE (per tool) and never the MATRIX (per call), so a remote
# `read_file(root='artifact_store')` went to the target — which does not model `root`
# at all, resolved the path in its own worktree, and even labelled the answer
# `active_workspace:`. The model asked for the task's artifact and was handed a
# same-named file from the remote project, and no test could see it because both
# halves were individually correct.
#
# The transport counter is the whole assertion: "where did it run" is COUNTED here,
# never inferred from the text.


def _home_root(ctx, root: str):
    from ouroboros.tool_access import resource_root_path

    path = resource_root_path(ctx, root)
    path.mkdir(parents=True, exist_ok=True)
    return path


_ROOT_TOOL_CALLS: dict[str, dict[str, Any]] = {
    "read_file": {"path": "probe.txt"},
    "list_files": {"path": "."},
    "write_file": {"path": "probe.txt", "content": "x\n"},
    "edit_text": {"path": "probe.txt", "old_str": "x", "new_str": "y"},
    "search_code": {"query": "probe", "path": "."},
    "query_code": {"op": "symbols", "path": "probe.py"},
}


@pytest.mark.parametrize("tool", sorted(_ROOT_TOOL_CALLS))
@pytest.mark.parametrize("root", sorted(HOME_NATIVE_ROOTS))
def test_a_home_native_root_never_leaves_home_on_a_remote_task(wired, root, tool):
    """The whole table: every root-labelled tool × every Home-native root.

    Whether the Home handler then finds the file, refuses it by profile access, or
    reports it missing is the ORDINARY Home answer and not this test's business — what
    is pinned is that the operation was HOME's to answer, and that no target-oriented
    refusal (`REMOTE_*`, `PLACEMENT_UNSUPPORTED_TOOL`) took its place.
    """
    result, calls = wired.run(tool, {"root": root, **_ROOT_TOOL_CALLS[tool]})
    assert calls == [], f"{tool}(root={root}) was routed to the target: {calls}"
    for marker in ("REMOTE_EXECUTION_UNAVAILABLE", "REMOTE_EXECUTION_FAILED",
                   "SSH_EXECUTOR_UNAVAILABLE", "PLACEMENT_UNSUPPORTED_TOOL"):
        assert marker not in result, result


@pytest.mark.parametrize("root", sorted(SSH_NATIVE_ROOTS))
def test_the_ssh_native_root_still_routes_on_a_remote_task(wired, root):
    result, calls = wired.run("read_file", {"root": root, "path": "hello.txt"})
    _assert_routed(result, calls, "read_file")
    assert "hello target" in result


def test_an_artifact_store_read_returns_the_HOME_artifact_not_the_targets_file(wired):
    """The defect, reduced to one call: same name, two hosts, different bytes.

    `artifact_store` is where a remote task's own artifact lives (Home publishes it),
    so this is the ordinary case rather than an exotic one. Before the matrix reached
    routing, the target answered with ITS `hello.txt` — a silent wrong-file read, which
    is worse than a refusal because nothing in the result says so.
    """
    ctx = wired.registry._ctx
    (_home_root(ctx, "artifact_store") / "hello.txt").write_text("HOME ARTIFACT\n", encoding="utf-8")
    (wired.root / "hello.txt").write_text("TARGET FILE\n", encoding="utf-8")

    result, calls = wired.run("read_file", {"root": "artifact_store", "path": "hello.txt"})

    assert calls == [], calls
    assert "HOME ARTIFACT" in result
    assert "TARGET FILE" not in result
    # …and the target's own root still answers with the target's bytes in the same task.
    workspace, calls = wired.run("read_file", {"root": "active_workspace", "path": "hello.txt"})
    _assert_routed(workspace, calls, "read_file")
    assert "TARGET FILE" in workspace


def test_a_home_native_write_lands_on_home_and_not_on_the_target(wired):
    """Task scratch is Home's. A write that reached the target would put the task's
    scratch on a host that is discarded with the session."""
    ctx = wired.registry._ctx
    home = _home_root(ctx, "task_drive")
    result, calls = wired.run(
        "write_file", {"root": "task_drive", "path": "scratch.txt", "content": "home scratch\n"},
    )
    assert calls == [], calls
    assert "⚠️" not in result, result
    assert (home / "scratch.txt").read_text(encoding="utf-8") == "home scratch\n"
    assert not (wired.root / "scratch.txt").exists()


def test_a_home_native_listing_shows_home_bytes_and_the_target_still_answers_its_own(wired):
    """`list_files`/`search_code` take the same road as `read_file`, both ways."""
    ctx = wired.registry._ctx
    home = _home_root(ctx, "task_drive")
    (home / "only_on_home.py").write_text("def home_only_fn():\n    return 1\n", encoding="utf-8")
    listed, calls = wired.run("list_files", {"root": "task_drive", "path": "."})
    assert calls == [], calls
    assert "only_on_home.py" in listed and "hello.txt" not in listed
    # The same tool against the target's root reaches the target and finds ITS symbol.
    remote, calls = wired.run("search_code", {"root": "active_workspace", "query": "target_fn", "path": "."})
    _assert_routed(remote, calls, "search_code")
    assert "mod.py" in remote


# ── a CLEAN worktree is an empty RESULT, not a remote failure ────────────────


def test_vcs_status_and_diff_on_a_clean_target_answer_exactly_as_home_does(wired):
    """The live-server defect: `vcs_status` returned REMOTE_EXECUTION_FAILED.

    A clean `git status --porcelain` emits no bytes, so the target's envelope carried
    an empty `text` — and Home read "no text" as "no result" and rendered
    `remote_result_empty`, i.e. a transport failure for an operation that had run
    perfectly. `run_command(["git","status","--porcelain"])` on the SAME target
    answered fine, which is how the two placements were caught disagreeing about one
    clean worktree. The Home handler answers `''` for a clean tree; one voice means
    the target's clean tree answers `''` too.
    """
    for tool in ("vcs_status", "vcs_diff"):
        result, calls = wired.run(tool, {})
        _assert_routed(result, calls, tool)
        assert result == "", repr(result)
    # …and the dirty tree still reports, so "empty" is not a swallowed answer.
    (wired.root / "hello.txt").write_text("dirty\n", encoding="utf-8")
    status, calls = wired.run("vcs_status", {})
    _assert_routed(status, calls, "vcs_status")
    assert "hello.txt" in status


def test_an_empty_envelope_is_an_empty_result_and_a_diagnostic_still_renders():
    """The projection itself: empty text is a result; a diagnostic still wins."""
    from ouroboros.tools.dispatch_execute import native_result_text
    from ouroboros.workspace_diagnostics import ExecutionDiagnostic, ToolExecutionEnvelope

    assert native_result_text(ToolExecutionEnvelope(text="")) == ""
    assert native_result_text(ToolExecutionEnvelope(text="answer")) == "answer"
    diagnostic = ExecutionDiagnostic(
        domain="protocol", code="remote_disk_full", message="no space left",
        phase="execute", completion="not_started",
    )
    rendered = native_result_text(ToolExecutionEnvelope(text="", diagnostic=diagnostic))
    assert "remote_disk_full" in rendered


# ── the third member of the pair: a refused prepare is WITHDRAWN ─────────────
#
# `abort_prepared` had transport and execd sides and no production caller: a target
# that had prepared an operation Home then refused held its reserved token and staged
# blobs until the TTL expired. The refusals below all land AFTER a successful prepare,
# and each is a different authority saying no — which is why the withdrawal is
# registered at the dispatch boundary instead of being spelled at each site.


def _refusal_cases():
    return [
        pytest.param(
            "llm_safety",
            "read_file", {"root": "active_workspace", "path": "hello.txt"},
            "SAFETY_SUPERVISOR_BLOCKED", id="llm_safety",
        ),
        pytest.param(
            "shell_guard",
            "run_command", {"cmd": ["sh", "-c", "rm -rf /etc/passwd"]},
            "WORKSPACE_SHELL_BLOCKED", id="shell_guard",
        ),
        pytest.param(
            "placement_blind_guard",
            "run_command", {"cmd": ["sudo", "apt-get", "install", "x"]},
            "SUDO_INTERACTIVE_BLOCKED", id="placement_blind_guard",
        ),
        # `arg_schema` used to be the fourth case here and is deliberately GONE. The
        # public-schema refusal moved to the placement-BLIND position, so it can no
        # longer be a refusal that lands after a successful prepare. Its replacement is
        # the stronger statement in
        # `test_a_malformed_call_is_refused_before_the_target_is_touched_at_all`: nothing
        # is prepared, so there is nothing to withdraw. The three cases left are the
        # authorities that genuinely do run after prepare, and they still prove the
        # withdrawal.
    ]


@pytest.mark.parametrize("kind,tool,args,marker", _refusal_cases())
def test_a_home_refusal_after_prepare_withdraws_the_prepared_operation(
    wired, monkeypatch, kind, tool, args, marker,
):
    if kind == "llm_safety":
        import ouroboros.safety as safety

        monkeypatch.setattr(
            safety, "check_safety",
            lambda *a, **k: (False, f"⚠️ {marker}: the supervisor refused this call."),
        )
    result, calls = wired.run(tool, dict(args))

    assert result.startswith("⚠️"), result
    if marker:
        assert marker in result
    assert f"prepare:{REMOTE_NATIVE_TOOL_OPERATION[tool]}" in calls, calls
    assert not any(call.startswith("execute_prepared") for call in calls), calls
    # The token the target reserved is withdrawn, not left to expire…
    assert calls.count("abort_prepared") == 1, calls
    # …and the target really dropped the prepared row.
    assert wired.transport.prepared == {}


def test_a_malformed_call_is_refused_before_the_target_is_touched_at_all(wired):
    """A call the PUBLIC SCHEMA rejects never reaches the other machine.

    The refusal was always correct — `TOOL_ARG_ERROR` won in the end — but it used to be
    decided AFTER `prepare_operation`, so a misspelled argument reserved a prepared token
    on the target (and, for a tool that stages blobs, staged them) before Home said no.
    A schema verdict is placement-BLIND: nothing about it can differ per host. So the
    assertion is not about the text, which never changed, but about the WORK: no prepare,
    hence nothing to abort.

    This is the negative half of the parity rule. The positive half — the same malformed
    call gets the same text on the local route — lives in `tests/test_route_refusal_parity.py`.
    """
    result, calls = wired.run("read_file", {"root": "active_workspace"})

    assert "TOOL_ARG_ERROR" in result, result
    assert not any(call.startswith("prepare:") for call in calls), calls
    assert "abort_prepared" not in calls, calls
    assert wired.transport.prepared == {}


def test_a_binding_mismatch_after_authorization_withdraws_the_prepared_operation(
    wired, monkeypatch,
):
    """The one refusal that fires between AUTHORIZE and EXECUTE.

    Nothing is supposed to rewrite argv or cwd there; the prepared token is what makes
    "supposed to" checkable. When it fires, the target is already holding the token
    that the mismatch proves must not be replayed — so this is the refusal that most
    needs the withdrawal, and it is the one the owner named.
    """
    from ouroboros.tools import registry as registry_mod

    real = registry_mod.project_dispatch_args

    def drifting(ctx, name, args, **kwargs):
        """Only the registry's POST-authorization re-projection is patched here.

        `bind_execution_args` reaches `project_dispatch_args` through its own module,
        so the token still binds the authorized command and this is a genuine
        after-the-fact drift rather than both sides moving together.
        """
        projections = real(ctx, name, args, **kwargs)
        return dataclasses.replace(
            projections,
            execution_args=dataclasses.replace(
                projections.execution_args, argv=(*projections.execution_args.argv, "--drifted"),
            ),
        )

    monkeypatch.setattr(registry_mod, "project_dispatch_args", drifting)
    result, calls = wired.run("run_command", {"cmd": ["true"]})

    assert result.startswith("⚠️ PREPARED_CALL_BINDING_MISMATCH")
    assert not any(call.startswith("execute_prepared") for call in calls), calls
    assert calls.count("abort_prepared") == 1, calls


def test_an_executed_operation_is_never_withdrawn(wired):
    """The hand-off: once the operation is the target's, Home does not withdraw it.

    Including when the target REFUSES the execute — an operation that may have started
    is the target's and the reconciliation path's to settle, and a Home abort there
    would be a second authority over one operation's completion.
    """
    _result, calls = wired.run("read_file", {"root": "active_workspace", "path": "hello.txt"})
    assert "execute_prepared:read_file" in calls
    assert "abort_prepared" not in calls, calls

    from ouroboros.workspace_diagnostics import RemoteWorkspaceError

    def refusing_execute(_message):
        raise RemoteWorkspaceError(
            "remote_disk_full", "no space", phase="execute", completion="not_started",
        )

    wired.transport.execute_prepared = refusing_execute
    result, calls = wired.run("write_file", {
        "root": "active_workspace", "path": "n.txt", "content": "x",
    })
    assert "remote_disk_full" in result
    assert "abort_prepared" not in calls, calls


def test_the_withdrawal_is_idempotent_and_never_masks_the_refusal(wired, monkeypatch):
    """Two properties of the seam itself, both required by the owner.

    Idempotence: the register is released BEFORE the transport call, so a second
    withdrawal of the same dispatch is a no-op rather than a second abort.

    Precedence: a dead transport must not replace the owner's diagnosis with a cleanup
    error. The refusal is why the tool call failed; "the abort also failed" is a fact
    about a token that will expire on its own anyway.
    """
    from ouroboros.tools.dispatch_execute import withdraw_outstanding_prepare
    from ouroboros.tools.dispatch_prepare import OutstandingPrepare, prepare_operation

    ctx = wired.registry._ctx
    register = OutstandingPrepare()
    prepared = prepare_operation(ctx, "read_file", {"path": "hello.txt"}, outstanding=register)
    assert prepared.native is not None and register.prepared is prepared

    assert withdraw_outstanding_prepare(ctx, register) is True
    assert register.prepared is None
    assert wired.transport.calls.count("abort_prepared") == 1
    # Idempotent: nothing left to withdraw, and no second RPC.
    assert withdraw_outstanding_prepare(ctx, register) is False
    assert wired.transport.calls.count("abort_prepared") == 1

    # A dead transport: swallowed, so the caller's own refusal survives.
    register.claim(prepare_operation(ctx, "read_file", {"path": "hello.txt"}))

    def dead(_message):
        raise RuntimeError("the ssh session is gone")

    monkeypatch.setattr(wired.transport, "abort_prepared", dead)
    assert withdraw_outstanding_prepare(ctx, register) is False
    assert register.prepared is None

    # …and through the WHOLE dispatch, which is where it matters: the owner must read
    # the refusal, not an error about the cleanup that followed it.
    import ouroboros.safety as safety

    monkeypatch.setattr(
        safety, "check_safety",
        lambda *a, **k: (False, "⚠️ SAFETY_SUPERVISOR_BLOCKED: the supervisor refused this."),
    )
    result = wired.registry.execute("read_file", {"root": "active_workspace", "path": "hello.txt"})
    assert result == "⚠️ SAFETY_SUPERVISOR_BLOCKED: the supervisor refused this."


def test_nothing_is_withdrawn_for_a_home_local_operation(wired, monkeypatch):
    """A Home refusal of a Home operation has no target state to release — including
    a Home-native root on a remote task, which never prepared anything."""
    import ouroboros.safety as safety

    monkeypatch.setattr(safety, "check_safety", lambda *a, **k: (False, "⚠️ NOPE"))
    for tool, args in (
        ("tree_read", {}),
        ("read_file", {"root": "artifact_store", "path": "probe.txt"}),
    ):
        _result, calls = wired.run(tool, dict(args))
        assert calls == [], (tool, calls)

# ── (i) the path Home authorized IS the path the target touches ────────────────
#
# The argv half of this contract shipped and the PATH half did not, and the token
# made that invisible: `ExecutionArgs` carried no path field, so the execution hash
# of a `read_file` was a CONSTANT per tool, and `execute_prepared` replays the
# TARGET's own `execution_args`. A target that answered a prepare with a different
# `path` was authorized for one file and ran on another with nothing anywhere
# disagreeing — Home authorized `write_file path="ok.txt"` and `PWNED.txt` appeared,
# `edit_text path="hello.txt"` edited `victim.txt`, `read_file path="hello.txt"`
# returned `SECRET.txt`.

_PATH_SUBSTITUTION_CASES = [
    pytest.param(
        "read_file", {"root": "active_workspace", "path": "hello.txt"},
        "SECRET.txt", id="read_file",
    ),
    pytest.param(
        "write_file", {"root": "active_workspace", "path": "ok.txt", "content": "x\n"},
        "PWNED.txt", id="write_file",
    ),
    pytest.param(
        "edit_text",
        {"root": "active_workspace", "path": "hello.txt", "old_str": "hello", "new_str": "bye"},
        "victim.txt", id="edit_text",
    ),
    pytest.param(
        "list_files", {"root": "active_workspace", "path": "src"},
        "secrets", id="list_files",
    ),
    pytest.param(
        "search_code", {"root": "active_workspace", "query": "target_fn", "path": "src"},
        "secrets", id="search_code",
    ),
    pytest.param(
        "query_code", {"op": "symbols", "root": "active_workspace", "path": "src/mod.py"},
        "secrets/leak.py", id="query_code",
    ),
]


def _substitute_prepared_path(monkeypatch, tool: str, substituted: str) -> None:
    from ouroboros import workspace_native

    real = workspace_native.prepare_native_operation

    def substituting(root, name, args, **kwargs):
        prepared = real(root, name, args, **kwargs)
        if name == tool:
            prepared.execution_args["path"] = substituted
        return prepared

    monkeypatch.setattr(workspace_native, "prepare_native_operation", substituting)


@pytest.mark.parametrize("tool,args,substituted", _PATH_SUBSTITUTION_CASES)
def test_a_substituted_path_is_refused_and_the_operation_never_runs(
    wired, monkeypatch, tool, args, substituted,
):
    (wired.root / "SECRET.txt").write_text("pwned read\n", encoding="utf-8")
    (wired.root / "victim.txt").write_text("victim content\n", encoding="utf-8")
    (wired.root / "secrets").mkdir(exist_ok=True)
    (wired.root / "secrets" / "leak.py").write_text("def leak():\n    pass\n", encoding="utf-8")
    _substitute_prepared_path(monkeypatch, tool, substituted)

    result, calls = wired.run(tool, dict(args))

    assert result.startswith("⚠️ REMOTE_PATH_SUBSTITUTED"), result
    assert substituted in result and str(args.get("path")) in result
    assert not any(call.startswith("execute_prepared") for call in calls), calls
    # Nothing the substituted path names was read, written or edited…
    assert "pwned read" not in result and "def leak" not in result
    assert not (wired.root / "PWNED.txt").exists()
    assert (wired.root / "victim.txt").read_text(encoding="utf-8") == "victim content\n"
    # …and the token the target reserved under the substituted path is withdrawn.
    assert calls.count("abort_prepared") == 1, calls


def test_a_pure_spelling_normalization_is_adopted_and_the_operation_runs(wired):
    """`./hello.txt` normalizes to `hello.txt` ON THE TARGET, and that is fine.

    The split is not "did the target disclose it" but something Home can check for
    itself: `native_relative_spelling` is the contract BOTH sides share, so Home
    derives the canonical spelling independently. A target path equal to it is a tidied
    spelling and is adopted into the authorized set; anything else is a substitution.
    """
    result, calls = wired.run("read_file", {"root": "active_workspace", "path": "./hello.txt"})
    _assert_routed(result, calls, "read_file")
    assert "hello target" in result
    executed = wired.transport.executed[-1]
    assert executed["execution_args"]["path"] == "hello.txt"


def test_a_target_may_not_fill_in_a_path_home_never_named(wired):
    """A target that rewrites the path is refused — on a WELL-FORMED call.

    The call used to be `read_file` with no `path` at all, which reached prepare only
    because the public-schema refusal ran after it. That refusal is now placement-blind
    and fires first, so a malformed call can no longer be the vehicle for this test —
    and it should not be: the substitution guard has nothing to do with argument shape.
    The vehicle is a call Home fully authorized, whose path the target then swaps.
    """
    from ouroboros import workspace_native

    real = workspace_native.prepare_native_operation

    def substituting(root, name, args, **kwargs):
        prepared = real(root, name, args, **kwargs)
        if name == "read_file":
            prepared.execution_args["path"] = "SECRET.txt"
        return prepared

    (wired.root / "SECRET.txt").write_text("pwned read\n", encoding="utf-8")

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(workspace_native, "prepare_native_operation", substituting)
        result, calls = wired.run(
            "read_file", {"root": "active_workspace", "path": "hello.txt"}
        )
    assert result.startswith("⚠️ REMOTE_PATH_SUBSTITUTED"), result
    assert "pwned read" not in result
    assert not any(call.startswith("execute_prepared") for call in calls), calls


def test_the_execution_hash_distinguishes_two_paths_of_one_tool(wired):
    """The CONTRACT the constant hash violated: two paths, two bindings.

    Pinned as its own case because the whole substitution class was reachable only
    because this was false — `_execution_hash("read_file", …)` answered the same
    digest for every file in the workspace.
    """
    from ouroboros.tools.dispatch_prepare import bind_execution_args, prepare_operation

    ctx = wired.registry._ctx
    digests = []
    for path in ("hello.txt", "src/mod.py"):
        prepared = prepare_operation(ctx, "read_file", {"root": "active_workspace", "path": path})
        assert prepared.available and prepared.native_routed
        bound = bind_execution_args(
            prepared, ctx, {"root": "active_workspace", "path": path}, runtime_mode="advanced",
        )
        assert bound.projections.execution_args.paths == (path,)
        assert bound.projections.execution_args.resource_root == "active_workspace"
        digests.append(bound.prepared_hash)
    assert digests[0] != digests[1]


def test_the_execution_hash_distinguishes_two_roots_of_one_path(wired):
    """`read_file(root='active_workspace')` and `root='artifact_store'` are not one call."""
    from ouroboros.tools.dispatch_args import project_dispatch_args
    from ouroboros.tools.dispatch_prepare import _execution_hash

    ctx = wired.registry._ctx
    digests = [
        _execution_hash(
            "read_file",
            project_dispatch_args(ctx, "read_file", {"root": root, "path": "x.txt"}).execution_args,
        )
        for root in ("active_workspace", "artifact_store")
    ]
    assert digests[0] != digests[1]


def test_a_rewritten_content_argument_is_refused_like_a_rewritten_path(wired):
    """The residue is CLOSED, not enumerated: the target may canonicalize argv, path
    and cwd, and every other argument it hands back must be Home's own.

    Enumerating permitted keys one at a time is precisely how the path key came to be
    missing for six tools, so `edit_text`'s `old_str`/`new_str` and `write_file`'s
    `content` are covered by the residual rule rather than by a table entry each.
    """
    from ouroboros import workspace_native

    real = workspace_native.prepare_native_operation

    def substituting(root, name, args, **kwargs):
        prepared = real(root, name, args, **kwargs)
        if name == "write_file":
            prepared.execution_args["content"] = "OWNED\n"
        return prepared

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(workspace_native, "prepare_native_operation", substituting)
        result, calls = wired.run(
            "write_file", {"root": "active_workspace", "path": "new.txt", "content": "mine\n"},
        )
    assert result.startswith("⚠️ REMOTE_ARGS_SUBSTITUTED"), result
    assert "content" in result
    assert not any(call.startswith("execute_prepared") for call in calls), calls
    assert not (wired.root / "new.txt").exists()


# ── (j) read_file / list_files APPLY the policy they were bound to ────────────
#
# Both were declared on the `workspace_query` channel, both had their echo hash
# verified at prepare, and neither called the evaluator: `read_file` shipped
# whatever byte it was pointed at and `list_files` named every entry, while
# `search_code` on the SAME channel filtered and disclosed. So a reader could not
# tell "the policy ran and allowed this" from "the policy never ran" — the worse of
# the two states, because the second one looks exactly like the first.


def test_a_read_of_a_policy_excluded_file_is_refused_and_says_why(wired):
    (wired.root / ".env").write_text("API_KEY=leak\n", encoding="utf-8")
    result, calls = wired.run("read_file", {"root": "active_workspace", "path": ".env"})
    assert "REMOTE_EXPORT_POLICY_EXCLUDED" in result, result
    assert "credential-like file" in result
    assert "API_KEY" not in result and "leak" not in result
    assert "execute_prepared:read_file" in calls, calls


def test_an_ordinary_read_is_untouched_and_declares_the_policy_that_ran(wired):
    result, calls = wired.run("read_file", {"root": "active_workspace", "path": "hello.txt"})
    _assert_routed(result, calls, "read_file")
    assert "hello target" in result
    assert "POLICY_FILTERED" not in result


def test_a_listing_excludes_the_sensitive_entries_and_discloses_the_count(wired):
    (wired.root / ".env").write_text("API_KEY=leak\n", encoding="utf-8")
    (wired.root / "credentials.json").write_text("{}\n", encoding="utf-8")
    result, calls = wired.run("list_files", {"root": "active_workspace", "path": "."})
    listing, _, note = result.partition("\n\n")
    assert ".env" not in listing and "credentials.json" not in listing, result
    assert "hello.txt" in listing, result
    assert "LIST_POLICY_FILTERED" in note, result
    assert "2 entries excluded" in note, result
    assert ".env: sensitive_file" in note and "credentials.json: sensitive_file" in note
    assert "execute_prepared:list_files" in calls, calls


def test_a_listing_with_nothing_to_exclude_is_byte_identical_to_before(wired):
    result, calls = wired.run("list_files", {"root": "active_workspace", "path": "src"})
    _assert_routed(result, calls, "list_files")
    assert "POLICY_FILTERED" not in result
    assert json.loads(result) == ["src/mod.py"]


def test_the_bulk_only_directory_rule_does_not_forbid_a_named_read(wired):
    """`.git/config` is readable on a local placement, so it stays readable here.

    The excluded-dirs rule exists so a TREE export does not ship `.git`; reading a
    path the model named by hand is a different question, and answering it with that
    rule would manufacture a placement divergence out of an enumeration rule.
    """
    result, calls = wired.run("read_file", {"root": "active_workspace", "path": ".git/config"})
    assert "REMOTE_EXPORT_POLICY_EXCLUDED" not in result, result
    listing, _calls = wired.run("list_files", {"root": "active_workspace", "path": "."})
    assert ".git/" in listing.partition("\n\n")[0], listing


def test_both_doors_disclose_through_the_same_wire_block_search_code_uses(wired):
    """The trace carries the D7 block, so Home's import side can VALIDATE it."""
    from ouroboros import workspace_native

    (wired.root / ".env").write_text("API_KEY=leak\n", encoding="utf-8")
    from ouroboros.export_policy_contract import build_policy_document

    prepared = workspace_native.prepare_native_operation(
        wired.root,
        "list_files",
        {"path": ".", "_export_policy": build_policy_document(channel="workspace_query")},
    )
    result = workspace_native.execute_native_operation(
        wired.root, "list_files", prepared.execution_args,
        native_facts=prepared.native_facts,
    )
    block = result.envelope.trace["export_policy"]
    assert block["policy_scope"] == "policy_filtered"
    assert block["excluded_count"] == 1
    # `judged` rides with every row so Home can re-derive the claim rather than trust it;
    # it equals `path` unless an ALIAS was the finding.
    assert block["excluded"] == [
        {"path": ".env", "reason": "sensitive_file", "judged": ".env"}
    ]
    assert block["policy_hash"] == prepared.native_facts["export_policy_hash"]
    # …and the listing DECLARES what it handed over, which is what Home re-evaluates.
    assert block["exported"] and ".env" not in block["exported"]


# ── (k) a protected artifact in a REMOTE workspace is not writable ────────────
#
# `resource_policy` black-box protection was enforced only inside the Home handlers
# `tools/core._write_file`/`_edit_text`, which the native route replaces — so on an
# ssh placement a protected artifact could be overwritten. Same class as the subagent
# secret denial: a policy that lived in a handler body and had no remote twin. Home
# cannot spell the check for a target path (it has no Home path to ask
# `protected_artifacts` about), and the document already carries those paths projected
# to target spellings, so the SOURCE applies it — §3.2's own division of labour.


def _protected_policy(*paths):
    from ouroboros.export_policy_contract import build_policy_document

    return build_policy_document(channel="workspace_query", protected_paths=list(paths))


@pytest.mark.parametrize(
    "tool,args",
    [
        pytest.param(
            "write_file", {"path": "model.bin", "content": "x"}, id="write_file",
        ),
        pytest.param(
            "write_file",
            {"files": [{"path": "ok.txt", "content": "a"}, {"path": "model.bin", "content": "b"}]},
            id="write_file_batch_row",
        ),
        pytest.param(
            "edit_text",
            {"path": "model.bin", "old_str": "weights", "new_str": "tampered"},
            id="edit_text",
        ),
    ],
)
def test_a_protected_artifact_is_not_mutable_on_the_target(wired, tool, args):
    (wired.root / "model.bin").write_text("weights\n", encoding="utf-8")
    prepared = workspace_native.prepare_native_operation(
        wired.root, tool, {**args, "_export_policy": _protected_policy("model.bin")},
    )
    result = workspace_native.execute_native_operation(
        wired.root, tool, prepared.execution_args, native_facts=prepared.native_facts,
    )
    assert "REMOTE_PROTECTED_ARTIFACT_BLOCKED" in result.envelope.text, result.envelope.text
    assert (wired.root / "model.bin").read_text(encoding="utf-8") == "weights\n"
    # A batch refuses as a whole rather than half-applying it.
    assert not (wired.root / "ok.txt").exists()


def test_an_unprotected_path_still_writes_under_the_same_policy(wired):
    prepared = workspace_native.prepare_native_operation(
        wired.root,
        "write_file",
        {"path": "notes.txt", "content": "fine\n", "_export_policy": _protected_policy("model.bin")},
    )
    result = workspace_native.execute_native_operation(
        wired.root, "write_file", prepared.execution_args,
        native_facts=prepared.native_facts,
    )
    assert "BLOCKED" not in result.envelope.text, result.envelope.text
    assert (wired.root / "notes.txt").read_text(encoding="utf-8") == "fine\n"


def test_the_credential_name_rules_do_NOT_block_a_write(wired):
    """A task writing a `.env` into its own workspace is ordinary on a local
    placement, so only the PROTECTED-ARTIFACT class participates on the write side —
    refusing the rest here would manufacture a divergence out of an export rule."""
    prepared = workspace_native.prepare_native_operation(
        wired.root,
        "write_file",
        {"path": ".env", "content": "A=1\n", "_export_policy": _protected_policy("model.bin")},
    )
    result = workspace_native.execute_native_operation(
        wired.root, "write_file", prepared.execution_args,
        native_facts=prepared.native_facts,
    )
    assert "BLOCKED" not in result.envelope.text, result.envelope.text
    assert (wired.root / ".env").read_text(encoding="utf-8") == "A=1\n"


# ── the byte-read ORACLE: bytes_equal is a read, and reads have a policy ──────
#
# `verify_and_record`'s `bytes_equal` reports sizes plus a hexdump around the first
# divergence, which is a read of both operands by any honest accounting. Home's half
# refuses a protected-artifact operand for exactly that reason
# (`tools/verify._bytes_equal_confinement_block`, protected_artifacts `read_bytes`);
# the target's half enforced workspace containment and nothing else, so a remote
# bytes_equal could hexdump a black-box reference binary the identical Home call
# refuses. The refusal that closes it is the operation's own bound export document.


def _verify_bytes_equal(root, a, b, policy):
    prepared = workspace_native.prepare_native_operation(
        root,
        "verify_remote_check",
        {
            "cmd": ["true"],
            "cwd": "",
            "artifact_paths": [a, b],
            "expected_match": "bytes_equal",
            "_export_policy": policy,
        },
    )
    return workspace_native.execute_native_operation(
        root, "verify_remote_check", prepared.execution_args,
        native_facts=prepared.native_facts,
    )


def test_a_protected_artifact_is_not_byte_comparable_on_the_target(wired):
    (wired.root / "model.bin").write_text("weights-SENTINEL\n", encoding="utf-8")
    (wired.root / "mine.bin").write_text("guess\n", encoding="utf-8")
    result = _verify_bytes_equal(
        wired.root, "model.bin", "mine.bin", _protected_policy("model.bin"),
    )
    assert "REMOTE_EXPORT_POLICY_EXCLUDED" in result.envelope.text, result.envelope.text
    # The point of the refusal: no size, no hexdump, no byte of the artifact.
    assert "weights" not in result.envelope.text
    assert "77 65 69" not in result.envelope.text


def test_the_second_operand_is_judged_too(wired):
    """Both operands, because the oracle reads both — a first-operand-only check
    would be satisfied by swapping the arguments."""
    (wired.root / "model.bin").write_text("weights\n", encoding="utf-8")
    (wired.root / "mine.bin").write_text("guess\n", encoding="utf-8")
    result = _verify_bytes_equal(
        wired.root, "mine.bin", "model.bin", _protected_policy("model.bin"),
    )
    assert "REMOTE_EXPORT_POLICY_EXCLUDED" in result.envelope.text, result.envelope.text


def test_an_ordinary_pair_still_compares_on_the_target(wired):
    """The guard must not be satisfiable by refusing every comparison."""
    (wired.root / "a.bin").write_text("same\n", encoding="utf-8")
    (wired.root / "b.bin").write_text("same\n", encoding="utf-8")
    result = _verify_bytes_equal(
        wired.root, "a.bin", "b.bin", _protected_policy("model.bin"),
    )
    assert "EXCLUDED" not in result.envelope.text, result.envelope.text
    assert result.envelope.trace["verification"]["bytes_equal"]["matched"] is True


# ── a Home-native root under an ssh placement is still Home's ────────────────


def _light_write(registry, root_name: str, path: str) -> str:
    return registry.execute(
        "write_file", {"root": root_name, "path": path, "content": "TAMPERED\n"}
    )


@pytest.mark.parametrize(
    ("root_name", "path"),
    [("system_repo", "notes.txt"), ("runtime_data", "memory/x.md")],
)
def test_light_mode_guards_a_home_native_root_the_same_on_both_placements(
    wired, tmp_path, monkeypatch, root_name, path
):
    """The ratified root matrix says these calls are HOME's, so their guards are too.

    A remote task calling against `system_repo` or `runtime_data` keeps its Home
    handler — that is the normal path, not a degradation, because that is where a
    remote task's artifacts, scratch and the Ouroboros body actually live. The
    dispatch's single Home-side resolution was gated on the PLACEMENT rather than on
    whether the operation leaves Home, so for every ssh task it was skipped and every
    guard keyed on the binding took its `None` fallback — which reads `workspace_mode`,
    true for any admitted remote task. `runtime_mode=light` exists to make Ouroboros
    self-repo and control-plane mutation impossible, and for ssh tasks it silently
    stopped: the model could rewrite the owner's live Ouroboros checkout and data drive
    from a task that was supposed to be operating on a remote project. One policy, two
    doors — with no conflict, no lint and no failing test.

    Asserted as PARITY rather than as a refusal, because a refusal alone would also be
    satisfied by a placement that refuses everything.
    """

    import ouroboros.config as config

    monkeypatch.setattr(config, "get_runtime_mode", lambda: "light")

    remote = _light_write(wired.registry, root_name, path)

    local_ctx = ToolContext(
        repo_dir=wired.registry._ctx.repo_dir,
        drive_root=wired.registry._ctx.drive_root,
        task_id="task-local",
        workspace_root=str(wired.root),
        workspace_mode="external",
    )
    local_registry = ToolRegistry(
        repo_dir=local_ctx.repo_dir, drive_root=local_ctx.drive_root
    )
    local_registry.set_context(local_ctx)
    local = _light_write(local_registry, root_name, path)

    assert "LIGHT_MODE_BLOCKED" in local, local
    assert "LIGHT_MODE_BLOCKED" in remote, remote
    assert not (wired.registry._ctx.repo_dir / path).exists()
    assert not (wired.registry._ctx.drive_root / path).exists()
