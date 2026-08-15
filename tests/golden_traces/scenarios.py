# tests/golden_traces/scenarios.py — the scenario matrix for the dispatch golden traces.
#
# Each scenario builds an isolated ToolRegistry + ToolContext under a caller
# tmp dir (pytest tmp_path — parallel-safe) and declares the execute() calls
# whose guard-call order and results are recorded. Scenarios that spawn REAL
# processes (run_command/run_script/services/verify) carry serial=True and are
# marked @pytest.mark.serial by the test module.
from __future__ import annotations

import os
import pathlib
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tools.registry import ToolContext, ToolRegistry


@dataclass
class ScenarioRun:
    registry: ToolRegistry
    calls: List[Tuple[str, dict]]
    roots: Dict[str, str]  # normalizer placeholder -> raw path
    # An ssh scenario owns a live broker and a patched module attribute, so it needs a
    # teardown a plain builder return value cannot express.
    cleanup: Optional[Callable[[], None]] = None
    # Record the REMOTE seam (prepare / binding / native execute / the two
    # placement-blind refusals / the listing filter). Off by default and therefore
    # off for every local and docker scenario: those functions run on a local
    # dispatch too, and recording them unconditionally would rewrite fixtures whose
    # byte-identity is the whole point of this directory.
    remote_seam: bool = False


@dataclass
class Scenario:
    name: str
    description: str
    build: Callable[[pathlib.Path], ScenarioRun]
    env: Dict[str, str] = field(default_factory=dict)
    serial: bool = False
    # Env vars that depend on the caller's tmp dir (a stub binary's PATH entry).
    # Applied with monkeypatch BEFORE build(), so the scenario body sees them.
    env_factory: Optional[Callable[[pathlib.Path], Dict[str, str]]] = None


# --------------------------------------------------------------------------- #
# Context builders
# --------------------------------------------------------------------------- #
def _git_init(path: pathlib.Path, files: Dict[str, str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        target = path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    run = lambda *argv: subprocess.run(  # noqa: E731 — tiny local helper
        argv, cwd=str(path), capture_output=True, text=True, timeout=30, check=False,
    )
    run("git", "init", "-q", "-b", "main")
    run("git", "-c", "user.email=golden@test", "-c", "user.name=golden", "add", "-A")
    run(
        "git", "-c", "user.email=golden@test", "-c", "user.name=golden",
        "commit", "-q", "-m", "golden base", "--no-gpg-sign",
    )


_REPO_FILES = {
    "hello.txt": "hello golden\n",
    "out.txt": "artifact-bytes\n",
    "src/mod.py": "def golden_fn():\n    return 41\n",
}


def _normal(base: pathlib.Path, **ctx_kwargs) -> Tuple[ToolRegistry, Dict[str, str]]:
    """Plain task context: temp git repo as active workspace + temp data drive."""
    repo = base / "repo"
    drive = base / "data"
    _git_init(repo, _REPO_FILES)
    task_drive = drive / "task_drives" / "golden-task"
    task_drive.mkdir(parents=True, exist_ok=True)
    (task_drive / "note.txt").write_text("drive note\n", encoding="utf-8")
    ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id="golden-task", **ctx_kwargs)
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry.set_context(ctx)
    return registry, {"REPO": str(repo), "DRIVE": str(drive)}


def _workspace(base: pathlib.Path, **ctx_kwargs) -> Tuple[ToolRegistry, Dict[str, str]]:
    """External-workspace task: git worktree in a separate temp folder."""
    repo = base / "repo"
    drive = base / "data"
    workspace = base / "ws"
    _git_init(repo, {"README.md": "system repo\n"})
    _git_init(workspace, {"app.py": "print('app')\n"})
    drive.mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(
        repo_dir=repo, drive_root=drive, task_id="golden-ws-task",
        workspace_root=workspace, workspace_mode="external", **ctx_kwargs,
    )
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry.set_context(ctx)
    return registry, {"REPO": str(repo), "DRIVE": str(drive), "WS": str(workspace)}


def _bare(base: pathlib.Path, **ctx_kwargs) -> Tuple[ToolRegistry, Dict[str, str]]:
    """Minimal context (no git) for pure early-exit scenarios."""
    repo = base / "repo"
    drive = base / "data"
    repo.mkdir(parents=True, exist_ok=True)
    drive.mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(repo_dir=repo, drive_root=drive, **ctx_kwargs)
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry.set_context(ctx)
    return registry, {"REPO": str(repo), "DRIVE": str(drive)}


# --------------------------------------------------------------------------- #
# docker_exec backend: a STUB `docker` on PATH
# --------------------------------------------------------------------------- #
# §9 asks for byte-identity on local + docker mapped/unmapped, and until now the
# whole matrix was `kind=local` — `workspace_executor_local` is the LOCAL branch,
# so the docker executor had no golden coverage at all. A real container is not
# usable here (the daemon is present but the registry is unreachable, so `pull`
# hangs), and a golden trace must not depend on a network anyway.
#
# The stub is enough because what the trace pins is the CONTRACT the dispatch
# pipeline computes and hands to the backend, not what a container does with it:
# whether the operation routes to the executor at all, the host→backend cwd
# projection, the container name, the network mode, and the recorded executor
# trace. It deliberately echoes ONLY the `--workdir` and container name: the real
# argv wraps the command in a `sh -lc` shell whose pidfile carries a fresh uuid4
# per run, which is exactly the kind of noise a byte-identical fixture must not
# contain.
#
# What the stub therefore does NOT cover, stated plainly rather than implied: the
# in-container process itself — signal delivery to the backend pid, the
# setsid/pidfile teardown path on timeout, `docker inspect` network enforcement
# against a live container, and anything about the container filesystem. Those
# need a real image and belong to an integration lane.
_DOCKER_STUB = """#!/bin/sh
# Golden-trace stub for `docker`. Deterministic by construction: it prints the
# backend workdir and container name and nothing else, so the recorded trace
# carries the path projection without the per-run uuid in the real wrapper.
if [ "$1" = "inspect" ]; then
  echo none
  exit 0
fi
if [ "$1" = "exec" ]; then
  shift
  workdir=""
  container=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --workdir) workdir="$2"; shift 2 ;;
      -*) shift ;;
      *) container="$1"; break ;;
    esac
  done
  echo "docker-stub workdir=$workdir container=$container"
  exit 0
fi
echo "docker-stub: unsupported docker invocation" >&2
exit 1
"""

_DOCKER_BACKEND_ROOT = "/workspace"
_DOCKER_CONTAINER = "golden-exec-container"


def docker_stub_env(base: pathlib.Path) -> Dict[str, str]:
    """Put a stub `docker` first on PATH for the docker_exec scenarios."""

    bindir = base / "stub_bin"
    bindir.mkdir(parents=True, exist_ok=True)
    stub = bindir / "docker"
    stub.write_text(_DOCKER_STUB, encoding="utf-8")
    stub.chmod(0o755)
    return {"PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"}


def _docker_workspace(base: pathlib.Path) -> Tuple[ToolRegistry, Dict[str, str]]:
    """External workspace whose ONLY executor mapping is the workspace root."""

    registry, roots = _workspace(base)
    # A real subdirectory, so the host→backend projection has something to carry
    # beyond the mapping base.
    (pathlib.Path(roots["WS"]) / "sub").mkdir(parents=True, exist_ok=True)
    registry._ctx.executor_ref = {
        "kind": "docker_exec",
        "container_name": _DOCKER_CONTAINER,
        "network": "host",
        "workspace_host_path": roots["WS"],
        "workspace_backend_path": _DOCKER_BACKEND_ROOT,
    }
    return registry, roots


# --------------------------------------------------------------------------- #
# Scenario builders (one function per fixture)
# --------------------------------------------------------------------------- #
def _read_file_roots(base):
    registry, roots = _normal(base)
    return ScenarioRun(registry, [
        ("read_file", {"root": "active_workspace", "path": "hello.txt"}),
        ("read_file", {"root": "system_repo", "path": "hello.txt"}),
        ("read_file", {"root": "task_drive", "path": "note.txt"}),
    ], roots)


def _write_file_roots(base):
    registry, roots = _normal(base)
    return ScenarioRun(registry, [
        ("write_file", {"root": "active_workspace", "path": "new.txt", "content": "ws write\n"}),
        ("write_file", {"root": "system_repo", "path": "sys_new.txt", "content": "sys write\n"}),
        ("write_file", {"root": "task_drive", "path": "drive_new.txt", "content": "drive write\n"}),
    ], roots)


def _edit_text_workspace(base):
    registry, roots = _normal(base)
    return ScenarioRun(registry, [
        ("edit_text", {
            "root": "active_workspace", "path": "hello.txt",
            "old_str": "hello golden", "new_str": "hello edited",
        }),
    ], roots)


def _list_files_roots(base):
    registry, roots = _normal(base)
    return ScenarioRun(registry, [
        ("list_files", {"root": "active_workspace", "path": "."}),
        ("list_files", {"root": "system_repo", "path": "src"}),
        ("list_files", {"root": "task_drive", "path": "."}),
    ], roots)


def _search_code_roots(base):
    registry, roots = _normal(base)
    return ScenarioRun(registry, [
        ("search_code", {"root": "active_workspace", "query": "golden_fn"}),
        ("search_code", {"root": "system_repo", "query": "hello", "path": "."}),
        ("search_code", {"root": "task_drive", "query": "drive note"}),
    ], roots)


def _query_code_roots(base):
    registry, roots = _normal(base)
    return ScenarioRun(registry, [
        ("query_code", {"op": "symbols", "root": "active_workspace", "path": "src"}),
        ("query_code", {"op": "definition", "root": "system_repo", "query": "golden_fn"}),
    ], roots)


def _run_command_echo(base):
    registry, roots = _normal(base)
    return ScenarioRun(registry, [
        ("run_command", {"cmd": ["echo", "golden"]}),
    ], roots)


def _run_command_sudo_blocked(base):
    registry, roots = _normal(base)
    return ScenarioRun(registry, [
        ("run_command", {"cmd": ["sudo", "whoami"]}),
    ], roots)


def _run_script_python(base):
    registry, roots = _normal(base)
    return ScenarioRun(registry, [
        ("run_script", {"script": "print('golden script')", "interpreter": "python3"}),
    ], roots)


def _services_lifecycle(base):
    registry, roots = _normal(base)
    return ScenarioRun(registry, [
        ("start_service", {"cmd": ["sleep", "5"], "name": "goldensvc"}),
        ("service_status", {"name": "goldensvc"}),
        ("stop_service", {"name": "goldensvc"}),
    ], roots)


def _verify_and_record(base):
    registry, roots = _normal(base)
    return ScenarioRun(registry, [
        ("verify_and_record", {
            "contract_kind": "explicit_command",
            "check": ["echo", "verify-ok"],
            "expected": "verify-ok",
            "artifact_paths": ["out.txt"],
        }),
        ("verify_and_record", {
            "contract_kind": "artifact_observation",
            "artifact_paths": ["out.txt", "missing.bin"],
        }),
    ], roots)


def _unknown_tool(base):
    registry, roots = _bare(base)
    return ScenarioRun(registry, [
        ("definitely_not_a_tool", {"anything": 1}),
    ], roots)


def _disabled_tool(base):
    registry, roots = _bare(base, task_contract={"disabled_tools": ["run_command", "web_search"]})
    return ScenarioRun(registry, [
        ("run_command", {"cmd": ["echo", "nope"]}),
    ], roots)


def _workspace_allowlist_block(base):
    registry, roots = _workspace(base)
    return ScenarioRun(registry, [
        ("write_file", {"root": "active_workspace", "path": "feature.txt", "content": "ok\n"}),
        ("commit_reviewed", {"message": "not allowed in workspace mode"}),
    ], roots)


def _workspace_executor_local(base):
    registry, roots = _workspace(base)
    ws = roots["WS"]
    registry._ctx.executor_ref = {
        "kind": "local",
        "workspace_host_path": ws,
        "workspace_backend_path": ws,
    }
    return ScenarioRun(registry, [
        ("run_command", {"cmd": ["echo", "executor-local"]}),
    ], roots)


def _docker_exec_mapped_cwd(base):
    """The MAPPED branch: the cwd is inside the mapping, so it goes to the backend.

    What the fixture pins: `covers()` said yes, the operation routed to
    `workspace_executor.execute`, the host cwd was projected to the backend
    spelling, and the executor trace records kind/container/network/backend cwd.
    """

    registry, roots = _docker_workspace(base)
    return ScenarioRun(registry, [
        ("run_command", {"cmd": ["echo", "executor-docker-mapped"]}),
        # A cwd in a SUBDIRECTORY of the mapped root: the projection must carry the
        # relative tail onto the backend base, not collapse to the base itself.
        ("run_command", {"cmd": ["echo", "executor-docker-subdir"], "cwd": "sub"}),
    ], roots)


def _docker_exec_unmapped_root(base):
    """The UNMAPPED branch: a cwd outside every mapping runs on the HOST.

    Not an error — `covers()` documents non-coverage as "fall back to host
    execution", so a task_drive cwd under a docker-backed workspace executes
    locally and the container is never invoked. That is a deliberate contract and
    exactly the kind a refactor can silently invert (into an error, or worse into
    running host paths inside the container), so it is worth a byte-identical
    fixture. The stub's absence from stdout is the evidence.
    """

    registry, roots = _docker_workspace(base)
    return ScenarioRun(registry, [
        ("run_command", {"cmd": ["echo", "executor-docker-unmapped"], "cwd": "task_drive"}),
    ], roots)


def _light_mode_repo_write_block(base):
    registry, roots = _normal(base)
    return ScenarioRun(registry, [
        ("write_file", {"root": "active_workspace", "path": "notes.md", "content": "blocked\n"}),
    ], roots)


def _protected_path_write_block(base):
    registry, roots = _normal(base)
    return ScenarioRun(registry, [
        ("write_file", {"root": "active_workspace", "path": "prompts/SAFETY.md", "content": "HACK\n"}),
    ], roots)


def _ephemeral_turn_block(base):
    registry, roots = _bare(base, is_ephemeral_turn=True)
    return ScenarioRun(registry, [
        ("write_file", {"root": "active_workspace", "path": "x.txt", "content": "no\n"}),
    ], roots)


def _local_readonly_subagent_block(base):
    registry, roots = _bare(
        base, task_constraint=TaskConstraint(mode="local_readonly_subagent"),
    )
    return ScenarioRun(registry, [
        ("write_file", {"root": "active_workspace", "path": "x.txt", "content": "no\n"}),
    ], roots)


def _acting_no_workspace_block(base):
    registry, roots = _bare(
        base,
        task_constraint=TaskConstraint(mode="acting_subagent", surface="self_worktree"),
        task_metadata={"delegation_role": "subagent"},
    )
    return ScenarioRun(registry, [
        ("write_file", {"root": "active_workspace", "path": "x.txt", "content": "no\n"}),
        ("run_command", {"cmd": ["echo", "no"]}),
    ], roots)


# --------------------------------------------------------------------------- #
# ssh backend: a live broker over a FAKE wire, a real native kernel behind it
# --------------------------------------------------------------------------- #
# ARCHITECTURE names three kinds of executor — local, docker_exec and ssh — and this
# directory covered two of them. The gap was not cosmetic: the ssh route is the one
# that REPLACES the built-in handler, so the guard ORDER on it is a different sequence
# from the local one, and the whole class of "a policy that stops at the placement
# fork" is invisible to a fixture set that never takes the fork. Pinning it here means
# a dispatch refactor has to keep the remote order and the remote result TEXT, exactly
# as it already has to keep the local ones.
#
# The wire is fake and the TARGET is not: `tests/test_registry_remote_dispatch`'s
# harness runs the real `ouroboros.workspace_native` kernel against a real temp git
# worktree, so the recorded texts are the ones the real target produces. The harness is
# REUSED rather than restated for the reason that file's own docstring gives — two
# copies of the wiring would drift, and the fixture would then pin the copy.


def _ssh_workspace(base: pathlib.Path) -> ScenarioRun:
    """A registry on an ssh placement, plus the teardown its broker needs."""

    from tests.test_registry_remote_dispatch import wire_ssh_registry

    saved: List[Tuple[Any, str, Any]] = []

    class _Patcher:
        """The two methods `wire_ssh_registry` uses from pytest's monkeypatch.

        A golden scenario has no fixture to receive one, and `capture` already stubs
        `check_safety` itself — but the harness must not be forked just to drop its
        one patch, so the shim satisfies it and the scenario undoes it.
        """

        def setattr(self, obj, attr, value):  # noqa: A003 — mirrors monkeypatch
            saved.append((obj, attr, getattr(obj, attr)))
            setattr(obj, attr, value)

    patcher = _Patcher()
    generator = wire_ssh_registry(base, patcher)
    wired = next(generator)

    def cleanup() -> None:
        for _ in generator:  # drive the harness through its own finally
            pass
        for obj, attr, original in reversed(saved):
            setattr(obj, attr, original)

    return ScenarioRun(
        wired.registry,
        [],
        {"TARGET": str(wired.root)},
        cleanup=cleanup,
        remote_seam=True,
    )


def _with_calls(run: ScenarioRun, calls: List[Tuple[str, dict]]) -> ScenarioRun:
    run.calls = calls
    return run


def _ssh_read_and_list(base):
    """The two read doors, on the route that replaces their handlers.

    Both carry an export policy now, so the fixture also pins the D7 disclosure block
    the target emits on EVERY call — an absent block would let a quiet omission look
    like an unfiltered export, and that is a property of the TEXT.
    """

    run = _ssh_workspace(base)
    drive_note = base / "data" / "task_drives" / "remote-dispatch-task" / "note.txt"
    drive_note.parent.mkdir(parents=True, exist_ok=True)
    drive_note.write_text("home drive note\n", encoding="utf-8")
    return _with_calls(run, [
        ("read_file", {"root": "active_workspace", "path": "hello.txt"}),
        ("list_files", {"root": "active_workspace", "path": "."}),
        # A HOME-native root on the same placement: it must NOT route, and the result
        # must be about Home's own store rather than the same-named remote file. This
        # is the root MATRIX half of routing, and it once sent both to the target.
        ("read_file", {"root": "task_drive", "path": "note.txt"}),
    ])


def _ssh_write_and_edit(base):
    run = _ssh_workspace(base)
    return _with_calls(run, [
        ("write_file", {"root": "active_workspace", "path": "new.txt", "content": "remote write\n"}),
        ("edit_text", {
            "root": "active_workspace", "path": "hello.txt",
            "old_str": "hello target", "new_str": "hello edited",
        }),
    ])


def _ssh_run_command(base):
    """A process on the target, through the whole guard chain and the binding."""

    run = _ssh_workspace(base)
    return _with_calls(run, [
        ("run_command", {"cmd": ["echo", "golden-remote"]}),
        # Refused after prepare by the shell guard: the sudo rule is
        # placement-independent, and the fixture pins that the remote branch reaches it.
        ("run_command", {"cmd": ["sudo", "whoami"]}),
        # The interpreter allowlist, lifted into the pipeline. The trace must show it
        # answering BEFORE `prepare_operation` — a prepare ahead of it would mean the
        # target had already been handed an argv the allowlist never judged.
        ("run_script", {"script": "print(1)", "interpreter": "/bin/not-allowed"}),
        # ...and the allowed one still runs on the target.
        ("run_script", {"script": "print('golden-remote-script')", "interpreter": "python3"}),
    ])


def _ssh_restricted_subagent(base):
    """A restricted subagent on an ssh placement — the founding instance of the class.

    The READ is decided before prepare, so the fixture pins that the target is never
    asked: a `prepare_operation` event on that call would be the bug itself. The LIST
    of an ordinary directory is the other half — it routes, the target filters and
    DISCLOSES what the policy declined, and Home's own redaction adds the exact hidden
    count on top. Both statements are in the result text, which is why it is pinned
    here rather than only asserted somewhere.
    """

    run = _ssh_workspace(base)
    (pathlib.Path(run.roots["TARGET"]) / ".env").write_text("SECRET=1\n", encoding="utf-8")
    run.registry._ctx.task_constraint = TaskConstraint(mode="local_readonly_subagent")
    return _with_calls(run, [
        ("read_file", {"root": "active_workspace", "path": ".env"}),
        ("read_file", {"root": "active_workspace", "path": "hello.txt"}),
        ("list_files", {"root": "active_workspace", "path": "."}),
    ])


# --------------------------------------------------------------------------- #
# The matrix
# --------------------------------------------------------------------------- #
SCENARIOS: List[Scenario] = [
    Scenario("read_file_roots", "read_file across active_workspace / system_repo / task_drive", _read_file_roots),
    Scenario("write_file_roots", "write_file across active_workspace / system_repo / task_drive (advanced mode)", _write_file_roots),
    Scenario("edit_text_workspace", "edit_text str-replace on an active_workspace file", _edit_text_workspace),
    Scenario("list_files_roots", "list_files across active_workspace / system_repo / task_drive", _list_files_roots),
    Scenario("search_code_roots", "search_code across active_workspace / system_repo / task_drive", _search_code_roots),
    Scenario("query_code_roots", "query_code symbols (active_workspace) + definition (system_repo)", _query_code_roots),
    Scenario("run_command_echo", "harmless run_command argv echo through the full shell guard chain", _run_command_echo, serial=True),
    Scenario("run_command_sudo_blocked", "run_command sudo WITHOUT -n blocked pre-dispatch (no process spawned)", _run_command_sudo_blocked),
    Scenario("run_script_python", "run_script with resolved python3 interpreter", _run_script_python, serial=True),
    Scenario("services_lifecycle", "start_service + service_status + stop_service on a sleeper", _services_lifecycle, serial=True),
    Scenario("verify_and_record", "verify_and_record run-kind check + artifact_observation", _verify_and_record, serial=True),
    Scenario("unknown_tool", "unknown tool name early exit listing available tools", _unknown_tool),
    Scenario("disabled_tool", "task_contract.disabled_tools withholds run_command", _disabled_tool),
    Scenario("workspace_allowlist_block", "external workspace: allowed write_file + commit_reviewed outside allowlist", _workspace_allowlist_block),
    Scenario("workspace_executor_local", "external workspace with executor_ref kind=local path mapping, run_command", _workspace_executor_local, serial=True),
    Scenario(
        "docker_exec_mapped_cwd",
        "executor_ref kind=docker_exec, cwd INSIDE the mapping: routed to the backend with the projected workdir",
        _docker_exec_mapped_cwd,
        serial=True,
        env_factory=docker_stub_env,
    ),
    Scenario(
        "docker_exec_unmapped_root",
        "executor_ref kind=docker_exec, cwd OUTSIDE every mapping: documented host fallback, container never invoked",
        _docker_exec_unmapped_root,
        serial=True,
        env_factory=docker_stub_env,
    ),
    Scenario("light_mode_repo_write_block", "runtime_mode=light blocks write_file into the repo", _light_mode_repo_write_block, env={"OUROBOROS_RUNTIME_MODE": "light"}),
    Scenario("protected_path_write_block", "advanced mode write_file to protected prompts/SAFETY.md blocked", _protected_path_write_block),
    Scenario("ephemeral_turn_block", "ephemeral decision turn blocks non-allowlisted write_file", _ephemeral_turn_block),
    Scenario("local_readonly_subagent_block", "local readonly subagent blocked from write_file", _local_readonly_subagent_block),
    Scenario("acting_no_workspace_block", "acting subagent without resolved workspace: write_file + run_command blocked", _acting_no_workspace_block),
    Scenario(
        "ssh_read_and_list",
        "ssh placement: read_file + list_files routed to the target with the D7 block, and a Home-native root that does not route",
        _ssh_read_and_list,
        serial=True,
    ),
    Scenario(
        "ssh_write_and_edit",
        "ssh placement: write_file + edit_text executed on the target under the bound export policy",
        _ssh_write_and_edit,
        serial=True,
    ),
    Scenario(
        "ssh_run_command",
        "ssh placement: a process on the target through the full guard chain, plus a pre-dispatch sudo refusal that never prepares",
        _ssh_run_command,
        serial=True,
    ),
    Scenario(
        "ssh_restricted_subagent",
        "ssh placement: a restricted subagent's secret read refused before prepare, its ordinary listing filtered and disclosed on the target",
        _ssh_restricted_subagent,
        serial=True,
    ),
]
