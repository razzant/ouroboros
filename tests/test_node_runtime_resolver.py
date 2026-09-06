"""Node branch of the process-interpreter resolver (process_interpreters.py).

Ladder contract under test (plan §2.1 + amendments R1-R8):
a healthy PATH node is a byte-identical no-op (argv AND env), a missing or
probe-dead PATH candidate falls back to the bundled runtime (argv rewrite only
for node/nodejs requests; an attested child-env PATH prepend for EVERY
triggered launch, npm-family and ``sh -c`` bodies included), a non-local
executor backend skips the ladder entirely, and no usable runtime is an
honest as-written launch with disclosed probe facts — never a typed pre-block.

Seam placement is itself part of the contract: the node health check is an
EXECUTION probe of an argv[0]-steered candidate, so it runs only AFTER the
dispatch gates (light fence / shell guard / safety) have approved the call —
a planted PATH shim named ``node`` must never execute on a refused call.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Any

import pytest

import ouroboros.process_interpreters as resolver
from ouroboros.platform_layer import PATH_SEP, NodeRuntimeHealth
from ouroboros.process_interpreters import (
    InterpreterResolutionTrace,
    apply_env_path_prepend,
    interpreter_path_overlay,
    record_interpreter_resolution,
    resolve_process_node,
)
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.tools.shell_guards import interpreter_family

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX stub executables drive the ladder"
)


@pytest.fixture(autouse=True)
def _isolated_node_health_memo():
    """T18: the probe memo is a module-level registry — reset it around every
    test so no verdict leaks between tests on the same xdist worker."""
    from ouroboros import node_runtime as _nr

    saved = dict(_nr._NODE_HEALTH_MEMO)
    _nr._NODE_HEALTH_MEMO.clear()
    try:
        yield
    finally:
        _nr._NODE_HEALTH_MEMO.clear()
        _nr._NODE_HEALTH_MEMO.update(saved)



def _stub(path: pathlib.Path, body: str) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}", encoding="utf-8")
    path.chmod(0o755)
    return path


def _healthy_stub(path: pathlib.Path) -> pathlib.Path:
    return _stub(path, "echo v24.16.0\n")


def _context(tmp_path: pathlib.Path) -> ToolContext:
    repo = tmp_path / "system_repo"
    data = tmp_path / "data"
    repo.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    return ToolContext(
        repo_dir=repo,
        system_repo_dir=repo,
        drive_root=data,
        task_id="node-resolver-test",
    )


def _tool_args(tool_name: str, *, token: str = "node") -> dict:
    if tool_name == "run_command":
        return {"cmd": [token, "--version"]}
    if tool_name == "run_script":
        return {"script": "console.log('ok')", "interpreter": token}
    if tool_name == "start_service":
        return {"name": "svc", "cmd": [token, "server.js"]}
    if tool_name == "verify_and_record":
        return {"contract_kind": "explicit_command", "check": [token, "--version"]}
    raise AssertionError(tool_name)


@pytest.fixture()
def quiet_bootstrap(monkeypatch):
    """Deterministic PATH: the resolver's idempotent bootstrap becomes a no-op
    so monkeypatched PATH is exactly what the ladder probes."""
    monkeypatch.setattr(resolver, "bootstrap_process_path", lambda: [])


@pytest.mark.parametrize(
    "tool_name", ["run_command", "run_script", "start_service", "verify_and_record"]
)
def test_healthy_path_node_is_byte_identical_noop(tmp_path, monkeypatch, quiet_bootstrap, tool_name):
    bin_dir = tmp_path / "bin"
    node = _healthy_stub(bin_dir / "node")
    monkeypatch.setenv("PATH", str(bin_dir))
    ctx = _context(tmp_path)
    args = _tool_args(tool_name)

    resolved, trace = resolve_process_node(ctx, tool_name, args, runtime_mode="advanced")

    assert resolved == args  # argv byte-identical
    assert trace is not None
    assert trace.family == "node"
    assert trace.reason == "path_node_healthy"
    assert not trace.changed
    assert trace.env_path_prepend == ""
    assert trace.runtime_path == str(node)
    assert trace.runtime_version == "24.16.0"
    assert trace.path_snapshot == str(bin_dir)
    # env byte-identical: no overlay, inherit-env stays inherit (None).
    assert interpreter_path_overlay(trace) is None
    assert apply_env_path_prepend(None, trace) is None


@pytest.mark.serial
def test_broken_path_node_falls_back_to_bundled_with_rewrite_and_prepend(
    tmp_path, monkeypatch, quiet_bootstrap,
):
    """A PATH node the kernel kills on launch (the incident class) loses to the
    healthy bundled runtime: argv[0] rewritten, bundled dir attested as prepend."""
    bin_dir = tmp_path / "bin"
    dead = _stub(bin_dir / "node", "kill -9 $$\n")
    bundled = _healthy_stub(tmp_path / "bundle" / "node-standalone" / "bin" / "node")
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setattr(resolver, "resolve_bundled_node", lambda: str(bundled))
    ctx = _context(tmp_path)

    resolved, trace = resolve_process_node(
        ctx, "run_command", {"cmd": ["node", "app.js"]}, runtime_mode="advanced"
    )

    assert resolved["cmd"] == [str(bundled), "app.js"]
    assert trace is not None and trace.reason == "bundled_node_fallback"
    assert trace.changed
    assert trace.fallback_reason == f"path_node_broken:signal:SIGKILL:{dead}"
    assert trace.env_path_prepend == str(bundled.parent)
    assert trace.runtime_path == str(bundled)
    assert trace.runtime_version == "24.16.0"
    overlay = interpreter_path_overlay(trace)
    assert overlay == {"PATH": f"{bundled.parent}{PATH_SEP}{bin_dir}"}


def test_missing_path_node_falls_back_to_bundled(tmp_path, monkeypatch, quiet_bootstrap):
    empty = tmp_path / "empty"
    empty.mkdir()
    bundled = _healthy_stub(tmp_path / "bundle" / "node-standalone" / "bin" / "node")
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(resolver, "resolve_bundled_node", lambda: str(bundled))
    ctx = _context(tmp_path)

    resolved, trace = resolve_process_node(
        ctx, "run_command", {"cmd": ["node", "--version"]}, runtime_mode="advanced"
    )

    assert resolved["cmd"][0] == str(bundled)
    assert trace is not None and trace.reason == "bundled_node_fallback"
    assert trace.fallback_reason == "path_node_missing:node"


def test_no_usable_node_is_noop_with_disclosed_facts(tmp_path, monkeypatch, quiet_bootstrap):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(resolver, "resolve_bundled_node", lambda: None)
    ctx = _context(tmp_path)
    args = {"cmd": ["node", "--version"]}

    resolved, trace = resolve_process_node(ctx, "run_command", args, runtime_mode="advanced")

    assert resolved == args  # honest as-written launch, no typed pre-block (R8)
    assert trace is not None and trace.reason == "no_usable_node"
    assert not trace.changed and trace.env_path_prepend == ""
    assert trace.fallback_reason == "path_node_missing:node;bundled_node_missing"
    assert trace.error_reason == ""  # never routed into the python fail-closed branch


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        ("run_command", {"cmd": ["/usr/bin/node", "--version"]}),
        ("run_command", {"cmd": ["node20", "--version"]}),
        ("run_command", {"cmd": ["nodemon", "app.js"]}),
        ("run_command", {"cmd": ["sh", "-c", "echo hello"]}),
        ("run_script", {"script": "x", "interpreter": "/opt/node/bin/node"}),
        ("run_script", {"script": "x", "interpreter": "python3"}),
        ("start_service", {"name": "svc", "cmd": ["node18", "server.js"]}),
        ("verify_and_record", {"contract_kind": "artifact_observation", "check": ["node", "-v"]}),
        ("remote_exec", {"cmd": ["node", "-v"]}),
    ],
)
def test_noneligible_invocations_are_byte_for_byte_unchanged(
    tmp_path, monkeypatch, quiet_bootstrap, tool_name, args
):
    """Explicit absolute paths and versioned names are never touched (bug-report
    requirement #5); lookalikes and non-node shells do not trigger the ladder."""
    monkeypatch.setattr(
        resolver, "node_runtime_health",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("probe must not run")),
    )
    ctx = _context(tmp_path)

    resolved, trace = resolve_process_node(ctx, tool_name, args, runtime_mode="advanced")

    assert resolved == args
    assert trace is None


def test_windows_launcher_suffixes_normalize_for_token_match(
    tmp_path, monkeypatch, quiet_bootstrap,
):
    """On Windows node.exe rewrites like node and NPM.CMD triggers the family
    prepend (R7); on POSIX the same spellings stay unclassified (T9: exec is
    case-sensitive there and launcher suffixes are a Windows convention)."""
    empty = tmp_path / "empty"
    empty.mkdir()
    bundled = _healthy_stub(tmp_path / "bundle" / "node-standalone" / "bin" / "node")
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(resolver, "resolve_bundled_node", lambda: str(bundled))
    ctx = _context(tmp_path)

    monkeypatch.setattr(resolver, "IS_WINDOWS", False)
    posix_args = {"cmd": ["node.exe", "app.js"]}
    unchanged, posix_trace = resolve_process_node(
        ctx, "run_command", posix_args, runtime_mode="advanced"
    )
    assert unchanged == posix_args and posix_trace is None

    monkeypatch.setattr(resolver, "IS_WINDOWS", True)

    rewritten, exe_trace = resolve_process_node(
        ctx, "run_command", {"cmd": ["node.exe", "app.js"]}, runtime_mode="advanced"
    )
    assert rewritten["cmd"][0] == str(bundled)
    assert exe_trace is not None and exe_trace.reason == "bundled_node_fallback"

    family_args = {"cmd": ["NPM.CMD", "ci"]}
    unchanged, npm_trace = resolve_process_node(
        ctx, "run_command", family_args, runtime_mode="advanced"
    )
    assert unchanged == family_args  # family tools are never rewritten
    assert npm_trace is not None and npm_trace.env_path_prepend == str(bundled.parent)


def test_npm_family_gets_prepend_without_rewrite(tmp_path, monkeypatch, quiet_bootstrap):
    empty = tmp_path / "empty"
    empty.mkdir()
    bundled = _healthy_stub(tmp_path / "bundle" / "node-standalone" / "bin" / "node")
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(resolver, "resolve_bundled_node", lambda: str(bundled))
    ctx = _context(tmp_path)
    args = {"cmd": ["npm", "ci"]}

    resolved, trace = resolve_process_node(ctx, "run_command", args, runtime_mode="advanced")

    assert resolved == args
    assert trace is not None and trace.reason == "bundled_node_fallback"
    assert not trace.changed
    assert trace.env_path_prepend == str(bundled.parent)
    env = apply_env_path_prepend({"PATH": "ignored-base", "HOME": "/h"}, trace)
    # The prepend rebuilds PATH from the resolver's FROZEN snapshot.
    assert env == {"PATH": f"{bundled.parent}{PATH_SEP}{empty}", "HOME": "/h"}


def test_sh_dash_c_body_triggers_prepend_only(tmp_path, monkeypatch, quiet_bootstrap):
    empty = tmp_path / "empty"
    empty.mkdir()
    bundled = _healthy_stub(tmp_path / "bundle" / "node-standalone" / "bin" / "node")
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(resolver, "resolve_bundled_node", lambda: str(bundled))
    ctx = _context(tmp_path)
    args = {"cmd": ["sh", "-c", "npm ci && node app.js"]}

    resolved, trace = resolve_process_node(ctx, "run_command", args, runtime_mode="advanced")

    assert resolved == args  # the shell wrapper argv is never rewritten
    assert trace is not None and trace.reason == "bundled_node_fallback"
    assert trace.env_path_prepend == str(bundled.parent)


def test_run_script_shell_body_triggers_prepend(tmp_path, monkeypatch, quiet_bootstrap):
    empty = tmp_path / "empty"
    empty.mkdir()
    bundled = _healthy_stub(tmp_path / "bundle" / "node-standalone" / "bin" / "node")
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(resolver, "resolve_bundled_node", lambda: str(bundled))
    ctx = _context(tmp_path)
    args = {"script": "corepack enable\nyarn install\n", "interpreter": "bash"}

    resolved, trace = resolve_process_node(ctx, "run_script", args, runtime_mode="advanced")

    assert resolved == args
    assert trace is not None and trace.env_path_prepend == str(bundled.parent)


def test_string_check_shell_body_triggers_prepend(tmp_path, monkeypatch, quiet_bootstrap):
    """A string verify check normalizes to ["sh","-c",text]; its body is scanned."""
    empty = tmp_path / "empty"
    empty.mkdir()
    bundled = _healthy_stub(tmp_path / "bundle" / "node-standalone" / "bin" / "node")
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(resolver, "resolve_bundled_node", lambda: str(bundled))
    ctx = _context(tmp_path)
    args = {"contract_kind": "explicit_command", "check": "npx --yes jest"}

    resolved, trace = resolve_process_node(ctx, "verify_and_record", args, runtime_mode="advanced")

    assert resolved == args
    assert trace is not None and trace.env_path_prepend == str(bundled.parent)


def test_docker_executor_skips_ladder_without_probing(tmp_path, monkeypatch, quiet_bootstrap):
    """R2/Q2-3: a non-local backend resolves node in its own filesystem — no
    host probe runs and no host path can leak into the container argv."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ctx = _context(tmp_path)
    ctx.workspace_root = workspace
    ctx.workspace_mode = "external"
    ctx.executor_ref = {
        "type": "docker_exec",
        "id": "bench",
        "container_name": "bench",
        "network": "none",
        "workspace_host_path": str(workspace),
        "workspace_backend_path": "/workspace",
    }
    monkeypatch.setattr(
        resolver, "node_runtime_health",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("probe must not run")),
    )
    args = {"cmd": ["node", "app.js"]}

    resolved, trace = resolve_process_node(ctx, "run_command", args, runtime_mode="advanced")

    assert resolved == args
    assert trace is not None and trace.reason == "executor_backend_node"
    assert trace.environment == "backend_path"
    assert trace.env_path_prepend == ""


def test_local_executor_continues_ladder(tmp_path, monkeypatch, quiet_bootstrap):
    """A local executor runs on THIS host, so skipping the ladder would keep the
    broken-PATH bug alive there (amendment R2)."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bin_dir = tmp_path / "bin"
    node = _healthy_stub(bin_dir / "node")
    monkeypatch.setenv("PATH", str(bin_dir))
    ctx = _context(tmp_path)
    ctx.workspace_root = workspace
    ctx.workspace_mode = "external"
    ctx.executor_ref = {
        "type": "local",
        "id": "local",
        "workspace_host_path": str(workspace),
        "workspace_backend_path": "/workspace",
    }

    resolved, trace = resolve_process_node(
        ctx, "run_command", {"cmd": ["node", "app.js"]}, runtime_mode="advanced"
    )

    assert trace is not None and trace.reason == "path_node_healthy"
    assert trace.runtime_path == str(node)


def test_verify_check_args_never_clobbered_by_rewrite(tmp_path, monkeypatch, quiet_bootstrap):
    """R4: the check text is the receipt's identity — the substitution lives in
    the trace (resolved_interpreter) and reaches execution via the attestation."""
    empty = tmp_path / "empty"
    empty.mkdir()
    bundled = _healthy_stub(tmp_path / "bundle" / "node-standalone" / "bin" / "node")
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(resolver, "resolve_bundled_node", lambda: str(bundled))
    ctx = _context(tmp_path)
    args = {"contract_kind": "explicit_command", "check": ["node", "--test"]}

    resolved, trace = resolve_process_node(ctx, "verify_and_record", args, runtime_mode="advanced")

    assert resolved == args  # args["check"] untouched
    assert trace is not None and trace.changed
    assert trace.resolved_interpreter == str(bundled)
    assert trace.env_path_prepend == str(bundled.parent)


def test_apply_env_path_prepend_windows_path_key_casing(monkeypatch):
    monkeypatch.setattr(resolver, "IS_WINDOWS", True)
    trace = InterpreterResolutionTrace(
        tool="run_command",
        requested_interpreter="node",
        resolved_interpreter="C:\\bundle\\node.exe",
        surface="system_repo",
        environment="bundled_node",
        reason="bundled_node_fallback",
        family="node",
        path_snapshot="C:\\base",
        env_path_prepend="C:\\bundle",
    )

    env = apply_env_path_prepend({"Path": "stale", "HOME": "h"}, trace)

    assert env is not None
    assert "Path" not in env  # no case-variant duplicate for CreateProcess
    assert env["PATH"] == f"C:\\bundle{PATH_SEP}C:\\base"
    assert env["HOME"] == "h"


def test_recorder_writes_family_specific_event_types(tmp_path):
    ctx = _context(tmp_path)
    base = dict(
        tool="run_command",
        requested_interpreter="node",
        resolved_interpreter="node",
        surface="system_repo",
        environment="host_path",
        reason="path_node_healthy",
    )
    record_interpreter_resolution(
        ctx, InterpreterResolutionTrace(family="node", path_snapshot="/bin", **base)
    )
    record_interpreter_resolution(
        ctx,
        InterpreterResolutionTrace(
            **{**base, "requested_interpreter": "python", "resolved_interpreter": "python",
               "environment": "ouroboros_agent", "reason": "agent_python"},
        ),
    )

    lines = (ctx.drive_logs() / "events.jsonl").read_text(encoding="utf-8").splitlines()
    node_event, python_event = (json.loads(line) for line in lines[-2:])
    assert node_event["type"] == "node_runtime_resolution"
    assert node_event["family"] == "node"
    assert node_event["path_snapshot"] == "/bin"
    assert python_event["type"] == "python_interpreter_resolution"
    # The historic python event payload gains no generalization keys.
    assert "family" not in python_event
    assert "path_snapshot" not in python_event
    assert "env_path_prepend" not in python_event


# ---- registry seam: gates first, then the probe; guard/handler family parity ----


def _advanced_registry(tmp_path, monkeypatch) -> tuple[ToolRegistry, ToolContext]:
    ctx = _context(tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    # Campaign owner: the safety check is a registry_guard_process module
    # function whose no-block value is None (upstream's method returned "").
    monkeypatch.setattr(
        "ouroboros.tools.registry_guard_process._run_shell_safety_check",
        lambda *a, **k: None,
    )
    return registry, ctx


def _mock_health(monkeypatch, verdicts: dict[str, NodeRuntimeHealth]) -> None:
    def fake_health(path: str, timeout_sec: float = 10) -> NodeRuntimeHealth:
        return verdicts[str(path)]

    monkeypatch.setattr(resolver, "node_runtime_health", fake_health)


def test_registry_guard_sees_family_stable_argv_and_handler_gets_rewrite(
    tmp_path, monkeypatch, quiet_bootstrap,
):
    """The guard inspects the ORIGINAL bare argv (the node step runs post-gates);
    the handler executes the resolver's substitution, which classifies into the
    SAME interpreter family — the disclosed guard/handler delta contract."""
    # Campaign call site: registry_core reads the guard-args builder through
    # the shell_guards module (upstream read it off the registry facade).
    import ouroboros.tools.shell_guards as registry_module

    bin_dir = tmp_path / "bin"
    dead = _stub(bin_dir / "node", "exit 1\n")
    bundled = _healthy_stub(tmp_path / "bundle" / "node-standalone" / "bin" / "node")
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setattr(resolver, "resolve_bundled_node", lambda: str(bundled))
    _mock_health(monkeypatch, {
        str(dead): NodeRuntimeHealth(status="broken", reason="signal:SIGKILL", path=str(dead)),
        str(bundled): NodeRuntimeHealth(status="healthy", version="24.16.0", path=str(bundled)),
    })
    registry, ctx = _advanced_registry(tmp_path, monkeypatch)
    captured: dict[str, list[str]] = {}
    original_guard = registry_module.process_shell_guard_args

    def capture_guard(name, args, **kwargs):
        guarded = original_guard(name, args, **kwargs)
        captured["guard"] = list(guarded["cmd"])
        return guarded

    def capture_handler(_ctx, cmd, _resolved_binding=None, **_kwargs):
        captured["handler"] = list(cmd)
        captured["attested"] = getattr(_ctx, "_active_interpreter_resolution", None)
        return "ok"

    monkeypatch.setattr(registry_module, "process_shell_guard_args", capture_guard)
    import dataclasses
    registry._entries["run_command"] = dataclasses.replace(
        registry._entries["run_command"], handler=capture_handler,
    )

    result = registry.execute("run_command", {"cmd": ["node", "--version"]})

    assert result == "ok"
    assert captured["guard"] == ["node", "--version"]
    assert captured["handler"] == [str(bundled), "--version"]
    assert interpreter_family(captured["guard"][0]) == "node"
    assert interpreter_family(captured["handler"][0]) == "node"
    attested = captured["attested"]
    assert isinstance(attested, InterpreterResolutionTrace)
    assert attested.family == "node" and attested.reason == "bundled_node_fallback"
    assert not hasattr(ctx, "_active_interpreter_resolution")
    event = json.loads(
        (ctx.drive_logs() / "events.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    assert event["type"] == "node_runtime_resolution"
    assert event["env_path_prepend"] == str(bundled.parent)


def test_registry_healthy_path_handler_argv_is_untouched(tmp_path, monkeypatch, quiet_bootstrap):
    bin_dir = tmp_path / "bin"
    node = _healthy_stub(bin_dir / "node")
    monkeypatch.setenv("PATH", str(bin_dir))
    _mock_health(monkeypatch, {
        str(node): NodeRuntimeHealth(status="healthy", version="24.16.0", path=str(node)),
    })
    registry, ctx = _advanced_registry(tmp_path, monkeypatch)
    observed: dict[str, Any] = {}

    def handler(_ctx, cmd, _resolved_binding=None, **_kwargs):
        observed["cmd"] = list(cmd)
        observed["attested"] = getattr(_ctx, "_active_interpreter_resolution", None)
        return "ok"

    import dataclasses
    registry._entries["run_command"] = dataclasses.replace(
        registry._entries["run_command"], handler=handler,
    )

    assert registry.execute("run_command", {"cmd": ["node", "--version"]}) == "ok"
    assert observed["cmd"] == ["node", "--version"]
    attested = observed["attested"]
    assert attested is not None and attested.reason == "path_node_healthy"
    assert attested.env_path_prepend == ""


@pytest.mark.serial
def test_light_fence_refuses_before_the_node_probe_can_execute(tmp_path, monkeypatch):
    """Seam-order pin for the probe hazard: a planted PATH shim named ``node``
    must not execute AT ALL (not even as ``node --version``) when the light
    fence refuses the call — pre-guard resolution would have run its payload."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    repo = tmp_path / "repo"
    repo.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=tmp_path / "drive")
    registry._ctx.task_id = "t-node-order"
    marker = tmp_path / "shim_executed"
    bin_dir = tmp_path / "bin"
    _stub(bin_dir / "node", f"printf ran > '{marker}'\n")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    result = registry.execute("run_command", {
        "cmd": ["node", f"--eval=require('node:fs').writeFileSync('{repo}/x.py','x')"],
    })

    assert "LIGHT_MODE_BLOCKED" in result, result[:300]
    assert not marker.exists(), "the node health probe executed a refused call's argv[0]"


# ---- handler env application ----


def test_run_shell_applies_attested_prepend_and_healthy_env_is_untouched(
    tmp_path, monkeypatch,
):
    import ouroboros.tools.shell as shell

    ctx = _context(tmp_path)
    seen: dict[str, Any] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        seen["env"] = kwargs.get("env")

        class _Res:
            returncode = 0
            stdout = "ok"
            stderr = ""
            args = cmd

        return _Res()

    monkeypatch.setattr(shell, "_tracked_subprocess_run", fake_run)
    prepend_dir = str(tmp_path / "bundle" / "bin")
    ctx._active_interpreter_resolution = InterpreterResolutionTrace(
        tool="run_command",
        requested_interpreter="npm",
        resolved_interpreter="npm",
        surface="system_repo",
        environment="bundled_node",
        reason="bundled_node_fallback",
        family="node",
        path_snapshot="/frozen",
        env_path_prepend=prepend_dir,
    )
    result = shell._run_shell(ctx, ["npm", "--version"], cwd="system_repo")
    assert "exit_code=0" in result
    assert seen["env"] is not None
    assert seen["env"]["PATH"] == f"{prepend_dir}{PATH_SEP}/frozen"

    del ctx._active_interpreter_resolution
    seen.clear()
    result = shell._run_shell(ctx, ["npm", "--version"], cwd="system_repo")
    assert "exit_code=0" in result
    assert seen["env"] is None  # in-repo cwd keeps today's inherit-env behavior


def test_verify_executes_resolved_argv_but_receipt_keeps_original_check(
    tmp_path, monkeypatch,
):
    from ouroboros.outcomes import verification_receipts_path
    from ouroboros.tools.verify import _verify_and_record

    ctx = _context(tmp_path)
    bundled = str(tmp_path / "bundle" / "node-standalone" / "bin" / "node")
    seen: dict[str, Any] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        seen["env"] = kwargs.get("env")

        class _Res:
            returncode = 0
            stdout = "v24.16.0"
            stderr = ""
            args = cmd

        return _Res()

    import ouroboros.tools.shell as shell

    monkeypatch.setattr(shell, "_tracked_subprocess_run", fake_run)
    ctx._active_interpreter_resolution = InterpreterResolutionTrace(
        tool="verify_and_record",
        requested_interpreter="node",
        resolved_interpreter=bundled,
        surface="system_repo",
        environment="bundled_node",
        reason="bundled_node_fallback",
        family="node",
        path_snapshot="/frozen",
        env_path_prepend=str(pathlib.Path(bundled).parent),
    )

    result = _verify_and_record(
        ctx,
        contract_kind="explicit_command",
        check=["node", "--version"],
        expected="v24",
    )

    assert "PASS" in result
    assert seen["cmd"] == [bundled, "--version"]
    assert seen["env"]["PATH"].startswith(f"{pathlib.Path(bundled).parent}{PATH_SEP}")
    receipts = verification_receipts_path(ctx.drive_root, "node-resolver-test")
    receipt = json.loads(receipts.read_text(encoding="utf-8").splitlines()[-1])
    # R4: the receipt's identity is the ORIGINAL check text, not the rewrite.
    assert receipt["check"] == "node --version"


def test_run_script_accepts_attested_bundled_node_and_allows_node_exe(
    tmp_path, monkeypatch,
):
    """The allowlist matches the interpreter BASENAME (base contract), so the
    bundled ``.../bin/node`` passes on its own; the generalized attestation is
    what admits a verified node resolution whose basename is NOT allowlisted."""
    import ouroboros.tools.shell as shell

    ctx = _context(tmp_path)
    monkeypatch.setattr(shell, "_run_shell", lambda *_a, **_k: "ok")
    bundled = str(_healthy_stub(tmp_path / "bundle" / "node-standalone" / "bin" / "node"))
    ctx._active_interpreter_resolution = InterpreterResolutionTrace(
        tool="run_script",
        requested_interpreter="node",
        resolved_interpreter=bundled,
        surface="system_repo",
        environment="bundled_node",
        reason="bundled_node_fallback",
        family="node",
        env_path_prepend=str(pathlib.Path(bundled).parent),
    )
    attested = shell._run_script(ctx, "console.log(1)", interpreter=bundled)
    assert "RUN_SCRIPT_BLOCKED" not in attested

    del ctx._active_interpreter_resolution
    plain = shell._run_script(ctx, "console.log(1)", interpreter="node.exe")
    assert "RUN_SCRIPT_BLOCKED" not in plain

    odd_basename = str(tmp_path / "opt" / "bundle" / "node24")
    blocked = shell._run_script(ctx, "console.log(1)", interpreter=odd_basename)
    assert blocked.startswith("⚠️ RUN_SCRIPT_BLOCKED:")
    ctx._active_interpreter_resolution = InterpreterResolutionTrace(
        tool="run_script",
        requested_interpreter="node",
        resolved_interpreter=odd_basename,
        surface="system_repo",
        environment="bundled_node",
        reason="bundled_node_fallback",
        family="node",
    )
    admitted = shell._run_script(ctx, "console.log(1)", interpreter=odd_basename)
    assert "RUN_SCRIPT_BLOCKED" not in admitted


@pytest.mark.serial
def test_workspace_executor_local_applies_env_overlay(tmp_path):
    from ouroboros.workspace_executor import ExecutorRef, PathMapping, _execute_local

    executor = ExecutorRef(
        kind="local",
        executor_id="t",
        network="host",
        mappings=(PathMapping(host_path=tmp_path, backend_path="/workspace"),),
    )
    argv = [
        sys.executable,
        "-c",
        "import os; print(os.environ.get('OURO_NODE_TEST', 'missing'))",
    ]

    plain = _execute_local(executor, argv, tmp_path, 30, drive_root=None)
    overlaid = _execute_local(
        executor, argv, tmp_path, 30, drive_root=None,
        env_overlay={"OURO_NODE_TEST": "prepended"},
    )

    assert plain.returncode == 0 and plain.stdout.strip() == "missing"
    assert overlaid.returncode == 0 and overlaid.stdout.strip() == "prepended"


# ---- rename hygiene ----


def test_no_stale_python_interpreter_module_references():
    """The module moved to process_interpreters.py with no compatibility shim;
    a stale import would crash at runtime on the next release."""
    import importlib.util

    # Built dynamically so this test's own source never matches its scan.
    stale_module = "python" + "_interpreter"
    stale_dotted = f"ouroboros.{stale_module}"
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    assert importlib.util.find_spec("ouroboros.process_interpreters") is not None
    assert not (repo_root / "ouroboros" / f"{stale_module}.py").exists()
    # An editable install registers a meta-path finder for its OWN checkout, so
    # in a dev environment `find_spec` can resurrect the old name from a
    # different tree; only a spec originating in THIS tree is a rename failure.
    stale = importlib.util.find_spec(stale_dotted)
    assert stale is None or not str(stale.origin or "").startswith(str(repo_root))
    offenders: list[str] = []
    for base in ("ouroboros", "supervisor", "tests"):
        for path in sorted((repo_root / base).rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            if stale_dotted in text or f"from ouroboros import {stale_module}" in text:
                offenders.append(str(path.relative_to(repo_root)))
    server = repo_root / "server.py"
    if server.is_file() and stale_dotted in server.read_text(encoding="utf-8", errors="replace"):
        offenders.append("server.py")
    assert offenders == []


def test_run_script_family_tokens_stay_blocked_on_healthy_path(tmp_path, monkeypatch):
    """A-F1 pin: a HEALTHY-path family trace (changed=False) must not widen the
    run_script interpreter allowlist — bare npm/npx/pnpm/yarn/corepack/nodejs
    keep the base RUN_SCRIPT_BLOCKED refusal; only an actual substitution (the
    emergency bundled rewrite, changed=True) earns attestation."""
    from types import SimpleNamespace

    from ouroboros.process_interpreters import InterpreterResolutionTrace
    from ouroboros.tools.shell import _run_script

    ctx = SimpleNamespace(
        repo_dir=tmp_path, drive_root=tmp_path,
        drive_logs=lambda: pathlib.Path(str(tmp_path)),
    )
    script = tmp_path / "s.js"
    script.write_text("console.log(1)\n", encoding="utf-8")
    for spelling in ("npm", "npx", "pnpm", "yarn", "corepack", "nodejs"):
        ctx._active_interpreter_resolution = InterpreterResolutionTrace(
            tool="run_script",
            requested_interpreter=spelling,
            resolved_interpreter=spelling,
            surface="external_workspace",
            # A VERIFIED healthy-path form: (path_node_healthy, host_path) is in
            # _VERIFIED_RESOLUTIONS, so this pin fails on the changed-gate, not
            # trivially on verified=False (T6).
            environment="host_path",
            reason="path_node_healthy",
            family="node",
        )
        assert ctx._active_interpreter_resolution.verified
        assert not ctx._active_interpreter_resolution.changed
        out = _run_script(ctx, str(script), interpreter=spelling)
        assert out.startswith("⚠️ RUN_SCRIPT_BLOCKED"), (spelling, out[:120])


def test_registry_bridges_resolved_runtime_slot_for_observability(tmp_path):
    """Synthesis pin (streams A+B): when the node trace records a substitution
    (argv rewrite or emergency prepend), `_invoke_builtin_handler` publishes the
    ONE string slot ``ctx._process_resolved_runtime`` for the duration of the
    handler call — the slot the typed process facts and the verify receipt
    disclose — and restores it afterwards; a no-op trace publishes nothing."""
    from types import SimpleNamespace

    from ouroboros.process_interpreters import InterpreterResolutionTrace
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry.__new__(ToolRegistry)
    registry._ctx = SimpleNamespace()
    seen = {}

    def handler(ctx, **kwargs):
        seen["slot"] = getattr(ctx, "_process_resolved_runtime", None)
        return "ok"

    entry = SimpleNamespace(handler=handler)
    changed = InterpreterResolutionTrace(
        tool="run_command", requested_interpreter="node",
        resolved_interpreter="/bundle/bin/node", surface="external_workspace",
        environment="bundled_node", reason="bundled_node_fallback", family="node",
    )
    err, result = registry._invoke_builtin_handler(
        "run_command", entry, {}, None, changed, None)
    assert err is None and result == "ok"
    assert seen["slot"] == "/bundle/bin/node"
    assert not hasattr(registry._ctx, "_process_resolved_runtime")

    healthy = InterpreterResolutionTrace(
        tool="run_command", requested_interpreter="node",
        resolved_interpreter="node", surface="external_workspace",
        environment="target_path", reason="path_node_healthy", family="node",
    )
    err, result = registry._invoke_builtin_handler(
        "run_command", entry, {}, None, healthy, None)
    assert err is None and seen["slot"] is None
    assert not hasattr(registry._ctx, "_process_resolved_runtime")


def test_run_script_schema_enum_is_subset_of_validator_allowlist():
    """T5 pin: every advertised interpreter enum option must pass the actual
    _run_script allowlist (the schema is advisory for the model; the allowlist
    is the validator). Windows launcher spellings (python.exe/node.exe) are
    accepted synonyms deliberately NOT advertised in the enum."""
    from ouroboros.tools import shell as shell_mod

    entry = next(e for e in shell_mod.get_tools() if e.name == "run_script")
    enum = entry.schema["parameters"]["properties"]["interpreter"]["enum"]
    assert set(enum) <= shell_mod.RUN_SCRIPT_INTERPRETER_ALLOWLIST
    assert {"node", "python3"} <= set(enum)
    # Launcher spellings are accepted by the validator but deliberately NOT
    # advertised (delta finding D2-7): re-adding one here must fail this pin.
    assert "node.exe" not in enum and "python.exe" not in enum


def test_whitespace_padded_head_is_not_classified(tmp_path, monkeypatch, quiet_bootstrap):
    """T8 pin: ' node ' must NOT produce a node trace — a padded head is run
    as written, so no attestation may claim a substituted runtime for it."""
    empty = tmp_path / "empty"
    empty.mkdir()
    bundled = _healthy_stub(tmp_path / "bundle" / "node-standalone" / "bin" / "node")
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(resolver, "resolve_bundled_node", lambda: str(bundled))
    ctx = _context(tmp_path)
    for tool, args in (
        ("run_command", {"cmd": [" node ", "--version"]}),
        ("start_service", {"cmd": [" node ", "server.js"]}),
        # explicit_command IS in _VERIFY_RUN_KINDS — the padded head must be
        # the ONLY reason this stays unclassified (delta finding D2-3).
        ("verify_and_record", {"contract_kind": "explicit_command", "check": [" node ", "--version"]}),
    ):
        resolved, trace = resolve_process_node(ctx, tool, args, runtime_mode="advanced")
        assert resolved == args and trace is None, tool
    # Positive control: the same verify kind with an UNPADDED head classifies,
    # proving the run-kind gate above is actually open for these cases.
    control = {"contract_kind": "explicit_command", "check": ["node", "--version"]}
    _resolved, control_trace = resolve_process_node(
        ctx, "verify_and_record", control, runtime_mode="advanced"
    )
    assert control_trace is not None


def test_relative_path_which_result_is_a_noop(tmp_path, monkeypatch, quiet_bootstrap):
    """T10 pin: a which() hit through a RELATIVE PATH entry is unprovable from
    the worker process (exec resolves it against the command cwd instead), so
    the resolver must run as written — never substitute bundled node."""
    bundled = _healthy_stub(tmp_path / "bundle" / "node-standalone" / "bin" / "node")
    monkeypatch.setattr(resolver, "resolve_bundled_node", lambda: str(bundled))
    monkeypatch.setattr(resolver.shutil, "which", lambda tok: "bin/node")
    ctx = _context(tmp_path)
    args = {"cmd": ["node", "app.js"]}
    resolved, trace = resolve_process_node(ctx, "run_command", args, runtime_mode="advanced")
    assert resolved == args
    assert trace is not None
    assert trace.reason == "path_node_relative_entry_unprovable"
    assert trace.env_path_prepend in ("", None)


def test_wrapper_matching_covers_abs_paths_and_zsh(tmp_path, monkeypatch, quiet_bootstrap):
    """F-1 pin: /bin/sh -c and zsh -c bodies naming node-family tools trigger
    the family prepend exactly like bare sh; a wrapper hit only rides the env
    prepend (argv untouched)."""
    empty = tmp_path / "empty"
    empty.mkdir()
    bundled = _healthy_stub(tmp_path / "bundle" / "node-standalone" / "bin" / "node")
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(resolver, "resolve_bundled_node", lambda: str(bundled))
    ctx = _context(tmp_path)
    for head in ("/bin/sh", "zsh", "dash"):
        args = {"cmd": [head, "-c", "npm ci"]}
        resolved, trace = resolve_process_node(ctx, "run_command", args, runtime_mode="advanced")
        assert resolved == args, head
        assert trace is not None and trace.env_path_prepend == str(bundled.parent), head
    # A non-wrapper absolute head with a node body stays unclassified.
    plain = {"cmd": ["/usr/bin/env", "node", "app.js"]}
    _r, none_trace = resolve_process_node(ctx, "run_command", plain, runtime_mode="advanced")
    assert none_trace is None
