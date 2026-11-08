"""Surface-aware interpreter/runtime selection for host process tools.

Only the four public process launch surfaces opt into these resolvers, and
the two families run at DIFFERENT points of the dispatch pipeline: Python
resolves pre-dispatch (never executes a candidate, so guards and handler see
the same argv), while Node resolves post-gates (its health probe EXECUTES the
candidate — see ``resolve_process_node``), so the deterministic guards always
inspect the original bare argv and the handler sees the attested rewrite.
Launchers must not rewrite the interpreter afterwards.

Two families share one trace/attestation contract:

* Python (``resolve_process_python``): the original five-step ladder ending in
  a typed pre-block when no interpreter can be proven.
* Node (``resolve_process_node``): PATH-first with an execution health probe
  (``shutil.which`` proves only that a file exists — the incident class is a
  PATH node the kernel SIGKILLs on launch), falling back to the bundled signed
  runtime only when the PATH candidate is missing or probe-dead.  Node never
  pre-blocks: with no usable runtime the argv runs as written and fails
  honestly, while the trace discloses the probe facts.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import shutil
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

from ouroboros.contracts.task_constraint import (
    TaskConstraint,
    normalize_task_constraint,
)
from ouroboros.platform_layer import (
    IS_WINDOWS,
    PATH_SEP,
    bootstrap_process_path,
    node_runtime_health,
    project_venv_python,
    resolve_bundled_node,
)
from ouroboros.shell_parse import normalize_check_argv, shell_command_string, shell_tokens
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    build_resolved_resource_binding,
    path_is_relative_to,
)
from ouroboros.utils import append_jsonl, utc_now_iso

_PYTHON_TOKENS = frozenset({"python", "python3"})
# argv[0] spellings the node resolver may REWRITE to the bundled runtime.
_NODE_TOKENS = frozenset({"node", "nodejs"})
# The wider node family that TRIGGERS the emergency PATH prepend (npm & co are
# not shipped in the bundle, so they are never rewritten — their env shebang
# `#!/usr/bin/env node` picks up the prepended runtime instead; a formula with
# a rewritten absolute shebang is a disclosed residual, Q2-1=A).
_NODE_FAMILY_TOKENS = frozenset({"node", "nodejs", "npm", "npx", "pnpm", "yarn", "corepack"})
# Shell wrappers whose ``-c`` body is seam-scanned for family tokens (R3).
_NODE_SHELL_WRAPPERS = frozenset({"sh", "bash", "zsh", "dash"})
_WINDOWS_LAUNCHER_SUFFIXES = (".exe", ".cmd", ".bat")
_PROCESS_TOOLS = frozenset({"run_command", "run_script", "start_service", "verify_and_record"})
# Mirrors the run-kind set of tools/verify.py (`_RUN_KINDS`); keep in sync when
# a new verify run-kind is introduced there.
_VERIFY_RUN_KINDS = frozenset({"visible_verifier", "explicit_command", "explicit_metric"})
_VERIFIED_RESOLUTIONS = frozenset(
    {
        ("reviewed_skill_environment", "isolated_skill"),
        ("executor_backend_python3", "backend_path"),
        ("project_venv", "project_venv"),
        ("agent_python", "ouroboros_agent"),
        # Node resolutions proved by an execution probe (or delegated to a
        # non-local backend, mirroring executor_backend_python3).
        ("executor_backend_node", "backend_path"),
        ("path_node_healthy", "host_path"),
        ("bundled_node_fallback", "bundled_node"),
    }
)


@dataclass(frozen=True)
class InterpreterResolutionTrace:
    """One resolver decision for one process-tool call, any family.

    ``resolved_interpreter`` is the EXECUTION identity: for a no-op resolution
    it equals ``requested_interpreter`` (argv stays byte-identical); when the
    resolver substitutes a runtime it is the absolute path execution uses.  For
    ``verify_and_record`` the node resolver never rewrites ``args["check"]``
    (the receipt keeps the original text as its identity, amendment R4) — the
    handler reads the substitution from this attestation instead.
    """

    tool: str
    requested_interpreter: str
    resolved_interpreter: str
    surface: str
    environment: str
    reason: str
    fallback_reason: str = ""
    error_reason: str = ""
    target_root: str = ""
    target_cwd: str = ""
    target_source: str = ""
    target_skill: str = ""
    family: str = "python"
    # Node-family provenance: the frozen PATH the resolver probed (and the base
    # of any attested child-env prepend), the physically identified executable,
    # its probed version, and the emergency bundled-runtime dir to prepend.
    path_snapshot: str = ""
    env_path_prepend: str = ""
    runtime_path: str = ""
    runtime_version: str = ""

    @property
    def changed(self) -> bool:
        return self.requested_interpreter != self.resolved_interpreter

    @property
    def verified(self) -> bool:
        """Whether the resolver proved the selected interpreter provenance."""

        return (self.reason, self.environment) in _VERIFIED_RESOLUTIONS

    def to_event(self) -> Dict[str, Any]:
        event = {**asdict(self), "changed": self.changed}
        if self.family == "python":
            # The long-standing python event payload stays byte-identical; the
            # generalization fields ride only on non-python families.
            for key in ("family", "path_snapshot", "env_path_prepend", "runtime_path", "runtime_version"):
                event.pop(key, None)
        return event


# Existing isinstance-consumers and tests keep working under the historic name.
PythonResolutionTrace = InterpreterResolutionTrace


def _python_request(tool_name: str, args: Mapping[str, Any]) -> tuple[str, list[str] | None]:
    """Return the exact eligible token and normalized argv, when applicable."""

    if tool_name in {"run_command", "start_service"}:
        raw = args.get("cmd")
        if not isinstance(raw, list) or not raw:
            return "", None
        argv = [str(part) for part in raw]
        requested = str(argv[0]).strip()
        return (requested, argv) if requested in _PYTHON_TOKENS else ("", None)

    if tool_name == "run_script":
        requested = str(args.get("interpreter") or "python3").strip() or "python3"
        return (requested, None) if requested in _PYTHON_TOKENS else ("", None)

    if tool_name == "verify_and_record":
        kind = str(args.get("contract_kind") or "").strip()
        if kind not in _VERIFY_RUN_KINDS:
            return "", None
        argv = normalize_check_argv(args.get("check")) or []
        if not argv:
            return "", None
        requested = str(argv[0]).strip()
        return (requested, argv) if requested in _PYTHON_TOKENS else ("", None)

    return "", None


def _usable_executable(path_text: str) -> str:
    """Validate an interpreter while preserving venv symlink semantics."""

    text = str(path_text or "").strip()
    if not text:
        return ""
    candidate = pathlib.Path(text).expanduser()
    if not candidate.is_absolute():
        located = shutil.which(text)
        if not located:
            return ""
        candidate = pathlib.Path(located)
    try:
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            return ""
    except OSError:
        return ""
    # Do not resolve a venv's python symlink: executing the lexical path is what
    # lets Python discover the adjacent pyvenv.cfg and preserve the environment.
    return os.path.abspath(os.fspath(candidate))


def _reviewed_skill_python(
    ctx: Any,
    binding: ResolvedResourceBinding | None = None,
) -> tuple[str, str]:
    """Return the lifecycle-proven isolated Python for the selected skill.

    A dispatch binding is authoritative and loads exactly its physical payload
    against its canonical state root.  Legacy task metadata is consulted only
    when a non-registry/direct caller supplied no binding.  A skill that declares
    Python packages whose env python exists but cannot run here (``moved``,
    ``other_python``, ``unreadable``) reports ``skill_env_python_unavailable``:
    another interpreter would run it without them, so the launch is refused
    rather than resolved further.  A never-installed (``absent``) env falls
    through as before, keeping authoring and repair runs possible.
    """

    try:
        from ouroboros.marketplace.isolated_deps import python_runtime_binary, read_deps_state
        from ouroboros.skill_dependencies import unusable_python_env
        from ouroboros.skill_loader import find_skill, load_skill
        from ouroboros.skill_readiness import skill_readiness_for_execution

        if binding is not None:
            if binding.root != "skill_payload":
                return "", ""
            drive_root = pathlib.Path(binding.state_drive_root)
            loaded = load_skill(pathlib.Path(binding.base_path), drive_root)
            if loaded is None or loaded.name != binding.skill_name:
                return "", "reviewed_skill_environment_unavailable"
        else:
            metadata = getattr(ctx, "task_metadata", {})
            metadata = metadata if isinstance(metadata, dict) else {}
            skill_name = str(metadata.get("skill") or "").strip()
            if not skill_name:
                return "", ""
            drive_root = pathlib.Path(getattr(ctx, "drive_root"))
            loaded = find_skill(drive_root, skill_name)
        if loaded is None:
            return "", "reviewed_skill_environment_unavailable"
        if unusable_python_env(drive_root, loaded) in {"moved", "other_python", "unreadable"}:
            return "", "skill_env_python_unavailable"
        if not skill_readiness_for_execution(drive_root, loaded).ready:
            return "", "reviewed_skill_environment_unavailable"
        deps_state = read_deps_state(drive_root, loaded.name, loaded.skill_dir)
        if str(deps_state.get("status") or "") != "installed":
            return "", "reviewed_skill_environment_unavailable"
        candidate = python_runtime_binary(loaded.skill_dir)
        usable = _usable_executable(str(candidate or ""))
        if usable:
            return usable, ""
    except Exception:
        return "", "reviewed_skill_environment_probe_failed"
    return "", "reviewed_skill_environment_unavailable"


def _executor_covers_kind(ctx: Any, work_dir: pathlib.Path) -> tuple[bool, str, str]:
    """Whether a configured executor covers ``work_dir``, plus its backend kind."""

    try:
        from ouroboros.workspace_executor import executor_ref_from_ctx, map_host_path

        executor = executor_ref_from_ctx(ctx)
        if executor is None:
            return False, "", ""
        map_host_path(executor, pathlib.Path(work_dir).resolve(strict=False))
        return True, str(executor.kind or ""), ""
    except ValueError:
        return False, "", ""
    except Exception:
        return False, "", "executor_resolution_failed"


def _executor_covers(ctx: Any, work_dir: pathlib.Path) -> tuple[bool, str]:
    covers, _kind, error = _executor_covers_kind(ctx, work_dir)
    return covers, error


def _surface_for(
    ctx: Any,
    binding: ResolvedResourceBinding,
    constraint: Optional[TaskConstraint],
) -> str:
    if binding.root != "active_workspace":
        return binding.root or "unresolved"
    if constraint and constraint.mode == "acting_subagent" and constraint.surface == "self_worktree":
        from ouroboros.workspace_copies import is_system_copy

        return "system_repo" if is_system_copy(ctx) else "external_workspace"
    mode = str(getattr(ctx, "workspace_mode", "") or "").strip().lower()
    if mode in {"external", "external_workspace", "genesis"}:
        return "external_workspace"
    system_repo = pathlib.Path(
        getattr(ctx, "system_repo_dir", None) or getattr(ctx, "repo_dir", binding.target_path)
    ).resolve(strict=False)
    return "system_repo" if path_is_relative_to(binding.target_path, system_repo) else "external_workspace"


def _trace_target(binding: ResolvedResourceBinding) -> Dict[str, str]:
    return {
        "target_root": binding.root,
        "target_cwd": str(binding.target_path),
        "target_source": binding.source,
        "target_skill": binding.skill_name,
    }


def _project_root(ctx: Any, surface: str, work_dir: pathlib.Path) -> pathlib.Path:
    if surface == "external_workspace":
        workspace_root = getattr(ctx, "workspace_root", None)
        if workspace_root:
            candidate = pathlib.Path(workspace_root).resolve(strict=False)
            if path_is_relative_to(work_dir, candidate):
                return candidate
    return work_dir


def _replace_request(
    tool_name: str,
    args: Mapping[str, Any],
    argv: list[str] | None,
    resolved: str,
) -> Dict[str, Any]:
    out = dict(args)
    if tool_name in {"run_command", "start_service"}:
        new_argv = list(argv or [])
        new_argv[0] = resolved
        out["cmd"] = new_argv
    elif tool_name == "run_script":
        out["interpreter"] = resolved
    elif tool_name == "verify_and_record":
        new_argv = list(argv or [])
        new_argv[0] = resolved
        out["check"] = new_argv
    return out


def resolve_process_python(
    ctx: Any,
    tool_name: str,
    args: Mapping[str, Any],
    *,
    runtime_mode: str,
    effective_constraint: Optional[TaskConstraint] = None,
    resolved_binding: ResolvedResourceBinding | None = None,
) -> tuple[Dict[str, Any], Optional[PythonResolutionTrace]]:
    """Resolve an exact ``python``/``python3`` request for one process tool."""

    name = str(tool_name or "").strip()
    original = dict(args or {})
    if name not in _PROCESS_TOOLS:
        return original, None
    requested, argv = _python_request(name, original)
    if not requested:
        return original, None

    constraint = normalize_task_constraint(effective_constraint)
    cwd_text = str(original.get("cwd") or "")
    binding = resolved_binding
    try:
        operation = "service" if name == "start_service" else "shell"
        if binding is None:
            binding = build_resolved_resource_binding(
                ctx,
                operation=operation,
                process_cwd=cwd_text,
                bucket=str(original.get("bucket") or ""),
                skill_name=str(original.get("skill_name") or ""),
            )
        work_dir = pathlib.Path(binding.target_path).resolve(strict=False)
    except Exception:
        trace = PythonResolutionTrace(
            tool=name,
            requested_interpreter=requested,
            resolved_interpreter=requested,
            surface="unresolved",
            environment="target_path",
            reason="target_path_fallback",
            fallback_reason="cwd_resolution_failed",
            error_reason="cwd_resolution_failed",
        )
        return original, trace

    fallback_reason = ""
    assert binding is not None
    skill_binding = resolved_binding
    if skill_binding is None and binding.root == "skill_payload":
        skill_binding = binding
    skill_python, skill_reason = _reviewed_skill_python(ctx, skill_binding)
    if skill_python:
        trace = PythonResolutionTrace(
            tool=name,
            requested_interpreter=requested,
            resolved_interpreter=skill_python,
            surface="reviewed_skill",
            environment="isolated_skill",
            reason="reviewed_skill_environment",
            **_trace_target(binding),
        )
        return _replace_request(name, original, argv, skill_python), trace
    if skill_reason == "skill_env_python_unavailable":
        return original, PythonResolutionTrace(
            tool=name, requested_interpreter=requested, resolved_interpreter=requested, surface="reviewed_skill",
            environment="isolated_skill", reason=skill_reason, error_reason=skill_reason, **_trace_target(binding))
    if skill_reason:
        fallback_reason = skill_reason

    executor_active, executor_error = _executor_covers(ctx, work_dir)
    if executor_active:
        trace = PythonResolutionTrace(
            tool=name,
            requested_interpreter=requested,
            resolved_interpreter="python3",
            surface="executor",
            environment="backend_path",
            reason="executor_backend_python3",
            fallback_reason=fallback_reason,
            **_trace_target(binding),
        )
        return _replace_request(name, original, argv, "python3"), trace
    if executor_error and not fallback_reason:
        fallback_reason = executor_error

    surface = _surface_for(ctx, binding, constraint)
    if surface in {"external_workspace", "user_files"}:
        project_python = project_venv_python(_project_root(ctx, surface, work_dir))
        if project_python:
            trace = PythonResolutionTrace(
                tool=name,
                requested_interpreter=requested,
                resolved_interpreter=project_python,
                surface=surface,
                environment="project_venv",
                reason="project_venv",
                fallback_reason=fallback_reason,
                **_trace_target(binding),
            )
            return _replace_request(name, original, argv, project_python), trace
        trace = PythonResolutionTrace(
            tool=name,
            requested_interpreter=requested,
            resolved_interpreter=requested,
            surface=surface,
            environment="target_path",
            reason="target_path_fallback",
            fallback_reason=fallback_reason or "project_venv_unavailable",
            **_trace_target(binding),
        )
        return original, trace

    configured_agent_python = _usable_executable(
        os.environ.get("OUROBOROS_AGENT_PYTHON", "")
    )
    agent_python = configured_agent_python or _usable_executable(sys.executable or "")
    if agent_python:
        trace = PythonResolutionTrace(
            tool=name,
            requested_interpreter=requested,
            resolved_interpreter=agent_python,
            surface=surface,
            environment="ouroboros_agent",
            reason="agent_python",
            fallback_reason=(
                fallback_reason
                or ("agent_env_unavailable_process_fallback" if not configured_agent_python else "")
            ),
            **_trace_target(binding),
        )
        return _replace_request(name, original, argv, agent_python), trace

    trace = PythonResolutionTrace(
        tool=name,
        requested_interpreter=requested,
        resolved_interpreter=requested,
        surface=surface,
        environment="target_path",
        reason="target_path_fallback",
        fallback_reason=fallback_reason or "agent_python_unavailable",
        error_reason="agent_python_unavailable",
        **_trace_target(binding),
    )
    return original, trace


def _normalize_runtime_token(text: str) -> str:
    """Trimmed token for family matching; case-folding and launcher-suffix
    stripping apply ONLY on Windows (R7/T9): there ``node.exe``/``NPM.CMD``
    classify like their bare spellings, while POSIX exec is case-sensitive so
    the exact token is preserved.  An absolute path or a versioned name
    (``node20``) never equals a family token, so both stay untouched by
    construction — same contract as python.
    """

    token = str(text or "").strip()
    if not IS_WINDOWS:
        # POSIX exec is case-sensitive: "NODE" is a different file, and
        # launcher suffixes are a Windows-only convention.
        return token
    token = token.lower()
    for suffix in _WINDOWS_LAUNCHER_SUFFIXES:
        if token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _shell_body_names_node_family(body: str) -> bool:
    """Deterministic seam-scan of a shell body for node-family tokens (R3).

    Reuses the guard-layer tokenizer (``shell_parse.shell_tokens``) — no regex
    over prose.  A transitive spawn (a python script that itself execs node)
    stays a documented residual.
    """

    tokens = shell_tokens(str(body or "")) or []
    return any(_normalize_runtime_token(token) in _NODE_FAMILY_TOKENS for token in tokens)


def _node_request(tool_name: str, args: Mapping[str, Any]) -> tuple[str, list[str] | None, str]:
    """Return ``(requested_token, argv, trigger)`` for a node-family launch.

    ``trigger`` is ``""`` (not a node-family request), ``"runtime"`` (argv[0]
    is node/nodejs — rewrite-eligible), ``"family"`` (npm/npx/pnpm/yarn/
    corepack — emergency prepend only), or ``"shell_body"`` (a shell wrapper —
    sh/bash/zsh/dash, matched by basename — whose command body names a family
    tool; emergency prepend only).
    """

    requested = ""
    argv: list[str] | None = None
    body = ""
    if tool_name in {"run_command", "start_service"}:
        raw = args.get("cmd")
        if not isinstance(raw, list) or not raw:
            return "", None, ""
        argv = [str(part) for part in raw]
        requested = argv[0].strip()
        if requested != argv[0]:
            # A whitespace-padded head is not a bare runtime token: classifying
            # it would let the trace attest a substitution that deferred
            # executors (verify) never apply — run it as written instead (T8).
            return "", None, ""
        body = shell_command_string(argv)
    elif tool_name == "run_script":
        requested = str(args.get("interpreter") or "python3").strip() or "python3"
        body = str(args.get("script") or "")
    elif tool_name == "verify_and_record":
        kind = str(args.get("contract_kind") or "").strip()
        if kind not in _VERIFY_RUN_KINDS:
            return "", None, ""
        argv = normalize_check_argv(args.get("check")) or []
        if not argv:
            return "", None, ""
        requested = str(argv[0]).strip()
        if requested != str(argv[0]):
            return "", None, ""  # padded head: not bare, run as written (T8)
        body = shell_command_string(argv)
    else:
        return "", None, ""

    normalized = _normalize_runtime_token(requested)
    if normalized in _NODE_TOKENS:
        return requested, argv, "runtime"
    if normalized in _NODE_FAMILY_TOKENS:
        return requested, argv, "family"
    # Wrapper matching is BASENAME-based (full-scope finding F-1): /bin/sh and
    # zsh -c bodies reproduce the incident class exactly like bare sh. This
    # grants no new exec power — a wrapper hit only scans the body and rides
    # the env prepend; family/runtime tokens above stay bare-only on purpose
    # (an absolute node path is the caller's explicit runtime choice).
    wrapper_token = _normalize_runtime_token(pathlib.PurePath(requested).name)
    if wrapper_token in _NODE_SHELL_WRAPPERS and _shell_body_names_node_family(body):
        return requested, argv, "shell_body"
    return "", None, ""


def _node_trace(**fields: Any) -> InterpreterResolutionTrace:
    return InterpreterResolutionTrace(family="node", **fields)


def resolve_process_node(
    ctx: Any,
    tool_name: str,
    args: Mapping[str, Any],
    *,
    runtime_mode: str,
    effective_constraint: Optional[TaskConstraint] = None,
    resolved_binding: ResolvedResourceBinding | None = None,
) -> tuple[Dict[str, Any], Optional[InterpreterResolutionTrace]]:
    """Resolve a node-family request for one process tool (D2/D3/D4 ladder).

    Ladder: (i) a NON-local executor backend resolves ``node`` in its own
    filesystem — argv untouched, no host path leaks into the container; a
    LOCAL executor runs on this host, so the ladder continues (Q2-3).  (ii) a
    PATH candidate that passes the execution health probe wins — argv and env
    stay byte-identical (the healthy-system no-op invariant).  (iii) a missing
    or probe-dead PATH candidate falls back to the healthy bundled runtime:
    node/nodejs argv[0] is rewritten, and EVERY triggered launch gets the
    bundled dir attested as a child-env PATH prepend (fixes npm & co and
    ``sh -c`` bodies).  (iv) with neither usable the argv runs as written and
    fails honestly — node never raises a typed pre-block (R8).

    Placement (post-gates, deliberate — differs from python's pre-guard seam):
    the ladder's health check is an EXECUTION probe (``<candidate> --version``)
    and argv[0] steers which executable it runs. Pre-guard it would execute an
    agent-influenced binary BEFORE the light fence / protected-write gates /
    shell guard / safety supervisor — a planted PATH shim named ``node`` would
    run its payload on a call those gates then refuse. Post-gates the probe
    holds strictly LESS power than the just-approved call itself. Guards
    therefore inspect the ORIGINAL bare argv; the substitution the handler
    executes is limited to a family-stable argv[0]
    (``shell_guards.interpreter_family`` classifies the bare token and the
    resolver's absolute path identically) and is disclosed via the trace event
    and the per-call attestation. Node never pre-blocks (R8): with no usable
    runtime the argv runs as written and fails honestly.
    """

    name = str(tool_name or "").strip()
    original = dict(args or {})
    if name not in _PROCESS_TOOLS:
        return original, None
    requested, argv, trigger = _node_request(name, original)
    if not trigger:
        return original, None

    # R1: the resolver must see the SAME PATH the handler will (both call the
    # idempotent bootstrap; whoever runs first performs the one mutation).  The
    # snapshot is the base of any attested child-env PATH prepend.
    bootstrap_process_path()
    from ouroboros.tools.process_facts import process_path_for_cwd, selected_process_environment
    path_snapshot = str(selected_process_environment().get("PATH", os.environ.get("PATH", "")) or "")

    constraint = normalize_task_constraint(effective_constraint)
    cwd_text = str(original.get("cwd") or "")
    binding = resolved_binding
    try:
        operation = "service" if name == "start_service" else "shell"
        if binding is None:
            binding = build_resolved_resource_binding(
                ctx,
                operation=operation,
                process_cwd=cwd_text,
                bucket=str(original.get("bucket") or ""),
                skill_name=str(original.get("skill_name") or ""),
            )
        work_dir = pathlib.Path(binding.target_path).resolve(strict=False)
    except Exception:
        # No typed pre-block for node (R8): execution proceeds as written and
        # the handler's own binding build reports the canonical cwd block.
        trace = _node_trace(
            tool=name,
            requested_interpreter=requested,
            resolved_interpreter=requested,
            surface="unresolved",
            environment="target_path",
            reason="target_path_fallback",
            fallback_reason="cwd_resolution_failed",
            path_snapshot=path_snapshot,
        )
        return original, trace

    fallback_reason = ""
    assert binding is not None
    covers, executor_kind, executor_error = _executor_covers_kind(ctx, work_dir)
    if covers and executor_kind != "local":
        # R2/Q2-3: only a non-local backend skips the ladder — bare ``node``
        # resolves inside the container and a host path must never leak there.
        trace = _node_trace(
            tool=name,
            requested_interpreter=requested,
            resolved_interpreter=requested,
            surface="executor",
            environment="backend_path",
            reason="executor_backend_node",
            path_snapshot=path_snapshot,
            **_trace_target(binding),
        )
        return original, trace
    if executor_error:
        fallback_reason = executor_error

    surface = _surface_for(ctx, binding, constraint)
    probe_token = requested if trigger == "runtime" else "node"
    located = shutil.which(probe_token, path=process_path_for_cwd(path_snapshot, work_dir)) or ""
    if located and not os.path.isabs(located):
        # Search entries are bound to the launch cwd above. If a platform still
        # cannot establish an absolute candidate, retain the as-written launch
        # rather than substituting a runtime on unprovable evidence (T10).
        trace = _node_trace(
            tool=name,
            requested_interpreter=requested,
            resolved_interpreter=requested,
            surface=surface,
            environment="host_path",
            reason="path_node_relative_entry_unprovable",
            fallback_reason=fallback_reason,
            path_snapshot=path_snapshot,
            runtime_path=located,
            **_trace_target(binding),
        )
        return original, trace
    if located:
        path_health = node_runtime_health(located)
        if path_health.healthy:
            trace = _node_trace(
                tool=name,
                requested_interpreter=requested,
                resolved_interpreter=requested,
                surface=surface,
                environment="host_path",
                reason="path_node_healthy",
                fallback_reason=fallback_reason,
                path_snapshot=path_snapshot,
                runtime_path=located,
                runtime_version=path_health.version,
                **_trace_target(binding),
            )
            return original, trace
        path_fact = f"path_node_broken:{path_health.reason}:{located}"
    else:
        path_fact = f"path_node_missing:{probe_token}"

    bundled = resolve_bundled_node() or ""
    bundled_health = node_runtime_health(bundled) if bundled else None
    if bundled_health is not None and bundled_health.healthy:
        prepend_dir = str(pathlib.Path(bundled).parent)
        resolved = bundled if trigger == "runtime" else requested
        trace = _node_trace(
            tool=name,
            requested_interpreter=requested,
            resolved_interpreter=resolved,
            surface=surface,
            environment="bundled_node",
            reason="bundled_node_fallback",
            fallback_reason=path_fact,
            path_snapshot=path_snapshot,
            env_path_prepend=prepend_dir,
            runtime_path=bundled,
            runtime_version=bundled_health.version,
            **_trace_target(binding),
        )
        if trigger == "runtime" and name != "verify_and_record":
            return _replace_request(name, original, argv, bundled), trace
        # verify_and_record keeps its ORIGINAL check text (receipt identity,
        # R4); the handler executes the resolved argv from this attestation.
        return original, trace

    if bundled_health is not None:
        bundled_fact = f"bundled_node_broken:{bundled_health.reason}"
    else:
        bundled_fact = "bundled_node_missing"
    trace = _node_trace(
        tool=name,
        requested_interpreter=requested,
        resolved_interpreter=requested,
        surface=surface,
        environment="target_path",
        reason="no_usable_node",
        fallback_reason=f"{path_fact};{bundled_fact}",
        path_snapshot=path_snapshot,
        **_trace_target(binding),
    )
    return original, trace


def active_node_resolution(ctx: Any) -> Optional[InterpreterResolutionTrace]:
    """The node-family attestation of the CURRENT handler call, if any.

    The registry scopes ``_active_interpreter_resolution`` to exactly one
    handler invocation, so a non-``None`` return always describes this call.
    """

    resolution = getattr(ctx, "_active_interpreter_resolution", None)
    if isinstance(resolution, InterpreterResolutionTrace) and resolution.family == "node":
        return resolution
    return None


def interpreter_path_overlay(
    trace: Optional[InterpreterResolutionTrace],
) -> Dict[str, str] | None:
    """The ``{"PATH": ...}`` overlay attested by an emergency bundled fallback.

    ``None`` on every healthy/no-op resolution — callers must then leave their
    child env byte-identical to today's behavior.
    """

    if trace is None or not getattr(trace, "env_path_prepend", ""):
        return None
    base = trace.path_snapshot or os.environ.get("PATH", "")
    return {"PATH": trace.env_path_prepend + ((PATH_SEP + base) if base else "")}


def apply_env_path_prepend(
    env: Mapping[str, str] | None,
    trace: Optional[InterpreterResolutionTrace],
) -> Dict[str, str] | None:
    """Child env with the attested bundled-runtime dir prepended to PATH.

    Without an attested prepend the input is returned UNCHANGED (``None`` stays
    ``None`` → inherit), keeping the healthy path byte-identical.  With one, a
    ``None``/inherit env becomes an explicit ``os.environ`` copy and PATH is
    rebuilt from the resolver's frozen snapshot.  Never mutates process-global
    ``os.environ``.
    """

    overlay = interpreter_path_overlay(trace)
    if overlay is None:
        return env if env is None else dict(env)
    out = dict(os.environ if env is None else env)
    if IS_WINDOWS:
        for key in [k for k in out if k.upper() == "PATH" and k != "PATH"]:
            del out[key]
    out["PATH"] = overlay["PATH"]
    return out


_RESOLUTION_EVENT_TYPES = {
    "python": "python_interpreter_resolution",
    "node": "node_runtime_resolution",
}


def resolve_node_postgates(
    ctx: Any,
    tool_name: str,
    args: "Dict[str, Any]",
    *,
    runtime_mode: str,
    effective_constraint: Any = None,
    resolved_binding: Any = None,
) -> "tuple[Dict[str, Any], Any]":
    """Registry seam: resolve node once post-gates and record the trace.

    Post-gates on purpose — the health check EXECUTES the candidate; the full
    placement rationale lives on ``resolve_process_node``.
    """
    args, node_resolution = resolve_process_node(
        ctx,
        tool_name,
        args,
        runtime_mode=runtime_mode,
        effective_constraint=effective_constraint,
        resolved_binding=resolved_binding,
    )
    record_interpreter_resolution(ctx, node_resolution)
    return args, node_resolution


@contextlib.contextmanager
def interpreter_attestation(ctx: Any, trace: Any):
    """Scope the per-call interpreter attestation to one handler invocation.

    Publishes BOTH slots the downstream consumers read — the trace itself
    (``ctx._active_interpreter_resolution``: verify_and_record's R4 substitution
    and the handlers' emergency env prepend)
    and, exactly when the resolver substituted execution (an argv rewrite or an
    attested emergency PATH prepend), the ONE string slot
    ``ctx._process_resolved_runtime`` that result_meta and the verify receipt
    disclose. Both slots are restored on exit whatever the handler did; on the
    healthy path the runtime slot is never set at all.
    """
    missing = object()
    prior = getattr(ctx, "_active_interpreter_resolution", missing)
    ctx._active_interpreter_resolution = trace
    prior_runtime = getattr(ctx, "_process_resolved_runtime", missing)
    resolved_runtime = ""
    if trace is not None and (
        getattr(trace, "changed", False) or getattr(trace, "env_path_prepend", "")
    ):
        resolved_runtime = str(
            getattr(trace, "runtime_path", "")
            or getattr(trace, "resolved_interpreter", "")
            or ""
        )
    if resolved_runtime:
        ctx._process_resolved_runtime = resolved_runtime
    try:
        yield
    finally:
        if prior is missing:
            try:
                delattr(ctx, "_active_interpreter_resolution")
            except AttributeError:
                pass
        else:
            ctx._active_interpreter_resolution = prior
        if resolved_runtime:
            if prior_runtime is missing:
                try:
                    delattr(ctx, "_process_resolved_runtime")
                except AttributeError:
                    pass
            else:
                ctx._process_resolved_runtime = prior_runtime


def record_interpreter_resolution(ctx: Any, trace: Optional[InterpreterResolutionTrace]) -> None:
    """Persist a compact, secret-free trace in the existing events log."""

    if trace is None:
        return
    try:
        event: Dict[str, Any] = {
            "ts": utc_now_iso(),
            "type": _RESOLUTION_EVENT_TYPES.get(trace.family, "interpreter_resolution"),
            "task_id": str(getattr(ctx, "task_id", "") or ""),
            **trace.to_event(),
        }
        metadata = getattr(ctx, "task_metadata", {})
        if isinstance(metadata, dict):
            for key in ("root_task_id", "parent_task_id", "delegation_role"):
                value = metadata.get(key)
                if value not in (None, ""):
                    event[key] = value
        correlation = getattr(ctx, "_current_llm_call_meta", {})
        if isinstance(correlation, dict):
            for key in ("execution_id", "round_id", "llm_call_id"):
                if correlation.get(key):
                    event[key] = correlation[key]
        drive_logs = getattr(ctx, "drive_logs", None)
        if callable(drive_logs):
            log_dir = pathlib.Path(drive_logs())
        else:
            log_dir = pathlib.Path(getattr(ctx, "drive_root")) / "logs"
        from ouroboros.tools.process_facts import redact_process_data
        append_jsonl(log_dir / "events.jsonl", redact_process_data(event))
    except Exception:
        # Trace persistence must not make an otherwise-valid process call fail.
        return


# Historic recorder name, kept for existing callers; the event type is chosen
# by the trace's family either way.
record_python_resolution = record_interpreter_resolution


__all__ = [
    "InterpreterResolutionTrace",
    "PythonResolutionTrace",
    "active_node_resolution",
    "apply_env_path_prepend",
    "interpreter_attestation",
    "resolve_node_postgates",
    "interpreter_path_overlay",
    "record_interpreter_resolution",
    "record_python_resolution",
    "resolve_process_node",
    "resolve_process_python",
]
