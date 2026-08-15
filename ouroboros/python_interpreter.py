"""Surface-aware Python selection for host process tools.

Only the four public process launch surfaces opt into this resolver.  It runs
once at registry pre-dispatch so the deterministic guards and the handler see
the same argv; launchers must not rewrite the interpreter afterwards.
"""

from __future__ import annotations

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
from ouroboros.platform_layer import project_venv_python
from ouroboros.shell_parse import normalize_check_argv
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    build_resolved_resource_binding,
    path_is_relative_to,
)
from ouroboros.utils import append_jsonl, utc_now_iso

_PYTHON_TOKENS = frozenset({"python", "python3"})
_PROCESS_TOOLS = frozenset({"run_command", "run_script", "start_service", "verify_and_record"})
# Mirrors the run-kind set of tools/verify.py (`_RUN_KINDS`); keep in sync when
# a new verify run-kind is introduced there.
_VERIFY_RUN_KINDS = frozenset({"visible_verifier", "explicit_command", "explicit_metric"})
_VERIFIED_RESOLUTIONS = frozenset(
    {
        ("reviewed_skill_environment", "isolated_skill"),
        # (RWS v2) A target-native interpreter proven by the bundled prepare
        # response is evidence of the SAME strength as a Home probe: it comes
        # from the host that will run the process.
        ("remote_prepare_interpreter", "target_native"),
        ("executor_backend_python3", "backend_path"),
        ("project_venv", "project_venv"),
        ("agent_python", "ouroboros_agent"),
    }
)


@dataclass(frozen=True)
class PythonResolutionTrace:
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

    @property
    def changed(self) -> bool:
        return self.requested_interpreter != self.resolved_interpreter

    @property
    def verified(self) -> bool:
        """Whether the resolver proved the selected interpreter provenance."""

        return (self.reason, self.environment) in _VERIFIED_RESOLUTIONS

    def to_event(self) -> Dict[str, Any]:
        return {**asdict(self), "changed": self.changed}


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
def _effective_cwd_text(ctx: Any, tool_name: str, args: Mapping[str, Any], runtime_mode: str) -> str:
    cwd = str(args.get("cwd") or "")
    if (
        tool_name == "run_script"
        and not cwd.strip()
        and str(runtime_mode or "").strip() == "light"
        and not bool(getattr(ctx, "is_workspace_mode", lambda: False)())
    ):
        try:
            return str(ctx.task_drive_root())
        except Exception:
            return cwd
    return cwd


def usable_executable(path_text: str) -> str:
    """Validate an interpreter while preserving venv symlink semantics.

    The SINGLE interpreter-provability probe: ``execution_facts`` exposes it as
    the placement-neutral ``interpreter_fact`` rather than re-implementing it."""

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
    when a non-registry/direct caller supplied no binding.
    """

    try:
        from ouroboros.marketplace.isolated_deps import python_runtime_binary, read_deps_state
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
        if loaded is None or not skill_readiness_for_execution(drive_root, loaded).ready:
            return "", "reviewed_skill_environment_unavailable"
        deps_state = read_deps_state(drive_root, loaded.name, loaded.skill_dir)
        if str(deps_state.get("status") or "") != "installed":
            return "", "reviewed_skill_environment_unavailable"
        candidate = python_runtime_binary(loaded.skill_dir)
        usable = usable_executable(str(candidate or ""))
        if usable:
            return usable, ""
    except Exception:
        return "", "reviewed_skill_environment_probe_failed"
    return "", "reviewed_skill_environment_unavailable"


def _executor_covers(ctx: Any, work_dir: pathlib.Path) -> tuple[bool, str]:
    """Coverage via the shared predicate; only the resolver's own ctx-error
    taxonomy stays local (ValueError = malformed ref reads as plain non-coverage,
    anything else is surfaced as a fallback_reason breadcrumb)."""
    from ouroboros.workspace_executor import covers, executor_ref_from_ctx

    try:
        executor = executor_ref_from_ctx(ctx)
    except ValueError:
        return False, ""
    except Exception:
        return False, "executor_resolution_failed"
    return covers(executor, work_dir), ""


def _surface_for(
    ctx: Any,
    binding: ResolvedResourceBinding,
    constraint: Optional[TaskConstraint],
) -> str:
    if binding.root != "active_workspace":
        return binding.root or "unresolved"
    if constraint and constraint.mode == "acting_subagent" and constraint.surface == "self_worktree":
        return "system_repo"
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
    facts: Any = None,
) -> tuple[Dict[str, Any], Optional[PythonResolutionTrace]]:
    """Resolve an exact ``python``/``python3`` request for one process tool.

    ``facts`` is the operation's prepare fact door. For a non-local placement the
    interpreter is a TARGET fact and is read from there: probing Home for the
    interpreter of a process that will run elsewhere is the same category error as
    probing Home for its cwd, and `check_safety` must see evidence of equal
    strength on both placements (RWS-02)."""

    name = str(tool_name or "").strip()
    original = dict(args or {})
    if name not in _PROCESS_TOOLS:
        return original, None
    requested, argv = _python_request(name, original)
    if not requested:
        return original, None
    if facts is not None and str(getattr(facts, "placement", "local")) != "local":
        return _resolve_target_native_python(name, original, argv, requested, facts)

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

    configured_agent_python = usable_executable(
        os.environ.get("OUROBOROS_AGENT_PYTHON", "")
    )
    agent_python = configured_agent_python or usable_executable(sys.executable or "")
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


def _resolve_target_native_python(
    name: str,
    original: Dict[str, Any],
    argv: list[str] | None,
    requested: str,
    facts: Any,
) -> tuple[Dict[str, Any], Optional[PythonResolutionTrace]]:
    """Interpreter selection for a non-local placement, from prepare facts only.

    The fact comes from the bundled prepare block, where execd recorded the
    interpreter it resolved ON the target — a venv under the workspace root if there
    is one, else the target's PATH. If the bundle cannot answer, the typed
    unresolved trace is returned and the dispatcher surfaces
    PYTHON_INTERPRETER_UNAVAILABLE: a Home interpreter is never smuggled into a
    remote launch, because a path that exists here says nothing about there.
    """
    try:
        fact = facts.interpreter_fact(requested)
    except Exception:
        return original, PythonResolutionTrace(
            tool=name,
            requested_interpreter=requested,
            resolved_interpreter=requested,
            surface=str(getattr(facts, "placement", "remote")),
            environment="target_path",
            reason="target_path_fallback",
            fallback_reason="remote_prepare_unavailable",
            error_reason="remote_prepare_unavailable",
        )
    surface = str(getattr(facts, "placement", "remote"))
    if not fact.usable:
        return original, PythonResolutionTrace(
            tool=name,
            requested_interpreter=requested,
            resolved_interpreter=requested,
            surface=surface,
            environment="target_path",
            reason="target_path_fallback",
            fallback_reason="remote_interpreter_unavailable",
            error_reason="remote_interpreter_unavailable",
        )
    trace = PythonResolutionTrace(
        tool=name,
        requested_interpreter=requested,
        resolved_interpreter=fact.resolved,
        surface=surface,
        environment="target_native",
        reason="remote_prepare_interpreter",
    )
    return _replace_request(name, original, argv, fact.resolved), trace


def record_python_resolution(ctx: Any, trace: Optional[PythonResolutionTrace]) -> None:
    """Persist a compact, secret-free trace in the existing events log."""

    if trace is None:
        return
    try:
        event: Dict[str, Any] = {
            "ts": utc_now_iso(),
            "type": "python_interpreter_resolution",
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
        append_jsonl(log_dir / "events.jsonl", event)
    except Exception:
        # Trace persistence must not make an otherwise-valid process call fail.
        return


__all__ = [
    "PythonResolutionTrace",
    "record_python_resolution",
    "resolve_process_python",
    "usable_executable",
]
