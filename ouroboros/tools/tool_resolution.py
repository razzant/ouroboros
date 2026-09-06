"""Argument normalization and physical target binding for tool dispatch.

Every span is extracted VERBATIM from the parent's tip bytes by
scripts/v7next_transplant.py (D18/D33 module-handle split, proof-checked);
the parent re-exports every moved name, so historical imports and
monkeypatch targets keep working unchanged.
"""

from __future__ import annotations

import inspect
import os
import pathlib

from dataclasses import dataclass

from typing import TYPE_CHECKING

from ouroboros.process_interpreters import record_interpreter_resolution, resolve_process_python

from ouroboros.tools.tool_result import ToolResult

if TYPE_CHECKING:  # annotation-only imports (inert at runtime)
    from typing import Any
    from typing import Callable
    from typing import Dict
    from typing import List
    from typing import Literal

    from ouroboros.tools.tool_catalog import ToolEntry


def _registry():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time (the owner-approved D18/D33 mechanical exception).
    """
    from ouroboros.tools import registry

    return registry


def _coerce_real_path(value: Any) -> pathlib.Path | None:
    if value is None or value.__class__.__module__.startswith("unittest.mock"):
        return None
    try:
        return pathlib.Path(os.fspath(value))
    except TypeError:
        return None


def active_repo_dir_for(ctx: Any) -> pathlib.Path:
    """Return the active repo/workspace root for real and lightweight test contexts."""
    active = getattr(ctx, "active_repo_dir", None)
    if callable(active):
        try:
            candidate = active()
        except Exception:
            candidate = None
        path = _coerce_real_path(candidate)
        if path is not None:
            return path

    workspace_root = getattr(ctx, "workspace_root", None)
    workspace_path = _coerce_real_path(workspace_root)
    if workspace_path is not None:
        workspace_mode = str(getattr(ctx, "workspace_mode", "") or "").strip()
        if workspace_mode:
            return workspace_path

    return pathlib.Path(getattr(ctx, "repo_dir"))


def system_repo_dir_for(ctx: Any) -> pathlib.Path:
    """Return the Ouroboros system repo root, not an external active workspace."""

    return pathlib.Path(getattr(ctx, "system_repo_dir", None) or getattr(ctx, "repo_dir"))


_PATH_NORMALIZED_TOOLS = frozenset({"read_file", "write_file", "edit_text", "list_files", "search_code", "query_code"})
_TOP_LEVEL_PATH_WRITE_TOOLS = frozenset({"write_file", "edit_text"})


_ROOT_ARG_REPO_WRITE_TOOLS = frozenset({"write_file", "edit_text", "apply_patch", "edit_batch"})


@dataclass(frozen=True)
class _DispatchPathNormalization:
    """Exact dispatch note plus any explicit root required before dispatch."""

    text: str = ""
    required_root: Literal["active_workspace"] | None = None


def _payload_write_paths(name: str, args: Dict[str, Any]) -> List[str]:
    """Repo paths a write tool will touch, in the spelling its guards must judge.

    write_file/edit_text carry `path`/`files[]` and were already canonicalized by
    `_normalize_dispatch_path_args`. apply_patch addresses files inside the patch
    text (`*** Update File: <path>`) and edit_batch inside `edits[]`, so their
    paths reach this point RAW and are canonicalized here — otherwise a
    protected-path gate reads `repo/BIBLE.md` (not a protected-table member)
    while the write lands on `BIBLE.md`.
    """

    paths: List[str] = []
    if name == "write_file":
        if isinstance(args.get("path"), str) and args["path"]:
            paths.append(args["path"])
        for entry in args.get("files") or []:
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                paths.append(entry["path"])
    elif name == "edit_text":
        if isinstance(args.get("path"), str):
            paths.append(args["path"])
    elif name == "edit_batch":
        for entry in args.get("edits") or []:
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                paths.append(entry["path"])
    elif name == "apply_patch":
        # Derived from the REAL parser (lazy import: edit_ops imports this
        # module), so the gate can never drift from what apply_patch will do.
        # An unparseable patch yields no paths and is refused by the handler
        # before any write, so the gate has nothing to miss.
        from ouroboros.tools.edit_ops import patch_target_paths

        paths.extend(patch_target_paths(str(args.get("patch") or "")))
    return [p for p in paths if str(p or "").strip()]


def _normalize_dispatch_path_args_result(
    ctx: Any,
    name: str,
    args: Dict[str, Any],
) -> _DispatchPathNormalization:
    """ROOT-FIX (v6.35.0): normalize an absolute / redundant-root-basename
    active_workspace|system_repo path arg IN PLACE at the dispatch boundary, so
    the handler AND every downstream guard (protected-path, protected-artifact,
    accidental-truncation shrink guard) resolve the SAME target. One authoritative
    normalization point is what makes a guard unable to desync from the operation.

    v6.54.3 root-label fix: returns a dispatch note ("" when nothing rerouted).
    When ``root='user_files'`` carries an ABSOLUTE path that resolves under the
    ACTIVE WORKSPACE root, the root label is wrong, not the intent: reads
    (read_file/list_files/search_code) are auto-routed to
    ``root='active_workspace'`` with a visible note appended AFTER the result
    (trailing, so first-line failure classification is never masked),
    and writes (write_file/edit_text) return an actionable
    ROOT_REQUIRED_ACTIVE_WORKSPACE redirect instead of a generic access denial.
    The destination root still passes every downstream gate (profile access
    decision, protected-path guards, subagent filters) — only the label is
    corrected, never the authority. ``query_code`` is excluded: its
    root=user_files external-target contract handles absolute paths natively."""
    if name not in _PATH_NORMALIZED_TOOLS:
        return _DispatchPathNormalization()
    root_arg = str(args.get("root") or "active_workspace")
    if root_arg in ("active_workspace", "system_repo"):
        try:
            norm_root = active_repo_dir_for(ctx) if root_arg == "active_workspace" else system_repo_dir_for(ctx)
            for _key in ("path", "dir"):
                if isinstance(args.get(_key), str) and args[_key]:
                    args[_key] = _registry().normalize_root_relative(norm_root, args[_key])
            if isinstance(args.get("files"), list):
                for _f in args["files"]:
                    if isinstance(_f, dict) and isinstance(_f.get("path"), str) and _f["path"]:
                        _f["path"] = _registry().normalize_root_relative(norm_root, _f["path"])
        except Exception:
            pass
        return _DispatchPathNormalization()
    if root_arg != "user_files" or name == "query_code":
        return _DispatchPathNormalization()
    try:
        workspace = pathlib.Path(active_repo_dir_for(ctx)).resolve(strict=False)
    except Exception:
        return _DispatchPathNormalization()

    def _under_workspace(text: str) -> bool:
        if not _registry().is_absolute_path_text(text):
            return False
        try:
            pathlib.Path(text).expanduser().resolve(strict=False).relative_to(workspace)
            return True
        except (ValueError, OSError, RuntimeError):
            return False

    candidates: list[str] = []
    for _key in ("path", "dir"):
        if isinstance(args.get(_key), str) and args[_key]:
            candidates.append(args[_key])
    if isinstance(args.get("files"), list):
        for _f in args["files"]:
            if isinstance(_f, dict) and isinstance(_f.get("path"), str) and _f["path"]:
                candidates.append(_f["path"])
    hits = [text for text in candidates if _under_workspace(text)]
    if not hits:
        return _DispatchPathNormalization()
    if name in _TOP_LEVEL_PATH_WRITE_TOOLS:
        return _DispatchPathNormalization(
            text=(
                "⚠️ ROOT_REQUIRED_ACTIVE_WORKSPACE: absolute path "
                f"{hits[0]!r} is under the active workspace, but root='user_files' does not "
                "write there. Retry the same call with root='active_workspace' (the same "
                "path is accepted)."
            ),
            required_root="active_workspace",
        )
    args["root"] = "active_workspace"
    try:
        for _key in ("path", "dir"):
            if isinstance(args.get(_key), str) and args[_key]:
                args[_key] = _registry().normalize_root_relative(workspace, args[_key])
        if isinstance(args.get("files"), list):
            for _f in args["files"]:
                if isinstance(_f, dict) and isinstance(_f.get("path"), str) and _f["path"]:
                    _f["path"] = _registry().normalize_root_relative(workspace, _f["path"])
    except Exception:
        pass
    return _DispatchPathNormalization(
        text=(
            "⚠️ AUTO_ROUTED_TO_ACTIVE_WORKSPACE: absolute path "
            f"{hits[0]!r} is under the active workspace; the call ran with "
            "root='active_workspace'. Pass root='active_workspace' directly for "
            "workspace paths."
        )
    )


def _normalize_dispatch_path_args(ctx: Any, name: str, args: Dict[str, Any]) -> str:
    """Compatibility projection of the typed dispatch-path normalization."""
    return _normalize_dispatch_path_args_result(ctx, name, args).text


_TOOL_ARG_ALIASES: dict[str, dict[str, str]] = {
    "*": {"max_entries": "max_results", "timeout": "timeout_sec"},
}


_IGNORE_ROOT_ARG_TOOLS = frozenset({
    "commit_reviewed",
    "vcs_commit_reviewed",
})


_GENERIC_VCS_TARGET_TOOLS = frozenset({
    "vcs_status",
    "vcs_diff",
    "vcs_pull_ff",
    "vcs_restore",
    "vcs_revert",
})


_TARGET_BINDING_OPERATIONS = {
    "read_file": "read",
    "list_files": "list",
    "search_code": "search",
    "query_code": "search",
    "write_file": "write",
    "edit_text": "edit",
    "apply_patch": "edit",
    "edit_batch": "edit",
    **{name: "vcs" for name in _GENERIC_VCS_TARGET_TOOLS},
}


_SKILL_LIFECYCLE_TARGET_TOOLS = frozenset({
    "skill_review",
    "skill_preflight",
    "submit_skill_to_hub",
})


_PROCESS_TARGET_TOOLS = frozenset({"run_command", "run_script", "start_service"})


_VERIFY_RUN_KINDS = frozenset({
    "visible_verifier",
    "explicit_command",
    "explicit_metric",
})


def _target_binding_operation(name: str, args: dict[str, Any]) -> str | None:
    operation = _TARGET_BINDING_OPERATIONS.get(name)
    if operation is not None:
        return operation
    if name in _SKILL_LIFECYCLE_TARGET_TOOLS:
        return "review"
    if name in _PROCESS_TARGET_TOOLS:
        return "service" if name == "start_service" else "shell"
    if name == "verify_and_record" and str(args.get("contract_kind") or "") in _VERIFY_RUN_KINDS:
        return "shell"
    # CONDITIONAL, never a static map entry (R1 item 1): delegate_start becomes
    # target-bound only when it explicitly selects an exact skill payload; a
    # plain or retry call keeps its current active-workspace behavior untouched.
    # ONLY the known selector value binds here — any other root value falls
    # through to the handler's TYPED unsupported_root refusal instead of an
    # untyped ValueError from binding construction (gate fix 9).
    if (name == "delegate_start"
            and str(args.get("root") or "").strip() == "skill_payload"
            and not str(args.get("retry_of") or "").strip()):
        return "write"
    return None


def _handler_public_params(handler: Callable[..., Any]) -> list[str]:
    try:
        params = list(inspect.signature(handler).parameters)
    except (TypeError, ValueError):
        return []
    return [name for name in params if name not in {"ctx", "_resolved_binding"}]


def _entry_public_params(entry: "ToolEntry") -> list[str]:
    try:
        params = entry.schema.get("parameters") or {}
        props = params.get("properties")
        if isinstance(props, dict):
            return [str(name) for name in props]
    except Exception:
        pass
    return _handler_public_params(entry.handler)


def _entry_has_public_param_schema(entry: "ToolEntry") -> bool:
    try:
        params = entry.schema.get("parameters") or {}
        return isinstance(params.get("properties"), dict)
    except Exception:
        return False


def _normalize_tool_call_args(entry: "ToolEntry", args: dict[str, Any]) -> None:
    tool_name = entry.name
    accepted = set(_entry_public_params(entry))
    aliases: dict[str, str] = {}
    aliases.update(_TOOL_ARG_ALIASES.get("*", {}))
    aliases.update(_TOOL_ARG_ALIASES.get(tool_name, {}))
    for alias, canonical in aliases.items():
        if alias in args and canonical in accepted and alias not in accepted and canonical not in args:
            args[canonical] = args.pop(alias)
    if tool_name in _IGNORE_ROOT_ARG_TOOLS and "root" in args and "root" not in accepted:
        args.pop("root", None)


def _prepare_public_builtin_args(entry: "ToolEntry", args: dict[str, Any]) -> str:
    """Normalize and validate only the model-visible builtin argument surface.

    This runs after capability/lineage availability checks but before path
    normalization, target selection, Python predispatch, or target-sensitive
    guards. Private dispatch carriers therefore cannot be supplied by the model
    and invalid public calls cannot trigger target work before rejection.
    """

    _normalize_tool_call_args(entry, args)
    public_params = set(_entry_public_params(entry))
    # A handler may name a bounded set of execution-only legacy parameters.  They
    # remain absent from its model-visible schema and are therefore usable only by
    # callers replaying the former wire shape through this real registry path.  The
    # handler still owns deterministic migration/refusal; this generic seam neither
    # chooses a route nor special-cases a tool name.
    hidden_legacy = {
        str(name)
        for name in (getattr(entry.handler, "_hidden_legacy_params", ()) or ())
        if str(name)
    }
    accepted_params = public_params | hidden_legacy
    if _entry_has_public_param_schema(entry) and any(key not in accepted_params for key in args):
        return _format_tool_arg_error(
            entry,
            rejected=tuple(sorted(
                str(key)
                for key in args
                if key not in accepted_params and not str(key).startswith("_")
            )),
        )
    try:
        inspect.signature(entry.handler).bind(object(), **args)
    except TypeError:
        return _format_tool_arg_error(entry)
    return ""


def _build_builtin_target_binding(ctx: Any, name: str, args: dict[str, Any]) -> Any:
    """Build the one private physical-target carrier for a builtin call."""

    operation = _target_binding_operation(name, args)
    if operation is None:
        return None
    if name in _SKILL_LIFECYCLE_TARGET_TOOLS:
        return _registry().build_resolved_resource_binding(
            ctx,
            root="skill_payload",
            operation="review",
            path=".",
            skill_name=str(args.get("skill") or ""),
        )
    if name in _PROCESS_TARGET_TOOLS or name == "verify_and_record":
        return _registry().build_resolved_resource_binding(
            ctx,
            operation=operation,
            process_cwd=str(args.get("cwd") or ""),
            bucket=str(args.get("bucket") or ""),
            skill_name=str(args.get("skill_name") or ""),
        )
    if name == "delegate_start":
        return _registry().build_resolved_resource_binding(
            ctx,
            root=str(args.get("root") or ""),
            operation="write",
            path=".",
            bucket=str(args.get("bucket") or ""),
            skill_name=str(args.get("skill_name") or ""),
        )
    root = str(args.get("root") or "active_workspace")
    if name in {"edit_batch", "apply_patch"} and root not in {"active_workspace", "system_repo"}:
        # The repo-only handler owns this typed argument refusal. Resolving an
        # unsupported payload first would ask for selectors it cannot accept.
        return None
    bucket = str(args.get("bucket") or "")
    skill_name = str(args.get("skill_name") or "")

    def _one(path: str) -> Any:
        return _registry().build_resolved_resource_binding(
            ctx,
            root=root,
            operation=operation,
            path=path or ".",
            bucket=bucket,
            skill_name=skill_name,
        )

    if name == "write_file" and args.get("files"):
        return tuple(
            _one(str(item.get("path") or ""))
            for item in args.get("files") or []
            if isinstance(item, dict)
        )
    if name == "apply_patch":
        from ouroboros.tools.edit_ops import patch_target_paths

        return tuple(_one(path) for path in patch_target_paths(str(args.get("patch") or "")))
    if name == "edit_batch":
        return tuple(
            _one(str(item.get("path") or ""))
            for item in args.get("edits") or []
            if isinstance(item, dict)
        )
    return _one(str(args.get("path") or "."))


def _binding_items(binding: Any) -> tuple[Any, ...]:
    if binding is None:
        return ()
    return binding if isinstance(binding, tuple) else (binding,)


def _binding_set_targets_system_repo(ctx: Any, binding: Any) -> bool:
    items = _binding_items(binding)
    return bool(items) and all(_registry().binding_targets_system_repo(ctx, item) for item in items)


def _binding_set_is_light_restricted(ctx: Any, binding: Any) -> bool:
    """Whether light mode must treat this file/VCS target as internal state."""
    items = _binding_items(binding)
    return bool(items) and all(
        _registry().binding_targets_system_repo(ctx, item)
        or (item.root == "runtime_data" and item.source == "runtime_data")
        for item in items
    )


def _light_binding_failure_redirect(name: str, args: dict[str, Any]) -> str:
    """Project an existing light-mode UX redirect after a failed target bind."""

    try:
        from ouroboros.config import get_runtime_mode

        if get_runtime_mode() == "light":
            return _registry().light_cognitive_or_root_redirect(name, args) or ""
    except Exception:
        pass
    return ""


def _light_binding_failure_result(
    name: str,
    args: dict[str, Any],
) -> str | ToolResult | None:
    """Retain cognitive text while typing the structurally distinct root redirect."""

    redirect = _light_binding_failure_redirect(name, args)
    if not redirect:
        return None
    try:
        root = _registry().normalize_root(str(args.get("root") or "active_workspace"))
    except ValueError:
        root = "active_workspace"
    if root == "active_workspace":
        # This branch is reached only for the user_files redirect (the cognitive
        # one needs root=runtime_data), so the code names the demanded root: the
        # recovery walk credits the retry only against the root it names.
        return ToolResult(
            status="blocked",
            code="ROOT_REQUIRED_USER_FILES",
            text=redirect,
        )
    return redirect


def _binding_error_text(name: str, root: str, exc: Exception) -> str | ToolResult:
    detail = str(exc)
    if detail.startswith("SKILL_REDIRECT_BLOCKED:"):
        return f"⚠️ {detail}"
    if detail.startswith("profile=") and " cannot " in detail:
        return ToolResult(
            status="blocked",
            code="ACCESS_BLOCKED",
            text=f"⚠️ TOOL_ACCESS_BLOCKED: {detail.rstrip('.')}.",
        )
    if isinstance(exc, _registry().UserFilesPathBlockedError) and name in {
        "read_file", "list_files", "search_code",
    }:
        return f"⚠️ USER_FILES_PATH_BLOCKED: {detail}"
    if root == "skill_payload" and name in {"write_file", "edit_text"}:
        return f"⚠️ SKILL_PAYLOAD_ARG_ERROR: {detail}"
    prefixes = {
        "read_file": "READ_FILE_ERROR",
        "list_files": "LIST_FILES_ERROR",
        "search_code": "SEARCH_ERROR",
        "query_code": "TOOL_ARG_ERROR (query_code)",
        "write_file": "WRITE_FILE_ERROR",
        "edit_text": "EDIT_TEXT_ERROR",
        "vcs_status": "GIT_ERROR",
        "vcs_diff": "GIT_ERROR",
        "vcs_pull_ff": "PULL_ERROR",
        "vcs_restore": "RESTORE_ERROR",
        "vcs_revert": "REVERT_ERROR",
        "skill_review": "SKILL_REVIEW_ERROR",
        "skill_preflight": "SKILL_PREFLIGHT_ERROR",
        "submit_skill_to_hub": "SUBMIT_BLOCKED",
        "run_command": "SHELL_CWD_BLOCKED",
        "run_script": "SCRIPT_CWD_BLOCKED",
        "start_service": "SHELL_CWD_BLOCKED",
        "verify_and_record": "VERIFY_ERROR",
    }
    text = f"⚠️ {prefixes.get(name, 'TOOL_ERROR')}: {type(exc).__name__}: {detail}"
    if name == "query_code":
        return ToolResult(status="error", code="TOOL_ARG_ERROR", text=text)
    if name in {"vcs_status", "vcs_diff"}:
        return ToolResult(status="ok", code="GIT_ERROR", text=text)
    if name not in prefixes:
        return ToolResult(status="error", code="TOOL_ERROR", text=text)
    return text


def _format_tool_arg_error(entry: "ToolEntry", *, rejected: tuple[str, ...] = ()) -> str:
    params = _entry_public_params(entry)
    accepted = ", ".join(params) if params else "none"
    # Naming the refused key is the actionable half of the repair hint; a
    # signature-bind refusal cannot name one, and a PRIVATE dispatch carrier is
    # never echoed back.
    named = f"unsupported argument(s): {', '.join(rejected)}. " if rejected else ""
    return (
        f"⚠️ TOOL_ARG_ERROR ({entry.name}): invalid arguments for {entry.name}. "
        f"{named}Accepted parameters: {accepted}."
    )


def _resolve_python_predispatch(
    registry: Any,
    name: str,
    args: Dict[str, Any],
    runtime_mode: str,
    effective_constraint: Any,
    resolved_binding: Any = None,
) -> tuple[Dict[str, Any], Any, str | ToolResult | None]:
    """Resolve an exact python/python3 request ONCE, before the shell guard.

    Every downstream guard and the handler therefore see byte-identical
    argv; launchers must not select an interpreter after this boundary.
    Python never EXECUTES a candidate; node probes post-gates instead.
    """
    args, python_resolution = resolve_process_python(
        registry._ctx,
        name,
        args,
        runtime_mode=runtime_mode,
        effective_constraint=effective_constraint,
        resolved_binding=resolved_binding,
    )
    record_interpreter_resolution(registry._ctx, python_resolution)
    if python_resolution is not None and python_resolution.error_reason:
        if python_resolution.error_reason == "cwd_resolution_failed":
            # The failure is the CWD CONFINEMENT policy, not interpreter
            # availability: the resolver could not prove the working directory
            # is inside an allowed root, so no interpreter question was ever
            # reachable. Reuse the canonical shell-CWD denial so the agent gets
            # the actionable root list instead of a misleading "interpreter
            # unavailable" (which twice sent agents hunting for python instead
            # of fixing cwd). The resolver refused the LAUNCH for a policy
            # reason, and the typed SHELL_CWD_BLOCKED status lands in the
            # policy-denial family instead of degrading execution.
            return args, python_resolution, _registry().shell_cwd_block_message(
                registry._ctx,
                str((args or {}).get("cwd") or ""),
                operation="service" if name == "start_service" else "shell",
            )
        return args, python_resolution, ToolResult(
            status="unavailable",
            code="CAPABILITY_UNAVAILABLE",
            text=(
                "⚠️ PYTHON_INTERPRETER_UNAVAILABLE: Ouroboros could not prove "
                "the target interpreter for this launch surface "
                f"({python_resolution.error_reason}). The process was not started."
            ),
            meta={"reason": python_resolution.error_reason},
        )
    return args, python_resolution, None
