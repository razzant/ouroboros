"""Argument normalization and physical target binding for tool dispatch.

The facade re-exports these definitions so existing imports and monkeypatch
targets retain the same bindings.
"""

from __future__ import annotations

import inspect
import json
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
    at import time.
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
        candidate = active()
        path = _coerce_real_path(candidate)
        if path is not None:
            return path

    workspace_root = getattr(ctx, "workspace_root", None)
    workspace_path = _coerce_real_path(workspace_root)
    if workspace_path is not None:
        workspace_mode = str(getattr(ctx, "workspace_mode", "") or "").strip()
        if workspace_mode:
            return workspace_path

    from ouroboros.tool_access import folderless_scratch_dir, project_room_lens_dir

    room = project_room_lens_dir(ctx)
    if room is not None:
        return room
    return folderless_scratch_dir(ctx) or pathlib.Path(getattr(ctx, "repo_dir"))


def system_repo_dir_for(ctx: Any) -> pathlib.Path:
    """Return the Ouroboros system repo root, not an external active workspace."""

    return pathlib.Path(getattr(ctx, "system_repo_dir", None) or getattr(ctx, "repo_dir"))


_PATH_NORMALIZED_TOOLS = frozenset({"read_file", "write_file", "edit_text", "apply_patch", "edit_batch", "list_files", "search_code", "query_code"})
_ROOT_SELECTED_READ_TOOLS = frozenset({"read_file", "list_files", "search_code"})


_ROOT_ARG_REPO_WRITE_TOOLS = frozenset({"write_file", "edit_text", "apply_patch", "edit_batch"})
# Every editor retains the same explicit wrong-root redirect before a write.
_TOP_LEVEL_PATH_WRITE_TOOLS = _ROOT_ARG_REPO_WRITE_TOOLS


@dataclass(frozen=True)
class _DispatchPathNormalization:
    """Exact dispatch note plus any explicit root required before dispatch."""

    text: str = ""
    required_root: Literal["active_workspace", "mixed"] | None = None


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


def protected_repo_write_paths(ctx: Any, name: str, args: Dict[str, Any],
                               binding: Any, *, workspace_mode: bool,
                               acting_system_worktree: bool) -> list[str]:
    """Only physical repo members of a mixed binding reach the protected gate."""
    if binding is not None:
        from ouroboros.tool_access import path_is_relative_to

        repo = system_repo_dir_for(ctx)
        return [pathlib.Path(item.target_path).relative_to(repo).as_posix()
                for item in _binding_items(binding)
                if item.target_path is not None
                and path_is_relative_to(pathlib.Path(item.target_path), repo)]
    root = str(args.get("root") or "active_workspace")
    if (workspace_mode and not acting_system_worktree) or root not in {"active_workspace", "system_repo"}:
        return []
    from ouroboros.tool_access import canonical_repo_relative_path

    return [canonical_repo_relative_path(ctx, root, path)
            for path in _payload_write_paths(name, args)]


def _root_containing_absolute_path(ctx: Any, name: str, text: str) -> str:
    """Owner 7A: the root that physically contains an absolute path given WITHOUT
    a root — among the roots THIS profile may use for the tool's operation plus
    the actor's lineage task roots — or "" when none holds it. The deepest
    containing base wins (``runtime_data`` holds the task roots, the owner home
    holds Deliverables); the path itself is never rewritten, so the `01aea0663`
    mirror stays closed. The file-tool twin of ``tool_access._select_process_target``."""
    if not _registry().is_absolute_path_text(text):
        return ""
    from ouroboros.tool_access import (
        active_tool_profile, decide_tool_access, lineage_read_base,
        path_is_relative_to, profile_readable_root_paths,
    )

    operation = _TARGET_BINDING_OPERATIONS[name]
    try:
        target = pathlib.Path(text).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return ""
    holders = [(label, base) for label, base in profile_readable_root_paths(ctx, operation=operation)
               if path_is_relative_to(target, base)]
    profile = active_tool_profile(ctx)
    for label in ("task_drive", "artifact_store"):
        if decide_tool_access(profile=profile, root=label, operation=operation).allow:
            base = lineage_read_base(ctx, label, target)
            if base is not None:
                holders.append((label, base))
    return max(holders, key=lambda item: len(str(item[1])), default=("", None))[0]


def inferred_file_root(ctx: Any, name: str, root: str | None, paths: list[str]) -> tuple[str, str]:
    """Use the existing physical root selector for an omitted file root."""
    if root:
        return root, ""
    selected = {
        _root_containing_absolute_path(ctx, name, path) or "active_workspace"
        for path in paths
    }
    if len(selected) > 1:
        return "", "⚠️ TOOL_ARG_ERROR: paths infer different roots; pass an explicit common root or use separate calls."
    return next(iter(selected), "active_workspace"), ""


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
    and all file editors return an actionable
    ROOT_REQUIRED_ACTIVE_WORKSPACE redirect instead of a generic access denial.
    The destination root still passes every downstream gate (profile access
    decision, protected-path guards, subagent filters) — only the label is
    corrected, never the authority. ``query_code`` is excluded: its
    root=user_files external-target contract handles absolute paths natively.
    Owner 7A: an absolute path with NO ``root`` runs under the permitted root
    that physically contains it (``_root_containing_absolute_path``), the path
    itself untouched; a NAMED root is never re-rooted."""
    if name not in _PATH_NORMALIZED_TOOLS:
        return _DispatchPathNormalization()
    root_arg = str(args.get("root") or "")
    if not root_arg and name in _ROOT_ARG_REPO_WRITE_TOOLS:
        paths = _payload_write_paths(name, args)
        selected = {
            _root_containing_absolute_path(ctx, name, path) or "active_workspace"
            for path in paths
        }
        if len(selected) > 1:
            return _DispatchPathNormalization(
                text="⚠️ TOOL_ARG_ERROR: paths infer different roots; pass an explicit common root or use separate calls.",
                required_root="mixed",
            )
        if selected and next(iter(selected)) != "active_workspace":
            args["root"] = root_arg = next(iter(selected))
    if not root_arg and name in _ROOT_SELECTED_READ_TOOLS:
        selected = _root_containing_absolute_path(ctx, name, str(args.get("path") or ""))
        if selected and selected != "active_workspace":
            args["root"] = root_arg = selected
    root_arg = root_arg or "active_workspace"
    if root_arg in ("active_workspace", "system_repo"):
        try:
            from ouroboros.body_candidate import serving_alias_root

            norm_root = active_repo_dir_for(ctx) if root_arg == "active_workspace" else system_repo_dir_for(ctx)
            # A bound candidate keeps the serving spelling of a body path (`repo/x`, the
            # absolute serving path) naming the same file it named on the binding write.
            serving_alias = serving_alias_root(ctx, norm_root)

            def _normalize(text: str) -> str:
                text = text.strip().replace("\\", "/")
                relative = _registry().normalize_root_relative(norm_root, text)
                if serving_alias is not None and relative == text:
                    relative = _registry().normalize_root_relative(serving_alias, text)
                return relative

            for _key in ("path", "dir"):
                if isinstance(args.get(_key), str) and args[_key]:
                    args[_key] = _normalize(args[_key])
            for entries in (args.get("files"), args.get("edits")):
                for _f in entries if isinstance(entries, list) else []:
                    if isinstance(_f, dict) and isinstance(_f.get("path"), str) and _f["path"]:
                        _f["path"] = _normalize(_f["path"])
            if name == "apply_patch" and isinstance(args.get("patch"), str):
                from ouroboros.tools.edit_ops import normalize_patch_paths

                args["patch"] = normalize_patch_paths(args["patch"], _normalize)
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
    for entries in (args.get("files"), args.get("edits")):
        for _f in entries if isinstance(entries, list) else []:
            if isinstance(_f, dict) and isinstance(_f.get("path"), str) and _f["path"]:
                candidates.append(_f["path"])
    if name == "apply_patch" and isinstance(args.get("patch"), str):
        from ouroboros.tools.edit_ops import patch_target_paths

        candidates.extend(patch_target_paths(args["patch"]))
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


# Empty: the commit tools declare ``root`` and refuse every root but the system
# repository themselves; the name stays for ``registry.py``'s re-export.
_IGNORE_ROOT_ARG_TOOLS: frozenset[str] = frozenset()


_GENERIC_VCS_TARGET_TOOLS = frozenset({
    "vcs_status",
    "vcs_diff",
    "vcs_pull_ff",
    "vcs_restore",
    "vcs_revert",
    "review_change",
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
    # ONLY a COMPLETE known selector binds here — any other root value or an
    # incomplete selector falls through to the handler's TYPED unsupported_root /
    # payload_selector_incomplete refusal instead of an untyped ValueError from
    # binding construction (gate fix 9, #1304).
    if (name == "delegate_start"
            and str(args.get("root") or "").strip() == "skill_payload"
            and str(args.get("bucket") or "").strip() and str(args.get("skill_name") or "").strip()
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
    if tool_name == "delegate_start" and str(args.get("root") or "").strip() == "active_workspace":
        # #1304: the schema's documented default IS omission (the #882 rule), removed
        # before the configured-session selector check and the payload binder read it.
        args.pop("root")


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
    bucket = str(args.get("bucket") or "")
    skill_name = str(args.get("skill_name") or "")

    def _one(path: str, selected_operation: str = operation) -> Any:
        return _registry().build_resolved_resource_binding(
            ctx,
            root=root,
            operation=selected_operation,
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
        from ouroboros.tools.edit_ops import _parse_patch

        ops, error = _parse_patch(str(args.get("patch") or ""))
        return tuple(_one(op.path, "edit" if op.kind == "update" else "write") for op in ops) if not error else ()
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


def editor_round_has_disjoint_effects(ctx: Any, calls: list[dict[str, Any]]) -> bool:
    """Preflight physical file and shared publication effects for editor-only rounds."""
    editors = {"edit_text", "edit_batch", "apply_patch"}
    if len(calls) < 2 or any(str(c.get("function", {}).get("name") or "") not in editors for c in calls):
        return False
    seen: set[tuple[Any, ...]] = set()
    for call in calls:
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        try:
            raw = function.get("arguments") or "{}"
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            if not isinstance(args, dict):
                return False
            if _normalize_dispatch_path_args_result(ctx, name, args).required_root:
                return False
            paths = _payload_write_paths(name, args)
            bindings = _binding_items(_build_builtin_target_binding(ctx, name, args))
            if not paths or len(paths) != len(bindings):
                return False
            effects: set[tuple[Any, ...]] = set()
            for binding in bindings:
                target = pathlib.Path(binding.target_path)
                effects.add(("path", str(target.resolve(strict=False))))
                try:
                    stat = target.stat()
                    effects.add(("inode", stat.st_dev, stat.st_ino))
                except OSError:
                    pass
                # The shared artifact manifest and skill revision are one
                # effect even when two paths are physically disjoint.
                if binding.root == "user_files":
                    effects.add(("user-artifacts", str(binding.state_drive_root)))
                if binding.skill_name:
                    effects.add(("skill-revision", str(binding.state_drive_root), binding.skill_name))
                effects.add(("workspace-output-name", target.name))
            if effects & seen:
                return False
            seen.update(effects)
        except Exception:  # Footprint preflight is advisory; actual dispatch owns typed refusals.
            return False
    return True


def _binding_set_targets_system_repo(ctx: Any, binding: Any) -> bool:
    items = _binding_items(binding)
    return bool(items) and any(_registry().binding_targets_system_repo(ctx, item) for item in items)


def _user_files_binding_reaches_repo(ctx: Any, binding: Any) -> bool:
    """Whether a ``user_files`` target physically lands inside the Ouroboros repo.

    ``binding_targets_system_repo`` compares the SELECTED ROOT's base, and
    ``user_files`` resolves to the owner's home (the whole host on a cyber_pro
    install), which can CONTAIN the repo. A repository path reached under that
    name therefore answers no there while still being self-repo mutation, so the
    light gate reads the resolved target instead of the root name. Only
    ``user_files`` needs this: every other root's base is a data/payload
    location the gate already classifies correctly.
    """
    from ouroboros.tool_access import path_is_relative_to

    repos = {system_repo_dir_for(ctx), pathlib.Path(getattr(ctx, "serving_repo_dir", None) or system_repo_dir_for(ctx))}
    return any(
        item.root == "user_files" and item.target_path is not None
        and any(path_is_relative_to(pathlib.Path(item.target_path), repo) for repo in repos)
        for item in _binding_items(binding)
    )


def _binding_set_is_light_restricted(ctx: Any, binding: Any) -> bool:
    """Whether light mode must treat this file/VCS target as internal state."""
    items = _binding_items(binding)
    return bool(items) and any(
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


def delegate_payload_binding_refusal(ctx: Any, exc: Exception) -> ToolResult:
    """A complete skill-payload selector the binder could not resolve (#1304): a typed,
    definite no-run that names the selector and its repair, plus the one durable
    START_BLOCKED attempt row a pre-custody refusal owes (D5)."""
    from ouroboros.delegate_evidence import record_start_blocked
    from ouroboros.delegate_shared import _fail

    record_start_blocked(ctx, str(getattr(ctx, "task_id", "") or ""), "payload_selector_unresolved")
    return _fail("delegate_start", "payload_selector_unresolved",
                 f"root='skill_payload' with this bucket/skill_name selects no payload you can write: {exc}. "
                 "Name an installed skill exactly, or omit root (root='active_workspace') for ordinary "
                 "workspace delegation.", definitely_unrun=True)


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
    if root == "skill_payload" and name in {"write_file", "edit_text", "edit_batch", "apply_patch"}:
        return f"⚠️ SKILL_PAYLOAD_ARG_ERROR: {detail}"
    prefixes = {
        "read_file": "READ_FILE_ERROR",
        "list_files": "LIST_FILES_ERROR",
        "search_code": "SEARCH_ERROR",
        "query_code": "TOOL_ARG_ERROR (query_code)",
        "write_file": "WRITE_FILE_ERROR",
        "edit_text": "EDIT_TEXT_ERROR",
        "edit_batch": "EDIT_BATCH_ERROR",
        "apply_patch": "APPLY_PATCH_ERROR",
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
    hint = (" Use cwd=system_repo or cwd=system_repo/subdir, not root."
            if "root" in rejected and entry.name in {"run_command", "run_script", "start_service"} else "")
    return (
        f"⚠️ TOOL_ARG_ERROR ({entry.name}): invalid arguments for {entry.name}. "
        f"{named}Accepted parameters: {accepted}.{hint}"
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
