"""Tool registry execution authority: load modules, expose schemas, dispatch safely.

The class body is the parent's tip bytes (D04 remainder), re-homed by the F3.1
typed-organ lane; the parent ``ouroboros/tools/registry.py`` stays as the
compatibility facade and re-exports every moved name, so historical imports and
monkeypatch targets keep working unchanged. The D02 deltas carried here are the
typed dispatch path (``execute_result``/``_execute_legacy_text`` + the
``LegacyTextResultAdapter`` boundary) and the first-party/scoped duplicate-name
hard-fail with visible extension/MCP collision evidence.
"""

from __future__ import annotations

import copy
import inspect
import logging
import pathlib
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional

import ouroboros.tools.extension_dispatch as extension_dispatch
import ouroboros.tools.registry_guard_process as registry_guard_process
import ouroboros.tools.registry_guards as registry_guards
import ouroboros.tools.shell_guards as shell_guards
import ouroboros.tools.tool_resolution as tool_resolution
from ouroboros.runtime_mode_policy import (
    mode_allows_protected_write,
    protected_paths_in,
    protected_write_block_message,
)
from ouroboros.tool_capabilities import (
    ACTING_SUBAGENT_MODE,
    ACTING_SUBAGENT_TOOL_NAMES,
    CORE_TOOL_NAMES,
    LOCAL_READONLY_SUBAGENT_MODE,
    LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
    META_TOOL_NAMES,
)
from ouroboros.tool_access import (
    active_tool_profile,
    build_resolved_resource_binding,
    canonical_repo_relative_path,
    decide_tool_access,
    light_cognitive_or_root_redirect,
    _path_is_relative_to_casefold,
    shell_cwd_block_message,
    resource_root_path,
    user_files_path_block_reason,
    workspace_mode_block_reason,
)
from ouroboros.tools.deliverables_shell import lexical_user_files_block_reason
from ouroboros.tools.tool_catalog import (
    DuplicateToolNameError as _DuplicateToolNameError,
    ToolCatalog as _ToolCatalog,
    ToolEntry,
    partition_shadowed_tools as _partition_shadowed_tools,
)
from ouroboros.tools.tool_context import ToolContext
from ouroboros.tools.tool_resolution import (
    _GENERIC_VCS_TARGET_TOOLS,
    _binding_set_is_light_restricted,
    _binding_set_targets_system_repo,
    _build_builtin_target_binding,
    _target_binding_operation,
    active_repo_dir_for,
    system_repo_dir_for,
)
from ouroboros.tools.tool_result import (
    LegacyTextResultAdapter,
    ToolResult,
    _TOOL_RESULT_ATTR,
    _compose_execute_result_result,
    _install_tool_result_sidecar,
    _published_tool_result,
    _restore_tool_result_sidecar,
)
from ouroboros.tools.registry_guards import (
    _EPHEMERAL_ALLOWED_TOOLS,
    _builtin_tool_availability,
    _disabled_tools,
    _resource_allowed,
)
from ouroboros.contracts.task_constraint import VALID_WRITE_SURFACES, normalize_task_constraint

# The logger name is pinned to the parent's literal namespace so the re-homing
# does not silently rename the log stream.
log = logging.getLogger("ouroboros.tools.registry")


_PROCESS_COMMAND_TOOLS = frozenset({"run_command", "run_script", "start_service"})
# verify_and_record runs the agent's declared `check` like a command, so it must clear the
# same PRE-EXECUTION shell guards (subagent-secret read, protected-artifact read, sudo,
# protected-root / workspace-state / light-mode writes) — that pre-exec filter is the
# security boundary and blocks a forbidden mutation BEFORE the handler runs, so a guarded
# check cannot mutate protected state and then leave a host-attested PASS receipt. It is
# deliberately NOT in _PROCESS_COMMAND_TOOLS: those POST-execution checks (owner-file
# restore, light-repo diff, git-ref tripwire) run AFTER the handler has already written the
# receipt, so they would only annotate the returned text, not gate the durable receipt —
# adding them would give false assurance while the pre-exec guards already do the gating.
_SHELL_GUARDED_TOOLS = _PROCESS_COMMAND_TOOLS | {"verify_and_record"}


def _shell_guard_required(name: str, args: Dict[str, Any]) -> bool:
    """Keep the command-less actor zero-run receipt outside shell-CWD guards."""

    return name in _SHELL_GUARDED_TOOLS and not (
        name == "verify_and_record"
        and str(args.get("contract_kind") or "").strip() == "delegation_zero_run"
    )


_REPO_MUTATION_TOOLS = frozenset({
    "write_file",
    "commit_reviewed",
    "vcs_commit_reviewed",
    "edit_text",
    "apply_patch",
    "edit_batch",
    "vcs_revert",
    "vcs_pull_ff",
    "vcs_restore",
    "vcs_rollback",
    "promote_to_stable",
    # PR integration tools mutate the local worktree/refs.
    "fetch_pr_ref",
    "create_integration_branch",
    "cherry_pick_pr_commits",
    "stage_adaptations",
    "stage_pr_merge",
})
_SYSTEM_INTRINSIC_REPO_MUTATION_TOOLS = frozenset({
    "commit_reviewed",
    "vcs_commit_reviewed",
    "vcs_rollback",
    "promote_to_stable",
    "fetch_pr_ref",
    "create_integration_branch",
    "cherry_pick_pr_commits",
    "stage_adaptations",
    "stage_pr_merge",
})


def _presence_tool_allowed(ctx: Any, name: str) -> bool:
    """Positive ceiling for host-admitted presence work; absent is byte-compatible."""

    from ouroboros.presence_authority import (
        presence_ceiling_allows_tool,
        presence_ceiling_from_context,
    )

    ceiling = presence_ceiling_from_context(ctx)
    if name in {"presence_finish", "presence_cancel_work"}:
        return ceiling is not None
    return ceiling is None or presence_ceiling_allows_tool(ceiling, name)


def _presence_binding_allowed(ctx: Any, binding: Any) -> bool:
    from ouroboros.presence_authority import (
        presence_ceiling_allows_binding,
        presence_ceiling_from_context,
    )

    ceiling = presence_ceiling_from_context(ctx)
    if ceiling is None or binding is None:
        return True
    items = binding if isinstance(binding, tuple) else (binding,)
    return bool(items) and all(presence_ceiling_allows_binding(ceiling, item) for item in items)


def _presence_bound_args(ctx: Any, name: str, args: Any) -> tuple[dict[str, Any], str]:
    try:
        from ouroboros.presence_authority import apply_presence_argument_bindings

        bound = apply_presence_argument_bindings(ctx, name, dict(args or {}))
        if not _presence_tool_allowed(ctx, name):
            return {}, (
                "⚠️ PRESENCE_CAPABILITY_BLOCKED: "
                f"{name!r} is outside this presence task's positive capability ceiling."
            )
        return bound, ""
    except Exception as exc:
        return {}, f"⚠️ PRESENCE_ARGUMENT_BINDING_BLOCKED: {exc}"


def _configured_delegate_selector(ctx: Any, name: str, args: dict[str, Any]) -> bool:
    """Configured-actor validation PRECEDES generic target binding.

    A configured session child passing any exact-resource selector gets the
    handler's precise configured_actor_resource_mismatch (which also records
    the START_BLOCKED attempt row), not the generic payload-binding
    TOOL_ACCESS_BLOCKED that twice read as "workspace authority lost" and
    pushed the nanny into native rebuilds."""
    return (
        name == "delegate_start"
        and isinstance(getattr(ctx, "_configured_actor_bootstrap", None), dict)
        and not str(args.get("retry_of") or "").strip()
        and any(str(args.get(key) or "").strip()
                for key in ("root", "bucket", "skill_name"))
    )


_ACTING_NO_WORKSPACE_REPO_RESULT = ToolResult(
    status="blocked",
    code="ACCESS_BLOCKED",
    text=(
        "⚠️ ACTING_NO_WORKSPACE_BLOCKED: this acting subagent has no resolved isolated "
        "workspace; write only to root=task_drive, root=artifact_store, or root=user_files. "
        "active_workspace/system_repo map to the live Ouroboros repo and are blocked."
    ),
)
_ACTING_NO_WORKSPACE_PROCESS_RESULT = ToolResult(
    status="blocked",
    code="ACCESS_BLOCKED",
    text=(
        "⚠️ ACTING_NO_WORKSPACE_BLOCKED: shell/coding/service/integration tools need an "
        "isolated workspace (their default target is the live repo). Schedule a self_worktree "
        "/ external_workspace child for that work."
    ),
)
_LIGHT_START_SERVICE_RESULT = ToolResult(
    status="blocked",
    code="LIGHT_MODE_BLOCKED",
    text="⚠️ LIGHT_MODE_BLOCKED: runtime_mode=light refuses start_service against the Ouroboros repository because long-running services can mutate after initial tool checks. For external services, set cwd under user_files, task_drive, or artifact_store; switch to advanced/pro only for reviewed Ouroboros self-modification.",
)



def _unknown_tool_result(entries: Dict[str, Any], name: str, extension_unavailable: bool) -> str | ToolResult:
    """The unknown-name answer, typed EXTENSION_UNAVAILABLE for a dead extension.

    A registered extension name whose payload is NOT live is a distinct fact
    from an unknown name (the D02 liveness bit), so it carries a typed code
    instead of a nameless text; a truly unknown name keeps the legacy text.
    """
    text = f"⚠️ Unknown tool: {name}. Available: {', '.join(sorted(n for n, e in entries.items() if not e.alias_for))}"
    if extension_unavailable:
        return ToolResult(
            status="unavailable",
            code="EXTENSION_UNAVAILABLE",
            text=text,
            meta={"dynamic_provider": True},
        )
    return text


def _protected_write_block_result(*, path: str, runtime_mode: str, action: str) -> ToolResult:
    return ToolResult(
        status="blocked",
        code="CORE_PROTECTION_BLOCKED",
        text=protected_write_block_message(
            path=path,
            runtime_mode=runtime_mode,
            action=action,
        ),
    )


class ToolRegistry:
    """Tool registry; modules export ``get_tools()``."""

    def __init__(self, repo_dir: pathlib.Path, drive_root: pathlib.Path):
        self._entries: Dict[str, ToolEntry] = {}
        self._ctx = ToolContext(repo_dir=repo_dir, drive_root=drive_root)
        # Load-time omissions are FACTS OF THIS PROCESS (a tool module that failed
        # to import stays missing until restart); schemas() rebuilds start from
        # them instead of an empty list so the ledger never forgets them (H3).
        self._module_load_omissions: List[Dict[str, Any]] = []
        self._capability_omissions: List[Dict[str, Any]] = []
        self._base_catalog = self._load_modules()
        self._capability_omissions = [dict(item) for item in self._module_load_omissions]
        self._entries.update(self._base_catalog.entries)
        self._entry_origins = dict(self._base_catalog.origins)
        self._scoped_entries: Dict[str, ToolEntry] = {}
        self._handler_overrides: Dict[str, Callable] = {}

    _FROZEN_TOOL_MODULES = [
        "browser", "ci", "claude_advisory_review", "compact_context", "control",
        "core", "delegate", "edit_ops", "evolution_stats", "followup", "git", "git_pr", "git_rollback", "github",
        "health", "join_ledger", "knowledge", "media", "memory_tools", "plan_review", "project_journal", "presence",
        "recent_tasks",
        "query_code", "review", "search", "services", "shell", "skill_exec", "skill_publish",
        "skill_preflight", "subagent_integration", "task_tree", "tool_discovery", "verify", "vision",
    ]

    def _load_modules(self) -> _ToolCatalog:
        """Load frozen or package-discovered tool modules."""
        import importlib
        import sys

        if getattr(sys, 'frozen', False):
            module_names = self._FROZEN_TOOL_MODULES
        else:
            import pkgutil
            import ouroboros.tools as tools_pkg
            module_names = [
                m for _, m, _ in pkgutil.iter_modules(tools_pkg.__path__)
                if not m.startswith("_") and m != "registry"
            ]

        catalog_entries = []
        for modname in module_names:
            try:
                mod = importlib.import_module(f"ouroboros.tools.{modname}")
                if hasattr(mod, "get_tools"):
                    for index, entry in enumerate(mod.get_tools()):
                        catalog_entries.append(
                            (f"ouroboros.tools.{modname}.get_tools[{index}]", entry)
                        )
            except Exception as exc:
                log.warning(
                    "Failed to load tool module %s", modname, exc_info=True)
                # A failed module silently omits EVERY tool it exports; record it
                # in the durable capability ledger, not only the process log (H3).
                self._module_load_omissions.append({
                    "surface": "tools",
                    "reason": "module_load_failed",
                    "module": modname,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        # Duplicate detection deliberately happens outside the import-degrade
        # boundary: a first-party name collision is a broken catalog, not an
        # optional module import failure that startup may silently omit.
        return _ToolCatalog(catalog_entries)

    def set_context(self, ctx: ToolContext) -> None:
        self._ctx = ctx

    def register(self, entry: ToolEntry, *, origin: str = "") -> None:
        """Register one task-scoped entry without mutating the base catalog."""
        scoped_origin = str(origin or "").strip()
        if not scoped_origin:
            handler = entry.handler
            handler_module = str(getattr(handler, "__module__", "") or "unknown")
            handler_name = str(
                getattr(handler, "__qualname__", "")
                or getattr(handler, "__name__", "")
                or type(handler).__qualname__
            )
            scoped_origin = f"{handler_module}.{handler_name}"
        if entry.name in self._entries:
            raise _DuplicateToolNameError(
                entry.name,
                self._entry_origins.get(entry.name, "unknown"),
                scoped_origin,
            )
        self._scoped_entries[entry.name] = entry
        self._entries[entry.name] = entry
        self._entry_origins[entry.name] = scoped_origin

    # Contract.

    def _ctx_is_delegated_subagent(self) -> bool:
        for attr in ("task_metadata", "task_contract"):
            data = getattr(self._ctx, attr, None)
            if isinstance(data, dict) and str(data.get("delegation_role") or "").strip() == "subagent":
                return True
        return False

    def _is_local_readonly_subagent(self) -> bool:
        tc = normalize_task_constraint(getattr(self._ctx, "task_constraint", None))
        if tc and tc.mode == LOCAL_READONLY_SUBAGENT_MODE:
            return True
        # Fail-closed (mirror active_tool_profile): a valid acting constraint is
        # acting; a malformed acting constraint, or any delegated subagent without
        # a valid acting constraint (incl. a missing constraint), resolves read-only.
        if self._is_acting_subagent():
            return False
        if tc and tc.mode == ACTING_SUBAGENT_MODE:
            return True
        return self._ctx_is_delegated_subagent()

    def _is_acting_subagent(self) -> bool:
        tc = normalize_task_constraint(getattr(self._ctx, "task_constraint", None))
        return bool(
            tc and tc.mode == ACTING_SUBAGENT_MODE
            and str(getattr(tc, "surface", "") or "") in VALID_WRITE_SURFACES
        )

    def _acting_self_worktree(self) -> bool:
        tc = normalize_task_constraint(getattr(self._ctx, "task_constraint", None))
        return bool(
            tc and getattr(tc, "mode", "") == ACTING_SUBAGENT_MODE
            and str(getattr(tc, "surface", "") or "") == "self_worktree"
        )

    def _deliverables_shell_target_allowed(
        self,
        candidate: pathlib.Path,
        *,
        lexical_candidate: pathlib.Path | None = None,
    ) -> bool:
        """Return whether a top-level user-files shell may write this target.

        The workspace shell guard owns the process-root boundary.  This narrow
        exception reuses the user-files policy and the configured Deliverables
        root for the one existing top-level profile that already has
        ``user_files:shell``.  Delegated children never inherit the carve-out.
        """
        if self._is_acting_subagent() or self._is_local_readonly_subagent():
            return False
        profile = active_tool_profile(self._ctx)
        if not decide_tool_access(
            profile=profile,
            root="user_files",
            operation="shell",
        ).allow:
            return False
        try:
            if lexical_user_files_block_reason(lexical_candidate or candidate):
                return False
            target = pathlib.Path(candidate).resolve(strict=False)
            deliverables = resource_root_path(self._ctx, "deliverables")
            # Validate the configured container itself before admitting a child.
            # A root that contains a protected repo/data drive is not a genuine
            # sibling; checking only the final file would otherwise turn its
            # harmless-looking sibling paths into a broad parent escape.
            if user_files_path_block_reason(self._ctx, deliverables):
                return False
            if not (
                target.is_relative_to(deliverables)
                or _path_is_relative_to_casefold(target, deliverables)
            ):
                return False
            try:
                deliverable_binding = build_resolved_resource_binding(
                    self._ctx,
                    root="user_files",
                    operation="shell",
                    path=str(target),
                )
            except (OSError, TypeError, ValueError, RuntimeError):
                return False
            if not _presence_binding_allowed(self._ctx, deliverable_binding):
                return False
            return not user_files_path_block_reason(self._ctx, target)
        except (OSError, TypeError, ValueError, RuntimeError):
            return False

    def _acting_tool_grants(self) -> set:
        tc = normalize_task_constraint(getattr(self._ctx, "task_constraint", None))
        return set(getattr(tc, "external_tool_grants", ()) or ()) if tc else set()

    def _readonly_tool_allowed(self, name: str) -> bool:
        """Return the read-only child allowlist, including the narrow actor receipt.

        ``verify_and_record`` is normally an acting/workspace code tool because its
        other contract kinds execute commands or inspect deliverables.  An
        actor-first configured session needs one exceptional, non-shell use: a
        typed ``delegation_zero_run`` receipt when its selected leaf is still
        pending.  Keep that exception bound to the private host bootstrap marker;
        the handler applies the same check again at execution time.
        """
        if name in LOCAL_READONLY_SUBAGENT_TOOL_NAMES:
            return True
        if name != "verify_and_record" or not self._is_local_readonly_subagent():
            return False
        bootstrap = getattr(self._ctx, "_configured_actor_bootstrap", None)
        return (
            isinstance(bootstrap, dict)
            and not bool(bootstrap.get("physical_started"))
            and not bool(bootstrap.get("zero_run_receipt_recorded"))
            and (
                bool(bootstrap.get("exact_start_pending", True))
                or str(bootstrap.get("zero_run_evidence_status") or "") == "unknown"
            )
        )

    def initial_tool_names(self) -> frozenset[str]:
        if self._is_local_readonly_subagent():
            names = set(LOCAL_READONLY_SUBAGENT_TOOL_NAMES)
            if self._readonly_tool_allowed("verify_and_record"):
                names.add("verify_and_record")
            return frozenset(names)
        if self._is_acting_subagent():
            return ACTING_SUBAGENT_TOOL_NAMES
        return frozenset(set(self.available_tools()) | set(META_TOOL_NAMES))

    def available_tools(self) -> List[str]:
        acting_subagent = self._is_acting_subagent()
        local_readonly_subagent = self._is_local_readonly_subagent()
        disabled = _disabled_tools(self._ctx)
        return [
            e.name
            for e in self._entries.values()
            if not e.alias_for  # compat aliases are callable, never advertised
            if e.name not in disabled  # declarative tool policy (task_contract.disabled_tools)
            if _presence_tool_allowed(self._ctx, e.name)
            if _builtin_tool_availability(e.name, self._ctx)[0]
            if not local_readonly_subagent or self._readonly_tool_allowed(e.name)
            if not acting_subagent or e.name in ACTING_SUBAGENT_TOOL_NAMES
        ]

    def _schema_for_entry(self, entry: ToolEntry) -> Dict[str, Any]:
        schema = entry.schema
        if self._is_local_readonly_subagent():
            if entry.name == "verify_and_record" and self._readonly_tool_allowed(entry.name):
                # The read-only actor is allowed to mint exactly one kind of
                # host receipt.  Do not expose the general verification schema,
                # whose other kinds can execute shell commands or inspect files.
                schema = copy.deepcopy(schema)
                schema["description"] = (
                    "Record the typed zero-run decision for this configured actor-first "
                    "session. No physical leaf may have started. This receipt is the "
                    "only verification available to a read-only actor."
                )
                parameters = schema.setdefault("parameters", {})
                properties = parameters.setdefault("properties", {})
                parameters["properties"] = {
                    key: properties[key]
                    for key in ("contract_kind", "criterion_id", "zero_run_decision", "zero_run_basis")
                    if key in properties
                }
                if "contract_kind" in parameters["properties"]:
                    parameters["properties"]["contract_kind"]["enum"] = ["delegation_zero_run"]
                parameters["required"] = ["contract_kind", "zero_run_decision", "zero_run_basis"]
            elif entry.name in {"read_file", "list_files", "search_code", "query_code"}:
                schema = copy.deepcopy(schema)
                root_schema = schema.get("parameters", {}).get("properties", {}).get("root", {})
                if entry.name == "search_code":
                    allowed = {"active_workspace", "system_repo", "skill_payload"}
                elif entry.name == "query_code":
                    # query_code itself rejects non-repo roots — do not advertise more.
                    allowed = {"active_workspace", "system_repo"}
                else:
                    allowed = {"active_workspace", "system_repo", "runtime_data", "task_drive", "skill_payload", "artifact_store"}
                if isinstance(root_schema.get("enum"), list): root_schema["enum"] = [root for root in root_schema["enum"] if root in allowed]
            elif entry.name in {"browse_page", "browser_action"}:
                schema = copy.deepcopy(entry.schema)
                if entry.name == "browse_page":
                    schema["description"] = "Open an HTTP(S) URL (external, or localhost on non-Ouroboros ports) or a file:// path under your workspace in a headless browser. Returns page content as text, html, markdown, or screenshot (base64 PNG) — use it with analyze_screenshot to visually verify your own built apps. The Ouroboros API ports, private/link-local IPs, and other URL schemes are blocked for subagents. Use viewport to test mobile layouts (e.g. '375x812')."
                if entry.name == "browser_action":
                    schema["description"] = "Perform action on the current browser page (external HTTP(S), localhost on non-Ouroboros ports, or a file:// page under your workspace). Actions: click (selector), fill (selector + value), select (selector + value), screenshot (base64 PNG), scroll (value: up/down/top/bottom). JavaScript evaluate is unavailable to local-readonly subagents."
                    props = schema.get("parameters", {}).get("properties", {})
                    action_schema = props.get("action", {})
                    if isinstance((action_enum := action_schema.get("enum")), list):
                        action_schema["enum"] = [name for name in action_enum if name != "evaluate"]
                    if isinstance((value_schema := props.get("value", {})), dict): value_schema["description"] = "Value for fill/select or direction for scroll"
            elif entry.name == "schedule_subagent":
                # A read-only subagent may delegate read-only children only — hide the
                # acting (mutative) fields so it cannot spawn an acting grandchild.
                schema = copy.deepcopy(schema)
                props = schema.get("parameters", {}).get("properties", {})
                for field in ("write_surface", "write_root", "protected_paths_grant", "external_tool_grants"):
                    props.pop(field, None)
            elif entry.name == "delegate_start":
                # Same selector trap the acting branch hides: a readonly child
                # can never start a skill-payload resource either.
                schema = copy.deepcopy(schema)
                props = schema.get("parameters", {}).get("properties", {})
                for field in ("root", "bucket", "skill_name"):
                    props.pop(field, None)
        elif self._is_acting_subagent():
            # Advertise only what the acting profile can actually execute: writes go
            # ONLY to the isolated surface (active_workspace); reads use the read roots;
            # browser evaluate remains available on the current page; the browser
            # handler retains its owner/self-lowering checks.
            if entry.name == "delegate_start":
                # The exact-resource selector is a TRAP here: an acting child
                # can never write root=skill_payload, yet the schema advertised
                # it as root's only enum value - twice a nanny took the bait,
                # was refused by generic binding before the precise
                # configured_actor_resource_mismatch could answer, and rebuilt
                # the work natively (the two Minecraft-widget incidents). Hide
                # the selector triple; bound starts and retry_of need none of
                # it; the top-level principal's schema is untouched.
                schema = copy.deepcopy(schema)
                props = schema.get("parameters", {}).get("properties", {})
                for field in ("root", "bucket", "skill_name"):
                    props.pop(field, None)
            elif entry.name in tool_resolution._ROOT_ARG_REPO_WRITE_TOOLS or entry.name in _GENERIC_VCS_TARGET_TOOLS:
                schema = copy.deepcopy(schema)
                root_schema = schema.get("parameters", {}).get("properties", {}).get("root", {})
                if isinstance(root_schema.get("enum"), list):
                    root_schema["enum"] = [root for root in root_schema["enum"] if root == "active_workspace"]
            elif entry.name in {"read_file", "list_files", "search_code", "query_code"}:
                # Acting profile reads its own surface + data roots, NOT the live
                # system_repo (no system_repo in _POLICY['acting_subagent']).
                schema = copy.deepcopy(schema)
                root_schema = schema.get("parameters", {}).get("properties", {}).get("root", {})
                allowed = {"active_workspace"} if entry.name in {"search_code", "query_code"} else {"active_workspace", "runtime_data", "task_drive", "artifact_store"}
                if isinstance(root_schema.get("enum"), list):
                    root_schema["enum"] = [root for root in root_schema["enum"] if root in allowed]
        return {"type": "function", "function": schema}

    def _schemas_for_entry(self, entry: ToolEntry) -> List[Dict[str, Any]]:
        return [self._schema_for_entry(entry)]

    def _visible_dynamic_tools(
        self, surface: str, tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        visible, shadowed = _partition_shadowed_tools(tools, self._entries)
        if not shadowed:
            return visible
        collisions = []
        for tool in shadowed:
            name = str(tool.get("name") or "")
            if surface == "extensions":
                dynamic_origin = str(tool.get("skill") or "unknown extension")
            else:
                server_id = str(tool.get("server_id") or "unknown server")
                raw_name = str(tool.get("raw_name") or "unknown tool")
                dynamic_origin = f"{server_id}:{raw_name}"
            collisions.append({
                "name": name,
                "authoritative_origin": self._entry_origins.get(name, "unknown"),
                "dynamic_origin": dynamic_origin,
            })
        collisions.sort(key=lambda item: (item["name"], item["dynamic_origin"]))
        names = sorted({item["name"] for item in collisions})
        log.error(
            "%s tool name collision omitted; authoritative catalog wins: %s",
            surface,
            ", ".join(names),
        )
        self._capability_omissions.append({
            "surface": surface,
            "reason": "name_collision",
            "kind": "registry_shadow",
            "tools": names,
            "collisions": collisions,
        })
        return visible

    def _record_mcp_slug_collisions(self, collisions: List[Dict[str, Any]]) -> None:
        if not collisions:
            return
        rows = [dict(item) for item in collisions]
        rows.sort(key=lambda item: (
            str(item.get("prefixed_name") or ""),
            str(item.get("dropped_raw_name") or ""),
        ))
        names = sorted({str(item.get("prefixed_name") or "") for item in rows})
        self._capability_omissions.append({
            "surface": "mcp",
            "reason": "name_collision",
            "kind": "provider_slug",
            "tools": [name for name in names if name],
            "collisions": rows,
        })

    def schemas(self, core_only: bool = False) -> List[Dict[str, Any]]:
        acting_subagent = self._is_acting_subagent()
        acting_grants = self._acting_tool_grants() if acting_subagent else set()
        local_readonly_subagent = self._is_local_readonly_subagent()
        ephemeral_turn = bool(getattr(self._ctx, "is_ephemeral_turn", False))
        disabled_tools = _disabled_tools(self._ctx)
        # Rebuild from the load-time facts, never from empty: a rebuilt schema
        # list must not erase module_load_failed omissions (H3, capinv-447).
        self._capability_omissions = [dict(item) for item in self._module_load_omissions]
        unavailable_tools = {
            entry.name: detail
            for entry in self._entries.values()
            for available, reason, detail in [_builtin_tool_availability(entry.name, self._ctx)]
            if not available and reason == "missing_credential" and entry.name not in disabled_tools
        }
        built_in = [
            schema
            for entry in self._entries.values()
            if not entry.alias_for  # compat aliases are callable, never advertised
            if entry.name not in disabled_tools  # declarative tool policy (task_contract.disabled_tools)
            if _presence_tool_allowed(self._ctx, entry.name)
            if entry.name not in unavailable_tools
            if not local_readonly_subagent or self._readonly_tool_allowed(entry.name)
            if not acting_subagent or entry.name in ACTING_SUBAGENT_TOOL_NAMES
            if not ephemeral_turn or entry.name in _EPHEMERAL_ALLOWED_TOOLS  # CW3: default-deny allowlist
            for schema in self._schemas_for_entry(entry)
        ]
        if disabled_tools:
            self._capability_omissions.append({"surface": "tools", "reason": "disabled_by_contract", "tools": sorted(disabled_tools)})
        if unavailable_tools:
            self._capability_omissions.append({
                "surface": "tools",
                "reason": "missing_credential",
                "tools": sorted(unavailable_tools),
                "details": {name: unavailable_tools[name] for name in sorted(unavailable_tools)},
            })
        # Include live extension tool schemas in normal tool discovery.
        extension_schemas: List[Dict[str, Any]] = []
        if ephemeral_turn:
            # CW3: a short decision turn answers/routes/spawns/steers only — it gets no
            # extension surfaces, which can have durable/reviewed side effects.
            self._capability_omissions.append({"surface": "extensions", "reason": "ephemeral_turn"})
        elif not _resource_allowed(self._ctx, "network"):
            self._capability_omissions.append({"surface": "extensions", "reason": "resource_blocked", "resource": "network=false"})
        else:
            try:
                from ouroboros.extension_loader import (
                    _tools as _ext_tools,
                    _lock as _ext_lock,
                    is_extension_live as _ext_is_live,
                )
                meta = getattr(self._ctx, "task_metadata", {})
                capability_root = pathlib.Path((meta.get("budget_drive_root") if isinstance(meta, dict) else "") or getattr(self._ctx, "budget_drive_root", "") or getattr(self._ctx, "drive_root", "") or ".").resolve(strict=False)
                with _ext_lock:
                    extension_tools = [
                        dict(tool)
                        for tool in _ext_tools.values()
                        if _ext_is_live(str(tool.get("skill") or ""), capability_root, repo_path=str(tool.get("skills_repo_path") or "") or None)
                        and _presence_tool_allowed(self._ctx, tool["name"])
                        and (not acting_subagent or tool["name"] in acting_grants)
                    ]
                extension_tools = self._visible_dynamic_tools("extensions", extension_tools)
                extension_schemas = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool.get("schema", {"type": "object", "properties": {}}),
                        },
                    }
                    for tool in extension_tools
                ]
            except Exception as exc:
                self._capability_omissions.append({"surface": "extensions", "reason": "discovery_error", "error": f"{type(exc).__name__}: {exc}"})

        if not core_only:
            mcp_schemas = []
            if ephemeral_turn:
                # CW3: MCP tools can have durable side effects — not for a decision turn.
                self._capability_omissions.append({"surface": "mcp", "reason": "ephemeral_turn"})
            elif not _resource_allowed(self._ctx, "network"):
                self._capability_omissions.append({"surface": "mcp", "reason": "resource_blocked", "resource": "network=false"})
            else:
                try:
                    from ouroboros.mcp_client import ensure_configured_from_settings as _mcp_ensure_configured, get_manager as _mcp_get_manager
                    _mcp_ensure_configured(refresh=True)
                    _mgr = _mcp_get_manager()
                    mcp_tools = [
                        tool
                        for tool in _mgr.list_tools_for_registry()
                        if _presence_tool_allowed(self._ctx, tool["name"])
                        if not acting_subagent or tool["name"] in acting_grants
                    ]
                    mcp_tools = self._visible_dynamic_tools("mcp", mcp_tools)
                    mcp_schemas = [
                        {
                            "type": "function",
                            "function": {"name": tool["name"], "description": tool.get("description", ""), "parameters": tool.get("schema", {"type": "object", "properties": {}})},
                        }
                        for tool in mcp_tools
                    ]
                    slug_collisions = getattr(
                        _mgr, "tool_name_collisions", lambda: []
                    )()
                    if acting_subagent:
                        slug_collisions = [
                            item
                            for item in slug_collisions
                            if str(item.get("prefixed_name") or "") in acting_grants
                        ]
                    self._record_mcp_slug_collisions(
                        slug_collisions
                    )
                    # D1: an enabled+configured server returning zero tools WITHOUT
                    # raising (unreachable/slow/auth-failed) is otherwise silent. Make
                    # the reason visible so the model/owner learns WHY an expected MCP
                    # server produced no tools, instead of "the agent can't see MCP".
                    # Checked unconditionally so a broken server is surfaced even when a
                    # co-located healthy server contributed tools (does not mask it).
                    _empty = _mgr.enabled_servers_without_tools()
                    if _empty:
                        self._capability_omissions.append({"surface": "mcp", "reason": "server_no_tools", "servers": _empty})
                except Exception as exc:
                    self._capability_omissions.append({"surface": "mcp", "reason": "discovery_error", "error": f"{type(exc).__name__}: {exc}"})
            combined = built_in + extension_schemas + mcp_schemas
            if disabled_tools:
                # Apply the declarative tool policy to dynamic extension/MCP schemas too, not just
                # built-ins, so a disabled name can never surface from any discovery source.
                combined = [
                    s for s in combined
                    if (s.get("function", {}) or {}).get("name") not in disabled_tools
                ]
            return combined
        # Core tools plus meta-tools for enabling extended tools.
        result = []
        for e in self._entries.values():
            if e.alias_for:  # compat aliases are callable, never advertised
                continue
            if e.name in disabled_tools:  # declarative tool policy (task_contract.disabled_tools)
                continue
            if not _presence_tool_allowed(self._ctx, e.name):
                continue
            if e.name in unavailable_tools:
                continue
            if local_readonly_subagent and not self._readonly_tool_allowed(e.name):
                continue
            if acting_subagent and e.name not in ACTING_SUBAGENT_TOOL_NAMES:
                continue
            if ephemeral_turn and e.name not in _EPHEMERAL_ALLOWED_TOOLS:
                continue  # CW3: the core/initial envelope is allowlisted too, not just schemas(core_only=False)
            if (
                (local_readonly_subagent and self._readonly_tool_allowed(e.name))
                or (acting_subagent and e.name in ACTING_SUBAGENT_TOOL_NAMES)
                or e.name in CORE_TOOL_NAMES
                or e.name in ("list_available_tools", "enable_tools")
            ):
                result.extend(self._schemas_for_entry(e))
        ext = extension_schemas
        if disabled_tools:
            ext = [s for s in ext if (s.get("function", {}) or {}).get("name") not in disabled_tools]
        return result + ext

    def capability_omissions(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._capability_omissions]

    def policy_hidden_reason(self, name: str) -> Optional[str]:
        """Why a REGISTERED built-in tool is invisible to THIS task, or None.

        Read-only companion to get_schema_by_name (same predicates, same order):
        it distinguishes "hidden by policy" from "does not exist" so discovery
        answers can stop reporting a policy-filtered tool as nonexistent (F3,
        2026-08-10 saga). None means visible OR unknown name — callers that got
        no schema and no reason may honestly say "not found".
        """
        requested = str(name or "").strip()
        if not requested:
            return None
        # BEFORE the registration check: the declarative contract policy applies
        # across ALL discovery sources (get_schema_by_name checks it first for the
        # same reason), so a contract-disabled extension/MCP name answers with its
        # reason instead of "not found" (2026-08-10 amendments). Deeper extension/
        # MCP policy reasons (grants, network) would need new plumbing — disclosed
        # residual, not built.
        if requested in _disabled_tools(self._ctx):
            return "disabled by this task's contract (disabled_tools)"
        if not _presence_tool_allowed(self._ctx, requested):
            return "outside this presence task's positive capability ceiling"
        if requested not in self._entries:
            return None
        if self._entries[requested].alias_for:
            # Mirrors get_schema_by_name: a compat alias is callable but never
            # advertised — discovery answers as for any non-public name.
            return None
        available, reason, _detail = _builtin_tool_availability(requested, self._ctx)
        if not available:
            return f"unavailable ({reason})"
        if getattr(self._ctx, "is_ephemeral_turn", False) and requested not in _EPHEMERAL_ALLOWED_TOOLS:
            return "hidden on this ephemeral decision turn (allowlist)"
        acting_subagent = self._is_acting_subagent()
        if self._is_local_readonly_subagent() and not self._readonly_tool_allowed(requested):
            return "hidden by the read-only subagent profile"
        if acting_subagent and requested not in ACTING_SUBAGENT_TOOL_NAMES:
            return "hidden by the acting subagent profile"
        return None

    def get_schema_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Return the full schema for a specific tool."""
        requested = str(name or "").strip()
        acting_subagent = self._is_acting_subagent()
        acting_grants = self._acting_tool_grants() if acting_subagent else set()
        local_readonly_subagent = self._is_local_readonly_subagent()
        # Declarative tool policy applies across ALL discovery sources (built-in, extension, MCP),
        # so enable_tools/discovery can never surface a disabled name — consistent with schemas()/execute().
        if requested in _disabled_tools(self._ctx):
            return None
        if not _presence_tool_allowed(self._ctx, requested):
            return None
        entry = self._entries.get(requested)
        if entry and entry.alias_for:
            # Compat aliases are callable, never advertised — discovery answers
            # as it does for any non-public name.
            return None
        if entry:
            available, reason, detail = _builtin_tool_availability(requested, self._ctx)
            if not available:
                if reason == "missing_credential":
                    self._capability_omissions.append({
                        "surface": "tools",
                        "reason": reason,
                        "tools": [requested],
                        "details": {requested: detail},
                    })
                return None
            if getattr(self._ctx, "is_ephemeral_turn", False) and requested not in _EPHEMERAL_ALLOWED_TOOLS:
                return None  # CW3: allowlist-consistent with schemas()/execute() (so enable_tools can't surface a denied tool)
            if local_readonly_subagent and not self._readonly_tool_allowed(requested):
                return None
            if acting_subagent and requested not in ACTING_SUBAGENT_TOOL_NAMES:
                return None
            return self._schema_for_entry(entry)
        try:
            from ouroboros.extension_loader import parse_extension_surface_name as _ext_parse_name
        except Exception:
            _ext_parse_name = None
        if _ext_parse_name and _ext_parse_name(name):
            if acting_subagent and requested not in acting_grants:
                return None
            if not _resource_allowed(self._ctx, "network"):
                self._capability_omissions.append({"surface": "extensions", "reason": "resource_blocked", "resource": "network=false"})
                return None
            try:
                from ouroboros.extension_loader import get_tool as _ext_get_tool, is_extension_live as _ext_is_live
                ext_tool = _ext_get_tool(name)
                meta = getattr(self._ctx, "task_metadata", {})
                capability_root = pathlib.Path((meta.get("budget_drive_root") if isinstance(meta, dict) else "") or getattr(self._ctx, "budget_drive_root", "") or getattr(self._ctx, "drive_root", "") or ".").resolve(strict=False)
            except Exception:
                ext_tool = None
            if (
                ext_tool
                and _ext_is_live(str(ext_tool.get("skill") or ""), capability_root, repo_path=str(ext_tool.get("skills_repo_path") or "") or None)
            ):
                return {
                    "type": "function",
                    "function": {
                        "name": ext_tool["name"],
                        "description": ext_tool.get("description", ""),
                        "parameters": ext_tool.get("schema", {"type": "object", "properties": {}}),
                    },
                }
        try:
            from ouroboros.mcp_client import (
                ensure_configured_from_settings as _mcp_ensure_configured,
                get_manager as _mcp_get_manager,
                is_mcp_tool_name as _mcp_is_name,
            )
            _mcp_ensure_configured(refresh=False)
        except Exception:
            _mcp_get_manager = None
            _mcp_is_name = None
        if _mcp_get_manager and _mcp_is_name and _mcp_is_name(requested):
            if acting_subagent and requested not in acting_grants:
                return None
            if not _resource_allowed(self._ctx, "network"):
                self._capability_omissions.append({"surface": "mcp", "reason": "resource_blocked", "resource": "network=false"})
                return None
            mcp_tool = _mcp_get_manager().get_tool(requested)
            if mcp_tool:
                return {
                    "type": "function",
                    "function": {
                        "name": mcp_tool["name"],
                        "description": mcp_tool.get("description", ""),
                        "parameters": mcp_tool.get("schema", {"type": "object", "properties": {}}),
                    },
                }
        return None

    def get_timeout(self, name: str) -> int:
        """Return timeout_sec for the named tool (default 360)."""
        entry = self._entries.get(str(name or "").strip())
        if entry is not None:
            return entry.timeout_sec
        # Extension tools carry timeout_sec in the loader descriptor.
        try:
            from ouroboros.extension_loader import parse_extension_surface_name as _ext_parse_name
        except Exception:
            _ext_parse_name = None
        if _ext_parse_name and _ext_parse_name(name):
            try:
                from ouroboros.extension_loader import get_tool as _ext_get_tool
                ext_tool = _ext_get_tool(name)
            except Exception:
                ext_tool = None
            if ext_tool:
                # Add cleanup grace around the inner async wait_for.
                return int(ext_tool.get("timeout_sec") or 60) + 3
        try:
            from ouroboros.mcp_client import (
                ensure_configured_from_settings as _mcp_ensure_configured,
                get_manager as _mcp_get_manager,
                is_mcp_tool_name as _mcp_is_name,
            )
            _mcp_ensure_configured(refresh=False)
        except Exception:
            _mcp_get_manager = None
            _mcp_is_name = None
        if _mcp_get_manager and _mcp_is_name and _mcp_is_name(name):
            try:
                return int(_mcp_get_manager().tool_timeout_sec()) + 3
            except Exception:
                return 63
        return 360

    def _invoke_builtin_handler(
        self,
        name: str,
        entry: Any,
        args: Dict[str, Any],
        resolved_binding: Any,
        interpreter_resolution: Any,
        worktree_before: Any,
    ) -> tuple[str | None, Any]:
        """Run one builtin handler under the scoped attestation."""
        from ouroboros.process_interpreters import interpreter_attestation

        missing = object()
        prior_tool_result_attr = getattr(self._ctx, _TOOL_RESULT_ATTR, missing)
        tool_result_sentinel = object()
        tool_result_token = _install_tool_result_sidecar(
            self._ctx,
            tool_result_sentinel,
        )
        try:
            with interpreter_attestation(self._ctx, interpreter_resolution):
                try:
                    handler_args = dict(args)
                    if resolved_binding is not None:
                        parameters = inspect.signature(entry.handler).parameters
                        if "_resolved_binding" not in parameters:
                            return (
                                f"⚠️ TOOL_INTERNAL_ERROR ({name}): target-sensitive handler "
                                "does not declare the private _resolved_binding keyword.",
                                None,
                            )
                        handler_args["_resolved_binding"] = resolved_binding
                    try:
                        inspect.signature(entry.handler).bind(self._ctx, **handler_args)
                    except TypeError:
                        return tool_resolution._format_tool_arg_error(entry), None
                    result = entry.handler(self._ctx, **handler_args)
                    published = _published_tool_result(
                        self._ctx,
                        tool_result_sentinel,
                    )
                    if (
                        isinstance(published, ToolResult)
                        and isinstance(result, str)
                        and published.text == result
                    ):
                        return None, published
                    return None, result
                except TypeError as e:
                    return f"⚠️ TOOL_ERROR ({name}): {e}", None
                except Exception as e:
                    return f"⚠️ TOOL_ERROR ({name}): {e}", None
        finally:
            _restore_tool_result_sidecar(tool_result_token)
            if prior_tool_result_attr is missing:
                try:
                    delattr(self._ctx, _TOOL_RESULT_ATTR)
                except AttributeError:
                    pass
            else:
                setattr(self._ctx, _TOOL_RESULT_ATTR, prior_tool_result_attr)
            # Central advisory invalidation by OBSERVED worktree diff: runs on
            # success, tool error, and exception paths alike (the per-tool
            # manual calls missed early-return/error paths), and skips
            # invalidation when a flagged tool ran read-only.
            if worktree_before is not None:
                self._invalidate_advisory_if_worktree_changed(name, worktree_before)

    def _execute_legacy_text(self, name: str, args: Dict[str, Any]) -> str | ToolResult:
        name = str(name or "").strip()
        args, presence_arg_error = _presence_bound_args(self._ctx, name, args)
        if presence_arg_error:
            return presence_arg_error
        _route_note = ""
        task_constraint = normalize_task_constraint(getattr(self._ctx, "task_constraint", None))
        local_readonly_subagent = self._is_local_readonly_subagent()
        acting_subagent = self._is_acting_subagent()
        acting_self_worktree = acting_subagent and str(getattr(task_constraint, "surface", "") or "") == "self_worktree"
        acting_protected_grant = acting_subagent and bool(getattr(task_constraint, "protected_paths_grant", False))
        acting_tool_grants = set(getattr(task_constraint, "external_tool_grants", ()) or ()) if acting_subagent else set()
        entry = self._entries.get(name)
        ext_tool, extension_unavailable = extension_dispatch._extension_dispatch_candidate(self._ctx, name) if entry is None else (None, False)
        _mcp_is_name = None
        if entry is None and ext_tool is None:
            try:
                from ouroboros.mcp_client import (
                    ensure_configured_from_settings as _mcp_ensure_configured,
                    is_mcp_tool_name as _mcp_is_name,
                )
                _mcp_ensure_configured(refresh=False)
            except Exception:
                _mcp_is_name = None
        is_mcp = bool(_mcp_is_name and _mcp_is_name(name))
        _eph = registry_guards._ephemeral_block_result(self._ctx, name, ext_tool, is_mcp)  # CW3: built-in deny set + extension/MCP
        if _eph is not None:
            return _eph
        _resource_gate = registry_guards._capability_resource_guard_result(
            self._ctx, name, args, ext_tool, is_mcp)
        if _resource_gate is not None:
            return _resource_gate
        # Cover the full repo-mutating surface explicitly (CODE_TOOLS ∪ _REPO_MUTATION_TOOLS):
        # write_file/edit_text AND shell/process tools (run_command/run_script/
        # start_service) are all is_code_tool=True, but gating on the union makes the
        # "no OTHER task writes the repo while a merge is staged" contract robust to flag drift.
        _gate = registry_guards._subagent_and_update_guard_result(
            self, name, entry, ext_tool, is_mcp, local_readonly_subagent,
            acting_subagent, acting_tool_grants,
            entry is not None and (name in self.CODE_TOOLS or name in _REPO_MUTATION_TOOLS),
        )
        if _gate is not None:
            return _gate
        workspace_block_reason = ""
        try:
            workspace_block_reason = workspace_mode_block_reason(self._ctx)
        except Exception as exc:
            workspace_block_reason = f"workspace metadata validation failed: {type(exc).__name__}: {exc}"
        if workspace_block_reason:
            return ToolResult(status="blocked", code="WORKSPACE_BLOCKED", text=(
                "⚠️ WORKSPACE_MODE_BLOCKED: invalid external workspace metadata: "
                f"{workspace_block_reason}. Workspace tasks must not overlap the "
                "Ouroboros repo, runtime data, or control plane."
            ))
        if entry is not None:
            public_arg_error = tool_resolution._prepare_public_builtin_args(entry, args)
            if public_arg_error:
                return public_arg_error
            path_normalization = tool_resolution._normalize_dispatch_path_args_result(self._ctx, name, args)
            _route_note = path_normalization.text
            if path_normalization.required_root == "active_workspace":
                return ToolResult(status="blocked", code="ROOT_REQUIRED_ACTIVE_WORKSPACE", text=_route_note, meta={"required_root": "active_workspace"})
        heal_no_enable = bool(task_constraint and task_constraint.mode == "skill_repair")
        if heal_no_enable:
            heal_block = registry_guards._heal_mode_guard_result(
                self._ctx, name, args, task_constraint, ext_tool, is_mcp)
            if heal_block is not None:
                return heal_block
        workspace_mode = bool(getattr(self._ctx, "is_workspace_mode", lambda: False)())
        effective_constraint = task_constraint
        if entry is not None and not (skip_binding := _configured_delegate_selector(self._ctx, name, args)):
            effective_constraint, payload_result = registry_guards._payload_dispatch_constraint(
                self._ctx,
                name=name,
                args=args,
                task_constraint=task_constraint,
                workspace_mode=workspace_mode,
            )
            if payload_result is not None:
                return payload_result
        resolved_binding = None
        if entry is not None and not skip_binding and _target_binding_operation(name, args) is not None:
            try:
                resolved_binding = _build_builtin_target_binding(self._ctx, name, args)
            except Exception as exc:
                redirect = tool_resolution._light_binding_failure_result(name, args)
                if redirect is not None:
                    return redirect
                operation = tool_resolution._target_binding_operation(name, args)
                if operation in {"shell", "service"}:
                    return shell_cwd_block_message(
                        self._ctx, str(args.get("cwd") or ""), operation=operation, error=exc)
                return tool_resolution._binding_error_text(
                    name, str(args.get("root") or "active_workspace"), exc)
        # Asked three times below (light start_service, protected writes, the
        # light repo tripwire snapshot) and always with the same answer: an
        # acting child's own worktree counts as the system repo.
        targets_system_repo = (
            _binding_set_targets_system_repo(self._ctx, resolved_binding) or acting_self_worktree
        )
        if not _presence_binding_allowed(self._ctx, resolved_binding):
            return (
                "⚠️ PRESENCE_RESOURCE_BLOCKED: the resolved target is outside "
                "this presence task's positive resource ceiling."
            )
        # Fail-closed: an acting child WITHOUT a resolved isolated workspace would
        # have active_workspace/system_repo fall back to the LIVE repo. Confine it
        # to data roots and block shell/coding/service (whose default target is the repo).
        if acting_subagent and not workspace_mode:
            if name in tool_resolution._ROOT_ARG_REPO_WRITE_TOOLS and str(args.get("root", "") or "active_workspace") in ("active_workspace", "system_repo"):
                return _ACTING_NO_WORKSPACE_REPO_RESULT
            if name in ("run_command", "run_script", "start_service",
                        "integrate_subagent_patch", "integrate_delegated_patch"):
                return _ACTING_NO_WORKSPACE_PROCESS_RESULT
        # Hardcoded sandbox: light blocks repo mutation; advanced protects
        # core/contracts/release; pro still relies on commit review.
        try:
            from ouroboros.config import get_runtime_mode as _get_runtime_mode
            _runtime_mode = _get_runtime_mode()
        except Exception:
            _runtime_mode = "advanced"
        if is_mcp:
            return extension_dispatch._dispatch_mcp_tool_result(self._ctx, name, args)
        if entry is None:
            if ext_tool and callable(ext_tool.get("handler")):
                return extension_dispatch._dispatch_extension_tool_result(self._ctx, name, ext_tool, args)
            return _unknown_tool_result(self._entries, name, extension_unavailable)
        args, interpreter_resolution, interpreter_block = tool_resolution._resolve_python_predispatch(
            self, name, args, _runtime_mode, effective_constraint, resolved_binding,
        )
        if interpreter_block is not None:
            return interpreter_block
        allow_short_relative = bool(
            effective_constraint and effective_constraint.mode == "skill_repair"
        )
        light_skill_scoped_str_replace = resolved_binding is None and (
            registry_guards._light_mode_payload_mutation_allowed(
                ctx=self._ctx, tool_name=name, args=args, runtime_mode=_runtime_mode,
                effective_constraint=effective_constraint,
                implicit_skill_cwd_allowed=heal_no_enable,
                allow_short_relative=allow_short_relative,
            ))
        if name in _SYSTEM_INTRINSIC_REPO_MUTATION_TOOLS:
            light_targets_system = True
        elif resolved_binding is not None:
            light_targets_system = (
                _binding_set_is_light_restricted(self._ctx, resolved_binding) or acting_self_worktree
            )
        else:
            light_targets_system = not workspace_mode or acting_self_worktree
        if (
            _runtime_mode == "light"
            and name in _REPO_MUTATION_TOOLS
            and light_targets_system
            and not light_skill_scoped_str_replace
            and not registry_guards._authorized_managed_update_resolver(self._ctx)
        ):
            light_redirect = light_cognitive_or_root_redirect(name, args)
            if light_redirect:
                return light_redirect
            return ToolResult(
                status="blocked",
                code="LIGHT_MODE_BLOCKED",
                text=(
                    "⚠️ LIGHT_MODE_BLOCKED: runtime_mode=light blocks Ouroboros "
                    f"self-repo/control-plane mutation via {name!r}. For user-visible "
                    "deliverables use root=user_files (for example Desktop/file.html), "
                    "root=artifact_store for the canonical task artifact, or root=task_drive "
                    "for scratch. Skill payload edits remain allowed only through "
                    "root=skill_payload with bucket and skill_name "
                    "(data/skills/<bucket>/<skill>/) or skill_repair constraints. "
                    "Switch to advanced/pro only for reviewed Ouroboros self-modification."
                ),
            )
        protected_write_paths = []
        if name in tool_resolution._ROOT_ARG_REPO_WRITE_TOOLS:
            root_name = str(args.get("root", "") or "active_workspace")
            protected_write_paths = [
                canonical_repo_relative_path(self._ctx, root_name, p)
                for p in tool_resolution._payload_write_paths(name, args)
            ]
            if resolved_binding is not None:
                protected_target = targets_system_repo
            else:
                protected_target = (not workspace_mode or acting_self_worktree) and (
                    root_name in {"active_workspace", "system_repo"}
                )
            protected_matches = (
                protected_paths_in(protected_write_paths) if protected_target else []
            )
            allow_protected = registry_guards._authorized_managed_update_resolver(self._ctx) or (
                mode_allows_protected_write(_runtime_mode)
                and (acting_protected_grant or not acting_subagent)
            )
            if protected_matches and not allow_protected:
                first = protected_matches[0]
                return _protected_write_block_result(
                    path=first.path,
                    runtime_mode=_runtime_mode,
                    action=f"run tool {name!r} against",
                )

        if _shell_guard_required(name, args):
            if name == "start_service" and _runtime_mode == "light" and targets_system_repo:
                return _LIGHT_START_SERVICE_RESULT
            block_result = registry_guard_process._run_shell_safety_check(
                self,
                shell_guards.process_shell_guard_args(name, args, ctx=self._ctx, runtime_mode=_runtime_mode),
                _runtime_mode,
                resolved_binding,
            )
            if block_result is not None:
                return block_result

        # LLM safety supervisor.
        from ouroboros.safety import check_safety
        is_safe, safety_msg = check_safety(
            name,
            args,
            messages=getattr(self._ctx, "messages", None),
            ctx=self._ctx,
            python_resolution=interpreter_resolution,
        )
        if not is_safe:
            return ToolResult(status="blocked", code="SAFETY_VIOLATION", text=safety_msg)
        light_repo_before = (
            registry_guard_process._light_repo_snapshot(system_repo_dir_for(self._ctx))
            if name in _PROCESS_COMMAND_TOOLS and _runtime_mode == "light" and targets_system_repo
            else None
        )
        workspace_refs_before = (
            registry_guard_process._git_ref_snapshot(active_repo_dir_for(self._ctx))
            if name in _PROCESS_COMMAND_TOOLS and workspace_mode and acting_self_worktree
            else None
        )
        worktree_before = self._worktree_status_snapshot() if entry.mutates_worktree else None
        settings_before = registry_guard_process._owner_settings_snapshot() if name in _PROCESS_COMMAND_TOOLS else None
        if interpreter_resolution is None:  # node: post-gates (A-F4)
            from ouroboros.process_interpreters import resolve_node_postgates

            args, interpreter_resolution = resolve_node_postgates(
                self._ctx, name, args, runtime_mode=_runtime_mode,
                effective_constraint=effective_constraint, resolved_binding=resolved_binding,
            )
        early_error, result = self._invoke_builtin_handler(
            name, entry, args, resolved_binding, interpreter_resolution, worktree_before,
        )
        if name in _PROCESS_COMMAND_TOOLS:
            # Tripwires run on the TOOL_ERROR path too: two early_error returns
            # fire AFTER the process already ran (#447 B2).
            checked = registry_guard_process._run_shell_post_checks(
                self, early_error if early_error is not None else result,
                light_repo_before=light_repo_before,
                workspace_refs_before=workspace_refs_before,
                settings_before=settings_before, tool_name=name,
            )
            if early_error is not None:
                return checked
            result = checked
        elif early_error is not None:
            return early_error

        return _compose_execute_result_result(name, result, _route_note, safety_msg) if _route_note or safety_msg else result

    def execute_result(self, name: str, args: Dict[str, Any]) -> ToolResult:
        """Dispatch once and adapt only producers that still return legacy text."""
        result = self._execute_legacy_text(name, args)
        if isinstance(result, ToolResult):
            return result
        return LegacyTextResultAdapter.from_text(name, result)

    def execute(self, name: str, args: Dict[str, Any]) -> str:
        """Compatibility ABI: return the exact model-facing text projection."""
        return self.execute_result(name, args).text

    def _worktree_status_snapshot(self) -> str:
        try:
            from ouroboros.utils import run_cmd

            return run_cmd(["git", "status", "--porcelain"], cwd=self._ctx.repo_dir, timeout=20)
        except Exception:
            return "<status-unavailable>"

    def _invalidate_advisory_if_worktree_changed(self, tool_name: str, before: str) -> None:
        after = self._worktree_status_snapshot()
        if after == before:
            return
        try:
            from ouroboros.review_state import invalidate_advisory_after_mutation

            invalidate_advisory_after_mutation(
                pathlib.Path(self._ctx.drive_root),
                mutation_root=pathlib.Path(self._ctx.repo_dir),
                source_tool=tool_name,
            )
        except Exception:
            logging.getLogger(__name__).debug(
                "Central advisory invalidation failed for %s", tool_name, exc_info=True
            )

    def override_handler(self, name: str, handler) -> None:
        """Override the handler for a registered tool (used for closure injection)."""
        entry = self._entries.get(name)
        if entry:
            projected = replace(entry, handler=handler)
            self._handler_overrides[name] = handler
            self._entries[name] = projected
            if name in self._scoped_entries:
                self._scoped_entries[name] = projected

    @property
    def CODE_TOOLS(self) -> frozenset:
        return frozenset(e.name for e in self._entries.values() if e.is_code_tool)
