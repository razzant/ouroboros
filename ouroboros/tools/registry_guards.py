"""Host-owned pre-dispatch guards: capability/resource, managed-update and skill-payload constraints.

The facade re-exports these definitions so existing imports and monkeypatch
targets retain the same bindings.
"""

from __future__ import annotations

import logging
import os
import pathlib

from typing import TYPE_CHECKING

from ouroboros.tools.tool_result import ToolResult

if TYPE_CHECKING:  # annotation-only imports (inert at runtime)
    from ouroboros.contracts.task_constraint import TaskConstraint
    from typing import Any
    from typing import Dict
    from typing import List
    from typing import Optional

# The logger name is pinned to the parent's literal namespace so the
# extraction does not silently rename the log stream.
log = logging.getLogger("ouroboros.tools.registry")


def _registry():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time.
    """
    from ouroboros.tools import registry

    return registry


def _executor_backend_candidate_allowed(ctx: Any, candidate: str, allowed_roots: List[pathlib.Path]) -> bool:
    try:
        from ouroboros.workspace_executor import executor_ref_from_ctx as _executor_ref_from_ctx
        from ouroboros.workspace_executor import map_backend_path as _executor_map_backend_path

        executor_ref = _executor_ref_from_ctx(ctx)
        if executor_ref is None:
            return False
        resolved = _executor_map_backend_path(executor_ref, candidate)
        return any(
            resolved.is_relative_to(root) or _registry()._path_is_relative_to_casefold(resolved, root)
            for root in allowed_roots
        )
    except Exception:
        return False




def _stray_skill_payload_failsoft(root_arg: str, workspace_mode: bool, task_constraint: Any) -> bool:
    """Whether stray bucket/skill_name on a write tool should be DROPPED rather than
    surfaced as SKILL_PAYLOAD_ARG_ERROR. Fail-soft ONLY for a WORKSPACE edit that is
    NOT skill-authoring: there bucket/skill_name are model noise (the B2 footgun —
    reflexive bucket="external" on an /app edit). In light/advanced non-workspace
    skill-authoring (or an explicit root=skill_payload / skill_repair) the specific
    error is the intended helpful signal."""
    skill_payload_intent = root_arg == "skill_payload" or bool(
        task_constraint and task_constraint.has_selected_skill
    )
    return bool(workspace_mode and not skill_payload_intent)


def _managed_update_code_tool_block_result(ctx: Any, name: str) -> ToolResult | None:
    """Block a repo-mutating code tool while a managed-update assisted merge is staged for
    ANOTHER task (P2/SC2). Returns a blocked result, or ``None`` when allowed (this is the
    authorized resolution task, or no managed tx is active). A corrupt tx marker fails closed."""
    from ouroboros.config import get_runtime_mode
    from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency

    cyber = mode_has_unrestricted_agency(get_runtime_mode())
    try:
        from supervisor.update_merge import managed_assisted_tx_for

        if managed_assisted_tx_for(
            getattr(ctx, "task_id", ""),
            getattr(ctx, "task_metadata", None),
        )[1]:
            if cyber:
                from ouroboros.safety import _emit_durable_safety_event

                _emit_durable_safety_event(ctx, {
                    "type": "safety_advisory", "tool": name,
                    "assessment": "A separate managed update is in progress.",
                    "execution_allowed": True,
                })
                return None
            return ToolResult(
                status="blocked",
                code="ACCESS_BLOCKED",
                text=(
                    f"⚠️ MANAGED_UPDATE_IN_PROGRESS: {name!r} is blocked while a managed update merge "
                    "is being resolved (only its authorized resolution task may write the repo). "
                    "Retry after the update lands or is rolled back."
                ),
            )
    except Exception:
        if cyber:
            log.warning("Managed update state unavailable for Cyber call %s; proceeding", name)
            return None
        return ToolResult(
            status="unavailable",
            code="CAPABILITY_UNAVAILABLE",
            text=(
                f"⚠️ MANAGED_UPDATE_STATE_UNAVAILABLE: {name!r} is blocked because the managed "
                "update transaction state could not be verified. Retry after the update state is "
                "available or repaired."
            ),
        )
    return None


def _managed_update_code_tool_block(ctx: Any, name: str) -> str:
    """Compatibility projection for direct callers of the legacy helper."""
    result = _managed_update_code_tool_block_result(ctx, name)
    return result.text if result is not None else ""


def _subagent_and_update_guard_result(
    registry: Any,
    name: str,
    entry: Any,
    ext_tool: Any,
    is_mcp: bool,
    local_readonly_subagent: bool,
    acting_subagent: bool,
    acting_tool_grants: Any,
    repo_mutation: bool,
) -> ToolResult | None:
    """Early dispatch gates (native denial, or ``None`` to allow): the read-only and
    acting subagent tool-name allowlists, and the managed-update merge write-exclusivity
    (P2/SC2 — only the authorized resolution task may run code tools while a merge is staged)."""
    if local_readonly_subagent and entry is not None and not registry._readonly_tool_allowed(name):
        return ToolResult(status="blocked", code="ACCESS_BLOCKED", text=(
            "⚠️ LOCAL_READONLY_SUBAGENT_BLOCKED: this subagent may inspect "
            "local repo/data/history plus web/browser surfaces and enabled "
            "external tools, but may not call first-party local tool "
            f"{name!r}. Parent tasks must perform writes, commits, review "
            "gates, tool expansion, runtime control, shell, and skills. "
            "Nested readonly delegation is allowed only through schedule_subagent "
            "within configured depth/cap limits."
        ))
    from ouroboros.tool_capabilities import acting_tool_names_for_context

    acting_allowed_names = acting_tool_names_for_context(registry._ctx, registry._entries)
    if acting_subagent and entry is not None and name not in acting_allowed_names:
        return ToolResult(status="blocked", code="ACCESS_BLOCKED", text=(
            "⚠️ ACTING_SUBAGENT_BLOCKED: this mutative subagent may read and "
            "write inside its assigned write root and run shell/services "
            f"there, but may not call first-party tool {name!r}. It cannot "
            "commit the live body, run review/runtime/skills lifecycle, enable "
            "tools, or write cognitive memory; the parent applies isolated patches "
            "or verifies shared external files and is the sole live-body committer."
        ))
    if acting_subagent and entry is None and (ext_tool or is_mcp) and acting_tool_grants is not None and name not in acting_tool_grants:
        return ToolResult(status="blocked", code="ACCESS_BLOCKED", text=(
            "⚠️ ACTING_SUBAGENT_TOOL_NOT_GRANTED: extension/MCP tool "
            f"{name!r} is not in this acting subagent's external_tool_grants. "
            "The parent must grant dynamic tools explicitly per child."
        ))
    # Cover the full repo-mutating surface explicitly (CODE_TOOLS ∪ _REPO_MUTATION_TOOLS):
    # write_file/edit_text AND shell/process tools (run_command/run_script/
    # start_service) are all is_code_tool=True, but gating on the union makes the
    # "no OTHER task writes the repo while a merge is staged" contract robust to flag drift.
    if entry is not None and repo_mutation:
        return _managed_update_code_tool_block_result(registry._ctx, name)
    return None


def _authorized_managed_update_resolver(ctx: Any) -> bool:
    """Whether this task is the durable tx-authorized assisted resolver.

    Fail-closed bool for every authority consumer (False = no extra powers).
    The AUTHORITY-READ failure is additionally distinguished from an honest
    "not the resolver" via a typed ctx marker (``_managed_authority_read_error``:
    set on an unreadable read AND on a corrupt tx marker, cleared on every
    healthy evaluation), so the review-subject builder can fail LOUDLY instead
    of silently reviewing a possibly-managed candidate as an ordinary full
    staged capture."""
    try:
        from supervisor.update_merge import authorized_assisted_task_strict

        marker_status, tx = authorized_assisted_task_strict(
            getattr(ctx, "task_id", ""),
            getattr(ctx, "task_metadata", None),
        )
        try:
            if marker_status == "corrupt":
                # A tx marker EXISTS but cannot be parsed: authority stays
                # False (fail-closed) for every bool consumer, but the loud
                # A4 channel must fire — clearing the marker here would let
                # the review subject silently treat a possibly-managed
                # candidate as an ordinary full staged diff.
                setattr(
                    ctx, "_managed_authority_read_error",
                    "update_tx_corrupt: the managed update transaction marker "
                    "exists but could not be parsed",
                )
            else:
                setattr(ctx, "_managed_authority_read_error", "")
        except Exception:
            pass
        return bool(tx)
    except Exception as exc:
        try:
            setattr(ctx, "_managed_authority_read_error", repr(exc))
        except Exception:
            pass
        return False


def _light_mode_payload_mutation_allowed(
    *,
    ctx: Any,
    tool_name: str,
    args: Dict[str, Any],
    runtime_mode: str,
    effective_constraint: Optional[TaskConstraint],
    implicit_skill_cwd_allowed: bool,
    allow_short_relative: bool,
) -> bool:
    """Return True for light-mode data skill payload edits that do not touch repo files."""

    # apply_patch/edit_batch are DELIBERATELY absent: they refuse data-plane roots
    # entirely (repo lanes only), so they can never be a payload edit — in light
    # mode they stay under the generic repo-mutation block like any repo write.
    if runtime_mode != "light" or tool_name not in {"edit_text", "write_file"}:
        return False
    requested_root = str(args.get("root", "") or "active_workspace")
    try:
        requested_root = _registry().normalize_root(requested_root)
    except Exception:
        requested_root = str(args.get("root", "") or "active_workspace")
    if requested_root in {"task_drive", "artifact_store", "user_files"}:
        return True
    legacy_data_skill_edit = False
    if tool_name == "edit_text" and requested_root == "active_workspace":
        try:
            legacy_target = _registry().resolve_skill_payload_target(
                pathlib.Path(ctx.drive_root),
                str(args.get("path", "") or ""),
            )
            legacy_data_skill_edit = legacy_target.target_path.exists() and not legacy_target.control_plane
        except Exception:
            legacy_data_skill_edit = False
    if requested_root not in {"runtime_data", "skill_payload"} and not legacy_data_skill_edit:
        return False
    return _registry().is_skill_payload_path(
        pathlib.Path(ctx.drive_root),
        str(args.get("path", "") or ""),
        constraint=effective_constraint,
        allow_short_relative=allow_short_relative,
        allow_control_plane=False,
    )


_WEB_TOOLS = frozenset({"web_search", "browse_page", "browser_action", "youtube_transcript"})


def _resource_allowed(ctx: Any, key: str) -> bool:
    metadata = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    contract = metadata.get("task_contract") if isinstance(metadata.get("task_contract"), dict) else {}
    if not contract and isinstance(getattr(ctx, "task_contract", None), dict):
        contract = getattr(ctx, "task_contract")
    resources = {}
    for source in (metadata, contract):
        raw = source.get("allowed_resources") if isinstance(source, dict) else None
        if isinstance(raw, dict):
            resources.update(raw)
    if not resources:
        return True
    for name in (key, f"allow_{key}"):
        value = resources.get(name)
        if isinstance(value, bool):
            return value
    if key == "web":
        for name in ("network", "allow_network", "internet", "external_network"):
            value = resources.get(name)
            if isinstance(value, bool) and not value:
                return False
    if key == "network":
        for name in ("internet", "external_network"):
            value = resources.get(name)
            if isinstance(value, bool) and not value:
                return False
    return True


def _disabled_tools(ctx: Any) -> frozenset:
    """Tool names the task contract withholds (declarative tool policy).

    Independent of ``allowed_resources``: a caller can disable specific tools
    (e.g. the agent's web_search/browser/VLM tools for a faithful benchmark)
    WITHOUT setting web/network=false — so shell network egress (git/pip) stays
    available. Withholding web tools does not withhold unrelated network tools.

    Enforced twice — the schema filters in ``registry_core`` hide the names and
    ``_capability_resource_guard_result`` refuses them at dispatch — EXCEPT for a
    consciousness-origin task (``disabled_tools_dispatch_only``): its list is
    enforced at dispatch only, so a wake-up's tool schemas and its capability
    manifest are byte-identical to an owner turn's and the provider prompt cache
    prefix is shared (owner decision В31=B). The model then sees tools it may not
    call and gets the typed refusal instead; the wake message names the level.
    """
    metadata = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    contract = metadata.get("task_contract") if isinstance(metadata.get("task_contract"), dict) else {}
    if not contract and isinstance(getattr(ctx, "task_contract", None), dict):
        contract = getattr(ctx, "task_contract")
    names: set = set()
    for source in (metadata, contract):
        raw = source.get("disabled_tools") if isinstance(source, dict) else None
        if isinstance(raw, (list, tuple)):
            names.update(str(n).strip() for n in raw if str(n).strip())
    # D10 compatibility: `claude_code_edit` was retired; saved contracts that
    # withheld the external coding gateway keep withholding its SUCCESSOR — the
    # delegated coding session's start verb. The dead name stays in the set
    # too (harmless: nothing registers it), so old contracts round-trip as-is.
    if "claude_code_edit" in names:
        names.add("delegate_start")
    # Rename compatibility: contracts that withheld `advisory_review` keep
    # withholding the SAME organ under its new name, and vice versa (a new
    # contract naming only the new spelling must also silence the alias).
    if "advisory_review" in names:
        names.add("preflight_review")
    if "preflight_review" in names:
        names.add("advisory_review")
    return frozenset(names)


def disabled_tools_dispatch_only(ctx: Any) -> bool:
    """Whether ``disabled_tools`` binds at dispatch only (a consciousness-origin task)."""
    from ouroboros.consciousness_authority import is_consciousness_origin

    return is_consciousness_origin(getattr(ctx, "task_metadata", None))


def task_session_block_reason(ctx: Any, *, dispatch: bool = True) -> str:
    """Why ``ctx`` may not use a ``session_scope=task`` MCP server ('' when it may).

    Such a server's session is owned by one task id and never shared
    (``mcp_task_sessions``). A delegated child would otherwise open its own
    connection to what the owner connected for a task — the owner's Chrome, for
    the Playwright Extension — so children never see these tools. A
    consciousness wake is refused at dispatch only, keeping its schema prefix
    identical to an owner turn's (``disabled_tools_dispatch_only``).
    """
    from ouroboros.contracts.task_constraint import normalize_task_constraint
    from ouroboros.tool_capabilities import ACTING_SUBAGENT_MODE, LOCAL_READONLY_SUBAGENT_MODE

    if not str(getattr(ctx, "task_id", "") or "").strip():
        return "this caller has no task id to own a session"
    constraint = normalize_task_constraint(getattr(ctx, "task_constraint", None))
    roles = {str(getattr(constraint, "mode", "") or "")}
    for data in (getattr(ctx, "task_metadata", None), getattr(ctx, "task_contract", None)):
        if isinstance(data, dict):
            lineage = data.get("lineage") if isinstance(data.get("lineage"), dict) else {}
            roles |= {str(data.get("delegation_role") or "").strip(), str(lineage.get("delegation_role") or "").strip()}
    if roles & {"subagent", ACTING_SUBAGENT_MODE, LOCAL_READONLY_SUBAGENT_MODE}:
        return "a delegated child task may not open a session to it"
    if dispatch and disabled_tools_dispatch_only(ctx):
        return "a background consciousness wake may not open a session to it"
    return ""


_GITHUB_TOKEN_TOOLS = frozenset({
    "list_github_prs",
    "get_github_pr",
    "comment_on_pr",
    "pr_merge",
    "list_github_issues",
    "get_github_issue",
    "comment_on_issue",
    "close_github_issue",
    "create_github_issue",
    "run_ci_tests",
    "submit_skill_to_hub",
    "generate_evolution_stats",
})


def _builtin_tool_availability(name: str, ctx: Any = None) -> tuple[bool, str, str]:
    """Return ``(available, reason, detail)`` for built-in tool credential gates.

    Predicates are lazy to avoid registry import cycles and discovery-time side effects.
    """
    # A bare registry (unit tests, static policy inventory, import-time introspection)
    # is a structural surface, not a running task capability envelope.
    if not str(getattr(ctx, "task_id", "") or "").strip():
        metadata = getattr(ctx, "task_metadata", {}) if ctx is not None else {}
        contract = getattr(ctx, "task_contract", {}) if ctx is not None else {}
        if not metadata and not contract:
            return True, "", ""
    tool = str(name or "").strip()
    if tool == "web_search":
        try:
            from ouroboros.tools.search import _available_web_search_backends

            if not _available_web_search_backends():
                return False, "missing_credential", "web_search_backend"
        except ImportError:
            return True, "", ""
        except Exception:
            return True, "", ""
    if tool in _GITHUB_TOKEN_TOOLS:
        detail = "GITHUB_TOKEN"
        if tool in {"run_ci_tests", "submit_skill_to_hub", "generate_evolution_stats"}:
            configured = bool(os.environ.get("GITHUB_TOKEN", "").strip())
        else:
            from ouroboros.tools.github import github_cli_configured

            configured = github_cli_configured()
            detail = "GITHUB_TOKEN or configured GitHub CLI"
        if not configured:
            return False, "missing_credential", detail
    return True, "", ""


def _capability_resource_guard_result(
    ctx: Any,
    name: str,
    args: dict[str, Any],
    ext_tool: Any = None,
    is_mcp: bool = False,
) -> ToolResult | None:
    """Apply direct task capability and resource admission in legacy order."""
    # Consciousness Observe keeps the names it needs for useful read-only research
    # visible, but their ARGUMENTS cannot carry back the mutating authority the
    # level removed. The policy lives with the level that owns it (a call-time
    # import, like every other D04->D15 edge); a failure to consult it raises
    # here rather than reading as permission.
    from ouroboros.consciousness_authority import observe_argument_refusal

    metadata = getattr(ctx, "task_metadata", {}) if isinstance(getattr(ctx, "task_metadata", {}), dict) else {}
    if refusal := observe_argument_refusal(metadata, name, args):
        return ToolResult(status="blocked", code="RESOURCE_CONSTRAINT_BLOCKED", text=refusal)
    if name in _disabled_tools(ctx):
        return ToolResult(
            status="blocked",
            code="RESOURCE_CONSTRAINT_BLOCKED",
            text=(
                "⚠️ RESOURCE_CONSTRAINT_BLOCKED: task_contract.disabled_tools "
                f"withholds {name!r} for this task."
            ),
        )
    available, unavailable_reason, unavailable_detail = _builtin_tool_availability(name, ctx)
    if not available:
        suffix = f" ({unavailable_detail})" if unavailable_detail else ""
        return ToolResult(
            status="unavailable",
            code="CAPABILITY_UNAVAILABLE",
            text=f"⚠️ CAPABILITY_UNAVAILABLE: {name!r} is unavailable: {unavailable_reason}{suffix}.",
        )
    if name == "vlm_query" and str(args.get("image_url") or "").strip() and (
        not _resource_allowed(ctx, "web") or not _resource_allowed(ctx, "network")
    ):
        return ToolResult(
            status="blocked",
            code="RESOURCE_CONSTRAINT_BLOCKED",
            text=(
                "⚠️ RESOURCE_CONSTRAINT_BLOCKED: remote image_url for vlm_query "
                "requires allowed_resources.web/network."
            ),
        )
    if name in _WEB_TOOLS and not _resource_allowed(ctx, "web"):
        return ToolResult(
            status="blocked",
            code="RESOURCE_CONSTRAINT_BLOCKED",
            text=(
                "⚠️ RESOURCE_CONSTRAINT_BLOCKED: task_contract.allowed_resources.web=false "
                f"blocks {name!r}."
            ),
        )
    if name == "vcs_pull_ff" and not _resource_allowed(ctx, "network"):
        return ToolResult(
            status="blocked",
            code="RESOURCE_CONSTRAINT_BLOCKED",
            text=(
                "⚠️ RESOURCE_CONSTRAINT_BLOCKED: task_contract.allowed_resources.network=false "
                "blocks 'vcs_pull_ff'."
            ),
        )
    if (is_mcp or ext_tool) and not _resource_allowed(ctx, "network"):
        return ToolResult(
            status="blocked",
            code="RESOURCE_CONSTRAINT_BLOCKED",
            text=(
                "⚠️ RESOURCE_CONSTRAINT_BLOCKED: task_contract.allowed_resources.network=false "
                f"blocks external tool {name!r}."
            ),
        )
    return None


def _payload_dispatch_constraint(
    ctx: Any,
    *,
    name: str,
    args: dict[str, Any],
    task_constraint: Optional[TaskConstraint],
    workspace_mode: bool,
) -> tuple[Optional[TaskConstraint], ToolResult | None]:
    """Preserve repair selectors without letting stray selectors retarget work."""

    raw_bucket = str(args.get("bucket", "") or "")
    raw_skill_name = str(args.get("skill_name", "") or "")
    explicit_skill_root = str(args.get("root", "") or "").strip().lower() == "skill_payload"
    short_form_decision = None if explicit_skill_root else _registry().decide_payload_short_form(
        bucket=raw_bucket,
        skill_name=raw_skill_name,
        path_text=str(args.get("path", "") or "."),
        repo_dir=pathlib.Path(ctx.repo_dir),
        drive_root=pathlib.Path(ctx.drive_root),
    )
    if explicit_skill_root:
        # Binding selection already handled the explicit target. This legacy
        # constraint exists only for the light-mode data-payload carve-out.
        synthesized = _registry().synthesize_payload_constraint(raw_bucket, raw_skill_name)
    else:
        synthesized = (
            short_form_decision.constraint
            if short_form_decision is not None
            and task_constraint
            and task_constraint.has_selected_skill
            else None
        )

    if (
        (raw_bucket or raw_skill_name)
        and short_form_decision is not None
        and short_form_decision.error
        and name in {"write_file", "edit_text"}
    ):
        root_arg = str(args.get("root", "") or "").strip().lower()
        if _stray_skill_payload_failsoft(root_arg, workspace_mode, task_constraint):
            log.info(
                "Ignoring stray bucket/skill_name on %s (workspace edit, root=%s): %s",
                name,
                root_arg or "active_workspace",
                short_form_decision.error[:80],
            )
            args.pop("bucket", None)
            args.pop("skill_name", None)
            synthesized = None
        else:
            return None, ToolResult(
                # The skill-payload selector refusal is a POLICY denial (v6.57.0),
                # which is what its own first line has always said; the generic
                # argument-error code contradicted it and would have promoted the
                # refusal to an execution failure once the loop reads the code.
                status="blocked",
                code="SKILL_PAYLOAD_BLOCKED",
                text=f"⚠️ SKILL_PAYLOAD_ARG_ERROR: {short_form_decision.error}",
            )

    redirect_err = _registry().cross_skill_redirect_error(task_constraint, synthesized)
    if redirect_err and name in {"write_file", "edit_text"}:
        return None, ToolResult(
            status="blocked",
            code="SKILL_PAYLOAD_BLOCKED",
            text=f"⚠️ SKILL_REDIRECT_BLOCKED: {redirect_err}",
        )
    if task_constraint and task_constraint.has_selected_skill:
        return task_constraint, None
    return synthesized or task_constraint, None


def _blocked_path_note(path_text: Any, spelled: Any = "") -> str:
    """The ``Blocked path:`` clause of both Guard-B messages: the RESOLVED path
    and, when the model's own spelling differs (a relative operand, a symlink
    alias, a POSIX-rooted spelling resolved natively on Windows), that spelling
    too. The windows-latest serial pass named only ``C:\\Users\\Shared\\x`` for
    a command that wrote ``/Users/Shared/x`` — a path the model never typed."""
    resolved = str(path_text or "").strip()
    written = str(spelled or "").strip()
    if not resolved:
        return f" Blocked path: {written}." if written else ""
    if written and written != resolved:
        return f" Blocked path: {resolved} (as written: {written})."
    return f" Blocked path: {resolved}."


def _workspace_write_block_runtime_message(path_text: Any = "", spelled: Any = "") -> str:
    """Guard-B block for a write-shaped command reaching a protected runtime root.

    Names the resolved offending path and the sanctioned route (the light-lane
    message at the runtime_data guard is the in-repo exemplar): five return sites
    used to emit one byte-identical reasonless string, so the log could not even
    say WHICH path fired, and the agent had no route to self-correct.
    """
    return (
        "⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell command mentions Ouroboros system/data paths."
        + _blocked_path_note(path_text, spelled)
        + " Use the gated read_file/write_file tools for runtime data (installed skill"
        " payloads: root=skill_payload with bucket/skill_name, or run the command with"
        " cwd=skill_payload), and keep shell writes inside the selected process root."
    )


def _workspace_write_block_outside_root_message(
    path_text: Any = "", work_dir: Any = "", spelled: Any = "",
) -> str:
    """Guard-B block for a write-shaped command targeting outside the process root."""
    path_note = _blocked_path_note(path_text, spelled)
    root_note = f" Selected process root: {work_dir}." if str(work_dir or "").strip() else ""
    return (
        "⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell commands may not target paths"
        " outside the selected process root." + path_note + root_note
        + " Write inside the process root, or use the file tools with an explicit root"
        " (user_files / task_drive / artifact_store)."
    )


def _executor_backend_candidate_path(ctx: Any, candidate: str) -> pathlib.Path | None:
    """Map one backend spelling lexically, preserving descendant symlinks."""
    try:
        from ouroboros.workspace_executor import executor_ref_from_ctx as _executor_ref_from_ctx
        from ouroboros.workspace_executor import map_backend_path_lexical as _executor_map_backend_path_lexical

        executor_ref = _executor_ref_from_ctx(ctx)
        if executor_ref is None:
            return None
        return _executor_map_backend_path_lexical(executor_ref, candidate)
    except Exception:
        return None


def _git_protected_roots(self) -> list:
    """Ouroboros runtime roots the target-aware git resolver protects, by
    enumeration: the system repo + EVERY data drive the task touches (parent
    drive plus any child / budget drive in task_metadata). Missing a child
    drive here would let git escape into the control plane. ONE enumeration
    for the external-workspace lane and the default (non-workspace) lane."""
    git_protected_roots = [
        pathlib.Path(getattr(self._ctx, "system_repo_dir", None) or self._ctx.repo_dir),
        pathlib.Path(self._ctx.repo_dir),
        pathlib.Path(self._ctx.drive_root),
    ]
    _meta = getattr(self._ctx, "task_metadata", {})
    if isinstance(_meta, dict):
        for _k in ("drive_root", "child_drive_root", "headless_child_drive_root", "budget_drive_root"):
            if _meta.get(_k):
                git_protected_roots.append(pathlib.Path(str(_meta.get(_k))))
    return git_protected_roots


def _resolved_shell_cwd(
    self, args: Dict[str, Any], binding: Any = None,
) -> pathlib.Path | ToolResult:
    """The command's working directory, resolved ONCE through the cwd SSOT.

    Returns a ``pathlib.Path``, or a native cwd denial when
    resolution fails. Every guard downstream takes this canonical path instead
    of re-resolving — or, worse, string-joining the raw cwd label onto a root,
    which is the D1 regression class (v6.74.0)."""
    items = _registry()._binding_items(binding)
    if items:
        return pathlib.Path(items[0].target_path)
    raw_cwd = str(args.get("cwd") or "")
    operation = "service" if str(args.get("__tool_name") or "") == "start_service" else "shell"
    try:
        work_dir, _cwd_root, _allowed = _registry().resolve_shell_cwd(self._ctx, raw_cwd, operation=operation)
    except Exception as exc:
        return ToolResult(
            status="blocked",
            code="SHELL_CWD_BLOCKED",
            text=_registry().shell_cwd_block_message(
                self._ctx, raw_cwd, operation=operation, error=exc,
            ),
        )
    return pathlib.Path(work_dir)


def _direct_shell_write_block(self, raw_cmd: Any, work_dir: pathlib.Path, runtime_mode: str, binding: Any) -> ToolResult | None:
    """Apply existing resource authority to certain direct writes, never mentions."""
    from dataclasses import replace
    from ouroboros.tool_access import _process_root_candidates, _resolve_target_in_selected_base, decide_tool_access, path_is_relative_to
    from ouroboros.tool_access_user_files import UserFilesPathBlockedError
    from ouroboros.tools.deliverables_shell import _command_path
    from ouroboros.tools.shell_guards import direct_utility_target_rows, directory_destination_pairs
    from ouroboros.tools.core import _binding_skill_control_plane_path, is_skill_control_plane_path
    from ouroboros.shell_parse import directory_destination_child_name
    from ouroboros.runtime_mode_policy import mode_allows_protected_write, protected_paths_in
    from ouroboros.tools.shell_audit import _presence_allows_user_output

    rows = direct_utility_target_rows(raw_cmd)
    if not any(row[1] for row in rows):
        return None
    # A command word alone (touch, rm, mkdir, ...) is a guess, not evidence of a
    # write (DEVELOPMENT §2: permission is never reconstructed from command
    # words). For a top-level principal the guess still meets light mode,
    # protected paths and skill trust state below, never the outside-every-root
    # fence (owner decision 5A); a subordinate child keeps its write confinement,
    # so the guess still fences it.
    guess_fences = self._is_acting_subagent() or self._is_local_readonly_subagent()
    certain_rows = direct_utility_target_rows(raw_cmd, certain_only=True)
    items = _registry()._binding_items(binding)
    selected = items[0] if items else _registry().build_resolved_resource_binding(
        self._ctx, operation="shell", process_cwd=str(work_dir))
    roots = [row for row in dict.fromkeys([(selected.root, selected.base_path, selected.source, selected.skill_name),
                                          *_process_root_candidates(self._ctx, "shell")])
             if decide_tool_access(profile=selected.profile, root=row[0], operation="write").allow]
    system_repo = pathlib.Path(getattr(self._ctx, "system_repo_dir", None) or self._ctx.repo_dir).resolve(strict=False)

    def _refuse_write(target: pathlib.Path, token: str, declined: list, light_internal: bool) -> ToolResult:
        # The refusal names the real reason and every root this task may write.
        writable = ", ".join(dict.fromkeys(
            f"{root}={pathlib.Path(base).resolve(strict=False)}" for root, base, _source, _skill in roots
            if not (light_internal and pathlib.Path(base).resolve(strict=False) == system_repo))) or "(none)"
        reason = ("runtime_mode=light keeps the Ouroboros repository and runtime control data read-only"
                  if light_internal else "explicit write target is outside every root this task may write")
        return ToolResult(
            status="blocked", code="LIGHT_MODE_BLOCKED" if light_internal else "WORKSPACE_BLOCKED",
            text=(f"⚠️ {'LIGHT_MODE_BLOCKED' if light_internal else 'WORKSPACE_SHELL_BLOCKED'}: {reason}."
                  + _blocked_path_note(target, token) + f" Writable roots: {writable}."
                  + "".join(f" {why}" for why in declined) + " The process was not started."))

    for (argv, targets, _inline, _unknown), (_argv, certain, _i, _u), cwd in zip(
            rows, certain_rows, _registry().sequential_effective_cwds(rows, work_dir)):
        guessed = set(targets) - set(certain)
        for command, destination, source in directory_destination_pairs(argv):
            directory = _command_path(self._ctx, cwd, destination)
            child = directory_destination_child_name(command, argv, source)
            if directory is not None and directory.is_dir() and child:
                targets = [token for token in targets if token != destination] + [str(directory / child)]
        for token in targets:
            if not token or token == "/dev/null" or any(char in token for char in "$`*?{}~"):
                continue  # Unexpanded/computed words are not concrete target evidence.
            target = _command_path(self._ctx, cwd, token)
            if target is None:
                continue
            # The light gate is root-independent: ``runtime_mode`` here is the
            # EFFECTIVE mode (the install mode capped per task), and a cyber_pro
            # install resolves user_files to the whole host, which would admit a
            # repository target under that name for a light-capped task.
            if runtime_mode == "light" and path_is_relative_to(target, system_repo):
                return _refuse_write(target, token, [], True)
            declined: list[str] = []
            for root, base, source, skill in roots:
                try:
                    resolved = _resolve_target_in_selected_base(
                        self._ctx, root=root, base_path=base, path=str(target), operation="write")
                    target_binding = replace(selected, root=root, base_path=base, target_path=resolved,
                                             operation="write", source=source, skill_name=skill)
                    if (root == "skill_payload" and _binding_skill_control_plane_path(target_binding)
                            or is_skill_control_plane_path(resolved, target_binding.state_drive_root)):
                        return ToolResult(status="blocked", code="SKILL_PAYLOAD_BLOCKED", text=(
                            f"⚠️ SKILL_PAYLOAD_BLOCKED: explicit write target {resolved} is skill control-plane state. "
                            "Edit user-authored payload files instead. The process was not started."))
                    if _registry().binding_targets_system_repo(self._ctx, target_binding):
                        if runtime_mode == "light":
                            continue
                        protected = protected_paths_in([resolved.relative_to(base).as_posix()])
                        if protected and not mode_allows_protected_write(runtime_mode):
                            return _registry()._protected_write_block_result(
                                path=protected[0].path, runtime_mode=runtime_mode, action="let run_command write")
                    # Process output uses the existing shell/write grant owner,
                    # including a remapped Deliverables logical path prefix.
                    if (not _presence_allows_user_output(self._ctx, resolved) if root == "user_files"
                            else not _registry()._presence_binding_allowed(self._ctx, target_binding)):
                        declined.append(f"{root} declines it under the Presence output policy.")
                        continue
                    break
                except UserFilesPathBlockedError as exc:
                    declined.append(f"user_files declines it: {str(exc).removeprefix('user_files path blocked: ')}")
                except (OSError, ValueError, RuntimeError):
                    continue
            else:
                light_internal = runtime_mode == "light" and any(
                    path_is_relative_to(target, root) for root in _git_protected_roots(self))
                if guess_fences or light_internal or token not in guessed:
                    return _refuse_write(target, token, declined, light_internal)
    return None


def _external_workspace_git_block(
    self, raw_cmd: Any, work_dir: pathlib.Path,
) -> ToolResult | None:
    from ouroboros.git_shell_policy import external_workspace_git_violation

    # External-workspace git is no longer confined to the active workspace
    # (host scratch is legitimate); only the enumerated runtime roots are
    # protected. ``work_dir`` is the ALREADY-RESOLVED cwd from the one
    # resolve_shell_cwd call in _shell_git_and_runtime_block — passing it as
    # the base with cwd="" keeps the D1 rule (resolve once, through the SSOT,
    # never re-join a raw cwd label onto a root).
    git_violation = external_workspace_git_violation(
        raw_cmd,
        active_root=work_dir,
        cwd="",
        protected_roots=_git_protected_roots(self),
        allow_network=_resource_allowed(self._ctx, "network"),
    )
    if not git_violation:
        return None
    if git_violation.startswith("task_contract.allowed_resources"):
        return ToolResult(
            status="blocked",
            code="RESOURCE_CONSTRAINT_BLOCKED",
            text=f"⚠️ RESOURCE_CONSTRAINT_BLOCKED: {git_violation}.",
        )
    return ToolResult(
        status="blocked",
        code="WORKSPACE_BLOCKED",
        text=f"⚠️ WORKSPACE_GIT_BLOCKED: {git_violation}.",
    )










def _shell_git_and_runtime_block(
    self, raw_cmd: Any, args: Dict[str, Any], cmd_path_lower: str,
    workspace_mode: bool, acting_self_worktree: bool, binding: Any,
) -> ToolResult | None:
    """Direct-git-via-shell policy + the external-workspace runtime/secret read
    guard. External workspaces AND the default (non-workspace) lane get full
    task-local git through ONE target-aware resolver — only the Ouroboros
    runtime is protected — while raw non-git shell
    in external workspaces still cannot read the runtime/secrets;
    self_worktree keeps the strict read-only git policy."""
    if not _registry().shell_argv(raw_cmd):
        return None
    if workspace_mode and not acting_self_worktree:
        work_dir = _resolved_shell_cwd(self, args, binding)
        if isinstance(work_dir, ToolResult):
            return work_dir
        if git_block := _external_workspace_git_block(self, raw_cmd, work_dir):
            return git_block
        return None
    if workspace_mode:
        # Acting self_worktree: a checkout of the Ouroboros repo itself; the
        # acting-child contract (no commits anywhere — a moved HEAD fails patch
        # capture closed; patch integration) keeps the strict read-only git
        # policy, UNWEAKENED by the target-aware default lane below: both the
        # workspace-escape check and the blanket mutating-git text classifier
        # keep running for this lane.
        work_dir = _resolved_shell_cwd(self, args, binding)
        if isinstance(work_dir, ToolResult):
            return work_dir
        binding_item = _registry()._binding_items(binding)[0]
        active_root = pathlib.Path(binding_item.base_path)
        try:
            binding_cwd = pathlib.Path(work_dir).relative_to(active_root).as_posix()
        except ValueError:
            binding_cwd = ""
        git_violation = _registry().workspace_git_safety_violation(
            raw_cmd,
            active_root=active_root,
            cwd=binding_cwd,
            allow_network=_resource_allowed(self._ctx, "network"),
        )
        if git_violation:
            if git_violation.startswith("task_contract.allowed_resources"):
                return ToolResult(
                    status="blocked",
                    code="RESOURCE_CONSTRAINT_BLOCKED",
                    text=f"⚠️ RESOURCE_CONSTRAINT_BLOCKED: {git_violation}.",
                )
            return ToolResult(
                status="blocked",
                code="WORKSPACE_BLOCKED",
                text=(
                    "⚠️ WORKSPACE_GIT_BLOCKED: run_command may only use read-only git "
                    f"operations inside the active workspace; blocked {git_violation}."
                ),
            )
        git_violation = _registry().run_shell_git_block_reason(
            raw_cmd,
            allow_network=_resource_allowed(self._ctx, "network"),
        )
        if git_violation:
            if git_violation.startswith("task_contract.allowed_resources"):
                return ToolResult(
                    status="blocked",
                    code="RESOURCE_CONSTRAINT_BLOCKED",
                    text=f"⚠️ RESOURCE_CONSTRAINT_BLOCKED: {git_violation}.",
                )
            subcmd = git_violation.removeprefix("git ").strip() or git_violation
            return ToolResult(
                status="blocked",
                code="GIT_VIA_SHELL_BLOCKED",
                text=(
                    f"⚠️ GIT_VIA_SHELL_BLOCKED: `git {subcmd}` is blocked for acting "
                    "self_worktree children (no commits; the parent integrates the "
                    "returned patch and is the sole committer). For read-only git: "
                    "vcs_status, vcs_diff tools, or run_command with git "
                    "log/show/diff/status/rev-list/show-ref/for-each-ref/listing branch-tag forms."
                ),
            )
        return None
    # DEFAULT (non-workspace) lane: mutating git
    # is free EVERYWHERE outside the Ouroboros runtime. The argv-text
    # blanket is replaced by the SAME target-aware resolver the external
    # lane runs since v6.27: read-only git allowed even at a runtime
    # target; mutating git blocked only when it TARGETS the runtime
    # (bidirectional/casefold/symlink-resolved); the contract network
    # fence rides along. The cwd resolves EXACTLY ONCE via the shared
    # resolver, passed canonical — never re-join a raw label onto a root
    # (the v6.74.0 D1 class). Disclosed residual (no shell-parser arms
    # race): git via wrapper (nice/xargs) or interpreter code is not
    # classified here; the LLM safety layer still reviews intent, and the
    # light-mode post-exec dirtiness tripwire backstops.
    work_dir = _resolved_shell_cwd(self, args, binding)
    if isinstance(work_dir, ToolResult):
        return work_dir
    from ouroboros.git_shell_policy import external_workspace_git_violation

    git_violation = external_workspace_git_violation(
        raw_cmd,
        active_root=work_dir,
        cwd="",
        protected_roots=_git_protected_roots(self),
        allow_network=_resource_allowed(self._ctx, "network"),
    )
    if not git_violation:
        return None
    if git_violation.startswith("task_contract.allowed_resources"):
        return ToolResult(
            status="blocked",
            code="RESOURCE_CONSTRAINT_BLOCKED",
            text=f"⚠️ RESOURCE_CONSTRAINT_BLOCKED: {git_violation}.",
        )
    return ToolResult(
        status="blocked",
        code="GIT_VIA_SHELL_BLOCKED",
        text=(
            f"⚠️ GIT_VIA_SHELL_BLOCKED: {git_violation}. Mutating git may not target "
            "the Ouroboros runtime (system repo / data drives): self-repo changes go "
            "through commit_reviewed, which enforces pre-commit checks and review. "
            "Read-only git (status/log/diff/show/rev-parse/branch- and tag-listing, "
            "or the vcs_status/vcs_diff tools) works everywhere, and mutating git is "
            "free in any tree OUTSIDE the runtime (e.g. ~/projects, /tmp, an attached "
            "project folder)."
        ),
    )
