"""Host-owned pre-dispatch guards: capability/resource, ephemeral, managed-update and skill-payload constraints.

Every span is extracted VERBATIM from the parent's tip bytes by
scripts/v7next_transplant.py (D18/D33 module-handle split, proof-checked);
the parent re-exports every moved name, so historical imports and
monkeypatch targets keep working unchanged.
"""

from __future__ import annotations

import logging
import os
import pathlib
import re

from typing import TYPE_CHECKING

from ouroboros.artifacts import task_artifact_dir_path, task_id_for_artifacts
from ouroboros.tools.tool_result import ToolResult
from ouroboros.tools.write_shape import _no_deliverables_decision, _workspace_write_candidates

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
    at import time (the owner-approved D18/D33 mechanical exception).
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


def _command_mentions_protected_root(cmd_path_lower: str, root_text: str) -> bool:
    """Boundary-aware path containment for the workspace shell guard.

    True only when ``root_text`` (a normalised, lower-cased protected root path)
    appears in the command as a whole path or a parent prefix at a real path
    boundary — NOT as an incidental substring of an unrelated path that merely
    shares the prefix (e.g. protected ``/x/data`` must not match ``/x/database``).
    Used as a coarse catch-all for runtime paths embedded in non-tokenised text
    (e.g. inside a ``python -c`` string); the precise per-token containment loop
    still does the authoritative active/protected classification.
    """
    if not root_text:
        return False
    norm = root_text.rstrip("/")
    if not norm:
        return False
    span = len(norm)
    limit = len(cmd_path_lower)
    start = 0
    while True:
        idx = cmd_path_lower.find(norm, start)
        if idx < 0:
            return False
        end = idx + span
        nxt = cmd_path_lower[end] if end < limit else ""
        # Boundary = end-of-string, a path separator (child path), or a shell
        # token delimiter (the exact path). A trailing path char (letter/digit/
        # ``.``/``-``/``_``) means a DIFFERENT sibling path → keep scanning.
        if nxt == "" or nxt == "/" or nxt in " \t\"')(;:,&|<>":
            return True
        start = end


def _stray_skill_payload_failsoft(root_arg: str, workspace_mode: bool, task_constraint: Any) -> bool:
    """Whether stray bucket/skill_name on a write tool should be DROPPED rather than
    surfaced as SKILL_PAYLOAD_ARG_ERROR. Fail-soft ONLY for a WORKSPACE edit that is
    NOT skill-authoring: there bucket/skill_name are model noise (the B2 footgun —
    reflexive bucket="external" on an /app edit). In light/advanced non-workspace
    skill-authoring (or an explicit root=skill_payload / skill_repair) the specific
    error is the intended helpful signal."""
    skill_payload_intent = root_arg == "skill_payload" or bool(
        task_constraint and getattr(task_constraint, "mode", "") == "skill_repair"
    )
    return bool(workspace_mode and not skill_payload_intent)


def _managed_update_code_tool_block_result(ctx: Any, name: str) -> ToolResult | None:
    """Block a repo-mutating code tool while a managed-update assisted merge is staged for
    ANOTHER task (P2/SC2). Returns a blocked result, or ``None`` when allowed (this is the
    authorized resolution task, or no managed tx is active). A corrupt tx marker fails closed."""
    try:
        from supervisor.update_merge import managed_assisted_tx_for

        if managed_assisted_tx_for(
            getattr(ctx, "task_id", ""),
            getattr(ctx, "task_metadata", None),
        )[1]:
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
    if acting_subagent and entry is not None and name not in _registry().ACTING_SUBAGENT_TOOL_NAMES:
        return ToolResult(status="blocked", code="ACCESS_BLOCKED", text=(
            "⚠️ ACTING_SUBAGENT_BLOCKED: this mutative subagent may read and "
            "write inside its isolated write root and run shell/services "
            f"there, but may not call first-party tool {name!r}. It cannot "
            "commit the live body, run review/runtime/skills lifecycle, enable "
            "tools, or write cognitive memory; the parent integrates the "
            "returned patch and is the sole committer."
        ))
    if acting_subagent and entry is None and (ext_tool or is_mcp) and name not in acting_tool_grants:
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


def _task_constraint_path_allowed(path_text: str, constraint: Optional[TaskConstraint], drive_root: pathlib.Path) -> bool:
    return _registry().is_skill_payload_path(
        drive_root,
        path_text or "",
        constraint=constraint,
        allow_short_relative=True,
        allow_control_plane=True,
    )


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


_HEAL_MODE_ALLOWED_TOOLS = frozenset({
    "read_file",
    "list_files",
    "write_file",
    "edit_text",
    "list_skills",
    "skill_review", "skill_preflight",
})


def _heal_protected_payload_sidecar(path_text: str) -> bool:
    return _registry().is_skill_payload_control_filename(path_text)


def _heal_mode_guard_result(
    ctx: Any,
    name: str,
    args: dict[str, Any],
    task_constraint: TaskConstraint | None,
    ext_tool: Any,
    is_mcp: bool,
) -> ToolResult | None:
    """Apply skill-repair confinement in the established pre-dispatch order."""
    heal_skill = task_constraint.skill_name if task_constraint else ""
    if (
        name in {"read_file", "list_files", "write_file", "edit_text"}
        and str(args.get("root", "") or "") == "skill_payload"
    ):
        expected_bucket, expected_skill = _registry().constraint_bucket_skill(task_constraint)
        requested_bucket = str(args.get("bucket", "") or "").strip()
        requested_skill = str(args.get("skill_name", "") or "").strip()
        if (
            (requested_bucket and requested_bucket != expected_bucket)
            or (requested_skill and requested_skill != expected_skill)
        ):
            if name in {"write_file", "edit_text"}:
                return ToolResult(
                    status="blocked",
                    code="HEAL_MODE_BLOCKED",
                    text=(
                        "⚠️ SKILL_REDIRECT_BLOCKED: active skill_repair "
                        "task is scoped to the selected skill payload."
                    ),
                )
            return ToolResult(
                status="blocked",
                code="HEAL_MODE_BLOCKED",
                text=(
                    "⚠️ HEAL_MODE_BLOCKED: Repair payload access is limited "
                    "to the selected skill payload."
                ),
            )
    if name in {"read_file", "write_file"} and str(args.get("root", "") or "") == "skill_payload":
        payload_paths = []
        maybe_path = str(args.get("path", "") or "")
        if maybe_path:
            payload_paths.append(maybe_path)
        for f_entry in args.get("files") or []:
            if isinstance(f_entry, dict):
                payload_paths.append(str(f_entry.get("path", "") or ""))
        for payload_path in payload_paths or ["."]:
            if not _task_constraint_path_allowed(
                payload_path,
                task_constraint,
                pathlib.Path(ctx.drive_root),
            ):
                return ToolResult(
                    status="blocked",
                    code="HEAL_MODE_BLOCKED",
                    text=(
                        "⚠️ HEAL_MODE_BLOCKED: Repair data access is limited "
                        "to the selected skill payload under data/skills/external "
                        "data/skills/clawhub, or data/skills/ouroboroshub."
                    ),
                )
            if name == "write_file" and _heal_protected_payload_sidecar(payload_path):
                return ToolResult(
                    status="blocked",
                    code="HEAL_MODE_BLOCKED",
                    text=(
                        "⚠️ HEAL_MODE_BLOCKED: Repair may not edit marketplace "
                        "or official provenance sidecars (.clawhub.json, "
                        ".ouroboroshub.json, SKILL.openclaw.md, .seed-origin). "
                        "Edit the user-authored payload files instead."
                    ),
                )
    if name == "list_files" and str(args.get("root", "") or "") == "skill_payload":
        data_dir = str(args.get("path", "") or "")
        if not _task_constraint_path_allowed(
            data_dir,
            task_constraint,
            pathlib.Path(ctx.drive_root),
        ):
            return ToolResult(
                status="blocked",
                code="HEAL_MODE_BLOCKED",
                text=(
                    "⚠️ HEAL_MODE_BLOCKED: Repair data listing is limited "
                    "to the selected skill payload under data/skills/external "
                    "data/skills/clawhub, or data/skills/ouroboroshub."
                ),
            )
    if name == "edit_text":
        edit_path = str(args.get("path", "") or "")
        if not _task_constraint_path_allowed(
            edit_path,
            task_constraint,
            pathlib.Path(ctx.drive_root),
        ):
            return ToolResult(
                status="blocked",
                code="HEAL_MODE_BLOCKED",
                text="⚠️ HEAL_MODE_BLOCKED: Repair edit_text is limited to the selected skill payload.",
            )
        if _heal_protected_payload_sidecar(edit_path):
            return ToolResult(
                status="blocked",
                code="HEAL_MODE_BLOCKED",
                text=(
                    "⚠️ HEAL_MODE_BLOCKED: Repair may not edit marketplace "
                    "or official provenance sidecars (.clawhub.json, "
                    ".ouroboroshub.json, SKILL.openclaw.md, .seed-origin). "
                    "Edit the user-authored payload files instead."
                ),
            )
    if name == "skill_review" and str(args.get("skill", "") or "").strip() != heal_skill:
        return ToolResult(
            status="blocked",
            code="HEAL_MODE_BLOCKED",
            text="⚠️ HEAL_MODE_BLOCKED: Repair may only review the selected skill.",
        )
    if name == "skill_preflight" and str(args.get("skill", "") or "").strip() != heal_skill:
        return ToolResult(
            status="blocked",
            code="HEAL_MODE_BLOCKED",
            text="⚠️ HEAL_MODE_BLOCKED: Repair may only preflight the selected skill.",
        )
    if ext_tool or is_mcp or name not in _HEAL_MODE_ALLOWED_TOOLS:
        return ToolResult(
            status="blocked",
            code="HEAL_MODE_BLOCKED",
            text=(
                "⚠️ HEAL_MODE_BLOCKED: Repair tasks may inspect/edit skill "
                "payloads and run skill_review only. Shell, browser automation, "
                "repo mutation, skill execution, extension tools, MCP tools, "
                "delegation, and enable/disable flows are unavailable. Use "
                "the Skills UI after a fresh executable review."
            ),
        )
    return None


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
        for name in ("web", "allow_web", "internet", "external_network"):
            value = resources.get(name)
            if isinstance(value, bool) and not value:
                return False
    return True


def _disabled_tools(ctx: Any) -> frozenset:
    """Tool names the task contract withholds (declarative tool policy).

    Independent of ``allowed_resources``: a caller can disable specific tools
    (e.g. the agent's web_search/browser/VLM tools for a faithful benchmark)
    WITHOUT setting web/network=false — so shell network egress (git/pip) stays
    available and the web<->network cross-implication in ``_resource_allowed``
    never fires.
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
    # Q1 rename compatibility: contracts that withheld `advisory_review` keep
    # withholding the SAME organ under its new name, and vice versa (a new
    # contract naming only the new spelling must also silence the alias).
    if "advisory_review" in names:
        names.add("preflight_review")
    if "preflight_review" in names:
        names.add("advisory_review")
    return frozenset(names)


_GITHUB_TOKEN_TOOLS = frozenset({
    "list_github_prs",
    "get_github_pr",
    "comment_on_pr",
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
    if tool in _GITHUB_TOKEN_TOOLS and not os.environ.get("GITHUB_TOKEN", "").strip():
        return False, "missing_credential", "GITHUB_TOKEN"
    return True, "", ""


def _capability_resource_guard_result(
    ctx: Any,
    name: str,
    args: dict[str, Any],
    ext_tool: Any = None,
    is_mcp: bool = False,
) -> ToolResult | None:
    """Apply direct task capability and resource admission in legacy order."""
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
            and task_constraint.mode == "skill_repair"
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
            code="HEAL_MODE_BLOCKED",
            text=f"⚠️ SKILL_REDIRECT_BLOCKED: {redirect_err}",
        )
    if task_constraint and task_constraint.mode == "skill_repair":
        return task_constraint, None
    return synthesized or task_constraint, None


_EPHEMERAL_ALLOWED_TOOLS = frozenset({
    # read / inspect
    "read_file", "query_code", "search_code", "list_files", "web_search", "browse_page",
    "chat_history", "recent_tasks", "get_task_result", "vcs_diff", "vcs_status",
    "analyze_screenshot", "vlm_query",
    # decide / route / spawn-owner-task / reply
    "route_to_project", "promote_chat_to_task", "steer_task", "list_projects", "send_photo",
})


def _ephemeral_block_result(
    ctx: Any,
    name: str,
    ext_tool: Any = None,
    is_mcp: bool = False,
) -> ToolResult | None:
    """CW3: a short ephemeral decision turn may call ONLY the allowlisted read/decision
    tools (_EPHEMERAL_ALLOWED_TOOLS); every other built-in (durable/control/review/skill
    mutator, run_command) AND all extension/MCP tools fail closed. Default-deny, so a new
    mutator can never silently become reachable. It answers inline or promote_chat_to_task's
    the durable work into a supervised task."""
    if not getattr(ctx, "is_ephemeral_turn", False):
        return None
    if ext_tool or is_mcp:
        text = (
            f"⚠️ EPHEMERAL_TURN_RESTRICTED: external tool '{name}' can have durable side "
            "effects, which a short same-route decision turn must not do. Answer inline, "
            "or promote_chat_to_task to do that work in a supervised task."
        )
    elif name not in _EPHEMERAL_ALLOWED_TOOLS:
        text = (
            f"⚠️ EPHEMERAL_TURN_RESTRICTED: '{name}' is not in the decision-turn allowlist "
            "(read/inspect + answer/route/spawn/steer only) — a short same-route turn must "
            "not do durable/control/review/skill work or run shell. Answer inline, or "
            "promote_chat_to_task to do it in a supervised task."
        )
    else:
        return None
    return ToolResult(status="blocked", code="ACCESS_BLOCKED", text=text)


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


def _workspace_write_block_runtime_result(path_text: Any = "", spelled: Any = "") -> ToolResult:
    """Typed carrier of the Guard-B runtime-path denial (same bytes in text)."""
    return ToolResult(
        status="blocked",
        code="WORKSPACE_BLOCKED",
        text=_workspace_write_block_runtime_message(path_text, spelled),
    )


def _workspace_write_block_outside_root_result(
    path_text: Any = "", work_dir: Any = "", spelled: Any = "",
) -> ToolResult:
    """Typed carrier of the Guard-B outside-root denial (same bytes in text)."""
    return ToolResult(
        status="blocked",
        code="WORKSPACE_BLOCKED",
        text=_workspace_write_block_outside_root_message(path_text, work_dir, spelled),
    )


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


def _external_runtime_protected_paths(
    self, binding: Any = None,
) -> tuple[list, list, list, list]:
    """Ouroboros runtime roots that an EXTERNAL-workspace task must not touch via
    shell (system repo + EVERY data drive incl child/budget + owner credential
    locations) plus the task's own exempt task_drive/artifact_store roots. Returns
    (protected_texts, allowed_texts, protected_paths, allowed_paths): the *_texts
    feed the embedded-string boundary check; the *_paths feed token resolution
    (relative->cwd, ~->home, symlink canonicalization) so relative/symlink bypasses
    are closed. SSOT for the read + write guards."""
    meta = getattr(self._ctx, "task_metadata", {}) if isinstance(getattr(self._ctx, "task_metadata", {}), dict) else {}
    protected_values = [getattr(self._ctx, "system_repo_dir", None) or getattr(self._ctx, "repo_dir", None),
                        getattr(self._ctx, "drive_root", None)]
    try:
        from ouroboros.config import DATA_DIR as _PARENT_DATA_DIR
        protected_values.append(_PARENT_DATA_DIR)
    except Exception:
        pass
    for _dk in ("drive_root", "child_drive_root", "headless_child_drive_root", "budget_drive_root"):
        if meta.get(_dk):
            protected_values.append(meta.get(_dk))
    # Owner/runtime credential locations, as ABSOLUTE paths. Blocking by
    # absolute containment (not a substring marker) means the OWNER's personal
    # secrets (~/.ssh/id_rsa, ~/.aws, ~/file1.txt) are off-limits while a
    # project-relative file merely NAMED like a credential (site/.ssh/config, a
    # project .env) stays the task's own — and a non-path token like
    # "os.environ" can never spuriously match.
    try:
        from ouroboros.credential_shapes import owner_credential_locations

        owner_paths, owner_configs = owner_credential_locations(pathlib.Path.home())
        protected_values.extend(owner_paths)
    except Exception:
        owner_configs = []
    def _text_forms(value: Any) -> list:
        # Both the as-given and the symlink-resolved form, so a command using
        # /var/... matches a root resolved to /private/var/... (macOS) and vice
        # versa. In production ($HOME paths) the two coincide.
        out = []
        for variant in (value, None):
            try:
                p = pathlib.Path(value)
                if variant is None:
                    p = p.resolve(strict=False)
                t = str(p).replace("\\", "/").lower().rstrip("/")
                if t and t not in out:
                    out.append(t)
            except Exception:
                continue
        return out

    def _resolved(value: Any):
        try:
            return pathlib.Path(value).resolve(strict=False)
        except Exception:
            return None

    protected_texts: list = []
    protected_paths: list = []
    for v in protected_values:
        if not v:
            continue
        for t in _text_forms(v):
            if t not in protected_texts:
                protected_texts.append(t)
        rp = _resolved(v)
        if rp is not None and rp not in protected_paths:
            protected_paths.append(rp)
    allowed_texts = [text for path in owner_configs for text in _text_forms(path)]
    allowed_paths: list = []  # owner_configs are exact files, not allowed subtrees
    task_id = task_id_for_artifacts(self._ctx)
    for data_root in (getattr(self._ctx, "drive_root", None), meta.get("drive_root"), meta.get("budget_drive_root")):
        if not data_root:
            continue
        for rp_src in (pathlib.Path(data_root) / "task_drives" / task_id, task_artifact_dir_path(pathlib.Path(data_root), task_id, create=False)):
            for t in _text_forms(rp_src):
                if t not in allowed_texts:
                    allowed_texts.append(t)
            rp = _resolved(rp_src)
            if rp is not None and rp not in allowed_paths:
                allowed_paths.append(rp)
    # An explicitly selected system repo or exact skill payload is an
    # authorized process target. Keep every other runtime/credential root
    # protected, but do not re-block that exact binding merely because the
    # task also has an external workspace focus.
    for item in _registry()._binding_items(binding):
        if item.root not in {"system_repo", "skill_payload"}:
            continue
        selected = pathlib.Path(item.base_path)
        for t in _text_forms(selected):
            if t not in allowed_texts:
                allowed_texts.append(t)
        rp = _resolved(selected)
        if rp is not None and rp not in allowed_paths:
            allowed_paths.append(rp)
    return protected_texts, allowed_texts, protected_paths, allowed_paths


def _external_shell_runtime_or_secret_block(
    self, raw_cmd: Any, cmd_path_lower: str, args: Dict[str, Any],
    work_dir: Optional[pathlib.Path] = None,
    binding: Any = None,
) -> ToolResult | None:
    """External-workspace shell guard for READ and write commands alike: block any
    command that targets the Ouroboros runtime (system repo / any data drive) or an
    owner credential path. read_file/user_files already enforce this; raw shell
    (cat, python -c open(...), etc.) would otherwise bypass it. Two layers, because
    string matching alone is bypassable by relative paths and symlinks:
      (1) embedded-string boundary match of ABSOLUTE protected roots (catches a path
          literal inside e.g. python -c "open('/abs/data/settings.json')");
      (2) path-token RESOLUTION — every path-like arg is expanduser'd, joined to the
          command cwd when relative, and resolve()'d (canonicalizing symlinks + ..),
          then containment-checked. This closes a relative path passed as its own
          argv token (`cat ../../data/settings.json`) and a workspace-internal symlink
          to the data drive (round-2 review).
    Both layers are best-effort DEFENSE-IN-DEPTH, not the primary control: a relative
    path hidden INSIDE an interpreter one-liner string (e.g. node -e
    "readFileSync('../../data/settings.json')") is not a standalone token, so it is
    not extracted here — and that residual is deliberately NOT chased with a regex
    over code strings (an unwinnable arms race; BIBLE P5 / no-string-gate doctrine).
    The PRIMARY control is the gated read_file/user_files path, which fully resolves
    and containment-checks every read against the protected drives, plus the LLM
    safety supervisor judging intent on each shell call."""
    _BLOCK = ToolResult(
        status="blocked",
        code="WORKSPACE_BLOCKED",
        text=(
        "⚠️ WORKSPACE_SHELL_BLOCKED: shell command targets the Ouroboros runtime "
        "(system repo / data drive) or an owner credential path. External-workspace "
        "tasks may not read or write those; use the gated read_file tool for any "
        "inspection you need. Run your command against the task's own surfaces "
        "instead: the active workspace root (e.g. /app) or scratch such as /tmp."
        ),
    )
    protected_texts, allowed_texts, protected_paths, allowed_paths = (
        _external_runtime_protected_paths(self, binding)
    )
    # (1) embedded-string boundary match (absolute roots only — no substring secret
    # markers, which would false-block the task's own project files / "os.environ").
    for pt in protected_texts:
        if _command_mentions_protected_root(cmd_path_lower, pt) and not any(
            _command_mentions_protected_root(cmd_path_lower, t) for t in allowed_texts
        ):
            return _BLOCK
    # (2) path-token resolution (relative -> cwd, ~ -> home, symlinks canonicalized).
    # The cwd is resolved ONCE per safety check by the caller (D1); resolve here
    # only when this guard is used standalone.
    if work_dir is None:
        resolved_cwd = _resolved_shell_cwd(self, args, binding)
        if isinstance(resolved_cwd, ToolResult):
            return resolved_cwd
        work_dir = pathlib.Path(resolved_cwd)
    work_dir = pathlib.Path(work_dir)

    def _within(child: pathlib.Path, parent: pathlib.Path) -> bool:
        try:
            child.relative_to(parent)
            return True
        except ValueError:
            return False

    for tok in _registry().shell_argv_with_path_tokens(raw_cmd):
        tok_text = str(tok or "").strip()
        if not tok_text or tok_text.startswith("-") or tok_text in {"|", "&&", "||", ";", ">", ">>", "<", "<<", "&"}:
            continue
        try:
            p = pathlib.Path(tok_text).expanduser()
            resolved = p.resolve(strict=False) if p.is_absolute() else (work_dir / p).resolve(strict=False)
        except Exception:
            continue
        if str(resolved).replace("\\", "/").lower() in allowed_texts:
            continue
        if any(_within(resolved, ap) for ap in allowed_paths):
            continue
        if any(_within(resolved, pp) for pp in protected_paths):
            return _BLOCK
    return None


def _protected_shell_block(
    self, raw_cmd, cmd_path_lower, binding, acting_self_worktree, writeish,
) -> ToolResult | None:
    """Apply payload/core write guards to the selected physical target."""
    items = _registry()._binding_items(binding)
    targets_skill = bool(items) and all(item.root == "skill_payload" for item in items)
    targets_system = (
        _registry()._binding_set_targets_system_repo(self._ctx, binding)
        or acting_self_worktree
    )
    if (targets_skill or targets_system) and any(
        name in cmd_path_lower
        for name in (
            *_registry().SKILL_PAYLOAD_CONTROL_FILENAMES,
            *(_registry().SKILL_PAYLOAD_CONTROL_DIRNAMES - {"__pycache__"}),
        )
    ) and writeish:
        return ToolResult(
            status="blocked",
            code="SAFETY_VIOLATION",
            text=(
                "⚠️ SAFETY_VIOLATION: Shell command would modify a skill "
                "provenance / launcher seed / dependency marker (.clawhub.json, "
                ".ouroboroshub.json, .self_authored.json, SKILL.openclaw.md, .seed-origin, "
                ".ouroboros_env, node_modules). "
                "Use marketplace lifecycle flows or edit user-authored "
                "payload files instead."
            ),
        )
    if _authorized_managed_update_resolver(self._ctx):
        return None
    if targets_system and _registry().shell_writer_targets_protected(raw_cmd):
        return ToolResult(
            status="blocked",
            code="SAFETY_VIOLATION",
            text=(
                "⚠️ CRITICAL SAFETY_VIOLATION: Shell command would modify "
                "a protected core/contract/release file. Protected: "
                + ", ".join(sorted(_registry().PROTECTED_RUNTIME_PATHS))
            ),
        )
    if targets_system:
        for cf in _registry().PROTECTED_RUNTIME_PATHS_LOWER:
            # The MODE-AWARE composition fact, not the coarse legacy scan: a
            # pure read that merely mentions a protected name (`grep -n delete
            # ouroboros/safety.py`, `sed -n 1,40p BIBLE.md`, a python open('r'))
            # is not a modification; every write shape still blocks here.
            if cf in cmd_path_lower and writeish:
                return ToolResult(
                    status="blocked",
                    code="SAFETY_VIOLATION",
                    text=(
                        "⚠️ CRITICAL SAFETY_VIOLATION: Shell command would modify "
                        "a protected core/contract/release file. Protected: "
                        + ", ".join(sorted(_registry().PROTECTED_RUNTIME_PATHS))
                    ),
                )
    return None


def _workspace_shell_write_block(
    self,
    args: Dict[str, Any],
    raw_cmd: Any,
    cmd_path_lower: str,
    explicit_write_targets: list[str],
    target_rows: list,
    executable_path_tokens: set[str],
    runtime_mode: str,
    acting_subagent: bool,
    binding: Any,
) -> ToolResult | None:
    """Authorize root writes by resource; keep children inside their write surface."""

    items = _registry()._binding_items(binding)
    if not items:
        return ToolResult(
            status="blocked",
            code="WORKSPACE_BLOCKED",
            text="⚠️ WORKSPACE_SHELL_BLOCKED: process target was not resolved.",
        )
    deliverables_block = ToolResult(
        status="blocked",
        code="WORKSPACE_BLOCKED",
        text="⚠️ WORKSPACE_SHELL_BLOCKED: write-like shell target is not an allowed Deliverables path.",
    )
    selected = items[0]
    work_dir = pathlib.Path(selected.target_path).resolve(strict=False)
    selected_base = pathlib.Path(selected.base_path).resolve(strict=False)
    allowed_relative_roots = list(dict.fromkeys((selected_base, work_dir)))
    allowed_data_roots: list[pathlib.Path] = []
    meta = (
        getattr(self._ctx, "task_metadata", {})
        if isinstance(getattr(self._ctx, "task_metadata", {}), dict)
        else {}
    )
    for data_root in (getattr(self._ctx, "drive_root", None), meta.get("budget_drive_root")):
        if not data_root:
            continue
        task_id = task_id_for_artifacts(self._ctx)
        for root_path in (
            pathlib.Path(data_root) / "task_drives" / task_id,
            task_artifact_dir_path(pathlib.Path(data_root), task_id, create=False),
        ):
            resolved_root = pathlib.Path(root_path).resolve(strict=False)
            if resolved_root not in allowed_data_roots:
                allowed_data_roots.append(resolved_root)
    if selected.root in {"task_drive", "artifact_store"}:
        allowed_data_roots.append(selected_base)
    # Executor-backed commands use backend path spellings (for example
    # ``/deliverables/report.html``), while the policy roots above are host
    # paths. Keep the configured Deliverables root separate from the generic
    # allow-root list: every descendant must still pass the target-specific
    # user-files policy (hidden/credential/symlink checks) below.
    deliverables_root_lexical: pathlib.Path | None = None
    deliverables_root_lexical_alias: pathlib.Path | None = None
    deliverables_root_physical: pathlib.Path | None = None
    try:
        candidate_deliverables = _registry().resource_root_path(self._ctx, "deliverables")
        # Retain the configured spelling even when the root itself is
        # malformed or protected. Descendants must then take the
        # target-specific path and fail closed before a broader
        # workspace/data allow-root can accidentally admit them.
        deliverables_root_physical = pathlib.Path(candidate_deliverables).resolve(strict=False)
        deliverables_root_lexical = _registry()._deliverables_root_lexical()
        deliverables_root_lexical_alias = _registry()._deliverables_root_lexical_alias()
    except (OSError, TypeError, ValueError, RuntimeError):
        pass

    def _deliverables_target_decision(path: pathlib.Path) -> bool | None:
        """Decide Deliverables descendants before generic root admission.

        Deliverables can be configured inside the selected workspace.  A
        generic workspace-root fast path must not skip the user-files
        hidden/credential/symlink checks for such a target. ``None`` means
        that the candidate is outside Deliverables and may use the normal
        workspace-root checks.
        """
        if deliverables_root_lexical is None:
            return None
        try:
            lexical_path = pathlib.Path(path).expanduser()
            if not lexical_path.is_absolute():
                lexical_path = pathlib.Path(os.path.abspath(lexical_path))
            in_deliverables = (
                deliverables_root_lexical is not None
                and (
                    lexical_path.is_relative_to(deliverables_root_lexical)
                    or _registry()._lexical_path_is_relative_to_casefold(lexical_path, deliverables_root_lexical)
                    or _registry()._lexical_path_is_relative_to_casefold(
                        lexical_path, deliverables_root_lexical_alias,
                    )
                    or _registry()._lexical_path_is_relative_to_casefold(
                        lexical_path, deliverables_root_physical,
                    )
                )
            )
        except (OSError, TypeError, ValueError):
            return False
        resolved_path = pathlib.Path(path).resolve(strict=False)
        physically_in_deliverables = (
            deliverables_root_physical is not None
            and (
                resolved_path.is_relative_to(deliverables_root_physical)
                or _registry()._path_is_relative_to_casefold(
                    resolved_path, deliverables_root_physical,
                )
            )
        )
        if not in_deliverables and not physically_in_deliverables:
            return None
        return self._deliverables_shell_target_allowed(
            resolved_path,
            lexical_candidate=lexical_path,
        )

    if direct_target_block := _registry().direct_deliverable_target_block(
        self._ctx,
        work_dir,
        [list(row[0]) for row in target_rows],
        deliverables_root_physical,
        _deliverables_target_decision,
    ):
        return ToolResult(
            status="blocked",
            code="WORKSPACE_BLOCKED",
            text=direct_target_block,
        )

    allowed_write_roots = [*allowed_relative_roots, *allowed_data_roots]
    # The root's existing user_files authority is independent of cwd.
    # Acting children retain their isolated write surface in every mode.
    pro_workspace_passthrough = (
        str(runtime_mode or "").strip().lower() == "pro" and not acting_subagent
    )
    protected_roots = [
        getattr(self._ctx, "system_repo_dir", None) or getattr(self._ctx, "repo_dir", None),
        getattr(self._ctx, "drive_root", None),
    ]
    try:
        from ouroboros.config import DATA_DIR as parent_data_dir

        protected_roots.append(parent_data_dir)
    except Exception:
        pass
    for key in ("drive_root", "child_drive_root", "headless_child_drive_root", "budget_drive_root"):
        if meta.get(key):
            protected_roots.append(meta.get(key))
    allowed_texts = [
        str(root).replace("\\", "/").lower().rstrip("/")
        for root in [*allowed_relative_roots, *allowed_data_roots]
    ]
    protected_paths = []
    for root_value in protected_roots:
        try:
            root_path = pathlib.Path(root_value).resolve(strict=False)
        except Exception:
            continue
        protected_paths.append(root_path)
        if any(root_path.is_relative_to(root) for root in allowed_relative_roots):
            continue
        root_text = str(root_path).replace("\\", "/").lower()
        if _command_mentions_protected_root(cmd_path_lower, root_text) and not any(
            _command_mentions_protected_root(cmd_path_lower, text)
            for text in allowed_texts
        ):
            return _workspace_write_block_runtime_result(root_path)
    # Deliverables is a TARGET policy: a merely mentioned path takes no
    # Deliverables decision, while every candidate keeps the
    # protected-runtime-root scans below.
    row_cwds = _registry().sequential_effective_cwds(target_rows, work_dir)
    for token_text, is_write, row_index in _workspace_write_candidates(
        target_rows, explicit_write_targets, raw_cmd,
    ):
        candidate_cwd = row_cwds[row_index] if 0 <= row_index < len(row_cwds) else work_dir
        decide_deliverables = (
            _deliverables_target_decision if is_write else _no_deliverables_decision
        )
        if token_text in executable_path_tokens and not is_write:
            continue
        candidates = [token_text] if _registry().is_absolute_path_text(token_text) else []
        if token_text.startswith(("./", "../")):
            candidates.append(token_text)
        elif (
            token_text
            and not token_text.startswith("-")
            and token_text not in {"|", "&&", "||", ";", ">", ">>", "<", "<<"}
            and (
                token_text in explicit_write_targets
                or "/" in token_text
                or "\\" in token_text
            )
        ):
            candidates.append(token_text)
        for candidate in candidates:
            root_write_allowed = pro_workspace_passthrough
            if is_write and not acting_subagent and not root_write_allowed:
                try:
                    _registry().build_resolved_resource_binding(
                        self._ctx, root="user_files", operation="write",
                        path=str((candidate_cwd / pathlib.Path(candidate)).resolve(strict=False)),
                    )
                    root_write_allowed = True
                except (OSError, ValueError, RuntimeError):
                    pass
            if candidate == "/dev/null":
                continue
            if _registry().is_absolute_path_text(candidate):
                mapped_executor_lexical = _executor_backend_candidate_path(self._ctx, candidate)
                if mapped_executor_lexical is not None:
                    mapped_executor = mapped_executor_lexical.resolve(strict=False)
                    deliverables_decision = decide_deliverables(mapped_executor_lexical)
                    if deliverables_decision is not None:
                        if deliverables_decision:
                            continue
                        return deliverables_block
                    if any(mapped_executor.is_relative_to(root) for root in allowed_write_roots):
                        continue
                    for protected_path in protected_paths:
                        try:
                            mapped_executor.relative_to(protected_path)
                            return _workspace_write_block_runtime_result(mapped_executor, candidate)
                        except Exception:
                            pass
                if _executor_backend_candidate_allowed(
                    self._ctx,
                    candidate,
                    allowed_write_roots,
                ):
                    continue
                windows_drive_path = bool(re.match(r"^[A-Za-z]:[\\/]", candidate))
                unc_path = candidate.startswith("\\\\")
                # On the native Windows host, resolve drive paths exactly as
                # POSIX paths are resolved below. This canonicalizes directory
                # symlinks/junctions before containment: a workspace alias stays
                # allowed, while an in-workspace spelling whose nested link exits
                # the root is blocked. Keep lexical handling for foreign Windows
                # spellings seen on POSIX and for UNC paths (which may require a
                # network lookup merely to evaluate the guard).
                if (not windows_drive_path and not unc_path) or (
                    os.name == "nt" and windows_drive_path
                ):
                    try:
                        resolved = pathlib.Path(candidate).resolve(strict=False)
                    except Exception:
                        continue
                    # Keep the pre-resolution spelling so a symlink child
                    # cannot resolve into another allowed root and bypass
                    # the Deliverables policy.
                    deliverables_decision = decide_deliverables(pathlib.Path(candidate))
                    if deliverables_decision is not None:
                        if deliverables_decision:
                            continue
                        return deliverables_block
                    if any(resolved.is_relative_to(root) for root in allowed_write_roots):
                        continue
                    for protected_path in protected_paths:
                        try:
                            resolved.relative_to(protected_path)
                            return _workspace_write_block_runtime_result(resolved, candidate)
                        except Exception:
                            pass
                    if is_write and not root_write_allowed:
                        return _workspace_write_block_outside_root_result(resolved, work_dir, candidate)
                    continue
                deliverables_decision = decide_deliverables(pathlib.Path(candidate))
                if deliverables_decision is not None:
                    if deliverables_decision:
                        continue
                    return deliverables_block
                if any(_registry().path_text_is_inside(candidate, root) for root in allowed_write_roots):
                    continue
                for protected_path in protected_paths:
                    if _registry().path_text_is_inside(candidate, protected_path):
                        return _workspace_write_block_runtime_result(candidate)
                if is_write and not root_write_allowed:
                    return _workspace_write_block_outside_root_result(candidate, work_dir)
                continue
            resolved = (candidate_cwd / pathlib.Path(candidate)).resolve(strict=False)
            # The lexical relative spelling is authoritative for detecting
            # a Deliverables-origin target; the helper then canonicalizes
            # it and rejects symlink escapes.
            deliverables_decision = decide_deliverables(
                candidate_cwd / pathlib.Path(candidate)
            )
            if deliverables_decision is not None:
                if deliverables_decision:
                    continue
                return deliverables_block
            if any(resolved.is_relative_to(root) for root in allowed_write_roots):
                continue
            for protected_path in protected_paths:
                try:
                    resolved.relative_to(protected_path)
                    return _workspace_write_block_runtime_result(resolved, candidate)
                except Exception:
                    pass
            if is_write and not root_write_allowed:
                return _workspace_write_block_outside_root_result(resolved, work_dir, candidate)
    return None


def _shell_git_and_runtime_block(
    self, raw_cmd: Any, args: Dict[str, Any], cmd_path_lower: str,
    workspace_mode: bool, acting_self_worktree: bool, binding: Any,
) -> ToolResult | None:
    """Direct-git-via-shell policy + the external-workspace runtime/secret read
    guard. External workspaces AND the default (non-workspace) lane get full
    task-local git through ONE target-aware resolver — only the Ouroboros
    runtime is protected (Q4=A unwind, 2026-08-08) — while raw non-git shell
    in external workspaces still cannot read the runtime/secrets;
    self_worktree keeps the strict read-only git policy."""
    from ouroboros.git_shell_policy import is_readonly_git_command

    if not _registry().shell_argv(raw_cmd):
        return None
    if workspace_mode and not acting_self_worktree:
        work_dir = _resolved_shell_cwd(self, args, binding)
        if isinstance(work_dir, ToolResult):
            return work_dir
        if git_block := _external_workspace_git_block(self, raw_cmd, work_dir):
            return git_block
        # Even READ-only, non-git shell (cat/head/grep/python) must not
        # reach the runtime or secrets — the raw-shell bypass of the
        # user_files path guard stays closed (top-level external tasks).
        # READ-ONLY GIT IS EXEMPT (owner Q4=A "read-only everywhere"; the
        # f14baf8f false-block class): `git -C <system repo>
        # status|log|diff|show|rev-parse` is the vcs_status-equivalent
        # lane, already readable via gated read_file. The secret surface
        # stays closed: the exemption is ALL-or-nothing per segment
        # (`git status && cat <data>/settings.json` is not exempt) and
        # write-aware — `is_readonly_git_command` refuses `--output=` and
        # `--no-index`, so neither a runtime write nor a settings dump
        # can ride "read-only git".
        if _registry().is_external_workspace(self._ctx) and not is_readonly_git_command(raw_cmd):
            if ext_block := _external_shell_runtime_or_secret_block(
                self, raw_cmd, cmd_path_lower, args, work_dir=work_dir,
                binding=binding,
            ):
                return ext_block
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
    # DEFAULT (non-workspace) lane. Q4=A (owner 2026-08-08): mutating git
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
    if "git" not in cmd_path_lower:
        return None
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
