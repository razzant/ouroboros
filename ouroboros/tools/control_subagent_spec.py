"""The published schedule_subagent parameter surface, and its validation.

One mapping is the SSOT: the public JSON schema the model sees is built from it,
and the handler's closed keyword set is derived from the same object, so what a
parent may pass and what the schema advertises cannot drift apart. Field
normalization and the refusals for a malformed request live here beside it.
"""

from __future__ import annotations

from typing import Any, Dict


VALID_SUBTASK_MEMORY_MODES = frozenset({"forked", "empty"})


def schedule_subagent_properties() -> Dict[str, Any]:
    """SSOT for the schedule_subagent parameter surface: ONE object, TWO derived consumers.

    The PUBLIC schema is the model contract (`ToolEntry("schedule_subagent", …)` in `get_tools`,
    with `additionalProperties: False`). Ordinary arguments are derived from this one mapping;
    the only non-public exception is the bounded D23 legacy selector set carried through the
    registry for deterministic migration. This avoids the former pair of hand-maintained public
    parameter lists drifting apart (BIBLE P7).

    Returns a FRESH mapping per call, exactly as the inline literal did, so a caller that mutates
    a returned schema cannot corrupt every later `get_tools()`."""
    from ouroboros.tool_access import SUBAGENT_CAPABILITIES
    from ouroboros.configured_subagents import SESSION_ACCESS_LOWERING
    from ouroboros.settings_scales import EFFORT_SCALE

    return {
        "subagent_id": {
            "type": "string",
            "description": (
                "Exact Available subagents actor id. The child snapshots that row; later Settings "
                "edits cannot retarget it."
            ),
        },
        "access": {"type": "string", "enum": ["inherit", *SESSION_ACCESS_LOWERING], "default": "inherit", "description":
            "inherit (default/omitted) preserves owner-selected Agent-session access; readonly or "
            "workspace_write may only lower it. API-model rows ignore it with disclosure. It grants "
            "no task authority or write_surface change; readonly tasks stay readonly and "
            "write_surface controls read/write authority."},
        "objective": {"type": "string", "description": "Focused OUTCOME and scope, not a step-by-step script. A harness-dispatched child forwards work to its delegated run; a scripted objective can read as orders to execute natively."},
        "expected_output": {"type": "string", "description": "Concrete handoff expected from the child."},
        "role": {"type": "string", "description": "Optional freeform lineage/UI role, e.g. architecture-reviewer; omission assigns no role."},
        "context": {"type": "string", "description": "Optional reference context, not instructions: write the orientation it lacks (purpose, prior decisions, material paths). The child starts with my identity, life's top-level account, its room page and my human's originating words, but not this conversation. For a harness-dispatched child this becomes its session's WORK ORDER; that session has none of my memory, so put recipes/details here, not in objective."},
        "input_sources": {
            "type": "string", "enum": ["shared", "declared"],
            "description": (
                "Omit or shared: the child starts with SYSTEM/BIBLE, book maps, my identity, life's top-level "
                "account (except Agent-session supervisors), the child's room page, full assignment, "
                "attachments and my human's originating words verbatim; knowledge and other material "
                "remain one read away. declared selects only the authored objective/context/constraints "
                "and full governance/task authority for an independent first position. It excludes "
                "automatic shared memory, dialogue, project knowledge, inherited parent context and "
                "attachments; put common evidence in context. Descendants cannot widen inherited "
                "declared. Selection lasts for the task; the assignment defines retention/collaboration, "
                "and tool results/messages can broaden input. It prescribes no exchange sequence or "
                "transport, provides no access isolation and makes no claim about provider context or priors."
            ),
        },
        "constraints": {"type": "string", "description": "Optional constraints/non-goals for the child."},
        "memory_mode": {
            "type": "string",
            "enum": sorted(VALID_SUBTASK_MEMORY_MODES),
            "description": (
                "Child execution-drive seed: forked (default) copies stable identity/WORLD/registry/knowledge "
                "files, or only shared patterns for a Project child; empty seeds that drive with nothing. "
                "Ordinary context still uses canonical governance (BIBLE/SYSTEM/reference books) and the "
                "canonical data root's shared memory: empty means a blank drive, not a blank context. "
                "shared is disabled for live local subagents. input_sources=declared "
                "selects automatic inputs independently."),
        },
        "workspace_root": {"type": "string", "description": "Starting folder (default: parent's). Read-only helpers start exactly here, including Git subdirectories; self_worktree copies this Git source's current files. external_workspace requires explicit write_root to match; genesis requires omission and creates an empty project. Must already be parent-readable; grants no read/write authority."},
        "write_surface": {
            "type": "string",
            # No empty-string member: Google Gemini's function-calling validator
            # rejects empty enum values (400 INVALID_ARGUMENT). Read-only is the
            # default by OMITTING this param; `read_only` is an explicit, provider-safe
            # (non-empty) alias for the SAME read-only path, so an audit/read-only child
            # can NAME its intent instead of reaching for an acting surface like
            # self_worktree (the trap behind the read-only-audit cancel-storm). It is NOT
            # an acting VALID_WRITE_SURFACES member — it normalizes to the omit path.
            "enum": ["read_only", "self_worktree", "external_workspace", "genesis"],
            "description": "read_only (default/omitted) starts read-only in workspace_root or the inherited folder. Acting surfaces: self_worktree copies that Git source's current files for parent patch integration; external_workspace: native children write shared files directly; genesis creates a standalone project. See tool description for integration. Acting surfaces require mutative subagents enabled (default ON in advanced/pro).",
        },
        "write_root": {"type": "string", "description": "external_workspace project directory, with or without Git; never runtime data. Delegate installed skills directly via delegate_start(root='skill_payload', bucket=..., skill_name=..., subagent_id=..., prompt=...). Omit for a new cooperative shared Git tree inherited by descendants; integrate_subagent_patch verifies combined files without reapplying. Ignored for self_worktree (source: workspace_root), read_only and genesis (own empty root)."},
        "directory_strategy": {
            "type": "string", "enum": ["direct", "copy"],
            "description": "Agent-session ordinary folders: direct (default/omitted) works there; copy uses a separate scope_paths copy and returns changes for application. Choose for the task/owner preference. Write-capable children only; read-only children omit this and scope_paths (direct without scope also means omitted). Native/API children use shared files, never copy.",
        },
        "scope_paths": {
            "type": "array", "items": {"type": "string"},
            "description": "Agent-session ordinary folders: relative inputs for copy (nonempty required), or direct capture footprint including future outputs. ['.'] selects the whole folder. Write-capable children only; read-only children omit this and directory_strategy because nothing is copied back/captured. Native children declare outputs on file/process tools.",
        },
        "protected_paths_grant": {"type": "boolean", "default": False, "description": "Allow the child to modify protected paths in its self_worktree. Honored only in pro runtime mode; you still re-check at integration."},
        "external_tool_grants": {"type": "array", "items": {"type": "string"}, "description": "Optional extension/MCP tool names to grant this mutative child. Denied by default."},
        "allowed_origins": {
            "type": "array", "items": {"type": "string"},
            "description": "Exact HTTP(S) origins for assigned private-service browser work, e.g. http://192.168.1.20:5173; no paths, credentials or wildcards. Ordinary root/Main may name task-authorized services; delegated/external Presence callers may only select inherited origins. Omit to inherit, [] to remove. Other resource/control-plane boundaries remain.",
        },
        "delegation_intent": {"type": "string", "description": "Whether/how this child should delegate further, including children/grandchildren. Propagates through its delegation budget and prompt; defaults to the parent's intent."},
        "may_mutate": {"type": "boolean", "default": False, "description": "Intent to allow this child acting descendants; ordinary mutative-subagent gating and depth/active caps still apply."},
        "may_fan_out": {"type": "boolean", "default": True, "description": "Whether this child may spawn multiple children, within the per-root active cap."},
        "max_children": {"type": "integer", "default": 0, "description": "Optional soft cap on this child's own direct children (0 = inherit / configured cap)."},
        "requested_depth": {
            "type": "integer", "default": 0,
            "description": "Intended absolute nesting depth from root=0 (children=1, grandchildren=2, great-grandchildren=3). Recorded as your request and reported as requested/permitted/achieved on the root result; never changes configured caps. 0/omitted means no request.",
        },
        "required_capabilities": {
            "type": "array",
            "items": {"type": "string", "enum": list(SUBAGENT_CAPABILITIES)},
            "description": "Required capabilities (e.g. shell/vcs/write/service), reconciled with the selected profile before spawning. Use this enum, not prose.",
        },
        "effort": {
            "type": "string", "enum": ["auto", *EFFORT_SCALE], "default": "auto",
            # `auto` is the provider-safe spelling of omission (a strict schema may fill it).
            "description": (
                "This child's reasoning effort, inside my human's effort range: auto/omitted = the "
                "recommended level; a request outside the range runs at its nearest bound. A row my "
                "human pinned or a level in its model name keeps that level outside Cyber Pro; the "
                "result says so. For a session row this is the delegated run's level."
            ),
        },
        "deadline_at": {
            "type": "string",
            "description": "ISO-8601 UTC instant after which this child's work is no longer useful. NARROWING ONLY: the earlier of this and the parent's deadline wins; omission inherits the parent's.",
        },
        "acceptance_claims": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Concrete, checkable 'done' claims for this child as plain strings. Become contract "
                "acceptance_claims, ids claim_1..N in list order; verify_and_record links receipts by "
                "criterion_id for per-claim support at absorption. Parent claims are NEVER inherited. "
                "Omit unless you can state real checks; omitted/empty/blank means no claims."
            ),
        },
    }


def schedule_subagent_param_names() -> frozenset:
    """The handler's closed keyword set, DERIVED from the public schema above.

    Anything the schema does not expose is refused with the strict v6 message instead of being
    silently accepted — and because the set is derived, "what the schema exposes" is the only
    definition of it there is."""
    return frozenset(schedule_subagent_properties())


_INTERNAL_SCHEDULE_OPTIONS: frozenset = frozenset()


def _validated_schedule_fields(params: Dict[str, Any], *, ctx: Any = None) -> tuple[Dict[str, Any], str]:
    """Normalize and validate the public schedule_subagent fields.

    Returns ``(fields, "")`` or ``({optional reason}, refusal)``. Extracted from ``_schedule_task`` so
    the handler stays inside the method-size gate — argument validation is a coherent
    phase with one job, not a slice taken to shed lines.
    """
    deadline_at = str(params.get("deadline_at") or "").strip()
    memory_mode = str(params.get("memory_mode") or "forked").strip().lower()
    if deadline_at:
        # `deadline_at` became MODEL-AUTHORED in v6.87.7; it used to be computed by
        # plan_review, where neither check could fail. Both failures below are SILENT
        # without them (BIBLE P1): an unparseable stamp rides into the child contract
        # verbatim and simply never fires, so the parent believes it bound a child that is
        # running deadline-blind; and a past stamp makes the child emit its canned
        # "produce your best answer NOW" on round one, having done no work at all.
        from ouroboros.deadline_utils import parse_deadline_ts, utc_now

        parsed = parse_deadline_ts(deadline_at)
        if parsed is None:
            return {}, (
                "⚠️ TOOL_ARG_ERROR (schedule_subagent): deadline_at must be an ISO-8601 UTC "
                f"instant such as 2026-08-02T18:30:00Z (got: {deadline_at!r})."
            )
        if parsed <= utc_now():
            return {}, (
                "⚠️ TOOL_ARG_ERROR (schedule_subagent): deadline_at is already in the past "
                f"({deadline_at}); a child bound to it would finalize before doing any work."
            )
    objective = str(params.get("objective") or "").strip()
    if not objective:
        return {}, "⚠️ TOOL_ARG_ERROR (schedule_subagent): objective is required."
    expected_output = str(params.get("expected_output") or "").strip()
    if not expected_output:
        return {}, "⚠️ TOOL_ARG_ERROR (schedule_subagent): expected_output is required."
    raw_claims = params.get("acceptance_claims")
    if raw_claims is not None and (
        not isinstance(raw_claims, list)
        or any(not isinstance(item, str) for item in raw_claims)
    ):
        return {}, (
            "⚠️ TOOL_ARG_ERROR (schedule_subagent): acceptance_claims must be an array "
            "of plain strings (one checkable claim per entry)."
        )
    # Vacuous claims normalize to ABSENT, never an error (the v6.65.1/.2 lesson:
    # min-constraints shape placeholder junk instead of preventing it).
    acceptance_claims = [
        item.strip() for item in (raw_claims or []) if isinstance(item, str) and item.strip()
    ]
    if memory_mode not in VALID_SUBTASK_MEMORY_MODES:
        allowed = ", ".join(sorted(VALID_SUBTASK_MEMORY_MODES))
        return {}, (
            f"⚠️ TOOL_ARG_ERROR (schedule_subagent): memory_mode must be one of: {allowed}. "
            "memory_mode=shared is disabled for live local subagents until a sanitized shared-context mode exists."
        )
    requested_effort, effort_error = requested_child_effort(params.get("effort"), "schedule_subagent")
    if effort_error:
        return {}, effort_error
    directory_options = {}
    if "directory_strategy" in params:
        strategy = params["directory_strategy"]
        if strategy not in ("direct", "copy"):
            return {}, "⚠️ TOOL_ARG_ERROR (schedule_subagent): directory_strategy must be direct or copy."
        directory_options["directory_strategy"] = strategy
    if "scope_paths" in params:
        from pathlib import PurePosixPath, PureWindowsPath

        paths = params["scope_paths"]
        if not isinstance(paths, list) or any(
            not isinstance(path, str) or not path.strip() or "\x00" in path
            or PurePosixPath(path).is_absolute() or PureWindowsPath(path).drive
            or PureWindowsPath(path).root or ".." in PurePosixPath(path).parts for path in paths
        ):
            return {}, "⚠️ TOOL_ARG_ERROR (schedule_subagent): scope_paths must be an array of nonempty relative paths."
        directory_options["scope_paths"] = list(paths)
    if directory_options.get("directory_strategy") == "copy" and not directory_options.get("scope_paths"):
        return {}, "⚠️ TOOL_ARG_ERROR (schedule_subagent): copy requires scope_paths; use ['.'] to select the whole folder."
    from ouroboros.delegate_directory import DIRECTORY_OPTIONS_NEED_WRITE, default_shaped_directory_options

    # Folder geometry is a WRITE-side request: a read-only child reads the selected
    # folder as it is, and the host's pre-start refuses geometry it can never serve —
    # after a worker, a queue row and the child's first paid round. Refusing it HERE
    # costs the parent one argument fix instead. `read_only` is the provider-safe
    # alias for omitting the surface, so both spellings take the read-only path the
    # scheduler's own constraint selector takes.
    if str(params.get("write_surface") or "").strip().lower() in {"", "read_only"} and (
        not default_shaped_directory_options(
            directory_options.get("directory_strategy"), directory_options.get("scope_paths"))
    ):
        return {}, "⚠️ TOOL_ARG_ERROR (schedule_subagent): " + DIRECTORY_OPTIONS_NEED_WRITE
    from ouroboros.contracts.task_contract import normalize_allowed_origins, normalize_browser_origin, normalize_resource_policy
    from ouroboros.presence_authority import presence_ceiling_from_context
    from ouroboros.tools.core import is_restricted_subagent_profile

    metadata = getattr(ctx, "task_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    # EMPTINESS decides, not type. `ToolContext.task_contract` defaults to `{}`, so testing
    # only `isinstance(..., dict)` let that empty default win over a contract that really is
    # in `task_metadata` — and the parent's `deadline_at` lives in the contract, so the miss
    # silently un-narrowed every child deadline. Same precedence the registry already uses.
    parent = metadata.get("task_contract") if isinstance(metadata.get("task_contract"), dict) else {}
    if not parent and isinstance(getattr(ctx, "task_contract", None), dict):
        parent = ctx.task_contract
    from ouroboros.contracts.task_contract import normalize_input_sources

    input_source_fields = {}
    try:
        if "input_sources" in parent:
            input_source_fields["input_sources"] = normalize_input_sources(parent["input_sources"])
        if "input_sources" in params:
            selected = normalize_input_sources(params["input_sources"])
            if input_source_fields.get("input_sources") == "declared" and selected != "declared":
                return {"reason": "INPUT_SOURCE_SELECTION_WIDENING"}, (
                    "⚠️ TOOL_ARG_ERROR (schedule_subagent): input_sources=shared cannot widen an inherited declared selection.")
            input_source_fields["input_sources"] = selected
    except ValueError as exc:
        return {"reason": "INPUT_SOURCE_SELECTION_INVALID"}, f"⚠️ TOOL_ARG_ERROR (schedule_subagent): {exc}."
    resource_policy = normalize_resource_policy(parent.get("resource_policy") or metadata.get("resource_policy"))
    if "allowed_origins" in params:
        requested = params["allowed_origins"]
        if not isinstance(requested, list) or any(not normalize_browser_origin(item, origin_only=True) for item in requested):
            return {}, "⚠️ TOOL_ARG_ERROR (schedule_subagent): allowed_origins must contain exact HTTP(S) origins without paths, credentials or wildcards."
        origins = normalize_allowed_origins(requested)
        # The host caller class supplies authority; neither a URL in page prose
        # nor an external Presence event may manufacture a new owner grant.
        subset_only = (ctx is None or is_restricted_subagent_profile(ctx)
                       or presence_ceiling_from_context(ctx) is not None)
        if subset_only and not set(origins).issubset(resource_policy.get("allowed_origins", [])):
            return {}, "⚠️ BROWSER_ORIGIN_NOT_GRANTED: this caller may only narrow its inherited allowed_origins."
        resource_policy["allowed_origins"] = origins
    return {
        "deadline_at": deadline_at, "objective": objective, "expected_output": expected_output,
        "role": str(params.get("role") or "").strip(),
        "context": str(params.get("context") or "").strip(),
        "constraints": str(params.get("constraints") or "").strip(),
        "memory_mode": memory_mode, "may_mutate": params.get("may_mutate", False),
        "requested_effort": requested_effort,
        "acceptance_claims": acceptance_claims,
        "resource_policy": resource_policy,
        "parent_contract": parent,
        **input_source_fields,
        **directory_options,
    }, ""


def requested_child_effort(value: Any, tool_name: str) -> tuple[str, str]:
    """``(tier, "")`` for a child's ``effort`` argument — ``auto``, blank and omission all mean
    no request — or ``("", refusal)`` for an unknown tier (``settings_scales.choose_effort``
    decides the level at dispatch; the schedule result and the child record carry the request)."""
    from ouroboros.settings_scales import EFFORT_SCALE

    text = str(value or "").strip().lower()
    if text in ("", "auto"):
        return "", ""
    if text not in EFFORT_SCALE:
        return "", (f"⚠️ TOOL_ARG_ERROR ({tool_name}): effort must be auto or one of: "
                    f"{', '.join(EFFORT_SCALE)} (got {value!r}).")
    return text, ""
