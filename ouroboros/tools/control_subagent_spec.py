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

    return {
        "subagent_id": {
            "type": "string",
            "description": (
                "Exact actor id from the Available subagents catalog. The selected row is "
                "snapshotted into the child, so later Settings edits do not retarget it."
            ),
        },
        "access": {"type": "string", "enum": ["inherit", *SESSION_ACCESS_LOWERING], "default": "inherit", "description":
            "Default inherit (or omit) preserves the owner's Agent-session access. "
            "readonly or workspace_write may only lower it. For API-model rows this field "
            "is ignored with a disclosure; write_surface controls read/write authority. "
            "This does not change write_surface or grant task authority; readonly tasks stay readonly."},
        "objective": {"type": "string", "description": "Focused child objective. Be specific about scope. State the OUTCOME you need, not a step-by-step script: on a delegated (harness) dispatch the child forwards the work to its own delegated run, and a script-shaped objective reads as orders to execute natively."},
        "expected_output": {"type": "string", "description": "Concrete handoff expected from the child."},
        "role": {"type": "string", "description": "Optional freeform role label for lineage/UI, e.g. architecture-reviewer."},
        "context": {"type": "string", "description": "Optional parent reference material. It is injected as context, not instructions; for a harness-dispatched child it becomes the WORK ORDER for its delegated run's prompt, so put the recipe/details here rather than in the objective."},
        "input_sources": {
            "type": "string", "enum": ["shared", "declared"],
            "description": (
                "Omit or shared for ordinary shared context. declared selects only the authored "
                "objective/context/constraints and full governance/task authority as automatic inputs; "
                "it excludes automatic shared memory, dialogue, project knowledge, inherited parent "
                "context and attachments. Put common evidence explicitly in context. Applies to "
                "API and configured-session children; inherited declared cannot be widened by descendants. Selection lasts "
                "for the task; the assignment defines first-position retention and collaboration. "
                "Tool results and messages can broaden the input. This selector prescribes no "
                "exchange sequence or transport and is not access isolation or a claim about "
                "provider context or learned priors."
            ),
        },
        "constraints": {"type": "string", "description": "Optional constraints/non-goals for the child."},
        "memory_mode": {
            "type": "string",
            "enum": sorted(VALID_SUBTASK_MEMORY_MODES),
            "description": (
                "Seed of the child's OWN execution drive. Default forked copies stable memory files there "
                "(identity, WORLD, registry and knowledge; a Project child gets only the shared patterns); "
                "empty seeds that drive with nothing. In ordinary mode the child's context is still built from the "
                "canonical governance (BIBLE, SYSTEM, reference books) and the canonical data root's shared "
                "memory, so empty is a blank drive, not a blank context. shared is disabled for live local subagents. "
                "input_sources=declared independently selects automatic inputs."),
        },
        "workspace_root": {"type": "string", "description": "Optional folder, inherited from the parent when omitted. Read-only helpers start exactly here, including Git subdirectories. For self_worktree this selects the Git source copied with its current files into the isolated working tree. For external_workspace an explicit write_root must name the same folder; omit workspace_root for genesis, which provisions an empty project. The folder must already be readable by the parent and grants no read or write authority."},
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
            "description": "read_only (or omit) = read-only child starting in workspace_root or the inherited folder. A MUTATIVE child uses self_worktree (isolated current-tree Git copy of workspace_root or the inherited source, returning a patch for parent integration), external_workspace (native children write shared files directly), or genesis (standalone project). See tool description for integration. Acting surfaces require mutative subagents enabled (default ON in advanced/pro).",
        },
        "write_root": {"type": "string", "description": "For write_surface=external_workspace: the external project directory, with or without Git, never runtime data. An installed skill payload has its own resource address: delegate it directly with delegate_start(subagent_id=..., prompt=..., root='skill_payload', bucket=..., skill_name=...). OMIT write_root to build COOPERATIVELY from scratch — the host mints ONE shared git tree the whole subagent tree writes into together (deeper descendants inherit it), and you verify the combined files with integrate_subagent_patch without reapplying them. Ignored for self_worktree (workspace_root selects its source), read_only and genesis (which provisions its own empty root)."},
        "directory_strategy": {
            "type": "string", "enum": ["direct", "copy"],
            "description": "For an agent_session in an ordinary folder: direct works in the selected folder; copy works in a separate copy of scope_paths and returns changes for application. Choose according to the task and any owner preference. Omit for direct ordinary-folder work. Write-capable children only: a read-only child omits both this and scope_paths (direct with no scope is the same as omitting). Native/API children use shared files directly and do not support copy.",
        },
        "scope_paths": {
            "type": "array", "items": {"type": "string"},
            "description": "For an agent_session in an ordinary folder: relative copied inputs for copy, or capture footprint for direct (including future outputs). ['.'] explicitly selects the whole folder. Copy requires nonempty scope. Write-capable children only: a read-only child omits both this and directory_strategy (there is nothing for it to copy back or capture). Native children declare process outputs on their file/process tools instead.",
        },
        "protected_paths_grant": {"type": "boolean", "default": False, "description": "Allow the child to modify protected paths in its self_worktree. Honored only in pro runtime mode; you still re-check at integration."},
        "external_tool_grants": {"type": "array", "items": {"type": "string"}, "description": "Optional extension/MCP tool names to grant this mutative child. Denied by default."},
        "allowed_origins": {
            "type": "array", "items": {"type": "string"},
            "description": "Exact HTTP(S) origins for the child's assigned private-service browser work, e.g. http://192.168.1.20:5173 (no path, credentials or wildcard). The ordinary root/Main may name task-authorized services; delegated and external Presence callers may only select from inherited origins. Omit to inherit; [] removes them. Other resource and control-plane boundaries still apply.",
        },
        "delegation_intent": {"type": "string", "description": "Optional: tell THIS child whether/how to delegate further (e.g. 'build the whole game; spawn your own children per subsystem and let them spawn too'). Propagated structurally into the child's delegation budget and surfaced in its prompt, so a 'use maximum subagents / grandchildren' intent is not lost. Defaults to inheriting the parent's intent."},
        "may_mutate": {"type": "boolean", "default": False, "description": "Optional: grant this child the intent to spawn MUTATIVE (acting) descendants of its own. Still bounded by the usual mutative-subagent gating and depth/active caps."},
        "may_fan_out": {"type": "boolean", "default": True, "description": "Optional: whether this child may spawn MULTIPLE children (a wave). Bounded by the per-root active cap."},
        "max_children": {"type": "integer", "default": 0, "description": "Optional soft cap on this child's own direct children (0 = inherit / configured cap)."},
        "requested_depth": {
            "type": "integer", "default": 0,
            "description": "Optional: how deep, counted ABSOLUTELY FROM THE ROOT, you intend this branch to nest (root=0, direct children=1; asking for children, grandchildren and great-grandchildren is 3). Recorded as your attested request and reported back as requested/permitted/achieved on the root result; it never widens or narrows the configured caps. 0 or omitted = no request.",
        },
        "required_capabilities": {
            "type": "array",
            "items": {"type": "string", "enum": list(SUBAGENT_CAPABILITIES)},
            "description": "Closed-enum capabilities this child must have (e.g. shell/vcs/write/service). The scheduler reconciles this with the selected profile before spawning; do not encode these needs in prose.",
        },
        # Per-call effort is retired. The selected Available-subagent row owns its
        # effort; a second request knob could contradict that immutable row or the
        # compound session route it pins.
        "deadline_at": {
            "type": "string",
            "description": "Optional ISO-8601 UTC instant after which this child's work is worthless to you (e.g. a scout whose handoff you can only consume inside a narrow window). NARROWING ONLY: the earlier of this and the parent's deadline wins, so it can tighten your own deadline but never extend it. Omit it to simply inherit the parent's.",
        },
        "acceptance_claims": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Optional concrete, checkable claims of what 'done' means for THIS child "
                "(plain strings, e.g. 'the collision module rejects overlapping hulls'). "
                "They become the child contract's acceptance_claims (ids claim_1..N in "
                "list order) — the child links verify_and_record receipts to them via "
                "criterion_id, and you see per-claim support at absorption. The child "
                "NEVER inherits your own claims: omitted means the child has none. Omit "
                "the field unless you can state real checks; empty/blank values are "
                "treated as absent."
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
        "role": str(params.get("role") or "researcher").strip() or "researcher",
        "context": str(params.get("context") or "").strip(),
        "constraints": str(params.get("constraints") or "").strip(),
        "memory_mode": memory_mode, "may_mutate": params.get("may_mutate", False),
        "acceptance_claims": acceptance_claims,
        "resource_policy": resource_policy,
        "parent_contract": parent,
        **input_source_fields,
        **directory_options,
    }, ""


RETIRED_SCHEDULE_PARAMS: Dict[str, str] = {"effort": "reasoning_effort"}
