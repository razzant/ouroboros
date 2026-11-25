"""The published schedule_subagent parameter surface, and its validation.

One mapping is the SSOT: the public JSON schema the model sees is built from it,
and the handler's closed keyword set is derived from the same object, so what a
parent may pass and what the schema advertises cannot drift apart. Field
normalization and the refusals for a malformed request live here beside it.
"""

from __future__ import annotations

from typing import Any, Dict


VALID_SUBTASK_MEMORY_MODES = frozenset({"forked", "empty"})
# The fields a parent may widen a child's memory view by: the ones its description offers, no other ViewSpec field.
MEMORY_VIEW_FIELDS = ("story", "room_page", "room_lanes", "origin_words", "other_conversations", "marks", "knowledge",
                      "owner_words")


def schedule_subagent_properties() -> Dict[str, Any]:
    """SSOT for the schedule_subagent parameter surface: ONE object, TWO derived consumers.

    The PUBLIC schema is the model contract (`ToolEntry("schedule_subagent", …)` in `get_tools`,
    with `additionalProperties: False`). Ordinary arguments are derived from this one mapping;
    the only non-public exceptions are the bounded D23 legacy selector set carried through the
    registry for deterministic migration and ``title``, refused there with the call that works. This avoids the former pair of hand-maintained public
    parameter lists drifting apart (BIBLE P7).

    Returns a FRESH mapping per call, exactly as the inline literal did, so a caller that mutates
    a returned schema cannot corrupt every later `get_tools()`."""
    from ouroboros.tool_access import SUBAGENT_CAPABILITIES
    from ouroboros.configured_subagents import SESSION_ACCESS_LOWERING
    from ouroboros.delegate_documents import attachment_parameters
    from ouroboros.delegate_readonly_inputs import input_files_parameter
    from ouroboros.settings_scales import EFFORT_SCALE

    return {
        "mode": {"type": "string", "enum": ["ask", "plan", "agent"], "description":
            "Agent-session execution mode. Omit for access-derived behavior; ask and plan run read-only. Plan questions return as artifacts for the starter."},
        "subagent_id": {
            "type": "string",
            "description": (
                "Exact Available subagents actor id. The child snapshots that row; later Settings "
                "edits cannot retarget it."
            ),
        },
        "supervisor": {
            "type": "string",
            "description": (
                "Optional enabled API-model row id from Available subagents for an Agent-session "
                "child's supervisor, including subscription-backed models. Captures its model, "
                "effort, account and processing preference. Omit to inherit my actual current "
                "send binding. Agent-session rows cannot supervise; API children have no separate supervisor."
            ),
        },
        "access": {"type": "string", "enum": ["inherit", *SESSION_ACCESS_LOWERING], "default": "inherit", "description":
            "inherit (default/omitted) preserves owner-selected Agent-session access; readonly or "
            "workspace_write may only lower it. API-model rows ignore it with disclosure. It grants "
            "no task or product-write authority: a non-acting session's evidence-only private checking copy owes no apply, "
            "so do not call integrate_delegated_patch to apply or reject its changes. "
            "Select readonly for read-only session execution; write_surface controls native child/nanny authority."},
        "objective": {"type": "string", "description": "Focused OUTCOME and scope, not a step-by-step script. A harness-dispatched child forwards work to its delegated run; a scripted objective can read as orders to execute natively."},
        "expected_output": {"type": "string", "description": "Concrete handoff expected from the child."},
        "role": {"type": "string", "description": "Optional freeform lineage/UI role, e.g. architecture-reviewer; omission assigns no role."},
        "context": {"type": "string", "description": "Optional reference context, not instructions: write the orientation it lacks (purpose, prior decisions, material paths). The child does not see this conversation; input_sources says what it starts with. For a harness-dispatched child this becomes its session's WORK ORDER; that session starts with my human's words, the marks, the knowledge outline and, unless declared, the heads of the notes that claim its room and of the notes their heads link, but none of my other memory, so put recipes/details here, not in objective."},
        "input_sources": {
            "type": "string", "enum": ["shared", "declared"],
            "description": (
                "Every child starts with my human's words for the work verbatim and dated, the marks in force "
                "and the knowledge outline; other knowledge stays one read away. Omit or shared: it also starts "
                "with the heads of the notes that claim its room and of the notes their heads link, SYSTEM/BIBLE, "
                "book maps, my identity, life's top-level account (except Agent-session "
                "supervisors), the child's room page, full assignment and attachments. declared selects only "
                "the authored objective/context/constraints and full governance/task authority for an independent "
                "first position; it excludes the rest of shared memory (identity, life account, room page), "
                "dialogue, inherited parent context and attachments; put common evidence in context. Descendants cannot widen inherited "
                "declared. Selection lasts for the task; the assignment defines retention/collaboration, "
                "and tool results/messages can broaden input. It prescribes no exchange sequence or "
                "transport, provides no access isolation and makes no claim about provider context or priors."
            ),
        },
        "constraints": {"type": "string", "description": "Optional constraints/non-goals for the child."},
        "memory_view": {
            "type": "object",
            "description": (
                "Optional widening of the child's memory view by field, e.g. {\"knowledge\": true} starts it with the whole "
                "knowledge base instead of the outline. Fields: story, room_page, room_lanes, origin_words, "
                "other_conversations, marks (all|room_and_global), knowledge, owner_words. It only widens but for the story, "
                "which the assignment may leave out: my human's words and the marks in force can be turned on, never off "
                "(owner_words false and marks none are refused), and any other field is refused here. Omitted: the role's "
                "default view. A value the view does not know is refused "
                "at the child's start with the role default and one event."),
        },
        "memory_mode": {
            "type": "string",
            "enum": sorted(VALID_SUBTASK_MEMORY_MODES),
            "description": (
                "Child execution-drive seed: forked (default) copies stable identity/WORLD/registry/knowledge "
                "files, and no knowledge file for a Project child; empty seeds that drive with nothing. "
                "Ordinary context still uses canonical governance (BIBLE/SYSTEM/reference books) and the "
                "canonical data root's shared memory: empty means a blank drive, not a blank context. "
                "shared is disabled for live local subagents. input_sources=declared "
                "selects automatic inputs independently."),
        },
        "workspace_root": {"type": "string", "description": "Starting folder (default: parent's). Native readers start here, including Git subdirectories. external_workspace requires explicit write_root to match; genesis requires omission. Must already be parent-readable; grants no read/write authority."},
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
            "description": "read_only (default/omitted) keeps the native child/nanny read-only. Acting surfaces self_worktree, external_workspace (native children write shared files directly) and genesis are defined with their integration in the tool description; they require mutative subagents enabled (default ON in advanced/pro).",
        },
        "write_root": {"type": "string", "description": "external_workspace project directory, with or without Git; never runtime data, including installed skills. Omit for a new cooperative shared Git tree. Ignored for self_worktree (source: workspace_root), read_only and genesis (own empty root)."},
        "directory_strategy": {
            "type": "string", "enum": ["direct", "copy"],
            "description": "Agent-session ordinary folders: direct (default/omitted) works there; copy uses a separate scope_paths copy and returns changes for application. Choose for the task/owner preference. Acting children only; non-acting children omit this and scope_paths (direct without scope also means omitted). Native/API children use shared files, never copy.",
        },
        "workspace": {
            "type": "string", "enum": ["copy", "in_place"],
            "description": "Agent-session Git writers (an acting write_surface): copy (default) gives the harness a private copy whose diff the parent applies or rejects; in_place lets it work directly in the child's own write root, for example a self_worktree copy that is already private: its edits land at once, nothing is captured for apply or reject, undo is through git, and two in-place writers in one folder mix their edits.",
        },
        "scope_paths": {
            "type": "array", "items": {"type": "string"},
            "description": "Agent-session ordinary folders: relative inputs for copy (nonempty required), or direct capture footprint including future outputs. ['.'] selects the whole folder. Acting children only, like directory_strategy. Native children declare outputs on file/process tools.",
        },
        **attachment_parameters(
            "Your retained inputs, for an agent_session child only (API children refuse both): the first start of "
            "the child's work order attaches them; its later starts, continuations and replacements select their own. "
            "With input_sources=declared only these inputs are copied to the child."),
        **input_files_parameter(
            "Omit write_surface or use read_only. Checked for caller and child before queueing; [] reports an empty "
            "receipt. API children read each copy with read_file. The first session start verifies retained bytes "
            "unless it explicitly replaces the selection."),
        "protected_paths_grant": {"type": "boolean", "default": False, "description": "Allow the child to modify protected paths in its self_worktree, directly or by applying its own children's patches. Pro runtime mode needs it, Cyber Pro does not, lower modes refuse either way; you still re-check at integration."},
        "external_tool_grants": {"type": "array", "items": {"type": "string"}, "description": "Optional extension/MCP tool names to grant this mutative child. Denied by default."},
        "allowed_origins": {
            "type": "array", "items": {"type": "string"},
            "description": "Exact HTTP(S) origins for assigned private-service browser work, e.g. http://192.168.1.20:5173; no paths, credentials or wildcards. Ordinary root/Main may name task-authorized services; delegated/external Presence callers may only select inherited origins. Omit to inherit, [] to remove. Other resource/control-plane boundaries remain.",
        },
        "delegation_intent": {"type": "string", "description": "Whether/how this child should delegate further, including children/grandchildren. Propagates through its delegation budget and prompt; defaults to the parent's intent."},
        "may_mutate": {"type": "boolean", "default": False, "description": "Intent to allow this child acting descendants; ordinary mutative-subagent gating and depth/active caps still apply."},
        "may_fan_out": {"type": "boolean", "default": True, "description": "Whether this child may have more than one direct child: false allows one admitted in its whole lifetime, a finished one included. The per-root active cap still applies."},
        "max_children": {"type": "integer", "default": 0, "description": "Optional cap on how many direct children this child may have admitted in its whole lifetime, finished ones included; a start past it is refused (0 = inherit your cap; never above it)."},
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
                "This child's reasoning effort: auto/omitted = this room's primary clamped into its "
                "helper bounds. A level in the model name always holds; otherwise in Cyber Pro my request is applied "
                "as asked; otherwise a row my human pinned keeps its level; otherwise a request beyond the bounds "
                "runs at the nearest bound. The result says so. For a session row this is the delegated run's level."
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
SCHEDULE_REFUSAL = "⚠️ TOOL_ARG_ERROR (schedule_subagent): "
_STRICT_SCHEMA = ("Use the strict schema: subagent_id, objective, expected_output, optional role/context/constraints/"
                  "memory_mode/workspace_root/effort and (for mutative children) write_surface/write_root/"
                  "protected_paths_grant/external_tool_grants.")


def one_refusal(problems: list) -> str:
    """Every problem of one request in one refusal, each worded as its own check words it (the W2 shape of
    ``arg_feedback.argument_refusal``; DEVELOPMENT 02: field, value and repair in one reply), so fixing one never
    meets the next on the following call. One problem reads exactly as it always did."""
    return SCHEDULE_REFUSAL + "; ".join(str(problem).rstrip(". ") for problem in problems) + "."


def _shape_problems(params: Dict[str, Any], unsupported: Any) -> tuple[list[str], dict]:
    """Independent problems and normalized selections before any route is selected.

    The shape pass checks each field on its own. Checks that depend on other fields, the parent's grants
    or the route follow once these pass.
    """
    problems = [f"unsupported argument(s): {', '.join(unsupported)}. {_STRICT_SCHEMA}"] if unsupported else []
    if params.get("mode") not in (None, "ask", "plan", "agent"):
        problems.append("mode must be ask, plan or agent.")
    deadline_at = str(params.get("deadline_at") or "").strip()
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
            problems.append("deadline_at must be an ISO-8601 UTC instant such as 2026-08-02T18:30:00Z "
                            f"(got: {deadline_at!r}).")
        elif parsed <= utc_now():
            problems.append(f"deadline_at is already in the past ({deadline_at}); a child bound to it would "
                            "finalize before doing any work.")
    if "title" in params:  # start_task's card name; this call names no card
        problems.append(f"title={params['title']!r} is no schedule_subagent field: objective names the work"
                        + ("" if str(params.get("objective") or "").strip() else " and is required")
                        + ", and the optional role becomes part of the child's assignment; send the call without title.")
    for field in ("objective", "expected_output"):
        if not str(params.get(field) or "").strip() and (field, "title" in params) != ("objective", True):
            problems.append(f"{field} is required.")
    raw_claims = params.get("acceptance_claims")
    if raw_claims is not None and (not isinstance(raw_claims, list) or any(not isinstance(item, str) for item in raw_claims)):
        problems.append("acceptance_claims must be an array of plain strings (one checkable claim per entry).")
    if str(params.get("memory_mode") or "forked").strip().lower() not in VALID_SUBTASK_MEMORY_MODES:
        problems.append(f"memory_mode must be one of: {', '.join(sorted(VALID_SUBTASK_MEMORY_MODES))}. memory_mode=shared "
                        "is disabled for live local subagents until a sanitized shared-context mode exists.")
    effort_error = requested_child_effort(params.get("effort"), "schedule_subagent")[1]
    if effort_error:
        problems.append(effort_error.removeprefix(SCHEDULE_REFUSAL))
    view = params.get("memory_view")
    if isinstance(view, dict) and (view.get("owner_words") is False or view.get("marks") == "none"):
        problems.append("memory_view only widens a child's view; it keeps my human's words and the marks in force.")
    if isinstance(view, dict) and not set(view) <= set(MEMORY_VIEW_FIELDS):
        problems.append(f"memory_view fields are {', '.join(MEMORY_VIEW_FIELDS)} "
                        f"(got: {', '.join(sorted(map(str, set(view) - set(MEMORY_VIEW_FIELDS))))}).")
    if "directory_strategy" in params and params["directory_strategy"] not in ("direct", "copy"):
        problems.append("directory_strategy must be direct or copy.")
    if "scope_paths" in params and not _relative_paths(params["scope_paths"]):
        problems.append("scope_paths must be an array of nonempty relative paths.")
    if params.get("workspace") not in (None, "copy", "in_place"):
        problems.append("workspace must be copy or in_place.")

    selections: dict = {}
    if params.get("input_files") is not None:
        from ouroboros.delegate_readonly_inputs import InputSelectionError, selection_entries

        try:
            selections["input_files"] = selection_entries(params["input_files"])
        except InputSelectionError as exc:
            problems.append(exc.detail)
    if "allowed_origins" in params:
        from ouroboros.contracts.task_contract import normalize_browser_origin

        requested = params["allowed_origins"]
        origins = [normalize_browser_origin(item, origin_only=True) for item in requested] if isinstance(requested, list) else None
        if origins is None or not all(origins):
            problems.append("allowed_origins must contain exact HTTP(S) origins without paths, credentials or wildcards.")
        else:
            selections["allowed_origins"] = list(dict.fromkeys(origins))
    return problems, selections


def _relative_paths(paths: Any) -> bool:
    """``scope_paths`` as the folder geometry takes them: a list of nonempty relative paths inside the folder."""
    from pathlib import PurePosixPath, PureWindowsPath

    return isinstance(paths, list) and not any(
        not isinstance(path, str) or not path.strip() or "\x00" in path
        or PurePosixPath(path).is_absolute() or PureWindowsPath(path).drive
        or PureWindowsPath(path).root or ".." in PurePosixPath(path).parts for path in paths)


def _validated_schedule_fields(params: Dict[str, Any], *, ctx: Any = None,
                               unsupported: Any = ()) -> tuple[Dict[str, Any], str]:
    """Normalize and validate the public schedule_subagent fields.

    Returns ``(fields, "")`` or ``({optional reason}, refusal)``. Extracted from ``_schedule_task`` so
    the handler stays inside the method-size gate — argument validation is a coherent
    phase with one job, not a slice taken to shed lines. The independent shape problems,
    with the ``unsupported`` names the handler found, are refused together (``_shape_problems``).
    """
    problems, selections = _shape_problems(params, unsupported)
    if problems:
        return {}, one_refusal(problems)
    deadline_at = str(params.get("deadline_at") or "").strip()
    memory_mode = str(params.get("memory_mode") or "forked").strip().lower()
    objective = str(params.get("objective") or "").strip()
    expected_output = str(params.get("expected_output") or "").strip()
    # Vacuous claims normalize to ABSENT, never an error (the v6.65.1/.2 lesson:
    # min-constraints shape placeholder junk instead of preventing it).
    acceptance_claims = [
        item.strip() for item in (params.get("acceptance_claims") or []) if isinstance(item, str) and item.strip()
    ]
    requested_effort = requested_child_effort(params.get("effort"), "schedule_subagent")[0]
    directory_options = {}  # the folder fields' own values passed the shape pass
    if "directory_strategy" in params:
        directory_options["directory_strategy"] = params["directory_strategy"]
    if "scope_paths" in params:
        directory_options["scope_paths"] = list(params["scope_paths"])
    if directory_options.get("directory_strategy") == "copy" and not directory_options.get("scope_paths"):
        return {}, "⚠️ TOOL_ARG_ERROR (schedule_subagent): copy requires scope_paths; use ['.'] to select the whole folder."
    from ouroboros.delegate_directory import DIRECTORY_OPTIONS_NEED_WRITE, default_shaped_directory_options

    if "attachment_paths" in params or "attachment_import" in params:  # membership is checked while copying
        from ouroboros.delegate_documents import DocumentSelectionError, selection_options

        try:
            new_import = selection_options(params.get("attachment_paths"), params.get("attachment_import"))
        except DocumentSelectionError as exc:
            return {}, f"⚠️ TOOL_ARG_ERROR (schedule_subagent): {exc.detail}"
        if "attachment_paths" in params:
            directory_options["attachment_paths"] = list(params["attachment_paths"])
        directory_options["attachment_import"] = "new" if new_import else "reuse"
        if new_import:  # the first start's one new intent, saved with the task: a restart never mints twice
            import uuid

            directory_options["document_import_intent"] = uuid.uuid4().hex
    if "input_files" in selections:  # permission and copying happen with the queued child's identity
        directory_options["input_files"] = selections["input_files"]
        if directory_options["input_files"] and str(params.get("write_surface") or "").strip().lower() not in {"", "read_only"}:
            return {}, ("⚠️ TOOL_ARG_ERROR (schedule_subagent): input_files_readonly_only: input_files prepares inputs "
                        "for a read-only child; omit it or write_surface.")
    if params.get("workspace") == "in_place":  # the documented default geometry is no folder request, as at delegate_start
        surface = str(params.get("write_surface") or "").strip().lower()
        in_the_way = [] if surface in {"self_worktree", "external_workspace", "genesis"} else [
            f"write_surface={params['write_surface']!r}" if params.get("write_surface") else "no write_surface"]
        if not default_shaped_directory_options(directory_options.get("directory_strategy"),
                                                directory_options.get("scope_paths")):
            in_the_way += ["directory_strategy='copy'"] * (directory_options.get("directory_strategy") == "copy") + [
                "scope_paths"] * bool(directory_options.get("scope_paths"))
        if in_the_way:
            return {}, ("⚠️ TOOL_ARG_ERROR (schedule_subagent): workspace='in_place' needs an acting Git write_surface "
                        f"and no directory options; in the way: {', '.join(in_the_way)}. Omitting workspace removes "
                        "this conflict.")
        directory_options["workspace"] = "in_place"

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
    from ouroboros.contracts.task_contract import normalize_resource_policy
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
    if "allowed_origins" in selections:
        origins = selections["allowed_origins"]
        # The host caller class supplies authority; neither a URL in page prose
        # nor an external Presence event may manufacture a new owner grant.
        subset_only = (ctx is None or is_restricted_subagent_profile(ctx)
                       or presence_ceiling_from_context(ctx) is not None)
        if subset_only and not set(origins).issubset(resource_policy.get("allowed_origins", [])):
            return {}, "⚠️ BROWSER_ORIGIN_NOT_GRANTED: this caller may only narrow its inherited allowed_origins."
        resource_policy["allowed_origins"] = origins
    view = params.get("memory_view")
    return {
        "deadline_at": deadline_at, "objective": objective, "expected_output": expected_output,
        "role": str(params.get("role") or "").strip(),
        "context": str(params.get("context") or "").strip(),
        "constraints": str(params.get("constraints") or "").strip(),
        "memory_mode": memory_mode, "may_mutate": params.get("may_mutate", False),
        "requested_effort": requested_effort,
        **({"delegated_mode": params["mode"]} if params.get("mode") is not None else {}),
        **({"memory_view": view} if view is not None else {}),  # its values are checked at the child's start
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
