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

    return {
        "subagent_id": {
            "type": "string",
            "description": (
                "Exact actor id from the Available subagents catalog. The selected row is "
                "snapshotted into the child, so later Settings edits do not retarget it."
            ),
        },
        "objective": {"type": "string", "description": "Focused child objective. Be specific about scope. State the OUTCOME you need, not a step-by-step script: on a delegated (harness) dispatch the child forwards the work to its own delegated run, and a script-shaped objective reads as orders to execute natively."},
        "expected_output": {"type": "string", "description": "Concrete handoff expected from the child."},
        "role": {"type": "string", "description": "Optional freeform role label for lineage/UI, e.g. architecture-reviewer."},
        "context": {"type": "string", "description": "Optional parent reference material. It is injected as context, not instructions; for a harness-dispatched child it becomes the WORK ORDER for its delegated run's prompt, so put the recipe/details here rather than in the objective."},
        "constraints": {"type": "string", "description": "Optional constraints/non-goals for the child."},
        "memory_mode": {
            "type": "string",
            "enum": sorted(VALID_SUBTASK_MEMORY_MODES),
            "description": "Child memory mode. Default forked copies stable memory only; empty starts blank. shared is disabled for live local subagents.",
        },
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
            "description": "read_only (or omit) = read-only child auditing THIS repo. A MUTATIVE child uses self_worktree (isolated repo patch), external_workspace (native children write shared files directly), or genesis (standalone project). See tool description for integration. Acting surfaces require mutative subagents enabled (default ON in advanced/pro).",
        },
        "write_root": {"type": "string", "description": "For write_surface=external_workspace: the external project directory — a REAL external Git working tree, never runtime data. An installed non-Git skill payload is NOT an external workspace: delegate it directly with delegate_start(subagent_id=..., prompt=..., root='skill_payload', bucket=..., skill_name=...). OMIT write_root to build COOPERATIVELY from scratch — the host mints ONE shared git tree the whole subagent tree writes into together (deeper descendants inherit it), and you verify the combined files with integrate_subagent_patch without reapplying them. Ignored for self_worktree and genesis (both auto-provisioned)."},
        "protected_paths_grant": {"type": "boolean", "default": False, "description": "Allow the child to modify protected paths in its self_worktree. Honored only in pro runtime mode; you still re-check at integration."},
        "external_tool_grants": {"type": "array", "items": {"type": "string"}, "description": "Optional extension/MCP tool names to grant this mutative child. Denied by default."},
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


def _validated_schedule_fields(params: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
    """Normalize and validate the public schedule_subagent fields.

    Returns ``(fields, "")`` or ``({}, refusal)``. Extracted from ``_schedule_task`` so
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
    return {
        "deadline_at": deadline_at, "objective": objective, "expected_output": expected_output,
        "role": str(params.get("role") or "researcher").strip() or "researcher",
        "context": str(params.get("context") or "").strip(),
        "constraints": str(params.get("constraints") or "").strip(),
        "memory_mode": memory_mode, "may_mutate": params.get("may_mutate", False),
        "acceptance_claims": acceptance_claims,
    }, ""


RETIRED_SCHEDULE_PARAMS: Dict[str, str] = {"effort": "reasoning_effort"}
