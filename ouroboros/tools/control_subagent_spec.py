"""The published schedule_subagent parameter surface, and its validation.

One mapping is the SSOT: the public JSON schema the model sees is built from it,
and the handler's closed keyword set is derived from the same object, so what a
parent may pass and what the schema advertises cannot drift apart. Field
normalization and the refusals for a malformed request live here beside it.
"""

from __future__ import annotations

from typing import Any, Dict

from ouroboros.subagents import (
    SUBAGENT_EXECUTORS,
    normalize_subagent_executor,
    normalize_subagent_model_lane,
)


VALID_SUBTASK_MEMORY_MODES = frozenset({"forked", "empty"})


def schedule_subagent_properties() -> Dict[str, Any]:
    """SSOT for the schedule_subagent parameter surface: ONE object, TWO derived consumers.

    The PUBLIC schema is the contract (`ToolEntry("schedule_subagent", …)` in `get_tools`, with
    `additionalProperties: False`), and the handler must refuse exactly what the schema does not
    expose. Those were previously two hand-maintained copies — this mapping and a frozenset of
    names sitting beside it, its own comment admitting it "mirrors" the schema. A mirror is only
    correct until someone adds a parameter to one side, at which point handler validation drifts
    from what the model can see: a newly published parameter gets refused as "unsupported", or a
    withdrawn one keeps being accepted. Now `get_tools` builds `properties` from this and
    `_schedule_task` builds its allowed-key set from `schedule_subagent_param_names()`, so the two
    cannot disagree (BIBLE P7).

    Returns a FRESH mapping per call, exactly as the inline literal did, so a caller that mutates
    a returned schema cannot corrupt every later `get_tools()`."""
    from ouroboros.tool_access import SUBAGENT_CAPABILITIES

    return {
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
        "model_lane": {
            "type": "string",
            "enum": ["auto", "main", "heavy", "light"],
            "default": "auto",
            "description": "How STRONG this child should be. It says nothing about what the child may DO — authority comes from write_surface. CHOOSE CONSCIOUSLY: light for a read-only micro-check, a mini-audit, a formatting or lookup task; heavy for the strong acting/coding slot; main for the ordinary strong model. Omitting it (auto) INHERITS YOUR OWN lane — the right answer when the child's answer gets committed, or when you will act on it without re-checking. Leave auto absent a specific API-strength need: an explicit lane OVERRIDES dispatch policy (a harness-dispatched nanny's own metered rounds default to the cheap light lane, and naming a lane cancels that economy). Cheap work must be NAMED light, not left unsaid. An empty Heavy/Light slot falls back to Main and the child reports the reduction in capability_delta. Depth does not change the lane; a child that finds the work harder raises itself with switch_model.",
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
            "description": "read_only (or omit) = read-only child auditing THIS repo. Otherwise the isolated write surface for a MUTATIVE child (see tool description). Acting surfaces require mutative subagents enabled (default ON in advanced/pro).",
        },
        "write_root": {"type": "string", "description": "For write_surface=external_workspace: the external project directory — a REAL external Git working tree, never runtime data. An installed non-Git skill payload is NOT an external workspace: delegate it directly with delegate_start(root='skill_payload', bucket=..., skill_name=...). OMIT write_root to build COOPERATIVELY from scratch — the host mints ONE shared git tree the whole subagent tree writes into together (deeper descendants inherit it), and you integrate the result as the sole committer. Ignored for self_worktree and genesis (both auto-provisioned)."},
        "protected_paths_grant": {"type": "boolean", "default": False, "description": "Allow the child to modify protected paths in its self_worktree. Honored only in pro runtime mode; you still re-check at integration."},
        "external_tool_grants": {"type": "array", "items": {"type": "string"}, "description": "Optional extension/MCP tool names to grant this mutative child. Denied by default."},
        "delegation_intent": {"type": "string", "description": "Optional: tell THIS child whether/how to delegate further (e.g. 'build the whole game; spawn your own children per subsystem and let them spawn too'). Propagated structurally into the child's delegation budget and surfaced in its prompt, so a 'use maximum subagents / grandchildren' intent is not lost. Defaults to inheriting the parent's intent."},
        "may_mutate": {"type": "boolean", "default": False, "description": "Optional: grant this child the intent to spawn MUTATIVE (acting) descendants of its own. Still bounded by the usual mutative-subagent gating and depth/active caps."},
        "may_fan_out": {"type": "boolean", "default": True, "description": "Optional: whether this child may spawn MULTIPLE children (a wave). Bounded by the per-root active cap."},
        "max_children": {"type": "integer", "default": 0, "description": "Optional soft cap on this child's own direct children (0 = inherit / configured cap)."},
        "required_capabilities": {
            "type": "array",
            "items": {"type": "string", "enum": list(SUBAGENT_CAPABILITIES)},
            "description": "Closed-enum capabilities this child must have (e.g. shell/vcs/write/service). The scheduler reconciles this with the selected profile before spawning; do not encode these needs in prose.",
        },
        "executor": {
            "type": "string",
            "enum": list(SUBAGENT_EXECUTORS),
            "default": "auto",
            "description": "WHO runs this child, a third axis alongside power (model_lane) and authority (write_surface). auto follows the owner's configured policy and is the right answer almost always — with no delegation route configured it simply runs native. native pins the child to Ouroboros' own metered loop. harness pins it to the configured delegation harness and is a REFUSAL to spend metered API money: when no harness route is available the child does NOT run, it ends with a typed executor-unavailable outcome (reported in capability_delta), because re-routing the pin to native would spend exactly what the pin prevents. Ask for auto if metered spend is acceptable.",
        },
        # `effort` was published here until v6.87.28 and is gone. A parent declares
        # the WORK, not the machinery: `model_lane` already answers "how good must
        # this answer be", so `model_lane: light` with `effort: max` was a request
        # nobody could resolve, and a harness route carries its own effort, so a
        # parent asking `low` against a route pinned to `xhigh` had no rule for who
        # wins. Effort is derived at dispatch from `config.resolve_effort(task_type)`
        # — the owner's control over it is exactly what it was before the knob.
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


# Runtime-INTERNAL scheduling options, deliberately absent from the public schema and
# structurally unreachable from a model tool call: they ride in the POSITIONAL-ONLY `internal`
# mapping, which no keyword argument produced from tool-call JSON can ever bind to. Keeping
# them out of the signature is also what holds the handler inside the <8-parameter contract.
#
# The set is EMPTY as of v6.87.7: its only member, `deadline_at`, became a public parameter
# once the caller judged to be the right one turned out to be the parent LLM itself — it is
# the parent that knows when a child's handoff stops being useful. The seam stays because it
# is closed and cheap, and an unknown key here still fails loudly rather than being ignored.
_INTERNAL_SCHEDULE_OPTIONS: frozenset = frozenset()


def _validated_schedule_fields(params: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
    """Normalize and validate the public schedule_subagent fields.

    Returns ``(fields, "")`` or ``({}, refusal)``. Extracted from ``_schedule_task`` so
    the handler stays inside the method-size gate — argument validation is a coherent
    phase with one job, not a slice taken to shed lines.
    """
    deadline_at = str(params.get("deadline_at") or "").strip()
    memory_mode = str(params.get("memory_mode") or "forked").strip().lower()
    try:
        model_lane = normalize_subagent_model_lane(params.get("model_lane", "auto"))
        executor = normalize_subagent_executor(params.get("executor", "auto"))
    except ValueError as exc:
        return {}, f"⚠️ TOOL_ARG_ERROR (schedule_subagent): {exc}."
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
        "model_lane": model_lane, "executor": executor,
        "acceptance_claims": acceptance_claims,
    }, ""


# A parameter this tool used to publish, mapped to the durable field it wrote.
# Separate from "unsupported" because a caller passing one is not guessing: it read
# a schema that was real, and "unsupported argument" hides that the capability still
# exists and is now derived. The REASON is not restated here — it is
# `LEGACY_SUBAGENT_FIELDS`, the same sentence the dispatch resolution puts on the
# record when it ignores a stored value, so the live refusal and the durable
# disclosure cannot come to disagree about why the field went away.
RETIRED_SCHEDULE_PARAMS: Dict[str, str] = {"effort": "reasoning_effort"}
