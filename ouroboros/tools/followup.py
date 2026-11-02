"""The agent's two tools over the EXISTING supervisor schedule table.

One table (``state/scheduled_tasks.json``), reached only through
``supervisor/queue_schedules.py``, which owns its locking, lifecycle and audit:
``schedule_followup`` WRITES a future row, and ``manage_schedules`` sees and
governs the rows that are already there — listing and getting are reads any task may do,
while changing one (including restoring a disabled or withdrawn row) stays with
the owner/root turn and is audited with a reason.  Observe visibility does not
change that original schedule authority.

``schedule_followup`` is the W=A wait affordance: when waiting for an
external instant (a subscription window reset, an embargo, a slow dependency) beats
burning rounds, the agent registers a one-shot follow-up in the supervisor's
scheduled-task table (``state/scheduled_tasks.json``). A task may also register a
recurring 5-field cron follow-up through that same table. The supervisor's ordinary
scheduler tick enqueues either form as an ordinary ROOT task (normal admission,
normal budget). No second scheduler exists: this module only reaches the table the
supervisor already consumes.

``notify=true`` leaves a NOTE instead of a wake-up: a ``kind: "notify"`` row holding
the agent's own words and when they were written, which the tick shows to the owner
verbatim at the trigger as a System row, with no model call
(``supervisor/schedule_notes.py``). Whether the words will still be true by then is
the agent's choice at scheduling time; the host never classifies the text. A note
reaches the owner directly, so Presence, agent-to-agent and delegated callers are
refused (``control_runtime.owner_contact_refusal``).

Authority is narrower than the parent's, not wider: a delegated subagent may not
mint future root tasks (typed refusal) and the objective is the agent's own plain
text (no host template). How many follow-ups a task registers is the agent's
judgment, not a host count: every registration reports this task's pending total,
and the owner sees and controls each record on the Schedules surface.
"""

from __future__ import annotations

from ouroboros.tools.tool_result import ToolResult, _publish_tool_result, publish_no_effect

import functools
import json
import uuid
from typing import Any, Dict, List

from ouroboros.consciousness_authority import consciousness_origin_metadata, work_kind
from ouroboros.deadline_utils import parse_deadline_ts
from ouroboros.dialogue_provenance import presence_caller_binding, presence_root_carrier
from ouroboros.settings_scales import EFFORT_SCALE, requested_effort
from ouroboros.tools.arg_feedback import ignored_argument_note
from ouroboros.tools.registry import ToolContext, ToolEntry

FOLLOWUP_SOURCE = "task_followup"
_MAX_OBJECTIVE_CHARS = 4_000
_MAX_CONTEXT_CHARS = 8_000


def _manage_schedules(
    ctx: ToolContext, action: str = "list", schedule_id: str = "", reason: str = "",
    offset: int = 0, limit: int = 20, expected_hold_id: str = "", relation: str = "",
    name: str = "", objective: str = "", context: str = "", run_at: str = "",
    cron: str = "", timezone: str = "", reasoning_effort: str = "",
) -> str:
    """See, or apply one narrow audited change to, the rows already in the table.

    The authority is argument-aware rather than name-wide: READING is allowed to any
    member of the tree, while CHANGING the owner's schedules stays with
    the root turn. A Presence conversation sees only the follow-ups its own binding
    scheduled, from any of its conversations, and may withdraw one (``delete``):
    an admitted guest room never reads or rewrites the rest of this mind's table
    (``queue_schedules.presence_promise``, rechecked on the stored row under the
    write's transaction).
    """
    from supervisor.queue import (
        SCHEDULE_ACTIONS, ScheduleRefused, ScheduleStoreUnreadable, load_schedule_store,
        mutate_scheduled_task, schedule_tool_projection,
    )
    from supervisor.queue_schedules import presence_promise, presence_promise_store

    operation = str(action or "list").strip().lower()
    binding = presence_caller_binding(ctx)  # a speaker, or work acting for its binding
    if binding is not None and (not binding or operation not in {"list", "get", "delete"}):
        return publish_no_effect(ctx, ToolResult(
            status="blocked", code="RESOURCE_CONSTRAINT_BLOCKED",
            text=("⚠️ RESOURCE_CONSTRAINT_BLOCKED: a Presence conversation lists, reads and withdraws (delete) "
                  "only the follow-ups its place scheduled; owner schedules and other changes stay out of reach."),
        ))
    owned_by = functools.partial(presence_promise, binding) if binding is not None else None
    if operation not in {"list", "get"} and not _root_schedule_mutation_authorized(ctx):
        return publish_no_effect(ctx, ToolResult(
            status="blocked", code="RESOURCE_CONSTRAINT_BLOCKED",
            text=(
                "⚠️ RESOURCE_CONSTRAINT_BLOCKED: a delegated task may only read schedules "
                "(action='list' or 'get'); changing one is the root turn's authority. Report the "
                f"schedule you would change instead of {operation!r}."
            ),
        ))
    root = getattr(ctx, "budget_drive_root", None) or getattr(ctx, "drive_root", None)
    try:
        from ouroboros.tool_capabilities import tool_result_limit
        result_limit = tool_result_limit("manage_schedules")
        if operation == "list":
            from supervisor.followup_policy import observed_store
            store = load_schedule_store(root)
            return json.dumps(schedule_tool_projection(observed_store(root, store if binding is None else
                                                                      presence_promise_store(store, binding)),
                                                       offset=offset, limit=limit,
                                                       result_limit=result_limit),
                              ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"))
        if operation == "get":
            from supervisor.followup_policy import observed_store

            row = next((row for row in load_schedule_store(root)["tasks"]
                        if str(row.get("id") or "") == str(schedule_id or "").strip()), None)
            if row is None or (owned_by is not None and not owned_by(row)):
                return publish_no_effect(ctx, json.dumps({"ok": False, "status": "not_found",
                                         "schedule_id": schedule_id}), tool_name="manage_schedules")
            row = observed_store(root, {"tasks": [row]})["tasks"][0]
            return _schedule_definition_reply(row, ok=True)
        if operation == "update":
            return _update_schedule(ctx, root, schedule_id, reason, {
                "name": name, "objective": objective, "context": context, "run_at": run_at,
                "cron": cron, "timezone": timezone, "reasoning_effort": reasoning_effort})
        if operation not in SCHEDULE_ACTIONS:
            return publish_no_effect(ctx, json.dumps({"ok": False, "status": "invalid_action",
                               "allowed": ["list", "get", "update", *sorted(SCHEDULE_ACTIONS)]}, sort_keys=True), tool_name="manage_schedules")
        # Preserve selectors verbatim. Refuse an identity that cannot fit in a
        # truthful receipt before changing anything, rather than cutting it.
        if len(json.dumps(str(schedule_id or ""), ensure_ascii=False)) > result_limit - 2_000:
            return publish_no_effect(ctx, json.dumps({"ok": False, "changed": False, "status": "identity_too_large",
                               "audit": "not_written", "detail": "Schedule identity exceeds the tool result limit; nothing changed."}), tool_name="manage_schedules")
        outcome = mutate_scheduled_task(
            operation, schedule_id, reason=reason, actor="agent",
            task_id=str(getattr(ctx, "task_id", "") or ""), drive_root=root,
            expected_hold_id=expected_hold_id, relation=relation,
            writer_kind=work_kind(getattr(ctx, "task_metadata", None)),
            writer_metadata={**(getattr(ctx, "task_metadata", None) or {}), "task_id": str(getattr(ctx, "task_id", "") or "")},
            owned_by=owned_by)
    except ScheduleRefused as exc:
        # Same typed marker the sibling refusals carry: the text channel is the
        # registered ABI, so a refusal must be legible there and not only in the
        # published sidecar.
        return _publish_tool_result(ctx, ToolResult(
            status="unavailable", code="CAPABILITY_UNAVAILABLE",
            text=f"⚠️ CAPABILITY_UNAVAILABLE: {exc.status}: {exc}"))
    except ScheduleStoreUnreadable as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="unavailable", code="CAPABILITY_UNAVAILABLE",
            text=f"⚠️ CAPABILITY_UNAVAILABLE: the schedule table could not be read: {exc}",
        ))
    # The HTTP/audit seam retains the lifecycle row. The model gets the same
    # lifecycle fact through a bounded projection, never an arbitrary task
    # template or an unbounded owner-authored label.
    try:
        schedule = outcome.get("schedule")
        if isinstance(schedule, dict):
            outcome["schedule"] = schedule_tool_projection(
                {"tasks": [_stored_row(root, schedule)]}, limit=1, result_limit=result_limit)["tasks"][0]
    except ScheduleRefused as exc:
        # The mutation is ALREADY durable and audited by the time the row is
        # rendered. A row this process cannot fit is a missing receipt detail,
        # never an action that did not run: reporting CAPABILITY_UNAVAILABLE
        # here would describe a change the owner's table already holds as
        # nothing having happened. Drop the row, keep the lifecycle receipt.
        outcome.pop("schedule", None)
        outcome["schedule_omitted"] = True
        outcome["schedule_omitted_reason"] = f"{exc.status}: {exc}"
    if "detail" in outcome:
        detail = str(outcome["detail"])
        outcome["detail"] = detail[:500]
        if len(detail) > 500:
            outcome["detail_truncated"] = True
    encoded = json.dumps(outcome, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(encoded) <= result_limit:
        return encoded
    # A manually edited row can still carry an unusually large identity or
    # metadata value. Keep the action receipt valid and explicit rather than
    # letting the outer transport cut JSON mid-string; selectors are retained.
    compact = {key: outcome[key] for key in (
        "ok", "changed", "status", "schedule_id", "operation_id", "running_or_queued", "audit",
        "detail", "detail_truncated", "schedule_omitted_reason",
    ) if key in outcome}
    # ``detail`` is already bounded above and often names the actual blocker;
    # keep it rather than replacing the one explanation
    # the receipt carries with a generic note about its own size.
    compact.setdefault(
        "detail", "schedule action completed; full row omitted to fit the tool result limit")
    compact["schedule_omitted"] = True
    return json.dumps(compact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stored_row(root: Any, audit_row: dict) -> dict:
    """The changed row as stored now: the audit row lacks the facts the view derives."""
    from supervisor.followup_policy import observed_store
    from supervisor.queue import ScheduleStoreUnreadable, load_schedule_store

    try:
        row = next((row for row in load_schedule_store(root)["tasks"]
                    if str(row.get("id") or "") == str(audit_row.get("id") or "")), None)
    except ScheduleStoreUnreadable:
        return audit_row
    return observed_store(root, {"tasks": [row]})["tasks"][0] if row else audit_row


def _schedule_definition_reply(row: dict, **facts) -> str:
    """Keep the full definition; tool-result delivery retains a source before any cut."""
    from supervisor.queue import schedule_activity_projection

    return json.dumps({**facts, "schedule": schedule_activity_projection({"tasks": [row]})["tasks"][0]},
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _update_schedule(ctx: ToolContext, root: Any, schedule_id: str, reason: str, fields: dict) -> str:
    from supervisor.queue import patch_scheduled_task

    patch = {key: str(value).strip() for key, value in fields.items() if value is not None and str(value).strip()}
    if patch.get("run_at") and patch.get("cron"):
        return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(
            "ERROR: FOLLOWUP_TRIGGER_REQUIRED: supply run_at or cron, not both. Nothing changed.")))
    for key in ("objective", "context"):
        if error := _text_size_error(key, patch.get(key, "")):
            return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=error))
    notes = ""
    if "reasoning_effort" in patch:
        try:
            effort, notes = _starting_effort(patch.pop("reasoning_effort"))
        except ValueError as exc:
            return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(
                f"ERROR: FOLLOWUP_EFFORT_INVALID: {exc}. Nothing changed.")))
        if effort:
            patch["reasoning_effort"] = effort
    if patch.get("timezone") and _naive_instant(patch.get("run_at", "")):
        return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(
            f"ERROR: FOLLOWUP_TIMEZONE_WITH_RUN_AT: run_at={patch['run_at']!r} carries no UTC offset and "
            "timezone applies only to recurring cron schedules. Put the offset into run_at "
            "(example: 2026-08-19T12:20:00+03:00) and omit timezone. Nothing changed.")))
    if "run_at" in patch:
        patch["trigger"] = {"type": "once", "run_at": patch.pop("run_at")}
    elif "cron" in patch:
        patch["trigger"] = {"type": "cron", "expr": patch.pop("cron")}
    outcome = patch_scheduled_task(schedule_id, patch, reason=reason, actor="agent",
                                   task_id=str(getattr(ctx, "task_id", "") or ""), drive_root=root,
                                   writer_kind=work_kind(getattr(ctx, "task_metadata", None)),
                                   writer_metadata={**(getattr(ctx, "task_metadata", None) or {}),
                                                    "task_id": str(getattr(ctx, "task_id", "") or "")})
    if outcome.get("ok") is False:
        return publish_no_effect(ctx, json.dumps(outcome, ensure_ascii=False), tool_name="manage_schedules")
    facts = {key: outcome[key] for key in ("status", "changed_fields", "audit", "reconsidered_claim")}
    row = {key: value for key, value in outcome.items() if key not in facts}
    if notes:
        facts["argument_note"] = notes.strip()
    return _schedule_definition_reply(row, ok=outcome["audit"] == "recorded",
                                      changed=outcome["status"] == "updated", **facts)


def _next_due(stored: dict, trigger: dict) -> str:
    """A one-shot is due at its own instant; a cron at its computed next run."""
    once = trigger.get("run_at") if trigger.get("type") == "once" else ""
    return str(stored.get("next_run_at") or once or "not determined")


def _text_size_error(field: str, text: str) -> str:
    limit = _MAX_OBJECTIVE_CHARS if field == "objective" else _MAX_CONTEXT_CHARS
    return (f"ERROR: FOLLOWUP_TEXT_TOO_LONG: {field} is {len(text)} chars; the limit is {limit}. "
            "Shorten it — nothing was truncated and nothing changed.") if len(text) > limit else ""


def _starting_effort(value: Any) -> tuple[str, str]:
    from ouroboros.runtime_mode_policy import effort_range_binds
    from ouroboros.settings_scales import effort_range

    effort = requested_effort(value)
    if effort_range_binds():
        return "", " " + ignored_argument_note(
            "reasoning_effort", effort,
            "outside Cyber Pro a future task I schedule starts at my human's recommended level "
            f"({effort_range()['recommended']})") + "."
    return effort, ""


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="schedule_followup",
            schema={
                "name": "schedule_followup",
                "description": (
                    "Create a named future task or note in the supervisor schedule table. Supply run_at "
                    "for a one-shot ISO 8601 instant (naive = UTC), or cron for a recurring 5-field "
                    "expression with an optional IANA timezone. The receipt includes the next due instant "
                    "and this task's pending count. Schedules fire only while Ouroboros runs; after downtime "
                    "a missed occurrence catches up once, without replaying every missed cron tick. "
                    "A task wakes as an ordinary root through normal admission; objective becomes its text "
                    "and can name the skill or knowledge containing the living procedure. With notify=true "
                    "nothing wakes: the host shows objective verbatim in the owner's chat as a System row "
                    "with written/due times, without a model call or context. Choose related work to keep "
                    "the original verified money/deadline and Stop controls, or independent work; notes are "
                    "independent. Either task relation can read this task's task_drive/artifact_store "
                    "read-only, one hop, at paths you name. Root tasks only; Presence cannot leave notes. "
                    "manage_schedules gets full definitions, updates a promise in place, disables future "
                    "dispatch, or withdraws it with delete; withdrawal remains restorable for a while."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Optional label for this promise, in your own words."},
                        "relation": {"type": "string", "enum": ["related", "independent"],
                                     "description": "Explicit relationship to this work. Related preserves its original money and hard deadline; independent is separate work. Author/cron/text do not decide this. A note is always independent."},
                        "run_at": {
                            "type": "string",
                            "description": "One-shot ISO 8601 instant to fire at/after (naive = UTC).",
                        },
                        "cron": {
                            "type": "string",
                            "description": "Recurring 5-field cron expression; mutually exclusive with run_at.",
                        },
                        "timezone": {
                            "type": "string",
                            "description": "Optional IANA timezone for cron (blank = system local timezone). Not for run_at: put the UTC offset into run_at itself (a zone beside an offset-carrying run_at is ignored).",
                        },
                        "objective": {
                            "type": "string",
                            "description": (
                                "Plain-language objective for the future task (or the note's exact text), in your own words "
                                f"(max {_MAX_OBJECTIVE_CHARS} chars; longer is a typed refusal, never truncated)."
                            ),
                        },
                        "notify": {
                            "type": "boolean",
                            "description": "true = a note shown to the owner at the trigger without a model call (objective only, no context). Default false = a wake-up task.",
                        },
                        "context": {
                            "type": "string",
                            "description": (
                                "Optional context the future task should start from (facts, ids, paths; "
                                f"max {_MAX_CONTEXT_CHARS} chars; longer is a typed refusal, never truncated)."
                            ),
                        },
                        "reasoning_effort": {
                            "type": "string", "enum": list(EFFORT_SCALE),
                            "description": "Optional: the reasoning effort each future task starts on; applied in "
                                           "Cyber Pro only — otherwise the task starts at my human's recommended "
                                           "level and the result says so. Omit for that level.",
                        },
                    },
                    "required": ["objective", "relation"],
                },
            },
            handler=_handle_schedule_followup,
            timeout_sec=30,
        ),
        ToolEntry("manage_schedules", {
            "name": "manage_schedules",
            "description": (
                "Read and change existing supervisor schedules. list pages bounded previews with "
                "offset/limit and total/next_offset. get returns one full authored definition and current "
                "lifecycle/outcome facts; oversized results stay readable through the persisted result source. "
                "update changes supplied name, objective, context, run_at or cron, timezone, and starting "
                "reasoning_effort in place; blank strings mean unchanged. objective is the future task's "
                "text, naming the skill or knowledge with its living procedure, or a note's verbatim words "
                "shown without waking a model. Every change requires a reason and is audited. Update "
                "preserves policy holds and paused state; its receipt says if an unstarted claim was "
                "reconsidered. A consumed one-shot re-arms only with a new run_at or a cron. "
                "disable/delete stop future dispatch; admitted work keeps running. delete withdraws the "
                "record, restorable for a while and retained until its accepted work settles and retention "
                "expires. Generic restore cancels withdrawal (the row comes back enabled, even if paused before) or enables a paused row; a consumed one-shot "
                "stays consumed. Held restore needs the observed hold_id; an unknown relation must first "
                "be resolved. Exact hold release preserves enabled/deletion state, pending task identity, "
                "money/deadline and other holds. Changes belong to a root turn; delegated tasks may "
                "list/get only; Observe cannot update; "
                "a Presence conversation lists, gets and deletes only the follow-ups its place scheduled. "
                "Schedules fire only while Ouroboros runs, with one late "
                "catch-up after downtime rather than replaying missed cron ticks."
            ),
            "parameters": {"type": "object", "properties": {
                "action": {"type": "string", "enum": ["list", "get", "update", "disable", "delete", "restore"]},
                "schedule_id": {"type": "string", "description": "Schedule id for get or a change."},
                "reason": {"type": "string", "description": "Why this change is needed."},
                "name": {"type": "string", "description": "Update: your label for the schedule. Blank means unchanged."},
                "objective": {"type": "string", "description": f"Update: task objective or exact note text (max {_MAX_OBJECTIVE_CHARS} chars). Blank means unchanged."},
                "context": {"type": "string", "description": f"Update: task context (max {_MAX_CONTEXT_CHARS} chars). Blank means unchanged; clearing context is not offered."},
                "run_at": {"type": "string", "description": "Update: replace the trigger with a one-shot ISO 8601 instant (naive = UTC). Mutually exclusive with cron; blank means unchanged."},
                "cron": {"type": "string", "description": "Update: replace the trigger with a recurring 5-field cron expression. Mutually exclusive with run_at; blank means unchanged."},
                "timezone": {"type": "string", "description": "Update: IANA timezone. Blank means unchanged."},
                "reasoning_effort": {"type": "string", "enum": [*EFFORT_SCALE], "description": "Update: starting effort for future tasks, applied in Cyber Pro only. Otherwise ignored with a disclosure. Omit it to leave the effort unchanged."},
                "expected_hold_id": {"type": "string", "description": "Exact hold_id from list, required to restore a held row. A stale identity refuses."},
                "relation": {"type": "string", "enum": ["related", "independent"], "description": "Resolve a legacy unknown relationship on restore. Related must bind the existing origin; no fresh budget is inferred."},
                "offset": {"type": "integer", "minimum": 0, "description": "Zero-based list offset."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Rows per list page (max 20; responses stay bounded)."},
            }, "required": ["action"], "additionalProperties": False},
        }, _manage_schedules, timeout_sec=30),
    ]


def _is_delegated_subagent(ctx: ToolContext) -> bool:
    """Fail closed for delegated profiles, even when lineage metadata is stale."""
    try:
        from ouroboros.tool_access import active_tool_profile

        profile = str(active_tool_profile(ctx) or "").strip()
    except Exception:
        # A profile resolution failure cannot prove owner authority.  Treat the
        # caller as delegated so every mutation door fails closed.
        return True
    if profile in {"acting_subagent", "local_readonly_subagent"}:
        return True
    for attr in ("task_metadata", "task_contract"):
        data = getattr(ctx, attr, None)
        if isinstance(data, dict) and str(data.get("delegation_role") or "").strip() == "subagent":
            return True
    return False


def _root_schedule_mutation_authorized(ctx: ToolContext) -> bool:
    """Only a resolved top-level principal may mutate the owner schedule table.

    This intentionally does not infer authority from ``delegation_role`` alone:
    stale/missing metadata must not turn an acting or read-only delegated profile
    into an owner.  Profile resolution itself is fail-closed.
    """
    try:
        from ouroboros.tool_access import _TOP_LEVEL_PRINCIPAL_PROFILES, active_tool_profile

        profile = str(active_tool_profile(ctx) or "").strip()
    except Exception:
        return False
    if profile not in _TOP_LEVEL_PRINCIPAL_PROFILES:
        return False
    return not _is_delegated_subagent(ctx)


def _pending_followups(records: List[Dict[str, Any]], task_id: str) -> List[Dict[str, Any]]:
    out = []
    for record in records:
        if not isinstance(record, dict) or not record.get("enabled", True):
            continue
        if str(record.get("source") or "") != FOLLOWUP_SOURCE:
            continue
        from supervisor.followup_policy import origin_of
        if origin_of(record).get("task_id") == task_id:
            out.append(record)
    return out


def _naive_instant(raw: str) -> bool:
    """True when an ISO instant names no UTC offset (such a time is read as UTC)."""
    from datetime import datetime

    try:
        return datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw).tzinfo is None
    except ValueError:
        return False


def _handle_schedule_followup(ctx: ToolContext, **params) -> str:
    if _is_delegated_subagent(ctx):
        return (
            publish_no_effect(ctx, ToolResult(status="blocked", code="ACCESS_BLOCKED", text=("ERROR: FOLLOWUP_SUBAGENT_REFUSED: a delegated subagent holds narrower-than-parent "
            "authority and may not mint future root tasks. Report the wait instant to your "
            "parent instead; the parent (or the owner) decides whether to schedule a follow-up.")))
        )
    task_id = str(getattr(ctx, "task_id", "") or "").strip()
    if not task_id:
        return publish_no_effect(ctx, ToolResult(status="unavailable", code="CAPABILITY_UNAVAILABLE", text=("ERROR: FOLLOWUP_TASK_ID_REQUIRED: a durable follow-up must belong to a real task.")))
    notify = False if params.get("notify") is None else params.get("notify")  # models may send null
    if not isinstance(notify, bool):
        return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(
            "ERROR: FOLLOWUP_NOTIFY_INVALID: notify must be true (a note) or false (a wake-up task). Nothing was scheduled.")))
    if notify:
        from ouroboros.tools.control_runtime import owner_contact_refusal

        if refusal := owner_contact_refusal(ctx, getattr(ctx, "current_chat_id", None)):
            return publish_no_effect(ctx, ToolResult(status="blocked", code="ACCESS_BLOCKED", text=(
                f"ERROR: FOLLOWUP_NOTIFY_REFUSED: {refusal}. No note was scheduled.")))
    run_at_raw = str(params.get("run_at") or "").strip()
    cron = str(params.get("cron") or "").strip()
    if bool(run_at_raw) == bool(cron):
        return (
            publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=("ERROR: FOLLOWUP_TRIGGER_REQUIRED: supply exactly one of run_at (one-shot) "
            "or cron (recurring).")))
        )
    timezone = str(params.get("timezone") or "").strip()
    timezone_note = ""
    if run_at_raw:
        instant = parse_deadline_ts(run_at_raw)
        if timezone and instant is not None:
            if _naive_instant(run_at_raw):
                # A zone beside a run_at WITHOUT an offset asks for something: ignoring it would
                # schedule the naive time as UTC, hours away from what was meant.
                return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(
                    f"ERROR: FOLLOWUP_TIMEZONE_WITH_RUN_AT: run_at={run_at_raw!r} carries no UTC offset and "
                    f"timezone={timezone!r} applies only to recurring cron follow-ups. Put the offset into run_at "
                    "(example: 2026-08-19T12:20:00+03:00) and omit timezone. Nothing was scheduled.")))
            # run_at with its own offset is an absolute instant: a zone beside it asks for nothing.
            timezone_note = " " + ignored_argument_note(
                "timezone", timezone, "it applies only to recurring cron follow-ups; run_at carries its own offset") + "."
            timezone = ""
        if instant is None:
            return (
                publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(f"ERROR: FOLLOWUP_RUN_AT_INVALID: {run_at_raw!r} is not a parseable ISO 8601 "
                "instant. Example: 2026-08-19T12:20:00+03:00 (naive times read as UTC).")))
            )
        trigger = {"type": "once", "run_at": instant.isoformat()}
    else:
        from ouroboros.schedule_contract import cron_error, timezone_error

        if error := cron_error(cron):
            return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(f"ERROR: FOLLOWUP_CRON_INVALID: {error}")))
        if error := timezone_error(timezone):
            return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(f"ERROR: FOLLOWUP_TIMEZONE_INVALID: {error}")))
        trigger = {"type": "cron", "expr": cron}
    objective = str(params.get("objective") or "").strip()
    if not objective:
        return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=("ERROR: FOLLOWUP_OBJECTIVE_REQUIRED: write the future task's objective in plain language.")))
    # Typed refusal, never a silent cut: the text rides VERBATIM into the future
    # task, so truncating it here would silently change what that task is.
    if error := _text_size_error("objective", objective):
        return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=error))
    context = str(params.get("context") or "").strip()
    if notify and context:
        return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(
            "ERROR: FOLLOWUP_NOTIFY_CONTEXT: a note is shown exactly as its objective; put what the owner "
            "should read there and omit context. Nothing was scheduled.")))
    if error := _text_size_error("context", context):
        return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=error))
    effort = ""
    if params.get("reasoning_effort") is not None and notify:
        timezone_note += " " + ignored_argument_note(
            "reasoning_effort", params["reasoning_effort"], "a note wakes no model") + "."
    elif params.get("reasoning_effort") is not None:
        try:
            effort, effort_note = _starting_effort(params["reasoning_effort"])
            timezone_note += effort_note
        except ValueError as exc:
            return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(
                f"ERROR: FOLLOWUP_EFFORT_INVALID: {exc}. Nothing was scheduled.")))
    from ouroboros.tool_access import canonical_data_root
    from supervisor.queue import schedule_transaction

    try:
        drive_root = canonical_data_root(ctx)
    except Exception as exc:
        return publish_no_effect(ctx, ToolResult(status="error", code="TOOL_ERROR", text=(f"ERROR: FOLLOWUP_DATA_ROOT_UNRESOLVED: {exc}")))
    from supervisor.followup_policy import resolve_relation
    origin = {"task_id": task_id, "root_task_id": str(getattr(ctx, "root_task_id", "") or
              (getattr(ctx, "task_metadata", None) or {}).get("root_task_id") or task_id)}
    if notify and str(params.get("relation") or "") == "related":
        timezone_note += " " + ignored_argument_note(
            "relation", "related", "a note carries no money or deadline, so it is recorded as independent") + "."
    try:
        relation = resolve_relation(drive_root, origin, "independent" if notify else params.get("relation"),
                                    declared_by=task_id)
    except ValueError as exc:
        return publish_no_effect(ctx, ToolResult(status="unavailable", code="CAPABILITY_UNAVAILABLE",
                                    text=f"FOLLOWUP_REFUSED: {exc}"))
    # The pending count is read from the table this call is about to write, so the
    # reported total and the write share ONE transaction.
    from supervisor.queue import ScheduleStoreUnreadable

    try:
        with schedule_transaction(drive_root):
            if notify:
                return _register_note(ctx, task_id, drive_root, objective, trigger, cron, timezone,
                                      timezone_note, origin, relation, name=str(params.get("name") or "").strip())
            return _register_followup(ctx, task_id, drive_root, objective, context,
                                      trigger, cron, timezone, timezone_note, origin, relation,
                                      reasoning_effort=effort, name=str(params.get("name") or "").strip())
    except ScheduleStoreUnreadable as exc:
        # A missed table lock (ScheduleLockTimeout) or an unparseable table: the
        # follow-up was NOT registered, and the refusal says so in the text ABI
        # instead of surfacing as a generic tool error.
        return _publish_tool_result(ctx, ToolResult(
            status="unavailable", code="CAPABILITY_UNAVAILABLE",
            text=f"⚠️ CAPABILITY_UNAVAILABLE: FOLLOWUP_STORE_UNAVAILABLE: {exc}"))


def followup_resource_intent(ctx: Any, metadata: Any, project_id: str) -> Dict[str, Any]:
    """The resource intent a follow-up carries: the origin's own stamped intent, or a
    direct turn's (its room's current folder, or Main's system repo). A queued task
    without a stamp carries none — the scheduler recovers only from its record."""
    stamped = metadata.get("resource_intent") if isinstance(metadata, dict) else None
    if isinstance(stamped, dict) and stamped.get("kind"):
        return dict(stamped)
    if not bool(getattr(ctx, "is_direct_chat", False)) or getattr(ctx, "workspace_root", None):
        return {}
    return {"kind": "room_default", "project_id": project_id} if project_id else {"kind": "system_repo"}


def _pending_for_task(task_id: str, drive_root: Any) -> list:
    """This task's pending follow-ups (notes included): a fact for the reply, never a cap."""
    from supervisor.queue import load_schedule_store

    return _pending_followups(load_schedule_store(drive_root)["tasks"], task_id)


def _register_note(ctx: ToolContext, task_id: str, drive_root: Any, objective: str,
                   trigger: Dict[str, Any], cron: str, timezone: str, notes: str,
                   origin: dict, relation: dict, *, name: str = "") -> str:
    """A note: the same table and lifecycle as a follow-up, but no task template.

    The row keeps the words and when they were written; the scheduler tick shows
    them at the trigger (``supervisor/schedule_notes.py``)."""
    from ouroboros.utils import utc_now_iso
    from supervisor.queue import ScheduleRefused, ScheduleStoreUnreadable, upsert_scheduled_task

    pending = _pending_for_task(task_id, drive_root)
    record = {
        "id": f"followup-{task_id}-{uuid.uuid4().hex[:6]}", "name": name or f"Reminder of task {task_id}",
        "kind": "notify", "source": FOLLOWUP_SOURCE, "enabled": True,
        "timezone": timezone, "trigger": trigger,
        "notification": {"text": objective, "set_at": utc_now_iso()},
    }
    try:
        stored = upsert_scheduled_task(
            record, drive_root=drive_root, actor="agent:schedule_followup", task_id=task_id,
            writer_kind=work_kind(getattr(ctx, "task_metadata", None)),
            writer_metadata={**(getattr(ctx, "task_metadata", None) or {}), "task_id": str(getattr(ctx, "task_id", "") or "")},
            reason=f"note registered by task {task_id}",
            host_followup={"followup_origin": origin, "followup_relation": relation})
    except ScheduleRefused as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="unavailable", code="CAPABILITY_UNAVAILABLE",
            text=f"⚠️ CAPABILITY_UNAVAILABLE: FOLLOWUP_REFUSED: {exc.status}: {exc}"))
    except ScheduleStoreUnreadable:
        raise  # the caller's typed FOLLOWUP_STORE_UNAVAILABLE refusal owns this
    except Exception as exc:
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ERROR", text=(
            f"ERROR: FOLLOWUP_PERSIST_FAILED: {type(exc).__name__}: {exc}; inspect schedule {record['id']} before retrying")))
    timing = (f"once at/after {trigger['run_at']}" if trigger["type"] == "once"
              else f"recurring cron {cron} ({timezone or 'system local time'})")
    audit = str(stored.get("audit") or "")
    audit_note = "" if audit == "recorded" else (
        f" AUDIT_INCOMPLETE: the row is durable, but its audit outcome fact could not be written (audit={audit or 'unknown'}).")
    return (
        f"FOLLOWUP_SCHEDULED: note {stored.get('id')} registered {timing}. "
        f"Next due: {_next_due(stored, trigger)}. Nothing wakes then: the host shows "
        "your objective verbatim in the owner's chat as a System row signed with when you wrote it and when "
        "it was due (after downtime, at the next start with both times). Pending follow-ups for this task: "
        f"{len(pending) + 1}; the owner can disable or delete it from the Schedules "
        f"surface.{audit_note}{notes}"
    )


def _register_followup(ctx: ToolContext, task_id: str, drive_root: Any,
                       objective: str, context: str, trigger: Dict[str, Any], cron: str,
                       timezone: str, timezone_note: str, origin: dict, relation: dict,
                       *, reasoning_effort: str = "", name: str = "") -> str:
    """The pending read, the record build and the write — all under the schedule lock."""
    from supervisor.queue import upsert_scheduled_task

    pending = _pending_for_task(task_id, drive_root)
    metadata_src = getattr(ctx, "task_metadata", None)
    root_task_id = metadata_src.get("root_task_id") if isinstance(metadata_src, dict) else None
    project_id = str(
        getattr(ctx, "project_id", "")
        or (metadata_src.get("project_id") if isinstance(metadata_src, dict) else "")
        or ""
    ).strip()
    source_chat_id = getattr(ctx, "current_chat_id", None)
    if source_chat_id in (None, "") and isinstance(metadata_src, dict):
        source_chat_id = metadata_src.get("chat_id")
    record = {
        "id": f"followup-{task_id}-{uuid.uuid4().hex[:6]}",
        "name": name or f"Follow-up of task {task_id}",
        "source": FOLLOWUP_SOURCE,
        "enabled": True,
        "timezone": timezone,
        "trigger": trigger,
        "task": {
            "type": "task",
            "text": objective,
            **({"context": context} if context else {}),
            **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
            **({"project_id": project_id} if project_id else {}),
            "metadata": {
                "source": FOLLOWUP_SOURCE,
                "origin_task_id": task_id,
                # `or ""` before str(): an absent/None root_task_id must fall back
                # to task_id, never become the literal string "None".
                "origin_root_task_id": str(root_task_id or "") or task_id,
                # TZ-2 B3: the same author stamp a promote carries — the successor's
                # first turn is this task's note, framed as such, never an owner directive.
                "objective_author": {"kind": "task", "task_id": task_id},
            },
            **({"chat_id": source_chat_id} if source_chat_id not in (None, "") else {}),
        },
    }
    # The host writer persists this origin separately; the template is not its
    # authority. Only independent work uses the root-start exception and cause.
    record["task"]["metadata"].update(consciousness_origin_metadata(
        metadata_src, started_root=relation["kind"] == "independent"))
    # The resource choice rides BY VALUE (#1315): the origin's producer-stamped intent,
    # else a direct turn's own (its room's default, or Main's system repo).
    intent = followup_resource_intent(ctx, metadata_src, project_id)
    if intent:
        record["task"]["metadata"]["resource_intent"] = intent
    # The same Presence carrier a promote keeps: a speaker's metadata, or the binding a
    # promoted descendant root acts for; the ceiling rides by value and no Project is chosen.
    # A new root authors its own objective, context and acceptance premises;
    # only the origin's authority and general constraints survive by value. The
    # origin's own predecessor is not this root's: it reads its direct origin
    # only (``tool_access.scheduled_origin``, #1488), never a second hop.
    contract = getattr(ctx, "task_contract", None)
    carrier = presence_root_carrier(metadata_src, task_contract=contract)
    if carrier and isinstance(contract, dict):
        record["task"]["metadata"].update(carrier)
        record["task"]["task_contract"] = {
            key: value for key, value in contract.items()
            if key not in {"objective", "context", "expected_output", "acceptance_claims", "success_criteria",
                           "predecessor_authority"}}
        record["task"].pop("project_id", None)
    from supervisor.queue import ScheduleRefused, ScheduleStoreUnreadable

    try:
        stored = upsert_scheduled_task(
            record, drive_root=drive_root, actor="agent:schedule_followup", task_id=task_id,
            writer_kind=work_kind(getattr(ctx, "task_metadata", None)),
            writer_metadata={**(getattr(ctx, "task_metadata", None) or {}), "task_id": str(getattr(ctx, "task_id", "") or "")},
            reason=f"follow-up registered by task {task_id}",
            host_followup={"followup_origin": origin, "followup_relation": relation},
            # Host-authored provenance of a NAMED continuation (never payload/prose):
            # it keeps a consciousness follow-up out of the spontaneous-start cap.
            continuation_of={"task_id": task_id, "root_task_id": str(root_task_id or "") or task_id})
    except ScheduleRefused as exc:
        return _publish_tool_result(ctx, ToolResult(
            status="unavailable", code="CAPABILITY_UNAVAILABLE",
            text=f"⚠️ CAPABILITY_UNAVAILABLE: FOLLOWUP_REFUSED: {exc.status}: {exc}"))
    except ScheduleStoreUnreadable:
        raise  # the caller's typed FOLLOWUP_STORE_UNAVAILABLE refusal owns this
    except Exception as exc:
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ERROR", text=(f"ERROR: FOLLOWUP_PERSIST_FAILED: {type(exc).__name__}: {exc}")))
    if trigger["type"] == "once":
        timing = f"once at/after {trigger['run_at']}"
        lifecycle = "fires exactly once"
    else:
        zone = timezone or "system local time"
        timing = f"recurring cron {cron} ({zone})"
        lifecycle = "remains active after each run"
    # The write seam returns what its audit achieved. A durable record whose
    # outcome fact was lost is not a clean success, and saying "registered" alone
    # would hide the one thing that makes the change accountable.
    audit = str(stored.get("audit") or "")
    audit_note = "" if audit == "recorded" else (
        f" AUDIT_INCOMPLETE: the record is durable, but its audit outcome fact could not "
        f"be written (audit={audit or 'unknown'}), so this registration is not fully "
        "accounted for in logs/events.jsonl.")
    return (
        f"FOLLOWUP_SCHEDULED: follow-up {stored.get('id')} registered as {timing}. "
        f"Next due: {_next_due(stored, trigger)}. Future enqueue "
        "requires eligible relationship/controls and ordinary supervisor admission; "
        f"pending follow-ups for this task: {len(pending) + 1}. The "
        f"record is durable in state/scheduled_tasks.json and {lifecycle}; the owner can "
        f"disable or delete it from the Schedules surface. Relationship: {relation['kind']}; "
        f"hold: {stored.get('followup_hold') or 'none'}; wait: {stored.get('followup_wait') or 'none'}."
        f"{audit_note}{timezone_note}"
    )
