"""The agent's two tools over the EXISTING supervisor schedule table.

One table (``state/scheduled_tasks.json``), reached only through
``supervisor/queue_schedules.py``, which owns its locking, lifecycle and audit:
``schedule_followup`` WRITES a future row, and ``manage_schedules`` sees and
governs the rows that are already there — listing is a read any task may do,
while changing one (including restoring a disabled/suppressed row) stays with
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

Authority is narrower than the parent's, not wider: a delegated subagent may not
mint future root tasks (typed refusal), the objective is the agent's own plain
text (no host template), and a task may hold at most ``_MAX_PENDING_FOLLOWUPS``
pending follow-ups — past the cap the refusal is typed and discloses the pending
records.
"""

from __future__ import annotations

from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

import json
import uuid
from typing import Any, Dict, List

from ouroboros.consciousness_authority import consciousness_origin_metadata
from ouroboros.deadline_utils import parse_deadline_ts
from ouroboros.tools.arg_feedback import ignored_argument_note
from ouroboros.tools.registry import ToolContext, ToolEntry

_MAX_PENDING_FOLLOWUPS = 2
FOLLOWUP_SOURCE = "task_followup"
_MAX_OBJECTIVE_CHARS = 4_000
_MAX_CONTEXT_CHARS = 8_000


def _manage_schedules(
    ctx: ToolContext, action: str = "list", schedule_id: str = "", reason: str = "",
    offset: int = 0, limit: int = 20,
) -> str:
    """See, or apply one narrow audited change to, the rows already in the table.

    The authority is argument-aware rather than name-wide: LISTING is a read any
    member of the tree may do, while CHANGING the owner's schedules stays with
    the root turn. A Presence conversation gets neither — an admitted guest room
    is not a place from which this mind's own schedule table is read or rewritten.
    """
    from supervisor.queue import (
        SCHEDULE_ACTIONS, ScheduleRefused, ScheduleStoreUnreadable, load_schedule_store,
        mutate_scheduled_task, schedule_tool_projection,
    )

    metadata = getattr(ctx, "task_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    operation = str(action or "list").strip().lower()
    if metadata.get("presence"):
        return _publish_tool_result(ctx, ToolResult(
            status="blocked", code="RESOURCE_CONSTRAINT_BLOCKED",
            text="⚠️ RESOURCE_CONSTRAINT_BLOCKED: a Presence conversation cannot read or change owner schedules.",
        ))
    if operation != "list" and not _root_schedule_mutation_authorized(ctx):
        return _publish_tool_result(ctx, ToolResult(
            status="blocked", code="RESOURCE_CONSTRAINT_BLOCKED",
            text=(
                "⚠️ RESOURCE_CONSTRAINT_BLOCKED: a delegated task may only read schedules "
                "(action='list'); changing one is the root turn's authority. Report the "
                f"schedule you would change instead of {operation!r}."
            ),
        ))
    root = getattr(ctx, "budget_drive_root", None) or getattr(ctx, "drive_root", None)
    try:
        from ouroboros.tool_capabilities import tool_result_limit
        result_limit = tool_result_limit("manage_schedules")
        if operation == "list":
            return json.dumps(schedule_tool_projection(load_schedule_store(root),
                                                       offset=offset, limit=limit,
                                                       result_limit=result_limit),
                              ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"))
        if operation not in SCHEDULE_ACTIONS:
            return json.dumps({"ok": False, "status": "invalid_action",
                               "allowed": ["list", *sorted(SCHEDULE_ACTIONS)]}, sort_keys=True)
        # Preserve selectors verbatim. Refuse an identity that cannot fit in a
        # truthful receipt before changing anything, rather than cutting it.
        if len(json.dumps(str(schedule_id or ""), ensure_ascii=False)) > result_limit - 2_000:
            return json.dumps({"ok": False, "changed": False, "status": "identity_too_large",
                               "audit": "not_written", "detail": "Schedule identity exceeds the tool result limit; nothing changed."})
        outcome = mutate_scheduled_task(
            operation, schedule_id, reason=reason, actor="agent",
            task_id=str(getattr(ctx, "task_id", "") or ""), drive_root=root)
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
                {"tasks": [schedule]}, limit=1, result_limit=result_limit)["tasks"][0]
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
    # ``detail`` is already bounded above and often names the actual blocker
    # (``skill_not_ready``); keep it rather than replacing the one explanation
    # the receipt carries with a generic note about its own size.
    compact.setdefault(
        "detail", "schedule action completed; full row omitted to fit the tool result limit")
    compact["schedule_omitted"] = True
    return json.dumps(compact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="schedule_followup",
            schema={
                "name": "schedule_followup",
                "description": (
                    "Register a deferred follow-up task that the supervisor scheduler "
                    "enqueues as an ordinary root task. Root tasks only; subagents report "
                    "the proposed follow-up to their parent. Supply exactly one trigger: run_at "
                    "for a one-shot ISO 8601 instant (naive = UTC), or cron for a recurring "
                    "5-field expression with an optional IANA timezone. Write the objective "
                    "in your own words — it becomes each future task's text verbatim. The "
                    "record is durable in state/scheduled_tasks.json, and the owner can "
                    "disable or delete it from the Schedules surface. A task may hold at "
                    f"most {_MAX_PENDING_FOLLOWUPS} pending follow-ups."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
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
                                "Plain-language objective for the future task, in your own words "
                                f"(max {_MAX_OBJECTIVE_CHARS} chars; longer is a typed refusal, never truncated)."
                            ),
                        },
                        "context": {
                            "type": "string",
                            "description": (
                                "Optional context the future task should start from (facts, ids, paths; "
                                f"max {_MAX_CONTEXT_CHARS} chars; longer is a typed refusal, never truncated)."
                            ),
                        },
                    },
                    "required": ["objective"],
                },
            },
            handler=_handle_schedule_followup,
            timeout_sec=30,
        ),
        ToolEntry("manage_schedules", {
            "name": "manage_schedules",
            "description": (
                "See and govern your own existing supervisor schedules. action='list' shows every "
                "row with its lifecycle: active, disabled, suppressed (a skill row you stopped) or "
                "consumed (a one-shot that already fired). disable/delete stop FUTURE dispatch — a "
                "task already admitted from the schedule keeps running — and restore brings a "
                "suppressed or disabled row back, re-checking the skill rather than enabling it. "
                "Every change needs a concrete reason and is audited. A consumed one-shot is "
                "history: schedule a new run_at instead of trying to re-arm it. Deleting a skill "
                "row keeps it as a suppressed record so the skill lifecycle cannot resurrect it. "
"A kind='notify' row is a skill's model-free reminder (its sentence is the preview): "
"the owner's disable or delete keeps it suppressed so the skill cannot re-arm that key, "
"and a second delete of the suppressed record removes it. "
                "Only your own root turn may change a schedule; a delegated task may only list. "
                "list accepts offset and limit (bounded pages) and returns total/next_offset; "
                "each row contains only a bounded objective preview, never full context."
            ),
            "parameters": {"type": "object", "properties": {
                "action": {"type": "string", "enum": ["list", "disable", "delete", "restore"]},
                "schedule_id": {"type": "string", "description": "Schedule id for a change."},
                "reason": {"type": "string", "description": "Why this change is needed."},
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
        template = record.get("task") if isinstance(record.get("task"), dict) else {}
        metadata = template.get("metadata") if isinstance(template.get("metadata"), dict) else {}
        if str(metadata.get("origin_task_id") or "") == task_id:
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
            _publish_tool_result(ctx, ToolResult(status="blocked", code="ACCESS_BLOCKED", text=("ERROR: FOLLOWUP_SUBAGENT_REFUSED: a delegated subagent holds narrower-than-parent "
            "authority and may not mint future root tasks. Report the wait instant to your "
            "parent instead; the parent (or the owner) decides whether to schedule a follow-up.")))
        )
    task_id = str(getattr(ctx, "task_id", "") or "").strip()
    if not task_id:
        return _publish_tool_result(ctx, ToolResult(status="unavailable", code="CAPABILITY_UNAVAILABLE", text=("ERROR: FOLLOWUP_TASK_ID_REQUIRED: a durable follow-up must belong to a real task.")))
    run_at_raw = str(params.get("run_at") or "").strip()
    cron = str(params.get("cron") or "").strip()
    if bool(run_at_raw) == bool(cron):
        return (
            _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=("ERROR: FOLLOWUP_TRIGGER_REQUIRED: supply exactly one of run_at (one-shot) "
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
                return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(
                    f"ERROR: FOLLOWUP_TIMEZONE_WITH_RUN_AT: run_at={run_at_raw!r} carries no UTC offset and "
                    f"timezone={timezone!r} applies only to recurring cron follow-ups. Put the offset into run_at "
                    "(example: 2026-08-19T12:20:00+03:00) and omit timezone. Nothing was scheduled.")))
            # run_at with its own offset is an absolute instant: a zone beside it asks for nothing.
            timezone_note = " " + ignored_argument_note(
                "timezone", timezone, "it applies only to recurring cron follow-ups; run_at carries its own offset") + "."
            timezone = ""
        if instant is None:
            return (
                _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(f"ERROR: FOLLOWUP_RUN_AT_INVALID: {run_at_raw!r} is not a parseable ISO 8601 "
                "instant. Example: 2026-08-19T12:20:00+03:00 (naive times read as UTC).")))
            )
        trigger = {"type": "once", "run_at": instant.isoformat()}
    else:
        from ouroboros.schedule_contract import cron_error, timezone_error

        if error := cron_error(cron):
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(f"ERROR: FOLLOWUP_CRON_INVALID: {error}")))
        if error := timezone_error(timezone):
            return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(f"ERROR: FOLLOWUP_TIMEZONE_INVALID: {error}")))
        trigger = {"type": "cron", "expr": cron}
    objective = str(params.get("objective") or "").strip()
    if not objective:
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=("ERROR: FOLLOWUP_OBJECTIVE_REQUIRED: write the future task's objective in plain language.")))
    # Typed refusal, never a silent cut: the text rides VERBATIM into the future
    # task, so truncating it here would silently change what that task is.
    if len(objective) > _MAX_OBJECTIVE_CHARS:
        return (
            _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(f"ERROR: FOLLOWUP_TEXT_TOO_LONG: objective is {len(objective)} chars; the limit is "
            f"{_MAX_OBJECTIVE_CHARS}. Shorten it — nothing was truncated and nothing was scheduled.")))
        )
    context = str(params.get("context") or "").strip()
    if len(context) > _MAX_CONTEXT_CHARS:
        return (
            _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ARG_ERROR", text=(f"ERROR: FOLLOWUP_TEXT_TOO_LONG: context is {len(context)} chars; the limit is "
            f"{_MAX_CONTEXT_CHARS}. Shorten it — nothing was truncated and nothing was scheduled.")))
        )
    from ouroboros.tool_access import canonical_data_root
    from supervisor.queue import schedule_transaction

    try:
        drive_root = canonical_data_root(ctx)
    except Exception as exc:
        return _publish_tool_result(ctx, ToolResult(status="error", code="TOOL_ERROR", text=(f"ERROR: FOLLOWUP_DATA_ROOT_UNRESOLVED: {exc}")))
    # The cap is read from the table this call is about to write, so the count
    # and the write share ONE transaction: two tasks registering at once would
    # otherwise each read the same under-cap count and both land.
    from supervisor.queue import ScheduleStoreUnreadable

    try:
        with schedule_transaction(drive_root):
            return _register_followup(ctx, task_id, drive_root, objective, context,
                                      trigger, cron, timezone, timezone_note)
    except ScheduleStoreUnreadable as exc:
        # A missed table lock (ScheduleLockTimeout) or an unparseable table: the
        # follow-up was NOT registered, and the refusal says so in the text ABI
        # instead of surfacing as a generic tool error.
        return _publish_tool_result(ctx, ToolResult(
            status="unavailable", code="CAPABILITY_UNAVAILABLE",
            text=f"⚠️ CAPABILITY_UNAVAILABLE: FOLLOWUP_STORE_UNAVAILABLE: {exc}"))


def _register_followup(ctx: ToolContext, task_id: str, drive_root: Any,
                       objective: str, context: str, trigger: Dict[str, Any], cron: str,
                       timezone: str, timezone_note: str) -> str:
    """The cap read, the record build and the write — all under the schedule lock."""
    from supervisor.queue import list_scheduled_tasks, upsert_scheduled_task

    records = [r for r in (list_scheduled_tasks(drive_root).get("tasks") or []) if isinstance(r, dict)]
    pending = _pending_followups(records, task_id)
    if len(pending) >= _MAX_PENDING_FOLLOWUPS:
        def _trigger_label(record: Dict[str, Any]) -> str:
            item = record.get("trigger") if isinstance(record.get("trigger"), dict) else {}
            if str(item.get("type") or "") == "cron":
                zone = str(record.get("timezone") or "").strip() or "system local time"
                return f"cron {item.get('expr')} ({zone})"
            return f"fires at/after {item.get('run_at')}"

        listing = "; ".join(
            f"{record.get('id')} ({_trigger_label(record)})" for record in pending
        )
        return (
            _publish_tool_result(ctx, ToolResult(status="blocked", code="RESOURCE_CONSTRAINT_BLOCKED", text=(f"ERROR: FOLLOWUP_CAP_REACHED: this task already holds {len(pending)} pending "
            f"follow-up(s) of the {_MAX_PENDING_FOLLOWUPS} allowed: {listing}. Wait for a "
            "one-shot to fire, or the owner can disable/delete records from the Schedules surface.")))
        )
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
        "name": f"Follow-up of task {task_id}",
        "description": objective,
        "source": FOLLOWUP_SOURCE,
        "enabled": True,
        "timezone": timezone,
        "trigger": trigger,
        "task": {
            "type": "task",
            "text": objective,
            "description": objective,
            **({"context": context} if context else {}),
            **({"project_id": project_id} if project_id else {}),
            "metadata": {
                "source": FOLLOWUP_SOURCE,
                "origin_task_id": task_id,
                # `or ""` before str(): an absent/None root_task_id must fall back
                # to task_id, never become the literal string "None".
                "origin_root_task_id": str(root_task_id or "") or task_id,
            },
            **({"chat_id": source_chat_id} if source_chat_id not in (None, "") else {}),
        },
    }
    # A follow-up from a consciousness turn/tree starts a consciousness root: the
    # origin, category and level ride the template; admission derives the rest.
    record["task"]["metadata"].update(consciousness_origin_metadata(metadata_src))
    presence = metadata_src.get("presence") if isinstance(metadata_src, dict) else None
    contract = getattr(ctx, "task_contract", None)
    if isinstance(presence, dict) and presence and isinstance(contract, dict):
        record["task"]["metadata"]["presence"] = dict(presence)
        record["task"]["task_contract"] = dict(contract)
    from supervisor.queue import ScheduleRefused, ScheduleStoreUnreadable

    try:
        stored = upsert_scheduled_task(
            record, drive_root=drive_root, actor="agent:schedule_followup", task_id=task_id,
            reason=f"follow-up registered by task {task_id}")
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
        f"FOLLOWUP_SCHEDULED: follow-up {stored.get('id')} registered as {timing}. It will "
        "enqueue ordinary root tasks through the supervisor scheduler under normal admission; "
        f"pending follow-ups for this task: {len(pending) + 1}/{_MAX_PENDING_FOLLOWUPS}. The "
        f"record is durable in state/scheduled_tasks.json and {lifecycle}; the owner can "
        f"disable or delete it from the Schedules surface.{audit_note}{timezone_note}"
    )
