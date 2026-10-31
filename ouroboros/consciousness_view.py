"""Read-only consciousness facts from places, current roots and durable results."""

import contextlib
from datetime import datetime
import json
import time
from typing import Any

from ouroboros.usage_origin import cause_label

RECENT_WAKE_LIMIT = 50
_DISPLAY_ZONE = None  # the server's local zone; the browser fixture generator pins one


@contextlib.contextmanager
def _reading(store_name: str):
    """Name the store of a ``StoreBusy`` leaving this read on the same exception, so its
    reason, cause and rollback facts travel unchanged; an inner read's name stands.
    Helpers both stores share raise identical messages, so only the reader knows the store."""
    from ouroboros.store import StoreBusy

    try:
        yield
    except StoreBusy as exc:
        exc.store_name = getattr(exc, "store_name", "") or store_name
        raise


def _clock_of(iso_value: Any) -> str:
    """Server-local clock, with a short date unless today (``?`` when absent)."""
    from ouroboros.deadline_utils import parse_deadline_ts

    parsed = parse_deadline_ts(str(iso_value or ""))
    if parsed is None:
        return "?"
    at = parsed.astimezone(_DISPLAY_ZONE)
    today = datetime.fromtimestamp(time.time()).astimezone(_DISPLAY_ZONE).date()
    date = f"{at:%b} {at.day} " if at.date() != today else ""
    return date + at.strftime("%H:%M")


def _when(iso_value: Any) -> str:
    """«at HH:MM» for a coming instant, «due since HH:MM» for a passed one: no line names a past time as next."""
    from ouroboros.deadline_utils import parse_deadline_ts

    parsed = parse_deadline_ts(str(iso_value or ""))
    return ("due since " if parsed is not None and parsed.timestamp() <= time.time() else "at ") + _clock_of(iso_value)


def describe_consciousness_state(snapshot: dict, requested_enabled: bool | None) -> dict:
    """Project the alarm clock's snapshot into one honest status + detail line
    for the client headline (``None``: stored control is unknown, never off — #1307).
    The appointment is the scheduled time; the next attempt is where skips and refusals moved it."""
    from ouroboros.deadline_utils import parse_deadline_ts

    outcome = str(snapshot.get("last_wake_outcome") or "")
    next_at = _when(snapshot.get("next_wake_at"))
    deadline = parse_deadline_ts(str(snapshot.get("next_wake_at") or ""))
    wake = snapshot.get("live_wake_task_id")
    if requested_enabled is None:
        status, detail = "unknown", "The stored consciousness control is unknown: runtime state is unavailable or recovering."
    elif not requested_enabled:
        status, detail = "disabled", "Background consciousness is off."
    elif not snapshot:
        status, detail = "stopped", "Enabled in state, but the alarm clock was not constructed (supervisor init failed)."
    elif wake and snapshot.get("live_wake_held"):
        status, detail = "wake_paused", (f"Wake-up {wake} is paused or waiting in the queue; its card shows why. "
                                         "The next wake-up follows it.")
    elif wake:
        status, detail = "thinking", f"Wake-up {wake} is running."
    elif outcome == "skipped:waiting_for_first_conversation":
        status, detail = "waiting_for_first_conversation", "No owner chat is bound yet; the first conversation binds it."
    elif outcome == "skipped:allowance_exhausted":
        spent, daily = snapshot.get("spent_24h_usd"), snapshot.get("daily_usd")
        status = "allowance_exhausted"
        at_least = "at least " if int(snapshot.get("unknown_unmetered") or 0) > 0 else ""
        degraded = "; ledger integrity degraded" if snapshot.get("integrity_degraded") else ""
        detail = (f"Daily allowance spent ({at_least}${float(spent or 0):.2f} of ${float(daily or 0):.2f} in the last 24 h{degraded}); "
                  f"next check {_when(snapshot.get('allowance_resets_at')) if snapshot.get('allowance_resets_at') else next_at}.")
    elif outcome == "skipped:allowance_unknown":
        status, detail = "allowance_unknown", f"The usage ledger could not be read ({snapshot.get('last_error') or 'unknown error'}); retry {next_at}."
    elif outcome.startswith("rejected:"):
        status, detail = "wake_rejected", f"The last wake-up was refused ({outcome.split(':', 1)[1]}); next attempt {next_at}."
    elif outcome in {"paused", "pausing"}:
        status = "wake_paused"
        detail = (f"The last wake-up returned while {outcome}; its task card shows the current state. "
                  f"Next wake check {next_at}.")
    elif outcome == "unknown":
        status, detail = "wake_outcome_unknown", f"The last wake-up's outcome is unconfirmed; next check {next_at}."
    elif outcome == "failed":
        status, detail = "wake_failed", f"The last wake-up failed ({snapshot.get('last_error') or 'runner error'}); next attempt {next_at}, backing off."
    elif deadline is not None and deadline.timestamp() <= time.time():
        # Only the scheduled time says since when; a legacy state's retry time cannot.
        since = f" since {_clock_of(snapshot['appointment_at'])}" if snapshot.get("appointment_at") else ""
        status, detail = "due", f"Due{since}; the wake-up has not started yet."
    else:
        status, detail = "sleeping", f"Sleeping until {_clock_of(snapshot.get('next_wake_at'))}."
    return {**snapshot, "enabled": requested_enabled, "status": status, "detail": detail}



def describe_consciousness(clock) -> dict:
    """Read the same stored control as /api/state, including unknown and no-clock boot states."""
    from supervisor.state import control_value, load_state

    known, enabled = control_value(load_state(), "bg_consciousness_enabled")
    requested = bool(enabled) if known else None
    if clock is not None:
        return clock.details_snapshot(requested)
    return {**describe_consciousness_state({}, requested), "recent_wakes": []}


def place_views(drive_root, allowance=None) -> list[dict]:
    """Stored PlaceSummary facts, using the caller's allowance without connector discovery."""
    from ouroboros.place_projection import place_view
    from ouroboros.places import load_places

    return [place_view(drive_root, place, allowance=allowance, brief=True) for place in load_places(drive_root)]


def places_text(drive_root, views) -> str:
    from ouroboros.place_arrivals import recent_identity
    from ouroboros.places import OwnerAccount, owner_account_matches
    if views is None:
        return "unknown (place index unavailable)"
    lines = []
    for view in views:
        identity = recent_identity(drive_root, view["place_id"])
        sending = (identity or {}).get("sends_as") or {}
        owner = sending.get("kind") == "user" and owner_account_matches(
            (OwnerAccount(**a) for a in view.get("owner_accounts", [])), sending["id"], view.get("account_id", ""))
        voice = ""
        if sending:
            voice = (f'; as Ouroboros (bot {sending["label"]})' if sending["kind"] == "bot"
                     else f'; writes as {sending["label"]}' + (" (your human's own account)" if owner else ""))
        count = view.get("unanswered")
        addresses = [str(room.get("address") or f'chat_history(chat_id={int(room["room_chat_id"])})')
                     for room in view.get("rooms") or []]
        address = "; ".join(addresses) or ("no captured room" if view.get("rooms") == [] else "room address unknown")
        lines.append(f'{view.get("label") or view["place_id"]} [{view["place_id"]}]: {view["state"]}; '
                     f'open turns: {count if count is not None else "unknown"}; {address}{voice}')
        for note in view.get("notes") or []:
            lines.extend(f'  note ({note["author"]}): {line}' for line in note["text"].splitlines())
    return "\n  ".join(lines) if lines else "none"


def cause_views(rows, places) -> list[dict] | None:
    if rows is None:
        return None
    return [{**row, "label": cause_label(row["cause"], places or ())} for row in rows]


def root_views(roots, places) -> list[dict]:
    return [{**root, "cause": root["consciousness_cause"],
             "cause_label": cause_label(root["consciousness_cause"], places or ())} for root in roots]


def spend_text(rows) -> str:
    if rows is None:
        return "unknown"
    return "; ".join(f'{row["label"]}: {"at least " if row["unmetered_calls"] else ""}'
                     f'${row["usd"]:.2f} ({row["unmetered_calls"]} unmetered calls)' for row in rows) or "none"


def roots_text(rows) -> str:
    if rows is None:
        return "unknown"
    return "; ".join(f'{row["task_id"]}: {row["title"]} ({row["cause_label"]})' for row in rows) or "none"


def first_reply_at(drive_root, task_id: str) -> str | None:
    """Earliest confirmed receipt or automatic release, using this task's dialogue index.

    Authored mode-0 replies prove release, not provider delivery. Failed/uncertain
    receipts, drafts and task summaries prove neither. Read pages, never a global log.
    """
    from ouroboros import dialogue_store
    from ouroboros.deadline_utils import parse_deadline_ts

    first = None
    with _reading("dialogue"):  # the lazy iterator reads at its first step and at every page
        for stored in dialogue_store.iter_probes(drive_root, "chat", [("task_id", task_id)]):
            row = stored.row
            transport = row.get("transport") or {}
            delivery = transport.get("delivery") or {}
            receipt = row.get("type") == "presence_delivery" and delivery.get("state") in {"accepted", "delivered"}
            released = (row.get("direction") == "out" and str(row.get("source") or "").startswith("presence:")
                        and delivery.get("state") == "authored")
            if row.get("task_id") != task_id or not (receipt or released):
                continue
            stamp = str(delivery.get("reported_at") or row.get("ts") or "")
            instant = parse_deadline_ts(stamp)
            if instant is not None and (first is None or instant < first[0]):
                first = instant, stamp
    return first[1] if first else None


def wake_ids(drive_root, now: float) -> list[str]:
    """Newest distinct wake roots off the allowance's category/time index.

    Its own wakes and the places' each keep half of the page; the half one side leaves unused goes
    to the other, so a flood of place wakes never hides its own (a cut always fills the page). The
    stamped cause decides the side; an unstamped legacy root is its own, as in the allowance.
    """
    from ouroboros import usage_store
    from ouroboros.consciousness_allowance import ROOT_HORIZON_SEC, _MONEY_KIND

    with usage_store.read(drive_root, allow_stale=True) as txn:
        rows = txn.conn.execute(
            "SELECT root_task_id, place FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY place "
            "ORDER BY last DESC, root_task_id DESC) AS n FROM (SELECT root_task_id, MAX(ts_last_epoch) AS last, "
            "COALESCE(MAX(json_extract(extra, '$.consciousness_cause')), '') LIKE 'place:%' AS place "
            "FROM attempts INDEXED BY attempts_category_time "
            "WHERE category = 'consciousness' AND ts_last_epoch >= ? AND root_task_id != '' "
            f"AND {_MONEY_KIND} AND state != 'released' AND (ts_dispatched IS NOT NULL "
            "OR kind IN ('subscription_session', 'external_unmetered')) GROUP BY root_task_id)) "
            f"WHERE n <= {RECENT_WAKE_LIMIT} ORDER BY last DESC, root_task_id DESC",
            (now - ROOT_HORIZON_SEC,)).fetchall()
    sides = [[root for root, place in rows if place == side] for side in (0, 1)]
    keep = [max(RECENT_WAKE_LIMIT // 2, RECENT_WAKE_LIMIT - len(other)) for other in reversed(sides)]
    kept = {root for side, room in zip(sides, keep) for root in side[:room]}
    return [root for root, _place in rows if root in kept]


def _wake_notes(drive_root, ids: list[str]) -> dict:
    """One PK-addressed read of the whole results excluded from display documents."""
    from ouroboros.store import open_store

    with open_store(drive_root, write=False) as txn:
        return dict(txn.conn.execute(
            "SELECT task_id, json_extract(body, '$.result') FROM task_bodies "
            "WHERE task_id IN (SELECT value FROM json_each(?))", (json.dumps(ids),)))


def _wake_phase(row: dict, clock_live: bool) -> str:
    """Terminal card outcome, or the existing work classifier over stored activity facts."""
    from ouroboros.gateway.work_status import classify_member
    from ouroboros.project_dialogue import outcome_phase, owner_wait_projection
    from ouroboros.task_status import FINAL_STATUSES
    from ouroboros.work_stop import stop_presentation

    status = str(row.get("status") or "")
    if status in FINAL_STATUSES:
        return outcome_phase(row, {})
    phase = {"running": "working", "requested": "queued", "scheduled": "queued",
             "queued": "queued", "paused": "budget_paused"}.get(status, "working" if clock_live else "unknown")
    stop = stop_presentation(row)
    if stop["stop"] or stop["waiting"]:
        phase = "budget_paused"
    facts = {**row, "activity_id": row["task_id"], "phase": phase}
    wait = facts.pop("owner_wait", None) or {}
    if wait.get("quiz_id"):
        quiz = (row.get("owner_quiz") or {}).get(wait["quiz_id"]) or {}
        facts["owner_wait"] = {**owner_wait_projection(wait["quiz_id"], wait, quiz),
                               "quiz_id": wait["quiz_id"], "quiz_state": quiz.get("state", "unknown")}
    activity = classify_member(facts, row)["activity"]
    return "running" if activity == "working" else activity


def recent_wakes(drive_root, places, *, live_wake_task_id: str = "", now: float | None = None,
                 ids: list[str] | None = None) -> list[dict]:
    """At most 50 wakes that reached a model call in the last 48 h, with room for both kinds, newest first.

    Wakes that never reached a model call are not listed. Selection uses the
    allowance's ledger horizon; reservations alone are not calls. Facts come
    from display documents, whole notes from one PK read, focus replies from
    the dialogue task index. Missing historical evidence stays unknown.
    """
    from ouroboros.consciousness_authority import wake_cause
    from ouroboros.task_results import TASK_COST_META_FIELDS
    from ouroboros.task_status import FINAL_STATUSES
    from ouroboros.task_store import display_documents
    from ouroboros.terminal_time import terminal_time_fact
    from ouroboros.task_finalization import quiet_finish

    if ids is None:
        ids = wake_ids(drive_root, time.time() if now is None else now)
    if not ids:
        return []
    with _reading("task"):
        documents = display_documents(drive_root, ids)
        notes = _wake_notes(drive_root, ids)
    rows = []
    for task_id in ids:
        if task_id not in documents:
            continue
        row = json.loads(documents[task_id])
        metadata = row.get("metadata") or {}
        # Category-only legacy roots use the allowance's own-clock fallback.
        cause = "focus" if wake_cause(metadata) == "focus" else "own"
        focus = metadata.get("focus") or {}
        sending = (focus.get("identity") or {}).get("sends_as") if cause == "focus" else None
        place_id = str(focus.get("place_id") or "")
        usage = (row.get("loop_outcome") or {}).get("usage") or {}
        rows.append({"task_id": task_id, "cause": cause,
                     **({"as": {"label": sending["label"], "owner": bool(focus.get("identity_owner"))}} if sending else {}),
                     "chat_id": row.get("chat_id"), "conversation": str(focus.get("conversation_key") or ""),
                     "place_label": cause_label(f"place:{place_id}", places or ()) if place_id else "",
                     "started_at": row.get("started_at"),
                     **({"first_reply_at": first_reply_at(drive_root, task_id)} if cause == "focus" else {}),
                     "ended_at": terminal_time_fact(row)["occurred_at"],
                     "live": live_wake_task_id == task_id or row.get("status") not in FINAL_STATUSES,
                     "phase": _wake_phase(row, live_wake_task_id == task_id), "quiet": quiet_finish(row),
                     **({"tool_calls": row["tool_calls"]} if "tool_calls" in row else {}),
                     **({"rounds": usage["total_rounds"]} if "total_rounds" in usage else {}),
                     "checks": row.get("safety_checks", []), "note": notes.get(task_id),
                     **({"summary": row["wake_summary"]} if isinstance(row.get("wake_summary"), str) and row["wake_summary"] else {}),
                     **({"said_owner": bool(row["owner_message_sent"])} if "owner_message_sent" in row else {}),
                     **{k: row[k] for k in TASK_COST_META_FIELDS if k in row}})
    return rows
