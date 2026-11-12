"""Read-only consciousness facts from places, current roots and durable results."""

import contextlib
from datetime import datetime
import json
import logging
import time
from typing import Any

from ouroboros.usage_origin import cause_label
from ouroboros.text_sentences import first_sentence

log = logging.getLogger(__name__)
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


def describe_consciousness_state(snapshot: dict, requested_enabled: bool | None, *, supervisor_failed: bool = False) -> dict:
    """Project the alarm clock's snapshot into one honest status + detail line
    for the client headline (``None``: stored control is unknown, never off — #1307).
    The appointment is the scheduled time; the next attempt is where skips and refusals moved it.
    Without a clock the server says why (``supervisor_failed``): a supervisor that failed to start
    cannot wake it until a restart; otherwise Ouroboros is still starting (or waits for a provider)."""
    from ouroboros.deadline_utils import parse_deadline_ts

    outcome = str(snapshot.get("last_wake_outcome") or "")
    note = ""  # what follows from the state, said beneath it (``detail_note``); only a proven allowance hold has one
    next_at = _when(snapshot.get("next_wake_at"))
    deadline = parse_deadline_ts(str(snapshot.get("next_wake_at") or ""))
    wake = snapshot.get("live_wake_task_id")
    if requested_enabled is None:
        status, detail = "unknown", "The stored consciousness control is unknown: runtime state is unavailable or recovering."
    elif not requested_enabled:  # the places it already read (a Paused one does not hear either); unread, as Stop says
        hearing = sum(row.get("state") != "paused" for row in snapshot.get("places_brief") or [])
        unread = "places_brief" in snapshot and snapshot["places_brief"] is None
        status, detail = "disabled", (f"Off: no own wakes; {_hearing(hearing)}." if hearing else
                                      "Off: no own wakes; its places only hear." if unread else "Off: no own wakes.")
    elif not snapshot and supervisor_failed:
        status, detail = "stopped", "Cannot wake: Ouroboros did not finish starting. Restart Ouroboros."
    elif not snapshot:
        status, detail = "starting", "Waits for Ouroboros to finish starting."
    elif wake and snapshot.get("live_wake_held"):
        status, (detail, note) = "wake_paused", held_sentence(wake, snapshot.get("live_wake_hold"))
    elif wake:
        status, detail = "thinking", f"Wake-up {wake} is running."
    elif outcome == "skipped:waiting_for_first_conversation":
        status, detail = "waiting_for_first_conversation", "No owner chat is bound yet; the first conversation binds it."
    elif outcome == "skipped:allowance_exhausted":
        spent, daily = snapshot.get("spent_24h_usd"), snapshot.get("daily_usd")
        status = "allowance_exhausted"
        at_least = "at least " if int(snapshot.get("unknown_unmetered") or 0) > 0 else ""
        degraded = "; ledger integrity degraded" if snapshot.get("integrity_degraded") else ""
        # Checked at the reset, or sooner once a changed daily limit moved the next attempt (``consciousness._due_at``).
        reset = parse_deadline_ts(str(snapshot.get("allowance_resets_at") or ""))
        sooner = deadline is not None and reset is not None and time.time() < deadline.timestamp() < reset.timestamp()
        check = _when(snapshot.get("allowance_resets_at")) if snapshot.get("allowance_resets_at") and not sooner else next_at
        # A later window read that failed leaves the amount unknown now, never $0; the skip stays what happened.
        counted = (f"({at_least}${float(spent):.2f} of ${float(daily or 0):.2f} in the last 24 h{degraded})"
                   if type(spent) in (int, float) else "at its last check; what it spent in the last 24 h is unknown now")
        detail = f"Daily allowance spent {counted}; next check {check}."
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
    return {**{key: value for key, value in snapshot.items() if key != "detail_note"},
            "enabled": requested_enabled, "status": status, "detail": detail, **({"detail_note": note} if note else {})}


def _hearing(n: int) -> str:
    return f"{n} place only hears" if n == 1 else f"{n} places only hear"


# A held wake the allowance it started with stopped: the owner's move (DESIGN "Status marks"), said in Costs too.
ALLOWANCE_HOLDS = frozenset({"allowance", "allowance_and_cap"})
_STAMPED_CAUSES = {"allowance": "allowance", "both": "allowance_and_cap", "per_task_cap": "per_task_cap"}


def held_sentence(wake: str, hold: Any) -> tuple:
    """``(detail, note)`` of a held own wake: its proven cause (``hold_cause``) or the card's word for it, and what
    follows from an allowance hold (``""`` otherwise). No clock: when room returns is Costs' forecast, in one zone."""
    hold = hold if isinstance(hold, dict) else {}
    cause, ceiling = hold.get("cause"), hold.get("ceiling_usd")
    if cause == "owner":
        return f"Wake-up {wake} is on an explicit pause; its card shows who paused it and how to continue.", ""
    if cause in ALLOWANCE_HOLDS:
        facts = [f"${float(ceiling):.2f}"] if type(ceiling) in (int, float) else []
        facts += ["the per-task cap too"] if cause == "allowance_and_cap" else []
        said = f" ({', '.join(facts)})" if facts else ""
        return (f"Wake-up {wake} stopped at the allowance it had left when it started{said}.",
                "Resume keeps that limit, so it pauses again while its known spend is at it, and a higher allowance "
                "does not raise it. Stop now on its card ends it and frees the clock for its next check.")
    # The limit it launched with, as far as its stamp names it (an older record names none).
    limit = {"per_task_cap": "the per-task cost cap", "launch_ceiling": "the spending limit"}.get(cause)
    said = f"paused at {limit} it started with" if limit else "paused or waiting in the queue"
    return f"Wake-up {wake} is {said}; its card shows why. The next wake-up follows it.", ""


def _at_launch_ceiling(pause: dict, ceiling, basis: str) -> bool:
    """A live budget pause at the graceful ceiling the wake launched with: its saved ceiling is that number under the
    producer's own basis (a member's inherited one), or, on a row older than the saved ceiling, the rail alone."""
    import math
    from ouroboros.budget_pause import RAIL_GRACEFUL_CEILING, STATE_PAUSED, STATE_PAUSING

    if (pause.get("state") not in (STATE_PAUSED, STATE_PAUSING) or pause.get("reason") != "budget"
            or pause.get("rail") != RAIL_GRACEFUL_CEILING or ceiling is None):
        return False
    saved = pause.get("cost_ceiling")
    if not isinstance(saved, dict):
        return True
    number = saved.get("ceiling_usd")
    return (saved.get("basis") == basis and type(number) in (int, float)
            and math.isclose(float(number), ceiling, rel_tol=0.0, abs_tol=1e-9))


def hold_cause(metadata: dict, own: dict, members=()) -> dict:
    """``{"cause", "ceiling_usd"}``: what holds my own held wake, only as far as its records prove it (pure).

    ``own`` is the wake's pause row, ``members`` the pause rows of its paused tree members. The wake paused at the
    ceiling it launched with (``root_cost_ceiling_usd``), or, child first, its tree's latch refused it while a member
    holding that inherited number paused at it: the launch stamp (``consciousness_wake.wake_task_metadata``) names
    ``allowance``, ``allowance_and_cap`` or ``per_task_cap``, and a record from before the stamp is only
    ``launch_ceiling``. Otherwise the pause's own reason or rail: ``owner``, ``sleep``, ``technical``,
    ``total_budget``, ``per_task_fence``. ``""`` proves nothing: no live pause, a Resume granted, a lifted allowance."""
    from ouroboros.budget_pause import (RAIL_DISPATCH_REFUSED, RAIL_GLOBAL_EXHAUSTED, REASON_AUTHORITY, REASON_OWNER,
                                        REASON_SLEEP, STATE_PAUSED, STATE_PAUSING)
    from ouroboros.task_pacing import recorded_producer_ceiling

    metadata = metadata if isinstance(metadata, dict) else {}
    own = own if isinstance(own, dict) else {}
    unproven = {"cause": "", "ceiling_usd": None}
    if own.get("state") not in (STATE_PAUSED, STATE_PAUSING):
        return unproven
    reason, rail = own.get("reason"), own.get("rail")
    other = {REASON_OWNER: "owner", REASON_SLEEP: "sleep", REASON_AUTHORITY: "technical"}.get(reason)
    if other or reason != "budget":
        return {"cause": other or "", "ceiling_usd": None}
    lifted = isinstance(metadata.get("producer_allowance_lifted"), dict)
    ceiling = None if lifted else recorded_producer_ceiling({"metadata": metadata})
    refused_root = rail == RAIL_DISPATCH_REFUSED and own.get("scope") == "root"
    tree = refused_root and any(_at_launch_ceiling(member, ceiling, "root_resolved_ceiling")
                                for member in members if isinstance(member, dict))
    if _at_launch_ceiling(own, ceiling, "producer_allowance") or tree:
        return {"cause": _STAMPED_CAUSES.get(metadata.get("root_cost_ceiling_source"), "launch_ceiling"),
                "ceiling_usd": ceiling}
    if rail == RAIL_GLOBAL_EXHAUSTED or rail == RAIL_DISPATCH_REFUSED and not refused_root:
        return {"cause": "total_budget", "ceiling_usd": None}
    return {"cause": "per_task_fence" if refused_root else "", "ceiling_usd": None}


def held_wake_cause(drive_root, task_id: str) -> dict:
    """``hold_cause`` of my held own wake from its records: its pool row's metadata (memory), its pause row and, only
    when its tree's latch refused it, its paused members' rows (one compact hot-facts read each, no task body). An
    unreadable record proves nothing."""
    from ouroboros.budget_pause import RAIL_DISPATCH_REFUSED, budget_pause_row
    from ouroboros.owner_pause import fence_closed
    from ouroboros.store_hot_facts import controls
    from supervisor.consciousness_carriers import live_task_rows, paused_tree_members

    try:
        metadata = (live_task_rows([task_id]).get(task_id) or {}).get("metadata") or {}
        facts = controls(drive_root, task_id)
        fence = facts.get("owner_pause") or {}
        if fence_closed(fence) and task_id not in (fence.get("selected_members") or {}):
            return {"cause": "owner", "ceiling_usd": None}
        own = facts.get("budget_pause") or {}
        members = ([budget_pause_row(root or drive_root, member) for member, root in paused_tree_members(task_id)]
                   if own.get("rail") == RAIL_DISPATCH_REFUSED and own.get("scope") == "root" else [])
    except Exception:
        log.debug("The pause of held wake %s is unreadable; its cause stays unknown", task_id, exc_info=True)
        return {"cause": "", "ceiling_usd": None}
    return hold_cause(metadata, own, members)


def off_sentence(places) -> str:
    """The switch's first words as it turns consciousness off (owner 09.11, variant A). ``places`` is one read of the
    place store, None when it could not be read; a Paused place does not hear either, so it is not counted."""
    if places is None:
        return "Consciousness off: no own wakes, and its places only hear until Start."
    hearing = sum(place.state != "paused" for place in places)
    return f"Consciousness off: no own wakes, and {_hearing(hearing)} until Start." if hearing else "Consciousness off: no own wakes."


def stopping_words(task_ids, *, wake: str, rows: dict, authors, places) -> str:
    """«Stopping now: …» naming each live item a stop ends, from what is in memory: its own wake-up, a reply in a
    place (named from the place store read beside it, else «a place»), an autonomous task by its queue title, else
    its id. ``rows`` maps a task id to its in-memory ``title`` and ``metadata``; ``authors`` are live place authors."""
    from ouroboros.consciousness_authority import is_focus_wake, is_own_clock_wake
    from ouroboros.place_projection import said_once

    named: dict[str, str] = {}
    for place in places or ():
        for key in (place.place_id, *place.legacy_binding_ids):
            named.setdefault(key, said_once(place.label or place.provider))
    woke, replies, tasks = False, [], []
    for task_id in task_ids:
        row = rows.get(task_id) or {}
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        focus, presence = (meta.get(key) if isinstance(meta.get(key), dict) else {} for key in ("focus", "presence"))
        if task_id == wake or is_own_clock_wake(meta):
            woke = True
        elif task_id in authors or is_focus_wake(meta) or presence:
            cause = str(meta.get("consciousness_cause") or "")
            key = str(focus.get("place_id") or (cause.removeprefix("place:") if cause.startswith("place:") else "")
                      or presence.get("binding_id") or "")
            replies.append(f"a reply in {named.get(key) or 'a place'}")
        else:
            tasks.append(f"«{row.get('title') or task_id}»")
    items = (["its wake-up"] if woke else []) + replies + tasks
    if not items:
        return ""
    return "Stopping now: " + (items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}") + "."


def describe_consciousness(clock, *, supervisor_failed: bool = False) -> dict:
    """Read the same stored control as /api/state, including unknown and no-clock boot states."""
    from supervisor.state import control_value, load_state

    known, enabled = control_value(load_state(), "bg_consciousness_enabled")
    requested = bool(enabled) if known else None
    if clock is not None:
        return clock.details_snapshot(requested)
    return {**describe_consciousness_state({}, requested, supervisor_failed=supervisor_failed), "recent_wakes": []}


def place_views(drive_root, allowance=None) -> list[dict]:
    """Stored PlaceSummary facts with each place's circle (one store read), the caller's allowance and one read of the
    switch, no connector discovery."""
    from ouroboros.place_projection import consciousness_off, place_view
    from ouroboros.places import place_records

    records = place_records(drive_root)
    off = consciousness_off(drive_root)
    return [place_view(drive_root, place, allowance=allowance, brief=True, records=records, off=off)
            for place in records.places.values()]


def places_text(drive_root, views) -> str:
    """The places as my own map: each circle once with its whole words and its places under it, then the places in
    no circle. A place I put in a circle says so: there my human's words still restrict, but consent needs their
    placement."""
    from ouroboros.places import PlaceError, place_owner_accounts
    if views is None:
        return "unknown (place index unavailable)"
    try:
        counted = place_owner_accounts(drive_root)  # one read: the accounts each place's connector account confirms
    except PlaceError:
        counted = {}
    groups: dict[str, list] = {}
    for view in views:
        groups.setdefault((view.get("circle") or {}).get("circle_id", ""), []).append(view)
    lines = []
    for key in sorted(groups, key=lambda key: (not key, str((groups[key][0].get("circle") or {}).get("name", "")))):
        indent = "  " if key or len(groups) > 1 else ""
        if key:  # its words like a note's lines, with their author, once for every place in it
            circle = groups[key][0]["circle"]
            lines.append(f'circle {circle["name"]} [{key}]' + ("" if circle["text"] else ": no words yet"))
            lines.extend(f'  words ({circle["text_author"]}): {line}' for line in circle["text"].splitlines())
        elif indent:
            lines.append("not in a circle:")
        for view in groups[key]:
            lines.extend(indent + line for line in _place_lines(drive_root, view, counted))
    if lines:
        lines.append("a place's rooms: configure_presence(action='inspect', place_id=<its id in brackets>)")
    return "\n  ".join(lines) if lines else "none"


def _place_lines(drive_root, view, counted) -> list[str]:
    from ouroboros.place_arrivals import recent_identity
    from ouroboros.places import OWNER_VOICE, OwnerAccount, owner_account_matches
    identity = recent_identity(drive_root, view["place_id"])
    sending = (identity or {}).get("sends_as") or {}
    owner = sending.get("kind") == "user" and owner_account_matches(
        counted.get(view["place_id"]) or [OwnerAccount(**a) for a in view.get("owner_accounts", [])],
        sending["id"], view.get("account_id", ""))
    voice = ""
    if sending:
        voice = (f'; as Ouroboros (bot {sending["label"]})' if sending["kind"] == "bot"
                 else f'; writes as {sending["label"]}' + (" (your human's own account)" if owner else ""))
    count, rooms = view.get("unanswered"), view.get("rooms")
    # A count; the closing line names the reader of every room, sealed ones included.
    held = f"rooms: {len(rooms)}" if rooms else "no captured room" if rooms == [] else "rooms unknown"
    circle = view.get("circle") or {}
    placed = ""
    if circle.get("text") and circle["placed_by"] not in OWNER_VOICE:  # words I wrote there stay my own record
        placed = ("; you placed it in this circle: your human's words there still restrict it, but consent needs their "
                  "placement" if circle.get("text_author") in OWNER_VOICE else "; you placed it in this circle")
    lines = [f'{view.get("label") or view["place_id"]} [{view["place_id"]}]: {view["state"]}; '
             f'open turns: {count if count is not None else "unknown"}; {held}{voice}{placed}']
    for note in view.get("notes") or []:
        lines.extend(f'  note ({note["author"]}): {line}' for line in note["text"].splitlines())
    return lines


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
                     **({"place_id": place_id} if place_id else {}),  # the Shade's door to that place's sheet
                     "started_at": row.get("started_at"),
                     **({"first_reply_at": first_reply_at(drive_root, task_id)} if cause == "focus" else {}),
                     "ended_at": terminal_time_fact(row)["occurred_at"],
                     "live": live_wake_task_id == task_id or row.get("status") not in FINAL_STATUSES,
                     "phase": _wake_phase(row, live_wake_task_id == task_id), "quiet": quiet_finish(row),
                     **({"tool_calls": row["tool_calls"]} if "tool_calls" in row else {}),
                     **({"rounds": usage["total_rounds"]} if "total_rounds" in usage else {}),
                     "checks": row.get("safety_checks", []), "note": notes.get(task_id),
                     **({"summary": first_sentence(row["wake_summary"])} if isinstance(row.get("wake_summary"), str) and row["wake_summary"] else {}),
                     **({"said_owner": bool(row["owner_message_sent"])} if "owner_message_sent" in row else {}),
                     **{k: row[k] for k in TASK_COST_META_FIELDS if k in row}})
    return rows
