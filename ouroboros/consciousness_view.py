"""Read-only consciousness facts from places, current roots, durable results and the ledger's period view."""

import contextlib
from datetime import datetime
import json
import logging
import pathlib
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


def _zone_of(iso_value: Any) -> str:
    """`` (UTC+03:00)``: the zone ``_clock_of`` prints that instant in (the server's), so a UTC server's time is not
    read as the reader's own; "" when there is no instant."""
    from ouroboros.deadline_utils import parse_deadline_ts, utc_offset_text

    parsed = parse_deadline_ts(str(iso_value or ""))
    return f" ({utc_offset_text(parsed.astimezone(_DISPLAY_ZONE))})" if parsed is not None else ""


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
        unread = "places_brief" in snapshot and snapshot["places_brief"] is None
        heard = heard_sentence(None if unread else snapshot.get("places_brief") or [])
        status, detail = "disabled", (f"Panic turned it off. {heard}" if snapshot.get("stop_source") == "panic" else heard)
    elif not snapshot and supervisor_failed:
        status, detail = "stopped", "Cannot wake: Ouroboros did not finish starting. Restart Ouroboros."
    elif not snapshot:
        status, detail = "starting", "Waits for Ouroboros to finish starting."
    elif wake and snapshot.get("live_wake_held"):
        status, (detail, note) = "wake_paused", held_sentence(wake, snapshot.get("live_wake_hold"))
    elif wake:
        status, detail = "thinking", f"Wake-up {wake} is running."
    elif outcome == "skipped:owner_chat_unknown":  # unknown, never empty: a known empty slot wakes in Main
        why = snapshot.get("last_error") or "unknown error"
        status, detail = "owner_chat_unknown", f"The owner chat could not be read ({why}); retry {next_at}."
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
        wake_at = snapshot.get("next_wake_at")
        status, detail = "sleeping", f"Sleeping until {_clock_of(wake_at)}{_zone_of(wake_at)}."
    return {**{key: value for key, value in snapshot.items() if key != "detail_note"},
            "enabled": requested_enabled, "status": status, "detail": detail, **({"detail_note": note} if note else {}),
            "position": position_words(requested_enabled, snapshot.get("places_brief")), "gloss": SWITCH_GLOSS}


# ---- The switch's words (owner 15.11, decision 5A; SPEC §2.14). One switch, named Consciousness and glossed as what
# Ouroboros does on its own, its two positions named by what they mean. This is their one home: the app shows them from
# the server's answers (/api/consciousness, /api/state, /api/places, the setup bootstrap), and the /bg replies (every
# surface, headless included) and the mind's own fact are written here. Before a turn the product names what changes
# from this install's facts; a place says on its human's screens why it is quiet, never inside the place.
SWITCH_GLOSS = "When Ouroboros acts"
POSITION_WORDS = {False: "Only when you write to it", True: "Also on its own and in places"}
# The owner's names of the stored levels (21.10); web/modules/consciousness_autonomy.js holds the same, a test keeps them equal.
LEVEL_NAMES = {"observe": "Advises", "act": "Acts", "full": "Evolves"}
PEOPLE_DAYS = 7  # "how many people wrote there lately": the week before the turn
_NAMED = 3  # places named before «and N more»; the dialog's details list every one


ON_WITHOUT_PLACES = "Also on its own"  # an install with no place says nothing of places (SPEC §2.17)


def position_words(enabled: Any, places=None) -> str:
    """The switch's position by meaning; ``""`` while the stored control is unknown (never Off). ``places`` is the
    install's place read (``None``: not read): on, an install that holds no place at all is not told of places."""
    if not isinstance(enabled, bool):
        return ""
    return ON_WITHOUT_PLACES if enabled and isinstance(places, list) and not places else POSITION_WORDS[enabled]


def _count(n: int, one: str, many: str = "") -> str:
    return f"{n} {one if n == 1 else many or one + 's'}"


def _same(a: str, b: str) -> bool:
    return "".join(str(a).lower().split()) == "".join(str(b).lower().split())


def _place_fact(place, circle: str = "") -> dict:
    """``{place_id, label, state}`` of a stored place or of a view row, its label as the Circles row names it (display
    only): said once, without the service it is on and without its circle's name when more is left («Telegram · Lab chat ·
    Research» in the circle Research reads «Lab chat»; «Slack» stays)."""
    from ouroboros.place_projection import said_once

    get = place.get if isinstance(place, dict) else lambda key, default="": getattr(place, key, default)
    provider = str(get("provider", "") or "")
    parts = said_once(str(get("label", "") or provider)).split(" · ")
    if len(parts) > 1 and provider and _same(parts[0], provider):
        parts = parts[1:]
    if len(parts) > 1 and circle and _same(parts[-1], circle):
        parts = parts[:-1]
    return {"place_id": str(get("place_id", "") or ""), "state": str(get("state", "") or ""), "label": " · ".join(parts)}


def _ranked(places, people=None, circles=None) -> list[dict]:
    """Places with the people who wrote there lately (``recent_people``), the busiest first, then by name; ``circles``
    maps a place id to its circle's name."""
    counts, circles = (people or {}).get("by_place") or {}, circles or {}
    capped = set((people or {}).get("capped_places") or ())
    facts = [_place_fact(place, circles.get(_place_fact(place)["place_id"], "")) for place in places or ()]
    facts = [{**fact, "people": counts.get(fact["place_id"]), **({"people_capped": True} if fact["place_id"] in capped else {})}
             for fact in facts]
    return sorted(facts, key=lambda fact: (-(fact["people"] or 0), fact["label"].lower()))


def _named(facts: list[dict]) -> str:
    """«A», «A and B», «A, B and C», «A, B, C and 28 more»."""
    labels = [fact["label"] or fact["place_id"] for fact in facts]
    if len(labels) > _NAMED:
        return ", ".join(labels[:_NAMED]) + f" and {len(labels) - _NAMED} more"
    return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]


def _people_line(people) -> str:
    """«At least 284 people wrote there in the last 7 days.» from ``recent_people`` (at least: a busy place's newest rows
    only); nothing when it was not read or is zero."""
    total = int((people or {}).get("total") or 0)
    if not total:
        return ""
    return (("At least " if people.get("capped") else "") + _count(total, "person", "people")
            + f" wrote there in the last {PEOPLE_DAYS} days.")


def heard_sentence(places) -> str:
    """What Off leaves in the places, said beside the position: ``places`` is one read of the place store (stored places
    or view rows), ``None`` when it could not be read. A Paused place does not hear either, so it is not counted."""
    if places is None:
        return "Its places are heard, not answered."
    hearing = [fact for fact in map(_place_fact, places) if fact["state"] != "paused"]
    if not hearing:
        return "Nothing wakes it between your messages."
    return f"{_count(len(hearing), 'place')} {'is' if len(hearing) == 1 else 'are'} heard, not answered."


def _running_line(running) -> str:
    """What a turn off ends at once, from the state line it reads (``describe_consciousness_state``): its wake running now
    and the own tasks it started; nothing when neither runs or the read failed."""
    wake = bool((running or {}).get("live_wake_task_id")) and not (running or {}).get("live_wake_held")
    tasks = int((running or {}).get("tasks_running") or 0)
    said = " and ".join((["the wake running now"] if wake else []) + ([f"its {_count(tasks, 'own task')}"] if tasks else []))
    if not said:
        return ""
    # The stop asks (a graceful end where there is room for one), so the words promise neither an instant end nor a
    # saved result (review r3).
    one = (wake and not tasks) or (tasks == 1 and not wake)
    return f"{said[0].upper()}{said[1:]} {'is' if one else 'are'} asked to stop now."


def switch_preview(to: str, places, *, people=None, circles=None, autonomy: str = "", daily_usd=None,
                   running=None) -> dict:
    """What turning the switch ``to`` ("on" | "off") changes, said before it happens, from this install's facts:
    ``places`` (stored places or view rows; ``None`` unread), ``people`` (``recent_people``; ``None`` unread),
    ``circles`` (place id → its circle's name), ``autonomy`` and ``daily_usd`` (turning on), ``running`` (turning off:
    the state line's live wake and own tasks). ``{"to", "lines", "places"}``: one sentence a line and every place the
    turn concerns with its people, for the dialog's details (``None`` when the places could not be read: the app still
    asks, with these lines). A fact absent says nothing."""
    if to == "off":
        hearing = None if places is None else _ranked([p for p in places if _place_fact(p)["state"] != "paused"],
                                                      people, circles)
        lines = ["Ouroboros will act only when you write to it."]
        if hearing is None:
            lines.append("Its places will be heard, not answered; which ones could not be read.")
        elif hearing:
            lines += [f"{_count(len(hearing), 'place')} will be heard, not answered: {_named(hearing)}.",
                      _people_line(people)]
        lines += [_running_line(running), "Your chats, your tasks and the schedules you asked for keep working."]
        return {"to": "off", "lines": [line for line in lines if line], "places": hearing}
    facts = _ranked(places or (), people, circles)
    active = [fact for fact in facts if fact["state"] == "active"]
    silent = [fact for fact in facts if fact["state"] == "silent"]
    if daily_usd == 0:
        lines = ["Ouroboros will also act on its own, but its allowance is $0.00: it cannot wake until you raise it in Costs."]
    else:
        lines = ["Ouroboros will also wake between your messages."]
        if active and autonomy == "observe":
            lines.append(f"Advises: in {_count(len(active), 'place')} it reads and thinks; what it would say waits for you.")
        elif active:
            lines.append(f"It may answer in {_count(len(active), 'place')}: {_named(active)}.")
    if silent and daily_usd != 0:
        lines.append(f"In {_count(len(silent), 'Silent place')} it only reads.")
    if places is None:
        lines.append("Where it may answer could not be read.")
    if active or silent:
        lines.append("What arrived while it was off waits; at its first wake it sees what is unread there and decides what "
                     "to read and answer.")
    return {"to": "on", "lines": lines, "places": None if places is None else active + silent}


def read_switch_preview(drive_root, to: str, running=None) -> dict:
    """``switch_preview`` from this install's stores (``GET /api/consciousness?switch=on|off``, read at the press): one
    place-store read with each place's circle and, turning off, the week's people of every hearing place; ``running``
    is the caller's state line. A read that fails says less, never a guess: no place list says «its places», no people
    count says nothing."""
    from ouroboros.place_arrivals import recent_people
    from ouroboros.places import place_records
    from ouroboros.runtime_limits import get_consciousness_autonomy, get_consciousness_daily_usd

    try:
        records = place_records(drive_root)
        places = list(records.places.values())
        circles = {p.place_id: str((records.circle_context(p.place_id) or {}).get("name") or "") for p in places}
    except Exception:
        log.warning("Places unreadable for the switch preview", exc_info=True)
        places, circles = None, {}
    people = None
    if places:
        try:
            people = recent_people(drive_root, [p.place_id for p in places if p.state != "paused"], days=PEOPLE_DAYS,
                                   services={p.place_id: p.provider for p in places})
        except Exception:
            log.warning("Recent people unreadable for the switch preview", exc_info=True)
    return switch_preview(to, places, people=people, circles=circles, autonomy=get_consciousness_autonomy(),
                          daily_usd=get_consciousness_daily_usd(), running=running)


def on_sentence(at: str, places, autonomy: str = "", daily_usd: Any = None, allowance: str = "") -> str:
    """The switch's first words as it turns consciousness on (every surface's /bg start): when it wakes next, then the
    sentences the app's dialog said before the turn (``switch_preview``), so Telegram and the CLI hear the same. A $0
    allowance, or one spent for the last 24 hours (``allowance``: the window's status), names no wake."""
    rest = switch_preview("on", places, autonomy=autonomy, daily_usd=daily_usd)["lines"][1:]
    if daily_usd == 0:
        head = "Consciousness on, but its allowance is $0.00: it cannot wake until you raise it in Costs."
    elif allowance == "exhausted":
        head = "Consciousness on, but its allowance for the last 24 hours is spent: it wakes once there is room in it again."
    elif allowance == "allowance_unknown":
        head = "Consciousness on, but its spending could not be read: it wakes once it can be read and there is room."
    else:
        head = f"Consciousness on: Ouroboros also acts on its own; next wake at {at}."
    return " ".join([head, *rest])


def status_reply(described: dict) -> str:
    """The /bg status reply on every surface: the switch's position (with the level while on), the host's state line
    and what follows from it."""
    level = LEVEL_NAMES.get(str(described.get("level") or "")) if described.get("enabled") else ""
    head = described.get("position") or described.get("status") or "state unknown"
    note = f" {described['detail_note']}" if described.get("detail_note") else ""
    return f"🧠 Consciousness: {head}{f' ({level})' if level else ''} — {described.get('detail', '')}{note}"


# Why a place is quiet, by its one cause, first match wins (the Panic or the switch before the place's own word, the
# allowance before the level): ``word`` for its row, ``note`` for its sheet, ``door`` where the cause is changed.
# A note promises only waking (what turning it on, or room in the allowance, brings): a Silent or Advises place still
# sends nothing then, so no note says it will answer.
QUIET = {
    "panic": ("Hears only · Panic", "Panic turned consciousness off: it only hears here, and nothing here wakes it until "
              "you turn it on.", "consciousness"),
    "off": ("Hears only · consciousness off", "Consciousness is off: it only hears here, and nothing here wakes it until "
            "you turn it on.", "consciousness"),
    "allowance": ("Waits · allowance spent", "Its allowance for the last 24 hours is spent: nothing here starts a new wake "
                  "until there is room in it again.", "costs"),
    "advises": ("Replies wait for you · Advises", "Advises: it reads and thinks here; what it would say waits for you in "
                "its conversation here.", "consciousness"),
}
# A $0 allowance has no reset to wait for: only the owner's change opens it.
ZERO_ALLOWANCE = "Its allowance is $0.00: nothing here starts a new wake until you raise it in Costs."


def place_quiet(state: str, *, switch: str, autonomy: str = "", allowance: str = "", limit_usd: Any = None) -> dict | None:
    """``{cause, word, note, door}`` when a place that would speak is quiet for a cause beyond its own stored word, else
    ``None``. ``switch``: "on" | "off" | "panic" | "unknown" (no claim); ``allowance``: the window's status and
    ``limit_usd`` its daily limit. A Paused place says Paused and a Silent one Silent: their own words already say it."""
    if state == "paused" or switch == "unknown":
        return None
    if switch in ("panic", "off"):
        cause = switch
    elif allowance == "exhausted":
        cause = "allowance"
    elif state == "active" and autonomy == "observe":
        cause = "advises"
    else:
        return None
    word, note, door = QUIET[cause]
    if cause == "allowance" and isinstance(limit_usd, (int, float)) and limit_usd <= 0:
        note = ZERO_ALLOWANCE
    return {"cause": cause, "word": word, "door": door, "note": note}


def connect_note(switch: str) -> str:
    """The Add flow's words about where a new place lands (it starts Silent), true for the switch as it stands."""
    if switch in ("off", "panic"):
        return "It starts Silent: it sends nothing there until you make it Active. While consciousness is off, it only hears."
    return "It starts Silent: it hears and thinks there and sends nothing until you make it Active."


def first_run_words(autonomy: str = "act") -> dict:
    """The setup step's words, true for the autonomy the offer names (Advises answers nowhere): the gloss, the two
    positions (an install being set up holds no place yet), what they mean, what a subscription alone pays with (said
    once: the step's limits leave it out there) and where to change it later. ``choice`` is the on position a page
    built before the step names (kept for such pages)."""
    places = ("reads and thinks in places you make Active; what it would say waits for you" if autonomy == "observe"
              else "answers in places you make Active")
    return {"gloss": SWITCH_GLOSS + ".", "choice": POSITION_WORDS[True], "off": POSITION_WORDS[False],
            "on": ON_WITHOUT_PLACES,
            # Start begins the first conversation either way (owner 21.11, 1C); on, the wakes go on before a reply.
            "meaning": ("Off: it works when you write to it, and places you connect are heard, not answered. On: it also "
                        f"keeps waking on its own within its daily allowance, even before your first reply, and {places}. "
                        "Either way, Start begins your first conversation."),
            "subscriptionCost": "On a subscription alone its wakes use the subscription's quota between your messages.",
            "door": "Change it later in Dashboard › Consciousness; the allowance is in Dashboard › Costs."}


FIRST_RUN = first_run_words()


def mind_fact(places, *, enabled: Any, stop_source: str = "", panic: bool = False) -> str:
    """The switch as a fact for the mind while its places only hear (the 7 Nov reading: places «work separately»):
    facts only, never a rule of conduct; only when the switch is known off (or a Panic waits) and a place is not
    Paused; else nothing."""
    if not (panic or enabled is False):
        return ""
    hearing = [fact for fact in map(_place_fact, places or ()) if fact["state"] != "paused"]
    if not hearing:
        return ""
    who = ("a Panic waits" if panic else {"owner": "my human turned it off", "panic": "my human pressed Panic",
                                          "agent_tool": "I turned it off"}.get(stop_source, ""))
    return (f"Consciousness is off{f' ({who})' if who else ''}: my own wakes and the schedules I set for myself have "
            f"stopped; my human's chats and tasks, and the schedules my human asked for, still reach me. In my "
            f"{_count(len(hearing), 'connected place')} what arrives is recorded and wakes nothing until it is on.")


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
    """The switch's first words as it turns consciousness off (owner 09.11, variant A; words 15.11, 5A). ``places`` is
    one read of the place store, None when it could not be read; a Paused place does not hear either."""
    said = "Consciousness off: Ouroboros acts only when you write to it."
    if places is None:
        return said + " Its places are heard, not answered."
    hearing = [fact for fact in map(_place_fact, places) if fact["state"] != "paused"]
    if not hearing:
        return said
    return said + f" {heard_sentence(places)}"


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
    """Stored PlaceSummary facts with each place's circle (one store read) and the caller's allowance, no connector
    discovery and no switch read: a brief view says no quiet cause."""
    from ouroboros.place_arrivals import place_facts
    from ouroboros.place_projection import place_view
    from ouroboros.places import place_records

    records = place_records(drive_root)
    facts = place_facts(drive_root, records.places.values(), full=False)
    return [place_view(drive_root, place, allowance=allowance, brief=True, records=records, facts=facts)
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


def cache_writes_text(drive_root, now: float = 0.0) -> str:
    """The install's cache writes of the last 24 hours as one wake fact (``gateway.cost_period``, the Costs page's own
    view, read once per wake). A view that cannot be read is said as unknown, never as none, and the wake goes on."""
    try:
        from ouroboros.gateway.cost_period import cost_period

        return _cache_writes_line(cost_period(pathlib.Path(drive_root), "24h", now=now or None)["cache_writes"])
    except Exception:
        log.warning("The last day's cache writes could not be read for this wake", exc_info=True)
        return "unknown (could not be read)"


def _cache_writes_line(rows) -> str:
    """``cost_period``'s cache-write rows as one line: the calls that wrote, their tokens, the writes' share of recorded
    cost, then every class of what the writes followed, in the Costs page's own order (share of cost, then tokens). A
    class goes without its detail (route and block names). A call whose cost no tariff could share is counted as
    unsplit: a class's dollars are of its other calls, and where no call's cost could be shared the share is unknown,
    never zero."""
    classes: dict[str, dict] = {}
    for row in rows:
        held = classes.setdefault(row["class"], {"calls": 0, "tokens": 0, "usd": 0.0, "unsplit": 0})
        held["calls"] += row["calls"]
        held["tokens"] += row["write_tokens"]
        held["usd"] += row["write_cost"]
        held["unsplit"] += row["unshared_calls"]
    if not classes:
        return "none recorded in the last 24 h"
    calls, tokens, usd, unsplit = (sum(held[key] for held in classes.values()) for key in ("calls", "tokens", "usd", "unsplit"))

    def named(name, held) -> str:
        if held["unsplit"] >= held["calls"]:
            return f'{name} {held["calls"]}'  # none of its calls' cost could be shared: no dollars, never $0.00
        return f'{name} {held["calls"]} ${held["usd"]:.2f}' + (f' ({held["unsplit"]} unsplit)' if held["unsplit"] else "")

    money = ("write share of cost unknown" if unsplit >= calls else
             f"${usd:.2f} write share of recorded cost" + (f", {unsplit} unsplit" if unsplit else ""))
    order = sorted(classes.items(), key=lambda item: (-item[1]["usd"], -item[1]["tokens"], item[0]))
    return (f"{calls} in the last 24 h, {tokens:,} tokens, {money}; by what they followed: "
            + ", ".join(named(name, held) for name, held in order))


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
