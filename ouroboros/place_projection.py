"""Owner-facing place views from current connector declarations and stored choices.

At the sending side of a linked Ouroboros (connector ``ouroboros``), a letter's card in the room of the work that
wrote it (section "Letters" below).
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Mapping
from ouroboros.places import (AUTHOR_OWNER, STATE_SILENT, PlaceError, confirmed_elsewhere, connect_place,
                              is_wake_signal, place_records, place_summary, note_identity, migration_problems)
from ouroboros.places_migration import declared_wake_facts, matching_declarations, merged_declared_facts, wake_review

log = logging.getLogger(__name__)


def place_declarations(root) -> list[dict]:
    from ouroboros.extension_contributions import live_place_declarations, project_place_status
    from ouroboros.skill_loader import discover_skills
    from ouroboros.extension_ui_validation import validate_contributions
    from ouroboros.icon_geometry import icon_spec
    rows = live_place_declarations()
    for row in rows:
        # ``published``: the declarations a wake save reads (``places.declared_fact_labels``). The rows added below are
        # only installed: a save refuses a new signal of theirs until their connector is on (``unpublished_signals``).
        row.update(readiness={"text": "Connector on", "tone": "ok"}, published=True)
    published = {row["skill"] for row in rows}
    for skill in discover_skills(root):
        if skill.name in published or skill.load_error or skill.identity_collision:
            continue
        clean, _warnings = validate_contributions(skill.manifest)
        readiness = {"text": "Off in Skills", "tone": "warn"} if not skill.enabled else {
            "text": "Connector not running" if skill.review.gate_for(skill.content_hash)["executable_review"]
            else "Review required", "tone": "warn"}
        for row in clean:
            if row.get("provides") != "place":
                continue
            source = None
            if row.get("icon") == "ui/icon.svg":
                path = skill.skill_dir / "ui/icon.svg"
                try:
                    if path.resolve().is_relative_to(skill.skill_dir.resolve()):
                        source = path.read_bytes()
                except OSError:
                    log.warning("Place icon unavailable for %s", skill.name, exc_info=True)
            # An unloaded connector has no callable status route.
            row.pop("source", None)
            rows.append(project_place_status({**row, "skill": skill.name, "readiness": readiness,
                         "icon_spec": icon_spec(row.get("icon", ""), source)}))
    return rows


def unpublished_signals(declarations, offered) -> list[str]:
    """The ``offered`` signals a wake save refuses now: it adds a new signal only when a published declaration of the
    place labels it (``places.declared_fact_labels``), so the signals an installed connector declares wait until it is
    on. Core kinds are never held, and a saved signal stays removable whatever this says."""
    from ouroboros.extension_contributions import place_fact_labels
    accepted = place_fact_labels([row for row in declarations if row.get("published")])
    return [fact for fact in dict.fromkeys(offered) if is_wake_signal(fact) and fact not in accepted]


def connect_declared_place(root, connector: str, *, account_id: str = "", conversation_id: str = "", label: str = "",
                           first=None, author: str = AUTHOR_OWNER):
    """Connect from the installed declaration; ``first`` is what an Add says about the new place (``places.connect_place``)."""
    if not isinstance(label, str):
        raise PlaceError("place_invalid_string", "label")
    declarations = [row for row in place_declarations(root) if row["skill"] == connector]
    if len(declarations) != 1:
        raise PlaceError("place_declaration_unavailable", "connector", "Enable a reviewed connector with one place declaration.")
    row = declarations[0]
    provider = str(row["destination"]).removeprefix("vendor:").lower()
    # A label that already starts with the connector's word is not prefixed again ("Telegram · Telegram · …").
    given, word = label.strip(), str(row["label"])
    named = given.casefold() == word.casefold() or given.casefold().startswith(f"{word} · ".casefold())
    title = given if named else f"{word} · {given}" if given else word
    return connect_place(root, connector=connector, provider=provider, label=title, state=STATE_SILENT,
                         account_id=account_id, conversation_id=conversation_id, scope=row.get("scope", "account"),
                         contribution_id=row["id"], wakes=row.get("wakes", []), first=first, author=author)


def said_once(label: str) -> str:
    """A label stored before its connector stopped prefixing its own word reads that word once (display only)."""
    parts = str(label or "").split(" · ")
    return " · ".join(part for index, part in enumerate(parts) if not index or part != parts[index - 1])


def switch_state(root) -> str:
    """The switch as a place's quiet line names it, from the reading intake folds (``presence_admission.switch_cause``:
    ``on`` | ``off`` | ``panic`` | ``unknown``), so the words why a place is quiet follow its admission. A read that
    fails is ``unknown``: no claim (intake stays closed on it, the view says nothing)."""
    from ouroboros.presence_admission import switch_cause
    try:
        return switch_cause(root)
    except Exception:
        log.warning("Consciousness switch unreadable for the place views", exc_info=True)
        return "unknown"


_NO_WORK_READ = object()  # place_view's default: no work or room read, and no ``peer_work`` / ``peer_room`` field


def place_view(root, place, *, declarations: list[dict] | None = None, allowance: dict | None = None,
               brief: bool = False, records=None, facts=None, switch: str | None = None,
               peer_work: Any = _NO_WORK_READ, peer_room: Any = _NO_WORK_READ) -> dict[str, Any]:
    """Stored facts for status/launch; the owner surface also resolves connector capabilities. ``records`` is one
    ``places.place_records`` read shared by a list; its circle and description facts join the place here, the
    circle's words only in a brief view (the list names them once, ``places_view``). ``switch`` is one
    ``switch_state`` read shared by a list (read here when absent): a full view says why the place is quiet now as
    ``quiet`` (``consciousness_view.place_quiet``: derived on every read, never stored). A linked Ouroboros's place
    also carries ``peer_effective`` (``peer_links.effective_permissions``), likewise derived. ``facts`` is one
    ``place_arrivals.place_facts`` read shared by a list (rooms only for a brief one; read here when absent)."""
    from ouroboros.consciousness_allowance import allowance_window
    from ouroboros.place_arrivals import place_facts
    from ouroboros.places import place_model_route
    from ouroboros.route_choice import following_route
    if allowance is None and not brief:
        allowance = allowance_window(root, allow_stale=True)
    records = place_records(root, persist_import=False) if records is None else records
    result = dict(place_summary(place))
    # Legacy notes predate ids; use stable content+ordinal ids without rewriting history.
    result["notes"] = [{**n, "note_id": note_identity(place.place_id, i, place.notes[i])}
                       for i, n in enumerate(result["notes"])]
    circle = records.circle_context(place.place_id)
    model, layer = place_model_route(place, circle)
    # No model of its own follows consciousness; the aliases are roles. Each resolves by its slot's one rule.
    model, local = following_route(model or "consciousness") if model in ("", "main", "light") else (model, False)
    result["model_route"] = {"value": model, "layer": layer, "local": local}
    result["circle"] = circle if brief or circle is None else {
        key: circle[key] for key in ("circle_id", "placed_by", "placed_at")}
    result["description"] = records.description(place.place_id)
    linked = None
    if place.connector == "ouroboros":
        # A linked Ouroboros (peer_links): my human's stored choices, what is in effect now beside them and the link
        # (no key). A paused, unconfirmed, unmarked or revoked link keeps its choices and reads as refusing here.
        from ouroboros.peer_links import link_card, peer_facts_line, place_permissions

        peer, switch = records.peer(place.place_id), switch_state(root) if switch is None else switch
        linked = place_permissions(root, place, peer, switch=switch)
        result.update(peer=peer, peer_effective=linked, peer_facts=peer_facts_line(peer, linked),
                      link=link_card(root, place.place_id))
    cause = "place:" + place.place_id
    spend = next((item.get("usd") for item in (allowance or {}).get("spend_by_cause") or []
                  if item.get("cause") == cause), None)
    own = (place_facts(root, [place], full=not brief) if facts is None else facts)[place.place_id]
    rooms = own["rooms"]
    if brief:
        # Status and launch need labels, notes and open turns: one index pass, counts unknown when it is unavailable.
        counts = [room.get("unanswered_count") for room in rooms or []]
        result.update(label=said_once(place.label or place.provider), rooms=rooms,
                      unanswered=None if rooms is None or any(c is None for c in counts) else sum(counts),
                      last_activity_at=max((str(room.get("last_at") or "") for room in rooms or []), default=""),
                      activity_coverage="captured_place_events", spent_24h_usd=spend)
        return result
    # Why it is quiet now, by its one cause (owner 15.11, 5A): on its human's screens, never inside the place.
    from ouroboros.consciousness_view import place_quiet
    from ouroboros.runtime_limits import get_consciousness_autonomy
    # A link that accepts nothing here (or cannot be read) is not "hearing": its own facts say why, never the switch.
    quiet = None if linked and not linked["messages"] else place_quiet(
        place.state, switch=switch_state(root) if switch is None else switch,
        autonomy=get_consciousness_autonomy(), allowance=str((allowance or {}).get("status") or ""),
        limit_usd=(allowance or {}).get("limit_usd"), assignments=linked["assignments"] if linked else None)
    if quiet:
        result["quiet"] = quiet
    if place.connector == "ouroboros" and peer_work is not _NO_WORK_READ:  # the owner's list read hands them in
        result["peer_work"] = peer_work
    if place.connector == "ouroboros" and peer_room is not _NO_WORK_READ:
        result["peer_room"] = peer_room
    result.update(label=said_once(place.label or place.provider),
                  rooms=rooms, unanswered=own["unanswered"] if rooms is not None else None,
                  last_activity_at=own["last_activity_at"],
                  activity_coverage="captured_place_events", spent_24h_usd=spend)
    rows = place_declarations(root) if declarations is None else declarations
    builtin = place.connector == "ouroboros"  # built in: letters arrive at /api/letters, no connector skill runs
    matches = [] if builtin else matching_declarations(place, rows)
    row = ({"readiness": {"text": "Built in", "tone": "ok"}, "destination": "vendor:ouroboros",
            "wakes": ["direct_message", "from_bot"]} if builtin
           else matches[0] if matches else {})
    suggestions = declared_wake_facts(matches) if place.legacy_binding_ids else list(row.get("wakes") or [])
    result.update(label=said_once(place.label or row.get("label") or place.provider),
                  reply_destination=place.destination,
                  destination=row.get("destination") or "vendor:" + place.provider,
                  icon=row.get("icon_spec") or {}, wake_suggestions=suggestions,
                  readiness=row.get("readiness") or {"text": "Connector unavailable", "tone": "warn"})
    review = wake_review(records, place, result["wake_suggestions"])
    if review:
        result["wake_review"] = review
    from ouroboros.extension_contributions import place_fact_labels
    from ouroboros.safety_config import get_safety_mode
    # The vocabulary combines all matching declarations; the status and settings doors belong to the first one.
    facts = merged_declared_facts(matches)
    result.update(safety_mode=get_safety_mode("consciousness"), labels=place_fact_labels(matches))
    if facts is not None:
        result["facts"] = facts
    offered = [item.get("id") if isinstance(item, dict) else item for item in facts or ()]
    if waiting := unpublished_signals(matches, [*offered, *suggestions]):
        result["unpublished_signals"] = waiting
    result.update({key: row[key] for key in ("source", "settings") if key in row})
    result["account_candidates"] = own["account_candidates"]
    # Accounts another place of this bot or app confirms count here too (``places.counted_owner_accounts``): named
    # with where they were confirmed, removed there.
    result["shared_owner_accounts"] = [
        {**{key: getattr(account, key) for key in ("platform_actor_id", "label", "confirmed_at", "account_id")},
         "place_id": other.place_id, "place_label": said_once(other.label or other.provider)}
        for other, account in confirmed_elsewhere(records.places.values(), place)
        if not account.account_id or place.account_id in ("", "*", account.account_id)]
    return result


def circle_summary(circle) -> dict[str, Any]:
    """A circle once, with its whole words and who wrote them; its places name it by ``circle.circle_id``."""
    return {**{key: circle[key] for key in ("circle_id", "name", "text", "text_author", "text_at", "revision")},
            "model": circle.get("model", "")}


def places_view(root, *, owner_reading: bool = False) -> dict:
    """Every place's full view, its circles and the installed connectors. ``owner_reading``: the owner's own list read
    (``gateway.places_api`` GET /api/places) also names each linked place's latest work its letters started here
    (``peer_works``) and, while its arrival index lists no room, the room this side wrote to it in (``peer_rooms``);
    any other reader of these views (a place's own tools) gets no such field."""
    declarations = place_declarations(root)
    records = place_records(root, persist_import=False)
    from ouroboros.consciousness_allowance import allowance_window
    from ouroboros.place_arrivals import place_facts
    allowance = allowance_window(root, allow_stale=True) or {}
    switch = switch_state(root)
    facts = place_facts(root, records.places.values())
    linked = [p for p in records.places.values() if p.connector == "ouroboros"] if owner_reading else []
    works = peer_works(root, [p.place_id for p in linked]) if linked else {}
    talks = peer_rooms(root, [p for p in linked if not (facts.get(p.place_id) or {}).get("rooms")]) if linked else {}
    from ouroboros.consciousness_view import connect_note
    return {"migration_problems": migration_problems(root, records), "connect_note": connect_note(switch),
            "places": [place_view(root, p, declarations=declarations, allowance=allowance, records=records,
                                  facts=facts, switch=switch,
                                  **({"peer_work": works.get(p.place_id), "peer_room": talks.get(p.place_id)}
                                     if owner_reading and p.connector == "ouroboros" else {}))
                       for p in records.places.values()],
            "circles": [circle_summary(circle) for circle in records.circles],
            "connectors": [{"connector": r["skill"],
                            "label": r["label"], "icon": r.get("icon_spec") or {}, "destination": r["destination"],
                            "readiness": r.get("readiness") or {"text": "Connector unavailable", "tone": "warn"},
                            **{key: r[key] for key in ("facts", "source", "settings", "labels") if key in r},
                            "wakes": r.get("wakes") or [],
                            "connected": any(matching_declarations(p, (r,)) for p in records.places.values())}
                           for r in declarations]}

# -- Letters ------------------------------------------------------------------------------------------------------
# A linked Ouroboros's letters as its human's screens state them, read only: never a key, a hash or the room. Its
# own conversation is the place's like any other (the place sheet reads the room's history); nothing here selects it.
#
# At the sender a letter is a ``peer`` delegate custody record of its work (``tools/peer_delegate``). Its card is one
# typed fact, ``peer_letter_card`` (``gateway/history_contracts.PeerLetterCardFact``), on that work's own progress line,
# so the room that work lives in receives it live and replays it from its history (``gateway/history``: a row of its
# own, never the task's carrier), and no other room carries it. It is published from the send's typed outcome before
# any Safety note is composed onto it (``card_of_send``), at each ``delegate_wait`` read (``observe_letter_run``: a
# question the read names carries its words) and at the complete retained end before a tool preview bounds it, at an
# answer whose delivery the receiver's receipt confirmed (``observe_letter_answer``) and at its Stop
# (``observe_letter_stop``), from records the sending side already keeps: the letter recorded at its start, its place
# and link, the read itself, the answer it sent. A state not stated stays unknown, and a reply shows only on the
# receiver's explicit word that it is this sender's (``result_available``, ``gateways/peer``), never as the receiving
# work's own final. Words longer than the card carries are read whole on demand from those records (``letter_words``).
# No journal of its own, and no read of the other installation for a browser.


def starts_work_here(place: Mapping[str, Any]) -> bool | None:
    """Whether a linked place can assign work, using the letter door's projection. Unread stays unknown."""
    if place.get("connector") != "ouroboros":
        return False
    effect = place.get("peer_effective")
    value = effect.get("assignments") if isinstance(effect, Mapping) else None
    return value if isinstance(value, bool) else None


def any_assignment_permission(answers) -> bool | None:
    """One affirmative is enough; otherwise an unread permission prevents a negative claim."""
    unknown = False
    for value in answers:
        if value is True:
            return True
        unknown |= value is None
    return None if unknown else False


def letters_can_start_work(root: Any, *, records=None) -> bool | None:
    """The letter door's own reading for the switch turn and preview."""
    from ouroboros.peer_links import place_permissions

    try:
        if records is None:
            records = place_records(root, persist_import=False)
        places = [place for place in records.places.values() if place.connector == "ouroboros"]
        if not places:
            return False
        switch = switch_state(root)
        return any_assignment_permission(place_permissions(root, place, records.peer(place.place_id), switch=switch)
                                         .get("assignments") for place in places)
    except Exception:  # noqa: BLE001 - unread is not a claim that letters do or do not start work
        log.debug("Linked places unreadable for the switch's words", exc_info=True)
        return None


# One row per linked place among the letter roots alone (``peer_letters.letter_task_id``: the ``letter-`` key range of
# the tasks' primary key, never every task): its work whose row changed last (``tasks.ts``, rewritten as the work
# moves, so never an acceptance date), with the facts its acceptance record holds. SQLite answers a bare column of a
# MAX() group from that row.
_PEER_WORKS = (
    "SELECT json_extract(b.body,'$.metadata.peer_letter.place_id') AS place, t.task_id AS task_id, t.status AS status, "
    "MAX(t.ts) AS changed, b.body AS body "
    "FROM tasks t JOIN task_bodies b ON b.task_id=t.task_id WHERE t.task_id>='letter-' AND t.task_id<'letter.' "
    "AND json_extract(b.body,'$.metadata.peer_letter.place_id') IN (SELECT value FROM json_each(?)) GROUP BY place")


def peer_works(root: Any, place_ids: list) -> dict:
    """``{place_id: work | None}``: each linked place's work its letters started here whose record changed last, as its
    acceptance record states it (``gateway.peer_letters_api._accept_new``: the letter root's task row): who sent it (the
    link's label recorded in ``metadata.peer_letter.sender_label``), its title, its stored status and the room it lives
    in (the row's own ``chat_id``: the place's room stored at acceptance). One read for all of them; None where no letter
    was accepted; ``{"unknown": True}`` for each when the task store cannot be read now (never a guessed absence)."""
    from pathlib import Path
    import json

    from ouroboros import store
    from ouroboros.project_dialogue import outcome_phase

    wanted = [str(pid) for pid in place_ids]
    if not wanted:
        return {}
    unknown = {pid: {"unknown": True} for pid in wanted}
    try:
        if not (Path(root) / store.STORE_REL).exists():
            return {pid: None for pid in wanted}  # no task store yet: nothing was ever accepted here
        if store.migrating(root):
            return unknown
        with store.open_store(root, write=False) as txn:
            rows = txn.conn.execute(_PEER_WORKS, (json.dumps(wanted),)).fetchall()
    except Exception:  # noqa: BLE001 - an unread store is no claim that nothing was accepted
        log.debug("Accepted letters of the linked places unreadable", exc_info=True)
        return unknown
    works: dict = {pid: None for pid in wanted}
    for row in rows:
        place_id = str(row["place"])
        try:
            body = json.loads(row["body"])
            room = body.get("chat_id")
            peer = (body.get("metadata") or {}).get("peer_letter") or {}
            status = str(row["status"] or "")
            phase = outcome_phase({**body, "status": status}, {}) if status else ""
            stop = body.get("current_stop")
            if phase == "working" and (status == "paused" or (isinstance(stop, dict) and stop.get("cause") == "pause")):
                phase = "paused"
            works[place_id] = {"task_id": str(row["task_id"]), "title": str(body.get("title") or ""),
                               "status": status, "phase": phase, "from": str(peer.get("sender_label") or ""),
                               "room_chat_id": room if isinstance(room, int) and not isinstance(room, bool) else None}
        except (ValueError, AttributeError, TypeError):
            works[place_id] = unknown[place_id]
    return works


# A linked place's room is listed by what arrived there (``place_arrivals``); before anything has, this side's own
# messages to it are what that room holds (each a ``presence_delivery`` receipt, ``gateway.peer_letters_api``).
_NEWEST_RECEIPT = ("SELECT ts FROM dialogue INDEXED BY dialogue_room WHERE stream='chat' AND chat_id=? "
                   "AND kind='presence_delivery' ORDER BY seq DESC LIMIT 1")


def peer_rooms(root: Any, places: list) -> dict:
    """``{place_id: {"room_chat_id", "wrote_at"} | None}`` for linked ``places`` whose arrival index lists no room: the
    place's own room, by the one derivation its readers and its letter door use (``places.conversation_chat_id`` of its
    conversation, ``gateway.peer_letters_api._room``), with the time of this side's newest message receipt there; None
    where this side wrote nothing there. ``{"unknown": True}`` for each when the history cannot be read now."""
    from ouroboros import dialogue_store
    from ouroboros.places import conversation_chat_id
    from ouroboros.presence_bindings import conversation_key

    rooms = {p.place_id: conversation_chat_id(conversation_key(p.provider, p.account_id, p.conversation_id, ""))
             for p in places}
    if not rooms:
        return {}
    try:
        if not dialogue_store.present(root):
            return {pid: None for pid in rooms}
        with dialogue_store.transaction(root, write=False) as conn:
            newest = {pid: conn.execute(_NEWEST_RECEIPT, (chat,)).fetchone() for pid, chat in rooms.items()}
    except Exception:  # noqa: BLE001 - an unread history is no claim that nothing was written
        log.debug("The rooms of the linked places unreadable", exc_info=True)
        return {pid: {"unknown": True} for pid in rooms}
    return {pid: {"room_chat_id": rooms[pid], "wrote_at": str(row[0] or "")} if row else None
            for pid, row in newest.items()}


def _money(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


CARD_TEXT_MAX = 8000  # the letter's and a given answer's words a card row carries whole; longer ones are read whole
# on demand (``letter_words``), and a receiver's failure reason past it keeps its first part, marked (``error_chars``)
_CARD_STATES = frozenset({"accepted_running", "needs_input", "waiting_owner", "done", "failed", "cancelled", "refused",
                          "not_sent", "unconfirmed", "unknown"})
_RUN_STATES = {"succeeded": "done", "failed": "failed", "interrupted": "failed", "cancelled": "cancelled",
               "running": "accepted_running"}
_STATE_WORDS = {"accepted_running": "sent, at work", "needs_input": "asked about it", "waiting_owner": "waiting for its human",
                "done": "finished", "failed": "failed", "cancelled": "cancelled", "refused": "refused",
                "not_sent": "not sent", "unconfirmed": "acceptance not confirmed", "unknown": "state unknown"}
# A Stop its read-back did not confirm (``delegate_custody.CANCEL_*``), as the card says it: never cancelled.
_STOPS = {"requested": "requested", "failed": "refused", "containment_fault_run_may_still_be_live": "unverified"}
_STOP_WORDS = {"requested": "stop requested", "refused": "stop refused", "unverified": "stop not verified"}
_PUBLISHED: dict = {}  # (task, letter) -> what this process last published: a repeated read adds no row
_PUBLISHED_LOCK = threading.Lock()


def _whole(text: Any, limit: int = CARD_TEXT_MAX) -> tuple[str | None, int]:
    """Words whole within ``limit`` (``None`` past it) and how many there are."""
    words = str(text or "")
    return (words if len(words) <= limit else None), len(words)


def _peer_names(root: Any, place_id: str) -> tuple[str, str]:
    """The linked place's name here (its label) and its face's seed (the other side's install id)."""
    from ouroboros.peer_links import link_of_place

    place = place_records(root, persist_import=False).places.get(place_id)
    found = link_of_place(root, place_id) or ("", {})
    return str(getattr(place, "label", "") or "Ouroboros"), str(found[1].get("peer_install_id") or "")


def _said(question: Any) -> dict | None:
    """A question or an answer to it on the card: its id and words, whole within the letter protocol's bound
    (``peer_contracts.TEXT_MAX_CHARS``; only the rows that name them carry them, and no other record here holds them);
    a question whose words the receiver holds now carries no words and its hold code (``held``, the receiver's own),
    and one whose answer is reserved there, its delivery not proven, says so (``delivery_state``, its own word)."""
    from ouroboros.gateway.peer_contracts import TEXT_MAX_CHARS

    if not isinstance(question, Mapping) or not question.get("question_id"):
        return None
    said: dict = {"question_id": str(question["question_id"])}
    if question.get("held") and not isinstance(question.get("text"), str):
        said.update(text=None, held=str(question["held"]))
    else:
        said["text"], chars = _whole(question.get("text"), TEXT_MAX_CHARS)
        said.update({"chars": chars} if said["text"] is None else {})
    return {**said, **({"delivery_state": "reserved"} if question.get("delivery_state") == "reserved" else {})}


def letter_card_fact(root: Any, letter_id: str, *, state: str, letter: Mapping[str, Any] | None = None, place_id: str = "",
                     phase: str = "", refusal_code: str = "", error: str = "", spent_usd: Any = None,
                     spent_final: bool = False, reply: str = "", answer: Any = None, at: str = "", held: str = "",
                     question: Mapping[str, Any] | None = None, answered: Mapping[str, Any] | None = None,
                     stop: str = "") -> dict:
    """The card of one letter (``gateway/history_contracts.PeerLetterCardFact``): to whom (the place's name and face
    seed), the letter as its start recorded it, its state in the letter contract's words with a typed refusal, and
    what is known of its spend under its ceiling (``None``: unknown, never 0; ``spent_final`` false: so far, never the
    total). Done, its ``reply`` is ``given`` only on the receiver's explicit word that a reply is this sender's,
    ``none`` on its word that there is none, ``held`` on its word that it holds one back for now, else ``unknown``; only
    a given reply carries its answer. The letter's and the answer's words ride whole up to ``CARD_TEXT_MAX``, else null beside
    their length (``text_chars``, ``answer_chars``), read whole on demand (``letter_words``); a failure reason past it
    keeps its first part beside its length (``error_chars``). ``question`` is the receiver's question a read names,
    ``answered`` this side's answer whose delivery the receiver confirmed, ``stop`` a Stop not confirmed."""
    if letter is None or not place_id:
        from ouroboros.delegate_invocation import invocation_record

        record = invocation_record(root, letter_id) or {}
        letter = record.get("request") if letter is None and isinstance(record.get("request"), Mapping) else letter or {}
        place_id = place_id or str((record.get("engine_ref") or {}).get("place_id") or "")
    to, seed = _peer_names(root, place_id)
    text, chars = _whole(letter.get("text"))
    fact = {"letter_id": str(letter_id), "place_id": place_id, "to": to, "to_seed": seed, "text": text,
            **({"text_chars": chars} if text is None else {}), "title": str(letter.get("title") or ""),
            "ceiling_usd": _money(letter.get("cost_ceiling_usd")), "spent_usd": _money(spent_usd),
            "spent_final": bool(spent_final) and _money(spent_usd) is not None,
            "state": state if state in _CARD_STATES else "unknown"}
    error = str(error or "")
    fact.update({key: value for key, value in (("phase", phase), ("refusal_code", refusal_code),
                                               ("error", error[:CARD_TEXT_MAX]), ("at", at),
                                               ("question", _said(question)), ("answered", _said(answered)),
                                               ("stop", stop if stop in _STOP_WORDS else "")) if value})
    fact.update({"error_chars": len(error)} if len(error) > CARD_TEXT_MAX else {})
    if fact["state"] == "done":
        fact["reply"] = reply if reply in ("given", "none", "held") else "unknown"
        if fact["reply"] == "held":
            fact["held"] = str(held or "held")
        if fact["reply"] == "given":
            fact["answer"], chars = _whole(answer)
            fact.update({"answer_chars": chars} if fact["answer"] is None else {})
    return fact


def publish_letter_card(ctx: Any, letter_id: str, **facts: Any) -> None:
    """Put the letter's card fact on its work's progress line with plain words for text-only readers. A fact equal to
    the one this process last published for the letter adds no row; a failure here never touches the letter."""
    emit = getattr(ctx, "emit_progress_fn", None)
    if not callable(emit) or not letter_id:
        return
    try:
        from ouroboros.delegate_custody import custody_root

        fact = letter_card_fact(custody_root(ctx), letter_id, **facts)
        key = (str(getattr(ctx, "task_id", "") or ""), str(letter_id))
        asked = fact.get("question") or {}
        said = (*(fact.get(name) for name in ("state", "phase", "refusal_code", "spent_usd", "spent_final", "reply",
                                              "stop")),
                *((fact.get(name) or {}).get("question_id") for name in ("question", "answered")),
                asked.get("held"), asked.get("delivery_state"), fact.get("held"))
        with _PUBLISHED_LOCK:
            if _PUBLISHED.get(key) == said:
                return
            _PUBLISHED[key] = said
            while len(_PUBLISHED) > 512:
                _PUBLISHED.pop(next(iter(_PUBLISHED)))
        words = _STATE_WORDS[fact["state"]] + (f" ({fact['refusal_code']})" if fact.get("refusal_code") else "") + (
            f", {_STOP_WORDS[fact['stop']]}" if fact.get("stop") else "")
        emit(f"Letter to {fact['to']}: {words}", _progress_meta_override={"peer_letter_card": fact})
    except Exception:  # noqa: BLE001 - the card is a view of the letter, never a condition of it
        log.debug("The card of letter %s was not published", letter_id, exc_info=True)


_SENT = {"unknown": "unconfirmed", "refused": "refused", "no": "not_sent"}  # ``tools/peer_delegate._out``


def card_of_send(ctx: Any, sent: Any, *, letter: Mapping[str, Any], place_id: str) -> None:
    """The card of a send's own outcome, from the typed fact its producer decided (``tools/peer_delegate._post``), never
    from the words the model reads: ``started`` with the receiver's state and phase as its answer named them;
    ``unknown``, acceptance not confirmed (a retry's refusal, or a receiver refusal that does not prove the letter unrun,
    included); ``refused``, the receiver's typed refusal that proves it unrun; ``no``, nothing left here (a refusal
    before the letter left, Safety's included, after it was recorded); each with its code."""
    if not isinstance(sent, Mapping):
        return
    kind, code = str(sent.get("sent") or ""), str(sent.get("code") or "")
    if kind == "started":
        state = str(sent.get("state") or "")
        facts = {"state": state if state in _CARD_STATES else _RUN_STATES.get(state, "unknown"),
                 "phase": str(sent.get("phase") or "")}
    elif kind in _SENT:
        facts = {"state": _SENT[kind], "refusal_code": code}
    else:
        return
    publish_letter_card(ctx, str(letter.get("letter_id") or ""), letter=letter, place_id=place_id, **facts)


def _end_facts(final: Any, output: Any, cost: Any = None, held: Any = None) -> dict:
    """What a letter's end says on its card, from its final summary as the transport carries it (live
    ``finalSummary``, retained ``final_summary``): a reply only on the receiver's explicit ``result_available`` with its
    words in hand, never the text alone (a stated reply this read holds no words of stays unknown, never empty); an
    answer the receiver holds now (``held``, the read's ``summary.peerHold`` of a result, or ``result_held`` beside
    ``result_available`` false) is ``held`` with its code, never none or an empty answer; and the receiver's spend as it
    states it (``peerCost``, else the summary's ``cost``: ``known_usd``; ``final`` false is so far, never the total)."""
    final = final if isinstance(final, Mapping) else {}
    cost = cost if isinstance(cost, Mapping) else final.get("cost") if isinstance(final.get("cost"), Mapping) else {}
    available = final.get("result_available")
    held = held or (final.get("result_held") if available is False else None)
    reply = "held" if held else ("given" if available else "none") if isinstance(available, bool) else "unknown"
    text = output.get("text") if reply == "given" and isinstance(output, Mapping) else None
    reply = "unknown" if reply == "given" and not isinstance(text, str) else reply
    return {"reply": reply, "answer": text, "spent_usd": cost.get("known_usd"), "spent_final": cost.get("final") is True,
            **({"held": str(held)} if reply == "held" else {})}


def observe_letter_run(ctx: Any, entry: Any, *, detail: Mapping[str, Any] | None = None,
                       payload: Mapping[str, Any] | None = None) -> None:
    """A read of a letter (a ``peer`` custody record) as its card: the receiver's state from the run ``detail``
    (``gateways/peer.get_run``) of a ``delegate_wait`` read, or a retained terminal ``payload`` whole, as it is retained
    (``delegate_terminal_evidence.retained_terminal_result``, before a tool preview bounds it), and at its end the reply
    and spend (``_end_facts``). A question the read names carries its words and the receiver's word that an answer to it
    is reserved there (``deliveryState``). A state the read does not name stays unknown."""
    from ouroboros.execution_domain import is_peer_ref

    if not is_peer_ref(getattr(entry, "engine_ref", None)):
        return
    run_id = str(getattr(entry, "run_id", "") or "")
    if payload is not None:
        state = _RUN_STATES.get(str(payload.get("state") or ""), "unknown")
        spent = (payload.get("cost") or {}).get("receiver_cost") if isinstance(payload.get("cost"), Mapping) else None
        # The retained end is the failure reason's own record on replay: its summary (as the live read's) or its
        # final summary, never a second journal; a reason it does not carry stays unstated.
        summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
        final = payload.get("final_summary") if isinstance(payload.get("final_summary"), Mapping) else {}
        publish_letter_card(ctx, run_id, state=state, phase=str(summary.get("phase") or ""),
                            refusal_code=str(summary.get("refusalCode") or ""),
                            error=str(summary.get("error") or final.get("error") or ""),
                            **_end_facts(payload.get("final_summary"), payload.get("primary_output"), spent))
        return
    detail = detail or {}
    summary = detail.get("summary") if isinstance(detail.get("summary"), Mapping) else {}
    # What the receiver holds now (``gateways/peer._held``): a question's words or a done letter's answer, its code.
    hold = summary.get("peerHold") if isinstance(summary.get("peerHold"), Mapping) else {}
    state = str(summary.get("peerState") or "unknown")
    ended = _end_facts(detail.get("finalSummary"), detail.get("primaryOutput"), detail.get("peerCost"),
                       held=hold.get("reason") if hold.get("what") == "result" else None) if state in (
        "done", "failed", "cancelled") else {}
    asked = next(({"question_id": question.get("id"), "text": question.get("question"),
                   "delivery_state": row.get("deliveryState")}
                  for row in detail.get("pendingInteractions") or () if isinstance(row, Mapping)
                  for question in row.get("questions") or () if isinstance(question, Mapping)
                  and question.get("id") and isinstance(question.get("question"), str)), None)
    if asked is None and hold.get("what") == "question" and hold.get("question_id"):
        asked = {"question_id": hold["question_id"], "text": None, "held": str(hold.get("reason") or "held"),
                 "delivery_state": hold.get("delivery_state")}
    publish_letter_card(ctx, run_id, state=state, phase=str(summary.get("phase") or ""),
                        refusal_code=str(summary.get("refusalCode") or ""), error=str(summary.get("error") or ""),
                        question=asked if state == "needs_input" else None, **ended)


def observe_letter_answer(ctx: Any, entry: Any, interaction_id: str, answers: Any) -> None:
    """This side's answer to the question of a letter as its card, once the receiver's receipt confirmed its delivery
    (``delegate_answer``: ``delivered`` with the mailbox ``msg_id``): the words that left, as the answer letter carries
    them (``gateways/peer._validated_answer_text``). The receiver's state after it is the next read's to say; until
    then the card keeps the question."""
    from ouroboros.execution_domain import is_peer_ref
    from ouroboros.gateways.peer import _validated_answer_text

    if not is_peer_ref(getattr(entry, "engine_ref", None)):
        return
    try:
        text = _validated_answer_text(interaction_id, answers)
    except Exception:  # noqa: BLE001 - rows that are no one answer delivered nothing the card could say
        return
    publish_letter_card(ctx, str(getattr(entry, "run_id", "") or ""), state="needs_input",
                        answered={"question_id": str(interaction_id), "text": text})


def observe_letter_stop(ctx: Any, entry: Any, outcome: Any) -> None:
    """A Stop of a letter as its card, from its typed outcome (``delegate_cancel``: ``custody.cancel_and_verify``): a
    confirmed Stop that cancelled it is ``cancelled``; one its read-back did not confirm keeps the state that read-back
    saw beside what is known of the Stop (``stop``: requested, refused, unverified), never cancelled and never a
    promised answer. A Stop over an end already reached publishes nothing: that end is its reads' to say."""
    from ouroboros.execution_domain import is_peer_ref

    if not is_peer_ref(getattr(entry, "engine_ref", None)) or not isinstance(outcome, Mapping):
        return
    verdict, state = str(outcome.get("outcome") or ""), str(outcome.get("state") or "")
    run_id = str(getattr(entry, "run_id", "") or "")
    if verdict == "confirmed" and state == "cancelled":
        publish_letter_card(ctx, run_id, state="cancelled")
    elif verdict in _STOPS:
        publish_letter_card(ctx, run_id, state=_RUN_STATES.get(state, "unknown"), stop=_STOPS[verdict])


class LetterWordsRefused(Exception):
    """Why ``letter_words`` serves no words: its typed ``code`` and the HTTP ``status`` that answers it."""

    def __init__(self, code: str, status: int, message: str) -> None:
        super().__init__(message)
        self.code, self.status = code, status


def letter_words(root: Any, task_id: str, letter_id: str, part: str) -> str:
    """The whole words of one part of a letter ``task_id`` wrote, for its human's reader of the card, re-read from
    this side's own records and never stored again: ``letter``, the letter as recorded at its start (its
    invocation's exact request, ``delegate_invocation.invocation_record``), or ``answer``, the reply the receiver
    explicitly gave this sender (``final_summary.result_available`` true) in the letter's complete retained end. Typed refusals
    (``LetterWordsRefused``): ``letter_not_found`` (no letter of this task by that id), ``reply_not_given`` (none, held,
    unknown or withheld: never served), ``answer_not_retained`` (a reply was given, but its complete retained end is not
    readable now), ``letter_words_unavailable`` (the records cannot be read now)."""
    from ouroboros.execution_domain import is_peer_ref

    missing = LetterWordsRefused("letter_not_found", 404, "this task wrote no letter by that id")
    try:
        if part == "letter":
            from ouroboros.delegate_invocation import invocation_record

            record = invocation_record(root, letter_id) or {}
            request = record.get("request") if isinstance(record.get("request"), Mapping) else {}
            if record.get("task_id") != task_id or not is_peer_ref(record.get("engine_ref")) \
                    or not isinstance(request.get("text"), str):
                raise missing
            return request["text"]
        from ouroboros.delegate_collection import retained_payload
        from ouroboros.delegate_custody_current import run

        entry = run(root, letter_id)
        if entry is None or entry.task_id != task_id or not is_peer_ref(entry.engine_ref):
            raise missing
        full = retained_payload(root, entry, require_complete=True)
    except LetterWordsRefused:
        raise
    except Exception as exc:  # noqa: BLE001 - an unread record is no claim about the letter
        raise LetterWordsRefused("letter_words_unavailable", 503, "the letter's records cannot be read now") from exc
    if not isinstance(full, Mapping) or full.get("run_id") != letter_id:
        raise LetterWordsRefused("answer_not_retained", 409, "no complete retained end of this letter is readable now")
    final = full.get("final_summary") if isinstance(full.get("final_summary"), Mapping) else {}
    output = full.get("primary_output") if isinstance(full.get("primary_output"), Mapping) else {}
    if final.get("result_available") is not True:
        raise LetterWordsRefused("reply_not_given", 409, "the receiver gave this sender no reply to read")
    if not isinstance(output.get("text"), str):
        raise LetterWordsRefused("answer_not_retained", 409, "the given reply's words are not in its retained end")
    return output["text"]
