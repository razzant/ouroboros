"""Owner-facing place views from current connector declarations and stored choices."""
from __future__ import annotations

import logging
from typing import Any
from ouroboros.places import (AUTHOR_OWNER, STATE_PAUSED, STATE_SILENT, PlaceError, confirmed_elsewhere, connect_place,
                              place_records, place_summary, note_identity, migration_problems)

log = logging.getLogger(__name__)


def place_declarations(root) -> list[dict]:
    from ouroboros.extension_contributions import live_place_declarations, project_place_status
    from ouroboros.skill_loader import discover_skills
    from ouroboros.extension_ui_validation import validate_contributions
    from ouroboros.icon_geometry import icon_spec
    rows = live_place_declarations()
    for row in rows:
        row["readiness"] = {"text": "Connector on", "tone": "ok"}
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


def consciousness_off(root) -> bool:
    """Whether consciousness cannot focus in any place now (``presence_admission.focus_state`` is ``off``: the owner's
    switch known Off or a pending Panic). A switch this read cannot know is no claim: intake stays closed on it, the
    view says nothing."""
    from ouroboros.presence_admission import focus_state
    try:
        return focus_state(root) == "off"
    except Exception:
        log.warning("Consciousness switch unreadable for the place views", exc_info=True)
        return False


def place_view(root, place, *, declarations: list[dict] | None = None, allowance: dict | None = None,
               brief: bool = False, records=None, off: bool | None = None) -> dict[str, Any]:
    """Stored facts for status/launch; the owner surface also resolves connector capabilities. ``records`` is one
    ``places.place_records`` read shared by a list; its circle and description facts join the place here, the
    circle's words only in a brief view (the list names them once, ``places_view``). ``off`` is one
    ``consciousness_off`` read shared by a list (read here when absent): while it holds, a place that is not Paused
    only hears, and its view says so as ``only_hears`` (derived on every read, never stored)."""
    from ouroboros.consciousness_allowance import allowance_window
    from ouroboros.place_arrivals import account_candidates
    from ouroboros.places import place_model_route, place_room_index
    from ouroboros.route_choice import following_route
    if allowance is None and not brief:
        allowance = allowance_window(root, allow_stale=True)
    records = place_records(root) if records is None else records
    result = dict(place_summary(place))
    if (consciousness_off(root) if off is None else off) and place.state != STATE_PAUSED:
        result["only_hears"] = True
    # Legacy notes predate ids; use stable content+ordinal ids without rewriting history.
    result["notes"] = [{**n, "note_id": note_identity(place.place_id, i, place.notes[i])}
                       for i, n in enumerate(result["notes"])]
    circle = records.circle_context(place.place_id)
    model, layer = place_model_route(place, circle)
    if not model:
        model = following_route("consciousness")[0]
    elif model in ("main", "light"):
        from ouroboros.config import get_light_model
        from ouroboros.model_slots import _main_model
        model = get_light_model() if model == "light" else _main_model()
    result["model_route"] = {"value": model, "layer": layer}
    result["circle"] = circle if brief or circle is None else {
        key: circle[key] for key in ("circle_id", "placed_by", "placed_at")}
    result["description"] = records.description(place.place_id)
    cause = "place:" + place.place_id
    spend = next((item.get("usd") for item in (allowance or {}).get("spend_by_cause") or []
                  if item.get("cause") == cause), None)
    try:
        rooms = place_room_index(root, place.place_id)
    except Exception:
        log.warning("Place room index unavailable for %s", place.place_id, exc_info=True)
        rooms = None
    if brief:
        # Status and launch need labels, notes and open turns: one index pass, counts unknown when it is unavailable.
        counts = [room.get("unanswered_count") for room in rooms or []]
        result.update(label=said_once(place.label or place.provider), rooms=rooms,
                      unanswered=None if rooms is None or any(c is None for c in counts) else sum(counts),
                      last_activity_at=max((str(room.get("last_at") or "") for room in rooms or []), default=""),
                      activity_coverage="captured_place_events", spent_24h_usd=spend)
        return result
    activity = place_activity(root, place)
    result.update(label=said_once(place.label or place.provider),
                  rooms=rooms, unanswered=activity["unanswered"] if rooms is not None else None,
                  last_activity_at=activity["last_activity_at"],
                  activity_coverage="captured_place_events", spent_24h_usd=spend)
    rows = place_declarations(root) if declarations is None else declarations
    row = next((r for r in rows if r["skill"] == place.connector and (
        not place.contribution_id or r["id"] == place.contribution_id)), {})
    result.update(label=said_once(place.label or row.get("label") or place.provider),
                  reply_destination=place.destination,
                  destination=row.get("destination") or "vendor:" + place.provider,
                  icon=row.get("icon_spec") or {}, wake_suggestions=list(row.get("wakes") or []),
                  readiness=row.get("readiness") or {"text": "Connector unavailable", "tone": "warn"})
    from ouroboros.safety_config import get_safety_mode
    result.update(safety_mode=get_safety_mode("consciousness"), labels=row.get("labels") or {})
    result.update({key: row[key] for key in ("facts", "source", "settings") if key in row})
    result["account_candidates"] = account_candidates(root, place.place_id)
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


def places_view(root) -> dict:
    declarations = place_declarations(root)
    records = place_records(root)
    from ouroboros.consciousness_allowance import allowance_window
    allowance = allowance_window(root, allow_stale=True) or {}
    off = consciousness_off(root)
    return {"migration_problems": migration_problems(root),
            "places": [place_view(root, p, declarations=declarations, allowance=allowance, records=records, off=off)
                       for p in records.places.values()],
            "circles": [circle_summary(circle) for circle in records.circles],
            "connectors": [{"connector": r["skill"],
                            "label": r["label"], "icon": r.get("icon_spec") or {}, "destination": r["destination"],
                            "readiness": r.get("readiness") or {"text": "Connector unavailable", "tone": "warn"},
                            **{key: r[key] for key in ("facts", "source", "settings", "labels") if key in r},
                            "wakes": r.get("wakes") or [], "connected": any(p.connector == r["skill"] for p in records.places.values())}
                           for r in declarations]}


def place_activity(root, place) -> dict:
    """Indexed newest arrival and nonterminal focus count; no historical row walk."""
    from ouroboros.place_arrivals import activity
    return activity(root, place)
