"""Owner-facing place views from current connector declarations and stored choices."""
from __future__ import annotations

import logging
from typing import Any
from ouroboros.places import STATE_SILENT, PlaceError, connect_place, load_places, place_summary, note_identity, migration_problems

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


def connect_declared_place(root, connector: str, *, account_id: str = "", conversation_id: str = "", label: str = ""):
    if not isinstance(label, str):
        raise PlaceError("place_invalid_string", "label")
    declarations = [row for row in place_declarations(root) if row["skill"] == connector]
    if len(declarations) != 1:
        raise PlaceError("place_declaration_unavailable", "connector", "Enable a reviewed connector with one place declaration.")
    row = declarations[0]
    provider = str(row["destination"]).removeprefix("vendor:").lower()
    title = f'{row["label"]} · {label.strip()}' if label.strip() else row["label"]
    return connect_place(root, connector=connector, provider=provider, label=title, state=STATE_SILENT,
                         account_id=account_id, conversation_id=conversation_id, scope=row.get("scope", "account"),
                         contribution_id=row["id"], wakes=row.get("wakes", []))


def place_view(root, place, *, declarations: list[dict] | None = None, allowance: dict | None = None,
               brief: bool = False) -> dict[str, Any]:
    """Stored facts for status/launch; the owner surface also resolves connector capabilities."""
    from ouroboros.consciousness_allowance import allowance_window
    from ouroboros.place_arrivals import account_candidates
    from ouroboros.places import place_room_index
    if allowance is None and not brief:
        allowance = allowance_window(root, allow_stale=True)
    result = dict(place_summary(place))
    # Legacy notes predate ids; use stable content+ordinal ids without rewriting history.
    result["notes"] = [{**n, "note_id": note_identity(place.place_id, i, place.notes[i])}
                       for i, n in enumerate(result["notes"])]
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
        result.update(label=place.label or place.provider, rooms=rooms,
                      unanswered=None if rooms is None or any(c is None for c in counts) else sum(counts),
                      last_activity_at=max((str(room.get("last_at") or "") for room in rooms or []), default=""),
                      activity_coverage="captured_place_events", spent_24h_usd=spend)
        return result
    activity = place_activity(root, place)
    result.update(label=place.label or place.provider,
                  rooms=rooms, unanswered=activity["unanswered"] if rooms is not None else None,
                  last_activity_at=activity["last_activity_at"],
                  activity_coverage="captured_place_events", spent_24h_usd=spend)
    rows = place_declarations(root) if declarations is None else declarations
    row = next((r for r in rows if r["skill"] == place.connector and (
        not place.contribution_id or r["id"] == place.contribution_id)), {})
    result.update(label=place.label or row.get("label") or place.provider,
                  reply_destination=place.destination,
                  destination=row.get("destination") or "vendor:" + place.provider,
                  icon=row.get("icon_spec") or {}, wake_suggestions=list(row.get("wakes") or []),
                  readiness=row.get("readiness") or {"text": "Connector unavailable", "tone": "warn"})
    from ouroboros.safety_config import get_safety_mode
    result.update(safety_mode=get_safety_mode("consciousness"), labels=row.get("labels") or {})
    result.update({key: row[key] for key in ("facts", "source", "settings") if key in row})
    result["account_candidates"] = account_candidates(root, place.place_id)
    return result


def places_view(root) -> dict:
    declarations = place_declarations(root)
    places = load_places(root)
    from ouroboros.consciousness_allowance import allowance_window
    allowance = allowance_window(root, allow_stale=True) or {}
    return {"migration_problems": migration_problems(root), "places": [place_view(root, p, declarations=declarations, allowance=allowance) for p in places],
            "connectors": [{"connector": r["skill"],
                            "label": r["label"], "icon": r.get("icon_spec") or {}, "destination": r["destination"],
                            "readiness": r.get("readiness") or {"text": "Connector unavailable", "tone": "warn"},
                            **{key: r[key] for key in ("facts", "source", "settings", "labels") if key in r},
                            "wakes": r.get("wakes") or [], "connected": any(p.connector == r["skill"] for p in places)}
                           for r in declarations]}


def place_activity(root, place) -> dict:
    """Indexed newest arrival and nonterminal focus count; no historical row walk."""
    from ouroboros.place_arrivals import activity
    return activity(root, place)
