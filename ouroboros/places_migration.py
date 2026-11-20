"""Import frozen Presence bindings once and finish their missing connector defaults on intake."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from ouroboros.places import (AUTHOR_CONNECTOR, AUTHOR_MIGRATION, FACT_LEGACY_ADDRESSED, WAKE_FACTS, NoteEntry,
                              Place, PlaceError, WakeItem, _decode, _with_choices, _write, place_summary,
                              places_path, wake_matches)
from ouroboros.utils import utc_now_iso


def import_recorded(document: dict) -> bool:
    """A provisional import already materialized its rows. Even a damaged proof cannot authorize a second scan."""
    migration = document.get("migration", {})
    return bool(migration.get("presence_bindings_v1") or "wake_import_open" in migration)


def import_bindings(root: Path, current: dict) -> dict:
    from ouroboros.extension_contributions import live_place_declarations
    from ouroboros.presence_bindings import PresenceBindingError, list_presence_bindings
    rows = dict(current["places"])
    # Only a row this import creates can receive its proof, even if an old choice has an orphan witness.
    choices = {key: {name: value for name, value in choice.items() if name != "wake_import"}
               for key, choice in current.get("place_choices", {}).items()}
    known = {bid for row in rows.values() for bid in row.get("legacy_binding_ids", ())}
    pending, problems = {}, []
    collision = False
    # A malformed whole binding document still requires repair; a bad member's
    # external profile must not prevent unrelated, valid bindings from importing.
    try:
        bindings = list_presence_bindings(root)
    except PresenceBindingError as exc:
        raise PlaceError("places_migration_failed", str(places_path(root)), f"Place migration needs repair: {exc}") from exc
    declarations = live_place_declarations() if bindings else []
    profiles = {}  # one observation per behavior for this import/preview, never shared across reads
    for binding in bindings:
        # A row this document already carries was imported once: its stored place is the record, and its source is
        # never re-read — a behavior skill that broke since cannot add a problem that would seal a fresh binding's
        # pending window and leave it a human review instead of its own proven completion.
        if binding.binding_id in known:
            continue
        try:
            name = binding.behavior_skill
            if name not in profiles:
                profiles[name] = _legacy_profile(root, name)
            place = _legacy_place(binding, profiles[name])
        except (PlaceError, OSError, ValueError, TypeError, AttributeError) as exc:
            code = getattr(exc, "code", "places_migration_source_invalid")
            detail = {
                "places_migration_source_missing": f"the behavior skill {binding.behavior_skill} is not installed",
                "places_migration_source_invalid": "its presence profile could not be read",
                "places_migration_instructions_missing": f"the behavior skill {binding.behavior_skill} has no instructions",
            }.get(code, "the old connection could not be imported")
            problems.append({"binding_id": binding.binding_id,
                             "code": code, "detail": detail})
            continue
        if place.place_id in rows:
            collision = True  # an unrelated row currently prevents the old import of this id
            continue
        matches = matching_declarations(place, declarations)
        place = replace(place, revision=current["revision"] + 1, wakes=_with_defaults(place.wakes, matches))
        rows[place.place_id] = place_summary(place)
        choices.pop(place.place_id, None)
        if not matches:
            pending[place.place_id] = {"connected_revision": place.revision, "wake_import": {
                "revision": place.revision, "wakes": [asdict(wake) for wake in place.wakes]}}
    migration = {"presence_bindings_v1": True, "at": utc_now_iso(), "problems": problems}
    if pending and not problems and not collision:
        migration.update(presence_bindings_v1=False, wake_import_open=True)
        choices.update(pending)
    return _with_choices({**current, "places": rows, "revision": current["revision"] + 1,
                          "migration": migration}, choices)


def declared_wake_facts(declarations) -> list[str]:
    """Default ids in contribution order. Each declaration has passed validate_contributions."""
    return list(dict.fromkeys(fact for row in declarations for fact in row.get("wakes", ())))


def merged_declared_facts(declarations):
    """The ordered union of effective vocabularies, independent of saved wakes. Omitted ``facts`` means
    the core kinds for that declaration; if all omit it, leave the sheet's ordinary core fallback in place."""
    if not any("facts" in declaration for declaration in declarations):
        return None
    facts, seen = [], set()
    for declaration in declarations:
        for item in declaration.get("facts", WAKE_FACTS):
            fact = item.get("id") if isinstance(item, dict) else item
            if isinstance(fact, str) and fact and fact not in seen:
                seen.add(fact)
                facts.append(item)
    return facts


def matching_declarations(place: Place, declarations) -> list[dict]:
    """A legacy binding names one provider of a connector, not all destinations it supports."""
    return [row for row in declarations if row.get("provides") == "place" and row.get("skill") == place.connector
            and row.get("destination", "").removeprefix("vendor:").lower() == place.provider
            and (not place.contribution_id or row["id"] == place.contribution_id)]


def _with_defaults(existing, declarations):
    chosen = {wake.fact for wake in existing}
    return existing + tuple(WakeItem(fact, AUTHOR_CONNECTOR) for fact in declared_wake_facts(declarations)
                            if fact not in chosen)


def _legacy_profile(root: Path, name: str) -> dict:
    from ouroboros.skill_loader import find_skill
    skill = find_skill(root, name)
    if skill is None:
        raise PlaceError("places_migration_source_missing", name)
    raw = skill.manifest.raw_extra.get("presence") or {}
    if not isinstance(raw, dict):
        raise PlaceError("places_migration_source_invalid", name)
    topics = raw.get("context_topics") or []
    if not isinstance(topics, list) or any(not isinstance(topic, str) for topic in topics):
        raise PlaceError("places_migration_source_invalid", "context_topics")
    instructions = raw.get("instructions", "")
    if raw.get("instructions_file"):
        source = (Path(skill.skill_dir) / raw["instructions_file"]).resolve()
        if not source.is_relative_to(Path(skill.skill_dir).resolve()):
            raise PlaceError("places_migration_source_invalid", str(source))
        instructions = source.read_text(encoding="utf-8")
    if not isinstance(instructions, str) or not instructions.strip():
        raise PlaceError("places_migration_instructions_missing", name)
    state_path = root / "state" / "skills" / name / "presence_profile_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if not isinstance(state, dict):
        raise PlaceError("places_migration_source_invalid", str(state_path))
    defaults = raw.get("runtime_defaults") or {}
    overrides = state.get("runtime_overrides") or {}
    if not isinstance(defaults, dict) or not isinstance(overrides, dict):
        raise PlaceError("places_migration_source_invalid", str(state_path))
    slot = overrides.get("model_slot") or defaults.get("model_slot") or "main"
    # Keep the old slot choice as a symbolic route; an absent override used Main,
    # not the (possibly different) consciousness model.
    return {"notes": (NoteEntry(instructions, AUTHOR_MIGRATION),), "context_topics": tuple(topics),
            "model": str(state.get("model") or slot), "working_folder": str(state.get("workspace_root") or "")}


def _legacy_place(binding, profile: dict) -> Place:
    origin = binding.origin
    return Place(
        place_id=binding.binding_id, connector=binding.transport_skill, provider=origin.transport,
        account_id=origin.account_id, conversation_id="" if origin.conversation_id == "*" else origin.conversation_id,
        thread_id=origin.thread_id, label=origin.transport,
        state="active" if binding.enabled else "paused", wakes=(WakeItem(FACT_LEGACY_ADDRESSED, AUTHOR_MIGRATION),),
        legacy_binding_ids=(binding.binding_id,), destination=binding.destination.__dict__,
        scope="account" if origin.conversation_id == "*" else "conversation", revision=1,
        **profile,
    )


def _open(migration):
    return migration.get("wake_import_open") is True and migration.get("presence_bindings_v1") is False


def _pending(migration, place, choice):
    marker = choice.get("wake_import")
    return (_open(migration) and bool(place.legacy_binding_ids) and 0 < place.connected_revision <= place.revision
            and "wakes_reviewed" not in choice
            and isinstance(marker, dict) and set(marker) == {"revision", "wakes"}
            and type(marker["revision"]) is int and marker["revision"] == place.revision
            and place.wakes == (WakeItem(FACT_LEGACY_ADDRESSED, AUTHOR_MIGRATION),)
            and marker["wakes"] == [asdict(wake) for wake in place.wakes])


def wake_review(records, place, suggestions):
    choice = records.choices.get(place.place_id, {})
    if choice.get("connected_revision") != place.connected_revision:
        choice = {}
    if _pending(records.migration, place, choice):
        return "pending"
    if choice.get("wakes_reviewed") is True:
        return None
    if place.legacy_binding_ids and set(suggestions).difference(wake.fact for wake in place.wakes):
        return "ambiguous"
    return None


def after_place_write(current, changed):
    """Every older Places setter calls load_places before its edit, even a same-list save which writes nothing.
    With v1 open that call replaces the migration object under the same store lock. A restart on this body
    preserves it. Seal before detaching a legacy binding so the old importer cannot reconnect it. The same
    precaution at first import preserves failed bindings and their original diagnostics."""
    if changed is None or "wake_import_open" not in current.get("migration", {}):
        return changed
    choices = dict(changed.get("place_choices", {}))
    removed = current["places"].keys() - changed["places"].keys()
    seal = any(current["places"][key].get("legacy_binding_ids") for key in removed)
    pending = False
    for key, choice in list(choices.items()):
        if "wake_import" not in choice:
            continue
        if not seal and key in current["places"] and key in changed["places"]:
            old = _decode(current["places"][key], current.get("connections", {}).get(key))
            new = _decode(changed["places"][key], changed.get("connections", {}).get(key))
            if (choice["connected_revision"] == old.connected_revision == new.connected_revision
                    and _pending(current["migration"], old, choice) and new.wakes == old.wakes):
                pending = True
                if new.revision != old.revision:
                    choices[key] = {**choice, "wake_import": {**choice["wake_import"], "revision": new.revision}}
                continue
        rest = {k: v for k, v in choice.items() if k != "wake_import"}
        if rest.keys() == {"connected_revision"}:
            del choices[key]
        else:
            choices[key] = rest
    if not pending:
        migration = {**current["migration"], "presence_bindings_v1": True}
        del migration["wake_import_open"]
        changed = {**changed, "migration": migration}
    return _with_choices(changed, choices)


def complete_wake_import(root, records, place) -> bool:
    """True if admission must re-read its snapshot after a completion or a proven-lost one.

    The write never guesses that the snapshot's revision is still current: it applies the declared defaults to the
    record as it stands under the store lock, so an edit racing the first event (a note, a state) neither fails a
    compare-and-set nor defers the import — and the event's own wake match — to a later event. ``finish`` bounds
    what may be applied: the same connection, the import still open and this place still pending on the current
    record. A proof that fails — an explicit human save of the list, which always wins, a reconnect, a window an
    older writer sealed, a concurrent disconnect — applies nothing; admission re-reads and proceeds on the record
    as it stands.
    """
    if not _pending(records.migration, place, records.choices.get(place.place_id, {})):
        return False
    from ouroboros.extension_contributions import live_place_declarations
    matches = matching_declarations(place, live_place_declarations())
    if not matches:
        return False

    def finish(choice, document):
        # An older writer may have closed the import without moving this place's revision.
        key = place.place_id
        actual = _decode(document["places"][key], document.get("connections", {}).get(key))
        if actual.connected_revision != place.connected_revision or not _pending(document.get("migration", {}), actual, choice):
            raise PlaceError("place_revision_conflict", "wake_import")
        return {name: value for name, value in choice.items() if name != "wake_import"}

    def completed(old):
        if old is None:  # a concurrent disconnect: the fresh read answers place_not_connected
            raise PlaceError("place_revision_conflict", "wake_import")
        return replace(old, wakes=_with_defaults(old.wakes, matches))

    try:
        _write(root, place.place_id, completed, choose=finish)
    except PlaceError as exc:
        if exc.code != "place_revision_conflict":
            raise
    return True


def wake_import_declarations_pending(root, records, place, facts) -> bool:
    """Could completing this proven import change a wake miss?"""
    from ouroboros.extension_contributions import place_declarations_for_retry

    if not _pending(records.migration, place, records.choices.get(place.place_id, {})):
        return False
    declarations = place_declarations_for_retry(root, place)
    if not declarations:
        return False
    possible = replace(place, wakes=_with_defaults(place.wakes, declarations))
    return bool(wake_matches(possible, facts))
