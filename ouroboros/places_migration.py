"""Read-only legacy Presence import. Source bytes and old task authority stay intact."""
from __future__ import annotations

import json
from pathlib import Path

from ouroboros.places import (AUTHOR_CONNECTOR, AUTHOR_MIGRATION, FACT_LEGACY_ADDRESSED,
                              WAKE_FACTS, NoteEntry, Place, PlaceError, WakeItem)


def legacy_places(root: Path) -> tuple[list[Place], list[dict]]:
    from ouroboros.presence_bindings import list_presence_bindings
    result, problems = [], []
    # A malformed whole binding document still requires repair; a bad member's
    # external profile must not prevent unrelated, valid bindings from importing.
    for binding in list_presence_bindings(root):
        try:
            result.append(_legacy_place(root, binding))
        except (PlaceError, OSError, ValueError, TypeError, AttributeError) as exc:
            code = getattr(exc, "code", "places_migration_source_invalid")
            detail = {
                "places_migration_source_missing": f"the behavior skill {binding.behavior_skill} is not installed",
                "places_migration_source_invalid": "its presence profile could not be read",
                "places_migration_instructions_missing": f"the behavior skill {binding.behavior_skill} has no instructions",
            }.get(code, "the old connection could not be imported")
            problems.append({"binding_id": binding.binding_id,
                             "code": code, "detail": detail})
    return result, problems


def _legacy_place(root: Path, binding) -> Place:
    from ouroboros.skill_loader import find_skill
    skill = find_skill(root, binding.behavior_skill)
    if skill is None:
        raise PlaceError("places_migration_source_missing", binding.behavior_skill)
    raw = skill.manifest.raw_extra.get("presence") or {}
    if not isinstance(raw, dict):
        raise PlaceError("places_migration_source_invalid", binding.behavior_skill)
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
        raise PlaceError("places_migration_instructions_missing", binding.behavior_skill)
    state_path = root / "state" / "skills" / binding.behavior_skill / "presence_profile_state.json"
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
    model = str(state.get("model") or slot)
    transport = find_skill(root, binding.transport_skill)
    wakes = [WakeItem(FACT_LEGACY_ADDRESSED, AUTHOR_MIGRATION)]
    if transport is not None:
        for contribution in transport.manifest.contributes:
            if contribution.get("provides") == "place":
                for item in contribution.get("wakes") or []:
                    fact = item.get("fact", item.get("kind")) if isinstance(item, dict) else item
                    if fact in WAKE_FACTS and fact not in {w.fact for w in wakes}:
                        wakes.append(WakeItem(fact, AUTHOR_CONNECTOR))
    origin = binding.origin
    return Place(
        place_id=binding.binding_id, connector=binding.transport_skill, provider=origin.transport,
        account_id=origin.account_id, conversation_id="" if origin.conversation_id == "*" else origin.conversation_id,
        thread_id=origin.thread_id, label=origin.transport,
        state="active" if binding.enabled else "paused", wakes=tuple(wakes),
        notes=(NoteEntry(instructions, AUTHOR_MIGRATION),),
        context_topics=tuple(raw.get("context_topics") or ()), model=model,
        working_folder=str(state.get("workspace_root") or ""),
        legacy_binding_ids=(binding.binding_id,), destination=binding.destination.__dict__,
        scope="account" if origin.conversation_id == "*" else "conversation", revision=1,
    )
