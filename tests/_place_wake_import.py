"""Synthetic legacy Places stores and old-reader observations shared by migration consumers."""
import json
from types import SimpleNamespace

from ouroboros import places as p
from ouroboros.presence_bindings import PresenceBinding, PresenceEndpoint
from tests.frozen_presence_bindings import save_presence_binding


def document(root):
    return json.loads(p.places_path(root).read_text(encoding="utf-8"))


def base_reader(value):
    """The place reader of a body predating circles (3e05 ``_document``/``_decode``): strict ``Place(**row)``, the
    connection witness, no knowledge of ``circles`` or ``place_choices``."""
    assert value["schema_version"] == 1 and isinstance(value["places"], dict)
    out = {}
    for key, row in value["places"].items():
        data = dict(row)
        connection = value.get("connections", {}).get(key)
        data["connected_revision"] = (connection["connected_revision"] if isinstance(connection, dict)
                                      and connection.get("written_revision") == data.get("revision", 0) else data.get("revision", 0))
        for name, cls in (("wakes", p.WakeItem), ("notes", p.NoteEntry), ("owner_accounts", p.OwnerAccount)):
            data[name] = tuple(cls(**item) for item in data.get(name, ()))
        for name in ("context_topics", "legacy_binding_ids"):
            data[name] = tuple(data.get(name, ()))
        out[key] = p.Place(**data)
    return out


def legacy_wake_install(root, monkeypatch, *, published=(), ids=("b" * 32,)):
    """Frozen bindings and the transport's published place declarations."""
    behavior = SimpleNamespace(skill_dir=root, manifest=SimpleNamespace(raw_extra={
        "presence": {"instructions": "Keep these words.", "context_topics": ["social"]}}))
    transport = SimpleNamespace(manifest=SimpleNamespace(contributes=[]))
    skills = {"mind": behavior, "slack-bridge": transport}
    monkeypatch.setattr("ouroboros.skill_loader.find_skill", lambda _root, name: skills.get(name))
    for key in ids:
        origin = PresenceEndpoint("slack", "team", key[:8])
        save_presence_binding(root, PresenceBinding(key, "slack-bridge", "mind", origin, origin))
    rows = list(published)
    monkeypatch.setattr("ouroboros.extension_registry_state.live_contribution_projection", lambda: rows)
    return rows


def wake_declaration(*facts, connector="slack-bridge"):
    return {"provides": "place", "id": "place", "skill": connector, "label": "Slack",
            "destination": "vendor:Slack", "wakes": list(facts)}


def admit_live(root, place):
    from ouroboros.gateway.host_service_places import _admit_presence
    return _admit_presence(SimpleNamespace(data_dir=root), place.connector, place.place_id,
                           provider=place.provider, account_id=place.account_id, conversation_id=place.conversation_id)


def old_load_places(root, frozen_rows):
    """The v1 reader before every historical Places edit (8a674efe283 through 8ea6b46f721).
    It checks the v1 flag again under the lock, keeps known places, then replaces the migration object."""
    def migrate(value):
        if value.get("migration", {}).get("presence_bindings_v1"):
            return value
        rows = dict(value["places"])
        known = {bid for row in rows.values() for bid in row.get("legacy_binding_ids", ())}
        for row in frozen_rows:
            if not known.intersection(row["legacy_binding_ids"]):
                rows.setdefault(row["place_id"], row)
        return {**value, "places": rows, "revision": value["revision"] + 1,
                "migration": {"presence_bindings_v1": True, "problems": [], "at": "old import"}}
    if not document(root)["migration"].get("presence_bindings_v1"):
        p.update_json_locked(p.places_path(root), migrate)
    return base_reader(document(root))
