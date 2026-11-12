import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import places as p
from ouroboros.presence_bindings import PresenceBinding, PresenceEndpoint
from tests.frozen_presence_bindings import save_presence_binding


def paused_inventory(root, kind):
    """A stored inventory without a migration stamp, shared with restart consumers."""
    path = p.places_path(root)
    if kind == "absent":
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "unreadable":
        path.write_text("{broken", encoding="utf-8")
        return "Paused places: unknown (place records could not be read)."
    rows = {key: p.place_summary(p.Place(key, "custom-bridge", "custom", label=label, state=state))
            for key, label, state in [("a", "Research room", "paused"), ("b", "", "paused"),
                                     ("c", "Active room", "active"), ("d", "Silent room", "silent")]}
    if kind == "empty":
        rows = {}
    path.write_text(json.dumps({"schema_version": 1, "revision": 4, "places": rows}), encoding="utf-8")
    return "Paused places (no writing there; tasks are not affected): Research room, b." if rows else ""


@pytest.mark.parametrize("kind", ["paused", "empty", "unreadable"])
def test_paused_places_sentence_reads_without_migration_or_writes(tmp_path, monkeypatch, kind):
    expected = paused_inventory(tmp_path, kind)
    path = p.places_path(tmp_path)
    before = path.read_bytes() if path.exists() else None
    monkeypatch.setattr(p, "migrate_places", lambda *_a: pytest.fail("display migrated places"))
    monkeypatch.setattr(p, "update_json_locked", lambda *_a, **_k: pytest.fail("display wrote places"))
    assert p.paused_places_sentence(tmp_path) == expected
    assert (path.read_bytes() if path.exists() else None) == before


@pytest.mark.parametrize("migration_saved", [False, True])
def test_paused_places_legacy_warning_ends_after_persisted_migration(tmp_path, monkeypatch, migration_saved):
    expected = paused_inventory(tmp_path, "paused")
    binding = PresenceBinding("a" * 32, "slack-bridge", "missing-mind", PresenceEndpoint("slack", "team", "*"),
                              PresenceEndpoint("slack", "team", "elsewhere"), False)
    save_presence_binding(tmp_path, binding)
    legacy = tmp_path / "state" / "presence_bindings.json"
    legacy_bytes = legacy.read_bytes()
    monkeypatch.setattr("ouroboros.skill_loader.find_skill", lambda *_a: None)
    if migration_saved:
        document = p.migrate_places(tmp_path)
        assert document["migration"]["presence_bindings_v1"] and document["migration"]["problems"]
    else:
        expected += " Paused legacy places: unknown (some old connections could not be read)."
    path = p.places_path(tmp_path)
    before = path.read_bytes()
    monkeypatch.setattr(p, "migrate_places", lambda *_a: pytest.fail("display persisted migration"))
    monkeypatch.setattr(p, "update_json_locked", lambda *_a, **_k: pytest.fail("display wrote places"))
    assert p.paused_places_sentence(tmp_path) == expected
    assert p.paused_places_sentence(tmp_path) == expected
    assert path.read_bytes() == before and legacy.read_bytes() == legacy_bytes


def connect(root, **kw):
    return p.connect_place(root, connector="slack-bridge", provider="slack", account_id="team", **kw)


def test_store_choices_revision_and_refresh(tmp_path):
    place = connect(tmp_path, wakes=[p.FACT_MENTION])
    changed = p.edit_place(tmp_path, place.place_id, "add_note", author=p.AUTHOR_OWNER, text="Owner words\nverbatim.")
    changed = p.edit_place(tmp_path, place.place_id, "set_state", author=p.AUTHOR_OUROBOROS, state="silent")
    again = connect(tmp_path, wakes=[p.FACT_ANY_MESSAGE], label="Updated connector")
    assert again.notes == changed.notes and again.wakes == place.wakes and again.state == "silent"
    assert p.get_place(tmp_path, place.place_id) == again
    with pytest.raises(p.PlaceError, match="Current revision"):
        p.edit_place(tmp_path, place.place_id, "set_model", author="owner", model="x", expected_revision=1)
    final = p.edit_place(tmp_path, place.place_id, "set_model", author="owner", model="x", expected_revision=again.revision)
    assert final.model == "x"


def test_facts_and_empty_wakes(tmp_path):
    place = connect(tmp_path, wakes=[p.FACT_MENTION])
    assert p.wake_matches(place, [p.FACT_MENTION])
    assert not p.wake_matches(place, ["please mention me"])
    assert not p.wake_matches(place, [p.FACT_MENTION, p.FACT_FROM_BOT])
    assert p.wake_matches(replace(place, wakes=place.wakes + (p.WakeItem(p.FACT_FROM_BOT, "owner"),)),
                          [p.FACT_MENTION, p.FACT_FROM_BOT])
    assert not p.wake_matches(replace(place, wakes=()), [p.FACT_LEGACY_ADDRESSED])


def test_resolution_and_owner_confirmation(tmp_path):
    broad = connect(tmp_path)
    narrow = connect(tmp_path, conversation_id="room", scope="conversation")
    assert p.resolve_place(tmp_path, connector="slack-bridge", provider="slack", account_id="team", conversation_id="room") == narrow
    assert p.resolve_place(tmp_path, connector="slack-bridge", provider="slack", account_id="team", conversation_id="other") == broad
    assert p.resolve_place(tmp_path, connector="other", binding_id=narrow.place_id) is None
    with pytest.raises(p.PlaceError, match="owner's door"):
        p.edit_place(tmp_path, narrow.place_id, "confirm_owner_account", author="ouroboros", platform_actor_id="anton")
    confirmed = p.edit_place(tmp_path, narrow.place_id, "confirm_owner_account", author="owner", platform_actor_id="anton")
    assert p.is_owner_account(confirmed, "anton", "team")
    assert not p.is_owner_account(confirmed, "anton", "other")
    assert not p.is_owner_account(confirmed, "stranger", "team")


@pytest.mark.parametrize("enabled", [True, False])
def test_migration_exact_sources_and_idempotence(tmp_path, monkeypatch, enabled):
    root = tmp_path
    behavior = SimpleNamespace(skill_dir=root, manifest=SimpleNamespace(raw_extra={"presence": {
        "instructions": "Keep these exact words.\n", "context_topics": ["social"],
        "runtime_defaults": {"model_slot": "main"}}}))
    connector = SimpleNamespace(manifest=SimpleNamespace(contributes=[{"provides": "place", "wakes": ["group_dm"]}]))
    monkeypatch.setattr("ouroboros.skill_loader.find_skill", lambda _root, name: behavior if name == "mind" else connector)
    state = root / "state" / "skills" / "mind" / "presence_profile_state.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"runtime_overrides": {"model_slot": "light"}, "workspace_root": "/work"}))
    binding = PresenceBinding("a" * 32, "slack-bridge", "mind", PresenceEndpoint("slack", "team", "*"),
                              PresenceEndpoint("slack", "team", "elsewhere"), enabled)
    save_presence_binding(root, binding)
    source = (root / "state" / "presence_bindings.json").read_bytes()
    assert p.paused_places_sentence(root) == (
        "" if enabled else "Paused places (no writing there; tasks are not affected): slack.")
    assert not p.places_path(root).exists()
    assert (root / "state" / "presence_bindings.json").read_bytes() == source
    place, = p.load_places(root)
    assert place.state == ("active" if enabled else "paused") and place.legacy_binding_ids == (binding.binding_id,)
    assert place.notes[0] == p.NoteEntry("Keep these exact words.\n", "migration")
    assert place.context_topics == ("social",) and place.working_folder == "/work" and place.model == "light"
    assert place.destination["conversation_id"] == "elsewhere"
    assert {w.fact for w in place.wakes} == {"legacy_addressed", "group_dm"}
    assert "connected_revision" not in json.loads(p.places_path(root).read_text(encoding="utf-8"))["places"][place.place_id]
    saved = p.edit_place(root, place.place_id, "set_state", author="owner", state="silent")
    assert p.load_places(root) == [saved]
    assert p.paused_places_sentence(root) == ""
    assert (root / "state" / "presence_bindings.json").read_bytes() == source


def test_empty_and_malformed_sources_preserved(tmp_path):
    assert p.load_places(tmp_path) == []
    assert p.load_places(tmp_path) == []
    other = tmp_path / "broken"
    path = other / "state" / "presence_bindings.json"
    path.parent.mkdir(parents=True)
    path.write_text("{broken")
    assert p.paused_places_sentence(other) == "Paused places: unknown (place records could not be read)."
    with pytest.raises(p.PlaceError, match="repair"):
        p.load_places(other)
    assert path.read_text(encoding="utf-8") == "{broken"
    assert not p.places_path(other).exists()
    path.unlink()
    assert p.load_places(other) == []


def test_chat_identity_matches_old_runner():
    from ouroboros.presence_runner import _stable_numeric_id
    assert p.conversation_chat_id("slack:team:room:0") == _stable_numeric_id("presence-conversation", "slack:team:room:0")


def test_declared_targets_keep_active_siblings_distinct(tmp_path):
    active = connect(tmp_path, conversation_id="a")
    closed = connect(tmp_path, conversation_id="b")
    closed = p.edit_place(tmp_path, closed.place_id, "set_state", author="owner", state="silent")
    assert p.place_for_target(tmp_path, connector="slack-bridge", targets={"to": ["a"]}) == (active, True)
    assert p.place_for_target(tmp_path, connector="slack-bridge", targets={"to": ["b"]}) == (closed, True)
    assert p.place_for_target(tmp_path, connector="slack-bridge", targets={"to": ["unknown"]}) == (None, False)
    assert p.place_for_target(tmp_path, connector="slack-bridge", targets={}) == (None, False)


def test_owner_door_settings_do_not_create_places(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"TELEGRAM_BOT_TOKEN": "test-only", "OWNER_CHAT_ID": "42"}))
    assert p.load_places(tmp_path) == []
    connected = connect(tmp_path)
    assert p.load_places(tmp_path) == [connected]


@pytest.mark.parametrize("thread_id,expected", [("topic", "thread"), ("", "room"), ("other", "room")])
def test_thread_scope_resolves_only_the_exact_thread(tmp_path, monkeypatch, thread_id, expected):
    room = p.Place("room", "slack-bridge", "slack", "team", "room")
    thread = replace(room, place_id="thread", thread_id="topic")
    monkeypatch.setattr(p, "load_places", lambda _root: [room, thread])
    found = p.resolve_place(tmp_path, connector="slack-bridge", provider="slack", account_id="team",
                            conversation_id="room", thread_id=thread_id)
    assert found.place_id == expected
    assert p.resolve_place(tmp_path, connector="slack-bridge", binding_id="thread") == thread


def test_place_storage_keys_are_rollback_compatible(tmp_path):
    place = connect(tmp_path)
    document = json.loads(p.places_path(tmp_path).read_text(encoding="utf-8"))
    row = document["places"][place.place_id]
    assert set(row) == {
        "place_id", "connector", "provider", "account_id", "conversation_id", "label", "state", "wakes",
        "notes", "owner_accounts", "working_folder", "model", "context_topics", "legacy_binding_ids", "revision",
        "thread_id", "destination", "scope", "contribution_id",
    }
    assert set(p.place_summary(place)) == set(row)
    assert document["connections"][place.place_id] == {
        "connected_revision": place.revision, "written_revision": place.revision}
    assert p._decode(row).connected_revision == place.revision


@pytest.mark.parametrize("wildcard_id,account_id", [("aa-wild", "zz-account"), ("zz-wild", "aa-account")])
def test_outgoing_wildcard_cannot_shadow_concrete_ingress(tmp_path, wildcard_id, account_id):
    wildcard = p.connect_place(tmp_path, connector="slack-bridge", provider="slack", account_id="*", place_id=wildcard_id)
    concrete = connect(tmp_path, place_id=account_id)
    assert p.resolve_place(tmp_path, connector="slack-bridge", provider="slack", account_id="team") == concrete
    assert p.resolve_place(tmp_path, connector="slack-bridge", provider="slack", account_id="other") is None
    assert p.resolve_place(tmp_path, connector="slack-bridge", binding_id=wildcard.place_id) == wildcard


def test_connect_state_is_atomic_and_repeat_preserves_owner_choice(tmp_path):
    active = connect(tmp_path, conversation_id="default")
    silent = connect(tmp_path, conversation_id="silent", state=p.STATE_SILENT)
    assert active.state == "active" and p.get_place(tmp_path, silent.place_id).state == "silent"
    changed = p.edit_place(tmp_path, silent.place_id, "set_state", author="owner", state="paused")
    assert connect(tmp_path, conversation_id="silent", state=p.STATE_SILENT) == changed
    with pytest.raises(p.PlaceError):
        connect(tmp_path, conversation_id="invalid", state="invented")
    assert len(p.load_places(tmp_path)) == 2


def test_reconnect_cannot_accept_a_previous_incarnations_revision(tmp_path):
    first = connect(tmp_path)
    p.edit_place(tmp_path, first.place_id, "disconnect", author="owner", expected_revision=first.revision)
    again = connect(tmp_path)
    assert again.place_id == first.place_id and again.revision > first.revision
    with pytest.raises(p.PlaceError, match="Current revision"):
        p.edit_place(tmp_path, again.place_id, "set_model", author="owner", model="stale", expected_revision=first.revision)
    fresh = p.edit_place(tmp_path, again.place_id, "set_model", author="owner", model="fresh", expected_revision=again.revision)
    assert fresh.model == "fresh"


def test_revision_is_per_record_and_disconnect_compares_it(tmp_path):
    first, other = connect(tmp_path, conversation_id="one"), connect(tmp_path, conversation_id="two")
    other = p.edit_place(tmp_path, other.place_id, "set_model", author="owner", model="other")
    saved = p.edit_place(tmp_path, first.place_id, "set_model", author="owner", model="one", expected_revision=first.revision)
    assert saved.revision > other.revision and p.get_place(tmp_path, other.place_id) == other
    with pytest.raises(p.PlaceError, match="Current revision"):
        p.edit_place(tmp_path, first.place_id, "disconnect", author="owner", expected_revision=first.revision)
    assert p.get_place(tmp_path, first.place_id) == saved
    assert p.edit_place(tmp_path, first.place_id, "disconnect", author="owner", expected_revision=saved.revision) is None
    assert p.load_places(tmp_path) == [other]


@pytest.mark.parametrize("revision", [True, "2", -1])
def test_invalid_revision_is_not_an_unconditional_write(tmp_path, revision):
    place = connect(tmp_path)
    with pytest.raises(p.PlaceError, match="place_revision_invalid"):
        p.edit_place(tmp_path, place.place_id, "disconnect", author="owner", expected_revision=revision)
    assert p.get_place(tmp_path, place.place_id) == place


@pytest.mark.parametrize("fact", ["signal.words", "signal.a-b_2", "signal." + "a" * 40])
def test_signal_choices_persist_and_matching_keeps_bot_gate(tmp_path, monkeypatch, fact):
    monkeypatch.setattr("ouroboros.extension_registry_state.live_contribution_projection", lambda: [{
        "provides": "place", "skill": "slack-bridge", "facts": [{"id": fact, "label": "Chosen words"}]}])
    place = connect(tmp_path)
    saved = p.edit_place(tmp_path, place.place_id, "add_wake", author="owner", fact=fact)
    assert saved.wakes == (p.WakeItem(fact, "owner"),)
    assert p.wake_matches(saved, [fact]) and not p.wake_matches(saved, ["signal.other"])
    assert not p.wake_matches(saved, [fact, "from_bot"])
    bots = p.edit_place(tmp_path, place.place_id, "add_wake", author="owner", fact="from_bot")
    assert p.wake_matches(bots, [fact, "from_bot"])
    assert p.wake_matches(bots, ["from_bot"]), "from_bot is itself an OR wake item"
    monkeypatch.setattr("ouroboros.extension_registry_state.live_contribution_projection", lambda: [])
    assert p.get_place(tmp_path, place.place_id) == bots  # off/removed connectors do not corrupt storage
    with pytest.raises(p.PlaceError, match="place_wake_undeclared"):
        p.edit_place(tmp_path, place.place_id, "add_wake", author="owner", fact="signal.undeclared")
    with pytest.raises(p.PlaceError, match="place_wake_undeclared"):
        p.edit_place(tmp_path, place.place_id, "set_wakes", author="owner", wakes=["signal.undeclared"])
    kept = p.edit_place(tmp_path, place.place_id, "set_wakes", author="owner", wakes=[fact])
    assert kept.wakes == saved.wakes
    assert not p.edit_place(tmp_path, place.place_id, "remove_wake", author="owner", fact=fact).wakes


@pytest.mark.parametrize("fact", ["signal.", None, {}])
def test_invalid_signal_never_enters_storage(tmp_path, fact):
    place = connect(tmp_path)
    with pytest.raises(p.PlaceError, match="place_wake_invalid"):
        p.edit_place(tmp_path, place.place_id, "add_wake", author="owner", fact=fact)
    assert p.load_places(tmp_path) == [place]


# ---- circles: sibling metadata in the same document, one locked writer -------------------------------------------
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


def test_a_choice_moves_the_place_revision_once_and_an_unchanged_choice_writes_nothing(tmp_path):
    place = connect(tmp_path)
    circle = p.create_circle(tmp_path, name="Team", text="Help by the facts.", author=p.AUTHOR_OWNER)
    with pytest.raises(p.PlaceError, match="Current revision"):  # a stale sheet meets a conflict
        p.edit_place(tmp_path, place.place_id, "set_description", author="owner", description="x",
                     expected_revision=place.revision + 7)
    saved = p.edit_place(tmp_path, place.place_id, "update", author="owner", expected_revision=place.revision,
                         changes=[("set_circle", {"circle_id": circle["circle_id"]}),
                                  ("set_description", {"description": "  Where   the team\nasks  "})])
    assert saved.revision > place.revision and saved.notes == place.notes  # the row itself is unchanged
    records = p.place_records(tmp_path)
    assert records.description(place.place_id)["text"] == "Where the team asks"
    assert document(tmp_path)["connections"][place.place_id] == {
        "connected_revision": place.connected_revision, "written_revision": saved.revision}
    before = p.places_path(tmp_path).read_bytes()
    again = p.edit_place(tmp_path, place.place_id, "set_description", author="owner", description="Where the team asks")
    assert again == saved and p.places_path(tmp_path).read_bytes() == before  # the no-op path writes nothing
    with pytest.raises(p.PlaceError, match="place_description_invalid"):
        p.edit_place(tmp_path, place.place_id, "set_description", author="owner", description="x" * (p.DESCRIPTION_MAX + 1))
    with pytest.raises(p.PlaceError, match="circle_not_found"):
        p.edit_place(tmp_path, place.place_id, "set_circle", author="owner", circle_id="missing")
    assert p.places_path(tmp_path).read_bytes() == before
    cleared = p.edit_place(tmp_path, place.place_id, "update", author="owner", changes=[
        ("set_circle", {"circle_id": ""}), ("set_description", {"description": ""})])
    assert cleared.revision > saved.revision and "place_choices" not in document(tmp_path)


def test_placement_and_words_keep_both_origins_and_only_my_humans_save_makes_them_theirs(tmp_path):
    place = connect(tmp_path)
    circle = p.create_circle(tmp_path, name="Work", text="Owner words for work.", author=p.AUTHOR_OWNER)
    p.edit_place(tmp_path, place.place_id, "set_circle", author=p.AUTHOR_OUROBOROS, circle_id=circle["circle_id"])
    context = p.place_records(tmp_path).circle_context(place.place_id)
    assert (context["text_author"], context["placed_by"], context["standing_delegation"]) == ("owner", "ouroboros", False)
    assert context["text"] == "Owner words for work." and context["name"] == "Work"  # shown intact, both origins
    owned = p.edit_place(tmp_path, place.place_id, "set_circle", author=p.AUTHOR_OWNER, circle_id=circle["circle_id"])
    context = p.place_records(tmp_path).circle_context(place.place_id)
    assert context["placed_by"] == "owner" and context["standing_delegation"] is True
    # Ouroboros saving the same circle again never makes my human's placement its own (no write at all).
    assert p.edit_place(tmp_path, place.place_id, "set_circle", author=p.AUTHOR_OUROBOROS,
                        circle_id=circle["circle_id"]) == owned
    reworded = p.edit_circle(tmp_path, circle["circle_id"], author=p.AUTHOR_OUROBOROS, text="Mind's words")
    assert reworded["text_author"] == "ouroboros" and reworded["revision"] > circle["revision"]
    assert p.place_records(tmp_path).circle_context(place.place_id)["standing_delegation"] is False
    renamed = p.edit_circle(tmp_path, circle["circle_id"], author=p.AUTHOR_OWNER, name="Work room")
    assert renamed["text_author"] == "ouroboros" and renamed["text"] == "Mind's words"  # a rename keeps authorship
    with pytest.raises(p.PlaceError, match="Current revision") as stale:
        p.edit_circle(tmp_path, circle["circle_id"], author="owner", text="late", expected_revision=reworded["revision"])
    assert stale.value.code == "circle_revision_conflict"
    with pytest.raises(p.PlaceError, match="circle_name_taken"):
        p.create_circle(tmp_path, name="work ROOM", author="owner")
    mine = p.edit_circle(tmp_path, circle["circle_id"], author=p.AUTHOR_OWNER, text="Mind's words")
    assert mine["text_author"] == "owner"  # my human's deliberate save of the same words makes them theirs
    assert p.edit_circle(tmp_path, circle["circle_id"], author=p.AUTHOR_OUROBOROS, text="Mind's words") == mine


def test_a_choice_belongs_to_one_connection_and_a_reconnected_place_starts_outside_every_circle(tmp_path):
    place = connect(tmp_path)
    circle = p.create_circle(tmp_path, name="Team", author="owner")
    p.edit_place(tmp_path, place.place_id, "update", author="owner", changes=[
        ("set_circle", {"circle_id": circle["circle_id"]}), ("set_description", {"description": "Team room"})])
    p.edit_place(tmp_path, place.place_id, "disconnect", author="owner")
    assert "place_choices" not in document(tmp_path)
    again = connect(tmp_path)
    records = p.place_records(tmp_path)
    assert again.place_id == place.place_id and records.circle_context(again.place_id) is None
    assert records.description(again.place_id) is None and records.members(circle["circle_id"]) == []


def test_an_older_body_reads_every_place_and_its_stale_choice_never_returns(tmp_path):
    place = connect(tmp_path, wakes=[p.FACT_MENTION])
    circle = p.create_circle(tmp_path, name="Team", text="Words", author="owner")
    p.edit_place(tmp_path, place.place_id, "update", author="owner", changes=[
        ("add_note", {"text": "Own note"}), ("set_circle", {"circle_id": circle["circle_id"]}),
        ("set_description", {"description": "Team room"})])
    value = document(tmp_path)
    assert set(value) >= {"circles", "place_choices"} and value["schema_version"] == 1
    old = base_reader(value)  # rollback keeps every place readable; it ignores the circle words
    assert old[place.place_id].notes == p.get_place(tmp_path, place.place_id).notes
    # The older writer disconnects and connects the same id again, keeping the document's unknown keys.
    del value["places"][place.place_id], value["connections"][place.place_id]
    value["revision"] += 1
    row = p.place_summary(replace(place, revision=value["revision"]))
    value["places"][place.place_id] = row
    p.places_path(tmp_path).write_text(json.dumps(value), encoding="utf-8")
    records = p.place_records(tmp_path)
    assert place.place_id in value["place_choices"] and records.circle_context(place.place_id) is None
    assert records.description(place.place_id) is None and records.circle(circle["circle_id"])["text"] == "Words"


def test_deleting_a_circle_detaches_its_places_atomically_and_keeps_their_own_words(tmp_path):
    first, second = connect(tmp_path, conversation_id="one"), connect(tmp_path, conversation_id="two")
    circle = p.create_circle(tmp_path, name="Team", text="Shared words", author="owner")
    other = p.create_circle(tmp_path, name="Other", author="owner")
    for place, state in ((first, "silent"), (second, "paused")):
        p.edit_place(tmp_path, place.place_id, "update", author="owner", changes=[
            ("set_state", {"state": state}), ("add_note", {"text": f"{place.conversation_id} note"}),
            ("set_circle", {"circle_id": circle["circle_id"]}), ("set_description", {"description": "kept"})])
    before = p.place_records(tmp_path)
    with pytest.raises(p.PlaceError, match="Current revision") as stale:
        p.delete_circle(tmp_path, circle["circle_id"], expected_revision=circle["revision"] + 99)
    assert stale.value.code == "circle_revision_conflict" and p.place_records(tmp_path) == before
    removed = p.delete_circle(tmp_path, circle["circle_id"], expected_revision=circle["revision"])
    assert removed["circle"]["text"] == "Shared words"
    assert sorted(row["place_id"] for row in removed["detached"]) == sorted([first.place_id, second.place_id])
    after = p.place_records(tmp_path)
    assert [row["circle_id"] for row in after.circles] == [other["circle_id"]]
    for place in (first, second):
        old, new = before.places[place.place_id], after.places[place.place_id]
        assert (new.notes, new.state, new.connected_revision) == (old.notes, old.state, old.connected_revision)
        assert new.revision > old.revision and after.circle_context(place.place_id) is None
        assert after.description(place.place_id)["text"] == "kept"
        assert document(tmp_path)["connections"][place.place_id]["written_revision"] == new.revision
    with pytest.raises(p.PlaceError, match="circle_not_found"):
        p.delete_circle(tmp_path, circle["circle_id"])


def test_an_adds_first_words_land_once_with_the_new_place_in_one_transaction(tmp_path):
    circle = p.create_circle(tmp_path, name="Team", author="owner")
    first = {"note": "What this room is for", "circle_id": circle["circle_id"], "description": "Team room"}
    place = connect(tmp_path, first=first)
    records = p.place_records(tmp_path)
    assert [(n.text, n.author) for n in place.notes] == [("What this room is for", "owner")]
    assert records.circle_context(place.place_id)["placed_by"] == "owner"
    assert records.description(place.place_id)["text"] == "Team room"
    before = p.places_path(tmp_path).read_bytes()
    with pytest.raises(p.PlaceError, match="place_exists") as refused:
        connect(tmp_path, first={**first, "note": "Other words"})
    assert refused.value.place_id == place.place_id and p.places_path(tmp_path).read_bytes() == before
    assert connect(tmp_path) == p.get_place(tmp_path, place.place_id)  # a plain repeat stays idempotent
    with pytest.raises(p.PlaceError, match="circle_not_found"):  # nothing is connected when the circle is gone
        connect(tmp_path, conversation_id="new", first={"note": "x", "circle_id": "gone"})
    assert [row.conversation_id for row in p.load_places(tmp_path)] == [""]


def test_malformed_circle_metadata_is_a_repair_error_never_an_empty_store(tmp_path):
    connect(tmp_path)
    value = document(tmp_path)
    value["circles"] = [{"circle_id": "c", "name": "", "text": "", "text_author": "", "text_at": "", "revision": 1}]
    p.places_path(tmp_path).write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(p.PlaceError, match="need repair") as unreadable:
        p.place_records(tmp_path)
    assert unreadable.value.code == "places_unreadable"


def test_circle_model_is_revisioned_without_changing_words_or_place_rows(tmp_path):
    place = connect(tmp_path)
    circle = p.create_circle(tmp_path, name="Team", text="Shared words", model="  future/route  ", author="owner")
    place = p.edit_place(tmp_path, place.place_id, "set_circle", author="owner", circle_id=circle["circle_id"])
    before = document(tmp_path)
    assert before["circles"][0]["model"] == "future/route"
    assert p.place_records(tmp_path).circle_context(place.place_id)["model"] == "future/route"
    changed = p.edit_circle(tmp_path, circle["circle_id"], author="ouroboros", model=" another/route ",
                            expected_revision=circle["revision"])
    assert changed["model"] == "another/route" and changed["revision"] > circle["revision"]
    assert {key: changed[key] for key in ("text", "text_author", "text_at")} == {
        key: circle[key] for key in ("text", "text_author", "text_at")}
    assert set(changed) == {"circle_id", "name", "text", "text_author", "text_at", "revision", "model"}
    assert document(tmp_path)["places"] == before["places"]  # frozen Place rows and their revisions stay intact
    saved = p.places_path(tmp_path).read_bytes()
    with pytest.raises(p.PlaceError, match="Current revision") as stale:
        p.edit_circle(tmp_path, circle["circle_id"], author="owner", model="stale", expected_revision=circle["revision"])
    assert stale.value.code == "circle_revision_conflict" and p.places_path(tmp_path).read_bytes() == saved
    assert p.edit_circle(tmp_path, circle["circle_id"], author="owner", model="another/route",
                          expected_revision=changed["revision"]) == changed
    assert p.places_path(tmp_path).read_bytes() == saved
    cleared = p.edit_circle(tmp_path, circle["circle_id"], author="owner", model="  ", expected_revision=changed["revision"])
    assert cleared["model"] == "" and cleared["revision"] > changed["revision"]
    assert p.place_records(tmp_path).circle_context(place.place_id)["model"] == ""


def test_old_circle_without_model_inherits_without_rewriting_the_store(tmp_path):
    from ouroboros.place_projection import circle_summary
    place = connect(tmp_path)
    circle = p.create_circle(tmp_path, name="Old circle", author="owner")
    p.edit_place(tmp_path, place.place_id, "set_circle", author="owner", circle_id=circle["circle_id"])
    value = document(tmp_path)
    del value["circles"][0]["model"]
    path = p.places_path(tmp_path)
    path.write_text(json.dumps(value), encoding="utf-8")
    before = path.read_bytes()
    records = p.place_records(tmp_path)
    assert circle_summary(records.circle(circle["circle_id"]))["model"] == ""
    assert records.circle_context(place.place_id)["model"] == ""
    assert path.read_bytes() == before
    assert base_reader(value)[place.place_id] == records.places[place.place_id]


@pytest.mark.parametrize("model", [7, False, [], {}])
def test_circle_model_requires_a_string_in_writers_and_stored_rows(tmp_path, model):
    circle = p.create_circle(tmp_path, name="Team", model="unlisted/model", author="owner")
    path = p.places_path(tmp_path)
    before = path.read_bytes()
    for write in (lambda: p.create_circle(tmp_path, name="Other", model=model, author="owner"),
                  lambda: p.edit_circle(tmp_path, circle["circle_id"], model=model, author="owner")):
        with pytest.raises(p.PlaceError) as error:
            write()
        assert error.value.code == "place_invalid_string" and error.value.field == "model"
        assert path.read_bytes() == before
    value = document(tmp_path)
    value["circles"][0]["model"] = model
    path.write_text(json.dumps(value), encoding="utf-8")
    malformed = path.read_bytes()
    with pytest.raises(p.PlaceError, match="need repair"):
        p.place_records(tmp_path)
    assert path.read_bytes() == malformed
