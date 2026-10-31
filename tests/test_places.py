import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from ouroboros import places as p
from ouroboros.presence_bindings import PresenceBinding, PresenceEndpoint
from tests.frozen_presence_bindings import save_presence_binding


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
    place, = p.load_places(root)
    assert place.state == ("active" if enabled else "paused") and place.legacy_binding_ids == (binding.binding_id,)
    assert place.notes[0] == p.NoteEntry("Keep these exact words.\n", "migration")
    assert place.context_topics == ("social",) and place.working_folder == "/work" and place.model == "light"
    assert place.destination["conversation_id"] == "elsewhere"
    assert {w.fact for w in place.wakes} == {"legacy_addressed", "group_dm"}
    saved = p.edit_place(root, place.place_id, "set_state", author="owner", state="silent")
    assert p.load_places(root) == [saved]
    assert (root / "state" / "presence_bindings.json").read_bytes() == source


def test_empty_and_malformed_sources_preserved(tmp_path):
    assert p.load_places(tmp_path) == []
    assert p.load_places(tmp_path) == []
    other = tmp_path / "broken"
    path = other / "state" / "presence_bindings.json"
    path.parent.mkdir(parents=True)
    path.write_text("{broken")
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


def test_place_storage_keys_are_rollback_compatible():
    assert set(asdict(p.Place("id", "bridge", "chat"))) == {
        "place_id", "connector", "provider", "account_id", "conversation_id", "label", "state", "wakes",
        "notes", "owner_accounts", "working_folder", "model", "context_topics", "legacy_binding_ids", "revision",
        "thread_id", "destination", "scope", "contribution_id",
    }


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


@pytest.mark.parametrize("revision", [True, "2", 2.0, -1])
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


@pytest.mark.parametrize("fact", ["signal.", "signal.Upper", "signal.a.b", "signal." + "a" * 41, "signal.a\n", None, {}])
def test_invalid_signal_never_enters_storage(tmp_path, fact):
    place = connect(tmp_path)
    with pytest.raises(p.PlaceError, match="place_wake_invalid"):
        p.edit_place(tmp_path, place.place_id, "add_wake", author="owner", fact=fact)
    assert p.load_places(tmp_path) == [place]
