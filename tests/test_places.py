import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import places as p
from ouroboros.presence_bindings import PresenceBinding, PresenceEndpoint, conversation_key
from tests._place_wake_import import admit_live, base_reader, document, legacy_wake_install, old_load_places, wake_declaration
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


@pytest.mark.parametrize("author", ["owner", "ouroboros"])
def test_same_wakes_on_an_ordinary_connection_preserve_the_place_and_file(tmp_path, author):
    facts = ["mention", "direct_message"]
    place = connect(tmp_path, wakes=facts)
    path = p.places_path(tmp_path)
    before = path.read_bytes()
    saved = p.edit_place(tmp_path, place.place_id, "set_wakes", author=author, wakes=facts, expected_revision=place.revision)
    assert saved == place
    assert path.read_bytes() == before


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
    monkeypatch.setattr("ouroboros.skill_loader.find_skill", lambda _root, name: behavior if name == "mind" else None)
    monkeypatch.setattr("ouroboros.extension_registry_state.live_contribution_projection",
                        lambda: [wake_declaration("group_dm")])
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


def test_core_first_import_finishes_once_at_live_admission_and_the_first_dm_focuses(tmp_path, monkeypatch):
    from ouroboros.place_intake import decide_attention
    from ouroboros.presence_events import PresenceTurnEvent
    published = legacy_wake_install(tmp_path, monkeypatch)
    place, = p.load_places(tmp_path)
    legacy = p.WakeItem("legacy_addressed", "migration")
    assert place.wakes == (legacy,)
    published.append(wake_declaration("any_message", connector="another-bridge"))
    assert admit_live(tmp_path, place).place == place
    published.append(wake_declaration("direct_message", "mention"))
    source = (tmp_path / "state" / "presence_bindings.json").read_bytes()
    admission = admit_live(tmp_path, place)
    assert admission.place.wakes == (legacy, p.WakeItem("direct_message", "connector"), p.WakeItem("mention", "connector"))
    assert replace(admission.place, wakes=place.wakes, revision=place.revision) == place
    key = conversation_key("slack", "team", place.conversation_id)
    event = PresenceTurnEvent("first-dm", "slack", "team", place.conversation_id, "", key,
                              {"platform_actor_id": "person"}, {}, {}, "hello", facts=("direct_message",))
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance", lambda _root: {"status": "available"})
    assert decide_attention(tmp_path, admission, event)[0] == "focus"
    saved = p.places_path(tmp_path).read_bytes()
    published[:] = [wake_declaration("any_message")]
    assert admit_live(tmp_path, place).place == admission.place and p.places_path(tmp_path).read_bytes() == saved
    assert (tmp_path / "state" / "presence_bindings.json").read_bytes() == source


@pytest.mark.parametrize("defaults", [[], ["direct_message"]])
def test_connector_first_import_takes_even_an_empty_declaration_as_final(tmp_path, monkeypatch, defaults):
    published = legacy_wake_install(tmp_path, monkeypatch, published=[wake_declaration(*defaults)])
    place, = p.load_places(tmp_path)
    assert [w.fact for w in place.wakes] == ["legacy_addressed", *defaults]
    published[:] = [wake_declaration("any_message")]
    assert admit_live(tmp_path, place).place == place
    assert document(tmp_path)["migration"]["presence_bindings_v1"] is True


@pytest.mark.parametrize("connector_first", [True, False])
def test_legacy_transport_keeps_declared_order_across_its_place_contributions(tmp_path, monkeypatch, connector_first):
    rows = [wake_declaration("mention"),
            {**wake_declaration("direct_message", "mention", "signal.inbox"), "id": "inbox",
             "facts": ["direct_message", "mention", {"id": "signal.inbox", "label": "Inbox item"}]}]
    published = legacy_wake_install(tmp_path, monkeypatch, published=rows if connector_first else ())
    place, = p.load_places(tmp_path)
    if not connector_first:
        published.extend(rows)
    assert admit_live(tmp_path, place).place.wakes == (
        p.WakeItem("legacy_addressed", "migration"), p.WakeItem("mention", "connector"),
        p.WakeItem("direct_message", "connector"), p.WakeItem("signal.inbox", "connector"))


@pytest.mark.parametrize("actions", [
    [("set_wakes", {"wakes": ["legacy_addressed"]})],
    [("add_wake", {"fact": "mention"}), ("remove_wake", {"fact": "mention"})],
    [("remove_wake", {"fact": "direct_message"})],
    [("set_wakes", {"wakes": []})],
])
@pytest.mark.parametrize("author", ["owner", "ouroboros"])
def test_every_explicit_wake_action_records_review_even_when_the_list_is_unchanged(tmp_path, monkeypatch, actions, author):
    published = legacy_wake_install(tmp_path, monkeypatch)
    place, = p.load_places(tmp_path)
    for action, args in actions:
        previous = place
        place = p.edit_place(tmp_path, place.place_id, action, author=author, expected_revision=place.revision, **args)
        assert place.revision > previous.revision
        assert p.place_records(tmp_path).choices[place.place_id] == {
            "connected_revision": place.connected_revision, "wakes_reviewed": True}
    before = p.places_path(tmp_path).read_bytes()
    assert p.edit_place(tmp_path, place.place_id, "set_wakes", author=author, expected_revision=place.revision,
                        wakes=[wake.fact for wake in place.wakes]) == place
    assert p.places_path(tmp_path).read_bytes() == before
    published.append(wake_declaration("direct_message"))
    assert admit_live(tmp_path, place).place == place


def test_note_state_and_circle_edits_keep_pending_across_fresh_reads(tmp_path, monkeypatch):
    published = legacy_wake_install(tmp_path, monkeypatch)
    place, = p.load_places(tmp_path)
    circle = p.create_circle(tmp_path, name="Friends", author="owner")
    noted = p.edit_place(tmp_path, place.place_id, "update", author="owner", changes=[
        ("set_circle", {"circle_id": circle["circle_id"]}), ("set_state", {"state": "silent"}),
        ("add_note", {"text": "Keep quiet here."})])
    p.delete_circle(tmp_path, circle["circle_id"], author="owner")
    published.append(wake_declaration("direct_message"))
    admitted = admit_live(tmp_path, place).place
    assert admitted.state == "silent" and admitted.notes == noted.notes
    assert admitted.wakes[-1] == p.WakeItem("direct_message", "connector")


def test_late_empty_declaration_closes_import_before_later_defaults(tmp_path, monkeypatch):
    published = legacy_wake_install(tmp_path, monkeypatch)
    place, = p.load_places(tmp_path)
    published.append(wake_declaration())
    closed = admit_live(tmp_path, place).place
    assert closed.revision > place.revision and closed.wakes == place.wakes
    published[:] = [wake_declaration("any_message")]
    assert admit_live(tmp_path, closed).place == closed


def old_same_list_save(root, place):
    """Two separate v1 locks. The setter reports success even when its second transaction writes nothing."""
    old_load_places(root, [p.place_summary(place)])
    path = p.places_path(root)
    before = path.read_bytes()
    submitted = [wake.fact for wake in place.wakes]
    seen = []

    def save(current):
        old = base_reader(current)[place.place_id]
        assert old.revision == place.revision
        wakes = tuple(next((item for item in old.wakes if item.fact == fact), p.WakeItem(fact, "owner"))
                      for fact in dict.fromkeys(submitted))
        updated = replace(old, wakes=wakes)
        assert updated == old
        seen.append(updated)
        return None  # v1 _write's unchanged-place/unchanged-choice branch

    p.update_json_locked(path, save)
    assert seen == [place] and path.read_bytes() == before


def test_restored_old_reader_closes_proof_before_a_successful_same_list_save(tmp_path, monkeypatch):
    from ouroboros.places_migration import wake_review
    published = legacy_wake_install(tmp_path, monkeypatch)
    place, = p.load_places(tmp_path)
    place = p.edit_place(tmp_path, place.place_id, "set_state", author="owner", state="paused")
    old_same_list_save(tmp_path, place)
    assert document(tmp_path)["migration"]["presence_bindings_v1"] is True
    published.append(wake_declaration("direct_message"))
    assert admit_live(tmp_path, place).place == place
    assert wake_review(p.place_records(tmp_path), place, ["direct_message"]) == "ambiguous"


@pytest.mark.parametrize("damage", ["revision", "author", "malformed"])
def test_a_stale_or_malformed_witness_only_offers_manual_review(tmp_path, monkeypatch, damage):
    from ouroboros.places_migration import wake_review
    published = legacy_wake_install(tmp_path, monkeypatch)
    place, = p.load_places(tmp_path)
    value = document(tmp_path)
    row = value["places"][place.place_id]
    if damage == "revision":
        value["revision"] += 1
        row["revision"] = value["revision"]
    elif damage == "author":
        row["wakes"][0]["author"] = "owner"
    else:
        value["place_choices"][place.place_id]["wake_import"]["revision"] = True
    path = p.places_path(tmp_path)
    path.write_text(json.dumps(value), encoding="utf-8")
    before = path.read_bytes()
    published.append(wake_declaration("direct_message"))
    live = admit_live(tmp_path, place).place
    assert wake_review(p.place_records(tmp_path), live, ["direct_message"]) == "ambiguous"
    assert path.read_bytes() == before


def test_legacy_disconnect_seals_before_an_old_reader_can_revive_it(tmp_path, monkeypatch):
    from ouroboros.places_migration import wake_review
    published = legacy_wake_install(tmp_path, monkeypatch, ids=("a" * 32, "b" * 32))
    first, other = p.load_places(tmp_path)
    p.edit_place(tmp_path, first.place_id, "disconnect", author="owner")
    assert set(old_load_places(tmp_path, [p.place_summary(first), p.place_summary(other)])) == {other.place_id}
    published.append(wake_declaration("direct_message"))
    assert admit_live(tmp_path, other).place == other
    assert wake_review(p.place_records(tmp_path), other, ["direct_message"]) == "ambiguous"
    again = connect(tmp_path, place_id=first.place_id)
    assert again.connected_revision > first.connected_revision and admit_live(tmp_path, again).place == again
    assert first.place_id not in p.place_records(tmp_path).choices


def test_partial_import_keeps_its_problem_and_cannot_revive_the_repaired_binding(tmp_path, monkeypatch):
    from ouroboros.places_migration import wake_review
    published = legacy_wake_install(tmp_path, monkeypatch)
    endpoint = PresenceEndpoint("slack", "team", "other")
    missing = PresenceBinding("c" * 32, "slack-bridge", "missing-mind", endpoint, endpoint)
    save_presence_binding(tmp_path, missing)
    place, = p.load_places(tmp_path)
    problems = p.migration_problems(tmp_path)
    assert problems[0]["binding_id"] == missing.binding_id
    fixed = replace(place, place_id=missing.binding_id, legacy_binding_ids=(missing.binding_id,))
    assert set(old_load_places(tmp_path, [p.place_summary(fixed)])) == {place.place_id}
    assert p.migration_problems(tmp_path) == problems
    published.append(wake_declaration("direct_message"))
    assert admit_live(tmp_path, place).place == place
    assert wake_review(p.place_records(tmp_path), place, ["direct_message"]) == "ambiguous"


def test_unrelated_disconnect_does_not_close_a_pending_legacy_import(tmp_path, monkeypatch):
    published = legacy_wake_install(tmp_path, monkeypatch)
    place, = p.load_places(tmp_path)
    ordinary = connect(tmp_path, conversation_id="another")
    p.edit_place(tmp_path, ordinary.place_id, "disconnect", author="owner")
    published.append(wake_declaration("direct_message"))
    assert admit_live(tmp_path, place).place.wakes[-1] == p.WakeItem("direct_message", "connector")


def test_a_known_binding_is_never_re_read_and_its_broken_source_blocks_no_fresh_window(tmp_path, monkeypatch):
    """A re-import (the migration stamp was lost, the rows stayed — a rollback's shape) skips bindings the stored
    rows already carry, without re-reading their sources: a behavior skill that broke since adds no problem, so a
    fresh binding's pending window stays proven and its review stays pending, never ambiguous. No wakes are added
    to the known row without proof; its own later suggestions stay a human choice."""
    from ouroboros.places_migration import wake_review
    behavior = SimpleNamespace(skill_dir=tmp_path, manifest=SimpleNamespace(raw_extra={
        "presence": {"instructions": "Keep these words.", "context_topics": ["social"]}}))
    transport = SimpleNamespace(manifest=SimpleNamespace(contributes=[]))
    skills = {"mind": behavior, "mind-2": behavior, "slack-bridge": transport}
    monkeypatch.setattr("ouroboros.skill_loader.find_skill", lambda _root, name: skills.get(name))
    published = []
    monkeypatch.setattr("ouroboros.extension_registry_state.live_contribution_projection", lambda: published)
    origin = PresenceEndpoint("slack", "team", "aaaaaaaa")
    save_presence_binding(tmp_path, PresenceBinding("a" * 32, "slack-bridge", "mind", origin, origin))
    imported, = p.load_places(tmp_path)  # the first import: one pending place, its window open
    assert imported.wakes == (p.WakeItem("legacy_addressed", "migration"),)
    value = document(tmp_path)
    del value["migration"]  # the stamp is lost but the row stays
    p.places_path(tmp_path).write_text(json.dumps(value), encoding="utf-8")
    skills.pop("mind")  # the known binding's behavior source is unreadable now
    second = PresenceEndpoint("slack", "team", "cccccccc")
    save_presence_binding(tmp_path, PresenceBinding("c" * 32, "slack-bridge", "mind-2", second, second))
    rows = {row.place_id: row for row in p.load_places(tmp_path)}  # the re-import runs
    assert p.migration_problems(tmp_path) == [], "a known binding's broken source is not this import's problem"
    assert document(tmp_path)["migration"]["wake_import_open"] is True, "the fresh binding's window stayed open"
    assert rows["a" * 32] == imported, "the known row is untouched: no wakes added without proof"
    fresh = rows["c" * 32]
    assert fresh.wakes == (p.WakeItem("legacy_addressed", "migration"),)
    records = p.place_records(tmp_path)
    assert wake_review(records, fresh, ["direct_message"]) == "pending"
    # The known row lost its witness with the stamp: later suggestions are a human choice, never automatic.
    assert wake_review(records, imported, ["direct_message"]) == "ambiguous"


@pytest.mark.parametrize("old_writer", [True, False])
def test_racing_choice_wins_even_if_an_old_writer_keeps_the_row_revision(tmp_path, monkeypatch, old_writer):
    legacy_wake_install(tmp_path, monkeypatch)
    place, = p.load_places(tmp_path)
    saved = []

    def publish_after_save():
        if old_writer:
            old_same_list_save(tmp_path, place)
        else:
            p.edit_place(tmp_path, place.place_id, "set_wakes", author="owner", wakes=[])
        saved.append(p.get_place(tmp_path, place.place_id))
        return [wake_declaration("direct_message")]

    monkeypatch.setattr("ouroboros.extension_registry_state.live_contribution_projection", publish_after_save)
    admission = admit_live(tmp_path, place)
    assert saved == [admission.place] and admission.place == p.get_place(tmp_path, place.place_id)
    assert admission.place.wakes == (place.wakes if old_writer else ())


def test_note_racing_the_first_typed_event_keeps_the_note_and_finishes_import(tmp_path, monkeypatch):
    legacy_wake_install(tmp_path, monkeypatch)
    place, = p.load_places(tmp_path)
    notes = []

    def publish_after_note():
        if not notes:
            changed = p.edit_place(tmp_path, place.place_id, "add_note", author="owner", text="Keep this.")
            notes.extend(changed.notes)
        return [wake_declaration("direct_message")]

    monkeypatch.setattr("ouroboros.extension_registry_state.live_contribution_projection", publish_after_note)
    admitted = admit_live(tmp_path, place).place
    assert list(admitted.notes) == notes and notes[-1].text == "Keep this."
    assert admitted.wakes[-1] == p.WakeItem("direct_message", "connector")


def test_racing_note_edits_never_defer_the_first_event_past_the_wake_import(tmp_path, monkeypatch):
    """Review item 7: two revision conflicts used to leave the admission on the migration placeholder, so a
    direct-message-only first event settled silent and its retry replayed that stored outcome; the import finished
    only on the next event. The completion now binds to the record as it stands under the store lock: the racing
    edit is kept, the import finishes in the first event's own admission, and that event is heard as a wake."""
    from ouroboros.place_intake import decide_attention
    from ouroboros.presence_events import PresenceTurnEvent
    published = legacy_wake_install(tmp_path, monkeypatch)
    place, = p.load_places(tmp_path)
    published.append(wake_declaration("direct_message"))
    edits = []

    def publish_under_edit():
        assert len(edits) < 2, "one completion under the lock finishes the import; no second declarations read"
        edits.append(p.edit_place(tmp_path, place.place_id, "add_note", author="owner", text=f"Note {len(edits) + 1}"))
        return published

    with monkeypatch.context() as racing:
        racing.setattr("ouroboros.extension_registry_state.live_contribution_projection", publish_under_edit)
        admission = admit_live(tmp_path, place)
    assert len(edits) == 1
    admitted = admission.place
    assert admitted.notes == edits[0].notes
    assert admitted.notes[:-1] == place.notes and admitted.notes[-1].text == "Note 1"
    assert admitted.wakes == (*place.wakes, p.WakeItem("direct_message", "connector"))
    assert admitted.revision > edits[0].revision
    assert document(tmp_path)["migration"]["presence_bindings_v1"] is True
    key = conversation_key("slack", "team", admitted.conversation_id)
    event = PresenceTurnEvent("first-dm", "slack", "team", admitted.conversation_id, "", key,
                              {"platform_actor_id": "person"}, {}, {}, "hello", facts=("direct_message",))
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance", lambda _root: {"status": "available"})
    assert decide_attention(tmp_path, admission, event)[0] == "focus", "the first direct message is never settled silent"


def test_unpublished_connector_declarations_repeat_the_first_dm_instead_of_settling_it(tmp_path, monkeypatch):
    """The open migration case: a connector updated while its live declarations are temporarily unavailable.
    The first real DM must not settle silent-heard forever on the placeholder list — the door repeats the event
    instead; once the registry publishes, the first repeat of that same event completes the import once and the
    event gets its real decision. A genuinely old connector (nothing declared in its installed manifest) still
    settles plain heard, so its events never loop."""
    from ouroboros.place_intake import WAKE_IMPORT_PENDING, author_hears, decide_attention
    from ouroboros.presence_events import PresenceTurnEvent
    from ouroboros.contracts.skill_manifest import parse_skill_manifest_text
    from ouroboros import extension_registry_state as registry
    published = legacy_wake_install(tmp_path, monkeypatch)
    place, = p.load_places(tmp_path)
    assert place.wakes == (p.WakeItem("legacy_addressed", "migration"),)
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance", lambda _root: {"status": "available"})
    key = conversation_key("slack", "team", place.conversation_id)
    event = PresenceTurnEvent("first-dm", "slack", "team", place.conversation_id, "", key,
                              {"platform_actor_id": "person"}, {}, {}, "hello", facts=("direct_message",))

    # Old connector: nothing installed, nothing published — the plain DM stays settled silent-heard.
    old = admit_live(tmp_path, place)
    assert decide_attention(tmp_path, old, event) == ("silent", {"reason": "no_wake_match"})
    assert decide_attention(tmp_path, old, replace(event, facts=None))[0] == "focus"

    # The connector updated on disk; the live registry has not published it yet (the temporary gap).
    entry = wake_declaration("direct_message", "group_dm")
    del entry["skill"]  # only the published row carries its connector's name
    manifest = parse_skill_manifest_text(json.dumps({"name": place.connector, "type": "extension",
        "permissions": ["presence"], "contributes": [entry]}))
    transport = SimpleNamespace(skill_dir=tmp_path, content_hash="reviewed", manifest=manifest)
    monkeypatch.setattr("ouroboros.skill_loader.find_skill",
                        lambda _root, name: transport if name == "slack-bridge" else None)
    monkeypatch.setattr("ouroboros.extension_liveness.runtime_state_for_loaded_skill",
                        lambda *_: {"desired_live": True, "reason": "ready", "live_loaded": False})
    before = p.places_path(tmp_path).read_bytes()
    admission = admit_live(tmp_path, place)
    assert admission.place.wakes == place.wakes, "unpublished file content applies nothing"
    pending = ("retry", {"reason": WAKE_IMPORT_PENDING, "retry_after": 60})
    assert decide_attention(tmp_path, admission, event) == pending
    assert author_hears(*pending, event) is True
    group = replace(event, facts=("group_dm",))
    assert decide_attention(tmp_path, admission, group) == pending
    assert author_hears(*pending, group) is None
    assert author_hears(*pending, replace(group, thread_id="topic")) is True
    with monkeypatch.context() as settled:
        for desired, reason in ((True, "load_error"), (False, "disabled"), (False, "review_stale")):
            settled.setattr("ouroboros.extension_liveness.runtime_state_for_loaded_skill", lambda *_, desired=desired, reason=reason: {
                "desired_live": desired, "reason": reason, "live_loaded": False})
            assert decide_attention(tmp_path, admission, event) == ("silent", {"reason": "no_wake_match"})
    bundle = registry._ExtensionRegistrations(content_hash=transport.content_hash, skill_dir=str(tmp_path.resolve()))
    monkeypatch.setitem(registry._extensions, place.connector, bundle)
    assert decide_attention(tmp_path, admission, event) == ("silent", {"reason": "no_wake_match"})
    with monkeypatch.context() as reload:
        reload.setattr(registry, "_unloading", {place.connector})
        reload.setattr("ouroboros.extension_liveness.runtime_state_for_loaded_skill",
                       lambda *_: {"desired_live": True, "reason": "ready", "live_loaded": True})
        assert decide_attention(tmp_path, admission, event) == pending
    # A bot not selected by the old choice gets no live mail; ordinary group chatter is only background.
    for other, hears in ((replace(event, facts=("any_message",)), False),
                         (replace(event, actor={"is_bot": True}), None)):
        decision, details = decide_attention(tmp_path, admission, other)
        assert decision == "silent" and author_hears(decision, details, other) is hears
    assert p.places_path(tmp_path).read_bytes() == before, "the repeated decision settled nothing"

    # The registry catches up: the first repeat of the same event id completes the import once, then decides for real.
    published.append(wake_declaration("direct_message", "group_dm"))
    admission = admit_live(tmp_path, place)
    assert admission.place.wakes == (p.WakeItem("legacy_addressed", "migration"),
                                     p.WakeItem("direct_message", "connector"), p.WakeItem("group_dm", "connector"))
    assert decide_attention(tmp_path, admission, event)[0] == "focus"
    assert document(tmp_path)["migration"]["presence_bindings_v1"] is True
    saved = p.places_path(tmp_path).read_bytes()
    assert admit_live(tmp_path, admission.place).place == admission.place
    assert p.places_path(tmp_path).read_bytes() == saved, "the import completes exactly once"


def test_refused_origin_does_not_complete_a_wake_import(tmp_path, monkeypatch):
    from ouroboros.presence_admission import PresenceAdmissionError
    ready = legacy_wake_install(tmp_path, monkeypatch)
    place, = p.load_places(tmp_path)
    ready.append(wake_declaration("direct_message"))
    before = p.places_path(tmp_path).read_bytes()
    with monkeypatch.context() as refused:
        refused.setattr("ouroboros.presence_admission.origin_outside", lambda *_: True)
        with pytest.raises(PresenceAdmissionError, match="presence_origin_mismatch"):
            admit_live(tmp_path, place)
    assert p.places_path(tmp_path).read_bytes() == before
    assert admit_live(tmp_path, place).place.wakes[-1].fact == "direct_message"


def test_empty_and_malformed_sources_preserved(tmp_path):
    assert p.load_places(tmp_path) == []
    assert p.load_places(tmp_path) == []
    other = tmp_path / "broken"
    path = other / "state" / "presence_bindings.json"
    path.parent.mkdir(parents=True)
    path.write_text("{broken")
    assert p.paused_places_sentence(other) == "Paused places: unknown (place records could not be read)."
    with pytest.raises(p.PlaceError, match="repair") as preview:
        p.place_records(other, persist_import=False)
    assert preview.value.code == "places_migration_failed" and not p.places_path(other).exists()
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
        "provides": "place", "skill": "slack-bridge", "destination": "vendor:Slack",
        "facts": [{"id": fact, "label": "Chosen words"}]}])
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
    reworded = p.edit_circle(tmp_path, circle["circle_id"], author=p.AUTHOR_OUROBOROS, text="Mind's words",
                             by={"owner_turn": True})  # my human's turn orders it; the words I compose are mine
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
        p.delete_circle(tmp_path, circle["circle_id"], author="owner", expected_revision=circle["revision"] + 99)
    assert stale.value.code == "circle_revision_conflict" and p.place_records(tmp_path) == before
    removed = p.delete_circle(tmp_path, circle["circle_id"], author="owner", expected_revision=circle["revision"])
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
        p.delete_circle(tmp_path, circle["circle_id"], author="owner")


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


# ---- my human's words leave only by their act; every word that leaves is kept ---------------------------------------
def journal(root):
    from ouroboros.place_words_history import journal_path
    return journal_path(root)


def test_my_humans_place_words_leave_only_by_their_act_and_mine_stay_mine(tmp_path):
    """14.11: a wake at Full removed my human's note of 11.11 from two places. Their note, a place's description and a
    circle's words leave only by their own act: their app, their turn ordering words I compose, or their words waking
    the focus of a place those words stand in. A wake, or my human speaking elsewhere, is refused and nothing changes;
    a stale edit still meets its revision first. My own words, and a circle's name and model, stay mine to change;
    disconnecting is an act on the place, never refused for its words."""
    place, other = connect(tmp_path, conversation_id="one"), connect(tmp_path, conversation_id="two")
    circle = p.create_circle(tmp_path, name="Team", text="Owner words for the team.", author="owner")
    p.edit_place(tmp_path, place.place_id, "update", author="owner", changes=[
        ("add_note", {"text": "Owner's standing note."}), ("set_circle", {"circle_id": circle["circle_id"]}),
        ("set_description", {"description": "Owner's line"})])
    owners, mine = (note.note_id for note in p.edit_place(
        tmp_path, place.place_id, "add_note", author="ouroboros", text="My own note.").notes)
    p.edit_place(tmp_path, other.place_id, "add_note", author="owner", text="Owner's note elsewhere.")
    before = p.places_path(tmp_path).read_bytes()
    for by in (None, {"owner_speaks_in": other.place_id}):  # a wake; my human speaking outside this place and circle
        for write in (
                lambda: p.edit_place(tmp_path, place.place_id, "remove_note", author="ouroboros", by=by, note_id=owners),
                lambda: p.edit_place(tmp_path, place.place_id, "set_description", author="ouroboros", by=by,
                                     description=""),
                lambda: p.edit_circle(tmp_path, circle["circle_id"], author="ouroboros", by=by, text="Mine"),
                lambda: p.delete_circle(tmp_path, circle["circle_id"], author="ouroboros", by=by)):
            with pytest.raises(p.PlaceError) as refused:
                write()
            assert refused.value.code == "place_owner_words_owner_only"
    with pytest.raises(p.PlaceError) as stale:
        p.edit_circle(tmp_path, circle["circle_id"], author="ouroboros", text="Mine", expected_revision=0)
    assert stale.value.code == "circle_revision_conflict"
    assert p.places_path(tmp_path).read_bytes() == before and not journal(tmp_path).exists()
    kept = p.edit_place(tmp_path, place.place_id, "remove_note", author="ouroboros", note_id=mine)
    assert [note.text for note in kept.notes] == ["Owner's standing note."]
    renamed = p.edit_circle(tmp_path, circle["circle_id"], author="ouroboros", name="Team room", model="circle/next")
    assert (renamed["text"], renamed["text_author"]) == ("Owner words for the team.", "owner")
    here = {"owner_speaks_in": place.place_id}  # my human's own words woke this place's focus now
    assert p.edit_place(tmp_path, place.place_id, "remove_note", author="ouroboros", by=here, note_id=owners).notes == ()
    assert p.edit_circle(tmp_path, circle["circle_id"], author="ouroboros", by=here, text="Mine")["text_author"] == (
        "ouroboros")
    p.edit_place(tmp_path, place.place_id, "set_description", author="ouroboros", by={"owner_turn": True},
                 description="")
    assert p.place_records(tmp_path).description(place.place_id) is None
    theirs = p.create_circle(tmp_path, name="Friends", text="Owner words for friends.", author="owner")
    assert p.delete_circle(tmp_path, theirs["circle_id"], author="owner")["circle"]["text"] == "Owner words for friends."
    assert p.edit_place(tmp_path, other.place_id, "disconnect", author="ouroboros") is None


def test_every_word_a_change_removes_or_replaces_is_kept_exactly_and_read_newest_first(tmp_path):
    """Before a change lands, each note, description and circle's words it removes or replaces is appended exactly:
    text, author and time, where it stood, the act, who made the change and the focus that wrote it. A place's earlier
    words read newest first, its circle's included. Words that stay, and a change that removes nothing, leave no row."""
    from ouroboros.place_words_history import earlier
    place = connect(tmp_path)
    circle = p.create_circle(tmp_path, name="Team", text="Circle words.", author="owner")
    p.edit_place(tmp_path, place.place_id, "update", author="owner", changes=[
        ("add_note", {"text": "First\n  note, verbatim. "}), ("add_note", {"text": "Second note."}),
        ("set_circle", {"circle_id": circle["circle_id"]}), ("set_description", {"description": "Owner's line"})])
    p.edit_place(tmp_path, place.place_id, "set_state", author="owner", state="silent")
    p.edit_circle(tmp_path, circle["circle_id"], author="owner", text="Circle words.")  # saved again: unchanged
    assert not journal(tmp_path).exists()
    first, second = p.get_place(tmp_path, place.place_id).notes
    line = p.place_records(tmp_path).description(place.place_id)
    writer = {"kind": "mind", "focus": {"role": "main", "task_id": "task-1"}}
    p.edit_place(tmp_path, place.place_id, "remove_note", author="owner", note_id=first.note_id)
    p.edit_place(tmp_path, place.place_id, "set_description", author="ouroboros", description="My line",
                 by={"owner_turn": True, "writer": writer})
    p.edit_circle(tmp_path, circle["circle_id"], author="owner", text="Circle words, reworded.")
    reworded = p.place_records(tmp_path).circle(circle["circle_id"])
    p.delete_circle(tmp_path, circle["circle_id"], author="owner")
    shown = earlier(tmp_path, place.place_id)
    assert [(word["action"], word["kind"], word["text"], word["author"], word["changed_by"]) for word in shown] == [
        ("delete_circle", "circle_words", "Circle words, reworded.", "owner", "owner"),
        ("edit_circle", "circle_words", "Circle words.", "owner", "owner"),
        ("set_description", "description", "Owner's line", "owner", "owner"),
        ("remove_note", "note", "First\n  note, verbatim. ", "owner", "owner")]
    assert (shown[0]["at"], shown[0]["circle_id"], shown[0]["circle_name"]) == (
        reworded["text_at"], circle["circle_id"], "Team")
    assert shown[1]["at"] == circle["text_at"] and shown[2]["at"] == line["at"] and shown[2]["writer"] == writer
    assert (shown[3]["note_id"], shown[3]["at"], shown[3]["place_id"]) == (first.note_id, first.at, place.place_id)
    rows = [json.loads(row) for row in journal(tmp_path).read_text(encoding="utf-8").splitlines()]
    assert [row["document_revision"] for row in rows] == sorted({row["document_revision"] for row in rows})
    assert rows[2]["words"][0]["places"] == [place.place_id] and "writer" not in rows[0]
    p.edit_place(tmp_path, place.place_id, "disconnect", author="owner")
    assert [(word["action"], word["kind"], word["text"]) for word in earlier(tmp_path, place.place_id)[:2]] == [
        ("disconnect", "note", second.text), ("disconnect", "description", "My line")]
    assert earlier(tmp_path, "another-place") == []


def test_a_history_that_cannot_be_written_changes_nothing(tmp_path):
    """The history row comes first: when the journal cannot take it, a change that removes words is refused
    (``place_history_unavailable``) and the place document keeps every byte, and a journal that cannot be read is
    an error, never "no earlier words". A change that removes nothing needs no row and still lands."""
    from ouroboros.place_words_history import earlier
    place = p.edit_place(tmp_path, connect(tmp_path).place_id, "add_note", author="owner", text="Keep me.")
    journal(tmp_path).mkdir()  # something in the journal's way: no row can be appended or read
    before = p.places_path(tmp_path).read_bytes()
    for action, values in (("remove_note", {"note_id": place.notes[0].note_id}), ("disconnect", {})):
        with pytest.raises(p.PlaceError) as unavailable:
            p.edit_place(tmp_path, place.place_id, action, author="owner", **values)
        assert unavailable.value.code == "place_history_unavailable"
    assert p.places_path(tmp_path).read_bytes() == before
    with pytest.raises(p.PlaceError) as unreadable:
        earlier(tmp_path, place.place_id)
    assert unreadable.value.code == "place_history_unreadable"
    added = p.edit_place(tmp_path, place.place_id, "add_note", author="owner", text="And me.")
    assert [note.text for note in added.notes] == ["Keep me.", "And me."]
