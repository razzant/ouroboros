"""The owner and model share one PlaceSummary, including unavailable evidence."""
import json

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from ouroboros import place_projection as projection, places
from ouroboros.gateway.places_api import places_routes
from ouroboros.icon_geometry import icon_spec
from ouroboros.tools.presence import _configure_presence
from ouroboros.tools.registry import ToolContext
from tests.test_extension_icons import PLACE, SVG, staged_icon


@pytest.fixture
def connector(tmp_path, monkeypatch):
    skill, staged = staged_icon(tmp_path, entries=[{
        **PLACE, "label": "GitHub", "destination": "vendor:GitHub", "wakes": ["mention", "signal.words"],
        "facts": [{"id": "mention", "label": "Mentions you"}, {"id": "signal.words", "label": "Words you chose appear"}],
        "source": {"route": "status", "refresh_s": 60}, "settings": "connection",
    }])
    monkeypatch.setattr("ouroboros.skill_loader.discover_skills", lambda _root: [skill])
    monkeypatch.setattr("ouroboros.extension_registry_state.live_contribution_projection", lambda: staged.contributions)
    return skill, staged


@pytest.fixture
def client(tmp_path, connector):
    app = Starlette(routes=places_routes())
    app.state.drive_root = tmp_path
    with TestClient(app) as value:
        yield value


@pytest.mark.parametrize("live", [True, False])
def test_both_declaration_branches_supply_sanitized_icon_and_service_label(tmp_path, connector, monkeypatch, live):
    skill, staged = connector
    if live:
        # Mutable payload changes cannot replace the live publication's captured mark.
        (skill.skill_dir / "ui/icon.svg").write_text('<svg><script/></svg>', encoding="utf-8")
    else:
        monkeypatch.setattr("ouroboros.extension_registry_state.live_contribution_projection", lambda: [])
    place = projection.connect_declared_place(tmp_path, skill.name)
    wire = projection.places_view(tmp_path)
    row, = wire["places"]
    offer, = wire["connectors"]
    assert row["icon"] == offer["icon"] == icon_spec("ui/icon.svg", SVG)
    assert row["label"] == offer["label"] == "GitHub"
    assert row["connector"] == offer["connector"] == skill.name
    assert "connector_label" not in row and "connector_label" not in offer
    assert row["readiness"] == offer["readiness"] == {
        "text": "Connector on" if live else "Off in Skills", "tone": "ok" if live else "warn"}
    assert row["revision"] == place.revision > 0
    assert row["facts"] == staged.contributions[0]["facts"]
    # Installed but not on: a save refuses its declared signal until it is (core kinds are never held).
    assert row.get("unpublished_signals") == (None if live else ["signal.words"])
    assert ("source" in row) is live and ("source" in offer) is live
    if live:
        assert row["source"]["url"] == "/api/extensions/connector/status"
    else:
        (skill.skill_dir / "ui/icon.svg").write_text('<svg><script/></svg>', encoding="utf-8")
        assert projection.places_view(tmp_path)["places"][0]["icon"] == {}


def test_loader_readiness_and_unavailable_connector_keep_choices(tmp_path, connector, monkeypatch):
    skill, _staged = connector
    place = projection.connect_declared_place(tmp_path, skill.name)
    monkeypatch.setattr("ouroboros.extension_registry_state.live_contribution_projection", lambda: [])
    skill.enabled = True
    assert projection.place_view(tmp_path, place)["readiness"] == {"text": "Review required", "tone": "warn"}
    monkeypatch.setattr(skill.review, "gate_for", lambda _hash: {"executable_review": True})
    held = projection.place_view(tmp_path, place)
    assert held["readiness"] == {"text": "Connector not running", "tone": "warn"}
    assert held["unpublished_signals"] == ["signal.words"]
    monkeypatch.setattr("ouroboros.skill_loader.discover_skills", lambda _root: [])
    row = projection.place_view(tmp_path, place)
    assert row["readiness"] == {"text": "Connector unavailable", "tone": "warn"}
    assert row["icon"] == {} and row["label"] == "GitHub" and row["wakes"]
    assert "unpublished_signals" not in row, "nothing declared is offered: the saved choice stays, removable"


def test_http_and_tool_connect_are_silent_and_instances_are_distinct(client, tmp_path):
    broad = client.post("/api/places", json={"connector": "connector"}).json()["place"]
    assert broad["state"] == "silent" and broad["label"] == "GitHub"
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    result = json.loads(_configure_presence(ctx, "connect", connector="connector", account_id="writer",
                                          conversation_id="owner/repo", label="owner/repo"))
    instance = result["place"]
    assert instance["state"] == "silent" and instance["label"] == "GitHub · owner/repo"
    assert instance["place_id"] != broad["place_id"]
    assert instance["wakes"] == [{"fact": "mention", "author": "connector"},
                                  {"fact": "signal.words", "author": "connector"}]
    path = "/api/places/" + instance["place_id"]
    saved = client.post(path, json={"state": "active", "expected_revision": instance["revision"]}).json()["place"]
    repeated = client.post("/api/places", json={"connector": "connector", "account_id": "writer",
                           "conversation_id": "owner/repo", "label": "Renamed by connector"}).json()["place"]
    assert repeated == saved
    assert places.get_place(tmp_path, broad["place_id"]).label == "GitHub"
    assert len(client.get("/api/places").json()["places"]) == 2
    assert client.post("/api/places", json={"connector": "connector", "label": None}).status_code == 400
    assert client.post("/api/places", json={"connector": "connector", "account_id": None}).status_code == 400


def test_reading_imported_places_and_their_sheet_never_materializes_or_finishes_the_import(tmp_path, monkeypatch):
    from tests._place_wake_import import legacy_wake_install, wake_declaration
    published = legacy_wake_install(tmp_path, monkeypatch)
    monkeypatch.setattr("ouroboros.skill_loader.discover_skills", lambda _root: [])
    app = Starlette(routes=places_routes())
    app.state.drive_root = tmp_path
    path = places.places_path(tmp_path)
    with TestClient(app) as owner:
        fresh, = owner.get("/api/places").json()["places"]
        assert fresh["wake_review"] == "pending" and not path.exists()
        assert owner.get(f"/api/places/{fresh['place_id']}/rooms").status_code == 200
        ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
        assert json.loads(_configure_presence(ctx, "inspect", fresh["place_id"]))["place"]["wake_review"] == "pending"
        assert not path.exists()
        persisted, = places.load_places(tmp_path)
        published.append(wake_declaration("direct_message"))
        before = path.read_bytes()
        assert "wake_review" not in json.loads(before)["places"][persisted.place_id]
        row, = owner.get("/api/places").json()["places"]
        assert row["wake_review"] == "pending" and row["wake_suggestions"] == ["direct_message"]
        assert row["revision"] == fresh["revision"] == persisted.revision and path.read_bytes() == before


def test_a_preview_reads_one_behavior_profile_for_its_several_legacy_bindings(tmp_path, monkeypatch):
    from ouroboros import skill_loader
    from tests._place_wake_import import legacy_wake_install

    legacy_wake_install(tmp_path, monkeypatch, ids=("a" * 32, "b" * 32))
    read, seen = skill_loader.find_skill, []

    def find(root, name):
        seen.append(name)
        return read(root, name)

    monkeypatch.setattr(skill_loader, "find_skill", find)
    for _ in range(2):
        first, second = places.place_records(tmp_path, persist_import=False).places.values()
        assert first.notes == second.notes and first.conversation_id != second.conversation_id
    assert seen == ["mind", "mind"]  # shared within one preview, not across previews
    assert not places.places_path(tmp_path).exists()


def test_a_finished_import_keeps_every_declared_kind_visible_and_removable(tmp_path, monkeypatch):
    """A finished legacy import offers each contribution's kinds even after removal, with one manifest label."""
    from tests._place_wake_import import admit_live, legacy_wake_install, wake_declaration
    foreign = {**wake_declaration("any_message", "signal.foreign"), "id": "other", "destination": "vendor:Other",
               "facts": ["any_message", {"id": "signal.foreign", "label": "Other item"}]}
    rows = [foreign, wake_declaration("mention"),
            {**wake_declaration("direct_message", "signal.inbox"), "id": "inbox",
             "facts": ["direct_message", {"id": "signal.inbox", "label": "Inbox item"}]},
            {**wake_declaration(), "id": "secondary-inbox",
             "facts": [{"id": "signal.inbox", "label": "Secondary inbox item"}]}]
    published = legacy_wake_install(tmp_path, monkeypatch, published=[foreign])
    monkeypatch.setattr("ouroboros.skill_loader.discover_skills", lambda _root: [])
    place, = places.load_places(tmp_path)
    assert [w.fact for w in place.wakes] == ["legacy_addressed"]
    published[:] = rows
    admitted = admit_live(tmp_path, place)  # the first live event finishes the import
    assert [w.fact for w in admitted.place.wakes] == [
        "legacy_addressed", "mention", "direct_message", "signal.inbox"]
    app = Starlette(routes=places_routes())
    app.state.drive_root = tmp_path
    with TestClient(app) as owner:
        view = owner.get("/api/places").json()
        row, = view["places"]
        assert "wake_review" not in row
        assert row["facts"] == [*places.WAKE_FACTS, {"id": "signal.inbox", "label": "Inbox item"}]
        assert row["labels"] == places.declared_fact_labels(place) == {"signal.inbox": "Inbox item"}
        assert {r["destination"]: r["connected"] for r in view["connectors"]} == {
            "vendor:Other": False, "vendor:Slack": True}
        assert row["destination"] == "vendor:Slack" and row["wake_suggestions"] == ["mention", "direct_message", "signal.inbox"]
        path = f"/api/places/{place.place_id}"
        kept = [w["fact"] for w in row["wakes"] if w["fact"] not in {"signal.inbox", "mention"}]
        saved = owner.post(path, json={"wakes": kept, "expected_revision": row["revision"]})
        assert saved.status_code == 200, "removing a saved kind is never an added undeclared signal"
        assert [w["fact"] for w in saved.json()["place"]["wakes"]] == kept
        without_legacy = [fact for fact in kept if fact != "legacy_addressed"]
        resp = owner.post(path, json={"wakes": without_legacy,
                                      "expected_revision": saved.json()["place"]["revision"]})
        assert resp.status_code == 200, "the migration's own kind stays removable after the explicit review"
        reread, = owner.get("/api/places").json()["places"]
        assert [w["fact"] for w in reread["wakes"]] == without_legacy
        assert reread["facts"] == row["facts"] and reread["labels"] == row["labels"]
        again = owner.post(path, json={"wakes": [*without_legacy, "mention", "signal.inbox"],
                                       "expected_revision": reread["revision"]})
        assert again.status_code == 200
        assert again.json()["place"]["wakes"] == [*reread["wakes"], {"fact": "mention", "author": "owner"},
                                                  {"fact": "signal.inbox", "author": "owner"}]


@pytest.mark.parametrize("store", ["original-v1", "restored-v1"])
def test_old_ambiguous_choice_can_be_kept_from_the_sheet_or_changed_by_the_tool(tmp_path, monkeypatch, store):
    from tests._place_wake_import import admit_live, legacy_wake_install, old_load_places, wake_declaration
    published = legacy_wake_install(tmp_path, monkeypatch, ids=("a" * 32, "b" * 32))
    monkeypatch.setattr("ouroboros.skill_loader.discover_skills", lambda _root: [])
    first, second = places.load_places(tmp_path)
    old_load_places(tmp_path, [places.place_summary(p) for p in (first, second)])
    if store == "original-v1":
        path = places.places_path(tmp_path)
        historical = json.loads(path.read_bytes())
        historical.pop("place_choices", None)
        path.write_text(json.dumps(historical), encoding="utf-8")
    published.append(wake_declaration("direct_message"))
    app = Starlette(routes=places_routes())
    app.state.drive_root = tmp_path
    with TestClient(app) as owner:
        before = places.places_path(tmp_path).read_bytes()
        rows = owner.get("/api/places").json()["places"]
        assert [row["wake_review"] for row in rows] == ["ambiguous", "ambiguous"]
        assert places.places_path(tmp_path).read_bytes() == before
        saved = owner.post(f"/api/places/{first.place_id}", json={"wakes": ["legacy_addressed"],
                                                                 "expected_revision": first.revision}).json()["place"]
        assert saved["revision"] > first.revision and saved["wakes"] == rows[0]["wakes"]
        assert "wake_review" not in saved
        ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
        saved = json.loads(_configure_presence(ctx, "add_wake", second.place_id, fact="mention"))["place"]
        assert saved["wakes"][-1] == {"fact": "mention", "author": "ouroboros"} and "wake_review" not in saved
        published[:] = [wake_declaration("any_message")]
        for place in (first, second):
            assert "any_message" not in [item.fact for item in admit_live(tmp_path, place).place.wakes]
        assert all("wake_review" not in row for row in owner.get("/api/places").json()["places"])


def test_an_old_place_holds_only_the_signals_its_installed_connector_has_not_published(tmp_path, monkeypatch):
    """Review 23.11: a connector installed but not on still declares its suggestions, yet a save adds a new signal
    only from a published declaration (``places.declared_fact_labels``), so adding the whole list was a certain
    refusal. The projection names the held signals; a core suggestion, the saved choice and its removal still save,
    and once the connector is on nothing is held and the signal saves."""
    from tests._place_wake_import import legacy_wake_install, wake_declaration
    published = legacy_wake_install(tmp_path, monkeypatch)
    monkeypatch.setattr("ouroboros.skill_loader.discover_skills", lambda _root: [])
    inbox = {**wake_declaration("direct_message", "signal.inbox"),
             "facts": ["direct_message", {"id": "signal.inbox", "label": "Inbox item"}]}
    declarations = projection.place_declarations
    # The row place_declarations gives an installed, reviewed connector that is not running: validated, unpublished.
    monkeypatch.setattr(projection, "place_declarations",
                        lambda _root: [{**inbox, "readiness": {"text": "Connector not running", "tone": "warn"}}])
    app = Starlette(routes=places_routes())
    app.state.drive_root = tmp_path
    with TestClient(app) as owner:
        row, = owner.get("/api/places").json()["places"]
        assert row["wake_review"] == "pending" and row["wake_suggestions"] == ["direct_message", "signal.inbox"]
        assert row["unpublished_signals"] == ["signal.inbox"]
        path = f"/api/places/{row['place_id']}"
        whole = owner.post(path, json={"wakes": ["legacy_addressed", "direct_message", "signal.inbox"]})
        assert whole.status_code == 400 and whole.json()["code"] == "place_wake_undeclared"
        core = owner.post(path, json={"wakes": ["legacy_addressed", "direct_message"]})
        assert core.status_code == 200, "a core suggestion saves while the connector is off"
        saved = core.json()["place"]
        assert saved["unpublished_signals"] == ["signal.inbox"] and "wake_review" not in saved
        kept = owner.post(path, json={"wakes": ["direct_message"], "expected_revision": saved["revision"]})
        assert kept.status_code == 200, "the saved choice stays removable"
        published.append(inbox)  # the connector is on: its declaration is published
        monkeypatch.setattr(projection, "place_declarations", declarations)
        row, = owner.get("/api/places").json()["places"]
        assert "unpublished_signals" not in row
        added = owner.post(path, json={"wakes": ["direct_message", "signal.inbox"], "expected_revision": row["revision"]})
        assert added.status_code == 200, added.text
        assert [wake["fact"] for wake in added.json()["place"]["wakes"]] == ["direct_message", "signal.inbox"]


def test_a_label_that_already_names_its_connector_is_not_prefixed_again(tmp_path, connector):
    """Audit 05.11 (U015): «GitHub · GitHub · …» came from prefixing a label that already said its connector."""
    named = projection.connect_declared_place(tmp_path, "connector", conversation_id="a/b", label="GitHub · a/b")
    plain = projection.connect_declared_place(tmp_path, "connector", conversation_id="c/d", label="c/d")
    assert (named.label, plain.label) == ("GitHub · a/b", "GitHub · c/d")
    # A label stored doubled before this fix reads its word once on every place surface; the store keeps its bytes.
    doubled = projection.connect_declared_place(tmp_path, "connector", conversation_id="e/f", label="GitHub · GitHub · e/f")
    assert projection.place_view(tmp_path, doubled, brief=True)["label"] == "GitHub · e/f"
    assert projection.place_view(tmp_path, doubled)["label"] == "GitHub · e/f", "the place sheet's full view too"
    assert places.get_place(tmp_path, doubled.place_id).label == "GitHub · GitHub · e/f"
    assert projection.place_view(tmp_path, plain, brief=True)["label"] == "GitHub · c/d"
    # A label that only begins like the word is still prefixed: «GitHubbers» is not «GitHub · …».
    assert projection.connect_declared_place(tmp_path, "connector", conversation_id="g/h", label="GitHubbers").label == "GitHub · GitHubbers"


def test_tool_accepts_only_this_live_connectors_declared_signal(tmp_path, connector, monkeypatch):
    place = projection.connect_declared_place(tmp_path, "connector")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    places.edit_place(tmp_path, place.place_id, "set_wakes", author="owner", wakes=[])
    saved = json.loads(_configure_presence(ctx, "add_wake", place.place_id, fact="signal.words"))
    assert saved["place"]["wakes"] == [{"fact": "signal.words", "author": "ouroboros"}]
    assert "place_wake_undeclared" in _configure_presence(ctx, "add_wake", place.place_id, fact="signal.unknown")
    monkeypatch.setattr("ouroboros.extension_registry_state.live_contribution_projection", lambda: [])
    assert "place_wake_undeclared" in _configure_presence(ctx, "add_wake", place.place_id, fact="signal.words")
    assert json.loads(_configure_presence(ctx, "remove_wake", place.place_id, fact="signal.words"))["place"]["wakes"] == []


def test_http_conflicts_answer_current_record_including_disconnect_and_reconnect(client):
    first = client.post("/api/places", json={"connector": "connector"}).json()["place"]
    path = "/api/places/" + first["place_id"]
    fresh = client.post(path, json={"state": "active", "expected_revision": first["revision"]}).json()["place"]
    for method in ("POST", "DELETE"):
        response = client.request(method, path, json={"expected_revision": first["revision"]})
        assert response.status_code == 409 and response.json()["code"] == "place_revision_conflict"
        assert response.json()["place"] == fresh
    assert client.request("DELETE", path, json={"expected_revision": fresh["revision"]}).status_code == 200
    missing = client.post(path, json={"state": "active", "expected_revision": fresh["revision"]})
    assert missing.status_code == 409 and missing.json()["place"] is None
    again = client.post("/api/places", json={"connector": "connector"}).json()["place"]
    assert again["revision"] > fresh["revision"]
    stale = client.post(path, json={"state": "active", "expected_revision": first["revision"]})
    assert stale.status_code == 409 and stale.json()["place"] == again
    assert client.request("DELETE", path, json={"expected_revision": again["revision"]}).status_code == 200
    assert client.get("/api/places").json()["places"] == []


@pytest.mark.parametrize("unknown", [None, {"spend_by_cause": None}])
def test_display_accounting_unknown_never_loses_a_write(client, tmp_path, monkeypatch, unknown):
    reads = []
    def money(_root, *, allow_stale=False):
        reads.append(allow_stale)
        return unknown
    monkeypatch.setattr("ouroboros.consciousness_allowance.allowance_window", money)
    connected = client.post("/api/places", json={"connector": "connector"})
    assert connected.status_code == 200
    row = connected.json()["place"]
    assert row["spent_24h_usd"] is None
    saved = client.post("/api/places/" + row["place_id"], json={"add_note": "Committed words"})
    assert saved.status_code == 200 and saved.json()["place"]["spent_24h_usd"] is None
    assert places.get_place(tmp_path, row["place_id"]).notes[0].text == "Committed words"
    assert client.get("/api/places").json()["places"][0]["spent_24h_usd"] is None
    assert reads == [True, True, True]
    from ouroboros.presence_admission import focus_allowance
    focus_allowance(tmp_path)
    assert reads[-1] is False, "a display outage must not relax admission's exact read"
    monkeypatch.setattr("ouroboros.consciousness_allowance.allowance_window", lambda _root, **kw: {
        "spend_by_cause": [{"cause": "place:" + row["place_id"], "usd": 0.0}]})
    assert client.get("/api/places").json()["places"][0]["spent_24h_usd"] == 0.0


def test_usage_store_outage_is_only_unknown_money(client, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise OSError("accounting store unavailable")
    monkeypatch.setattr("ouroboros.usage_store.read", unavailable)
    response = client.post("/api/places", json={"connector": "connector"})
    assert response.status_code == 200 and response.json()["place"]["spent_24h_usd"] is None
    assert client.get("/api/places").status_code == 200


def test_model_reads_stored_fields_of_the_same_place_projection(tmp_path, connector):
    from ouroboros.consciousness_view import place_views
    projection.connect_declared_place(tmp_path, "connector")
    brief, = place_views(tmp_path, {"spend_by_cause": []})
    full, = projection.places_view(tmp_path)["places"]
    # Full owner views replace the stored reply destination with the declared service destination.
    assert {key: value for key, value in brief.items() if key != "destination"} == {
        key: full[key] for key in brief if key != "destination"}
    assert not {"icon", "readiness", "account_candidates"} & brief.keys()


def test_places_endpoint_keeps_full_connector_projection(client, connector, monkeypatch):
    skill, _ = connector
    client.post("/api/places", json={"connector": "connector"})
    calls = []
    monkeypatch.setattr("ouroboros.skill_loader.discover_skills", lambda root: calls.append(root) or [skill])
    response = client.get("/api/places")
    assert response.status_code == 200 and len(calls) == 1
    row, = response.json()["places"]
    assert row["icon"] == icon_spec("ui/icon.svg", SVG)
    assert row["readiness"] == {"text": "Connector on", "tone": "ok"}
    assert row["account_candidates"] == []


@pytest.mark.parametrize("switch", [False, True, None, "panic"])
def test_a_place_says_why_it_is_quiet_while_consciousness_is_off(tmp_path, monkeypatch, switch):
    """Owner 06.11: Off left every place that was not Paused only hearing for 28 hours while each surface, the mind's
    configure_presence included, read «active». The place's own view now says why it is quiet, by one cause (owner
    15.11, 5A: the switch off, Panic); the switch is read once a list, a brief view reads none, and a switch this read
    cannot know (state recovering or unreadable) is no claim, though intake stays closed on it."""
    from ouroboros.consciousness_view import place_views
    from supervisor.state import StateRead
    for state in ("active", "silent", "paused"):
        places.connect_place(tmp_path, place_id=state, connector="bridge", provider="chat", conversation_id=state,
                             label=state, state=state)
    checks = []

    def read_state(root=None):
        if switch is None:
            return StateRead("unavailable", "none", {}, reason="state unreadable")
        return StateRead("current", "primary", {"bg_consciousness_enabled": switch is not False})
    monkeypatch.setattr("supervisor.state.read_state", read_state)
    monkeypatch.setattr("ouroboros.consciousness.panic_blocks_wake", lambda root: checks.append(root) or switch == "panic")
    monkeypatch.setattr(projection, "place_declarations", lambda _root: [])
    hearing = {"active", "silent"} if switch in (False, "panic") else set()
    cause = "panic" if switch == "panic" else "off"
    listed = projection.places_view(tmp_path)["places"]
    brief = place_views(tmp_path, {})
    assert len(checks) == 1, "one switch read for the list, never one per place, none for a brief view"
    assert {row["place_id"]: row["quiet"]["cause"] for row in listed if "quiet" in row} == dict.fromkeys(hearing, cause)
    assert all("quiet" not in row for row in brief), "a brief view says no quiet cause"
    # A single view (a write's answer, the mind's inspect) reads the switch itself; nothing is stored.
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    inspected = json.loads(_configure_presence(ctx, "inspect", "active"))["place"]
    assert ("quiet" in inspected) is bool(hearing)
    assert "quiet" not in json.loads(places.places_path(tmp_path).read_text(encoding="utf-8"))["places"]["active"]


def test_brief_view_reads_one_index_pass_and_keeps_words_when_the_store_is_busy(tmp_path, monkeypatch):
    from ouroboros import place_projection, places

    place = places.connect_place(tmp_path, connector="bridge", provider="chat", label="Inbox")
    places.edit_place(tmp_path, place.place_id, "add_note", author="owner", text="Answer politely.")
    place = places.get_place(tmp_path, place.place_id)  # the stored record carries the note
    from ouroboros.place_arrivals import place_facts as read_facts
    rooms = [{"room_chat_id": 7, "unanswered_count": 2, "last_at": "2027-01-01T00:00:00Z"},
             {"room_chat_id": 8, "unanswered_count": None, "last_at": ""}]

    def brief_facts(root, rows, full=True):
        assert full is False, "the brief view must not read activity"
        return {p.place_id: {"rooms": rooms} for p in rows}
    monkeypatch.setattr("ouroboros.place_arrivals.place_facts", brief_facts)
    view = place_projection.place_view(tmp_path, place, allowance={}, brief=True)
    assert view["label"] == "Inbox" and view["notes"][0]["text"] == "Answer politely."
    assert view["unanswered"] is None, "an unknown room count makes the total unknown, never a smaller number"
    assert view["last_activity_at"] == "2027-01-01T00:00:00Z"
    rooms[1]["unanswered_count"] = 1
    assert place_projection.place_view(tmp_path, place, allowance={}, brief=True)["unanswered"] == 3

    from contextlib import contextmanager
    from ouroboros import dialogue_store
    from ouroboros.store import StoreBusy
    dialogue_store.ensure(tmp_path)  # a dialogue store that is busy, not one that holds nothing

    @contextmanager
    def busy(*_args, **_kwargs):
        raise StoreBusy("store busy")
        yield  # pragma: no cover
    monkeypatch.setattr("ouroboros.place_arrivals.place_facts", read_facts)
    monkeypatch.setattr(dialogue_store, "transaction", busy)
    view = place_projection.place_view(tmp_path, place, allowance={}, brief=True)
    assert view["label"] == "Inbox" and view["notes"][0]["text"] == "Answer politely."
    assert view["rooms"] is None and view["unanswered"] is None and view["last_activity_at"] == ""
    # The owner's full view raises with its store, as before: an unreadable list is no answer there.
    with pytest.raises(StoreBusy):
        place_projection.place_view(tmp_path, place, allowance={}, declarations=[])


@pytest.mark.parametrize("mode", ["off", "light", "full"])
def test_owner_views_keep_manifest_labels_safety_and_accounts_without_identity(client, connector, monkeypatch, mode):
    skill, _ = connector
    monkeypatch.setattr("ouroboros.config.runtime_setting", lambda key, default='': mode if key == "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE" else default)
    place = client.post("/api/places", json={"connector": skill.name, "account_id": "account"}).json()["place"]
    assert place["safety_mode"] == mode
    assert place["labels"] == {"mention": "Mentions you", "signal.words": "Words you chose appear"}
    assert "identity" not in place and "can_confirm" not in place
    path = "/api/places/" + place["place_id"]
    confirmed = client.post(path, json={"confirm_owner_account": {"platform_actor_id": "writer"}}).json()["place"]
    assert confirmed["owner_accounts"][0]["platform_actor_id"] == "writer"
    response = client.get("/api/places").json()
    assert response["places"][0]["owner_accounts"] == confirmed["owner_accounts"]
    assert response["connectors"][0]["labels"] == place["labels"]
    for row in response["places"] + response["connectors"]:
        assert "identity" not in row and "can_confirm" not in row
    revoked = client.post(path, json={"revoke_owner_account": "writer"}).json()["place"]
    assert revoked["owner_accounts"] == []


@pytest.mark.parametrize("place_model,circle_model,consciousness_model,value,layer,local", [
    ("place/model", "circle/model", "consciousness/model", "place/model", "place", False),
    ("", "circle/model", "consciousness/model", "circle/model", "circle", False),
    ("", "", "consciousness/model", "consciousness/model", "consciousness", False),
    ("", "", "", "main/model", "consciousness", False),
    ("place/model", None, "", "place/model", "place", False),
    ("", None, "consciousness/model", "consciousness/model", "consciousness", False),
    ("main", "circle/model", "consciousness/model", "main/model", "place", False),
    ("", "light", "consciousness/model", "light/model", "circle", True),  # the alias runs on Light's own box
])
def test_model_route_is_shared_by_brief_full_and_inspect(tmp_path, monkeypatch, place_model, circle_model,
                                                       consciousness_model, value, layer, local):
    monkeypatch.setattr("ouroboros.route_choice.runtime_setting", lambda key, default="": {
        "OUROBOROS_MODEL_CONSCIOUSNESS": consciousness_model, "OUROBOROS_MODEL_LIGHT": "light/model",
        "USE_LOCAL_LIGHT": "true"}.get(key, default))
    monkeypatch.setattr("ouroboros.model_slots._main_model", lambda: "main/model")
    monkeypatch.setattr(projection, "place_declarations", lambda _root: [])
    place = places.connect_place(tmp_path, connector="bridge", provider="chat")
    place = places.edit_place(tmp_path, place.place_id, "set_model", author="owner", model=place_model)
    if circle_model is not None:
        circle = places.create_circle(tmp_path, name="Team", model=circle_model, author="owner")
        place = places.edit_place(tmp_path, place.place_id, "set_circle", author="owner", circle_id=circle["circle_id"])
        assert projection.places_view(tmp_path)["circles"][0]["model"] == circle_model
    before = places.places_path(tmp_path).read_bytes()
    brief = projection.place_view(tmp_path, place, brief=True)
    full = projection.place_view(tmp_path, place)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    inspected = json.loads(_configure_presence(ctx, "inspect", place.place_id))["place"]
    assert brief["model_route"] == full["model_route"] == inspected["model_route"] == {
        "value": value, "layer": layer, "local": local}
    assert full["model"] == place_model
    assert places.places_path(tmp_path).read_bytes() == before  # a projection never freezes the inherited choice


def test_owner_bot_place_does_not_inherit_separate_bot_green_readiness_when_bridge_is_off(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ouroboros import telegram_owner_bot as owner_bot
    from ouroboros.extension_contributions import stage_contributions
    from ouroboros.skill_loader import load_skill

    folder = tmp_path / "skills" / "native" / "telegram-bot"
    (folder / "ui").mkdir(parents=True)
    (folder / "plugin.py").write_text("def register(api): pass\n", encoding="utf-8")
    (folder / "skill.json").write_text(json.dumps({
        "name": "telegram-bot", "type": "extension", "entry": "plugin.py", "permissions": ["presence"],
        "icon": "ui/icon.svg",
        "contributes": [{
            **PLACE, "label": "Telegram", "destination": "vendor:telegram",
            "wakes": ["mention", "signal.words"],
            "facts": [{"id": "mention", "label": "Mentions you"},
                      {"id": "signal.words", "label": "Words you chose appear"}],
            "source": {"route": "status", "refresh_s": 15}, "settings": "connection",
        }],
        "tools": [{"name": "send", "effect": "message", "target": ["to", "cc", "bcc"]}],
    }), encoding="utf-8")
    (folder / "ui/icon.svg").write_text(SVG, encoding="utf-8")
    skill = load_skill(folder, tmp_path)
    api = SimpleNamespace(_staged=SimpleNamespace())
    stage_contributions(api, skill, skill.content_hash)
    staged = api._staged
    expected_icon = icon_spec("ui/icon.svg", SVG)

    monkeypatch.setattr("ouroboros.skill_loader.discover_skills", lambda _root: [skill])
    monkeypatch.setattr("ouroboros.extension_registry_state.live_contribution_projection", lambda: staged.contributions)
    monkeypatch.setattr(owner_bot, "_live_status_route", lambda: True)
    monkeypatch.setattr(owner_bot, "official_connector", lambda _root: True)
    monkeypatch.setattr(owner_bot, "owner_bridge_bot_id", lambda _root: "")  # personal telegram bridge off

    circle = places.create_circle(tmp_path, name="Family", text="Be brief.", author="owner")
    owner_place = places.connect_place(
        tmp_path, connector="telegram-bot", provider="telegram", contribution_id="owner_bot",
        account_id="101", conversation_id="-100101", label="Family group", scope="conversation",
        first={"note": "Keep family chat private.", "description": "Family chat", "circle_id": circle["circle_id"]},
        author="owner",
    )
    separate_place = places.connect_place(
        tmp_path, connector="telegram-bot", provider="telegram", contribution_id="place",
        account_id="202", conversation_id="-100202", label="Ops group", scope="conversation",
        author="owner",
    )
    allowance = {"spend_by_cause": [{"cause": f"place:{owner_place.place_id}", "usd": 1.25}]}

    # 1. Separate bot connector is live ("Connector on"), personal bridge is off -> owner_bot place must NOT look green,
    # must NOT probe separate bot status, and must keep icon, notes, circle/group, spend and Signals.
    view = projection.place_view(tmp_path, owner_place, allowance=allowance)
    assert view["readiness"] == {"text": "Connector unavailable", "tone": "warn"}
    assert "source" not in view
    assert view["icon"] == expected_icon
    assert view["label"] == "Family group"
    assert view["notes"][0]["text"] == "Keep family chat private."
    assert view["circle"]["circle_id"] == circle["circle_id"]
    assert view["description"]["text"] == "Family chat"
    assert view["spent_24h_usd"] == 1.25
    assert view["wake_suggestions"] == ["mention", "signal.words"]
    assert view["facts"] == staged.contributions[0]["facts"]
    assert view["labels"] == {"mention": "Mentions you", "signal.words": "Words you chose appear"}
    assert "unpublished_signals" not in view
    assert view["settings"] == "connection"

    # Separate bot's own group stays green and keeps its own status URL.
    sep_view = projection.place_view(tmp_path, separate_place, allowance=allowance)
    assert sep_view["readiness"] == {"text": "Connector on", "tone": "ok"}
    assert sep_view["source"]["url"] == "/api/extensions/telegram-bot/status"

    # 2. When telegram-bot itself is Off in Skills (non-ok readiness), owner_bot keeps that truthful warn readiness.
    monkeypatch.setattr("ouroboros.extension_registry_state.live_contribution_projection", lambda: [])
    skill.enabled = False
    off_view = projection.place_view(tmp_path, owner_place, allowance=allowance)
    assert off_view["readiness"] == {"text": "Off in Skills", "tone": "warn"}
    assert "source" not in off_view
    assert off_view["icon"] == expected_icon
    assert off_view["notes"][0]["text"] == "Keep family chat private."
    assert off_view["circle"]["circle_id"] == circle["circle_id"]
    assert off_view["spent_24h_usd"] == 1.25
    assert off_view["unpublished_signals"] == ["signal.words"]

    # 3. When personal bridge is back on and owner_bot is projected, direct match restores owner_bot_status and readiness.
    skill.enabled = True
    monkeypatch.setattr("ouroboros.extension_registry_state.live_contribution_projection", lambda: staged.contributions)
    monkeypatch.setattr(owner_bot, "owner_bridge_bot_id", lambda _root: "101")
    monkeypatch.setattr("ouroboros.extension_companion.is_server_process", lambda: True)
    monkeypatch.setattr(owner_bot, "current_bridge", lambda: None)
    stopped_view = projection.place_view(tmp_path, owner_place, allowance=allowance)
    assert stopped_view["readiness"] == {"text": "Bridge not running", "tone": "warn"}
    assert stopped_view["source"]["url"] == "/api/extensions/telegram-bot/owner_bot_status"
    monkeypatch.setattr(owner_bot, "current_bridge", lambda: object())
    live_view = projection.place_view(tmp_path, owner_place, allowance=allowance)
    assert live_view["readiness"] == {"text": "Connector on", "tone": "ok"}
    assert live_view["source"]["url"] == "/api/extensions/telegram-bot/owner_bot_status"
