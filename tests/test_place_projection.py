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
    assert projection.place_view(tmp_path, place)["readiness"] == {"text": "Connector not running", "tone": "warn"}
    monkeypatch.setattr("ouroboros.skill_loader.discover_skills", lambda _root: [])
    row = projection.place_view(tmp_path, place)
    assert row["readiness"] == {"text": "Connector unavailable", "tone": "warn"}
    assert row["icon"] == {} and row["label"] == "GitHub" and row["wakes"]


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


def test_brief_view_reads_one_index_pass_and_keeps_words_when_the_store_is_busy(tmp_path, monkeypatch):
    from ouroboros import place_projection, places

    place = places.connect_place(tmp_path, connector="bridge", provider="chat", label="Inbox")
    places.edit_place(tmp_path, place.place_id, "add_note", author="owner", text="Answer politely.")
    place = places.get_place(tmp_path, place.place_id)  # the stored record carries the note
    monkeypatch.setattr(place_projection, "place_activity",
                        lambda root, p: pytest.fail("the brief view must not read activity"))
    rooms = [{"room_chat_id": 7, "unanswered_count": 2, "last_at": "2027-01-01T00:00:00Z"},
             {"room_chat_id": 8, "unanswered_count": None, "last_at": ""}]
    monkeypatch.setattr(places, "place_room_index", lambda root, place_id: rooms)
    view = place_projection.place_view(tmp_path, place, allowance={}, brief=True)
    assert view["label"] == "Inbox" and view["notes"][0]["text"] == "Answer politely."
    assert view["unanswered"] is None, "an unknown room count makes the total unknown, never a smaller number"
    assert view["last_activity_at"] == "2027-01-01T00:00:00Z"
    rooms[1]["unanswered_count"] = 1
    assert place_projection.place_view(tmp_path, place, allowance={}, brief=True)["unanswered"] == 3

    def busy(root, place_id):
        raise RuntimeError("store busy")
    monkeypatch.setattr(places, "place_room_index", busy)
    view = place_projection.place_view(tmp_path, place, allowance={}, brief=True)
    assert view["label"] == "Inbox" and view["notes"][0]["text"] == "Answer politely."
    assert view["rooms"] is None and view["unanswered"] is None and view["last_activity_at"] == ""


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
