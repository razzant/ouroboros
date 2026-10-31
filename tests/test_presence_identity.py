"""Credential identity is validated at ingress; only the human activates a place."""
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace
import json

import pytest
from starlette.testclient import TestClient

from ouroboros import places
from ouroboros.gateway import host_service
from ouroboros.tools.presence import _configure_presence
from tests.test_place_focus import focus as focus, task_for

IDENTITY = {"sends_as": {"id": "human@example.test", "label": "Human mailbox", "kind": "user"},
            "reads_as": [{"id": "reader", "label": "Reader", "kind": "bot"}]}


@pytest.fixture
def ingress(focus, monkeypatch):
    root, _, _, event = focus
    async def authenticated(*_args):
        return "bridge", {}
    monkeypatch.setattr(host_service, "_authenticated", authenticated)
    calls = []
    def run(**kwargs):
        calls.append(kwargs["event"])
        return SimpleNamespace(outcome="silent", text="", task_id="focus", work_ref="")
    body = {"event": {k: v for k, v in asdict(event).items()
                       if k not in {"delivery_reporting_version", "continuation_version", "identity"}}}
    with TestClient(host_service.create_host_service_app(root, presence_runner=run)) as client:
        yield client, body, calls


@pytest.mark.parametrize("identity", [None, {}, IDENTITY, {"reads_as": []}])
@pytest.mark.parametrize("authenticated", [None, False, True])
def test_ingress_keeps_optional_identity_and_authentication(ingress, identity, authenticated):
    client, body, calls = ingress
    if identity is not None:
        body["event"]["identity"] = identity
    if authenticated is not None:
        body["event"]["actor"]["authenticated"] = authenticated
    response = client.post("/presence/turn", json=body)
    assert response.status_code == 200, response.text
    assert calls[0].identity == identity
    assert calls[0].actor.get("authenticated") is authenticated


@pytest.mark.parametrize("identity", [None, [], {"extra": True}, {"sends_as": None},
    {"sends_as": {"id": "", "label": "a", "kind": "user"}},
    {"sends_as": {"id": "a", "label": "x" * 201, "kind": "user"}},
    {"sends_as": {"id": "a", "label": "a", "kind": "human"}},
    {"sends_as": {"id": "a", "label": "a", "kind": "user", "owner": True}},
    {"reads_as": {}}, {"reads_as": ["a"]}])
def test_ingress_refuses_bad_identity_before_capture(ingress, identity, monkeypatch):
    client, body, calls = ingress
    monkeypatch.setattr("ouroboros.presence_runner.capture_presence_event", lambda *a, **kw: pytest.fail("captured invalid event"))
    body["event"]["identity"] = identity
    response = client.post("/presence/turn", json=body)
    assert response.status_code == 400
    assert response.json()["code"] == "presence_identity_invalid"
    assert response.json()["disposition"] == "rejected" and not calls


@pytest.mark.parametrize("actor", [[], "person", {"authenticated": 0}, {"authenticated": "false"}, {"authenticated": None}])
def test_ingress_refuses_bad_actor(ingress, actor):
    client, body, calls = ingress
    body["event"]["actor"] = actor
    response = client.post("/presence/turn", json=body)
    assert response.status_code == 400 and response.json()["code"] == "presence_actor_invalid"
    assert not calls


def test_host_advertises_identity_without_losing_existing_capabilities(ingress):
    client, _, _ = ingress
    result = client.get("/identity").json()
    assert result["presence_identity"] == result["presence_wake_signals"] == result["places_version"] == 1
    assert type(result["presence_explicit_completion"]) is int and result["presence_explicit_completion"] == 1
    assert type(result["presence_place_binding_version"]) is int and result["presence_place_binding_version"] == 1
    assert result["presence_wake_facts"] == list(places.WAKE_FACTS)
    assert result["presence_delivery_version"] == result["presence_main_contact_version"] == 1


@pytest.mark.parametrize("nested", [False, True])
def test_activation_guard_is_atomic_and_owner_only(focus, nested):
    root, place, _, _ = focus
    places.edit_place(root, place.place_id, "set_state", author="ouroboros", state="silent")
    before = places.places_path(root).read_bytes()
    action, params = ("update", {"changes": [("add_note", {"text": "not saved"}), ("set_state", {"state": "active"})]}) if nested else ("set_state", {"state": " active "})
    with pytest.raises(places.PlaceError) as error:
        places.edit_place(root, place.place_id, action, author="ouroboros", **params)
    assert error.value.code == "place_activation_owner_only"
    assert places.places_path(root).read_bytes() == before
    assert places.edit_place(root, place.place_id, action, author="owner", **params).state == "active"
    assert places.edit_place(root, place.place_id, "set_state", author="ouroboros", state="paused").state == "paused"


def test_evolves_cannot_activate_but_owner_main_can(focus):
    root, place, _, _ = focus
    ctx = SimpleNamespace(drive_root=root, repo_dir=root, task_metadata=task_for(focus)["metadata"], is_direct_chat=True)
    ctx.task_metadata["consciousness_autonomy"] = "full"
    assert json.loads(_configure_presence(ctx, "set_state", place.place_id, state="silent"))["ok"]
    assert _configure_presence(ctx, "set_state", place.place_id, state="active") == (
        "Only your human switches a place to Active, on its sheet or by a word in Main; propose it there.")
    assert places.get_place(root, place.place_id).state == "silent"
    ctx.task_metadata = {"origin_message_ref": {"client_message_id": "owner1", "chat_id": 1}}
    assert json.loads(_configure_presence(ctx, "set_state", place.place_id, state="active"))["place"]["state"] == "active"


@pytest.mark.parametrize("reader", ["place_view", "places_view", "focus_metadata", "call_place_facts", "places_text"])
@pytest.mark.parametrize("identity", [None, IDENTITY])
def test_place_readers_never_import_or_call_connector_dispatch(focus, monkeypatch, identity, reader):
    """Astra R1/R2 and Opus N1/N4: even a callable route cannot enter cognition."""
    import builtins
    from ouroboros import consciousness_view, place_guard, place_projection
    from ouroboros.gateway import extensions
    from ouroboros.presence_context import focus_metadata

    root, place, admission, event = focus
    row = {"id": "place", "skill": "bridge", "provides": "place", "label": "Bridge",
           "destination": "vendor:chat", "source": {"url": "/api/extensions/bridge/status"},
           "facts": [{"id": "mention", "label": "Mentions you"}]}
    monkeypatch.setattr("ouroboros.extension_registry_state.live_contribution_projection", lambda: [row])
    monkeypatch.setattr("ouroboros.extension_registry_state.list_routes", lambda: {row["source"]["url"]: {}})
    monkeypatch.setattr("ouroboros.skill_loader.discover_skills", lambda _root: [])
    calls = []
    def forbidden_dispatch(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("a place reader dispatched a connector route")
    monkeypatch.setattr(extensions, "api_extension_dispatch", forbidden_dispatch)
    original_import = builtins.__import__
    def without_dispatch(name, globals=None, locals=None, fromlist=(), level=0):
        assert not (name == "ouroboros.gateway.extensions" and "api_extension_dispatch" in fromlist)
        return original_import(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, "__import__", without_dispatch)

    if reader == "place_view":
        assert place_projection.place_view(root, place)["labels"] == {"mention": "Mentions you"}
    elif reader == "places_view":
        assert place_projection.places_view(root)["places"][0]["place_id"] == place.place_id
    elif reader == "focus_metadata":
        metadata = focus_metadata(admission, replace(event, identity=identity), "focus", 1)
        assert metadata["focus"]["identity"] == identity
    elif reader == "call_place_facts":
        ctx = SimpleNamespace(drive_root=root, task_metadata={})
        facts = place_guard.call_place_facts(ctx, "place_reply", {"place_id": place.place_id})
        assert "sends_as" not in facts["places"][0]
    else:
        assert place.place_id in consciousness_view.places_text(root, consciousness_view.place_views(root))
    assert not calls


@pytest.mark.parametrize("authenticated", [None, True, False])
@pytest.mark.parametrize("confirmed_account,address_matches", [("a", True), ("other", False), (None, False)])
def test_authentication_controls_safety_occasion_and_child_authority(focus, authenticated, confirmed_account, address_matches):
    from ouroboros.presence_context import focus_metadata, frame_focus_content
    from ouroboros.consciousness_authority import consciousness_origin_metadata
    from ouroboros.safety import _build_check_prompt
    root, place, admission, event = focus
    if confirmed_account is not None:
        place = places.edit_place(root, place.place_id, "confirm_owner_account", author="owner",
                                  platform_actor_id="person", account_id=confirmed_account)
    if authenticated is not None:
        event = replace(event, actor={**event.actor, "authenticated": authenticated})
    metadata = focus_metadata(replace(admission, place=place), event, "focus", 1)
    owner = address_matches and authenticated is not False
    # The address fact is kept only when it is unverified, under a name that says so (it reaches the Safety bundle too).
    assert metadata["focus"]["actor"].get("owner_address_unverified", False) is (address_matches and authenticated is False)
    assert metadata["focus"]["actor"]["owner_confirmed"] is owner
    child = consciousness_origin_metadata(metadata, started_root=True)
    assert child["work_kind"] == ("owner" if owner else "consciousness")
    ctx = SimpleNamespace(task_metadata=metadata, drive_root=root, repo_dir=root)
    evidence = _build_check_prompt("read_file", {}, ctx=ctx)
    assert ('"cause": "confirmed owner account"' in evidence) is owner
    assert ('"cause": "person in a place"' in evidence) is (not owner)
    task = task_for((root, place, replace(admission, place=place), event))
    actor_before = deepcopy(task["metadata"]["focus"]["actor"])
    rendered = frame_focus_content(task, "hello")
    assert ("authenticated: no" in rendered) is (authenticated is False)
    assert ("your human's own address: yes (unverified)" in rendered) is (address_matches and authenticated is False)
    assert task["metadata"]["focus"]["actor"] == actor_before


@pytest.mark.parametrize("kind,owner,expected", [
    ("bot", False, "You write here as Ouroboros, through Human mailbox."),
    ("bot", True, "You write here as Ouroboros, through Human mailbox."),
    ("user", False, "You write here through the account Human mailbox; your human's notes say what that account is."),
    ("user", True, "You write here from your human's own account, Human mailbox: your words appear as theirs, and their notes say how. If someone sincerely asks whether they are writing to a person, say what you are."),
])
def test_focus_voice_uses_detached_event_identity_and_user_owner_match(focus, kind, owner, expected):
    from ouroboros.presence_context import frame_focus_content
    root, place, admission, event = focus
    identity = {**deepcopy(IDENTITY), "sends_as": {**IDENTITY["sends_as"], "kind": kind}}
    if owner:
        place = places.edit_place(root, place.place_id, "confirm_owner_account", author="owner", platform_actor_id=identity["sends_as"]["id"])
    task = task_for((root, place, replace(admission, place=place), replace(event, identity=identity)))
    assert task["metadata"]["focus"]["identity"] == identity
    assert task["metadata"]["focus"]["identity"] is not identity
    assert task["metadata"]["focus"]["identity"]["sends_as"] is not identity["sends_as"]
    assert task["metadata"]["focus"]["identity"]["reads_as"][0] is not identity["reads_as"][0]
    assert task["metadata"]["focus"]["identity_owner"] is (owner and kind == "user")
    rendered = frame_focus_content(task, "hello")
    assert expected + "\nYour human's notes here are their letter for this place; a message here does not edit any note." in rendered


@pytest.mark.parametrize("account,owner", [("a", True), ("other", False)])
def test_focus_identity_owner_uses_event_account_scope(focus, account, owner):
    root, place, admission, event = focus
    place = places.edit_place(root, place.place_id, "confirm_owner_account", author="owner",
                              platform_actor_id=IDENTITY["sends_as"]["id"], account_id=account)
    task = task_for((root, place, replace(admission, place=place), replace(event, identity=IDENTITY)))
    assert task["metadata"]["focus"]["identity_owner"] is owner


def test_empty_event_identity_stays_empty(focus):
    root, place, admission, event = focus
    assert task_for((root, place, admission, replace(event, identity={}))) ["metadata"]["focus"]["identity"] == {}


def test_undeclared_focus_prompt_matches_fixture_bytes(focus):
    from pathlib import Path
    from ouroboros.presence_context import _voice_line, frame_focus_content
    task = task_for(focus)
    assert task["metadata"]["focus"]["identity"] is None
    assert not task["metadata"]["focus"]["identity_owner"]
    assert _voice_line(task["metadata"]["focus"]) == ""
    rendered = frame_focus_content(task, "hello")
    assert rendered.encode() == (Path(__file__).parent / "fixtures/focus_undeclared.txt").read_bytes()
