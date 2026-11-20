import json
import pathlib
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from ouroboros.task_results import write_task_result

from ouroboros.gateway.host_service import AUTH_TOKEN_FILENAME, create_host_service_app
from ouroboros.event_bus import CHAT_OUTBOUND, publish_event
from ouroboros.skill_loader import (compute_content_hash, requested_skill_permissions, save_enabled, save_review_state,
                                    save_skill_grants, SkillReviewState)
from ouroboros.utils import atomic_write_json
from tests._store_shared import seed_result
from tests.test_launcher_handoff_stop import native_guard  # noqa: F401

@pytest.fixture(autouse=True)
def _host_lane_roots(tmp_path, monkeypatch):
    """A Host-only process: its turns run on the disclosed degraded executor in this process (no pool generation)."""
    from tests.conversation_carrier_support import host_lane, join_host_lane_carriers

    host_lane(monkeypatch, tmp_path)
    yield
    join_host_lane_carriers()  # its carriers settle after the waiters: that tail stays in this world


@pytest.fixture(autouse=True)
def _isolated_presence_runner(monkeypatch):
    """The injected-runner registry is process-global; keep each test's own out of the next."""
    from ouroboros import presence_carrier
    monkeypatch.setattr(presence_carrier, "_RUNNERS", {})


def _inject(tmp_path, runner):
    """Create a Host app whose conversation-lane carrier runs this test's presence runner (1A seam)."""
    from ouroboros.presence_carrier import set_presence_runner
    set_presence_runner(tmp_path, runner)
    return create_host_service_app(tmp_path, presence_runner=runner)


class FakeBridge:
    def __init__(self) -> None:
        self.messages = []
        self._subs = {}

    def enqueue_local_message(self, text, **kwargs):
        self.messages.append({"text": text, **kwargs})
        for callback in list(self._subs.values()):
            callback("reply from host")

    def subscribe_response(self, chat_id, callback):
        self._subs["sub"] = callback
        return "sub"

    def unsubscribe_response(self, subscription_id):
        self._subs.pop(subscription_id, None)


@pytest.fixture
def panic_bridge(native_guard, monkeypatch):  # noqa: F811 — imported pytest fixture
    from ouroboros.server_control import PanicIngress

    bridge = FakeBridge()
    bridge.stopped = threading.Event()
    bridge.panic = PanicIngress(bridge.stopped.set)  # no process coordinator
    monkeypatch.setattr('ouroboros.panic_source.record_source', lambda *a, **kw: None)
    yield bridge
    for thread in threading.enumerate():
        if thread.name == 'panic-ingress':
            thread.join(timeout=3)
            assert not thread.is_alive()


def _seed_presence_behavior(data_dir: pathlib.Path, *, account_wide: bool = False) -> str:
    from ouroboros.presence_bindings import PresenceBinding, PresenceEndpoint
    from tests.frozen_presence_bindings import new_presence_binding_id, save_presence_binding
    from tests.frozen_presence_capabilities import (
        PresenceSelection,
        PresenceState,
        PresenceToolTarget,
        presence_state_fingerprint,
        save_presence_state,
    )
    from tests.frozen_presence_profile import parse_presence_profile, presence_request_fingerprint
    from ouroboros.skill_loader import load_skill

    skill_dir = data_dir / "skills" / "external" / "community-helper"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: community-helper\ndescription: Neutral helper.\nversion: 0.1.0\n"
        "type: instruction\npresence:\n  instructions: Participate helpfully.\n"
        "  capability_requests:\n    - id: history\n      kind: tool\n      required: true\n"
        "      purpose: Read history.\n---\n# Helper\n",
        encoding="utf-8",
    )
    loaded = load_skill(skill_dir, data_dir)
    assert loaded is not None
    save_enabled(data_dir, loaded.name, True)
    save_review_state(data_dir, loaded.name, SkillReviewState(status="pass", content_hash=loaded.content_hash))
    profile = parse_presence_profile(loaded.manifest, skill_dir)
    assert profile is not None
    empty = PresenceState()
    save_presence_state(
        data_dir,
        loaded.name,
        PresenceState((PresenceSelection(
            presence_request_fingerprint(profile.capability_requests[0]),
            PresenceToolTarget("builtin", "chat_history"),
        ),)),
        expected_state_fingerprint=presence_state_fingerprint(empty),
    )
    origin = PresenceEndpoint(
        "telegram", "bot-1", "*" if account_wide else "room-1", "" if account_wide else "topic-1"
    )
    destination = PresenceEndpoint("telegram", "bot-1", "room-1", "topic-1")
    binding = save_presence_binding(
        data_dir,
        PresenceBinding(
            new_presence_binding_id(),
            "telegram-bot",
            loaded.name,
            origin,
            destination,
        ),
    )
    return binding.binding_id


def _seed_token(
    data_dir: pathlib.Path,
    skill: str = "test_skill",
    token: str = "token",
    permissions=None,
    review_status: str = "pass",
    subscribe_events=None,
    manifest_permissions=None,
    contributes=None,
) -> None:
    topics = list(subscribe_events or ["chat.outbound"])
    manifest_perms = list(manifest_permissions or ["inject_chat", "subscribe_event"])
    declarations = f"contributes: {json.dumps(contributes)}\n" if contributes is not None else ""
    skill_dir = data_dir / "skills" / "external" / skill
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {skill}
description: test skill
version: 0.1
type: extension
entry: plugin.py
permissions: [{", ".join(manifest_perms)}]
subscribe_events: [{", ".join(topics)}]
{declarations}---
# Test
""",
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text("def register(api): pass\n", encoding="utf-8")
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_review_state(
        data_dir,
        skill,
        SkillReviewState(status=review_status, content_hash=content_hash),
    )
    save_enabled(data_dir, skill, True)
    save_skill_grants(
        data_dir,
        skill,
        [],
        content_hash=content_hash,
        requested_keys=[],
        granted_permissions=list(permissions or []),
        requested_permissions=requested_skill_permissions(manifest_perms, topics),
    )
    atomic_write_json(
        data_dir / "state" / "skills" / skill / AUTH_TOKEN_FILENAME,
        {
            "token": token,
            "content_hash": content_hash,
            "issued_at": "now",
        },
    )


def test_ui_ws_message_relays_namespaced_event_from_token_identity(tmp_path: pathlib.Path) -> None:
    from ouroboros.extension_loader import extension_surface_name

    _seed_token(tmp_path, skill="wsskill", token="tok", manifest_permissions=["ws_handler"])
    sent: list[dict] = []
    app = create_host_service_app(tmp_path, ws_broadcaster_getter=lambda: sent.append)
    client = TestClient(app)

    # Spoofed body skill/type must be ignored — identity/namespace come from the token.
    resp = client.post(
        "/ui/ws-message",
        headers={"X-Skill-Token": "tok"},
        json={"message_type": "progress", "data": {"pct": 5}, "skill": "evil", "type": "evil"},
    )
    assert resp.status_code == 202
    assert len(sent) == 1
    assert sent[0]["skill"] == "wsskill"
    assert sent[0]["type"] == extension_surface_name("wsskill", "progress")
    assert sent[0]["data"] == {"pct": 5}


def test_ui_ws_message_requires_ws_handler_manifest_permission(tmp_path: pathlib.Path) -> None:
    _seed_token(tmp_path, skill="nows", token="tok", manifest_permissions=["inject_chat"])
    sent: list[dict] = []
    app = create_host_service_app(tmp_path, ws_broadcaster_getter=lambda: sent.append)
    client = TestClient(app)

    resp = client.post("/ui/ws-message", headers={"X-Skill-Token": "tok"}, json={"message_type": "progress", "data": {}})
    assert resp.status_code == 403
    assert sent == []


def test_ui_ws_message_rejects_missing_or_bad_token(tmp_path: pathlib.Path) -> None:
    _seed_token(tmp_path, skill="wsskill", token="tok", manifest_permissions=["ws_handler"])
    app = create_host_service_app(tmp_path, ws_broadcaster_getter=lambda: (lambda m: None))
    client = TestClient(app)

    assert client.post("/ui/ws-message", json={"message_type": "progress"}).status_code == 403
    assert client.post("/ui/ws-message", headers={"X-Skill-Token": "wrong"}, json={"message_type": "progress"}).status_code == 403


def test_identity_requires_skill_token(tmp_path: pathlib.Path) -> None:
    _seed_token(tmp_path, permissions=["inject_chat"])
    app = create_host_service_app(tmp_path, bridge_getter=FakeBridge)
    client = TestClient(app)

    assert client.get("/identity").status_code == 403
    assert client.get("/identity", headers={"X-Skill-Token": "token"}).status_code == 200


def test_identity_accepts_advisory_pass_review(tmp_path: pathlib.Path) -> None:
    _seed_token(tmp_path, permissions=["inject_chat"], review_status="advisory_pass")
    app = create_host_service_app(tmp_path, bridge_getter=FakeBridge)
    client = TestClient(app)

    assert client.get("/identity", headers={"X-Skill-Token": "token"}).status_code == 200


@pytest.mark.parametrize('text', [' /panic', '/status'])
def test_chat_inject_reviewed_skill_cannot_bind_panic_owner(tmp_path: pathlib.Path, panic_bridge, text) -> None:
    _seed_token(tmp_path, permissions=["inject_chat"])
    bridge = panic_bridge
    app = create_host_service_app(tmp_path, bridge_getter=lambda: bridge)
    client = TestClient(app)

    response = client.post(
        "/chat/inject",
        headers={"X-Skill-Token": "token"},
        json={"text": text, "chat_id": 1},
    )

    if text.strip() == '/panic':
        assert response.status_code == 503
        assert 'not authorized' in response.json()['error']
        assert bridge.messages == []
    else:
        assert response.status_code == 202
        assert bridge.messages[0]['text'] == text
    assert not bridge.stopped.is_set()


def test_presence_turn_resolves_binding_and_returns_typed_result(tmp_path: pathlib.Path) -> None:
    _seed_token(
        tmp_path,
        skill="telegram-bot",
        token="presence-token",
        permissions=["presence"],
        manifest_permissions=["presence"],
    )
    binding_id = _seed_presence_behavior(tmp_path)
    captured = {}

    def run_presence(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(outcome="message", text="Hello back", task_id="turn-1", work_ref="")

    app = _inject(tmp_path, run_presence)
    client = TestClient(app)
    response = client.post(
        "/presence/turn",
        headers={"X-Skill-Token": "presence-token"},
        json={
            "binding_id": binding_id,
            "event": {
                "source_event_id": "telegram:bot-1:42",
                "provider": "telegram",
                "account_id": "bot-1",
                "conversation_id": "room-1",
                "thread_id": "topic-1",
                "conversation_key": "caller-controlled-key-is-ignored",
                "actor": {"platform_actor_id": "user-7"},
                "conversation": {"title": "Community"},
                "message": {"message_id": "42"},
                "text": "Hello",
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "status": "completed",
        "outcome": "message",
        "text": "Hello back",
        "turn_ref": "turn-1",
        "work_ref": "",
        "delivery_reporting_version": 0,
    }
    assert captured["admission"].transport_skill == "telegram-bot"
    assert captured["event"].actor["platform_actor_id"] == "user-7"
    assert captured["event"].conversation_key == "telegram:bot-1:room-1:topic-1"


def _install_inbox(root, *, declares=True, wakes=("direct_message",)):
    from ouroboros.extension_contributions import stage_contributions
    from ouroboros.extension_registry_state import _ExtensionRegistrations
    from ouroboros.skill_loader import find_skill

    entry = {"provides": "place", "id": "inbox", "label": "Inbox", "destination": "vendor:Telegram",
             "facts": [*wakes, {"id": "signal.inbox", "label": "Inbox item"}], "wakes": list(wakes)}
    permissions = ["presence", "call_peer"]
    _seed_token(root, skill="telegram-bot", token="inbox-token", permissions=permissions,
                manifest_permissions=permissions, contributes=[entry] if declares else [])
    skill = find_skill(root, "telegram-bot")
    bundle = _ExtensionRegistrations(content_hash=skill.content_hash, skill_dir=str(skill.skill_dir.resolve()))
    stage_contributions(SimpleNamespace(_staged=bundle), skill, skill.content_hash)
    assert not bundle.contribution_problems
    return bundle


def _inbox_event(source, *facts):
    return {"source_event_id": source, "provider": "telegram", "account_id": "bot-1", "conversation_id": "dm",
            "thread_id": "", "conversation_key": "telegram:bot-1:dm:0", "actor": {"platform_actor_id": "sender"},
            "conversation": {}, "message": {"message_id": source}, "text": "Hello", "facts": list(facts)}


@pytest.mark.parametrize("connector_first", [True, False], ids=["connector-first", "core-first"])
@pytest.mark.parametrize("extra_signal", [False, True], ids=["core-fact", "with-connector-signal"])
def test_first_dm_waits_for_its_published_place(tmp_path, monkeypatch, connector_first, extra_signal):
    from ouroboros import dialogue_store, extension_registry_state as registry, places
    from ouroboros.place_intake import HANDED_TO_AUTHOR, ROUTE_TO_MIND

    starts = []
    monkeypatch.delitem(registry._extensions, "telegram-bot", raising=False)
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance", lambda _root: {"status": "available"})
    binding = _seed_presence_behavior(tmp_path, account_wide=True)
    bundle = _install_inbox(tmp_path, declares=connector_first)
    place, = places.load_places(tmp_path)
    assert place.wakes == (places.WakeItem("legacy_addressed", "migration"),)
    if not connector_first:
        bundle = _install_inbox(tmp_path)
    before = places.places_path(tmp_path).read_bytes()
    facts = ["direct_message", "signal.inbox"] if extra_signal else ["direct_message"]
    event, meeting = _inbox_event("first-dm", *facts), _inbox_event("meeting", *facts)

    def run(admission, event, **_kwargs):
        starts.append((event.source_event_id, event.facts, admission.fact_labels))
        return SimpleNamespace(outcome="message", text="Hello back", task_id="focus", work_ref="")

    with TestClient(_inject(tmp_path, run), headers={"x-skill-token": "inbox-token"}) as client:
        call = client.post("/call/open", json={"role": "meeting", "binding_id": binding, "event": meeting})
        assert (call.status_code, call.json()["code"]) == (429, "wake_import_declarations_pending"), call.text
        assert set(call.json()) == {"ok", "code", "error"} and call.json()["ok"] is False
        turn = client.post("/presence/turn", json={"event": event})
        assert turn.status_code == 429, turn.text
        assert (turn.json()["code"], turn.json()["disposition"], turn.json()["retry_after"]) == (
            "wake_import_declarations_pending", "retry", 60)
        assert "field" not in turn.json()
        if not extra_signal:
            with monkeypatch.context() as readers:
                readers.setattr("ouroboros.place_intake.offer_arrival", lambda *_args, **_kwargs: (ROUTE_TO_MIND, ""))
                routed = client.post("/presence/turn", json={"event": event})
                assert (routed.status_code, routed.json()) == (429, turn.json())
                readers.setattr("ouroboros.place_intake.offer_arrival", lambda *_args, **_kwargs: (HANDED_TO_AUTHOR, "reader"))
                followup = {**event, "source_event_id": "author-dm", "message": {"message_id": "3"}}
                follow = client.post("/presence/turn", json={"event": followup})
                assert follow.status_code == 200 and follow.json()["outcome"] == "silent", follow.text
        assert not starts and places.places_path(tmp_path).read_bytes() == before
        assert len(dialogue_store.rows_by_client_id(tmp_path, "first-dm")) == (0 if extra_signal else 1)
        with monkeypatch.context() as declined:
            declined.setitem(registry._extensions, "telegram-bot",
                             replace(bundle, contributions=[], contribution_problems=["Place declaration declined"]))
            later = {**event, "source_event_id": "next-dm", "message": {"message_id": "4"}}
            ordinary = client.post("/presence/turn", json={"event": later})
            if extra_signal:
                assert (ordinary.status_code, ordinary.json()["code"]) == (400, "presence_payload_invalid")
                assert ordinary.json()["disposition"] == "rejected"
            else:
                assert ordinary.status_code == 200 and ordinary.json()["outcome"] == "silent", ordinary.text
        assert not starts and places.places_path(tmp_path).read_bytes() == before
        # Release precisely the contribution the publication owner staged for this reviewed version.
        monkeypatch.setitem(registry._extensions, "telegram-bot", bundle)
        repeated = client.post("/presence/turn", json={"event": event})
        assert repeated.status_code == 200 and repeated.json()["text"] == "Hello back", repeated.text
        assert starts == [("first-dm", tuple(facts), {"signal.inbox": "Inbox item"})]
        assert len(dialogue_store.rows_by_client_id(tmp_path, "first-dm")) == 1
        records = places.place_records(tmp_path)
        assert records.places[binding].wakes == (*place.wakes, places.WakeItem("direct_message", "connector"))
        assert records.migration["presence_bindings_v1"] is True


def _post_inbox(client, path, binding, event, **payload):
    if path == "/call/open":
        payload["role"] = "meeting"
    return client.post(path, json={"binding_id": binding, "event": event, **payload})


@pytest.mark.parametrize("choice", ["new-selected", "migrated", "explicit-empty", "old-writer"])
def test_connected_place_retries_its_connector_signal_independently_of_wake_import(tmp_path, monkeypatch, choice):
    from ouroboros import dialogue_store, extension_registry_state as registry, places
    from tests._place_wake_import import old_load_places

    connector = "telegram-bot"
    monkeypatch.delitem(registry._extensions, connector, raising=False)
    bundle = _install_inbox(tmp_path)
    if choice == "new-selected":
        monkeypatch.setitem(registry._extensions, connector, bundle)
        place = places.connect_place(tmp_path, connector=connector, provider="telegram", account_id="bot-1",
                                     conversation_id="dm", scope="conversation", contribution_id="inbox")
        places.edit_place(tmp_path, place.place_id, "add_wake", author="owner", fact="signal.inbox")
        monkeypatch.delitem(registry._extensions, connector)
    else:
        _seed_presence_behavior(tmp_path, account_wide=True)
        if choice == "migrated":
            monkeypatch.setitem(registry._extensions, connector, bundle)
        place, = places.load_places(tmp_path)
        if choice == "migrated":
            monkeypatch.delitem(registry._extensions, connector)
        elif choice == "explicit-empty":
            places.edit_place(tmp_path, place.place_id, "set_wakes", author="owner", wakes=[])
        else:
            old_load_places(tmp_path, [])
    assert places.place_records(tmp_path).migration["presence_bindings_v1"] is True
    before = places.places_path(tmp_path).read_bytes()
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance", lambda _root: {"status": "available"})
    starts = []

    def run(event, **_kwargs):
        starts.append(event.source_event_id)
        return SimpleNamespace(outcome="message", text="Hello back", task_id="focus", work_ref="")

    event = _inbox_event("inbox-item", "signal.inbox")
    with TestClient(_inject(tmp_path, run), headers={"x-skill-token": "inbox-token"}) as client:
        def responses(incoming):
            replies = [_post_inbox(client, path, place.place_id, incoming) for path in ("/presence/turn", "/call/open")]
            return [(reply.status_code, reply.json().get("code")) for reply in replies]

        retry = [(429, "wake_import_declarations_pending")] * 2
        rejected = [(400, "presence_payload_invalid")] * 2
        assert responses(event) == retry
        assert responses(_inbox_event("other", "signal.inbox", "signal.unknown")) == rejected
        monkeypatch.setitem(registry._extensions, connector, bundle)
        with monkeypatch.context() as reload:
            reload.setattr(registry, "_unloading", {connector})
            assert responses(event) == retry
        with monkeypatch.context() as dropped:
            dropped.setitem(registry._extensions, connector, replace(bundle, contributions=[]))
            assert responses(event) == rejected
        assert not starts and not dialogue_store.rows_by_client_id(tmp_path, event["source_event_id"])
        assert places.places_path(tmp_path).read_bytes() == before
        turn = _post_inbox(client, "/presence/turn", place.place_id, event)
        assert turn.status_code == 200, turn.text
        if choice == "new-selected":
            assert starts == ["inbox-item"] and turn.json()["outcome"] == "message"
        else:
            assert not starts and turn.json()["outcome"] == "silent"
            refused = _post_inbox(client, "/call/open", place.place_id, _inbox_event("meeting", "signal.inbox"))
            assert (refused.status_code, refused.json()["code"]) == (403, "place_no_wake_match")
        assert len(dialogue_store.rows_by_client_id(tmp_path, "inbox-item")) == 1
        assert places.places_path(tmp_path).read_bytes() == before


def test_unpublished_signal_retries_an_unreadable_place_store(tmp_path, monkeypatch):
    from ouroboros import dialogue_store, extension_registry_state as registry, places

    monkeypatch.delitem(registry._extensions, "telegram-bot", raising=False)
    binding = _seed_presence_behavior(tmp_path, account_wide=True)
    _install_inbox(tmp_path)
    places.load_places(tmp_path)
    path = places.places_path(tmp_path)
    event = _inbox_event("unreadable-place", "signal.inbox")
    read = pathlib.Path.read_text

    def unavailable(file, *args, **kwargs):
        if file == path:
            raise OSError("temporary place store outage")
        return read(file, *args, **kwargs)

    with TestClient(create_host_service_app(tmp_path), headers={"x-skill-token": "inbox-token"}) as client:
        with monkeypatch.context() as outage:
            outage.setattr(pathlib.Path, "read_text", unavailable)
            turn = _post_inbox(client, "/presence/turn", binding, event)
            assert (turn.status_code, turn.json()["code"], turn.json()["disposition"]) == (503, "places_unreadable", "retry")
            call = _post_inbox(client, "/call/open", binding, event)
            assert (call.status_code, call.json()["code"]) == (503, "places_unreadable")
        retry = _post_inbox(client, "/presence/turn", binding, event)
        assert (retry.status_code, retry.json()["code"]) == (429, "wake_import_declarations_pending")
    assert not dialogue_store.rows_by_client_id(tmp_path, event["source_event_id"])


def test_pending_group_message_waits_while_a_previous_mailbox_delivery_replays(tmp_path, monkeypatch):
    from ouroboros import dialogue_store, extension_registry_state as registry, places
    from tests.test_place_reactions import author_in, mail

    connector = "telegram-bot"
    monkeypatch.delitem(registry._extensions, connector, raising=False)
    binding = _seed_presence_behavior(tmp_path, account_wide=True)
    bundle = _install_inbox(tmp_path, wakes=("any_message",))
    place, = places.load_places(tmp_path)
    assert [wake.fact for wake in place.wakes] == ["legacy_addressed"]
    before = places.places_path(tmp_path).read_bytes()
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance", lambda _root: {"status": "available"})
    group = {**_inbox_event("group-post", "any_message"), "conversation_id": "group",
             "conversation": {"type": "group"}, "conversation_key": "telegram:bot-1:group:0"}
    previous = {**group, "source_event_id": "previous-post", "message": {"message_id": "previous-post"}}
    lease = author_in(SimpleNamespace(root=tmp_path), group["conversation_key"])
    try:
        with TestClient(create_host_service_app(tmp_path), headers={"x-skill-token": "inbox-token"}) as client:
            def post(event):
                return _post_inbox(client, "/presence/turn", binding, event)

            def unavailable(*_args, **_kwargs):
                raise OSError("settlement unavailable")

            with monkeypatch.context() as earlier:
                earlier.setitem(registry._extensions, connector, replace(bundle, contributions=[]))
                earlier.setattr("ouroboros.place_intake.settle_heard", unavailable)
                assert post(previous).json()["code"] == "presence_io_unavailable"
            assert [m["place_arrival"]["wake"] for m in mail(tmp_path)] == [False]
            waiting = post(group)
            assert (waiting.status_code, waiting.json()["code"]) == (429, "wake_import_declarations_pending")
            assert post(previous).json()["outcome"] == "silent"
            assert len(mail(tmp_path)) == 1 and places.places_path(tmp_path).read_bytes() == before

            monkeypatch.setitem(registry._extensions, connector, bundle)
            assert post(group).json()["outcome"] == "silent"
            assert post(group).json()["outcome"] == "silent"
            assert [m["place_arrival"]["wake"] for m in mail(tmp_path)] == [False, True]
            assert len(dialogue_store.rows_by_client_id(tmp_path, "group-post")) == 1
    finally:
        lease.release()


@pytest.mark.parametrize("materialized", [False, True], ids=["first-import", "pending-import"])
def test_presence_ingress_refusals_preserve_the_import(tmp_path, monkeypatch, materialized):
    from ouroboros import dialogue_store, extension_registry_state as registry, places
    from ouroboros.gateway import host_service
    from ouroboros.presence_events import PresenceTurnError

    connector = "telegram-bot"
    monkeypatch.delitem(registry._extensions, connector, raising=False)
    binding = _seed_presence_behavior(tmp_path, account_wide=True)
    bundle = _install_inbox(tmp_path)
    if materialized:
        place, = places.load_places(tmp_path)
        assert [wake.fact for wake in place.wakes] == ["legacy_addressed"]
    path = places.places_path(tmp_path)
    before = path.read_bytes() if materialized else None
    monkeypatch.setitem(registry._extensions, connector, bundle)
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: False)
    event = _inbox_event("incoming", "direct_message")
    attachment = tmp_path / "state" / "skills" / connector / "note.txt"
    attachment.write_text("Meeting notes", encoding="utf-8")
    with TestClient(create_host_service_app(tmp_path), headers={"x-skill-token": "inbox-token"}) as client:
        def refuse_files(*_args):
            raise ValueError("staged files unavailable")

        with monkeypatch.context() as refused:
            refused.setattr(host_service, "_presence_staged_files", refuse_files)
            turn = _post_inbox(client, "/presence/turn", binding, event, staged_files=[str(attachment)])
            assert (turn.status_code, turn.json()["code"]) == (400, "presence_payload_invalid")
        with monkeypatch.context() as refused:
            refused.setattr("ouroboros.presence_runner.stage_task_attachments",
                            lambda *_args: [{"name": "note.txt", "status": "rejected"}])
            turn = _post_inbox(client, "/presence/turn", binding, event, staged_files=[str(attachment)])
            assert (turn.status_code, turn.json()["code"]) == (409, "presence_attachment_admission_rejected")
        def unknown_prior(*_args):
            raise PresenceTurnError("presence_event_identity_unproven", "source_event_id")

        with monkeypatch.context() as refused:
            refused.setattr("ouroboros.presence_call._stored_turn", unknown_prior)
            call = _post_inbox(client, "/call/open", binding, event)
            assert (call.status_code, call.json()["code"]) == (409, "presence_event_identity_unproven")
        for door in ("/presence/turn", "/call/open"):
            wrong_room = _post_inbox(client, door, binding, {**event, "account_id": "another-bot"})
            assert (wrong_room.status_code, wrong_room.json()["code"]) == (403, "presence_origin_mismatch")
        assert (path.read_bytes() if path.exists() else None) == before
        assert not dialogue_store.rows_by_client_id(tmp_path, "incoming")
        assert _post_inbox(client, "/presence/turn", binding, event).status_code == 200
        assert [w.fact for w in places.get_place(tmp_path, binding).wakes] == ["legacy_addressed", "direct_message"]


def test_presence_turn_attachment_refusal_returns_complete_typed_manifest(
    tmp_path: pathlib.Path,
) -> None:
    from ouroboros.presence_runner import PresenceTurnGate, run_presence_turn

    _seed_token(
        tmp_path,
        skill="telegram-bot",
        token="presence-token",
        permissions=["presence"],
        manifest_permissions=["presence"],
    )
    binding_id = _seed_presence_behavior(tmp_path)
    skill_state = tmp_path / "state" / "skills" / "telegram-bot"
    available = skill_state / "available.txt"
    available.write_text("available", encoding="utf-8")
    rejected = skill_state / ".env"
    rejected.write_text("must not be staged", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    agent_calls = []

    class Agent:
        def handle_task(self, task):
            agent_calls.append(task)
            write_task_result(tmp_path, task["id"], "completed", metadata=task["metadata"], result="ok")
            return [{"type": "presence_result", "outcome": "message", "text": "ok"}]

    def run_real_presence(**kwargs):
        # The carrier's body calls the runner with its own repo, drive and factory; the test's win.
        kwargs.update(repo_dir=repo, drive_root=tmp_path, agent_factory=lambda **_kwargs: Agent())
        return run_presence_turn(gate=PresenceTurnGate(1), **kwargs)

    client = TestClient(_inject(tmp_path, run_real_presence))
    response = client.post(
        "/presence/turn",
        headers={"X-Skill-Token": "presence-token"},
        json={
            "binding_id": binding_id,
            "event": {
                "source_event_id": "telegram:bot-1:42",
                "provider": "telegram",
                "account_id": "bot-1",
                "conversation_id": "room-1",
                "thread_id": "topic-1",
                "conversation_key": "ignored",
                "actor": {"platform_actor_id": "user-7"},
                "conversation": {"title": "Community"},
                "message": {"message_id": "42"},
                "text": "Hello",
            },
            "staged_files": [str(available), str(rejected)],
        },
    )

    # В25c (capinv-447): partial staging is the default — the turn proceeds and
    # the secret-shaped source stays a typed rejected row on the task manifest.
    assert response.status_code == 200
    assert agent_calls, "partial staging must let the turn reach the agent"
    manifest = agent_calls[0]["attachments"]
    assert [row["status"] for row in manifest] == ["staged", "rejected"]
    assert manifest[1]["reason"] == "secret_source"
    assert manifest[1]["rule"], "the exact rule that fired must be named"
    assert pathlib.Path(manifest[0]["abs_path"]).is_file()


def test_presence_turn_host_passes_attachment_limit_to_canonical_staging_owner(
    tmp_path: pathlib.Path,
) -> None:
    """Host shape/confinement validation must not discard ordinal staging rows."""
    from ouroboros.presence_runner import PresenceTurnGate, run_presence_turn

    _seed_token(
        tmp_path,
        skill="telegram-bot",
        token="presence-token",
        permissions=["presence"],
        manifest_permissions=["presence"],
    )
    binding_id = _seed_presence_behavior(tmp_path)
    skill_state = tmp_path / "state" / "skills" / "telegram-bot"
    files = []
    for index in range(26):
        path = skill_state / f"input-{index}.txt"
        path.write_text(str(index), encoding="utf-8")
        files.append(str(path))
    repo = tmp_path / "repo"
    repo.mkdir()
    agent_calls = []

    class Agent:
        def handle_task(self, task):
            agent_calls.append(task)
            write_task_result(tmp_path, task["id"], "completed", metadata=task["metadata"], result="ok")
            return [{"type": "presence_result", "outcome": "message", "text": "ok"}]

    def run_real_presence(**kwargs):
        # The carrier's body calls the runner with its own repo, drive and factory; the test's win.
        kwargs.update(repo_dir=repo, drive_root=tmp_path, agent_factory=lambda **_kwargs: Agent())
        return run_presence_turn(gate=PresenceTurnGate(1), **kwargs)

    response = TestClient(
        _inject(tmp_path, run_real_presence)
    ).post(
        "/presence/turn",
        headers={"X-Skill-Token": "presence-token"},
        json={
            "binding_id": binding_id,
            "event": {
                "source_event_id": "telegram:bot-1:42",
                "provider": "telegram",
                "account_id": "bot-1",
                "conversation_id": "room-1",
                "thread_id": "topic-1",
                "conversation_key": "ignored",
                "actor": {"platform_actor_id": "user-7"},
                "conversation": {"title": "Community"},
                "message": {"message_id": "42"},
                "text": "Hello",
            },
            "staged_files": files,
        },
    )

    # All captured inputs survive behind the full manifest; only inline view is bounded.
    from ouroboros.artifacts import resolve_attachment_manifest
    assert response.status_code == 200
    assert agent_calls
    task = agent_calls[0]
    assert len(task["attachments"]) == 25
    manifest = resolve_attachment_manifest(tmp_path, task["id"], task["task_contract"])
    assert len(manifest) == 26
    assert [row["ordinal"] for row in manifest] == list(range(26))
    assert all(row["status"] == "staged" for row in manifest)


def test_presence_turn_host_passes_internal_missing_and_directory_to_staging_owner(
    tmp_path: pathlib.Path,
) -> None:
    from ouroboros.presence_runner import PresenceTurnGate, run_presence_turn

    _seed_token(
        tmp_path,
        skill="telegram-bot",
        token="presence-token",
        permissions=["presence"],
        manifest_permissions=["presence"],
    )
    binding_id = _seed_presence_behavior(tmp_path)
    skill_state = tmp_path / "state" / "skills" / "telegram-bot"
    missing = skill_state / "missing.txt"
    directory = skill_state / "directory"
    directory.mkdir()
    good = skill_state / "good.txt"
    good.write_text("payload", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    agent_calls = []

    class Agent:
        def handle_task(self, task):
            agent_calls.append(task)
            write_task_result(tmp_path, task["id"], "completed", metadata=task["metadata"], result="ok")
            return [{"type": "presence_result", "outcome": "message", "text": "ok"}]

    def run_real_presence(**kwargs):
        # The carrier's body calls the runner with its own repo, drive and factory; the test's win.
        kwargs.update(repo_dir=repo, drive_root=tmp_path, agent_factory=lambda **_kwargs: Agent())
        return run_presence_turn(gate=PresenceTurnGate(1), **kwargs)

    response = TestClient(
        _inject(tmp_path, run_real_presence)
    ).post(
        "/presence/turn",
        headers={"X-Skill-Token": "presence-token"},
        json={
            "binding_id": binding_id,
            "event": {
                "source_event_id": "telegram:bot-1:42",
                "provider": "telegram",
                "account_id": "bot-1",
                "conversation_id": "room-1",
                "thread_id": "topic-1",
                "conversation_key": "ignored",
                "actor": {"platform_actor_id": "user-7"},
                "conversation": {"title": "Community"},
                "message": {"message_id": "42"},
                "text": "Hello",
            },
            "staged_files": [str(good), str(missing), str(directory)],
        },
    )

    # В25c (capinv-447): the typed rejections reach the staging owner and the
    # turn proceeds with them disclosed (partial staging; a FULLY-rejected set
    # would stay atomic — pinned elsewhere).
    assert response.status_code == 200
    assert agent_calls
    manifest = agent_calls[0]["attachments"]
    assert [row["ordinal"] for row in manifest] == [0, 1, 2]
    assert [row["reason"] for row in manifest][1:] == ["source_missing", "source_not_file"]
    assert manifest[0]["status"] == "staged"


def test_presence_turn_host_rejects_symlink_escape_before_staging(
    tmp_path: pathlib.Path,
) -> None:
    _seed_token(
        tmp_path,
        skill="telegram-bot",
        token="presence-token",
        permissions=["presence"],
        manifest_permissions=["presence"],
    )
    binding_id = _seed_presence_behavior(tmp_path)
    skill_state = tmp_path / "state" / "skills" / "telegram-bot"
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    escaped = skill_state / "escaped.txt"
    escaped.symlink_to(outside)
    called = []
    app = create_host_service_app(tmp_path, presence_runner=lambda **kwargs: called.append(kwargs))
    response = TestClient(app).post(
        "/presence/turn",
        headers={"X-Skill-Token": "presence-token"},
        json={
            "binding_id": binding_id,
            "event": {
                "source_event_id": "telegram:bot-1:43",
                "provider": "telegram",
                "account_id": "bot-1",
                "conversation_id": "room-1",
                "thread_id": "topic-1",
                "conversation_key": "ignored",
                "actor": {"platform_actor_id": "user-7"},
                "conversation": {"title": "Community"},
                "message": {"message_id": "43"},
                "text": "Hello",
            },
            "staged_files": [str(escaped)],
        },
    )
    assert response.status_code == 400
    assert called == []


def test_presence_turn_rejects_event_outside_binding(tmp_path: pathlib.Path) -> None:
    _seed_token(
        tmp_path,
        skill="telegram-bot",
        token="presence-token",
        permissions=["presence"],
        manifest_permissions=["presence"],
    )
    binding_id = _seed_presence_behavior(tmp_path)
    app = create_host_service_app(tmp_path, presence_runner=lambda **_kwargs: None)
    client = TestClient(app)
    response = client.post(
        "/presence/turn",
        headers={"X-Skill-Token": "presence-token"},
        json={
            "binding_id": binding_id,
            "event": {
                "source_event_id": "telegram:bot-1:42",
                "provider": "telegram",
                "account_id": "bot-1",
                "conversation_id": "another-room",
                "thread_id": "topic-1",
                "conversation_key": "telegram:bot-1:another-room:topic-1",
                "actor": {"platform_actor_id": "user-7"},
                "conversation": {},
                "message": {"message_id": "42"},
                "text": "Hello",
            },
        },
    )
    assert response.status_code == 403


def test_presence_turn_account_wide_binding_accepts_any_room_on_exact_account(
    tmp_path: pathlib.Path,
) -> None:
    _seed_token(
        tmp_path,
        skill="telegram-bot",
        token="presence-token",
        permissions=["presence"],
        manifest_permissions=["presence"],
    )
    binding_id = _seed_presence_behavior(tmp_path, account_wide=True)
    captured = {}

    def run_presence(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(outcome="silent", text="", task_id="turn-2", work_ref="")

    client = TestClient(create_host_service_app(tmp_path, presence_runner=run_presence))
    event = {
        "source_event_id": "telegram:bot-1:99",
        "provider": "telegram",
        "account_id": "bot-1",
        "conversation_id": "new-room",
        "thread_id": "topic-9",
        "conversation_key": "ignored",
        "actor": {"platform_actor_id": "user-9"},
        "conversation": {},
        "message": {"message_id": "99"},
        "text": "Hello",
    }
    accepted = client.post(
        "/presence/turn",
        headers={"X-Skill-Token": "presence-token"},
        json={"binding_id": binding_id, "event": event},
    )
    assert accepted.status_code == 200
    assert captured["event"].conversation_key == "telegram:bot-1:new-room:topic-9"

    event["account_id"] = "another-bot"
    refused = client.post(
        "/presence/turn",
        headers={"X-Skill-Token": "presence-token"},
        json={"binding_id": binding_id, "event": event},
    )
    assert refused.status_code == 403


def test_presence_work_returns_only_correlated_terminal_result(tmp_path: pathlib.Path) -> None:
    _seed_token(
        tmp_path,
        skill="telegram-bot",
        token="presence-token",
        permissions=["presence"],
        manifest_permissions=["presence"],
    )
    binding_id = _seed_presence_behavior(tmp_path)
    work_ref = "presence-work-1"
    seed_result(tmp_path, work_ref, {
            "_schema_version": 1,
            "task_id": work_ref,
            "status": "completed",
            "result": "late answer",
            "metadata": {
                "presence": {"binding_id": binding_id},
                "presence_outcome": "message",
                "presence_result_text": "late answer",
            },
        })
    app = create_host_service_app(tmp_path, presence_runner=lambda **_kwargs: None)
    client = TestClient(app)

    response = client.get(
        f"/presence/work/{work_ref}",
        params={"binding_id": binding_id},
        headers={"X-Skill-Token": "presence-token"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "status": "completed",
        "outcome": "message",
        "text": "late answer",
        "work_ref": work_ref,
        "task_id": work_ref,
        "output_ref": "",
        "child_work_ref": "",
        "delivery_reporting_version": 0,
    }

    seed_result(tmp_path, work_ref, {
            "_schema_version": 1,
            "task_id": work_ref,
            "status": "completed",
            "result": "late deferred answer",
            "metadata": {
                "presence": {"binding_id": binding_id},
                "presence_outcome": "deferred",
                "presence_result_text": "late deferred answer",
            },
        })
    deferred = client.get(
        f"/presence/work/{work_ref}",
        params={"binding_id": binding_id},
        headers={"X-Skill-Token": "presence-token"},
    )
    assert deferred.status_code == 200
    assert deferred.json()["text"] == "late deferred answer"
    assert {key: deferred.json()[key] for key in ("task_id", "output_ref", "child_work_ref")} == {
        "task_id": work_ref, "output_ref": "", "child_work_ref": ""}

    wrong = client.get(
        f"/presence/work/{work_ref}",
        params={"binding_id": "0" * 32},
        headers={"X-Skill-Token": "presence-token"},
    )
    assert wrong.status_code == 404


def test_chat_inject_tags_skill_source(tmp_path: pathlib.Path) -> None:
    _seed_token(tmp_path, skill="telegram_bridge", token="token", permissions=["inject_chat"])
    bridge = FakeBridge()
    app = create_host_service_app(tmp_path, bridge_getter=lambda: bridge)
    client = TestClient(app)

    response = client.post(
        "/chat/inject",
        headers={"X-Skill-Token": "token"},
        json={"text": "hello", "chat_id": 1234, "sender_label": "Telegram"},
    )

    assert response.status_code == 202
    assert bridge.messages[0]["source"] == "skill:telegram_bridge"
    assert bridge.messages[0]["chat_id"] == 1234


def test_chat_inject_preserves_transport_metadata(tmp_path: pathlib.Path) -> None:
    _seed_token(tmp_path, skill="transport_bridge", token="token", permissions=["inject_chat"])
    bridge = FakeBridge()
    app = create_host_service_app(tmp_path, bridge_getter=lambda: bridge)
    client = TestClient(app)

    response = client.post(
        "/chat/inject",
        headers={"X-Skill-Token": "token"},
        json={
            "text": "hello",
            "chat_id": 1234,
            "transport": {"kind": "messenger", "conversation_id": "abc", "sender_label": "Messenger"},
        },
    )

    assert response.status_code == 202
    assert bridge.messages[0]["transport"] == {"kind": "messenger", "conversation_id": "abc", "sender_label": "Messenger"}


@pytest.mark.parametrize('text', ['/panic', 'hello'])
def test_chat_inject_defaults_missing_ids_to_non_owner_sentinel(tmp_path: pathlib.Path, panic_bridge, text) -> None:
    _seed_token(tmp_path, skill="transport_bridge", token="token", permissions=["inject_chat"])
    bridge = panic_bridge
    bridge.panic._owner = [True, (17, 23)]  # positively observed owner; absent ids cannot borrow it
    app = create_host_service_app(tmp_path, bridge_getter=lambda: bridge)
    client = TestClient(app)

    response = client.post(
        "/chat/inject",
        headers={"X-Skill-Token": "token"},
        json={"text": text},
    )

    if text == '/panic':
        assert response.status_code == 503
        assert bridge.messages == []
    else:
        assert response.status_code == 202
        assert bridge.messages[0]["chat_id"] == 0
        assert bridge.messages[0]["user_id"] == 0
    assert not bridge.stopped.is_set()


def test_chat_inject_wait_for_response_unsubscribes(tmp_path: pathlib.Path) -> None:
    _seed_token(tmp_path, skill="waiter", token="token", permissions=["inject_chat"])
    bridge = FakeBridge()
    app = create_host_service_app(tmp_path, bridge_getter=lambda: bridge)
    client = TestClient(app)

    response = client.post(
        "/chat/inject",
        headers={"X-Skill-Token": "token"},
        json={"text": "hello", "chat_id": -1234, "wait_for_response": True, "timeout_sec": 5},
    )

    assert response.status_code == 200
    assert response.json()["response"] == "reply from host"
    assert bridge._subs == {}

    # A wait on a human/project chat is refused: the subscription resolves on
    # the FIRST non-progress frame, which on a shared chat can be any
    # concurrent task's (or live proactive) frame, never reliably "the reply".
    refused = client.post(
        "/chat/inject",
        headers={"X-Skill-Token": "token"},
        json={"text": "hello", "chat_id": 1234, "wait_for_response": True, "timeout_sec": 5},
    )
    assert refused.status_code == 400
    assert "A2A-allocated" in refused.json()["error"]
    assert bridge._subs == {}


def test_allocate_internal_chat_ids_are_distinct(tmp_path: pathlib.Path) -> None:
    _seed_token(tmp_path, skill="a2a", token="token", permissions=["inject_chat"])
    app = create_host_service_app(tmp_path, bridge_getter=FakeBridge)
    client = TestClient(app)

    first = client.post(
        "/chat/allocate-internal",
        headers={"X-Skill-Token": "token"},
        json={"range_name": "a2a"},
    ).json()["chat_id"]
    second = client.post(
        "/chat/allocate-internal",
        headers={"X-Skill-Token": "token"},
        json={"range_name": "a2a"},
    ).json()["chat_id"]

    assert first < 0
    assert second < 0
    assert second != first


def test_chat_inject_requires_permission_grant(tmp_path: pathlib.Path) -> None:
    _seed_token(tmp_path, skill="unprivileged", token="token", permissions=[])
    bridge = FakeBridge()
    app = create_host_service_app(tmp_path, bridge_getter=lambda: bridge)
    client = TestClient(app)

    response = client.post(
        "/chat/inject",
        headers={"X-Skill-Token": "token"},
        json={"text": "hello", "chat_id": 1},
    )

    assert response.status_code == 403
    assert bridge.messages == []


def test_chat_inject_rejects_disabled_skill_token(tmp_path: pathlib.Path) -> None:
    _seed_token(tmp_path, skill="disabled", token="token", permissions=["inject_chat"])
    save_enabled(tmp_path, "disabled", False)
    app = create_host_service_app(tmp_path, bridge_getter=FakeBridge)
    client = TestClient(app)

    response = client.post(
        "/chat/inject",
        headers={"X-Skill-Token": "token"},
        json={"text": "hello", "chat_id": 1},
    )

    assert response.status_code == 403


def _seed_blocker_token(data_dir: pathlib.Path, strict_source: bool) -> None:
    """A token of a skill whose review found blockers; a source that reviews each skill keeps them for the owner."""
    from ouroboros.skill_install_record import write_install_record

    _seed_token(data_dir, skill="failed", token="token", permissions=["inject_chat"], review_status="blockers")
    if strict_source:
        write_install_record(data_dir, "failed", {"basis": "review", "release": {"digest": "d"}, "source": {"id": "hub"}})


@pytest.mark.parametrize("strict_source,status", [(False, 202), (True, 403)])
def test_chat_inject_follows_the_review_gate_of_a_token_with_blocker_findings(tmp_path: pathlib.Path, strict_source,
                                                                              status) -> None:
    _seed_blocker_token(tmp_path, strict_source)
    app = create_host_service_app(tmp_path, bridge_getter=FakeBridge)
    client = TestClient(app)

    response = client.post(
        "/chat/inject",
        headers={"X-Skill-Token": "token"},
        json={"text": "hello", "chat_id": 1},
    )

    assert response.status_code == status


@pytest.mark.parametrize("strict_source,status", [(False, 200), (True, 403)])
def test_identity_follows_the_review_gate_of_a_token_with_blocker_findings(tmp_path: pathlib.Path, strict_source,
                                                                           status) -> None:
    _seed_blocker_token(tmp_path, strict_source)
    app = create_host_service_app(tmp_path, bridge_getter=FakeBridge)
    client = TestClient(app)

    response = client.get("/identity", headers={"X-Skill-Token": "token"})

    assert response.status_code == status


def test_events_websocket_receives_granted_topic(tmp_path: pathlib.Path) -> None:
    _seed_token(tmp_path, skill="listener", token="token", permissions=["subscribe_event:chat.outbound"])
    app = create_host_service_app(tmp_path, bridge_getter=FakeBridge)
    client = TestClient(app)

    with client.websocket_connect("/events", headers={"X-Skill-Token": "token"}) as ws:
        ws.send_json({"type": "subscribe", "topic": CHAT_OUTBOUND})
        assert ws.receive_json()["type"] == "subscribed"
        publish_event(CHAT_OUTBOUND, {"text": "hello"})
        message = ws.receive_json()

    assert message["type"] == "event"
    assert message["topic"] == CHAT_OUTBOUND
    assert message["data"]["text"] == "hello"


def test_events_websocket_allows_manifest_declared_skill_lifecycle_without_grant(tmp_path: pathlib.Path) -> None:
    _seed_token(tmp_path, skill="listener", token="token", permissions=[], subscribe_events=["skill.lifecycle"])
    app = create_host_service_app(tmp_path, bridge_getter=FakeBridge)
    client = TestClient(app)

    with client.websocket_connect("/events", headers={"X-Skill-Token": "token"}) as ws:
        ws.send_json({"type": "subscribe", "topic": "skill.lifecycle"})
        assert ws.receive_json()["type"] == "subscribed"


def test_events_websocket_rejects_ungranted_topic(tmp_path: pathlib.Path) -> None:
    _seed_token(tmp_path, skill="listener", token="token", permissions=[])
    app = create_host_service_app(tmp_path, bridge_getter=FakeBridge)
    client = TestClient(app)

    with client.websocket_connect("/events", headers={"X-Skill-Token": "token"}) as ws:
        ws.send_json({"type": "subscribe", "topic": CHAT_OUTBOUND})
        message = ws.receive_json()

    assert message["type"] == "error"
    assert "lacks grant" in message["error"]


@pytest.mark.parametrize('user_id,chat_id', [(0, 1), (17, 99), (99, 23)])
def test_chat_inject_panic_caption_requires_bound_owner(tmp_path: pathlib.Path, panic_bridge, user_id, chat_id) -> None:
    _seed_token(tmp_path, permissions=["inject_chat"])
    bridge = panic_bridge
    bridge.panic._owner = [True, (17, 23)]  # same cached binding as the production emergency door
    app = create_host_service_app(tmp_path, bridge_getter=lambda: bridge)
    client = TestClient(app)

    assert client.post('/chat/inject', json={'text': '/panic', 'chat_id': 23, 'user_id': 17}).status_code == 403
    assert not bridge.stopped.is_set() and bridge.messages == []
    response = client.post(
        "/chat/inject",
        headers={"X-Skill-Token": "token"},
        json={"text": "", "image_caption": "/panic", "chat_id": chat_id, "user_id": user_id},
    )

    if (user_id, chat_id) == (17, 23):
        assert response.status_code == 202
        assert response.json() == {"ok": True, "status": "accepted"}
        assert bridge.stopped.wait(3)
    else:
        assert response.status_code == 503
        assert not bridge.stopped.is_set()
    assert bridge.messages == [], 'accepted and refused Panic both bypass ordinary admission'


def test_chat_inject_allows_slash_command_caption_even_with_text(tmp_path: pathlib.Path) -> None:
    _seed_token(tmp_path, permissions=["inject_chat"])
    bridge = FakeBridge()
    app = create_host_service_app(tmp_path, bridge_getter=lambda: bridge)
    client = TestClient(app)

    response = client.post(
        "/chat/inject",
        headers={"X-Skill-Token": "token"},
        json={"text": "photo", "image_caption": "/panic", "chat_id": 1},
    )

    assert response.status_code == 202
    assert bridge.messages[0]["text"] == "photo"
    assert bridge.messages[0]["image_caption"] == "/panic"


@pytest.fixture(autouse=True)
def _focus_authority_available(monkeypatch):
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance", lambda _root: {"status": "available", "remaining_usd": 10})
