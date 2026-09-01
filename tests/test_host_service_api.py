import pathlib
from types import SimpleNamespace

from starlette.testclient import TestClient

from ouroboros.gateway.host_service import AUTH_TOKEN_FILENAME, create_host_service_app
from ouroboros.event_bus import CHAT_OUTBOUND, publish_event
from ouroboros.skill_loader import compute_content_hash, save_enabled, save_review_state, save_skill_grants, SkillReviewState
from ouroboros.utils import atomic_write_json


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


def _seed_presence_behavior(data_dir: pathlib.Path, *, account_wide: bool = False) -> str:
    from ouroboros.presence_bindings import (
        PresenceBinding,
        PresenceEndpoint,
        new_presence_binding_id,
        save_presence_binding,
    )
    from ouroboros.presence_capabilities import (
        PresenceSelection,
        PresenceState,
        PresenceToolTarget,
        presence_state_fingerprint,
        save_presence_state,
    )
    from ouroboros.presence_profile import parse_presence_profile, presence_request_fingerprint
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
) -> None:
    topics = list(subscribe_events or ["chat.outbound"])
    manifest_perms = list(manifest_permissions or ["inject_chat", "subscribe_event"])
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
---
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
        requested_permissions=[
            *(permission for permission in ("inject_chat", "presence") if permission in manifest_perms),
            *(
                f"subscribe_event:{topic}"
                for topic in topics
                if topic != "skill.lifecycle" and "subscribe_event" in manifest_perms
            ),
        ],
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


def test_chat_inject_allows_slash_commands_from_reviewed_skill(tmp_path: pathlib.Path) -> None:
    _seed_token(tmp_path, permissions=["inject_chat"])
    bridge = FakeBridge()
    app = create_host_service_app(tmp_path, bridge_getter=lambda: bridge)
    client = TestClient(app)

    response = client.post(
        "/chat/inject",
        headers={"X-Skill-Token": "token"},
        json={"text": " /panic", "chat_id": 1},
    )

    assert response.status_code == 202
    assert bridge.messages[0]["text"] == " /panic"


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

    app = create_host_service_app(tmp_path, presence_runner=run_presence)
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
    }
    assert captured["admission"].transport_skill == "telegram-bot"
    assert captured["event"].actor["platform_actor_id"] == "user-7"
    assert captured["event"].conversation_key == "telegram:bot-1:room-1:topic-1"


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
            return [{"type": "presence_result", "outcome": "message", "text": "ok"}]

    def run_real_presence(**kwargs):
        return run_presence_turn(
            repo_dir=repo,
            drive_root=tmp_path,
            agent_factory=lambda **_kwargs: Agent(),
            gate=PresenceTurnGate(1),
            **kwargs,
        )

    client = TestClient(create_host_service_app(tmp_path, presence_runner=run_real_presence))
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
            return [{"type": "presence_result", "outcome": "message", "text": "ok"}]

    def run_real_presence(**kwargs):
        return run_presence_turn(
            repo_dir=repo,
            drive_root=tmp_path,
            agent_factory=lambda **_kwargs: Agent(),
            gate=PresenceTurnGate(1),
            **kwargs,
        )

    response = TestClient(
        create_host_service_app(tmp_path, presence_runner=run_real_presence)
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

    # В25c (capinv-447): the over-limit row is a disclosed rejection; the 25
    # in-limit rows stage and the turn proceeds. Ordinal rows stay complete.
    assert response.status_code == 200
    assert agent_calls
    manifest = agent_calls[0]["attachments"]
    assert len(manifest) == 26
    assert [row["ordinal"] for row in manifest] == list(range(26))
    assert manifest[25]["status"] == "rejected"
    assert manifest[25]["reason"] == "attachment_limit_exceeded"
    assert all(row["status"] == "staged" for row in manifest[:25])


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
            return [{"type": "presence_result", "outcome": "message", "text": "ok"}]

    def run_real_presence(**kwargs):
        return run_presence_turn(
            repo_dir=repo,
            drive_root=tmp_path,
            agent_factory=lambda **_kwargs: Agent(),
            gate=PresenceTurnGate(1),
            **kwargs,
        )

    response = TestClient(
        create_host_service_app(tmp_path, presence_runner=run_real_presence)
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
    atomic_write_json(
        tmp_path / "task_results" / f"{work_ref}.json",
        {
            "task_id": work_ref,
            "status": "completed",
            "result": "late answer",
            "metadata": {
                "presence": {"binding_id": binding_id},
                "presence_outcome": "message",
                "presence_result_text": "late answer",
            },
        },
    )
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
    }

    atomic_write_json(
        tmp_path / "task_results" / f"{work_ref}.json",
        {
            "task_id": work_ref,
            "status": "completed",
            "result": "late deferred answer",
            "metadata": {
                "presence": {"binding_id": binding_id},
                "presence_outcome": "deferred",
                "presence_result_text": "late deferred answer",
            },
        },
    )
    deferred = client.get(
        f"/presence/work/{work_ref}",
        params={"binding_id": binding_id},
        headers={"X-Skill-Token": "presence-token"},
    )
    assert deferred.status_code == 200
    assert deferred.json()["text"] == "late deferred answer"

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


def test_chat_inject_defaults_missing_ids_to_non_owner_sentinel(tmp_path: pathlib.Path) -> None:
    _seed_token(tmp_path, skill="transport_bridge", token="token", permissions=["inject_chat"])
    bridge = FakeBridge()
    app = create_host_service_app(tmp_path, bridge_getter=lambda: bridge)
    client = TestClient(app)

    response = client.post(
        "/chat/inject",
        headers={"X-Skill-Token": "token"},
        json={"text": "/panic"},
    )

    assert response.status_code == 202
    assert bridge.messages[0]["chat_id"] == 0
    assert bridge.messages[0]["user_id"] == 0


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


def test_chat_inject_rejects_failed_review_token(tmp_path: pathlib.Path, monkeypatch) -> None:
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    _seed_token(tmp_path, skill="failed", token="token", permissions=["inject_chat"], review_status="blockers")
    app = create_host_service_app(tmp_path, bridge_getter=FakeBridge)
    client = TestClient(app)

    response = client.post(
        "/chat/inject",
        headers={"X-Skill-Token": "token"},
        json={"text": "hello", "chat_id": 1},
    )

    assert response.status_code == 403


def test_identity_rejects_failed_review_token(tmp_path: pathlib.Path, monkeypatch) -> None:
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    _seed_token(tmp_path, skill="failed", token="token", permissions=["inject_chat"], review_status="blockers")
    app = create_host_service_app(tmp_path, bridge_getter=FakeBridge)
    client = TestClient(app)

    response = client.get("/identity", headers={"X-Skill-Token": "token"})

    assert response.status_code == 403


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


def test_chat_inject_allows_slash_command_caption(tmp_path: pathlib.Path) -> None:
    _seed_token(tmp_path, permissions=["inject_chat"])
    bridge = FakeBridge()
    app = create_host_service_app(tmp_path, bridge_getter=lambda: bridge)
    client = TestClient(app)

    response = client.post(
        "/chat/inject",
        headers={"X-Skill-Token": "token"},
        json={"text": "", "image_caption": "/panic", "chat_id": 1},
    )

    assert response.status_code == 202
    assert bridge.messages[0]["image_caption"] == "/panic"


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
