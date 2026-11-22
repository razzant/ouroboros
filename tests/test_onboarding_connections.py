"""A built-in skill's declared connection card (ouroboros/onboarding_connections.py; DESIGN §7, CREATING_SKILLS).

A fixture skill stands for the Telegram skill: its card reaches the Skills step without its function, keys or
routes; the token check calls the declared function of a built-in skill only and never echoes the token or the
address; the completion saves the token alone under its declared key in its one write, keeps the code out of Settings,
and refuses a card's values from the Telegram Mini App; after the write the owner's consent is recorded, the token key
granted where reviewed skills are not granted automatically, and the code with the options posted once to the route
they declare, after which the code leaves the record.
"""
from __future__ import annotations

import os
import pathlib

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import onboarding_connections as cards
from ouroboros import skill_starter

CODE = "A" * 10 + "b_-" + "9" * 19  # 32 characters, the shape a skill mints
CARD = """connection_card:
  title: "Phone via Demo"
  pitch: "Chat anywhere"
  steps:
    - {text: "Open the maker", link: "https://example.test/maker", action: "Open the maker"}
    - {text: "Paste the token here"}
  token: botToken
  pairing: {code: {key: DEMO_CODE, store: skill_settings, route: settings/save}, link: "https://example.test/{username}?start={code}", identity: "IDENTITY", hint: "Scan it", note: "Finish now"}
  options:
    - {key: DEMO_NOTIFY, label: "Send results here", "on": "on", "off": "off", new_connection: true, store: skill_settings, route: settings/save}
"""
MANIFEST = """---
name: demo
description: Test integration.
version: 1
type: script
runtime: python3
timeout_sec: 30
env_from_settings: [DEMO_TOKEN]
secret_fields:
  botToken: DEMO_TOKEN
CARD---
Demo.
"""
IDENT = '''
CALLS = []


def check(token, api_base=None, *, timeout=8.0):
    CALLS.append(timeout)
    if token == "1:boom":
        raise RuntimeError("secret 1:boom at https://api.example.test/bot1:boom/getMe")
    if token.startswith("1:"):
        return {"ok": True, "username": "demo_bot", "id": 1}
    return {"ok": False, "cause": "refused"}
'''


def _skill(root: pathlib.Path, card: str = CARD, identity: str = "lib/ident.py:check") -> pathlib.Path:
    (root / "demo" / "lib").mkdir(parents=True, exist_ok=True)
    (root / "demo" / "SKILL.md").write_text(MANIFEST.replace("CARD", card.replace("IDENTITY", identity)), encoding="utf-8")
    (root / "demo" / "lib" / "ident.py").write_text(IDENT, encoding="utf-8")
    return root


@pytest.fixture
def skills(tmp_path, monkeypatch):
    root = _skill(tmp_path / "skills")
    monkeypatch.setattr("ouroboros.skill_builtin.bundled_skills_dir", lambda: root)
    return root


def _entry(**changes):
    return {"demo": {"fields": {"botToken": "1:abc"}, "code": CODE, "options": {"DEMO_NOTIFY": "on"}, **changes}}


def test_the_step_sees_the_card_s_words_but_never_its_function_keys_or_routes(skills):
    seen = cards.starter_connection("demo")
    assert seen == {
        "skill": "demo", "token": "botToken", "title": "Phone via Demo", "pitch": "Chat anywhere",
        "steps": [{"text": "Open the maker", "link": "https://example.test/maker", "action": "Open the maker"},
                  {"text": "Paste the token here", "link": "", "action": ""}],
        "pairing": {"link": "https://example.test/{username}?start={code}", "hint": "Scan it", "note": "Finish now"},
        "options": [{"key": "DEMO_NOTIFY", "label": "Send results here", "on": "on", "off": "off", "new_connection": True}],
    }
    assert cards.card_keys("demo") == ("DEMO_TOKEN", "DEMO_CODE")
    # A card missing any part the host needs is no card: the skill stays an ordinary row.
    for broken in (CARD.replace("?start={code}", ""), CARD.replace("store: skill_settings, route: settings/save}, link", "store: core, route: settings/save}, link"),
                   CARD.replace("store: skill_settings, route: settings/save}\n", "store: core, route: settings/save}\n"),
                   CARD.replace("key: DEMO_NOTIFY", "key: DEMO_CODE"), CARD.replace("token: botToken", "token: apiKey"),
                   CARD.replace("https://example.test/{username}", "http://example.test/{username}")):
        _skill(skills, broken)
        assert cards.declared_card("demo") is None and cards.starter_connection("demo") is None, broken
    _skill(skills, CARD, identity="../outside.py:check")
    assert cards.declared_card("demo") is None
    _skill(skills, "")
    assert cards.declared_card("demo") is None, "no card declared"
    assert cards.declared_card("nope") is None and cards.declared_card("../demo") is None


def test_the_starter_item_carries_the_card_and_drops_the_code_from_its_prerequisites(skills, tmp_path, monkeypatch):
    offer = [{"id": "builtin/demo", "name": "demo", "source_id": "builtin", "category": ["messaging"],
              "needs_keys": ["DEMO_TOKEN", "DEMO_CODE", "DEMO_REGION"]},
             {"id": "builtin/other", "name": "other", "source_id": "builtin", "category": ["messaging"],
              "needs_keys": ["DEMO_CODE"]}]
    monkeypatch.setattr(skill_starter, "_offer", lambda *a, **k: [dict(item) for item in offer])
    monkeypatch.setattr(skill_starter, "starter_ids", lambda *a, **k: [])
    monkeypatch.setattr("ouroboros.skill_source_catalog.snapshot_for", lambda *a, **k: (None, {}))
    items = {item["id"]: item for group in skill_starter.starter_groups(tmp_path, rows=[])["groups"]
             for item in group["items"]}
    assert items["builtin/demo"]["connection"]["skill"] == "demo"
    assert items["builtin/demo"]["needs_keys"] == ["DEMO_TOKEN", "DEMO_REGION"], "the code the card fills is no prerequisite"
    assert "connection" not in items["builtin/other"] and items["builtin/other"]["needs_keys"] == ["DEMO_CODE"]


def _client(*routes: Route, drive_root: pathlib.Path | None = None) -> TestClient:
    app = Starlette(routes=list(routes))
    if drive_root is not None:
        app.state.drive_root = drive_root
    return TestClient(app)


def test_the_token_check_runs_the_declared_function_of_a_built_in_skill_and_names_nothing_else(skills):
    client = _client(Route("/check", endpoint=cards.api_connection_identity, methods=["POST"]))
    assert client.post("/check", json={"skill": "demo", "token": " 1:abc "}).json() == {"ok": True, "username": "demo_bot"}
    assert client.post("/check", json={"skill": "demo", "token": "2:abc"}).json() == {"ok": False, "cause": "refused"}
    assert client.post("/check", json={"skill": "demo", "token": ""}).json() == {"ok": False, "cause": "refused"}
    failed = client.post("/check", json={"skill": "demo", "token": "1:boom"})
    assert failed.status_code == 500 and "1:boom" not in failed.text and "example.test" not in failed.text
    assert client.post("/check", json={"skill": "nope", "token": "1:abc"}).status_code == 404
    assert client.post("/check", content=b"[1]").status_code == 400
    # The function is loaded from the app's own copy of the skill: a link out of it runs nothing.
    outside = skills.parent / "outside.py"
    outside.write_text(IDENT, encoding="utf-8")
    module = skills / "demo" / "lib" / "ident.py"
    module.unlink()
    os.symlink(outside, module)
    escaped = client.post("/check", json={"skill": "demo", "token": "1:abc"})
    assert escaped.status_code == 500 and "username" not in escaped.text


def test_finish_saves_the_token_alone_and_refuses_anything_else(skills):
    body = {"connections": _entry(fields={"botToken": " 1:abc "}, options={"DEMO_NOTIFY": "off"})}
    assert cards.completion_settings(body) == {"DEMO_TOKEN": "1:abc"}, "the code and the options stay out of Settings"
    assert skill_starter.onboarding_skill_choices(body, {}) == {"DEMO_TOKEN": "1:abc"}
    assert cards.completion_settings({}) == {} and cards.completion_settings({"connections": {}}) == {}
    for bad in ({"nope": _entry()["demo"]},
                _entry(fields={}),
                _entry(fields={"botToken": "1:a", "pairingCode": CODE}),
                _entry(fields={"botToken": "1 a"}),
                _entry(code="short_code"),  # not the shape a skill mints
                _entry(code=CODE + "!"),
                {"demo": {"fields": {"botToken": "1:a"}, "options": {}}},
                _entry(options={"DEMO_NOTIFY": "yes"}),
                _entry(options={"OTHER": "on"}),
                ["demo"]):
        with pytest.raises(ValueError) as refused:
            cards.completion_settings({"connections": bad})
        assert CODE not in str(refused.value) and "1:a" not in str(refused.value)


def test_the_telegram_mini_app_cannot_arm_a_code_and_a_saved_completion_records_the_consent(skills, tmp_path, monkeypatch):
    reached, settled = [], []
    answer = {"status": 200, "body": {"ok": True, "status": "saved"}}

    async def completion(request):
        reached.append(await request.json())
        return JSONResponse(answer["body"], status_code=answer["status"])

    monkeypatch.setattr(cards, "settle_later", lambda drive_root, repo_dir=None: settled.append(drive_root))
    drive = tmp_path / "data"
    client = _client(Route("/complete", endpoint=cards.guard_completion(completion), methods=["POST"]), drive_root=drive)
    connections = {"connections": _entry()}
    refused = client.post("/complete", json=connections, headers={cards.MINIAPP_HEADER: "1"})
    assert refused.status_code == 403 and refused.json()["code"] == "connection_owner_only" and reached == []
    assert client.post("/complete", json={"OUROBOROS_MODEL": "x"}, headers={cards.MINIAPP_HEADER: "1"}).status_code == 200
    assert "connections" not in skill_starter.read_marker(drive) and settled == []
    # A completion that saved nothing records nothing; one that saved records the owner's consent and settles it.
    answer.update(status=400, body={"ok": False, "saved": False, "error": "no Main model"})
    assert client.post("/complete", json=connections).status_code == 400
    assert "connections" not in skill_starter.read_marker(drive) and settled == []
    answer.update(status=200, body={"ok": True, "status": "saved"})
    assert client.post("/complete", json=connections).status_code == 200, "the app and browsers pass"
    record = skill_starter.read_marker(drive)["connections"][0]
    assert (record["token_key"], record["code"], record["options"]) == ("DEMO_TOKEN", CODE, {"DEMO_NOTIFY": "on"})
    assert settled == [drive] and len(reached) == 3 and reached[-1] == connections


def test_after_finish_the_token_is_granted_and_the_code_posted_once_with_the_options(skills, tmp_path, monkeypatch):
    drive = tmp_path / "data"
    assert len(cards.record_consent(drive, _entry())) == 1
    installed, grants, posts = [False], [], []
    monkeypatch.setattr(cards, "_installed", lambda _drive, _name: installed[0])
    monkeypatch.setattr("ouroboros.config.get_auto_grant_enabled", lambda: False)

    def grant(ctx, name, action, **kwargs):
        grants.append((name, action, kwargs["items"], kwargs["_owner_actor"], kwargs["_enable_after_grant"]))
        return {"ok": True}

    monkeypatch.setattr("ouroboros.skill_lifecycle_actions.run_skill_action", grant)
    answers = iter([409, 200, 200])

    def post(path, body):
        posts.append((path, dict(body)))
        return next(answers)

    record = cards.settle(drive, post=post)[0]
    assert (record["grant"], record["post"], grants, posts) == ("pending", "pending", [], []), "not installed yet"
    installed[0] = True  # the starter install's end: granted, but the skill's route does not serve yet
    record = cards.settle(drive, post=post)[0]
    assert grants == [("demo", "grant", ["DEMO_TOKEN"], "owner_ui", True)], "the token alone; the code needs no grant"
    assert (record["post"], record["code"]) == ("waiting", CODE), "no timer, no retry: the next event posts"
    record = cards.settle(drive, post=post)[0]  # the extension's load
    assert posts == [("/api/extensions/demo/settings/save", {"DEMO_CODE": CODE, "DEMO_NOTIFY": "on"})] * 2, \
        "one body for the route, posted once it serves"
    assert (record["grant"], record["post"], "code" in record) == ("granted", "posted", False)
    assert "code" not in skill_starter.read_marker(drive)["connections"][0], "the code leaves the host's record"
    cards.settle(drive, post=post)
    assert len(grants) == 1 and len(posts) == 2, "settled cards are not settled again"
    # Where reviewed skills are granted automatically, the install's own grant covers the token.
    cards.record_consent(drive, _entry())
    monkeypatch.setattr("ouroboros.config.get_auto_grant_enabled", lambda: True)
    assert cards.settle(drive, post=post)[0]["grant"] == "automatic" and len(grants) == 1


def test_failures_are_recorded_never_raised_and_a_refused_post_drops_the_code(skills, tmp_path, monkeypatch):
    drive = tmp_path / "data"
    cards.record_consent(drive, _entry())
    monkeypatch.setattr(cards, "_installed", lambda _drive, _name: True)
    monkeypatch.setattr("ouroboros.config.get_auto_grant_enabled", lambda: False)
    monkeypatch.setattr("ouroboros.skill_lifecycle_actions.run_skill_action", lambda *a, **k: {"error": "missing grants"})
    posts = []
    record = cards.settle(drive, post=lambda path, body: posts.append(path) or 200)[0]
    assert record["grant"].startswith("failed: missing grants") and record["post"] == "pending" and posts == []
    assert record["code"] == CODE, "a skill that cannot run keeps its code for a later grant"
    monkeypatch.setattr("ouroboros.config.get_auto_grant_enabled", lambda: True)
    cards.record_consent(drive, _entry())
    refused = cards.settle(drive, post=lambda path, body: 400)[0]
    assert refused["post"] == "failed: the route answered 400" and "code" not in refused
    cards.record_consent(drive, _entry())
    waiting = cards.settle(drive, post=lambda path, body: 409)[0]
    assert waiting["post"] == "waiting" and waiting["code"] == CODE, "a route not serving yet is tried again at the next run"

    def boom(*_a, **_k):
        raise RuntimeError("lifecycle down")

    monkeypatch.setattr("ouroboros.config.get_auto_grant_enabled", lambda: False)
    monkeypatch.setattr("ouroboros.skill_lifecycle_actions.run_skill_action", boom)
    cards.record_consent(drive, _entry())
    assert cards.settle(drive, post=lambda path, body: 200) == []  # logged by type, never raised


def test_a_skill_finishing_its_load_in_the_server_settles_the_cards_waiting_for_it(tmp_path, monkeypatch):
    """The extension's load is one of the events a waiting card posts on (no timer): only a load in the server
    process, where the route lives; a failed load or a worker's reconcile settles nothing."""
    from ouroboros import extension_loader

    calls = []
    monkeypatch.setattr(cards, "settle_later", lambda drive_root, repo_dir=None: calls.append(drive_root))
    monkeypatch.setattr(extension_loader, "_announce_extension_state_change", lambda *a, **k: "")
    monkeypatch.setattr("ouroboros.extension_health.record_health_for_runtime_state", lambda *a, **k: None)
    for server, reason, expected in ((True, "loaded", [tmp_path]), (True, "load_error", []), (False, "loaded", [])):
        calls.clear()
        monkeypatch.setattr(extension_loader, "is_server_process", lambda server=server: server)
        extension_loader._finalize_extension_reconcile({}, tmp_path, "demo", reason=reason)
        assert calls == expected, (server, reason)
