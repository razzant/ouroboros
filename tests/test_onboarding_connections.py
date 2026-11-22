"""A built-in skill's declared connection card (ouroboros/onboarding_connections.py; DESIGN §7, CREATING_SKILLS).

A fixture skill stands for the Telegram skill: its card reaches the Skills step without its function, keys or
routes; the token check calls the declared function of a built-in skill only and never echoes the token or the
address; the completion saves the token and the code under the declared keys in its one write and refuses a card's
values from the Telegram Mini App; after the write the owner's consent is recorded, the skill granted its keys once
it is installed, and the options posted once to the declared route.
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

CARD = """connection_card:
  title: "Phone via Demo"
  pitch: "Chat anywhere"
  steps:
    - {text: "Open the maker", link: "https://example.test/maker", action: "Open the maker"}
    - {text: "Paste the token here"}
  token: botToken
  pairing: {code: pairingCode, link: "https://example.test/{username}?start={code}", identity: "IDENTITY", hint: "Scan it", note: "Finish now"}
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
env_from_settings: [DEMO_TOKEN, DEMO_CODE]
secret_fields:
  botToken: DEMO_TOKEN
  pairingCode: DEMO_CODE
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


def test_the_step_sees_the_card_s_words_but_never_its_function_keys_or_routes(skills):
    seen = cards.starter_connection("demo")
    assert seen == {
        "skill": "demo", "token": "botToken", "title": "Phone via Demo", "pitch": "Chat anywhere",
        "steps": [{"text": "Open the maker", "link": "https://example.test/maker", "action": "Open the maker"},
                  {"text": "Paste the token here", "link": "", "action": ""}],
        "pairing": {"code": "pairingCode", "link": "https://example.test/{username}?start={code}", "hint": "Scan it",
                    "note": "Finish now"},
        "options": [{"key": "DEMO_NOTIFY", "label": "Send results here", "on": "on", "off": "off", "new_connection": True}],
    }
    assert cards.card_keys("demo") == ("DEMO_TOKEN", "DEMO_CODE")
    # A card missing any part the host needs is no card: the skill stays an ordinary row.
    for broken in (CARD.replace("?start={code}", ""), CARD.replace("store: skill_settings", "store: core"),
                   CARD.replace("token: botToken", "token: apiKey"), CARD.replace("https://example.test/{username}", "http://example.test/{username}")):
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
    assert items["builtin/demo"]["needs_keys"] == ["DEMO_TOKEN", "DEMO_REGION"], "the code the card writes is no prerequisite"
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


def test_finish_saves_the_token_and_code_under_the_declared_keys_and_refuses_anything_else(skills):
    body = {"connections": {"demo": {"fields": {"botToken": " 1:abc ", "pairingCode": "Abc_-1"},
                                     "options": {"DEMO_NOTIFY": "off"}}}}
    assert cards.completion_settings(body) == {"DEMO_TOKEN": "1:abc", "DEMO_CODE": "Abc_-1"}, "options stay out"
    assert skill_starter.onboarding_skill_choices(body, {}) == {"DEMO_TOKEN": "1:abc", "DEMO_CODE": "Abc_-1"}
    assert cards.completion_settings({}) == {} and cards.completion_settings({"connections": {}}) == {}
    for bad in ({"nope": {"fields": {"botToken": "1:a", "pairingCode": "c"}}},
                {"demo": {"fields": {"botToken": "1:a"}}},
                {"demo": {"fields": {"botToken": "1:a", "pairingCode": "c", "extra": "x"}}},
                {"demo": {"fields": {"botToken": "1:a", "pairingCode": "c d"}}},
                {"demo": {"fields": {"botToken": "1 a", "pairingCode": "c"}}},
                {"demo": {"fields": {"botToken": "1:a", "pairingCode": "c"}, "options": {"DEMO_NOTIFY": "yes"}}},
                {"demo": {"fields": {"botToken": "1:a", "pairingCode": "c"}, "options": {"OTHER": "on"}}},
                ["demo"]):
        with pytest.raises(ValueError):
            cards.completion_settings({"connections": bad})


def test_the_telegram_mini_app_cannot_arm_a_code_and_a_saved_completion_records_the_consent(skills, tmp_path, monkeypatch):
    reached, settled = [], []
    answer = {"status": 200, "body": {"ok": True, "status": "saved"}}

    async def completion(request):
        reached.append(await request.json())
        return JSONResponse(answer["body"], status_code=answer["status"])

    monkeypatch.setattr(cards, "settle_later", lambda drive_root, repo_dir=None: settled.append(drive_root))
    drive = tmp_path / "data"
    client = _client(Route("/complete", endpoint=cards.guard_completion(completion), methods=["POST"]), drive_root=drive)
    connections = {"connections": {"demo": {"fields": {"botToken": "1:a", "pairingCode": "c"}}}}
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
    assert [record["keys"] for record in skill_starter.read_marker(drive)["connections"]] == [["DEMO_TOKEN", "DEMO_CODE"]]
    assert settled == [drive] and len(reached) == 3 and reached[-1] == connections


def test_after_finish_the_consent_is_kept_then_the_keys_granted_and_the_options_posted_once(skills, tmp_path, monkeypatch):
    drive = tmp_path / "data"
    raw = {"demo": {"fields": {"botToken": "1:abc", "pairingCode": "Abc"}, "options": {"DEMO_NOTIFY": "on"}}}
    assert [record["keys"] for record in cards.record_consent(drive, raw)] == [["DEMO_TOKEN", "DEMO_CODE"]]
    installed, grants, posts = [False], [], []
    monkeypatch.setattr(cards, "_installed", lambda _drive, _name: installed[0])

    def grant(ctx, name, action, **kwargs):
        grants.append((name, action, kwargs["items"], kwargs["_owner_actor"], kwargs["_enable_after_grant"]))
        return {"ok": True}

    monkeypatch.setattr("ouroboros.skill_lifecycle_actions.run_skill_action", grant)
    answers = iter([409, 200])

    def post(path, body):
        posts.append((path, body))
        return next(answers)

    record = cards.settle(drive, post=post, pause=lambda _s: None)[0]
    assert (record["grant"], record["options_state"], grants, posts) == ("pending", "pending", [], []), "not installed yet"
    installed[0] = True
    record = cards.settle(drive, post=post, pause=lambda _s: None)[0]
    assert grants == [("demo", "grant", ["DEMO_TOKEN", "DEMO_CODE"], "owner_ui", True)]
    assert posts == [("/api/extensions/demo/settings/save", {"DEMO_NOTIFY": "on"})] * 2, "it waits for the route, once"
    assert (record["grant"], record["options_state"]) == ("granted", "posted")
    assert skill_starter.read_marker(drive)["connections"][0]["options_state"] == "posted"
    cards.settle(drive, post=post, pause=lambda _s: None)
    assert len(grants) == 1 and len(posts) == 2, "settled cards are not settled again"


def test_a_failed_grant_posts_nothing_and_no_failure_escapes(skills, tmp_path, monkeypatch):
    drive = tmp_path / "data"
    cards.record_consent(drive, {"demo": {"fields": {"botToken": "1:abc", "pairingCode": "Abc"}, "options": {"DEMO_NOTIFY": "on"}}})
    monkeypatch.setattr(cards, "_installed", lambda _drive, _name: True)
    monkeypatch.setattr("ouroboros.skill_lifecycle_actions.run_skill_action", lambda *a, **k: {"error": "missing grants"})
    posts = []
    record = cards.settle(drive, post=lambda path, body: posts.append(path) or 200, pause=lambda _s: None)[0]
    assert record["grant"].startswith("failed: missing grants") and record["options_state"] == "pending" and posts == []

    def boom(*_a, **_k):
        raise RuntimeError("lifecycle down")

    cards.record_consent(drive, {"demo": {"fields": {"botToken": "1:abc", "pairingCode": "Abc"}}})
    monkeypatch.setattr("ouroboros.skill_lifecycle_actions.run_skill_action", boom)
    assert cards.settle(drive, post=lambda path, body: 200, pause=lambda _s: None) == []  # recorded in the log, never raised
