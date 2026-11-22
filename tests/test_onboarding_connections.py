"""A built-in skill's declared connection card (ouroboros/onboarding_connections.py; DESIGN §7, CREATING_SKILLS).

A fixture skill stands for the Telegram skill: its card reaches the Skills step without its function, keys or
routes; the token check calls the declared function of a built-in skill only and never echoes the token or the
address, and asks the server a declared reader names; the completion saves the token alone under its declared key in
its one write, keeps the code out of Settings, and refuses a card's values from the Telegram Mini App; after the write
the owner's consent is recorded and, once the skill serves, the code with the options is posted once to the route they
declare, after which the code leaves the record. Where the install grants reviewed skills on its own, the consent turns
on a skill its install never granted (a payload a Reset kept) through the lifecycle's grant; where it grants nothing
on its own, neither does the card, and the step is told the skill starts once the owner allows it.
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
CALLS, BASES, FAIL = [], [], []


def base(state_dir):
    BASES.append(str(state_dir))
    if FAIL:
        raise RuntimeError("the skill's settings cannot be read")
    return "http://127.0.0.1:8081"


def check(token, api_base=None, *, timeout=8.0):
    CALLS.append((api_base, timeout))
    if token == "1:boom":
        raise RuntimeError("secret 1:boom at https://api.example.test/bot1:boom/getMe")
    if token.startswith("1:"):
        return {"ok": True, "username": "demo_bot", "id": 1}
    if token.startswith("3:"):
        return {"ok": False, "cause": "unreachable"}  # the skill's own answer: it asked and could not reach the service
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


def test_the_step_sees_the_card_s_words_but_never_its_function_keys_or_routes(skills, monkeypatch):
    monkeypatch.setattr("ouroboros.config.get_auto_grant_enabled", lambda: True)
    seen = cards.starter_connection("demo")
    assert seen == {
        "skill": "demo", "token": "botToken", "title": "Phone via Demo", "pitch": "Chat anywhere",
        "steps": [{"text": "Open the maker", "link": "https://example.test/maker", "action": "Open the maker"},
                  {"text": "Paste the token here", "link": "", "action": ""}],
        "pairing": {"link": "https://example.test/{username}?start={code}", "hint": "Scan it", "note": "Finish now"},
        "options": [{"key": "DEMO_NOTIFY", "label": "Send results here", "on": "on", "off": "off", "new_connection": True}],
        "needs_allow": False,
    }
    # Where the install grants reviewed skills nothing on its own, the step learns the skill starts once allowed.
    monkeypatch.setattr("ouroboros.config.get_auto_grant_enabled", lambda: False)
    assert cards.starter_connection("demo") == {**seen, "needs_allow": True}
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
    for reader in ("../outside.py:base", "base", ""):  # a declared reader out of shape is no card, never the cloud
        _skill(skills, CARD.replace('identity: "IDENTITY",', f'identity: "IDENTITY", api_base: "{reader}",'))
        assert cards.declared_card("demo") is None, reader
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
    assert skill_starter.starter_groups(tmp_path, rows=[])["catalog_more"] is None, "no source read: no count"


def _client(*routes: Route, drive_root: pathlib.Path | None = None) -> TestClient:
    app = Starlette(routes=list(routes))
    if drive_root is not None:
        app.state.drive_root = drive_root
    return TestClient(app)


def test_the_token_check_runs_the_declared_function_of_a_built_in_skill_and_names_nothing_else(skills):
    client = _client(Route("/check", endpoint=cards.api_connection_identity, methods=["POST"]))
    assert client.post("/check", json={"skill": "demo", "token": " 1:abc "}).json() == {"ok": True, "username": "demo_bot"}
    assert client.post("/check", json={"skill": "demo", "token": "2:abc"}).json() == {"ok": False, "cause": "refused"}
    # No token at all is this host's refusal, never the service's «refused»: the function is not called.
    module = cards._declared_function(cards.declared_card("demo")).__globals__
    calls = len(module["CALLS"])
    for draft in ("", "   ", "1:a b", "1:a\nb", "1:" + "a" * 511):
        refused = client.post("/check", json={"skill": "demo", "token": draft})
        assert refused.status_code == 400 and "cause" not in refused.json() and "1:a" not in refused.text, draft
    assert len(module["CALLS"]) == calls
    assert client.post("/check", json={"skill": "demo", "token": "1:" + "a" * 510}).json()["ok"] is True, "512 is a token"
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


def test_a_host_deadline_is_could_not_be_checked_and_only_the_skills_own_unreachable_is_unreachable(skills, monkeypatch):
    """The host's watchdog covers its own worker's wait too: expiring before the declared function was reached (a busy
    worker pool) or while the function waits, the check did not finish (503; the page says it could not be checked):
    the host has no evidence a request reached the service, so never «unreachable». A function's own «unreachable»
    answer passes through (rv2 C7)."""
    import asyncio
    import time

    module = cards._declared_function(cards.declared_card("demo")).__globals__
    module["CALLS"].clear()
    client = _client(Route("/check", endpoint=cards.api_connection_identity, methods=["POST"]))
    monkeypatch.setattr(cards, "_IDENTITY_WAIT", 0.3)

    async def pool_still_full(*args, **kwargs):
        await asyncio.Event().wait()

    with monkeypatch.context() as patch:
        patch.setattr(asyncio, "to_thread", pool_still_full)
        held = client.post("/check", json={"skill": "demo", "token": "1:abc"})
    assert held.status_code == 503 and "cause" not in held.json() and "1:abc" not in held.text
    assert module["CALLS"] == []
    real = module["check"]

    def silent(token, api_base=None, *, timeout=8.0):
        time.sleep(1.0)  # past the watchdog: the service never answers in time
        return real(token, api_base, timeout=timeout)

    monkeypatch.setitem(module, "check", silent)
    answer = client.post("/check", json={"skill": "demo", "token": "1:abc"})
    assert answer.status_code == 503 and "cause" not in answer.json() and "1:abc" not in answer.text
    monkeypatch.setitem(module, "check", real)
    own = client.post("/check", json={"skill": "demo", "token": "3:down"})
    assert own.status_code == 200 and own.json() == {"ok": False, "cause": "unreachable"}


def test_a_declared_reader_names_the_server_the_check_asks_and_a_failing_one_asks_nobody(skills, tmp_path):
    """``pairing.api_base`` is read like ``identity``: given the skill's state directory (read, never made), its answer
    is the check's second argument. Without it the check is called as before."""
    from ouroboros.skill_loader import skill_state_dir_path

    drive = tmp_path / "data"
    client = _client(Route("/check", endpoint=cards.api_connection_identity, methods=["POST"]), drive_root=drive)
    assert client.post("/check", json={"skill": "demo", "token": "1:abc"}).json()["ok"] is True
    module = cards._declared_function(cards.declared_card("demo")).__globals__
    assert (module["CALLS"], module["BASES"]) == ([(None, 8.0)], []), "no reader declared: the function's own default"
    _skill(skills, CARD.replace('identity: "IDENTITY",', 'identity: "IDENTITY", api_base: "lib/ident.py:base",'))
    module = cards._declared_function(cards.declared_card("demo")).__globals__
    module["CALLS"].clear()
    assert client.post("/check", json={"skill": "demo", "token": "1:abc"}).json() == {"ok": True, "username": "demo_bot"}
    state_dir = skill_state_dir_path(drive, "demo")
    assert module["BASES"] == [str(state_dir)] and module["CALLS"] == [("http://127.0.0.1:8081", 8.0)]
    assert not state_dir.exists()
    assert "ident.py" not in str(cards.starter_connection("demo")), "the step never sees a declared function"
    module["FAIL"].append(True)
    failed = client.post("/check", json={"skill": "demo", "token": "1:abc"})
    assert failed.status_code == 500 and "1:abc" not in failed.text
    assert module["CALLS"] == [("http://127.0.0.1:8081", 8.0)], "a reader that fails sends the token nowhere"


def test_finish_saves_the_token_alone_and_refuses_anything_else(skills):
    body = {"connections": _entry(fields={"botToken": " 1:abc "}, options={"DEMO_NOTIFY": "off"})}
    assert cards.completion_settings(body) == {"DEMO_TOKEN": "1:abc"}, "the code and the options stay out of Settings"
    assert skill_starter.onboarding_skill_choices(body, {}) == {"DEMO_TOKEN": "1:abc"}
    assert cards.completion_settings({}) == {} and cards.completion_settings({"connections": {}}) == {}
    # A card whose QR was never drawn sends its token alone: saved, with no code and no options.
    alone = {"demo": {"fields": {"botToken": "1:abc"}}}
    assert cards.completion_settings({"connections": alone}) == {"DEMO_TOKEN": "1:abc"}
    assert cards.parse_connections(alone)[0][1:] == ("1:abc", "", {})
    assert cards.parse_connections({"demo": {"fields": {"botToken": "1:abc"}, "options": {}}})[0][2] == ""
    for bad in ({"nope": _entry()["demo"]},
                _entry(fields={}),
                _entry(fields={"botToken": "1:a", "pairingCode": CODE}),
                _entry(fields={"botToken": "1 a"}),
                _entry(code="short_code"),  # not the shape a skill mints
                _entry(code=CODE + "!"),
                {"demo": {"fields": {"botToken": "1:a"}, "options": {"DEMO_NOTIFY": "on"}}},  # options without their code
                _entry(code=None),
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


def test_after_finish_the_code_is_posted_once_the_skill_serves(skills, tmp_path, monkeypatch):
    drive = tmp_path / "data"
    assert len(cards.record_consent(drive, _entry())) == 1
    installed, posts = [False], []
    monkeypatch.setattr(cards, "_installed", lambda _drive, _name: installed[0])

    def grant(*_a, **_k):
        raise AssertionError("no payload on disk here: nothing for the consent to turn on")

    monkeypatch.setattr("ouroboros.skill_lifecycle_actions.run_skill_action", grant)
    answers = iter([(409, {}), (200, {"ok": True, "owner_ignored": False}), (200, {"ok": True})])

    def post(path, body):
        posts.append((path, dict(body)))
        return next(answers)

    record = cards.settle(drive, post=post)[0]
    assert (record["post"], posts, "grant" in record) == ("pending", [], False), "not installed yet"
    installed[0] = True  # installed, not serving yet: still loading, or a skill the owner has still to allow
    record = cards.settle(drive, post=post)[0]
    assert (record["post"], record["code"]) == ("waiting", CODE), "no timer, no retry: the next event posts"
    record = cards.settle(drive, post=post)[0]  # the extension's load
    assert posts == [("/api/extensions/demo/settings/save", {"DEMO_CODE": CODE, "DEMO_NOTIFY": "on"})] * 2, \
        "one body for the route, posted once it serves"
    assert (record["post"], "code" in record) == ("posted", False)
    assert "code" not in skill_starter.read_marker(drive)["connections"][0], "the code leaves the host's record"
    cards.settle(drive, post=post)
    assert len(posts) == 2, "settled cards are not settled again"


def test_where_reviewed_skills_are_granted_on_their_own_the_consent_turns_on_a_skill_its_install_never_granted(
        skills, tmp_path, monkeypatch):
    """A payload already on disk (a Reset keeps them) is installed but was never granted. Where the install grants
    reviewed skills on its own, the consent given at Finish grants what the skill asks through the lifecycle and turns
    it on, and the grant exists; where it does not, the card grants nothing; a skill granted and on is left alone."""
    import shutil

    from ouroboros import config, skill_lifecycle_actions
    from ouroboros.skill_loader import SkillReviewState, find_skill, load_enabled, load_skill_grants, save_review_state

    drive = tmp_path / "data"
    shutil.copytree(skills / "demo", drive / "skills" / "external" / "demo")  # first, a namesake a person added
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(tmp_path / "empty"))
    read = config.load_settings  # Finish saved the token: Settings now hold it, which is what makes it grantable
    monkeypatch.setattr(config, "load_settings", lambda *a, **k: {**read(*a, **k), "DEMO_TOKEN": "1:abc"})
    automatic = [False]
    monkeypatch.setattr(config, "get_auto_grant_enabled", lambda: automatic[0])
    save_review_state(drive, "demo", SkillReviewState(status="clean", content_hash=find_skill(drive, "demo").content_hash))
    calls, real = [], skill_lifecycle_actions.run_skill_action
    monkeypatch.setattr(skill_lifecycle_actions, "run_skill_action", lambda *a, **k: calls.append(k) or real(*a, **k))

    def post(path, body):  # the route does not serve here: the post waits for the next event
        return 409, {}

    cards.record_consent(drive, _entry())
    assert cards.settle(drive, post=post)[0]["post"] == "pending", "a namesake outside the built-in bucket gets nothing"
    (drive / "skills" / "native").mkdir()
    shutil.move(str(drive / "skills" / "external" / "demo"), str(drive / "skills" / "native" / "demo"))
    record = cards.settle(drive, post=post)[0]
    assert (calls, "grant" in record, record["post"]) == ([], False, "waiting"), "the owner allows it in Extensions"
    assert load_skill_grants(drive, "demo")["granted_keys"] == [] and load_enabled(drive, "demo") is False
    automatic[0] = True
    record = cards.settle(drive, post=post)[0]
    assert record["grant"] == "granted", record
    assert load_skill_grants(drive, "demo")["granted_keys"] == ["DEMO_TOKEN"] and load_enabled(drive, "demo") is True
    assert (calls[0]["items"], calls[0]["_enable_after_grant"], calls[0]["_owner_actor"]) == (["DEMO_TOKEN"], True, "owner_ui")
    cards.settle(drive, post=post)
    assert len(calls) == 1, "granted and on: the next event grants nothing again"


def test_a_card_that_sent_its_token_alone_records_no_code_and_posts_nothing(skills, tmp_path, monkeypatch):
    drive = tmp_path / "data"
    monkeypatch.setattr(cards, "_installed", lambda _drive, _name: True)
    record = cards.record_consent(drive, {"demo": {"fields": {"botToken": "1:abc"}}})[0]
    assert "code" not in record and record["options"] == {} and record["post"] == "pending"
    posts = []
    settled = cards.settle(drive, post=lambda path, body: (posts.append(path), (200, {}))[1])[0]
    assert (settled["post"], posts) == ("none", []), "no code was armed: the skill's own first-writer rule applies"
    cards.record_consent(drive, _entry())  # with its drawn QR the code goes out as before
    assert cards.settle(drive, post=lambda path, body: (posts.append(path), (200, {}))[1])[0]["post"] == "posted"
    assert posts == ["/api/extensions/demo/settings/save"]


def test_failures_are_recorded_never_raised_and_a_refused_post_drops_the_code(skills, tmp_path, monkeypatch):
    drive = tmp_path / "data"
    monkeypatch.setattr(cards, "_installed", lambda _drive, _name: True)
    cards.record_consent(drive, _entry())
    refused = cards.settle(drive, post=lambda path, body: (400, {}))[0]
    assert refused["post"] == "failed: the route answered 400" and "code" not in refused
    # A 200 that says the route left the owner's values unchanged saved nothing of them: never a success.
    cards.record_consent(drive, _entry())
    ignored = cards.settle(drive, post=lambda path, body: (200, {"ok": True, "owner_ignored": True}))[0]
    assert ignored["post"].startswith("failed:") and "owner_ignored" in ignored["post"] and "code" not in ignored
    cards.record_consent(drive, _entry())
    waiting = cards.settle(drive, post=lambda path, body: (409, {}))[0]
    assert waiting["post"] == "waiting" and waiting["code"] == CODE, "a route not serving yet is tried again at the next run"

    def boom(*_a, **_k):
        raise RuntimeError("route down")

    assert cards.settle(drive, post=boom) == []  # logged by type, never raised


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
        extension_loader._finalize_extension_reconcile({"skill": "demo"}, tmp_path, None, reason=reason)
        assert calls == expected, (server, reason)
