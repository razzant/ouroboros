"""Skills' service sign-ins against an in-memory OpenID provider (``ouroboros/service_sign_in.py``).

The provider publishes no RFC 8414 document (only OpenID discovery), checks PKCE S256, rotates refresh
tokens (a reused one is refused), answers ``/userinfo`` for the account an access token belongs to and
counts revocations. Two skills declare the same issuer; the owner's routes (``gateway/sign_in.py``) and the
Host Service routes (``gateway/host_sign_in.py``) run as the card and a skill process would.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import pathlib
import secrets
import threading
import urllib.parse

import httpx
import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from ouroboros import config, oauth_engine, service_sign_in
from ouroboros.gateway.host_service import AUTH_TOKEN_FILENAME, create_host_service_app
from ouroboros.gateway.sign_in import sign_in_routes
from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_enabled, save_review_state
from ouroboros.utils import atomic_write_json

ISSUER = "https://issuer.example"
CAL, MAIL = "calendar", "mail"


class Provider:
    """OpenID provider in memory: one signed-in person at a time, tokens bound to (client, subject, scope)."""

    def __init__(self) -> None:
        self.person = {"sub": "sub-1", "email": "owner@example.com"}
        self.codes, self.access, self.refresh, self.lock = {}, {}, {}, threading.Lock()
        self.counts = {"refresh": 0, "revoke": 0, "token_calls": 0}
        self.down, self.dead = False, set()

    def document(self) -> dict:
        return {"issuer": ISSUER, "authorization_endpoint": ISSUER + "/authorize", "token_endpoint": ISSUER + "/token",
                "revocation_endpoint": ISSUER + "/revoke", "userinfo_endpoint": ISSUER + "/userinfo",
                "response_types_supported": ["code"], "code_challenge_methods_supported": ["S256"],
                "token_endpoint_auth_methods_supported": ["client_secret_post"]}

    def approve(self, url: str) -> str:
        """The owner's browser at the provider: the address it is sent back to."""
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        code = secrets.token_urlsafe(8)
        self.codes[code] = {**query, "person": dict(self.person)}
        return f"{query['redirect_uri']}?" + urllib.parse.urlencode({"code": code, "state": query["state"], "iss": ISSUER})

    def _issue(self, client_id: str, person: dict, scope: str) -> dict:
        access, refresh = secrets.token_urlsafe(12), secrets.token_urlsafe(12)
        self.access[access] = self.refresh[refresh] = {"client": client_id, "person": person, "scope": scope}
        return {"access_token": access, "refresh_token": refresh, "token_type": "Bearer", "expires_in": 3600,
                "scope": scope}

    def handle(self, request: httpx.Request) -> httpx.Response:
        with self.lock:
            return self._handle(request)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if self.down:
            return httpx.Response(503)
        if path == "/.well-known/openid-configuration":
            return httpx.Response(200, json=self.document())
        if path.startswith("/.well-known/"):
            return httpx.Response(404)
        form = dict(urllib.parse.parse_qsl(request.content.decode())) if request.content else {}
        if path == "/token":
            self.counts["token_calls"] += 1
            if form["grant_type"] == "authorization_code":
                grant = self.codes.pop(form["code"], None)
                challenge = base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"].encode()).digest()).rstrip(b"=")
                if grant is None or challenge.decode() != grant["code_challenge"] or form["client_id"] != grant["client_id"]:
                    return httpx.Response(400, json={"error": "invalid_grant"})
                return httpx.Response(200, json=self._issue(form["client_id"], grant["person"], grant.get("scope", "")))
            held = self.refresh.pop(form.get("refresh_token"), None)
            if held is None or form.get("refresh_token") in self.dead:
                return httpx.Response(400, json={"error": "invalid_grant"})
            self.counts["refresh"] += 1
            issued = self._issue(held["client"], held["person"], held["scope"])
            return httpx.Response(200, json={k: v for k, v in issued.items() if k != "refresh_token"} | {
                "refresh_token": issued["refresh_token"]})
        if path == "/userinfo":  # an account only for a token that carries the identity scope
            held = self.access.get(request.headers.get("authorization", "").removeprefix("Bearer "))
            return httpx.Response(200, json=held["person"]) if held and "openid" in held["scope"].split() \
                else httpx.Response(403 if held else 401)
        if path == "/revoke":
            self.counts["revoke"] += 1
            self.access.pop(form.get("token"), None)
            self.refresh.pop(form.get("token"), None)
            return httpx.Response(200)
        raise AssertionError(str(request.url))


def _declaration(scopes: list, optional=(), notes=None) -> str:
    entry = {"id": "google", "label": "Google", "issuer": ISSUER, "scopes": scopes,
             "authorize_params": {"access_type": "offline"}, "accounts": "many", "optional": True,
             "optional_scopes": [{"scope": s, "label": s.title(), "default": False} for s in optional],
             "notes": notes or {"refresh_rejected": "Testing apps end sign-ins after 7 days."}}
    return "sign_in: " + json.dumps([entry])


def install(data: pathlib.Path, skill: str, sign_in: str, token: str) -> None:
    """A reviewed, enabled extension skill with its Host Service token (as installation leaves it)."""
    folder = data / "skills" / "external" / skill
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(f"---\nname: {skill}\ndescription: test\nversion: 0.1\ntype: extension\n"
                                     f"entry: plugin.py\npermissions: []\n{sign_in}\n---\n# {skill}\n", encoding="utf-8")
    (folder / "plugin.py").write_text("def register(api): pass\n", encoding="utf-8")
    content_hash = compute_content_hash(folder, manifest_entry="plugin.py")
    save_review_state(data, skill, SkillReviewState(status="pass", content_hash=content_hash))
    save_enabled(data, skill, True)
    atomic_write_json(data / "state" / "skills" / skill / AUTH_TOKEN_FILENAME,
                      {"token": token, "content_hash": content_hash, "issued_at": "now"})


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(oauth_engine, "_server_port", lambda: 8765)
    provider = Provider()
    monkeypatch.setattr(oauth_engine, "http_client",
                        lambda: httpx.AsyncClient(transport=httpx.MockTransport(provider.handle)))
    install(tmp_path, CAL, _declaration(["openid", "email", "calendar.events"], optional=["calendar.acls"]), "cal-token")
    install(tmp_path, MAIL, _declaration(["openid", "email", "gmail.modify"], optional=["gmail.settings"]), "mail-token")
    owner = TestClient(Starlette(routes=sign_in_routes()), base_url="http://127.0.0.1:8765")
    host = TestClient(create_host_service_app(tmp_path))
    return provider, owner, host


def _shown(owner, skill: str, client: str = "") -> dict:
    """What the card shows of the saved client a sign-in starts with (the named one, else the only one): its registration."""
    row = next(s for s in owner.get("/api/sign-in/status").json()["services"] if s["skill"] == skill)
    chosen = [c for c in row["clients"] if client in ("", c["id"])]
    return {"registration": chosen[0]["registration"]} if len(chosen) == 1 else {}


def _sign_in(provider, owner, skill: str, **begin) -> dict:
    words = ["openid", "email", "calendar.events" if skill == CAL else "gmail.modify"]
    started = owner.post("/api/sign-in", json={"action": "begin", "skill": skill, "service": "google", "scope_words": words,
                                               **_shown(owner, skill, begin.get("client", "")), **begin}).json()
    assert started["ok"] is True, started
    returned = provider.approve(started["url"])
    return owner.post("/api/sign-in", json={"action": "complete", "returned_url": returned}).json()


def _save_client(owner, label="Personal", client_id="personal-client", secret="personal-secret-0001") -> str:
    saved = owner.post("/api/sign-in", json={"action": "client_save", "issuer": ISSUER, "client": {
        "label": label, "client_id": client_id, "client_secret": secret, "return": "loopback"}}).json()
    assert saved["ok"] is True and "secret" not in json.dumps(saved).replace("secret_set", ""), saved
    return saved["client"]["id"]


def _token(host, skill: str, **body) -> httpx.Response:
    return host.post("/sign-in/token", headers={"x-skill-token": {CAL: "cal-token", MAIL: "mail-token"}[skill]},
                     json={"service": "google", **body})


def test_a_skill_signs_in_with_only_its_own_scopes_and_reads_only_its_own_grant(world):
    provider, owner, host = world
    assert owner.post("/api/sign-in", json={"action": "begin", "skill": CAL, "service": "google"}).json()["reason"] \
        == "no_client"  # no client and no registration: the owner adds one first
    _save_client(owner)
    started = owner.post("/api/sign-in", json={"action": "begin", "skill": CAL, "service": "google",
                                               "optional_scopes": ["calendar.acls", "gmail.modify"], **_shown(owner, CAL),
                                               "scope_words": ["openid", "email", "calendar.events", "calendar.acls"]}).json()
    asked = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(started["url"]).query))
    assert asked["scope"] == "openid email calendar.events calendar.acls"  # never another skill's scope
    assert asked["access_type"] == "offline" and asked["redirect_uri"].endswith(oauth_engine.RETURN_PATH)
    done = owner.post("/api/sign-in", json={"action": "complete", "returned_url": provider.approve(started["url"])}).json()
    assert done["ok"] is True and done["account"]["label"] == "owner@example.com" and done["missing_required"] == []
    answer = _token(host, CAL)
    assert answer.status_code == 200 and answer.json()["scope"] == asked["scope"]
    assert answer.json()["account"]["key"] == done["account"]["key"]
    assert _token(host, MAIL).status_code == 404  # the same issuer and account, but no grant of this skill
    assert host.post("/sign-in/token", json={"service": "google"}).status_code == 403
    listing = host.get("/sign-in/accounts", headers={"x-skill-token": "cal-token"}).json()
    assert listing["services"][0]["accounts"][0]["label"] == "owner@example.com"
    leaked = json.dumps([owner.get("/api/sign-in/status").json(), listing])
    grant = json.loads(next((config.DATA_DIR / "state" / "sign_in" / "grants" / CAL).glob("*/a-*.json")).read_text(
        encoding="utf-8"))
    assert grant["tokens"]["refresh_token"] not in leaked and "personal-secret-0001" not in leaked


def test_two_skills_on_one_client_and_account_hold_two_independent_grants(world):
    provider, owner, host = world
    _save_client(owner)
    calendar, mail = _sign_in(provider, owner, CAL), _sign_in(provider, owner, MAIL)
    assert calendar["ok"] and mail["ok"]
    key = calendar["account"]["key"]
    assert key == mail["account"]["key"]  # one account, two grants
    accounts = {s["skill"]: s["accounts"] for s in owner.get("/api/sign-in/status").json()["services"]}
    for skill, sibling, scope in ((CAL, MAIL, "calendar.events"), (MAIL, CAL, "gmail.modify")):
        answer = _token(host, skill, account=key)
        assert answer.status_code == 200
        expected = {"openid", "email", scope}
        assert set(answer.json()["scope"].split()) == expected
        [row] = accounts[skill]
        assert (row["key"], set(row["scope"]), row["siblings"]) == (key, expected, [{"skill": sibling, "service": "google"}])


def test_bearer_scope_is_a_subset_of_literal_consent():
    ceiling = {"read", "comment"}
    oauth_engine.check_bearer({"token_type": "Bearer", "scope": "read"}, ceiling)
    with pytest.raises(oauth_engine.SignInRequired) as exc:
        oauth_engine.check_bearer({"token_type": "Bearer", "scope": "read write"}, ceiling)
    assert exc.value.reason == "provider_scope_mismatch"


def test_concurrent_callers_refresh_once_and_a_rejected_token_refreshes_once(world):
    provider, owner, host = world
    _save_client(owner)
    key = _sign_in(provider, owner, CAL)["account"]["key"]
    grant = service_sign_in._grant(CAL, "google", key)
    doc = oauth_engine.load(grant)
    oauth_engine.save(grant, {**doc, "tokens": {**doc["tokens"], "expires_at": 1}})
    tokens = []
    callers = [threading.Thread(target=lambda: tokens.append(asyncio.run(service_sign_in.token(CAL, "google"))))
               for _ in range(2)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(30)
    assert len({answer["access_token"] for answer in tokens}) == 1 and provider.counts["refresh"] == 1
    current = tokens[0]["access_token"]
    assert _token(host, CAL, rejected="an-older-token").json()["access_token"] == current  # not the stored one
    assert _token(host, CAL, rejected=current).json()["access_token"] != current and provider.counts["refresh"] == 2


def test_a_refused_refresh_asks_for_sign_in_with_the_skills_note_and_an_unreachable_provider_keeps_the_grant(world):
    provider, owner, host = world
    _save_client(owner)
    key = _sign_in(provider, owner, CAL)["account"]["key"]
    grant = service_sign_in._grant(CAL, "google", key)
    oauth_engine.save(grant, {**oauth_engine.load(grant), "tokens": {**oauth_engine.load(grant)["tokens"], "expires_at": 1}})
    provider.down = True
    refused = _token(host, CAL)
    assert refused.status_code == 503 and refused.json()["code"] == "SIGN_IN_UNAVAILABLE"
    assert oauth_engine.load(grant)["tokens"]["refresh_token"] and oauth_engine.load(grant)["last_error"]
    provider.down = False
    provider.dead.add(oauth_engine.load(grant)["tokens"]["refresh_token"])
    refused = _token(host, CAL)
    body = refused.json()
    assert refused.status_code == 409 and (body["code"], body["reason"]) == ("SIGN_IN_REQUIRED", "refresh_rejected")
    assert body["note"] == "Testing apps end sign-ins after 7 days." and "owner@example.com" in body["hint"]
    row = owner.get("/api/sign-in/status").json()["services"][0]["accounts"][0]
    assert (row["state"], row["reason"]) == ("sign_in_required", "refresh_rejected")


def test_disconnect_asks_nobody_and_revoke_ends_only_this_grant_with_its_own_client(world, monkeypatch):
    provider, owner, host = world
    _save_client(owner)
    key = _sign_in(provider, owner, CAL)["account"]["key"]
    _sign_in(provider, owner, MAIL)
    grant_path = service_sign_in._grant(CAL, "google", key).path
    provider_calls = []
    handle = provider.handle

    def observe(request: httpx.Request) -> httpx.Response:
        provider_calls.append((request.url.path, grant_path.is_file()))
        return handle(request)

    monkeypatch.setattr(provider, "handle", observe)
    consent = next(a for s in owner.get("/api/sign-in/status").json()["services"] if s["skill"] == MAIL for a in s["accounts"])["consent"]
    out = owner.post("/api/sign-in", json={"action": "disconnect", "skill": MAIL, "service": "google", "account": key, "consent": consent})
    assert out.status_code == 200 and out.json() == {"ok": True}
    assert not provider_calls
    assert _token(host, MAIL).status_code == 404 and _token(host, CAL).status_code == 200
    _sign_in(provider, owner, MAIL)
    provider_calls.clear()
    consent = next(a for s in owner.get("/api/sign-in/status").json()["services"] if s["skill"] == CAL for a in s["accounts"])["consent"]
    revoked = owner.post("/api/sign-in", json={"action": "revoke", "skill": CAL, "service": "google", "account": key, "consent": consent})
    assert revoked.status_code == 200 and revoked.json() == {"ok": True, "revoked": True}
    assert provider_calls == [("/revoke", True), ("/revoke", True)]
    assert not grant_path.exists()
    assert _token(host, CAL).status_code == 404 and _token(host, MAIL).status_code == 200  # the sibling is untouched


def test_two_clients_sit_side_by_side_and_a_removed_client_leaves_its_grants_working(world):
    provider, owner, host = world
    personal, work = _save_client(owner), _save_client(owner, "Work, approved by IT", "work-client", "work-secret-01")
    assert owner.post("/api/sign-in", json={"action": "begin", "skill": CAL, "service": "google"}).json()["reason"] \
        == "client_required"  # chosen before the provider opens
    done = _sign_in(provider, owner, CAL, client=work)
    row = owner.get("/api/sign-in/status").json()["services"][0]
    assert {c["id"] for c in row["clients"]} == {personal, work} and row["accounts"][0]["client"]["label"].startswith("Work")
    assert owner.post("/api/sign-in", json={"action": "client_remove", "client": work}).json()["removed"] is True
    assert _token(host, CAL, account=done["account"]["key"]).status_code == 200


def test_a_needed_scope_inside_the_declaration_is_recorded_for_the_next_sign_in_and_one_outside_is_ignored(world):
    provider, owner, host = world
    _save_client(owner)
    key = _sign_in(provider, owner, CAL)["account"]["key"]
    assert _token(host, CAL, required_scope="gmail.modify").status_code == 200  # another skill's scope: ignored
    step_up = _token(host, CAL, required_scope="calendar.acls")
    assert step_up.status_code == 409 and step_up.json()["reason"] == "insufficient_scope"
    row = next(s for s in owner.get("/api/sign-in/status").json()["services"] if s["skill"] == CAL)["accounts"][0]
    assert set(row["required_scope"].split()) == {"openid", "email", "calendar.events", "calendar.acls"}
    assert row["missing_step_up"] == ["calendar.acls"]
    again = owner.post("/api/sign-in", json={"action": "begin", "skill": CAL, "service": "google", "account": key,
                                             "optional_scopes": ["calendar.acls"], **_shown(owner, CAL),
                                             "scope_words": ["openid", "email", "calendar.events", "calendar.acls"]}).json()
    assert "calendar.acls" in dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(again["url"]).query))["scope"].split()


def test_the_account_is_the_one_the_provider_names_and_another_person_gets_their_own_row(world):
    provider, owner, host = world
    _save_client(owner)
    first = _sign_in(provider, owner, CAL)
    provider.person = {"sub": "sub-2", "email": "work@example.com"}
    second = _sign_in(provider, owner, CAL, login_hint="owner@example.com")  # a hint is not an identity
    assert second["account"]["label"] == "work@example.com" and second["account"]["key"] != first["account"]["key"]
    assert len(owner.get("/api/sign-in/status").json()["services"][0]["accounts"]) == 2
    assert _token(host, CAL).status_code == 404  # two accounts: the skill names one
    again = owner.post("/api/sign-in", json={"action": "complete", "returned_url": provider.approve(
        owner.post("/api/sign-in", json={"action": "begin", "skill": CAL, "service": "google",
                                         "scope_words": ["openid", "email", "calendar.events"], **_shown(owner, CAL)}).json()["url"])}).json()
    assert again["account"]["key"] == second["account"]["key"]  # the same person replaces that account's grant
    replay = owner.post("/api/sign-in", json={"action": "complete", "state": "e30", "code": "x"})
    assert replay.status_code == 400  # not a sign-in return of a declared service


def test_a_connection_keeps_its_id_through_refresh_and_consent_and_a_sign_in_after_disconnect_gets_another(world):
    provider, owner, host = world
    personal, work = _save_client(owner), _save_client(owner, "Work, approved by IT", "work-client", "work-secret-01")
    key = _sign_in(provider, owner, CAL, client=personal)["account"]["key"]

    def connection() -> str:
        [row] = host.get("/sign-in/accounts", headers={"x-skill-token": "cal-token"}).json()["services"][0]["accounts"]
        return row["connection_id"]

    first, grant = connection(), service_sign_in._grant(CAL, "google", key)
    saved = oauth_engine.load(grant)
    oauth_engine.save(grant, {**saved, "tokens": {**saved["tokens"], "expires_at": 1}})
    assert _token(host, CAL).status_code == 200 and connection() == first  # a refresh keeps it
    assert _sign_in(provider, owner, CAL, client=work, account=key)["ok"] and connection() == first  # consent, another client
    assert _token(host, CAL, connection_id=first).json()["connection_id"] == first  # an expected connection still matches
    consent = next(a["consent"] for s in owner.get("/api/sign-in/status").json()["services"] if s["skill"] == CAL
                   for a in s["accounts"] if a["key"] == key)
    disconnected = owner.post("/api/sign-in", json={"action": "disconnect", "skill": CAL, "service": "google", "account": key, "consent": consent})
    assert disconnected.json() == {"ok": True}
    _sign_in(provider, owner, CAL, client=personal)
    assert connection() not in ("", first)
    changed = _token(host, CAL, connection_id=first)  # a write that expects the old one is refused before any refresh
    assert (changed.status_code, changed.json()["reason"]) == (409, "connection_changed")
