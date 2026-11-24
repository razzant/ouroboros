"""What the owner is shown is what a sign-in, a Disconnect and a Revoke act on (``ouroboros/service_sign_in.py``).

Ordinary use against the loopback suite's in-memory provider and its fictional accounts: the card names a saved client by its
registration and an account by its consent; a skill whose current bytes may not act starts no sign-in while leaving one stays
possible; a Revoke is final only when the provider confirmed it; a skill told there is no account to use is told which; an explicit
deletion forgets only its own skill's sign-ins and is refused, before any change, while a sign-in step holds its lock.
"""

from __future__ import annotations

import json
import pathlib

import httpx

from ouroboros import config, oauth_engine, service_sign_in, skill_loader, store
from ouroboros.skill_loader import SkillReviewState, save_enabled, save_review_state
from ouroboros.skill_uninstall_state import delete_local_skill
from tests.test_service_sign_in_loopback import CAL, ISSUER, MAIL, _save_client, _shown, _sign_in, _token, world  # noqa: F401 -- the fixture
from tests.test_skill_uninstall_tombstone import _MANIFEST, _seed_sign_in_records, _seed_state

WORDS = ["openid", "email", "calendar.events"]
CYBER = "ouroboros.tools.review_helpers.cyber_review_authority"


def _service(owner, skill: str = CAL) -> dict:
    return next(row for row in owner.get("/api/sign-in/status").json()["services"] if row["skill"] == skill)


def _post(owner, action: str, skill: str = CAL, **body) -> dict:
    return owner.post("/api/sign-in", json={"action": action, "skill": skill, "service": "google", **body}).json()


def _begin(owner, **body) -> dict:
    return _post(owner, "begin", scope_words=WORDS, **body)


def _withdraw(owner, action: str, key: str) -> dict:
    """Disconnect or Revoke as the card sends it: for the consent it showed."""
    consent = next(row for row in _service(owner)["accounts"] if row["key"] == key)["consent"]
    return _post(owner, action, account=key, consent=consent)


def _save(owner, client: dict) -> dict:
    return owner.post("/api/sign-in", json={"action": "client_save", "issuer": ISSUER, "client": client}).json()


def _isolated(monkeypatch) -> None:
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)
    monkeypatch.setenv("OUROBOROS_SKILL_SOURCES", "[]")


def test_a_begin_answers_for_the_registration_the_card_showed(world):  # noqa: F811
    provider, owner, _host = world
    client = _save_client(owner)
    shown = _service(owner)["clients"][0]["registration"]
    assert len(shown) == 64 and "personal-secret-0001" not in repr(_service(owner)["clients"])  # a hash of public facts
    assert _save(owner, {"id": client, "label": "Home", "return": "loopback"})["ok"] is True
    assert _service(owner)["clients"][0]["registration"] == shown  # a label is not the registration
    for echo in ("", "0" * 64):  # nothing shown, or something else
        refused = _begin(owner, client=client, registration=echo)
        assert (refused["ok"], refused["reason"]) == (False, "sign_in_choices_changed")
    assert not oauth_engine.load(service_sign_in._grant(CAL, "google", "pending"))  # nothing was reserved
    assert _save(owner, {"id": client, "label": "Home", "client_id": "personal-client-two", "return": "loopback"})["ok"] is True
    stale = _begin(owner, client=client, registration=shown)
    assert (stale["ok"], stale["reason"]) == (False, "sign_in_choices_changed")
    fresh = _service(owner)["clients"][0]["registration"]
    assert fresh != shown
    started = _begin(owner, client=client, registration=fresh)
    done = owner.post("/api/sign-in", json={"action": "complete", "returned_url": provider.approve(started["url"])}).json()
    assert started["ok"] is True and done["ok"] is True and done["account"]["label"] == "owner@example.com"


def _registering(provider, monkeypatch) -> list:
    """The loopback provider also registers clients (RFC 7591) and keeps each registration it receives."""
    registered, answers, document = [], provider.handle, provider.document

    def handle(request):
        if request.url.path == "/register":
            registered.append(json.loads(request.content))
            return httpx.Response(201, json={**registered[-1], "client_id": "host-registered"})
        return answers(request)

    monkeypatch.setattr(provider, "document", lambda: {**document(), "registration_endpoint": ISSUER + "/register"})
    monkeypatch.setattr(provider, "handle", handle)
    return registered


def test_a_client_the_host_registers_answers_for_the_provider_the_card_showed(world, monkeypatch):  # noqa: F811
    provider, owner, _host = world
    registered = _registering(provider, monkeypatch)
    shown = _service(owner)["issuer"]
    assert shown == ISSUER and _service(owner)["clients"] == []
    for echo in ("", "https://other.example"):  # nothing shown, or another provider's address
        refused = _begin(owner, register_client=True, issuer=echo)
        assert (refused["ok"], refused["reason"]) == (False, "sign_in_choices_changed")
    assert registered == [] and not oauth_engine.load(service_sign_in._grant(CAL, "google", "pending"))  # nothing asked, nothing reserved
    started = _begin(owner, register_client=True, issuer=shown)
    assert started["ok"] is True and [row["redirect_uris"] for row in registered] == [[started["redirect_uri"]]]
    [saved] = _service(owner)["clients"]
    assert (saved["id"], saved["source"]) == (started["client"]["id"], "registered")
    done = owner.post("/api/sign-in", json={"action": "complete", "returned_url": provider.approve(started["url"])}).json()
    assert done["ok"] is True and done["account"]["label"] == "owner@example.com"
    assert _begin(owner, register_client=True, issuer=shown)["reason"] == "client_required"  # a client exists now: it is chosen on the card
    plain = _begin(owner, client=saved["id"], registration=saved["registration"])  # a saved client still answers by its registration alone
    other = _begin(owner, client=saved["id"], registration=saved["registration"], issuer="https://other.example")
    assert plain["ok"] is True and (other["ok"], other["reason"]) == (False, "sign_in_choices_changed")


def test_a_registration_whose_answer_never_came_is_not_called_nothing_but_a_connection_that_never_opened_is(world, monkeypatch):  # noqa: F811
    provider, owner, _host = world
    _registering(provider, monkeypatch)
    answers = provider.handle
    for fault, reason in ((httpx.ReadTimeout, "registration_unconfirmed"), (httpx.ConnectError, "provider_unreachable")):
        def broken(request, fault=fault):
            if request.url.path == "/register":
                raise fault("no answer", request=request)
            return answers(request)

        monkeypatch.setattr(provider, "handle", broken)
        out = _begin(owner, register_client=True, issuer=ISSUER)
        assert (out["ok"], out["reason"]) == (False, reason)
        assert _service(owner)["clients"] == []  # no new client is available here


def test_a_token_answer_that_never_came_is_not_sent_again_and_its_attempt_stays_readable(world, monkeypatch):  # noqa: F811
    provider, owner, _host = world
    client = _save_client(owner)
    started = _begin(owner, client=client, **_shown(owner, CAL))
    returned = provider.approve(started["url"])
    answers, sent, silent = provider.handle, [], {"on": True}

    def transport(request):  # counts the token requests at the boundary to the provider
        if request.url.path == "/token":
            sent.append(request)
            if silent["on"]:
                raise httpx.ReadTimeout("no answer", request=request)
        return answers(request)

    monkeypatch.setattr(provider, "handle", transport)
    complete = {"action": "complete", "returned_url": returned, "attempt_id": started["attempt_id"]}
    lost = owner.post("/api/sign-in", json=complete).json()
    assert (lost["ok"], lost["reason"]) == (False, "provider_unreachable") and len(sent) == 1
    silent["on"] = False
    again = owner.post("/api/sign-in", json=complete).json()
    assert (again["ok"], again["reason"]) == (False, "exchange_not_reusable") and len(sent) == 1  # the same return is never sent a second time
    [row] = [s for s in owner.get("/api/sign-in/status").json()["services"] if s["skill"] == CAL]
    assert row["accounts"] == [] and row["pending"]["attempt_id"] == started["attempt_id"]  # the attempt's own state can be read
    assert _post(owner, "cancel", attempt_id=started["attempt_id"])["cancelled"] is True
    assert _sign_in(provider, owner, CAL)["ok"] is True  # the next step is a new attempt


def test_a_saved_token_answer_is_read_again_from_its_saved_copy_and_the_provider_issued_the_code_once(world, monkeypatch):  # noqa: F811
    provider, owner, _host = world
    client = _save_client(owner)
    started = _begin(owner, client=client, **_shown(owner, CAL))
    returned = provider.approve(started["url"])
    answers, down = provider.handle, {"identity": True}
    monkeypatch.setattr(provider, "handle", lambda request: httpx.Response(503) if down["identity"] and request.url.path == "/userinfo" else answers(request))
    complete = {"action": "complete", "returned_url": returned, "attempt_id": started["attempt_id"]}
    held = owner.post("/api/sign-in", json=complete).json()
    assert (held["ok"], held["reason"], held["attempt_id"]) == (False, "answer_saved", started["attempt_id"])
    down["identity"] = False
    done = owner.post("/api/sign-in", json=complete).json()
    assert done["ok"] is True and done["account"]["label"] == "owner@example.com"
    assert provider.counts["token_calls"] == 1  # the provider issued the one-time code once


def test_a_new_sign_in_needs_the_skills_current_bytes_admitted_and_leaving_one_never_does(world, monkeypatch):  # noqa: F811
    provider, owner, _host = world
    client = _save_client(owner)
    key = _sign_in(provider, owner, CAL)["account"]["key"]
    assert _service(owner)["admission"] == ""
    started = _begin(owner, client=client, **_shown(owner, CAL))  # an attempt waits at the provider
    monkeypatch.setattr(CYBER, lambda: False)
    save_review_state(config.DATA_DIR, CAL, SkillReviewState(status="pending"))  # its review no longer covers these bytes
    assert _service(owner)["admission"] == "not_reviewed"
    refused = _begin(owner, client=client, **_shown(owner, CAL))
    assert (refused["ok"], refused["reason"]) == (False, "skill_not_reviewed")
    late = owner.post("/api/sign-in", json={"action": "complete", "returned_url": provider.approve(started["url"])}).json()
    assert (late["ok"], late["reason"]) == (False, "skill_not_reviewed")  # the open attempt does not complete either
    cancelled = _post(owner, "cancel", attempt_id=started["attempt_id"])
    assert (cancelled["ok"], cancelled["cancelled"]) == (True, True)  # but it can be ended, in its own slot
    monkeypatch.setattr(CYBER, lambda: True)  # Cyber Pro's own judgment admits the same bytes
    assert _service(owner)["admission"] == "" and _begin(owner, client=client, **_shown(owner, CAL))["ok"] is True
    monkeypatch.setattr(CYBER, lambda: False)
    assert _withdraw(owner, "revoke", key) == {"ok": True, "revoked": True}  # nor does a Revoke
    save_enabled(config.DATA_DIR, MAIL, False)
    assert _service(owner, MAIL)["admission"] == "disabled"
    off = _post(owner, "begin", MAIL, scope_words=["openid", "email", "gmail.modify"])
    assert (off["ok"], off["reason"]) == (False, "skill_disabled")


def test_a_replayed_return_of_a_filed_sign_in_answers_the_same_after_the_skill_is_switched_off(world):  # noqa: F811
    provider, owner, _host = world
    client = _save_client(owner)
    started = _begin(owner, client=client, **_shown(owner, CAL))
    returned = provider.approve(started["url"])
    first = owner.post("/api/sign-in", json={"action": "complete", "returned_url": returned}).json()
    save_enabled(config.DATA_DIR, CAL, False)
    again = owner.post("/api/sign-in", json={"action": "complete", "returned_url": returned}).json()
    assert first["ok"] is True and again["ok"] is True and again["account"] == first["account"]


def test_a_revoke_is_final_only_when_the_provider_confirmed_it(world, monkeypatch):  # noqa: F811
    provider, owner, _host = world
    _save_client(owner)
    answers = provider.handle

    def silent(request):
        raise httpx.ConnectError("no route", request=request)

    def half(request):  # the refresh token's end is confirmed, the access token's is refused
        if request.url.path == "/revoke" and b"token_type_hint=access_token" in request.content:
            return httpx.Response(400, json={"error": "unsupported_token_type"})
        return answers(request)

    for handle in (silent, half):  # each unknown outcome on its own new sign-in, one Revoke each
        key = _sign_in(provider, owner, CAL)["account"]["key"]
        grant = service_sign_in._grant(CAL, "google", key)
        monkeypatch.setattr(provider, "handle", handle)
        kept = _withdraw(owner, "revoke", key)
        assert (kept["ok"], kept["reason"], kept["revocation_reason"]) == (False, "revocation_not_confirmed", "not_confirmed")
        assert "revoked" not in kept  # a lost or partial answer is not reported as a refusal
        assert grant.path.is_file() and oauth_engine.load(grant)["tokens"]["refresh_token"]  # kept, tokens included
        monkeypatch.setattr(provider, "handle", answers)
        assert _withdraw(owner, "disconnect", key) == {"ok": True} and not grant.path.exists()  # the one local next step


def test_a_revoke_the_provider_was_never_asked_keeps_the_sign_in_and_is_not_a_disconnect(world):  # noqa: F811
    provider, owner, _host = world
    _save_client(owner)
    key = _sign_in(provider, owner, CAL)["account"]["key"]
    grant = service_sign_in._grant(CAL, "google", key)
    doc = oauth_engine.load(grant)
    cases = {"no_endpoint": {**doc, "metadata": {k: v for k, v in doc["metadata"].items() if k != "revocation_endpoint"}},
             "no_tokens": {**doc, "tokens": {}}}
    for reason, changed in cases.items():
        oauth_engine.save(grant, changed)
        out = _withdraw(owner, "revoke", key)
        assert (out["ok"], out["reason"], out["revocation_reason"]) == (False, "revocation_not_asked", reason)
        assert grant.path.is_file() and provider.counts["revoke"] == 0  # nothing was sent, nothing was forgotten
    assert _withdraw(owner, "disconnect", key) == {"ok": True}  # only the owner's own Disconnect forgets it
    assert not grant.path.exists()


def test_a_disconnect_and_a_revoke_answer_for_the_consent_the_card_showed(world):  # noqa: F811
    provider, owner, host = world
    _save_client(owner)
    key = _sign_in(provider, owner, CAL)["account"]["key"]
    shown = _service(owner)["accounts"][0]["consent"]
    for action in ("disconnect", "revoke"):
        missing = _post(owner, action, account=key)  # nothing shown: nothing confirmed
        assert (missing["ok"], missing["reason"]) == (False, "consent_changed")
    grant = service_sign_in._grant(CAL, "google", key)
    doc = oauth_engine.load(grant)
    oauth_engine.save(grant, {**doc, "tokens": {**doc["tokens"], "expires_at": 1}})
    assert _token(host, CAL).status_code == 200 and provider.counts["refresh"] == 1  # an ordinary token rotation
    assert _service(owner)["accounts"][0]["consent"] == shown
    _sign_in(provider, owner, CAL)  # the same account signs in again elsewhere
    for action in ("disconnect", "revoke"):
        stale = _post(owner, action, account=key, consent=shown)
        assert (stale["ok"], stale["reason"]) == (False, "consent_changed")
    assert grant.path.is_file() and provider.counts["revoke"] == 0
    assert _withdraw(owner, "revoke", key) == {"ok": True, "revoked": True}


def test_a_token_request_without_a_usable_account_says_which_and_what_to_do_next(world):  # noqa: F811
    provider, owner, host = world
    _save_client(owner)
    none = _token(host, CAL)
    assert none.status_code == 404 and none.json()["reason"] == "not_signed_in" and "Extensions" in none.json()["hint"]
    first = _sign_in(provider, owner, CAL)["account"]
    provider.person = {"sub": "sub-2", "email": "work@example.com"}
    second = _sign_in(provider, owner, CAL)["account"]
    many = _token(host, CAL)
    assert many.status_code == 404 and many.json()["reason"] == "account_required"
    assert first["key"] in many.json()["hint"] and second["key"] in many.json()["hint"]  # the accounts this very pick read
    assert _token(host, CAL, account="a-" + "0" * 20).json()["reason"] == "unknown_account"
    assert _token(host, CAL, account=first["key"]).status_code == 200
    for account in (first, second):  # both lose their tokens: the accounts exist, none can be used
        grant = service_sign_in._grant(CAL, "google", account["key"])
        oauth_engine.save(grant, {**oauth_engine.load(grant), "tokens": {}})
    dead = _token(host, CAL)
    assert dead.status_code == 404 and dead.json()["reason"] == "no_usable_account" and first["key"] in dead.json()["hint"]


def test_a_skill_naming_an_undeclared_service_is_told_which_and_a_storage_fault_stays_a_503(world, monkeypatch):  # noqa: F811
    _provider, owner, host = world
    _save_client(owner)
    nope = _token(host, CAL, service="nope")
    assert nope.status_code == 404 and nope.json()["reason"] == "unknown_service" and "GET /sign-in/accounts" in nope.json()["hint"]

    def unreadable(skill, service):
        raise oauth_engine.SignInRequired("sign_in_storage_unavailable", "the saved accounts could not be listed")

    monkeypatch.setattr(service_sign_in, "_account_files", unreadable)
    broken = _token(host, CAL)
    assert broken.status_code == 503 and broken.json()["code"] == "SIGN_IN_UNAVAILABLE"  # never a 404 "not signed in"


def test_status_says_what_it_could_not_read(world, monkeypatch):  # noqa: F811
    from ouroboros.gateway import sign_in_contracts
    from ouroboros.gateway.endpoint_index import HTTP_SHAPES

    _provider, owner, _host = world

    def unreadable(**_kwargs):
        raise oauth_engine.SignInRequired("sign_in_storage_unavailable", "the saved clients could not be read")

    monkeypatch.setattr(service_sign_in, "status", unreadable)
    answer = owner.get("/api/sign-in/status")
    assert answer.status_code == 503
    assert answer.json() == {"ok": False, "code": "SIGN_IN_UNAVAILABLE", "reason": "sign_in_storage_unavailable",
                             "error": "the saved clients could not be read"}
    assert set(answer.json()) == set(sign_in_contracts.SignInStatusProblem.__annotations__)
    assert "503:SignInStatusProblem" in HTTP_SHAPES["GET /api/sign-in/status"].responses


def test_deleting_a_skill_forgets_its_grants_and_its_own_handover_clients_only(world, monkeypatch, tmp_path):  # noqa: F811
    _isolated(monkeypatch)
    provider, owner, host = world
    _save_client(owner)
    keys = {skill: _sign_in(provider, owner, skill)["account"]["key"] for skill in (CAL, MAIL)}
    for skill, token in ((CAL, "cal-token"), (MAIL, "mail-token")):
        sent = host.post("/sign-in/handover", headers={"x-skill-token": token}, json={
            "service": "google", "grants": [], "clients": [{"client_id": f"{skill}-own", "label_hint": skill}]}).json()
        assert sent["clients"][0]["result"] == "imported"
    assert sorted(row["source"] for row in service_sign_in.clients()) == ["handover:calendar", "handover:mail", "owner"]
    result = delete_local_skill(tmp_path, skill_loader.find_skill(tmp_path, CAL), payload_root="skills/external/calendar")
    assert result["ok"] is True and "sign_ins_left" not in result, result
    assert not service_sign_in._grant(CAL, "google", keys[CAL]).path.parent.parent.exists()  # this skill's grants folder is gone
    assert service_sign_in._grant(MAIL, "google", keys[MAIL]).path.is_file() and _token(host, MAIL).status_code == 200
    assert sorted(row["source"] for row in service_sign_in.clients()) == ["handover:mail", "owner"]  # owner rows and the other skill's stay


def test_a_deletion_is_refused_before_any_change_while_a_sign_in_step_holds_its_lock(world, monkeypatch, tmp_path):  # noqa: F811
    _isolated(monkeypatch)
    provider, owner, _host = world
    _save_client(owner)
    key = _sign_in(provider, owner, CAL)["account"]["key"]
    loaded = skill_loader.find_skill(tmp_path, CAL)
    control = service_sign_in._grant(CAL, "google", "pending")
    with store.name_lock(tmp_path, control.lock, timeout_sec=0):  # a sign-in step holds the service's pending lock
        refused = delete_local_skill(tmp_path, loaded, payload_root="skills/external/calendar")
    assert refused["ok"] is False and refused["status_code"] == 409 and "nothing was deleted" in refused["error"]
    assert service_sign_in._grant(CAL, "google", key).path.is_file() and (tmp_path / "skills" / "external" / CAL).is_dir()
    assert delete_local_skill(tmp_path, loaded, payload_root="skills/external/calendar")["ok"] is True  # free again: it goes


def test_a_deletion_that_cannot_forget_every_sign_in_says_some_may_be_gone_and_keeps_the_skill(tmp_path, monkeypatch):
    _isolated(monkeypatch)
    data = tmp_path / "data"
    payload = data / "skills" / "external" / "demo"
    (payload / "scripts").mkdir(parents=True)
    (payload / "SKILL.md").write_text(_MANIFEST, encoding="utf-8")
    (payload / "scripts" / "check.py").write_text("print('fixture')\n", encoding="utf-8")
    state = _seed_state(data, "demo")
    account, pending = sorted(_seed_sign_in_records(data, "demo"))  # the account file, then pending.json
    loaded = skill_loader.find_skill(data, "demo")
    unlink = pathlib.Path.unlink

    def stuck(self, missing_ok=False):
        if self.name == "pending.json":
            raise OSError("fixture: cannot remove")
        return unlink(self, missing_ok=missing_ok)

    with monkeypatch.context() as patched:
        patched.setattr(pathlib.Path, "unlink", stuck)
        refused = delete_local_skill(data, loaded, payload_root="skills/external/demo")
    assert refused["ok"] is False and "some may already be gone" in refused["error"]
    assert payload.is_dir() and state.is_dir() and not account.exists() and pending.is_file()  # the skill stays; one file is gone
    assert delete_local_skill(data, loaded, payload_root="skills/external/demo")["ok"] is True  # delete again to finish
    assert not payload.exists() and not pending.exists()
