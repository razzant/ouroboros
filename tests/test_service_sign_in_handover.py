"""A skill hands the host the sign-ins it already held (``service_sign_in.handover``, ``POST /sign-in/handover``).

The account is the one the provider names for the handed token (a claim that differs is refused); a token
without the identity scope is kept unconfirmed under its skill's own key; a token the provider refuses is
kept as needing sign-in; a repeat answers ``already_present`` and a grant the owner made on the card wins.
A transfer resumes after a lost identity answer without a second refresh, a single-account service's
confirmed sign-in replaces its unconfirmed moved row, and a moved grant's first refresh reaches the skill
once the provider names the same account, due again only when its request never left. A refused grant says what this
host knows of its token (``exchange``), ``recover`` answers from the slot first and starts nothing, an exchange that
left is never displaced, a receipt keeps its transfer's connection and each owner's break reaches the skill. A verified offer's receipt reads back without the provider, an imported grant's own consent with it, after a lost reply or a late writer of the same attempt too, and a token request may bind the consent it saw; a verified generation whose fresh bearer's check failed reads back as that refusal until anything newer, and a later generation's own refusal before exchange then reads back as its own. A record of withdrawals that cannot be read makes both handover answers unavailable; two legacy keys of one person read back by their own transfers; after another person's withdrawal a strict first offer cut off before its attempt keeps its exact fresh bearer, with the registration saved from that same first POST, for a recover that checks only its identity, and one without such a bearer, cut off there or before its own refresh left, is recorded as ``resubmit`` and moves when the same offer comes again; a new sign-in finishes an interrupted withdrawal without waiting for another account. No answer,
log line or printed text carries a token or a secret.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace

import httpx
import pytest

from tests.test_service_sign_in_loopback import CAL, ISSUER, _save_client, _shown, _sign_in, world  # noqa: F401 -- the fixture

from ouroboros import oauth_engine, service_sign_in

SECRET = "personal-secret-0001"
HEADERS = {"x-skill-token": "cal-token"}


def _handover(host, provider, *, scope="openid email calendar.events", subject=None, expires_at=None, refresh=None):
    person = dict(provider.person)
    issued = provider._issue("personal-client", person, scope)
    item = {"legacy_key": f"google:{person['email']}", "label": person["email"], "client_id": "personal-client",
            "tokens": {"refresh_token": refresh or issued["refresh_token"], "access_token": issued["access_token"],
                       "expires_at": expires_at, "scope": scope}}
    if subject:
        item["subject"] = subject
    body = {"service": "google", "grants": [item], "clients": [
        {"client_id": "personal-client", "client_secret": SECRET, "label_hint": "Calendar"}]}
    answer = host.post("/sign-in/handover", headers={"x-skill-token": "cal-token"}, json=body)
    assert answer.status_code == 200, answer.text
    return answer.json(), issued, body


def test_a_handed_grant_moves_once_and_the_provider_names_its_account(world, caplog, capsys):  # noqa: F811
    provider, owner, host = world
    caplog.set_level(logging.DEBUG)
    first, issued, body = _handover(host, provider)
    assert first["clients"][0]["result"] == "imported" and first["grants"][0]["result"] == "imported"
    key = first["grants"][0]["account"]["key"]
    assert key == service_sign_in.account_key(ISSUER, "sub-1")  # confirmed by the provider, not by the skill
    assert first["grants"][0]["account"]["label"] == "owner@example.com"
    again = host.post("/sign-in/handover", headers={"x-skill-token": "cal-token"}, json=body).json()
    assert again["grants"][0]["result"] == "already_present" and again["clients"][0]["result"] == "already_present"
    token = host.post("/sign-in/token", headers={"x-skill-token": "cal-token"}, json={"service": "google"}).json()
    assert token["access_token"] == issued["access_token"]  # still valid: no refresh happened
    row = owner.get("/api/sign-in/status").json()["services"][0]
    assert row["accounts"][0]["via"] == "handover" and row["clients"][0]["label"] == "Imported from Calendar"
    printed = capsys.readouterr()
    seen = caplog.text + printed.out + printed.err + json.dumps([first, again, row])
    assert issued["refresh_token"] not in seen and SECRET not in seen


def test_the_owners_own_sign_in_wins_and_a_forged_account_is_refused(world):  # noqa: F811
    provider, owner, host = world
    _handover(host, provider)
    _sign_in(provider, owner, CAL)  # the owner signs the same account in on the card
    assert _handover(host, provider)[0]["grants"][0]["result"] == "kept_newer"
    forged = _handover(host, provider, subject="someone-else")[0]["grants"][0]
    assert (forged["result"], forged["reason"]) == ("refused", "identity_mismatch")


def test_a_token_without_identity_is_kept_unconfirmed_and_a_dead_one_asks_for_sign_in(world):  # noqa: F811
    provider, owner, host = world
    unconfirmed = _handover(host, provider, scope="calendar.events")[0]["grants"][0]
    assert unconfirmed["account"]["key"] == service_sign_in.account_key(ISSUER, "google:owner@example.com", skill=CAL)
    provider.person = {"sub": "sub-3", "email": "old@example.com"}
    dead = _handover(host, provider, refresh="no-longer-known", expires_at="2020-01-01T00:00:00Z")[0]["grants"][0]
    assert dead["result"] == "imported" and dead["state"] == "sign_in_required"
    rows = {row["key"]: row for row in owner.get("/api/sign-in/status").json()["services"][0]["accounts"]}
    held = rows[unconfirmed["account"]["key"]]
    assert held["verified"] is False and held["state"] == "signed_in"
    assert rows[dead["account"]["key"]]["state"] == "sign_in_required"  # the row stays, with Sign in again


def test_late_consent_replaces_only_its_saved_predecessor(request):
    provider, _owner, host = request.getfixturevalue("world")  # MockTransport and ASGI TestClients; no network
    provider.person = {'sub': 'calendar-owner', 'email': 'first@example.test'}
    headers = {'x-skill-token': 'cal-token'}
    scope = 'openid email calendar.events'
    def issued():
        return provider._issue('personal-client', provider.person, scope)
    def send(grant, clients=()):
        response = host.post('/sign-in/handover', headers=headers,
                             json={'service': 'google', 'grants': [grant], 'clients': list(clients)})
        assert response.status_code == 200
        return response.json()['grants'][0]
    first = {'legacy_key': 'calendar-row', 'legacy_generation': 'first-generation', 'subject': 'calendar-owner',
             'label': 'first@example.test', 'client_id': 'personal-client', 'tokens': issued()}
    original = send(first, [{'client_id': 'personal-client', 'client_secret': SECRET}])
    assert (original['result'], original['legacy_generation']) == ('imported', 'first-generation')
    provider.person['email'] = 'renamed@example.test'
    second = {**first, 'legacy_generation': 'second-generation', 'previous_legacy_generation': 'first-generation',
              'label': 'renamed@example.test', 'tokens': issued()}
    saved = send(second)
    assert (saved['result'], saved['legacy_generation'], saved['account']['key']) == (
        'imported', 'second-generation', original['account']['key'])
    [shown] = host.get('/sign-in/accounts', headers=headers).json()['services'][0]['accounts']
    assert (shown['key'], shown['subject'], shown['legacy_generation']) == (
        original['account']['key'], 'calendar-owner', 'second-generation')
    assert send(second)['result'] == 'already_present'
    token = host.post('/sign-in/token', headers=headers, json={'service': 'google', 'account': shown['key']}).json()
    assert token['access_token'] == second['tokens']['access_token']


def test_a_handed_web_client_keeps_its_exact_return_and_one_that_is_not_https_is_refused_alone(world):  # noqa: F811
    _provider, owner, host = world
    good = "exact:https://box.example/onboarding/static/provider-return.html"
    answer = host.post("/sign-in/handover", headers={"x-skill-token": "cal-token"}, json={"service": "google", "clients": [
        {"client_id": "web-client", "label_hint": "Calendar", "return_hint": good},
        {"client_id": "lan-client", "label_hint": "Calendar", "return_hint": "exact:http://192.168.1.5/back"}],
        "grants": []}).json()
    assert [item["result"] for item in answer["clients"]] == ["imported", "refused"]
    assert answer["clients"][1]["reason"] == "invalid_return"
    rows = owner.get("/api/sign-in/status").json()["services"][0]["clients"]
    assert [row["return"] for row in rows] == [good]


def test_unknown_return_moves_a_verified_grant_and_a_configured_client_can_sign_in(world):  # noqa: F811
    provider, owner, host = world
    scope = "openid email calendar.events"
    issued = provider._issue("old-application", dict(provider.person), scope)
    body = {"service": "google", "clients": [{"client_id": "old-application", "return_hint": "unknown"}],
            "grants": [{"legacy_key": "earlier-account", "legacy_generation": "g1", "client_id": "old-application",
                        "require_verified": True, "tokens": issued}]}
    moved = host.post("/sign-in/handover", headers=HEADERS, json=body).json()["grants"][0]
    assert (moved["result"], moved["state"]) == ("imported", "signed_in") and moved["connection_id"]
    recovered = host.post("/sign-in/handover", headers=HEADERS, json={**body, "recover": True}).json()["grants"][0]
    assert (recovered["result"], recovered["consent"], recovered["connection_id"]) == (
        "imported", moved["consent"], moved["connection_id"])
    assert provider.counts["refresh"] == 0
    key = moved["account"]["key"]

    def view():
        return owner.get("/api/sign-in/status").json()["services"][0]

    [client] = view()["clients"]
    assert client["return"] == "unknown" and "registration" not in client
    renamed = owner.post("/api/sign-in", json={"action": "client_save", "issuer": ISSUER,
                         "client": {"id": client["id"], "label": "Earlier app"}}).json()
    assert renamed["client"]["return"] == "unknown"
    read = host.post("/sign-in/token", headers=HEADERS, json={"service": "google", "account": key,
                     "connection_id": moved["connection_id"], "rejected": issued["access_token"]}).json()
    assert read["ok"] and read["connection_id"] == moved["connection_id"]
    assert read["access_token"] != issued["access_token"] and provider.counts["refresh"] == 1
    configured = _save_client(owner, client_id="configured-application")
    provider.person = {"sub": "sub-2", "email": "second@example.test"}
    started = owner.post("/api/sign-in", json={"action": "begin", "skill": CAL, "service": "google",
                         "scope_words": scope.split(), **_shown(owner, CAL, configured)}).json()
    assert started["ok"] and started["client"]["id"] == configured
    assert owner.post("/api/sign-in", json={"action": "complete", "returned_url": provider.approve(started["url"])}).json()["ok"]
    consent = next(row["consent"] for row in view()["accounts"] if row["key"] == key)
    revoked = owner.post("/api/sign-in", json={"action": "revoke", "skill": CAL, "service": "google",
                         "account": key, "consent": consent}).json()
    assert revoked["revoked"] and provider.counts["revoke"] == 2
    assert [row["label"] for row in view()["accounts"]] == ["second@example.test"]


@pytest.mark.parametrize("expired", [False, True], ids=["current", "renewed"])
def test_a_transfer_resumes_after_a_lost_identity_answer_without_a_second_refresh(world, monkeypatch, expired):  # noqa: F811
    provider, _owner, host = world
    handle, outage = provider.handle, ["/userinfo"]

    def once_unavailable(request):
        if outage and request.url.path == outage[0]:
            outage.clear()
            return httpx.Response(503)
        return handle(request)

    monkeypatch.setattr(provider, "handle", once_unavailable)
    first, _issued, body = _handover(host, provider, expires_at="2020-01-01T00:00:00Z" if expired else None)
    assert (first["grants"][0]["result"], first["grants"][0]["reason"], first["grants"][0]["exchange"]) == (
        "refused", "provider_unreachable", "saved" if expired else "none")  # a fresh bearer's lost identity answer sent no refresh
    again = host.post("/sign-in/handover", headers={"x-skill-token": "cal-token"}, json=body).json()["grants"][0]
    assert (again["result"], again["state"]) == ("imported", "signed_in") and provider.counts["refresh"] == int(expired)
    assert again["account"]["key"] == service_sign_in.account_key(ISSUER, "sub-1") and again["connection_id"]


def test_a_single_account_services_confirmed_sign_in_replaces_its_unconfirmed_moved_row(world, monkeypatch):  # noqa: F811
    provider, owner, host = world
    declared = service_sign_in._declared
    monkeypatch.setattr(service_sign_in, "_declared", lambda *names: replace(declared(*names), accounts="one"))
    moved = _handover(host, provider, scope="calendar.events")[0]["grants"][0]  # no identity scope: unconfirmed
    assert moved["result"] == "imported" and "connection_id" not in moved
    done = _sign_in(provider, owner, CAL)  # the owner's Sign in again on that row, with the moved client
    assert done["ok"] is True and done["account"]["key"] == service_sign_in.account_key(ISSUER, "sub-1")
    [row] = host.get("/sign-in/accounts", headers={"x-skill-token": "cal-token"}).json()["services"][0]["accounts"]
    assert (row["key"], row["verified"], row["via"]) == (done["account"]["key"], True, "card") and row["connection_id"]


def test_a_moved_grants_first_refresh_reaches_the_skill_once_the_provider_names_its_account(world, monkeypatch):  # noqa: F811
    provider, _owner, host = world
    paths, handle = [], provider.handle
    monkeypatch.setattr(provider, "handle", lambda request: paths.append(request.url.path) or handle(request))
    moved = _handover(host, provider)[0]["grants"][0]
    grant = service_sign_in._grant(CAL, "google", moved["account"]["key"])
    saved = oauth_engine.load(grant)
    oauth_engine.save(grant, {**saved, "tokens": {**saved["tokens"], "expires_at": 1}})
    paths.clear()

    def token():
        return host.post("/sign-in/token", headers={"x-skill-token": "cal-token"}, json={"service": "google"}).json()

    first = token()
    assert first["ok"] is True and paths == ["/token", "/userinfo"]
    assert token()["access_token"] == first["access_token"] and paths == ["/token", "/userinfo"]  # checked once


@pytest.mark.parametrize("known_age", [False, True], ids=["unknown-age", "known-age"])
def test_a_handed_bearer_the_provider_refuses_moves_only_after_one_ordinary_refresh(world, known_age):  # noqa: F811
    provider, _owner, host = world
    issued = provider._issue("personal-client", dict(provider.person), "openid email calendar.events")
    provider.access.pop(issued["access_token"])  # the provider no longer honours the handed access token
    tokens = {"refresh_token": issued["refresh_token"], "access_token": issued["access_token"],
              "expires_at": "2999-01-01T00:00:00Z" if known_age else None, "scope": "openid email calendar.events"}
    body = {"service": "google", "clients": [{"client_id": "personal-client", "client_secret": SECRET, "label_hint": "Calendar"}],
            "grants": [{"legacy_key": "google:owner@example.com", "label": "owner@example.com", "client_id": "personal-client",
                        "tokens": tokens}]}
    moved = host.post("/sign-in/handover", headers={"x-skill-token": "cal-token"}, json=body).json()["grants"][0]
    listed = host.get("/sign-in/accounts", headers={"x-skill-token": "cal-token"}).json()["services"][0]["accounts"]
    if known_age:  # no extra refresh: nothing moves and the skill keeps its own route
        assert (moved["result"], moved["reason"], listed, provider.counts["refresh"]) == ("refused", "token_rejected", [], 0)
    else:  # its age was unknown: one ordinary refresh, then the provider names the account
        assert (moved["result"], moved["state"], provider.counts["refresh"]) == ("imported", "signed_in", 1)
        assert [row["key"] for row in listed] == [service_sign_in.account_key(ISSUER, "sub-1")]


def test_a_moved_grants_first_refresh_is_confirmed_even_when_the_provider_keeps_the_bearer(world, monkeypatch):  # noqa: F811
    provider, _owner, host = world
    moved = _handover(host, provider)[0]["grants"][0]
    grant = service_sign_in._grant(CAL, "google", moved["account"]["key"])
    saved = oauth_engine.load(grant)
    kept = saved["tokens"]["access_token"]
    oauth_engine.save(grant, {**saved, "tokens": {**saved["tokens"], "expires_at": 1}})
    handle, paths, outage = provider.handle, [], ["/userinfo"]

    def keeps_bearer(request):  # the provider renews the grant and answers with the same access token
        paths.append(request.url.path)
        if outage and request.url.path == outage[0]:
            outage.clear()
            return httpx.Response(503)
        answer = handle(request)
        return httpx.Response(200, json={**answer.json(), "access_token": kept}) if request.url.path == "/token" else answer

    monkeypatch.setattr(provider, "handle", keeps_bearer)
    calls = provider.counts["token_calls"]

    def token():
        return host.post("/sign-in/token", headers={"x-skill-token": "cal-token"}, json={"service": "google"})

    assert token().status_code == 503 and paths == ["/token", "/userinfo"]  # renewed; its identity answer was lost
    [row] = host.get("/sign-in/accounts", headers={"x-skill-token": "cal-token"}).json()["services"][0]["accounts"]
    assert row["state"] == "signed_in"  # the saved renewal waits only for its account check
    answer = token().json()  # only the identity read repeats
    assert answer["access_token"] == kept and paths == ["/token", "/userinfo", "/userinfo"]
    assert provider.counts["token_calls"] == calls + 1


@pytest.mark.parametrize("failure", [httpx.ConnectError, httpx.ReadTimeout], ids=["never-left", "may-have-left"])
def test_a_moved_grants_first_refresh_is_due_again_only_when_its_request_never_left(world, monkeypatch, failure):  # noqa: F811
    provider, _owner, host = world
    moved = _handover(host, provider)[0]["grants"][0]
    grant = service_sign_in._grant(CAL, "google", moved["account"]["key"])
    saved = oauth_engine.load(grant)
    oauth_engine.save(grant, {**saved, "tokens": {**saved["tokens"], "expires_at": 1}})
    handle, outage = provider.handle, ["/token"]

    def once_failing(request):
        if outage and request.url.path == outage[0]:
            outage.clear()
            raise failure("fixture", request=request)
        return handle(request)

    monkeypatch.setattr(provider, "handle", once_failing)
    lost = host.post("/sign-in/token", headers=HEADERS, json={"service": "google"})
    assert (lost.status_code, lost.json()["reason"]) == (503, "provider_unreachable")
    again = host.post("/sign-in/token", headers=HEADERS, json={"service": "google"})
    if failure is httpx.ConnectError:  # it never left: the first refresh is due again and goes out once
        assert again.status_code == 200 and provider.counts["refresh"] == 1
    else:  # the provider may have seen it: never sent twice
        assert (again.status_code, again.json()["reason"], provider.counts["refresh"]) == (409, "handover_unconfirmed", 0)


def test_a_refused_grant_says_what_left_and_an_unresolved_exchange_is_never_displaced(world, monkeypatch):  # noqa: F811
    provider, _owner, host = world
    handle, failures = provider.handle, []

    def failing(request):
        if failures and request.url.path.startswith(failures[0][0]):
            raise failures.pop(0)[1]("fixture", request=request)
        return handle(request)

    monkeypatch.setattr(provider, "handle", failing)
    failures.append(("/.well-known/", httpx.ConnectError))
    first, _issued, body = _handover(host, provider, expires_at="2020-01-01T00:00:00Z")
    assert (first["grants"][0]["reason"], first["grants"][0]["exchange"]) == ("provider_unreachable", "none")
    failures.append(("/token", httpx.ReadTimeout))
    lost = host.post("/sign-in/handover", headers=HEADERS, json=body).json()["grants"][0]  # the same grant, sent again
    assert (lost["reason"], lost["exchange"]) == ("provider_unreachable", "unknown")
    failures.append(("/.well-known/", httpx.ConnectError))
    offline = host.post("/sign-in/handover", headers=HEADERS, json=body).json()["grants"][0]
    assert (offline["reason"], offline["exchange"]) == ("provider_unreachable", "unknown")  # today's outage proves nothing
    newer = provider._issue("personal-client", dict(provider.person), "openid email calendar.events")
    other = {**body["grants"][0], "tokens": {**body["grants"][0]["tokens"], "refresh_token": newer["refresh_token"]}}
    displaced = host.post("/sign-in/handover", headers=HEADERS, json={**body, "grants": [other]}).json()["grants"][0]
    assert (displaced["reason"], displaced["exchange"]) == ("handover_unconfirmed", "unknown")
    failures.append(("/", httpx.ConnectError))
    again = host.post("/sign-in/handover", headers=HEADERS, json={**body, "recover": True}).json()["grants"][0]
    assert (again["reason"], again["exchange"], provider.counts["refresh"]) == ("handover_unconfirmed", "unknown", 0)
    assert failures  # recover answered from the slot: the provider was not asked


def test_recover_files_a_kept_answer_past_an_outage_and_starts_nothing_without_one(world, monkeypatch):  # noqa: F811
    provider, _owner, host = world
    handle, plan = provider.handle, [("/userinfo", None), ("/.well-known/", httpx.ConnectError)]

    def flaky(request):
        if plan and request.url.path.startswith(plan[0][0]):
            failure = plan.pop(0)[1]
            if failure is None:
                return httpx.Response(503)
            raise failure("fixture", request=request)
        return handle(request)

    monkeypatch.setattr(provider, "handle", flaky)
    first, _issued, body = _handover(host, provider, expires_at="2020-01-01T00:00:00Z")
    assert (first["grants"][0]["reason"], first["grants"][0]["exchange"]) == ("provider_unreachable", "saved")
    hidden = host.post("/sign-in/handover", headers=HEADERS, json={**body, "recover": True}).json()["grants"][0]
    assert (hidden["reason"], hidden["exchange"]) == ("provider_unreachable", "saved")  # an outage never hides it
    recovered = host.post("/sign-in/handover", headers=HEADERS, json={**body, "recover": True}).json()
    assert recovered["clients"] == [] and recovered["grants"][0]["result"] == "imported" and provider.counts["refresh"] == 1
    provider.person = {"sub": "sub-9", "email": "late@example.com"}
    issued = provider._issue("personal-client", dict(provider.person), "openid email calendar.events")
    item = {"legacy_key": "google:late@example.com", "label": "late@example.com", "client_id": "personal-client",
            "tokens": {"refresh_token": issued["refresh_token"], "access_token": issued["access_token"],
                       "scope": "openid email calendar.events"}}
    unstarted = host.post("/sign-in/handover", headers=HEADERS,
                          json={"service": "google", "recover": True, "grants": [item]}).json()["grants"][0]
    assert (unstarted["reason"], unstarted["exchange"]) == ("handover_not_started", "unknown")
    pending = oauth_engine.load(service_sign_in._grant(CAL, "google", "pending"))
    assert oauth_engine._digest(item["legacy_key"]) not in pending.get("handovers", {})  # nothing was recorded
    moved = host.post("/sign-in/handover", headers=HEADERS, json={"service": "google", "grants": [item]}).json()["grants"][0]
    assert moved["result"] == "imported" and provider.counts["refresh"] == 1  # its fresh bearer needed no refresh


def test_a_transfers_receipt_keeps_its_connection_and_each_owners_break_reaches_the_skill(world):  # noqa: F811
    provider, owner, host = world
    moved, issued, body = _handover(host, provider)
    key, connection = moved["grants"][0]["account"]["key"], moved["grants"][0]["connection_id"]

    def disconnect() -> dict:
        row = next(s for s in owner.get("/api/sign-in/status").json()["services"] if s["skill"] == CAL)
        consent = next(a["consent"] for a in row["accounts"] if a["key"] == key)
        return owner.post("/api/sign-in", json={"action": "disconnect", "skill": CAL, "service": "google",
                                                "account": key, "consent": consent}).json()

    assert host.post("/sign-in/handover", headers=HEADERS, json=body).json()["grants"][0]["connection_id"] == connection
    assert disconnect() == {"ok": True}
    after = host.post("/sign-in/handover", headers=HEADERS, json=body).json()["grants"][0]
    assert (after["reason"], after["exchange"]) == ("account_disconnected", "unknown")  # an explicit break: never none
    _sign_in(provider, owner, CAL)  # the same person signs in again on the card: another connection
    assert disconnect() == {"ok": True}
    path = service_sign_in._withdrawals_file(CAL).path
    text = path.read_text(encoding="utf-8")
    rows = json.loads(text)["accounts"]
    assert [(row["via"], row.get("legacy_key"), row["subject"], row["connection_id"] == connection) for row in rows] == [
        ("handover", "google:owner@example.com", "sub-1", True), ("card", None, "sub-1", False)]  # the first break stays
    assert SECRET not in text and issued["refresh_token"] not in text
    _sign_in(provider, owner, CAL)
    for kept in ("{not json", '{"v": 2, "other": []}'):  # a record that cannot be read, or of another form, is never replaced
        path.write_text(kept, encoding="utf-8")
        assert disconnect()["reason"] == "sign_in_storage_unavailable" and path.read_text(encoding="utf-8") == kept
    assert [a["key"] for a in host.get("/sign-in/accounts", headers=HEADERS).json()["services"][0]["accounts"]] == [key]


def test_disconnect_ends_access_before_publishing_the_break(world, monkeypatch):  # noqa: F811
    provider, owner, _host = world
    _save_client(owner)
    assert _sign_in(provider, owner, CAL)["ok"] is True
    services = owner.get("/api/sign-in/status").json()["services"]
    [account] = next(row["accounts"] for row in services if row["skill"] == CAL)
    key = account["key"]
    grant = service_sign_in._grant(CAL, "google", key)
    ledger = service_sign_in._withdrawals_file(CAL).path
    had_bearer = bool((oauth_engine.load(grant).get("tokens") or {}).get("access_token"))
    no_prior_break = service_sign_in._withdrawals(CAL) == []
    assert had_bearer and no_prior_break
    write, observed = service_sign_in.atomic_write_json, []

    def observe(path, payload, **kwargs):
        if path != ledger:
            return write(path, payload, **kwargs)
        saved = oauth_engine.load(grant)
        ended = bool(saved) and not saved.get("tokens")
        result = write(path, payload, **kwargs)
        rows = service_sign_in._withdrawals(CAL) or []
        published = any(row.get("service") == "google" and row.get("account") == key for row in rows)
        observed.append((ended, published, grant.path.is_file()))
        return result

    with monkeypatch.context() as probe:
        probe.setattr(service_sign_in, "atomic_write_json", observe)
        response = owner.post("/api/sign-in", json={"action": "disconnect", "skill": CAL, "service": "google",
                                                    "account": key, "consent": account["consent"]})
    assert (response.status_code, response.json().get("ok")) == (200, True)
    assert observed == [(True, True, True)]  # no local tokens, break on disk, account file still present
    assert not grant.path.exists()
    assert _sign_in(provider, owner, CAL)["state"] == "signed_in"


def test_only_the_rate_limits_refusal_before_reading_a_handover_says_it_never_started(world, monkeypatch):  # noqa: F811
    _provider, _owner, host = world
    stranger = host.post("/sign-in/handover", headers={"x-skill-token": "nobody"}, json={"service": "google", "grants": []})
    assert stranger.status_code == 403 and "handover_started" not in stranger.json()
    monkeypatch.setattr(host.app.state.host_service_context.rate_limiter, "allow", lambda _key: False)
    limited = host.post("/sign-in/handover", headers=HEADERS, json={"service": "google", "grants": []})
    assert (limited.status_code, limited.json()) == (
        429, {"ok": False, "error": "rate limit exceeded", "handover_started": False})
    token = host.post("/sign-in/token", headers=HEADERS, json={"service": "google"})
    assert token.status_code == 429 and "handover_started" not in token.json()


def test_recover_returns_a_kept_receipt_only_for_the_exact_handed_sign_in(world, monkeypatch):  # noqa: F811
    provider, _owner, host = world
    issued = provider._issue("personal-client", dict(provider.person), "openid email calendar.events")
    item = {"legacy_key": "google:owner@example.com", "legacy_generation": "g1", "label": "owner@example.com",
            "client_id": "personal-client", "tokens": {"refresh_token": issued["refresh_token"],
                                                       "access_token": issued["access_token"], "scope": issued["scope"]}}
    moved = host.post("/sign-in/handover", headers=HEADERS, json={"service": "google", "grants": [item], "clients": [
        {"client_id": "personal-client", "client_secret": SECRET}]}).json()["grants"][0]

    def offline(request):
        raise AssertionError(f"recover asked the provider: {request.url}")

    monkeypatch.setattr(provider, "handle", offline)

    def recover(grant):
        return host.post("/sign-in/handover", headers=HEADERS,
                         json={"service": "google", "recover": True, "grants": [grant]}).json()["grants"][0]

    exact = recover(item)
    assert (exact["result"], exact["legacy_generation"], exact["connection_id"]) == ("already_present", "g1", moved["connection_id"])
    rotated = recover({**item, "tokens": {**item["tokens"], "refresh_token": "rotated-since"}})  # another token after a lost answer
    untagged = recover({name: value for name, value in item.items() if name != "legacy_generation"})
    assert [(answer["result"], answer["reason"], answer["exchange"]) for answer in (rotated, untagged)] == [
        ("refused", "handover_unconfirmed", "unknown")] * 2


def test_a_renewal_replaces_only_the_connection_its_transfer_recorded(world):  # noqa: F811
    provider, owner, host = world

    def send(grant, **extra):
        return host.post("/sign-in/handover", headers=HEADERS,
                         json={"service": "google", "grants": [grant], **extra}).json()["grants"][0]

    def tokens():
        return provider._issue("personal-client", dict(provider.person), "openid email calendar.events")

    first = {"legacy_key": "calendar-row", "legacy_generation": "g1", "subject": "sub-1", "label": "owner@example.com",
             "client_id": "personal-client", "tokens": tokens()}
    connection = send(first, clients=[{"client_id": "personal-client", "client_secret": SECRET}])["connection_id"]
    _sign_in(provider, owner, CAL)  # an ordinary re-sign-in of the same person on the card keeps the connection
    second = {**first, "legacy_generation": "g2", "previous_legacy_generation": "g1", "tokens": tokens()}
    renewed = send(second)
    assert (renewed["result"], renewed["legacy_generation"], renewed["connection_id"]) == ("kept_newer", "g2", connection)
    control, slot_key = service_sign_in._grant(CAL, "google", "pending"), oauth_engine._digest("calendar-row")
    saved = oauth_engine.load(control)
    older = {name: value for name, value in saved["handovers"][slot_key].items() if name not in ("source", "connection_id")}
    oauth_engine.save(control, {**saved, "handovers": {**saved["handovers"], slot_key: older}})  # as an older core wrote it
    unproven = send({**second, "legacy_generation": "g3", "previous_legacy_generation": "g2", "tokens": tokens()})
    lost = send(second, recover=True)
    assert [(answer["result"], answer["reason"], answer["exchange"]) for answer in (unproven, lost)] == [
        ("refused", "connection_changed", "unknown"), ("refused", "handover_unconfirmed", "unknown")]


def _strict_offer(provider, **tokens):
    scope = "openid email calendar.events"
    issued = provider._issue("personal-client", dict(provider.person), scope)
    offer = {"legacy_key": "temporary-calendar-row", "legacy_generation": "local-offer-1"}
    body = {"service": "google", "clients": [{"client_id": "personal-client", "client_secret": SECRET}],
            "grants": [{**offer, "client_id": "personal-client", "require_verified": True, "tokens": {**issued, **tokens}}]}
    return issued, offer, body


def _read_without_the_provider(host, provider, monkeypatch, offer):
    paths, handle = [], provider.handle
    monkeypatch.setattr(provider, "handle", lambda request: paths.append(request.url.path) or handle(request))
    read = host.post("/sign-in/handover", headers=HEADERS, json={"service": "google", "receipt": offer}).json()
    assert read["ok"] is True and paths == []  # discovery, identity and token addresses were never asked
    assert "access_token" not in json.dumps(read) and "refresh_token" not in json.dumps(read)
    return read["receipt"]


def test_a_verified_import_reads_back_with_its_consent_and_a_write_may_bind_that_consent(world, monkeypatch):  # noqa: F811
    provider, _owner, host = world
    issued, offer, body = _strict_offer(provider)
    first = host.post("/sign-in/handover", headers=HEADERS, json=body).json()["grants"][0]
    assert first["result"] == "imported" and first["consent"] and first["connection_id"]
    [row] = host.get("/sign-in/accounts", headers=HEADERS).json()["services"][0]["accounts"]
    assert row["consent"] == first["consent"]
    receipt = _read_without_the_provider(host, provider, monkeypatch, offer)
    assert (receipt["result"], receipt["legacy_key"], receipt["legacy_generation"], receipt["consent"]) == (
        "imported", offer["legacy_key"], offer["legacy_generation"], first["consent"])
    assert (receipt["account"]["key"], receipt["connection_id"]) == (row["key"], row["connection_id"])
    bound = host.post("/sign-in/token", headers=HEADERS, json={
        "service": "google", "account": row["key"], "connection_id": row["connection_id"], "consent": row["consent"]}).json()
    assert bound["ok"] and bound["consent"] == first["consent"] and bound["access_token"] == issued["access_token"]


def test_a_kept_newer_receipt_reads_back_without_becoming_write_consent(world, monkeypatch):  # noqa: F811
    provider, owner, host = world
    _save_client(owner)
    _sign_in(provider, owner, CAL)
    [selected] = host.get("/sign-in/accounts", headers=HEADERS).json()["services"][0]["accounts"]
    _issued, offer, body = _strict_offer(provider)
    first = host.post("/sign-in/handover", headers=HEADERS, json=body).json()["grants"][0]
    assert first["result"] == "kept_newer" and "consent" not in first
    receipt = _read_without_the_provider(host, provider, monkeypatch, offer)
    assert (receipt["result"], receipt["legacy_key"], receipt["legacy_generation"]) == (
        "kept_newer", offer["legacy_key"], offer["legacy_generation"])
    assert (receipt["account"]["key"], receipt["connection_id"]) == (selected["key"], selected["connection_id"])
    assert "consent" not in receipt
    token = host.post("/sign-in/token", headers=HEADERS, json={
        "service": "google", "account": selected["key"], "connection_id": selected["connection_id"],
        "consent": selected["consent"]}).json()  # a new operation binds the consent its own row names
    assert token["ok"] and token["consent"] == selected["consent"]


@pytest.mark.parametrize("after", ["offered_again", "card_break"])
def test_a_fresh_bearers_failed_identity_check_ends_its_generation_with_no_exchange(world, monkeypatch, after):  # noqa: F811
    provider, owner, host = world
    _issued, offer, body = _strict_offer(provider, expires_at="2999-01-01T00:00:00Z")
    handle, outage = provider.handle, ["/userinfo"]

    def once_unavailable(request):
        if outage and request.url.path == outage[0]:
            outage.clear()
            return httpx.Response(503)
        return handle(request)

    monkeypatch.setattr(provider, "handle", once_unavailable)
    first = host.post("/sign-in/handover", headers=HEADERS, json=body).json()["grants"][0]
    assert (first["result"], first["reason"], first["exchange"]) == ("refused", "provider_unreachable", "none")
    receipt = _read_without_the_provider(host, provider, monkeypatch, offer)  # as a process that never saw that answer
    assert receipt == {**offer, "result": "refused", "reason": "provider_unreachable", "exchange": "none"}
    lost = host.post("/sign-in/handover", headers=HEADERS, json={**body, "recover": True}).json()["grants"][0]
    assert (lost["reason"], lost["exchange"], provider.counts["refresh"]) == ("handover_not_started", "unknown", 0)
    assert host.get("/sign-in/accounts", headers=HEADERS).json()["services"][0]["accounts"] == []
    if after == "offered_again":  # a new attempt of that generation supersedes the refusal
        again = host.post("/sign-in/handover", headers=HEADERS, json=body).json()["grants"][0]
        assert (again["result"], again["legacy_generation"], provider.counts["refresh"]) == (
            "imported", offer["legacy_generation"], 0)
        assert _read_without_the_provider(host, provider, monkeypatch, offer)["result"] == "imported"
    else:  # the owner signs someone in on the card and disconnects them: an offer naming no subject cannot rule that out
        key = _sign_in(provider, owner, CAL)["account"]["key"]
        shown = next(s for s in owner.get("/api/sign-in/status").json()["services"] if s["skill"] == CAL)
        consent = next(a["consent"] for a in shown["accounts"] if a["key"] == key)
        assert owner.post("/api/sign-in", json={"action": "disconnect", "skill": CAL, "service": "google",
                                                "account": key, "consent": consent}).json() == {"ok": True}
        assert _read_without_the_provider(host, provider, monkeypatch, offer)["exchange"] == "unknown"


def test_recover_verifies_an_exact_fresh_bearer_after_a_lost_reply_without_a_refresh(world, monkeypatch):  # noqa: F811
    provider, _owner, host = world
    _issued, offer, body = _strict_offer(provider, expires_at="2999-01-01T00:00:00Z")
    handle, outage, refused = provider.handle, ["/userinfo"], service_sign_in._refused

    def once_unavailable(request):
        if outage and request.url.path == outage[0]:
            outage.clear()
            return httpx.Response(503)
        return handle(request)

    async def exited(*_args, **_kwargs):  # the process ended before this refusal was recorded: no answer arrived
        return {}

    monkeypatch.setattr(provider, "handle", once_unavailable)
    monkeypatch.setattr(service_sign_in, "_refused", exited)
    host.post("/sign-in/handover", headers=HEADERS, json=body)  # its claim is kept, nothing is settled
    monkeypatch.setattr(service_sign_in, "_refused", refused)
    unsettled = _read_without_the_provider(host, provider, monkeypatch, offer)
    assert (unsettled["result"], unsettled["exchange"]) == ("refused", "unknown")  # a claim nobody ended proves nothing
    recovered = host.post("/sign-in/handover", headers=HEADERS, json={**body, "recover": True}).json()["grants"][0]
    assert (recovered["result"], recovered["state"], provider.counts["refresh"]) == ("imported", "signed_in", 0)
    receipt = _read_without_the_provider(host, provider, monkeypatch, offer)
    assert (receipt["result"], receipt["consent"], receipt["connection_id"]) == (
        "imported", recovered["consent"], recovered["connection_id"])


@pytest.mark.parametrize("pointer", ["recorded", "older", "late_writer"])
def test_a_lost_reply_settles_with_its_transfers_own_consent_never_a_later_sign_ins(world, monkeypatch, pointer):  # noqa: F811
    provider, owner, host = world
    _issued, offer, body = _strict_offer(provider)
    control, slot_key = service_sign_in._grant(CAL, "google", "pending"), oauth_engine._digest(offer["legacy_key"])
    account = service_sign_in._grant(CAL, "google", service_sign_in.account_key(ISSUER, "sub-1"))
    pointers, save = [], oauth_engine._save_sign_in

    def observed(grant, doc):  # each control write naming a target, and whether the account file existed by then
        if grant.path == control.path and (((doc.get("handovers") or {}).get(slot_key) or {}).get("attempt") or {}).get("target"):
            pointers.append((json.loads(json.dumps(doc)), account.path.exists()))
        save(grant, doc)

    monkeypatch.setattr(oauth_engine, "_save_sign_in", observed)
    first = host.post("/sign-in/handover", headers=HEADERS, json=body).json()["grants"][0]
    monkeypatch.setattr(oauth_engine, "_save_sign_in", save)
    [(saved, published)] = pointers
    attempt = saved["handovers"][slot_key]["attempt"]
    assert (published, attempt["consent"]) == (False, first["consent"])  # the pointer and its consent precede the account
    if pointer == "older":
        attempt.pop("consent")  # the same pointer as an older core wrote it
    oauth_engine.save(control, saved)  # the account write landed; the reply and the slot's settling were lost
    if pointer == "late_writer":  # an accepted writer of that same attempt arrives after the write: the same receipt
        late = asyncio.run(service_sign_in._file_handover(CAL, service_sign_in._declared(CAL, "google"), control, slot_key,
                                                          attempt["generation"], account.path.stem, offer["legacy_key"],
                                                          None, oauth_engine.load(account)))
        assert (late["result"], late["consent"]) == ("imported", first["consent"])
    _sign_in(provider, owner, CAL)  # then the same person signs in again on the card: a later consent, the same origin
    [row] = host.get("/sign-in/accounts", headers=HEADERS).json()["services"][0]["accounts"]
    assert row["consent"] != first["consent"] and row["legacy_generation"] == offer["legacy_generation"]
    receipt = _read_without_the_provider(host, provider, monkeypatch, offer)
    assert receipt["connection_id"] == first["connection_id"]
    if pointer == "older":  # never given a consent it did not record
        assert receipt["result"] == "already_present" and "consent" not in receipt
    else:  # of the two consents this account was given, the transfer's own
        assert (receipt["result"], receipt["consent"]) == ("imported", first["consent"]) and receipt["consent"] != row["consent"]


def test_a_write_bound_to_a_saved_consent_must_confirm_again_after_the_persons_new_consent(world, monkeypatch):  # noqa: F811
    provider, owner, host = world
    _issued, _offer, body = _strict_offer(provider)
    saved = host.post("/sign-in/handover", headers=HEADERS, json=body).json()["grants"][0]
    assert saved["result"] == "imported" and saved["consent"] and saved["connection_id"]

    def request_token(account, connection, consent):
        return host.post("/sign-in/token", headers=HEADERS, json={
            "service": "google", "account": account, "connection_id": connection, "consent": consent})

    held = (saved["account"]["key"], saved["connection_id"], saved["consent"])
    began = request_token(*held)
    assert (began.status_code, began.json()["consent"]) == (200, saved["consent"])
    assert _sign_in(provider, owner, CAL)["ok"] is True  # the same person consents again on the card
    [row] = host.get("/sign-in/accounts", headers=HEADERS).json()["services"][0]["accounts"]
    assert (row["key"], row["connection_id"]) == held[:2] and row["consent"] != held[2]
    handle, paths = provider.handle, []
    monkeypatch.setattr(provider, "handle", lambda request: paths.append(request.url.path) or handle(request))
    stale = request_token(*held)
    assert (stale.status_code, stale.json()["code"], stale.json()["reason"]) == (409, "SIGN_IN_REQUIRED", "consent_changed")
    assert not {"access_token", "refresh_token", "client_secret"}.intersection(stale.json()) and paths == []
    current = request_token(row["key"], row["connection_id"], row["consent"])
    assert (current.status_code, current.json()["connection_id"], current.json()["consent"]) == (
        200, row["connection_id"], row["consent"])
    assert current.json()["access_token"]


def test_a_strict_offer_is_not_moved_when_the_providers_ordinary_answer_names_no_account(world, monkeypatch):  # noqa: F811
    provider, _owner, host = world
    _issued, offer, body = _strict_offer(provider, expires_at="2999-01-01T00:00:00Z")
    handle = provider.handle

    def names_no_account(request):
        if request.url.path == "/userinfo":  # a display address without a provider subject
            return httpx.Response(200, json={"email": provider.person["email"]})
        return handle(request)

    monkeypatch.setattr(provider, "handle", names_no_account)
    refused = host.post("/sign-in/handover", headers=HEADERS, json=body).json()["grants"][0]
    assert (refused["result"], refused["reason"], refused["exchange"]) == ("refused", "identity_unavailable", "none")
    assert provider.counts["token_calls"] == 0
    assert host.get("/sign-in/accounts", headers=HEADERS).json()["services"][0]["accounts"] == []
    receipt = _read_without_the_provider(host, provider, monkeypatch, offer)
    assert receipt == {**offer, "result": "refused", "reason": "identity_unavailable", "exchange": "none"}


def test_a_later_generations_refusal_before_exchange_reads_back_as_its_own(world, monkeypatch):  # noqa: F811
    provider, _owner, host = world
    _issued, offer, body = _strict_offer(provider)
    tokens = body["grants"][0]["tokens"]

    def submit(generation, handed):
        grant = {**body["grants"][0], "legacy_generation": generation, "tokens": handed}
        return host.post("/sign-in/handover", headers=HEADERS, json={**body, "grants": [grant]}).json()["grants"][0]

    unscoped = {name: value for name, value in tokens.items() if name != "scope"}  # the old sign-in's permissions are not known
    answers = [submit(generation, unscoped) for generation in ("local-offer-1", "local-offer-2")]
    assert [(answer["reason"], answer["exchange"]) for answer in answers] == [("provider_scope_unconfirmed", "none")] * 2
    later = {**offer, "legacy_generation": "local-offer-2"}
    assert _read_without_the_provider(host, provider, monkeypatch, later) == {
        **later, "result": "refused", "reason": "provider_scope_unconfirmed", "exchange": "none"}
    assert _read_without_the_provider(host, provider, monkeypatch, offer)["exchange"] == "unknown"  # the later refusal replaced it
    assert submit("local-offer-2", tokens)["result"] == "imported" and provider.counts["refresh"] == 0
    assert _read_without_the_provider(host, provider, monkeypatch, later)["result"] == "imported"


def test_a_record_of_withdrawals_that_cannot_be_read_makes_both_handover_answers_unavailable(world, monkeypatch):  # noqa: F811
    provider, _owner, host = world
    _issued, offer, body = _strict_offer(provider)
    grant = {**body["grants"][0], "tokens": {k: v for k, v in body["grants"][0]["tokens"].items() if k != "scope"}}
    first = host.post("/sign-in/handover", headers=HEADERS, json={**body, "grants": [grant]}).json()["grants"][0]
    assert (first["reason"], first["exchange"]) == ("provider_scope_unconfirmed", "none")
    record = service_sign_in._withdrawals_file(CAL).path
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text("{not json", encoding="utf-8")  # the record both answers depend on cannot be read
    answers = [host.post("/sign-in/handover", headers=HEADERS, json=request) for request in (
        {"service": "google", "receipt": offer}, {"service": "google", "grants": [grant]})]
    assert [(answer.status_code, answer.json()["reason"]) for answer in answers] == [(503, "sign_in_storage_unavailable")] * 2
    record.unlink()
    assert _read_without_the_provider(host, provider, monkeypatch, offer)["exchange"] == "none"  # nothing was guessed meanwhile
    service_sign_in._clients_file().path.write_text("{not json", encoding="utf-8")  # unreadable saved clients, no grant offered
    alone = host.post("/sign-in/handover", headers=HEADERS, json={"service": "google", "clients": body["clients"], "grants": []})
    assert (alone.status_code, alone.json()["reason"]) == (503, "sign_in_storage_unavailable")  # never 200 with refused clients


def test_two_legacy_keys_of_one_person_read_back_by_their_own_transfers(world):  # noqa: F811
    provider, _owner, host = world
    _first, _issued, body = _handover(host, provider)  # imported: the account holds this transfer's own grant

    def send(legacy):
        issued = provider._issue("personal-client", dict(provider.person), "openid email calendar.events")
        item = {"legacy_key": legacy, "label": "owner@example.com", "client_id": "personal-client", "tokens": issued}
        return host.post("/sign-in/handover", headers=HEADERS, json={"service": "google", "grants": [item]}).json()["grants"][0]

    assert send("google:second-copy")["result"] == "kept_newer"  # that person's account already holds the first one's grant
    again = send("google:second-copy")
    assert again["result"] == "kept_newer" and "consent" not in again  # by its own record, never as a transfer of that grant
    assert send(body["grants"][0]["legacy_key"])["result"] == "already_present"  # the unchanged first transfer, as before


@pytest.mark.parametrize(("bearer", "cut"), [(True, "/.well-known/"), (False, "/.well-known/"), (False, "/token")])
def test_a_strict_first_offer_cut_off_before_its_attempt_continues_after_another_withdrawal(world, monkeypatch, bearer, cut):  # noqa: F811
    provider, owner, host = world
    _save_client(owner)
    withdrawn = _sign_in(provider, owner, CAL)["account"]["key"]  # the first confirmed person, then the owner's Disconnect
    [row] = next(s for s in owner.get("/api/sign-in/status").json()["services"] if s["skill"] == CAL)["accounts"]
    assert owner.post("/api/sign-in", json={"action": "disconnect", "skill": CAL, "service": "google", "account": withdrawn,
                                            "consent": row["consent"]}).json() == {"ok": True}
    provider.person = {"sub": "sub-2", "email": "second@example.com"}  # another person's own fresh grant and registration
    issued = provider._issue("personal-client", dict(provider.person), "openid email calendar.events")
    body = {"service": "google", "clients": [{"client_id": "personal-client", "client_secret": SECRET}], "grants": [{
        "legacy_key": "second-row", "legacy_generation": "second-1", "client_id": "personal-client", "require_verified": True,
        "tokens": {**issued, "expires_at": "2999-01-01T00:00:00Z"}}]}
    if not bearer:  # an older saved grant: only its refresh token and its recorded permissions go along
        body["grants"][0]["tokens"] = {"refresh_token": issued["refresh_token"], "scope": issued["scope"]}
    handle, outage = provider.handle, [cut]  # the discovery, or the refresh that then never leaves (NOT_SENT)

    def unreachable_once(request):
        if outage and request.url.path.startswith(outage[0]):
            outage.clear()
            raise httpx.ConnectError("fixture", request=request)
        return handle(request)

    monkeypatch.setattr(provider, "handle", unreachable_once)
    first = host.post("/sign-in/handover", headers=HEADERS, json=body).json()
    held = first["grants"][0]
    assert first["clients"][0]["result"] == "imported"  # its registration, saved in that same first POST under the declaration
    assert (held["reason"], held["exchange"]) == ("provider_unreachable", "saved" if bearer else "resubmit")  # never none: that break may be its account's
    slot = oauth_engine.load(service_sign_in._grant(CAL, "google", "pending"))["handovers"][oauth_engine._digest("second-row")]
    if bearer:  # its exact bearer is kept as the attempt a recover checks
        attempt = slot["attempt"]
        assert (attempt["legacy_generation"], "refresh" in attempt, bool(attempt["bearer_offer"])) == ("second-1", False, True)
    else:  # no attempt is kept: its own recorded word reads back, and only an ordinary repeat of the same offer goes on
        assert "attempt" not in slot and _read_without_the_provider(host, provider, monkeypatch, {
            "legacy_key": "second-row", "legacy_generation": "second-1"})["exchange"] == "resubmit"
    moved = host.post("/sign-in/handover", headers=HEADERS, json={**body, "recover": bearer}).json()["grants"][0]
    assert (moved["result"], moved["account"]["key"], provider.counts["refresh"]) == (
        "imported", service_sign_in.account_key(ISSUER, "sub-2"), 0 if bearer else 1)  # its identity alone, or its one refresh


def test_a_new_sign_in_finishes_an_interrupted_withdrawal_without_waiting_for_another_account(world, monkeypatch):  # noqa: F811
    provider, owner, host = world
    _save_client(owner)
    ended = _sign_in(provider, owner, CAL)["account"]["key"]
    provider.person = {"sub": "sub-2", "email": "second@example.com"}
    other = _sign_in(provider, owner, CAL)["account"]["key"]  # an unrelated account of the same service
    grant = service_sign_in._grant(CAL, "google", ended)
    doc = oauth_engine.load(grant)
    row = service_sign_in._withdrawal_row("google", doc)
    oauth_engine.save(grant, {"account": doc["account"], "sign_in": {  # its withdrawal's first write, no tokens: nothing more
        "state": "sign_in_required", "reason": "account_disconnected", "since": row["at"], "withdrawal": row}})
    offer = {"legacy_key": "another-row", "label": "third@example.com", "client_id": "never-handed", "tokens": {
        "refresh_token": "synthetic-refresh", "access_token": "synthetic-access", "scope": "openid email calendar.events"}}
    refused = host.post("/sign-in/handover", headers=HEADERS, json={"service": "google", "grants": [offer]}).json()["grants"][0]
    assert (refused["reason"], refused["exchange"]) == ("unknown_client", "unknown")  # an unpublished break counts for v1 too
    taken, locked = [], oauth_engine.locked
    monkeypatch.setattr(oauth_engine, "locked", lambda held: taken.append(held.path.stem) or locked(held))
    provider.person = {"sub": "sub-3", "email": "third@example.com"}
    assert _sign_in(provider, owner, CAL)["state"] == "signed_in"
    assert ended in taken and other not in taken  # it finished that withdrawal and waited for no other account's lock
    rows = json.loads(service_sign_in._withdrawals_file(CAL).path.read_text(encoding="utf-8"))["accounts"]
    assert [(entry["account"], entry["subject"]) for entry in rows] == [(ended, "sub-1")] and not grant.path.exists()


@pytest.mark.parametrize("fault", ["token_503", "token_408", "token_admin_policy", "userinfo_503_then_stale"])
def test_a_moved_grant_whose_first_renewal_met_an_outage_renews_again(world, monkeypatch, fault):  # noqa: F811
    """An answer that issued no token (503, 408, a provider's refusal) is not a lost answer, and a saved renewal that
    went stale before its identity read is renewed again from what was saved: none strands a moved sign-in as
    ``handover_unconfirmed``."""
    provider, _owner, host = world
    key = _handover(host, provider)[0]["grants"][0]["account"]["key"]
    grant = service_sign_in._grant(CAL, "google", key)
    saved = oauth_engine.load(grant)
    oauth_engine.save(grant, {**saved, "tokens": {**saved["tokens"], "expires_at": 1}})
    served, outage = provider.handle, [fault]
    answers = {"token_503": (503, {"error": "temporarily_unavailable"}), "token_408": (408, {}),
               "token_admin_policy": (400, {"error": "admin_policy_enforced"}),
               "userinfo_503_then_stale": (503, {"error": "temporarily_unavailable"})}

    def once(request):
        if outage and request.url.path == ("/userinfo" if fault == "userinfo_503_then_stale" else "/token"):
            outage.clear()
            status, body = answers[fault]
            return httpx.Response(status, json=body)
        return served(request)

    monkeypatch.setattr(provider, "handle", once)
    first = host.post("/sign-in/token", headers={"x-skill-token": "cal-token"}, json={"service": "google"})
    assert first.status_code == (409 if fault == "token_admin_policy" else 503)
    if fault == "token_408":  # a request timeout is an outage, never an administrator's refusal
        assert first.json()["reason"] == oauth_engine.load(grant)["last_error"]["reason"] == "provider_unreachable"
    if fault == "userinfo_503_then_stale":  # the saved renewal outlives its access token before the next read
        renewed = oauth_engine.load(grant)
        assert renewed["origin"]["first_refresh"] == "renewed"
        oauth_engine.save(grant, {**renewed, "tokens": {**renewed["tokens"], "expires_at": 1}})
    second = host.post("/sign-in/token", headers={"x-skill-token": "cal-token"}, json={"service": "google"})
    assert second.status_code == 200, second.text
    assert "first_refresh" not in oauth_engine.load(grant)["origin"]


@pytest.mark.parametrize("loss", ["read_timeout", "gateway_504", "bare_503", "gateway_json_503"])
def test_a_moved_grants_first_renewal_whose_answer_was_lost_is_never_repeated(world, monkeypatch, loss):  # noqa: F811
    """A timeout after sending, or a 504 or a bare 503 a gateway may give after the exchange happened (and a provider
    that rotates refresh tokens may have rotated it): never repeated, so a working sign-in is not spent twice."""
    provider, _owner, host = world
    key = _handover(host, provider)[0]["grants"][0]["account"]["key"]
    grant = service_sign_in._grant(CAL, "google", key)
    saved = oauth_engine.load(grant)
    oauth_engine.save(grant, {**saved, "tokens": {**saved["tokens"], "expires_at": 1}})
    served = provider.handle

    def lost(request):
        if request.url.path == "/token":
            if loss in ("gateway_504", "bare_503"):  # no OAuth error of the provider's own: whoever answered
                return httpx.Response(504 if loss == "gateway_504" else 503)
            if loss == "gateway_json_503":  # a web stack's own error object, not the OAuth code
                return httpx.Response(503, json={"error": "Service Unavailable", "status": 503})
            raise httpx.ReadTimeout("the answer never came back", request=request)
        return served(request)

    monkeypatch.setattr(provider, "handle", lost)
    assert host.post("/sign-in/token", headers={"x-skill-token": "cal-token"}, json={"service": "google"}).status_code == 503
    monkeypatch.setattr(provider, "handle", served)
    again = host.post("/sign-in/token", headers={"x-skill-token": "cal-token"}, json={"service": "google"})
    assert again.status_code == 409 and again.json()["reason"] == "handover_unconfirmed"
