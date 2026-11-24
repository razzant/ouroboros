"""A skill hands the host the sign-ins it already held (``service_sign_in.handover``, ``POST /sign-in/handover``).

The account is the one the provider names for the handed token (a claim that differs is refused); a token
without the identity scope is kept unconfirmed under its skill's own key; a token the provider refuses is
kept as needing sign-in; a repeat answers ``already_present`` and a grant the owner made on the card wins.
A transfer resumes after a lost identity answer without a second refresh, a single-account service's
confirmed sign-in replaces its unconfirmed moved row, and a moved grant's first refresh reaches the skill
once the provider names the same account, due again only when its request never left. A refused grant says what this
host knows of its token (``exchange``), ``recover`` answers from the slot first and starts nothing, an exchange that
left is never displaced, a receipt keeps its transfer's connection and each owner's break reaches the skill. No answer,
log line or printed text carries a token or a secret.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace

import httpx
import pytest

from tests.test_service_sign_in_loopback import CAL, ISSUER, _sign_in, world  # noqa: F401 -- the fixture

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
