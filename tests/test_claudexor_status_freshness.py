"""The Claudexor status read: one fetch in flight per variant, each facet its own freshness.

Concurrent status requests share the read already in flight instead of fanning the
daemon out again, an owner wake reads anew, and every facet keeps its own last answered
value: a failed or unasked facet serves it again as stale, under its own observation
time and this read's error, while the facets that answered stay as read.
"""

from __future__ import annotations

import threading
import asyncio
import json
from concurrent.futures import Future

import httpx

import pytest

from ouroboros.gateway import claudexor_accounts as accounts


@pytest.fixture(autouse=True)
def _fresh_status_state(monkeypatch):
    monkeypatch.setattr(accounts, "_STATUS_IN_FLIGHT", {}, raising=False)
    monkeypatch.setattr(accounts, "_FACET_MEMORY", {}, raising=False)
    monkeypatch.setattr(accounts, "_ACCOUNTS_REVISION", 0)


def _payload(*, home="/home-a", reads=None, errors=None, harnesses=("codex",), accounts_value="a1",
             quota=("q1",)):
    reads = {"catalog": "ok", "accounts": "ok", "quota": "ok", **(reads or {})}
    body = {
        "daemon": {"state": "running"}, "config_dir": home, "reads": reads,
        "harnesses": [{"id": name} for name in harnesses] if reads["catalog"] == "ok" else [],
        "profiles": {"profiles": [accounts_value]} if reads["accounts"] == "ok" else {},
        "quota": [{"id": row} for row in quota] if reads["quota"] == "ok" else [],
        "quota_absences": [],
    }
    if errors:
        body["_facet_errors"] = dict(errors)
    return body


def test_concurrent_requests_share_one_read_and_get_their_own_copy(monkeypatch):
    entered, release, reads, waiting = threading.Event(), threading.Event(), [], []

    def reader(include_models):
        reads.append(include_models)
        entered.set()
        release.wait(10)
        return _payload()

    class CountingFuture(Future):
        def result(self, timeout=None):
            if not self.done():
                waiting.append(1)
            return super().result(timeout)

    monkeypatch.setattr(accounts, "_read_status_payload", reader)
    monkeypatch.setattr(accounts, "Future", CountingFuture)
    results = []
    first = threading.Thread(target=lambda: results.append(accounts._status_payload(False)))
    first.start()
    assert entered.wait(10)
    joiners = [threading.Thread(target=lambda: results.append(accounts._status_payload(False))) for _ in range(3)]
    for thread in joiners:
        thread.start()
    for _ in range(1000):
        if len(waiting) == 3:
            break
        threading.Event().wait(0.01)
    assert len(waiting) == 3, "every later request joined the read in flight"
    release.set()
    for thread in (first, *joiners):
        thread.join(10)
    assert reads == [False], "one daemon fan-out served four requests"
    assert len(results) == 4 and all(result == results[0] for result in results)
    assert len({id(result) for result in results}) == 4, "each caller owns its copy"
    assert accounts._STATUS_IN_FLIGHT == {}, "a finished read is never joined later"


def test_an_owner_wake_and_another_variant_read_anew(monkeypatch):
    entered, release, reads = threading.Event(), threading.Event(), []

    def reader(include_models):
        reads.append(include_models)
        if len(reads) == 1:
            entered.set()
            release.wait(10)
        return _payload()

    monkeypatch.setattr(accounts, "_read_status_payload", reader)
    background = threading.Thread(target=lambda: accounts._status_payload(False))
    background.start()
    assert entered.wait(10)
    accounts._status_payload(True)  # the models variant is its own read
    accounts._status_payload(False, join=False)  # an owner wake reads after its own start
    release.set()
    background.join(10)
    assert sorted(reads) == [False, False, True]


def test_a_failed_facet_serves_its_last_read_without_refreshing_its_time(monkeypatch):
    clock = iter(["2026-10-10T10:00:00Z", "2026-10-10T10:05:00Z", "2026-10-10T10:10:00Z"])
    monkeypatch.setattr("ouroboros.utils.utc_now_iso", lambda: next(clock))
    script = iter([
        _payload(harnesses=("codex",), accounts_value="a1", quota=("q1",)),
        _payload(reads={"accounts": "failed"}, errors={"accounts": "daemon_busy"}, harnesses=("codex", "claude")),
        _payload(reads={"accounts": "failed", "quota": "failed"},
                 errors={"accounts": "daemon_busy", "quota": "malformed_response"}),
    ])
    monkeypatch.setattr(accounts, "_read_status_payload", lambda include_models: next(script))

    first = accounts._status_payload(False)
    assert first["facets"]["accounts"] == {"observed_at": "2026-10-10T10:00:00Z", "stale": False, "error": None}
    second = accounts._status_payload(False)
    # The refused facet is not blanked: its last read returns, stale, at its own time.
    assert second["profiles"] == {"profiles": ["a1"]} and second["reads"]["accounts"] == "failed"
    assert second["facets"]["accounts"] == {"observed_at": "2026-10-10T10:00:00Z", "stale": True,
                                            "error": "daemon_busy"}
    # One facet's error leaves the others as read, fresh, at this read's time.
    assert [row["id"] for row in second["harnesses"]] == ["codex", "claude"]
    assert second["facets"]["catalog"] == {"observed_at": "2026-10-10T10:05:00Z", "stale": False, "error": None}
    assert second["daemon"]["state"] == "running" and "_facet_errors" not in second
    third = accounts._status_payload(False)
    assert third["facets"]["accounts"]["observed_at"] == "2026-10-10T10:00:00Z", "stale never refreshes its time"
    assert third["quota"] == [{"id": "q1"}] and third["facets"]["quota"] == {
        "observed_at": "2026-10-10T10:05:00Z", "stale": True, "error": "malformed_response"}


def test_a_facet_never_read_stays_empty_and_homes_never_share_memory(monkeypatch):
    script = iter([
        _payload(home="/home-a"),
        _payload(home="/home-b", reads={"accounts": "not_read"}, errors={"accounts": "daemon_unreachable"}),
    ])
    monkeypatch.setattr(accounts, "_read_status_payload", lambda include_models: next(script))
    accounts._status_payload(False)
    other_home = accounts._status_payload(False)
    assert other_home["profiles"] == {}, "another daemon home never inherits this one's accounts"
    assert other_home["facets"]["accounts"] == {"observed_at": None, "stale": False, "error": "daemon_unreachable"}


def test_the_catalog_is_remembered_per_variant_and_the_accounts_are_shared(monkeypatch):
    script = iter([
        _payload(harnesses=("codex",), accounts_value="a1"),  # a read without per-harness models
        _payload(reads={"catalog": "failed", "accounts": "failed"},
                 errors={"catalog": "daemon_busy", "accounts": "daemon_busy"}),  # the models variant
    ])
    monkeypatch.setattr(accounts, "_read_status_payload", lambda include_models: next(script))
    accounts._status_payload(False)
    with_models = accounts._status_payload(True)
    assert with_models["harnesses"] == [] and with_models["facets"]["catalog"] == {
        "observed_at": None, "stale": False, "error": "daemon_busy"}, "rows without models never stand in"
    assert with_models["profiles"] == {"profiles": ["a1"]} and with_models["facets"]["accounts"]["stale"] is True


def _named_payload(*, enabled=True, **kwargs):
    payload = _payload(**kwargs)
    payload["profiles"] = {"profiles": [{"profile": {
        "harness_id": "codex", "profile_id": "work", "enabled": enabled}}],
        "accountPools": [{"harness_id": "codex", "next_up": {"profileId": "work"}}],
        "routing_evidence": {"old": True}}
    payload["unified_accounts"] = True
    return payload


@pytest.mark.parametrize("deleted", [False, True])
def test_a_late_full_status_joiner_cannot_return_accounts_from_before_confirmation(monkeypatch, tmp_path, deleted):
    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateways import claudexor as wire

    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path)
    entered, release, consumed, deliver = [threading.Event() for _ in range(4)]

    class DelayedConsumer(Future):
        def result(self, timeout=None):
            value = super().result(timeout)
            if threading.current_thread().name == "delayed-status-joiner":
                consumed.set()
                assert deliver.wait(10)
            return value

    def reader(_models):
        entered.set()
        assert release.wait(10)
        return _named_payload(home=str(tmp_path))

    class Gateway:
        def __init__(self, _endpoint): pass
        def __enter__(self): return self
        def __exit__(self, *_exc): pass
        def handshake(self): pass
        def update_credential_profile(self, *_args, **_kw): return {"ok": True}
        def delete_credential_profile(self, *_args): return {"removed": True}

    monkeypatch.setattr(wire, "discover_daemon_at", lambda _home: object())
    monkeypatch.setattr(wire, "ClaudexorGateway", Gateway)
    monkeypatch.setattr(accounts, "Future", DelayedConsumer)
    monkeypatch.setattr(accounts, "_read_status_payload", reader)
    answers = []
    owner = threading.Thread(target=lambda: accounts._status_payload(False))
    joiner = threading.Thread(name="delayed-status-joiner",
                              target=lambda: answers.append(accounts._status_payload(False)))
    owner.start()
    try:
        assert entered.wait(10)
        joiner.start()
        release.set()
        assert consumed.wait(10)
        if deleted:
            accounts._remove_credential_profile("codex", "work")
        else:
            accounts._update_credential_profile("codex", "work", False)
    finally:
        release.set()
        deliver.set()
        owner.join(10)
        if joiner.ident:
            joiner.join(10)
    answer = answers[0]
    assert answer["reads"]["accounts"] == "not_read"
    assert answer["unified_accounts"] is False
    assert not any(row["profile"]["enabled"] for row in answer["profiles"].get("profiles", []))
    assert "account_revision" not in answer and "accountRevision" not in answer


def test_an_older_full_variant_does_not_replace_newer_shared_facets(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def reader(models):
        if not models:
            entered.set()
            assert release.wait(10)
        payload = _named_payload(enabled=not models, quota=("new" if models else "old",))
        if not models:
            payload.update(resources=[{"obsolete": True}], availability_scope={"obsolete": True})
        return payload

    monkeypatch.setattr(accounts, "_read_status_payload", reader)
    answers = []
    old = threading.Thread(target=lambda: answers.append(accounts._status_payload(False)))
    old.start()
    try:
        assert entered.wait(10)
        newest = accounts._status_payload(True)
    finally:
        release.set()
        old.join(10)
    assert answers[0]["profiles"] == newest["profiles"]
    assert answers[0]["quota"] == [{"id": "new"}]
    assert answers[0]["reads"] == {"catalog": "ok", "accounts": "not_read", "quota": "not_read"}
    assert answers[0]["facets"]["quota"]["observed_at"] == newest["facets"]["quota"]["observed_at"]
    assert "resources" not in answers[0] and "availability_scope" not in answers[0]


def test_full_facets_keep_their_individual_read_times(monkeypatch):
    monkeypatch.setattr("ouroboros.utils.utc_now_iso", lambda: "2026-11-13T19:00:30Z")
    payload = _payload()
    payload["_facet_observed_at"] = {
        "accounts": "2026-11-13T19:00:01Z", "quota": "2026-11-13T19:00:02Z",
        "catalog": "2026-11-13T19:00:29Z"}
    answer = accounts._with_facet_memory(payload)
    assert {facet: info["observed_at"] for facet, info in answer["facets"].items()} == {
        "accounts": "2026-11-13T19:00:01Z", "quota": "2026-11-13T19:00:02Z",
        "catalog": "2026-11-13T19:00:29Z"}
    assert "_facet_observed_at" not in answer


def test_manifest_policy_is_ordered_independently_from_retained_membership():
    first = _named_payload()
    first["profiles"]["login_capable_harnesses"] = ["codex", "other"]
    initial = accounts._with_facet_memory(first, read_sequence=1)
    failed = _payload(reads={"accounts": "failed"})
    failed["profiles"] = {"login_capable_harnesses": ["codex"]}
    current = accounts._with_facet_memory(failed, read_sequence=3)
    assert current["profiles"]["profiles"] == initial["profiles"]["profiles"]
    assert current["facets"]["accounts"]["observed_at"] == initial["facets"]["accounts"]["observed_at"]
    assert current["reads"]["accounts"] == "failed"
    assert current["profiles"]["login_capable_harnesses"] == ["codex"]
    # An older successful roster may fill membership, but cannot restore old policy.
    older = _named_payload(enabled=False)
    older["profiles"]["login_capable_harnesses"] = ["codex", "other"]
    answer = accounts._with_facet_memory(older, read_sequence=2)
    assert answer["profiles"]["login_capable_harnesses"] == ["codex"]
    assert answer["profiles"]["profiles"][0]["profile"]["enabled"] is False
    revision = accounts.account_read_revision()
    accounts._account_changed("codex", "work", False, home="/home-a")
    fenced = accounts.coherent_accounts(answer, revision)
    assert fenced["profiles"]["login_capable_harnesses"] == ["codex"]


def test_account_mutation_rpc_and_memory_confirmation_have_one_order(monkeypatch, tmp_path):
    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateways import claudexor as wire

    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path)
    accounts._with_facet_memory(_named_payload(home=str(tmp_path)))
    entered, release, second_entered = [threading.Event() for _ in range(3)]
    engine = {"enabled": True}

    class Gateway:
        def __init__(self, _endpoint): pass
        def __enter__(self): return self
        def __exit__(self, *_exc): pass
        def handshake(self): pass
        def update_credential_profile(self, _harness, _profile, *, enabled):
            engine["enabled"] = enabled
            if not enabled:
                entered.set()
                assert release.wait(10)
            else:
                second_entered.set()

    monkeypatch.setattr(wire, "discover_daemon_at", lambda _home: object())
    monkeypatch.setattr(wire, "ClaudexorGateway", Gateway)
    first = threading.Thread(target=lambda: accounts._update_credential_profile("codex", "work", False))
    second = threading.Thread(target=lambda: accounts._update_credential_profile("codex", "work", True))
    first.start()
    try:
        assert entered.wait(10)
        second.start()
        # A second RPC may not overtake confirmation of the first mutation.
        assert not second_entered.wait(0.1)
    finally:
        release.set()
        first.join(10)
        if second.ident:
            second.join(10)
    held = accounts._FACET_MEMORY[(str(tmp_path), "accounts", False)]["value"]["profiles"]
    assert engine["enabled"] is True and held["profiles"][0]["profile"]["enabled"] is True
    assert held["accountPools"] == [] and "routing_evidence" not in held


@pytest.mark.parametrize("operation,view", [
    ("status", b""), ("status", b"view=quota"), ("wake", b""), ("restart", b"")])
def test_handler_fences_a_confirmation_while_resuming_the_thread_result(monkeypatch, tmp_path, operation, view):
    from starlette.requests import Request
    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateway import claudexor_passive

    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path)
    from types import SimpleNamespace
    from ouroboros import claudexor_restart

    monkeypatch.setattr(owned, "get_owned_daemon", lambda: SimpleNamespace(clear_start_failure_latch=lambda **_: None))
    monkeypatch.setattr(owned, "ensure_owned_gateway", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(claudexor_restart, "restart_engine", lambda _token: {"outcome": "restarted"})
    monkeypatch.setattr(accounts, "_read_status_payload", lambda _models: _named_payload(home=str(tmp_path)))
    monkeypatch.setattr(claudexor_passive, "_quota_payload", lambda: _named_payload(home=str(tmp_path)))

    async def resumed_after_confirmation(function, *args):
        payload = function(*args)  # worker completed with pre-mutation accounts
        accounts._account_changed("codex", "work", False)
        return payload  # HTTP coroutine resumes after the other client's write

    monkeypatch.setattr(accounts.asyncio, "to_thread", resumed_after_confirmation)
    async def receive():
        return {"type": "http.request", "body": b'{"token":"test-preview"}', "more_body": False}

    handler = getattr(accounts, "api_claudexor_" + operation)
    response = asyncio.run(handler(Request({
        "type": "http", "method": "GET" if operation == "status" else "POST",
        "path": "/api/claudexor/" + operation, "query_string": view, "headers": []}, receive)))
    payload = json.loads(response.body)
    if operation == "restart":
        assert payload["restart"] == {"outcome": "restarted"}
        payload = payload["status"]
    assert response.status_code == 200 and payload["reads"]["accounts"] == "not_read"
    assert not any(row["profile"]["enabled"] for row in payload["profiles"].get("profiles", []))


@pytest.mark.parametrize("transport,code", [
    (httpx.ReadTimeout, "daemon_not_answering"),  # accepted the connection, never answered
    (httpx.ConnectError, "daemon_starting"),  # nothing listening yet: a startup to join
])
def test_a_live_daemon_that_accepts_and_stays_silent_is_not_answering(monkeypatch, tmp_path, transport, code):
    from ouroboros.gateways import claudexor as gateway_mod
    from tests.test_claudexor_startup_failure import _OOM_REACHED, _Stand

    stand = _Stand(monkeypatch, tmp_path, returncode=None, banner=_OOM_REACHED)
    descriptor = stand.config_dir / "daemon" / "control-api.json"  # it served: control was published
    descriptor.parent.mkdir(parents=True, exist_ok=True)
    descriptor.write_text('{"host": "127.0.0.1", "port": 45690, "tokenPath": "token"}', encoding="utf-8")

    class Silent:
        def __init__(self, _endpoint): pass
        def __enter__(self): return self
        def __exit__(self, *_exc): pass
        def handshake(self, *, timeout_sec=None):  # the real transport keeps the httpx class as the cause
            raise gateway_mod.ClaudexorUnavailable("daemon_unreachable", "no answer") from transport("fixture")

    monkeypatch.setattr(gateway_mod, "ClaudexorGateway", Silent)
    monkeypatch.setattr(stand.manager, "_startup_pids", lambda: {4242})  # a daemon already in custody
    error = stand.fail_once()
    assert (error.code, error.status_code) == (code, 503)
    assert stand.manager._last_error.startswith(f"{code}: ")
    assert ("retry joins the same daemon" if code == "daemon_not_answering"
            else "retry joins the same startup") in str(error)
    assert stand.spawned == [], "a live daemon is joined, never respawned"
