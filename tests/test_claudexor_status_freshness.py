"""The Claudexor status read: one fetch in flight per variant, each facet its own freshness.

Concurrent status requests share the read already in flight instead of fanning the
daemon out again, an owner wake reads anew, and every facet keeps its own last answered
value: a failed or unasked facet serves it again as stale, under its own observation
time and this read's error, while the facets that answered stay as read.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future

import httpx

import pytest

from ouroboros.gateway import claudexor_accounts as accounts


@pytest.fixture(autouse=True)
def _fresh_status_state(monkeypatch):
    monkeypatch.setattr(accounts, "_STATUS_IN_FLIGHT", {}, raising=False)
    monkeypatch.setattr(accounts, "_FACET_MEMORY", {}, raising=False)


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
