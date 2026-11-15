"""Thin foreground quota transport regressions."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event

import httpx
import pytest
from starlette.requests import Request

from ouroboros.gateway.endpoint_index import HTTP_ENDPOINTS
from ouroboros.gateway.router import collect_routes
from ouroboros.gateways.claudexor import (
    ClaudexorGateway,
    ClaudexorUnavailable,
    DaemonEndpoint,
)


def _envelope() -> dict:
    return {
        "snapshots": [{
            "subject": {
                "harness": "claude",
                "credential_route": "local",
                "plan_label": "Max",
                "subject_id": "mironov",
            },
            "freshness": "fresh",
            "source": "claude_oauth_usage",
            "observed_at": "2026-09-01T11:59:00Z",
            "constraints": [{
                "id": "weekly_scoped:fable",
                "label": "Fable weekly",
                "applies_to_models": ["fable"],
                "used_ratio": 0.83,
                "window_seconds": 604_800,
                "resets_at": "2026-09-02T12:00:00Z",
                "cooldown_until": None,
            }],
        }],
        "absences": [{
            "subject": {
                "harness": "agy",
                "credential_route": "local",
                "plan_label": None,
                "subject_id": "sdventures",
            },
            "reason": "rate_limited",
            "detail": None,
            "observed_at": "2026-09-01T12:00:00Z",
            "retry_after_ms": 60_000,
        }],
        "refreshed_at": "2026-09-01T12:00:00Z",
        "refresh_skipped": [{
            "vendor": "claude",
            "not_before": "2026-09-01T12:01:00Z",
        }],
    }


def _request() -> Request:
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request({
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/claudexor/quota/refresh",
        "raw_path": b"/api/claudexor/quota/refresh",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 8765),
    }, receive)


@pytest.fixture(autouse=True)
def _fresh_refresh_state(monkeypatch):
    from ouroboros.gateway import claudexor_quota

    monkeypatch.setattr(claudexor_quota, "_REFRESH_FLIGHTS", {})


def _joined_future(monkeypatch, quota):
    """Future whose result() reports a caller that had to wait for an unfinished attempt."""
    joined = Event()

    class JoinedFuture(Future):
        def result(self, timeout=None):
            if not self.done():
                joined.set()
            return super().result(timeout)

    monkeypatch.setattr(quota, "Future", JoinedFuture)
    return joined


def test_shared_refresh_joins_only_an_identical_attempt_in_flight(monkeypatch, tmp_path):
    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateway import claudexor_quota as quota

    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path)
    joined = _joined_future(monkeypatch, quota)
    entered, release = Event(), Event()
    calls = []

    def refresh(target=None):
        calls.append(target)
        entered.set()
        assert release.wait(10)
        return _envelope()

    monkeypatch.setattr(quota, "_refresh_quota", refresh)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(quota._shared_refresh)
        try:
            assert entered.wait(10)
            second = pool.submit(quota._shared_refresh)
            assert joined.wait(10)
        finally:
            release.set()
        answers = [call.result(10) for call in (first, second)]
    assert calls == [None]
    attempts = [answer.pop("refresh_attempt") for answer in answers]
    assert [item["reused"] for item in attempts] == [False, True]
    assert attempts[0]["started_at"] == attempts[1]["started_at"] <= attempts[0]["completed_at"]
    assert attempts[0]["completed_at"] == attempts[1]["completed_at"]
    answers[0]["snapshots"].clear()
    assert answers[1] == _envelope()
    # A completed attempt is never served again: the next owner request asks the engine anew.
    fresh = quota._shared_refresh()
    assert fresh.pop("refresh_attempt")["reused"] is False
    assert fresh == _envelope() and calls == [None, None] and quota._REFRESH_FLIGHTS == {}


def test_a_failed_refresh_is_neither_cached_nor_retried(monkeypatch, tmp_path):
    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateway import claudexor_quota as quota

    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path)
    calls = []

    def refuse(target=None):
        calls.append(target)
        raise ClaudexorUnavailable("quota_refresh_paced", "paced", status_code=429)

    monkeypatch.setattr(quota, "_refresh_quota", refuse)
    target = {"harness": "codex", "profile_id": "a"}
    for _ in range(2):
        with pytest.raises(ClaudexorUnavailable, match="paced"):
            quota._shared_refresh(target)
    # Each owner request reaches the engine once; its pacing answer is not replayed here.
    assert calls == [target, target] and quota._REFRESH_FLIGHTS == {}


def test_a_refresh_is_retired_before_its_outcome_is_published(monkeypatch, tmp_path):
    """No identical request may join a completed attempt: the flight leaves the registry first."""
    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateway import claudexor_quota as quota

    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path)
    registered_at_publish = []

    class WatchedFuture(Future):
        def set_result(self, result):
            registered_at_publish.append(self in quota._REFRESH_FLIGHTS.values())
            super().set_result(result)

        def set_exception(self, exception):
            registered_at_publish.append(self in quota._REFRESH_FLIGHTS.values())
            super().set_exception(exception)

    monkeypatch.setattr(quota, "Future", WatchedFuture)
    monkeypatch.setattr(quota, "_refresh_quota", lambda target=None: _envelope())
    assert quota._shared_refresh()["refresh_attempt"]["reused"] is False

    def refuse(target=None):
        raise ClaudexorUnavailable("quota_refresh_paced", "paced", status_code=429)

    monkeypatch.setattr(quota, "_refresh_quota", refuse)
    with pytest.raises(ClaudexorUnavailable):
        quota._shared_refresh()
    assert registered_at_publish == [False, False]
    assert quota._REFRESH_FLIGHTS == {}


def test_a_retiring_refresh_never_removes_a_newer_flight(monkeypatch, tmp_path):
    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateway import claudexor_quota as quota

    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path)
    newer = Future()

    def refresh(target=None):
        # Model a newer identical flight registered under the same key before this one retires.
        key = next(iter(quota._REFRESH_FLIGHTS))
        quota._REFRESH_FLIGHTS[key] = newer
        return _envelope()

    monkeypatch.setattr(quota, "_refresh_quota", refresh)
    quota._shared_refresh()
    assert list(quota._REFRESH_FLIGHTS.values()) == [newer]


def test_refresh_scopes_join_only_the_same_home_and_exact_target(monkeypatch, tmp_path):
    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateway import claudexor_quota as quota

    home, calls = [tmp_path / "a"], []
    monkeypatch.setattr(owned, "owned_config_dir", lambda: home[0])
    joined = _joined_future(monkeypatch, quota)
    entered, release = Event(), Event()
    a = {"harness": "codex", "profile_id": "a"}

    def refresh(target=None):
        calls.append((str(home[0]), target))
        if len(calls) == 1:
            entered.set()
            assert release.wait(10)
        return {"snapshots": [], "target": target}

    monkeypatch.setattr(quota, "_refresh_quota", refresh)
    with ThreadPoolExecutor(max_workers=2) as pool:
        held = pool.submit(quota._shared_refresh, a)
        try:
            assert entered.wait(10)
            # Full, another account and another home never join the held exact attempt.
            assert quota._shared_refresh()["target"] is None
            assert quota._shared_refresh({"harness": "codex", "profile_id": "b"})["target"]["profile_id"] == "b"
            home[0] = tmp_path / "b"
            assert quota._shared_refresh(a)["refresh_attempt"]["reused"] is False
            home[0] = tmp_path / "a"
            same = pool.submit(quota._shared_refresh, {"profile_id": "a", "harness": "codex"})
            assert joined.wait(10)
        finally:
            release.set()
        first, second = held.result(10), same.result(10)
    assert first["refresh_attempt"]["reused"] is False and second["refresh_attempt"]["reused"] is True
    assert calls == [(str(tmp_path / "a"), a), (str(tmp_path / "a"), None),
                     (str(tmp_path / "a"), {"harness": "codex", "profile_id": "b"}), (str(tmp_path / "b"), a)]


def test_gateway_refresh_quota_posts_once_and_keeps_token_host_side(caplog, monkeypatch):
    requests: list[tuple[str, str, dict, str, float]] = []
    token = "host-only-test-token"
    envelope = _envelope()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((
            request.method,
            request.url.path,
            json.loads(request.content),
            request.headers["Authorization"],
            request.extensions["timeout"]["read"],
        ))
        return httpx.Response(200, json=envelope)

    with ClaudexorGateway(DaemonEndpoint("127.0.0.1", 1, token)) as gateway:
        gateway._client.close()
        gateway._client = httpx.Client(
            base_url="http://127.0.0.1:1",
            transport=httpx.MockTransport(handler),
            headers={"Authorization": f"Bearer {token}"},
        )
        answer = gateway.refresh_quota()

    assert answer == envelope
    assert requests == [("POST", "/v2/quota", {}, f"Bearer {token}", 90.0)]
    assert token not in json.dumps(answer)
    assert token not in caplog.text


def test_gateway_quota_timeout_does_not_widen_handshake_or_cached_get(monkeypatch):
    monkeypatch.delenv("OUROBOROS_CLAUDEXOR_QUOTA_REFRESH_TIMEOUT_SEC", raising=False)
    observed: list[tuple[str, str, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append((
            request.method,
            request.url.path,
            request.extensions["timeout"]["read"],
        ))
        if request.url.path == "/v2/handshake":
            return httpx.Response(200, json={
                "compatible": True,
                "protocolMajor": 3,
                "engine": {"version": "3.9.4", "sha": "test"},
            })
        return httpx.Response(200, json=_envelope())

    with ClaudexorGateway(DaemonEndpoint("127.0.0.1", 1, "host-token")) as gateway:
        gateway._client.close()
        gateway._client = httpx.Client(
            base_url="http://127.0.0.1:1",
            transport=httpx.MockTransport(handler),
            timeout=httpx.Timeout(60.0, connect=5.0),
        )
        gateway.handshake()
        gateway.quota_state()
        gateway.refresh_quota()

    assert observed == [
        ("POST", "/v2/handshake", 60.0),
        ("GET", "/v2/quota", 60.0),
        ("POST", "/v2/quota", 90.0),
    ]


def test_quota_refresh_timeout_setting_can_only_tighten_outer_invariant(monkeypatch):
    from ouroboros.config import get_claudexor_quota_refresh_timeout_sec

    monkeypatch.setenv("OUROBOROS_CLAUDEXOR_QUOTA_REFRESH_TIMEOUT_SEC", "123")
    assert get_claudexor_quota_refresh_timeout_sec() == 90
    monkeypatch.setenv("OUROBOROS_CLAUDEXOR_QUOTA_REFRESH_TIMEOUT_SEC", "45")
    assert get_claudexor_quota_refresh_timeout_sec() == 45


def test_gateway_refresh_quota_preserves_transport_and_upstream_failures():
    endpoint = DaemonEndpoint("127.0.0.1", 1, "host-token")

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("quota read timed out", request=request)

    with ClaudexorGateway(endpoint) as gateway:
        gateway._client.close()
        gateway._client = httpx.Client(
            base_url="http://127.0.0.1:1",
            transport=httpx.MockTransport(timeout),
        )
        with pytest.raises(ClaudexorUnavailable) as unavailable:
            gateway.refresh_quota()
    assert unavailable.value.code == "daemon_unreachable"

    def refused(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={
            "code": "quota_refresh_paced",
            "message": "try later",
        })

    with ClaudexorGateway(endpoint) as gateway:
        gateway._client.close()
        gateway._client = httpx.Client(
            base_url="http://127.0.0.1:1",
            transport=httpx.MockTransport(refused),
        )
        with pytest.raises(ClaudexorUnavailable) as upstream:
            gateway.refresh_quota()
    assert upstream.value.code == "quota_refresh_paced"
    assert upstream.value.status_code == 429


def test_owned_refresh_handshakes_then_calls_foreground_quota_once(monkeypatch, tmp_path):
    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateway.claudexor_quota import _refresh_quota
    from ouroboros.gateways import claudexor as gateway_module

    envelope = _envelope()
    calls: list[str] = []
    endpoint = object()

    class Gateway:
        def __init__(self, actual_endpoint):
            assert actual_endpoint is endpoint

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            calls.append("close")

        def handshake(self):
            calls.append("handshake")
            return {"compatible": True}

        def operations(self):
            calls.append("operations")
            return []

        def refresh_quota(self):
            calls.append("refresh_quota")
            return envelope

    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "owned")
    monkeypatch.setattr(
        gateway_module,
        "discover_daemon_at",
        lambda path: endpoint if path == tmp_path / "owned" else None,
    )
    monkeypatch.setattr(gateway_module, "ClaudexorGateway", Gateway)

    assert _refresh_quota() is envelope
    assert calls == ["handshake", "operations", "refresh_quota", "close"]


def test_owned_refresh_stops_on_protocol_failure(monkeypatch, tmp_path):
    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateway.claudexor_quota import _refresh_quota
    from ouroboros.gateways import claudexor as gateway_module

    class Gateway:
        def __init__(self, _endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def handshake(self):
            raise ClaudexorUnavailable("protocol_incompatible", "wrong protocol")

        def refresh_quota(self):  # pragma: no cover - handshake must gate the POST
            raise AssertionError("foreground refresh ran after failed handshake")

    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "owned")
    monkeypatch.setattr(gateway_module, "discover_daemon_at", lambda _path: object())
    monkeypatch.setattr(gateway_module, "ClaudexorGateway", Gateway)

    with pytest.raises(ClaudexorUnavailable) as failure:
        _refresh_quota()
    assert failure.value.code == "protocol_incompatible"


def test_inbound_refresh_returns_exact_foreground_envelope_via_thread(monkeypatch):
    from ouroboros.gateway import claudexor_quota

    envelope = _envelope()
    calls: list[tuple] = []

    def refresh():
        calls.append(("refresh",))
        return envelope

    async def to_thread(function, *args, **kwargs):
        calls.append(("to_thread", function, args, kwargs))
        return function(*args, **kwargs)

    monkeypatch.setattr(claudexor_quota, "_refresh_quota", refresh)
    monkeypatch.setattr(claudexor_quota.asyncio, "to_thread", to_thread)

    response = asyncio.run(claudexor_quota.api_claudexor_quota_refresh(_request()))
    assert response.status_code == 200
    actual = json.loads(response.body)
    attempt = actual.pop("refresh_attempt")
    assert attempt["reused"] is False
    assert attempt["started_at"] <= attempt["completed_at"]
    assert actual == envelope
    assert calls == [("to_thread", claudexor_quota._shared_refresh, (), {}), ("refresh",)]


@pytest.mark.parametrize("code", [
    "daemon_not_discovered",
    "protocol_incompatible",
    "rate_limited",
])
def test_inbound_refresh_preserves_typed_engine_failure(monkeypatch, code):
    from ouroboros.gateway import claudexor_quota

    def refuse():
        raise ClaudexorUnavailable(code, "typed upstream refusal", status_code=429)

    monkeypatch.setattr(claudexor_quota, "_refresh_quota", refuse)
    response = asyncio.run(claudexor_quota.api_claudexor_quota_refresh(_request()))
    assert response.status_code == 429
    assert json.loads(response.body) == {"error": "typed upstream refusal", "code": code,
                                        "required_actions": []}


@pytest.mark.parametrize("answer", [None, [], "text"])
def test_inbound_refresh_types_a_malformed_engine_answer_instead_of_blaming_the_request(monkeypatch, answer):
    from ouroboros.gateway import claudexor_quota

    monkeypatch.setattr(claudexor_quota, "_refresh_quota", lambda: answer)
    response = asyncio.run(claudexor_quota.api_claudexor_quota_refresh(_request()))
    assert response.status_code == 503, "a 2xx engine answer without an object is not the owner's bad request"
    body = json.loads(response.body)
    assert body["code"] == "malformed_response" and body["required_actions"] == []
    assert "Invalid JSON" not in body["error"]
    assert claudexor_quota._REFRESH_FLIGHTS == {}, "the failed attempt is retired, never joined"


def test_quota_refresh_route_is_post_only_and_indexed(tmp_path):
    matching = [
        route for route in collect_routes(data_dir=tmp_path)
        if getattr(route, "path", "") == "/api/claudexor/quota/refresh"
    ]
    assert len(matching) == 1
    assert matching[0].methods == {"POST"}
    assert "POST /api/claudexor/quota/refresh" in HTTP_ENDPOINTS
    assert "GET /api/claudexor/quota/refresh" not in HTTP_ENDPOINTS
