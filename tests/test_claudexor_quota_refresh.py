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
    monkeypatch.setattr(claudexor_quota, "_REFRESH_RECENT", {})


def test_shared_refresh_joins_live_manual_and_reuses_only_lazy_recent(monkeypatch, tmp_path):
    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateway import claudexor_quota as quota

    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path)
    entered, release, joined = Event(), Event(), Event()
    calls = []

    class JoinedFuture(Future):
        def result(self, timeout=None):
            if not self.done():
                joined.set()
            return super().result(timeout)

    monkeypatch.setattr(quota, "Future", JoinedFuture)

    def refresh(target=None):
        calls.append(target)
        entered.set()
        assert release.wait(10)
        return _envelope()

    monkeypatch.setattr(quota, "_refresh_quota", refresh)
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(quota._shared_refresh, lazy=True)
        try:
            assert entered.wait(10)
            second = pool.submit(quota._shared_refresh)
            assert joined.wait(10)
            joined.clear()
            third = pool.submit(quota._shared_refresh, lazy=True)
            assert joined.wait(10)
        finally:
            release.set()
        answers = [call.result(10) for call in (first, second, third)]
    assert calls == [None]
    attempts = [answer.pop("refresh_attempt") for answer in answers]
    assert [item["reused"] for item in attempts] == [False, True, True]
    assert attempts[0]["started_at"] == attempts[1]["started_at"] == attempts[2]["started_at"]
    assert attempts[0]["completed_at"] == attempts[1]["completed_at"] == attempts[2]["completed_at"]
    assert attempts[0]["started_at"] <= attempts[0]["completed_at"]
    answers[0]["snapshots"].clear()
    assert answers[1] == _envelope() and answers[2] == _envelope()
    replay = quota._shared_refresh(lazy=True)
    assert replay.pop("refresh_attempt") == attempts[1]
    assert replay == _envelope() and calls == [None]
    fresh = quota._shared_refresh()
    assert fresh.pop("refresh_attempt")["reused"] is False
    assert fresh == _envelope() and calls == [None, None]


def test_lazy_refresh_failure_is_an_attempt_throttle_and_expired_keys_are_pruned(monkeypatch, tmp_path):
    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateway import claudexor_quota as quota

    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path)
    clock, calls = [100.0], []
    monkeypatch.setattr(quota.time, "monotonic", lambda: clock[0])

    def refuse(target=None):
        calls.append(target)
        raise ClaudexorUnavailable("quota_refresh_paced", "paced", status_code=429)

    monkeypatch.setattr(quota, "_refresh_quota", refuse)
    targets = [{"harness": "codex", "profile_id": value} for value in ("a", "b")]
    for _ in range(2):
        with pytest.raises(ClaudexorUnavailable, match="paced"):
            quota._shared_refresh(targets[0], lazy=True)
    assert calls == [targets[0]] and quota._REFRESH_FLIGHTS == {}
    # Manual ignores completed attempts but never retries an unknown call itself.
    with pytest.raises(ClaudexorUnavailable):
        quota._shared_refresh(targets[0])
    assert len(calls) == 2
    clock[0] += 61
    with pytest.raises(ClaudexorUnavailable):
        quota._shared_refresh(targets[1], lazy=True)
    assert len(quota._REFRESH_RECENT) == 1, "expired targets/homes must not accumulate forever"


def test_lazy_refresh_scopes_do_not_mix_full_exact_accounts_or_homes(monkeypatch, tmp_path):
    from ouroboros import claudexor_daemon as owned
    from ouroboros.gateway import claudexor_quota as quota

    home, calls = [tmp_path / "a"], []
    monkeypatch.setattr(owned, "owned_config_dir", lambda: home[0])

    def refresh(target=None):
        calls.append((str(home[0]), target))
        return {"snapshots": [], "target": target}

    monkeypatch.setattr(quota, "_refresh_quota", refresh)
    a = {"harness": "codex", "profile_id": "a"}
    b = {"harness": "codex", "profile_id": "b"}
    for target in (None, a, b, {"profile_id": "a", "harness": "codex"}):
        assert quota._shared_refresh(target, lazy=True)["target"] == target
    assert len(calls) == 3, "only the same exact semantic target joins"
    home[0] = tmp_path / "b"
    assert quota._shared_refresh(a, lazy=True)["target"] == a
    assert len(calls) == 4, "a different owned home cannot reuse another daemon's result"


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
    assert calls == [("to_thread", claudexor_quota._shared_refresh, (), {"lazy": False}), ("refresh",)]


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


def test_quota_refresh_route_is_post_only_and_indexed(tmp_path):
    matching = [
        route for route in collect_routes(data_dir=tmp_path)
        if getattr(route, "path", "") == "/api/claudexor/quota/refresh"
    ]
    assert len(matching) == 1
    assert matching[0].methods == {"POST"}
    assert "POST /api/claudexor/quota/refresh" in HTTP_ENDPOINTS
    assert "GET /api/claudexor/quota/refresh" not in HTTP_ENDPOINTS
