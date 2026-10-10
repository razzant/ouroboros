"""An expired output fact survives an outage without probing on every send."""
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from ouroboros import capability_evidence as ce, usage_accounting as ua, utils
from ouroboros.response_limits import record_response_ack, resolve_response_limit
from tests import test_processing_transport as transport_fixtures

transport = transport_fixtures.transport


@pytest.mark.parametrize("provider,outage", [
    ("openrouter", "transport"), ("openrouter", "http_error"), ("openrouter", "omitted"),
    ("openai-compatible", "transport"), ("openai-compatible", "http_error"),
])
def test_expired_maximum_throttles_failed_catalog_refresh_then_recovers_on_wire(
        transport, monkeypatch, provider, outage):
    root, client, sent = transport
    clock = [datetime.now(timezone.utc)]
    monkeypatch.setattr(ce, "utc_now", lambda: clock[0])
    monkeypatch.setattr(ce, "utc_now_iso", lambda: clock[0].isoformat())
    monkeypatch.setattr(utils, "utc_now_iso", lambda: clock[0].isoformat())
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://limits.invalid/v1")
    model = "vendor/listed" if provider == "openrouter" else "openai-compatible::listed"
    target = client._resolve_remote_target(model)
    route = dict(provider=provider, model=model, base_url=target["base_url"])
    # The independent window acknowledgement must not hide output-limit refreshes.
    ce.record_owner_ack(root, **route, window_tokens=100_000)
    state = {"outage": "", "maximum": 4096}
    gets = []

    def get(url, **kwargs):
        gets.append(url)
        assert kwargs["timeout"] in (5, 5.0)
        if state["outage"] == "transport":
            raise OSError("metadata offline")
        status = 503 if state["outage"] == "http_error" else 200

        def raise_for_status():
            if status != 200:
                raise OSError("metadata unavailable")

        row = {"id": "vendor/listed" if provider == "openrouter" else "listed",
               "context_length": 100_000,
               "max_output_tokens": state["maximum"],
               "top_provider": {"max_completion_tokens": state["maximum"]}}
        return SimpleNamespace(status_code=status, raise_for_status=raise_for_status,
                               json=lambda: {"data": [] if state["outage"] == "omitted" else [row]})

    monkeypatch.setattr("requests.get", get)
    monkeypatch.setattr("httpx.get", get)

    def probe_and_send(expected):
        evidence = ce.probe(root, **route)
        client.chat([{"role": "user", "content": "reply"}], model, max_tokens=65536)
        assert sent[-1]["max_tokens"] == expected
        assert ua.last_physical_attempt_capture().max_completion_tokens == expected
        return evidence.response_limit

    original = probe_and_send(4096)
    assert not original["stale"] and len(gets) == 1
    clock[0] += timedelta(seconds=ce._CONFIRMED_TTL_SEC + 1)
    state["outage"] = outage
    stale = probe_and_send(65536)
    assert stale == {**original, "stale": True}
    assert len(gets) == 2
    assert probe_and_send(65536) == stale
    clock[0] += timedelta(seconds=ce._FAILED_TTL_SEC - 1)
    assert probe_and_send(65536) == stale
    assert len(gets) == 2  # A failed refresh is throttled; its old fact stays stale.
    assert ce._load(root)["response_limits"][original["route_fp"]] == original

    # A different exact route still reads its catalog during this route's outage.
    other = {**route, "model": "vendor/other" if provider == "openrouter" else "openai-compatible::other"}
    assert resolve_response_limit(root, **other, allow_fetch=True).ceiling(65536) == 65536
    assert len(gets) == 3
    # Explicit limits take precedence even while the failed refresh is cooling down.
    record_response_ack(root, **route, max_output_tokens=2048)
    assert probe_and_send(2048)["source"] == "owner_ack" and len(gets) == 3
    record_response_ack(root, **route, max_output_tokens=0)
    assert probe_and_send(65536) == stale and len(gets) == 3

    clock[0] += timedelta(seconds=2)
    assert probe_and_send(65536) == stale and len(gets) == 4
    assert probe_and_send(65536) == stale and len(gets) == 4
    state.update(outage="", maximum=8192)
    clock[0] += timedelta(seconds=ce._FAILED_TTL_SEC + 1)
    # A hot-path read never fetches, even after the retry interval elapses.
    assert resolve_response_limit(root, **route).ceiling(65536) == 65536 and len(gets) == 4
    recovered = probe_and_send(8192)
    assert recovered["observed_at"] != original["observed_at"] and not recovered["stale"]
    assert recovered["max_output_tokens"] == 8192 and len(gets) == 5
    assert asdict(resolve_response_limit(root, **route, allow_fetch=True)) == recovered
    assert len(gets) == 5
