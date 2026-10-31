"""Window declarations, observations and their consumer-facing provenance."""
from types import SimpleNamespace

import pytest

from ouroboros import capability_evidence as ce, context_budget as cb
from ouroboros.context_window import record_observation, record_refusal, resolve_window
from ouroboros.loop_llm_call import classify_llm_exception


SCHEDULER = ("Input length (412345 tokens) exceeds the maximum allowed length (400000 tokens). "
             "Use a shorter input or enable --allow-auto-truncate.")


@pytest.mark.parametrize("text,counts", [
    (SCHEDULER, (412345, 0, 0)),
    ("The input (412345 tokens) is longer than the model's context length (400000 tokens).", (412345, 0, 0)),
    ("Requested token count exceeds the model's maximum context length of 400000 tokens. "
     "You requested a total of 458000 tokens: 392464 tokens from the input messages and "
     "65536 tokens for the completion.", (392464, 65536, 458000)),
])
def test_sglang_size_refusals_keep_reply_room(tmp_path, monkeypatch, text, counts):
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    error = RuntimeError(text)
    error.status_code = 400
    error.body = {"error": {"code": 400, "message": text}}
    assert classify_llm_exception(error).kind == "context_overflow"
    assert cb.context_overflow_message(text)
    assert ce._parse_ctx_limit_number(text) == 400000
    assert ce._parse_ctx_requested(text) == counts
    record_refusal({"_last_llm_provider_message": text}, SimpleNamespace(route_fp="route"))
    record = resolve_window(ce.CapabilityEvidence(0, "unprobeable", "none", "route"), 65536, root=tmp_path)
    assert record.total.tokens == 400000 and record.input is None
    assert record.usable == 334464


@pytest.mark.parametrize("text", [
    "maximum allowed length (64 characters)", "maximum length is 128",
    "Request header field exceeds the maximum allowed length (8192)", "Upstream request failed",
])
def test_unrelated_length_errors_create_no_window(tmp_path, monkeypatch, text):
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    error = RuntimeError(text)
    error.status_code = 400
    assert classify_llm_exception(error).kind == "bad_request"
    assert not cb.context_overflow_message(text)
    assert ce._parse_ctx_limit_number(text) == 0
    record_refusal({"_last_llm_provider_message": text}, SimpleNamespace(route_fp="route"))
    assert resolve_window(ce.CapabilityEvidence(0, "unprobeable", "none", "route"), 65536, root=tmp_path).usable is None


@pytest.mark.parametrize("code,kind", [
    ("invalid_api_key", "auth_error"), ("insufficient_quota", "quota_exhausted"),
    ("rate_limit_exceeded", "provider_transient"), (400, "context_overflow"),
])
def test_typed_errors_keep_precedence(code, kind):
    error = RuntimeError(SCHEDULER)
    error.status_code = 400
    error.body = {"error": {"code": code, "message": SCHEDULER}}
    assert classify_llm_exception(error).kind == kind


@pytest.mark.parametrize("prefix", ["max_tokens: ", "request body too large: ", ""])
def test_output_and_body_errors_keep_precedence(prefix):
    error = RuntimeError(prefix + SCHEDULER)
    error.status_code = 400
    assert classify_llm_exception(error).kind == ("request_too_large" if prefix else "context_overflow")


@pytest.mark.parametrize("note,window,retired", [
    ("owner-confirmed via settings save", 1000000, True),
    ("owner-confirmed via context-mode toggle", 1000000, True),
    ("owner-confirmed scope reviewer window", 1000000, True),
    ("owner-confirmed via settings save", 400000, False),
    ("owner-confirmed via context-mode toggle", 400000, False),
    ("owner-confirmed scope reviewer window", 400000, False),
    ("typed by owner", 1000000, False),
    ("owner-confirmed via settings save ", 1000000, False),
    ("owner-confirmed scope reviewer window ", 1000000, False),
])
def test_only_forced_dialog_acks_retire_without_a_read_rewrite(tmp_path, monkeypatch, note, window, retired):
    from ouroboros.response_limits import record_response_ack

    route = dict(provider="openai-compatible", model="openai-compatible::fixture", base_url="https://fixture.invalid/v1")
    ack = ce.record_owner_ack(tmp_path, **route, window_tokens=window, note=note)
    record_response_ack(tmp_path, **route, max_output_tokens=4096)
    before = ce._store_path(tmp_path).read_bytes()
    evidence = ce.probe(tmp_path, **route, allow_fetch=False)
    assert evidence.window_tokens == (0 if retired else window)
    assert bool(evidence.retired_ack) is retired
    assert evidence.response_limit["max_output_tokens"] == 4096
    assert ce.list_owner_acks(tmp_path)[0]["retired"] is retired
    assert ce._store_path(tmp_path).read_bytes() == before
    assert ce._load(tmp_path)["owner_acks"][ack["route_fp"]] == ack


@pytest.mark.parametrize("mismatch", [False, True])
def test_clear_window_ack_is_exact_route_and_audited(tmp_path, monkeypatch, mismatch):
    import asyncio
    import json
    from ouroboros.gateway import settings as gateway
    from ouroboros.response_limits import record_response_ack, resolve_response_limit

    route = dict(provider="openai-compatible", model="openai-compatible::fixture", base_url="https://fixture.invalid/v1")
    ack = ce.record_owner_ack(tmp_path, **route, window_tokens=1000000)
    other = ce.record_owner_ack(tmp_path, **{**route, "model": "other"}, window_tokens=400000)
    record_response_ack(tmp_path, **route, max_output_tokens=4096)
    audit = []
    monkeypatch.setattr(gateway, "_owner_audit", lambda *args: audit.append(args[1:]))

    async def body():
        return {**route, "window_tokens": 0, "route_fp": "wrong" if mismatch else ack["route_fp"]}

    request = SimpleNamespace(json=body, app=SimpleNamespace(state=SimpleNamespace(drive_root=tmp_path)))
    response = asyncio.run(gateway.api_acknowledge_capability(request))
    assert response.status_code == (400 if mismatch else 200)
    rows = ce._load(tmp_path)["owner_acks"]
    assert (ack["route_fp"] in rows) is mismatch
    assert other["route_fp"] in rows
    assert resolve_response_limit(tmp_path, **route).max_output_tokens == 4096
    assert bool(audit) is not mismatch
    if not mismatch:
        assert json.loads(response.body)["ack"]["removed"]
        assert audit[0][0] == "capability_ack" and audit[0][1]["window_tokens"] == 0


@pytest.mark.parametrize("ack,role,expected,conflict", [
    (0, 0, 400000, False), (1000000, 0, 1000000, True),
    (1000000, 250000, 250000, True), (1000000, 400000, 400000, False),
])
def test_discovery_continues_beside_owner_declarations(tmp_path, monkeypatch, ack, role, expected, conflict):
    from datetime import datetime, timedelta, timezone

    now, calls = datetime.now(timezone.utc), []
    monkeypatch.setattr(ce, "utc_now", lambda: now)
    monkeypatch.setattr(ce, "utc_now_iso", lambda: now.isoformat())
    monkeypatch.setattr(ce, "_provider_metadata_window", lambda *a, **k: calls.append(1) or (400000, now.isoformat()))
    route = dict(provider="openai", model="openai::fixture")
    if ack:
        ce.record_owner_ack(tmp_path, **route, window_tokens=ack)
    def read(fetch):
        return ce.with_role_window(ce.probe(tmp_path, **route, allow_fetch=fetch), role)

    evidence = read(True)
    assert evidence.window_tokens == expected and len(calls) == 1
    record = resolve_window(evidence, 65536, root=tmp_path)
    assert bool(record.contradicted_by) is conflict
    if conflict:
        assert record.contradicted_by.tokens == 400000
    if role:
        assert evidence.ts == ""  # No metadata date invented for an owner's edit.
    read(True)
    assert len(calls) == 1
    now += timedelta(seconds=ce._CONFIRMED_TTL_SEC + 1)
    hot = read(False)
    assert len(calls) == 1 and resolve_window(hot, 65536, root=tmp_path).contradicted_by is None
    refreshed = read(True)
    assert len(calls) == 2 and refreshed.window_tokens == expected


@pytest.mark.parametrize("declared,expected", [(250000, 250000), (1000000, 400000)])
def test_refusal_caps_but_never_raises_a_declaration(tmp_path, monkeypatch, declared, expected):
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    monkeypatch.setattr(ce, "utc_now", lambda: now)
    monkeypatch.setattr(ce, "utc_now_iso", lambda: now.isoformat())
    evidence = ce.CapabilityEvidence(declared, "asserted", "user_setting", "route")
    record_observation(tmp_path, "route", refused=400000)
    assert resolve_window(evidence, 65536, root=tmp_path).total.tokens == expected
    now += timedelta(seconds=ce._CONFIRMED_TTL_SEC + 1)
    assert resolve_window(evidence, 65536, root=tmp_path).total.tokens == declared


def test_process_cached_metadata_keeps_its_date_and_refreshes_after_expiry(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from ouroboros import utils, vision_routing
    from ouroboros.llm import LLMClient
    import requests

    now, calls = datetime.now(timezone.utc), []
    monkeypatch.setattr(ce, "utc_now", lambda: now)
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(utils, "utc_now_iso", lambda: now.isoformat())
    monkeypatch.setattr(vision_routing, "utc_now_iso", lambda: now.isoformat())
    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_FETCHED", False)
    monkeypatch.setattr(LLMClient, "_CAPABILITIES_FETCH_OK", False)
    monkeypatch.setattr(LLMClient, "_CONTEXT_LENGTH_CACHE", {})
    monkeypatch.setattr(LLMClient, "_CONTEXT_LENGTH_OBSERVED_AT", "")
    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_CACHE", {})
    monkeypatch.setattr(requests, "get", lambda *a, **k: calls.append(1) or SimpleNamespace(
        status_code=200, json=lambda: {"data": [{"id": "fixture/model", "context_length": 400000,
                                               "top_provider": {"max_completion_tokens": 65536}}]}))
    route = dict(provider="openrouter", model="fixture/model")
    first = ce.probe(tmp_path, **route)
    now += timedelta(hours=1)
    cached = ce.probe(tmp_path, **route, force=True)
    assert cached.ts == first.ts and len(calls) == 1
    now += timedelta(seconds=ce._CONFIRMED_TTL_SEC)
    stale = ce.probe(tmp_path, **route, allow_fetch=False)
    assert stale.stale and stale.ts == first.ts and len(calls) == 1
    fresh = ce.probe(tmp_path, **route)
    assert fresh.ts == now.isoformat() and not fresh.stale and len(calls) == 2


@pytest.mark.parametrize("window", [400000, 0])
def test_catalog_refresh_publishes_complete_windows_with_their_date(tmp_path, monkeypatch, window):
    from ouroboros import utils, vision_routing
    from ouroboros.llm import LLMClient
    import requests

    old = {"fixture/first": 128000, "fixture/second": 64000, "fixture/removed": 16000}
    old_date, new_date = "2026-10-20T00:00:00+00:00", "2026-10-22T00:00:00+00:00"
    monkeypatch.setattr(LLMClient, "_CONTEXT_LENGTH_CACHE", old)
    monkeypatch.setattr(LLMClient, "_CONTEXT_LENGTH_OBSERVED_AT", old_date)
    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_CACHE", {})
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(utils, "utc_now_iso", lambda: new_date)
    monkeypatch.setattr(vision_routing, "utc_now_iso", lambda: new_date)
    reads = []

    class ObservedRow(dict):
        def get(self, key, default=None):
            if key == "context_length":
                # Interleave a hot-path reader with parsing each new catalog row.
                reads.append(({model: LLMClient.openrouter_context_length(model, allow_fetch=False)
                               for model in old}, LLMClient._CONTEXT_LENGTH_OBSERVED_AT))
            return super().get(key, default)

    rows = [ObservedRow(id="fixture/first", context_length=window),
            ObservedRow(id="fixture/second", context_length=window * 2)]
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(
        status_code=200, json=lambda: {"data": rows}))
    LLMClient._fetch_openrouter_capabilities()

    assert reads == [(old, old_date), (old, old_date)]
    expected = {"fixture/first": window, "fixture/second": window * 2} if window else {}
    assert LLMClient._CONTEXT_LENGTH_CACHE == expected
    assert LLMClient._CONTEXT_LENGTH_OBSERVED_AT == new_date
    assert LLMClient._CAPABILITIES_FETCH_OK
    assert LLMClient.openrouter_context_length("fixture/first", allow_fetch=False) == window
    assert LLMClient.openrouter_context_length("fixture/removed", allow_fetch=False) == 0
