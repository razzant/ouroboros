"""Window declarations, observations and their consumer-facing provenance."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from ouroboros import capability_evidence as ce
from ouroboros import context_budget as cb
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
    "maximum allowed length (64 characters)", "Request header field exceeds the maximum allowed length (8192)", "Upstream request failed",
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
    ("rate_limit_exceeded", "provider_transient"),
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


@pytest.mark.parametrize("accepted,capped", [(841_181, False), (600_000, True), (500_000, True), (0, True)])
def test_a_stated_limit_below_an_input_the_scope_accepted_teaches_no_ceiling(tmp_path, monkeypatch, accepted, capped):
    """One endpoint behind a gateway model said «maximum context length is 600000 tokens» an hour after the same
    scope had 841,181 accepted (15.11). The scope does not name the serving endpoint, so that statement is the
    endpoint's: it writes no refusal row and every room keeps the declared window. With no larger acceptance on
    record the 24-hour rule caps as before (an equal one included)."""
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    if accepted:
        record_observation(tmp_path, "route", accepted=accepted)
    record_refusal({"_last_llm_provider_message": "This endpoint's maximum context length is 600000 tokens."},
                   SimpleNamespace(route_fp="route"))
    record = resolve_window(ce.CapabilityEvidence(1_048_576, "confirmed", "metadata", "route"), 65_536, root=tmp_path)
    assert (record.total.source, record.total.tokens) == (("refusal", 600_000) if capped else ("metadata", 1_048_576))
    assert record.usable == (534_464 if capped else 983_040)


VLLM = ("'max_tokens' or 'max_completion_tokens' is too large: {reply}. This model's maximum context length is {window} "
        "tokens and your request has {size} input tokens ({reply} > {window} - {size}).")


@pytest.mark.parametrize("window,size,kind", [
    (200_000, 150_000, "context_overflow"),   # the input fits the window it names: the sum is what it refuses
    (32_768, 1_000, "request_too_large"),     # the reply alone passes that window: an output limit, as before
])
def test_a_sentence_that_counts_the_input_under_its_window_is_the_windows_refusal(tmp_path, monkeypatch, window, size, kind):
    """A server that validates input plus ``max_tokens`` against its window names ``max_tokens`` first (vLLM). Its
    refusal is the window's own, with the server's count of the request, so the size recovery reads it and the
    record learns that this total holds the reply; a reply above the window it names stays an output limit."""
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    text = VLLM.format(reply=65_536, window=window, size=size)
    error = RuntimeError(text)
    error.status_code = 400
    error.body = {"error": {"code": 400, "message": text}}
    assert classify_llm_exception(error).kind == kind
    assert cb.context_overflow_message(text) is (kind == "context_overflow")
    assert cb.output_or_body_size_message(text) is (kind == "request_too_large")
    assert ce._parse_ctx_limit_number(text) == window and ce._parse_ctx_requested(text) == (size, 65_536, 0)
    # "is too large: N" is the reply only where max_tokens is what is too large; an input worded so states no reply.
    assert ce._parse_ctx_requested("Input is too large: 250000 tokens. The maximum context length is 200000 tokens.")[1] == 0
    if kind == "context_overflow":
        record_refusal({"_last_llm_provider_message": text}, SimpleNamespace(route_fp="route"))
        record = resolve_window(ce.CapabilityEvidence(window, "confirmed", "metadata", "route"), 65_536, root=tmp_path)
        assert record.total.source == "refusal" and record.admitted == record.usable == window - 65_536


@pytest.mark.parametrize("note,window,retired", [
    ("owner-confirmed via settings save", 1000000, True),
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
    (1000000, 250000, 250000, True),
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

    import requests

    from ouroboros import utils, vision_routing
    from ouroboros.llm import LLMClient

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
    import requests

    from ouroboros import utils, vision_routing
    from ouroboros.llm import LLMClient

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


def test_cached_and_fresh_equal_reads_do_not_release_a_refusal(tmp_path):
    from dataclasses import replace
    first = ce.CapabilityEvidence(80000, 'confirmed', ce.SOURCE_SERVER_METADATA, 'r',
                                  provider='openai-compatible', ts=ce.utc_now_iso(), window_read_id='one')
    resolve_window(first, 8192, root=tmp_path)
    second = replace(first, window_read_id='two', window_sequence=0, window_sequence_route='')
    resolve_window(second, 8192, root=tmp_path)
    record_observation(tmp_path, 'r', refused=40000)
    cached = replace(second, window_sequence=0, window_sequence_route='')
    fresh = replace(cached, window_read_id='three')
    for evidence in (second, cached, fresh):
        assert resolve_window(evidence, 8192, root=tmp_path).total.tokens == 40000


@pytest.mark.parametrize('raiseable,owner,expected', [(True, 80000, None), (False, 80000, 80000), (True, 120000, 120000)])
def test_acceptance_disproves_each_smaller_raiseable_statement(tmp_path, raiseable, owner, expected):
    evidence = ce.CapabilityEvidence(owner, 'asserted', 'user_setting', 'r', provider='openai-compatible',
        route_facts={'acceptance_can_raise': raiseable},
        server_window=dict(tokens=80000, kind='total', source='server_metadata', observed_at=ce.utc_now_iso()))
    resolve_window(evidence, 8192, root=tmp_path)
    record_observation(tmp_path, 'r', accepted=100000)
    result = resolve_window(evidence, 8192, root=tmp_path)
    assert (result.total.tokens if result.total else None) == expected
    assert result.input.tokens == 100000


def test_window_preview_does_not_publish(tmp_path):
    evidence = ce.CapabilityEvidence(80000, 'confirmed', ce.SOURCE_SERVER_METADATA, 'r',
                                     provider='openai-compatible', ts=ce.utc_now_iso())
    assert resolve_window(evidence, 8192, root=tmp_path, publish=False).total.tokens == 80000
    assert not ce._store_path(tmp_path).exists()


def test_enumerated_window_authority():
    """All 111,151 histories up to length four, three sizes, six event classes.

    The oracle selects surviving claims by their future witnesses, independently
    of the reducer's event fold. Identical reads are observations of one claim.
    """
    from itertools import product

    from ouroboros.context_window import WindowLimit, window_authority
    alphabet = [(source, kind, size, raises) for size in (40000, 80000, 120000)
                for source, kind, raises in [('user_setting', 'total', True), ('owner_ack', 'total', False),
                    ('server_metadata', 'total', True), ('refusal', 'total', True),
                    ('refusal', 'input', True), ('accepted', 'input', True)]]
    for length in range(5):
        for history in product(alphabet, repeat=length):
            events = [WindowLimit(n, k, s, acceptance_can_raise=r, sequence=i+1) for i, (s,k,n,r) in enumerate(history)]
            actual, _ = window_authority(events)
            # Repeating the current physical statement can never alter the result.
            read = next((e for e in reversed(events) if e.source == 'server_metadata'), None)
            if read:
                assert window_authority([*events, read])[0] == actual, history
            # Every surviving raiseable statement must lack a larger later acceptance.
            for limit in actual.values():
                if limit.source != 'accepted' and limit.acceptance_can_raise:
                    assert not any(e.source == 'accepted' and e.sequence > limit.sequence and e.tokens > limit.tokens
                                   for e in events), history
            # Latest owner claim survives unless a later larger acceptance raises it;
            # a smaller read/refusal may cap it, but cannot enlarge it.
            owner = next((e for e in reversed(events) if e.source in {'user_setting','owner_ack'}), None)
            if owner and not (owner.acceptance_can_raise and any(e.source == 'accepted' and e.tokens > owner.tokens
                              and e.sequence > owner.sequence for e in events)):
                assert actual.get('total') and actual['total'].tokens <= owner.tokens, history
            # A refusal after the current (changed) read stays, even after equal reads.
            changed = []
            for e in events:
                if e.source == 'server_metadata' and (not changed or e.tokens != changed[-1].tokens):
                    changed.append(e)
            for kind in ('input', 'total'):
                refused = next((e for e in reversed(events) if e.source == 'refusal' and e.kind == kind), None)
                if refused and (not changed or refused.sequence > changed[-1].sequence) and not any(
                    e.source == 'accepted' and e.sequence > refused.sequence and e.tokens > refused.tokens for e in events):
                    assert kind in actual and actual[kind].tokens <= refused.tokens, history


@pytest.fixture
def clock(monkeypatch):
    now = [datetime.now(timezone.utc)]
    monkeypatch.setattr(ce, "utc_now", lambda: now[0])
    monkeypatch.setattr(ce, "utc_now_iso", lambda: now[0].isoformat())
    monkeypatch.setattr("ouroboros.utils.utc_now_iso", lambda *a, **kw: now[0].isoformat())
    return now


def server(clock, window=80000, provider="openai-compatible", source="server_metadata"):
    stamp = clock[0].isoformat()
    result = ce.CapabilityEvidence(window, "confirmed", source, "r", provider=provider, ts=stamp)
    result.server_window = dict(tokens=window, kind="total", source=source, observed_at=stamp)
    return result


def test_new_read_overrules_old_success_and_later_success_can_disprove_it(tmp_path, clock):
    record_observation(tmp_path, "r", accepted=200000)
    clock[0] += timedelta(seconds=1)
    evidence = server(clock)
    record = resolve_window(evidence, 8192, root=tmp_path, accepted_input_tokens=200000)
    assert record.total.tokens == 80000 and record.usable == 71808
    # A post-restart success below the historic maximum must still be recorded.
    clock[0] += timedelta(seconds=1)
    record_observation(tmp_path, "r", accepted=100000)
    record = resolve_window(evidence, 8192, root=tmp_path)
    assert record.total is None and record.input.tokens == 100000
    clock[0] += timedelta(seconds=1)
    record_observation(tmp_path, "r", accepted=70000)
    assert resolve_window(evidence, 8192, root=tmp_path).input.tokens == 100000
    clock[0] += timedelta(seconds=1)
    evidence = server(clock, 90000)
    assert resolve_window(evidence, 8192, root=tmp_path).total.tokens == 90000


def test_clock_rollback_uses_publication_order_not_timestamp_order(tmp_path, clock):
    start = clock[0]
    for seconds, tokens in ((200, 80000), (100, 90000), (300, 70000)):
        clock[0] = start + timedelta(seconds=seconds)
        record_observation(tmp_path, "r", accepted=tokens)
    clock[0] = start + timedelta(seconds=150)
    evidence = server(clock, 75000)
    record = resolve_window(evidence, 8192, root=tmp_path)
    # The read is published last, regardless of its wall timestamp.
    assert record.total.tokens == 75000 and record.accepted == 90000
    clock[0] = start + timedelta(seconds=50)
    record_observation(tmp_path, "r", accepted=80000)
    assert resolve_window(evidence, 8192, root=tmp_path).input.tokens == 80000
    # A still later smaller success cannot replace the useful answer.
    clock[0] = start + timedelta(seconds=400)
    record_observation(tmp_path, "r", accepted=70000)
    assert resolve_window(evidence, 8192, root=tmp_path).input.tokens == 80000


def test_identical_acceptances_save_once_and_again_only_after_a_new_fact(tmp_path, clock, monkeypatch):
    saves = []
    original = ce._save
    monkeypatch.setattr(ce, "_save", lambda *args: (saves.append(1), original(*args))[1])
    for _ in range(40):
        record_observation(tmp_path, "r", accepted=50000)
        clock[0] += timedelta(seconds=1)
    assert len(saves) == 1
    record_observation(tmp_path, "r", refused=40000)
    record_observation(tmp_path, "r", accepted=50000)
    for _ in range(40):
        record_observation(tmp_path, "r", accepted=50000)
    assert len(saves) == 3
    evidence = server(clock, 40000)
    resolve_window(evidence, 8192, root=tmp_path)
    for _ in range(40):
        record_observation(tmp_path, "r", accepted=50000)
    assert len(saves) == 5  # one read, one success answering it
    assert resolve_window(evidence, 8192, root=tmp_path).input.tokens == 50000


def test_captured_read_keeps_its_order_when_replayed_after_new_read(tmp_path, clock):
    from dataclasses import replace
    evidence = server(clock)
    captured = resolve_window(evidence, 8192, root=tmp_path)
    clock[0] -= timedelta(seconds=5)
    record_observation(tmp_path, "r", accepted=100000)
    new = server(clock, 90000)
    assert resolve_window(new, 8192, root=tmp_path).total.tokens == 90000
    plan = SimpleNamespace(window_record=captured, window_tokens=80000, status="confirmed",
                           provider="openai-compatible")
    assert resolve_window(plan, 8192, root=tmp_path).input.tokens == 100000
    assert resolve_window(replace(evidence), 8192, root=tmp_path).input.tokens == 100000
    assert resolve_window(new, 8192, root=tmp_path).total.tokens == 90000


def test_capture_keeps_acceptance_witness_after_another_read_and_refusal(tmp_path, clock):
    record_observation(tmp_path, "r", accepted=200000)
    captured = resolve_window(server(clock), 8192, root=tmp_path)
    plan = SimpleNamespace(window_record=captured, window_tokens=80000, status="confirmed",
                           provider="openai-compatible")
    record_observation(tmp_path, "r", accepted=100000)
    assert resolve_window(plan, 8192, root=tmp_path).usable == 100000
    resolve_window(server(clock, 120000), 8192, root=tmp_path)
    record_observation(tmp_path, "r", refused=110000)
    assert resolve_window(plan, 8192, root=tmp_path).usable == 100000


def test_captured_witness_survives_enumerated_other_observer_events(tmp_path, monkeypatch):
    """Project an older capture after each event; the oracle keeps the full history."""
    from itertools import product

    from ouroboros.context_window import WindowLimit, window_authority
    from tests._window_history import ObserverHistory
    observer = ObserverHistory(monkeypatch, tmp_path)
    for history in product(("read120", "accept100", "accept200", "refuse_input110", "refuse_total110"), repeat=3):
        observer.reset()
        record_observation(tmp_path, "r", accepted=200000)
        captured = resolve_window(ce.CapabilityEvidence(80000, "confirmed", "server_metadata", "r",
            provider="openai-compatible", ts=ce.utc_now_iso(), window_instance_id="captured"), 8192, root=tmp_path)
        plan = SimpleNamespace(window_record=captured, window_tokens=80000, status="confirmed",
                               provider="openai-compatible")
        record_observation(tmp_path, "r", accepted=100000)
        events = [WindowLimit(200000, "input", "accepted", sequence=2),
                  captured.total, WindowLimit(100000, "input", "accepted", sequence=4)]
        for step, action in enumerate(history, 5):
            if action == "read120":
                resolve_window(ce.CapabilityEvidence(120000, "confirmed", "server_metadata", "r",
                    provider="openai-compatible", ts=ce.utc_now_iso()), 8192, root=tmp_path)
            if action.startswith("accept") or action.startswith("refuse"):
                accepted = action.startswith("accept")
                tokens = int(action.removeprefix("accept")) * 1000 if accepted else 110000
                kind = "input" if accepted or "input" in action else "total"
                record_observation(tmp_path, "r", **({"accepted": tokens} if accepted else {"refused": tokens, "kind": kind}))
                events.append(WindowLimit(tokens, kind, "accepted" if accepted else "refusal", sequence=step))
            limits, _ = window_authority(events)
            expected = min((x.tokens - (8192 if x.kind == "total" else 0) for x in limits.values()), default=None)
            assert resolve_window(plan, 8192, root=tmp_path).usable == expected, history


def test_future_acceptance_is_retained_but_cannot_overrule_a_current_read(tmp_path, clock):
    now = clock[0]
    clock[0] += timedelta(hours=1)
    record_observation(tmp_path, "r", accepted=120000)
    clock[0] = now
    assert resolve_window(server(clock), 8192, root=tmp_path).total.tokens == 80000


def test_same_size_later_acceptance_must_be_saved_to_answer_an_intervening_read(tmp_path, clock):
    record_observation(tmp_path, "r", accepted=50000)
    clock[0] += timedelta(seconds=1)
    evidence = server(clock, 40000)
    assert resolve_window(evidence, 8192, root=tmp_path).total.tokens == 40000
    clock[0] += timedelta(seconds=1)
    record_observation(tmp_path, "r", accepted=50000)
    record = resolve_window(evidence, 8192, root=tmp_path)
    assert record.total is None and record.input.tokens == 50000


def test_a_dominated_out_of_order_acceptance_does_not_rewrite_the_store(tmp_path, clock, monkeypatch):
    now = clock[0]
    record_observation(tmp_path, "r", accepted=50000)
    clock[0] = now - timedelta(seconds=1)
    def unexpected_save(*args):
        pytest.fail("an older smaller receipt changes no frontier")
    monkeypatch.setattr(ce, "_save", unexpected_save)
    record_observation(tmp_path, "r", accepted=40000)


@pytest.mark.parametrize("declaration", ["role", "ack"])
def test_serving_buffer_caps_large_declaration_and_discloses_it(tmp_path, clock, declaration):
    from dataclasses import replace

    from ouroboros.context_view_facts import window_projection

    evidence = server(clock)
    evidence = ce.with_role_window(evidence, 262000) if declaration == "role" else replace(
        evidence, window_tokens=262000, source=ce.SOURCE_OWNER_ACK, status="asserted")
    record = resolve_window(evidence, 8192, root=tmp_path)
    assert record.total.tokens == 80000 and record.admitted == 71808
    assert record.declared[0].tokens == 262000 and record.contradicted_by.tokens == 80000
    text = window_projection(record)["text"]
    assert "262,000" in text and "80,000" in text and "\n" not in text
    # The plan's captured record must retain the ceiling through subsequent rounds.
    plan = SimpleNamespace(window_record=record, window_tokens=262000, status="asserted",
                           provider="openai-compatible", evidence_source=evidence.source)
    assert resolve_window(plan, 8192, root=tmp_path).total.tokens == 80000


@pytest.mark.parametrize("provider,source,role,expected", [
    ("openai-compatible", "server_metadata", 60000, 60000),
    ("openai-compatible", "provider_metadata", 262000, 262000),
    ("openai", "server_metadata", 262000, 262000),
    ("claudexor", "server_metadata", 262000, 262000),
])
def test_scope_and_smaller_settings_remain_intact(tmp_path, clock, provider, source, role, expected):
    evidence = ce.with_role_window(server(clock, provider=provider, source=source), role)
    assert resolve_window(evidence, 8192, root=tmp_path).total.tokens == expected


def test_new_server_read_releases_an_old_instances_refusal(tmp_path, clock):
    record_observation(tmp_path, "r", refused=40000)
    clock[0] += timedelta(seconds=1)
    evidence = server(clock)
    assert resolve_window(evidence, 8192, root=tmp_path).total.tokens == 80000
    clock[0] += timedelta(seconds=1)
    record_observation(tmp_path, "r", refused=60000)
    assert resolve_window(evidence, 8192, root=tmp_path).total.tokens == 60000


def test_expired_server_statement_cannot_cap_an_owner_setting(tmp_path, clock):
    evidence = ce.with_role_window(server(clock), 262000)
    clock[0] += timedelta(seconds=ce._CONFIRMED_TTL_SEC + 1)
    record = resolve_window(evidence, 8192, root=tmp_path)
    assert record.total.tokens == 262000 and record.contradicted_by is None


def test_local_task_start_rereads_but_round_resolution_uses_cache(tmp_path, monkeypatch, clock):
    import httpx

    from ouroboros import config, model_slots, vision_routing
    from ouroboros import context_fit as cf

    reads, window = [], [80000]
    route = dict(provider="openai-compatible", model="openai-compatible::fixture", model_role="main",
                 base_url="http://127.0.0.1:18080/v1", options={}, use_local=False)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(ce, "_route_api_key", lambda model: "")
    monkeypatch.setattr(cf, "_context_route", lambda task: (dict(route), {}))
    monkeypatch.setattr(model_slots, "model_role_option", lambda *a, **kw: 0)
    monkeypatch.setattr(vision_routing, "record_catalog_image_input", lambda *a, **kw: None)

    def get(url, **kw):
        reads.append((url, kw["timeout"]))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "data": [{"id": "fixture", "max_output_tokens": 4096, "meta": {"n_ctx": window[0]}}]})

    monkeypatch.setattr(httpx, "get", get)
    def start():
        return cf.RouteQuestion({}, resolver=cf.resolve_context_fit_route)({}, allow_fetch=True)[1]

    assert start().window_tokens == 80000
    assert len(reads) == 1, reads
    window[0] = 40000
    clock[0] += timedelta(seconds=1)
    assert cf.resolve_context_fit_route({}, allow_fetch=True)[1].window_tokens == 80000
    assert len(reads) == 1
    assert start().window_tokens == 40000
    assert len(reads) == 2 and all(timeout == 5 for _, timeout in reads)
    cf.resolve_context_fit_route({}, allow_fetch=True)
    assert len(reads) == 2


def test_real_probe_cache_keeps_read_identity_across_clock_rollback(tmp_path, monkeypatch, clock):
    import httpx
    reads = []
    route = dict(provider="openai-compatible", model="openai-compatible::fixture",
                 base_url="http://127.0.0.1:18080/v1", api_key="")
    def get(url, **kwargs):
        reads.append(url)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "data": [{"id": "fixture", "max_output_tokens": 4096, "meta": {"n_ctx": 80000}}]})
    monkeypatch.setattr(httpx, "get", get)
    first = ce.probe(tmp_path, **route)
    resolve_window(first, 8192, root=tmp_path)
    record_observation(tmp_path, first.route_fp, accepted=100000)
    clock[0] -= timedelta(hours=1)
    before_cache = len(reads)
    cached = ce.probe(tmp_path, **route, allow_fetch=False)
    assert len(reads) == before_cache
    assert not cached.stale
    assert cached.window_read_id == first.window_read_id
    assert resolve_window(cached, 8192, root=tmp_path).input.tokens == 100000
    second = ce.probe(tmp_path, **route, force=True)
    assert second.window_read_id != first.window_read_id
    assert resolve_window(second, 8192, root=tmp_path).input.tokens == 100000
    before_cache = len(reads)
    assert resolve_window(ce.probe(tmp_path, **route, allow_fetch=False), 8192, root=tmp_path).input.tokens == 100000
    assert len(reads) == before_cache
    # A new HTTP receipt is still the same serving claim.
    record_observation(tmp_path, first.route_fp, accepted=100000)
    third = ce.probe(tmp_path, **route, force=True)
    assert third.ts == second.ts and third.window_read_id != second.window_read_id
    assert resolve_window(third, 8192, root=tmp_path).input.tokens == 100000


def test_unchanged_reads_keep_one_frontier_and_do_not_rewrite(tmp_path, clock, monkeypatch):
    from uuid import uuid4
    saves = []
    original = ce._save
    monkeypatch.setattr(ce, "_save", lambda *args: (saves.append(1), original(*args))[1])
    for _ in range(1000):
        evidence = server(clock)
        evidence.window_read_id = uuid4().hex
        evidence.server_window["read_id"] = evidence.window_read_id
        assert resolve_window(evidence, 8192, root=tmp_path).total.tokens == 80000
        clock[0] += timedelta(seconds=1)
    assert len(saves) == 1
    assert len(ce._read(tmp_path)["main_windows"]["r"]["reads"]) == 1
    record_observation(tmp_path, "r", accepted=100000)
    old = evidence
    evidence = server(clock)
    assert resolve_window(evidence, 8192, root=tmp_path).input.tokens == 100000
    assert resolve_window(old, 8192, root=tmp_path).input.tokens == 100000
    assert len(ce._read(tmp_path)["main_windows"]["r"]["reads"]) == 1


def test_refreshed_equal_statement_retains_order_and_records_new_witness(tmp_path, clock):
    record_observation(tmp_path, 'r', accepted=200000)
    first = resolve_window(server(clock), 8192, root=tmp_path)
    clock[0] += timedelta(seconds=ce._CONFIRMED_TTL_SEC + 1)
    fresh = server(clock)
    assert resolve_window(fresh, 8192, root=tmp_path).total.sequence == first.total.sequence
    record_observation(tmp_path, 'r', accepted=100000)
    assert resolve_window(fresh, 8192, root=tmp_path).input.tokens == 100000


def test_expired_identical_refusal_gets_a_fresh_receipt(tmp_path, clock):
    evidence = ce.CapabilityEvidence(120000, 'asserted', 'owner_ack', 'r')
    record_observation(tmp_path, 'r', refused=80000)
    clock[0] += timedelta(seconds=ce._CONFIRMED_TTL_SEC + 1)
    record_observation(tmp_path, 'r', refused=80000)
    assert resolve_window(evidence, 8192, root=tmp_path).total.tokens == 80000


def test_only_a_changed_instance_releases_equal_size_refusal():
    from ouroboros.context_window import WindowLimit, window_authority
    first = WindowLimit(80000, 'total', 'server_metadata', sequence=1, instance_id='one')
    refusal = WindowLimit(40000, 'total', 'refusal', sequence=2)
    same = WindowLimit(80000, 'total', 'server_metadata', sequence=3, instance_id='one')
    changed = WindowLimit(80000, 'total', 'server_metadata', sequence=4, instance_id='two')
    assert window_authority([first, refusal, same])[0]['total'].tokens == 40000
    assert window_authority([first, refusal, changed])[0]['total'].tokens == 80000


@pytest.mark.parametrize('failure', ['lock', 'timeout', 'save'])
def test_unpublished_read_keeps_ceiling_over_historical_acceptance(tmp_path, clock, monkeypatch, failure):
    """Real four-second lock contention once; other failures use the store seam."""
    from contextlib import nullcontext
    record_observation(tmp_path, 'r', accepted=200000)
    evidence = server(clock)
    evidence.server_window = {}
    if failure == 'timeout':
        def unavailable(*args, **kwargs):
            raise TimeoutError('capability writer busy')
        monkeypatch.setattr(ce, '_store_transaction', unavailable)
    elif failure == 'save':
        monkeypatch.setattr(ce, '_save', lambda *args: False)
    lock = ce.json_update_lock(ce._store_path(tmp_path)) if failure == 'lock' else nullcontext()
    with lock:
        result = resolve_window(evidence, 8192, root=tmp_path)
    assert result.total is not None and result.total.tokens == 80000
    assert result.total.sequence == 0
    assert result.usable <= 80000 - 8192
    assert evidence.window_sequence == 0


def test_expired_frontier_survives_refusal_and_equal_cache_restore(tmp_path, clock):
    """An equal fresh receipt outlives the store frontier's timestamp."""
    from copy import deepcopy
    first = resolve_window(server(clock), 8192, root=tmp_path)
    clock[0] += timedelta(seconds=ce._CONFIRMED_TTL_SEC // 2)
    cached = server(clock)
    resolve_window(deepcopy(cached), 8192, root=tmp_path)
    clock[0] += timedelta(seconds=ce._CONFIRMED_TTL_SEC // 2 + 1)
    record_observation(tmp_path, 'r', refused=60000)
    result = resolve_window(deepcopy(cached), 8192, root=tmp_path)
    assert result.total.tokens == 60000
    frontier = ce._read(tmp_path)['main_windows']['r']['reads']
    assert len(frontier) == 1 and frontier[0]['sequence'] == first.total.sequence


def test_enumerated_publication_and_expiry_histories(tmp_path, monkeypatch):
    """41,371 histories after an initial read; 14 events, lengths zero to four.

    Compare persistence/recovery to full true chronology. Faults may restrict
    capacity, never enlarge it; histories without faults must be identical.
    Every prefix is checked, including the publication-failure call itself.
    """
    from itertools import product

    from tests._window_history import ALPHABET, ObserverHistory
    observer = ObserverHistory(monkeypatch, tmp_path)
    checked = ambiguous = widened = 0
    for length in range(5):
        for history in product(ALPHABET, repeat=length):
            observer.reset()
            for index, action in enumerate(history):
                actual, oracle = observer.observe(action)
                checked += 1
                # An unrecorded changed claim followed by a fresh transport read
                # can look like an equal replay of the persisted frontier.
                # Keep these histories in the alphabet; check their observable
                # equivalence explicitly in the companion information-loss test.
                if not observer.lost_order:
                    assert actual <= oracle, (history[:index + 1], actual, oracle)
                else:
                    ambiguous += 1
                    widened += actual > oracle
                    assert observer.faulted
                    assert actual <= observer.observable_oracle, (history[:index + 1], actual, observer.observable_oracle)
                if not observer.faulted:
                    assert actual == oracle, (history[:index + 1], actual, oracle)
    assert checked == 162302
    assert ambiguous == 6508
    assert widened == 700  # unrecorded claim excursions; pinned explicitly below



def test_unknown_publication_cannot_be_disproved_by_an_owner_witness():
    from ouroboros.context_window import WindowLimit, window_authority
    owner = WindowLimit(70000, 'total', 'user_setting')
    physical = WindowLimit(80000, 'total', 'server_metadata')
    accepted = WindowLimit(200000, 'input', 'accepted', sequence=1)
    limits, _ = window_authority([owner, physical, accepted])
    assert limits['total'] == physical


def test_failed_publication_then_process_loss_is_chronologically_ambiguous(tmp_path, monkeypatch):
    """The same retained inputs can mean either a replay or a genuinely changed read.

    Without storing the failed claim (which failed!) or an additional observer
    log, exact chronology after process loss is not recoverable. Keep this
    counterexample explicit rather than silently removing failed publications from enumeration.
    """
    from copy import deepcopy

    from tests._window_history import ObserverHistory
    def snapshot():
        # UUIDs distinguish receipts, but encode no chronology after restart.
        def without_ids(value):
            if isinstance(value, dict):
                return {k: without_ids(v) for k, v in value.items() if k not in {'read_id', 'window_read_id'}}
            if isinstance(value, list):
                return [without_ids(v) for v in value]
            return value
        return without_ids(deepcopy(observer.store)), without_ids(observer.receipt.to_json())
    observer = ObserverHistory(monkeypatch, tmp_path)
    observer.reset()
    observer.observe('fail120')
    observer.observe('refuse_input60')
    before = snapshot()[0]
    a = observer.observe('restart')
    lost = snapshot()
    observer.reset()
    observer.observe('refuse_input60')
    assert before == snapshot()[0]
    b = observer.observe('read120')
    assert lost == snapshot()
    assert a == (111808, 60000) and b == (111808, 111808)

    # An unrecorded excursion and return also looks like an equal replay.
    observer.reset()
    observer.observe('fail120')
    observer.observe('accept100')
    a = observer.observe('read80')
    lost = snapshot()
    observer.reset()
    observer.observe('accept100')
    b = observer.observe('refresh')
    assert lost == snapshot()
    assert a == (100000, 71808) and b == (100000, 100000)


@pytest.mark.parametrize("failure", ["lock", "save"])
@pytest.mark.parametrize("before_attempt", [False, True])
def test_cached_receipt_keeps_failed_attempt(tmp_path, monkeypatch, failure, before_attempt):
    from copy import deepcopy

    from tests._window_history import ObserverHistory
    observer = ObserverHistory(monkeypatch, tmp_path)
    observer.reset()
    observer.observe('fail120' if failure == 'lock' else 'save120')
    record = deepcopy(observer.receipt if before_attempt else observer.live).to_json()
    observer.observe('refuse_input60')
    restored = ce._cached_evidence(record, 'r', '', 'openai-compatible')
    result = resolve_window(restored, 8192, root=tmp_path)
    assert result.usable == 60000
    assert restored.window_sequence == 0
    assert restored.window_sequence_route == 'r'


def test_failed_receipt_cache_does_not_block_new_reads_or_acceptances(tmp_path, monkeypatch):
    from tests._window_history import ObserverHistory
    observer = ObserverHistory(monkeypatch, tmp_path)
    observer.reset()
    observer.observe('fail120')
    observer.observe('refuse_input60')
    before = observer.store.copy()
    assert observer.observe('cache') == (60000, 60000)
    assert observer.store == before  # recovery has no extra IO or persisted state
    assert observer.observe('read120')[0] == 111808
    assert observer.live.window_sequence > 0
    assert observer.observe('accept200')[0] == 200000


def _captured(record, window=80000):
    return SimpleNamespace(window_record=record, window_tokens=window, status="confirmed", provider="openai-compatible")


@pytest.mark.parametrize("other_read_first", [False, True])
def test_a_captured_plan_keeps_a_later_acceptance_whatever_read_came_between(tmp_path, clock, other_read_first):
    record_observation(tmp_path, "r", accepted=200000)
    plan = _captured(resolve_window(server(clock, 80000), 8192, root=tmp_path))
    if other_read_first:
        resolve_window(server(clock, 120000), 8192, root=tmp_path)
    record_observation(tmp_path, "r", accepted=100000)
    if not other_read_first:
        resolve_window(server(clock, 120000), 8192, root=tmp_path)
    assert resolve_window(plan, 8192, root=tmp_path).usable == 100000


def test_a_captured_serving_read_past_its_ttl_is_unknown(tmp_path, clock):
    plan = _captured(resolve_window(server(clock, 80000), 8192, root=tmp_path))
    assert resolve_window(plan, 8192, root=tmp_path).total.tokens == 80000
    clock[0] += timedelta(seconds=ce._CONFIRMED_TTL_SEC + 1)
    record = resolve_window(plan, 8192, root=tmp_path)
    assert record.total is None and record.admitted is None
