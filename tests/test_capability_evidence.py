"""Capability Evidence (v6.33.0 WS4): sourced, route-fingerprinted window proof."""

from __future__ import annotations

import json

import ouroboros.capability_evidence as ce


def test_route_fingerprint_stable_and_route_sensitive():
    a = ce.route_fingerprint(provider="anthropic", model="claude-opus-4-8", options={"beta": "1m"})
    b = ce.route_fingerprint(provider="anthropic", model="claude-opus-4-8", options={"beta": "1m"})
    assert a == b
    # Any route change yields a new fingerprint.
    assert a != ce.route_fingerprint(provider="anthropic", model="claude-opus-4-8")  # no beta
    assert a != ce.route_fingerprint(provider="anthropic", model="claude-opus-4-7", options={"beta": "1m"})
    assert a != ce.route_fingerprint(provider="openai", model="claude-opus-4-8", options={"beta": "1m"})


def test_route_fingerprint_does_not_embed_or_rotate_with_credentials():
    first = ce.route_fingerprint(
        provider="openai", model="gpt-5.5",
        headers={"Authorization": "Bearer secret-one", "X-Route-Beta": "1m"},
    )
    second = ce.route_fingerprint(
        provider="openai", model="gpt-5.5",
        headers={"Authorization": "Bearer secret-two", "X-Route-Beta": "1m"},
    )
    assert first == second
    assert first == ce.route_fingerprint(
        provider="openai", model="gpt-5.5",
        headers={"X-Route-Beta": "1m"},
    )
    assert first != ce.route_fingerprint(
        provider="openai", model="gpt-5.5",
        headers={"Authorization": "Bearer secret-two", "X-Route-Beta": "200k"},
    )


def test_unknown_is_fail_closed(tmp_path):
    ev = ce.probe(tmp_path, provider="anthropic", model="claude-opus-4-8", allow_fetch=False)
    assert ev.status == ce.STATUS_UNPROBEABLE
    assert ce.confirms_at_least(ev, ce.ONE_MILLION) is False
    assert ce.confirms_at_least(None) is False


def test_owner_ack_is_asserted_and_route_scoped(tmp_path):
    ce.record_owner_ack(tmp_path, provider="anthropic", model="claude-opus-4-8",
                        window_tokens=1_000_000, options={"beta": "1m"}, note="beta header on")
    ev = ce.probe(tmp_path, provider="anthropic", model="claude-opus-4-8", options={"beta": "1m"}, allow_fetch=False)
    assert ev.status == ce.STATUS_ASSERTED
    assert ev.window_tokens == 1_000_000
    assert ce.confirms_at_least(ev) is True
    # The SAME model on a DIFFERENT route (no beta) is NOT covered by the ack.
    other = ce.probe(tmp_path, provider="anthropic", model="claude-opus-4-8", allow_fetch=False)
    assert other.status == ce.STATUS_UNPROBEABLE
    assert ce.confirms_at_least(other) is False


def test_confirmed_probe_below_threshold_fails_closed(tmp_path):
    # Seed a confirmed-but-small window via owner-ack (asserted counts as known).
    ce.record_owner_ack(tmp_path, provider="gigachat", model="GigaChat-3-Ultra", window_tokens=131_072)
    ev = ce.probe(tmp_path, provider="gigachat", model="GigaChat-3-Ultra", allow_fetch=False)
    assert ev.window_tokens == 131_072
    assert ce.confirms_at_least(ev, ce.ONE_MILLION) is False  # known but < 1M
    assert ce.confirms_at_least(ev, 131_072) is True


def test_revoke_owner_ack(tmp_path):
    ce.record_owner_ack(tmp_path, provider="openai", model="gpt-5.5", window_tokens=1_000_000)
    fp = ce.route_fingerprint(provider="openai", model="gpt-5.5")
    assert any(a["route_fp"] == fp for a in ce.list_owner_acks(tmp_path))
    assert ce.revoke_owner_ack(tmp_path, fp) is True
    ev = ce.probe(tmp_path, provider="openai", model="gpt-5.5", allow_fetch=False)
    assert ev.status == ce.STATUS_UNPROBEABLE


def test_local_health_confirmed(tmp_path, monkeypatch):
    monkeypatch.setattr(ce, "_local_health_window", lambda model: 256_000)
    ev = ce.probe(tmp_path, provider="local", model="qwen", use_local=True, allow_fetch=True)
    assert ev.status == ce.STATUS_CONFIRMED
    assert ev.source == ce.SOURCE_LOCAL_HEALTH
    assert ev.window_tokens == 256_000


def test_provider_outage_keeps_prior_confirmed(tmp_path, monkeypatch):
    """A transient provider outage must NEVER erase a prior CONFIRMED record
    (module invariant) — it is kept, surfaced as stale (v6.33.0 P4)."""
    monkeypatch.setattr(ce, "_provider_metadata_window", lambda *a, **k: 1_000_000)
    ev1 = ce.probe(tmp_path, provider="openrouter", model="x/y", allow_fetch=True)
    assert ev1.status == ce.STATUS_CONFIRMED and ev1.window_tokens == 1_000_000
    # Provider now unreachable: metadata 0 + a transport failure.
    monkeypatch.setattr(ce, "_provider_metadata_window", lambda *a, **k: 0)
    monkeypatch.setattr(ce, "_metadata_fetch_transport_failed", lambda *a, **k: True)
    ev2 = ce.probe(tmp_path, provider="openrouter", model="x/y", allow_fetch=True, force=True)
    assert ev2.window_tokens == 1_000_000          # not erased
    assert ev2.status == ce.STATUS_CONFIRMED
    assert ev2.stale is True
    assert ce.confirms_at_least(ev2, ce.ONE_MILLION) is True


def test_transport_failure_records_status_failed(tmp_path, monkeypatch):
    """Provider unreachable (no prior record) -> STATUS_FAILED, distinct from a
    route with no metadata source. STATUS_FAILED is fail-closed for the >=1M gate."""
    monkeypatch.setattr(ce, "_provider_metadata_window", lambda *a, **k: 0)
    monkeypatch.setattr(ce, "_metadata_fetch_transport_failed", lambda *a, **k: True)
    ev = ce.probe(tmp_path, provider="openrouter", model="x/y", allow_fetch=True)
    assert ev.status == ce.STATUS_FAILED
    assert ce.confirms_at_least(ev, ce.ONE_MILLION) is False


def test_no_metadata_source_is_unprobeable_not_failed(tmp_path, monkeypatch):
    """A route with no metadata source and no outage stays UNPROBEABLE (owner-ack
    path), NOT FAILED — the two must not be conflated."""
    monkeypatch.setattr(ce, "_provider_metadata_window", lambda *a, **k: 0)
    monkeypatch.setattr(ce, "_metadata_fetch_transport_failed", lambda *a, **k: False)
    ev = ce.probe(tmp_path, provider="anthropic", model="claude-opus-4-8", allow_fetch=True)
    assert ev.status == ce.STATUS_UNPROBEABLE


def test_openrouter_metadata_retries_after_transport_failure(monkeypatch):
    """A failed /models fetch is NOT one-shot-poisoned: the next allow_fetch=True
    probe retries, and a model unresolved while the provider is unreachable reads
    as a transport failure (not silently unprobeable) (v6.33.0 triad fix)."""
    from ouroboros.llm import LLMClient

    monkeypatch.setattr(LLMClient, "_SUPPORTED_PARAMS_FETCHED", False)
    monkeypatch.setattr(LLMClient, "_CAPABILITIES_FETCH_OK", False)
    monkeypatch.setattr(LLMClient, "_CONTEXT_LENGTH_CACHE", {})
    state = {"n": 0}

    def fake_fetch():
        state["n"] += 1
        LLMClient._SUPPORTED_PARAMS_FETCHED = True
        if state["n"] == 1:
            LLMClient._CAPABILITIES_FETCH_OK = False  # provider unreachable
        else:
            LLMClient._CAPABILITIES_FETCH_OK = True
            LLMClient._CONTEXT_LENGTH_CACHE["x/y"] = 1_000_000

    monkeypatch.setattr(LLMClient, "_fetch_openrouter_capabilities", fake_fetch)
    # 1st probe: fetch attempted but failed -> 0 + transport-failed signalled.
    assert LLMClient.openrouter_context_length("x/y") == 0
    assert state["n"] == 1
    assert LLMClient.metadata_fetch_attempted_and_failed() is True
    # 2nd probe: retries (the prior fetch failed), provider recovered, model present.
    assert LLMClient.openrouter_context_length("x/y") == 1_000_000
    assert state["n"] == 2
    assert LLMClient.metadata_fetch_attempted_and_failed() is False


# --- CW6: OpenAI-compatible /models metadata probe (vLLM/Ollama/...) ---

def test_openai_compatible_metadata_window_parses_max_model_len(monkeypatch):

    import ouroboros.config as cfg

    monkeypatch.setattr(cfg, "load_settings", lambda: {"OPENAI_COMPATIBLE_API_KEY": "k"})

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [
                {"id": "other-model", "max_model_len": 8192},
                {"id": "my-model", "max_model_len": 1048576},
            ]}

    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())

    win = ce._openai_compatible_metadata_window("my-model", "http://localhost:8000/v1", allow_fetch=True)
    assert win == 1048576
    # The saved model is normally provider-prefixed; /models lists the BARE id (CW6 fix).
    win_prefixed = ce._openai_compatible_metadata_window(
        "openai-compatible::my-model", "http://localhost:8000/v1", allow_fetch=True)
    assert win_prefixed == 1048576


def test_openai_compatible_metadata_window_fail_closed(monkeypatch):
    import httpx

    # hot path (allow_fetch=False) and no base_url => no network, 0.
    assert ce._openai_compatible_metadata_window("m", "http://x/v1", allow_fetch=False) == 0
    assert ce._openai_compatible_metadata_window("m", "", allow_fetch=True) == 0

    # transport error => fail-closed to 0 (never raises).
    def _boom(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("ouroboros.config.load_settings", lambda: {})
    monkeypatch.setattr(httpx, "get", _boom)
    assert ce._openai_compatible_metadata_window("m", "http://x/v1", allow_fetch=True) == 0


def test_provider_metadata_window_routes_openai_compatible(monkeypatch):
    seen = {}

    def _fake(model, base_url, allow_fetch, api_key=None):
        seen["hit"] = (model, base_url)
        return 4096

    monkeypatch.setattr(ce, "_openai_compatible_metadata_window", _fake)
    win = ce._provider_metadata_window("openai-compatible", "m", "http://x/v1", allow_fetch=True)
    assert win == 4096 and seen["hit"] == ("m", "http://x/v1")
    # gigachat stays unprobeable (no per-model window in its /models)
    assert ce._provider_metadata_window("gigachat", "GigaChat", "", allow_fetch=True) == 0


def test_probe_threads_api_key_through_metadata_and_generative(tmp_path, monkeypatch):
    """First-run onboarding passes the in-flight OPENAI_COMPATIBLE_API_KEY to
    probe(api_key=...) because it is not yet on disk. Confirm the key reaches BOTH
    the metadata probe and (on fall-through) the generative probe for an
    openai-compatible route."""
    seen = {}

    def _fake_meta(model, base_url, allow_fetch, api_key=None):
        seen["meta_key"] = api_key
        return 0  # force fall-through to the generative probe

    def _fake_gen(provider, model, base_url="", api_key=None):
        seen["gen_key"] = api_key
        return 0, ce.STATUS_UNPROBEABLE, "stub"

    monkeypatch.setattr(ce, "_openai_compatible_metadata_window", _fake_meta)
    monkeypatch.setattr(ce, "_generative_probe_window", _fake_gen)
    ce.probe(
        tmp_path, provider="openai-compatible", model="m", base_url="http://x/v1",
        allow_fetch=True, allow_generative=True, force=True, api_key="THEKEY",
    )
    assert seen["meta_key"] == "THEKEY"
    assert seen["gen_key"] == "THEKEY"


# --- v6.46.0: generative context-window probe + cloudru base_url fix ---

def test_classify_generative_probe_response_no_network():
    from ouroboros.capability_evidence import (
        classify_generative_probe_response as C,
        STATUS_CONFIRMED, STATUS_UNPROBEABLE, STATUS_FAILED,
    )
    # 4xx overflow reject WITH a parseable limit -> CONFIRMED at that number (free path).
    win, st, _ = C(400, "This model's maximum context length is 1048576 tokens. However you requested 2000000 tokens")
    assert (win, st) == (1048576, STATUS_CONFIRMED)
    win, st, _ = C(400, "Input length (160062 tokens) is longer than the model's context length (59862 tokens)")
    assert (win, st) == (59862, STATUS_CONFIRMED)
    # 4xx WITHOUT a number (e.g. Zhipu code 1261, or a 413 size reject) -> owner-ack.
    assert C(400, "error code 1261 prompt too long")[1] == STATUS_UNPROBEABLE
    # 200: the oversized input was ACCEPTED (possibly paid) -> never auto-confirm; owner-ack.
    assert C(200, "", canaries=["A"], echoed_text="A", usage_prompt_tokens=100, sent_token_estimate=100)[1] == STATUS_UNPROBEABLE
    # transport / 5xx / unknown -> transient FAILED.
    assert C(503, "bad gateway")[1] == STATUS_FAILED
    assert C(None, "timeout")[1] == STATUS_FAILED


def test_active_main_route_sets_cloudru_base_url():
    from ouroboros.gateway.settings import _active_main_route
    route = _active_main_route({
        "OUROBOROS_MODEL": "cloudru::glm-5.2",
        "CLOUDRU_FOUNDATION_MODELS_BASE_URL": "https://foundation-models.api.cloud.ru/v1",
    })
    assert route["provider"] == "cloudru"
    # The bug was base_url='' for cloudru (only openai/openai-compatible/gigachat set it).
    assert route["base_url"] == "https://foundation-models.api.cloud.ru/v1"


# --- v6.80.0: measured tokenizer density (token_density namespace) --------------

def test_token_density_is_measured_throttled_and_bounded(tmp_path, monkeypatch):
    """RS4: observations are measured, conservative (MAX of fresh pairs), bounded in
    retention, and throttled — the store is one file behind one lock shared with the
    scope-review path, and a lock-starvation incident under bench load is on record."""
    from ouroboros.capability_evidence import (
        COLD_START_TOKEN_DENSITY,
        MEASURED_DENSITY_SAFETY_FACTOR,
        _DENSITY_MEMO,
        _TOKEN_DENSITY_MAX_PAIRS,
        get_token_density,
        record_token_density,
        resolve_token_density,
    )

    _DENSITY_MEMO.clear()
    # No observation anywhere -> the documented conservative cold-start density.
    assert get_token_density(tmp_path, "some/model") == 0.0
    density, source = resolve_token_density(tmp_path, "some/model")
    assert source == "cold_conservative"
    assert density == COLD_START_TOKEN_DENSITY

    chars = 4_000_000  # 1,000,000 estimated tokens
    record_token_density(tmp_path, "m/one", prompt_chars=chars, prompt_tokens=1_500_000)
    assert abs(get_token_density(tmp_path, "m/one") - 1.5) < 1e-6
    density, source = resolve_token_density(tmp_path, "m/one")
    assert source == "measured"
    # Successor pin (issue #284): a fresh exact-model witness is authoritative
    # and MAY lower the effective review density below the cold floor; the
    # floor keeps governing only stale/absent/cross-model evidence.
    assert density == 1.5 * MEASURED_DENSITY_SAFETY_FACTOR
    assert density < COLD_START_TOKEN_DENSITY

    # Above the floor, the measured value (with the safety factor) is what applies.
    _DENSITY_MEMO.clear()
    record_token_density(tmp_path, "m/dense", prompt_chars=chars, prompt_tokens=2_000_000)
    density, source = resolve_token_density(tmp_path, "m/dense")
    assert source == "measured"
    assert abs(density - 2.0 * MEASURED_DENSITY_SAFETY_FACTOR) < 1e-6

    # A DENSER observation always wins (conservative); a lighter one never lowers it
    # inside the window.
    _DENSITY_MEMO.clear()
    record_token_density(tmp_path, "m/one", prompt_chars=chars, prompt_tokens=1_800_000)
    assert abs(get_token_density(tmp_path, "m/one") - 1.8) < 1e-6
    _DENSITY_MEMO.clear()
    record_token_density(tmp_path, "m/one", prompt_chars=chars, prompt_tokens=1_100_000)
    assert abs(get_token_density(tmp_path, "m/one") - 1.8) < 1e-6

    # Raw-pair retention is bounded.
    for extra in range(12):
        _DENSITY_MEMO.clear()
        record_token_density(
            tmp_path, "m/one", prompt_chars=chars, prompt_tokens=1_200_000 + extra * 40_000,
        )
    entry = json.loads((tmp_path / "state" / "capability_evidence.json").read_text())
    assert len(entry["token_density"]["m/one"]["pairs"]) <= _TOKEN_DENSITY_MAX_PAIRS

    # Throttle: a repeat observation within tolerance must not touch the shared store
    # AT ALL (it is one file behind one lock shared with the scope-review hot path).
    saves: list = []
    monkeypatch.setattr(ce, "_save", lambda root, data: saves.append(1))
    _DENSITY_MEMO.clear()
    record_token_density(
        tmp_path, "m/one", prompt_chars=chars, prompt_tokens=int(1.64 * chars / 4),
    )
    assert saves == [], "a within-tolerance repeat must not write the shared store"
    # ...and a real drift IS persisted.
    _DENSITY_MEMO.clear()
    record_token_density(tmp_path, "m/one", prompt_chars=chars, prompt_tokens=int(3.0 * chars / 4))
    assert len(saves) == 1
    monkeypatch.undo()

    # Junk and too-small observations are ignored (fixed scaffolding dominates them).
    _DENSITY_MEMO.clear()
    record_token_density(tmp_path, "m/two", prompt_chars=100, prompt_tokens=90)
    record_token_density(tmp_path, "m/two", prompt_chars=chars, prompt_tokens=0)
    record_token_density(tmp_path, "m/two", prompt_chars=chars, prompt_tokens=99_000_000)
    assert get_token_density(tmp_path, "m/two") == 0.0

    # The cold-conservative fallback IS the constant: the fresh m/one (1.8) and
    # m/dense (2.0) witnesses are other tokenizers, no evidence about never/seen —
    # neither optimistic (a light foreign row) nor pessimistic (a dense one).
    density, source = resolve_token_density(tmp_path, "never/seen")
    assert source == "cold_conservative"
    assert density == COLD_START_TOKEN_DENSITY


def test_density_reducers_use_newest_route_or_model_but_densest_review_witness(
    tmp_path, monkeypatch,
):
    from datetime import datetime, timezone

    from ouroboros.capability_evidence import (
        _DENSITY_MEMO,
        COLD_START_TOKEN_DENSITY,
        MEASURED_DENSITY_SAFETY_FACTOR,
        record_token_density,
        resolve_main_token_density,
        resolve_review_token_density,
    )

    # Windows clock granularity can stamp back-to-back records identically.
    # Persisted observation order, not test-synthesised time, breaks that tie.
    base = datetime.now(timezone.utc)
    monkeypatch.setattr(ce, "utc_now_iso", lambda: base.isoformat())

    _DENSITY_MEMO.clear()
    chars = 400_000
    record_token_density(
        tmp_path, "m/one", prompt_chars=chars, prompt_tokens=180_000, route_fp="route-a",
        basis="bounded_proxy",
    )
    record_token_density(
        tmp_path, "m/one", prompt_chars=chars, prompt_tokens=110_000, route_fp="route-a",
        basis="bounded_proxy",
    )

    assert resolve_main_token_density(tmp_path, "route-a", "m/one") == (
        1.1, "fresh_route_usage",
    )
    assert resolve_main_token_density(tmp_path, "missing-route", "m/one") == (
        1.1, "fresh_model_usage",
    )
    review, source = resolve_review_token_density(tmp_path, "m/one")
    assert source == "measured"
    assert review == max(COLD_START_TOKEN_DENSITY, 1.8 * MEASURED_DENSITY_SAFETY_FACTOR)
    # m/one's 1.8 x 1.05 = 1.89 is m/one's tokenizer; m/missing gets the floor.
    assert resolve_review_token_density(tmp_path, "m/missing") == (
        COLD_START_TOKEN_DENSITY, "cold_conservative",
    )

    stored = json.loads((tmp_path / "state" / "capability_evidence.json").read_text())
    assert set(stored["token_density"]["m/one"]) == {"pairs"}
    assert resolve_main_token_density(tmp_path, "", "never/seen") == (1.0, "cold_estimate")


def test_foreign_density_witness_leaves_the_cold_floor_untouched(tmp_path):
    """Paid run 2026-09-04 (lane SM1_a1): gemini-3.8-flash's witness 407,767 /
    (900,708 / 4) = 1.81 governed gpt-5.6-terra (no exact witness yet) at
    1.81 x 1.05 = 1.90, cutting the scope cap to 499,627 of a 1,050,000 window
    where the floor alone gives 575,757. Another model's tokenizer is no
    evidence about this route: the cross-model branch returns the floor."""
    from ouroboros.capability_evidence import (
        _DENSITY_MEMO,
        COLD_START_TOKEN_DENSITY,
        MEASURED_DENSITY_SAFETY_FACTOR,
        record_token_density,
        resolve_review_token_density,
    )

    _DENSITY_MEMO.clear()
    record_token_density(
        tmp_path, "google/gemini-3.8-flash", prompt_chars=900_708, prompt_tokens=407_767,
    )
    assert resolve_review_token_density(tmp_path, "openai/gpt-5.6-terra") == (
        COLD_START_TOKEN_DENSITY, "cold_conservative",
    )
    density, source = resolve_review_token_density(tmp_path, "google/gemini-3.8-flash")
    assert source == "measured"
    assert abs(density - 407_767 / (900_708 / 4) * MEASURED_DENSITY_SAFETY_FACTOR) < 1e-9


def test_doubled_cached_usage_row_is_not_a_density_witness(tmp_path):
    """The same paid run: 22 openrouter gemini usage rows (every lane) reported
    prompt_tokens = 2 x cached_tokens - 1 — a cache read added on top of an
    already cache-inclusive total — beside honest rows of the same prompt at
    1.00-1.17 x cached. Accepted, one such row is the densest fresh pair and
    the review reducer promotes it for the 90-day TTL, so it is no witness.
    Uncached ABOVE cached (a cold prompt with a small cache hit) is ordinary
    accounting and stays measurable."""
    from ouroboros import usage_accounting as ua
    from ouroboros.capability_evidence import _DENSITY_MEMO, get_token_density, observe_token_density

    def observe(root, usage, estimate):
        _DENSITY_MEMO.clear()
        observe_token_density(
            ua.AttemptRequest(
                model="google/gemini-3.8-flash", provider="openrouter",
                prompt_tokens_estimate=estimate, drive_root=root,
            ),
            usage,
            drive_root_resolver=lambda root: root,
        )
        return get_token_density(root, "google/gemini-3.8-flash")

    doubled = tmp_path / "doubled"
    assert observe(doubled, {"prompt_tokens": 407_767, "cached_tokens": 203_884}, 225_177) == 0.0
    assert observe(doubled, {"prompt_tokens": 407_768, "cached_tokens": 203_884}, 225_177) == 0.0
    honest = observe(tmp_path / "honest", {"prompt_tokens": 205_299, "cached_tokens": 203_884}, 226_443)
    assert abs(honest - 205_299 / 226_443) < 1e-6
    cold_hit = observe(tmp_path / "cold_hit", {"prompt_tokens": 412_884, "cached_tokens": 203_884}, 225_177)
    assert abs(cold_hit - 412_884 / 225_177) < 1e-6


def test_density_throttle_uses_legacy_list_order_for_equal_clock_ties(tmp_path):
    from ouroboros.capability_evidence import _DENSITY_MEMO, record_token_density

    observed_at = ce.utc_now_iso()
    ce._save(tmp_path, {"token_density": {"m/one": {"pairs": [
        {
            "prompt_chars": 400_000,
            "prompt_tokens": 100_000,
            "observed_at": observed_at,
            "route_fp": "route-a",
        },
        {
            "prompt_chars": 400_000,
            "prompt_tokens": 200_000,
            "observed_at": observed_at,
            "observation_seq": "invalid-legacy-value",
            "route_fp": "route-a",
        },
    ]}}})

    _DENSITY_MEMO.clear()
    record_token_density(
        tmp_path, "m/one", prompt_chars=400_000, prompt_tokens=200_000,
        route_fp="route-a",
    )
    stored = json.loads((tmp_path / "state" / "capability_evidence.json").read_text())
    assert len(stored["token_density"]["m/one"]["pairs"]) == 2

    _DENSITY_MEMO.clear()
    record_token_density(
        tmp_path, "m/one", prompt_chars=400_000, prompt_tokens=100_000,
        route_fp="route-a",
    )
    stored = json.loads((tmp_path / "state" / "capability_evidence.json").read_text())
    assert len(stored["token_density"]["m/one"]["pairs"]) == 3


def test_density_memo_does_not_claim_a_witness_when_persistence_failed(tmp_path, monkeypatch):
    from ouroboros.capability_evidence import _DENSITY_MEMO, record_token_density

    _DENSITY_MEMO.clear()
    monkeypatch.setattr(ce, "_save", lambda root, data: False)
    record_token_density(
        tmp_path, "m/one", prompt_chars=400_000, prompt_tokens=150_000,
        route_fp="route-a",
    )
    assert "m/one\0route-a" not in _DENSITY_MEMO


def test_orphan_density_scalar_has_no_authority_without_a_retained_witness(tmp_path):
    from ouroboros.capability_evidence import (
        COLD_START_TOKEN_DENSITY,
        get_token_density,
        resolve_main_token_density,
        resolve_review_token_density,
    )

    ce._save(tmp_path, {"token_density": {"m/one": {
        "density": 3.0,
        "observed_at": ce.utc_now_iso(),
        "pairs": [],
    }}})
    assert get_token_density(tmp_path, "m/one") == 0.0
    assert resolve_main_token_density(tmp_path, "route-a", "m/one") == (1.0, "cold_estimate")
    assert resolve_review_token_density(tmp_path, "m/one") == (
        COLD_START_TOKEN_DENSITY, "cold_conservative",
    )


def test_density_retention_preserves_fresh_high_witness_without_refreshing_its_ttl(
    tmp_path, monkeypatch,
):
    import datetime

    from ouroboros.capability_evidence import (
        _DENSITY_MEMO,
        MEASURED_DENSITY_SAFETY_FACTOR,
        _TOKEN_DENSITY_MAX_PAIRS,
        _TOKEN_DENSITY_TTL_SEC,
        record_token_density,
        resolve_main_token_density,
        resolve_review_token_density,
    )

    _DENSITY_MEMO.clear()
    now = datetime.datetime.now(datetime.timezone.utc)
    old_high_ts = (now - datetime.timedelta(seconds=_TOKEN_DENSITY_TTL_SEC - 60)).isoformat()
    ce._save(tmp_path, {"token_density": {"m/one": {"pairs": [{
        "prompt_chars": 400_000,
        "prompt_tokens": 250_000,
        "observed_at": old_high_ts,
        "source": "dispatch_usage",
        "route_fp": "route-high",
    }]}}})
    monkeypatch.setattr(ce, "utc_now_iso", lambda: now.isoformat())

    for index, density in enumerate((1.0, 1.1, 1.2, 1.3, 1.4, 1.5)):
        _DENSITY_MEMO.clear()
        record_token_density(
            tmp_path, "m/one", prompt_chars=400_000,
            prompt_tokens=int(density * 100_000), route_fp=f"route-low-{index}",
            basis="bounded_proxy",
        )
    stored = json.loads((tmp_path / "state" / "capability_evidence.json").read_text())
    pairs = stored["token_density"]["m/one"]["pairs"]
    assert len(pairs) == _TOKEN_DENSITY_MAX_PAIRS
    assert any(pair["observed_at"] == old_high_ts for pair in pairs)
    fresh_sequences = [
        pair["observation_seq"] for pair in pairs if pair["observed_at"] != old_high_ts
    ]
    assert fresh_sequences and all(sequence > 0 for sequence in fresh_sequences)

    monkeypatch.setattr(ce, "utc_now", lambda: now + datetime.timedelta(seconds=120))
    assert resolve_main_token_density(tmp_path, "route-high", "m/one") == (
        1.5, "fresh_model_usage",
    )
    # Successor pin (issue #284): the densest FRESH exact-model witness (1.5,
    # with the safety factor) undercuts the cold floor and is honored; the
    # stale 2.5 witness no longer counts toward review sizing.
    assert resolve_review_token_density(tmp_path, "m/one") == (
        1.5 * MEASURED_DENSITY_SAFETY_FACTOR, "measured",
    )
    _DENSITY_MEMO.clear()
    record_token_density(
        tmp_path, "m/one", prompt_chars=400_000, prompt_tokens=155_000, route_fp="route-new",
    )
    stored = json.loads((tmp_path / "state" / "capability_evidence.json").read_text())
    assert all(pair["observed_at"] != old_high_ts for pair in stored["token_density"]["m/one"]["pairs"])


def test_token_density_is_concurrency_safe_under_bench_like_parallelism(tmp_path):
    """The density writer must not corrupt or lose the shared store under the kind of
    parallelism a benchmark server produces."""
    import threading

    from ouroboros.capability_evidence import (
        _DENSITY_MEMO, get_token_density, record_token_density,
    )

    _DENSITY_MEMO.clear()
    errors: list = []

    def worker(index: int) -> None:
        try:
            for _ in range(15):
                _DENSITY_MEMO.clear()
                record_token_density(
                    tmp_path, f"m/{index}",
                    prompt_chars=4_000_000, prompt_tokens=1_400_000 + index * 10_000,
                )
        except Exception as exc:  # pragma: no cover - the assertion below reports it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    data = json.loads((tmp_path / "state" / "capability_evidence.json").read_text())
    assert len(data["token_density"]) == 8
    for index in range(8):
        assert get_token_density(tmp_path, f"m/{index}") > 0


def test_cold_density_never_demotes_the_main_loop_context_fit(tmp_path, monkeypatch):
    """Д7 CRITICAL decoupling: the conservative cold-start density is for review-pack
    sizing ONLY. Every fresh install and EVERY isolated benchmark server starts with an
    empty observation store, so applying it to the main loop would silently demote
    `initial_mode` from Max to Low and lose information (BIBLE P1)."""
    from ouroboros import context_fit
    from ouroboros.capability_evidence import _DENSITY_MEMO, record_token_density

    _DENSITY_MEMO.clear()
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)

    # Empty store: a no-observation route keeps the neutral 1.0 baseline.
    assert context_fit._route_calibration_ratio(tmp_path, "fp", "openai/gpt-5.5") == 1.0
    assert context_fit._route_calibration_ratio(tmp_path, "fp", "anthropic/claude-fable-5") == 1.0

    # Only a MEASURED density for that exact model may raise it.
    record_token_density(
        tmp_path, "anthropic/claude-fable-5", prompt_chars=4_000_000, prompt_tokens=1_580_000,
        basis="bounded_proxy",
    )
    assert context_fit._route_calibration_ratio(tmp_path, "fp", "openai/gpt-5.5") == 1.0
    assert context_fit._route_calibration_ratio(
        tmp_path, "fp", "anthropic/claude-fable-5"
    ) > 1.5


def test_no_observation_non_claude_main_route_keeps_todays_initial_mode(tmp_path, monkeypatch):
    """Named anti-regression test: empty store + KNOWN window + non-Claude main route
    ⇒ `initial_mode` identical to today's (Max)."""
    from types import SimpleNamespace

    from ouroboros import context_fit
    from ouroboros.capability_evidence import _DENSITY_MEMO

    _DENSITY_MEMO.clear()
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    core = context_fit.ContextCore(
        base_prompt="p", bible_md="b", architecture_md="a", development_md="d",
        semi_stable_text="s", dynamic_text="y", user_content_json='"u"',
        docs_need_development=False,
    )
    env = SimpleNamespace(drive_root=str(tmp_path))
    monkeypatch.setattr(context_fit, "reference_doc_sections", lambda *a, **k: [])

    def resolver(task, *, allow_fetch):
        return (
            {"provider": "openai", "model": "openai/gpt-5.5", "base_url": "", "use_local": False},
            SimpleNamespace(route_fp="fp", status="confirmed", stale=False, window_tokens=400_000),
        )

    plan = context_fit.build_context_fit_plan(
        env, core, {}, preferred_mode="max", route_resolver=resolver,
    )
    assert plan.initial_mode == "max"
    assert plan.max_projection.calibration_ratio == 1.0


def test_first_successful_call_seeds_density_so_the_next_projection_is_measured(
    tmp_path, monkeypatch,
):
    """Bounds the cold-baseline exposure to a SINGLE round (v6.80.0, Д7).

    The main-loop baseline for an unmeasured route is the neutral 1.0, so a cold
    Claude-family route is no longer proactively demoted max->low. This test verifies
    the protection comes back by MEASUREMENT rather than by guess: round 1 dispatches
    at baseline 1.0, the first settled send records this model's density through the
    real ``execute_physical_attempt`` boundary, and the very next
    ``build_context_fit_plan`` for the same model already carries a measured baseline.

    Critically, the recorded send carries CACHE tokens — the main loop always marks a
    stable prefix — on a cache-INCLUSIVE provider (OpenRouter, whose prompt_tokens
    semantics `pricing.py` already assumes). An over-broad skip-on-any-cache-token
    rule would make the whole measurement path vacuous and freeze every projection and
    review pack at the cold density forever, which would be a much wider exposure than
    one round.

    RESIDUAL EXPOSURE (accepted, disclosed): the only window left open is a FIRST
    round whose prompt ALREADY exceeds the calibrated window on a fresh evidence
    store. That round fails ``infra_failed`` with ``context_overflow``; ``loop.py``
    then reprojects the transcript into task-local Low exactly once and retries. From
    the next projection onward the density is measured, so the exposure does not
    recur for that model.
    """
    from types import SimpleNamespace

    from ouroboros import context_fit
    from ouroboros.capability_evidence import _DENSITY_MEMO, get_token_density
    from ouroboros.usage_accounting import AttemptRequest, execute_physical_attempt

    _DENSITY_MEMO.clear()
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TOTAL_BUDGET", "500")  # the reservation rail is not under test
    # One observation store: the plan builder reads density from the CANONICAL
    # evidence root, so bind it to this test's root before seeding witnesses.
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    model = "anthropic/claude-fable-5"  # Claude-family, routed through OpenRouter
    core = context_fit.ContextCore(
        base_prompt="p", bible_md="b", architecture_md="a", development_md="d",
        semi_stable_text="s", dynamic_text="y", user_content_json='"u"',
        docs_need_development=False,
    )
    env = SimpleNamespace(drive_root=str(tmp_path))
    monkeypatch.setattr(context_fit, "reference_doc_sections", lambda *a, **k: [])

    def resolver(task, *, allow_fetch):
        return (
            {"provider": "openrouter", "model": model, "base_url": "", "use_local": False},
            SimpleNamespace(
                route_fp="fp-cold", status="confirmed", stale=False, window_tokens=1_000_000,
            ),
        )

    def _plan():
        return context_fit.build_context_fit_plan(
            env, core, {}, preferred_mode="max", route_resolver=resolver,
        )

    # --- Round 1: no observation anywhere -> neutral baseline, NO demotion.
    assert get_token_density(tmp_path, model) == 0.0
    first = _plan()
    assert first.max_projection.calibration_ratio == 1.0
    assert first.initial_mode == "max"

    # --- The first settled send records the density through the REAL boundary,
    # cache tokens and all.
    request = AttemptRequest(
        model=model,
        provider="openrouter",
        prompt_tokens_estimate=50_000,     # == 200,000 prompt chars
        # The modern producer stamps the fit-estimator basis (no images here,
        # so it equals the raw estimate); a 0 legacy field records basis="raw",
        # which the MAIN fit resolver deliberately refuses to calibrate on.
        prompt_tokens_bounded_estimate=50_000,
        max_completion_tokens=1024,
        drive_root=tmp_path,
    )
    response = SimpleNamespace(usage={
        "prompt_tokens": 80_000,           # cache-INCLUSIVE (pricing.py's assumption)
        "completion_tokens": 10,
        "cached_tokens": 45_000,           # the stable governance prefix was cached
        "cache_write_tokens": 600,         # ...and part of it was written this call
    })
    execute_physical_attempt(
        request,
        lambda: response,
        extractor=lambda resp: (dict(resp.usage), 0.0, True),
    )

    measured = get_token_density(tmp_path, model)
    assert measured > 1.0, (
        "a cache-bearing send on a cache-inclusive provider must still be measurable; "
        f"got {measured}"
    )
    assert abs(measured - 1.6) < 1e-6

    # --- Round 2 for the SAME model: the projection is now measured, not guessed.
    second = _plan()
    assert second.max_projection.calibration_ratio > 1.0
    assert abs(second.max_projection.calibration_ratio - 1.6) < 1e-6
    assert second.max_projection.calibrated_tokens == int(
        second.max_projection.estimated_tokens * second.max_projection.calibration_ratio
    )

    # A route whose prompt_tokens semantics are NOT known cache-inclusive stays
    # skipped while cache tokens are present (a falsely low density would LOOSEN the
    # review-pack cap — the only dangerous direction). GigaChat is that route: its
    # `precached_prompt_tokens` semantics are undocumented.
    #
    # This example used to be direct-Anthropic, and that was correct when this test was
    # written: the path reported bare `input_tokens` then. v6.77.0 made it fold cache
    # reads and writes into `prompt_tokens`, so the premise expired inside this very
    # release and the assertion became a pin on a stale world rather than on a contract.
    _DENSITY_MEMO.clear()
    execute_physical_attempt(
        AttemptRequest(
            model="gigachat-model", provider="gigachat",
            prompt_tokens_estimate=50_000, max_completion_tokens=1024,
            drive_root=tmp_path,
        ),
        lambda: response,
        extractor=lambda resp: (dict(resp.usage), 0.0, True),
    )
    assert get_token_density(tmp_path, "gigachat-model") == 0.0

    # ...and the direct-Anthropic route, now cache-inclusive, IS measured. Leaving it
    # out made the measurement vacuous on the main and heavy slots, which are exactly
    # the routes this machinery exists to calibrate.
    _DENSITY_MEMO.clear()
    execute_physical_attempt(
        AttemptRequest(
            model="direct-anthropic-model", provider="anthropic",
            prompt_tokens_estimate=50_000, max_completion_tokens=1024,
            drive_root=tmp_path,
        ),
        lambda: response,
        extractor=lambda resp: (dict(resp.usage), 0.0, True),
    )
    assert abs(get_token_density(tmp_path, "direct-anthropic-model") - 1.6) < 1e-6


def test_freshness_is_an_explicit_axis_of_the_owned_predicate(tmp_path, monkeypatch):
    """The >=1M question and the "is this still current" question are ONE predicate
    with an explicit knob, not one predicate plus a re-derivation at each call site.

    `probe` already documents that a stale record "reads as unknown for >=1M gates",
    but `confirms_at_least` never checked `stale` — so the gate that AUTHORIZES a
    blocking review accepted an outage-carried record, while the gate beside it
    hand-rolled the freshness test in a local boolean."""
    monkeypatch.setattr(ce, "_provider_metadata_window", lambda *a, **k: 1_000_000)
    fresh = ce.probe(tmp_path, provider="openrouter", model="x/y", allow_fetch=True)
    assert fresh.stale is False
    assert ce.is_known(fresh, require_fresh=True) is True
    assert ce.confirms_at_least(fresh, ce.ONE_MILLION, require_fresh=True) is True

    # Provider now unreachable: the prior record is KEPT (module invariant) but stale.
    monkeypatch.setattr(ce, "_provider_metadata_window", lambda *a, **k: 0)
    monkeypatch.setattr(ce, "_metadata_fetch_transport_failed", lambda *a, **k: True)
    stale = ce.probe(tmp_path, provider="openrouter", model="x/y", allow_fetch=True, force=True)
    assert stale.stale is True and stale.window_tokens == 1_000_000

    # Restricting gates keep it (never downgrade the owner's horizon on a blip)...
    assert ce.confirms_at_least(stale, ce.ONE_MILLION) is True
    # ...authorizing gates do not.
    assert ce.confirms_at_least(stale, ce.ONE_MILLION, require_fresh=True) is False
    assert ce.is_known(stale, require_fresh=True) is False
    assert ce.is_known(stale) is True

    # A window of 0 is never "known", whatever the status claims.
    assert ce.is_known(ce.CapabilityEvidence(0, ce.STATUS_CONFIRMED, "", "")) is False
    assert ce.is_known(None) is False


def test_no_surface_restates_the_freshness_predicate_it_does_not_own():
    """Class guard (v6.87.45): `is_known` is the ONE place that says what counts as
    current, sourced evidence.

    Making freshness an argument of the owned predicate is only worth anything if the
    restatements it replaced are gone. Four survived the change that claimed them —
    `context_fit` twice (once on a real `CapabilityEvidence`, once on the plan built
    from it) and `loop` twice — each spelling out the same
    `status in {confirmed, asserted} and not stale and window > 0`. They agree today;
    a fifth caller that drops the `not stale` clause is exactly how the >=1M gate lost
    it in the first place."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent / "ouroboros"
    pattern = re.compile(r"""in\s*\{\s*["'](?:confirmed|asserted)["']\s*,\s*["'](?:confirmed|asserted)["']\s*\}""")
    offenders = [
        f"{path.relative_to(root)}:{i}"
        for path in sorted(root.rglob("*.py"))
        if path.name != "capability_evidence.py"
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert offenders == [], (
        "these re-derive capability_evidence.is_known(...) instead of calling it: "
        + ", ".join(offenders)
    )


def test_a_failed_probe_is_retried_once_its_throttle_expires_and_not_before(tmp_path, monkeypatch):
    """The FAILED cache is a short throttle, and both of its edges are load-bearing.

    `probe` serves a cached record without touching the network while it is inside its
    TTL, and the TTL a FAILED record gets is a tenth of an hour against a confirmed
    record's day. Nothing measured either number: both could move to any value with a
    green suite, and each direction breaks something this branch has already had to fix
    once. Too LONG and one transport blip reads as a dead route for the whole window —
    every resolution past it takes the no-fetch answer, `blocking_authority_allowed`
    goes False, and scope blocks every commit until it expires, which is v6.87.45's
    wedge arriving through the record instead of through a process-lifetime memo. Too
    SHORT and the throttle stops existing: a provider that is genuinely down is re-asked
    on every resolution, on the hot path of every review.

    So the behaviour is driven from both sides of the edge rather than named."""
    import datetime

    fetches = []

    def fake_metadata(*_a, **_k):
        fetches.append(1)
        return 1_000_000

    monkeypatch.setattr(ce, "_provider_metadata_window", fake_metadata)
    monkeypatch.setattr(ce, "_metadata_fetch_transport_failed", lambda *a, **k: True)

    def seed_failure(age_seconds):
        fp = ce.route_fingerprint(provider="openrouter", base_url="", model="x/y")
        ts = (datetime.datetime.now(datetime.timezone.utc)
              - datetime.timedelta(seconds=age_seconds)).isoformat()
        store = tmp_path / "state" / "capability_evidence.json"
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(json.dumps({"probes": {fp: {
            "window_tokens": 0, "status": ce.STATUS_FAILED, "source": ce.SOURCE_NONE,
            "route_fp": fp, "model": "x/y", "provider": "openrouter", "ts": ts,
            "detail": "provider unreachable during probe",
        }}}), encoding="utf-8")

    # Inside the throttle: the failure is served from the cache, untouched, and the
    # provider is not asked again.
    seed_failure(5 * 60)
    throttled = ce.probe(tmp_path, provider="openrouter", model="x/y", allow_fetch=True)
    assert throttled.status == ce.STATUS_FAILED
    assert fetches == [], "a fresh failure record must not re-ask the provider"

    # Past it: the route is retried, and a provider that has come back replaces the
    # failure with confirmed evidence instead of staying poisoned.
    seed_failure(20 * 60)
    retried = ce.probe(tmp_path, provider="openrouter", model="x/y", allow_fetch=True)
    assert fetches, "an expired failure record must be retried"
    assert retried.status == ce.STATUS_CONFIRMED and retried.window_tokens == 1_000_000
    assert ce.confirms_at_least(retried, ce.ONE_MILLION, require_fresh=True) is True


def test_expired_probe_records_drop_on_write(tmp_path):
    """CPL4-C8: the write seam drops probe records the reader already treats as
    expired — failed/unprobeable past their TTL, confirmed past GC retention —
    while owner acks, in-retention stale confirmed (the blip-keep invariant),
    and records with unreadable timestamps survive untouched."""
    from ouroboros.utils import utc_now_iso

    store = tmp_path / "state" / "capability_evidence.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    old_failed = {"status": ce.STATUS_FAILED, "ts": "2020-01-01T00:00:00+00:00"}
    old_unprobeable = {"status": ce.STATUS_UNPROBEABLE, "ts": "2020-01-01T00:00:00+00:00"}
    ancient_confirmed = {
        "status": ce.STATUS_CONFIRMED, "window_tokens": 1_000_000,
        "ts": "2020-01-01T00:00:00+00:00",
    }
    stale_confirmed_within_gc = {
        "status": ce.STATUS_CONFIRMED, "window_tokens": 1_000_000, "ts": utc_now_iso(),
    }
    broken_ts = {"status": ce.STATUS_FAILED, "ts": "not-a-time"}
    store.write_text(json.dumps({
        "probes": {
            "old-failed": old_failed,
            "old-unprobeable": old_unprobeable,
            "ancient-confirmed": ancient_confirmed,
            "fresh-confirmed": stale_confirmed_within_gc,
            "broken-ts": broken_ts,
        },
        "owner_acks": {"acked-route": {"tokens": 1_000_000, "ts": "2020-01-01T00:00:00+00:00"}},
    }), encoding="utf-8")

    ce._store_evidence(tmp_path, "probes", "new-route", {
        "status": ce.STATUS_CONFIRMED, "window_tokens": 200_000, "ts": utc_now_iso(),
    })

    on_disk = json.loads(store.read_text(encoding="utf-8"))
    assert set(on_disk["probes"]) == {"fresh-confirmed", "broken-ts", "new-route"}
    assert on_disk["owner_acks"] == {
        "acked-route": {"tokens": 1_000_000, "ts": "2020-01-01T00:00:00+00:00"},
    }
