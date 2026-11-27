"""The Costs page's period view: the recorded cost of a period's model calls by token class, and its cache
writes by what each followed (``ouroboros/gateway/cost_period.py``).

The ledger stays the one money authority: the classes share the RECORDED cost out, so they add up to it; a model
the route's catalog does not price keeps its tokens and gets no invented split; a period is found by position
and filtered by time, so the order of rows decides speed and never membership.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import time

import pytest

from ouroboros import llm_capability_policy, pricing, usage_accounting as ua, usage_store
from ouroboros.gateway import cost_period
from ouroboros.physical_attempt_context import PhysicalAttemptContext
from ouroboros.provider_models import normalize_model_identity

OPUS, UNLISTED = "anthropic/claude-opus-5.5", "vendor/unlisted-model"
# The route's catalog as the price reader keys it; dollars per million: input, cache read, 5-minute cache write, output.
CATALOG = {normalize_model_identity(OPUS): pricing.PricingSchedule((4.0, 0.2, 5.0, 20.0), cache_write_1h=8.0)}


@pytest.fixture
def root(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "state").mkdir(parents=True)
    (data / "logs").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(data / "settings.json"))
    usage_store.migrate_from_journal(data)
    monkeypatch.setattr(pricing, "get_pricing", lambda *, provider="openrouter", **_kw: dict(CATALOG) if provider == "openrouter" else {})
    monkeypatch.setattr(llm_capability_policy, "recorded_openrouter_catalog", lambda: ({}, ""))
    return data


def _call(data, *, model=OPUS, provider="openrouter", prompt, read=0, write=0, output=0, cost, ttl="1h",
          prefix=None, settle=True, dispatch=True):
    physical = PhysicalAttemptContext(
        profile="owner_max", rendered_mode="max", measurement_basis="fresh_route_usage", route_fp="fp",
        round_id="exec:round:1", target_total_tokens=None, capacity_total_tokens=1_000_000, context_target_miss=False,
        automatic_pass_used=False, prefix=prefix)
    attempt = ua.reserve_attempt(ua.AttemptRequest(
        model=model, provider=provider, reservation_usd=5.0, global_limit_usd=1000.0, drive_root=data,
        task_id="task", root_task_id="root", category="task", source="test.period", prompt_cache_ttl=ttl,
        candidate_measurement_kind="canonical_json_v1", candidate_raw_sha256="0" * 64, candidate_context_sha256="1" * 64,
        candidate_raw_size_bytes=2, candidate_context_size_bytes=1, physical_context=physical))
    if dispatch:
        ua.mark_dispatched(attempt)
    if settle:
        ua.settle_attempt(attempt, {"prompt_tokens": prompt, "completion_tokens": output, "cached_tokens": read,
                                    "cache_write_tokens": write, "prompt_cache_ttl": ttl}, cost_usd=cost, cost_final=True)
    return attempt


def _settle(attempt, *, prompt, read=0, write=0, output=0, cost, ttl="1h"):
    ua.settle_attempt(attempt, {"prompt_tokens": prompt, "completion_tokens": output, "cached_tokens": read,
                                "cache_write_tokens": write, "prompt_cache_ttl": ttl}, cost_usd=cost, cost_final=True)


def _day_after(moment):
    """``now`` for a 24-hour period that starts at ``moment``."""
    return moment + cost_period.PERIODS["24h"]


def test_the_classes_share_the_recorded_cost_and_add_up_to_it(root):
    # 900k read, 100k written at the 1-hour lifetime, 2k out, nothing uncached. The catalog would charge
    # 0.18 + 0.80 + 0.04 = 1.02; the provider recorded 1.10 (another tier, a regional endpoint): 1.10 is the amount.
    _call(root, prompt=1_000_000, read=900_000, write=100_000, output=2_000, cost=1.10)
    view = cost_period.cost_period(root, "24h")
    model, = view["models"]
    assert (model["model"], model["calls"], model["cost"]) == (OPUS, 1, 1.10)
    assert model["tokens"] == {"cache_write": 100_000, "cache_read": 900_000, "output": 2_000, "uncached_input": 0}
    assert sum(model["by_class"].values()) == pytest.approx(1.10)
    assert model["by_class"]["cache_write"] == pytest.approx(1.10 * 0.80 / 1.02)
    assert model["by_class"]["cache_read"] == pytest.approx(1.10 * 0.18 / 1.02, abs=1e-6)
    assert (model["write_price_in_reads"], model["write_price_lifetime"]) == (40.0, "1h")  # the ratio names its lifetime
    # The same tokens at the 5-minute lifetime weigh the write at 5, not 8, dollars a million.
    _call(root, prompt=1_000_000, read=900_000, write=100_000, output=2_000, cost=1.0, ttl="5m")
    both, = cost_period.cost_period(root, "24h")["models"]
    assert both["cost"] == pytest.approx(2.10) and sum(both["by_class"].values()) == pytest.approx(2.10)
    assert both["by_class"]["cache_write"] == pytest.approx(1.10 * 0.80 / 1.02 + 1.0 * 0.50 / 0.72, abs=1e-6)
    # The price ratio is the one of the lifetime most tokens were written at, and names it: 300k more at five minutes.
    _call(root, prompt=400_000, read=100_000, write=300_000, output=10, cost=1.7, ttl="5m")
    mostly_short, = cost_period.cost_period(root, "24h")["models"]
    assert (mostly_short["write_price_in_reads"], mostly_short["write_price_lifetime"]) == (25.0, "5m")


def test_a_model_the_catalog_does_not_price_keeps_its_tokens_and_gets_no_split(root):
    _call(root, model=UNLISTED, prompt=500, read=100, write=50, output=7, cost=0.4)
    _call(root, prompt=1_000, output=10, cost=0.2, settle=False)  # sent, not answered yet: an open row, no money
    by_model = {row["model"]: row for row in cost_period.cost_period(root, "24h")["models"]}
    unlisted = by_model[UNLISTED]
    assert unlisted["by_class"] is None and unlisted["unshared_cost"] == 0.4 and unlisted["cost"] == 0.4
    assert unlisted["tokens"] == {"cache_write": 50, "cache_read": 100, "output": 7, "uncached_input": 350}
    assert (unlisted["write_price_in_reads"], unlisted["write_price_lifetime"]) == (None, "1h")
    # A model none of whose calls has a recorded price has an unknown cost, never $0.
    assert (by_model[OPUS]["calls"], by_model[OPUS]["cost"], by_model[OPUS]["open_rows"]) == (1, None, 1)


@pytest.mark.parametrize("routes", [("anthropic", "openrouter"), ("openrouter", "anthropic")])
def test_dominant_lifetime_counts_writes_on_every_route_of_the_model(root, routes):
    for provider in routes:
        _call(root, provider=provider, prompt=20, write=20, cost=0.1)
    _call(root, prompt=30, write=30, ttl="5m", cost=0.1)
    model, = cost_period.cost_period(root, "24h")["models"]
    assert (model["write_price_lifetime"], model["write_price_in_reads"]) == ("1h", None)
    assert model["tokens"]["cache_write"] == 70


@pytest.mark.parametrize("read_price,short_price,hour_price,ratio", [
    (0.5, None, 13.0, 26.0),
    (0.5, None, 0.0, 0.0),
    (0.5, None, 0.001, 0.002),
    (0.5, 5.0, None, None),
    (0.0, 5.0, 13.0, None),
])
def test_period_reads_the_hourly_price_and_preserves_settled_cash(root, monkeypatch, read_price, short_price, hour_price, ratio):
    attempt = _call(root, prompt=1000, cost=None, settle=False)
    ua.settle_attempt(attempt, {"prompt_tokens": 1000, "completion_tokens": 0, "cached_tokens": 100,
                                "cache_write_tokens": 900, "prompt_cache_ttl": "1h"}, cost_usd=0.125, cost_final=False)
    ledger_before = ua.read_usage_records(root)
    assert ledger_before[0]["cost_estimated"]
    rate = pricing.PricingSchedule((2.0, read_price, short_price, 8.0), cache_write_1h=hour_price)
    monkeypatch.setitem(CATALOG, normalize_model_identity(OPUS), rate)
    model, = cost_period.cost_period(root, "24h")["models"]
    assert (model["write_price_in_reads"], model["write_price_lifetime"]) == (ratio, "1h")
    assert model["cost"] == 0.125
    if hour_price is None:
        assert model["by_class"] is None and model["unshared_cost"] == 0.125
    else:
        share = 900 * hour_price / (900 * hour_price + 100 * read_price)
        assert model["by_class"]["cache_write"] == pytest.approx(0.125 * share, abs=1e-6)
    assert ua.read_usage_records(root) == ledger_before
    # A few short writes cannot substitute their price for the predominant hourly lifetime.
    _call(root, prompt=10, write=10, cost=0.01, ttl="5m")
    combined, = cost_period.cost_period(root, "24h")["models"]
    assert (combined["write_price_in_reads"], combined["write_price_lifetime"]) == (ratio, "1h")


def test_a_period_holds_the_calls_started_in_it_whatever_the_order_of_rows(root):
    earlier = _call(root, prompt=10, write=5, cost=0.5)
    time.sleep(0.05)
    start = time.time()
    time.sleep(0.05)
    _call(root, prompt=20, write=7, cost=0.25)
    # Rows that carry no reservation time (an unmetered dispatch, a delegated session) stand in the same tail: they
    # are not model calls of the ledger, are never counted here and never mislead the search for the period's start.
    ua.record_unmetered_external_dispatch("external-call", drive_root=root, provider="external-skill", task_id="external",
                                          root_task_id="root", category="skill", source="test.external")
    view = cost_period.cost_period(root, "24h", now=_day_after(start))
    assert [(row["calls"], row["cost"]) for row in view["models"]] == [(1, 0.25)]
    assert [cause["write_tokens"] for cause in view["cache_writes"]] == [7]
    with usage_store.read(root) as txn:  # the view equals the plain filter over every row
        plain = txn.conn.execute("SELECT COUNT(*), SUM(CAST(cost_usd AS REAL)) FROM attempts WHERE COALESCE(kind, 'attempt')='attempt' "
                                 "AND ts_reserved>=?", (view["since"],)).fetchone()
        stampless = txn.conn.execute("SELECT COUNT(*) FROM attempts WHERE ts_reserved IS NULL").fetchone()[0]
    assert tuple(plain) == (1, 0.25) and stampless >= 1
    assert cost_period.cost_period(root, "24h", now=_day_after(time.time() + 1))["models"] == []
    # A row that stands before the period's first row but was reserved inside the period still counts:
    # position only decides where the read starts.
    with usage_store.hold(root) as txn:
        inside = txn.conn.execute("SELECT MAX(ts_reserved) FROM attempts").fetchone()[0]
        txn.conn.execute("UPDATE attempts SET ts_reserved=? WHERE attempt_id=?", (inside, earlier.attempt_id))
    moved = cost_period.cost_period(root, "24h", now=_day_after(start))
    assert [(row["calls"], row["cost"]) for row in moved["models"]] == [(2, 0.75)]


def test_cache_writes_are_grouped_by_what_each_followed(root):
    _call(root, prompt=100, write=90, cost=0.9, prefix={"departs": "first"})
    _call(root, prompt=100, write=80, cost=0.8, prefix={"departs": "system", "index": 1, "of": 3, "block": "my story and identity"})
    _call(root, prompt=100, write=70, cost=0.7, prefix={"departs": "messages", "index": 4, "of": 9, "origins": ["compaction"]})
    # A pause is measured from the base's answer (``base_at``) to the row's own dispatch, against the row's lifetime.
    _call(root, prompt=100, write=60, cost=0.6, prefix={"departs": "append", "base_at": time.time() - 4000})
    _call(root, prompt=100, write=50, cost=0.5, prefix={"departs": "append", "base_at": time.time() - 400}, ttl="5m")
    _call(root, prompt=100, write=40, cost=0.4, prefix={"departs": "append", "base_at": time.time() - 12, "from_route": "codex"})
    _call(root, prompt=100, write=30, cost=0.3)  # a row written before the record existed
    _call(root, prompt=100, read=100, cost=0.1, prefix={"departs": "append"})  # wrote nothing: not a cache write
    writes = {(cause["class"], cause["detail"]): cause for cause in cost_period.cost_period(root, "24h")["cache_writes"]}
    assert set(writes) == {("first_request", ""), ("system_changed", "my story and identity"), ("history_rewritten", "compaction"),
                           ("paused", ""), ("returned_to_model", "codex"), ("not_recorded", "")}
    assert writes[("paused", "")]["calls"] == 2 and writes[("paused", "")]["write_tokens"] == 110
    first = writes[("first_request", "")]
    # 90 written and 10 uncached tokens: the write's share of the recorded 0.9 is 90*8 / (90*8 + 10*4).
    assert first["write_cost"] == pytest.approx(0.9 * 720 / 760, abs=1e-6) and first["unshared_calls"] == 0


def test_the_endpoint_adds_the_period_only_when_asked(root, monkeypatch):
    from types import SimpleNamespace

    from ouroboros.gateway.history import make_cost_breakdown_endpoint

    _call(root, prompt=100, read=40, write=30, output=5, cost=0.25, prefix={"departs": "first"})
    endpoint = make_cost_breakdown_endpoint(root)

    def get(**query):
        return json.loads(asyncio.run(endpoint(SimpleNamespace(query_params=query))).body)

    plain = get()
    assert "period" not in plain and plain == json.loads(asyncio.run(endpoint(None)).body)
    asked = get(period="7d")
    assert {key: value for key, value in asked.items() if key != "period"} == plain  # the page's totals are untouched
    assert asked["period"]["available"] and asked["period"]["models"][0]["cost"] == 0.25
    assert get(period="year")["period"] == {"available": False, "period": "year", "error_code": "unknown_period",
                                            "periods": ["24h", "7d", "30d"]}
    monkeypatch.setattr(cost_period, "cost_period", lambda *_a, **_k: 1 / 0)
    assert get(period="7d")["period"] == {"available": False, "period": "7d", "error_code": "ledger_unavailable"}


def test_class_weights_are_the_addends_the_estimate_itself_sums(root):
    usage = {"prompt_tokens": 1_200_000, "completion_tokens": 3_000, "cached_tokens": 900_000, "cache_write_tokens": 100_000,
             "prompt_cache_ttl": "1h"}
    tariff = CATALOG[normalize_model_identity(OPUS)]
    weights = pricing.cost_class_weights(tariff, usage)
    estimate = pricing.estimate_cost_optional(OPUS, 1_200_000, 3_000, cache_usage=usage, provider="openrouter", allow_live_fetch=False)
    assert sum(weights.values()) == pytest.approx(estimate) and set(weights) == set(pricing.COST_CLASSES)
    assert weights == pytest.approx({"uncached_input": 0.8, "cache_read": 0.18, "cache_write": 0.8, "output": 0.06})
    # The provider's own split of the writes by lifetime wins over the requested one: 40k of the 100k at one hour.
    split = pricing.cost_class_weights(tariff, {**usage, "cache_write_tokens_by_ttl": {"1h": 40_000, "5m": 60_000}})
    assert split["cache_write"] == pytest.approx(40_000 * 8 / 1e6 + 60_000 * 5 / 1e6)
    # Unknown stays unknown: no tariff, or a call that reported no token counts, has no weights.
    assert pricing.cost_class_weights(None, usage) is None
    assert pricing.cost_class_weights(tariff, {**usage, "prompt_tokens": None}) is None
    assert pricing.cost_class_weights(tariff, {**usage, "cached_tokens": None}) is None  # absent counts are not zero
    assert pricing.cost_class_weights(tariff, {**usage, "cache_write_tokens": None}) is None
    # The provider's split counts only when it accounts for every write; then it wins over an unnamed lifetime.
    unnamed = {**usage, "prompt_cache_ttl": "default"}
    assert pricing.cost_class_weights(tariff, {**unnamed, "cache_write_tokens_by_ttl": {"5m": 60_000, "1h": 40_000}})["cache_write"] \
        == pytest.approx(60_000 * 5 / 1e6 + 40_000 * 8 / 1e6)
    for partial in ({}, {"5m": 20_000}, {"5m": 60_000, "1h": "40000"}, {"2h": 100_000}):
        assert pricing.cost_class_weights(tariff, {**unnamed, "cache_write_tokens_by_ttl": partial}) is None
    assert pricing.cost_class_weights((4.0, None, 5.0, 20.0), usage) is None  # reads happened, their price is unknown
    assert pricing.cost_class_weights(tariff, {**usage, "prompt_cache_ttl": "default"}) is None  # a write at an unnamed lifetime
    assert pricing.cost_class_weights(tariff, {**usage, "prompt_cache_ttl": "", "cache_write_tokens": 0}) is not None


def test_a_display_reads_tariffs_without_a_fetch_and_falls_back_to_the_recorded_catalogue(root, monkeypatch):
    from ouroboros import llm_capability_policy, llm_pricing

    assert pricing.display_tariffs("openrouter") == (CATALOG, "")           # what this process holds
    assert pricing.display_tariffs("anthropic") == ({}, "")                  # a route with no catalogue of its own
    monkeypatch.setattr(pricing, "get_pricing", lambda *, allow_live_fetch, **_kw: {} if not allow_live_fetch else 1 / 0)
    record = {OPUS: {"context_length": 1_000_000, "pricing": {"prompt": "0.000004", "completion": "0.00002",
                        "input_cache_read": "0.0000002", "input_cache_write": "0.000005", "input_cache_write_1h": "0.000008"}},
              "vendor/no-prices": {"context_length": 8}}
    monkeypatch.setattr(llm_capability_policy, "recorded_openrouter_catalog", lambda: (record, "2026-11-12T10:00:00+00:00"))
    rows, observed = pricing.display_tariffs("openrouter")
    assert observed == "2026-11-12T10:00:00+00:00" and tuple(rows[normalize_model_identity(OPUS)]) == (4.0, 0.2, 5.0, 20.0)
    assert llm_pricing.recorded_openrouter_pricing()[0].keys() >= {OPUS} and "vendor/no-prices" not in rows
    # The period view names that date beside the shares it priced with the record.
    _call(root, prompt=1_000, read=900, write=100, output=2, cost=0.01)
    view = cost_period.cost_period(root, "24h")
    assert view["tariff_observed_at"] == "2026-11-12T10:00:00+00:00" and view["models"][0]["by_class"] is not None
    monkeypatch.setattr(llm_capability_policy, "recorded_openrouter_catalog", lambda: ({}, ""))
    bare = cost_period.cost_period(root, "24h")
    assert bare["tariff_observed_at"] == "" and bare["models"][0]["by_class"] is None and bare["models"][0]["unshared_cost"] == 0.01


def test_a_rebuilt_store_gives_the_same_period(root):
    """An import inserts attempts in the order of their LAST record (``usage_store._build``): a call reserved before
    the period and settled inside it then stands after the period's calls, and position alone would misplace the
    start. The period is still exactly the calls reserved in it, before and after the rebuild."""
    early = _call(root, prompt=10, write=5, cost=None, settle=False)        # reserved before the period...
    time.sleep(0.05)
    start = time.time()
    time.sleep(0.05)
    inside = _call(root, prompt=20, write=7, cost=None, settle=False)       # reserved inside it, answered late
    _call(root, prompt=30, write=9, cost=0.125)                             # reserved and settled inside it
    _settle(inside, prompt=20, write=7, cost=0.25)
    _settle(early, prompt=10, write=5, cost=0.5)                            # ...and settled last of all
    before = cost_period.cost_period(root, "24h", now=_day_after(start))
    assert [(row["calls"], row["cost"]) for row in before["models"]] == [(2, 0.375)]
    assert sorted(cause["write_tokens"] for cause in before["cache_writes"]) == [16]
    usage_store.export_journal(root)
    usage_store.migrate_from_journal(root)  # the store rebuilt from its journal
    assert cost_period.cost_period(root, "24h", now=_day_after(start)) == before
    with usage_store.hold(root) as txn:  # and with the early call standing last, where its last record puts it
        columns = [column[1] for column in txn.conn.execute("PRAGMA table_info(attempts)")]
        row = tuple(txn.conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (early.attempt_id,)).fetchone())
        txn.conn.execute("DELETE FROM attempts WHERE attempt_id=?", (early.attempt_id,))
        txn.conn.execute(f"INSERT INTO attempts ({','.join(columns)}) VALUES ({','.join('?' * len(columns))})", row)
        last = txn.conn.execute("SELECT attempt_id FROM attempts ORDER BY rowid DESC LIMIT 1").fetchone()[0]
    assert last == early.attempt_id
    assert cost_period.cost_period(root, "24h", now=_day_after(start)) == before
    whole = cost_period.cost_period(root, "24h")
    assert [(row["calls"], row["cost"]) for row in whole["models"]] == [(3, 0.875)]


def test_the_serving_upstream_and_the_write_split_reach_the_settled_row(root):
    from types import SimpleNamespace

    from ouroboros._usage_response import usage_from_response
    from ouroboros._usage_settlement import settlement_fields

    raw = {"provider": "Claude Platform on AWS",
           "usage": {"prompt_tokens": 100, "completion_tokens": 5, "response_provider": "a nested claim",
                     "prompt_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 30},
                     "cache_creation": {"ephemeral_1h_input_tokens": 20, "ephemeral_5m_input_tokens": 10}}}
    usage, _cost, _final = usage_from_response(raw, payload=raw)
    assert usage["response_provider"] == "Claude Platform on AWS"        # the response's own outer label
    assert usage["cache_write_tokens_by_ttl"] == {"5m": 10, "1h": 20}
    unnamed = {key: value for key, value in raw.items() if key != "provider"}
    assert "response_provider" not in usage_from_response(unnamed, payload=unnamed)[0]  # a nested label names nothing
    reservation = SimpleNamespace(provider="openrouter", model=OPUS, processing_preference=None, submitted_processing_mode=None)
    fields = settlement_fields(reservation, usage, 0.01, True, estimate_cost_optional=lambda *_a, **_k: None)
    assert fields["response_provider"] == "Claude Platform on AWS" and fields["cache_write_tokens_by_ttl"] == {"5m": 10, "1h": 20}
    # On the row, read back through the store: the period view prices the writes by the provider's own split.
    attempt = _call(root, prompt=100, cost=None, settle=False)
    ua.settle_attempt(attempt, {**usage, "prompt_cache_ttl": "1h"}, cost_usd=0.01, cost_final=True)
    with usage_store.read(root) as txn:
        row = txn.attempt(attempt.attempt_id)
    assert row["response_provider"] == "Claude Platform on AWS" and row["cache_write_tokens_by_ttl"] == {"5m": 10, "1h": 20}
    assert row["cache_write_tokens"] == 30 and row["prompt_tokens"] == 100


@pytest.mark.parametrize("nested,expected,cash", [
    (True, 12, 0.025), ("unavailable", 12, 0.025), (-1, 12, None), (1.5, 12, None),
    (-0.5, 12, 0.0), (0, 0, None), ("7", 7, None),
])
def test_unreadable_nested_write_count_preserves_the_reported_aggregate(root, nested, expected, cash):
    raw = {"usage": {"prompt_tokens": 20, "completion_tokens": 1, "cache_write_tokens": 12,
                     "prompt_tokens_details": {"cache_write_tokens": nested}, "prompt_cache_ttl": "1h", "cost": cash}}
    usage, cost, final = ua.usage_from_response(raw)
    reservation = _call(root, prompt=20, cost=None, settle=False)
    ua.settle_attempt(reservation, usage, cost_usd=cost, cost_final=final)
    with usage_store.read(root) as txn:
        actual = txn.attempt(reservation.attempt_id)
    assert actual["cache_write_tokens"] == expected
    estimate = ((20 - expected) * 4 + expected * 8 + 20) / 1_000_000
    assert actual["cost_usd"] == pytest.approx(estimate if cash is None else cash, rel=1e-12, abs=0)
    assert actual["cost_final"] is (cash is not None)


@pytest.mark.parametrize("fallback,write,prompt", [
    ({"cache_write_tokens": 8}, 8, 14),
    ({"cache_creation": {"ephemeral_5m_input_tokens": 3, "ephemeral_1h_input_tokens": 5}}, 8, 14),
    pytest.param({}, None, None, id="unreadable-write-cannot-complete-native-prompt"),
])
def test_an_invalid_native_write_total_falls_back_to_the_next_report(fallback, write, prompt):
    body = {"type": "message", "usage": {
        "input_tokens": 4, "cache_read_input_tokens": 2, "output_tokens": 1, "cache_creation_input_tokens": -1,
        **fallback,
    }}
    facts, _, _ = ua.usage_from_response(body)
    assert facts["cache_write_tokens"] == write and facts["prompt_tokens"] == prompt


@pytest.mark.parametrize("writes,total,written", [
    pytest.param({"cache_write_tokens": 8}, 14, 8, id="aggregate-under-its-alias"),
    pytest.param({"cache_creation": {"ephemeral_5m_input_tokens": 3, "ephemeral_1h_input_tokens": 5}}, 14, 8, id="both-bins"),
    pytest.param({"cache_creation": {"ephemeral_1h_input_tokens": 7}}, None, None, id="one-bin-no-aggregate"),
    pytest.param({"cache_write_tokens_by_ttl": {"5m": 7}}, None, None, id="one-normalized-bin-no-aggregate"),
])
def test_a_native_reply_has_one_inclusive_total_in_the_ledger_the_adapter_and_its_event(writes, total, written):
    """Messages usage of 4 fresh and 2 read tokens: whichever field names the 8 written ones, the ledger, the adapter
    and the adapter's event fact read the same total of 14; one lifetime bin without an aggregate leaves it unknown."""
    from ouroboros.event_usage import event_token_counts
    from ouroboros.llm import LLMClient

    body = {"type": "message", "content": [],
            "usage": {"input_tokens": 4, "cache_read_input_tokens": 2, "output_tokens": 1, **writes}}
    ledger, _, _ = ua.usage_from_response(body)
    _, reply = LLMClient(api_key="unused")._normalize_anthropic_response(
        body, {"provider": "anthropic", "usage_model": "anthropic/claude-test"})
    event = event_token_counts(reply)
    assert ledger["prompt_tokens"] == reply["prompt_tokens"] == event["prompt_tokens"] == total
    assert ledger["cache_write_tokens"] == reply["cache_write_tokens"] == event["cache_write_tokens"] == written


_BOTH_BINS = {"cache_creation": {"ephemeral_5m_input_tokens": 3, "ephemeral_1h_input_tokens": 5}}


@pytest.mark.parametrize("bill", [None, 0, 0.031], ids=["no-bill", "reported-zero", "reported-charge"])
@pytest.mark.parametrize("usage,total", [
    pytest.param({}, None, id="empty-usage"),
    pytest.param({"output_tokens": 5}, None, id="output-alone"),
    pytest.param({"cache_read_input_tokens": 20_000}, None, id="reads-alone"),
    pytest.param(_BOTH_BINS, None, id="both-bins-alone"),
    pytest.param({"input_tokens": 0, **_BOTH_BINS}, 8, id="reported-zero-fresh-input-and-both-bins"),
    pytest.param({"input_tokens": 4, "cache_read_input_tokens": 2, "cache_creation_input_tokens": 8, "output_tokens": 1},
                 14, id="fresh-read-and-write"),
    pytest.param({"input_tokens": 4, "cache_creation": {"ephemeral_1h_input_tokens": 7}}, None, id="one-bin"),
])
def test_a_native_reply_without_fresh_input_has_no_inclusive_prompt(root, usage, total, bill):
    """A Messages reply's ``input_tokens`` is its fresh input. A reply that did not report it (no usage, its output
    alone, 20,000 reads alone, its write bins alone) has no known inclusive prompt in the ledger, the adapter or
    the settled row; a reported zero is a measurement. Each reported counter and an independent bill stay."""
    from ouroboros.event_usage import event_token_counts
    from ouroboros.llm import LLMClient

    body = {"type": "message", "content": [], "usage": dict(usage), **({} if bill is None else {"total_cost_usd": bill})}
    ledger, cost, final = ua.usage_from_response(body)
    _, reply = LLMClient(api_key="unused")._normalize_anthropic_response(
        body, {"provider": "anthropic", "usage_model": "anthropic/claude-test"})
    attempt = _call(root, prompt=1, cost=None, settle=False)
    ua.settle_attempt(attempt, ledger, cost_usd=cost, cost_final=final)
    with usage_store.read(root) as txn:
        row = txn.attempt(attempt.attempt_id)
    assert ledger["prompt_tokens"] == reply["prompt_tokens"] == row["prompt_tokens"] == total
    assert event_token_counts(reply)["prompt_tokens"] in (None, total)
    assert row["completion_tokens"] == reply["completion_tokens"] == usage.get("output_tokens")
    assert row["cached_tokens"] == usage.get("cache_read_input_tokens")
    if "cache_creation" in usage:
        assert row["cache_write_tokens_by_ttl"] == reply["cache_write_tokens_by_ttl"] != {}
    assert row["cost_final"] is reply["cost_final"] is (bill is not None)
    if bill is not None:
        assert row["cost_usd"] == reply["cost"] == bill


def test_a_reply_without_native_fields_keeps_its_input_count_as_the_whole_prompt():
    """Only a native reply's ``input_tokens`` excludes cache tokens: with no native field it stays the total."""
    facts, _, _ = ua.usage_from_response({"usage": {"input_tokens": 100, "cached_tokens": 40, "cache_write_tokens": 8}})
    assert (facts["prompt_tokens"], facts["cached_tokens"], facts["cache_write_tokens"]) == (100, 40, 8)


@pytest.mark.parametrize("shape", ["native", "native-normalized-bins", "responses", "chat"])
@pytest.mark.parametrize("bins,written", [({"5m": 0, "1h": 30}, 30), ({"5m": 0, "1h": 0}, 0), ({"5m": 7}, None)])
def test_settlement_keeps_inclusive_prompt_separate_from_native_uncached_input(root, shape, bins, written):
    native_bins = {f"ephemeral_{tier}_input_tokens": n for tier, n in bins.items()}
    if shape.startswith("native"):
        usage = {"input_tokens": 10, "cache_read_input_tokens": 20, "output_tokens": 1}
        body = {"type": "message", "usage": usage}
        prompt = 30 + written if written is not None else None
    elif shape == "responses":
        usage = {"input_tokens": 60, "input_tokens_details": {"cached_tokens": 20}, "output_tokens": 1}
        body = {"object": "response", "usage": usage}
        prompt = 60
    else:
        usage = {"prompt_tokens": 60, "prompt_tokens_details": {"cached_tokens": 20}, "completion_tokens": 1}
        body = {"usage": usage}
        prompt = 60
    if shape == "native-normalized-bins":
        usage["cache_write_tokens_by_ttl"] = bins
    else:
        usage["cache_creation"] = native_bins
    usage["cost"] = 0.025
    facts, cost, final = ua.usage_from_response(body)
    attempt = _call(root, prompt=60, cost=None, settle=False)
    ua.settle_attempt(attempt, facts, cost_usd=cost, cost_final=final)
    with usage_store.read(root) as txn:
        row = txn.attempt(attempt.attempt_id)
    assert row["prompt_tokens"] == prompt and row["cache_write_tokens"] == written
    assert row["cache_write_tokens_by_ttl"] == bins
    assert row["cost_usd"] == 0.025 and row["cost_final"]


def test_partial_bins_preserve_confirmed_money_and_unclassified_input(root):
    _call(root, prompt=20, write=20, cost=0.5, ttl="5m")
    partial = {"prompt_tokens": 100, "completion_tokens": 1, "cached_tokens": 20,
               "prompt_cache_ttl": "1h", "cache_write_tokens_by_ttl": {"5m": 30}}
    for cost in (0.25, None):
        attempt = _call(root, prompt=100, cost=None, settle=False)
        ua.settle_attempt(attempt, partial, cost_usd=cost, cost_final=cost is not None)
        with usage_store.read(root) as txn:
            row = txn.attempt(attempt.attempt_id)
        assert row["cache_write_tokens_by_ttl"] == {"5m": 30} and row["cache_write_tokens"] is None
        assert row["cost_usd"] == cost and row["cost_final"] == (cost is not None)
    model, = cost_period.cost_period(root, "24h")["models"]
    assert (model["cost"], model["priced_rows"], model["unpriced_rows"]) == (0.75, 2, 1)
    assert model["unshared_cost"] == 0.25 and sum(model["by_class"].values()) == pytest.approx(0.5)
    assert model["uncounted_input_tokens"] == 160  # each inclusive prompt reported only 20 reads and no write total


def test_ten_reported_writes_with_lifetime_bins_three_and_five_keep_only_reported_money(root):
    from ouroboros.event_usage import event_token_counts

    raw = {"prompt_tokens": 20, "completion_tokens": 1, "cached_tokens": 4, "cache_write_tokens": 10,
           "prompt_cache_ttl": "1h", "cache_creation": {"ephemeral_5m_input_tokens": 3, "ephemeral_1h_input_tokens": 5}}
    for payment in (None, 0.125):
        usage, cost, final = ua.usage_from_response({"usage": {**raw, "cost": payment}})
        attempt = _call(root, prompt=20, cost=None, settle=False)
        ua.settle_attempt(attempt, usage, cost_usd=cost, cost_final=final)
        with usage_store.read(root) as txn:
            row = txn.attempt(attempt.attempt_id)
        assert event_token_counts(usage)["cache_write_tokens"] == row["cache_write_tokens"] == 10
        assert row["cache_write_tokens_by_ttl"] == {"5m": 3, "1h": 5}
        assert row["cost_usd"] == payment and row["cost_final"] is (payment is not None)
    view = cost_period.cost_period(root, "24h")
    model, = view["models"]
    assert model["tokens"] == {"cache_write": 20, "cache_read": 8, "output": 2, "uncached_input": 12}
    assert model["unpriced_rows"] == 1 and model["cost"] == model["unshared_cost"] == 0.125
    assert model["by_class"] is None and model["write_price_in_reads"] is None
    cause, = view["cache_writes"]
    assert (cause["write_tokens"], cause["shared_write_tokens"], cause["unshared_calls"]) == (20, 0, 2)


@pytest.mark.parametrize("preference,submitted,tier,can_share", [
    pytest.param("", "", "priority", False, id="tier-priority-observed-fast"),
    pytest.param("fast", "priority", "default", True, id="requested-fast-observed-default"),
    pytest.param("fast", "", "", False, id="intent-only-host-request-receipt"),
    pytest.param("", "", "", True, id="no-receipt-no-tier-base-router"),
])
def test_period_uses_the_recorded_processing_mode_for_the_paid_subset(root, preference, submitted, tier, can_share):
    _call(root, prompt=10, write=10, cost=1.0)
    payment = 0.00000025
    attempt = ua.reserve_attempt(ua.AttemptRequest(
        model=OPUS, provider="openrouter", reservation_usd=5.0, global_limit_usd=1000.0, drive_root=root,
        task_id="task", root_task_id="root", category="task", source="test.period", prompt_cache_ttl="1h",
        candidate_measurement_kind="canonical_json_v1", candidate_raw_sha256="0" * 64,
        candidate_context_sha256="1" * 64, candidate_raw_size_bytes=2, candidate_context_size_bytes=1,
        processing_preference=preference, submitted_processing_mode=submitted, allow_live_fetch=False))
    ua.mark_dispatched(attempt)
    wire = {"usage": {
        "prompt_tokens": 90, "cached_tokens": 0, "cache_write_tokens": 90, "completion_tokens": 0,
        "prompt_cache_ttl": "1h", "cost": payment}}
    if tier:
        wire["service_tier"] = tier
    usage, cost, final = ua.usage_from_response(wire, provider="openrouter")
    ua.settle_attempt(attempt, usage, cost_usd=cost, cost_final=final)
    with usage_store.read(root) as txn:
        row = txn.attempt(attempt.attempt_id)
    if tier:
        assert row["service_tier"] == tier and row["processing"]["observedNative"] == [tier]
    elif preference:
        assert row["processing"]["requested"] == preference
    else:
        assert row.get("processing") is None
    view = cost_period.cost_period(root, "24h")
    model, = view["models"]
    cause, = view["cache_writes"]
    assert model["cost"] == 1.0 + payment and model["tokens"]["cache_write"] == cause["write_tokens"] == 100
    assert model["unshared_cost"] == (0 if can_share else payment)
    assert model["by_class"]["cache_write"] == cause["write_cost"] == (1.0 + payment if can_share else 1.0)
    assert model["unshared_calls"] == cause["unshared_calls"] == (0 if can_share else 1)
    assert model["tokens_shared"]["cache_write"] == cause["shared_write_tokens"] == (100 if can_share else 10)
    assert model["write_price_in_reads"] == (40 if can_share else None)


@pytest.mark.parametrize("receipt,tier,can_share", [
    pytest.param({"observed": "standard", "observedNative": ["default"]}, "", True, id="recorded-same-attempt-default"),
    pytest.param({"observed": "standard", "observedNative": ["default"]}, "priority", False,
                 id="recorded-receipt-disagrees-with-tier"),
    pytest.param({"observed": "fast", "observedNative": ["priority"]}, "", False, id="recorded-fast"),
    pytest.param({"observed": "mixed", "observedNative": ["default", "priority"]}, "", False, id="recorded-mixed"),
])
def test_period_shares_only_a_persisted_receipt_that_matches_the_same_attempt_default(root, receipt, tier, can_share):
    """An imported receipt is a recorded fact; mixed or contradictory evidence cannot set the class weights."""
    _call(root, prompt=10, write=10, cost=1.0)
    payment = 0.00000025
    attempt = _call(root, prompt=90, write=90, cost=payment)
    with usage_store.hold(root) as txn:
        txn.conn.execute("UPDATE attempts SET extra=json_set(extra,'$.processing',json(?)) WHERE attempt_id=?",
                         (json.dumps(receipt), attempt.attempt_id))
        if tier:
            txn.conn.execute("UPDATE attempts SET extra=json_set(extra,'$.service_tier',json(?)) WHERE attempt_id=?",
                             (json.dumps(tier), attempt.attempt_id))
    model, = cost_period.cost_period(root, "24h")["models"]
    assert model["cost"] == 1.0 + payment
    assert model["unshared_cost"] == (0 if can_share else payment)
    assert model["write_price_in_reads"] == (40 if can_share else None)
    assert model["tokens_shared"]["cache_write"] == (100 if can_share else 10)


@pytest.mark.parametrize("written,stored_split", [(100, {"5m": 40}), (0, {"5m": 2}), (100, '{"5m":100}')])
def test_costs_keeps_recorded_spend_unshared_when_stored_bins_cannot_explain_it(root, written, stored_split):
    attempt = _call(root, prompt=100, write=written, cost=0.25, ttl="1h")
    # Imported rows can contain a partial map, conflicting counts, or a string that resembles a map.
    with usage_store.hold(root) as txn:
        txn.conn.execute("UPDATE attempts SET extra=json_set(extra,'$.cache_write_tokens_by_ttl',json(?)) WHERE attempt_id=?",
                         (json.dumps(stored_split), attempt.attempt_id))
    before = ua.read_usage_records(root)
    model, = cost_period.cost_period(root, "24h")["models"]
    assert model["cost"] == model["unshared_cost"] == 0.25 and model["by_class"] is None
    assert model["write_price_lifetime"] == "" and model["write_price_in_reads"] is None
    assert ua.read_usage_records(root) == before


@pytest.mark.parametrize("hour_rate,expected,small_cap_fits", [
    ("0", 0.0, True), ("0.00000025", 2.5e-7, False), (None, None, None),
])
def test_catalog_rate_reaches_reservation_wrapup_and_the_sum_of_small_attempts(
    root, monkeypatch, hour_rate, expected, small_cap_fits,
):
    from ouroboros.llm_pricing import openrouter_price_row
    from ouroboros.task_pacing import wrapup_reservation_fits

    fields = {"prompt": "0", "completion": "0", "input_cache_read": "0.00000000002",
              "input_cache_write": "0.000001"}
    if hour_rate is not None:
        fields["input_cache_write_1h"] = hour_rate
    tariff = openrouter_price_row({"id": OPUS, "pricing": fields})
    monkeypatch.setitem(CATALOG, normalize_model_identity(OPUS), tariff)
    monkeypatch.setattr(ua, "last_task_cache_split", lambda *a, **kw: None)
    request = ua.AttemptRequest(OPUS, "openrouter", prompt_tokens_estimate=1, prompt_cache_ttl="1h", allow_live_fetch=False)
    with ua.usage_scope(ua.UsageScope(drive_root=root, task_id="root", root_task_id="root")):
        bound = ua._reservation_cost(request)
        assert wrapup_reservation_fits(request=request, root_cap_usd=1e-7, deciding_usd=0) is small_cap_fits
    for index in range(4):
        attempt = _call(root, prompt=1, write=1, output=0, cost=None)
        with usage_store.read(root) as txn:
            row = txn.attempt(attempt.attempt_id)
        assert row["cost_usd"] == bound and not row["cost_final"]
        if index == 0:
            single = cost_period.cost_period(root, "24h")
            only, = single["models"]
            cause, = single["cache_writes"]
            assert only["cost"] == bound
            if expected:
                assert only["by_class"]["cache_write"] == cause["write_cost"] == bound
    model, = cost_period.cost_period(root, "24h")["models"]
    if expected is None:
        assert bound is None and model["cost"] is None and model["write_price_in_reads"] is None
    else:
        assert bound == pytest.approx(expected, rel=1e-12, abs=0)
        assert model["cost"] == pytest.approx(4 * expected, rel=1e-12, abs=0)
        assert model["write_price_in_reads"] == pytest.approx(expected / 2e-11, rel=1e-12, abs=0)


def test_named_router_wrapup_uses_the_model_tariff_and_the_connections_own_cache(root, monkeypatch):
    from ouroboros import _usage_cache_splits
    from ouroboros.task_pacing import wrapup_reservation_fits

    monkeypatch.setitem(CATALOG, normalize_model_identity(OPUS), pricing.PricingSchedule((4.0, 0.2, 5.0, 20.0)))
    monkeypatch.setattr(_usage_cache_splits, "_SPLITS", {})
    named = f"openrouter@byok::{OPUS}"
    request = ua.AttemptRequest(named, "openrouter", prompt_tokens_estimate=100, prompt_cache_ttl="1h", allow_live_fetch=False)
    with ua.usage_scope(ua.UsageScope(drive_root=root, task_id="wrap", root_task_id="wrap")):
        ua.stash_task_cache_split("wrap", OPUS, 100, provider="openrouter", ttl_seconds=300)
        assert wrapup_reservation_fits(request=request, root_cap_usd=1, deciding_usd=0) is None
        cold = ua.reserve_attempt(request)
        assert cold.model == named and cold.reservation_upper_bound_usd is None
        ua.release_attempt(cold)
        ua.stash_task_cache_split("wrap", named, 100, provider="openrouter", ttl_seconds=300)
        warm = ua.reserve_attempt(request)
        assert warm.reservation_upper_bound_usd == pytest.approx(100 * 0.2 / 1_000_000)
        ua.release_attempt(warm)


def test_a_busy_ledger_is_asked_again_and_a_ledger_that_stays_busy_is_said_to_be_unreadable(root, monkeypatch):
    """A display read gives way to the ledger's writers; one chunk refused must not lose a whole period."""
    import contextlib

    from ouroboros.usage_ledger import UsageLockUnavailable

    _call(root, prompt=100, write=30, cost=0.25)
    real, refusals = usage_store.read, {"left": 3}

    @contextlib.contextmanager
    def read(path, **kwargs):
        if refusals["left"] > 0:
            refusals["left"] -= 1
            raise UsageLockUnavailable("the ledger's writer holds the store")
        with real(path, **kwargs) as txn:
            yield txn

    monkeypatch.setattr(usage_store, "read", read)
    assert cost_period.cost_period(root, "24h")["models"][0]["cost"] == 0.25 and refusals["left"] == 0
    refusals["left"] = 10 ** 6
    with pytest.raises(UsageLockUnavailable):
        cost_period.cost_period(root, "24h")


def test_calls_without_cache_counts_are_counted_apart_and_never_split(root):
    _call(root, prompt=1_000, read=400, write=100, output=10, cost=0.5)
    attempt = _call(root, prompt=100, cost=None, settle=False)
    ua.settle_attempt(attempt, {"prompt_tokens": 100, "completion_tokens": 10}, cost_usd=1.0, cost_final=True)  # no cache counts
    model, = cost_period.cost_period(root, "24h")["models"]
    assert model["tokens"] == {"cache_write": 100, "cache_read": 400, "output": 20, "uncached_input": 500}  # its 100 are in no class
    assert (model["uncounted_calls"], model["uncounted_input_tokens"]) == (1, 100)
    assert model["cost"] == pytest.approx(1.5) and model["unshared_cost"] == pytest.approx(1.0)  # its dollar stays unshared
    assert sum(model["by_class"].values()) == pytest.approx(0.5, abs=1e-6)
    # One count reported, one absent: what it did report is classed, only the rest of its input is apart.
    partial = _call(root, prompt=100, cost=None, settle=False)
    ua.settle_attempt(partial, {"prompt_tokens": 100, "completion_tokens": 0, "cached_tokens": 60}, cost_usd=0.2, cost_final=True)
    model, = cost_period.cost_period(root, "24h")["models"]
    assert model["tokens"]["cache_read"] == 460 and (model["uncounted_calls"], model["uncounted_input_tokens"]) == (2, 140)


def test_the_bounds_search_reads_points_and_a_stamp_spelling_does_not_move_membership(root):
    """Rows without a parseable time never make the search walk the ledger (each probe is one row), and membership
    compares instants: a ``Z`` stamp half a second before the period's start stays out."""
    first = _call(root, prompt=10, write=1, cost=0.1)
    with usage_store.hold(root) as txn:
        txn.conn.execute("UPDATE attempts SET ts_last_epoch=NULL")  # an imported tail without parseable times
    with usage_store.read(root) as txn:
        statements = []
        txn.conn.set_trace_callback(statements.append)
        bounds = txn.period_bounds(time.time() - 60)
        txn.conn.set_trace_callback(None)
    assert bounds[0] <= 1 and all("ts_last_epoch IS NOT NULL" not in sql for sql in statements)
    start = time.time()
    stamp = _dt.datetime.fromtimestamp(start - 0.5, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    with usage_store.hold(root) as txn:
        txn.conn.execute("UPDATE attempts SET ts_reserved=?, ts_last_epoch=? WHERE attempt_id=?", (stamp, start, first.attempt_id))
    assert cost_period.cost_period(root, "24h", now=start + cost_period.PERIODS["24h"])["models"] == []
    later = _dt.datetime.fromtimestamp(start + 0.5, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    with usage_store.hold(root) as txn:
        txn.conn.execute("UPDATE attempts SET ts_reserved=? WHERE attempt_id=?", (later, first.attempt_id))
    assert [row["calls"] for row in cost_period.cost_period(root, "24h", now=start + cost_period.PERIODS["24h"])["models"]] == [1]


def test_the_price_ratio_follows_the_lifetimes_the_provider_says_the_writes_were_made_at(root):
    attempt = _call(root, prompt=1_000, cost=None, settle=False, ttl="1h")
    ua.settle_attempt(attempt, {"prompt_tokens": 1_000, "completion_tokens": 5, "cached_tokens": 0, "cache_write_tokens": 900,
                                "cache_write_tokens_by_ttl": {"5m": 900}, "prompt_cache_ttl": "1h"}, cost_usd=0.01, cost_final=True)
    model, = cost_period.cost_period(root, "24h")["models"]
    assert (model["write_price_in_reads"], model["write_price_lifetime"]) == (25.0, "5m")  # requested 1h, written at 5m


def test_a_long_prompt_tier_weighs_its_own_one_hour_write_price():
    tier = pricing.PricingSchedule((6.0, 0.6, 7.5, 22.5), cache_write_1h=15.0)
    schedule = pricing.PricingSchedule((3.0, 0.3, 3.75, 15.0), ((200_000, tier),), cache_write_1h=6.0)
    usage = {"prompt_tokens": 300_000, "completion_tokens": 1_000, "cached_tokens": 100_000, "cache_write_tokens": 50_000,
             "prompt_cache_ttl": "1h"}
    weights = pricing.cost_class_weights(schedule, usage)
    assert weights == pytest.approx({"cache_write": 0.75, "cache_read": 0.06, "output": 0.0225, "uncached_input": 0.9})


def test_released_reservations_start_no_call_and_a_z_stamp_reads(root):
    ua.release_attempt(_call(root, model="vendor/released", prompt=0, cost=None, settle=False, dispatch=False))
    _call(root, prompt=1_000, read=900, write=50, output=5, cost=0.01)
    assert [model["model"] for model in cost_period.cost_period(root, "24h")["models"]] == [OPUS]
    assert cost_period._epoch("2026-11-12T10:00:00Z") == cost_period._epoch("2026-11-12T10:00:00+00:00") \
        == cost_period._epoch("2026-11-12T10:00:00")


def test_a_money_line_shows_the_tokens_its_shared_dollars_are_of(root):
    """A call whose cost cannot be shared adds its tokens to the period's totals, never to the shared dollars' tokens."""
    _call(root, prompt=1_000, read=600, write=200, output=10, cost=1.0)
    unshared = _call(root, prompt=200, cost=None, settle=False)
    ua.settle_attempt(unshared, {"prompt_tokens": 200, "completion_tokens": 0, "cached_tokens": None, "cache_write_tokens": 200,
                                 "prompt_cache_ttl": "1h"}, cost_usd=100.0, cost_final=True)
    model, = cost_period.cost_period(root, "24h")["models"]
    assert model["tokens"]["cache_write"] == 400 and model["tokens_shared"]["cache_write"] == 200
    assert (model["unshared_calls"], model["unshared_cost"]) == (1, pytest.approx(100.0))
    cause, = cost_period.cost_period(root, "24h")["cache_writes"]
    assert (cause["write_tokens"], cause["shared_write_tokens"], cause["unshared_calls"]) == (400, 200, 1)


def test_reported_hourly_writes_share_their_recorded_cost_through_the_period_read(root, monkeypatch):
    """The stored split, its explicit zero bin included, reads back from the period SQL as the provider's map, and a
    rate for a class the call has no tokens of (output here) leaves none of its recorded cost unshared."""
    unknown_output = pricing.PricingSchedule((4.0, 0.2, 5.0, None), cache_write_1h=11.0)
    monkeypatch.setitem(CATALOG, normalize_model_identity(OPUS), unknown_output)
    attempt = _call(root, prompt=100, cost=None, settle=False)
    ua.settle_attempt(attempt, {"prompt_tokens": 100, "completion_tokens": 0, "cached_tokens": 0, "cache_write_tokens": 100,
                                "cache_write_tokens_by_ttl": {"5m": 0, "1h": 100}, "prompt_cache_ttl": "1h"},
                      cost_usd=0.0011, cost_final=True)
    with usage_store.read(root) as txn:
        call, = txn.period_settled_calls(*txn.period_bounds(0), "1970-01-01")
    assert json.loads(call["write_split"]) == {"5m": 0, "1h": 100}
    model, = cost_period.cost_period(root, "24h")["models"]
    assert model["unshared_cost"] == 0 and model["by_class"]["cache_write"] == pytest.approx(0.0011, rel=1e-12, abs=0)
    assert (model["write_price_lifetime"], model["write_price_in_reads"]) == ("1h", 55.0)


_TEXT_REPLY = {"prompt_tokens": 100, "completion_tokens": 20, "cached_tokens": 0, "cache_write_tokens": 30,
               "prompt_cache_ttl": "1h", "cost": 0.031}


@pytest.mark.parametrize("beside,split", [
    pytest.param({"prompt_tokens_details": {"audio_tokens": 40}}, False, id="audio-input-beside-the-text-counts"),
    pytest.param({"completion_tokens_details": {"image_tokens": 7}}, False, id="image-output-beside-the-text-counts"),
    pytest.param({}, True, id="text-alone"),
])
def test_a_confirmed_bill_the_text_counts_cannot_explain_stays_whole_and_is_never_split(root, beside, split):
    """The provider billed $0.031 for 100 prompt tokens (30 of them written for an hour) and 20 output tokens. When
    the same reply also reports audio or image tokens, the row retains that usage and no text tariff can say how
    the bill divides: the period keeps the whole $0.031 as the model's cost and as its unshared part, the write's
    cause row too, and invents no share. The reply with text alone shares the $0.031 out by the price list."""
    from types import SimpleNamespace

    from ouroboros.gateway.history import make_cost_breakdown_endpoint

    attempt = _call(root, prompt=100, cost=None, settle=False)
    usage, cost, final = ua.usage_from_response({"usage": {**_TEXT_REPLY, **beside}})
    ua.settle_attempt(attempt, usage, cost_usd=cost, cost_final=final)
    with usage_store.read(root) as txn:
        row = txn.attempt(attempt.attempt_id)
    assert (row["state"], row["cost_usd"], row["cost_final"]) == ("settled", 0.031, True)
    assert ("usage" in (row.get("cost_evidence") or {})) is (not split)  # what the text counters cannot carry
    before = ua.read_usage_records(root)
    view = cost_period.cost_period(root, "24h")
    (model,), (cause,) = view["models"], view["cache_writes"]
    assert (model["cost"], model["priced_rows"], model["unpriced_rows"]) == (0.031, 1, 0)
    assert model["tokens"] == {"cache_write": 30, "cache_read": 0, "output": 20, "uncached_input": 70}
    if split:  # 70 uncached at $4, 30 written at $8 and 20 output at $20 a million weigh 280 : 240 : 400
        assert model["by_class"] == pytest.approx({"uncached_input": 0.031 * 280 / 920, "cache_write": 0.031 * 240 / 920,
                                                   "output": 0.031 * 400 / 920, "cache_read": 0.0})
        assert (model["unshared_calls"], model["unshared_cost"]) == (0, 0)
        assert model["tokens_shared"] == model["tokens"]
        assert (cause["unshared_calls"], cause["shared_write_tokens"]) == (0, 30)
        assert cause["write_cost"] == pytest.approx(0.031 * 240 / 920)
    else:
        assert model["by_class"] is None and (model["unshared_calls"], model["unshared_cost"]) == (1, 0.031)
        assert model["tokens_shared"] == dict.fromkeys(cost_period._CLASSES, 0)
        assert (cause["write_tokens"], cause["unshared_calls"], cause["shared_write_tokens"], cause["write_cost"]) \
            == (30, 1, 0, 0.0)
    assert ua.read_usage_records(root) == before  # the display reprices and rewrites nothing
    # The Costs endpoint serves that very view beside the page's unchanged total.
    served = json.loads(asyncio.run(make_cost_breakdown_endpoint(root)(SimpleNamespace(query_params={"period": "24h"}))).body)
    assert served["total_cost"] == 0.031 and served["period"]["models"] == json.loads(json.dumps(view["models"]))


def _record_dated_attempt(txn, attempt_id, root_id, start, *, task_id=None, finish=None, **price):
    row = {"kind": "attempt", "attempt_id": attempt_id, "task_id": task_id or root_id,
           "root_task_id": root_id, "parent_task_id": root_id if task_id else "",
           "category": "task", "state": "reserved", "ts": start,
           "reservation_upper_bound_usd": 4, "pricing_known": True}
    reserved = txn.write(row)
    if not price:
        return
    dispatched = txn.write({**row, "state": "dispatched"}, previous=reserved)
    txn.write({**row, "state": "settled", "ts": finish or start,
               "cost_final": True, **price}, previous=dispatched)


def test_project_days_at_the_costs_endpoint_uses_root_room_and_the_recorded_money(root, monkeypatch):
    import typing
    from types import SimpleNamespace

    from ouroboros import projects_registry as pr, store, store_import
    from ouroboros.gateway import management_contracts as contracts
    from ouroboros.gateway.cost_breakdown import make_cost_breakdown_endpoint

    since, until = "2026-02-02T00:00:00+00:00", "2026-02-08T12:00:00+00:00"
    today, previous_day = "2026-02-08T06:00:00Z", "2026-02-07T12:30:00+03:00"
    a = pr.create_project(root, "alpha")["chat_id"]
    b = pr.create_project(root, "beta")["chat_id"]
    pr.create_project(root, "empty")
    store_import.ensure_current_store(root, with_children=False)
    with store.open_store(root, write=True) as txn:
        for task, chat in (("alpha-root", a), ("child", b), ("beta-root", 1), ("main-root", 1),
                           ("nolink", None), ("place", 2**40 + 273)):
            is_child = task == "child"
            txn.store_task({"task_id": task, "chat_id": chat, "status": "running",
                            "root_task_id": "alpha-root" if is_child else task,
                            "parent_task_id": "alpha-root" if is_child else None,
                            "delegation_role": "subagent" if is_child else "root"})

    with usage_store.hold(root) as txn:
        # Exact literal that the codec leaves in extra; its receipt lands on a later day.
        _record_dated_attempt(txn, "exact", "alpha-root", since,
                              finish="2026-02-09T11:00:00Z", cost_usd="1.23456789")
        assert txn.conn.execute("SELECT cost_usd FROM attempts WHERE attempt_id='exact'").fetchone()[0] is None
        _record_dated_attempt(txn, "estimated", "alpha-root", since, task_id="child", cost_usd=0.5, cost_final=False)
        _record_dated_attempt(txn, "partial", "alpha-root", since, cost_usd=0.0625,
                              cost_final=False, cost_estimated=False, cost_partial=True)
        _record_dated_attempt(txn, "held", "alpha-root", today)  # priced upper bound, never spent money
        txn.write({"kind": "subscription_session", "attempt_id": "free", "state": "settled",
                   "task_id": "child", "root_task_id": "alpha-root", "parent_task_id": "alpha-root",
                   "category": "subagent", "ts": previous_day, "cost_usd": 0, "cost_final": True})
        _record_dated_attempt(txn, "converted", "beta-root", today, cost_usd=2)
        _record_dated_attempt(txn, "known-main", "main-root", today, cost_usd=3)
        _record_dated_attempt(txn, "unknown-room", "nolink", today, cost_usd=5)
        _record_dated_attempt(txn, "unknown-price", "place", today,
                              cost_usd=None, reservation_upper_bound_usd=None, cost_final=False)
        txn.write({"kind": "external_unmetered", "attempt_id": "no-root", "state": "unresolved",
                   "category": None, "ts": today, "pricing_known": False})
        # A fresh ledger timestamp on a compact/imported row is not a day's spend.
        txn.write({"kind": "legacy_usage", "attempt_id": "undated", "state": "settled", "root_task_id": "nolink",
                   "task_id": "nolink", "ts": today, "cost_usd": "13.1", "cost_final": True})
        _record_dated_attempt(txn, "too-early", "main-root", "2026-02-02T02:59:59.999+03:00", finish=today, cost_usd=7)
        _record_dated_attempt(txn, "at-until", "main-root", until, cost_usd=11)
        subtotal = txn.bucket("root", "alpha-root").render_breakdown()

    pr.bind_task_to_project(root, "beta-root", "beta", b, origin={"absent": "system"})
    pr.update_project(root, "beta", closed={"at": today, "by": "test", "note": "Synthetic archive"})
    monkeypatch.setattr(cost_period, "time", SimpleNamespace(time=lambda: _dt.datetime.fromisoformat(until).timestamp()))
    monkeypatch.setattr(cost_period, "_DAY_PAGE", 2)  # cross page boundaries even when several rows share a stamp
    endpoint = make_cost_breakdown_endpoint(root)

    def get(**params):
        result = asyncio.run(endpoint(SimpleNamespace(query_params=params)))
        assert result.status_code == 200
        return json.loads(result.body)

    plain = get()
    combined = get(project_days="project:alpha")
    assert {key: value for key, value in combined.items() if key != "project_days"} == plain
    view = combined["project_days"]
    assert {key: view[key] for key in ("since", "until", "timezone", "days")} == {
        "since": since, "until": until, "timezone": "UTC", "days": 7}
    # The served fields match the Python declarations, the source of the published shapes.
    day_fields = set(typing.get_type_hints(contracts.ProjectDayCost))
    assert set(view) == set(typing.get_type_hints(contracts.ProjectDaysView))
    assert view["integrity_degraded"] is False
    assert all(set(row) == day_fields for row in view["rows"]) and set(view["total"]) == day_fields - {"day"}
    hold, free, cash = view["rows"]
    assert (hold["day"], hold["settled_usd"], hold["unknown_unmetered"], hold["accounting_open_rows"]) == (
        "2026-02-08", None, 0, 1)
    assert (free["day"], free["settled_usd"], free["physical_calls"], free["subscription_sessions"]) == (
        "2026-02-07", 0.0, 0, 1)
    assert cash["day"] == "2026-02-02" and cash["settled_usd"] == 1.79706789 and cash["confirmed_usd"] == 1.29706789
    assert cash["price_basis"] == "mixed" and free["price_basis"] == "confirmed" and hold["price_basis"] == "unknown"
    # Settled estimates and the partial receipt keep the ledger's accounting open; only the held call has no outcome.
    assert (hold["open_rows"], free["open_rows"], cash["open_rows"], cash["accounting_open_rows"]) == (1, 0, 0, 2)
    assert (view["total"]["open_rows"], view["total"]["unresolved_rows"]) == (1, 0)
    for field in ("settled_usd", "confirmed_usd", "estimated_usd"):
        assert round(view["total"][field], 6) == subtotal[field], field  # compatibility remains rounded
    for field in ("unknown_unmetered", "non_final_rows", "accounting_open_rows", "physical_calls", "subscription_sessions"):
        assert view["total"][field] == subtotal[field], field
    assert [row["settled_usd"] for row in get(project_days="project:beta")["project_days"]["rows"]] == [2]
    assert [row["settled_usd"] for row in get(project_days="main")["project_days"]["rows"]] == [3]
    assert get(project_days="project:empty")["project_days"]["rows"] == []
    (gap,) = get(project_days="unassigned")["project_days"]["rows"]
    assert (gap["settled_usd"], gap["unknown_unmetered"]) == (5, 2)
    # The unresolved external call ended: its money is not reconciled, and it is no call with no outcome.
    assert (gap["open_rows"], gap["unresolved_rows"], gap["accounting_open_rows"]) == (0, 1, 1)
    month = get(project_days="main", days="30")["project_days"]
    assert month["since"] == "2026-01-10T00:00:00+00:00"
    assert [row["settled_usd"] for row in month["rows"]] == [3, 7]
    assert get(project_days="project:missing")["project_days"]["error_code"] == "unknown_selector"
    assert get(project_days="main", days="3000")["project_days"]["error_code"] == "unknown_days"

    # The room card's own read answers each view alone, field for field the combined answer's, and a room or span
    # choice never reads the page's totals (``usage_breakdown``) again; the ordinary answer is unchanged.
    totals = []
    breakdown = ua.usage_breakdown
    monkeypatch.setattr(ua, "usage_breakdown", lambda *args, **kwargs: totals.append(args) or breakdown(*args, **kwargs))
    choices = [("project:alpha", "7"), ("main", "7"), ("unassigned", "7"), ("project:beta", "30"),
               ("project:missing", "7"), ("main", "3000")]
    views = [get(project_days=selector, days=days)["project_days"] for selector, days in choices]
    assert len(totals) == len(choices)
    totals.clear()
    assert [get(view="project_days", project_days=selector, days=days) for selector, days in choices] == [
        {"project_days": answer} for answer in views]
    assert totals == [], "a room or span choice recomputed the page's totals"
    assert get() == plain and len(totals) == 1

    # A real unreadable room registry with a perfectly readable physical-attempt ledger is not a ledger outage.
    registry = pr._registry_path(root)
    saved_registry = registry.read_bytes()
    try:
        registry.write_text('{"projects":null}', encoding="utf-8")
        refused = {"available": False, "selector": "project:alpha", "error_code": "project_source_unavailable"}
        assert get(project_days="project:alpha") == {**plain, "project_days": refused}
        assert get(view="project_days", project_days="project:alpha") == {"project_days": refused}
    finally:
        registry.write_bytes(saved_registry)

    def cannot_read(*_args, **_kwargs):
        raise RuntimeError("the optional view is unavailable")

    monkeypatch.setattr(cost_period, "project_days", cannot_read)
    assert get() == plain
    refused = {"available": False, "selector": "main", "error_code": "ledger_unavailable"}
    assert get(project_days="main") == {**plain, "project_days": refused}
    assert get(view="project_days", project_days="main") == {"project_days": refused}  # its refusal answers 200


def test_project_days_span_adds_unrounded_prices_and_keeps_zero_estimates(root):
    from decimal import Decimal

    end = "2026-02-08T12:00:00+00:00"
    with usage_store.hold(root) as txn:
        for attempt, stamp, price in (
            ("tiny", "2026-02-08T06:00:00Z", {"cost_usd": 0.00000025}),
            ("string", "2026-02-07T06:00:00Z", {"cost_usd": "0.0000002500000007"}),
            ("unknown", "2026-02-06T06:00:00Z", {"cost_usd": None, "cost_final": False}),
            ("zero", "2026-02-05T06:00:00Z", {"cost_usd": 0}),
        ):
            _record_dated_attempt(txn, attempt, "", stamp, **price)
        for attempt, stamp in (("estimated", "2026-02-08T06:00:00Z"), ("estimate-only", "2026-02-04T06:00:00Z")):
            txn.write({"attempt_id": attempt, "kind": "subscription_session", "state": "settled", "ts": stamp,
                       "cost_usd": "0", "cost_final": False})

    view = cost_period.project_days(root, "unassigned", "7", now=_dt.datetime.fromisoformat(end).timestamp())
    mixed, small, unknown, confirmed, estimated = view["rows"]
    assert (mixed["settled_usd"], mixed["estimated_usd"], mixed["price_basis"]) == (0.00000025, 0, "mixed")
    assert (small["settled_usd"], small["price_basis"]) == (0.0000002500000007, "confirmed")
    assert (unknown["settled_usd"], unknown["price_basis"], unknown["unknown_unmetered"]) == (None, "unknown", 1)
    assert (confirmed["settled_usd"], confirmed["price_basis"]) == (0, "confirmed")
    assert (estimated["settled_usd"], estimated["price_basis"], estimated["subscription_sessions"]) == (0, "estimated", 1)
    assert Decimal(str(view["total"]["settled_usd"])) == Decimal("0.0000005000000007")
    assert (view["total"]["price_basis"], view["total"]["estimated_usd"], view["total"]["unknown_unmetered"]) == ("mixed", 0, 1)


def test_project_days_tells_calls_ended_unreconciled_from_calls_with_no_outcome(root, monkeypatch):
    """An answer can arrive and its accounting still fail (``usage_accounting._account_response``): that call ended
    ``unresolved``, so a day counts it apart from reserved and dispatched calls, which have no outcome. The ledger's own
    ``accounting_open_rows`` keeps counting all of them and the settled estimate; the money stays as recorded."""
    fixed = "2026-02-08T12:00:00+00:00"
    monkeypatch.setattr(usage_store, "utc_now_iso", lambda: fixed)

    def request(task):
        return ua.AttemptRequest(model=OPUS, provider="openrouter", reservation_usd=0.5, global_limit_usd=1000.0,
                                 drive_root=root, task_id=task, root_task_id="", category="task", source="test.days")

    def accounting_fails(_response):
        raise ValueError("synthetic accounting failure after the answer")

    answer = {"choices": [{"message": {"content": "done"}}]}
    assert ua.execute_physical_attempt(request("answered"), lambda: answer, extractor=accounting_fails) == answer
    ua.reserve_attempt(request("reserved"))
    ua.mark_dispatched(ua.reserve_attempt(request("dispatched")))
    unknown = ua.reserve_attempt(request("unknown-outcome"))
    ua.mark_dispatched(unknown)
    ua.mark_unresolved(unknown, "provider outcome unknown")
    for task, cost, final in (("estimate", 0.25, False), ("zero", 0, True), ("tiny", 0.00000025, True)):
        settled = ua.reserve_attempt(request(task))
        ua.mark_dispatched(settled)
        ua.settle_attempt(settled, {"prompt_tokens": 10, "completion_tokens": 1}, cost_usd=cost, cost_final=final)
    answered = next(row for row in ua.read_usage_records(root, final_only=True) if row.get("task_id") == "answered")
    assert answered["state"] == "unresolved" and "post_response_accounting_failed" in json.dumps(answered, default=str)

    view = cost_period.project_days(
        root, "unassigned", "7", now=_dt.datetime.fromisoformat(fixed).timestamp() + 1,
    )
    (day,) = view["rows"]
    assert day["day"] == "2026-02-08"
    assert (day["open_rows"], day["unresolved_rows"], day["accounting_open_rows"]) == (2, 2, 5)
    assert day["physical_calls"] == 6  # every dispatched call; the reservation never left
    assert (day["settled_usd"], day["confirmed_usd"], day["estimated_usd"], day["price_basis"]) == (
        0.25000025, 0.00000025, 0.25, "mixed")
    ledger = ua.usage_breakdown(root)
    for field in ("accounting_open_rows", "non_final_rows", "unknown_unmetered", "physical_calls"):
        assert view["total"][field] == ledger[field], field  # the ledger's own counts, unchanged
    assert {key: view["total"][key] for key in ("open_rows", "unresolved_rows")} == {"open_rows": 2, "unresolved_rows": 2}


def test_project_days_reports_ledger_integrity_degraded_without_global_aggregation(root, monkeypatch):
    """Ledger-wide quarantine state rides on both non-empty and empty room slices without global aggregation,
    keeping every known sum intact."""
    from types import SimpleNamespace

    from ouroboros.gateway.cost_breakdown import make_cost_breakdown_endpoint
    from ouroboros.usage_ledger import QUARANTINE_REL

    now = _dt.datetime(2026, 2, 8, 12, tzinfo=_dt.timezone.utc).timestamp()
    with usage_store.hold(root) as txn:
        _record_dated_attempt(txn, "known", "", "2026-02-08T08:00:00Z", cost_usd=2.5)

    monkeypatch.setattr(cost_period, "time", SimpleNamespace(time=lambda: now))
    global_reads = []
    monkeypatch.setattr(ua, "usage_breakdown", lambda *args, **kwargs: global_reads.append((args, kwargs)))
    endpoint = make_cost_breakdown_endpoint(root)

    def get_alone(selector):
        result = asyncio.run(endpoint(SimpleNamespace(query_params={"view": "project_days", "project_days": selector})))
        assert result.status_code == 200
        return json.loads(result.body)["project_days"]

    intact_nonempty = get_alone("unassigned")
    intact_empty = get_alone("main")
    assert intact_nonempty["integrity_degraded"] is False and intact_empty["integrity_degraded"] is False
    assert intact_nonempty["total"]["settled_usd"] == 2.5 and intact_empty["rows"] == []

    quarantine = root / QUARANTINE_REL
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    quarantine.write_text('{"corrupt": true}\n', encoding="utf-8")

    degraded_nonempty = get_alone("unassigned")
    degraded_empty = get_alone("main")
    assert degraded_nonempty["integrity_degraded"] is True
    assert degraded_nonempty["rows"] == intact_nonempty["rows"]
    assert degraded_nonempty["total"] == intact_nonempty["total"]
    assert degraded_empty["integrity_degraded"] is True
    assert degraded_empty["rows"] == [] and degraded_empty["total"]["settled_usd"] is None
    assert global_reads == [], "reading integrity_degraded must not trigger global aggregation"


def test_project_days_reads_only_a_recent_index_range(root, monkeypatch):
    from contextlib import contextmanager

    from ouroboros import store_import

    store_import.ensure_current_store(root, with_children=False)
    now = _dt.datetime(2026, 2, 8, 12, tzinfo=_dt.timezone.utc).timestamp()
    today, old = "2026-02-08T08:00:00Z", "2025-01-01T00:00:00Z"
    with usage_store.hold(root) as txn:
        _record_dated_attempt(txn, "recent", "", today, cost_usd=None,
                              reservation_upper_bound_usd=None, cost_final=False)
    read = usage_store.read
    work = []

    @contextmanager
    def measured(*args, **kwargs):
        with read(*args, **kwargs) as txn:
            txn.conn.set_progress_handler(lambda: work.append(1) or 0, 100)
            try:
                yield txn
            finally:
                txn.conn.set_progress_handler(None, 0)

    monkeypatch.setattr(usage_store, "read", measured)
    before = cost_period.project_days(root, "unassigned", "7", now=now)
    (only,) = before["rows"]
    assert only["settled_usd"] is None and only["unknown_unmetered"] == 1
    baseline = len(work)
    with usage_store.hold(root) as txn:
        # Exercise the realistic index shape with a cold-history tail; these rows need no summary writes to be a scan cost.
        txn.conn.executemany(
            "INSERT INTO attempts (attempt_id,revision,seq,seq_first,kind,state,category,ts_reserved,ts_last_epoch) "
            "VALUES (?,1,?,?,'attempt','settled','task',?,?)",
            ((f"past-{i}", i, i, old, cost_period._epoch(old)) for i in range(10_000, 20_000)),
        )
    work.clear()
    assert cost_period.project_days(root, "unassigned", "7", now=now) == before
    assert len(work) <= baseline + 2, "reading a seven-day slice must seek past the cold history"
