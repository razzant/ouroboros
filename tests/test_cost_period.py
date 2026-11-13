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
CATALOG = {normalize_model_identity(OPUS): (4.0, 0.2, 5.0, 20.0)}


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


def _call(data, *, model=OPUS, prompt, read=0, write=0, output=0, cost, ttl="1h", prefix=None, settle=True, dispatch=True):
    physical = PhysicalAttemptContext(
        profile="owner_max", rendered_mode="max", measurement_basis="fresh_route_usage", route_fp="fp",
        round_id="exec:round:1", target_total_tokens=None, capacity_total_tokens=1_000_000, context_target_miss=False,
        automatic_pass_used=False, prefix=prefix)
    attempt = ua.reserve_attempt(ua.AttemptRequest(
        model=model, provider="openrouter", reservation_usd=5.0, global_limit_usd=1000.0, drive_root=data,
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
    assert model["by_class"]["cache_write"] == pytest.approx(1.10 * 0.80 / 1.02, abs=1e-6)  # amounts keep six decimals
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
    assert (unlisted["write_price_in_reads"], unlisted["write_price_lifetime"]) == (None, "")
    # A model none of whose calls has a recorded price has an unknown cost, never $0.
    assert (by_model[OPUS]["calls"], by_model[OPUS]["cost"], by_model[OPUS]["open_rows"]) == (1, None, 1)


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
                                                              "input_cache_read": "0.0000002", "input_cache_write": "0.000005"}},
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
    exact = pricing._cost_addends(schedule, 300_000, 1_000, usage, exact_cache_tiers=True)
    weights = pricing.cost_class_weights(schedule, usage)
    assert weights["cache_write"] == pytest.approx(exact[2] + exact[3]) == pytest.approx(50_000 * 15.0 / 1e6)


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
