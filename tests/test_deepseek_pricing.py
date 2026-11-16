"""DeepSeek's conditional catalog price: pinned per physical attempt, settled from that pin alone.

The fixture holds the DeepSeek-provider entries of the public LiteLLM catalog exactly as fetched on
2026-11-15 (``model_prices_and_context_window.json``, sha256 3b5829b9…3889 of the whole file). Every
balance read goes through the real reader over a scripted ``requests.get``; nothing reaches the network.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ouroboros import _usage_settlement, usage_store
from ouroboros import pricing_catalog as catalog
from ouroboros import usage_accounting as ua
from ouroboros._usage_response import usage_from_response
from ouroboros.utils import utc_now_iso
from tests._tariff_guard import _no_unscripted_tariff_reads  # noqa: F401  (autouse: DeepSeek sends here)
from tests._usage_store_testing import ledger_rows

FIXTURE = Path(__file__).parent / "fixtures" / "pricing" / "litellm-deepseek-20261115.json"
UTC = timezone.utc
THURSDAY = datetime(2026, 11, 12, tzinfo=UTC)
SATURDAY = datetime(2026, 11, 14, tzinfo=UTC)
KEY = "sk-synthetic-deepseek"
TARGET = {"provider": "deepseek", "base_url": "https://api.deepseek.com/v1", "api_key": KEY,
          "resolved_model": "deepseek-v4-pro", "usage_model": "deepseek/deepseek-v4-pro"}
_BALANCE_READ = catalog._fetch_currency  # the real reader, before the guard below denies it to each case
PRO_OFF_PEAK = (1000 - 600) * 0.66e-6 + 600 * 0.022e-6 + 100 * 1.98e-6
REPLY = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100, "prompt_cache_hit_tokens": 600,
         "prompt_cache_miss_tokens": 400, "prompt_tokens_details": {"cached_tokens": 600},
         "completion_tokens_details": {"reasoning_tokens": 40}}


def _snapshot(entries: dict) -> dict:
    raw = json.dumps(entries).encode("utf-8")
    return catalog._decode({"url": catalog.CATALOG_URL, "content": raw.decode("utf-8"), "fetched_at": utc_now_iso(),
                            "sha256": hashlib.sha256(raw).hexdigest()})


def _entries() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def tariff(tmp_path, monkeypatch):
    """The retained entries behind the catalog owner, read once; a second fetch fails the test."""
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    for name, value in (("_cache", {}), ("_retry_after", {}), ("_fetching", set()), ("_currency", {}),
                        ("_currency_fetching", set())):
        monkeypatch.setattr(catalog, name, value)
    monkeypatch.setattr(catalog, "_fetch_catalog", lambda: _snapshot(_entries()))
    monkeypatch.setattr(catalog, "_fetch_currency", _BALANCE_READ)  # its transport is each test's script
    assert catalog.get_catalog_quote("deepseek-v4-pro", provider="deepseek", drive_root=tmp_path) is not None
    monkeypatch.setattr(catalog, "_fetch_catalog", lambda: pytest.fail("catalog fetched twice"))
    return tmp_path


class _Response:
    def __init__(self, status, body, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _balance(monkeypatch, *answers):
    """Answer the documented balance read from a script, recording each request; never the network."""
    calls = []

    def get(url, **kwargs):
        calls.append({"url": url, **kwargs})
        answer = answers[min(len(calls), len(answers)) - 1]
        if isinstance(answer, Exception) and not isinstance(answer, ValueError):
            raise answer
        return answer if isinstance(answer, _Response) else _Response(200, answer)

    import requests
    monkeypatch.setattr(requests, "get", get)
    return calls


USD = {"is_available": True, "balance_infos": [{"currency": "USD", "total_balance": "9.99"}]}


# --- the catalog entry ----------------------------------------------------------------------------------------

def test_the_exact_entry_is_read_for_every_route_spelling_and_no_alias_is_guessed(tariff):
    for spelling in ("deepseek::deepseek-v4-pro", "deepseek@work::deepseek-v4-pro", "deepseek/deepseek-v4-pro",
                     "deepseek-v4-pro"):
        quote = catalog.get_catalog_quote(spelling, provider="deepseek", drive_root=tariff)
        assert quote["catalog_id"] == "deepseek/deepseek-v4-pro", spelling
        assert quote["sha256"] and quote["fetched_at"] and quote["url"] == catalog.CATALOG_URL
    for spelling in ("deepseek-pro-latest", "deepseek::deepseek-flash-latest", "openai::deepseek-v4-pro",
                     "deepseek@bad name::deepseek-v4-pro", "deepseek-v4-pro (local)", "deepseek/x/y", ""):
        assert catalog.off_peak_terms(catalog.get_catalog_quote(spelling, provider="deepseek", drive_root=tariff)) \
            is None, spelling


def test_only_a_current_chat_entry_with_complete_off_peak_terms_is_priced(tariff):
    def terms(model, change=None):
        quote = catalog.get_catalog_quote(model, provider="deepseek", drive_root=tariff)
        if change:
            change(quote["entry"])
        return catalog.off_peak_terms(quote)

    pro = terms("deepseek-v4-pro")
    assert pro["rates"] == {"input_cost_per_token": 6.6e-07, "cache_read_input_token_cost": 2.2e-08,
                            "output_cost_per_token": 1.98e-06}
    assert (1, 0, 60) in pro["windows"] and (6, 0, 1440) in pro["windows"] and (1, 600, 1440) in pro["windows"]
    assert terms("deepseek-v4-flash")
    assert terms("deepseek-chat") is None, "retired 2026-07-24"
    assert terms("deepseek-v3.2") is None, "no published off-peak terms"
    off = lambda **values: lambda entry: entry["off_peak_pricing"].update(values)  # noqa: E731
    for change in (
        lambda entry: entry.update(input_cost_per_token_cache_hit=9e-9),  # the two cache-hit rates disagree
        lambda entry: entry.update(deprecation_date="2026-13-01"),  # an unreadable retirement date
        lambda entry: entry.update(input_cost_per_token_above_200k_tokens=1e-6),  # a prompt-length tier
        lambda entry: entry.update(mode="embedding"),
        lambda entry: entry.pop("output_cost_per_token"),
        off(timezone="Asia/Shanghai"), off(input_cost_per_token="0.66"), off(output_cost_per_token=-1.0),
        off(windows=[]), off(windows=[{"hours_utc": "10:00-04:00", "weekdays": [1]}]),
        off(windows=[{"hours_utc": "24:00-01:00", "weekdays": [1]}]),
        off(windows=[{"hours_utc": "00:00-01:00", "weekdays": [0]}]),
        off(windows=[{"hours_utc": "00:00-01:00", "weekdays": [1], "holidays": "CN"}]),
    ):
        assert terms("deepseek-v4-pro", change) is None
    single = terms("deepseek-v4-pro", off(windows=[{"hours_utc": "00:00-00:00", "weekdays": [6, 7]}]))
    assert single["windows"] == ((6, 0, 1440), (7, 0, 1440)), "a single string is one span"


def test_search_is_one_entry_with_one_unit(tariff):
    search = catalog.get_catalog_quote("search", provider="deepseek", drive_root=tariff)
    assert catalog.per_query_terms(search) == 0.00075 and catalog.off_peak_terms(search) is None
    assert catalog.per_query_terms(catalog.get_catalog_quote("deepseek-v4-pro", provider="deepseek",
                                                             drive_root=tariff)) is None
    search["entry"]["output_cost_per_token"] = 1e-6
    assert catalog.per_query_terms(search) is None, "a second price component is not this body's contract"


@pytest.mark.parametrize("start, end, inside", [
    (THURSDAY.replace(hour=0, minute=10), THURSDAY.replace(hour=0, minute=50), True),
    (THURSDAY.replace(hour=0, minute=59), THURSDAY.replace(hour=1, minute=0), False),  # 01:00 is already peak
    (THURSDAY.replace(hour=4, minute=30), THURSDAY.replace(hour=5, minute=59, second=59), True),
    (THURSDAY.replace(hour=2), THURSDAY.replace(hour=2, minute=1), False),
    (THURSDAY.replace(hour=9, minute=59), THURSDAY.replace(hour=10, minute=1), False),  # crossing into off-peak
    (THURSDAY.replace(hour=23), THURSDAY.replace(hour=23) + timedelta(hours=2, minutes=30), False),  # into Fri 01:00
    (SATURDAY - timedelta(hours=1), SATURDAY + timedelta(hours=30), True),  # Friday night through Sunday
    (SATURDAY + timedelta(days=1, hours=23), SATURDAY + timedelta(days=2, minutes=59), True),  # Sun into Mon 00:59
    (SATURDAY + timedelta(days=1, hours=23), SATURDAY + timedelta(days=2, hours=1), False),  # ... and Mon 01:00
    (THURSDAY.replace(hour=3), THURSDAY.replace(hour=2), False),  # an inverted clock
])
def test_only_an_interval_wholly_inside_the_windows_is_off_peak(tariff, start, end, inside):
    windows = catalog.off_peak_terms(catalog.get_catalog_quote("deepseek-v4-pro", provider="deepseek",
                                                               drive_root=tariff))["windows"]
    assert catalog.wholly_off_peak(windows, start, end) is inside
    assert catalog.wholly_off_peak(windows, start.replace(tzinfo=None), end.replace(tzinfo=None)) is False


# --- the selected key's currency ---------------------------------------------------------------------------------

def test_the_currency_is_one_bounded_read_of_the_selected_key_kept_only_in_this_process(tariff, monkeypatch):
    calls = _balance(monkeypatch, USD)
    fact = catalog.account_currency(KEY)
    assert fact["currency"] == "USD" and fact["source"] == "deepseek_user_balance" and fact["observed_at"]
    assert set(fact) == {"currency", "source", "observed_at"}, "no balance, key or fingerprint"
    assert catalog.account_currency(KEY) == fact and len(calls) == 1, "held for the pricing TTL"
    [call] = calls
    assert call["url"] == "https://api.deepseek.com/user/balance"
    assert call["headers"]["Authorization"] == f"Bearer {KEY}"
    assert (call["allow_redirects"], call["timeout"]) == (False, 5.0)
    assert KEY not in json.dumps(catalog._currency) and "9.99" not in json.dumps(catalog._currency)
    other = _balance(monkeypatch, {"balance_infos": [{"currency": "CNY", "total_balance": "1"}]})
    assert catalog.account_currency("sk-another")["currency"] == "CNY" and len(other) == 1, "per key"


@pytest.mark.parametrize("answer, reason", [
    (_Response(302, None, {"Location": "https://elsewhere.invalid/user/balance"}), "http_302"),
    (_Response(401, {"error": {}}), "http_401"),
    ({"balance_infos": [{"currency": "USD"}, {"currency": "CNY"}]}, "currency_not_single"),
    ({"balance_infos": []}, "balance_unreadable"),
    ({"balance_infos": [{"currency": "EUR"}]}, "balance_unreadable"),
    ({"is_available": True}, "balance_unreadable"),
    (_Response(200, ValueError("not json")), "ValueError"),
    (TimeoutError("synthetic"), "TimeoutError"),
])
def test_an_unreadable_currency_is_unknown_and_retried_only_after_a_pause(tariff, monkeypatch, answer, reason):
    calls = _balance(monkeypatch, answer)
    fact = catalog.account_currency(KEY)
    assert (fact["currency"], fact["reason"]) == (None, reason)
    assert catalog.account_currency(KEY)["currency"] is None and len(calls) == 1, "a 30-second pause, not a storm"
    monkeypatch.setattr(catalog.time, "monotonic", lambda: 10**9)
    _balance(monkeypatch, USD)
    assert catalog.account_currency(KEY)["currency"] == "USD"


def test_one_read_in_flight_per_key(tariff, monkeypatch):
    catalog._currency_fetching.add(hashlib.sha256(KEY.encode()).hexdigest())
    calls = _balance(monkeypatch, USD)
    assert catalog.account_currency(KEY) is None and calls == []


# --- the pin ---------------------------------------------------------------------------------------------------

def test_the_pin_is_made_for_the_selected_deepseek_key_only(tariff, monkeypatch):
    calls = _balance(monkeypatch, USD)
    pinned = catalog.pin_tariff(TARGET, text_only=True)
    assert pinned["tariff_quote"]["catalog_id"] == "deepseek/deepseek-v4-pro"
    assert (pinned["operation"], pinned["input"]) == ("chat", "text")
    assert pinned["account_currency"]["currency"] == "USD" and len(calls) == 1
    unclassified = catalog.pin_tariff(TARGET)  # an input not proven text alone is pinned as such, never priced
    assert (unclassified["input"], unclassified["account_currency"]) == ("media", None)
    searched = catalog.pin_tariff(TARGET, search=True)
    assert (searched["tariff_quote"]["catalog_id"], searched["operation"]) == ("deepseek/search", "search")
    for target in ({**TARGET, "provider": "openai"}, {**TARGET, "base_url": "https://proxy.invalid/v1"},
                   {**TARGET, "resolved_model": "deepseek-chat"}, {**TARGET, "resolved_model": "deepseek-v3.2"}):
        assert catalog.pin_tariff(target) is None
    assert len(calls) == 1, "no key leaves for a route without an applicable entry"


def test_an_expired_catalog_pins_nothing_new(tariff, monkeypatch):
    monkeypatch.setenv("OUROBOROS_PRICING_TTL_SEC", "60")
    monkeypatch.setattr(catalog.time, "time", lambda: datetime.now(UTC).timestamp() + 3600)
    monkeypatch.setattr(catalog, "_fetch_catalog", lambda: (_ for _ in ()).throw(OSError("offline")))
    calls = _balance(monkeypatch, USD)
    assert catalog.pin_tariff(TARGET) is None and calls == []
    assert catalog.price_condition("deepseek-v4-pro") == "", "a display claims no coverage from an expired record"
    assert calls == [], "nor does it fetch or read an account"


def _publish(entries: dict, fetched_at: str) -> None:
    """The record a refresh in another worker process publishes: written over the path, never through this cache."""
    from ouroboros.utils import atomic_write_json

    raw = json.dumps(entries)
    atomic_write_json(catalog._cache_path(None), {"url": catalog.CATALOG_URL, "fetched_at": fetched_at,
                                                  "sha256": hashlib.sha256(raw.encode()).hexdigest(), "content": raw})


def test_every_read_follows_the_record_now_published_and_decodes_it_once_per_change(tariff, monkeypatch):
    without_pro = {key: value for key, value in _entries().items() if key != "deepseek/deepseek-v4-pro"}
    _publish(without_pro, utc_now_iso())
    assert catalog.price_condition("deepseek-v4-pro") == ""
    decoded, real = [], catalog._decode
    monkeypatch.setattr(catalog, "_decode", lambda snapshot: decoded.append(1) or real(snapshot))
    _publish(_entries(), utc_now_iso())
    assert [catalog.price_condition(model) for model in ("deepseek-v4-pro", "deepseek-v4-pro", "deepseek-v4-flash")] \
        == ["off_peak_usd"] * 3
    assert decoded == [1], "a display decodes a changed record once, never on every read"
    (catalog._cache_path(None)).write_text("{not json", encoding="utf-8")
    assert [catalog.price_condition("deepseek-v4-pro") for _ in range(2)] == ["", ""] and decoded == [1]
    # A stale resident record yields to a fresh one published meanwhile, without a fetch of its own.
    monkeypatch.setenv("OUROBOROS_PRICING_TTL_SEC", "60")
    later = datetime.now(UTC) + timedelta(hours=1)
    _publish(_entries(), utc_now_iso())
    assert catalog.get_catalog_quote("deepseek-v4-pro", provider="deepseek") is not None
    monkeypatch.setattr(catalog.time, "time", lambda: later.timestamp())
    assert catalog.get_catalog_quote("deepseek-v4-pro", provider="deepseek", allow_live_fetch=False) is None
    _publish(_entries(), later.isoformat())
    assert catalog.get_catalog_quote("deepseek-v4-pro", provider="deepseek", allow_live_fetch=False)[
        "fetched_at"] == later.isoformat()


def test_a_record_published_right_after_this_processs_own_write_is_the_one_read_next(tariff, monkeypatch):
    """Another worker's refresh may replace the file between this fetch's write and any later look at it: the fetched
    bytes answer this call only, never under that later file's identity."""
    without_pro = {key: value for key, value in _entries().items() if key != "deepseek/deepseek-v4-pro"}
    real_write = catalog.atomic_write_json

    def write_then_another_worker_publishes(target, payload, **kwargs):
        real_write(target, payload, **kwargs)
        _publish(without_pro, utc_now_iso())

    monkeypatch.setattr(catalog, "atomic_write_json", write_then_another_worker_publishes)
    monkeypatch.setattr(catalog, "_fetch_catalog", lambda: _snapshot(_entries()))
    catalog._cache_path(None).unlink()
    assert catalog.get_catalog_quote("deepseek-v4-pro", provider="deepseek") is not None, "this call's own fetch"
    monkeypatch.setattr(catalog, "_fetch_catalog", lambda: pytest.fail("a read after the race fetches nothing"))
    decoded, real = [], catalog._decode
    monkeypatch.setattr(catalog, "_decode", lambda snapshot: decoded.append(1) or real(snapshot))
    assert [catalog.price_condition("deepseek-v4-pro") for _ in range(3)] == ["", "", ""]
    assert catalog.get_catalog_quote("deepseek-v4-pro", provider="deepseek", allow_live_fetch=False) is None
    assert catalog.price_condition("deepseek-v4-flash") == "off_peak_usd" and decoded == [1], "decoded once"


# --- settlement over the pinned row -------------------------------------------------------------------------------

@pytest.fixture
def ledger(tariff, monkeypatch):
    root = tariff / "data"
    (root / "state").mkdir(parents=True)
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    ua._reset_task_cache_splits()
    yield root
    usage_store.forget(root)


def _pinned(tariff, currency="USD", model="deepseek-v4-pro"):
    """A text candidate's pin, as ``pin_tariff`` makes it."""
    quote = catalog.get_catalog_quote(model, provider="deepseek", drive_root=tariff)
    return {"tariff_quote": quote, "operation": "chat", "input": "text",
            "account_currency": {"currency": currency, "source": "deepseek_user_balance",
                                 "observed_at": "2026-11-12T00:00:00+00:00"}}


def _clock(monkeypatch, dispatched: datetime, settled: datetime) -> None:
    """The ledger stamps its rows at ``dispatched``; settlement reads its own clock as ``settled``."""
    class _Settled(datetime):
        @classmethod
        def now(cls, tz=None):
            return settled

    monkeypatch.setattr(usage_store, "utc_now_iso", lambda *_args, **_kwargs: dispatched.isoformat())
    monkeypatch.setattr(_usage_settlement, "datetime", _Settled)


def _request(root, evidence, **overrides):
    return ua.AttemptRequest(**{"model": "deepseek/deepseek-v4-pro", "provider": "deepseek", "drive_root": root,
                                "prompt_tokens_estimate": 1000, "max_completion_tokens": 100, "task_id": "t",
                                "root_task_id": "r", "source": "test", "cost_evidence": evidence, **overrides})


def _settled_row(root, evidence, reply=REPLY, *, body_error=False, **settle):
    reservation = ua.reserve_attempt(_request(root, evidence))
    ua.mark_dispatched(reservation)
    usage, cost, final = usage_from_response({"usage": copy.deepcopy(reply), **({"error": {"code": 429}}
                                                                               if body_error else {})})
    ua.settle_attempt(reservation, usage, **({"cost_usd": cost, "cost_final": final} | settle))
    row, = [row for row in ledger_rows(root) if row["attempt_id"] == reservation.attempt_id]
    return reservation, row


@pytest.mark.parametrize("dispatched, settled", [
    (SATURDAY.replace(hour=2), SATURDAY.replace(hour=2, minute=3)),
    (THURSDAY.replace(hour=4, minute=30), THURSDAY.replace(hour=5, minute=10)),
    (THURSDAY.replace(hour=23, minute=59), (THURSDAY + timedelta(days=1)).replace(minute=0, second=30)),
])
def test_a_wholly_off_peak_usd_attempt_settles_its_estimate_from_the_pin(ledger, tariff, monkeypatch, dispatched,
                                                                          settled):
    _clock(monkeypatch, dispatched, settled)
    evidence = _pinned(tariff)
    _, row = _settled_row(ledger, evidence)
    assert row["reservation_upper_bound_usd"] is None and row["reservation_basis"] == "unknown_pricing"
    # Reasoning tokens are part of completion_tokens (40 of 100): priced once, never added again.
    assert row["cost_usd"] == pytest.approx(PRO_OFF_PEAK, rel=1e-12) and row["cached_tokens"] == 600
    assert (row["cost_estimated"], row["cost_final"]) == (True, False)
    assert row["price_source"] == f"litellm@{evidence['tariff_quote']['fetched_at']}"
    assert row["cost_evidence"]["tariff_quote"] == evidence["tariff_quote"]
    assert row["cost_evidence"]["account_currency"] == evidence["account_currency"]
    pricing = row["cost_evidence"]["tariff_pricing"]
    assert pricing["interval"] == [dispatched.isoformat(), settled.isoformat()] and pricing["period"] == "off_peak"
    assert pricing["tokens"] == {"prompt": 1000, "cache_hit": 600, "completion": 100}


@pytest.mark.parametrize("dispatched, settled", [
    (THURSDAY.replace(hour=2), THURSDAY.replace(hour=2, minute=3)),  # peak (or a Chinese holiday: unknown)
    (THURSDAY.replace(hour=9, minute=59), THURSDAY.replace(hour=10, minute=1)),  # crossing out of peak
    (THURSDAY.replace(hour=0, minute=59), THURSDAY.replace(hour=1, minute=1)),  # crossing into peak
])
def test_any_peak_instant_leaves_the_amount_unknown_not_zero(ledger, tariff, monkeypatch, dispatched, settled):
    _clock(monkeypatch, dispatched, settled)
    _, row = _settled_row(ledger, _pinned(tariff))
    assert (row["cost_usd"], row["cost_estimated"], row["cost_final"]) == (None, False, False)
    assert row["cost_evidence"]["tariff_pricing"]["unpriced"] == "not_wholly_off_peak"
    assert "price_source" not in row
    spent = ua.usage_breakdown(ledger)
    assert (spent["priced_rows"], spent["unknown_unmetered"]) == (0, 1)


@pytest.mark.parametrize("currency, reason", [("CNY", "currency_not_usd"), (None, "currency_unknown")])
def test_a_key_not_proven_to_bill_in_usd_leaves_the_amount_unknown(ledger, tariff, monkeypatch, currency, reason):
    _clock(monkeypatch, SATURDAY.replace(hour=2), SATURDAY.replace(hour=2, minute=3))
    _, row = _settled_row(ledger, _pinned(tariff, currency))
    assert row["cost_usd"] is None and row["cost_evidence"]["tariff_pricing"] == {"unpriced": reason}


@pytest.mark.parametrize("change, reason", [
    (lambda r: r.pop("prompt_cache_hit_tokens") and r.pop("prompt_tokens_details"), "cache_hit_unreported"),
    (lambda r: r.update(prompt_cache_hit_tokens=500), "cache_hit_reports_disagree"),
    (lambda r: r.update(prompt_cache_miss_tokens=401), "cache_split_inconsistent"),
    (lambda r: r.update(prompt_cache_hit_tokens=1001, prompt_tokens_details={"cached_tokens": 1001},
                        prompt_cache_miss_tokens=None), "cache_split_inconsistent"),
    (lambda r: r.update(prompt_cache_hit_tokens="600"), "usage_malformed"),
    (lambda r: r.pop("completion_tokens"), "usage_unreported"),
    (lambda r: r.update(prompt_tokens_details={"cached_tokens": 600, "cache_write_tokens": 10}), "usage_unpriced_class"),
])
def test_an_incomplete_or_inconsistent_usage_stays_unknown(ledger, tariff, monkeypatch, change, reason):
    _clock(monkeypatch, SATURDAY.replace(hour=2), SATURDAY.replace(hour=2, minute=3))
    reply = copy.deepcopy(REPLY)
    change(reply)
    _, row = _settled_row(ledger, _pinned(tariff), reply)
    assert row["cost_usd"] is None and row["cost_evidence"]["tariff_pricing"]["unpriced"] == reason


def test_an_explicit_zero_hit_and_a_tiny_charge_are_measurements(ledger, tariff, monkeypatch):
    _clock(monkeypatch, SATURDAY.replace(hour=2), SATURDAY.replace(hour=2, minute=3))
    zero_hit = {"prompt_tokens": 1, "completion_tokens": 0, "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 1}
    _, row = _settled_row(ledger, _pinned(tariff, model="deepseek-v4-flash"), zero_hit)
    assert row["cost_usd"] == pytest.approx(1.5e-07, rel=1e-12) and row["cost_usd"] > 0, "never rounded to free"
    assert row["cached_tokens"] == 0, "the top-level hit count alone is the ledger's cache count"
    assert row["cost_evidence"]["tariff_pricing"]["tokens"] == {"prompt": 1, "cache_hit": 0, "completion": 0}


def test_reported_cash_wins_and_invalid_cash_is_not_estimated(ledger, tariff, monkeypatch):
    _clock(monkeypatch, SATURDAY.replace(hour=2), SATURDAY.replace(hour=2, minute=3))
    _, row = _settled_row(ledger, _pinned(tariff), {**REPLY, "cost": 0.0})
    assert (row["cost_usd"], row["cost_final"], row["cost_estimated"]) == (0.0, True, False)
    assert "tariff_pricing" not in row["cost_evidence"]
    _, row = _settled_row(ledger, _pinned(tariff), {**REPLY, "cost": "free"})
    assert (row["cost_usd"], row["cost_estimated"], row.get("cost_invalid")) == (None, False, True)


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens", "cache_write_tokens",
                                    "cache_creation_input_tokens", "detail_write", "detail_creation", "input_write",
                                    "ttl_write"])
@pytest.mark.parametrize("bad", [-1, 0.9, False, True, "0", "1"])
@pytest.mark.parametrize("body_error", [False, True])
def test_raw_token_counts_are_not_repaired_into_a_tariff(field, bad, body_error, ledger, tariff, monkeypatch):
    _clock(monkeypatch, SATURDAY.replace(hour=2), SATURDAY.replace(hour=2, minute=3))
    reply = {"prompt_tokens": 0, "completion_tokens": 0, "prompt_cache_hit_tokens": 0}
    if field == "ttl_write":
        reply["cache_creation"] = {"ephemeral_5m_input_tokens": bad}
    elif field == "input_write":
        reply["input_tokens_details"] = {"cache_creation_input_tokens": bad}
    elif field.startswith("detail_"):
        key = "cache_write_tokens" if field == "detail_write" else "cache_creation_tokens"
        reply["prompt_tokens_details"] = {"cached_tokens": 0, key: bad}
    else:
        reply[field] = bad
    _, row = _settled_row(ledger, _pinned(tariff), reply, body_error=body_error)
    assert (row["cost_usd"], row["cost_estimated"], row["cost_final"]) == (None, False, False)
    assert row["cost_evidence"]["tariff_pricing"]["unpriced"] == "usage_malformed"
    if field in {"prompt_tokens", "completion_tokens", "cache_write_tokens"}:
        expected = None if isinstance(bad, bool) else max(0, int(bad))
        assert row[field] == expected, "the historical ledger counter projection is unchanged"


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens"])
@pytest.mark.parametrize("absent", [True, False])
def test_missing_raw_token_counts_are_not_zero(field, absent, ledger, tariff, monkeypatch):
    _clock(monkeypatch, SATURDAY.replace(hour=2), SATURDAY.replace(hour=2, minute=3))
    reply = {"prompt_tokens": 0, "completion_tokens": 0, "prompt_cache_hit_tokens": 0}
    if absent:
        reply.pop(field)
    else:
        reply[field] = None
    # Other providers' aliases cannot invent a DeepSeek count it did not report.
    reply.update(input_tokens=0, output_tokens=0)
    _, row = _settled_row(ledger, _pinned(tariff), reply)
    assert (row["cost_usd"], row["cost_estimated"], row["cost_final"]) == (None, False, False)


@pytest.mark.parametrize("cash", [0.0, 0.125])
@pytest.mark.parametrize("body_error", [False, True])
def test_raw_malformed_counts_do_not_discard_provider_cash(cash, body_error, ledger, tariff, monkeypatch):
    _clock(monkeypatch, SATURDAY.replace(hour=2), SATURDAY.replace(hour=2, minute=3))
    _, row = _settled_row(ledger, _pinned(tariff), {"prompt_tokens": -1, "completion_tokens": 0.9,
                                                        "prompt_cache_hit_tokens": "0", "cost": cash},
                          body_error=body_error)
    assert (row["cost_usd"], row["cost_estimated"], row["cost_final"]) == (cash, False, True)
    assert "tariff_pricing" not in row["cost_evidence"]


@pytest.mark.parametrize("body_error", [False, True])
def test_exact_integer_zero_usage_remains_known(body_error, ledger, tariff, monkeypatch):
    _clock(monkeypatch, SATURDAY.replace(hour=2), SATURDAY.replace(hour=2, minute=3))
    _, row = _settled_row(ledger, _pinned(tariff), {"prompt_tokens": 0, "completion_tokens": 0,
                                                        "prompt_cache_hit_tokens": 0, "cache_write_tokens": 0},
                          body_error=body_error)
    assert (row["cost_usd"], row["cost_estimated"], row["cost_final"]) == (0.0, not body_error, body_error)


def test_the_raw_witness_contains_only_token_fields_and_cannot_be_supplied_by_the_reply():
    reply = {**REPLY, "_reported_token_usage": {"prompt_tokens": 0}, "private_extension": "not a token",
             "prompt_tokens_details": {"cached_tokens": 600, "private_extension": "not a token"}}
    usage, _, _ = usage_from_response({"usage": reply})
    witness = usage["_reported_token_usage"]
    assert witness["prompt_tokens"] == 1000 and "private_extension" not in json.dumps(witness)
    reply["prompt_tokens_details"]["cached_tokens"] = 0
    assert witness["prompt_tokens_details"]["cached_tokens"] == 600, "the witness is not a mutable view"


def test_a_later_catalog_never_reprices_a_pinned_attempt(ledger, tariff, monkeypatch):
    _clock(monkeypatch, SATURDAY.replace(hour=2), SATURDAY.replace(hour=2, minute=3))
    evidence = _pinned(tariff)
    reservation = ua.reserve_attempt(_request(ledger, evidence))
    ua.mark_dispatched(reservation)
    dearer = _entries()
    dearer["deepseek/deepseek-v4-pro"]["off_peak_pricing"]["output_cost_per_token"] = 1.0
    monkeypatch.setattr(catalog, "_cache", {})
    monkeypatch.setattr(catalog, "_fetch_catalog", lambda: _snapshot(dearer))
    monkeypatch.setattr(catalog, "_fresh", lambda _snapshot: False)  # the TTL ran out during the call
    assert catalog.get_catalog_quote("deepseek-v4-pro", provider="deepseek", drive_root=tariff)["entry"][
        "off_peak_pricing"]["output_cost_per_token"] == 1.0
    # A restarted settler rebuilds the reservation from its row: the row is the carrier.
    current, = ledger_rows(ledger)
    rebuilt = ua.AttemptReservation(reservation.attempt_id, ledger, current["model"], current["provider"],
                                    current["reservation_upper_bound_usd"])
    ua.settle_attempt(rebuilt, usage_from_response({"usage": copy.deepcopy(REPLY)})[0])
    row, = ledger_rows(ledger)
    assert row["cost_usd"] == pytest.approx(PRO_OFF_PEAK, rel=1e-12)


def test_an_administrative_close_keeps_the_pin_and_a_late_receipt_is_priced_from_it(ledger, tariff, monkeypatch):
    _clock(monkeypatch, SATURDAY.replace(hour=2), SATURDAY.replace(hour=2, minute=3))
    evidence = _pinned(tariff)
    reservation = ua.reserve_attempt(_request(ledger, evidence))
    ua.mark_dispatched(reservation)
    ua.mark_unresolved(reservation, "response lost")
    assert ua.terminalize_abandoned_attempt(reservation, reason="owner_task_terminal") == "settled"
    row, = ledger_rows(ledger)
    assert (row["settle_reason"], row["cost_usd"], row["cost_final"]) == ("abandoned", None, False)
    assert {key: row["cost_evidence"].get(key) for key in evidence} == evidence, "the pre-send facts stay"
    assert "tariff_pricing" not in row["cost_evidence"], "the close prices nothing"
    ua.settle_attempt(reservation, usage_from_response({"usage": copy.deepcopy(REPLY)})[0])
    row, = ledger_rows(ledger)
    assert row["settle_reason"] == "late_receipt" and row["cost_usd"] == pytest.approx(PRO_OFF_PEAK, rel=1e-12)


def test_a_reply_cannot_supply_its_own_tariff(ledger, tariff, monkeypatch):
    _clock(monkeypatch, SATURDAY.replace(hour=2), SATURDAY.replace(hour=2, minute=3))
    forged = copy.deepcopy(_pinned(tariff))
    forged["tariff_quote"]["entry"]["off_peak_pricing"]["output_cost_per_token"] = 0.0
    reservation = ua.reserve_attempt(_request(ledger, None))  # nothing was pinned before the send
    ua.mark_dispatched(reservation)
    usage = usage_from_response({"usage": copy.deepcopy(REPLY)})[0]
    ua.settle_attempt(reservation, {**usage, "cost_evidence": forged, "price_source": "litellm@forged"})
    row, = ledger_rows(ledger)
    assert row["cost_usd"] is None and row.get("price_source") is None


def test_a_receipt_recorded_without_its_dispatch_has_no_interval_and_stays_unknown(ledger, tariff, monkeypatch):
    _clock(monkeypatch, SATURDAY.replace(hour=2), SATURDAY.replace(hour=2, minute=3))
    usage = usage_from_response({"usage": copy.deepcopy(REPLY)})[0]
    ua.record_received_attempt(_request(ledger, _pinned(tariff)), usage)
    row, = ledger_rows(ledger)
    assert (row["cost_usd"], row["cost_estimated"]) == (None, False) and "tariff_pricing" not in row["cost_evidence"]
    assert _usage_settlement._pinned_price(_pinned(tariff), usage, "")[1]["unpriced"] == "dispatch_instant_unknown"


def test_an_unknown_price_still_needs_paid_admission_at_a_reached_limit(ledger, tariff, monkeypatch):
    monkeypatch.setenv("TOTAL_BUDGET", "0")
    with pytest.raises(ua.BudgetExceeded):
        ua.reserve_attempt(_request(ledger, _pinned(tariff), global_limit_usd=0.0, root_task_id=""))


# --- the reply shows its own attempt's settlement ------------------------------------------------------------------

DS_TARGET = {"provider": "deepseek", "usage_model": "deepseek/deepseek-v4-pro", "resolved_model": "deepseek-v4-pro",
             "base_url": "https://api.deepseek.com/v1"}


def _reply_usage(root, evidence, reply):
    from ouroboros.llm import LLMClient

    body = {"id": "r", "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": copy.deepcopy(reply)}
    ua.execute_physical_attempt(_request(root, evidence), lambda: copy.deepcopy(body))
    return LLMClient(api_key="unused")._normalize_remote_response(copy.deepcopy(body), DS_TARGET,
                                                                  skip_cost_fetch=True)[1]


def test_the_reply_shows_the_same_amount_its_attempt_row_settled(ledger, tariff, monkeypatch):
    from ouroboros.llm import LLMClient
    from ouroboros.pricing import normalize_usage_cost

    _clock(monkeypatch, SATURDAY.replace(hour=2), SATURDAY.replace(hour=2, minute=3))
    usage = _reply_usage(ledger, _pinned(tariff), REPLY)
    row, = ledger_rows(ledger)
    assert (usage["cost"], usage["cost_estimated"], usage["cost_final"]) == (row["cost_usd"], True, False)
    assert usage["price_source"] == row["price_source"]
    assert normalize_usage_cost(dict(usage), model=DS_TARGET["usage_model"], use_local=False,
                                provider="deepseek")[0] == row["cost_usd"], "the loop lane keeps it, no repricing"
    other = LLMClient(api_key="unused")._normalize_remote_response(
        {"choices": [{"message": {"role": "assistant", "content": "x"}}],
         "usage": {**REPLY, "completion_tokens": 7}}, DS_TARGET, skip_cost_fetch=True)[1]
    assert other["cost"] is None and other["cost_final"] is False, "a neighbour's row is never borrowed"


def test_a_peak_reply_stays_unknown_in_its_usage_too(ledger, tariff, monkeypatch):
    _clock(monkeypatch, THURSDAY.replace(hour=2), THURSDAY.replace(hour=2, minute=3))
    usage = _reply_usage(ledger, _pinned(tariff), REPLY)
    assert usage["cost"] is None and usage["cost_final"] is False and not usage.get("cost_estimated")


def test_an_sdk_reply_whose_detail_fields_are_null_is_priced_as_text(ledger, tariff, monkeypatch):
    """The OpenAI SDK model fills the detail fields a reply leaves out with None (``audio_tokens``): unreported."""
    from openai.types.chat import ChatCompletion

    from ouroboros.llm import LLMClient

    _clock(monkeypatch, SATURDAY.replace(hour=2), SATURDAY.replace(hour=2, minute=3))
    body = {"id": "r", "object": "chat.completion", "created": 1, "model": "deepseek-v4-pro",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": copy.deepcopy(REPLY)}
    sdk = ChatCompletion.model_validate(body)
    assert sdk.model_dump()["usage"]["prompt_tokens_details"]["audio_tokens"] is None
    ua.execute_physical_attempt(_request(ledger, _pinned(tariff)), lambda: sdk)
    usage = LLMClient(api_key="unused")._normalize_remote_response(sdk.model_dump(), DS_TARGET,  # as llm.py does
                                                                   skip_cost_fetch=True)[1]
    row, = ledger_rows(ledger)
    assert row["cost_usd"] == pytest.approx(PRO_OFF_PEAK, rel=1e-12) and usage["cost"] == row["cost_usd"]


@pytest.mark.parametrize("usage, nontext", [
    ({"prompt_tokens_details": {"audio_tokens": None, "cached_tokens": 3}}, False),
    ({"prompt_tokens_details": {"audio_tokens": 5}}, True),
    ({"completion_tokens_details": {"audio_tokens": "5"}}, True),  # not a count: not text either
    ({"image_tokens": -1}, True),
    ({"promptTokensDetails": [{"modality": "IMAGE", "tokenCount": None}]}, True),
    ({"promptTokensDetails": [{"modality": "IMAGE", "tokenCount": 0}]}, False),
])
def test_only_a_stated_non_text_count_is_non_text_usage(usage, nontext):
    assert _usage_settlement.has_nontext_usage(usage) is nontext


# --- the send path pins before reserving ------------------------------------------------------------------------------

def test_the_send_pins_the_selected_targets_tariff_before_its_reservation(ledger, tariff, monkeypatch):
    from ouroboros import llm_fallback
    from ouroboros.llm import LLMClient

    class _Observed(BaseException):
        pass

    def observe(request, send, **_kwargs):
        seen.append(request)
        raise _Observed

    calls, seen = _balance(monkeypatch, USD), []
    monkeypatch.setattr(llm_fallback, "_execute_candidate", lambda request, *_args: observe(request, None))
    with pytest.raises(_Observed):
        LLMClient(api_key="unused")._create_chat_completion_with_retries(
            lambda **_kw: None, {"model": "deepseek-v4-pro", "messages": [{"role": "user", "content": "hi"}]}, TARGET)
    [request] = seen
    assert request.cost_evidence["tariff_quote"]["catalog_id"] == "deepseek/deepseek-v4-pro"
    assert request.cost_evidence["account_currency"]["currency"] == "USD" and len(calls) == 1


# --- the static disclosure: applicability of the published entry, never an account read ------------------------------

def test_spend_coverage_names_conditional_groups_apart_from_unknown_ones(tariff, monkeypatch):
    from ouroboros.pricing import price_source, spend_coverage

    calls = _balance(monkeypatch, USD)
    deepseek = {"DEEPSEEK_API_KEY": "k", "OUROBOROS_MODEL": "deepseek::deepseek-v4-pro",
                "OUROBOROS_MODEL_LIGHT": "deepseek@work::deepseek-v4-flash", "OUROBOROS_WEBSEARCH_BACKEND": "ddgs"}
    gaps, partial = spend_coverage(deepseek)
    assert gaps == ["Vision"], "Vision follows Main, yet every call of it carries an image: the text tariff prices none"
    assert {"name": "Main", "condition": "off_peak_usd"} in partial and {"name": "Light", "condition": "off_peak_usd"} \
        in partial and all(entry["name"] != "all model roles" for entry in partial)
    mixed = {**deepseek, "OUROBOROS_MODEL_LIGHT": "openai::gpt-6", "OPENAI_API_KEY": "k"}
    gaps, partial = spend_coverage(mixed)
    assert "Light" in gaps and "Main" not in gaps and {"name": "Main", "condition": "off_peak_usd"} in partial
    assert "Light" in spend_coverage({**deepseek, "OUROBOROS_MODEL_LIGHT": "deepseek::deepseek-chat"})[0], "retired"
    gaps, partial = spend_coverage({**deepseek, "OUROBOROS_MODEL_VISION": "deepseek::deepseek-v4-flash"})
    assert gaps == ["Vision"] and all(entry["name"] != "Vision" for entry in partial), "an explicit Vision route too"
    searched = {**deepseek, "OUROBOROS_WEBSEARCH_BACKEND": "deepseek"}
    assert {"name": "Web search", "condition": "usd"} in spend_coverage(searched)[1], "search inherits no off-peak"
    helper = {"subagent_id": "h", "enabled": True, "route": {"kind": "api_model", "target_id": "deepseek::deepseek-v4-pro"}}
    other = {**helper, "subagent_id": "o", "route": {"kind": "api_model", "target_id": "openai::gpt-6"}}
    assert {"name": "Agents", "condition": "off_peak_usd"} in spend_coverage(deepseek, {"items": [helper]})[1]
    assert "Agents" in spend_coverage(deepseek, {"items": [helper, other]})[0], "an unknown member is not counted"
    assert price_source("deepseek", {}, "deepseek-v3.2") == "unknown"
    assert price_source("deepseek", {}, search=True) == "usd"
    assert calls == [], "a display never reads an account"


def test_without_a_catalog_record_the_disclosure_says_unknown(tmp_path, monkeypatch):
    from ouroboros.pricing import spend_coverage

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(catalog, "_cache", {})
    assert spend_coverage({"DEEPSEEK_API_KEY": "k", "OUROBOROS_MODEL": "deepseek::deepseek-v4-pro",
                           "OUROBOROS_WEBSEARCH_BACKEND": "ddgs"}) == (["all model roles"], [])


def test_the_model_catalog_marks_only_rows_its_entry_prices_conditionally(tariff):
    import asyncio

    import httpx

    from ouroboros.gateway import models

    listed = {"object": "list", "data": [{"id": "deepseek-v4-pro"}, {"id": "deepseek-chat"}, {"id": "deepseek-flash"}]}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=listed))

    async def read():
        async with httpx.AsyncClient(transport=transport) as client:
            return await models._fetch_openai_compatible_model_catalog(
                client, "deepseek", "DeepSeek", "k", "https://api.deepseek.com/v1")

    items = {item["id"]: item for item in asyncio.run(read())}
    assert items["deepseek-v4-pro"]["price_condition"] == "off_peak_usd" and "price" not in items["deepseek-v4-pro"]
    assert items["deepseek-flash"]["price_condition"] == "off_peak_usd" and "price" not in items["deepseek-flash"]
    assert "price_condition" not in items["deepseek-chat"]


# --- applicability and preserved evidence ------------------------------------------------------------------------

def test_only_the_scoped_entries_are_applied(tariff):
    """Only the three text-model entries and per-query search entry belong to this scope; new IDs stay unpriced."""
    quote = lambda model: catalog.get_catalog_quote(model, provider="deepseek", drive_root=tariff)  # noqa: E731
    assert catalog.off_peak_terms(quote("deepseek-v4-pro")) and catalog.off_peak_terms(quote("deepseek-v4-flash"))
    assert catalog.per_query_terms(quote("search")) == 0.00075
    for model in ("deepseek-v4-flash-vision-exp",):  # complete off-peak terms, not selected
        assert quote(model) is not None and catalog.off_peak_terms(quote(model)) is None, model
        assert catalog.price_condition(model) == "", model
    assert catalog.price_condition("deepseek-v4-flash") == "off_peak_usd"


def test_the_exact_flash_entry_is_read_with_its_own_figures_and_no_neighbour_is_borrowed(tariff):
    """``deepseek-flash`` is its own exact entry (the snapshot's own rates and windows), never an alias of
    ``deepseek-v4-flash``; its ``-latest`` alias, its vision variant and a prefix spelling stay unpriced."""
    quote = lambda model: catalog.get_catalog_quote(model, provider="deepseek", drive_root=tariff)  # noqa: E731
    for spelling in ("deepseek-flash", "deepseek::deepseek-flash", "deepseek@work::deepseek-flash",
                     "deepseek/deepseek-flash"):
        assert quote(spelling)["catalog_id"] == "deepseek/deepseek-flash", spelling
        assert catalog.off_peak_terms(quote(spelling)) is not None, spelling
    # Its own record: the snapshot's v4-flash entry carries framing fields this one does not.
    assert "token_count_framing" in quote("deepseek-v4-flash")["entry"]
    assert "token_count_framing" not in quote("deepseek-flash")["entry"]
    assert catalog.price_condition("deepseek-flash") == "off_peak_usd"
    for model in ("deepseek-flash-latest", "deepseek::deepseek-flash-latest", "deepseek-flash-vision-exp",
                  "deepseek-flas", "deepseek-flash-2", "openai::deepseek-flash"):
        assert catalog.off_peak_terms(quote(model)) is None, model
        assert catalog.price_condition(model) == "", model


@pytest.mark.parametrize("change, reason", [
    (lambda r: r.update(total_tokens=1101), "usage_total_inconsistent"),
    (lambda r: r.update(total_tokens=True), "usage_malformed"),
    (lambda r: r.update(total_tokens=-1100), "usage_malformed"),
    (lambda r: r.update(total_tokens="1100"), "usage_malformed"),
    (lambda r: r.update(completion_tokens_details={"reasoning_tokens": 101}), "reasoning_exceeds_completion"),
    (lambda r: r.update(completion_tokens_details={"reasoning_tokens": 4.5}), "usage_malformed"),
])
def test_a_total_or_reasoning_count_contradicting_the_split_leaves_row_and_reply_unknown(ledger, tariff, monkeypatch,
                                                                                         change, reason):
    _clock(monkeypatch, SATURDAY.replace(hour=2), SATURDAY.replace(hour=2, minute=3))
    reply = copy.deepcopy(REPLY)
    change(reply)
    usage = _reply_usage(ledger, _pinned(tariff), reply)
    row, = ledger_rows(ledger)
    assert (row["cost_usd"], row["cost_estimated"]) == (None, False)
    assert row["cost_evidence"]["tariff_pricing"]["unpriced"] == reason
    assert usage["cost"] is None and not usage.get("cost_estimated"), "the same attempt's reply stays unknown too"


def test_an_absent_total_is_not_invented_and_does_not_discard_the_measured_split(ledger, tariff, monkeypatch):
    _clock(monkeypatch, SATURDAY.replace(hour=2), SATURDAY.replace(hour=2, minute=3))
    reply = {key: value for key, value in REPLY.items() if key not in ("total_tokens", "completion_tokens_details")}
    _, row = _settled_row(ledger, _pinned(tariff), reply)
    assert row["cost_usd"] == pytest.approx(PRO_OFF_PEAK, rel=1e-12)


@pytest.mark.parametrize("infos", [
    [{"currency": ["USD"]}], [{"currency": {"code": "USD"}}], [{"currency": None}], {"currency": "USD"},
    [{"currency": "USD"}] * 40, [{"currency": "USD"}, "USD"],
])
def test_a_malformed_balance_answer_is_an_unread_currency_with_its_pause(tariff, monkeypatch, infos):
    calls = _balance(monkeypatch, {"balance_infos": infos})
    fact = catalog.account_currency(KEY)
    assert fact["currency"] is None and fact["reason"], fact
    assert catalog.account_currency(KEY)["currency"] is None and len(calls) == 1, "malformed is paused like a failure"


def test_a_reader_fault_is_an_unread_currency_never_a_raise(tariff, monkeypatch):
    def broken(_key):
        raise RuntimeError("synthetic reader fault")

    monkeypatch.setattr(catalog, "_fetch_currency", broken)
    fact = catalog.account_currency(KEY)
    assert (fact["currency"], fact["reason"]) == (None, "RuntimeError")
    assert hashlib.sha256(KEY.encode()).hexdigest() not in catalog._currency_fetching


def test_the_read_result_is_published_before_its_in_flight_mark_is_released(tariff, monkeypatch):
    """No instant holds neither the in-flight mark nor the answer, so a concurrent caller never reads the key twice."""
    digest = hashlib.sha256(KEY.encode()).hexdigest()
    seen = []

    class _Published(dict):
        def __setitem__(self, key, value):
            seen.append((key in catalog._currency_fetching, catalog._lock.locked()))
            super().__setitem__(key, value)

    monkeypatch.setattr(catalog, "_currency", _Published())
    _balance(monkeypatch, USD)
    assert catalog.account_currency(KEY)["currency"] == "USD"
    assert seen == [(True, True)] and digest not in catalog._currency_fetching


def _observed_send(monkeypatch, target, messages, wrap=None):
    """The request one send would reserve, observed at its execution and never dispatched."""
    from ouroboros import llm_fallback
    from ouroboros.llm import LLMClient

    class _Observed(BaseException):
        pass

    seen = []

    def observe(request, *_args):
        seen.append(request)
        raise _Observed

    monkeypatch.setattr(llm_fallback, "_execute_candidate", observe)
    if wrap is not None:
        original = llm_fallback._attempt_request
        monkeypatch.setattr(llm_fallback, "_attempt_request", lambda *a, **k: wrap(original(*a, **k)))
    with pytest.raises(_Observed):
        LLMClient(api_key="unused")._create_chat_completion_with_retries(
            lambda **_kw: None, {"model": target["resolved_model"], "messages": messages}, target)
    return seen[0]


def test_the_pin_merges_into_evidence_the_request_already_carries(tariff, monkeypatch):
    from dataclasses import replace

    calls = _balance(monkeypatch, USD)
    kept = lambda request: replace(request, cost_evidence={"kept": "earlier evidence"})  # noqa: E731
    text = [{"role": "user", "content": "hi"}]
    request = _observed_send(monkeypatch, TARGET, text, kept)
    assert request.cost_evidence["kept"] == "earlier evidence"
    assert request.cost_evidence["tariff_quote"]["catalog_id"] == "deepseek/deepseek-v4-pro" and len(calls) == 1
    other = {"provider": "openai", "base_url": "https://api.openai.com/v1", "api_key": "k", "resolved_model": "gpt-x",
             "usage_model": "openai/gpt-x"}
    assert _observed_send(monkeypatch, other, text, kept).cost_evidence == {"kept": "earlier evidence"}, "no erasure"


def test_a_candidate_carrying_media_is_pinned_as_such_and_never_priced_as_text(ledger, tariff, monkeypatch):
    calls = _balance(monkeypatch, USD)
    image = [{"role": "user", "content": [{"type": "text", "text": "what is this?"},
                                          {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
    flash = {**TARGET, "resolved_model": "deepseek-v4-flash", "usage_model": "deepseek/deepseek-v4-flash"}
    request = _observed_send(monkeypatch, flash, image)
    assert request.cost_evidence["input"] == "media" and "data:image" not in json.dumps(request.cost_evidence)
    assert calls == [], "no key leaves for a call that will not be priced"
    assert _observed_send(monkeypatch, flash, [{"role": "user", "content": "hi"}]).cost_evidence["input"] == "text"
    _clock(monkeypatch, SATURDAY.replace(hour=2), SATURDAY.replace(hour=2, minute=3))
    _, row = _settled_row(ledger, {**_pinned(tariff, model="deepseek-v4-flash"), "input": "media"})
    assert row["cost_usd"] is None and row["cost_evidence"]["tariff_pricing"]["unpriced"] == "media_input_unqualified"


def test_retirement_is_judged_when_the_call_was_sent_not_when_it_settles(ledger, tariff, monkeypatch):
    sent, settled = SATURDAY.replace(hour=23, minute=59, second=30), SATURDAY + timedelta(days=1, seconds=30)
    _clock(monkeypatch, sent, settled)
    evidence = _pinned(tariff)
    evidence["tariff_quote"]["entry"]["deprecation_date"] = settled.date().isoformat()  # retired at that midnight
    _, row = _settled_row(ledger, evidence)
    assert row["cost_usd"] == pytest.approx(PRO_OFF_PEAK, rel=1e-12), row["cost_evidence"].get("tariff_pricing")
    assert catalog.off_peak_terms(evidence["tariff_quote"]) is None, "a new quote that day is no longer admitted"
