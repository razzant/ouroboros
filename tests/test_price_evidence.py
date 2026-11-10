"""Direct-provider price evidence: the peak clock, the FX cross, and unknown-on-stale.

Offline by construction: the CBR table is a fixture on disk and every call passes
``allow_live_fetch=False``, so a machine without network still proves the arithmetic.
The datasheet's own observation date is pinned to today inside the fixture, because a
test of arithmetic must not fail on the calendar — staleness has its own test.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from ouroboros import price_evidence
from ouroboros.pricing import estimate_cost_optional, get_pricing

UTC = timezone.utc
MONDAY = datetime(2026, 11, 2, tzinfo=UTC)  # the day this evidence was read
SATURDAY = datetime(2026, 11, 7, tzinfo=UTC)
SUNDAY = datetime(2026, 11, 8, tzinfo=UTC)
PEAK = MONDAY.replace(hour=2)      # inside 01:00-04:00
OFF_PEAK = MONDAY.replace(hour=5)  # between the two peak windows
USD_RUB = 80.5573                  # CBR, effective 31.10.2026
CNY_RUB = 12.0873


def _cbr_payload(currency_date: str = "") -> str:
    """A CBR record, dated today unless a test wants it old.

    The freshness rule (``MAX_FX_AGE_DAYS``) is part of what this module promises, so a
    fixture pinned to the day the rate was read would make a test of arithmetic fail on
    the calendar instead.
    """
    currency_date = currency_date or date.today().strftime("%d.%m.%Y")
    return json.dumps({
        "schema": 1,
        "currency_date": currency_date,
        "fetched_at": datetime.now(UTC).isoformat(),
        "source": price_evidence.CBR_URL,
        "rates": {"USD": USD_RUB, "CNY": CNY_RUB},
    })


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    """A dated CBR record on disk, and a datasheet observed today."""
    path = tmp_path / "cbr_rates.json"
    path.write_text(_cbr_payload(), encoding="utf-8")
    monkeypatch.setattr(price_evidence, "_fx_state_path", lambda: path)
    today = date.today().isoformat()
    for sheet in price_evidence._DATASHEET["providers"].values():
        monkeypatch.setitem(sheet, "observed_at", today)
    price_evidence._fx_memo.clear()
    yield path
    price_evidence._fx_memo.clear()


# --- the clock ---------------------------------------------------------------

@pytest.mark.parametrize("moment, expected", [
    (MONDAY.replace(hour=0, minute=59), "off_peak"),
    (MONDAY.replace(hour=1), "peak"),
    (MONDAY.replace(hour=3, minute=59), "peak"),
    (MONDAY.replace(hour=4), "off_peak"),
    (MONDAY.replace(hour=5, minute=59), "off_peak"),
    (MONDAY.replace(hour=6), "peak"),
    (MONDAY.replace(hour=9, minute=59), "peak"),
    (MONDAY.replace(hour=10), "off_peak"),
    (MONDAY.replace(hour=23), "off_peak"),
    (SATURDAY.replace(hour=2), "off_peak"),
    (SUNDAY.replace(hour=7), "off_peak"),
])
def test_peak_follows_the_providers_own_windows(moment, expected):
    assert price_evidence.deepseek_tier(moment) == expected


def test_a_weekend_never_pays_the_peak_rate(evidence):
    saturday_night = SATURDAY.replace(hour=2)
    peak = estimate_cost_optional("deepseek::deepseek-flash", 1_000_000, 0,
                                  allow_live_fetch=False, at=PEAK)
    weekend = estimate_cost_optional("deepseek::deepseek-flash", 1_000_000, 0,
                                     allow_live_fetch=False, at=saturday_night)
    assert peak == pytest.approx(0.30, abs=1e-9)
    assert weekend == pytest.approx(0.15, abs=1e-9)


# --- DeepSeek: the published rates ------------------------------------------

def test_deepseek_bills_miss_and_output_per_tier(evidence):
    peak = estimate_cost_optional("deepseek::deepseek-flash", 1_000_000, 1_000_000,
                                  allow_live_fetch=False, at=PEAK)
    off = estimate_cost_optional("deepseek::deepseek-flash", 1_000_000, 1_000_000,
                                 allow_live_fetch=False, at=OFF_PEAK)
    assert peak == pytest.approx(0.30 + 1.20, abs=1e-9)
    assert off == pytest.approx(0.15 + 0.60, abs=1e-9)


def test_deepseek_cache_hit_is_billed_at_the_hit_rate(evidence):
    cost = estimate_cost_optional("deepseek::deepseek-flash", 1_000_000, 0,
                                  cache_usage={"cached_tokens": 1_000_000},
                                  allow_live_fetch=False, at=PEAK)
    assert cost == pytest.approx(0.006, abs=1e-9)


def test_a_reported_cache_write_does_not_blank_the_estimate(evidence):
    # DeepSeek bills no separate write: the write path IS the cache-miss path.
    cost = estimate_cost_optional("deepseek::deepseek-flash", 1_000_000, 0,
                                  cache_usage={"cache_write_tokens": 500_000},
                                  allow_live_fetch=False, at=OFF_PEAK)
    assert cost == pytest.approx(0.15 * 0.5, abs=1e-9)


def test_the_pro_model_is_four_times_the_flash_peak(evidence):
    flash = price_evidence.prices("deepseek", "deepseek-flash", at=PEAK, allow_live_fetch=False)
    pro = price_evidence.prices("deepseek", "deepseek-v4-pro", at=PEAK, allow_live_fetch=False)
    assert pro.input_per_million == pytest.approx(flash.input_per_million * 4.4, rel=1e-6)
    assert pro.output_per_million == pytest.approx(3.96, abs=1e-9)


def test_a_retired_alias_bills_as_its_successor(evidence):
    alias = price_evidence.prices("deepseek", "deepseek-v4-flash", at=PEAK, allow_live_fetch=False)
    canonical = price_evidence.prices("deepseek", "deepseek-flash", at=PEAK, allow_live_fetch=False)
    assert alias.input_per_million == canonical.input_per_million


# --- GigaChat: roubles, VAT, and the CBR cross -------------------------------

def test_gigachat_uses_the_appendix_the_vat_factor_and_the_cbr_rate(evidence):
    unit = price_evidence.prices("gigachat", "GigaChat-3-Ultra", allow_live_fetch=False)
    assert unit is not None
    assert unit.native_currency == "RUB"
    assert unit.vat_factor == pytest.approx(1.22)
    assert unit.tier == "flat"
    assert unit.input_per_million == pytest.approx(78.87 * 1.22 / USD_RUB, abs=1e-4)
    assert unit.output_per_million == pytest.approx(236.56 * 1.22 / USD_RUB, abs=1e-4)
    assert unit.as_of == date.today().isoformat()
    assert unit.source["ref"] == "foundation-models.pdf"
    assert unit.source["doc_version"] == "261029"


def test_gigachat_cost_is_the_converted_appendix(evidence):
    cost = estimate_cost_optional("gigachat::GigaChat-3-Pro", 1_000_000, 1_000_000,
                                  allow_live_fetch=False)
    expected = (59.86 + 144.58) * 1.22 / USD_RUB
    assert cost == pytest.approx(expected, abs=1e-4)


def test_gigachat_has_no_night_and_no_weekend_tier(evidence):
    day = price_evidence.prices("gigachat", "GigaChat-3-Ultra", at=PEAK, allow_live_fetch=False)
    night = price_evidence.prices("gigachat", "GigaChat-3-Ultra", at=SATURDAY.replace(hour=2),
                                  allow_live_fetch=False)
    assert night.input_per_million == day.input_per_million


def test_lightning_is_the_owners_one_fifth_of_ultra(evidence):
    lightning = price_evidence.prices("gigachat", "GigaChat-3-Lightning", allow_live_fetch=False)
    ultra = price_evidence.prices("gigachat", "GigaChat-3-Ultra", allow_live_fetch=False)
    assert lightning.input_per_million == pytest.approx(ultra.input_per_million / 5, abs=2e-6)
    assert lightning.output_per_million == pytest.approx(ultra.output_per_million / 5, abs=2e-6)
    assert any("PROXY" in note for note in lightning.provenance)
    assert any("owner decision" in note for note in lightning.provenance)


def test_the_rouble_is_the_single_cross_for_the_yuan(evidence):
    table = price_evidence.cbr_rates(allow_live_fetch=False)
    assert price_evidence.to_usd(1.0, "CNY", table) == pytest.approx(CNY_RUB / USD_RUB, abs=1e-6)
    assert price_evidence.to_usd(USD_RUB, "USD", table) == pytest.approx(USD_RUB, abs=1e-6)
    # A real amount, not the identity 1.0: one hundred yuan is not one hundred dollars.
    assert price_evidence.to_usd(100.0, "CNY", table) == pytest.approx(
        100.0 * CNY_RUB / USD_RUB, abs=1e-6)
    assert price_evidence.to_usd(100.0, "RUB", table) == pytest.approx(
        100.0 / USD_RUB, abs=1e-6)


def test_the_daily_fetch_carries_verify_on_the_client_and_nothing_on_the_request(monkeypatch):
    """The one line that made every rouble price unknown.

    ``verify`` is a CLIENT parameter in httpx. Passing it to ``Client.get`` raised
    TypeError, the broad ``except`` around the fetch swallowed it, and the daily
    record was never written — while a direct GET of the same URL answered 200.
    """
    import httpx

    from ouroboros import net_transport

    seen = {}

    class _Response:
        content = b"<ValCurs/>"

        def raise_for_status(self) -> None:
            return None

    class _Client:
        def __init__(self, **kwargs):
            seen["client"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def get(self, url, **kwargs):
            seen["get"] = {"url": url, **kwargs}
            return _Response()

    monkeypatch.setattr(httpx, "Client", _Client)
    monkeypatch.setattr(net_transport, "extra_ca_bundle", lambda: "C:/tmp/bundle.pem")
    monkeypatch.setattr(price_evidence, "_parse_cbr_xml", lambda raw: {"currency_date": "31.10.2026",
                                                                      "rates": {"USD": USD_RUB}})

    assert price_evidence._fetch_cbr()["rates"] == {"USD": USD_RUB}
    assert seen["client"].get("verify") == "C:/tmp/bundle.pem"
    assert seen["get"] == {"url": price_evidence.CBR_URL}


# --- unknown stays unknown ---------------------------------------------------

def test_a_model_the_datasheet_does_not_cover_is_unknown(evidence):
    assert price_evidence.prices("gigachat", "GigaChat-2-Pro", allow_live_fetch=False) is None
    assert estimate_cost_optional("gigachat::GigaChat-2-Pro", 10, 10,
                                  allow_live_fetch=False) is None


def test_an_unknown_provider_is_not_priced(evidence):
    assert price_evidence.prices("anthropic", "claude-haiku", allow_live_fetch=False) is None
    assert get_pricing(provider="anthropic", allow_live_fetch=False) == {}


def test_stale_evidence_becomes_unknown(evidence, monkeypatch):
    monkeypatch.setitem(price_evidence._DATASHEET["providers"]["deepseek"], "observed_at", "2020-01-01")
    assert price_evidence.prices("deepseek", "deepseek-flash", at=PEAK, allow_live_fetch=False) is None
    assert estimate_cost_optional("deepseek::deepseek-flash", 1000, 100,
                                  allow_live_fetch=False, at=PEAK) is None


def test_an_expired_fx_rate_becomes_unknown(evidence):
    evidence.write_text(_cbr_payload(currency_date="01.01.2026"), encoding="utf-8")
    price_evidence._fx_memo.clear()
    assert price_evidence.prices("gigachat", "GigaChat-3-Ultra", allow_live_fetch=False) is None
    assert estimate_cost_optional("gigachat::GigaChat-3-Ultra", 1000, 100,
                                  allow_live_fetch=False) is None


def test_a_missing_fx_table_becomes_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(price_evidence, "_fx_state_path", lambda: tmp_path / "absent.json")
    monkeypatch.setitem(price_evidence._DATASHEET["providers"]["gigachat"], "observed_at",
                        date.today().isoformat())
    price_evidence._fx_memo.clear()
    try:
        assert price_evidence.prices("gigachat", "GigaChat-3-Ultra", allow_live_fetch=False) is None
    finally:
        price_evidence._fx_memo.clear()


def test_a_stale_fx_record_becomes_unknown(tmp_path, monkeypatch):
    """A daily rate keeps meaning for days, not weeks: past MAX_FX_AGE_DAYS it is a guess."""
    path = tmp_path / "cbr_rates.json"
    path.write_text(_cbr_payload("01.01.2020"), encoding="utf-8")
    monkeypatch.setattr(price_evidence, "_fx_state_path", lambda: path)
    monkeypatch.setitem(price_evidence._DATASHEET["providers"]["gigachat"], "observed_at",
                        date.today().isoformat())
    price_evidence._fx_memo.clear()
    try:
        assert price_evidence.prices("gigachat", "GigaChat-3-Ultra", allow_live_fetch=False) is None
    finally:
        price_evidence._fx_memo.clear()


def test_a_us_priced_route_needs_no_fx_at_all(evidence):
    evidence.unlink()
    price_evidence._fx_memo.clear()
    assert price_evidence.prices("deepseek", "deepseek-flash", at=PEAK,
                                 allow_live_fetch=False) is not None


# --- wiring into the money substrate ----------------------------------------

def test_catalog_keys_are_the_route_identities(evidence):
    from ouroboros.provider_models import normalize_model_identity

    deepseek = get_pricing(provider="deepseek", allow_live_fetch=False)
    assert set(deepseek) == {
        normalize_model_identity("deepseek::deepseek-flash"),
        normalize_model_identity("deepseek::deepseek-v4-pro"),
    }
    gigachat = get_pricing(provider="gigachat", allow_live_fetch=False)
    assert normalize_model_identity("gigachat::GigaChat-3-Ultra") in gigachat
    assert normalize_model_identity("gigachat::GigaChat-3-Lightning") in gigachat


def test_the_schedule_carries_four_slots_and_its_provenance(evidence):
    schedule = get_pricing(provider="deepseek", allow_live_fetch=False)[
        "deepseek/deepseek-flash"]
    assert len(schedule) == 4
    assert schedule[0] is not None and schedule[3] is not None
    assert schedule.source["kind"] == "direct_price_evidence"
    assert schedule.source["tier"] in {"peak", "off_peak"}


def test_an_estimate_never_becomes_a_final_cost(evidence):
    from ouroboros.pricing import normalize_usage_cost

    usage = {"prompt_tokens": 1000, "completion_tokens": 100}
    cost, _, provider, estimated = normalize_usage_cost(
        usage, model="deepseek::deepseek-flash", use_local=False, provider="deepseek")
    assert provider == "deepseek"
    assert cost is not None and estimated is True
    assert usage["cost_final"] is False


# --- the datasheet is evidence, not a table ---------------------------------

def test_every_row_names_its_source_and_deepseek_halves_off_peak():
    for name, sheet in price_evidence._DATASHEET["providers"].items():
        source = sheet["source"]
        assert source.get("ref"), name
        assert date.fromisoformat(str(sheet["observed_at"]))
        for row_key, row in sheet["rows"].items():
            assert row.get("provenance"), (name, row_key)
            assert row.get("id"), (name, row_key)
    for row in price_evidence._DATASHEET["providers"]["deepseek"]["rows"].values():
        for kind in ("cache_hit", "cache_miss", "output"):
            assert row[kind]["off_peak"] == pytest.approx(row[kind]["peak"] / 2, rel=1e-9)
