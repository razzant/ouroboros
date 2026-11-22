"""What one review by one reviewer row costs, measured from the usage ledger.

A review is every attempt of one seat in one wave — a reading reviewer's turns and the light
verdict-extraction call summed — so the price of a row is the median of its recent waves on its
model, with their count and range. No history is "cost unknown", never a one-call estimate;
Settings → Agents prints the same words.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ouroboros.review_price import PLAN_USAGE_CATEGORY, REVIEW_USAGE_CATEGORY, review_price, review_price_text
from tests._usage_store_testing import ledger_rows, root, write_journal  # noqa: F401 -- the fixture

REPO = Path(__file__).resolve().parents[1]
MODEL, LIGHT = "openrouter/fake-reviewer", "openrouter/fake-light"


def _attempt(index, *, wave, row, model, cost, priced=True, category=REVIEW_USAGE_CATEGORY, settled=True, **closing):
    """One attempt's journal records. ``priced`` is the reservation's fact (did the catalog know a price before the
    send); ``cost`` is what its settlement recorded (``None``: closed without a sum); ``settled=False`` leaves it open."""
    base = dict(attempt_id=f"a-{index}", root_task_id="r", task_id="t", **({"kind": kind} if (
                    kind := closing.pop("kind", "attempt")) else {}),
                provider=closing.pop("provider", "openrouter"), model=model, category=category, review_wave_id=wave,
                review_slot_id=row, reservation_upper_bound_usd=5.0, pricing_known=priced, ts=f"2026-10-17T10:{index:02d}:00Z")
    closed = {**base, "state": "settled", "cost_usd": None if cost is None else float(cost), "cost_final": True, **closing}
    return [{**base, "state": "reserved"}, {**base, "state": "dispatched"}, *([closed] if settled else [])]


def _seed(data_root):
    write_journal(data_root, _history())
    ledger_rows(data_root)  # the server's one-time import; a display read never imports


def _history():
    rows, index = [], 0
    for wave, (first, second, extraction) in enumerate([(0.10, 0.20, 0.01), (0.50, 0.40, 0.01), (1.00, 1.50, 0.02)]):
        # The last wave ran on a model the catalog had no price for before the send: its exact sums are the bill.
        for model, cost in ((MODEL, first), (MODEL, second), (LIGHT, extraction)):  # two turns + the extraction call
            rows += _attempt(index, wave=f"w{wave}", row="critic", model=model, cost=cost, priced=wave != 2)
            index += 1
    rows += _attempt(index, wave="w-old", row="critic", model="openrouter/previous-model", cost=9.0)  # another model
    rows += _attempt(index + 1, wave="w-plan", row="critic", model=MODEL, cost=7.0, category="plan_review_review")
    # A subscription call inside a review (its verdict read by a Light model on a subscription) closes without a
    # sum: a seat and time, no API money, so the wave keeps its measured API cost.
    rows += _attempt(index + 2, wave="w-subscription", row="critic", model=MODEL, cost=0.91)
    rows += _attempt(index + 3, wave="w-subscription", row="critic", model="claudexor::codex=light", cost=None,
                     provider="claudexor")
    # Seats with a paid call whose sum is not all there stay unmeasured, whatever the catalog knew beforehand:
    # still open, closed without a sum (also in a record that names no ``kind``, which reads as an attempt), a known
    # subtotal with an unpriced remainder, an estimate, a subscription call closed administratively.
    abandoned = dict(cost=None, provider="claudexor", cost_final=False, settle_reason="abandoned")
    for offset, (wave, unsure) in enumerate((("w-open", dict(settled=False)), ("w-no-sum", dict(cost=None)),
                                             ("w-no-kind", dict(cost=None, kind=None)),
                                             ("w-partial", dict(cost_partial=True)), ("w-abandoned", abandoned),
                                             ("w-estimated", dict(cost_final=False, cost_estimated=True)))):
        rows += _attempt(index + 4 + 2 * offset, wave=wave, row="critic", model=MODEL, cost=40.0)
        rows += _attempt(index + 5 + 2 * offset, wave=wave, row="critic", model=MODEL, **{"cost": 40.0, **unsure})
    # A row whose every call runs on a subscription: its review cost no API money. A seat that sent nothing has no cost.
    rows += _attempt(index + 20, wave="w-seat-only", row="seated", model=MODEL, cost=None, provider="claudexor")
    rows += [record for record in _attempt(index + 21, wave="w-unsent", row="idle", model=MODEL, cost=None)
             if record["state"] == "reserved"] + [dict(attempt_id=f"a-{index + 21}", kind="attempt", root_task_id="r",
             task_id="t", provider="openrouter", model=MODEL, category=REVIEW_USAGE_CATEGORY, review_wave_id="w-unsent",
             review_slot_id="idle", reservation_upper_bound_usd=5.0, pricing_known=True, state="released",
             ts="2026-10-17T10:59:00Z")]
    # A released reservation on another model beside this seat's settled call: only the
    # model whose send was witnessed enters the seat's `models`.
    rows += _attempt(index + 22, wave="w-mixed", row="mixed", model=MODEL, cost=0.60)
    rows += [dict(attempt_id=f"a-{index + 23}", kind="attempt", root_task_id="r", task_id="t",
             provider="openrouter", model="openrouter/reserved-only", category=REVIEW_USAGE_CATEGORY,
             review_wave_id="w-mixed", review_slot_id="mixed", reservation_upper_bound_usd=5.0,
             pricing_known=True, state="released", ts="2026-10-17T11:00:00Z")]
    return rows


def test_one_review_costs_the_median_of_its_measured_waves(root):  # noqa: F811
    from ouroboros import usage_store
    from ouroboros._usage_rows import summary_delta
    from ouroboros.usage_ledger import is_abandoned_settlement

    _seed(root)

    price = review_price(root, "critic", MODEL)

    assert price is not None and price.reviews == 4, (
        "a wave is measured by its recorded sums (a price unknown before the send is still a bill, a subscription "
        "call is no API money); another model, other surfaces and a wave with an open, sum-less or partial paid "
        "call are not this price")
    assert (round(price.usd, 2), round(price.low, 2), round(price.high, 2)) == (0.91, 0.31, 2.52)
    assert review_price_text(price) == "≈$0.91 per review (median of 4 measured, $0.31–$2.52)"
    assert review_price(root, "critic", "openrouter/never-used") is None
    assert review_price(root, "stranger", MODEL) is None
    assert review_price_text(None) == "cost unknown"
    # «Priced» for this view is stricter than landed spend: it accepts only final, non-partial sums and
    # subscription calls that settled without a sum. The accounting ledger still books estimates.
    with usage_store.read(root) as txn:
        seat_rows = txn.review_seat_costs(REVIEW_USAGE_CATEGORY, limit=50)
    seats = {seat["wave_id"]: bool(seat["priced"]) for seat in seat_rows}
    mixed = next(seat for seat in seat_rows if seat["wave_id"] == "w-mixed")
    by_wave: dict = {}
    for row in ledger_rows(root):
        if row.get("category") == REVIEW_USAGE_CATEGORY and row.get("state") != "released":
            delta = summary_delta(row)
            by_wave.setdefault(row["review_wave_id"], []).append(row["state"] == "settled" and (
                (row.get("provider") == "claudexor" and not is_abandoned_settlement(row))
                or (bool(delta.priced) and not delta.unknown and row.get("cost_final") is True)))
    assert {wave: measured for wave, measured in seats.items() if wave != "w-unsent"} == {
        wave: all(flags) for wave, flags in by_wave.items()}
    assert {wave for wave, measured in seats.items() if not measured} == {
        "w-open", "w-no-sum", "w-no-kind", "w-partial", "w-abandoned", "w-estimated"}
    # A review made only of subscription calls cost no API money; a seat that sent nothing is not a review.
    assert review_price_text(review_price(root, "seated", MODEL)) == "no API cost per review"
    assert review_price(root, "idle", MODEL) is None
    # A released reservation beside this seat's settled call: the send was never witnessed, so the
    # seat's `models` names only the model that ran, and only that one carries the seat's price.
    assert mixed["models"] == MODEL and mixed["priced"]
    assert review_price(root, "mixed", "openrouter/reserved-only") is None
    assert review_price(root, "mixed", MODEL).reviews == 1


def test_a_plan_ask_has_its_own_price_and_never_moves_a_reviews(root):  # noqa: F811
    _seed(root)

    plan = review_price(root, "critic", MODEL, category=PLAN_USAGE_CATEGORY)

    assert plan is not None and (plan.reviews, plan.usd) == (1, 7.0)
    assert review_price_text(plan, "plan ask") == "≈$7.00 per plan ask (1 measured)"
    assert review_price(root, "critic", MODEL).reviews == 4, "the plan wave never enters a review's median"
    assert review_price(root, "critic", LIGHT, category=PLAN_USAGE_CATEGORY) is None


def test_settings_prints_the_same_words(root, monkeypatch):  # noqa: F811
    from ouroboros.gateway import settings as settings_mod

    _seed(root)
    monkeypatch.setattr(settings_mod, "DATA_DIR", root)
    rows = [{"subagent_id": "critic", "review_eligible": True, "route": {"kind": "api_model", "target_id": MODEL}},
            {"subagent_id": "newcomer", "review_eligible": True, "route": {"kind": "api_model", "target_id": MODEL}}]
    costs = settings_mod._review_pool_costs(rows)
    assert costs["critic"]["basis"] == "observed_reviews" and costs["critic"]["reviews"] == 4
    assert costs["newcomer"] == {"usd_per_review": None, "basis": "unknown"}
    script = ("import fs from 'node:fs'; import { reviewCostText } from './web/modules/subagent_status_primitives.js';"
              "const { rows, costs } = JSON.parse(fs.readFileSync(0, 'utf8'));"
              "console.log(JSON.stringify(rows.map((row) => reviewCostText(row, costs[row.subagent_id] || null))));")
    shown = json.loads(subprocess.run(["node", "--input-type=module", "-e", script], cwd=REPO, check=True,
                                      capture_output=True, text=True, encoding="utf-8",
                                      input=json.dumps({"rows": rows, "costs": costs})).stdout)
    assert shown == [review_price_text(review_price(root, "critic", MODEL)), "cost unknown"]


def test_a_charge_below_the_printed_precision_is_never_a_zero():
    from ouroboros.review_price import ReviewPrice

    tiny = ReviewPrice(usd=0.00003, reviews=1, low=0.00003, high=0.00003)
    assert review_price_text(tiny, "first call") == "<$0.0001 per first call (1 measured)"
    spread = ReviewPrice(usd=0.02, reviews=3, low=0.00003, high=0.05)
    assert review_price_text(spread) == "≈$0.02 per review (median of 3 measured, <$0.0001–$0.05)"
    assert review_price_text(ReviewPrice(usd=0.0, reviews=2, low=0.0, high=0.0)) == "no API cost per review"
