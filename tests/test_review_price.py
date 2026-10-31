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

from ouroboros.review_price import REVIEW_USAGE_CATEGORY, review_price, review_price_text
from tests._usage_store_testing import ledger_rows, root, write_journal  # noqa: F401 -- the fixture

REPO = Path(__file__).resolve().parents[1]
MODEL, LIGHT = "openrouter/fake-reviewer", "openrouter/fake-light"


def _attempt(index, *, wave, row, model, cost, priced=True, category=REVIEW_USAGE_CATEGORY):
    base = dict(attempt_id=f"a-{index}", kind="attempt", root_task_id="r", task_id="t", provider="openrouter",
                model=model, category=category, review_wave_id=wave, review_slot_id=row,
                reservation_upper_bound_usd=5.0, pricing_known=priced, ts=f"2026-10-17T10:{index:02d}:00Z")
    return [{**base, "state": "reserved"}, {**base, "state": "dispatched"},
            {**base, "state": "settled", "cost_usd": float(cost), "cost_final": True}]


def _seed(data_root):
    write_journal(data_root, _history())
    ledger_rows(data_root)  # the server's one-time import; a display read never imports


def _history():
    rows, index = [], 0
    for wave, (first, second, extraction) in enumerate([(0.10, 0.20, 0.01), (0.50, 0.40, 0.01), (1.00, 1.50, 0.02)]):
        for model, cost in ((MODEL, first), (MODEL, second), (LIGHT, extraction)):  # two turns + the extraction call
            rows += _attempt(index, wave=f"w{wave}", row="critic", model=model, cost=cost)
            index += 1
    rows += _attempt(index, wave="w-old", row="critic", model="openrouter/previous-model", cost=9.0)  # another model
    rows += _attempt(index + 1, wave="w-unpriced", row="critic", model=MODEL, cost=0.0, priced=False)
    rows += _attempt(index + 2, wave="w-plan", row="critic", model=MODEL, cost=7.0, category="plan_review_review")
    return rows


def test_one_review_costs_the_median_of_its_measured_waves(root):  # noqa: F811
    _seed(root)

    price = review_price(root, "critic", MODEL)

    assert price is not None and price.reviews == 3, "another model, an unpriced wave and other surfaces are not this price"
    assert (round(price.usd, 2), round(price.low, 2), round(price.high, 2)) == (0.91, 0.31, 2.52)
    assert review_price_text(price) == "≈$0.91 per review (median of 3 measured, $0.31–$2.52)"
    assert review_price(root, "critic", "openrouter/never-used") is None
    assert review_price(root, "stranger", MODEL) is None
    assert review_price_text(None) == "cost unknown"


def test_settings_prints_the_same_words(root, monkeypatch):  # noqa: F811
    from ouroboros.gateway import settings as settings_mod

    _seed(root)
    monkeypatch.setattr(settings_mod, "DATA_DIR", root)
    rows = [{"subagent_id": "critic", "review_eligible": True, "route": {"kind": "api_model", "target_id": MODEL}},
            {"subagent_id": "newcomer", "review_eligible": True, "route": {"kind": "api_model", "target_id": MODEL}}]
    costs = settings_mod._review_pool_costs(rows)
    assert costs["critic"]["basis"] == "observed_reviews" and costs["critic"]["reviews"] == 3
    assert costs["newcomer"] == {"usd_per_review": None, "basis": "unknown"}
    script = ("import fs from 'node:fs'; import { reviewCostText } from './web/modules/subagent_status_primitives.js';"
              "const { rows, costs } = JSON.parse(fs.readFileSync(0, 'utf8'));"
              "console.log(JSON.stringify(rows.map((row) => reviewCostText(row, costs[row.subagent_id] || null))));")
    shown = json.loads(subprocess.run(["node", "--input-type=module", "-e", script], cwd=REPO, check=True,
                                      capture_output=True, text=True, encoding="utf-8",
                                      input=json.dumps({"rows": rows, "costs": costs})).stdout)
    assert shown == [review_price_text(review_price(root, "critic", MODEL)), "cost unknown"]
