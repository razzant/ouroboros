"""One review row, one price: Settings → Agents and ``## Review`` show the same full call.

Both price a catalog row through ``review_helpers.review_row_call_usd`` — the
``GET /api/review-pool`` body Settings renders with ``reviewCostText`` and the seat's
``cost_hint`` in ``## Review`` — so the owner and the model read one number in one wording,
and neither calls it a cap on a whole review.
"""
import json
import subprocess
import time
from pathlib import Path

import pytest

from ouroboros import pricing, reviewer_window
from ouroboros.configured_subagents import SUBAGENTS_SETTING
from ouroboros.gateway.settings import review_pool_payload
from ouroboros.subagent_runtime import COST_UNKNOWN_HINT, review_facts_block
from ouroboros.tools.review_helpers import REVIEW_PROMPT_TOKEN_BUDGET

REPO = Path(__file__).resolve().parents[1]
MODEL = "openai/fake-reviewer"
TARIFF = {MODEL: (2.0, None, None, 8.0)}
RENDER = """
import fs from 'node:fs';
import { reviewCostText } from './web/modules/subagents_settings.js';
const { rows, costs } = JSON.parse(fs.readFileSync(0, 'utf8'));
console.log(JSON.stringify(rows.map((row) => reviewCostText(row, costs[row.subagent_id] || null))));
"""
ROWS = [
    {"subagent_id": "reading-critic", "recommended_use": "Reviews diffs.", "review_eligible": True,
     "route": {"kind": "api_model", "target_id": MODEL}},
    {"subagent_id": "packet-critic", "recommended_use": "Reviews diffs.", "review_eligible": True,
     "route": {"kind": "api_model", "target_id": MODEL}, "delivery": "packet"},
]


@pytest.fixture
def pool_on_a_272k_route(monkeypatch):
    """Two marked rows on one route whose window reads 272K; no tariff cached, none fetchable."""
    for key in ("OUROBOROS_REVIEWER_SLOTS", "OUROBOROS_REVIEW_MAX_TOKENS", "OUROBOROS_PROCESSING_PREFERENCE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(SUBAGENTS_SETTING, json.dumps({"enabled": True, "items": ROWS}))
    for name, empty in (("_cached_pricing", {}), ("_pricing_fetched_at", {}), ("_pricing_retry_after", {}),
                        ("_pricing_fetch_in_progress", set())):
        monkeypatch.setattr(pricing, name, empty)
    monkeypatch.setattr(pricing, "_fetch_live_rows", lambda provider, model="": {})
    real = reviewer_window.resolve_reviewer_window
    monkeypatch.setattr(reviewer_window, "resolve_reviewer_window", lambda model, **kw: (
        reviewer_window.ReviewerWindow(window_tokens=272_000, status="confirmed", model=model)
        if model == MODEL else real(model, **kw)))


def _seat_hints() -> dict:
    text = review_facts_block()
    return {seat["seat_id"]: seat["cost_hint"] for seat in json.loads(text.split("\n\n", 1)[1])["pool"]}


def _settings_texts(costs: dict, rows: list = ROWS) -> list:
    result = subprocess.run(["node", "--input-type=module", "-e", RENDER], cwd=REPO, capture_output=True,
                            input=json.dumps({"rows": rows, "costs": costs}), text=True, encoding="utf-8", check=True)
    return json.loads(result.stdout)


def test_one_row_shows_one_price_in_settings_and_in_review(pool_on_a_272k_route, monkeypatch):
    monkeypatch.setitem(pricing._cached_pricing, "openrouter", dict(TARIFF))
    monkeypatch.setitem(pricing._pricing_fetched_at, "openrouter", time.time())
    costs = review_pool_payload()["row_costs"]
    hints, shown = _seat_hints(), _settings_texts(costs)

    usd = costs["reading-critic"]["usd_per_review"]
    assert costs["packet-critic"] == costs["reading-critic"] == {"usd_per_review": usd, "basis": "route_tariff"}
    for row, settings_text in zip(ROWS, shown):
        assert settings_text == hints[row["subagent_id"]], "Settings and ## Review show one row differently"
    reading, packet = shown
    assert reading.startswith("≈$") and reading.endswith(" per full call (route tariff); a reading reviewer makes several")
    assert packet == reading.split(";")[0], "a packet row has no several-calls clause"
    assert "worst-case" not in reading + packet
    # One call inside the route's 272K window, never a 920K-token prompt the model cannot take.
    assert usd < pricing.estimate_cost_optional(MODEL, REVIEW_PROMPT_TOKEN_BUDGET, 0, provider="openrouter")


@pytest.mark.parametrize("target, words", [
    ("qwen-coder (local)", "no API cost per review"),
    ("claudexor::codex-models=gpt", "uses a session seat and time"),
])
def test_a_row_without_an_api_tariff_reads_the_same_in_both_places(pool_on_a_272k_route, monkeypatch, target, words):
    rows = [{**ROWS[0], "route": {"kind": "api_model", "target_id": target}}]
    monkeypatch.setenv(SUBAGENTS_SETTING, json.dumps({"enabled": True, "items": rows}))
    assert _settings_texts(review_pool_payload()["row_costs"], rows) == [words] == list(_seat_hints().values())


def test_review_context_never_waits_on_a_tariff_fetch(pool_on_a_272k_route, monkeypatch):
    fetched = []
    monkeypatch.setattr(pricing, "_fetch_live_rows", lambda provider, model="": fetched.append(provider) or dict(TARIFF))

    assert set(_seat_hints().values()) == {COST_UNKNOWN_HINT}
    assert fetched == [], "## Review prices from cached tariffs only"
    costs = review_pool_payload()["row_costs"]
    assert fetched == ["openrouter"] and costs["reading-critic"]["basis"] == "route_tariff"
    assert set(_seat_hints().values()) == set(_settings_texts(costs)), "once cached, both show the same price"
