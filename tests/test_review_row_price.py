"""One review row, one price: Settings → Agents and ``## Review seat activity`` show the same words.

Both read what one review by the row actually cost (``review_price``: the median of its recent
reviews on its model, with their count and range) — ``GET /api/review-pool``'s row cost rendered by
``reviewCostText`` and the seat's ``cost_hint`` in ``## Review seat activity`` — so the owner and the model read one
number in one wording; a session row spends a seat and time, a local route no API money, a row
without history is "cost unknown".
"""
import json
import subprocess
from pathlib import Path

import pytest

from ouroboros.configured_subagents import SUBAGENTS_SETTING
from ouroboros.gateway import settings as settings_mod
from ouroboros.subagent_runtime import COST_UNKNOWN_HINT, review_facts_block
from tests._usage_store_testing import ledger_rows, root, write_journal  # noqa: F401 -- the fixture
from tests.test_review_price import MODEL, _history

REPO = Path(__file__).resolve().parents[1]
RENDER = """
import fs from 'node:fs';
import { reviewCostText } from './web/modules/subagent_status_primitives.js';
const { rows, costs } = JSON.parse(fs.readFileSync(0, 'utf8'));
console.log(JSON.stringify(rows.map((row) => reviewCostText(row, costs[row.subagent_id] || null))));
"""


def _row(row_id, target, **extra):
    return {"subagent_id": row_id, "recommended_use": "Reviews diffs.", "review_eligible": True,
            "route": {"kind": "api_model", "target_id": target}, **extra}


@pytest.fixture
def pool(root, monkeypatch):  # noqa: F811
    write_journal(root, _history())
    ledger_rows(root)  # the server's one-time import
    monkeypatch.setattr(settings_mod, "DATA_DIR", root)
    for key in ("OUROBOROS_REVIEWER_SLOTS", "OUROBOROS_REVIEW_MAX_TOKENS", "OUROBOROS_PROCESSING_PREFERENCE"):
        monkeypatch.delenv(key, raising=False)

    def use(rows):
        monkeypatch.setenv(SUBAGENTS_SETTING, json.dumps({"enabled": True, "items": rows}))
        return rows
    return use


def _seat_hints() -> dict:
    """Each seat's price as ``## Review seat activity`` (the changing part beside ``## Review``) prints it."""
    activity: list = []
    review_facts_block(activity_out=activity)
    return {seat["seat_id"]: seat["cost_hint"] for seat in json.loads(activity[0].split("\n\n", 1)[1])["seats"]}


def _settings_texts(costs: dict, rows: list) -> list:
    result = subprocess.run(["node", "--input-type=module", "-e", RENDER], cwd=REPO, capture_output=True,
                            input=json.dumps({"rows": rows, "costs": costs}), text=True, encoding="utf-8", check=True)
    return json.loads(result.stdout)


def test_one_row_shows_one_measured_price_in_settings_and_in_review(pool):
    rows = pool([_row("critic", MODEL), _row("newcomer", MODEL, delivery="packet")])
    costs = settings_mod.review_pool_payload()["row_costs"]
    shown, hints = _settings_texts(costs, rows), _seat_hints()
    assert shown == [hints["critic"], hints["newcomer"]], "Settings and ## Review show one row differently"
    assert shown[0] == "≈$0.91 per review (median of 3 measured, $0.31–$2.52)"
    assert shown[1] == COST_UNKNOWN_HINT, "a row without measured reviews has no price, never a one-call guess"


@pytest.mark.parametrize("target, words", [
    ("qwen-coder (local)", "no API cost per review"),
    ("claudexor::codex-models=gpt", "uses a session seat and time"),
])
def test_a_row_without_api_money_reads_the_same_in_both_places(pool, target, words):
    rows = pool([_row("critic", target)])
    assert _settings_texts(settings_mod.review_pool_payload()["row_costs"], rows) == [words] == list(_seat_hints().values())
