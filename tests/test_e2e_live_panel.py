"""The stand's review panel (owner decision 2026-09-06): three model families at effort low on every reviewer,
task and evolution at medium, written into every paid lane's settings as the ONE structured reviewer surface the
product reads; ``--production-panel`` leaves the tree's own defaults in place and the stub lane keeps its loopback
rows. The document must parse with the product's own parser, or the review organ would fall back silently."""
from __future__ import annotations

import json

from devtools.e2e_live import run_live_lanes, scenarios
from ouroboros.reviewer_slot_config import parse_reviewer_slots

FAKE_KEY = "sk-or-v1-e2e-live-test-key-value-never-printed-0123456789"


def test_the_stand_panel_parses_with_the_product_parser_and_names_three_families():
    cfg = parse_reviewer_slots(scenarios.STAND_PANEL_SETTINGS["OUROBOROS_REVIEWER_SLOTS"])
    triad = [(slot.target_id, slot.effort) for slot in cfg.triad]
    assert triad == [("google/gemini-3.8-flash", "low"), ("openai/gpt-5.6-luna", "low"), ("deepseek/deepseek-v4-pro", "low")]
    assert [(slot.target_id, slot.effort) for slot in cfg.scope] == [("deepseek/deepseek-v4-pro", "low")]
    assert cfg.advisory is not None and cfg.advisory.target_id == "anthropic/claude-sonnet-5" and cfg.advisory.effort == "low"
    assert {m.split("/")[0] for m, _ in triad} == {"google", "openai", "deepseek"}
    assert scenarios.STAND_PANEL_SETTINGS["OUROBOROS_EFFORT_TASK"] == "medium"
    assert scenarios.STAND_PANEL_SETTINGS["OUROBOROS_EFFORT_EVOLUTION"] == "medium"
    assert scenarios.STAND_PANEL_SETTINGS["OUROBOROS_EFFORT_REVIEW"] == scenarios.STAND_PANEL_SETTINGS["OUROBOROS_EFFORT_SCOPE_REVIEW"] == "low"


def test_paid_lanes_carry_the_panel_unless_production_panel_or_stub(monkeypatch):
    monkeypatch.setenv("TMPDIR", "/tmp")
    paid = run_live_lanes.effective_settings(run_live_lanes.parse_args(["--out", "/tmp/x"]), FAKE_KEY)
    assert json.loads(paid["OUROBOROS_REVIEWER_SLOTS"]) == scenarios.STAND_REVIEW_PANEL
    assert paid["OUROBOROS_EFFORT_REVIEW"] == "low" and paid["OUROBOROS_EFFORT_TASK"] == "medium"
    production = run_live_lanes.effective_settings(run_live_lanes.parse_args(["--out", "/tmp/x", "--production-panel"]), FAKE_KEY)
    assert not production.get("OUROBOROS_REVIEWER_SLOTS") and "OUROBOROS_EFFORT_REVIEW" not in production
    stub = run_live_lanes.effective_settings(run_live_lanes.parse_args(["--stub", "--out", "/tmp/x"]), "")
    assert stub.get("OUROBOROS_REVIEWER_SLOTS") != scenarios.STAND_PANEL_SETTINGS["OUROBOROS_REVIEWER_SLOTS"]
    assert "OUROBOROS_EFFORT_REVIEW" not in stub
