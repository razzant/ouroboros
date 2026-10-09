"""The stand's review panel (owner decision 2026-09-06): three model families at effort low on every reviewer,
task and evolution at medium, written into every paid lane's settings; ``--production-panel`` leaves the tree's
own defaults in place and the stub lane keeps its loopback rows. The panel IS the review pool: catalog rows
marked Reviewer. The isolated settings builder drops the retired lane-era keys, so a panel written under them
vanished and the paid stand silently reviewed with the factory rows; what a lane runs is read back through the
product's read seam (``normalize_settings_raw`` -> ``review_pool_rows``)."""
from __future__ import annotations

import json

from devtools.e2e_live import run_live_lanes, scenarios
from ouroboros.config import normalize_settings_raw
from ouroboros.review_model_routes import adaptive_quorum
from ouroboros.reviewer_slot_config import review_pool_rows
from ouroboros.settings_defaults import OPENROUTER_REVIEW_DEFAULTS

FAKE_KEY = "sk-or-v1-e2e-live-test-key-value-never-printed-0123456789"


def test_the_stand_panel_is_a_pool_of_three_families_in_marked_catalog_rows():
    from ouroboros.review_pool_migration import migration_trigger

    assert migration_trigger(dict(scenarios.STAND_PANEL_SETTINGS)) == ""  # already a pool: nothing to migrate
    document = normalize_settings_raw(dict(scenarios.STAND_PANEL_SETTINGS))
    assert document["OUROBOROS_SUBAGENTS"] == scenarios.STAND_PANEL_SETTINGS["OUROBOROS_SUBAGENTS"]
    pool = [(row.target_id, row.effort, row.retrieves) for row in review_pool_rows(document)]
    # Three rows pack the brief at effort low; the fourth reads the work itself.
    assert pool == [("google/gemini-3.8-flash", "low", False), ("openai/gpt-5.6-luna", "low", False),
                    ("deepseek/deepseek-v4-pro", "low", False), ("deepseek/deepseek-v4-pro", "low", True)]
    assert {m.split("/")[0] for m, _, _ in pool} == {"google", "openai", "deepseek"}
    # The advisory reviewer is an unmarked catalog row the author may still name for a preflight.
    catalog = json.loads(document["OUROBOROS_SUBAGENTS"])["items"]
    advisory = [row for row in catalog if row["route"]["target_id"] == "anthropic/claude-sonnet-5"]
    assert len(advisory) == 1 and not advisory[0].get("review_eligible") and advisory[0]["effort"] == "low"
    assert document["OUROBOROS_EFFORT_TASK"] == "medium" and document["OUROBOROS_EFFORT_EVOLUTION"] == "medium"


def test_paid_lanes_carry_the_panel_unless_production_panel_or_stub(monkeypatch):
    # parse_args reads tempfile.gettempdir(), which caches the session's temp dir; a TMPDIR
    # env change never reaches it (test_e2e_live_runner.py's _short_tmp patches the same seam).
    monkeypatch.setattr(run_live_lanes.tempfile, "gettempdir", lambda: "/tmp")
    retired = {"OUROBOROS_REVIEWER_SLOTS", "OUROBOROS_EFFORT_REVIEW", "OUROBOROS_EFFORT_SCOPE_REVIEW"}
    paid = run_live_lanes.effective_settings(run_live_lanes.parse_args(["--out", "/tmp/x"]), FAKE_KEY)
    assert paid["OUROBOROS_SUBAGENTS"] == scenarios.STAND_PANEL_SETTINGS["OUROBOROS_SUBAGENTS"]
    assert not retired & set(paid) and paid["OUROBOROS_EFFORT_TASK"] == "medium"
    # What the paid lane RUNS: the stand's cheap rows, never the factory panel.
    assert [row.target_id for row in review_pool_rows(normalize_settings_raw(dict(paid)))] == [
        "google/gemini-3.8-flash", "openai/gpt-5.6-luna", "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-pro"]
    production = run_live_lanes.effective_settings(run_live_lanes.parse_args(["--out", "/tmp/x", "--production-panel"]), FAKE_KEY)
    assert not retired & set(production)
    # The production document names neither lanes nor a catalog: through the read seam it is
    # a never-configured install and runs the factory OpenRouter triad (quorum 2 of 3), so
    # the lane reviews with the tree's own default panel instead of an empty pool.
    assert "OUROBOROS_SUBAGENTS" not in production
    pool = review_pool_rows(normalize_settings_raw(dict(production)))
    assert [row.target_id for row in pool] == list(OPENROUTER_REVIEW_DEFAULTS["triad"]) and adaptive_quorum(len(pool)) == 2
    stub = run_live_lanes.effective_settings(run_live_lanes.parse_args(["--stub", "--out", "/tmp/x"]), "")
    assert stub.get("OUROBOROS_SUBAGENTS") != scenarios.STAND_PANEL_SETTINGS["OUROBOROS_SUBAGENTS"]
    assert not retired & set(stub)


def _lane_document(template: dict, sid: str) -> dict:
    """The document ``run_lane`` writes: the template, then the scenario's overrides over it."""
    cfg = dict(template)
    cfg.update(scenarios.SCENARIOS[sid].overrides(str(cfg.get("OUROBOROS_MODEL") or ""), cfg))
    return cfg


def _templates(monkeypatch) -> dict:
    import types

    from devtools.e2e_live import stub_lane

    monkeypatch.setattr(run_live_lanes.tempfile, "gettempdir", lambda: "/tmp")
    stub = types.SimpleNamespace(base_url="http://127.0.0.1:1")
    return {
        "stand": run_live_lanes.effective_settings(run_live_lanes.parse_args(["--out", "/tmp/x"]), FAKE_KEY),
        "production": run_live_lanes.effective_settings(
            run_live_lanes.parse_args(["--out", "/tmp/x", "--production-panel"]), FAKE_KEY),
        "stub": stub_lane.stub_settings(
            stub, run_live_lanes.effective_settings(run_live_lanes.parse_args(["--stub", "--out", "/tmp/x"]), "")),
    }


def test_t2b_every_scenarios_final_document_runs_an_executable_pool(monkeypatch):
    """The lane reviews with the document ``run_lane`` WRITES — template plus the scenario's
    overrides — not with the template alone. SW1's override replaced the catalog with its
    scout: under ``--production-panel`` (no lanes, no catalog: the never-configured template
    whose factory rows the seam would mint) and in the stub lane (keyless reviewers in the
    template's catalog) that left a structural catalog with no marked row — a loud EMPTY pool,
    no plan reviewer for the swarm root (T2b). The final document now keeps both roles: the
    scout beside the reviewers the template stood for."""
    from ouroboros.configured_subagents import parse_configured_subagents

    for name, template in _templates(monkeypatch).items():
        for sid in scenarios.SCENARIOS:
            document = _lane_document(template, sid)
            pool = review_pool_rows(normalize_settings_raw(dict(document)))
            assert pool, (name, sid, "the final document runs no reviewer")
            if name == "production":
                assert [row.target_id for row in pool] == list(OPENROUTER_REVIEW_DEFAULTS["triad"]), (name, sid)
            elif name == "stand":
                assert [row.target_id for row in pool][:3] == ["google/gemini-3.8-flash", "openai/gpt-5.6-luna",
                                                                "deepseek/deepseek-v4-pro"], (name, sid)
            else:
                assert {row.target_id for row in pool} == {"openai-compatible::mock-model"} or all(
                    row.target_id.startswith("openai-compatible::") for row in pool), (name, sid)
            if sid == "SW1":
                roster = parse_configured_subagents(document["OUROBOROS_SUBAGENTS"])
                assert roster.enabled, "the scout must stay dispatchable"
                scout = [row for row in roster.items if row.subagent_id == scenarios.SW1_ROSTER_ID]
                assert len(scout) == 1 and not scout[0].review_eligible, "the scout is a helper, not a reviewer"
                assert scout[0].route.target_id == str(template.get("OUROBOROS_MODEL") or "")
                assert all(row.subagent_id != scenarios.SW1_ROSTER_ID for row in pool)


def test_t2b_the_production_sw1_document_is_the_factory_pool_beside_the_scout(monkeypatch):
    """Exactly the tree's factory reviewer rows (``factory_review_rows`` of the production
    template) and the scout, nothing else: the catalog is a pool document already (no
    migration at the read seam), so what the file says is what runs."""
    from ouroboros import review_pool_migration as m
    from ouroboros.subscription_install_presets import factory_review_rows

    template = _templates(monkeypatch)["production"]
    document = _lane_document(template, "SW1")
    items = json.loads(document["OUROBOROS_SUBAGENTS"])["items"]
    assert items[0]["subagent_id"] == scenarios.SW1_ROSTER_ID and "review_eligible" not in items[0]
    assert items[1:] == factory_review_rows(template)
    assert m.migration_trigger(document) == "" and m.migrate_review_lanes(dict(document)) is None
    assert [row.slot_id for row in review_pool_rows(normalize_settings_raw(dict(document)))] == ["review-1", "review-2", "review-3"]


def test_t2b_a_template_catalog_without_a_mark_still_reads_as_an_empty_pool():
    """Not weakened: composing the scout beside a template catalog that marks no reviewer is
    still a structural catalog with no marked row — a loud empty pool, nothing minted."""
    helper_only = json.dumps({"enabled": True, "items": [{
        "subagent_id": "helper", "recommended_use": "Helps.", "route": {"kind": "api_model", "target_id": "x/y"}}]})
    document = {"OPENROUTER_API_KEY": FAKE_KEY, "OUROBOROS_SUBAGENTS": scenarios.sw1_roster("x/child", {
        "OPENROUTER_API_KEY": FAKE_KEY, "OUROBOROS_SUBAGENTS": helper_only})}
    loaded = normalize_settings_raw(dict(document))
    assert loaded["OUROBOROS_SUBAGENTS"] == document["OUROBOROS_SUBAGENTS"] and review_pool_rows(loaded) == []
    assert [row["subagent_id"] for row in json.loads(loaded["OUROBOROS_SUBAGENTS"])["items"]] == ["scout", "helper"]
