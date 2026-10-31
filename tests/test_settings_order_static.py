"""The section orders the owner approved under docs/DESIGN.md §5 "Order by importance and expected
use" (answers 1A 2A 3A 4A): Settings → Behavior, Advanced and Accounts, the budget pair first in
Runtime Limits, and the Dashboard tabs. A new section takes the place the rule gives it, so adding
one is a conscious edit here, never a side effect."""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
SETTINGS_UI = (ROOT / "web" / "modules" / "settings_ui.js").read_text(encoding="utf-8")
DASHBOARD = (ROOT / "web" / "modules" / "dashboard.js").read_text(encoding="utf-8")


def _panel(name: str) -> str:
    start = SETTINGS_UI.index(f'data-settings-panel="{name}"')
    return SETTINGS_UI[start:SETTINGS_UI.index("</section>", start)]


def _titles(name: str) -> list[str]:
    return re.findall(r"<h3>(.*?)</h3>", _panel(name))


def test_behavior_runs_from_review_and_power_to_the_host_tail() -> None:
    assert _titles("behavior") == [
        "Review Enforcement", "Task Result Review", "Max Review Cycles",
        "Access", "Safety Supervisor", "Post-Task Self-Evolution", "Consciousness",
        "Context Mode", "Image Input", "Skills",
        "Prompt Cache TTL", "External Skills Repo", "ClawHub Marketplace",
        "Update Channel", "Startup &amp; background",
    ]


def test_advanced_opens_with_the_limits_and_ends_with_the_danger_zone() -> None:
    assert _titles("advanced") == [
        "Runtime Limits", "MCP Servers", "Source Control", "Local Model Runtime", "Extension Settings",
        "Extra CA Certificates", "Main Interface", "Cleanup", "Danger Zone",
    ]


def test_runtime_limits_put_the_budget_pair_first() -> None:
    panel = _panel("advanced")
    limits = panel[panel.index("<h3>Runtime Limits</h3>"):panel.index("<h3>MCP Servers</h3>")]
    fields = re.findall(r'<label for="(s-[a-z0-9-]+)">', limits)
    assert fields == [
        "s-total-budget", "s-settings-per-task-cost",
        "s-workers", "s-max-rounds", "s-task-lifetime", "s-tool-timeout", "s-model-max-concurrency",
    ]
    # Concurrent conversations is a bound of consciousness in places: it lives in Behavior › Consciousness › More.
    assert "s-presence-max-active" not in panel


def test_the_concurrency_field_declares_no_minimum_so_a_saved_cap_of_zero_or_less_survives_any_save() -> None:
    """``model_concurrency._cap()`` has always read a cap of 0 or less as "no queue". A ``min`` on the field
    would make a hand-saved negative cap fail the Settings save gate for every unrelated save."""
    limits = _panel("advanced")
    limits = limits[limits.index("<h3>Runtime Limits</h3>"):limits.index("<h3>MCP Servers</h3>")]
    tag = re.search(r'<input id="s-model-max-concurrency"[^>]*>', limits).group(0)
    assert 'type="number"' in tag and 'step="1"' in tag and "min=" not in tag and "max=" not in tag, tag
    assert "<code>0</code> or less = no limit" in limits


def test_accounts_end_with_the_network_gate_then_the_legacy_escape_hatch() -> None:
    assert _titles("providers")[-2:] == ["Network Gate", "Legacy Compatibility"]


def test_dashboard_tabs_follow_expected_use_and_logs_stays_the_default() -> None:
    tabs = DASHBOARD[DASHBOARD.index("const DASHBOARD_TABS = ["):]
    tabs = tabs[:tabs.index("];")]
    assert re.findall(r"value: '([a-z]+)'", tabs) == [
        "logs", "activity", "consciousness", "costs", "updates", "storage", "evolution"]
    app = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    assert "dashboardActiveSubtab: 'logs'" in app
