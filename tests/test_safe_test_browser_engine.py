"""A person's browser choice reaches the real consumer through the launcher, and nothing else does (#2413).

The real ``scripts/safe_test.py`` runs real Accounts consumers. What they ran is read from Playwright's
launched browser: the environment value, or a report named after it, only repeats what was asked for.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.ui_browser
SELECTOR = "OUROBOROS_UI_BROWSER_ENGINE"
CONSUMERS = ("tests/test_accounts_model_browser.py::test_onboarding_accounts_step_explains_a_pre_unified_engine",
             "tests/test_subscription_setup_browser.py::test_cursor_login_and_catalog_are_not_execution_readiness")
# Synthetic owner configuration and a credential, and the selector's nearest neighbour: only the selector crosses.
SCRUBBED = {"OPENAI_API_KEY": "synthetic-2413", "OUROBOROS_MODEL": "synthetic-2413",
            "OUROBOROS_UI_BROWSER_EXECUTABLE": "/synthetic-2413/browser"}
# Run by the launcher with this interpreter: it replaces the environment, so PYTHONPATH could not carry a plugin.
_RECORDING_PYTEST = """
import json, os, sys
import pytest


class Recorder:
    def __init__(self):
        self.report = {"launched": [], "environment": sorted(os.environ)}

    def pytest_configure(self, config):
        from playwright.sync_api import BrowserType

        launch = BrowserType.launch

        def recorded(browser_type, *args, **kwargs):
            browser = launch(browser_type, *args, **kwargs)
            self.report["launched"].append(browser.browser_type.name)
            return browser

        BrowserType.launch = recorded

    def pytest_unconfigure(self, config):
        with open(os.environ["ENGINE_REPORT"], "w", encoding="utf-8") as stream:
            json.dump(self.report, stream)


sys.exit(pytest.main(sys.argv[1:], plugins=[Recorder()]))
"""


@pytest.mark.parametrize("selected,launched", [("webkit", "webkit"), (None, "chromium")], ids=["selected", "default"])
def test_the_launcher_carries_only_the_selected_engine_to_real_accounts_consumers(tmp_path, selected, launched):
    driver, report = tmp_path / "recording_pytest.py", tmp_path / "engine.json"
    driver.write_text(_RECORDING_PYTEST, encoding="utf-8")
    env = {**os.environ, **SCRUBBED, "OUROBOROS_RUN_UI_SMOKE": "1", "ENGINE_REPORT": str(report)}
    env.pop(SELECTOR, None)
    if selected:
        env[SELECTOR] = selected
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(REPO / "scripts" / "safe_test.py"), "--evidence-dir", str(tmp_path / "evidence"),
         "--timeout", "400", "--", sys.executable, str(driver), *CONSUMERS, "-m", "ui_browser"],
        cwd=REPO, env=env, text=True, capture_output=True, timeout=420)
    assert result.returncode == 0 and "2 passed" in result.stdout, result.stdout + result.stderr
    facts = json.loads(report.read_text(encoding="utf-8"))
    assert facts["launched"] == [launched] * len(CONSUMERS), facts
    assert not SCRUBBED.keys() & set(facts["environment"]), facts
