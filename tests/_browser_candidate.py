"""A source-bound checkout for browser integration consumers, without downloading an engine."""
import os
from pathlib import Path

import pytest

from tests.candidate_checkout import candidate_checkout


@pytest.fixture
def wait_clone(tmp_path):
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import Error, sync_playwright
    from ouroboros.tools.browser import _set_playwright_browsers_path_if_bundled

    _set_playwright_browsers_path_if_bundled()
    try:
        # launch() never downloads an engine, and completing a real launch also
        # settles the Playwright driver before this preflight context closes.
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            browser.close()
    except Error as exc:
        if os.environ.get("OUROBOROS_EXPECT_BROWSER_ENGINES"):
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    source = Path(__file__).resolve().parents[1]
    with candidate_checkout(source, tmp_path / "clone", origin_proof=True) as candidate:
        yield candidate
