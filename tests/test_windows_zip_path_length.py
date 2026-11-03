from __future__ import annotations

import pathlib
import os

import pytest


pytestmark = pytest.mark.portable_detail


def _bundled_browsers_root() -> pathlib.Path:
    """Where the shipped artifact keeps its Playwright browsers.

    Packaged builds install into the Playwright package tree
    (``PLAYWRIGHT_BROWSERS_PATH=0``); the Docker image hands the runtime a
    shared directory through the same variable. Either way the test measures
    the artifact as shipped, never a download it made itself.
    """
    configured = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
    if configured and configured != "0":
        return pathlib.Path(configured)
    playwright = pytest.importorskip("playwright", reason="Playwright is not installed")
    return pathlib.Path(playwright.__file__).parent / "driver" / "package" / ".local-browsers"


def test_bundled_playwright_headless_shell_paths_stay_short():
    from ouroboros import chromium_runtime

    root = _bundled_browsers_root()
    if not root.is_dir():
        if os.environ.get("OUROBOROS_EXPECT_HEADLESS_SHELL") == "1":
            pytest.fail(f"Expected Playwright browser bundle at {root} in this CI lane")
        pytest.skip("Playwright local browser bundle not present")
    shells = [root / chromium_runtime.host_pin().directory]
    if chromium_runtime.installed_executable(root, chromium_runtime.host_pin()) is None:
        if os.environ.get("OUROBOROS_EXPECT_HEADLESS_SHELL") == "1":
            pytest.fail(f"Expected the pinned Chromium headless shell under {root} in this CI lane")
        pytest.skip("Pinned Chromium headless shell not present")
    too_long = []
    for shell in shells:
        for path in shell.rglob("*"):
            if not path.is_file():
                continue
            rel_len = len(str(path.relative_to(root)))
            if rel_len > 200:
                too_long.append((rel_len, path.relative_to(root).as_posix()))
    assert not too_long, f"Headless-shell paths exceed 200 chars: {too_long[:10]}"
