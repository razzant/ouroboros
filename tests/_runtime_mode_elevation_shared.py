"""Fixtures and context helpers shared by the runtime-mode elevation suites.

Split out of ``tests/test_runtime_mode_elevation.py`` when that module was divided by
theme; the definitions are verbatim, so every sibling suite keeps the exact isolation
and seeding semantics it was written against.
"""

from __future__ import annotations

import json
import pathlib

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """Point ``SETTINGS_PATH`` and ``DATA_DIR`` at a fresh temp dir so each
    test starts with no on-disk settings.json. The fixture monkeypatches
    the module-level constants; downstream modules that import
    ``SETTINGS_PATH`` at module load (e.g., ``ouroboros.tools.core``) get
    the live patched value through ``ouroboros.config.SETTINGS_PATH``.

    Also clears ``_BOOT_RUNTIME_MODE`` between tests so each case starts
    with a fresh baseline. Tests that need a pinned boot baseline call
    ``initialize_runtime_mode_baseline`` explicitly.
    """
    from ouroboros import config as cfg

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    settings_path = data_dir / "settings.json"

    monkeypatch.setattr(cfg, "DATA_DIR", data_dir, raising=True)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path, raising=True)
    # The lock path derives from SETTINGS_PATH at call time.
    cfg.reset_runtime_mode_baseline_for_tests()
    yield settings_path
    cfg.reset_runtime_mode_baseline_for_tests()


def _seed_disk(settings_path: pathlib.Path, payload: dict) -> None:
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_drive_ctx(tmp_path):
    """Minimal ToolContext pointing drive_root at tmp_path/data."""
    from ouroboros.tools.registry import ToolContext

    drive_root = tmp_path / "data"
    drive_root.mkdir(exist_ok=True)
    return ToolContext(repo_dir=tmp_path / "repo", drive_root=drive_root)
