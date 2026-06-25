from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_legacy_skill_migrations_module_is_retired():
    """The v5.8 native-to-external topology repair window is closed."""
    assert importlib.util.find_spec("ouroboros.skill_migrations") is None


def test_server_startup_no_longer_runs_native_topology_migration():
    source = (REPO / "server.py").read_text(encoding="utf-8")
    assert "skill_migrations" not in source
    assert "migrate_unseeded_native_skills_to_external" not in source


def test_extensions_index_no_longer_mutates_skill_topology_on_read():
    source = (REPO / "ouroboros" / "gateway" / "extensions.py").read_text(encoding="utf-8")
    assert "skill_migrations" not in source
    assert "migrate_unseeded_native_skills_to_external" not in source





