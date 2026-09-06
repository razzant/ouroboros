"""ABI 7.0 (ABI-10, owner 5.4=A): the comma-list remnant sweep — the phase CI gate.

Grep-level checker pinning that the legacy reviewer comma-list migration read
stays gone: no migration-read branches, the settings vocabulary carries the
comma keys only as RETIRED, and no bench settings template configures
reviewers through them. The comma ENV spellings legitimately survive as the
derived runtime projection (``project_reviewer_slots_into_env`` + the
API-pinned getters) — the sweep therefore pins SETTINGS-plane and
migration-branch absence, not env-name absence.
"""

from __future__ import annotations

import json
import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parents[1]

_COMMA_SETTINGS_KEYS = (
    "OUROBOROS_REVIEW_MODELS",
    "OUROBOROS_SCOPE_REVIEW_MODELS",
    "OUROBOROS_SCOPE_REVIEW_MODEL",
)
_PHASE5_ROUTE_KEYS = (
    "OUROBOROS_REVIEW_ROUTES",
    "OUROBOROS_SCOPE_REVIEW_ROUTES",
    "OUROBOROS_ADVISORY_REVIEW_ROUTE",
)


def _python_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for base in ("ouroboros", "supervisor"):
        files.extend((_ROOT / base).rglob("*.py"))
    return files


def test_no_migration_read_branches_remain():
    """The reviewer_slot_config legacy block and its helpers stay deleted."""
    retired_symbols = ("_legacy_rows", "_legacy_config", "_shared_session_route_spec")
    hits: list[str] = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for symbol in retired_symbols:
            if re.search(rf"\b{symbol}\b", text):
                hits.append(f"{path.relative_to(_ROOT)}: {symbol}")
    assert not hits, f"migration-read remnants: {hits}"


def test_reviewer_config_source_is_never_legacy():
    """`ReviewerSlotConfig.source` vocabulary is structured|default now."""
    text = (_ROOT / "ouroboros" / "reviewer_slot_config.py").read_text(encoding="utf-8")
    assert 'source="legacy"' not in text
    assert 'source="default"' in text and 'source="structured"' in text


def test_settings_vocabulary_retired_the_comma_keys():
    from ouroboros.settings_defaults import RETIRED_SETTING_KEYS, SETTINGS_DEFAULTS

    for key in (*_COMMA_SETTINGS_KEYS, *_PHASE5_ROUTE_KEYS):
        assert key not in SETTINGS_DEFAULTS, f"{key} must not be a settings key"
        assert key in RETIRED_SETTING_KEYS, f"{key} must be RETIRED (ghost purge)"


def test_load_settings_purges_a_comma_only_install(tmp_path, monkeypatch):
    """An install that configured reviewers only through comma keys gets them
    stripped on load (5.4=A: it runs the shipped default panel; the RC auditor
    names this migration)."""
    import ouroboros.config as config

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({
        "OUROBOROS_REVIEW_MODELS": "x/custom,y/custom",
        "OUROBOROS_SCOPE_REVIEW_MODEL": "z/custom",
    }), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", settings_path)
    for key in (*_COMMA_SETTINGS_KEYS, *_PHASE5_ROUTE_KEYS):
        monkeypatch.delenv(key, raising=False)
    loaded = config.load_settings()
    for key in _COMMA_SETTINGS_KEYS:
        assert key not in loaded, f"retired {key} leaked through load_settings"


def test_no_bench_settings_template_configures_reviewers_via_comma_keys():
    """Bench templates migrated to the structured key with the SAME models."""
    hits: list[str] = []
    for path in (_ROOT / "devtools" / "benchmarks").rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        for key in _COMMA_SETTINGS_KEYS:
            if key in data:
                hits.append(f"{path.relative_to(_ROOT)}: {key}")
    assert not hits, f"bench templates still carry retired comma keys: {hits}"


def test_prose_no_longer_promises_a_comma_list_migration():
    """The rewritten prose markers stay rewritten (review.py:965 class)."""
    for rel in ("ouroboros/tools/review.py", "ouroboros/reviewer_slot_config.py",
                "ouroboros/review_model_routes.py"):
        text = (_ROOT / rel).read_text(encoding="utf-8")
        assert "migrated comma-lists" not in text, rel
        assert "старый читается" not in text, rel


def test_derived_projection_survives():
    """5.4=A removes the migration READ; the derived projection STAYS."""
    from ouroboros.reviewer_slot_config import project_reviewer_slots_into_env

    assert callable(project_reviewer_slots_into_env)
