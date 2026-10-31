"""Current review-pool states and compatibility with retired comma-list settings."""

from __future__ import annotations

import json
import pathlib

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


def test_review_pool_state_vocabulary_is_structured_empty_error():
    """The pool's state vocabulary is structured|empty|error — never ``legacy``."""
    from ouroboros.reviewer_slot_config import review_pool_state
    from tests.review_pool_rosters import pool_roster, pool_seat

    assert review_pool_state(None)["state"] == "empty"
    assert review_pool_state(pool_roster(pool_seat("a", "m/one")))["state"] == "structured"
    assert review_pool_state("{not json")["state"] == "error"


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
