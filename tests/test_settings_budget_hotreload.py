"""Settings save budget hot-reload regression tests."""

from __future__ import annotations

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient


def _settings_client(monkeypatch, tmp_path, current: dict):
    import server as srv
    import ouroboros.gateway.settings as gateway_settings

    monkeypatch.setattr(srv, "load_settings", lambda: dict(current))

    def fake_save_settings(settings, *args, **kwargs):
        current.clear()
        current.update(settings)

    monkeypatch.setattr(srv, "save_settings", fake_save_settings)
    monkeypatch.setattr(gateway_settings, "_owner_write_settings", fake_save_settings)
    monkeypatch.setattr(srv, "_apply_settings_to_env", lambda *_a, **_k: None)
    monkeypatch.setattr(srv, "_start_supervisor_if_needed", lambda *_a, **_k: False)
    monkeypatch.setattr(srv, "apply_runtime_provider_defaults", lambda s: (dict(s), False, []))
    monkeypatch.setattr(srv, "_mcp_reconfigure_startup", lambda *_a, **_k: None, raising=False)

    app = Starlette(routes=[Route("/api/settings", endpoint=srv.api_settings_post, methods=["POST"])])
    app.state.drive_root = tmp_path / "drive"
    app.state.repo_dir = tmp_path / "repo"
    return TestClient(app)


def test_settings_post_updates_budget_limits_and_per_task_threshold(monkeypatch, tmp_path):
    import supervisor.message_bus as bus_mod
    import supervisor.state as state_mod

    from ouroboros.config import SETTINGS_DEFAULTS as _defaults
    current = dict(_defaults)
    current["TOTAL_BUDGET"] = 10.0
    monkeypatch.setattr(state_mod, "TOTAL_BUDGET_LIMIT", 10.0)
    monkeypatch.setattr(bus_mod, "TOTAL_BUDGET_LIMIT", 10.0)

    client = _settings_client(monkeypatch, tmp_path, current)

    resp = client.post("/api/settings", json={"TOTAL_BUDGET": 25.0})

    assert resp.status_code == 200, resp.text
    assert resp.json().get("immediate_changed") is True
    assert state_mod.TOTAL_BUDGET_LIMIT == 25.0
    assert bus_mod.TOTAL_BUDGET_LIMIT == 25.0

    resp = client.post("/api/settings", json={"OUROBOROS_PER_TASK_COST_USD": "7.5"})

    assert resp.status_code == 200, resp.text
    assert resp.json().get("immediate_changed") is not True
    assert resp.json().get("next_task_changed") is True
    assert current["OUROBOROS_PER_TASK_COST_USD"] == 7.5

    invalid_cases = [
        ({"TOTAL_BUDGET": 0}, "greater than zero"),
        ({"TOTAL_BUDGET": 0.005}, "at least 0.01"),
        (["TOTAL_BUDGET", 25], "JSON body must be an object."),
        ({"OUROBOROS_PER_TASK_COST_USD": "nan"}, "must be a number"),
        ({"OUROBOROS_PER_TASK_COST_USD": "0.005"}, "at least 0.01"),
        ({"TOTAL_BUDGET": True}, "must be a number"),
    ]
    clean_budget_state = dict(current)
    clean_budget_state["TOTAL_BUDGET"] = 10.0
    clean_budget_state["OUROBOROS_PER_TASK_COST_USD"] = 20.0
    for payload, error in invalid_cases:
        current.clear()
        current.update(clean_budget_state)
        resp = client.post("/api/settings", json=payload)

        assert resp.status_code == 400
        assert error in resp.json()["error"]
        assert current["TOTAL_BUDGET"] == 10.0
        assert current["OUROBOROS_PER_TASK_COST_USD"] == 20.0


def test_settings_post_rejects_malformed_evolution_cadence(monkeypatch, tmp_path):
    """A direct API client must not be able to persist a malformed post-task evolution
    cadence (e.g. every_n:0) — backend half of the strict every_n validation contract."""

    key = "OUROBOROS_POST_TASK_EVOLUTION_CADENCE"
    from ouroboros.config import SETTINGS_DEFAULTS as _defaults
    current = dict(_defaults)
    current[key] = "llm"
    client = _settings_client(monkeypatch, tmp_path, current)

    for good in ("off", "llm", "every_n:1", "every_n:25"):
        resp = client.post("/api/settings", json={key: good})
        assert resp.status_code == 200, (good, resp.text)
        assert current[key] == good

    current[key] = "llm"
    for bad in ("every_n:0", "every_n:-1", "every_n:", "every_nonsense", "daily"):
        resp = client.post("/api/settings", json={key: bad})
        assert resp.status_code == 400, (bad, resp.text)
        assert "every_n:<positive int>" in resp.json()["error"]
        assert current[key] == "llm", bad  # not persisted


def test_settings_post_validates_and_applies_update_channel(monkeypatch, tmp_path):
    from ouroboros.config import SETTINGS_DEFAULTS as _defaults

    key = "OUROBOROS_UPDATE_CHANNEL"
    current = dict(_defaults)
    client = _settings_client(monkeypatch, tmp_path, current)

    for value in ("stable", "qa", "development"):
        resp = client.post("/api/settings", json={key: value.upper()})
        assert resp.status_code == 200, (value, resp.text)
        assert bool(resp.json().get("immediate_changed")) is (value != "stable")
        assert current[key] == value

    previous = current[key]
    for value in ("", "nightly", None):
        resp = client.post("/api/settings", json={key: value})
        assert resp.status_code == 400, (value, resp.text)
        assert "stable, qa, development" in resp.json()["error"]
        assert current[key] == previous


def test_settings_post_model_change_preserves_max_without_capability_probe(monkeypatch, tmp_path):
    """A model save changes only the model; persistent context intent is untouched."""
    import ouroboros.capability_evidence as ce
    from ouroboros.config import SETTINGS_DEFAULTS as _defaults

    current = dict(_defaults)
    current.update({
        "OUROBOROS_CONTEXT_MODE": "max",
        "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "false",
        "OUROBOROS_MODEL": "openai/gpt-5.5",
    })
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE_AUTO_LOW", "false")

    def unexpected_probe(*_args, **_kwargs):
        raise AssertionError("main-route capability must not gate a settings save")

    monkeypatch.setattr(ce, "probe", unexpected_probe)
    client = _settings_client(monkeypatch, tmp_path, current)

    resp = client.post("/api/settings", json={"OUROBOROS_MODEL": "openai/gpt-4o-mini"})

    assert resp.status_code == 200, resp.text
    assert current["OUROBOROS_MODEL"] == "openai/gpt-4o-mini"
    assert current["OUROBOROS_CONTEXT_MODE"] == "max"
    assert current["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"
    assert "context_mode_downgraded" not in resp.json()
    assert "notice" not in resp.json()


def test_context_mode_compat_migration_truth_table_and_second_start(
    monkeypatch, tmp_path, caplog,
):
    """First load persists only the raw migration; a fresh second start is silent."""
    import json
    import logging
    import os

    import ouroboros.config as cfg
    import ouroboros.context_mode_compat as context_mode_compat

    cases = [
        ({"OUROBOROS_CONTEXT_MODE": "max"}, "max", False),
        ({"OUROBOROS_CONTEXT_MODE": "max", "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "true"}, "max", False),
        ({"OUROBOROS_CONTEXT_MODE": "max", "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "off"}, "max", False),
        ({"OUROBOROS_CONTEXT_MODE": "low", "OUROBOROS_CONTEXT_MODE_AUTO_LOW": False}, "low", False),
        ({"OUROBOROS_CONTEXT_MODE": "low", "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "false"}, "low", False),
        ({"OUROBOROS_CONTEXT_MODE": "low", "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "off"}, "low", False),
        ({"OUROBOROS_CONTEXT_MODE": "low", "OUROBOROS_CONTEXT_MODE_AUTO_LOW": 0}, "low", False),
        ({"OUROBOROS_CONTEXT_MODE": "low"}, "max", True),
        ({"OUROBOROS_CONTEXT_MODE": "low", "OUROBOROS_CONTEXT_MODE_AUTO_LOW": True}, "max", True),
        ({"OUROBOROS_CONTEXT_MODE": "low", "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "true"}, "max", True),
        ({"OUROBOROS_CONTEXT_MODE": "low", "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "no"}, "max", True),
        ({"OUROBOROS_CONTEXT_MODE": "low", "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "unknown"}, "max", True),
    ]

    monkeypatch.setattr(os, "environ", dict(os.environ))
    for index, (raw, expected_mode, warning_expected) in enumerate(cases):
        settings_path = tmp_path / f"settings-{index}.json"
        raw_document = {
            **raw,
            "OPENAI_API_KEY": "test-secret-preserved",
            "UNKNOWN_MIGRATION_KEY": {"nested": [index, False]},
        }
        settings_path.write_text(json.dumps(raw_document), encoding="utf-8")
        monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path)
        monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
        monkeypatch.setattr(cfg, "_BOOT_RUNTIME_MODE", None)
        for key in (
            "OUROBOROS_CONTEXT_MODE",
            "OUROBOROS_CONTEXT_MODE_AUTO_LOW",
            "OUROBOROS_RUNTIME_MODE",
            cfg.BOOT_RUNTIME_MODE_ENV_KEY,
        ):
            os.environ.pop(key, None)

        caplog.clear()
        caplog.set_level(logging.WARNING, logger=context_mode_compat.__name__)
        context_mode_compat._MIGRATION_WARNED_PATHS.clear()
        loaded = cfg.load_settings()
        assert loaded["OUROBOROS_CONTEXT_MODE"] == expected_mode
        assert loaded["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"
        warnings = [
            record for record in caplog.records
            if "normalized Low to Max" in record.getMessage()
        ]
        assert len(warnings) == int(warning_expected)
        if warning_expected:
            assert "Re-select Low in Settings" in warnings[0].getMessage()

        canonical = json.loads(settings_path.read_text(encoding="utf-8"))
        assert canonical["OUROBOROS_CONTEXT_MODE"] == expected_mode
        assert canonical["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"
        assert canonical["OPENAI_API_KEY"] == raw_document["OPENAI_API_KEY"]
        assert canonical["UNKNOWN_MIGRATION_KEY"] == raw_document["UNKNOWN_MIGRATION_KEY"]
        assert set(canonical) == set(raw_document) | {
            "OUROBOROS_CONTEXT_MODE",
            "OUROBOROS_CONTEXT_MODE_AUTO_LOW",
        }, "the migration must not persist defaults or remove unknown raw keys"

        # A distinct process would start with empty module-local warning state. Clearing it
        # proves silence comes from the first load's disk canonicalization, not memoization.
        context_mode_compat._MIGRATION_WARNED_PATHS.clear()
        caplog.clear()
        assert cfg.load_settings()["OUROBOROS_CONTEXT_MODE"] == expected_mode
        assert not [
            record for record in caplog.records
            if "normalized Low to Max" in record.getMessage()
        ], "the canonical second start must be silent"


def test_context_mode_compat_migration_write_failure_is_honest_and_nonfatal(
    monkeypatch, tmp_path, caplog,
):
    """An atomic-write failure leaves disk intact and the current load normalized."""
    import json
    import logging

    import ouroboros.config as cfg
    import ouroboros.context_mode_compat as context_mode_compat

    settings_path = tmp_path / "settings.json"
    original = {
        "OUROBOROS_CONTEXT_MODE": "low",
        "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "true",
        "UNKNOWN_MIGRATION_KEY": {"preserved": True},
    }
    settings_path.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    context_mode_compat._MIGRATION_WARNED_PATHS.clear()

    def fail_atomic_write(*_args, **_kwargs):
        raise OSError("simulated migration write failure")

    monkeypatch.setattr(context_mode_compat, "atomic_write_json", fail_atomic_write)
    caplog.set_level(logging.WARNING, logger=context_mode_compat.__name__)

    loaded = cfg.load_settings()

    assert loaded["OUROBOROS_CONTEXT_MODE"] == "max"
    assert loaded["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"
    assert json.loads(settings_path.read_text(encoding="utf-8")) == original
    assert any(
        "could not be persisted; migration will retry" in record.getMessage()
        for record in caplog.records
    )


def test_context_mode_compat_migration_does_not_rewrite_unchanged_raw_pair(
    monkeypatch, tmp_path,
):
    """No pair and an already-canonical pair remain non-authoring/non-writing."""
    import json

    import ouroboros.config as cfg
    import ouroboros.context_mode_compat as context_mode_compat

    writes = []
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        context_mode_compat,
        "atomic_write_json",
        lambda *_args, **_kwargs: writes.append((_args, _kwargs)),
    )

    documents = [
        {"UNKNOWN_MIGRATION_KEY": {"untouched": True}},
        {
            "OUROBOROS_CONTEXT_MODE": "max",
            "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "false",
            "UNKNOWN_MIGRATION_KEY": {"untouched": True},
        },
    ]
    for index, document in enumerate(documents):
        settings_path = tmp_path / f"unchanged-{index}.json"
        settings_path.write_text(json.dumps(document), encoding="utf-8")
        monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path)

        cfg.load_settings()

        assert json.loads(settings_path.read_text(encoding="utf-8")) == document

    assert writes == []


def test_owner_raw_reader_uses_the_same_pre_default_migration(monkeypatch, tmp_path):
    import json

    import ouroboros.config as cfg
    from ouroboros.gateway.owner_settings import _owner_read_settings_raw

    path = tmp_path / "settings.json"
    path.write_text(json.dumps({
        "OUROBOROS_CONTEXT_MODE": "low",
        "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "true",
    }), encoding="utf-8")
    monkeypatch.setattr(cfg, "SETTINGS_PATH", path)

    loaded = _owner_read_settings_raw()

    assert loaded["OUROBOROS_CONTEXT_MODE"] == "max"
    assert loaded["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"


def test_context_mode_env_and_disk_owner_semantics(monkeypatch, tmp_path):
    """Bare env Low stays effective Low/owner Max; explicit false proves owner Low."""
    import json
    import os

    import ouroboros.config as cfg
    from ouroboros.tools import scope_review as sr

    monkeypatch.setattr(os, "environ", dict(os.environ))
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

    def clear_pair():
        os.environ.pop("OUROBOROS_CONTEXT_MODE", None)
        os.environ.pop("OUROBOROS_CONTEXT_MODE_AUTO_LOW", None)

    # Bare env Low: sizing Low, owner/P3 Max.
    clear_pair()
    os.environ["OUROBOROS_CONTEXT_MODE"] = "low"
    cfg.apply_settings_to_env(cfg.load_settings())
    assert cfg.get_context_mode() == "low"
    assert cfg.get_owner_context_mode() == "max"
    assert sr._scope_review_skipped_in_low_context() is False

    # Explicit forwarded false: benchmark/operator owner Low.
    clear_pair()
    os.environ["OUROBOROS_CONTEXT_MODE"] = "low"
    os.environ["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] = "false"
    cfg.apply_settings_to_env(cfg.load_settings())
    assert cfg.get_context_mode() == "low"
    assert cfg.get_owner_context_mode() == "low"
    assert sr._scope_review_skipped_in_low_context() is True

    # Explicit persisted Low + false remains owner Low after projection.
    settings_path.write_text(json.dumps({
        "OUROBOROS_CONTEXT_MODE": "low",
        "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "false",
    }), encoding="utf-8")
    clear_pair()
    cfg.apply_settings_to_env(cfg.load_settings())
    assert cfg.get_context_mode() == "low"
    assert cfg.get_owner_context_mode() == "low"

    # Ambiguous legacy persisted Low is restored to canonical owner Max.
    settings_path.write_text(json.dumps({
        "OUROBOROS_CONTEXT_MODE": "low",
        "OUROBOROS_CONTEXT_MODE_AUTO_LOW": "true",
    }), encoding="utf-8")
    clear_pair()
    migrated = cfg.load_settings()
    cfg.apply_settings_to_env(migrated)
    assert migrated["OUROBOROS_CONTEXT_MODE"] == "max"
    assert cfg.get_context_mode() == "max"
    assert cfg.get_owner_context_mode() == "max"


def test_auto_low_source_inventory_has_no_true_writer_or_ghost_reader():
    """The tombstone has a closed live-source inventory for its one window."""
    import pathlib
    import re

    repo = pathlib.Path(__file__).resolve().parent.parent
    roots = [
        repo / "ouroboros",
        repo / "web" / "modules",
        repo / "devtools" / "benchmarks" / "common",
        repo / "devtools" / "benchmarks" / "terminal_bench",
        repo / "devtools" / "benchmarks" / "continual_learning",
    ]
    sources = {}
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".js"} or "operator_patches" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            sources[path.relative_to(repo).as_posix()] = text

    key = "OUROBOROS_CONTEXT_MODE_AUTO_LOW"
    key_paths = {path for path, text in sources.items() if key in text}
    assert key_paths == {
        "ouroboros/config.py",
        "ouroboros/settings_defaults.py",
        "ouroboros/context_mode_compat.py",
        "ouroboros/gateway/owner_settings.py",
        "ouroboros/gateway/settings.py",
        "devtools/benchmarks/common/server_runner.py",
        "devtools/benchmarks/terminal_bench/harbor_installed_agent.py",
        "devtools/benchmarks/continual_learning/run_clb.py",
    }
    response_paths = {
        path for path, text in sources.items()
        if "context_mode_auto_low" in text
    }
    assert response_paths == {
        "ouroboros/gateway/contracts.py",
        "ouroboros/gateway/state.py",
        "web/modules/api_types.js",
    }, "no UI, CLI, or health reader may infer a derived mode"

    writer_patterns = (
        rf'["\']{key}["\']\s*\]\s*=\s*["\']true["\']',
        rf'["\']{key}["\']\s*:\s*["\']true["\']',
        rf'{key}\s*=\s*true\b',
    )
    for path, text in sources.items():
        for line in text.splitlines():
            if key in line:
                assert re.search(r"\btrue\b", line, re.IGNORECASE) is None, (
                    f"{path} still couples the tombstone to true"
                )
        for pattern in writer_patterns:
            assert re.search(pattern, text, re.IGNORECASE) is None, (
                f"{path} still writes the retired true marker"
            )

    settings_source = sources["ouroboros/gateway/settings.py"]
    control_source = sources["ouroboros/tools/control.py"]
    assert "_max_context_block" not in settings_source
    assert "_apply_max_context_auto_downgrade" not in settings_source
    assert "context_mode_downgraded" not in settings_source
    assert "_active_route_confirms_max" not in settings_source
    assert "SWITCH_BLOCKED" not in control_source
