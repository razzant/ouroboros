"""#285: the save response tells the truth about WHEN a change takes effect.

Three honesty surfaces:
- the classification tables (immediate / next task / restart / retired) match
  what the code actually does with each key;
- a retired key is reported as retired instead of pretending to apply;
- a failed task-start settings reload is disclosed loudly instead of leaving
  the task on the previous configuration in silence.
"""

from __future__ import annotations

import json


import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    from ouroboros import config as cfg

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    settings_path = data_dir / "settings.json"
    monkeypatch.setattr(cfg, "DATA_DIR", data_dir, raising=True)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path, raising=True)
    cfg.reset_runtime_mode_baseline_for_tests()
    from ouroboros import server_process
    monkeypatch.setattr(server_process, "_applied_restart_settings", {
        key: cfg.SETTINGS_DEFAULTS[key] for key in (
            "OUROBOROS_SERVER_HOST", "OUROBOROS_HOST_SERVICE_PORT", "OUROBOROS_UI_DIR",
            "OUROBOROS_SKILLS_REPO_PATH")})
    yield settings_path
    cfg.reset_runtime_mode_baseline_for_tests()


def _settings_app(monkeypatch, settings_path):
    from ouroboros.gateway import settings as settings_mod

    monkeypatch.setattr(settings_mod, "apply_runtime_provider_defaults", lambda s: (s, False, []))
    monkeypatch.setattr(settings_mod, "_start_supervisor_if_needed_for_request",
                        lambda *_a, **_k: False)
    monkeypatch.setattr(settings_mod, "_apply_settings_to_env", lambda *_a, **_k: None)
    monkeypatch.setattr(settings_mod, "_apply_settings_save_side_effects", lambda *_a, **_k: None)
    app = Starlette(routes=[
        Route("/api/settings", endpoint=settings_mod.api_settings_post, methods=["POST"])])
    app.state.drive_root = settings_path.parent
    app.state.repo_dir = settings_path.parent
    return app


def _save(monkeypatch, isolated_settings, payload):
    app = _settings_app(monkeypatch, isolated_settings)
    resp = TestClient(app).post("/api/settings", json=payload)
    assert resp.status_code == 200, resp.text
    return json.loads(resp.text)


def test_a_retired_key_is_absorbed_silently_and_claimed_by_no_bucket(monkeypatch, isolated_settings):
    """D04 (owner 1B) finished what #285 started: the flat timeout pair is
    RETIRED, not a typed no-op the save has to apologise for. The merge only
    walks SETTINGS_DEFAULTS, so a stored value cannot reach an effect bucket
    at all — and the RC auditor, not the save response, is where an upgrading
    install learns its key is gone."""
    data = _save(monkeypatch, isolated_settings, {"OUROBOROS_SOFT_TIMEOUT_SEC": "1234"})
    assert not data.get("immediate_changed")
    assert not data.get("next_task_changed")
    assert not data.get("restart_required")
    assert "OUROBOROS_SOFT_TIMEOUT_SEC" not in json.dumps(data)


def test_hot_reconfigured_mcp_keys_are_classified_immediate(monkeypatch, isolated_settings):
    """MCP keys are hot-reconfigured by the save handler in the server process,
    and worker processes re-check the settings mtime on their next tool-schema
    read (mcp_client.ensure_configured_from_settings). This test pins the
    CLASSIFICATION; the hot apply itself and its failure disclosure are pinned
    by test_failed_mcp_reconfigure_becomes_a_save_warning below."""
    from ouroboros.settings_scales import IMMEDIATE_SETTINGS

    for key in ("MCP_ENABLED", "MCP_SERVERS", "MCP_TOOL_TIMEOUT_SEC"):
        assert key in IMMEDIATE_SETTINGS
    data = _save(monkeypatch, isolated_settings, {"MCP_ENABLED": "true"})
    assert data.get("immediate_changed") is True
    assert not data.get("next_task_changed")


def test_tool_timeout_is_immediate_the_outer_cap_reads_settings_live(monkeypatch, isolated_settings):
    """loop_tool_execution reads OUROBOROS_TOOL_TIMEOUT_SEC from settings.json
    BEFORE env on every tool call in every process — a saved change bites the
    running task's next tool call, so "next task" would be a lie."""
    data = _save(monkeypatch, isolated_settings, {"OUROBOROS_TOOL_TIMEOUT_SEC": "120"})
    assert data.get("immediate_changed") is True
    assert not data.get("next_task_changed")


def test_a_float_setting_resent_as_text_is_not_a_change(monkeypatch, isolated_settings):
    """The Settings page sends every float field as its input text on each save;
    "20" against the loaded 20.0 must not report a next-task change, or every
    immediate-only save would also promise the next task (and its skill-key rule)."""
    isolated_settings.write_text(json.dumps({"OUROBOROS_CONSCIOUSNESS_DAILY_USD": "20",
                                             "OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD": "0"}), encoding="utf-8")
    resent = {"OUROBOROS_CONSCIOUSNESS_DAILY_USD": "20", "OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD": "0"}
    data = _save(monkeypatch, isolated_settings, {**resent, "OUROBOROS_TOOL_TIMEOUT_SEC": "120"})
    assert data.get("immediate_changed") is True and not data.get("next_task_changed")
    assert _save(monkeypatch, isolated_settings, resent).get("no_changes") is True
    data = _save(monkeypatch, isolated_settings, {**resent, "OUROBOROS_CONSCIOUSNESS_DAILY_USD": "20.5"})
    assert data.get("next_task_changed") is True  # a real value change still counts


def test_write_preconditions_are_not_custom_settings(monkeypatch, isolated_settings):
    first = _save(monkeypatch, isolated_settings, {"TOTAL_BUDGET": 201, "SERVICE_API_KEY": "synthetic-key"})
    token = first["changes"]["TOTAL_BUDGET"]["token"]
    before = isolated_settings.read_text(encoding="utf-8")
    same = _save(monkeypatch, isolated_settings, {
        "TOTAL_BUDGET": 201, "if_unchanged": {"TOTAL_BUDGET": token}, "unset": []})
    assert same.get("no_changes") is True
    assert isolated_settings.read_text(encoding="utf-8") == before
    stored = json.loads(before)
    assert stored["SERVICE_API_KEY"] == "synthetic-key"
    assert not {"IF_UNCHANGED", "UNSET"}.intersection(stored)
    _save(monkeypatch, isolated_settings, {"TOTAL_BUDGET": 202})
    client = TestClient(_settings_app(monkeypatch, isolated_settings))
    refused = client.post("/api/settings", json={"TOTAL_BUDGET": 203, "if_unchanged": {"TOTAL_BUDGET": token}})
    assert refused.status_code == 409, refused.text
    assert json.loads(isolated_settings.read_text(encoding="utf-8"))["TOTAL_BUDGET"] == 202


def test_skills_repo_path_requires_a_restart_for_pooled_workers(monkeypatch, isolated_settings):
    """Pooled workers load the extension registry once at spawn and never
    reload it per task; the server-side hot reload alone cannot make the key
    honest as "immediate"."""
    data = _save(monkeypatch, isolated_settings,
                 {"OUROBOROS_SKILLS_REPO_PATH": str(isolated_settings.parent)})
    assert data.get("restart_required") is True
    assert "OUROBOROS_SKILLS_REPO_PATH" in data.get("restart_keys", [])


def test_failed_mcp_reconfigure_becomes_a_save_warning(monkeypatch, tmp_path):
    """An immediate-classed key whose hot apply BROKE must not let the save
    report "took effect immediately" without saying so."""
    import types

    from ouroboros import mcp_client
    from ouroboros.gateway import settings as settings_mod

    monkeypatch.setattr(mcp_client, "configuration_changed", lambda *_a, **_k:
                        (_ for _ in ()).throw(RuntimeError("mcp exploded")))

    class _Req:
        app = types.SimpleNamespace(state=types.SimpleNamespace(drive_root=str(tmp_path)))

    warnings = settings_mod._apply_settings_save_side_effects(
        _Req(), {"MCP_ENABLED": "true"}, {}, ["MCP_ENABLED"])
    joined = " ".join(warnings)
    assert "MCP reconfigure failed" in joined
    assert "RuntimeError" in joined


def test_failed_skills_reload_becomes_a_save_warning(monkeypatch, tmp_path):
    import types

    from ouroboros import extension_loader
    from ouroboros.gateway import settings as settings_mod

    monkeypatch.setattr(extension_loader, "reload_all",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("reload exploded")))

    class _Req:
        app = types.SimpleNamespace(state=types.SimpleNamespace(drive_root=str(tmp_path)))

    warnings = settings_mod._apply_settings_save_side_effects(
        _Req(), {"OUROBOROS_SKILLS_REPO_PATH": str(tmp_path)}, {}, ["OUROBOROS_SKILLS_REPO_PATH"])
    joined = " ".join(warnings)
    assert "Skills repo reload failed" in joined
    assert "RuntimeError" in joined


def test_provider_base_url_keys_apply_at_once_without_a_restart(monkeypatch, isolated_settings):
    """The resolver reads base URLs per call from the live route view (``route_choice``),
    so they need neither a restart nor the next task."""
    from ouroboros.gateway.settings import _RESTART_REQUIRED_KEYS

    for key in ("OPENAI_BASE_URL", "OPENAI_COMPATIBLE_BASE_URL",
                "CLOUDRU_FOUNDATION_MODELS_BASE_URL", "MINIMAX_REGION",
                "GIGACHAT_SCOPE", "GIGACHAT_BASE_URL", "GIGACHAT_VERIFY_SSL_CERTS",
                "OUROBOROS_EXTRA_CA_BUNDLE"):
        assert key not in _RESTART_REQUIRED_KEYS
    data = _save(monkeypatch, isolated_settings, {"OPENAI_BASE_URL": "https://example.test/v1"})
    assert not data.get("restart_required")
    # A route key is read live by every process: immediate, never "the next task".
    assert data.get("immediate_changed") is True and not data.get("next_task_changed")
    assert _save(monkeypatch, isolated_settings, {"OUROBOROS_MODEL": "openai::next-main"}).get("immediate_changed") is True
    # A window is read when a task's context plan is built: the next task, or the running one's next route change.
    data = _save(monkeypatch, isolated_settings, {"OUROBOROS_MODEL_CONTEXT_WINDOWS": {"main": 200000}})
    assert data.get("next_task_changed") is True and not data.get("immediate_changed")


def test_consciousness_wake_bounds_apply_without_a_restart(monkeypatch, isolated_settings):
    """The alarm clock reads OUROBOROS_BG_WAKEUP_MIN/MAX through the config getters at each
    decision (consciousness.tick / set_next_wakeup), like the other consciousness keys, so
    a save never claims a restart for them (Background Consciousness redesign P2)."""
    from ouroboros.settings_scales import RESTART_REQUIRED_SETTINGS

    assert not {"OUROBOROS_BG_WAKEUP_MIN", "OUROBOROS_BG_WAKEUP_MAX"} & RESTART_REQUIRED_SETTINGS
    data = _save(monkeypatch, isolated_settings, {"OUROBOROS_BG_WAKEUP_MIN": "1200", "OUROBOROS_BG_WAKEUP_MAX": "7200"})
    assert not data.get("restart_required")
    assert not ({"OUROBOROS_BG_WAKEUP_MIN", "OUROBOROS_BG_WAKEUP_MAX"} & set(data.get("restart_keys") or []))


def test_unrelated_save_migrates_a_legacy_wakeup_min_to_the_effective_floor(
    monkeypatch, isolated_settings,
):
    """A legacy 30-second raw value must not make an otherwise unrelated
    Settings save fail HTML min=60 validation; the write carries the same
    normalized value the runtime already uses, without changing it to 900."""
    isolated_settings.write_text(json.dumps({
        "OUROBOROS_BG_WAKEUP_MIN": 30,
        "OUROBOROS_BG_WAKEUP_MAX": 7200,
    }), encoding="utf-8")
    _save(monkeypatch, isolated_settings, {"TOTAL_BUDGET": 123.0})
    from ouroboros import config as cfg

    persisted = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert persisted["OUROBOROS_BG_WAKEUP_MIN"] == 60
    assert persisted["OUROBOROS_BG_WAKEUP_MAX"] == 7200
    assert cfg.load_settings()["OUROBOROS_BG_WAKEUP_MIN"] == 60


def test_host_service_port_requires_a_restart(monkeypatch, isolated_settings):
    """The host-service port is bound once at server startup."""
    data = _save(monkeypatch, isolated_settings, {"OUROBOROS_HOST_SERVICE_PORT": "18999"})
    assert data.get("restart_required") is True
    assert "OUROBOROS_HOST_SERVICE_PORT" in data.get("restart_keys", [])


def _live_log_collector():
    events = []

    def emit(event_type, **fields):
        events.append({"type": event_type, **fields})

    return events, emit


def test_failed_task_start_reload_is_disclosed_loudly(monkeypatch):
    """agent.py used to swallow apply_task_start_settings failures with a bare
    ``except: pass`` — the task then ran on the previous task's env while the
    save UI promised "applies from the next task"."""
    from ouroboros import subagent_runtime

    def _boom():
        raise RuntimeError("settings.json unreadable")

    monkeypatch.setattr(subagent_runtime, "apply_task_start_settings", _boom)
    events, emit = _live_log_collector()
    subagent_runtime.apply_task_start_settings_or_disclose("task-9", emit)

    assert len(events) == 1
    event = events[0]
    assert event["type"] == "task_start_settings_reload_failed"
    assert event["task_id"] == "task-9"
    assert "RuntimeError" in event["error"]
    assert "previously applied configuration" in event["message"]


def test_corrupt_settings_file_is_disclosed_not_silently_defaulted(monkeypatch, tmp_path):
    """load_settings falls back to defaults+env on a malformed settings.json
    instead of raising — exactly the silence the wrapper exists to break. The
    wrapper probes the file itself, so the common corruption case is loud."""
    from ouroboros import config as cfg
    from ouroboros import subagent_runtime

    bad = tmp_path / "settings.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(cfg, "SETTINGS_PATH", bad, raising=True)
    events, emit = _live_log_collector()
    subagent_runtime.apply_task_start_settings_or_disclose("task-9", emit)
    assert len(events) == 1
    assert events[0]["type"] == "task_start_settings_reload_failed"
    assert "JSONDecodeError" in events[0]["error"]


def test_missing_settings_file_is_a_legitimate_defaults_install(monkeypatch, tmp_path):
    from ouroboros import config as cfg
    from ouroboros import subagent_runtime

    monkeypatch.setattr(cfg, "SETTINGS_PATH", tmp_path / "absent.json", raising=True)
    applied = []
    monkeypatch.setattr(subagent_runtime, "apply_task_start_settings",
                        lambda: applied.append(True))
    events, emit = _live_log_collector()
    subagent_runtime.apply_task_start_settings_or_disclose("task-9", emit)
    assert applied == [True]
    assert events == []


def test_reload_failure_event_is_persisted_durably(tmp_path):
    """The supervisor persists the disclosure to events.jsonl — without that
    the fact evaporates on the next page load (it must outlive the live feed)."""
    import types

    from supervisor import events as events_mod

    written = []

    class _Bridge:
        def push_log(self, payload):
            pass

    ctx = types.SimpleNamespace(
        bridge=_Bridge(),
        DRIVE_ROOT=tmp_path,
        append_jsonl=lambda path, payload: written.append((path, payload)),
    )
    events_mod._handle_log_event(
        {"data": {"type": "task_start_settings_reload_failed", "task_id": "t9",
                  "error": "RuntimeError: boom"}},
        ctx,
    )
    assert len(written) == 1
    assert written[0][1]["type"] == "task_start_settings_reload_failed"


def test_successful_task_start_reload_stays_silent(monkeypatch):
    from ouroboros import subagent_runtime

    calls = []
    monkeypatch.setattr(subagent_runtime, "apply_task_start_settings",
                        lambda: calls.append(True))
    events, emit = _live_log_collector()
    subagent_runtime.apply_task_start_settings_or_disclose("task-9", emit)
    assert calls == [True]
    assert events == []


def test_handle_task_wires_the_disclosing_reload():
    """The task-start seam must go through the disclosing wrapper — a revert
    to a bare apply_task_start_settings() call would resurrect the silence.
    The stale chat binding of a reused worker agent must be cleared BEFORE the
    wrapper emits, or the disclosure lands in the previous task's thread."""
    import inspect

    from ouroboros.agent import OuroborosAgent

    source = inspect.getsource(OuroborosAgent.handle_task)
    assert "apply_task_start_settings_or_disclose" in source
    reset_at = source.index("self._current_chat_id = None")
    disclose_at = source.index("apply_task_start_settings_or_disclose")
    assert reset_at < disclose_at


from tests.test_consciousness import clock as clock_fixture

clock = clock_fixture


def test_every_default_has_one_effect_without_changing_actor_snapshot_policy():
    from ouroboros.config import SETTINGS_DEFAULTS
    from ouroboros.gateway.settings import _effect_buckets
    from ouroboros.settings_integrity import _next_task_setting
    from ouroboros.settings_scales import setting_effect

    # Target snapshot exclusions, independent of the presentation function (including the live worker cap).
    original_live = {"TOTAL_BUDGET", "OUROBOROS_SUBAGENTS_MODE", "OUROBOROS_UI_LANGUAGE",
                     "OUROBOROS_TOOL_TIMEOUT_SEC", "OUROBOROS_MODEL_MAX_CONCURRENCY", "OUROBOROS_MAX_WORKERS",
                     "GITHUB_TOKEN", "GITHUB_REPO",
                     "OUROBOROS_UPDATE_CHANNEL", "MCP_ENABLED", "MCP_SERVERS", "MCP_TOOL_TIMEOUT_SEC"}
    original_restart = {"OUROBOROS_SERVER_HOST", "OUROBOROS_HOST_SERVICE_PORT",
                        "OUROBOROS_UI_DIR", "OUROBOROS_SKILLS_REPO_PATH", "LOCAL_MODEL_SOURCE", "LOCAL_MODEL_FILENAME",
                        "LOCAL_MODEL_PORT", "LOCAL_MODEL_N_GPU_LAYERS", "LOCAL_MODEL_CONTEXT_LENGTH", "LOCAL_MODEL_CHAT_FORMAT"}
    classes = {"now", "next_call", "next_wake", "next_task", "restart", "local_model"}
    for key in SETTINGS_DEFAULTS:
        effect = setting_effect(key)
        assert effect in classes, key
        live, later = _effect_buckets([key])
        assert bool(live) == (effect in {"now", "next_call"}), key
        assert bool(later) == (effect in {"next_wake", "next_task"}), key
        assert _next_task_setting(key) == (key not in original_live | original_restart | {"OUROBOROS_RUNTIME_MODE"}), key
    assert setting_effect("UNRECOGNIZED_KEY") == "next_task"
    assert setting_effect("OUROBOROS_SAFETY_OWNER_RULES") == "next_task"
    assert setting_effect("OUROBOROS_NETWORK_PASSWORD") == "now"
    assert setting_effect("OUROBOROS_FINALIZATION_GRACE_SEC") == "now"
    assert setting_effect("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS") == "now"


@pytest.mark.parametrize("case", ["scheduled_wake", "launched_wake", "following_model", "explicit_model", "restart", "local_model"])
def test_effects_at_the_reader_boundary_keep_running_actor_facts(clock, isolated_settings, monkeypatch, case):
    import os
    from types import SimpleNamespace
    from ouroboros import config, consciousness, route_choice, local_model
    from ouroboros.gateway import settings
    from ouroboros.settings_integrity import task_settings_scope, task_settings_snapshot, runtime_setting
    from ouroboros.settings_scales import setting_effect

    initial = {"OPENROUTER_API_KEY": "fixture", "OUROBOROS_MODEL": "openai/old", "OUROBOROS_MODEL_SAFETY": "openai/checker",
               "OUROBOROS_CONSCIOUSNESS_DAILY_USD": 20, "OUROBOROS_CONSCIOUSNESS_AUTONOMY": "act",
               "OUROBOROS_CONSCIOUSNESS_MAX_TASKS": 2, "OUROBOROS_BG_WAKEUP_MIN": 900, "OUROBOROS_BG_WAKEUP_MAX": 14400,
               "LOCAL_MODEL_CONTEXT_LENGTH": 16384, "OUROBOROS_HOST_SERVICE_PORT": 8767}
    isolated_settings.write_text(json.dumps(initial), encoding="utf-8")
    for key in config.SETTINGS_DEFAULTS:
        monkeypatch.setenv(key, os.environ.get(key, ""))  # restore environment projection at fixture teardown
    config.apply_settings_to_env(config.load_settings())
    snapshot = task_settings_snapshot(config.load_settings(), dict(os.environ))
    client = TestClient(_settings_app(monkeypatch, isolated_settings))
    monkeypatch.setattr(settings, "_apply_settings_to_env", config.apply_settings_to_env)
    if case.endswith("wake"):
        monkeypatch.setattr(consciousness, "allowance_window", lambda *a, **kw: {
            "status": "available", "limit_usd": config.get_consciousness_daily_usd(),
            "remaining_usd": config.get_consciousness_daily_usd(), "settled_usd": 0, "accounted_usd": 0})
        alarm, due = clock.clock, clock.clock.next_wake_at
        if case == "launched_wake":
            assert alarm.tick(now=due + 1) == "launched"
            admitted = dict(clock.launches[0]["prepare"])
        patch = {"OUROBOROS_CONSCIOUSNESS_DAILY_USD": 40, "OUROBOROS_CONSCIOUSNESS_AUTONOMY": "observe",
                 "OUROBOROS_CONSCIOUSNESS_MAX_TASKS": 3, "OUROBOROS_BG_WAKEUP_MIN": 1200, "OUROBOROS_BG_WAKEUP_MAX": 7200}
        response = client.post("/api/settings", json=patch)
        assert response.status_code == 200 and response.json()["next_task_changed"], response.text
        assert all(setting_effect(key) == "next_wake" for key in patch)
        with task_settings_scope(snapshot):
            assert all(runtime_setting(key) == snapshot.environ[key] for key in patch)
        assert alarm.next_wake_at == due  # bounds never silently move an existing appointment
        if case == "scheduled_wake":
            assert alarm.tick(now=due - 1) == "not_due"
            assert alarm.tick(now=due + 1) == "launched"
            preparation = clock.launches[-1]["prepare"]
            assert preparation["level"] == "observe" and preparation["max_tasks"] == 3
            assert clock.launches[-1]["metadata"]["root_cost_ceiling_usd"] == 40
            assert alarm.floor == 1200 and alarm.ceiling == 7200
        else:
            assert alarm.tick(now=due + 2) == "wake_live"
            assert clock.launches[0]["prepare"] == admitted
            assert admitted["level"] == "act" and admitted["max_tasks"] == 2
            assert clock.launches[0]["metadata"]["root_cost_ceiling_usd"] == 20
    elif case.endswith("model") and case != "local_model":
        follows = case == "following_model"
        ctx = SimpleNamespace(route_choice=route_choice.following("main") if follows else route_choice.explicit("task_argument"))
        with task_settings_scope(snapshot):
            assert route_choice.round_route(ctx, None, ("openai/old", False)) is None
            response = client.post("/api/settings", json={"OUROBOROS_MODEL": "openai/new"})
            assert response.status_code == 200 and response.json()["immediate_changed"], response.text
            assert setting_effect("OUROBOROS_MODEL") == "next_call"
            assert route_choice.round_route(ctx, None, ("openai/old", False)) == (("openai/new", False) if follows else None)
            assert snapshot.settings["OUROBOROS_MODEL"] == "openai/old"
    else:
        manager = local_model.LocalModelManager()
        monkeypatch.setattr(local_model, "get_manager", lambda: manager)
        monkeypatch.setattr(manager, "get_status", lambda: "ready")
        manager._applied_settings = local_model.local_model_settings(initial)
        key, value = ("OUROBOROS_HOST_SERVICE_PORT", 8768) if case == "restart" else ("LOCAL_MODEL_CONTEXT_LENGTH", 32768)
        assert setting_effect(key) == case
        response = client.post("/api/settings", json={key: value})
        assert response.status_code == 200, response.text
        pending = response.json()["restart_state"]
        if case == "restart":
            assert pending["running"][key] == initial[key] and key in pending["restart_keys"]
        else:
            assert key in pending["local_model"]["pending_keys"] and key not in pending["restart_keys"]
            assert manager._applied_settings[key] == 16384
        # Saving the applied value clears the pending fact, without changing the actor snapshot.
        back = client.post("/api/settings", json={key: initial[key]}).json()["restart_state"]
        assert back["running"] == {} and not back["local_model"]["pending_keys"]
        assert snapshot.settings[key] == initial[key]


def test_worker_cap_applies_now_to_the_pool_without_changing_actor_snapshot_policy(isolated_settings, monkeypatch):
    import os
    from ouroboros import config
    from ouroboros.gateway import settings
    from ouroboros.settings_integrity import _next_task_setting, task_settings_scope, task_settings_snapshot
    from ouroboros.settings_scales import OWNER_CAP_KEY, OWNER_CAP_FORMAT_KEY, setting_effect
    from supervisor.worker_pool_lifecycle import acting_room

    isolated_settings.write_text(json.dumps({OWNER_CAP_KEY: 2, OWNER_CAP_FORMAT_KEY: 1}), encoding="utf-8")
    for key in config.SETTINGS_DEFAULTS:
        monkeypatch.setenv(key, os.environ.get(key, ""))
    monkeypatch.setenv(OWNER_CAP_KEY, "2")
    snapshot = task_settings_snapshot({OWNER_CAP_KEY: 2}, {OWNER_CAP_KEY: "2"})
    client = TestClient(_settings_app(monkeypatch, isolated_settings))
    monkeypatch.setattr(settings, "_apply_settings_to_env", config.apply_settings_to_env)
    assert setting_effect(OWNER_CAP_KEY) == "now"
    assert _next_task_setting(OWNER_CAP_KEY) is False
    with task_settings_scope(snapshot):
        assert acting_room({}, {}) == 2
        for value, room in ((5, 5), ("", None), (2, 2)):
            response = client.post("/api/settings", json={OWNER_CAP_KEY: value})
            assert response.status_code == 200, response.text
            assert response.json()["immediate_changed"] is True
            assert not response.json().get("next_task_changed") and not response.json().get("restart_required")
            assert acting_room({}, {}) == room  # real pool reader follows the save, even inside an old actor scope
            assert snapshot.settings[OWNER_CAP_KEY] == 2 and snapshot.environ[OWNER_CAP_KEY] == "2"


def test_metadata_covers_added_settings_but_never_tokens_a_new_secret(isolated_settings, monkeypatch):
    from ouroboros import config
    from ouroboros.gateway import settings
    from ouroboros.settings_setup_contract import SECRET_SETTING_KEYS
    from ouroboros.settings_scales import OWNER_CAP_KEY, setting_effect

    # A future ordinary key and a new custom secret must not need a second metadata inventory.
    added, secret = "OUROBOROS_TEST_SETTING", "NEW_SERVICE_API_KEY"
    monkeypatch.setitem(config.SETTINGS_DEFAULTS, added, 7)
    isolated_settings.write_text(json.dumps({secret: "private-test-key"}), encoding="utf-8")
    app = _settings_app(monkeypatch, isolated_settings)
    app.routes.append(Route("/api/settings", settings.api_settings_get, methods=["GET"]))
    response = TestClient(app).get("/api/settings")
    assert response.status_code == 200, response.text
    shown = response.json()
    meta = shown.pop("_meta")
    assert set(config.SETTINGS_DEFAULTS) <= shown.keys()
    assert meta["effects"] == {key: setting_effect(key) for key in shown}
    assert set(meta["tokens"]) == shown.keys() - set(SECRET_SETTING_KEYS) - {secret}
    assert shown[added] == 7 and meta["effects"][added] == "next_task" and meta["tokens"][added]
    assert meta["effects"][OWNER_CAP_KEY] == "now" and meta["tokens"][OWNER_CAP_KEY]
    assert secret in meta["effects"] and secret not in meta["tokens"]
    assert "private-test-key" not in response.text
    assert not {"OUROBOROS_DELEGATE_WAIT_SEC", "OUROBOROS_DELEGATE_WAIT_MAX_SEC"} & shown.keys()


@pytest.mark.parametrize("key,old,new", [
    ("OUROBOROS_NETWORK_PASSWORD", "old-pass", "new-pass"),
    ("OUROBOROS_FINALIZATION_GRACE_SEC", 120, 180),
    ("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", "true", "false"),
])
def test_now_labels_read_the_new_value_even_with_an_old_actor_snapshot(isolated_settings, monkeypatch, key, old, new):
    import os
    from ouroboros import config, server_auth
    from ouroboros.gateway import owner_settings
    from ouroboros.settings_integrity import task_settings_scope, task_settings_snapshot, runtime_setting
    from ouroboros.settings_scales import setting_effect
    from supervisor import queue

    isolated_settings.write_text(json.dumps({key: old}), encoding="utf-8")
    monkeypatch.setenv(key, str(old))
    snapshot = task_settings_snapshot(config.load_settings(), dict(os.environ))
    assert setting_effect(key) == "now"
    owner_settings._owner_update_settings(owner_settings.fields_on_saved_document({key: new}))
    monkeypatch.setenv(key, str(new))
    # The clock/queue runs outside an actor's scope. Its hot refresh changes the consumer's shared value.
    monkeypatch.setattr(queue, "FINALIZATION_GRACE_SEC", 120)
    queue.refresh_timeouts_from_settings({key: new})
    with task_settings_scope(snapshot):
        assert str(runtime_setting(key)) == str(old)  # presentation did not loosen snapshot semantics
        if key == "OUROBOROS_NETWORK_PASSWORD":
            assert server_auth.get_configured_network_password() == new
        elif key == "OUROBOROS_FINALIZATION_GRACE_SEC":
            assert queue.FINALIZATION_GRACE_SEC == new
        else:
            assert config.get_auto_grant_enabled() is False
    owner_settings._owner_update_settings(owner_settings.fields_on_saved_document({key: old}))
    monkeypatch.setenv(key, str(old))
    queue.refresh_timeouts_from_settings({key: old})
    assert (server_auth.get_configured_network_password() == old if key == "OUROBOROS_NETWORK_PASSWORD" else
            queue.FINALIZATION_GRACE_SEC == old if key == "OUROBOROS_FINALIZATION_GRACE_SEC" else config.get_auto_grant_enabled())



def test_settings_ui_save_flow_pins():
    """Static pins for the #285 UI surfaces (no page-wide Save, the one Restart, the standing
    reviewer hint): the strings the flow depends on must survive edits."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "web" / "modules"
    # Settings saves row by row through the one writer; the shell keeps no Save, Reload or Restart bar.
    shell = (root / "settings.js").read_text(encoding="utf-8") + (root / "settings_ui.js").read_text(encoding="utf-8")
    for retired in ("btn-save-settings", "btn-reload-settings", "btn-restart-now", "settings-footer"):
        assert retired not in shell, retired
    assert "cmd: '/restart'" in (root / "chat_activity.js").read_text(encoding="utf-8")

    # Reviewers are catalog rows marked Reviewer; their editor carries the standing hint.
    subagents_js = " ".join((root / "subagents_settings.js").read_text(encoding="utf-8").split())
    assert "a task that is already running keeps the reviewers it started with" in subagents_js

    # A Costs limit is a setting row: a write in flight is its control, disabled and busy; no "Saving…" is drawn.
    costs_js = (root / "costs.js").read_text(encoding="utf-8")
    controls_js = (root / "settings_controls.js").read_text(encoding="utf-8")
    assert "Saving…" not in costs_js
    assert "element.setAttribute('aria-busy', 'true')" in controls_js
