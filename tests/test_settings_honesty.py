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
    from ouroboros.gateway.settings import _IMMEDIATE_KEYS

    for key in ("MCP_ENABLED", "MCP_SERVERS", "MCP_TOOL_TIMEOUT_SEC"):
        assert key in _IMMEDIATE_KEYS
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

    from ouroboros.gateway import settings as settings_mod

    fake_mcp = types.SimpleNamespace(
        reconfigure_from_settings=lambda *_a, **_k: (_ for _ in ()).throw(
            RuntimeError("mcp exploded")),
        refresh_all_background=lambda *_a, **_k: None,
    )
    monkeypatch.setitem(__import__("sys").modules, "ouroboros.mcp_client", fake_mcp)

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


def test_provider_base_url_keys_apply_on_the_next_task_not_restart(monkeypatch, isolated_settings):
    """llm.py resolves base URLs per call via configured() (env refreshed at
    every task start), so a restart was never required for them."""
    from ouroboros.gateway.settings import _RESTART_REQUIRED_KEYS

    for key in ("OPENAI_BASE_URL", "OPENAI_COMPATIBLE_BASE_URL",
                "CLOUDRU_FOUNDATION_MODELS_BASE_URL", "MINIMAX_REGION",
                "GIGACHAT_SCOPE", "GIGACHAT_BASE_URL", "GIGACHAT_VERIFY_SSL_CERTS"):
        assert key not in _RESTART_REQUIRED_KEYS
    data = _save(monkeypatch, isolated_settings, {"OPENAI_BASE_URL": "https://example.test/v1"})
    assert not data.get("restart_required")
    assert data.get("next_task_changed") is True


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


def test_settings_ui_save_flow_pins():
    """Static pins for the #285 UI surfaces (busy save, Restart now, standing
    Review-lanes hint): the strings the flow depends on must survive edits."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "web" / "modules"
    settings_js = (root / "settings.js").read_text(encoding="utf-8")
    assert "setStatus('Saving…', 'muted')" in settings_js
    assert "setButtonBusy(saveButton, true)" in settings_js
    assert "setButtonBusy(saveButton, false)" in settings_js
    assert "cmd: '/restart'" in settings_js
    assert "btn-restart-now" in settings_js

    settings_ui_js = (root / "settings_ui.js").read_text(encoding="utf-8")
    assert 'id="btn-restart-now" hidden' in settings_ui_js

    reviewer_slots_js = (root / "reviewer_slots.js").read_text(encoding="utf-8")
    assert "keeps the reviewer configuration it started with" in reviewer_slots_js

    costs_js = (root / "costs.js").read_text(encoding="utf-8")
    assert "'Saving…'" in costs_js
