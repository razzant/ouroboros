"""The settings API body and the owner endpoints that may change a mode.

Split verbatim out of ``tests/test_runtime_mode_elevation.py`` by theme. This module
owns ``_merge_settings_payload`` dropping the owner-only keys from a generic POST, the
owner runtime-mode, auto-grant and context-mode endpoints, their pending/restart
reporting, and the refusals they raise while a task is running.

Hermetic — no network, no supervisor boot. Uses temp dirs for ``DATA_DIR`` /
``SETTINGS_PATH`` overrides via monkeypatching ``ouroboros.config`` module-level
constants.
"""

from __future__ import annotations

import json
import os

import pytest


from tests._runtime_mode_elevation_shared import (
    _seed_disk,
)
from tests._runtime_mode_elevation_shared import isolated_settings as _isolated_settings

# The fixture is requested by name as a test parameter, so it is re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
isolated_settings = _isolated_settings


# ---------------------------------------------------------------------------
# 3. /api/settings drops OUROBOROS_RUNTIME_MODE from the body
# ---------------------------------------------------------------------------


def test_merge_settings_payload_skips_runtime_mode():
    """``_merge_settings_payload`` is the chokepoint for /api/settings POST."""
    from ouroboros.gateway import settings as server_mod

    old = {"OUROBOROS_RUNTIME_MODE": "light", "OPENAI_API_KEY": "old-key"}
    body = {"OUROBOROS_RUNTIME_MODE": "pro", "OPENAI_API_KEY": "new-key"}
    merged = server_mod._merge_settings_payload(old, body)
    # Mode comes from old (= disk), NOT from body.
    assert merged["OUROBOROS_RUNTIME_MODE"] == "light"
    # Other keys still flow through.
    assert merged["OPENAI_API_KEY"] == "new-key"


def test_merge_settings_payload_skips_auto_grant_reviewed_skills():
    """Auto-grant changes use the dedicated owner endpoint, not /api/settings."""
    from ouroboros.gateway import settings as server_mod

    old = {"OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS": "false"}
    body = {"OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS": "true"}
    merged = server_mod._merge_settings_payload(old, body)

    assert merged["OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS"] == "false"


def test_merge_settings_payload_skips_context_mode():
    """Context mode is owner-only (BIBLE P1 cognitive horizon): the agent-reachable
    /api/settings POST must not be able to lower it; it flows through the
    dedicated /api/owner/context-mode endpoint instead."""
    from ouroboros.gateway import settings as server_mod

    old = {"OUROBOROS_CONTEXT_MODE": "max"}
    body = {"OUROBOROS_CONTEXT_MODE": "low"}
    merged = server_mod._merge_settings_payload(old, body)

    assert merged["OUROBOROS_CONTEXT_MODE"] == "max"


def test_owner_runtime_mode_endpoint_persists_next_boot_without_env_elevation(isolated_settings, monkeypatch):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import config as cfg
    from ouroboros.gateway.settings import api_owner_runtime_mode

    _seed_disk(isolated_settings, {"OUROBOROS_RUNTIME_MODE": "advanced"})
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    cfg.initialize_runtime_mode_baseline("advanced")

    app = Starlette(routes=[Route("/api/owner/runtime-mode", endpoint=api_owner_runtime_mode, methods=["POST"])])
    app.state.drive_root = isolated_settings.parent
    response = TestClient(app).post("/api/owner/runtime-mode", json={"mode": "pro"})

    assert response.status_code == 200, response.text
    assert response.json() == {"ok": True, "runtime_mode": "pro", "restart_required": True}
    on_disk = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert on_disk["OUROBOROS_RUNTIME_MODE"] == "pro"
    assert os.environ["OUROBOROS_RUNTIME_MODE"] == "advanced"
    assert os.environ["OUROBOROS_BOOT_RUNTIME_MODE"] == "advanced"


def test_owner_runtime_mode_endpoint_reports_no_restart_when_mode_unchanged(isolated_settings, monkeypatch):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import config as cfg
    from ouroboros.gateway.settings import api_owner_runtime_mode

    _seed_disk(isolated_settings, {"OUROBOROS_RUNTIME_MODE": "advanced"})
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    cfg.initialize_runtime_mode_baseline("advanced")

    app = Starlette(routes=[Route("/api/owner/runtime-mode", endpoint=api_owner_runtime_mode, methods=["POST"])])
    app.state.drive_root = isolated_settings.parent
    before = isolated_settings.stat().st_mtime_ns
    response = TestClient(app).post("/api/owner/runtime-mode", json={"mode": "advanced"})

    assert response.status_code == 200, response.text
    assert response.json() == {"ok": True, "runtime_mode": "advanced", "restart_required": False}
    assert os.environ["OUROBOROS_RUNTIME_MODE"] == "advanced"
    # A no-change POST must not rewrite settings.json: the rewrite raced a
    # concurrent generic save (last-writer-wins over a stale read).
    assert isolated_settings.stat().st_mtime_ns == before
    assert json.loads(isolated_settings.read_text(encoding="utf-8")) == {
        "OUROBOROS_RUNTIME_MODE": "advanced",
    }


def test_owner_runtime_mode_endpoint_reports_restart_until_pending_mode_is_active(isolated_settings, monkeypatch):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import config as cfg
    from ouroboros.gateway.settings import api_owner_runtime_mode

    _seed_disk(isolated_settings, {"OUROBOROS_RUNTIME_MODE": "pro"})
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    cfg.initialize_runtime_mode_baseline("advanced")

    app = Starlette(routes=[Route("/api/owner/runtime-mode", endpoint=api_owner_runtime_mode, methods=["POST"])])
    app.state.drive_root = isolated_settings.parent
    response = TestClient(app).post("/api/owner/runtime-mode", json={"mode": "pro"})

    assert response.status_code == 200, response.text
    assert response.json() == {"ok": True, "runtime_mode": "pro", "restart_required": True}
    assert os.environ["OUROBOROS_RUNTIME_MODE"] == "advanced"


@pytest.mark.parametrize("next_mode", ["pro", "light"])
def test_generic_settings_save_preserves_pending_runtime_mode_without_hot_apply(
    isolated_settings,
    monkeypatch,
    next_mode,
):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import config as cfg
    from ouroboros.gateway import settings as settings_mod
    from ouroboros.gateway.settings import api_owner_runtime_mode, api_settings_post

    _seed_disk(isolated_settings, {
        "OUROBOROS_RUNTIME_MODE": "advanced",
        "TOTAL_BUDGET": "10",
    })
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    cfg.initialize_runtime_mode_baseline("advanced")
    monkeypatch.setattr(settings_mod, "apply_runtime_provider_defaults", lambda s: (s, False, []))
    monkeypatch.setattr(settings_mod, "_start_supervisor_if_needed_for_request", lambda *_a, **_k: False)

    app = Starlette(routes=[
        Route("/api/owner/runtime-mode", endpoint=api_owner_runtime_mode, methods=["POST"]),
        Route("/api/settings", endpoint=api_settings_post, methods=["POST"]),
    ])
    app.state.drive_root = isolated_settings.parent
    client = TestClient(app)

    owner_resp = client.post("/api/owner/runtime-mode", json={"mode": next_mode})
    assert owner_resp.status_code == 200, owner_resp.text
    save_resp = client.post("/api/settings", json={"TOTAL_BUDGET": "77"})

    assert save_resp.status_code == 200, save_resp.text
    on_disk = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert on_disk["OUROBOROS_RUNTIME_MODE"] == next_mode
    assert on_disk["TOTAL_BUDGET"] == 77.0
    assert os.environ["OUROBOROS_RUNTIME_MODE"] == "advanced"
    assert os.environ["OUROBOROS_BOOT_RUNTIME_MODE"] == "advanced"


def test_settings_save_warns_when_an_agent_task_is_running(isolated_settings, monkeypatch):
    """Owner decision (2026-08-05, option B): the task-start snapshot boundary
    stays, and a save landing while an agent task runs must SAY that the running
    task keeps its previous reviewer/subagent config — a bare "Settings saved"
    read as "applied to the task you are watching"."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros import config as cfg
    from ouroboros.gateway import settings as settings_mod
    from ouroboros.gateway.settings import api_settings_post

    _seed_disk(isolated_settings, {"OUROBOROS_RUNTIME_MODE": "advanced"})
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    # The REAL save endpoint applies saved keys to os.environ; register the key
    # with monkeypatch so teardown restores it (the save below writes "claude",
    # which otherwise leaks into later tests' route resolution). delenv on an
    # ABSENT key records nothing — setenv is what makes teardown restore.
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "")
    cfg.initialize_runtime_mode_baseline("advanced")
    monkeypatch.setattr(settings_mod, "apply_runtime_provider_defaults", lambda s: (s, False, []))
    monkeypatch.setattr(settings_mod, "_start_supervisor_if_needed_for_request", lambda *_a, **_k: False)
    monkeypatch.setattr(settings_mod, "_has_started_agent_tasks", lambda: True)

    app = Starlette(routes=[Route("/api/settings", endpoint=api_settings_post, methods=["POST"])])
    app.state.drive_root = isolated_settings.parent
    client = TestClient(app)

    # A next-task-class key (the delegation route) changed while a task runs.
    resp = client.post("/api/settings", json={"OUROBOROS_SUBAGENT_HARNESS": "off"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "saved"
    assert data.get("agent_task_running") is True
    warnings = data.get("warnings") or []
    assert any("keeps the configuration it started with" in w for w in warnings), warnings
    assert any("next task" in w for w in warnings), warnings

    # No running task -> no warning noise.
    monkeypatch.setattr(settings_mod, "_has_started_agent_tasks", lambda: False)
    resp2 = client.post("/api/settings", json={"OUROBOROS_SUBAGENT_HARNESS": "claude"})
    assert resp2.status_code == 200, resp2.text
    data2 = resp2.json()
    assert "agent_task_running" not in data2
    assert not any("keeps the configuration" in w for w in (data2.get("warnings") or []))


def test_started_predicate_is_read_only_and_never_constructs_the_agent(monkeypatch):
    """Negative pin (delta gate 2026-08-05): _has_started_agent_tasks must never
    call workers._get_chat_agent() — that CONSTRUCTS the agent and inserts the
    canonical repo into sys.path (proven test-isolation poison). Reading the
    existing instance (or its absence) is the whole contract."""
    import supervisor.workers as workers
    from ouroboros.gateway.settings import _has_started_agent_tasks

    def _boom():
        raise AssertionError("predicate constructed the agent")

    monkeypatch.setattr(workers, "_get_chat_agent", _boom)
    monkeypatch.setattr(workers, "RUNNING", {}, raising=False)
    monkeypatch.setattr(workers, "_chat_agent", None, raising=False)
    assert _has_started_agent_tasks() is False

    class _Busy:
        _busy = True

    monkeypatch.setattr(workers, "_chat_agent", _Busy(), raising=False)
    assert _has_started_agent_tasks() is True


def test_owner_auto_grant_endpoint_persists_outside_generic_settings(isolated_settings, monkeypatch):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.settings import api_owner_auto_grant

    _seed_disk(isolated_settings, {
        "OUROBOROS_RUNTIME_MODE": "pro",
        "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS": "false",
    })
    monkeypatch.delenv("OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", raising=False)

    app = Starlette(routes=[Route("/api/owner/auto-grant", endpoint=api_owner_auto_grant, methods=["POST"])])
    app.state.drive_root = isolated_settings.parent
    response = TestClient(app).post("/api/owner/auto-grant", json={"enabled": True})

    assert response.status_code == 200, response.text
    assert response.json() == {"ok": True, "enabled": True}
    on_disk = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert on_disk["OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS"] == "true"
    assert on_disk["OUROBOROS_RUNTIME_MODE"] == "pro"
    assert os.environ["OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS"] == "true"


def test_owner_context_mode_endpoint_persists_and_hot_applies(isolated_settings, monkeypatch):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway import settings as settings_mod

    api_owner_context_mode = settings_mod.api_owner_context_mode
    # This positive case owns its idle precondition. Other xdist tests exercise
    # the process-global supervisor queues and must not make the endpoint test
    # order-dependent; the following test covers the busy rejection explicitly.
    monkeypatch.setattr(settings_mod, "_has_running_agent_tasks", lambda: False)

    _seed_disk(isolated_settings, {
        "OUROBOROS_RUNTIME_MODE": "pro",
        "OUROBOROS_CONTEXT_MODE": "max",
    })
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    # Own the compatibility tombstone because the endpoint writes os.environ directly.
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE_AUTO_LOW", "false")

    app = Starlette(routes=[Route("/api/owner/context-mode", endpoint=api_owner_context_mode, methods=["POST"])])
    app.state.drive_root = isolated_settings.parent
    client = TestClient(app)

    response = client.post("/api/owner/context-mode", json={"mode": "low"})

    assert response.status_code == 200, response.text
    assert response.json() == {"ok": True, "context_mode": "low"}
    on_disk = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert on_disk["OUROBOROS_CONTEXT_MODE"] == "low"
    assert on_disk["OUROBOROS_RUNTIME_MODE"] == "pro"
    assert os.environ["OUROBOROS_CONTEXT_MODE"] == "low"
    # Owner selection atomically carries the one-window false provenance tombstone,
    # so stored Low still means the P3 scope review is not performed.
    assert on_disk["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"
    assert os.environ["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"

    invalid = client.post("/api/owner/context-mode", json={"mode": "huge"})
    assert invalid.status_code == 400, invalid.text
    assert "'mode' must be one of: low, max" in invalid.text
    assert os.environ["OUROBOROS_CONTEXT_MODE"] == "low"


def test_owner_context_mode_endpoint_refuses_lowering_while_task_runs(isolated_settings, monkeypatch):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway import settings as settings_mod
    from ouroboros.gateway.settings import api_owner_context_mode

    _seed_disk(isolated_settings, {"OUROBOROS_CONTEXT_MODE": "max"})
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    monkeypatch.setattr(settings_mod, "_has_running_agent_tasks", lambda: True)

    app = Starlette(routes=[Route("/api/owner/context-mode", endpoint=api_owner_context_mode, methods=["POST"])])
    app.state.drive_root = isolated_settings.parent
    response = TestClient(app).post("/api/owner/context-mode", json={"mode": "low"})

    assert response.status_code == 409, response.text
    assert "only be lowered while Ouroboros is idle" in response.text
    assert "queued or running work" in response.text
    assert json.loads(isolated_settings.read_text(encoding="utf-8"))["OUROBOROS_CONTEXT_MODE"] == "max"


def test_owner_context_mode_idle_predicate_covers_pending_and_direct_chat_busy(monkeypatch):
    from types import SimpleNamespace

    from ouroboros.gateway import settings as settings_mod
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "PENDING", [{"id": "queued"}])
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(workers, "_get_chat_agent", lambda: SimpleNamespace(_busy=False))
    assert settings_mod._has_running_agent_tasks() is True

    monkeypatch.setattr(workers, "PENDING", [])
    monkeypatch.setattr(workers, "_get_chat_agent", lambda: SimpleNamespace(_busy=True))
    assert settings_mod._has_running_agent_tasks() is True

    monkeypatch.setattr(workers, "_get_chat_agent", lambda: SimpleNamespace(_busy=False))
    assert settings_mod._has_running_agent_tasks() is False


def test_save_settings_refuses_context_mode_lowering_without_owner_flag(isolated_settings, monkeypatch):
    from ouroboros.config import save_settings

    _seed_disk(isolated_settings, {"OUROBOROS_CONTEXT_MODE": "max"})
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")

    with pytest.raises(PermissionError) as exc:
        save_settings({"OUROBOROS_CONTEXT_MODE": "low"})

    assert "OUROBOROS_CONTEXT_MODE lowering refused" in str(exc.value)
    assert json.loads(isolated_settings.read_text(encoding="utf-8"))["OUROBOROS_CONTEXT_MODE"] == "max"
