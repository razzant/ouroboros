"""The launcher bridges: runtime mode, auto-grant and skill grants after confirmation.

Split verbatim out of ``tests/test_runtime_mode_elevation.py`` by theme. This module
owns the launcher's confirmation-gated saves, its pending-restart and cancel
reporting, and the skill key/permission grants it validates against review and
manifest before reconciling.

Hermetic — no network, no supervisor boot. Uses temp dirs for ``DATA_DIR`` /
``SETTINGS_PATH`` overrides via monkeypatching ``ouroboros.config`` module-level
constants.
"""

from __future__ import annotations

import types



from tests._runtime_mode_elevation_shared import isolated_settings as _isolated_settings

# The fixture is requested by name as a test parameter, so it is re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
isolated_settings = _isolated_settings


def test_launcher_runtime_mode_bridge_saves_after_confirmation(monkeypatch):
    import launcher

    saved = {}
    monkeypatch.setattr(launcher, "_load_settings", lambda: {"OUROBOROS_RUNTIME_MODE": "advanced"})
    monkeypatch.setattr(launcher, "_save_settings", lambda settings: saved.update(settings))
    monkeypatch.setattr(launcher, "get_runtime_mode", lambda: "advanced")

    result = launcher._request_runtime_mode_change("pro", lambda _title, _message: True)

    assert result["ok"] is True
    assert result["runtime_mode"] == "pro"
    assert result["restart_required"] is True
    assert saved["OUROBOROS_RUNTIME_MODE"] == "pro"


def test_launcher_runtime_mode_bridge_reports_pending_restart_against_active(monkeypatch):
    import launcher

    saved = {}
    monkeypatch.setattr(launcher, "_load_settings", lambda: {"OUROBOROS_RUNTIME_MODE": "pro"})
    monkeypatch.setattr(launcher, "_save_settings", lambda settings: saved.update(settings))
    monkeypatch.setattr(launcher, "get_runtime_mode", lambda: "advanced")

    result = launcher._request_runtime_mode_change("pro", lambda _title, _message: False)

    assert result == {"ok": True, "runtime_mode": "pro", "restart_required": True}
    assert saved == {}


def test_launcher_runtime_mode_bridge_can_cancel_pending_mode_without_restart(monkeypatch):
    import launcher

    saved = {}
    monkeypatch.setattr(launcher, "_load_settings", lambda: {"OUROBOROS_RUNTIME_MODE": "pro"})
    monkeypatch.setattr(launcher, "_save_settings", lambda settings: saved.update(settings))
    monkeypatch.setattr(launcher, "get_runtime_mode", lambda: "advanced")

    result = launcher._request_runtime_mode_change("advanced", lambda _title, _message: True)

    assert result == {"ok": True, "runtime_mode": "advanced", "restart_required": False}
    assert saved["OUROBOROS_RUNTIME_MODE"] == "advanced"


def test_launcher_auto_grant_bridge_saves_after_confirmation(monkeypatch):
    import launcher

    saved = {}
    monkeypatch.setattr(launcher, "_load_settings", lambda: {"OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS": "false"})
    monkeypatch.setattr(launcher, "_save_settings", lambda settings: saved.update(settings))

    result = launcher._request_auto_grant_reviewed_skills_change(True, lambda _title, _message: True)

    assert result == {"ok": True, "enabled": True}
    assert saved["OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS"] == "true"


def test_launcher_auto_grant_bridge_disables_truthy_alias(monkeypatch):
    import launcher

    saved = {}
    monkeypatch.setattr(launcher, "_load_settings", lambda: {"OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS": "1"})
    monkeypatch.setattr(launcher, "_save_settings", lambda settings: saved.update(settings))

    result = launcher._request_auto_grant_reviewed_skills_change(False, lambda _title, _message: True)

    assert result == {"ok": True, "enabled": False}
    assert saved["OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS"] == "false"


def test_launcher_skill_key_grant_validates_review_and_manifest(monkeypatch, tmp_path):
    import launcher

    class _Manifest:
        env_from_settings = ["OPENROUTER_API_KEY"]
        def is_script(self):
            return True
        def is_extension(self):
            return False

    class _Review:
        status = "advisory_pass"
        def is_stale_for(self, _hash):
            return False

    loaded = types.SimpleNamespace(
        name="demo",
        manifest=_Manifest(),
        review=_Review(),
        content_hash="hash-a",
    )
    captured = {}
    monkeypatch.setattr(launcher, "DATA_DIR", tmp_path)
    monkeypatch.setattr(launcher, "_load_settings", lambda: {"OUROBOROS_SKILLS_REPO_PATH": ""})
    monkeypatch.setattr("ouroboros.skill_loader.find_skill", lambda *_a, **_kw: loaded)
    monkeypatch.setattr(
        "ouroboros.skill_loader.save_skill_grants",
        lambda drive, name, keys, **kw: captured.update(
            {"drive": drive, "name": name, "keys": keys, **kw}
        ),
    )

    result = launcher._request_skill_key_grant(
        "demo",
        ["OPENROUTER_API_KEY"],
        lambda _title, _message: True,
    )

    assert result["ok"] is True
    assert captured["name"] == "demo"
    assert captured["keys"] == ["OPENROUTER_API_KEY"]
    assert captured["content_hash"] == "hash-a"
    assert captured["requested_keys"] == ["OPENROUTER_API_KEY"]
    # v5.2.2: scripts pick up grants on next ``_scrub_env`` call so no
    # server reconcile is invoked. ``extension_action`` and
    # ``extension_reason`` therefore stay ``None`` for script-type
    # skills.
    assert result.get("extension_action") is None
    assert result.get("extension_reason") is None


def test_launcher_skill_grant_supports_permission_grants(monkeypatch, tmp_path):
    import launcher

    class _Manifest:
        env_from_settings = []
        permissions = ["inject_chat", "subscribe_event"]
        subscribe_events = ["chat.outbound"]
        def is_script(self):
            return False
        def is_extension(self):
            return True

    class _Review:
        status = "pass"
        def is_stale_for(self, _hash):
            return False

    loaded = types.SimpleNamespace(
        name="bridge",
        manifest=_Manifest(),
        review=_Review(),
        content_hash="hash-a",
    )
    captured = {}
    monkeypatch.setattr(launcher, "DATA_DIR", tmp_path)
    monkeypatch.setattr(launcher, "_load_settings", lambda: {"OUROBOROS_SKILLS_REPO_PATH": ""})
    monkeypatch.setattr("ouroboros.skill_loader.find_skill", lambda *_a, **_kw: loaded)
    monkeypatch.setattr(
        "ouroboros.skill_loader.save_skill_grants",
        lambda drive, name, keys, **kw: captured.update(
            {"drive": drive, "name": name, "keys": keys, **kw}
        ),
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_kw: types.SimpleNamespace(read=lambda: b'{"ok": true}'))

    result = launcher._request_skill_key_grant(
        "bridge",
        ["inject_chat", "subscribe_event:chat.outbound"],
        lambda _title, _message: True,
    )

    assert result["ok"] is True
    assert captured["keys"] == []
    assert captured["granted_permissions"] == ["inject_chat", "subscribe_event:chat.outbound"]
    assert captured["requested_permissions"] == ["inject_chat", "subscribe_event:chat.outbound"]


def test_launcher_skill_key_grant_supports_extensions(monkeypatch, tmp_path):
    """v5.2.2 dual-track grants: ``type: extension`` skills can be
    granted core keys and the launcher posts to the agent server's
    /api/skills/<name>/reconcile so the new grant reaches the live
    plugin without forcing a manual disable/enable.

    The launcher and server are independent OS processes — this test
    verifies the cross-process contract by stubbing ``urllib.request.urlopen``
    instead of stubbing ``reconcile_extension`` directly (which only
    runs in the launcher process and would not affect the server).
    """
    import launcher

    class _Manifest:
        env_from_settings = ["OPENROUTER_API_KEY"]
        def is_script(self):
            return False
        def is_extension(self):
            return True

    class _Review:
        status = "pass"
        def is_stale_for(self, _hash):
            return False

    loaded = types.SimpleNamespace(
        name="demo_ext",
        manifest=_Manifest(),
        review=_Review(),
        content_hash="ext-hash",
    )
    captured: dict = {}
    reconcile_calls: list = []
    monkeypatch.setattr(launcher, "DATA_DIR", tmp_path)
    monkeypatch.setattr(launcher, "_load_settings", lambda: {"OUROBOROS_SKILLS_REPO_PATH": ""})
    monkeypatch.setattr(launcher, "_read_port_file", lambda: 8765)
    monkeypatch.setattr("ouroboros.skill_loader.find_skill", lambda *_a, **_kw: loaded)
    monkeypatch.setattr(
        "ouroboros.skill_loader.save_skill_grants",
        lambda drive, name, keys, **kw: captured.update(
            {"drive": drive, "name": name, "keys": keys, **kw}
        ),
    )

    class _FakeResponse:
        def __init__(self, body: bytes):
            self._body = body
        def read(self):
            return self._body
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False

    def _fake_urlopen(req, timeout=10):
        reconcile_calls.append({
            "url": req.full_url,
            "method": req.get_method(),
            "data": req.data,
        })
        return _FakeResponse(
            b'{"skill":"demo_ext","extension_action":"extension_loaded",'
            b'"extension_reason":"ready","live_loaded":true,"load_error":null}'
        )

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    result = launcher._request_skill_key_grant(
        "demo_ext",
        ["OPENROUTER_API_KEY"],
        lambda _title, _message: True,
    )

    assert result["ok"] is True
    assert captured["name"] == "demo_ext"
    assert captured["keys"] == ["OPENROUTER_API_KEY"]
    assert len(reconcile_calls) == 1
    call = reconcile_calls[0]
    assert call["url"] == "http://127.0.0.1:8765/api/skills/demo_ext/reconcile"
    assert call["method"] == "POST"
    assert result.get("extension_action") == "extension_loaded"


def test_launcher_skill_key_grant_handles_reconcile_http_error(monkeypatch, tmp_path):
    """If the server-side reconcile HTTP call fails, the grant write
    succeeded but the response carries ``extension_reason='reconcile_call_failed'``
    so the UI can warn the user without throwing away the persisted grant."""
    import launcher

    class _Manifest:
        env_from_settings = ["OPENROUTER_API_KEY"]
        def is_script(self):
            return False
        def is_extension(self):
            return True

    class _Review:
        status = "pass"
        def is_stale_for(self, _hash):
            return False

    loaded = types.SimpleNamespace(
        name="demo_ext",
        manifest=_Manifest(),
        review=_Review(),
        content_hash="ext-hash",
    )
    monkeypatch.setattr(launcher, "DATA_DIR", tmp_path)
    monkeypatch.setattr(launcher, "_load_settings", lambda: {"OUROBOROS_SKILLS_REPO_PATH": ""})
    monkeypatch.setattr(launcher, "_read_port_file", lambda: 8765)
    monkeypatch.setattr("ouroboros.skill_loader.find_skill", lambda *_a, **_kw: loaded)
    monkeypatch.setattr(
        "ouroboros.skill_loader.save_skill_grants",
        lambda *_a, **_kw: None,
    )

    def _broken_urlopen(*_a, **_kw):
        raise ConnectionError("server not reachable")

    monkeypatch.setattr("urllib.request.urlopen", _broken_urlopen)

    result = launcher._request_skill_key_grant(
        "demo_ext",
        ["OPENROUTER_API_KEY"],
        lambda _title, _message: True,
    )

    # Grant itself succeeded (file persisted)
    assert result["ok"] is True
    assert result.get("granted_keys") == ["OPENROUTER_API_KEY"]
    # But the server reconcile failed and the UI is told
    assert result.get("extension_reason") == "reconcile_call_failed"
    assert result.get("extension_action") is None


def test_launcher_skill_key_grant_rejects_instruction_skill(monkeypatch, tmp_path):
    import launcher

    class _Manifest:
        env_from_settings = ["OPENROUTER_API_KEY"]
        def is_script(self):
            return False
        def is_extension(self):
            return False

    class _Review:
        status = "pass"
        def is_stale_for(self, _hash):
            return False

    loaded = types.SimpleNamespace(
        name="instr",
        manifest=_Manifest(),
        review=_Review(),
        content_hash="instr-hash",
    )
    monkeypatch.setattr(launcher, "DATA_DIR", tmp_path)
    monkeypatch.setattr(launcher, "_load_settings", lambda: {"OUROBOROS_SKILLS_REPO_PATH": ""})
    monkeypatch.setattr("ouroboros.skill_loader.find_skill", lambda *_a, **_kw: loaded)

    result = launcher._request_skill_key_grant(
        "instr",
        ["OPENROUTER_API_KEY"],
        lambda _title, _message: True,
    )
    assert result["ok"] is False
    assert "script and extension" in result["error"]
