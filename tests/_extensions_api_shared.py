"""Fixtures and client builders shared by the extension HTTP-surface suites.

Split out of ``tests/test_extensions_api.py`` when that module was divided by theme;
every definition is verbatim, so each sibling suite keeps the exact runtime cleanup,
extension layout and TestClient wiring it was written against. ``_clean_extensions``
is autouse, so importing it into a test module re-applies it there.
"""

from __future__ import annotations

import json
import pathlib

import pytest


from tests._shared import clean_extension_runtime_state


@pytest.fixture(autouse=True)
def _clean_extensions():
    clean_extension_runtime_state()
    yield
    clean_extension_runtime_state()


def _write_ext(
    repo_root: pathlib.Path,
    name: str,
    *,
    permissions: list[str],
    plugin: str,
    env_from_settings: list[str] | None = None,
    conflicts: list[str] | None = None,
) -> pathlib.Path:
    skill_dir = repo_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    perms_yaml = json.dumps(permissions)
    env_yaml = json.dumps(env_from_settings or [])
    conflicts_yaml = json.dumps(conflicts or [])
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            f"name: {name}\n"
            "description: Test ext.\n"
            "version: 0.1.0\n"
            "type: extension\n"
            "entry: plugin.py\n"
            f"permissions: {perms_yaml}\n"
            f"env_from_settings: {env_yaml}\n"
            f"conflicts: {conflicts_yaml}\n"
            "---\n"
            "body\n"
        ),
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text(plugin, encoding="utf-8")
    return skill_dir


def _make_client(tmp_path: pathlib.Path, monkeypatch):
    """Return ``(client, drive_root, patches)`` — Starlette TestClient with drive_root pinned.

    Tests that prefer the auto-cleanup variant should use the ``client_env``
    fixture below instead of calling this directly.
    """
    from unittest.mock import patch
    from starlette.testclient import TestClient

    import server as srv

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    # ``srv.app`` is the NetworkAuthGate wrapper; the inner Starlette is at
    # ``srv.app.app``. Pin ``drive_root`` / ``repo_dir`` on the inner state.
    srv.app.app.state.drive_root = drive_root  # type: ignore[attr-defined]
    srv.app.app.state.repo_dir = tmp_path / "repo"  # type: ignore[attr-defined]

    patches = [
        patch.object(srv, "_start_supervisor_if_needed", lambda *_a, **_k: None),
        patch.object(srv, "_apply_settings_to_env", lambda *_a, **_k: None),
        patch.object(srv, "apply_runtime_provider_defaults", lambda s: (s, False, [])),
        patch("ouroboros.server_auth.get_configured_network_password", return_value=""),
    ]
    for p in patches:
        p.start()
    client = TestClient(srv.app)
    return client, drive_root, patches


def _stop_patches(patches):
    for p in patches:
        try:
            p.stop()
        except RuntimeError:
            pass
