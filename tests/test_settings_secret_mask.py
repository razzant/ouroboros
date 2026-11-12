"""A display placeholder must never be stored as a credential.

The Settings API answers a GET with a placeholder instead of the stored secret,
so any client can post that placeholder back. Persisting it replaces the working
credential with a placeholder, and every consumer (env apply, provider catalogs,
capability probes) then sends it as an ``Authorization`` value — which is what an
OpenAI-compatible gateway rejects with "expected to start with 'sk-'".

Values here are fixtures, never real credentials.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.secret_masking import (
    looks_masked_mcp_secret,
    looks_masked_secret,
    looks_masked_settings_secret,
)
from tests._cases import case_folders

REAL_KEY = "sk-original-secret"
TOKEN_MASKS = ("***", "***set***", "sk-origi...")


@pytest.fixture()
def settings_client(tmp_path, monkeypatch):
    """TestClient over the real GET/POST settings endpoints with an in-memory disk."""
    import server as srv

    on_disk: Dict[str, Any] = {
        "OPENAI_COMPATIBLE_API_KEY": REAL_KEY,
        "OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:4000/v1",
        "OPENROUTER_API_KEY": "sk-or-original",
        "OPENAI_API_KEY": "sk-openai-original",
        "ANTHROPIC_API_KEY": "sk-ant-original",
        "GITHUB_TOKEN": "ghp-original",
    }
    saved: Dict[str, Any] = {}

    def fake_load_settings():
        return copy.deepcopy(on_disk)

    def fake_save_settings(settings, *_a, **_k):
        saved.clear()
        saved.update(settings)
        on_disk.update(settings)

    gateway = srv._gateway_settings

    def provider_defaults(settings):
        return dict(settings), False, []

    monkeypatch.setattr(gateway, "load_settings", fake_load_settings)
    # server's compatibility shim still assigns this retired module attribute.
    # Record its absence so monkeypatch removes the temporary assignment again.
    monkeypatch.setattr(gateway, "save_settings", fake_save_settings, raising=False)
    monkeypatch.setattr(gateway, "_apply_settings_to_env", lambda *a, **k: None)
    monkeypatch.setattr(gateway, "apply_runtime_provider_defaults", provider_defaults)
    monkeypatch.setattr(gateway, "_owner_read_settings_raw", fake_load_settings)
    monkeypatch.setattr(gateway, "settings_read_view", lambda: (fake_load_settings(), {}))
    monkeypatch.setattr(gateway, "_owner_write_settings", fake_save_settings)

    # server's compatibility wrapper copies these four globals into the gateway
    # on every request. Patch the gateway first so monkeypatch restores both
    # modules to their real functions after the fixture.
    monkeypatch.setattr(srv, "load_settings", fake_load_settings)
    monkeypatch.setattr(srv, "save_settings", fake_save_settings)
    monkeypatch.setattr(srv, "_apply_settings_to_env", lambda *a, **k: None)
    monkeypatch.setattr(srv, "apply_runtime_provider_defaults", provider_defaults)

    app = Starlette(
        routes=[
            Route("/api/settings", endpoint=srv.api_settings_get, methods=["GET"]),
            Route("/api/settings", endpoint=srv.api_settings_post, methods=["POST"]),
        ]
    )
    app.state.drive_root = tmp_path
    app.state.repo_dir = tmp_path
    with TestClient(app) as client:
        yield client, on_disk, saved


def test_get_masks_the_stored_key(settings_client):
    client, _on_disk, _saved = settings_client
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    assert REAL_KEY not in resp.text
    assert looks_masked_settings_secret("OPENAI_COMPATIBLE_API_KEY", resp.json()["OPENAI_COMPATIBLE_API_KEY"])


def test_reposting_the_served_mask_keeps_the_real_key(settings_client):
    """The exact UI flow: load, then save without touching the field."""
    client, on_disk, saved = settings_client
    mask = client.get("/api/settings").json()["OPENAI_COMPATIBLE_API_KEY"]

    resp = client.post("/api/settings", json={"OPENAI_COMPATIBLE_API_KEY": mask})

    assert resp.status_code == 200, resp.text
    assert saved["OPENAI_COMPATIBLE_API_KEY"] == REAL_KEY
    assert on_disk["OPENAI_COMPATIBLE_API_KEY"] == REAL_KEY


@pytest.mark.parametrize("mask", TOKEN_MASKS)
def test_a_mask_is_never_persisted_even_with_nothing_stored(settings_client, mask):
    """Without this, an empty field turns the placeholder into the credential."""
    client, on_disk, saved = settings_client
    on_disk["OPENAI_COMPATIBLE_API_KEY"] = ""

    resp = client.post("/api/settings", json={"OPENAI_COMPATIBLE_API_KEY": mask})

    assert resp.status_code == 200, resp.text
    assert saved["OPENAI_COMPATIBLE_API_KEY"] == ""


@pytest.mark.parametrize("mask", TOKEN_MASKS)
def test_a_mask_is_never_persisted_as_a_custom_secret(settings_client, mask):
    """Custom secret keys share the contract; they are masked by the same GET."""
    client, _on_disk, saved = settings_client

    resp = client.post("/api/settings", json={"MY_CUSTOM_TOKEN": mask})

    assert resp.status_code == 200, resp.text
    assert not saved.get("MY_CUSTOM_TOKEN")


def test_password_mask_is_never_persisted_with_nothing_stored(settings_client):
    client, on_disk, saved = settings_client
    on_disk["OUROBOROS_NETWORK_PASSWORD"] = ""

    resp = client.post("/api/settings", json={"OUROBOROS_NETWORK_PASSWORD": "***set***"})

    assert resp.status_code == 200, resp.text
    assert saved["OUROBOROS_NETWORK_PASSWORD"] == ""


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("OUROBOROS_NETWORK_PASSWORD", "correct-horse..."),
        ("MY_CUSTOM_TOKEN", "owner-chosen..."),
    ],
)
def test_a_real_secret_ending_in_ellipsis_is_persisted(settings_client, key, value):
    client, on_disk, saved = settings_client
    on_disk[key] = ""

    resp = client.post("/api/settings", json={key: value})

    assert resp.status_code == 200, resp.text
    assert saved[key] == value


def test_a_new_key_replaces_the_old_one(settings_client):
    client, on_disk, saved = settings_client

    resp = client.post("/api/settings", json={"OPENAI_COMPATIBLE_API_KEY": "sk-new-secret"})

    assert resp.status_code == 200, resp.text
    assert saved["OPENAI_COMPATIBLE_API_KEY"] == "sk-new-secret"
    assert on_disk["OPENAI_COMPATIBLE_API_KEY"] == "sk-new-secret"


def test_explicit_clear_removes_the_key(settings_client):
    """The Clear button posts an empty string; that must really delete it."""
    client, on_disk, saved = settings_client

    resp = client.post("/api/settings", json={"OPENAI_COMPATIBLE_API_KEY": ""})

    assert resp.status_code == 200, resp.text
    assert saved["OPENAI_COMPATIBLE_API_KEY"] == ""
    assert on_disk["OPENAI_COMPATIBLE_API_KEY"] == ""


def test_absent_key_keeps_the_stored_secret(settings_client):
    client, _on_disk, saved = settings_client

    resp = client.post("/api/settings", json={"OUROBOROS_MODEL": "openai-compatible::local-reason"})

    assert resp.status_code == 200, resp.text
    assert saved["OPENAI_COMPATIBLE_API_KEY"] == REAL_KEY


@pytest.mark.parametrize(
    "key",
    [
        "OPENROUTER_API_KEY",
        "GITHUB_TOKEN",
        "OPENAI_COMPATIBLE_API_KEY",
    ],
)
def test_every_provider_secret_round_trips(settings_client, key):
    """One contract for all masked secrets, not a per-provider patch."""
    client, on_disk, saved = settings_client
    original = on_disk[key]
    mask = client.get("/api/settings").json()[key]
    assert looks_masked_settings_secret(key, mask)

    assert client.post("/api/settings", json={key: mask}).status_code == 200
    assert saved[key] == original

    assert client.post("/api/settings", json={key: "brand-new-value-123"}).status_code == 200
    assert saved[key] == "brand-new-value-123"


def test_a_stored_mask_reads_back_as_unset(subtests, tmp_path, monkeypatch):
    """Repair for an install an older round-trip already poisoned."""
    case_paths = case_folders(tmp_path)
    for mask in TOKEN_MASKS:
        tmp_path = next(case_paths)
        with subtests.test(mask=mask), monkeypatch.context() as monkeypatch:
            from ouroboros import config as cfg

            settings_path = tmp_path / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "OPENAI_COMPATIBLE_API_KEY": mask,
                        "OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:4000/v1",
                    }
                ),
                encoding="utf-8",
            )
            monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path)
            monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)

            loaded = cfg.load_settings()

            assert loaded["OPENAI_COMPATIBLE_API_KEY"] == ""
            assert loaded["OPENAI_COMPATIBLE_BASE_URL"] == "http://127.0.0.1:4000/v1"


def test_a_stored_password_mask_reads_back_as_unset(tmp_path, monkeypatch):
    from ouroboros import config as cfg

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"OUROBOROS_NETWORK_PASSWORD": "***set***"}), encoding="utf-8")
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path)

    assert cfg.load_settings()["OUROBOROS_NETWORK_PASSWORD"] == ""


def test_disk_mask_does_not_block_a_real_environment_secret(tmp_path, monkeypatch):
    from ouroboros import config as cfg

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"OPENAI_COMPATIBLE_API_KEY": "***"}), encoding="utf-8")
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path)
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "sk-real-from-env")

    assert cfg.load_settings()["OPENAI_COMPATIBLE_API_KEY"] == "sk-real-from-env"


def test_environment_secret_ending_in_ellipsis_is_not_repaired(tmp_path, monkeypatch):
    from ouroboros import config as cfg

    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path)
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "sk-real-ending...")

    assert cfg.load_settings()["OPENAI_COMPATIBLE_API_KEY"] == "sk-real-ending..."


def test_real_stored_password_ending_in_ellipsis_is_not_repaired(tmp_path, monkeypatch):
    from ouroboros import config as cfg

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"OUROBOROS_NETWORK_PASSWORD": "correct-horse..."}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path)

    assert cfg.load_settings()["OUROBOROS_NETWORK_PASSWORD"] == "correct-horse..."


def test_custom_stored_masks_read_back_as_unset(tmp_path, monkeypatch):
    from ouroboros import config as cfg

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "MY_CUSTOM_TOKEN": "***",
                "TELEGRAM_BOT_TOKEN": "telegram...",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path)

    loaded = cfg.load_settings()

    assert loaded["MY_CUSTOM_TOKEN"] == ""
    assert loaded["TELEGRAM_BOT_TOKEN"] == ""


def test_common_writer_blanks_top_level_masks(tmp_path, monkeypatch):
    from ouroboros import config as cfg

    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path)

    cfg.save_settings(
        {
            "OPENAI_COMPATIBLE_API_KEY": "sk-origi...",
            "MY_CUSTOM_TOKEN": "my-token...",
        }
    )

    raw = json.loads(settings_path.read_text(encoding="utf-8"))
    assert raw["OPENAI_COMPATIBLE_API_KEY"] == ""
    assert raw["MY_CUSTOM_TOKEN"] == ""


def test_a_stored_mask_never_reaches_the_environment(tmp_path, monkeypatch):
    """apply_settings_to_env writes the real process env, so restore it after."""
    from ouroboros import config as cfg

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "OPENAI_COMPATIBLE_API_KEY": "***",
                "OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:4000/v1",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path)
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)

    saved_env = dict(os.environ)
    try:
        cfg.apply_settings_to_env(cfg.load_settings())
        assert os.environ.get("OPENAI_COMPATIBLE_API_KEY", "") == ""
    finally:
        os.environ.clear()
        os.environ.update(saved_env)


@pytest.mark.parametrize("value", ["***", "sk-origi...", "abcd..."])
def test_looks_masked_secret_accepts_every_placeholder(value):
    assert looks_masked_secret(value) is True


@pytest.mark.parametrize(
    "value",
    ["", "p@ssw0rd", "correct-horse..."],
)
def test_looks_masked_secret_rejects_real_values(value):
    assert looks_masked_secret(value) is False


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("OPENAI_API_KEY", "***", True),
        ("OUROBOROS_NETWORK_PASSWORD", "***", False),
        ("OUROBOROS_NETWORK_PASSWORD", "sk-origi...", False),
    ],
)
def test_settings_masks_are_key_class_specific(key, value, expected):
    assert looks_masked_settings_secret(key, value) is expected


@pytest.mark.parametrize("value", ["***", "abcd...", "sk-origi..."])
def test_mcp_recognizer_accepts_both_emitted_prefix_lengths(value):
    assert looks_masked_mcp_secret(value) is True


@pytest.mark.parametrize("value", ["***set***", "abcde...", "correct-horse..."])
def test_mcp_recognizer_rejects_other_shapes(value):
    assert looks_masked_mcp_secret(value) is False
