"""Keep running after the window closes: the host fact, its owner endpoint and the model's view.

The choice lives in settings.json (``OUROBOROS_DESKTOP_KEEP_RUNNING``) and is available only
where the running launcher says it can hide its window and still be reached
(``OUROBOROS_DESKTOP_BACKGROUND=1``: Windows and macOS desktop windows). Everything runs
against an isolated settings file; nothing touches the live install.
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import desktop_autostart

KEY = "OUROBOROS_DESKTOP_KEEP_RUNNING"


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A capable desktop host: packaged window on macOS, launcher exporting background support."""
    from ouroboros import config as cfg

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path, raising=True)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", tmp_path / "settings.json", raising=True)
    cfg.reset_runtime_mode_baseline_for_tests()
    monkeypatch.setattr(desktop_autostart, "sys", SimpleNamespace(platform="darwin"))
    for name, value in (("OUROBOROS_PRESENTATION", "desktop_window"), ("OUROBOROS_MANAGED_BY_LAUNCHER", "1"),
                        (desktop_autostart.BACKGROUND_ENV, "1")):
        monkeypatch.setenv(name, value)
    yield tmp_path / "settings.json"
    cfg.reset_runtime_mode_baseline_for_tests()


def _app(tmp_path):
    from ouroboros.gateway import desktop_autostart as endpoints

    app = Starlette(routes=[
        Route("/api/desktop/background", endpoint=endpoints.api_desktop_background_get, methods=["GET"]),
        Route("/api/desktop/background", endpoint=endpoints.api_desktop_background_post, methods=["POST"]),
    ])
    app.state.drive_root = tmp_path
    return TestClient(app)


def test_the_choice_reads_undecided_until_the_owner_answers(host):
    assert desktop_autostart.keep_running_choice() == ""
    assert desktop_autostart.background_status() == {"state": "off"}
    desktop_autostart.set_keep_running(True)
    assert json.loads(host.read_text(encoding="utf-8"))[KEY] == "true"
    assert desktop_autostart.keep_running_choice() == "true"
    assert desktop_autostart.background_status() == {"state": "on"}
    desktop_autostart.set_keep_running(False)
    assert desktop_autostart.keep_running_choice() == "false", "declined is remembered, unlike undecided"


@pytest.mark.parametrize("platform,env,reason", [
    ("darwin", {"OUROBOROS_PRESENTATION": "web"}, "packaged desktop app"),  # Docker, browser, Colab, server.py
    ("darwin", {"OUROBOROS_PRESENTATION": "browser_fallback"}, "packaged desktop app"),
    ("linux", {}, "Not available on Linux yet"),
    ("win32", {desktop_autostart.BACKGROUND_ENV: ""}, "newer app build"),  # a launcher from before this mode
])
def test_unavailable_hosts_say_why_and_record_nothing(host, monkeypatch, platform, env, reason):
    monkeypatch.setattr(desktop_autostart, "sys", SimpleNamespace(platform=platform))
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    status = desktop_autostart.background_status(True)
    assert status["state"] == "unavailable" and reason in status["reason"]
    assert not host.exists()


def test_the_endpoint_records_the_choice_and_audits_it(host, tmp_path):
    client = _app(tmp_path)
    assert client.get("/api/desktop/background").json() == {"state": "off"}
    response = client.post("/api/desktop/background", json={"enabled": True})
    assert response.status_code == 200 and response.json() == {"state": "on"}
    assert json.loads(host.read_text(encoding="utf-8"))[KEY] == "true"
    audits = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert audits[-1]["action"] == "desktop_background" and audits[-1]["enabled"] is True


@pytest.mark.parametrize("body", [{"enabled": "yes"}, {"enabled": True, "extra": 1}, ["enabled"]])
def test_a_malformed_body_is_refused_before_anything_is_saved(host, tmp_path, body):
    response = _app(tmp_path).post("/api/desktop/background", json=body)
    assert response.status_code == 400 and response.json()["saved"] is False
    assert not host.exists()


def test_an_unavailable_host_refuses_the_write(host, tmp_path, monkeypatch):
    monkeypatch.setattr(desktop_autostart, "sys", SimpleNamespace(platform="linux"))
    response = _app(tmp_path).post("/api/desktop/background", json={"enabled": True})
    assert response.status_code == 409 and response.json()["saved"] is False
    assert "Linux" in response.json()["error"]
    assert not host.exists()


def test_a_contended_settings_lock_is_a_typed_refusal(host, tmp_path):
    lock_path = pathlib.Path(str(host) + ".lock")
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        response = _app(tmp_path).post("/api/desktop/background", json={"enabled": True})
    finally:
        os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()
    assert response.status_code == 503 and response.json() == {**response.json(), "code": "settings_locked", "saved": False}
    assert not host.exists()


def test_the_model_sees_the_effective_choice(host, monkeypatch):
    monkeypatch.setattr(desktop_autostart, "autostart_status", lambda: {"state": "off"})
    assert desktop_autostart.runtime_facts()["keep_running_after_close"] is False
    desktop_autostart.set_keep_running(True)
    assert desktop_autostart.runtime_facts()["keep_running_after_close"] is True
    monkeypatch.setenv(desktop_autostart.BACKGROUND_ENV, "")  # an older launcher quits on close whatever the setting
    assert desktop_autostart.runtime_facts()["keep_running_after_close"] is False
