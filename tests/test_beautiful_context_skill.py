"""Bundled optional Beautiful Context skill: default-off registration, single viewer SSOT, and clean boundaries."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
from starlette.applications import Starlette
from starlette.routing import Route

from ouroboros import extension_loader, skill_builtin, skill_starter
from ouroboros.contracts.skill_manifest import parse_skill_manifest_text
from ouroboros.gateway.extensions import api_extension_dispatch
from ouroboros.gateway.widgets import api_widgets, widget_tabs
from ouroboros.skill_grant_follow import requested_skill_permissions
from ouroboros.skill_loader import find_skill, load_enabled, save_enabled
from ouroboros.skill_source_catalog import builtin_snapshot
from ouroboros.tools.skill_preflight import _handle_skill_preflight

REPO = Path(__file__).resolve().parents[1]
SKILL_DIR = REPO / "skills" / "beautiful_context"
BUNDLED_SOURCE = {"id": "builtin", "name": "Built in", "kind": "builtin", "trust": "trusted", "confirmed": True}


def test_beautiful_context_manifest_catalog_and_authorship(tmp_path):
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    manifest = parse_skill_manifest_text(text)

    assert manifest.name == "beautiful_context"
    assert manifest.type == "extension"
    assert manifest.entry == "plugin.py"
    assert manifest.plugin_api == {"version": "2.0", "capabilities": []}
    assert manifest.runtime == "python3"
    assert manifest.os == "any"
    assert manifest.permissions == ["widget", "route"]
    assert requested_skill_permissions(manifest.permissions, manifest.subscribe_events) == []
    assert manifest.ui_tab == {
        "id": "viewer",
        "title": "Beautiful Context",
        "render": {"kind": "iframe", "route": "viewer", "start": "auto", "height": 680, "span": 2},
    }
    assert manifest.model_experience.get("what_model_sees")
    assert manifest.model_experience.get("token_effect")
    assert "Egor Kosyrev" in text and "@josephsteuerjr" in text

    rows = {row["name"]: row for row in json.loads((REPO / "skills" / "builtin.json").read_text(encoding="utf-8"))["skills"]}
    row = rows["beautiful_context"]
    assert row.get("default_on") is not True
    assert row.get("recommended") is not True
    assert "beautiful_context" not in skill_builtin.default_on_names(REPO / "skills")
    assert "builtin/beautiful_context" not in skill_starter.default_starter_ids(
        tmp_path, rows=[BUNDLED_SOURCE], repo_dir=REPO, settings={},
    )

    offer = skill_starter.starter_groups(tmp_path, rows=[BUNDLED_SOURCE], repo_dir=REPO)
    items = {item["id"]: item for group in offer["groups"] for item in group["items"]}
    assert "builtin/beautiful_context" in items
    assert items["builtin/beautiful_context"]["recommended"] is False

    catalog = {entry["name"]: entry for entry in builtin_snapshot(REPO)["entries"]}
    assert "beautiful_context" in catalog
    assert catalog["beautiful_context"]["recommended"] is False


def test_beautiful_context_skill_preflight_and_boundaries(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true")
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)

    installed = skill_builtin.install_builtin_skill("beautiful_context", drive_root=tmp_path, seed_dir=REPO / "skills")
    assert installed["ok"] is True
    assert load_enabled(tmp_path, "beautiful_context") is False

    ctx = SimpleNamespace(drive_root=tmp_path, repo_dir=REPO)
    preflight = json.loads(_handle_skill_preflight(ctx, skill="beautiful_context"))
    assert preflight["ok"] is True, preflight
    assert not any(f.get("ok") is False for f in preflight.get("findings") or []), preflight

    plugin_source = (SKILL_DIR / "plugin.py").read_text(encoding="utf-8")
    assert "context_report_archive" not in plugin_source
    assert "launcher_context_report" not in plugin_source
    assert "pywebview" not in plugin_source


def test_beautiful_context_lifecycle_serves_shared_viewer_and_vanishes_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_TRUST_NATIVE_SEEDED_SKILLS", "true")
    monkeypatch.setattr("ouroboros.config.SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.delenv("OUROBOROS_SKILLS_REPO_PATH", raising=False)
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    monkeypatch.setattr("ouroboros.config.REPO_DIR", REPO)

    app = Starlette(routes=[
        Route("/api/widgets", api_widgets, methods=["GET"]),
        Route("/api/extensions/{skill}/{rest:path}", api_extension_dispatch, methods=["GET", "POST", "HEAD"]),
    ])
    app.state.drive_root = tmp_path
    app.state.repo_dir = REPO

    async def request_app(method: str, path: str) -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://fixture") as client:
            return await client.request(method, path)

    try:
        # Fresh install: absent from live widgets and GET /api/widgets.
        assert not any(tab["skill"] == "beautiful_context" for tab in widget_tabs())
        before = asyncio.run(request_app("GET", "/api/widgets"))
        assert before.status_code == 200
        assert not any(t["skill"] == "beautiful_context" for t in before.json()["ui_tabs"])

        installed = skill_builtin.install_builtin_skill("beautiful_context", drive_root=tmp_path, seed_dir=REPO / "skills")
        assert installed["ok"] is True
        skill = find_skill(tmp_path, "beautiful_context")
        assert skill is not None and skill.enabled is False
        assert skill.review.gate_for(skill.content_hash)["blocking_reason"] == "trusted_source"

        # Enable -> tab and GET route appear; no tools or ws handlers.
        save_enabled(tmp_path, "beautiful_context", True, actor="owner_ui")
        state = extension_loader.reconcile_extension("beautiful_context", tmp_path, lambda: {}, repo_path="")
        assert state["live_loaded"] is True, state

        live = extension_loader.snapshot()
        assert not [t for t in live["tools"] if "beautiful_context" in t]
        assert not [w for w in live["ws_handlers"] if "beautiful_context" in w]
        assert "/api/extensions/beautiful_context/viewer" in live["routes"]
        tabs = [tab for tab in widget_tabs() if tab["skill"] == "beautiful_context"]
        assert len(tabs) == 1
        assert tabs[0]["tab_id"] == "viewer"
        assert tabs[0]["title"] == "Beautiful Context"
        assert tabs[0]["span"] == 2
        assert tabs[0]["render"] == {
            "kind": "iframe", "route": "viewer", "start": "auto", "height": 680, "span": 2,
        }

        # The extension route serves the exact shared web/context-report.html without caching or writes.
        expected_html = (REPO / "web" / "context-report.html").read_text(encoding="utf-8")
        widgets_resp = asyncio.run(request_app("GET", "/api/widgets"))
        assert widgets_resp.status_code == 200
        assert any(t["skill"] == "beautiful_context" and t["tab_id"] == "viewer" for t in widgets_resp.json()["ui_tabs"])

        viewer_resp = asyncio.run(request_app("GET", "/api/extensions/beautiful_context/viewer"))
        assert viewer_resp.status_code == 200
        assert viewer_resp.text == expected_html
        assert viewer_resp.headers.get("cache-control") == "no-store"
        assert viewer_resp.headers.get("referrer-policy") == "no-referrer"

        post_resp = asyncio.run(request_app("POST", "/api/extensions/beautiful_context/viewer"))
        assert post_resp.status_code == 405

        # Disable -> tab and route disappear cleanly.
        save_enabled(tmp_path, "beautiful_context", False, actor="owner_ui")
        off_state = extension_loader.reconcile_extension("beautiful_context", tmp_path, lambda: {}, repo_path="")
        assert off_state["live_loaded"] is False
        assert not any(tab["skill"] == "beautiful_context" for tab in widget_tabs())
        assert "/api/extensions/beautiful_context/viewer" not in extension_loader.list_routes()

        widgets_off = asyncio.run(request_app("GET", "/api/widgets"))
        assert widgets_off.status_code == 200
        assert not any(t["skill"] == "beautiful_context" for t in widgets_off.json()["ui_tabs"])
    finally:
        extension_loader.unload_extension("beautiful_context")
