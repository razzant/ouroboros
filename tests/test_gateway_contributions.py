"""Contributions at the live HTTP, publication, preference and lifecycle boundaries."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.testclient import TestClient

from ouroboros import extension_loader
from ouroboros.extension_registry_state import live_contribution_projection, live_host_asset
from ouroboros.gateway.router import collect_routes
from ouroboros.skill_loader import (SkillReviewState, load_skill, save_enabled, save_review_state,
                                   save_skill_grants, grant_status_for_skill)
from tests._extension_loader_shared import _clear_loader_state  # noqa: F401
from tests.test_skill_contributions import TEMPLATE, STT, CHIP, CLIENT


def installed(tmp_path, name="demo", *, entries=None, kind="extension", permissions=(), grant=True,
              enabled=True, reviewed=True, plugin="def register(api): pass\n"):
    skill_dir = tmp_path / "skills/native" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / ".seed-origin").write_text("test seed")
    (skill_dir / "skill.json").write_text(json.dumps({"name": "Display " + name, "type": kind,
        "entry": "plugin.py" if kind == "extension" else "", "plugin_api": "2.0",
        "permissions": list(permissions), "contributes": entries or [TEMPLATE]}))
    (skill_dir / "plugin.py").write_text(plugin)
    (skill_dir / "ui/lib").mkdir(parents=True, exist_ok=True)
    (skill_dir / "ui/voice.js").write_text("import './lib/helper.js'; export function mount() {}")
    (skill_dir / "ui/lib/helper.js").write_text("export const value = 'original';")
    (skill_dir / "ui/theme.css").write_text("body { color: red; }")
    (skill_dir / "ui/data.json").write_text('{"ok":true}')
    (skill_dir / "ui/pixel.png").write_bytes(b"\x89PNG\r\n")
    (skill_dir / "ui/private.py").write_text("value = 'not a page asset'")
    loaded = load_skill(skill_dir, tmp_path)
    save_enabled(tmp_path, name, enabled)
    if reviewed:
        save_review_state(tmp_path, name, SkillReviewState(status="clean", content_hash=loaded.content_hash))
    if grant and "page_module" in permissions:
        save_skill_grants(tmp_path, name, [], content_hash=loaded.content_hash, requested_keys=[],
                          requested_permissions=["page_module"], granted_permissions=["page_module"])
    return load_skill(skill_dir, tmp_path)


def client(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: "")
    app = Starlette(routes=collect_routes(data_dir=tmp_path))
    app.state.drive_root = tmp_path
    app.state.repo_dir = tmp_path
    app.state.events = []
    app.state.broadcast_ws_sync = app.state.events.append
    return TestClient(app)


def test_live_projection_instruction_review_and_disabled_offers(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    live = installed(tmp_path, entries=[STT, CHIP])
    installed(tmp_path, "templates", kind="instruction")
    installed(tmp_path, "off", kind="instruction", enabled=False)
    installed(tmp_path, "pending", kind="instruction", reviewed=False)
    installed(tmp_path, "unloaded", entries=[STT])
    installed(tmp_path, "browser", entries=[{**CLIENT, "provides": "stt", "title": "Chrome sends audio to Google"}],
              permissions=["page_module"], enabled=False)
    assert extension_loader.load_extension(live, lambda: {}, drive_root=tmp_path) is None
    with client(tmp_path, monkeypatch) as http:
        response = http.get("/api/ui/contributions")
        assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
        payload = response.json()
        rows = {row["key"]: row for row in payload["contributions"]}
        assert set(rows) == {"demo:speech", "demo:quota", "templates:simpler"}
        speech = rows["demo:speech"]
        assert speech["skill_label"] == "Display demo"
        assert speech["route_url"] == "/api/extensions/demo/transcribe"
        assert (speech["max_bytes"], speech["max_seconds"], speech["formats"]) == (393216, 60, STT["formats"])
        assert rows["demo:quota"]["source"] == {"url": "/api/extensions/demo/chip", "refresh_s": 60}
        assert speech["revision"] == live.content_hash
        assert "sparkle" in payload["icons"]
        assert {item["skill"] for item in payload["offers"]["stt"]} == {"browser", "unloaded"}
        assert all(item["enabled"] is False for item in payload["offers"]["stt"])
        # An offer carries its skill's own words on where audio goes, for the chooser before the engine runs.
        assert {item["skill"]: item["title"] for item in payload["offers"]["stt"]} == {
            "browser": "Chrome sends audio to Google", "unloaded": ""}
        index = http.get("/api/extensions").json()
        assert index["contributions"] == payload["contributions"]
        assert index["live"]["contributions"] == extension_loader.snapshot()["contributions"]
        assert http.get("/api/extensions/demo/manifest").json()["manifest"]["contributes"] == [STT, CHIP]
        extension_loader.unload_extension("demo")
        assert [row["key"] for row in http.get("/api/ui/contributions").json()["contributions"]] == ["templates:simpler"]


def test_host_assets_are_captured_revision_bound_and_confined(tmp_path, monkeypatch):
    live = installed(tmp_path, entries=[CLIENT], permissions=["page_module"])
    assert extension_loader.load_extension(live, lambda: {}, drive_root=tmp_path) is None
    with client(tmp_path, monkeypatch) as http:
        row = http.get("/api/ui/contributions").json()["contributions"][0]
        url = row["render"]["url"]
        assert url == f"/api/extensions/demo/ui/{live.content_hash}/ui/voice.js"
        source = http.get(url)
        assert source.status_code == 200 and source.headers["content-type"].startswith("text/javascript")
        assert source.headers["cache-control"] == "no-store"
        base = url.rsplit("/", 1)[0]
        for path, mime in (("theme.css", "text/css"), ("data.json", "application/json"), ("pixel.png", "image/png")):
            assert http.get(base + "/" + path).headers["content-type"].startswith(mime)
        (live.skill_dir / "ui/lib/helper.js").write_text("export const value = 'edited';")
        assert "original" in http.get(base + "/lib/helper.js").text
        for path in ("private.py", ".hidden.js", "missing.js", "%2e%2e/plugin.py", "lib%5chelper.js"):
            assert http.get(base + "/" + path).status_code == 404
        assert http.get(url.replace(live.content_hash, "stale")).status_code == 404
        assert http.get(url.replace("/ui/voice.js", "/plugin.py")).status_code == 404
        extension_loader.unload_extension("demo")
        assert http.get(url).status_code == 404
        changed = load_skill(live.skill_dir, tmp_path)
        save_review_state(tmp_path, "demo", SkillReviewState(status="clean", content_hash=changed.content_hash))
        save_skill_grants(tmp_path, "demo", [], content_hash=changed.content_hash, requested_keys=[],
                          requested_permissions=["page_module"], granted_permissions=["page_module"])
        changed = load_skill(live.skill_dir, tmp_path)
        assert extension_loader.load_extension(changed, lambda: {}, drive_root=tmp_path) is None
        new_url = http.get("/api/ui/contributions").json()["contributions"][0]["render"]["url"]
        assert new_url != url and http.get(url).status_code == 404
        assert "edited" in http.get(new_url.rsplit("/", 1)[0] + "/lib/helper.js").text


def test_ungranted_page_power_is_visible_and_blocks_load(tmp_path, monkeypatch):
    skill = installed(tmp_path, entries=[CLIENT], permissions=["page_module"], grant=False)
    assert grant_status_for_skill(tmp_path, skill)["missing_permissions"] == ["page_module"]
    assert "missing owner grants" in extension_loader.load_extension(skill, lambda: {}, drive_root=tmp_path)
    assert not live_contribution_projection()
    save_skill_grants(tmp_path, "demo", [], content_hash=skill.content_hash, requested_keys=[],
                      requested_permissions=["page_module"], granted_permissions=["page_module"])
    assert extension_loader.load_extension(skill, lambda: {}, drive_root=tmp_path) is None
    assert live_contribution_projection()[0]["key"] == "demo:voice"


def test_invalid_entry_is_reported_without_losing_working_entry(tmp_path, monkeypatch):
    skill = installed(tmp_path, entries=[TEMPLATE, {**CLIENT, "render": {"kind": "host", "entry": "ui/missing.js"}}],
                      permissions=["page_module"])
    assert extension_loader.load_extension(skill, lambda: {}, drive_root=tmp_path) is None
    with client(tmp_path, monkeypatch) as http:
        payload = http.get("/api/ui/contributions").json()
        assert [row["key"] for row in payload["contributions"]] == ["demo:simpler"]
        assert "render.entry" in payload["problems"]["demo"][0]


def test_out_of_process_publication_and_recovery_keep_capture(tmp_path):
    skill = installed(tmp_path, entries=[CLIENT], permissions=["page_module"])
    args = dict(catalog={}, drive_root=tmp_path, state_dir=tmp_path / "state/skills/demo", settings_reader=lambda: {}, granted_keys=[], dependency_site_dirs_enabled=False)
    extension_loader._publish_out_of_process_registration(skill, current_hash=skill.content_hash, **args)
    generation = extension_loader._extensions["demo"].generation_digest
    before = live_host_asset("demo", skill.content_hash, "ui/voice.js")
    (skill.skill_dir / "ui/voice.js").write_text("edited")
    extension_loader._publish_out_of_process_registration(skill, expected_generation=generation, **args)
    assert live_host_asset("demo", skill.content_hash, "ui/voice.js") == before
    assert live_contribution_projection()[0]["revision"] == skill.content_hash
    extension_loader.unload_extension("demo")
    assert live_host_asset("demo", skill.content_hash, "ui/voice.js") is None


def test_failed_registration_publishes_no_contributions(tmp_path):
    failed = installed(tmp_path, entries=[CLIENT], permissions=["page_module"],
                       plugin="def register(api): raise RuntimeError('refused')\n")
    assert "refused" in extension_loader.load_extension(failed, lambda: {}, drive_root=tmp_path)
    assert live_contribution_projection() == []
    good = installed(tmp_path, "working", entries=[CLIENT], permissions=["page_module"])
    assert extension_loader.load_extension(good, lambda: {}, drive_root=tmp_path) is None
    assert live_contribution_projection()[0]["key"] == "working:voice"


def test_owner_toggle_invalidates_instruction_contributions(tmp_path, monkeypatch):
    skill = installed(tmp_path, kind="instruction", enabled=False)
    with client(tmp_path, monkeypatch) as http:
        assert http.get("/api/ui/contributions").json()["contributions"] == []
        for enabled in (True, False):
            response = http.post("/api/skills/demo/toggle", json={"enabled": enabled, "expected_content_hash": skill.content_hash})
            assert response.status_code == 200, response.text
            assert bool(http.get("/api/ui/contributions").json()["contributions"]) is enabled
            assert http.app.state.events[-1]["type"] == "extension_lifecycle"
            assert http.app.state.events[-1]["action"] == ("enable" if enabled else "disable")


def test_preferences_merge_and_missing_provider_stays_selected(tmp_path, monkeypatch):
    live = installed(tmp_path, entries=[STT])
    assert extension_loader.load_extension(live, lambda: {}, drive_root=tmp_path) is None
    with client(tmp_path, monkeypatch) as http:
        assert http.post("/api/ui/preferences", json={"voice": {"stt": "demo:speech"}}).status_code == 200
        assert http.post("/api/ui/preferences", json={"voice": {"tts": "missing:voice"}}).json()["voice"] == {"stt": "demo:speech", "tts": "missing:voice", "tts_voice": ""}
        # The chosen voice id merges beside the engine choice and is never read as a provider kind.
        saved = http.post("/api/ui/preferences", json={"voice": {"tts_voice": "com.apple.voice.Milena"}}).json()["voice"]
        assert saved == {"stt": "demo:speech", "tts": "missing:voice", "tts_voice": "com.apple.voice.Milena"}
        assert sorted(http.get("/api/ui/contributions").json()["voice"]) == ["stt", "tts"]
        assert http.post("/api/ui/preferences", json={"contributions": {"hidden": ["demo:speech"]}}).status_code == 200
        assert http.post("/api/ui/preferences", json={"contributions": {"order": ["gone:key"]}}).json()["contributions"] == {"hidden": ["demo:speech"], "order": ["gone:key"]}
        assert http.get("/api/ui/contributions").json()["voice"]["stt"]["available"] is True
        extension_loader.unload_extension("demo")
        choice = http.get("/api/ui/contributions").json()["voice"]["stt"]
        assert choice == {"key": "demo:speech", "available": False, "reason": "provider_unavailable"}
        assert http.get("/api/ui/preferences").json()["voice"]["stt"] == "demo:speech"
        for bad in ({"voice": {"stt": "x" * 121}}, {"voice": {"tts": False}}, {"voice": {"other": "x"}},
                    {"voice": {"tts_voice": "x" * 121}}, {"voice": {"tts_voice": None}},
                    {"contributions": {"hidden": ["x"] * 65}}, {"contributions": {"order": [3]}},
                    {"contributions": {"order": ["x" * 121]}}):
            assert http.post("/api/ui/preferences", json=bad).status_code == 400
        assert http.post("/api/ui/preferences", json={"voice": {"stt": "x" * 120}, "contributions": {"hidden": [str(i) for i in range(64)]}}).status_code == 200


def test_a_stored_preference_this_release_refuses_never_fails_the_skills_page(tmp_path, monkeypatch):
    installed(tmp_path, entries=[STT])
    prefs = tmp_path / "state" / "ui_preferences.json"
    prefs.parent.mkdir(parents=True, exist_ok=True)
    with client(tmp_path, monkeypatch) as http:
        prefs.write_text(json.dumps({"voice": {"stt": "demo:speech"}}), encoding="utf-8")
        assert http.get("/api/ui/contributions").json()["voice"]["stt"]["key"] == "demo:speech", "a good value is read"
        prefs.write_text(json.dumps({"voice": {"stt": 5}, "contributions": {"hidden": "x"}}), encoding="utf-8")
        catalog = http.get("/api/ui/contributions")
        assert catalog.status_code == 200 and catalog.json()["voice"]["stt"]["reason"] == "not_selected"
        assert http.get("/api/extensions").status_code == 200


def test_speech_and_ordinary_uploads_keep_existing_cap(tmp_path, monkeypatch):
    skill = installed(tmp_path, entries=[STT], permissions=["route"])
    extension_loader._publish_out_of_process_registration(skill, catalog={"routes": [
        {"skill": "demo", "path": f"/api/extensions/demo/{name}", "methods": ["POST"]}
        for name in ("transcribe", "ordinary")]}, drive_root=tmp_path, state_dir=tmp_path / "state/skills/demo",
        settings_reader=lambda: {}, granted_keys=[], dependency_site_dirs_enabled=False, current_hash=skill.content_hash)
    seen = []
    def dispatch(spec, request, **kwargs):
        from starlette.responses import JSONResponse
        seen.append(request)
        return JSONResponse({"text": "recognized"})
    monkeypatch.setattr("ouroboros.extension_process_runner.dispatch_extension_route_subprocess", dispatch)
    with client(tmp_path, monkeypatch) as http:
        for name in ("transcribe", "ordinary"):
            assert http.post(f"/api/extensions/demo/{name}?lang=en", content=b"a" * 393216, headers={"Content-Type": "audio/webm"}).json() == {"text": "recognized"}
            assert http.post(f"/api/extensions/demo/{name}", content=b"a" * (512 * 1024 + 1)).status_code == 413
    assert len(seen) == 2 and all(row["query_string"] == "lang=en" for row in seen)


def test_instruction_review_changes_follow_install_enforcement(tmp_path, monkeypatch):
    skill = installed(tmp_path, kind="instruction")
    save_review_state(tmp_path, "demo", SkillReviewState(status="blockers", content_hash=skill.content_hash))
    monkeypatch.setattr("ouroboros.config.get_review_enforcement", lambda: "blocking")
    with client(tmp_path, monkeypatch) as http:
        assert http.get("/api/ui/contributions").json()["contributions"] == []
        monkeypatch.setattr("ouroboros.config.get_review_enforcement", lambda: "advisory")
        assert http.get("/api/ui/contributions").json()["contributions"][0]["key"] == "demo:simpler"
        (skill.skill_dir / "plugin.py").write_text("# changed payload")
        assert http.get("/api/ui/contributions").json()["contributions"] == []


def test_capture_must_match_publication_hash_and_keeps_data_entry(tmp_path):
    from ouroboros.extension_contributions import stage_contributions
    from ouroboros.extension_registry_state import _StagedRegistrations

    skill = installed(tmp_path, entries=[CLIENT, TEMPLATE], permissions=["page_module"])
    api = SimpleNamespace(_staged=_StagedRegistrations())
    stage_contributions(api, skill, skill.content_hash)
    assert len(api._staged.contributions) == 2 and api._staged.host_sources
    (skill.skill_dir / "ui/voice.js").write_text("export function mount() { return {}; }")
    api = SimpleNamespace(_staged=_StagedRegistrations())
    stage_contributions(api, skill, skill.content_hash)
    assert [row["id"] for row in api._staged.contributions] == ["simpler"]
    assert api._staged.host_sources == {}
    assert "payload changed" in api._staged.contribution_problems[0]


def test_chunked_speech_upload_has_no_cap_exception():
    from ouroboros.gateway.extensions import _read_child_dispatch_body

    async def read(size):
        chunks = iter([b"x" * (size // 2), b"x" * (size - size // 2)])
        async def receive():
            chunk = next(chunks, b"")
            return {"type": "http.request", "body": chunk, "more_body": bool(chunk)}
        request = Request({"type": "http", "headers": []}, receive)
        return await _read_child_dispatch_body(request)
    assert len(asyncio.run(read(524288))) == 524288
    with pytest.raises(ValueError, match="too large"):
        asyncio.run(read(524289))


def test_live_instruction_provider_is_not_a_disabled_offer(tmp_path, monkeypatch):
    installed(tmp_path, kind="instruction", entries=[STT])
    with client(tmp_path, monkeypatch) as http:
        payload = http.get("/api/ui/contributions").json()
        assert [row["key"] for row in payload["contributions"]] == ["demo:speech"]
        assert payload["offers"]["stt"] == []
        save_enabled(tmp_path, "demo", False)
        payload = http.get("/api/ui/contributions").json()
        assert payload["contributions"] == []
        assert payload["offers"]["stt"][0]["skill"] == "demo"
