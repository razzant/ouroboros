"""The extension catalogue surface, and the client lifespan it is served through.

Covers ``GET /api/extensions`` and ``GET /api/extensions/<skill>/manifest``: the
catalogue snapshot, the collision rows that skip lifecycle projections and must not
reconcile a stale review job, the runtime load error the manifest prefers, and the
widget-only extension marked ui_pending — plus the TestClient lifespan that binds
``reload_all`` and the settings hot reload to the app-state drive root.

The skill lifecycle, the dispatcher, grants/reconcile/review and the websocket
endpoint were split verbatim into ``tests/test_extensions_skill_lifecycle.py``,
``tests/test_extensions_dispatcher.py``, ``tests/test_extensions_skill_grants.py``
and ``tests/test_extensions_websocket.py``; the fixtures and client builders they all
use live in ``tests/_extensions_api_shared.py``.

Uses Starlette TestClient so the full request path is exercised.
"""
from __future__ import annotations

import json
import pathlib
import asyncio

from tests._extensions_api_shared import (  # noqa: F401  (autouse fixture applies on import)
    _clean_extensions,
    _make_client,
    _stop_patches,
    _write_ext,
)




class _FakeUvicornServer:
    def __init__(self, _config):
        self.should_exit = False

    async def serve(self):
        await asyncio.sleep(0)


def _patch_lifespan_for_drive_root_test(monkeypatch, srv, settings: dict):
    monkeypatch.setattr(srv, "load_settings", lambda: dict(settings))
    monkeypatch.setattr(srv, "save_settings", lambda *_a, **_k: None)
    monkeypatch.setattr(srv, "apply_runtime_provider_defaults", lambda s: (s, False, []))
    monkeypatch.setattr(srv, "_apply_settings_to_env", lambda *_a, **_k: None)
    monkeypatch.setattr(srv, "has_startup_ready_provider", lambda *_a, **_k: False)
    # has_local_routing now lives only in server_runtime (server.py stopped
    # importing it after the provider-check consolidation).
    monkeypatch.setattr(
        "ouroboros.server_runtime.has_local_routing", lambda *_a, **_k: False
    )
    monkeypatch.setattr(srv, "_start_supervisor_if_needed", lambda *_a, **_k: None)
    monkeypatch.setattr(srv.uvicorn, "Server", _FakeUvicornServer)
    monkeypatch.setattr("ouroboros.launcher_bootstrap.ensure_data_skills_seeded", lambda: None)
    monkeypatch.setattr("ouroboros.server_auth.get_configured_network_password", lambda: "")


def test_testclient_lifespan_reload_all_uses_app_state_drive_root(tmp_path, monkeypatch):
    from starlette.testclient import TestClient
    import server as srv
    from ouroboros import extension_loader

    drive_root = tmp_path / "drive"
    repo_root = tmp_path / "skills"
    drive_root.mkdir()
    repo_root.mkdir()
    srv.app.app.state.drive_root = drive_root  # type: ignore[attr-defined]
    srv.app.app.state.repo_dir = tmp_path / "repo"  # type: ignore[attr-defined]
    _patch_lifespan_for_drive_root_test(
        monkeypatch,
        srv,
        {"OUROBOROS_SKILLS_REPO_PATH": str(repo_root), "OUROBOROS_RUNTIME_MODE": "advanced"},
    )
    monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: str(repo_root))
    calls: list[tuple[pathlib.Path, str | None]] = []
    monkeypatch.setattr(
        extension_loader,
        "reload_all",
        lambda root, _reader, *, repo_path=None: calls.append((pathlib.Path(root), repo_path)) or {},
    )

    with TestClient(srv.app):
        pass

    assert calls == [(drive_root, str(repo_root))]


def test_testclient_settings_hot_reload_uses_app_state_drive_root(tmp_path, monkeypatch):
    from starlette.testclient import TestClient
    import server as srv
    from ouroboros import extension_loader

    drive_root = tmp_path / "drive"
    old_repo = tmp_path / "skills-old"
    new_repo = tmp_path / "skills-new"
    drive_root.mkdir()
    old_repo.mkdir()
    new_repo.mkdir()
    srv.app.app.state.drive_root = drive_root  # type: ignore[attr-defined]
    srv.app.app.state.repo_dir = tmp_path / "repo"  # type: ignore[attr-defined]
    settings = {"OUROBOROS_SKILLS_REPO_PATH": str(old_repo), "OUROBOROS_RUNTIME_MODE": "advanced"}
    _patch_lifespan_for_drive_root_test(monkeypatch, srv, settings)
    monkeypatch.setattr("ouroboros.config.get_skills_repo_path", lambda: str(old_repo))
    calls: list[tuple[pathlib.Path, str | None]] = []
    monkeypatch.setattr(
        extension_loader,
        "reload_all",
        lambda root, _reader, *, repo_path=None: calls.append((pathlib.Path(root), repo_path)) or {},
    )

    with TestClient(srv.app) as client:
        response = client.post("/api/settings", json={"OUROBOROS_SKILLS_REPO_PATH": str(new_repo)})

    assert response.status_code == 200, response.text
    assert calls
    assert all(root == drive_root for root, _repo_path in calls)
    assert (drive_root, str(new_repo)) in calls


def test_api_extensions_index_lists_extension_skills(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    plugin = (
        "def register(api):\n"
        "    api.register_tool('t', lambda ctx: 'ok', description='', schema={})\n"
    )
    _write_ext(skills_root, "ext_a", permissions=["tool"], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    broadcasts = []
    client.app.app.state.broadcast_ws_sync = lambda payload: broadcasts.append(payload)  # type: ignore[attr-defined]
    try:
        resp = client.get("/api/extensions")
        assert resp.status_code == 200
        data = resp.json()
        names = {s["name"] for s in data.get("skills", [])}
        assert "ext_a" in names
        assert "live" in data
        ext_meta = next(s for s in data["skills"] if s["name"] == "ext_a")
        assert ext_meta["live_reason"] == "disabled"
        assert ext_meta["executable_review"] is False
        assert ext_meta["review_gate"]["blocking_reason"] == "review_pending"
    finally:
        _stop_patches(patches)


def test_extensions_index_collision_row_skips_lifecycle_projections(
    tmp_path, monkeypatch
):
    import ouroboros.extension_health as extension_health
    import ouroboros.extension_loader as extension_loader
    import ouroboros.gateway.extensions as extensions_api
    import ouroboros.marketplace.provenance as marketplace_provenance
    import ouroboros.skill_review_runner as skill_review_runner
    import ouroboros.tools.github as github_tools
    import supervisor.queue as supervisor_queue
    from ouroboros.contracts.skill_manifest import SkillManifest
    from ouroboros.skill_loader import LoadedSkill

    drive_root = tmp_path / "drive"
    skill_dir = drive_root / "skills" / "clawhub" / "alpha"
    skill_dir.mkdir(parents=True)
    collision = LoadedSkill(
        name="alpha",
        skill_dir=skill_dir,
        manifest=SkillManifest(
            name="alpha",
            description="",
            version="",
            type="extension",
            entry="plugin.py",
        ),
        content_hash="",
        load_error="Skill name collision: clawhub and user_repo",
        source="clawhub",
        identity_collision=True,
    )

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("collision row invoked a lifecycle/runtime projection")

    monkeypatch.setattr(extensions_api, "discover_skills", lambda *_a, **_kw: [collision])
    monkeypatch.setattr(extensions_api, "snapshot", lambda: {})
    monkeypatch.setattr(extensions_api, "_review_fields", _unexpected)
    monkeypatch.setattr(extensions_api, "skill_conflict_status", _unexpected)
    monkeypatch.setattr(extensions_api, "grant_status_for_skill", _unexpected)
    monkeypatch.setattr(extension_loader, "runtime_state_for_loaded_skill", _unexpected)
    monkeypatch.setattr(extension_health, "read_extension_health", _unexpected)
    monkeypatch.setattr(skill_review_runner, "skill_review_ui_projection", _unexpected)
    monkeypatch.setattr(marketplace_provenance, "read_provenance", _unexpected)
    schedule_inputs = []
    monkeypatch.setattr(
        supervisor_queue,
        "sync_skill_schedules",
        lambda skills, **_kwargs: schedule_inputs.append(list(skills)),
    )
    monkeypatch.setattr(github_tools, "github_token_from_env_or_settings", _unexpected)

    payload = extensions_api._build_extensions_index(drive_root, repo_path="")

    assert len(payload["skills"]) == 1
    row = payload["skills"][0]
    assert row["name"] == "alpha"
    assert row["load_error"] == collision.load_error
    assert row["source"] == "clawhub"
    assert row["live_reason"] == "load_error"
    assert row["live_loaded"] is False
    assert row["dispatch_live"] is False
    assert row["conflict"] is None
    assert row["skill_review"] == {}
    assert row["grants"] == {}
    assert schedule_inputs == [[collision]]
    assert not (drive_root / "state").exists()


def test_extensions_index_collision_does_not_reconcile_stale_review_job(
    tmp_path, monkeypatch,
):
    import ouroboros.skill_review_runner as review_runner

    checkout = tmp_path / "user-skills"
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(checkout))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    _write_ext(
        drive_root / "skills" / "external", "alpha",
        permissions=[], plugin="def register(api):\n    pass\n",
    )
    _write_ext(
        checkout, "alpha", permissions=[],
        plugin="def register(api):\n    pass\n",
    )
    job_path = review_runner.review_job_state_path(drive_root, "alpha")
    job_path.write_text(json.dumps({
        "status": "running",
        "skill": "alpha",
        "content_hash": "old-hash",
        "job_id": "stale-alpha",
        "started_at": "2020-01-01T00:00:00+00:00",
        "last_heartbeat_at": "2020-01-01T00:00:00+00:00",
        "pid": 999999,
    }), encoding="utf-8")
    before = job_path.read_bytes()
    monkeypatch.setattr(review_runner, "_pid_alive", lambda _pid: False)

    try:
        response = client.get("/api/extensions")
        assert response.status_code == 200
        assert len(response.json()["skills"]) == 2
        assert all("collision" in row["load_error"].lower() for row in response.json()["skills"])
        assert job_path.read_bytes() == before
        assert not (job_path.parent / "review_history.jsonl").exists()
    finally:
        _stop_patches(patches)


def test_api_extension_manifest_returns_metadata(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    plugin = "def register(api):\n    pass\n"
    _write_ext(skills_root, "ext_b", permissions=[], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        resp = client.get("/api/extensions/ext_b/manifest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "ext_b"
        assert data["manifest"]["type"] == "extension"
        assert data["executable_review"] is False
        assert data["review_gate"]["blocking_reason"] == "review_pending"
    finally:
        _stop_patches(patches)


def test_api_extension_manifest_prefers_runtime_load_error(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        find_skill,
        save_enabled,
        save_review_state,
    )

    skills_root = tmp_path / "skills"
    skill_dir = _write_ext(
        skills_root,
        "ext_manifest_error",
        permissions=["route"],
        plugin=(
            "def _hello(request):\n"
            "    return {'hello': 'world'}\n"
            "def register(api):\n"
            "    api.register_route('/absolute', _hello, methods=('GET',))\n"
        ),
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    broadcasts = []
    client.app.app.state.broadcast_ws_sync = lambda payload: broadcasts.append(payload)  # type: ignore[attr-defined]
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_enabled(drive_root, "ext_manifest_error", True)
        save_review_state(
            drive_root,
            "ext_manifest_error",
            SkillReviewState(status="pass", content_hash=content_hash),
        )
        loaded = find_skill(drive_root, "ext_manifest_error", repo_path=str(skills_root))
        assert loaded is not None
        state = extension_loader.reconcile_extension(
            "ext_manifest_error",
            drive_root,
            lambda: {},
            repo_path=str(skills_root),
            retry_load_error=True,
        )
        assert state["action"] == "extension_load_error"

        resp = client.get("/api/extensions/ext_manifest_error/manifest")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "absolute" in str(data["load_error"])
    finally:
        _stop_patches(patches)


def test_api_extensions_index_marks_widget_only_extensions_as_ui_pending(
    tmp_path, monkeypatch
):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        find_skill,
        save_enabled,
        save_review_state,
    )

    skills_root = tmp_path / "skills"
    skill_dir = _write_ext(
        skills_root,
        "ext_widget",
        permissions=["widget"],
        plugin=(
            "def register(api):\n"
            "    api.register_ui_tab('weather', 'Weather', render={'kind': 'declarative', 'schema_version': 1, 'components': [{'type': 'markdown', 'text': 'ok'}]})\n"
        ),
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_enabled(drive_root, "ext_widget", True)
        save_review_state(
            drive_root,
            "ext_widget",
            SkillReviewState(status="pass", content_hash=content_hash),
        )
        loaded = find_skill(drive_root, "ext_widget", repo_path=str(skills_root))
        assert loaded is not None
        err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
        assert err is None, err

        resp = client.get("/api/extensions")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        entry = next(s for s in data["skills"] if s["name"] == "ext_widget")
        assert entry["live_loaded"] is True
        assert entry["dispatch_live"] is False
        assert entry["ui_tabs_pending"] == []
        assert data["live"]["ui_tabs"][0]["key"] == "ext_widget:weather"
        assert data["live"]["ui_tabs"][0]["render"]["kind"] == "declarative"
        assert data["live"]["ui_tabs_pending"] == []
    finally:
        _stop_patches(patches)
