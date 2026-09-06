"""Phase 5 regression tests for the extension HTTP surface.

Covers:
- ``GET  /api/extensions``               catalogue snapshot
- ``GET  /api/extensions/<skill>/manifest``
- ``ALL  /api/extensions/<skill>/<rest>`` dispatcher
- ``POST /api/skills/<skill>/toggle``    UI-facing enable/disable

Uses Starlette TestClient so the full request path is exercised.
"""
from __future__ import annotations

import json
import pathlib
import asyncio

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
    presence: str = "",
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
            f"{presence}"
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


@pytest.fixture
def client_env(tmp_path, monkeypatch):
    """Yield ``(client, drive_root)`` and stop lifecycle patches at teardown."""
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        yield client, drive_root
    finally:
        _stop_patches(patches)


def test_api_extensions_index_lists_extension_skills(tmp_path, monkeypatch):
    import ouroboros.gateway.skill_publish as skill_publish_preflight

    def unexpected_scan(*_args, **_kwargs):
        raise AssertionError("passive extension listing ran Betterleaks")

    monkeypatch.setattr(skill_publish_preflight, "scan_named_bytes", unexpected_scan)
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
        assert ext_meta["process"] in {"server", "worker"}
        assert ext_meta["server_reconcile"] in {"", "requested", "request_failed"}
        assert ext_meta["executable_review"] is False
        assert ext_meta["review_gate"]["blocking_reason"] == "review_pending"
        assert ext_meta["submit_hub"]["publication_ready"] is False
        assert ext_meta["submit_hub"]["disabled"] is (
            not ext_meta["submit_hub"]["task_start_allowed"]
        )
    finally:
        _stop_patches(patches)


def test_reviewed_presence_runtime_card_projection_and_owner_cas(tmp_path, monkeypatch):
    from ouroboros.presence_capabilities import load_presence_state
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    skills_root = tmp_path / "skills"
    skill_dir = _write_ext(
        skills_root,
        "presence_ext",
        permissions=[],
        plugin="def register(api):\n    pass\n",
        presence=(
            "presence:\n"
            "  instructions: Participate when useful.\n"
            "  runtime_defaults:\n"
            "    model_slot: light\n"
            "    inline_max_rounds: 10\n"
        ),
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_review_state(
            drive_root,
            "presence_ext",
            SkillReviewState(status="pass", content_hash=content_hash),
        )
        review_path = drive_root / "state" / "skills" / "presence_ext" / "review.json"
        review_before = review_path.read_bytes()

        row = next(
            item for item in client.get("/api/extensions").json()["skills"]
            if item["name"] == "presence_ext"
        )
        runtime = row["presence_runtime"]
        assert runtime["defaults"] == {"model_slot": "light", "inline_max_rounds": 10}
        assert runtime["overrides"] == {"model_slot": None, "inline_max_rounds": None}

        update = client.post(
            "/api/owner/skills/presence_ext/presence-runtime",
            json={
                "expected_state_fingerprint": runtime["state_fingerprint"],
                "runtime_overrides": {"model_slot": "main", "inline_max_rounds": 7},
            },
        )
        assert update.status_code == 200, update.text
        updated_runtime = update.json()["presence_runtime"]
        assert updated_runtime["overrides"] == {"model_slot": "main", "inline_max_rounds": 7}
        assert review_path.read_bytes() == review_before
        assert load_presence_state(drive_root, "presence_ext").runtime_overrides.model_slot == "main"

        conflict = client.post(
            "/api/owner/skills/presence_ext/presence-runtime",
            json={
                "expected_state_fingerprint": runtime["state_fingerprint"],
                "runtime_overrides": {"model_slot": "light", "inline_max_rounds": 5},
            },
        )
        assert conflict.status_code == 409
        assert conflict.json()["code"] == "presence_state_conflict"

        reset = client.post(
            "/api/owner/skills/presence_ext/presence-runtime",
            json={
                "expected_state_fingerprint": updated_runtime["state_fingerprint"],
                "runtime_overrides": {"model_slot": None, "inline_max_rounds": None},
            },
        )
        assert reset.status_code == 200, reset.text
        assert reset.json()["presence_runtime"]["overrides"] == {
            "model_slot": None,
            "inline_max_rounds": None,
        }
        refreshed = next(
            item for item in client.get("/api/extensions").json()["skills"]
            if item["name"] == "presence_ext"
        )
        assert refreshed["review_stale"] is False
    finally:
        _stop_patches(patches)


def test_presence_runtime_controls_require_fresh_executable_review(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    _write_ext(
        skills_root,
        "presence_pending",
        permissions=[],
        plugin="def register(api):\n    pass\n",
        presence="presence:\n  instructions: Participate when useful.\n",
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, _drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        row = next(
            item for item in client.get("/api/extensions").json()["skills"]
            if item["name"] == "presence_pending"
        )
        assert "presence_runtime" not in row
        response = client.post(
            "/api/owner/skills/presence_pending/presence-runtime",
            json={
                "expected_state_fingerprint": "0" * 64,
                "runtime_overrides": {"model_slot": None, "inline_max_rounds": None},
            },
        )
        assert response.status_code == 409
    finally:
        _stop_patches(patches)


def test_api_extensions_index_uses_one_request_local_repo_identity(
    tmp_path,
    monkeypatch,
):
    import ouroboros.gateway.extensions as extensions_api
    import ouroboros.skill_review_runner as review_runner

    drive_root = tmp_path / "drive"
    request_repo = str(tmp_path / "request-skills")
    later_repo = str(tmp_path / "later-skills")
    config_reads: list[str] = []
    reconcile_calls: list[tuple[pathlib.Path, str]] = []
    build_calls: list[tuple[pathlib.Path, str]] = []

    def changing_repo_path() -> str:
        value = request_repo if not config_reads else later_repo
        config_reads.append(value)
        return value

    def reconcile_stale_review_jobs(root, *, repo_path=None, **_kwargs):
        effective_repo_path = changing_repo_path() if repo_path is None else repo_path
        reconcile_calls.append((pathlib.Path(root), effective_repo_path))

    def build_extensions_index(root, repo_path):
        build_calls.append((pathlib.Path(root), repo_path))
        return {"skills": [], "live": {}}

    monkeypatch.setattr(
        "ouroboros.config.get_skills_repo_path",
        changing_repo_path,
    )
    monkeypatch.setattr(
        review_runner,
        "reconcile_stale_review_jobs",
        reconcile_stale_review_jobs,
    )
    monkeypatch.setattr(
        extensions_api,
        "_build_extensions_index",
        build_extensions_index,
    )
    monkeypatch.setattr(
        extensions_api,
        "_request_drive_root",
        lambda _request: drive_root,
    )

    response = asyncio.run(extensions_api.api_extensions_index(object()))

    assert response.status_code == 200
    assert config_reads == [request_repo]
    assert reconcile_calls == [(drive_root, request_repo)]
    assert build_calls == [(drive_root, request_repo)]


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
    assert row["submit_hub"] == {
        "visible": False,
        "publication_ready": False,
        "task_start_allowed": False,
        "disabled": True,
        "state": "hard_block",
        "reason": "",
    }
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


def test_extensions_index_rows_carry_publication_receipt_fields(tmp_path, monkeypatch):
    """Every unique row projects the state-plane OuroborosHub publication
    receipt: absent -> published null / not malformed; valid record -> the
    published object as stored; malformed record -> null + malformed flag."""
    from ouroboros.marketplace.provenance import write_publication_record

    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    _write_ext(
        drive_root / "skills" / "external", "alpha",
        permissions=[], plugin="def register(api):\n    pass\n",
    )
    published = {
        "slug": "alpha",
        "version": "0.1.0",
        "content_hash": "d" * 64,
        "repository": "hub/project",
        "pr_number": 7,
        "pr_url": "https://github.com/hub/project/pull/7",
        "published_at": "2026-08-23T00:00:00+00:00",
    }

    def _row():
        response = client.get("/api/extensions")
        assert response.status_code == 200
        rows = [row for row in response.json()["skills"] if row["name"] == "alpha"]
        assert len(rows) == 1
        return rows[0]

    try:
        row = _row()
        assert row["published"] is None
        assert row["published_malformed"] is False

        write_publication_record(drive_root, "alpha", published)
        row = _row()
        assert row["published"] == published
        assert row["published_malformed"] is False

        record_path = drive_root / "state" / "skills" / "alpha" / "ouroboroshub.json"
        record_path.write_text("not json", encoding="utf-8")
        row = _row()
        assert row["published"] is None
        assert row["published_malformed"] is True
    finally:
        _stop_patches(patches)


def test_extensions_index_collision_rows_omit_publication_receipt_fields(tmp_path, monkeypatch):
    """Collision rows never read per-skill state (existing pin), so the
    publication-receipt fields are absent there rather than null."""
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
    try:
        response = client.get("/api/extensions")
        assert response.status_code == 200
        rows = response.json()["skills"]
        assert len(rows) == 2
        assert all("collision" in row["load_error"].lower() for row in rows)
        for row in rows:
            assert "published" not in row
            assert "published_malformed" not in row
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


def test_api_extensions_index_lists_widget_only_extension_tabs(
    tmp_path, monkeypatch
):
    from ouroboros import extension_health, extension_loader
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
        extension_health.record_extension_health(
            drive_root, loaded.name, status="live", sha="server-sha",
        )
        extension_health.record_extension_health(
            drive_root, loaded.name, status="broken", sha="worker-sha",
            process="worker", server_reconcile="request_failed",
        )

        resp = client.get("/api/extensions")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        entry = next(s for s in data["skills"] if s["name"] == "ext_widget")
        assert entry["live_loaded"] is True
        assert entry["dispatch_live"] is False
        assert entry["health_regressed"] is False
        assert entry["last_known_good"]["sha"] == "server-sha"
        assert entry["health_observations"]["worker"]["status"] == "broken"
        assert entry["health_observations"]["worker"]["server_reconcile"] == "request_failed"
        assert data["live"]["ui_tabs"][0]["key"] == "ext_widget:weather"
        assert data["live"]["ui_tabs"][0]["render"]["kind"] == "declarative"
    finally:
        _stop_patches(patches)


def test_api_skill_toggle_enables_and_loads_extension(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import SkillReviewState, save_review_state
    from ouroboros.skill_loader import compute_content_hash

    skills_root = tmp_path / "skills"
    plugin = (
        "def register(api):\n"
        "    api.register_tool('t', lambda ctx: 'ok', description='', schema={})\n"
    )
    skill_dir = _write_ext(skills_root, "ext_toggle", permissions=["tool"], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    broadcasts = []
    client.app.app.state.broadcast_ws_sync = lambda payload: broadcasts.append(payload)  # type: ignore[attr-defined]
    try:
        # Pre-mark review PASS so enable actually loads.
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_review_state(
            drive_root,
            "ext_toggle",
            SkillReviewState(status="pass", content_hash=content_hash),
        )
        resp = client.post(
            "/api/skills/ext_toggle/toggle",
            json={"enabled": True},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["enabled"] is True
        assert data["extension_action"] == "extension_loaded"
        assert data["process"] in {"server", "worker"}
        assert data["server_reconcile"] in {"", "requested", "request_failed"}
        assert broadcasts[-1]["type"] == "extension_lifecycle"
        assert broadcasts[-1]["skill"] == "ext_toggle"
        assert broadcasts[-1]["action"] == "extension_loaded"
        assert "ext_toggle" in extension_loader.snapshot()["extensions"]

        # Disable → unload.
        resp = client.post(
            "/api/skills/ext_toggle/toggle",
            json={"enabled": False},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["enabled"] is False
        assert data["extension_action"] == "extension_unloaded"
        assert data["process"] in {"server", "worker"}
        assert data["server_reconcile"] in {"", "requested", "request_failed"}
        assert broadcasts[-1]["action"] == "extension_unloaded"
        assert "ext_toggle" not in extension_loader.snapshot()["extensions"]
    finally:
        _stop_patches(patches)


def test_api_skill_toggle_load_error_carries_process_qualified_receipt(
    tmp_path, monkeypatch
):
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    skills_root = tmp_path / "skills"
    skill_dir = _write_ext(
        skills_root,
        "ext_toggle_error",
        permissions=["route"],
        plugin=(
            "def register(api):\n"
            "    api.register_route('/absolute', lambda request: {}, methods=('GET',))\n"
        ),
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        save_review_state(
            drive_root,
            "ext_toggle_error",
            SkillReviewState(
                status="pass",
                content_hash=compute_content_hash(skill_dir, manifest_entry="plugin.py"),
            ),
        )

        resp = client.post(
            "/api/skills/ext_toggle_error/toggle",
            json={"enabled": True},
        )

        assert resp.status_code == 409, resp.text
        data = resp.json()
        assert data["extension_action"] == "extension_load_error"
        assert data["process"] in {"server", "worker"}
        assert data["server_reconcile"] in {"", "requested", "request_failed"}
    finally:
        _stop_patches(patches)


def test_api_projects_conflict_and_refuses_enable_until_peer_is_disabled(
    tmp_path, monkeypatch
):
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        save_enabled,
        save_review_state,
    )

    skills_root = tmp_path / "skills"
    plugin = "def register(api):\n    pass\n"
    telegram_dir = _write_ext(
        skills_root,
        "telegram",
        permissions=[],
        plugin=plugin,
        conflicts=["telegram-bridge"],
    )
    _write_ext(
        skills_root,
        "telegram-bridge",
        permissions=[],
        plugin=plugin,
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        save_review_state(
            drive_root,
            "telegram",
            SkillReviewState(
                status="pass",
                content_hash=compute_content_hash(telegram_dir, manifest_entry="plugin.py"),
            ),
        )
        save_enabled(drive_root, "telegram-bridge", True)

        index = client.get("/api/extensions")
        assert index.status_code == 200, index.text
        row = next(item for item in index.json()["skills"] if item["name"] == "telegram")
        assert row["conflicts"] == ["telegram-bridge"]
        assert row["conflict"] == {
            "code": "skill_conflict",
            "skills": ["telegram-bridge"],
            "omitted": 0,
        }

        blocked = client.post("/api/skills/telegram/toggle", json={"enabled": True})
        assert blocked.status_code == 409, blocked.text
        assert blocked.json()["conflict"] == row["conflict"]

        disabled = client.post(
            "/api/skills/telegram-bridge/toggle",
            json={"enabled": False},
        )
        assert disabled.status_code == 200, disabled.text
        enabled = client.post("/api/skills/telegram/toggle", json={"enabled": True})
        assert enabled.status_code == 200, enabled.text
        assert enabled.json()["enabled"] is True
    finally:
        _stop_patches(patches)


def test_api_skill_delete_removes_external_payload_state_and_unloads(client_env):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    client, drive_root = client_env
    skill_dir = _write_ext(
        drive_root / "skills" / "external",
        "local_delete",
        permissions=["tool"],
        plugin="def register(api):\n    api.register_tool('t', lambda ctx: 'ok', description='', schema={})\n",
    )
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_review_state(drive_root, "local_delete", SkillReviewState(status="pass", content_hash=content_hash))

    enabled = client.post("/api/skills/local_delete/toggle", json={"enabled": True})
    assert enabled.status_code == 200, enabled.text
    assert "local_delete" in extension_loader.snapshot()["extensions"]
    assert (drive_root / "state" / "skills" / "local_delete").is_dir()

    resp = client.post("/api/skills/local_delete/delete", json={"payload_root": "skills/external/local_delete"})

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["ok"] is True
    assert data["deleted_payload_root"] == "skills/external/local_delete"
    assert not skill_dir.exists()
    assert not (drive_root / "state" / "skills" / "local_delete").exists()
    assert "local_delete" not in extension_loader.snapshot()["extensions"]

    hub_skill_dir = _write_ext(
        drive_root / "skills" / "clawhub",
        "hub_delete",
        permissions=[],
        plugin="def register(api):\n    pass\n",
    )
    (hub_skill_dir / ".clawhub.json").write_text("{}", encoding="utf-8")

    resp = client.post("/api/skills/hub_delete/delete", json={"payload_root": "skills/clawhub/hub_delete"})

    assert resp.status_code == 403
    assert hub_skill_dir.exists()


def test_api_skill_delete_rejects_external_symlink_bucket(client_env, tmp_path):
    client, drive_root = client_env
    external_target = tmp_path / "outside-external"
    _write_ext(
        external_target,
        "symlink_delete",
        permissions=[],
        plugin="def register(api):\n    pass\n",
    )
    skills_root = drive_root / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    try:
        (skills_root / "external").symlink_to(external_target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"directory symlinks unavailable in this environment: {exc}")

    resp = client.post(
        "/api/skills/symlink_delete/delete",
        json={"payload_root": "skills/external/symlink_delete"},
    )

    assert resp.status_code == 403
    assert (external_target / "symlink_delete").exists()


def test_api_skill_delete_rejects_name_collision_before_state_delete(client_env):
    client, drive_root = client_env
    external_dir = _write_ext(
        drive_root / "skills" / "external",
        "collide_delete",
        permissions=[],
        plugin="def register(api):\n    pass\n",
    )
    native_dir = _write_ext(
        drive_root / "skills" / "native",
        "collide_delete",
        permissions=[],
        plugin="def register(api):\n    pass\n",
    )
    state_dir = drive_root / "state" / "skills" / "collide_delete"
    state_dir.mkdir(parents=True)
    (state_dir / "enabled.json").write_text('{"enabled": true}', encoding="utf-8")

    resp = client.post(
        "/api/skills/collide_delete/delete",
        json={"payload_root": "skills/external/collide_delete"},
    )

    assert resp.status_code == 409
    assert external_dir.exists()
    assert native_dir.exists()
    assert state_dir.exists()


def test_api_skill_delete_accepts_unsanitized_external_directory_leaf(client_env):
    client, drive_root = client_env
    skill_dir = _write_ext(
        drive_root / "skills" / "external",
        "hello world",
        permissions=[],
        plugin="def register(api):\n    pass\n",
    )
    state_dir = drive_root / "state" / "skills" / "hello_world"
    state_dir.mkdir(parents=True)

    resp = client.post(
        "/api/skills/hello_world/delete",
        json={"payload_root": "skills/external/hello world"},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["deleted_payload_root"] == "skills/external/hello world"
    assert not skill_dir.exists()
    assert not state_dir.exists()


def test_api_skill_toggle_allows_warnings_review(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import SkillReviewState, save_review_state
    from ouroboros.skill_loader import compute_content_hash

    skills_root = tmp_path / "skills"
    plugin = (
        "def register(api):\n"
        "    api.register_tool('t', lambda ctx: 'ok', description='', schema={})\n"
    )
    skill_dir = _write_ext(skills_root, "ext_advisory", permissions=["tool"], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_review_state(
            drive_root,
            "ext_advisory",
            SkillReviewState(status="warnings", content_hash=content_hash),
        )
        resp = client.post("/api/skills/ext_advisory/toggle", json={"enabled": True})

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["enabled"] is True
        assert data["review_status"] == "warnings"
        assert data["extension_action"] == "extension_loaded"
        assert "ext_advisory" in extension_loader.snapshot()["extensions"]
    finally:
        _stop_patches(patches)


def test_api_skill_toggle_allows_warnings_under_blocking(tmp_path, monkeypatch):
    from ouroboros.skill_loader import SkillReviewState, save_review_state, compute_content_hash

    skills_root = tmp_path / "skills"
    plugin = (
        "def register(api):\n"
        "    api.register_tool('t', lambda ctx: 'ok', description='', schema={})\n"
    )
    skill_dir = _write_ext(skills_root, "ext_blocked", permissions=["tool"], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_review_state(
            drive_root,
            "ext_blocked",
            SkillReviewState(status="warnings", content_hash=content_hash),
        )
        resp = client.post("/api/skills/ext_blocked/toggle", json={"enabled": True})

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["executable_review"] is True
        assert data["review_gate"]["blocking_reason"] == "warnings_do_not_block_execution"
    finally:
        _stop_patches(patches)


def test_api_skill_toggle_blocks_missing_isolated_deps_env(tmp_path, monkeypatch):
    from ouroboros.marketplace.install_specs import install_specs_hash
    from ouroboros.marketplace.isolated_deps import DEPS_STATE_FILENAME
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        save_review_state,
        skill_state_dir,
    )

    skills_root = tmp_path / "skills"
    plugin = "def register(api):\n    api.register_tool('t', lambda ctx: 'ok', description='', schema={})\n"
    skill_dir = _write_ext(skills_root, "ext_deps", permissions=["tool"], plugin=plugin)
    manifest = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    (skill_dir / "SKILL.md").write_text(
        manifest.replace(
            "permissions: [\"tool\"]\n",
            "permissions: [\"tool\"]\n"
            "install_specs:\n"
            "  - kind: pip\n"
            "    package: wheel\n",
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_review_state(
            drive_root,
            "ext_deps",
            SkillReviewState(status="pass", content_hash=content_hash),
        )
        state_dir = skill_state_dir(drive_root, "ext_deps")
        state_dir.mkdir(parents=True, exist_ok=True)
        specs = [{"kind": "pip", "package": "wheel"}]
        (state_dir / DEPS_STATE_FILENAME).write_text(
            json.dumps({"status": "installed", "specs_hash": install_specs_hash(specs)}),
            encoding="utf-8",
        )

        resp = client.post("/api/skills/ext_deps/toggle", json={"enabled": True})

        assert resp.status_code == 409, resp.text
        data = resp.json()
        assert data["deps_status"] == "missing"
        assert not (state_dir / "enabled.json").exists()
    finally:
        _stop_patches(patches)


def test_api_skill_toggle_collision_disable_does_not_write_shared_state(
    tmp_path, monkeypatch
):
    skills_root = tmp_path / "skills"
    plugin = "def register(api):\n    return None\n"
    _write_ext(skills_root, "hello world", permissions=[], plugin=plugin)
    _write_ext(skills_root, "hello_world", permissions=[], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        resp = client.post("/api/skills/hello_world/toggle", json={"enabled": False})
        assert resp.status_code == 400, resp.text
        data = resp.json()
        assert data["extension_reason"] == "name_collision"
        state_file = drive_root / "state" / "skills" / "hello_world" / "enabled.json"
        assert not state_file.exists()
    finally:
        _stop_patches(patches)


def test_api_extension_dispatcher_routes_to_registered_handler(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        save_enabled,
        save_review_state,
    )

    skills_root = tmp_path / "skills"
    plugin = (
        "from starlette.responses import JSONResponse\n"
        "def _hello(request):\n"
        "    return JSONResponse({'hello': 'world'})\n"
        "def register(api):\n"
        "    api.register_route('greet', _hello, methods=('GET',))\n"
    )
    skill_dir = _write_ext(skills_root, "ext_route", permissions=["route"], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_enabled(drive_root, "ext_route", True)
        save_review_state(
            drive_root,
            "ext_route",
            SkillReviewState(status="pass", content_hash=content_hash),
        )
        from ouroboros.skill_loader import find_skill
        from ouroboros.config import load_settings
        refreshed = find_skill(drive_root, "ext_route", repo_path=str(skills_root))
        err = extension_loader.load_extension(refreshed, load_settings, drive_root=drive_root)
        assert err is None, err

        resp = client.get("/api/extensions/ext_route/greet")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"hello": "world"}
    finally:
        _stop_patches(patches)


def test_api_extension_module_serves_live_reviewed_js(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        find_skill,
        save_enabled,
        save_review_state,
    )

    skills_root = tmp_path / "skills"
    plugin = (
        "def register(api):\n"
        "    api.register_ui_tab('module', 'Module', render={'kind': 'module', 'entry': 'widget.js'})\n"
    )
    skill_dir = _write_ext(skills_root, "ext_module", permissions=["widget"], plugin=plugin)
    (skill_dir / "widget.js").write_text("window.__ok = true;\n", encoding="utf-8")
    (skill_dir / "other.js").write_text("window.__other = true;\n", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_enabled(drive_root, "ext_module", True)
        save_review_state(
            drive_root,
            "ext_module",
            SkillReviewState(status="pass", content_hash=content_hash),
        )
        loaded = find_skill(drive_root, "ext_module", repo_path=str(skills_root))
        assert loaded is not None
        err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
        assert err is None, err

        ok = client.get("/api/extensions/ext_module/module/widget.js")
        assert ok.status_code == 200, ok.text
        assert "window.__ok" in ok.text
        assert ok.headers["cache-control"] == "no-store"
        assert ok.headers["access-control-allow-origin"] == "*"

        # Q21=A: every reviewed .js of the payload is served, not only the declared entry.
        other = client.get("/api/extensions/ext_module/module/other.js")
        assert other.status_code == 200 and other.text == "window.__other = true;\n"
        assert client.get("/api/extensions/ext_module/module/../widget.js").status_code in {400, 404}
    finally:
        _stop_patches(patches)


def test_api_extension_module_rejects_non_live_extension(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    plugin = (
        "def register(api):\n"
        "    api.register_ui_tab('module', 'Module', render={'kind': 'module', 'entry': 'widget.js'})\n"
    )
    _write_ext(skills_root, "ext_module", permissions=["widget"], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, _, patches = _make_client(tmp_path, monkeypatch)
    try:
        resp = client.get("/api/extensions/ext_module/module/widget.js")
        assert resp.status_code == 409
    finally:
        _stop_patches(patches)


def test_api_extension_settings_section_returns_only_requested_skill(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        find_skill,
        save_enabled,
        save_review_state,
    )

    skills_root = tmp_path / "skills"
    plugin_a = (
        "def register(api):\n"
        "    api.register_settings_section('config', 'Config A', schema={'components': [\n"
        "        {'type': 'markdown', 'text': 'A'}\n"
        "    ]})\n"
    )
    plugin_b = (
        "def register(api):\n"
        "    api.register_settings_section('config', 'Config B', schema={'components': [\n"
        "        {'type': 'markdown', 'text': 'B'}\n"
        "    ]})\n"
    )
    skill_a = _write_ext(skills_root, "settings_a", permissions=["widget"], plugin=plugin_a)
    skill_b = _write_ext(skills_root, "settings_b", permissions=["widget"], plugin=plugin_b)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        for name, skill_dir in {"settings_a": skill_a, "settings_b": skill_b}.items():
            content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
            save_enabled(drive_root, name, True)
            save_review_state(drive_root, name, SkillReviewState(status="pass", content_hash=content_hash))
            loaded = find_skill(drive_root, name, repo_path=str(skills_root))
            assert loaded is not None
            err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
            assert err is None, err

        resp = client.get("/api/extensions/settings_a/settings_section")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["skill"] == "settings_a"
        assert [section["skill"] for section in data["sections"]] == ["settings_a"]
        assert data["sections"][0]["title"] == "Config A"
    finally:
        _stop_patches(patches)


def test_api_extension_dispatcher_allows_head_for_get_route(tmp_path, monkeypatch):
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
        "ext_head",
        permissions=["route"],
        plugin=(
            "from starlette.responses import JSONResponse\n"
            "def _hello(request):\n"
            "    return JSONResponse({'hello': 'world'})\n"
            "def register(api):\n"
            "    api.register_route('greet', _hello, methods=('GET',))\n"
        ),
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_enabled(drive_root, "ext_head", True)
        save_review_state(
            drive_root,
            "ext_head",
            SkillReviewState(status="pass", content_hash=content_hash),
        )
        loaded = find_skill(drive_root, "ext_head", repo_path=str(skills_root))
        assert loaded is not None
        err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
        assert err is None, err

        resp = client.head("/api/extensions/ext_head/greet")
        assert resp.status_code == 200, resp.text
    finally:
        _stop_patches(patches)


def test_api_extension_dispatcher_404_for_unknown_route(tmp_path, monkeypatch):
    client, _, patches = _make_client(tmp_path, monkeypatch)
    try:
        resp = client.get("/api/extensions/nope/xyz")
        assert resp.status_code == 404
    finally:
        _stop_patches(patches)


def test_api_extension_dispatcher_surfaces_lazy_load_error(tmp_path, monkeypatch):
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        save_enabled,
        save_review_state,
    )

    skills_root = tmp_path / "skills"
    plugin = (
        "def _hello(request):\n"
        "    return {'hello': 'world'}\n"
        "def register(api):\n"
        "    api.register_route('/absolute', _hello, methods=('GET',))\n"
    )
    skill_dir = _write_ext(skills_root, "ext_broken", permissions=["route"], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_enabled(drive_root, "ext_broken", True)
        save_review_state(
            drive_root,
            "ext_broken",
            SkillReviewState(status="pass", content_hash=content_hash),
        )

        resp = client.get("/api/extensions/ext_broken/greet")
        assert resp.status_code == 409, resp.text
        data = resp.json()
        assert data["state"]["action"] == "extension_load_error"
        assert data["state"]["reason"] == "load_error"
    finally:
        _stop_patches(patches)


def test_api_extension_dispatcher_rejects_not_live_route(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        save_enabled,
        save_review_state,
    )

    skills_root = tmp_path / "skills"
    plugin = (
        "from starlette.responses import JSONResponse\n"
        "def _hello(request):\n"
        "    return JSONResponse({'hello': 'world'})\n"
        "def register(api):\n"
        "    api.register_route('greet', _hello, methods=('GET',))\n"
    )
    skill_dir = _write_ext(skills_root, "ext_guarded", permissions=["route"], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_enabled(drive_root, "ext_guarded", True)
        save_review_state(
            drive_root,
            "ext_guarded",
            SkillReviewState(status="pass", content_hash=content_hash),
        )
        from ouroboros.skill_loader import find_skill

        loaded = find_skill(drive_root, "ext_guarded", repo_path=str(skills_root))
        assert loaded is not None
        err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
        assert err is None, err
        assert "ext_guarded" in extension_loader.snapshot()["extensions"]

        # Leave stale registrations in memory but mark the skill disabled on disk.
        save_enabled(drive_root, "ext_guarded", False)

        resp = client.get("/api/extensions/ext_guarded/greet")
        assert resp.status_code == 409, resp.text
        data = resp.json()
        assert data["state"]["reason"] == "disabled"
        assert "ext_guarded" not in extension_loader.snapshot()["extensions"]
    finally:
        _stop_patches(patches)


def test_api_extension_dispatcher_reloads_stale_live_route(tmp_path, monkeypatch):
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
        "ext_route_reload",
        permissions=["route"],
        plugin=(
            "from starlette.responses import JSONResponse\n"
            "def _hello(request):\n"
            "    return JSONResponse({'hello': 'v1'})\n"
            "def register(api):\n"
            "    api.register_route('greet', _hello, methods=('GET',))\n"
        ),
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_enabled(drive_root, "ext_route_reload", True)
        save_review_state(
            drive_root,
            "ext_route_reload",
            SkillReviewState(status="pass", content_hash=content_hash),
        )
        loaded = find_skill(drive_root, "ext_route_reload", repo_path=str(skills_root))
        assert loaded is not None
        err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
        assert err is None, err

        (skill_dir / "plugin.py").write_text(
            (
                "from starlette.responses import JSONResponse\n"
                "def _hello(request):\n"
                "    return JSONResponse({'hello': 'v2'})\n"
                "def register(api):\n"
                "    api.register_route('greet', _hello, methods=('GET',))\n"
            ),
            encoding="utf-8",
        )
        refreshed = find_skill(drive_root, "ext_route_reload", repo_path=str(skills_root))
        assert refreshed is not None
        save_review_state(
            drive_root,
            "ext_route_reload",
            SkillReviewState(status="pass", content_hash=refreshed.content_hash),
        )

        resp = client.get("/api/extensions/ext_route_reload/greet")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"hello": "v2"}
    finally:
        _stop_patches(patches)


def test_api_skill_toggle_rejects_non_boolean_enabled(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    plugin = "def register(api):\n    pass\n"
    _write_ext(skills_root, "ext_toggle_bad", permissions=[], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, _, patches = _make_client(tmp_path, monkeypatch)
    try:
        resp = client.post("/api/skills/ext_toggle_bad/toggle", json={"enabled": "definitely"})
        assert resp.status_code == 400
        assert "boolean" in resp.text
    finally:
        _stop_patches(patches)


def test_api_skill_grants_saves_keys_and_permissions(tmp_path, monkeypatch):
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, load_skill_grants, save_review_state

    skills_root = tmp_path / "skills"
    skill_dir = _write_ext(
        skills_root,
        "grant_api",
        permissions=["tool", "read_settings", "inject_chat"],
        plugin="def register(api):\n    pass\n",
        env_from_settings=["OPENROUTER_API_KEY"],
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_review_state(
            drive_root,
            "grant_api",
            SkillReviewState(status="pass", content_hash=content_hash),
        )
        resp = client.post(
            "/api/skills/grant_api/grants",
            json={"items": ["OPENROUTER_API_KEY", "inject_chat"]},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["ok"] is True
        assert data["granted_keys"] == ["OPENROUTER_API_KEY"]
        assert data["granted_permissions"] == ["inject_chat"]
        grants = load_skill_grants(drive_root, "grant_api")
        assert grants["granted_keys"] == ["OPENROUTER_API_KEY"]
        assert grants["granted_permissions"] == ["inject_chat"]
        assert data["extension_reason"] in {"disabled", "not_extension", "name_collision", None}
    finally:
        _stop_patches(patches)


def test_api_skill_grants_soft_fails_extension_reconcile_after_persist(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, load_skill_grants, save_review_state

    skills_root = tmp_path / "skills"
    skill_dir = _write_ext(
        skills_root,
        "grant_reconcile_soft_fail",
        permissions=["inject_chat"],
        plugin="def register(api):\n    pass\n",
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_review_state(
            drive_root,
            "grant_reconcile_soft_fail",
            SkillReviewState(status="pass", content_hash=content_hash),
        )

        def fail_reconcile(*_args, **_kwargs):
            raise RuntimeError("reconcile exploded")

        monkeypatch.setattr(extension_loader, "reconcile_extension", fail_reconcile)
        resp = client.post(
            "/api/skills/grant_reconcile_soft_fail/grants",
            json={"items": ["inject_chat"]},
        )

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["ok"] is True
        assert data["extension_reason"] == "reconcile_call_failed"
        assert "reconcile exploded" in data["load_error"]
        grants = load_skill_grants(drive_root, "grant_reconcile_soft_fail")
        assert grants["granted_permissions"] == ["inject_chat"]
    finally:
        _stop_patches(patches)


def test_api_skill_grants_rejects_blocking_blocker_review(tmp_path, monkeypatch):
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    skills_root = tmp_path / "skills"
    skill_dir = _write_ext(
        skills_root,
        "grant_blocked",
        permissions=["tool", "read_settings"],
        plugin="def register(api):\n    pass\n",
        env_from_settings=["OPENROUTER_API_KEY"],
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_review_state(
            drive_root,
            "grant_blocked",
            SkillReviewState(status="blockers", content_hash=content_hash),
        )
        resp = client.post("/api/skills/grant_blocked/grants", json={"items": ["OPENROUTER_API_KEY"]})
        assert resp.status_code == 409
        assert "fresh executable review" in resp.json()["error"]
    finally:
        _stop_patches(patches)


def test_api_skill_reconcile_clears_cached_load_error(tmp_path, monkeypatch):
    """v5.2.2 dual-track grants: ``POST /api/skills/<name>/reconcile``
    is the loopback endpoint the desktop launcher pings after a
    successful core-key grant. It must clear the server's cached
    ``_load_failures`` entry and re-run ``load_extension`` so the
    plugin picks up the freshly-granted key without forcing the user
    to disable/enable.
    """
    from ouroboros import extension_loader
    from ouroboros.skill_loader import (
        SkillReviewState,
        find_skill,
        save_enabled,
        save_review_state,
        save_skill_grants,
    )

    skills_root = tmp_path / "skills"
    plugin = (
        "def register(api):\n"
        "    api.register_tool('n', lambda ctx: 'ok', description='n', schema={})\n"
    )
    _write_ext(
        skills_root,
        "reconcile_demo",
        permissions=["tool", "read_settings"],
        plugin=plugin,
        env_from_settings=["OPENROUTER_API_KEY"],
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    broadcasts = []
    client.app.app.state.broadcast_ws_sync = lambda payload: broadcasts.append(payload)  # type: ignore[attr-defined]
    try:
        first = find_skill(drive_root, "reconcile_demo", repo_path=str(skills_root))
        assert first is not None
        save_enabled(drive_root, "reconcile_demo", True)
        save_review_state(
            drive_root,
            "reconcile_demo",
            SkillReviewState(status="pass", content_hash=first.content_hash),
        )
        loaded = find_skill(drive_root, "reconcile_demo", repo_path=str(skills_root))
        assert loaded is not None and loaded.enabled

        # First load attempt — no grant on disk → fails with the new
        # informative error and seeds ``_load_failures``.
        err = extension_loader.load_extension(
            loaded, lambda: {"OPENROUTER_API_KEY": "sk-secret"}, drive_root=drive_root,
        )
        assert err is not None
        assert "missing owner grants" in err
        with extension_loader._lock:
            extension_loader._load_failures["reconcile_demo"] = (
                extension_loader._ExtensionLoadFailure(
                    content_hash=loaded.content_hash,
                    skill_dir=str(loaded.skill_dir.resolve()),
                    error=err,
                )
            )

        # Owner grants → simulate the launcher writing grants.json.
        save_skill_grants(
            drive_root,
            "reconcile_demo",
            ["OPENROUTER_API_KEY"],
            content_hash=loaded.content_hash,
            requested_keys=["OPENROUTER_API_KEY"],
        )

        # The endpoint must clear the cached failure and load the plugin.
        resp = client.post("/api/skills/reconcile_demo/reconcile")
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["skill"] == "reconcile_demo"
        assert payload["live_loaded"] is True
        assert payload["extension_action"] == "extension_loaded"
        assert payload["process"] in {"server", "worker"}
        assert payload["server_reconcile"] in {"", "requested", "request_failed"}
        assert broadcasts[-1]["type"] == "extension_lifecycle"
        assert broadcasts[-1]["skill"] == "reconcile_demo"
        assert broadcasts[-1]["action"] == "extension_loaded"
        with extension_loader._lock:
            assert "reconcile_demo" in extension_loader._extensions
            assert "reconcile_demo" not in extension_loader._load_failures
    finally:
        _stop_patches(patches)


def test_api_skill_reconcile_load_error_carries_process_qualified_receipt(
    tmp_path, monkeypatch
):
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        save_enabled,
        save_review_state,
    )

    skills_root = tmp_path / "skills"
    skill_dir = _write_ext(
        skills_root,
        "reconcile_error",
        permissions=["route"],
        plugin=(
            "def register(api):\n"
            "    api.register_route('/absolute', lambda request: {}, methods=('GET',))\n"
        ),
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        save_enabled(drive_root, "reconcile_error", True)
        save_review_state(
            drive_root,
            "reconcile_error",
            SkillReviewState(
                status="pass",
                content_hash=compute_content_hash(skill_dir, manifest_entry="plugin.py"),
            ),
        )

        resp = client.post("/api/skills/reconcile_error/reconcile")

        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["extension_action"] == "extension_load_error"
        assert data["process"] in {"server", "worker"}
        assert data["server_reconcile"] in {"", "requested", "request_failed"}
    finally:
        _stop_patches(patches)


def test_api_skill_reconcile_rejects_missing_skill_name(tmp_path, monkeypatch):
    client, _drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        # Starlette path params with empty trailing segment → 404 path,
        # but explicit empty skill via direct call returns 400 from the
        # endpoint's own validation.
        resp = client.post("/api/skills/ /reconcile")
        # Whitespace-only path param hits the endpoint with stripped
        # empty name → 400.
        assert resp.status_code == 400
    finally:
        _stop_patches(patches)


def test_api_skill_review_offloads_to_thread_and_returns_outcome(tmp_path, monkeypatch):
    """Phase 5 regression: ``POST /api/skills/<skill>/review`` must
    trigger the tri-model review and return the outcome. The async
    Starlette endpoint offloads to ``asyncio.to_thread`` so the event
    loop stays responsive."""
    from unittest.mock import patch

    from ouroboros.skill_review import SkillReviewOutcome

    skills_root = tmp_path / "skills"
    plugin = "def register(api): pass\n"
    _write_ext(skills_root, "ext_r", permissions=[], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        canned = SkillReviewOutcome(
            skill_name="ext_r",
            status="pass",
            findings=[{"item": "manifest_schema", "verdict": "PASS"}],
            reviewer_models=["openai/gpt-5.5"],
            content_hash="abcd",
            error="",
        )
        with patch(
            "ouroboros.gateway.extensions._review_skill_impl",
            create=True,
            return_value=canned,
        ), patch(
            "ouroboros.skill_review.review_skill", return_value=canned,
        ):
            resp = client.post("/api/skills/ext_r/review", json={})
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert data["status"] == "clean"
            assert data["skill"] == "ext_r"
    finally:
        _stop_patches(patches)


def test_lifecycle_queue_endpoint_marks_stale_review_job_interrupted(tmp_path, monkeypatch):
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    job_dir = drive_root / "state" / "skills" / "alpha"
    job_dir.mkdir(parents=True)
    job_path = job_dir / "review_job.json"
    job_path.write_text(
        json.dumps(
            {
                "status": "running",
                "skill": "alpha",
                "content_hash": "abc",
                "job_id": "skill-job-old",
                "started_at": "2026-01-01T00:00:00+00:00",
                "last_heartbeat_at": "2026-01-01T00:00:00+00:00",
                "pid": 123456,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("ouroboros.skill_review_runner._pid_alive", lambda _pid: False)
    try:
        resp = client.get("/api/skills/lifecycle-queue")
        assert resp.status_code == 200
        data = json.loads(job_path.read_text(encoding="utf-8"))
        assert data["status"] == "interrupted"
        assert data["interrupt_reason"] == "owner_process_exited"
        progress = [
            json.loads(line)
            for line in (drive_root / "logs" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert progress[-1]["lifecycle"]["status"] == "interrupted"
        assert progress[-1]["task_id"] == "skill_lifecycle_review_alpha_skill-job-old"
    finally:
        _stop_patches(patches)


def test_ws_endpoint_dispatches_ext_prefixed_messages():
    """Phase 5 regression: gateway.ws::ws_endpoint must route
    provider-safe extension WS messages through ``extension_loader.list_ws_handlers()``.
    AST-level check — the full runtime round-trip requires a live
    supervisor which is out of scope for this file."""
    import ast
    src = (
        pathlib.Path(__file__).resolve().parent.parent
        / "ouroboros"
        / "gateway"
        / "ws.py"
    ).read_text(encoding="utf-8")
    assert "parse_extension_surface_name" in src, "gateway WS module has no extension dispatch branch"
    assert "list_ws_handlers" in src, (
        "gateway WS module does not look up extension WS handlers via "
        "``extension_loader.list_ws_handlers``."
    )
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "ws_endpoint":
            return
    assert False, "ws_endpoint not found in gateway/ws.py"


def test_ws_endpoint_reconciles_and_unloads_not_live_extension(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        find_skill,
        save_enabled,
        save_review_state,
    )

    skills_root = tmp_path / "skills"
    plugin = (
        "async def _handler(payload):\n"
        "    return {'acked': True}\n"
        "def register(api):\n"
        "    api.register_ws_handler('message', _handler)\n"
    )
    skill_dir = _write_ext(skills_root, "ext_ws_guarded", permissions=["ws_handler"], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_enabled(drive_root, "ext_ws_guarded", True)
        save_review_state(
            drive_root,
            "ext_ws_guarded",
            SkillReviewState(status="pass", content_hash=content_hash),
        )
        loaded = find_skill(drive_root, "ext_ws_guarded", repo_path=str(skills_root))
        assert loaded is not None
        err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
        assert err is None, err
        assert "ext_ws_guarded" in extension_loader.snapshot()["extensions"]

        save_enabled(drive_root, "ext_ws_guarded", False)

        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": extension_loader.extension_surface_name("ext_ws_guarded", "message")}))
            reply = json.loads(ws.receive_text())
        assert reply["type"] == "log"
        assert "not live" in reply["data"]["message"]
        assert "ext_ws_guarded" not in extension_loader.snapshot()["extensions"]
    finally:
        _stop_patches(patches)


def test_ws_endpoint_dispatches_first_message_after_lazy_load(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        save_enabled,
        save_review_state,
    )

    skills_root = tmp_path / "skills"
    plugin = (
        "async def _handler(payload):\n"
        "    return {'acked': payload.get('payload')}\n"
        "def register(api):\n"
        "    api.register_ws_handler('message', _handler)\n"
    )
    skill_dir = _write_ext(skills_root, "ext_ws_lazy", permissions=["ws_handler"], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_enabled(drive_root, "ext_ws_lazy", True)
        save_review_state(
            drive_root,
            "ext_ws_lazy",
            SkillReviewState(status="pass", content_hash=content_hash),
        )
        extension_loader.unload_extension("ext_ws_lazy")
        msg_type = extension_loader.extension_surface_name("ext_ws_lazy", "message")
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": msg_type, "payload": "first"}))
            reply = json.loads(ws.receive_text())
        assert reply == {"type": f"{msg_type}.reply", "data": {"acked": "first"}}
    finally:
        _stop_patches(patches)


def test_ws_endpoint_surfaces_extension_load_error(tmp_path, monkeypatch):
    skills_root = tmp_path / "skills"
    skill_dir = _write_ext(
        skills_root,
        "ext_ws_broken",
        permissions=["ws_handler"],
        plugin=(
            "async def _handler(payload):\n"
            "    return {'acked': True}\n"
            "def register(api):\n"
            "    api.register_ws_handler('bad-type', _handler)\n"
        ),
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        from ouroboros import extension_loader
        from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_enabled, save_review_state

        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_enabled(drive_root, "ext_ws_broken", True)
        save_review_state(
            drive_root,
            "ext_ws_broken",
            SkillReviewState(status="pass", content_hash=content_hash),
        )

        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": extension_loader.extension_surface_name("ext_ws_broken", "message")}))
            reply = json.loads(ws.receive_text())
        assert reply["type"] == "log"
        assert "failed to go live" in reply["data"]["message"]
    finally:
        _stop_patches(patches)


def test_tool_registry_execute_dispatches_ext_tool(tmp_path, monkeypatch):
    """Phase 5 regression: ``ToolRegistry.execute`` falls back to
    ``extension_loader.get_tool`` for extension names, but only for
    reviewed/live extensions that are surfaced through the normal
    registry schema lookup."""
    from ouroboros.tools import registry as tools_registry
    from ouroboros import extension_loader
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        find_skill,
        save_enabled,
        save_review_state,
    )

    skills_root = tmp_path / "skills"
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    plugin = (
        "def _echo(ctx, who='world'):\n"
        "    return f'hello {who}'\n"
        "def register(api):\n"
        "    api.register_tool('echo', _echo, description='echo', schema={}, timeout_sec=10)\n"
    )
    skill_dir = _write_ext(skills_root, "testskill", permissions=["tool"], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_enabled(drive_root, "testskill", True)
    save_review_state(
        drive_root,
        "testskill",
        SkillReviewState(status="pass", content_hash=content_hash),
    )
    loaded = find_skill(drive_root, "testskill", repo_path=str(skills_root))
    assert loaded is not None
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=drive_root)
    assert err is None, err
    try:
        tmp_reg = tools_registry.ToolRegistry(repo_dir=tmp_path, drive_root=drive_root)
        tool_name = extension_loader.extension_surface_name("testskill", "echo")
        schema = tmp_reg.get_schema_by_name(tool_name)
        assert schema is not None
        assert schema["function"]["name"] == tool_name
        result = tmp_reg.execute(tool_name, {"who": "phase5"})
        # v5.1.2 iter-2: extension dispatch now goes through
        # ``ouroboros.safety.check_safety``. In test envs without a
        # safety backend, the supervisor returns a visible
        # ``SAFETY_WARNING`` prefix while still letting the call run
        # (fail-open). Assert the handler ran and produced its output;
        # the warning prefix is acceptable.
        assert "hello phase5" in result, result
        # get_timeout honours the extension's declared timeout plus the v5.7.0
        # cleanup buffer used by async handlers (so the outer tool executor
        # does not time out before inner wait_for cancellation can finish).
        assert tmp_reg.get_timeout(tool_name) == 13
    finally:
        extension_loader.unload_extension("testskill")


def test_extensions_index_carries_the_preflight_failed_fact(tmp_path, monkeypatch):
    """#335: /api/extensions is the skill-card feed — a persisted deterministic
    preflight FAIL must reach the card as review_gate.preflight_failed so the
    renderer can offer Repair instead of a Review that fails the same way."""
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    skills_root = tmp_path / "skills"
    plugin = (
        "def register(api):\n"
        "    api.register_tool('t', lambda ctx: 'ok', description='', schema={})\n"
    )
    skill_dir = _write_ext(skills_root, "ext_preflight", permissions=["tool"], plugin=plugin)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    client, drive_root, patches = _make_client(tmp_path, monkeypatch)
    try:
        content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        save_review_state(
            drive_root,
            "ext_preflight",
            SkillReviewState(
                status="pending",
                content_hash=content_hash,
                findings=[{
                    "item": "skill_preflight",
                    "verdict": "FAIL",
                    "severity": "critical",
                    "reason": '{"manifest": [{"item": "manifest_entry_exists", "ok": false}], "ok": false}',
                    "model": "deterministic_preflight",
                }],
            ),
        )
        rows = client.get("/api/extensions").json()["skills"]
        row = next(item for item in rows if item["name"] == "ext_preflight")
        assert row["review_gate"]["preflight_failed"] is True
        assert row["review_gate"]["executable_review"] is False

        # A plain pending skill (no persisted findings) carries the fact
        # honestly as False — the producer KNOWS the findings here.
        save_review_state(
            drive_root, "ext_preflight",
            SkillReviewState(status="pending", content_hash=content_hash),
        )
        rows = client.get("/api/extensions").json()["skills"]
        row = next(item for item in rows if item["name"] == "ext_preflight")
        assert row["review_gate"]["preflight_failed"] is False
    finally:
        _stop_patches(patches)
