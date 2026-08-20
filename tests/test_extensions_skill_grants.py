"""Grants, reconcile and review over the extension HTTP surface.

Split verbatim out of ``tests/test_extensions_api.py`` by theme. This module owns the
keys and permissions a grant persists, the extension reconcile that may soft-fail
after that persist, the blocking review it refuses, the cached load error a reconcile
clears, the review that offloads to a thread, and the lifecycle queue that marks a
stale review job interrupted.
"""

from __future__ import annotations

import json




from tests._extensions_api_shared import (  # noqa: F401  (autouse fixture applies on import)
    _clean_extensions,
    _make_client,
    _stop_patches,
    _write_ext,
)


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
        assert broadcasts[-1]["type"] == "extension_lifecycle"
        assert broadcasts[-1]["skill"] == "reconcile_demo"
        assert broadcasts[-1]["action"] == "extension_loaded"
        with extension_loader._lock:
            assert "reconcile_demo" in extension_loader._extensions
            assert "reconcile_demo" not in extension_loader._load_failures
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
