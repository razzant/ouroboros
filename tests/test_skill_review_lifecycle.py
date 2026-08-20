"""The review job's lifecycle, and the live extension it reconciles when the verdict lands.

Split out of ``tests/test_skill_exec.py`` by theme: the job state and events the review tool
records, the stale job marked interrupted, the dead running job healed by reconciliation,
the cancellation that waits for the review thread, and the extension plugin loaded,
unloaded and reconciled around a review.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import threading
from unittest.mock import patch

from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_enabled, save_review_state
from ouroboros.tools import skill_exec as skill_exec_mod

from tests._skill_exec_shared import (
    _build_skill,
    _make_ctx,
)
from tests._skill_exec_shared import (  # noqa: F401  (autouse fixture applies on import)
    _clean_extension_runtime,
)


def test_review_skill_tool_records_lifecycle_job_state_and_events(tmp_path, monkeypatch):
    from ouroboros.skill_review import SkillReviewOutcome
    import ouroboros.skill_lifecycle_queue as lifecycle_queue

    lifecycle_queue._events.clear()
    lifecycle_queue._active = None
    lifecycle_queue._lock = None
    lifecycle_queue._dedupe_jobs.clear()

    ctx = _make_ctx(tmp_path)
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    skill_dir = _build_skill(skills_root, "alpha")
    content_hash = compute_content_hash(skill_dir)

    monkeypatch.setattr(
        skill_exec_mod,
        "_review_skill_impl",
        lambda _ctx, skill_name: SkillReviewOutcome(
            skill_name=skill_name,
            status="pass",
            content_hash=content_hash,
            reviewer_models=["fake/reviewer"],
            findings=[],
            error="",
        ),
    )

    from ouroboros.skill_review_runner import run_skill_review_lifecycle_blocking
    result = run_skill_review_lifecycle_blocking(
        ctx, "alpha", source="tool",
        review_impl=lambda rc, rn: skill_exec_mod._review_skill_impl(rc, rn),
    )

    assert result["status"] == "clean"
    assert result["deps_status"] == "not_required"
    review_job = json.loads(
        (ctx.drive_root / "state" / "skills" / "alpha" / "review_job.json").read_text(encoding="utf-8")
    )
    assert review_job["status"] == "completed"
    assert review_job["review_status"] == "clean"
    assert review_job["job_id"].startswith("skill-job-")
    lifecycle_event = lifecycle_queue.queue_snapshot()["events"][-1]
    assert lifecycle_event["kind"] == "review"
    assert lifecycle_event["target"] == "alpha"
    events_text = (ctx.drive_root / "logs" / "events.jsonl").read_text(encoding="utf-8")
    assert "skill_review_started" in events_text
    assert "skill_review_completed" in events_text


def test_stale_review_job_is_marked_interrupted(tmp_path, monkeypatch):
    from ouroboros.skill_review_runner import (
        mark_stale_review_job_interrupted,
        review_job_state_path,
    )

    ctx = _make_ctx(tmp_path)
    job_path = review_job_state_path(ctx.drive_root, "alpha")
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

    mark_stale_review_job_interrupted(ctx.drive_root, "alpha", current_content_hash="abc")

    data = json.loads(job_path.read_text(encoding="utf-8"))
    assert data["status"] == "interrupted"
    assert data["interrupt_reason"] == "owner_process_exited"
    events_text = (ctx.drive_root / "logs" / "events.jsonl").read_text(encoding="utf-8")
    assert "skill_review_interrupted" in events_text
    progress = [
        json.loads(line)
        for line in (ctx.drive_root / "logs" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert progress[-1]["task_id"] == "skill_lifecycle_review_alpha_skill-job-old"
    assert progress[-1]["lifecycle"]["status"] == "interrupted"
    assert progress[-1]["lifecycle"]["phase"] == "interrupted"


def test_reconcile_stale_review_jobs_heals_dead_running_job(tmp_path, monkeypatch):
    # The periodic supervisor reconcile (server.py) calls this to heal a worker
    # that died mid-review and left review_job.json at status=running in a
    # headless/no-UI run where boot/extensions-API reconciles never fire.
    from ouroboros.skill_review_runner import (
        reconcile_stale_review_jobs,
        review_job_state_path,
    )

    ctx = _make_ctx(tmp_path)
    job_path = review_job_state_path(ctx.drive_root, "beta")
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(
        json.dumps(
            {
                "status": "running",
                "skill": "beta",
                "content_hash": "h1",
                "job_id": "skill-job-dead",
                "started_at": "2026-01-01T00:00:00+00:00",
                "last_heartbeat_at": "2026-01-01T00:00:00+00:00",
                "pid": 999999,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("ouroboros.skill_review_runner._pid_alive", lambda _pid: False)

    healed = reconcile_stale_review_jobs(ctx.drive_root)

    assert healed == 1
    data = json.loads(job_path.read_text(encoding="utf-8"))
    assert data["status"] == "interrupted"
    assert data["interrupt_reason"] == "owner_process_exited"


def test_async_review_cancellation_waits_for_review_thread(tmp_path, monkeypatch):
    from ouroboros.skill_review import SkillReviewOutcome
    from ouroboros.skill_review_runner import run_skill_review_lifecycle
    import ouroboros.skill_lifecycle_queue as lifecycle_queue

    lifecycle_queue._events.clear()
    lifecycle_queue._active = None
    lifecycle_queue._lock = None
    lifecycle_queue._dedupe_jobs.clear()

    ctx = _make_ctx(tmp_path)
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    skill_dir = _build_skill(skills_root, "alpha")
    content_hash = compute_content_hash(skill_dir)
    started = threading.Event()
    release = threading.Event()

    def fake_review(_ctx, skill_name):
        started.set()
        release.wait(2)
        return SkillReviewOutcome(
            skill_name=skill_name,
            status="pass",
            content_hash=content_hash,
            reviewer_models=["fake/reviewer"],
            findings=[],
            error="",
        )

    async def main():
        task = asyncio.create_task(
            run_skill_review_lifecycle(ctx, "alpha", source="test", review_impl=fake_review)
        )
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0.05)
        active = lifecycle_queue.queue_snapshot()["active"]
        assert active is not None
        assert active["target"] == "alpha"
        quick = asyncio.create_task(
            lifecycle_queue.run_lifecycle_job(
                kind="review",
                target="beta",
                dedupe_key="review:beta:hash",
                runner=lambda: asyncio.sleep(0, result={"quick": True}),
                options=lifecycle_queue.LifecycleJobOptions(drive_root=ctx.drive_root),
            )
        )
        await asyncio.sleep(0.05)
        assert not quick.done()
        release.set()
        result = await asyncio.wait_for(task, timeout=2)
        assert result["status"] == "clean"
        assert await asyncio.wait_for(quick, timeout=2) == {"quick": True}
        assert lifecycle_queue.queue_snapshot()["active"] is None

    asyncio.run(main())


def test_toggle_skill_loads_and_unloads_extension_plugin(tmp_path, monkeypatch):
    """Phase 4 regression: enabling a type=extension skill via
    toggle_skill must actually call extension_loader.load_extension,
    and disabling must call unload_extension — otherwise the extension
    surface is mystery state relative to what the Skills UI says."""
    from ouroboros import extension_loader
    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "ext_live"
    skill_dir.mkdir(parents=True)
    import json as _json
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            "name: ext_live\n"
            "description: Runtime ext.\n"
            "version: 0.1.0\n"
            "type: extension\n"
            "entry: plugin.py\n"
            f"permissions: {_json.dumps(['tool'])}\n"
            "---\n"
            "body\n"
        ),
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text(
        (
            "def _t(ctx): return 'ok'\n"
            "def register(api):\n"
            "    api.register_tool('t', _t, description='', schema={})\n"
        ),
        encoding="utf-8",
    )
    ctx = _make_ctx(tmp_path)
    content_hash = compute_content_hash(
        skill_dir, manifest_entry="plugin.py", manifest_scripts=None
    )
    save_review_state(
        ctx.drive_root,
        "ext_live",
        SkillReviewState(status="pass", content_hash=content_hash),
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))

    # Clean slate.
    extension_loader.unload_extension("ext_live")
    assert "ext_live" not in extension_loader.snapshot()["extensions"]

    # Enable → plugin gets loaded into the runtime registry.
    enable_resp = _json.loads(
        skill_exec_mod._handle_toggle_skill(ctx, skill="ext_live", enabled=True)
    )
    assert enable_resp["extension_action"] == "extension_loaded"
    snap = extension_loader.snapshot()
    assert "ext_live" in snap["extensions"]
    assert extension_loader.extension_surface_name("ext_live", "t") in snap["tools"]

    # Disable → the plugin is torn down.
    disable_resp = _json.loads(
        skill_exec_mod._handle_toggle_skill(ctx, skill="ext_live", enabled=False)
    )
    assert disable_resp["extension_action"] == "extension_unloaded"
    snap = extension_loader.snapshot()
    assert "ext_live" not in snap["extensions"]


def test_review_skill_reconciles_live_extension_after_review(tmp_path, monkeypatch):
    from ouroboros import extension_loader
    from ouroboros.skill_loader import find_skill
    from ouroboros.skill_review import SkillReviewOutcome

    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "ext_reviewed"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            "name: ext_reviewed\n"
            "description: Runtime ext.\n"
            "version: 0.1.0\n"
            "type: extension\n"
            "entry: plugin.py\n"
            "permissions: [\"tool\"]\n"
            "---\n"
            "body\n"
        ),
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text(
        (
            "def _t(ctx): return 'v1'\n"
            "def register(api):\n"
            "    api.register_tool('t', _t, description='', schema={})\n"
        ),
        encoding="utf-8",
    )
    ctx = _make_ctx(tmp_path)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py", manifest_scripts=None)
    save_enabled(ctx.drive_root, "ext_reviewed", True)
    save_review_state(
        ctx.drive_root,
        "ext_reviewed",
        SkillReviewState(status="pass", content_hash=content_hash),
    )
    loaded = find_skill(ctx.drive_root, "ext_reviewed")
    assert loaded is not None
    err = extension_loader.load_extension(loaded, lambda: {}, drive_root=ctx.drive_root)
    assert err is None, err
    tool = extension_loader.get_tool(extension_loader.extension_surface_name("ext_reviewed", "t"))
    assert tool is not None
    assert tool["handler"](None) == "v1"

    (skill_dir / "plugin.py").write_text(
        (
            "def _t(ctx): return 'v2'\n"
            "def register(api):\n"
            "    api.register_tool('t', _t, description='', schema={})\n"
        ),
        encoding="utf-8",
    )

    def _fake_review(ctx_arg, skill_name):
        refreshed = find_skill(pathlib.Path(ctx_arg.drive_root), skill_name)
        assert refreshed is not None
        save_review_state(
            pathlib.Path(ctx_arg.drive_root),
            skill_name,
            SkillReviewState(status="pass", content_hash=refreshed.content_hash),
        )
        return SkillReviewOutcome(
            skill_name=skill_name,
            status="pass",
            findings=[],
            reviewer_models=["fake/reviewer"],
            content_hash=refreshed.content_hash,
            error="",
        )

    from ouroboros.skill_review_runner import run_skill_review_lifecycle_blocking
    with patch.object(skill_exec_mod, "_review_skill_impl", side_effect=_fake_review):
        result = run_skill_review_lifecycle_blocking(
            ctx, "ext_reviewed", source="tool",
            review_impl=lambda rc, rn: skill_exec_mod._review_skill_impl(rc, rn),
        )
    assert result["extension_action"] == "extension_loaded"
    tool = extension_loader.get_tool(extension_loader.extension_surface_name("ext_reviewed", "t"))
    assert tool is not None
    assert tool["handler"](None) == "v2"
