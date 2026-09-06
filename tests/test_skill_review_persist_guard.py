from __future__ import annotations

import asyncio
import json
import threading

from ouroboros.skill_loader import (
    SkillReviewState,
    compute_content_hash,
    load_review_state,
    save_review_state,
)
from ouroboros.skill_review import review_skill
from ouroboros.tools.registry import ToolContext
from ouroboros.utils import atomic_write_json, utc_now_iso

from tests._shared import reconcile_receipt


def _build_script_skill(root, name: str = "alpha"):
    skill_dir = root / name
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            f"name: {name}\n"
            "description: Test skill.\n"
            "version: 0.1.0\n"
            "type: script\n"
            "runtime: python3\n"
            "scripts:\n"
            "  - name: run.py\n"
            "    description: Run.\n"
            "---\n"
        ),
        encoding="utf-8",
    )
    (skill_dir / "scripts" / "run.py").write_text("print('ok')\n", encoding="utf-8")
    return skill_dir


def _pass_actor(model: str):
    findings = [
        {"item": "manifest_schema", "verdict": "PASS", "severity": "critical", "reason": "ok"},
        {"item": "permissions_honesty", "verdict": "PASS", "severity": "critical", "reason": "ok"},
        {"item": "no_repo_mutation", "verdict": "PASS", "severity": "critical", "reason": "ok"},
        {"item": "path_confinement", "verdict": "PASS", "severity": "critical", "reason": "ok"},
        {"item": "env_allowlist", "verdict": "PASS", "severity": "critical", "reason": "ok"},
        {"item": "timeout_and_output_discipline", "verdict": "PASS", "severity": "advisory", "reason": "ok"},
        {"item": "extension_namespace_discipline", "verdict": "PASS", "severity": "critical", "reason": "n/a"},
        {"item": "widget_module_safety", "verdict": "PASS", "severity": "critical", "reason": "n/a"},
        {"item": "inject_chat_minimization", "verdict": "PASS", "severity": "critical", "reason": "n/a"},
        {"item": "event_subscription_minimization", "verdict": "PASS", "severity": "critical", "reason": "n/a"},
        {"item": "companion_process_safety", "verdict": "PASS", "severity": "critical", "reason": "n/a"},
        {"item": "host_token_handling", "verdict": "PASS", "severity": "critical", "reason": "n/a"},
        {"item": "error_handling", "verdict": "PASS", "severity": "advisory", "reason": "ok"},
        {"item": "integration_preflight", "verdict": "PASS", "severity": "advisory", "reason": "ok"},
        {"item": "bug_hunting", "verdict": "PASS", "severity": "advisory", "reason": "ok"},
        {"item": "completion_notification", "verdict": "PASS", "severity": "advisory", "reason": "n/a"},
    ]
    return {
        "model": model,
        "request_model": model,
        "provider": "openrouter",
        "verdict": "REVIEW",
        "text": json.dumps(findings),
        "tokens_in": 1,
        "tokens_out": 1,
    }


def test_interrupted_review_job_cannot_late_write_pass(tmp_path, monkeypatch):
    from ouroboros.skill_review_runner import review_job_state_path

    repo_dir = tmp_path / "repo"
    drive_root = tmp_path / "drive"
    skills_root = tmp_path / "skills"
    repo_dir.mkdir()
    drive_root.mkdir()
    skill_dir = _build_script_skill(skills_root)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    ctx = ToolContext(repo_dir=repo_dir, drive_root=drive_root)
    ctx._skill_review_lifecycle_guard = True
    content_hash = compute_content_hash(skill_dir)

    save_review_state(
        drive_root,
        "alpha",
        SkillReviewState(status="fail", content_hash="old-hash", findings=[{"item": "old"}]),
    )
    atomic_write_json(
        review_job_state_path(drive_root, "alpha"),
        {
            "status": "interrupted",
            "skill": "alpha",
            "content_hash": content_hash,
            "job_id": "skill-job-old",
            "last_heartbeat_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
        },
        trailing_newline=True,
    )
    monkeypatch.setattr(
        "ouroboros.tools.review._handle_multi_model_review",
        lambda *_a, **_kw: json.dumps(
            {"results": [_pass_actor("fake/a"), _pass_actor("fake/b")]}
        ),
    )

    outcome = review_skill(ctx, "alpha")

    assert outcome.status == "pending"
    assert "not persisted" in outcome.error
    persisted = load_review_state(drive_root, "alpha")
    assert persisted.status == "blockers"
    assert persisted.content_hash == "old-hash"
    events = (drive_root / "logs" / "events.jsonl").read_text(encoding="utf-8")
    assert "skill_review_persist_skipped" in events


def test_mismatched_review_job_hash_cannot_late_write_pass(tmp_path, monkeypatch):
    from ouroboros.skill_review_runner import review_job_state_path

    repo_dir = tmp_path / "repo"
    drive_root = tmp_path / "drive"
    skills_root = tmp_path / "skills"
    repo_dir.mkdir()
    drive_root.mkdir()
    skill_dir = _build_script_skill(skills_root)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    ctx = ToolContext(repo_dir=repo_dir, drive_root=drive_root)
    ctx._skill_review_lifecycle_guard = True
    compute_content_hash(skill_dir)

    save_review_state(
        drive_root,
        "alpha",
        SkillReviewState(status="fail", content_hash="old-hash", findings=[{"item": "old"}]),
    )
    atomic_write_json(
        review_job_state_path(drive_root, "alpha"),
        {
            "status": "completed",
            "skill": "alpha",
            "content_hash": "newer-content-hash",
            "job_id": "skill-job-newer",
            "last_heartbeat_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
        },
        trailing_newline=True,
    )
    monkeypatch.setattr(
        "ouroboros.tools.review._handle_multi_model_review",
        lambda *_a, **_kw: json.dumps(
            {"results": [_pass_actor("fake/a"), _pass_actor("fake/b")]}
        ),
    )

    outcome = review_skill(ctx, "alpha")

    assert outcome.status == "pending"
    assert "not persisted" in outcome.error
    assert load_review_state(drive_root, "alpha").content_hash == "old-hash"


def test_mismatched_lifecycle_job_id_cannot_late_write_pass(tmp_path, monkeypatch):
    from ouroboros.skill_review_runner import review_job_state_path

    repo_dir = tmp_path / "repo"
    drive_root = tmp_path / "drive"
    skills_root = tmp_path / "skills"
    repo_dir.mkdir()
    drive_root.mkdir()
    skill_dir = _build_script_skill(skills_root)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    ctx = ToolContext(repo_dir=repo_dir, drive_root=drive_root)
    ctx._skill_review_lifecycle_guard = True
    ctx._skill_review_lifecycle_job_id = "skill-job-old"
    content_hash = compute_content_hash(skill_dir)

    save_review_state(
        drive_root,
        "alpha",
        SkillReviewState(status="fail", content_hash="old-hash", findings=[{"item": "old"}]),
    )
    atomic_write_json(
        review_job_state_path(drive_root, "alpha"),
        {
            "status": "running",
            "skill": "alpha",
            "content_hash": content_hash,
            "job_id": "skill-job-new",
            "last_heartbeat_at": utc_now_iso(),
            "finished_at": "",
        },
        trailing_newline=True,
    )
    monkeypatch.setattr(
        "ouroboros.tools.review._handle_multi_model_review",
        lambda *_a, **_kw: json.dumps(
            {"results": [_pass_actor("fake/a"), _pass_actor("fake/b")]}
        ),
    )

    outcome = review_skill(ctx, "alpha")

    assert outcome.status == "pending"
    assert "not persisted" in outcome.error
    assert load_review_state(drive_root, "alpha").content_hash == "old-hash"


def test_old_lifecycle_finish_cannot_overwrite_new_review_job_owner(tmp_path):
    from ouroboros.skill_lifecycle_queue import LifecycleJob
    from ouroboros.skill_review_runner import _on_finished, review_job_state_path

    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    job_path = review_job_state_path(drive_root, "alpha")
    atomic_write_json(
        job_path,
        {
            "status": "running",
            "skill": "alpha",
            "content_hash": "same-hash",
            "job_id": "skill-job-new",
            "last_heartbeat_at": utc_now_iso(),
            "finished_at": "",
        },
        trailing_newline=True,
    )
    old_job = LifecycleJob(
        id="skill-job-old",
        kind="review",
        target="alpha",
        dedupe_key="review:alpha:same-hash",
    )
    old_job.status = "succeeded"
    old_job.started_at = utc_now_iso()
    old_job.finished_at = utc_now_iso()

    _on_finished(drive_root, "alpha", "same-hash", {"value": 0.0})(old_job, None, None)

    data = json.loads(job_path.read_text(encoding="utf-8"))
    assert data["job_id"] == "skill-job-new"
    assert data["status"] == "running"
    events = (drive_root / "logs" / "events.jsonl").read_text(encoding="utf-8")
    assert "skill_review_finish_skipped" in events


def test_lifecycle_finish_preserves_interrupted_same_job(tmp_path):
    from ouroboros.skill_lifecycle_queue import LifecycleJob
    from ouroboros.skill_review import SkillReviewOutcome
    from ouroboros.skill_review_runner import _on_finished, review_job_state_path

    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    job_path = review_job_state_path(drive_root, "alpha")
    atomic_write_json(
        job_path,
        {
            "status": "interrupted",
            "skill": "alpha",
            "content_hash": "same-hash",
            "job_id": "skill-job-old",
            "interrupt_reason": "heartbeat_stale",
            "last_heartbeat_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
        },
        trailing_newline=True,
    )
    old_job = LifecycleJob(id="skill-job-old", kind="review", target="alpha")
    old_job.status = "succeeded"
    old_job.started_at = utc_now_iso()
    old_job.finished_at = utc_now_iso()
    result = SkillReviewOutcome(skill_name="alpha", status="pass", content_hash="same-hash")

    _on_finished(drive_root, "alpha", "same-hash", {"value": 0.0})(old_job, result, None)

    data = json.loads(job_path.read_text(encoding="utf-8"))
    assert data["status"] == "interrupted"
    assert data["interrupt_reason"] == "heartbeat_stale"
    assert "review_status" not in data
    events = (drive_root / "logs" / "events.jsonl").read_text(encoding="utf-8")
    assert "skill_review_finish_skipped" in events


def test_lifecycle_finish_without_start_does_not_create_review_job(tmp_path):
    from ouroboros.skill_lifecycle_queue import LifecycleJob
    from ouroboros.skill_review_runner import _on_finished, review_job_state_path

    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    job_path = review_job_state_path(drive_root, "alpha")
    job = LifecycleJob(id="skill-job-waiting", kind="review", target="alpha")
    job.status = "cancelled"
    job.finished_at = utc_now_iso()

    _on_finished(drive_root, "alpha", "same-hash", {})(job, None, RuntimeError("cancelled"))

    assert not job_path.exists()
    events = (drive_root / "logs" / "events.jsonl").read_text(encoding="utf-8")
    assert "review job never acquired lifecycle file lock" in events


def test_direct_review_ignores_historical_completed_job_hash(tmp_path, monkeypatch):
    from ouroboros.skill_review_runner import review_job_state_path

    repo_dir = tmp_path / "repo"
    drive_root = tmp_path / "drive"
    skills_root = tmp_path / "skills"
    repo_dir.mkdir()
    drive_root.mkdir()
    skill_dir = _build_script_skill(skills_root)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    ctx = ToolContext(repo_dir=repo_dir, drive_root=drive_root)
    content_hash = compute_content_hash(skill_dir)

    atomic_write_json(
        review_job_state_path(drive_root, "alpha"),
        {
            "status": "completed",
            "skill": "alpha",
            "content_hash": "old-marketplace-hash",
            "job_id": "skill-job-old",
            "last_heartbeat_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
        },
        trailing_newline=True,
    )
    monkeypatch.setattr(
        "ouroboros.tools.review._handle_multi_model_review",
        lambda *_a, **_kw: json.dumps(
            {"results": [_pass_actor("fake/a"), _pass_actor("fake/b")]}
        ),
    )

    outcome = review_skill(ctx, "alpha")

    assert outcome.status == "clean"
    assert load_review_state(drive_root, "alpha").content_hash == content_hash


def test_cancellation_during_extension_reconcile_keeps_lifecycle_lane(tmp_path, monkeypatch):
    from ouroboros.skill_review import SkillReviewOutcome
    from ouroboros.skill_review_runner import run_skill_review_lifecycle
    import ouroboros.skill_lifecycle_queue as lifecycle_queue
    import ouroboros.skill_review_runner as runner

    lifecycle_queue._events.clear()
    lifecycle_queue._active = None
    lifecycle_queue._lock = None
    lifecycle_queue._dedupe_jobs.clear()

    repo_dir = tmp_path / "repo"
    drive_root = tmp_path / "drive"
    skills_root = tmp_path / "skills"
    repo_dir.mkdir()
    drive_root.mkdir()
    skill_dir = _build_script_skill(skills_root)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    ctx = ToolContext(repo_dir=repo_dir, drive_root=drive_root)
    content_hash = compute_content_hash(skill_dir)
    reconcile_started = threading.Event()
    release_reconcile = threading.Event()

    def fake_review(_ctx, skill_name):
        return SkillReviewOutcome(
            skill_name=skill_name,
            status="pass",
            content_hash=content_hash,
            reviewer_models=["fake/reviewer"],
            findings=[],
            error="",
        )

    def fake_reconcile(*_args, **_kwargs):
        reconcile_started.set()
        release_reconcile.wait(2)
        return reconcile_receipt("extension_loaded", "ok")

    monkeypatch.setattr(runner, "_reconcile_extension_payload", fake_reconcile)

    async def main():
        task = asyncio.create_task(
            run_skill_review_lifecycle(ctx, "alpha", source="test", review_impl=fake_review)
        )
        assert await asyncio.to_thread(reconcile_started.wait, 2)
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
                options=lifecycle_queue.LifecycleJobOptions(drive_root=drive_root),
            )
        )
        await asyncio.sleep(0.05)
        assert not quick.done()
        release_reconcile.set()
        result = await asyncio.wait_for(task, timeout=2)
        assert result["status"] == "clean"
        assert await asyncio.wait_for(quick, timeout=2) == {"quick": True}
        assert lifecycle_queue.queue_snapshot()["active"] is None

    asyncio.run(main())


def test_heartbeat_continues_during_extension_reconcile(tmp_path, monkeypatch):
    from ouroboros.skill_review import SkillReviewOutcome
    from ouroboros.skill_review_runner import (
        review_job_state_path,
        run_skill_review_lifecycle,
    )
    import ouroboros.skill_lifecycle_queue as lifecycle_queue
    import ouroboros.skill_review_runner as runner

    lifecycle_queue._events.clear()
    lifecycle_queue._active = None
    lifecycle_queue._lock = None
    lifecycle_queue._dedupe_jobs.clear()

    repo_dir = tmp_path / "repo"
    drive_root = tmp_path / "drive"
    skills_root = tmp_path / "skills"
    repo_dir.mkdir()
    drive_root.mkdir()
    skill_dir = _build_script_skill(skills_root)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(skills_root))
    monkeypatch.setattr(runner, "_HEARTBEAT_INTERVAL_SEC", 0.01)
    ctx = ToolContext(repo_dir=repo_dir, drive_root=drive_root)
    content_hash = compute_content_hash(skill_dir)
    reconcile_started = threading.Event()
    release_reconcile = threading.Event()

    def fake_review(_ctx, skill_name):
        return SkillReviewOutcome(
            skill_name=skill_name,
            status="pass",
            content_hash=content_hash,
            reviewer_models=["fake/reviewer"],
            findings=[],
            error="",
        )

    def fake_reconcile(*_args, **_kwargs):
        reconcile_started.set()
        release_reconcile.wait(2)
        return reconcile_receipt("extension_loaded", "ok")

    monkeypatch.setattr(runner, "_reconcile_extension_payload", fake_reconcile)

    async def main():
        task = asyncio.create_task(
            run_skill_review_lifecycle(ctx, "alpha", source="test", review_impl=fake_review)
        )
        assert await asyncio.to_thread(reconcile_started.wait, 2)
        job_path = review_job_state_path(drive_root, "alpha")
        before = json.loads(job_path.read_text(encoding="utf-8"))["last_heartbeat_at"]
        # A bounded wait, not 20 x 10 ms: on windows-latest the heartbeat's atomic replace can
        # lose a few rounds to this very poll holding the file open (sharing violation, logged
        # and retried by the beat), and the clock ticks at ~15.6 ms — 200 ms saw no change.
        for _ in range(60):
            await asyncio.sleep(0.05)
            after = json.loads(job_path.read_text(encoding="utf-8"))["last_heartbeat_at"]
            if after != before:
                break
        assert after != before, "no heartbeat within 3 s while the reconcile blocks"
        release_reconcile.set()
        result = await asyncio.wait_for(task, timeout=2)
        assert result["status"] == "clean"
        final = json.loads(job_path.read_text(encoding="utf-8"))
        assert final["status"] == "completed"

    asyncio.run(main())
