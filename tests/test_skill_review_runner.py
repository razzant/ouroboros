from __future__ import annotations

import pathlib
from types import SimpleNamespace

import ouroboros.skill_lifecycle_queue as lifecycle_queue
from ouroboros.skill_loader import (
    SkillReviewState,
    compute_content_hash,
    load_enabled,
    load_review_state,
    load_skill_grants,
    save_review_state,
)
from ouroboros.skill_review import SkillReviewOutcome
from ouroboros.skill_review_runner import _review_result_message, run_skill_review_lifecycle_blocking

from tests._shared import reconcile_receipt


def _reset_queue() -> None:
    lifecycle_queue._events.clear()
    lifecycle_queue._active = None
    lifecycle_queue._lock = None
    lifecycle_queue._dedupe_jobs.clear()


def _build_extension(skills_root: pathlib.Path, name: str) -> pathlib.Path:
    skill_dir = skills_root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            f"name: {name}\n"
            "description: Review runner test.\n"
            "version: 0.1.0\n"
            "type: extension\n"
            "entry: plugin.py\n"
            "permissions: []\n"
            "---\n"
            "body\n"
        ),
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text("def register(api):\n    return None\n", encoding="utf-8")
    return skill_dir


def _build_keyed_extension(skills_root: pathlib.Path, name: str) -> pathlib.Path:
    skill_dir = _build_extension(skills_root, name)
    manifest = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    manifest = manifest.replace("permissions: []\n", "permissions: [read_settings]\nenv_from_settings: [OPENROUTER_API_KEY]\n")
    (skill_dir / "SKILL.md").write_text(manifest, encoding="utf-8")
    return skill_dir


def _mark_self_authored(skill_dir: pathlib.Path) -> None:
    payload = {
        "schema_version": 1,
        "origin": "self_authored",
        "task_id": "task-1",
        "created_at": "2026-05-07T00:00:00+00:00",
    }
    (skill_dir / ".self_authored.json").write_text(
        __import__("json").dumps(payload) + "\n",
        encoding="utf-8",
    )
    state = skill_dir.parents[2] / "state" / "skills" / skill_dir.name
    state.mkdir(parents=True, exist_ok=True)
    (state / "self_authored.json").write_text(__import__("json").dumps(payload) + "\n", encoding="utf-8")


def test_review_lifecycle_rejects_identity_collision_before_state_write(tmp_path):
    _reset_queue()
    drive_root = tmp_path / "drive"
    repo_dir = tmp_path / "repo"
    checkout = tmp_path / "checkout"
    repo_dir.mkdir()
    _build_extension(drive_root / "skills" / "external", "alpha")
    _build_extension(checkout, "alpha")
    ctx = SimpleNamespace(drive_root=drive_root, repo_dir=repo_dir, messages=[])

    def unexpected_review(_ctx, _skill_name):
        raise AssertionError("ambiguous skill reached the review implementation")

    payload = run_skill_review_lifecycle_blocking(
        ctx,
        "alpha",
        source="test",
        review_impl=unexpected_review,
        repo_path=str(checkout),
    )

    assert payload["status"] == "pending"
    assert payload["executable_review"] is False
    assert "collision" in payload["error"].lower()
    assert payload["deps_status"] == "not_run"
    assert not (drive_root / "state" / "skills" / "alpha").exists()


def test_blocking_review_lifecycle_uses_single_progress_card(tmp_path, monkeypatch):
    _reset_queue()
    sent = []
    reconcile_calls = []
    drive_root = tmp_path / "drive"
    repo_dir = tmp_path / "repo"
    skills_root = tmp_path / "skills"
    drive_root.mkdir()
    repo_dir.mkdir()
    skills_root.mkdir()
    skill_dir = _build_extension(skills_root, "alpha")
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    ctx = SimpleNamespace(drive_root=drive_root, repo_dir=repo_dir, messages=[])

    def fake_send(*args, **kwargs):
        sent.append((args, kwargs))

    def fake_review(_ctx, skill_name):
        return SkillReviewOutcome(
            skill_name=skill_name,
            status="pass",
            content_hash=content_hash,
            reviewer_models=["fake/reviewer"],
            findings=[{"item": "manifest_schema", "verdict": "PASS"}],
            error="",
        )

    def fake_reconcile(_ctx, skill_name, **_kwargs):
        reconcile_calls.append(lifecycle_queue.queue_snapshot()["active"]["target"])
        return reconcile_receipt("extension_loaded", "review_passed")

    monkeypatch.setattr("supervisor.message_bus.send_with_budget", fake_send)
    monkeypatch.setattr("ouroboros.skill_review_runner._reconcile_deps_after_pass_review", lambda *_a, **_k: ("installed", ""))
    monkeypatch.setattr("ouroboros.skill_review_runner._reconcile_extension_payload", fake_reconcile)

    payload = run_skill_review_lifecycle_blocking(
        ctx,
        "alpha",
        source="test",
        review_impl=fake_review,
        repo_path=str(skills_root),
    )

    assert payload["status"] == "clean"
    assert payload["deps_status"] == "installed"
    assert payload["extension_action"] == "extension_loaded"
    assert reconcile_calls == ["alpha"]

    progress_messages = [
        args[1]
        for args, kwargs in sent
        if kwargs.get("is_progress")
        and str(kwargs.get("task_id") or "").startswith("skill_lifecycle_review_alpha_")
    ]
    assert any("Running tri-model review" in message for message in progress_messages)
    assert any("Installing dependencies" in message for message in progress_messages)
    assert any("Reloading extension" in message for message in progress_messages)
    assert any("completed" in message and "Review executable (clean): PASS manifest_schema" in message for message in progress_messages)
    assert not any(kwargs.get("task_id") in {"skill_lifecycle_review", "api_skill_review"} for _args, kwargs in sent)


def test_review_lifecycle_installs_deps_after_warnings(tmp_path, monkeypatch):
    _reset_queue()
    deps_calls = []
    drive_root = tmp_path / "drive"
    repo_dir = tmp_path / "repo"
    skills_root = tmp_path / "skills"
    drive_root.mkdir()
    repo_dir.mkdir()
    skills_root.mkdir()
    skill_dir = _build_extension(skills_root, "alpha")
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    ctx = SimpleNamespace(drive_root=drive_root, repo_dir=repo_dir, messages=[])

    def fake_review(_ctx, skill_name):
        return SkillReviewOutcome(
            skill_name=skill_name,
            status="warnings",
            content_hash=content_hash,
            reviewer_models=["fake/reviewer#1", "fake/reviewer#2"],
            findings=[{"item": "error_handling", "verdict": "FAIL", "severity": "advisory"}],
            error="",
        )

    def fake_deps(*_args, **_kwargs):
        deps_calls.append("alpha")
        return "installed", ""

    monkeypatch.setattr("supervisor.message_bus.send_with_budget", lambda *a, **kw: None)
    monkeypatch.setattr("ouroboros.skill_review_runner._reconcile_deps_after_pass_review", fake_deps)
    monkeypatch.setattr("ouroboros.skill_review_runner._reconcile_extension_payload", lambda *_a, **_k: reconcile_receipt())

    payload = run_skill_review_lifecycle_blocking(
        ctx,
        "alpha",
        source="test",
        review_impl=fake_review,
        repo_path=str(skills_root),
    )

    assert payload["status"] == "warnings"
    assert payload["executable_review"] is True
    assert payload["review_gate"]["blocking_reason"] == "warnings_do_not_block_execution"
    assert payload["deps_status"] == "installed"
    assert deps_calls == ["alpha"]


def test_review_result_message_prefers_non_pass_findings_and_marks_omissions(monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    long_reason = "x" * 400
    outcome = SkillReviewOutcome(
        skill_name="alpha",
        status="blockers",
        findings=[
            {"item": "manifest_schema", "verdict": "PASS", "reason": "ok"},
            {"item": "extension_namespace_discipline", "verdict": "FAIL", "reason": long_reason},
        ],
    )

    message = _review_result_message(outcome)

    assert message.startswith("Review blocked: blocker findings (blockers): FAIL extension_namespace_discipline")
    assert "manifest_schema" not in message
    assert "[omitted " in message
    assert "full findings in Skills page" in message


def test_review_result_message_allows_warnings_status(monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    outcome = SkillReviewOutcome(
        skill_name="alpha",
        status="warnings",
        findings=[{"item": "bug_hunting", "verdict": "FAIL", "reason": "soft"}],
    )

    assert _review_result_message(outcome).startswith(
        "Review executable with findings (warnings):"
    )


def test_review_result_message_includes_auto_granted_keys(monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    outcome = SkillReviewOutcome(
        skill_name="alpha",
        status="pass",
        findings=[{"item": "manifest_schema", "verdict": "PASS", "reason": "ok"}],
        auto_granted_keys=["OPENROUTER_API_KEY"],
    )

    assert _review_result_message(outcome).endswith(
        "| auto-granted: OPENROUTER_API_KEY"
    )


def test_review_result_message_includes_auto_granted_permissions(monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    outcome = SkillReviewOutcome(
        skill_name="alpha",
        status="pass",
        findings=[{"item": "manifest_schema", "verdict": "PASS", "reason": "ok"}],
        auto_granted_permissions=["inject_chat"],
    )

    assert _review_result_message(outcome).endswith(
        "| auto-granted: permissions: inject_chat"
    )


def test_self_authored_review_lifecycle_uses_triad(tmp_path, monkeypatch):
    _reset_queue()
    sent = []
    drive_root = tmp_path / "drive"
    repo_dir = tmp_path / "repo"
    skills_root = drive_root / "skills" / "external"
    drive_root.mkdir()
    repo_dir.mkdir()
    skills_root.mkdir(parents=True)
    skill_dir = _build_keyed_extension(skills_root, "alpha")
    _mark_self_authored(skill_dir)
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    ctx = SimpleNamespace(drive_root=drive_root, repo_dir=repo_dir, messages=[])

    monkeypatch.setattr("supervisor.message_bus.send_with_budget", lambda *a, **kw: sent.append((a, kw)))
    monkeypatch.setattr(
        "ouroboros.skill_review_runner.load_settings",
        lambda: {"OPENROUTER_API_KEY": "sk-test"},
    )
    monkeypatch.setattr(
        "ouroboros.skill_review_runner._reconcile_deps_after_pass_review",
        lambda *_a, **_k: ("not_required", ""),
    )
    monkeypatch.setattr(
        "ouroboros.skill_review_runner._reconcile_extension_payload",
        lambda *_a, **_k: reconcile_receipt("extension_loaded", "ready"),
    )

    def fake_review(_ctx, _skill_name):
        outcome = SkillReviewOutcome(
            skill_name="alpha",
            status="pass",
            content_hash=content_hash,
            reviewer_models=["reviewer-a", "reviewer-b", "reviewer-c"],
            findings=[],
        )
        save_review_state(
            drive_root,
            "alpha",
            SkillReviewState(
                status=outcome.status,
                content_hash=outcome.content_hash,
                findings=outcome.findings,
                reviewer_models=outcome.reviewer_models,
            ),
        )
        return outcome

    payload = run_skill_review_lifecycle_blocking(
        ctx,
        "alpha",
        source="test",
        review_impl=fake_review,
        repo_path=str(drive_root / "skills"),
    )

    assert payload["status"] == "clean"
    assert payload["auto_flow"] is False
    assert load_enabled(drive_root, "alpha") is False
    review = load_review_state(drive_root, "alpha")
    assert review.status == "clean"
    assert review.content_hash == content_hash
    assert review.reviewer_models == ["reviewer-a", "reviewer-b", "reviewer-c"]
    grants = load_skill_grants(drive_root, "alpha")
    assert grants["granted_keys"] == []


def test_review_lifecycle_payload_surfaces_auto_flow_grants(tmp_path, monkeypatch):
    _reset_queue()
    drive_root = tmp_path / "drive"
    repo_dir = tmp_path / "repo"
    skills_root = drive_root / "skills" / "external"
    drive_root.mkdir()
    repo_dir.mkdir()
    skills_root.mkdir(parents=True)
    skill_dir = _build_keyed_extension(skills_root, "alpha")
    _mark_self_authored(skill_dir)
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    ctx = SimpleNamespace(drive_root=drive_root, repo_dir=repo_dir, messages=[])

    monkeypatch.setattr("supervisor.message_bus.send_with_budget", lambda *a, **kw: None)
    monkeypatch.setattr(
        "ouroboros.skill_review_runner._reconcile_deps_after_pass_review",
        lambda *_a, **_k: ("not_required", ""),
    )
    monkeypatch.setattr(
        "ouroboros.skill_review_runner._reconcile_extension_payload",
        lambda *_a, **_k: reconcile_receipt("extension_loaded", "ready"),
    )

    def fake_review(_ctx, _skill_name):
        return SkillReviewOutcome(
            skill_name="alpha",
            status="pass",
            content_hash=content_hash,
            reviewer_models=["reviewer"],
            auto_flow=True,
            requested_keys=["OPENROUTER_API_KEY"],
            auto_granted_keys=["OPENROUTER_API_KEY"],
            requested_permissions=["inject_chat"],
            auto_granted_permissions=["inject_chat"],
        )

    payload = run_skill_review_lifecycle_blocking(
        ctx,
        "alpha",
        source="test",
        review_impl=fake_review,
        repo_path=str(drive_root / "skills"),
    )

    assert payload["status"] == "clean"
    assert payload["auto_flow"] is True
    assert payload["requested_keys"] == ["OPENROUTER_API_KEY"]
    assert payload["auto_granted_keys"] == ["OPENROUTER_API_KEY"]
    assert payload["requested_permissions"] == ["inject_chat"]
    assert payload["auto_granted_permissions"] == ["inject_chat"]
    assert load_enabled(drive_root, "alpha") is True


def test_self_authored_review_does_not_enable_when_deps_fail(tmp_path, monkeypatch):
    _reset_queue()
    drive_root = tmp_path / "drive"
    repo_dir = tmp_path / "repo"
    skills_root = drive_root / "skills" / "external"
    drive_root.mkdir()
    repo_dir.mkdir()
    skills_root.mkdir(parents=True)
    skill_dir = _build_extension(skills_root, "alpha")
    _mark_self_authored(skill_dir)
    ctx = SimpleNamespace(drive_root=drive_root, repo_dir=repo_dir, messages=[])

    monkeypatch.setattr("supervisor.message_bus.send_with_budget", lambda *a, **kw: None)
    monkeypatch.setattr("ouroboros.skill_review_runner._reconcile_deps_after_pass_review", lambda *_a, **_k: ("failed", "pip exploded"))

    def fake_review(_ctx, _skill):
        outcome = SkillReviewOutcome(
            skill_name="alpha",
            status="pass",
            content_hash=compute_content_hash(skill_dir, manifest_entry="plugin.py"),
            reviewer_models=["reviewer"],
        )
        outcome.auto_flow = True
        return outcome

    payload = run_skill_review_lifecycle_blocking(
        ctx,
        "alpha",
        source="test",
        review_impl=fake_review,
        repo_path=str(drive_root / "skills"),
    )

    assert payload["status"] == "pending"
    assert payload["deps_status"] == "failed"
    assert "pip exploded" in payload["deps_error"]
    assert load_enabled(drive_root, "alpha") is False


def test_lifecycle_finish_writes_compact_provenance_to_chat_jsonl(tmp_path, monkeypatch):
    import json

    _reset_queue()
    drive_root = tmp_path / "drive"
    repo_dir = tmp_path / "repo"
    skills_root = tmp_path / "skills"
    drive_root.mkdir()
    repo_dir.mkdir()
    skills_root.mkdir()
    skill_dir = _build_extension(skills_root, "alpha")
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    ctx = SimpleNamespace(
        drive_root=drive_root,
        repo_dir=repo_dir,
        messages=[],
        task_id="child-task",
        current_chat_id=17,
        task_metadata={"root_task_id": "root-task"},
    )

    long_reason = (
        "ffmpeg invocation in handler.py:42 spawns a subprocess that exits within "
        "the request scope. Not a long-lived companion process — this finding "
        "should be advisory at most, see CHECKLISTS.md item 11."
    )
    raw_failure = "partial reviewer output that failed JSON parsing"

    def fake_review(_ctx, skill_name):
        return SkillReviewOutcome(
            skill_name=skill_name,
            status="fail",
            content_hash=content_hash,
            reviewer_models=["openai/gpt-5.5", "google/gemini-3.5-flash"],
            findings=[
                {
                    "item": "companion_process_safety",
                    "verdict": "FAIL",
                    "severity": "critical",
                    "reason": long_reason,
                    "model": "openai/gpt-5.5",
                },
                {
                    "item": "companion_process_safety",
                    "verdict": "PASS",
                    "severity": "critical",
                    "reason": "Transient subprocess in handler scope.",
                    "model": "google/gemini-3.5-flash",
                },
            ],
            raw_actor_records=[{
                "model_id": "anthropic/claude-opus-4.6",
                "status": "parse_failure",
                "raw_text": raw_failure,
            }],
            error="",
        )

    monkeypatch.setattr("supervisor.message_bus.send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(
        "ouroboros.skill_review_runner._reconcile_deps_after_pass_review",
        lambda *_a, **_k: ("not_required", ""),
    )
    monkeypatch.setattr(
        "ouroboros.skill_review_runner._reconcile_extension_payload",
        lambda *_a, **_k: reconcile_receipt("noop", "review_failed"),
    )

    run_skill_review_lifecycle_blocking(
        ctx,
        "alpha",
        source="test",
        review_impl=fake_review,
        repo_path=str(skills_root),
    )

    chat_path = drive_root / "logs" / "chat.jsonl"
    assert chat_path.exists(), "Expected chat.jsonl to be created on lifecycle finish"
    lines = [line for line in chat_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    skill_rows = [
        json.loads(line) for line in lines
        if json.loads(line).get("type") == "skill_review"
    ]
    assert len(skill_rows) == 1
    row = skill_rows[0]
    assert row["direction"] == "system"
    assert row["skill"] == "alpha"
    assert row["status"] == "blockers"
    assert row["review_round"] == 1
    assert row["snapshot_attempt"] == 1
    assert row["task_id"] == "child-task"
    assert row["root_task_id"] == "root-task"
    assert row["origin_task_id"] == "child-task"
    assert row["origin_root_task_id"] == "root-task"
    assert row["presentation_owner_task_id"] == "root-task"
    assert row["group_id"] == "task:root-task:alpha"
    assert row["chat_id"] == 17
    assert row["source"] == "test"
    assert "Skill review round 1" in row["text"]
    assert long_reason not in row["text"]
    assert raw_failure not in row["text"]

    history = [
        json.loads(line)
        for line in (drive_root / "state" / "skills" / "alpha" / "review_history.jsonl")
        .read_text(encoding="utf-8").splitlines()
    ]
    assert len(history) == 1
    assert history[0]["job_id"] == row["job_id"]
    assert history[0]["presentation_owner_task_id"] == "root-task"
    assert history[0]["raw_actor_records"][0]["raw_text"] == raw_failure


def test_lifecycle_finish_keeps_raw_only_review_private(tmp_path, monkeypatch):
    import json

    _reset_queue()
    drive_root = tmp_path / "drive"
    repo_dir = tmp_path / "repo"
    skills_root = tmp_path / "skills"
    drive_root.mkdir()
    repo_dir.mkdir()
    skills_root.mkdir()
    skill_dir = _build_extension(skills_root, "alpha")
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    ctx = SimpleNamespace(drive_root=drive_root, repo_dir=repo_dir, messages=[])
    raw_text = "raw reviewer text from a parse failure"

    def fake_review(_ctx, skill_name):
        return SkillReviewOutcome(
            skill_name=skill_name,
            status="pending",
            content_hash=content_hash,
            reviewer_models=["fake/reviewer"],
            findings=[],
            raw_actor_records=[{
                "model_id": "fake/reviewer",
                "status": "parse_failure",
                "raw_text": raw_text,
            }],
            error="quorum failure",
        )

    monkeypatch.setattr("supervisor.message_bus.send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(
        "ouroboros.skill_review_runner._reconcile_extension_payload",
        lambda *_a, **_k: reconcile_receipt("noop", "review_pending"),
    )
    run_skill_review_lifecycle_blocking(
        ctx, "alpha", source="test", review_impl=fake_review, repo_path=str(skills_root),
    )

    rows = [
        json.loads(line)
        for line in (drive_root / "logs" / "chat.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[-1]["type"] == "skill_review"
    assert raw_text not in rows[-1]["text"]
    history = [
        json.loads(line)
        for line in (drive_root / "state" / "skills" / "alpha" / "review_history.jsonl")
        .read_text(encoding="utf-8").splitlines()
    ]
    assert len(history) == 1
    assert history[0]["raw_actor_records"][0]["raw_text"] == raw_text


def test_lifecycle_history_redacts_secret_shaped_reviewer_prose(tmp_path, monkeypatch):
    import json

    _reset_queue()
    drive_root = tmp_path / "drive"
    repo_dir = tmp_path / "repo"
    skills_root = tmp_path / "skills"
    drive_root.mkdir()
    repo_dir.mkdir()
    skills_root.mkdir()
    skill_dir = _build_extension(skills_root, "alpha")
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    candidate = "sk-" + "A1" * 20
    ctx = SimpleNamespace(drive_root=drive_root, repo_dir=repo_dir, messages=[])

    def fake_review(_ctx, skill_name):
        return SkillReviewOutcome(
            skill_name=skill_name,
            status="fail",
            content_hash=content_hash,
            reviewer_models=["fake/reviewer"],
            findings=[{
                "item": "secret_handling",
                "verdict": "FAIL",
                "severity": "critical",
                "reason": f"Remove {candidate} from the fixture.",
                "model": "fake/reviewer",
            }],
            raw_actor_records=[{
                "model_id": "fake/reviewer",
                "status": "ok",
                "raw_text": f"The candidate is {candidate}.",
            }],
        )

    monkeypatch.setattr("supervisor.message_bus.send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(
        "ouroboros.skill_review_runner._reconcile_extension_payload",
        lambda *_a, **_k: reconcile_receipt("noop", "review_failed"),
    )
    run_skill_review_lifecycle_blocking(
        ctx, "alpha", source="test", review_impl=fake_review, repo_path=str(skills_root),
    )

    history_path = drive_root / "state" / "skills" / "alpha" / "review_history.jsonl"
    raw_history = history_path.read_text(encoding="utf-8")
    history = [json.loads(line) for line in raw_history.splitlines()]
    assert candidate not in raw_history
    assert "***REDACTED***" in history[0]["fail_findings"][0]["reason_excerpt"]
    assert "***REDACTED***" in history[0]["raw_actor_records"][0]["raw_text"]


def test_self_authored_review_requires_configured_requested_keys(tmp_path, monkeypatch):
    _reset_queue()
    drive_root = tmp_path / "drive"
    repo_dir = tmp_path / "repo"
    skills_root = drive_root / "skills" / "external"
    drive_root.mkdir()
    repo_dir.mkdir()
    skills_root.mkdir(parents=True)
    skill_dir = _build_keyed_extension(skills_root, "alpha")
    _mark_self_authored(skill_dir)
    ctx = SimpleNamespace(drive_root=drive_root, repo_dir=repo_dir, messages=[])

    monkeypatch.setattr("supervisor.message_bus.send_with_budget", lambda *a, **kw: None)
    monkeypatch.setattr("ouroboros.skill_review_runner.load_settings", lambda: {})

    payload = run_skill_review_lifecycle_blocking(
        ctx,
        "alpha",
        source="test",
        review_impl=lambda _ctx, _skill: SkillReviewOutcome(
            skill_name="alpha",
            status="pass",
            content_hash=compute_content_hash(skill_dir, manifest_entry="plugin.py"),
            reviewer_models=["reviewer"],
        ),
        repo_path=str(drive_root / "skills"),
    )

    assert payload["status"] == "clean"
    assert load_enabled(drive_root, "alpha") is False


def test_review_round_and_snapshot_attempt_are_group_scoped(tmp_path, monkeypatch):
    _reset_queue()
    drive_root = tmp_path / "drive"
    repo_dir = tmp_path / "repo"
    skills_root = tmp_path / "skills"
    drive_root.mkdir()
    repo_dir.mkdir()
    skills_root.mkdir()
    skill_dir = _build_extension(skills_root, "alpha")
    monkeypatch.setattr("supervisor.message_bus.send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(
        "ouroboros.skill_review_runner._reconcile_deps_after_pass_review",
        lambda *_a, **_k: ("not_required", ""),
    )
    monkeypatch.setattr(
        "ouroboros.skill_review_runner._reconcile_extension_payload",
        lambda *_a, **_k: reconcile_receipt("noop", "test"),
    )

    def fake_review(_ctx, skill_name):
        return SkillReviewOutcome(
            skill_name=skill_name,
            status="pass",
            content_hash=compute_content_hash(skill_dir, manifest_entry="plugin.py"),
            reviewer_models=["fake/reviewer"],
            findings=[{"item": "manifest_schema", "verdict": "PASS"}],
        )

    ctx = SimpleNamespace(
        drive_root=drive_root,
        repo_dir=repo_dir,
        messages=[],
        task_id="child-a",
        task_metadata={"root_task_id": "root-a"},
    )
    first = run_skill_review_lifecycle_blocking(
        ctx, "alpha", source="tool", review_impl=fake_review, repo_path=str(skills_root),
    )
    (skill_dir / "plugin.py").write_text("def register(api):\n    return 2\n", encoding="utf-8")
    second = run_skill_review_lifecycle_blocking(
        ctx, "alpha", source="tool", review_impl=fake_review, repo_path=str(skills_root),
    )
    third = run_skill_review_lifecycle_blocking(
        ctx, "alpha", source="tool", review_impl=fake_review, repo_path=str(skills_root),
    )

    other_ctx = SimpleNamespace(
        drive_root=drive_root,
        repo_dir=repo_dir,
        messages=[],
        task_id="child-b",
        task_metadata={"root_task_id": "root-b"},
    )
    other = run_skill_review_lifecycle_blocking(
        other_ctx, "alpha", source="tool", review_impl=fake_review,
        repo_path=str(skills_root),
    )

    assert (first["review_round"], first["snapshot_attempt"], first["snapshot_revised"]) == (1, 1, False)
    assert (second["review_round"], second["snapshot_attempt"], second["snapshot_revised"]) == (2, 1, True)
    assert (third["review_round"], third["snapshot_attempt"], third["snapshot_revised"]) == (3, 2, False)
    assert (other["review_round"], other["snapshot_attempt"], other["snapshot_revised"]) == (1, 1, False)
    assert first["group_id"] == "task:root-a:alpha"
    assert other["group_id"] == "task:root-b:alpha"


def test_review_rebinds_snapshot_hash_after_waiting_for_lifecycle_lock(tmp_path, monkeypatch):
    import json

    import ouroboros.skill_review_runner as review_runner

    _reset_queue()
    drive_root = tmp_path / "drive"
    repo_dir = tmp_path / "repo"
    skills_root = tmp_path / "skills"
    drive_root.mkdir()
    repo_dir.mkdir()
    skills_root.mkdir()
    skill_dir = _build_extension(skills_root, "alpha")
    plugin_path = skill_dir / "plugin.py"
    initial_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    real_skill_content_hash = review_runner._skill_content_hash
    observed_hashes = []

    def mutate_after_initial_hash(*args, **kwargs):
        current_hash = real_skill_content_hash(*args, **kwargs)
        observed_hashes.append(current_hash)
        if len(observed_hashes) == 1:
            plugin_path.write_text("def register(api):\n    return 2\n", encoding="utf-8")
        return current_hash

    monkeypatch.setattr(review_runner, "_skill_content_hash", mutate_after_initial_hash)
    monkeypatch.setattr("supervisor.message_bus.send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(
        "ouroboros.skill_review_runner._reconcile_deps_after_pass_review",
        lambda *_a, **_k: ("not_required", ""),
    )
    monkeypatch.setattr(
        "ouroboros.skill_review_runner._reconcile_extension_payload",
        lambda *_a, **_k: reconcile_receipt("noop", "test"),
    )
    ctx = SimpleNamespace(drive_root=drive_root, repo_dir=repo_dir, messages=[])

    def fake_review(review_ctx, skill_name):
        current_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
        assert review_ctx._skill_review_content_hash == current_hash
        return SkillReviewOutcome(
            skill_name=skill_name,
            status="pass",
            content_hash=current_hash,
            reviewer_models=["fake/reviewer"],
            findings=[{"item": "manifest_schema", "verdict": "PASS"}],
        )

    payload = run_skill_review_lifecycle_blocking(
        ctx,
        "alpha",
        source="skills",
        review_impl=fake_review,
        repo_path=str(skills_root),
    )

    rebound_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    assert initial_hash != rebound_hash
    assert observed_hashes == [initial_hash, rebound_hash]
    assert payload["content_hash"] == rebound_hash
    assert (payload["review_round"], payload["snapshot_attempt"]) == (1, 1)
    job = json.loads(
        (drive_root / "state" / "skills" / "alpha" / "review_job.json").read_text(
            encoding="utf-8",
        )
    )
    assert job["content_hash"] == rebound_hash
    rows = [
        json.loads(line)
        for line in (
            drive_root / "state" / "skills" / "alpha" / "review_history.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert [(row["content_hash"], row["review_round"], row["snapshot_attempt"]) for row in rows] == [
        (rebound_hash, 1, 1),
    ]


def test_legacy_history_ordinals_are_computed_without_rewrite(tmp_path):
    import json

    from ouroboros.skill_review import _load_skill_review_history
    from ouroboros.skill_review_history import review_history_path

    drive_root = tmp_path / "drive"
    path = review_history_path(drive_root, "alpha")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"status": "pending", "content_hash": "hash-a"}) + "\n"
        + json.dumps({"status": "clean", "content_hash": "hash-b"}) + "\n",
        encoding="utf-8",
    )
    before = path.read_bytes()

    history = _load_skill_review_history(drive_root, "alpha", limit=10)

    assert [(row["review_round"], row["snapshot_attempt"]) for row in history] == [(1, 1), (2, 1)]
    assert history[1]["snapshot_revised"] is True
    assert all(row["group_id"] == "manual:alpha" for row in history)
    assert path.read_bytes() == before


def test_started_failure_consumes_one_round_and_one_terminal_row(tmp_path, monkeypatch):
    import json

    _reset_queue()
    drive_root = tmp_path / "drive"
    repo_dir = tmp_path / "repo"
    skills_root = tmp_path / "skills"
    drive_root.mkdir()
    repo_dir.mkdir()
    skills_root.mkdir()
    _build_extension(skills_root, "alpha")
    ctx = SimpleNamespace(drive_root=drive_root, repo_dir=repo_dir, messages=[])
    monkeypatch.setattr("supervisor.message_bus.send_with_budget", lambda *a, **k: None)

    def fail_review(_ctx, _skill_name):
        raise RuntimeError("review infrastructure failed")

    try:
        run_skill_review_lifecycle_blocking(
            ctx, "alpha", source="skills", review_impl=fail_review,
            repo_path=str(skills_root),
        )
    except RuntimeError as exc:
        assert "review infrastructure failed" in str(exc)
    else:  # pragma: no cover - lifecycle must propagate the runner failure
        raise AssertionError("expected lifecycle failure")

    history_path = drive_root / "state" / "skills" / "alpha" / "review_history.jsonl"
    rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["review_round"] == 1
    assert rows[0]["snapshot_attempt"] == 1
    assert rows[0]["job_status"] == "failed"
    assert "RuntimeError" in rows[0]["terminal_reason"]


def test_skill_review_ui_projection_is_group_scoped_bounded_and_sanitized(tmp_path):
    import json

    from ouroboros.skill_review_runner import review_job_state_path, skill_review_ui_projection
    from ouroboros.utils import atomic_write_json

    drive_root = tmp_path / "drive"
    history_path = drive_root / "state" / "skills" / "alpha" / "review_history.jsonl"
    history_path.parent.mkdir(parents=True)
    rows = []
    for idx in range(12):
        rows.append({
            "status": "clean",
            "content_hash": f"hash-{idx}",
            "group_id": "manual:alpha",
            "review_round": idx + 1,
            "snapshot_attempt": 1,
            "job_id": f"job-{idx}",
            "raw_actor_records": [{"raw_text": "private"}],
        })
    rows.append({
        "status": "clean",
        "content_hash": "task-hash",
        "group_id": "task:root:alpha",
        "review_round": 1,
        "snapshot_attempt": 1,
        "job_id": "task-job",
    })
    history_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    atomic_write_json(
        review_job_state_path(drive_root, "alpha"),
        {
            "status": "completed",
            "review_status": "clean",
            "content_hash": "hash-11",
            "group_id": "manual:alpha",
            "review_round": 12,
            "snapshot_attempt": 1,
            "job_id": "job-11",
        },
    )

    projection = skill_review_ui_projection(drive_root, "alpha")

    assert projection["current"]["review_round"] == 12
    assert len(projection["history"]) == 10
    assert projection["history"][0]["review_round"] == 3
    assert all(row["group_id"] == "manual:alpha" for row in projection["history"])
    assert all("raw_actor_records" not in row for row in projection["history"])
    # The ten-row window is a disclosed bound: 12 group rows minus 10 shown.
    # The foreign-group row must not count into the omitted number.
    assert projection["history_omitted"] == 2


def test_cancel_and_timeout_each_write_one_idempotent_terminal_row(tmp_path):
    import asyncio
    import json

    from ouroboros.skill_lifecycle_queue import LifecycleJob
    from ouroboros.skill_review_runner import (
        _mark_review_job_timeout,
        _on_finished,
        _on_started,
        review_job_state_path,
    )

    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    provenance = {
        "group_id": "manual:alpha", "source": "skills", "task_id": "api_skill_review",
        "root_task_id": "", "chat_id": 0,
    }
    started = {}
    cancelled = LifecycleJob(id="cancel-job", kind="review", target="alpha")
    cancelled.status = "running"
    cancelled.started_at = "2026-07-16T00:00:00+00:00"
    _on_started(drive_root, "alpha", "hash-a", started, provenance)(cancelled)
    cancelled.status = "cancelled"
    cancelled.finished_at = "2026-07-16T00:00:01+00:00"
    finish = _on_finished(drive_root, "alpha", "hash-a", started)
    finish(cancelled, None, asyncio.CancelledError())
    finish(cancelled, None, asyncio.CancelledError())

    beta_job = {
        "status": "running", "lifecycle_status": "running", "skill": "beta",
        "content_hash": "hash-b", "job_id": "timeout-job", "group_id": "manual:beta",
        "review_round": 1, "snapshot_attempt": 1, "source": "skills",
    }
    from ouroboros.utils import atomic_write_json
    atomic_write_json(review_job_state_path(drive_root, "beta"), beta_job)
    _mark_review_job_timeout(
        drive_root, "beta", "hash-b", reason="TimeoutError: lifecycle deadline",
    )
    _mark_review_job_timeout(
        drive_root, "beta", "hash-b", reason="TimeoutError: lifecycle deadline",
    )

    for skill, expected_status, expected_job in (
        ("alpha", "cancelled", "cancel-job"),
        ("beta", "timeout", "timeout-job"),
    ):
        path = drive_root / "state" / "skills" / skill / "review_history.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        assert rows[0]["status"] == expected_status
        assert rows[0]["job_id"] == expected_job


def test_success_without_typed_verdict_stays_pending_in_history(tmp_path):
    """Lifecycle success is not a review verdict when the result omits one."""
    import json

    from ouroboros.skill_lifecycle_queue import LifecycleJob
    from ouroboros.skill_review_runner import (
        _on_finished,
        _on_started,
        review_job_state_path,
    )

    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    started = {}
    provenance = {
        "group_id": "manual:alpha", "source": "skills", "task_id": "",
        "root_task_id": "", "chat_id": 0,
    }
    job = LifecycleJob(id="success-job", kind="review", target="alpha")
    job.status = "running"
    _on_started(drive_root, "alpha", "hash-a", started, provenance)(job)
    job.status = "succeeded"
    _on_finished(drive_root, "alpha", "hash-a", started)(job, None, None)

    state = json.loads(review_job_state_path(drive_root, "alpha").read_text())
    assert state["status"] == "completed"
    assert state["review_status"] == "pending"
    history = [
        json.loads(line)
        for line in (drive_root / "state/skills/alpha/review_history.jsonl").read_text().splitlines()
    ]
    assert history[0]["status"] == "pending"
    assert history[0]["job_status"] == "succeeded"


def test_skill_review_response_typedef_carries_qualified_reconcile_fields():
    api_types = (
        pathlib.Path(__file__).resolve().parents[1] / "web" / "modules" / "api_types.js"
    ).read_text(encoding="utf-8")
    review_declaration = api_types.split("@typedef {Object} SkillReviewResponse", 1)[1].split(
        "*/", 1
    )[0]
    grant_declaration = api_types.split("@typedef {Object} SkillGrantResponse", 1)[1].split(
        "*/", 1
    )[0]

    assert "@property {string=} extension_process" in review_declaration
    assert "@property {string=} extension_server_reconcile" in review_declaration
    assert "extension_process" not in grant_declaration
    assert "extension_server_reconcile" not in grant_declaration
