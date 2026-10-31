import json
import inspect

import pytest


def test_commit_attempt_does_not_infer_author_finish_from_success():
    from ouroboros.tools import commit_gate

    source = inspect.getsource(commit_gate._record_commit_attempt)
    assert 'author_disposition = _req("author_disposition", None)' in source
    assert 'if author_disposition is None and status == "succeeded"' not in source
    assert 'disposition="accepted"' not in source


def test_author_disposition_is_hash_bound_and_rejects_malformed():
    from ouroboros.review_records import (
        build_author_disposition,
        validate_author_disposition,
    )

    record = build_author_disposition(
        disposition="rejected",
        rationale="The remaining note is outside this task.",
        subject_hash="abc123",
        reviewer_signal="REVISE_PLAN",
        enforcement="advisory",
    )
    assert record["subject_hash"] == "abc123"
    assert validate_author_disposition(record, subject_hash="abc123") == record
    assert validate_author_disposition(record, subject_hash="stale") is None
    with pytest.raises(ValueError, match="rationale"):
        build_author_disposition(
            disposition="accepted", rationale="", subject_hash="abc123",
        )


def test_plan_author_finish_is_projected_without_closing_blocking_gate():
    from ouroboros.task_results import _validated_plan_review_state

    state = {
        "schema_version": 2,
        "series_id": "s",
        "cycles_paid": 1,
        "need_evidence_seen": [],
        "current_attempt": {"fingerprint": "a" * 64, "status": "open", "reason": ""},
        "waves": [{
            "request_fingerprint": "a" * 64,
            "aggregate": "REVISE_PLAN",
            "closed": False,
            "spec": {"goal": "g"},
            "findings": [{"finding_id": "f", "class": "blocking"}],
            "dispositions": [],
            "author_disposition": {
                "disposition": "rejected",
                "rationale": "The reviewer request is out of scope.",
                "subject_hash": "a" * 64,
                "reviewer_signal": "REVISE_PLAN",
                "enforcement": "advisory",
                "recorded_at": "2026-01-01T00:00:00Z",
                "source": "author",
            },
        }],
    }
    loaded = _validated_plan_review_state(state)
    assert loaded["waves"][0]["author_disposition"]["disposition"] == "rejected"
    assert loaded["waves"][0]["closed"] is False
    bad = json.loads(json.dumps(state))
    bad["waves"][0]["author_disposition"]["subject_hash"] = "b" * 64
    with pytest.raises(ValueError, match="author_disposition"):
        _validated_plan_review_state(bad)


def test_skill_review_state_round_trips_current_hash_author_finish(tmp_path):
    from ouroboros.skill_loader import SkillReviewState, load_review_state, save_review_state

    state = SkillReviewState(
        status="blockers",
        content_hash="c" * 64,
        findings=[{"item": "bug_hunting", "verdict": "FAIL", "severity": "advisory"}],
        author_disposition={
            "disposition": "partial",
            "rationale": "The advisory finding is understood and accepted for this revision.",
            "subject_hash": "c" * 64,
            "reviewer_signal": "blockers",
            "enforcement": "advisory",
            "recorded_at": "2026-01-01T00:00:00Z",
            "source": "author",
        },
    )
    save_review_state(tmp_path, "demo", state)
    loaded = load_review_state(tmp_path, "demo")
    assert loaded.author_disposition["subject_hash"] == "c" * 64
    assert loaded.to_dict()["author_disposition"]["disposition"] == "partial"


def test_skill_author_finish_uses_existing_review_without_dispatch_same_hash(
    monkeypatch, tmp_path,
):
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state
    from ouroboros.tool_access_types import ResolvedResourceBinding
    from ouroboros.tools import skill_exec as skill_exec_mod
    from tests.test_skill_exec import _build_skill, _make_ctx

    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(skills_root, "demo")
    ctx = _make_ctx(tmp_path)
    binding = ResolvedResourceBinding(
        profile="self_modification", root="skill_payload", operation="review",
        base_path=skill_dir, target_path=skill_dir, source="test",
        skill_name="demo", state_drive_root=tmp_path,
    )
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    monkeypatch.setattr(skill_exec_mod, "build_resolved_resource_binding", lambda *a, **k: binding)
    monkeypatch.setattr(skill_exec_mod, "_skill_tool_preflight", lambda *a, **k: "")
    prior_hash = compute_content_hash(skill_dir)
    save_review_state(tmp_path, "demo", SkillReviewState(
        status="blockers", content_hash=prior_hash,
        findings=[{"item": "bug_hunting", "verdict": "FAIL", "severity": "critical", "reason": "review finding"}],
        raw_actor_records=[{"slot_id": "s0", "status": "ok"}],
    ))
    monkeypatch.setattr(
        skill_exec_mod, "run_skill_review_lifecycle_blocking",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("author finish dispatched a new panel")),
        raising=False,
    )
    out = skill_exec_mod._handle_review_skill(
        ctx, skill="demo", _resolved_binding=binding,
        author_disposition="rejected", author_rationale="The finding is outside this task.",
    )
    assert "Author finish recorded" in out
    assert "review finding" in out
    loaded = __import__("ouroboros.skill_loader", fromlist=["load_review_state"]).load_review_state(tmp_path, "demo")
    assert loaded.author_disposition["subject_hash"] == prior_hash
    assert loaded.status == "blockers"


def test_skill_author_finish_binds_changed_hash_after_preflight_without_panel(
    monkeypatch, tmp_path,
):
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, load_review_state, save_review_state
    from ouroboros.tool_access_types import ResolvedResourceBinding
    from ouroboros.tools import skill_exec as skill_exec_mod
    from tests.test_skill_exec import _build_skill, _make_ctx

    skills_root = tmp_path / "skills"
    skill_dir = _build_skill(skills_root, "demo", script_body="print('old')\n")
    ctx = _make_ctx(tmp_path)
    binding = ResolvedResourceBinding(
        profile="self_modification", root="skill_payload", operation="review",
        base_path=skill_dir, target_path=skill_dir, source="test",
        skill_name="demo", state_drive_root=tmp_path,
    )
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    monkeypatch.setattr(skill_exec_mod, "build_resolved_resource_binding", lambda *a, **k: binding)
    monkeypatch.setattr(skill_exec_mod, "_skill_tool_preflight", lambda *a, **k: "")
    old_hash = compute_content_hash(skill_dir)
    save_review_state(tmp_path, "demo", SkillReviewState(
        status="blockers", content_hash=old_hash,
        findings=[{"item": "bug_hunting", "verdict": "FAIL", "severity": "critical", "reason": "old finding"}],
        raw_actor_records=[{"slot_id": "s0", "status": "ok"}],
    ))
    (skill_dir / "scripts" / "hello.py").write_text("print('fixed')\n", encoding="utf-8")
    monkeypatch.setattr(
        skill_exec_mod, "run_skill_review_lifecycle_blocking",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("changed-hash finish dispatched a panel")),
        raising=False,
    )
    out = skill_exec_mod._handle_review_skill(
        ctx, skill="demo", _resolved_binding=binding,
        author_disposition="partial", author_rationale="The fix addresses the actionable part.",
    )
    current_hash = compute_content_hash(skill_dir)
    loaded = load_review_state(tmp_path, "demo")
    assert current_hash != old_hash
    assert loaded.content_hash == old_hash
    assert loaded.gate_for(current_hash)["reviewed_content_hash"] == old_hash
    assert loaded.is_stale_for(current_hash)
    assert loaded.gate_for(current_hash)["executable_review"]
    assert loaded.gate_for(current_hash)["author_accepted"] is True
    assert loaded.gate_for(current_hash, enforcement="blocking")["author_accepted"] is False
    assert loaded.gate_for(current_hash, enforcement="blocking")["executable_review"] is False
    assert loaded.author_disposition["subject_hash"] == current_hash
    assert loaded.status == "blockers"
    assert "raw reviewer findings" in out


def test_ordinary_advisory_commit_does_not_invent_author_finish(tmp_path):
    from types import SimpleNamespace
    from ouroboros.review_state import load_state
    from ouroboros.tools.commit_gate import _record_commit_attempt

    ctx = SimpleNamespace(
        drive_root=tmp_path,
        repo_dir=tmp_path,
        task_id="",
        _review_advisory=["advisory finding"],
        _current_review_attempt_number=0,
    )
    _record_commit_attempt(ctx, commit_message="ordinary advisory", status="succeeded")
    attempts = load_state(tmp_path).attempts
    assert attempts
    assert attempts[-1].author_disposition == {}


@pytest.mark.parametrize("status", ["clean", "warnings"])
@pytest.mark.parametrize("kind", ["script", "extension"])
def test_repeated_skill_finish_preserves_critic_and_independent_blocking(tmp_path, monkeypatch, status, kind):
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, load_skill, save_enabled, save_review_state
    from ouroboros.tool_access_types import ResolvedResourceBinding
    from ouroboros.tools import skill_exec
    from ouroboros.gateway.extensions import _review_fields
    from ouroboros.extension_liveness import _extension_runtime_state
    from tests.test_skill_exec import _build_skill, _make_ctx

    ctx = _make_ctx(tmp_path)
    skills = ctx.drive_root / "skills" / "external"
    manifest = None if kind == "script" else (
        "---\nname: demo\ndescription: Test extension.\nversion: 0.1.0\ntype: extension\nentry: plugin.py\nplugin_api_version: '2.0'\n---\nTest.\n"
    )
    directory = _build_skill(skills, "demo", manifest=manifest)
    changed = directory / "scripts/hello.py" if kind == "script" else directory / "plugin.py"
    if kind == "extension":
        changed.write_text("def register(api):\n    pass\n")
    original_hash = compute_content_hash(directory)
    findings = [] if status == "clean" else [{"item": "bug_hunting", "verdict": "FAIL", "severity": "advisory", "reason": "Original note"}]
    save_review_state(ctx.drive_root, "demo", SkillReviewState(
        status=status, content_hash=original_hash, findings=findings,
        raw_actor_records=[{"slot_id": "critic", "status": "ok"}],
    ))
    save_enabled(ctx.drive_root, "demo", True)
    binding = ResolvedResourceBinding(profile="self_modification", root="skill_payload", operation="review",
        base_path=directory, target_path=directory, source="test", skill_name="demo", state_drive_root=ctx.drive_root)
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    for revision in (1, 2):
        changed.write_text(f"print('author-{revision}')\n" if kind == "script" else f"def register(api):\n    value = {revision}\n")
        result = skill_exec._author_finish_existing_skill_review(ctx, binding, "demo",
            disposition="partial", rationale=f"Accepted revision {revision} after correcting the issue.")
        assert "error" not in result, result
        loaded = load_skill(directory, ctx.drive_root)
        assert loaded.review.content_hash == original_hash
        assert loaded.review.findings == findings
        assert loaded.review.is_stale_for(loaded.content_hash)
        assert loaded.review.gate_for(loaded.content_hash)["executable_review"]
        projected = _review_fields(loaded, github_token_configured=False)
        assert projected["review_stale"] and projected["executable_review"]
        assert projected["reviewed_content_hash"] == original_hash
        if kind == "script":
            actual = json.loads(skill_exec._handle_skill_exec(ctx, skill="demo", script="hello.py"))
            assert actual["exit_code"] == 0
            assert f"author-{revision}" in actual["stdout"]
        else:
            assert _extension_runtime_state(loaded, drive_root=ctx.drive_root, skills=[loaded])["desired_live"]
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    loaded = load_skill(directory, ctx.drive_root)
    assert not loaded.review.gate_for(loaded.content_hash)["executable_review"]
    if kind == "script":
        assert "SKILL_EXEC_BLOCKED" in skill_exec._handle_skill_exec(ctx, skill="demo", script="hello.py")
    else:
        assert not _extension_runtime_state(loaded, drive_root=ctx.drive_root, skills=[loaded])["desired_live"]
    save_review_state(ctx.drive_root, "demo", SkillReviewState(status="clean", content_hash=loaded.content_hash))
    refreshed = load_skill(directory, ctx.drive_root)
    assert refreshed.review.gate_for(refreshed.content_hash)["executable_review"]
