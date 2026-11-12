"""Focused T4.6 builtin ToolResult bridge and native producer contracts."""

from __future__ import annotations



from ouroboros.tools.tool_result import (
    ToolResult,
    _install_tool_result_sidecar,
    _published_tool_result,
    _restore_tool_result_sidecar,
)


def test_generic_builtin_sidecar_none_read_denies_unpublished_and_wrong_context() -> None:
    from types import SimpleNamespace

    stale = ToolResult(
        status="error",
        code="TOOL_ERROR",
        text="stale",
    )
    ctx = SimpleNamespace(_active_builtin_tool_result=stale)
    wrong_ctx = SimpleNamespace(_active_builtin_tool_result=stale)
    sentinel = object()
    token = _install_tool_result_sidecar(ctx, sentinel)
    try:
        assert _published_tool_result(ctx, None) is None
        assert _published_tool_result(wrong_ctx, None) is None
        wrong_sentinel = object()
        assert _published_tool_result(ctx, wrong_sentinel) is wrong_sentinel
    finally:
        _restore_tool_result_sidecar(token)

    assert _published_tool_result(ctx, None) is stale


def test_no_review_verdict_publishes_a_refusal_and_an_owner_stop_publishes_its_own(
    tmp_path,
    monkeypatch,
) -> None:
    """No review verdict publishes a structural refusal: a critical-findings or a scope-only
    outcome returns before Git for the author's decision and leaves the published result alone;
    only the owner's stop publishes its LEGACY_BLOCKED refusal.

    Tip adaptation of the reference pin: this tree's stage cycle reads its
    collaborators through the call-time facade handle _git() and gates with
    _free_cycle_gate/_preflight_and_tests_gate, so the facade is the patch point,
    the aggregate verdict is stubbed per scenario and the handback is captured."""
    from types import SimpleNamespace

    import ouroboros.review_custody as review_custody
    from ouroboros.tools import git as git_facade
    from ouroboros.tools import git_review_cycle as git_tools

    sentinel = object()
    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        _active_builtin_tool_result=sentinel,
        _review_advisory=[],
        emit_progress_fn=lambda *_a, **_k: None,
    )
    fingerprint = {"ok": True, "fingerprint": "f", "binding": {}}
    monkeypatch.setattr(
        git_facade,
        "_stage_candidate_for_review",
        lambda *_args, **_kwargs: (["x.py"], ["x.py"], None),
    )
    monkeypatch.setattr(git_facade, "protected_paths_in", lambda _paths: [])
    monkeypatch.setattr(git_facade, "_current_runtime_mode", lambda: "advanced")
    monkeypatch.setattr(git_facade, "_fingerprint_staged_diff", lambda *_a: fingerprint)
    monkeypatch.setattr(git_facade, "_free_cycle_gate", lambda *_a, **_k: None)
    monkeypatch.setattr(git_facade, "_preflight_and_tests_gate", lambda *_a, **_k: None)
    monkeypatch.setattr(git_facade, "_review_binding_precondition_error", lambda *_a, **_k: "")
    monkeypatch.setattr(git_facade, "_record_commit_attempt", lambda *_a, **_k: None)
    monkeypatch.setattr(git_facade, "_install_paid_dispatch_stamp", lambda *_a, **_k: None)
    monkeypatch.setattr(git_facade, "_reconcile_and_clear_review_roster", lambda *_a, **_k: None)
    monkeypatch.setattr(git_facade, "_review_custody_pending", lambda *_a, **_k: False)
    monkeypatch.setattr(git_facade, "_subject_binding_mismatch_outcome", lambda *_a, **_k: None)
    monkeypatch.setattr(git_facade, "classify_review_block", lambda *_a, **_k: "verdict")
    monkeypatch.setattr(
        git_facade,
        "_finalize_blocked_review",
        lambda *_a, combined_msg, **_k: combined_msg,
    )
    handed_back = []
    monkeypatch.setattr(git_tools, "_return_commit_feedback", lambda *_a, reason="", **_k: (
        handed_back.append(reason) or {"status": "reviewed", "message": "returned before commit"}))

    for message, reason in (("⚠️ REVIEW_BLOCKED: critical findings", "critical_findings"),
                            ("⚠️ SCOPE_REVIEW_BLOCKED: scope only", "scope_blocked")):
        monkeypatch.setattr(git_facade, "_run_parallel_review", lambda *_a, _m=message, _r=reason, **_k: (_m, None, _r, []))
        monkeypatch.setattr(git_facade, "_aggregate_review_verdict",
                            lambda *_a, _m=message, _r=reason, **_k: (True, _m, _r, [], []))
        outcome = git_tools._run_reviewed_stage_cycle(ctx, "fixture", 0.0)
        assert outcome == {"status": "reviewed", "message": "returned before commit"}
        assert handed_back[-1] == reason and ctx._active_builtin_tool_result is sentinel

    monkeypatch.setattr(review_custody, "review_stop_cause", lambda _ctx: "stopped")
    stopped = git_tools._run_reviewed_stage_cycle(ctx, "fixture", 0.0)
    published = ctx._active_builtin_tool_result
    assert stopped["status"] == "blocked" and stopped["block_reason"] == "owner_stopped"
    assert stopped["message"].startswith("⚠️ REVIEW_STOPPED: this work was stopped; this commit was not made.")
    assert isinstance(published, ToolResult) and published.code == "LEGACY_BLOCKED"
    assert published.text == stopped["message"] and len(handed_back) == 2
