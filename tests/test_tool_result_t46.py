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


def test_review_cycle_publishes_only_structural_critical_finding_rejection(
    tmp_path,
    monkeypatch,
) -> None:
    """Only a critical-findings VERDICT publishes REVIEW_BLOCKED; other blocks do not.

    Tip adaptation of the reference pin: this tree's stage cycle reads its
    collaborators through the call-time facade handle _git() and gates with
    _free_cycle_gate/_preflight_and_tests_gate (the reference's
    _check_advisory_freshness/advisory_gate_unavailable/_refuse_capped_attempt
    spellings do not exist here), so the facade is the patch point and the
    aggregate verdict is stubbed per scenario. The durable fact is unchanged."""
    from types import SimpleNamespace

    from ouroboros.tools import git as git_facade
    from ouroboros.tools import git_review_cycle as git_tools

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
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
        "_run_parallel_review",
        lambda *_a, **_k: ("⚠️ REVIEW_BLOCKED: critical findings", None, "critical_findings", []),
    )
    monkeypatch.setattr(
        git_facade,
        "_aggregate_review_verdict",
        lambda *_a, **_k: (
            True,
            "⚠️ REVIEW_BLOCKED: critical findings",
            "critical_findings",
            [],
            [],
        ),
    )
    monkeypatch.setattr(
        git_facade,
        "_finalize_blocked_review",
        lambda *_a, combined_msg, **_k: combined_msg,
    )

    outcome = git_tools._run_reviewed_stage_cycle(ctx, "fixture", 0.0)
    published = ctx._active_builtin_tool_result
    assert outcome["message"] == "⚠️ REVIEW_BLOCKED: critical findings"
    assert isinstance(published, ToolResult)
    assert published.code == "REVIEW_BLOCKED"
    assert published.text == outcome["message"]

    ctx._active_builtin_tool_result = sentinel
    monkeypatch.setattr(
        git_facade,
        "_run_parallel_review",
        lambda *_a, **_k: (None, object(), "critical_findings", []),
    )
    monkeypatch.setattr(
        git_facade,
        "_aggregate_review_verdict",
        lambda *_a, **_k: (
            True,
            "⚠️ SCOPE_REVIEW_BLOCKED: scope only",
            "scope_blocked",
            [],
            [],
        ),
    )
    scope_only = git_tools._run_reviewed_stage_cycle(ctx, "fixture", 0.0)
    assert scope_only["message"] == "⚠️ SCOPE_REVIEW_BLOCKED: scope only"
    assert ctx._active_builtin_tool_result is sentinel
