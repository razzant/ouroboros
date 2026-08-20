"""Focused T4.6 builtin ToolResult bridge and native producer contracts."""

from __future__ import annotations

import contextvars
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ouroboros.tools.registry import ToolEntry, ToolRegistry
from ouroboros.tools.tool_result import (
    LegacyTextResultAdapter,
    ToolResult,
    _install_tool_result_sidecar,
    _publish_tool_result,
    _published_tool_result,
    _restore_tool_result_sidecar,
)


def _entry(name: str, handler) -> ToolEntry:
    return ToolEntry(
        name,
        {
            "name": name,
            "description": "fixture",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        handler,
    )


def test_generic_builtin_result_sidecar_preserves_string_handler_abi(
    tmp_path,
    monkeypatch,
) -> None:
    import ouroboros.safety as safety

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    adapter_calls = []
    original = LegacyTextResultAdapter.from_text
    monkeypatch.setattr(safety, "check_safety", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setattr(
        LegacyTextResultAdapter,
        "from_text",
        classmethod(
            lambda _cls, tool_name, text: (
                adapter_calls.append((tool_name, text))
                or original(tool_name, text)
            )
        ),
    )
    native = ToolResult(
        status="ok",
        code="OK",
        text="exact builtin text",
        meta={"native_fact": True},
    )
    registry.register(_entry(
        "fixture_builtin",
        lambda ctx: _publish_tool_result(ctx, native),
    ))

    assert registry.execute_result("fixture_builtin", {}) == native
    assert registry.execute("fixture_builtin", {}) == native.text
    assert adapter_calls == []
    assert not hasattr(registry._ctx, "_active_builtin_tool_result")


def test_generic_builtin_sidecar_rejects_mismatch_restores_stale_and_preserves_direct_result(
    tmp_path,
    monkeypatch,
) -> None:
    import ouroboros.safety as safety

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    monkeypatch.setattr(safety, "check_safety", lambda *_args, **_kwargs: (True, ""))
    stale = ToolResult(
        status="error",
        code="TOOL_ERROR",
        text="stale",
        meta={"stale": True},
    )
    registry._ctx._active_builtin_tool_result = stale

    def mismatched(ctx):
        _publish_tool_result(
            ctx,
            ToolResult(
                status="error",
                code="TOOL_ERROR",
                text="different",
                meta={"forged": True},
            ),
        )
        return "legacy text"

    registry.register(_entry("fixture_mismatch", mismatched))
    mismatch = registry.execute_result("fixture_mismatch", {})
    assert mismatch == ToolResult(status="ok", code="OK", text="legacy text")
    assert registry._ctx._active_builtin_tool_result is stale

    direct = ToolResult(
        status="error",
        code="TOOL_ERROR",
        text="direct typed result",
        meta={"direct": True},
    )
    registry.register(_entry("fixture_direct", lambda _ctx: direct))
    assert registry.execute_result("fixture_direct", {}) == direct
    assert registry._ctx._active_builtin_tool_result is stale


def test_generic_builtin_sidecar_is_isolated_between_parallel_calls(
    tmp_path,
    monkeypatch,
) -> None:
    import ouroboros.safety as safety

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    monkeypatch.setattr(safety, "check_safety", lambda *_args, **_kwargs: (True, ""))
    barrier = threading.Barrier(2)

    def handler(ctx, label):
        result = ToolResult(
            status="ok",
            code="OK",
            text=f"text-{label}",
            meta={"label": label},
        )
        text = _publish_tool_result(ctx, result)
        barrier.wait(timeout=5)
        return text

    def bound(label):
        def invoke(ctx):
            return handler(ctx, label)
        return invoke

    for label in ("a", "b"):
        registry.register(_entry(
            f"fixture_parallel_{label}",
            bound(label),
        ))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(registry.execute_result, f"fixture_parallel_{label}", {})
            for label in ("a", "b")
        ]
    results = [future.result() for future in futures]

    assert {(result.text, result.meta["label"]) for result in results} == {
        ("text-a", "a"),
        ("text-b", "b"),
    }
    assert not hasattr(registry._ctx, "_active_builtin_tool_result")


def test_generic_builtin_sidecar_shares_context_copy_and_restores_nested_slot() -> None:
    ctx = object()
    outer_sentinel = object()
    outer_token = _install_tool_result_sidecar(ctx, outer_sentinel)
    copied_result = ToolResult(
        status="ok",
        code="OK",
        text="copied",
        meta={"source": "context-copy"},
    )
    inner_result = ToolResult(
        status="ok",
        code="OK",
        text="inner",
        meta={"source": "nested"},
    )
    try:
        copied = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(copied.run, _publish_tool_result, ctx, copied_result).result() == "copied"
        assert _published_tool_result(ctx, outer_sentinel) is copied_result

        inner_sentinel = object()
        inner_token = _install_tool_result_sidecar(ctx, inner_sentinel)
        try:
            assert _publish_tool_result(ctx, inner_result) == "inner"
            assert _published_tool_result(ctx, inner_sentinel) is inner_result
        finally:
            _restore_tool_result_sidecar(inner_token)

        assert _published_tool_result(ctx, outer_sentinel) is copied_result
    finally:
        _restore_tool_result_sidecar(outer_token)

    missing = object()
    assert _published_tool_result(ctx, missing) is missing


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


def test_registry_run_script_wrapper_preserves_native_process_meta(
    tmp_path,
    monkeypatch,
) -> None:
    import ouroboros.safety as safety
    from ouroboros.tools import shell
    from ouroboros.tools.tool_result import _publish_process_result

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    drive.mkdir()
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr(safety, "check_safety", lambda *_args, **_kwargs: (True, ""))
    captured = {}
    base_text = "⚠️ SHELL_EXIT_ERROR: synthetic signal"

    def fake_run_shell(ctx, argv, **_kwargs):
        captured["script_path"] = argv[1]
        return _publish_process_result(
            ctx,
            "SHELL_EXIT_ERROR",
            base_text,
            exit_code=-15,
            artifact_registered=True,
        )

    monkeypatch.setattr(shell, "_run_shell", fake_run_shell)
    result = registry.execute_result(
        "run_script",
        {"script": "print('wrapper regression')"},
    )

    assert result == ToolResult(
        status="error",
        code="SHELL_EXIT_ERROR",
        text=f"{base_text}\n# script_path={captured['script_path']}",
        meta={
            "exit_code": -15,
            "signal": "SIGTERM",
            "artifact_registered": True,
        },
    )


def test_plan_handler_propagates_native_sidecar_through_running_loop_thread(
    tmp_path,
    monkeypatch,
) -> None:
    import asyncio

    import ouroboros.safety as safety
    from ouroboros.tools import plan_review

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    monkeypatch.setattr(safety, "check_safety", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setattr(
        plan_review,
        "_planning_state_location",
        lambda _ctx: (tmp_path, "task"),
    )
    monkeypatch.setattr(
        plan_review,
        "_record_raw_plan_request_attempt",
        lambda *_args, **_kwargs: None,
    )

    async def fake_review(ctx, _request):
        return plan_review._publish_plan_review_projection(
            ctx,
            {"aggregate_signal": "GREEN", "closed": True},
            "exact async plan result",
        )

    monkeypatch.setattr(plan_review, "_run_plan_review_async", fake_review)
    monkeypatch.setattr(
        LegacyTextResultAdapter,
        "from_text",
        classmethod(
            lambda *_args, **_kwargs: pytest.fail(
                "native async plan result reached the legacy adapter"
            )
        ),
    )

    async def exercise():
        return registry.execute_result(
            "plan_task",
            {"plan": "P", "goal": "G"},
        )

    result = asyncio.run(exercise())
    assert result == ToolResult(
        status="ok",
        code="OK",
        text="exact async plan result",
        meta={"plan_review_outcome": "GREEN", "plan_review_closed": True},
    )


@pytest.mark.parametrize(
    ("mode", "review", "args", "note_name"),
    (
        (
            "fresh",
            {"aggregate_signal": "GREEN", "closed": True},
            {"plan": "P", "goal": "G", "spec": {"acceptance_claims": []}},
            "_VACUOUS_CLAIMS_NOTE",
        ),
        (
            "cached",
            {"aggregate_signal": "REVIEW_REQUIRED", "closed": False},
            {"plan": "P", "goal": "G", "review_disposition": {}},
            "_VACUOUS_DISPOSITION_NOTE",
        ),
        (
            "disposition",
            {"aggregate_signal": "REVIEW_REQUIRED", "closed": True},
            {"review_disposition": {"review_fingerprint": "a" * 64, "items": []}},
            "",
        ),
    ),
)
def test_plan_handler_wrapper_preserves_native_meta_for_all_projection_paths(
    tmp_path,
    monkeypatch,
    mode,
    review,
    args,
    note_name,
) -> None:
    import ouroboros.safety as safety
    from ouroboros.tools import plan_review

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    monkeypatch.setattr(safety, "check_safety", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setattr(
        plan_review,
        "_planning_state_location",
        lambda _ctx: (tmp_path, "task"),
    )
    monkeypatch.setattr(
        plan_review,
        "_record_raw_plan_request_attempt",
        lambda *_args, **_kwargs: None,
    )
    base_text = f"{mode} public projection"

    def reuse(ctx, *_args, **_kwargs):
        return plan_review._publish_plan_review_projection(ctx, review, base_text)

    async def review_async(ctx, _request):
        if mode == "cached":
            return plan_review._reuse_or_disposition_plan_review(ctx, "a" * 64, None)
        return plan_review._publish_plan_review_projection(ctx, review, base_text)

    monkeypatch.setattr(plan_review, "_reuse_or_disposition_plan_review", reuse)
    monkeypatch.setattr(plan_review, "_run_plan_review_async", review_async)
    monkeypatch.setattr(
        LegacyTextResultAdapter,
        "from_text",
        classmethod(
            lambda *_args, **_kwargs: pytest.fail(
                "native plan wrapper result reached the legacy adapter"
            )
        ),
    )

    result = registry.execute_result("plan_task", args)
    note = getattr(plan_review, note_name) if note_name else ""
    assert result == ToolResult(
        status="ok",
        code="OK",
        text=base_text + note,
        meta={
            "plan_review_outcome": review["aggregate_signal"],
            "plan_review_closed": review["closed"],
        },
    )


def test_native_review_and_git_producers_bypass_adapter_and_keep_legacy_loop_fields(
    tmp_path,
    monkeypatch,
) -> None:
    import ouroboros.safety as safety
    from ouroboros.loop_tool_execution import _execute_single_tool
    from ouroboros.outcomes import reviewable_effect_projection
    from ouroboros.tools import git as git_tools

    drive = tmp_path / "drive"
    logs = drive / "logs"
    logs.mkdir(parents=True)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=drive)
    monkeypatch.setattr(safety, "check_safety", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setattr(
        LegacyTextResultAdapter,
        "from_text",
        classmethod(
            lambda *_args, **_kwargs: pytest.fail(
                "native review/git result reached the legacy adapter"
            )
        ),
    )
    review_text = "⚠️ REVIEW_BLOCKED: critical findings"
    git_text = "⚠️ GIT_ERROR: repository unavailable"
    registry.override_handler(
        "commit_reviewed",
        lambda ctx, **_kwargs: git_tools._publish_review_blocked(ctx, review_text),
    )
    registry.register(_entry(
        "fixture_git",
        lambda ctx: git_tools._publish_git_error(ctx, git_text),
    ))

    review_row = _execute_single_tool(
        registry,
        {
            "id": "review",
            "function": {
                "name": "commit_reviewed",
                "arguments": json.dumps({"commit_message": "fixture"}),
            },
        },
        logs,
    )
    git_row = _execute_single_tool(
        registry,
        {
            "id": "git",
            "function": {"name": "fixture_git", "arguments": "{}"},
        },
        logs,
    )

    assert review_row["tool_result"] == ToolResult(
        status="ok", code="REVIEW_BLOCKED", text=review_text,
    )
    assert review_row["is_error"] is False
    # T1 §A.17: both refusals get their own bucket; is_error is unchanged.
    assert review_row["result_meta"]["status"] == "review_blocked"
    assert git_row["tool_result"] == ToolResult(
        status="ok", code="GIT_ERROR", text=git_text,
    )
    assert git_row["is_error"] is False
    assert git_row["result_meta"]["status"] == "git_error"
    trace = {
        "tool_calls": [{
            "tool": "commit_reviewed",
            "is_error": review_row["is_error"],
            **review_row["result_meta"],
        }]
    }
    assert reviewable_effect_projection(trace) == []


def test_physical_vcs_status_exception_is_native(
    tmp_path,
    monkeypatch,
) -> None:
    import ouroboros.safety as safety
    from ouroboros.tools import git_vcs_ops as git_tools

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    monkeypatch.setattr(safety, "check_safety", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setattr(
        git_tools,
        "run_cmd",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fixture")),
    )
    monkeypatch.setattr(
        LegacyTextResultAdapter,
        "from_text",
        classmethod(
            lambda *_args, **_kwargs: pytest.fail(
                "physical GIT_ERROR reached the legacy adapter"
            )
        ),
    )

    result = registry.execute_result("vcs_status", {"root": "system_repo"})

    assert result == ToolResult(
        status="ok",
        code="GIT_ERROR",
        text="⚠️ GIT_ERROR: fixture",
    )


def test_review_cycle_publishes_only_structural_critical_finding_rejection(
    tmp_path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from ouroboros.tools import git_review_cycle as git_tools

    sentinel = object()
    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        _active_builtin_tool_result=sentinel,
        _review_advisory=[],
    )
    fingerprint = {"ok": True, "fingerprint": "f", "binding": {}}
    monkeypatch.setattr(
        git_tools,
        "_stage_candidate_for_review",
        lambda *_args, **_kwargs: (["x.py"], ["x.py"], None),
    )
    monkeypatch.setattr(git_tools, "_current_runtime_mode", lambda: "advanced")
    monkeypatch.setattr(git_tools, "_check_advisory_freshness", lambda *_a, **_k: None)
    monkeypatch.setattr(git_tools, "advisory_gate_unavailable", lambda: False)
    monkeypatch.setattr(git_tools, "_fingerprint_staged_diff", lambda *_a: fingerprint)
    monkeypatch.setattr(git_tools, "_review_binding_precondition_error", lambda *_a, **_k: "")
    monkeypatch.setattr(git_tools, "_refuse_capped_attempt", lambda *_a, **_k: None)
    monkeypatch.setattr(git_tools, "_record_commit_attempt", lambda *_a, **_k: None)
    monkeypatch.setattr(
        git_tools,
        "_run_parallel_review",
        lambda *_a, **_k: ("⚠️ REVIEW_BLOCKED: critical findings", None, "critical_findings", []),
    )
    monkeypatch.setattr(
        git_tools,
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
        git_tools,
        "_run_parallel_review",
        lambda *_a, **_k: (None, object(), "critical_findings", []),
    )
    monkeypatch.setattr(
        git_tools,
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
