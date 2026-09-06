"""Tests for advisory_pre_review observability, model-drift fix, and budget gate.

Split from test_commit_gate.py to keep each test module within the ~1000-line limit (P7).
"""
import importlib
import json
import pathlib
import os
import sys


import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_advisory_module():
    sys.path.insert(0, REPO)
    return importlib.import_module("ouroboros.tools.claude_advisory_review")


def _fake_native_result(**overrides):
    """Advisory delivery result shape (the native episode's rehydrated result)."""
    from types import SimpleNamespace
    base = dict(success=True, result_text="", error=None, stderr_tail="",
                session_id="", cost_usd=0.0, usage={})
    base.update(overrides)
    return SimpleNamespace(**base)


def test_advisory_uses_the_shipped_default_model_helper():
    """_run_claude_advisory resolves its model through _advisory_native_model()
    (the configured row or the shipped routed default, each through the
    same-model payable-spelling fallback) — never a hardcoded model and never
    the retired Claude-SDK resolver."""
    import inspect
    adv_mod = _get_advisory_module()
    source = inspect.getsource(adv_mod._run_claude_advisory)
    assert "_advisory_native_model" in source
    assert "resolve_claude_code_model" not in source
    native_src = inspect.getsource(adv_mod._advisory_native_model)
    assert "_same_model_payable_spelling" in native_src


def test_advisory_row_effort_reaches_the_native_slot(monkeypatch, tmp_path):
    """D-5b survivor: the ADVISORY row's own effort rides the native episode —
    the scope reviewer's configured effort never leaks in. (The retired
    Claude-SDK effort plumbing test this replaces asserted the same contract
    on the removed transport.)"""
    adv_mod = _get_advisory_module()

    captured = {}

    def _capture(prompt, repo_dir, ctx_, slot, model, **_):
        from types import SimpleNamespace

        captured["effort"] = slot.effort or "low"
        return SimpleNamespace(
            success=True,
            result_text='[{"item":"bible_compliance","verdict":"PASS","reason":"ok","severity":"critical"}]',
            session_id="", cost_usd=0.0, usage={}, error="", stderr_tail="",
        ), model

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OUROBOROS_EFFORT_SCOPE_REVIEW", "xhigh")
    monkeypatch.delenv("OUROBOROS_REVIEWER_SLOTS", raising=False)
    monkeypatch.setattr(adv_mod, "_run_advisory_native", _capture)
    from ouroboros.tools.registry import ToolContext

    repo = tmp_path / "repo"; repo.mkdir()
    drive = tmp_path / "data"; drive.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=drive)
    _items, raw, _model, _chars = adv_mod._run_claude_advisory(
        repo, "msg", ctx, options={"include_repo_diff": False},
    )
    assert not raw.startswith("⚠️ ADVISORY_ERROR"), raw
    # The legacy default row carries effort "low"; scope xhigh never leaks in.
    assert captured["effort"] == "low"

def test_empty_advisory_result_is_error(monkeypatch, tmp_path):
    adv_mod = _get_advisory_module()
    from types import SimpleNamespace

    def fake_run_readonly(**kwargs):
        return _fake_native_result(
            success=True,
            result_text="(no output)",
            session_id="sess-empty",
            cost_usd=1.23,
            usage={"prompt_tokens": 100, "completion_tokens": 0},
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(_get_advisory_module(), "_run_advisory_native",
                        lambda prompt, repo_dir, ctx_, slot, model, **_: (
                            fake_run_readonly(), model,
                        ))
    monkeypatch.setattr(adv_mod, "_get_staged_diff", lambda *a, **kw: "diff")
    monkeypatch.setattr(adv_mod, "_get_changed_file_list", lambda *a, **kw: "M file.py")
    monkeypatch.setattr(adv_mod, "build_advisory_changed_context", lambda *a, **kw: (["file.py"], "pack", []))
    monkeypatch.setattr(adv_mod, "_build_advisory_prompt", lambda *a, **kw: "prompt")
    ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, pending_events=[], emit_progress_fn=lambda *_: None)

    items, raw, _model, _chars = adv_mod._run_claude_advisory(tmp_path, "msg", ctx)

    assert items == []
    assert raw.startswith("⚠️ ADVISORY_ERROR:")
    assert "empty output" in raw
    assert any(ev.get("type") == "advisory_suspect_result" for ev in ctx.pending_events)


def test_handle_advisory_error_persists_session_id(monkeypatch, tmp_path):
    adv_mod = _get_advisory_module()
    from types import SimpleNamespace
    from ouroboros.review_state import load_state

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(
        adv_mod,
        "_advisory_pre_sdk_gate",
        lambda **kwargs: ([], "M file.py", None),
    )

    def fake_run(*args, **kwargs):
        ctx = args[2]
        ctx._last_claude_advisory_meta = {"session_id": "sess-paid-empty"}
        return [], "⚠️ ADVISORY_ERROR: paid empty output", "claude-opus", 12345

    monkeypatch.setattr(adv_mod, "_run_claude_advisory", fake_run)
    ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, pending_events=[], emit_progress_fn=lambda *_: None, task_id="t")

    raw = adv_mod._handle_advisory_pre_review(ctx, commit_message="msg", skip_tests=True)
    result = json.loads(raw)

    assert result["status"] == "error"
    assert result["session_id"] == "sess-paid-empty"
    state = load_state(tmp_path)
    assert state.advisory_runs
    run = state.advisory_runs[-1]
    assert run.status == "error"
    assert run.session_id == "sess-paid-empty"
    assert run.prompt_chars == 12345


def test_skill_advisory_duplicate_expected_items_warn_not_error(monkeypatch, tmp_path):
    adv_mod = _get_advisory_module()
    from types import SimpleNamespace

    def fake_run_readonly(**kwargs):
        return _fake_native_result(
            success=True,
            result_text=json.dumps([
                {"item": "manifest_schema", "verdict": "PASS", "reason": "ok", "severity": "critical"},
                {"item": "permissions_honesty", "verdict": "PASS", "reason": "ok", "severity": "critical"},
                {"item": "permissions_honesty", "verdict": "FAIL", "reason": "second issue", "severity": "critical"},
            ]),
            session_id="sess-duplicate",
            cost_usd=0.2,
            usage={"prompt_tokens": 100, "completion_tokens": 20},
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(_get_advisory_module(), "_run_advisory_native",
                        lambda prompt, repo_dir, ctx_, slot, model, **_: (
                            fake_run_readonly(), model,
                        ))
    monkeypatch.setattr(adv_mod, "_build_advisory_prompt", lambda *a, **kw: "prompt")
    ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, pending_events=[], emit_progress_fn=lambda *_: None)

    items, raw, _model, _chars = adv_mod._run_claude_advisory(
        tmp_path,
        "skill advisory",
        ctx,
        scope="plugin.py",
        options={
            "include_repo_diff": False,
            "review_surface": "skill",
            "expected_items": ["manifest_schema", "permissions_honesty"],
        },
    )

    assert len(items) == 3
    assert not raw.startswith("⚠️ ADVISORY_ERROR:")
    assert any(ev.get("type") == "advisory_contract_warning" for ev in ctx.pending_events)
    assert not any(ev.get("type") == "advisory_suspect_result" for ev in ctx.pending_events)


def test_skill_advisory_repeated_bug_hunting_no_contract_warning(monkeypatch, tmp_path):
    # C6: bug_hunting is severity-driven and legitimately emits one row per
    # distinct bug; repeated rows must NOT trip duplicates=/count= contract
    # warnings or the advisory_suspect_result marker.
    adv_mod = _get_advisory_module()
    from types import SimpleNamespace

    def fake_run_readonly(**kwargs):
        return _fake_native_result(
            success=True,
            result_text=json.dumps([
                {"item": "manifest_schema", "verdict": "PASS", "reason": "ok", "severity": "critical"},
                {"item": "bug_hunting", "verdict": "FAIL", "reason": "bug A", "severity": "advisory"},
                {"item": "bug_hunting", "verdict": "FAIL", "reason": "bug B", "severity": "advisory"},
                {"item": "bug_hunting", "verdict": "FAIL", "reason": "bug C", "severity": "critical"},
            ]),
            session_id="sess-bughunt",
            cost_usd=0.2,
            usage={"prompt_tokens": 100, "completion_tokens": 20},
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(_get_advisory_module(), "_run_advisory_native",
                        lambda prompt, repo_dir, ctx_, slot, model, **_: (
                            fake_run_readonly(), model,
                        ))
    monkeypatch.setattr(adv_mod, "_build_advisory_prompt", lambda *a, **kw: "prompt")
    ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, pending_events=[], emit_progress_fn=lambda *_: None)

    items, raw, _model, _chars = adv_mod._run_claude_advisory(
        tmp_path,
        "skill advisory",
        ctx,
        scope="plugin.py",
        options={
            "include_repo_diff": False,
            "review_surface": "skill",
            "expected_items": ["manifest_schema", "bug_hunting"],
        },
    )

    assert len(items) == 4  # every bug row preserved in the output
    assert not raw.startswith("⚠️ ADVISORY_ERROR:")
    assert not any(ev.get("type") == "advisory_contract_warning" for ev in ctx.pending_events)
    assert not any(ev.get("type") == "advisory_suspect_result" for ev in ctx.pending_events)


def test_skill_advisory_missing_expected_items_still_errors(monkeypatch, tmp_path):
    adv_mod = _get_advisory_module()
    from types import SimpleNamespace

    def fake_run_readonly(**kwargs):
        return _fake_native_result(
            success=True,
            result_text='[{"item":"manifest_schema","verdict":"PASS","reason":"ok","severity":"critical"}]',
            session_id="sess-partial",
            cost_usd=0.2,
            usage={"prompt_tokens": 100, "completion_tokens": 20},
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(_get_advisory_module(), "_run_advisory_native",
                        lambda prompt, repo_dir, ctx_, slot, model, **_: (
                            fake_run_readonly(), model,
                        ))
    monkeypatch.setattr(adv_mod, "_build_advisory_prompt", lambda *a, **kw: "prompt")
    ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, pending_events=[], emit_progress_fn=lambda *_: None)

    items, raw, _model, _chars = adv_mod._run_claude_advisory(
        tmp_path,
        "skill advisory",
        ctx,
        scope="plugin.py",
        options={
            "include_repo_diff": False,
            "review_surface": "skill",
            "expected_items": ["manifest_schema", "permissions_honesty"],
        },
    )

    assert items == []
    assert raw.startswith("⚠️ ADVISORY_ERROR:")
    assert "checklist contract mismatch" in raw
    assert any(ev.get("type") == "advisory_suspect_result" for ev in ctx.pending_events)


# ---------------------------------------------------------------------------
# Observability: _format_advisory_error / _get_runtime_diagnostics
# ---------------------------------------------------------------------------

def test_advisory_error_message_includes_diagnostic_fields():
    """_format_advisory_error must include all required diagnostic fields."""
    adv_mod = _get_advisory_module()
    diag = {
        "model": "opus",
        "python": "/usr/bin/python3",
        "prompt_chars": 120000,
        "prompt_tokens_approx": 30000,
        "touched_paths": ["ouroboros/tools/foo.py"],
    }
    msg = adv_mod._format_advisory_error(
        prefix="test failure",
        result_error="exit code 1",
        stderr_tail="some stderr line",
        session_id="sess-123",
        diag=diag,
    )
    assert "⚠️ ADVISORY_ERROR:" in msg
    assert "opus" in msg
    assert "/usr/bin/python3" in msg
    assert "120000" in msg
    assert "30000" in msg or "30,000" in msg
    assert "sess-123" in msg
    assert "some stderr line" in msg
    assert "ouroboros/tools/foo.py" in msg


def test_get_runtime_diagnostics_never_raises():
    """_get_runtime_diagnostics must return partial data on any error, never raise."""
    adv_mod = _get_advisory_module()
    diag = adv_mod._get_runtime_diagnostics("opus", 50000, ["file.py"])
    assert isinstance(diag, dict)
    assert diag["model"] == "opus"
    assert diag["prompt_chars"] == 50000
    assert diag["prompt_tokens_approx"] == 12500
    assert diag["touched_paths"] == ["file.py"]
    # The retired Claude-SDK/CLI probes must never resurface here.
    assert "sdk_version" not in diag and "cli_version" not in diag


# ---------------------------------------------------------------------------
# Budget gate: skip path and durable state
# ---------------------------------------------------------------------------

def _make_minimal_git_repo(tmp_path):
    import subprocess
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    (tmp_path / "BIBLE.md").write_text("bible", encoding="utf-8")
    (tmp_path / "VERSION").write_text("5.99.0-rc.1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "5.99.0rc1"\n', encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "[![Version 5.99.0-rc.1](https://img.shields.io/badge/version-5.99.0--rc.1-green.svg)](VERSION)\n\n"
        "## Version History\n\n| Version | Date | Description |\n|---------|------|-------------|\n"
        "| 5.99.0-rc.1 | 2026-05-16 | test row |\n",
        encoding="utf-8",
    )
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "CHECKLISTS.md").write_text("# Repo Commit Checklist\n", encoding="utf-8")
    (tmp_path / "docs" / "ARCHITECTURE.md").write_text(
        "# Ouroboros v5.99.0-rc.1 — Architecture & Reference\n",
        encoding="utf-8",
    )
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)


@pytest.fixture
def budget_gate_env(monkeypatch, tmp_path):
    """Pin the prompt budget to 10 chars and prepare a minimal env+repo.

    Yields ``(adv_mod, tmp_path)`` so callers can build their own ctx and
    invoke the advisory entrypoint they care about. Restores the original
    ``_ADVISORY_PROMPT_MAX_CHARS`` at teardown.
    """
    adv_mod = _get_advisory_module()
    _make_minimal_git_repo(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "opus")
    original_limit = adv_mod._ADVISORY_PROMPT_MAX_CHARS
    adv_mod._ADVISORY_PROMPT_MAX_CHARS = 10
    try:
        yield adv_mod, tmp_path
    finally:
        adv_mod._ADVISORY_PROMPT_MAX_CHARS = original_limit


def _budget_ctx(tmp_path, *, task_id=None):
    from types import SimpleNamespace
    ctx_kwargs = {
        "repo_dir": tmp_path,
        "drive_root": tmp_path,
        "emit_progress_fn": lambda _: None,
        "pending_events": [],
    }
    if task_id is not None:
        ctx_kwargs["task_id"] = task_id
    return SimpleNamespace(**ctx_kwargs)


def test_advisory_budget_gate_returns_skipped_on_large_prompt(budget_gate_env):
    """_run_claude_advisory must return ADVISORY_SKIPPED when prompt exceeds budget gate."""
    adv_mod, tmp_path = budget_gate_env
    ctx = _budget_ctx(tmp_path)
    items, raw, _model, _chars = adv_mod._run_claude_advisory(tmp_path, "test commit", ctx)
    assert items == []
    assert raw.startswith("⚠️ ADVISORY_SKIPPED:")
    assert "chars" in raw


def test_handle_advisory_pre_review_returns_skipped_status_on_budget_gate(budget_gate_env):
    """_handle_advisory_pre_review must surface ADVISORY_SKIPPED as status='skipped'."""
    adv_mod, tmp_path = budget_gate_env
    ctx = _budget_ctx(tmp_path, task_id="t-test")
    raw_json = adv_mod._handle_advisory_pre_review(ctx, commit_message="test commit")
    result = json.loads(raw_json)
    assert result["status"] == "skipped"
    assert "ADVISORY_SKIPPED" in result["message"]


def test_budget_gate_skip_persists_durable_state(budget_gate_env):
    """Budget-gate skip must write status='skipped' to state; is_fresh() must return True."""
    adv_mod, tmp_path = budget_gate_env
    ctx = _budget_ctx(tmp_path, task_id="t-bg")
    raw_json = adv_mod._handle_advisory_pre_review(ctx, commit_message="budget gate test")

    result = json.loads(raw_json)
    assert result["status"] == "skipped"
    snapshot_hash = result["snapshot_hash"]

    from ouroboros.review_state import load_state
    state = load_state(tmp_path)
    assert state.is_fresh(snapshot_hash), (
        "is_fresh() must be True after budget-gate skip so commit gate does not re-block"
    )
    run = state.find_by_hash(snapshot_hash)
    assert run is not None
    assert run.status == "skipped"


def test_next_step_guidance_for_skipped_advisory():
    """_next_step_guidance must return a distinct message for status='skipped' runs."""
    adv_mod = _get_advisory_module()
    from ouroboros.review_state import AdvisoryRunRecord, AdvisoryReviewState

    skipped_run = AdvisoryRunRecord(
        snapshot_hash="abc123",
        commit_message="test",
        status="skipped",
        ts="2026-01-01T00:00:00",
    )
    state = AdvisoryReviewState(advisory_runs=[skipped_run])
    msg = adv_mod._next_step_guidance(
        latest=skipped_run,
        state=state,
        stale_from_edit=False,
        stale_from_edit_ts=None,
        open_obs=[],
        open_debts=[],
        effective_is_fresh=True,
    )
    # Must NOT say "fresh" or "no critical findings" — that would mislead
    assert "skip" in msg.lower() or "budget" in msg.lower(), (
        "skipped advisory must produce a distinct message, not the generic fresh-pass message"
    )
    assert "commit_reviewed" in msg, "message should still indicate commit can proceed"


def test_next_step_guidance_requires_reaudit_when_obligations_remain():
    """Open obligations after a blocked review should trigger explicit re-audit guidance."""
    adv_mod = _get_advisory_module()
    from ouroboros.review_state import AdvisoryRunRecord, AdvisoryReviewState, ObligationItem

    fresh_run = AdvisoryRunRecord(
        snapshot_hash="abc123",
        commit_message="test",
        status="fresh",
        ts="2026-01-01T00:00:00",
    )
    state = AdvisoryReviewState(advisory_runs=[fresh_run])
    open_obs = [ObligationItem(
        obligation_id="ob-1",
        item="code_quality",
        severity="critical",
        reason="Need broader fix",
        source_attempt_ts="2026-01-01T00:00:01",
        source_attempt_msg="blocked",
        repo_key="repo",
    )]
    msg = adv_mod._next_step_guidance(
        latest=fresh_run,
        state=state,
        stale_from_edit=False,
        stale_from_edit_ts=None,
        open_obs=open_obs,
        open_debts=[],
        effective_is_fresh=True,
    )
    lowered = msg.lower()
    assert "re-read the full diff" in lowered
    assert "group obligations by root cause" in lowered
    assert "rewrite the plan" in lowered


@pytest.mark.parametrize(
    "case_id,status,effective_is_fresh,stale_from_edit,items,open_debts",
    [
        ("missing", None, False, False, [], []),
        ("stale", "fresh", False, True, [], []),
        ("parse_failure", "parse_failure", False, False, [], []),
        ("error", "error", False, False, [], []),
        (
            "critical",
            "fresh",
            True,
            False,
            [{"item": "correctness", "verdict": "FAIL", "severity": "critical"}],
            [],
        ),
        ("open_debt", "fresh", True, False, [], [object()]),
    ],
)
def test_next_step_guidance_offers_both_advisory_choices_and_retains_gates(
    case_id,
    status,
    effective_is_fresh,
    stale_from_edit,
    items,
    open_debts,
):
    del case_id
    adv_mod = _get_advisory_module()
    from ouroboros.review_state import AdvisoryRunRecord, AdvisoryReviewState

    latest = None
    if status is not None:
        latest = AdvisoryRunRecord(
            snapshot_hash="abc123",
            commit_message="test",
            status=status,
            ts="2026-01-01T00:00:00",
            items=items,
        )
    state = AdvisoryReviewState(advisory_runs=[] if latest is None else [latest])
    message = adv_mod._next_step_guidance(
        latest=latest,
        state=state,
        stale_from_edit=stale_from_edit,
        stale_from_edit_ts="2026-01-01T00:01:00" if stale_from_edit else None,
        open_obs=[],
        open_debts=open_debts,
        effective_is_fresh=effective_is_fresh,
    )

    assert adv_mod.ADVISORY_REVIEW_CHOICE_GUIDANCE in message
    assert "advisory_review" in message
    assert "skip_advisory_review=True" in message
    assert "bypasses only the requirements for advisory freshness" in message
    assert "records remain visible" in message
    assert "tests, triad review" in message
    assert "snapshot/fingerprint revalidation" in message
    assert "final commit/tag/SHA binding" in message


def test_skipped_run_hash_mismatch_reported_as_stale(monkeypatch, tmp_path):
    """A skipped run with a different snapshot hash must be reported as stale (hash_mismatch path)."""
    adv_mod = _get_advisory_module()
    _make_minimal_git_repo(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "opus")

    import subprocess
    # Commit BIBLE.md so git has a real HEAD
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@test.com"], capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], capture_output=True)

    # Write a skipped run with a fake (stale) hash directly into state
    from ouroboros.review_state import (
        AdvisoryRunRecord, AdvisoryReviewState, save_state,
    )
    old_hash = "000000000000000000000000000000000000000000000000"
    run = AdvisoryRunRecord(
        snapshot_hash=old_hash,
        commit_message="skipped test",
        status="skipped",
        ts="2026-01-01T00:00:00",
    )
    state = AdvisoryReviewState(advisory_runs=[run])
    save_state(tmp_path, state)

    # Now add a file to the worktree so the real snapshot hash differs from old_hash
    (tmp_path / "new_file.py").write_text("x = 1\n", encoding="utf-8")

    # review_status must report stale (hash mismatch), not fresh
    raw_json = adv_mod._handle_review_status(
        ctx=__import__("types").SimpleNamespace(
            repo_dir=tmp_path, drive_root=tmp_path,
            emit_progress_fn=lambda _: None, pending_events=[],
        )
    )
    import json as _json
    result = _json.loads(raw_json)
    latest_status = result.get("latest_advisory_status", "")
    assert latest_status in ("stale", "no_advisory"), (
        f"Expected stale/no_advisory for skipped run with hash mismatch, got: {latest_status!r}\n"
        f"Full result: {result}"
    )


def test_advisory_context_build_failure_is_surfaced(monkeypatch, tmp_path):
    """Phase 4: changed-file context build failures must surface as explicit advisory errors."""
    adv_mod = _get_advisory_module()
    _make_minimal_git_repo(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "opus")

    monkeypatch.setattr(adv_mod, "_get_staged_diff", lambda *args, **kwargs: "(no diff)")
    monkeypatch.setattr(adv_mod, "_get_changed_file_list", lambda *args, **kwargs: "M foo.py")
    monkeypatch.setattr(
        adv_mod,
        "build_advisory_changed_context",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("context pack exploded")),
    )

    from types import SimpleNamespace
    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        emit_progress_fn=lambda _: None,
        pending_events=[],
        task_id="ctx-fail",
    )
    items, raw, _model, _chars = adv_mod._run_claude_advisory(tmp_path, "test commit", ctx)
    assert items == []
    assert raw.startswith("⚠️ ADVISORY_ERROR:")
    assert "failed to build advisory prompt" in raw


def test_budget_gate_skip_becomes_stale_after_edit(monkeypatch, tmp_path):
    """A budget-gate skip must be invalidated (marked stale) by a subsequent worktree edit."""
    adv_mod = _get_advisory_module()
    _make_minimal_git_repo(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_CODE_MODEL", "opus")

    original_limit = adv_mod._ADVISORY_PROMPT_MAX_CHARS
    try:
        adv_mod._ADVISORY_PROMPT_MAX_CHARS = 10
        from types import SimpleNamespace
        ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, task_id="t-stale",
                              emit_progress_fn=lambda _: None, pending_events=[])
        raw_json = adv_mod._handle_advisory_pre_review(ctx, commit_message="skip stale test")
    finally:
        adv_mod._ADVISORY_PROMPT_MAX_CHARS = original_limit

    result = json.loads(raw_json)
    assert result["status"] == "skipped"
    snapshot_hash = result["snapshot_hash"]

    # Simulate a worktree edit invalidating the advisory
    from ouroboros.review_state import load_state, mark_advisory_stale_after_edit
    mark_advisory_stale_after_edit(tmp_path)

    state = load_state(tmp_path)
    assert not state.is_fresh(snapshot_hash), (
        "is_fresh() must be False after mark_advisory_stale_after_edit() — edit invalidates skip"
    )
    run = state.find_by_hash(snapshot_hash)
    assert run is not None
    assert run.status == "stale"


# ---------------------------------------------------------------------------
# SDK break-after-ResultMessage fix (spurious exit code 1 prevention)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# _parse_advisory_output — JSON array extraction heuristics
# ---------------------------------------------------------------------------

class TestIsChecklistArray:
    """Unit tests for the _is_checklist_array helper."""

    def setup_method(self, _=None):
        import importlib
        self.mod = importlib.import_module("ouroboros.tools.claude_advisory_review")
        self.fn = self.mod._is_checklist_array

    def test_empty_list_rejected(self):
        assert self.fn([]) is False

    def test_stray_int_array_rejected(self):
        assert self.fn([1, 2, 3]) is False

    def test_stray_string_array_rejected(self):
        assert self.fn(["a", "b"]) is False

    def test_dict_missing_item_rejected(self):
        assert self.fn([{"verdict": "PASS"}]) is False

    def test_dict_missing_verdict_rejected(self):
        assert self.fn([{"item": "bible_compliance"}]) is False

    def test_valid_single_item_accepted(self):
        assert self.fn([{"item": "bible_compliance", "verdict": "PASS"}]) is True

    def test_valid_multi_item_accepted(self):
        items = [
            {"item": "bible_compliance", "verdict": "PASS"},
            {"item": "code_quality", "verdict": "FAIL", "reason": "bug"},
        ]
        assert self.fn(items) is True

    def test_mixed_valid_invalid_rejected(self):
        # One bad element should disqualify the whole array
        items = [
            {"item": "bible_compliance", "verdict": "PASS"},
            {"not_item": "x"},
        ]
        assert self.fn(items) is False


class TestParseAdvisoryOutput:
    """Tests for the JSON array parser used to extract checklist items
    from advisory SDK output, including cases where code blocks contain
    brackets that could confuse a naïve find/rfind approach."""

    @pytest.fixture(autouse=True)
    def _import(self):
        import importlib
        self.mod = importlib.import_module(
            "ouroboros.tools.claude_advisory_review"
        )

    def _parse(self, text: str) -> list:
        return self.mod._parse_advisory_output(text)

    def _item(self, item: str, verdict: str = "PASS") -> dict:
        return {"item": item, "verdict": verdict, "reason": "ok"}

    def test_plain_json_array(self):
        items = [self._item("bible_compliance")]
        text = json.dumps(items)
        assert self._parse(text) == items

    def test_json_after_prose(self):
        items = [self._item("secrets_check"), self._item("code_quality")]
        text = "Here is my analysis.\n\n" + json.dumps(items)
        assert self._parse(text) == items

    def test_json_after_code_block_with_brackets(self):
        """Code block containing '[' and ']' before the actual JSON array."""
        items = [self._item("bible_compliance", "PASS"), self._item("version_bump", "PASS")]
        code_block = (
            "```python\n"
            "result = [x for x in range(10)]\n"
            "nested = [[1, 2], [3, 4]]\n"
            "```\n\n"
        )
        text = "Let me think.\n" + code_block + "Final answer:\n" + json.dumps(items)
        result = self._parse(text)
        assert result == items

    def test_json_in_markdown_fence(self):
        items = [self._item("tests_affected")]
        text = "Review:\n```json\n" + json.dumps(items) + "\n```"
        assert self._parse(text) == items

    def test_empty_input(self):
        assert self._parse("") == []

    def test_no_json(self):
        assert self._parse("This is prose with no JSON.") == []

    def test_multiple_code_blocks_json_last(self):
        """Multiple code blocks followed by the JSON findings array — the
        real production scenario that caused parse_failure."""
        items = [
            self._item("bible_compliance"),
            self._item("code_quality"),
            self._item("version_bump"),
        ]
        text = (
            "Checking files...\n"
            "```python\n"
            "checks = [{'key': 'val'}, {'key2': [1, 2, 3]}]\n"
            "```\n"
            "More analysis with [inline] brackets and [another].\n"
            "Final findings:\n"
            + json.dumps(items)
        )
        result = self._parse(text)
        assert result == items

    def test_stray_array_after_real_checklist_returns_checklist(self):
        """When a real checklist array is followed by a stray unrelated array,
        the parser must return the checklist, not the stray array."""
        items = [
            self._item("bible_compliance"),
            self._item("code_quality"),
        ]
        # The stray [1,2,3] appears AFTER the real checklist — a bracket-scan
        # without shape validation would return [1,2,3] because rfind("]")
        # finds the last "]" which belongs to [1,2,3].
        text = json.dumps(items) + "\n\nSee also config option [1,2,3]."
        result = self._parse(text)
        assert result == items, (
            "Parser must prefer the checklist array over a later stray array"
        )

    def test_stray_int_array_alone_returns_empty(self):
        """A stray [1,2,3] with no real checklist must yield empty list (parse_failure)."""
        result = self._parse("some text [1,2,3] end")
        assert result == []


# ---------------------------------------------------------------------------
# LLM fallback extraction (_llm_extract_advisory_items)
# ---------------------------------------------------------------------------

class TestLLMFallbackExtraction:
    """Tests for the LLM-first parse-failure fallback in _run_claude_advisory."""

    @pytest.fixture(autouse=True)
    def _import(self):
        import importlib
        self.mod = importlib.import_module("ouroboros.tools.claude_advisory_review")

    def _make_ctx(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            repo_dir="/tmp",
            drive_root="/tmp",
            emit_progress_fn=lambda _: None,
            pending_events=[],
            task_id="fallback-test",
        )

    def _item(self, item: str, verdict: str = "PASS") -> dict:
        return {"item": item, "verdict": verdict, "reason": "ok"}

    def test_fallback_succeeds_when_direct_parse_fails(self, monkeypatch):
        """When _parse_advisory_output returns [] but raw_text has JSON,
        _llm_extract_advisory_items should return the checklist items."""
        expected = [self._item("bible_compliance"), self._item("code_quality")]

        # Patch LLMClient.chat to return the JSON as if the LLM extracted it
        def fake_chat(self_llm, messages, model, **kwargs):
            return {"content": json.dumps(expected)}, {"cost": 0.001}

        import ouroboros.llm as llm_mod
        monkeypatch.setattr(llm_mod.LLMClient, "chat", fake_chat)

        raw_text = (
            "I've reviewed the code carefully.\n"
            "Let me check each file...\n"
            "Here are my findings in JSON:\n"
            + json.dumps(expected)
        )
        result = self.mod._llm_extract_advisory_items(raw_text, self._make_ctx())
        assert result == expected

    def test_fallback_extraction_inherits_the_owner_deadline(self, monkeypatch):
        from datetime import datetime, timedelta, timezone

        captured = {}
        expected = [self._item("code_quality")]

        def fake_chat(_self, **kwargs):
            captured.update(kwargs)
            return {"content": json.dumps(expected)}, {"cost": 0.001}

        import ouroboros.llm as llm_mod
        monkeypatch.setattr(llm_mod.LLMClient, "chat", fake_chat)
        monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "1")
        ctx = self._make_ctx()
        ctx.task_metadata = {
            "deadline_at": (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat(),
        }
        assert self.mod._llm_extract_advisory_items("narrative", ctx) == expected
        assert 0 < captured["timeout"] <= 4.1

    def test_mid_artifact_critical_survives_the_verdict(self, monkeypatch):
        """A critical raised in the MIDDLE of a long advisory must reach the verdict.

        Extraction used to read a 4K head + 60K tail window: everything between was
        dropped, so a mid-artifact critical vanished — and because entries may carry
        `obligation_id`, a surviving advisory row could then close an obligation whose
        critical had just been cut away. The shared SSOT reads the WHOLE artifact, so
        either the array is read from it directly or the whole text reaches the
        extractor — the middle is never cut away before the verdict.
        """
        seen = {}
        expected = [{"item": "bug_hunting", "verdict": "FAIL", "severity": "critical",
                     "reason": "off-by-one in the mid-artifact path"}]

        def fake_chat(_self, **kwargs):
            seen["prompt"] = kwargs["messages"][0]["content"]
            return {"content": json.dumps(expected)}, {"cost": 0.0001}

        import ouroboros.llm as llm_mod
        monkeypatch.setattr(llm_mod.LLMClient, "chat", fake_chat)

        marker = "MID_ARTIFACT_CRITICAL_MARKER"
        raw_text = ("A" * 80_000) + marker + json.dumps(expected) + ("B" * 80_000)
        result = self.mod._llm_extract_advisory_items(raw_text, self._make_ctx())

        assert result == expected
        if "prompt" in seen:  # if a light-model call was needed, it saw everything
            assert marker in seen["prompt"]
            assert "OMISSION NOTE" not in seen["prompt"]
            assert raw_text in seen["prompt"]

    def test_oversize_artifact_refuses_typed_instead_of_guessing(self, monkeypatch):
        """Beyond the one-send extraction bound the answer is a typed refusal — never
        a verdict fabricated from whatever part happened to be visible."""
        calls = []

        def counting_chat(_self, **kwargs):
            # Counted, NOT raised: an exception here would be swallowed by the
            # extractor's own guard and the test would pass for the wrong reason.
            calls.append(kwargs)
            return {"content": "[]"}, {"cost": 0.0001}

        import ouroboros.llm as llm_mod
        from ouroboros.review_execution import _EXTRACT_MAX_CHARS
        monkeypatch.setattr(llm_mod.LLMClient, "chat", counting_chat)

        raw_text = "x" * (_EXTRACT_MAX_CHARS + 1)
        assert self.mod._llm_extract_advisory_items(raw_text, self._make_ctx()) == []
        # No windowed read was attempted at all — the refusal is typed, not a guess.
        assert calls == [], "extraction ran on an artifact over the one-send bound"

    def test_bare_empty_array_from_the_light_model_is_not_a_clean_verdict(self):
        """TRAP: the canonicalizer legitimately returns "[]" for "no findings here".
        Cleanliness is judged on the RAW artifact, so moving it onto the canonical
        text would fabricate a verified-clean verdict out of a failed extraction."""
        # The light model's "[]" yields no items...
        assert self.mod._parse_advisory_output("[]") == []
        # ...and the raw artifact it came from is NOT clean: no NO_FINDINGS sentinel,
        # no bare-`[]` body — so the run stays parse_failure rather than clean.
        assert self.mod._is_clean_verdict("I could not complete the review.") is False
        assert self.mod._is_clean_verdict("blah blah [] blah") is False
        # The genuine clean shapes still are clean.
        assert self.mod._is_clean_verdict("[]") is True
        assert self.mod._is_clean_verdict(json.dumps({"result": "[]\nNO_FINDINGS"})) is True

    def test_delegated_advisory_asks_the_schema_and_discloses_off_pin(self, monkeypatch):
        """The delegated advisory rides the SHARED AgentSessionReviewExecutor
        (phase C unification, 2=B): the schema is asked like every session
        slot, conformance is trusted only on "passed", and the substrate's
        own capability-delta vocabulary discloses the landings — the advisory
        no longer carries a transport dialect of its own."""
        from ouroboros.review_execution import review_session_output_schema

        monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "claude")
        asked = {}

        def fake_runner(**kwargs):
            invocation = kwargs["invocation"]
            asked.update({f: getattr(invocation, f) for f in
                          ("task_id", "surface", "slot_id", "output_schema")})
            return {
                "text": '[]', "run_id": "run-9", "route_id": "pinned-route",
                "model": "m", "spend": None, "spend_estimated": False,
                "settlement": {}, "schema_asked": True, "conformance": "failed",
                "effective_route_ids": ["other-route"],
            }

        import ouroboros.review_execution as rx
        monkeypatch.setattr(rx, "run_delegated_review_session", fake_runner)
        result, _model = self.mod._run_advisory_delegated(
            "prompt", pathlib.Path("/tmp"), self._make_ctx(),
        )

        assert asked["output_schema"] == review_session_output_schema("advisory_review")
        assert "minItems" not in asked["output_schema"]["properties"]["findings"]
        usage = result.usage
        assert usage["output_conformance"] == "failed"
        assert usage["verdict_provenance"]["conformance_trusted"] is False
        reasons = {d["reason"] for d in usage["capability_delta"]}
        assert {"schema_not_conformed_on_effective_route",
                "session_ran_off_pinned_route"} <= reasons, reasons

    def test_delegated_advisory_unwraps_a_conformant_clean_envelope(self, monkeypatch):
        """A schema-conformant session answers with the SESSION envelope
        ({"findings": [...]}) while every advisory consumer downstream reads the
        advisory's own ARRAY contract — so a clean {"findings": []} used to land as
        a paid extraction and then a parse_failure. Conformance-trusted envelopes
        unwrap to the array shape: clean becomes the bare "[]" sentinel the
        contract calls clean, findings become the bare array."""
        import ouroboros.review_execution as rx

        def runner_for(text, conformance):
            def fake_runner(**kwargs):
                return {
                    "text": text, "run_id": "run-9", "route_id": "r", "model": "m",
                    "spend": None, "spend_estimated": False, "settlement": {},
                    "schema_asked": True, "conformance": conformance,
                    "effective_route_ids": ["r"],
                }
            return fake_runner

        # monkeypatch, not bare os.environ: a leaked OUROBOROS_SUBAGENT_HARNESS
        # makes UNRELATED dispatch tests on the same xdist worker walk the
        # delegation route-health path (parallel-safety rule, DEVELOPMENT.md).
        monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "claude")
        cases = [
            # (session text, conformance, expected result_text)
            (json.dumps({"findings": []}), "passed", "[]"),
            (json.dumps({"findings": [{"item": "i", "verdict": "FAIL",
                                       "severity": "critical", "reason": "x"}]}),
             "passed",
             json.dumps([{"item": "i", "verdict": "FAIL",
                          "severity": "critical", "reason": "x"}],
                        ensure_ascii=False)),
            # NOT conformed but strictly parseable: the shared D19 order's
            # strict parse settles it — no paid extraction, no envelope trust.
            ("[]", "failed", "[]"),
        ]
        original = rx.run_delegated_review_session
        try:
            for text, conformance, expected in cases:
                rx.run_delegated_review_session = runner_for(text, conformance)
                result, _model = self.mod._run_advisory_delegated(
                    "prompt", pathlib.Path("/tmp"), self._make_ctx(),
                )
                assert result.result_text == expected, (text, conformance)
        finally:
            rx.run_delegated_review_session = original
        # And the unwrapped clean shape IS the contract's clean verdict.
        assert self.mod._is_clean_verdict("[]") is True

    def test_resolve_fallback_model_no_env_uses_config_default(self, monkeypatch):
        """When OUROBOROS_MODEL_LIGHT is unset, the fallback model falls back to Main
        (role-model v6.39: empty Light -> Main), never an empty model id."""
        monkeypatch.delenv("OUROBOROS_MODEL_LIGHT", raising=False)

        model = self.mod._resolve_fallback_model()
        from ouroboros.config import get_light_model
        expected = get_light_model()
        assert model == expected, (
            f"Expected role-model light->main {expected!r}, got: {model!r}"
        )
        assert model, "Fallback model must be non-empty"

    def test_resolve_fallback_model_uses_env_var(self, monkeypatch):
        """OUROBOROS_MODEL_LIGHT must take priority over auto-detection."""
        monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "openai::gpt-4o-mini")
        model = self.mod._resolve_fallback_model()
        assert model == "openai::gpt-4o-mini"

    def test_fallback_normalises_fail_without_severity_to_critical(self, monkeypatch):
        """FAIL items missing 'severity' from the LLM fallback must be normalised to 'critical'
        so _handle_advisory_pre_review() never silently downgrades blocking findings."""
        # Simulate LLM returning a FAIL item with no severity (schema-incomplete output)
        raw_items = [
            {"item": "code_quality", "verdict": "FAIL", "reason": "bug found"},
        ]

        def fake_chat(self_llm, messages, model, **kwargs):
            return {"content": json.dumps(raw_items)}, {"cost": 0.001}

        import ouroboros.llm as llm_mod
        monkeypatch.setattr(llm_mod.LLMClient, "chat", fake_chat)

        result = self.mod._llm_extract_advisory_items("narrative with no json", self._make_ctx())
        assert len(result) == 1
        assert result[0]["verdict"] == "FAIL"
        assert result[0]["severity"] == "critical", (
            "FAIL without severity must be normalised to 'critical' — not left empty"
        )

    def test_fallback_returns_empty_on_llm_failure(self, monkeypatch):
        """When the LLM call raises an exception, fallback must return [] gracefully."""
        import ouroboros.llm as llm_mod

        def fake_chat_raises(self_llm, messages, model, **kwargs):
            raise RuntimeError("API unavailable")

        monkeypatch.setattr(llm_mod.LLMClient, "chat", fake_chat_raises)

        result = self.mod._llm_extract_advisory_items("some narrative text", self._make_ctx())
        assert result == []

    def test_direct_parse_success_skips_fallback(self, monkeypatch):
        """When _parse_advisory_output succeeds, _llm_extract_advisory_items must NOT be called."""
        expected = [self._item("secrets_check"), self._item("version_bump")]

        fallback_called = []
        original_fallback = self.mod._llm_extract_advisory_items

        def tracking_fallback(raw_text, ctx):
            fallback_called.append(raw_text)
            return original_fallback(raw_text, ctx)

        monkeypatch.setattr(self.mod, "_llm_extract_advisory_items", tracking_fallback)

        # _run_claude_advisory calls _parse_advisory_output first; mock run_readonly
        # to return a clean JSON array (direct parse succeeds)

        def fake_run_readonly(**kwargs):
            return _fake_native_result(
                success=True,
                result_text=json.dumps(expected),
                session_id="sess-test",
            )

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setattr(_get_advisory_module(), "_run_advisory_native",
                        lambda prompt, repo_dir, ctx_, slot, model, **_: (
                            fake_run_readonly(), model,
                        ))
        monkeypatch.setattr(self.mod, "build_advisory_changed_context",
                            lambda *a, **kw: ([], "", set()))
        monkeypatch.setattr(self.mod, "_get_staged_diff",
                            lambda *a, **kw: "diff --git a/foo.py b/foo.py")
        monkeypatch.setattr(self.mod, "_get_changed_file_list",
                            lambda *a, **kw: "M foo.py")
        monkeypatch.setattr(self.mod, "_build_advisory_prompt",
                            lambda *a, **kw: "prompt text")

        import pathlib
        items, raw, _model, _chars = self.mod._run_claude_advisory(
            pathlib.Path("/tmp"), "test commit", self._make_ctx()
        )

        assert items == expected
        assert fallback_called == [], "Fallback must NOT be called when direct parse succeeds"


class TestAdvisoryCleanSentinel:
    """The advisory prompt asks for `[] + NO_FINDINGS` as a clean verdict
    (REVIEW_JSON_ARRAY_CONTRACT). Before this contract was honored here, a
    reviewer that found nothing was recorded as `parse_failure`, which burned a
    full serial preflight suite per retry and never surfaced the real cause.
    Triad already implemented the sentinel; advisory did not.
    """

    def _run_handler(self, tmp_path, monkeypatch, raw_text):
        from unittest import mock
        from ouroboros.tools import claude_advisory_review as adv

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-for-test")
        monkeypatch.setattr(
            adv, "_run_claude_advisory",
            lambda repo_dir, commit_message, ctx, **kwargs: ([], raw_text, "opus", 10),
        )
        # Release-metadata preflight (BIBLE P9) runs before the SDK branch under
        # test, so the fake change set must carry the release artifacts.
        monkeypatch.setattr(adv, "_get_staged_diff", lambda repo_dir, paths=None: "diff --git a/x.py b/x.py")
        monkeypatch.setattr(
            adv, "_get_changed_file_list",
            lambda repo_dir, paths=None: (
                "M  x.py\nM  VERSION\nM  pyproject.toml\n"
                "M  web/package.json\nM  README.md\nM  docs/ARCHITECTURE.md"
            ),
        )
        monkeypatch.setattr(adv, "check_worktree_readiness", lambda *a, **kw: [])
        monkeypatch.setattr(adv, "_check_worktree_version_sync_shared", lambda *a, **kw: "")
        monkeypatch.setattr(adv, "compute_snapshot_hash", lambda *a, **kw: "deadbeef")

        ctx = mock.MagicMock()
        ctx.repo_dir = str(tmp_path)
        ctx.drive_root = tmp_path
        ctx.emit_progress_fn = lambda *a, **kw: None
        ctx.task_id = "t-clean"
        return json.loads(adv._handle_advisory_pre_review(ctx, commit_message="test commit"))

    def test_sentinel_qualified_empty_array_is_a_clean_run(self, tmp_path, monkeypatch):
        result = self._run_handler(tmp_path, monkeypatch, "[]\nNO_FINDINGS")
        assert result.get("status") == "fresh", f"clean advisory misclassified: {result!r}"
        assert result.get("critical_count") == 0
        assert "No findings" in (result.get("message") or "")
        assert "Fix issues" not in (result.get("message") or "")

    def test_bare_empty_array_is_clean_matching_triad(self, tmp_path, monkeypatch):
        result = self._run_handler(tmp_path, monkeypatch, "[]")
        assert result.get("status") == "fresh", f"bare [] must match triad semantics: {result!r}"

    def test_empty_array_in_refusal_prose_stays_parse_failure(self, tmp_path, monkeypatch):
        raw = "I cannot review this diff because it is too large. []"
        result = self._run_handler(tmp_path, monkeypatch, raw)
        assert result.get("status") == "parse_failure", (
            f"refusal prose must not pass as clean: {result!r}"
        )

    def test_sentinel_without_an_array_is_not_clean(self, tmp_path, monkeypatch):
        """The sentinel alone is refusal prose. Accepting it would let any
        reviewer opt out of the gate by saying the word."""
        result = self._run_handler(
            tmp_path, monkeypatch, "I cannot review this diff. NO_FINDINGS")
        assert result.get("status") == "parse_failure", (
            f"sentinel-only response must not pass as clean: {result!r}"
        )

    def test_unparseable_array_with_sentinel_is_not_clean(self, tmp_path, monkeypatch):
        result = self._run_handler(tmp_path, monkeypatch, '[{"item": broken\nNO_FINDINGS')
        assert result.get("status") == "parse_failure"


class TestEmptyArrayIsVerifiedClean:
    """The shared predicate both advisory and triad classify with. A reviewer
    must not be able to opt out of the gate by emitting the sentinel word."""

    @pytest.mark.parametrize("raw,expected", [
        ("[]\nNO_FINDINGS", True),
        ("[]", True),
        ("[] NO_FINDINGS", True),          # sentinel must never make it worse
        ("[]\r\nNO_FINDINGS", True),        # CRLF
        ("[ ]\nNO_FINDINGS", True),
        ("[]\nNO_FINDINGS\nHope that helps!", False),
        ("```json\n[]\n```", True),
        ("```json\n[]\n```\nNO_FINDINGS", True),   # fencing model puts the sentinel after the fence
        ("```\n[]\n```\nNO_FINDINGS", True),
        ("```JSON\n[]\n```", True),               # tag case must not matter
        ("```json\n[]\n```\nI cannot review this", False),
        ("prose ```[]``` NO_FINDINGS", False),
        ("```json\n[1]\n```\nNO_FINDINGS", False),
        ("I cannot review this diff. NO_FINDINGS", False),
        ("NO_FINDINGS", False),
        ("I cannot review this. [] Please retry.", False),
        ("I cannot review this diff. []\nNO_FINDINGS", False),
        ("Everything checks out.\n[]\nNO_FINDINGS", False),
        ("[] NO_FINDINGS trailing prose", False),
        ('[{"item": "x", "verdict": "FAIL"}]\nNO_FINDINGS', False),
        ('[{"item": broken\nNO_FINDINGS', False),
        ("", False),
    ])
    def test_clean_verdict_requires_a_real_empty_array(self, raw, expected):
        from ouroboros.triad_review import empty_array_is_verified_clean
        assert empty_array_is_verified_clean(raw) is expected, repr(raw)

    @pytest.mark.parametrize("raw,expected", [
        ('{"result": "[]\\nNO_FINDINGS"}', True),
        ('{"result": "[]"}', True),
        ('{"result": "I cannot review this. []\\nNO_FINDINGS"}', False),
        ("[]\nNO_FINDINGS", True),
        ("I cannot review this diff. NO_FINDINGS", False),
    ])
    def test_clean_check_sees_through_the_sdk_envelope(self, raw, expected):
        """_parse_advisory_output unwraps {"result": ...}; the clean check must
        read the same payload or the fix misses exactly the wrapped shape."""
        from ouroboros.tools.claude_advisory_review import _is_clean_verdict
        assert _is_clean_verdict(raw) is expected, repr(raw)

    def test_clean_sentinel_skips_the_paid_fallback_model(self, monkeypatch, tmp_path):
        """A sentinel-qualified clean verdict has nothing to extract, so the
        fallback extraction model must not be paid for."""
        from types import SimpleNamespace
        adv_mod = _get_advisory_module()
        called = []
        monkeypatch.setattr(adv_mod, "_llm_extract_advisory_items",
                            lambda raw, ctx: called.append(raw) or [])
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setattr(adv_mod, "_run_advisory_native",
                            lambda prompt, repo_dir, ctx_, slot, model, **_: (
                                _fake_native_result(
                                    result_text="[]\nNO_FINDINGS", session_id="s",
                                    usage={"prompt_tokens": 10, "completion_tokens": 1},
                                ), model))
        monkeypatch.setattr(adv_mod, "_get_staged_diff", lambda *a, **kw: "diff")
        monkeypatch.setattr(adv_mod, "_get_changed_file_list", lambda *a, **kw: "M f.py")
        monkeypatch.setattr(adv_mod, "build_advisory_changed_context", lambda *a, **kw: (["f.py"], "pack", []))
        monkeypatch.setattr(adv_mod, "_build_advisory_prompt", lambda *a, **kw: "prompt")
        ctx = SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path,
                              pending_events=[], emit_progress_fn=lambda *_: None)

        adv_mod._run_claude_advisory(tmp_path, "msg", ctx)

        assert called == [], "clean sentinel must not pay for fallback extraction"


class TestReviewRunFailureReason:
    """`review_status` used to drop the per-run cause, so N identical
    deterministic failures read as N generic `parse_failure` rows."""

    def _run(self, status="parse_failure", raw=""):
        from types import SimpleNamespace
        return SimpleNamespace(status=status, raw_result=raw, items=[], snapshot_hash="h",
                               commit_message="m", ts="t", snapshot_summary="s", attempt=1,
                               bypass_reason="", repo_key="", tool_name="", task_id="",
                               model_used="opus", duration_sec=48.35, prompt_chars=786401)

    def test_rejected_clean_sentinel_is_named(self):
        from ouroboros.review_evidence import _review_status_run_to_dict
        data = _review_status_run_to_dict(self._run(raw="[]\nNO_FINDINGS"))
        assert data["failure_reason"] == "clean_sentinel_rejected"

    def test_sentinel_bearing_prose_is_not_called_a_rejected_clean_verdict(self):
        """The diagnostic asks the shared predicate, so refusal prose carrying
        the sentinel is reported as prose — not as a contract regression."""
        from ouroboros.review_evidence import _review_status_run_to_dict
        data = _review_status_run_to_dict(
            self._run(raw="I cannot review this diff. []\nNO_FINDINGS"))
        assert data["failure_reason"] == "non_json_prose"

    def test_shapes_are_distinguished(self):
        from ouroboros.review_evidence import _review_status_run_to_dict
        assert _review_status_run_to_dict(self._run(raw=""))["failure_reason"] == "empty_response"
        assert _review_status_run_to_dict(self._run(raw="[{bad"))["failure_reason"] == "malformed_array"
        assert _review_status_run_to_dict(self._run(raw="sorry"))["failure_reason"] == "non_json_prose"

    def test_fresh_run_has_no_failure_reason_but_keeps_diagnostics(self):
        from ouroboros.review_evidence import _review_status_run_to_dict
        data = _review_status_run_to_dict(self._run(status="fresh", raw="[]"))
        assert data["failure_reason"] is None
        assert not any("raw" in k and k != "failure_reason" for k in data), (
            "the projection must not echo untrusted reviewer text to the model")
        assert data["duration_sec"] == 48.35
        assert data["model_used"] == "opus"
        assert data["prompt_chars"] == 786401


class TestReviewContractModes:
    """Findings-only mode offers an all-clear; required-matrix mode must not,
    because its parser rejects an empty array as missing every row."""

    def test_matrix_contract_has_no_all_clear_branch(self):
        from ouroboros.triad_review import (
            REVIEW_JSON_ARRAY_CONTRACT, REVIEW_JSON_MATRIX_CONTRACT,
        )
        assert "NO_FINDINGS" in REVIEW_JSON_ARRAY_CONTRACT
        assert "NO_FINDINGS" not in REVIEW_JSON_MATRIX_CONTRACT
        assert "one entry per required checklist item" in REVIEW_JSON_MATRIX_CONTRACT

    def test_rendered_scope_prompt_offers_no_all_clear(self):
        """The scope parser rejects an empty array as all eight items missing,
        so the prompt it is paired with must not advertise the sentinel."""
        from ouroboros.tools.review_synthesis import build_scope_review_prompt

        rendered = "".join(
            part for part in build_scope_review_prompt(
                "files", scope_checklist="cl", canonical_docs="docs",
                intent_context="intent", history_block="hist", diff_text="diff",
                repo_pack_placeholder="atlas", critical_calibration="calib",
            ) if isinstance(part, str)
        )
        assert "NO_FINDINGS" not in rendered
        assert "one entry per required checklist item" in rendered
