"""`_advisory_and_tests_gate` recovers a pure advisory-freshness gap inline
(runs the real `preflight_review` against the current snapshot, re-checks once)
instead of bouncing the task out of the gate — but a block that reproduces on a
re-run (SyntaxError preflight, open obligations) is still returned unchanged.
"""

import importlib

import pytest


def _get_git_module():
    """Import ouroboros.tools.git fresh (mirrors tests/test_commit_gate.py)."""
    import ouroboros.tools.git as git_mod

    importlib.reload(git_mod)
    return git_mod




def test_advisory_gate_runs_advisory_inline_on_freshness_gap(monkeypatch, tmp_path):
    """when the only block is 'no fresh advisory run found',
    _advisory_and_tests_gate runs advisory_review inline against the current
    snapshot and proceeds — instead of bouncing the agent out to run it by
    hand. A real advisory record is written, so the compensating test-preflight
    coupling is unchanged (advisory_gate_unavailable stays the arbiter)."""
    git_mod = _get_git_module()

    calls = {"freshness": 0, "inline_advisory": 0, "preflight": 0}

    def fake_freshness(ctx, commit_message, skip, paths=None):
        calls["freshness"] += 1
        # First call: freshness gap. After the inline advisory runs, it clears.
        if calls["inline_advisory"] == 0:
            return (
                "⚠️ ADVISORY_PRE_REVIEW_REQUIRED: No fresh advisory run found for "
                "this snapshot (hash=abc123).\nNo advisory runs recorded yet.\n"
            )
        return None

    def fake_inline_advisory(ctx, commit_message, paths=None, skip_tests=False):
        calls["inline_advisory"] += 1
        return "{}"

    monkeypatch.setattr(git_mod, "_check_advisory_freshness", fake_freshness)
    monkeypatch.setattr(git_mod, "_handle_advisory_pre_review", fake_inline_advisory)
    monkeypatch.setattr(git_mod, "advisory_gate_unavailable", lambda: False)
    monkeypatch.setattr(git_mod, "_diff_is_doc_only", lambda paths: False)
    monkeypatch.setattr(git_mod, "_managed_candidate_needs_proof", lambda ctx: False)
    monkeypatch.setattr(
        git_mod, "_run_review_preflight_tests",
        lambda ctx: pytest.fail("preflight must not run when advisory is real"),
    )

    class FakeCtx:
        repo_dir = tmp_path
        drive_root = tmp_path

        def emit_progress_fn(self, *_a, **_k):
            pass

    result = git_mod._advisory_and_tests_gate(
        FakeCtx(),
        "test commit",
        0.0,
        classification_paths=["ouroboros/foo.py"],
        advisory_paths=["ouroboros/foo.py"],
        skip_advisory_pre_review=False,
        skip_tests=False,
    )

    assert result is None, f"gate should proceed after inline advisory, got: {result}"
    assert calls["inline_advisory"] == 1, "inline advisory_review must run exactly once"
    assert calls["freshness"] == 2, "freshness must be re-checked after the inline run"


def test_advisory_gate_does_not_retry_inline_on_syntax_preflight_block(monkeypatch, tmp_path):
    """A SyntaxError preflight block reproduces identically on a re-run, so the
    gate must NOT burn a ~2min inline advisory on it — it returns the block."""
    git_mod = _get_git_module()

    calls = {"inline_advisory": 0}

    def fake_freshness(ctx, commit_message, skip, paths=None):
        return (
            "⚠️ ADVISORY_PRE_REVIEW_REQUIRED: Last advisory run for this snapshot "
            "was blocked by the syntax preflight (hash=abc123). The Claude SDK "
            "advisory was skipped because a staged `.py` file has a SyntaxError.\n"
        )

    def fake_inline_advisory(*a, **k):
        calls["inline_advisory"] += 1
        return "{}"

    monkeypatch.setattr(git_mod, "_check_advisory_freshness", fake_freshness)
    monkeypatch.setattr(git_mod, "_handle_advisory_pre_review", fake_inline_advisory)
    monkeypatch.setattr(git_mod, "run_cmd", lambda *a, **k: "")
    monkeypatch.setattr(git_mod, "_record_commit_attempt", lambda *a, **k: None)

    class FakeCtx:
        repo_dir = tmp_path
        drive_root = tmp_path

        def emit_progress_fn(self, *_a, **_k):
            pass

    result = git_mod._advisory_and_tests_gate(
        FakeCtx(),
        "test commit",
        0.0,
        classification_paths=["ouroboros/foo.py"],
        advisory_paths=["ouroboros/foo.py"],
        skip_advisory_pre_review=False,
        skip_tests=False,
    )

    assert result is not None and result["block_reason"] == "no_advisory"
    assert calls["inline_advisory"] == 0, "must not run inline advisory for a syntax block"
