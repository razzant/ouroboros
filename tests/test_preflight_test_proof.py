"""Policy skips never become proofs; managed candidates run the forced suite."""

from types import SimpleNamespace

import pytest

from ouroboros.commit_admission import run_tests_preflight_with_proof
from ouroboros.tools.review_helpers import _run_review_preflight_tests


@pytest.mark.parametrize("managed,error", [(False, None), (True, None), (True, "red suite")])
def test_env_skip_and_managed_force_share_one_proof_owner(tmp_path, monkeypatch, managed, error):
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "0")
    monkeypatch.setattr("ouroboros.tools.registry._authorized_managed_update_resolver", lambda ctx: managed)
    suites, proofs = [], []
    monkeypatch.setattr("ouroboros.preflight_runner.run_hermetic_pytest", lambda *a, **kw: suites.append(kw) or error)
    monkeypatch.setattr("supervisor.update_merge.record_managed_tests_proof", lambda ctx, **kw: proofs.append(ctx))
    ctx = SimpleNamespace(repo_dir=tmp_path)
    result = run_tests_preflight_with_proof(ctx, runner=_run_review_preflight_tests)
    assert result == (error if managed else None)
    assert len(suites) == int(managed)
    assert len(proofs) == int(managed and error is None)
    assert ctx._preflight_tests_passed is bool(managed and error is None)


def test_advisory_failure_does_not_borrow_main_physical_capture(monkeypatch):
    from ouroboros.tools.preflight_review_run import _advisory_failure

    monkeypatch.setattr("ouroboros.usage_accounting.last_physical_attempt_capture", lambda: SimpleNamespace(state="unresolved", provider_status_code=None))
    result = _advisory_failure(RuntimeError("critic unavailable"), SimpleNamespace(failure_custody=lambda: {}))
    assert result.usage["operation_state"] == "settled"
    assert "physical_attempt_state" not in result.usage


@pytest.mark.parametrize("setting", ["0", "1"])
def test_actual_managed_proof_owner_reuses_one_forced_run(tmp_path, monkeypatch, setting):
    from tests.test_managed_review_subject import _managed_resolution_repo, _git
    from supervisor import update_merge
    from ouroboros.tools import git

    repo, ctx, tx = _managed_resolution_repo(tmp_path, monkeypatch)
    ctx.drive_root = tmp_path / "data"
    ctx.drive_logs = lambda: ctx.drive_root / "logs"
    ctx.emit_progress_fn = lambda *args: None
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", setting)
    suites = []
    monkeypatch.setattr("ouroboros.preflight_runner.run_hermetic_pytest", lambda *a, **kw: suites.append(kw) or None)
    if setting == "0":
        # The old lower owner already prevented a skipped run from becoming proof.
        assert update_merge.record_managed_tests_proof(ctx) == ""
        assert not getattr(ctx, "_managed_tests_proof_trees", set())
    assert git._managed_candidate_needs_proof(ctx)
    result = git._advisory_and_tests_gate(
        ctx, "managed candidate", 0,
        classification_paths=["docs/note.md"], advisory_paths=None,
        skip_advisory_pre_review=True, skip_tests=True,
    )
    assert result is None and len(suites) == 1
    evidence = update_merge.read_update_tx()["tests_evidence"]
    assert evidence["tree"] in ctx._managed_tests_proof_trees
    assert not git._managed_candidate_needs_proof(ctx)
    committed = _git(repo, "commit", "-qm", "managed candidate")
    assert committed.returncode == 0
    assert _git(repo, "rev-parse", "HEAD^{tree}").stdout.strip() == evidence["tree"]
    assert git._managed_post_commit_tests_gate(ctx, "managed candidate", 0, True, [""], tx) is None
    assert len(suites) == 1
    foreign = SimpleNamespace(task_id="foreign", task_metadata=ctx.task_metadata)
    assert update_merge.record_managed_tests_proof(foreign, force=True) == ""
    assert not getattr(foreign, "_managed_tests_proof_trees", set())
