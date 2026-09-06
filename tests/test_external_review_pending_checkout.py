"""The wrapper leaves one pending review's prepared checkout alive."""

import json
from types import SimpleNamespace

import pytest

from scripts import run_external_review as runner
from ouroboros.review_state import AdvisoryRunRecord, compute_snapshot_hash, make_repo_key, update_state
from ouroboros.tools import git
from tests.test_advisory_inline_freshness import candidate  # noqa: F401


@pytest.mark.parametrize("mode", ["preflight", "wave", "exception_pending", "finished", "exception_empty"])
def test_wrapper_retains_only_unresolved_custody(candidate, monkeypatch, tmp_path, mode):  # noqa: F811
    repo = candidate.repo_dir
    (repo / "VERSION").write_text("1.0.0\n")
    output, drive = tmp_path / "output", tmp_path / "review-data"
    monkeypatch.setattr(runner, "REPO", repo)
    args = SimpleNamespace(contributor=False, commit_message="candidate", goal="", scope="",
                           output=str(output), drive_root=str(drive), no_isolated_checkout=False)
    monkeypatch.setattr(runner, "_parse_args", lambda: args)
    monkeypatch.setattr(runner, "_prepare_review_configuration", lambda args: (None, "HEAD", {}))
    monkeypatch.setattr(runner, "_advisory_unavailability_warning", lambda: "")
    monkeypatch.setattr("ouroboros.tools.claude_advisory_review._handle_advisory_pre_review", lambda *a, **kw: pytest.fail("wrapper must not pay before cycle admission"))
    created, removed = [], []
    create, remove = runner._create_isolated_checkout, runner._remove_isolated_checkout

    def capture_create(*args, **kwargs):
        paths = create(*args, **kwargs)
        created.append(paths)
        return paths

    def capture_remove(*args):
        removed.append(args)
        remove(*args)

    monkeypatch.setattr(runner, "_create_isolated_checkout", capture_create)
    monkeypatch.setattr(runner, "_remove_isolated_checkout", capture_remove)
    full_source = "complete received review\n" * 100

    def cycle(ctx, message, **kwargs):
        assert kwargs["skip_advisory_review"] is False
        (ctx.repo_dir / "late-untracked.txt").write_text("preserve without staging\n")
        if mode in {"preflight", "exception_pending"}:
            update_state(ctx.drive_root, lambda state: state.add_run(AdvisoryRunRecord(
                snapshot_hash=compute_snapshot_hash(ctx.repo_dir), commit_message=message,
                repo_key=make_repo_key(ctx.repo_dir), status="pending", ts="2026-09-06T00:00:00Z",
                raw_result=full_source, execution={"invocation_id": "inv", "pending_invocation_id": "inv", "operation_state": "in_flight"},
            )))
        if mode == "wave":
            ctx._last_triad_raw_results = [{"slot_id": "s", "operation_id": "op", "operation_state": "in_flight", "late_result_pending": True}]
        if mode.startswith("exception"):
            raise RuntimeError("cycle interrupted")
        return {"status": "blocked", "block_reason": "preflight", "message": "recorded refusal"}

    monkeypatch.setattr(git, "_run_non_committing_review_cycle", cycle)
    try:
        if mode.startswith("exception"):
            with pytest.raises(RuntimeError, match="cycle interrupted"):
                runner.main()
        else:
            assert runner.main() == 3
        _, checkout = created[0]
        pending = mode in {"preflight", "wave", "exception_pending"}
        assert checkout.exists() == pending
        assert bool(removed) == (not pending)
        if pending:
            assert "late-untracked.txt" not in runner._git_text(["diff", "--cached", "--name-only"], cwd=checkout)
            result = json.loads((output / "outcome.json").read_text())
            assert result["exit_code"] == 3
            assert result["outcome"]["retained_checkout"] == str(checkout)
            assert result["outcome"]["retained_custody"]
        if mode == "preflight":
            assert json.loads((output / "advisory.txt").read_text())["raw_result"] == full_source
        if mode == "finished":
            assert json.loads((output / "advisory.txt").read_text())["status"] == "not_run"
    finally:
        for root, checkout in created:
            if checkout.exists():
                remove(root, checkout)
