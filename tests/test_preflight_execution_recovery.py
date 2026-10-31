"""C2 refusal and crash-before-ledger exhibits through their real consumers."""

import copy
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import preflight_receipts as receipts
from ouroboros import review_ledger
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests import _contributor_packet_shared as shared
from tests.test_preflight_receipts_consumers import _foreign_review, _recording
from tests.test_preflight_receipts import _git, _tiny_repo
from tests.test_review_change_end_to_end import (
    _brief_text, _paid_rows, _pending_then_answering_seat, staged_body,  # noqa: F401
)

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("refusal", [
    "⚠️ PREFLIGHT_BLOCKED: staged release carriers disagree",
    "⚠️ PREFLIGHT_UNAVAILABLE: staged README could not be read",
], ids=["preflight", "infra"])
@pytest.mark.parametrize("prior_proof", ["absent", "matching", "stale"])
@pytest.mark.parametrize("skip_tests", [False, True], ids=["run", "skip"])
def test_deterministic_refusal_reaches_every_seat_without_starting_tests(
        staged_body, tmp_path, monkeypatch, refusal, prior_proof, skip_tests):  # noqa: F811
    import ouroboros.review_substrate as substrate
    from ouroboros.commit_admission import preflight_test_proof_matches
    from ouroboros.tools import commit_gate, git
    from ouroboros.tools.git_review_cycle import _run_reviewed_stage_cycle

    repo = Path(staged_body["repo"])
    # Unlike the fixture's Git helper, the real commit consumer uses repository identity.
    shared.git(repo, "config", "user.name", "Preflight fixture")
    shared.git(repo, "config", "user.email", "preflight@example.invalid")
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "gate-drive")
    if prior_proof != "absent":
        # Model a context that still owns a prior attestation. The selected gate
        # below must not mistake it for an invocation that got past preflight.
        # A worker mints it after its registry imported every tool module, so the browser
        # module's PLAYWRIGHT_BROWSERS_PATH default is already in the environment the proof
        # binds; a first registry built by the review below would change it (#2363).
        ToolRegistry(repo_dir=repo, drive_root=ctx.drive_root)
        shared.passing_test_runner(ctx)
        if prior_proof == "stale":
            (repo / "ouroboros" / "helper.py").write_text("VALUE = 3\n", encoding="utf-8")
        assert preflight_test_proof_matches(ctx, repo) is (prior_proof == "matching")
    proof = getattr(ctx, "_preflight_test_proof", None)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    checked, launched, briefs, evidence = [], [], [], []

    def refuse(*args, **kwargs):
        checked.append(args)
        return refusal

    def no_launch(*args, **kwargs):
        launched.append(args)
        pytest.fail("deterministic refusal must not start tests")

    monkeypatch.setattr(commit_gate, "deterministic_preflight", refuse)
    monkeypatch.setattr(git, "_run_review_preflight_tests", no_launch)
    monkeypatch.setattr(substrate, "run_review_request", _recording(shared.golden_substrate(briefs), evidence))

    head = shared.git(repo, "rev-parse", "HEAD")
    git._reset_commit_review_state(ctx)
    outcome = _run_reviewed_stage_cycle(ctx, "fix: proposal", time.time(), skip_advisory_review=True,
                                        skip_tests=skip_tests, goal="Ship the proposal", scope="helper.py")

    # Cyber continues to the real panel, whose received outcome returns to its author before any Git effect.
    assert outcome["status"] == "reviewed", outcome
    assert outcome["message"].startswith("Review outcome returned before commit."), outcome["message"]
    assert shared.git(repo, "rev-parse", "HEAD") == head, "no commit, tag or push before the author decides"
    reference = outcome["review_reference"]
    assert (reference["surface"], reference["tool_name"]) == ("commit", "commit_reviewed") and reference["review_record_id"]
    assert len(checked) == 1 and not launched
    assert getattr(ctx, "_preflight_test_proof", None) is proof  # do not destroy a prior proof
    record = review_ledger.load_record(ctx.drive_root, reference["review_record_id"])
    exhibit = record["preflight_execution"]
    assert exhibit["status"] == "not_run"
    # The same record's tests fact follows the exhibit this attempt froze, not the kept proof.
    assert record["tests"] == {"policy": "NOT_RUN", "result": "unknown", **(
        {"reason": "tests_not_started"} if prior_proof == "matching" else
        {"reason": "tests_proof_not_for_this_candidate"} if prior_proof == "stale" else {})}
    assert refusal in exhibit["reason"] and "not started" in exhibit["reason"]
    assert "skipped by the author" not in exhibit["reason"] and "suite failed" not in exhibit["reason"]
    assert sorted(brief["slot_id"] for brief in briefs) == ["s1", "t1", "t2"]
    assert all(_brief_text(brief).count(receipts.execution_section(exhibit)) == 1 for brief in briefs)
    assert len(evidence) == 3 and all(item["preflight_execution"] == exhibit for item in evidence)
    # Only the author's explicit continuation commits: the same record, no new critic, the refusal rechecked
    # and still disclosed, and still no test started.
    shared.git(repo, "add", "-A")
    committed = git._commit_reviewed(ctx, "fix: proposal", skip_advisory_review=True, skip_tests=skip_tests,
                                     goal="Ship the proposal", scope="helper.py", review_reference=reference,
                                     author_disposition={"disposition": "accepted",
                                                         "rationale": "Read all three seats; the refusal is known."})
    if prior_proof == "stale":  # the continuation's current check remains visible to its Cyber author
        assert "PREFLIGHT_BLOCKED" in committed and "ouroboros/helper.py" in committed, committed
    assert shared.git(repo, "rev-parse", "HEAD") != head and ctx._commit_review_status == "author_continued", committed
    assert f"review_record_id: {reference['review_record_id']}" in committed and refusal in committed
    assert len(briefs) == 3 and len(checked) == 2 and not launched
    assert ctx._preflight_execution["status"] == "not_run" and refusal in ctx._preflight_execution["reason"]
    assert review_ledger.load_record(ctx.drive_root, reference["review_record_id"])["preflight_execution"] == exhibit


@pytest.mark.parametrize("outcome", ["green", "skip", "failed", "skip_after_green", "unbound", "skip_after_unbound",
                                     "managed_unbound"])
def test_nonrefused_gate_keeps_real_test_and_skip_exhibits(tmp_path, monkeypatch, outcome):
    from ouroboros import commit_admission, preflight_runner
    from ouroboros.tools import commit_gate, git

    repo = _tiny_repo(tmp_path)
    # This fixture reaches the real runner; release policy is exercised separately.
    monkeypatch.setattr(commit_admission, "release_metadata_diagnostics", lambda *a, **kw: {
        "status": "not_applicable", "findings": [], "unavailable": []})
    if outcome == "failed":
        (repo / "tests" / "test_parallel.py").write_text("def test_red():\n    assert False\n", encoding="utf-8")
        _git(repo, "add", "-A")
    if outcome.endswith("unbound"):  # real green lanes whose tested checkout cannot be bound mint no proof
        monkeypatch.setattr(commit_admission, "capture_preflight_test_subject", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "1")
    monkeypatch.setenv("OUROBOROS_PREFLIGHT_TEST_WORKERS", "2")
    monkeypatch.setenv("OURO_TEST_WORKER_BUDGET", "2")  # nested slots: the gate must really run two workers
    execute, lanes = preflight_runner._execute_pytest_pass, []

    def counted(*args):
        lanes.append(tuple(args[3]))
        return execute(*args)

    monkeypatch.setattr(preflight_runner, "_execute_pytest_pass", counted)
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path / "gate-drive")
    if outcome == "managed_unbound":
        monkeypatch.setattr(git, "_managed_candidate_needs_proof", lambda _ctx: True)
    for skip_tests in ([False] if outcome.startswith("skip_after") else []) + [outcome.startswith(("skip", "managed"))]:
        assert git._preflight_and_tests_gate(ctx, "candidate", time.time(),
                                             classification_paths=["tests/test_parallel.py"], skip_tests=skip_tests) is None
    assert len(lanes) == {"green": 2, "skip": 0, "failed": 1, "skip_after_green": 2, "unbound": 2,
                          "skip_after_unbound": 2, "managed_unbound": 2}[outcome]
    tests = commit_gate._review_tests_facts(ctx, getattr(ctx, "_preflight_tests_passed", None))
    if outcome in ("green", "skip_after_green"):  # a skip still shows a kept proof of this candidate
        assert ctx._preflight_execution["status"] == "passed"
        assert ctx._preflight_execution["lanes"] == ["parallel", "serial"]
        assert tests == {"policy": "run", "result": "passed", "proof": "candidate_bound"}
    else:  # the author's skip outranks a completed run that minted no proof
        reason = {"failed": "tests preflight did not pass",
                  "unbound": "the run completed but minted no proof of the tested subject",
                  "managed_unbound": "the run completed but minted no proof of the tested subject"}.get(
            outcome, "skipped by the author (skip_tests)")
        exhibit = dict(ctx._preflight_execution)
        observed = exhibit.pop("observed", [])
        assert exhibit == {"status": "not_run", "reason": reason}
        assert tests == {"policy": "NOT_RUN", "result": "unknown", **(
            {"reason": "tests_proof_not_for_this_candidate"} if outcome.endswith("unbound") else {})}
        section = receipts.execution_section(ctx._preflight_execution)
        if outcome != "failed":  # nothing ran red: the compatible NOT_RUN line stays the whole fact
            assert not observed and section.endswith(
                f"NOT_RUN: {reason}. No passing test result is claimed for this candidate.")
            return
        # The red run reaches every seat as what it was, on these bytes, never as "not run".
        [run] = observed
        assert (run["outcome"], run["relation"], run["tested"]["head"]) == (
            "failed", "same_bytes", _git(repo, "rev-parse", "HEAD"))
        assert run["evidence"] == ctx._preflight_test_evidence and run["terminal"] == "read"
        assert "test_red" in run["diagnosis"] and "assert False" in run["diagnosis"]
        assert "FAILED: a lane exited red, on the bytes of this candidate" in section
        assert "NOT_RUN names the missing proof only; the runs and checks below did happen." in section
        assert "test_red" in section and "never proof" in section


@pytest.mark.parametrize("legacy", [False, True], ids=["original-exhibit", "legacy-absent"])
def test_crash_before_first_ledger_write_rejoins_the_original_request_exhibit(tmp_path, monkeypatch, legacy):
    import ouroboros.review_substrate as substrate
    from ouroboros import review_custody
    from ouroboros.observability import read_call_payload
    from ouroboros.tools import review_change
    from ouroboros.tools.review_change_custody import recorded_commit_assignment

    ctx, project, original = _foreign_review(tmp_path, monkeypatch)
    later = {**original, "evidence": "receipt", "receipt": "f" * 64,
             "receipt_origin": {"task_id": "later-call"}}
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    sends, requests = [], []
    answer, starts = _pending_then_answering_seat("t2", "invocation-before-ledger")
    physical = shared.golden_physical_seam(sends, answer=answer)

    def transport(self, request, slot, **kwargs):
        if legacy:
            request.evidence.pop("preflight_execution", None)  # an actual old request without the field
        requests.append((slot.slot_id, copy.deepcopy(request)))
        return physical(self, request, slot, **kwargs)

    def crash(*args, **kwargs):
        raise SystemExit("crash before first ledger write")

    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", transport)
    ask = dict(subject="index", goal="Bump", scope="app.py")
    with monkeypatch.context() as fault:
        fault.setattr(review_change, "_settle", crash)
        with pytest.raises(SystemExit, match="crash before first ledger write"):
            review_change.run_review_change(ctx, **ask, _preflight_execution=None if legacy else original)

    [paid] = _paid_rows(ctx.drive_root, project)
    assert not review_ledger.load_record(ctx.drive_root, paid.review_record_id)
    assert len(sends) == 3 and len(starts) == 1
    assert all(row.get("operation_id") and not row.get("prompt_ref") for row in paid.triad_raw_results)
    collector = SimpleNamespace(**vars(ctx))
    collector._pending_review_attempt = paid
    collector._current_review_retry_key = paid.review_retry_key
    retained = {}
    for row in paid.triad_raw_results:
        request, slot = recorded_commit_assignment(collector, "multi_model_review", row)
        retained[slot.slot_id] = (row["operation_id"], copy.deepcopy(request.evidence))
        if legacy:
            assert "preflight_execution" not in request.evidence
        else:
            assert request.evidence["preflight_execution"] == original
        # These bytes were sent to each seat before the crash, not rebuilt by rejoin.
        sent = next(item for seat, item in requests if seat == slot.slot_id)
        assert sent.retry_key == request.retry_key and sent.evidence == request.evidence

    # Drop process-local custody and supply a conflicting newer fact on a fresh ctx.
    review_custody._ACTIVE.clear()
    review_custody._NO_RESEND.clear()
    restarted = ToolContext(repo_dir=ctx.repo_dir, system_repo_dir=ctx.system_repo_dir, drive_root=ctx.drive_root,
                            workspace_root=project, workspace_mode="external", task_id=ctx.task_id)
    result = review_change.run_review_change(restarted, **ask, _preflight_execution=later)

    assert (result["state"], result["record_id"]) == ("settled", paid.review_record_id), result
    assert len(sends) == 4 and sends[-1]["reconcile_only"]
    assert sends[-1]["operation_id"] == retained["t2"][0]
    assert sends[-1]["retry_state"] == {"pending_invocation_id": "invocation-before-ledger"}
    assert len(starts) == 1  # only the exact invocation was collected; no paid resend
    [settled] = _paid_rows(ctx.drive_root, project)
    assert settled.review_record_id == paid.review_record_id and not settled.late_result_pending
    record = review_ledger.load_record(ctx.drive_root, result["record_id"])
    assert record["preflight_execution"] == ({"status": "unobserved"} if legacy else original)
    for operation, evidence in retained.values():
        _manifest, payload, _refs = read_call_payload(ctx.drive_root, task_id=ctx.task_id,
                                                     call_id=operation + "_prompt")
        assert payload["request"]["evidence"] == evidence
