"""Owner Pause on the real managed sibling of commit_reviewed, then rejoin its tail."""
import json
from types import SimpleNamespace

import pytest

from ouroboros import owner_pause
from ouroboros.review_overrides import COUNTER_REL
from ouroboros.task_results import write_task_result
from ouroboros.tools import git
from ouroboros.tools.registry import ToolContext
from ouroboros.tools.tool_result import (
    _install_tool_result_sidecar,
    _published_tool_result,
    _restore_tool_result_sidecar,
)
from ouroboros.utils import read_json_dict
from supervisor import update_merge
from tests import test_update_merge_assisted as assisted
from tests._store_shared import read_result

# The author's explicit continuation over the returned outcome (the public envelope).
DECISION = {"disposition": "accepted", "rationale": "I read the returned outcome and accept the current bytes."}


@pytest.fixture
def managed_commit(tmp_path, monkeypatch):
    repo, branch, plan, tx = assisted._materialized_conflict_tx(tmp_path, monkeypatch)
    (repo / "VERSION").write_text("1.0.0\n")
    assert assisted._git(repo, "add", "-A").returncode == 0
    drive = tmp_path / "data"
    write_task_result(drive, "resolver", "running")
    ctx = ToolContext(repo_dir=repo, drive_root=drive, branch_dev=branch,
                      task_id="resolver", task_metadata=assisted._authority_metadata(tx))
    ctx.current_task_type = "task"
    ctx.task_lifecycle_bound = True
    effects, records = [], []
    native = git.run_cmd
    mode = {"pause": "commit", "red": ""}

    def command(argv, **kwargs):
        result = native(argv, **kwargs)
        if argv[1:2] == ["commit"]:
            effects.append("commit")
            if mode["pause"] == "commit":
                owner_pause.install_fence(drive, "resolver", request_id="after-commit")
        return result

    def stage(*_a, **_kw):
        effects.append("review")
        fingerprint = git._fingerprint_staged_diff(repo)
        assert fingerprint["ok"], fingerprint
        return {"status": "passed", "pre_fingerprint": fingerprint, "post_fingerprint": fingerprint}

    def tests(*_a, **_kw):
        effects.append("tests")
        if mode["pause"] == "tests":
            owner_pause.install_fence(drive, "resolver", request_id="after-tests")
        return "red tests" if mode["red"] == "tests" else None

    def smoke():
        effects.append("smoke")
        if mode["pause"] == "smoke":
            owner_pause.install_fence(drive, "resolver", request_id="during-smoke")
        return {"ok": mode["red"] != "smoke", "stderr": "red smoke", "returncode": 1}

    monkeypatch.setattr(git, "run_cmd", command)
    monkeypatch.setattr(git, "_check_overlapping_review_attempt", lambda *_a: "")
    monkeypatch.setattr(git, "_run_reviewed_stage_cycle", stage)
    monkeypatch.setattr(git, "_git_commit_with_tests", tests)
    monkeypatch.setattr(git, "_log_test_failure", lambda *_a: None)
    monkeypatch.setattr(git, "_record_commit_attempt", lambda _ctx, _msg, status, **kw: records.append({"status": status, **kw}))
    monkeypatch.setattr(git, "record_bound_commit_success", lambda *_a: records.append({"status": "committed"}))
    monkeypatch.setattr(update_merge, "update_restart_smoke", smoke)
    assisted._stub_worker_gates(monkeypatch)
    return SimpleNamespace(repo=repo, plan=plan, tx=tx, ctx=ctx, drive=drive,
                           effects=effects, records=records, mode=mode)


def _typed_commit(ctx):
    """The text and the typed result one commit_reviewed invocation publishes."""
    sentinel = object()
    token = _install_tool_result_sidecar(ctx, sentinel)
    try:
        return git._repo_commit_push(ctx, "reviewed managed resolution"), _published_tool_result(ctx, sentinel)
    finally:
        _restore_tool_result_sidecar(token)


@pytest.mark.parametrize("boundary", ["commit", "tests"])
def test_managed_pause_retains_commit_and_resumes_unfinished_gates(managed_commit, boundary):
    f = managed_commit
    f.mode["pause"] = boundary
    result, typed = _typed_commit(f.ctx)
    assert (typed.status, typed.code, typed.text) == ("blocked", "LEGACY_BLOCKED", result)
    commit = assisted._git(f.repo, "rev-parse", "HEAD").stdout.strip()
    assert commit != f.plan["base_sha"]
    assert "OWNER_PAUSE" in result and commit in result, result
    assert not result.startswith("OK")
    assert f.effects == (["review", "commit"] if boundary == "commit" else ["review", "commit", "tests"])
    durable = update_merge.read_update_tx()
    assert durable["merge_commit"] == commit
    assert durable["phase"] == "committing_assisted"
    assert not any(row["status"] in {"failed", "committed"} for row in f.records)
    boot = update_merge._recover_assisted_on_boot(durable, supervisor_ready=False)
    assert boot == {"finalized": False, "reason": "assisted_postcommit_paused"}, boot
    assert assisted._git(f.repo, "rev-parse", "HEAD").stdout.strip() == commit
    # A retry under the same closed fence neither runs gates nor duplicates the commit.
    assert "OWNER_PAUSE" in git._repo_commit_push(f.ctx, "reviewed managed resolution")
    owner_pause.release_fence(f.drive, "resolver", reason="owner_resume")
    f.mode["pause"] = ""
    # Recreate the context to prove the durable tx supplies the unfinished boundary.
    resumed = ToolContext(repo_dir=f.repo, drive_root=f.drive, branch_dev=f.ctx.branch_dev,
                          task_id="resolver", task_metadata=f.ctx.task_metadata)
    resumed.current_task_type = "task"
    resumed.task_lifecycle_bound = True
    result = git._repo_commit_push(resumed, "reviewed managed resolution")
    assert result.startswith("OK: committed"), result
    assert f.effects.count("commit") == f.effects.count("review") == 1
    assert f.effects[-2:] == ["tests", "smoke"]
    assert update_merge.read_update_tx()["phase"] == "pending_boot_smoke"
    assert update_merge.read_update_tx()["pre_restart_smoke"] == "passed"
    assert assisted._git(f.repo, "rev-parse", "HEAD").stdout.strip() == commit


@pytest.mark.parametrize("phase", ["tests", "smoke"])
@pytest.mark.parametrize("pause_during", [False, True])
def test_real_failed_managed_gates_keep_existing_rollback(managed_commit, phase, pause_during):
    f = managed_commit
    f.mode.update(pause=phase if pause_during else "", red=phase)
    result = git._repo_commit_push(f.ctx, "reviewed managed resolution")
    assert not result.startswith("OK"), result
    assert "rolled back" in result.lower(), result
    assert assisted._git(f.repo, "rev-parse", "HEAD").stdout.strip() == f.plan["base_sha"]
    assert not update_merge.read_update_tx()
    assert f.effects == (["review", "commit", "tests"] if phase == "tests"
                         else ["review", "commit", "tests", "smoke"])
    assert any(row["status"] == "failed" for row in f.records)
    assert not any(row["status"] == "committed" for row in f.records)


@pytest.mark.parametrize("change", ["head", "authority", "binding"])
def test_managed_resume_does_not_authorize_changed_subject_or_other_resolver(managed_commit, change):
    f = managed_commit
    assert "OWNER_PAUSE" in git._repo_commit_push(f.ctx, "reviewed managed resolution")
    commit = assisted._git(f.repo, "rev-parse", "HEAD").stdout.strip()
    owner_pause.release_fence(f.drive, "resolver", reason="owner_resume")
    f.mode["pause"] = ""
    if change == "dirty":
        (f.repo / "a.txt").write_text("later unrelated work\n")
    elif change == "head":
        assert assisted._git(f.repo, "commit", "--allow-empty", "-m", "later work").returncode == 0
    elif change == "authority":
        f.ctx.task_metadata = {}
    else:
        tx = update_merge.read_update_tx()
        tx["postcommit_resume"]["post_fingerprint"]["binding"]["tree_sha"] = "bad-tree"
        update_merge.write_update_tx(tx)
    result, typed = _typed_commit(f.ctx)
    assert "MANAGED_UPDATE" in result and not result.startswith("OK"), result
    if change != "authority":  # A changed retained subject is a failure, never a successful call.
        assert (typed.status, typed.code, typed.text) == ("error", "LEGACY_TOOL_ERROR", result)
        assert "MANAGED_UPDATE_POSTCOMMIT_CHANGED" in result
    assert f.effects == ["review", "commit"]
    assert update_merge.read_update_tx()["merge_commit"] == commit


def test_explicit_stop_of_paused_managed_resolver_keeps_existing_rollback(managed_commit):
    f = managed_commit
    assert "OWNER_PAUSE" in git._repo_commit_push(f.ctx, "reviewed managed resolution")
    result = update_merge.abort_orphaned_assisted_tx("resolver", f.ctx.task_metadata)
    assert result.get("rolled_back"), result
    assert assisted._git(f.repo, "rev-parse", "HEAD").stdout.strip() == f.plan["base_sha"]
    assert not update_merge.read_update_tx()


def test_managed_pause_before_commit_keeps_precommit_resolution(managed_commit, monkeypatch):
    f = managed_commit
    native_phase = git._managed_committing_phase_error
    def pause_before_commit(tx):
        result = native_phase(tx)
        owner_pause.install_fence(f.drive, "resolver", request_id="before-commit")
        return result
    monkeypatch.setattr(git, "_managed_committing_phase_error", pause_before_commit)
    result = git._repo_commit_push(f.ctx, "reviewed managed resolution")
    assert "OWNER_PAUSE_NOT_STARTED" in result
    assert f.effects == ["review"]
    assert not update_merge.read_update_tx().get("postcommit_resume")
    assert assisted._git(f.repo, "rev-parse", "HEAD").stdout.strip() == f.plan["base_sha"]
    assert update_merge.managed_assisted_precommit_verify(update_merge.read_update_tx())[0]


def test_managed_resume_never_reads_or_rewrites_new_commit_review_custody(managed_commit, monkeypatch):
    """The retained managed commit resumes its own gates. The NEW-commit custody lookup
    (pending-attempt flags and its stale-row expiry write) is not part of that resume."""
    from ouroboros.review_state import make_repo_key
    from ouroboros.tools.commit_gate import _check_overlapping_review_attempt

    f = managed_commit
    assert "OWNER_PAUSE" in git._repo_commit_push(f.ctx, "reviewed managed resolution")
    owner_pause.release_fence(f.drive, "resolver", reason="owner_resume")
    f.mode["pause"] = ""
    row = {"commit_message": "reviewed managed resolution", "status": "reviewing", "task_id": "resolver",
           "repo_key": make_repo_key(f.repo), "tool_name": "commit_reviewed"}
    state = f.drive / "state" / "advisory_review.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"attempts": [
        {**row, "ts": "2026-08-24T00:00:00Z", "attempt": 4, "paid": True, "review_retry_key": "commit_review:exact"},
        # An unpaid legacy row the lookup would expire, i.e. rewrite the store.
        {**row, "ts": "2020-01-01T00:00:00Z", "attempt": 1, "task_id": "other-task"},
    ]}), encoding="utf-8")
    before = state.read_bytes()
    monkeypatch.setattr(git, "_check_overlapping_review_attempt", _check_overlapping_review_attempt)
    result = git._repo_commit_push(f.ctx, "reviewed managed resolution")
    assert result.startswith("OK: committed"), result
    assert state.read_bytes() == before
    assert not getattr(f.ctx, "_review_resume_pending", False)
    assert getattr(f.ctx, "_current_review_attempt_number", None) is None
    assert f.effects.count("commit") == f.effects.count("review") == 1


def _real_review_binding(f, monkeypatch):
    """The review this managed commit continues over: a real NOT_DISPATCHED ledger record bound
    to a real durable attempt, and the review cycle's own binding of that record."""
    from ouroboros import review_ledger as ledger
    from ouroboros.review_records import RECORDED_REVIEW_AUTHORITY, build_author_disposition_from_mapping
    from ouroboros.review_state import load_state, make_repo_key
    from ouroboros.tools import commit_gate

    ctx = f.ctx
    # The real writers: the landing annotation on the record and the counter row are the effect
    # under proof, so neither is mocked here.
    monkeypatch.setattr(git, "record_bound_commit_success", commit_gate.record_bound_commit_success)
    monkeypatch.setattr(git, "_record_commit_attempt", commit_gate._record_commit_attempt)
    commit_gate._record_commit_attempt(ctx, "reviewed managed resolution", "reviewing", phase="review")
    attempt = ctx._current_review_attempt_number
    commit_gate.settle_commit_review_ledger(
        ctx, "reviewed managed resolution", pre_fingerprint={}, blocked=False, block_reason="",
        combined_findings=[], advisory_replay={"replay_reason": "free_replay",
                                                "advisory_replay": "No new review wave was bought."})
    record_id = ctx._current_review_record_id
    assert record_id, "the review wrote a record"
    source = load_state(ctx.drive_root).latest_attempt_for(
        repo_key=make_repo_key(ctx.repo_dir), task_id=ctx.task_id, tool_name="commit_reviewed",
        attempt=attempt)
    assert source is not None and source.review_record_id == record_id, source
    reference = {"surface": "commit", "repo_key": make_repo_key(ctx.repo_dir), "task_id": ctx.task_id,
                 "tool_name": "commit_reviewed", "attempt": attempt, "review_record_id": record_id,
                 "pre_review_fingerprint": source.pre_review_fingerprint}

    def stage(ctx, commit_message, commit_start, **_kw):
        f.effects.append("review")
        fingerprint = git._fingerprint_staged_diff(f.repo)
        assert fingerprint["ok"], fingerprint
        # What the review cycle does for an author continuation: the validated reference, the
        # decision, and the record it answers.
        ctx._author_commit_source = source
        ctx._author_commit_reference = dict(reference)
        ctx._author_commit_decision = dict(DECISION)
        # The author's decision on that record (the production builder; the release-metadata
        # preflight that follows it is not part of this seam).
        ctx._author_commit_record = build_author_disposition_from_mapping(
            DECISION, subject_hash=fingerprint["fingerprint"],
            reviewer_signal=str((ledger.load_record(ledger.ledger_root(ctx), record_id) or {})
                                .get("verdict", {}).get("aggregate") or ""),
            enforcement=RECORDED_REVIEW_AUTHORITY)
        commit_gate.settle_commit_review_ledger(
            ctx, commit_message, pre_fingerprint=fingerprint, blocked=False, block_reason="",
            combined_findings=[], author_source=source)
        assert ctx._current_review_record_id == record_id, ctx._current_review_record_id
        return {"status": "passed", "pre_fingerprint": fingerprint, "post_fingerprint": fingerprint}

    monkeypatch.setattr(git, "_run_reviewed_stage_cycle", stage)
    return record_id, source


def _paused_then_resumed(f, monkeypatch):
    """The real pause after Git and a fresh worker's resume of that exact retained commit."""
    f.mode["pause"] = "commit"
    result = git._repo_commit_push(f.ctx, "reviewed managed resolution")
    assert "OWNER_PAUSE" in result, result
    commit = assisted._git(f.repo, "rev-parse", "HEAD").stdout.strip()
    owner_pause.release_fence(f.drive, "resolver", reason="owner_resume")
    f.mode["pause"] = ""
    resumed = ToolContext(repo_dir=f.repo, drive_root=f.drive, branch_dev=f.ctx.branch_dev,
                          task_id="resolver", task_metadata=f.ctx.task_metadata)
    resumed.current_task_type = "task"
    resumed.task_lifecycle_bound = True
    result = git._repo_commit_push(resumed, "reviewed managed resolution")
    assert result.startswith("OK: committed"), result
    return resumed, commit


def test_managed_resume_lands_on_the_review_record_it_continued_over(managed_commit, monkeypatch):
    """#3451: a retained managed commit resumed by a fresh worker lands on the review record its
    author continuation answered, with that outcome, once — the old verdict stands."""
    from ouroboros import review_ledger as ledger

    f = managed_commit
    record_id, _source = _real_review_binding(f, monkeypatch)
    before = ledger.load_record(ledger.ledger_root(f.ctx), record_id)
    assert before["verdict"]["aggregate"] == "NOT_DISPATCHED", before["verdict"]
    assert not before.get("commit_landings"), before.get("commit_landings")

    resumed, commit = _paused_then_resumed(f, monkeypatch)

    after = ledger.load_record(ledger.ledger_root(resumed), record_id)
    assert after["verdict"] == before["verdict"], "the old verdict stands"
    assert [row["commit_sha"] for row in after.get("commit_landings") or []] == [commit], after.get("commit_landings")
    counter = read_json_dict(f.drive / COUNTER_REL) or {}
    assert [row["sha"] for row in counter.get("landed") or []] == [commit], counter
    expected = str((before.get("dispatch_refusal") or {}).get("kind") or "")
    assert expected, before.get("dispatch_refusal")
    assert counter["landed"][0]["reason"] == expected, counter
    fact = (read_result(f.drive, "resolver") or {}).get("review_overrides") or {}
    assert fact["landed"]["count"] == 1 and fact["landed"]["latest"]["sha"] == commit, fact


def test_managed_resume_without_the_receipt_fields_attributes_nothing(managed_commit, monkeypatch):
    """A receipt written before these fields stays unknown: no record, no outcome, no row."""
    from ouroboros import review_ledger as ledger

    f = managed_commit
    record_id, _source = _real_review_binding(f, monkeypatch)
    f.mode["pause"] = "commit"
    assert "OWNER_PAUSE" in git._repo_commit_push(f.ctx, "reviewed managed resolution")
    commit = assisted._git(f.repo, "rev-parse", "HEAD").stdout.strip()
    tx = update_merge.read_update_tx()
    tx["postcommit_resume"]["review_context"].pop("_current_review_record_id", None)
    tx["postcommit_resume"]["review_context"].pop("_landing_override_reason", None)
    update_merge.write_update_tx(tx)
    owner_pause.release_fence(f.drive, "resolver", reason="owner_resume")
    f.mode["pause"] = ""
    resumed = ToolContext(repo_dir=f.repo, drive_root=f.drive, branch_dev=f.ctx.branch_dev,
                          task_id="resolver", task_metadata=f.ctx.task_metadata)
    resumed.current_task_type = "task"
    resumed.task_lifecycle_bound = True
    assert git._repo_commit_push(resumed, "reviewed managed resolution").startswith("OK: committed")

    after = ledger.load_record(ledger.ledger_root(resumed), record_id)
    assert not after.get("commit_landings"), after.get("commit_landings")
    counter = read_json_dict(f.drive / COUNTER_REL) or {}
    assert not counter.get("landed"), counter
    assert assisted._git(f.repo, "rev-parse", "HEAD").stdout.strip() == commit


def test_managed_resume_keeps_its_landing_outcome_out_of_the_next_call(managed_commit, monkeypatch):
    """#3451: the outcome a resumed managed commit landed over belongs to THAT call.

    The next commit call on the same ToolContext starts without it: its own per-call reset
    clears the field the resume restored it into, so no later landing is attributed to the
    old review outcome.
    """
    from ouroboros import review_ledger as ledger
    from ouroboros.review_overrides import landing_reason

    f = managed_commit
    record_id, _source = _real_review_binding(f, monkeypatch)
    before = ledger.load_record(ledger.ledger_root(f.ctx), record_id)
    expected = str((before.get("dispatch_refusal") or {}).get("kind") or "")
    assert expected, before.get("dispatch_refusal")

    resumed, commit = _paused_then_resumed(f, monkeypatch)

    # The resumed call landed over the outcome the receipt carried, and only over that one.
    counter = read_json_dict(f.drive / COUNTER_REL) or {}
    assert [row["sha"] for row in counter.get("landed") or []] == [commit], counter
    assert counter["landed"][0]["reason"] == expected, counter
    assert landing_reason(resumed) == expected, "the restored outcome is in force for this call"

    # The next commit call on the SAME context.
    result = git._repo_commit_push(resumed, "later work")
    assert not result.startswith("OK"), result
    assert landing_reason(resumed) == "", "the resumed outcome did not cross into the next call"

    # And a landing on that later call records nothing over the old outcome.
    git.note_override_landing(resumed, "0" * 40)
    counter = read_json_dict(f.drive / COUNTER_REL) or {}
    assert [row["sha"] for row in counter.get("landed") or []] == [commit], counter
