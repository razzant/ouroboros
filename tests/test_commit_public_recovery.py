"""The public commit door collects its paid contract despite live catalog drift."""
from dataclasses import asdict
from types import SimpleNamespace
import subprocess

import pytest

from ouroboros import review_custody
from ouroboros.observability import persist_call, read_call_payload
from ouroboros.review_state import CommitAttemptRecord, load_state, make_repo_key, update_state
from ouroboros.tools import git
from ouroboros.utils import utc_now_iso
from tests import test_commit_historical_collection as historical
from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool
from tests.test_git_review_preflight_gate import candidate  # noqa: F401


MESSAGE = "fix the original candidate"
INTENT = dict(goal="keep the original goal", scope="change.py only", review_rebuttal="original evidence")
# The test execution fact the paid seats were dispatched with (``preflight_receipts`` exhibit).
DISPATCHED = {"status": "passed", "authenticity": "unsigned", "evidence": "executed", "lanes": ["serial"],
              "empty_lanes": ["parallel"], "tested": {"head": "a" * 40, "tree": "b" * 40, "index_tree": "b" * 40}}


def _git(ctx, *args):
    return subprocess.check_output(["git", *args], cwd=ctx.repo_dir, text=True).strip()


@pytest.fixture
def paid_commit(candidate, monkeypatch):  # noqa: F811
    ctx = candidate
    ctx.task_id = historical._TASK
    fingerprint = git._fingerprint_staged_diff(ctx.repo_dir)
    assert fingerprint["ok"]
    identity = SimpleNamespace(_current_review_contract_fingerprint=historical._CONTRACT)
    retry = git._commit_review_retry_key(identity, MESSAGE, binding_fingerprint=fingerprint["fingerprint"], **INTENT)
    monkeypatch.setattr(historical, "_RETRY", retry)
    sent = {"preflight_execution": DISPATCHED}
    ready, prompt = historical._seat(ctx.drive_root, ctx.repo_dir, "multi_model_review", "old-triad", evidence=sent)
    unknown, _ = historical._seat(ctx.drive_root, ctx.repo_dir, "multi_model_review", "old-unknown", complete=False,
                                  custody=False, evidence=sent)
    peer, _ = historical._seat(ctx.drive_root, ctx.repo_dir, "multi_model_review", "old-peer", evidence=sent)
    attempt = CommitAttemptRecord(
        ts=utc_now_iso(), commit_message=MESSAGE, status="reviewing", task_id=ctx.task_id,
        repo_key=make_repo_key(ctx.repo_dir), tool_name="commit_reviewed", attempt=1, paid=True,
        late_result_pending=True, root_task_id=ctx.task_id, pre_review_fingerprint=fingerprint["fingerprint"],
        review_retry_key=retry, review_contract_fingerprint=historical._CONTRACT,
        triad_raw_results=[ready, unknown, peer],
    )
    update_state(ctx.drive_root, lambda state: state.record_attempt(attempt))
    set_review_pool(monkeypatch, pool_roster(pool_seat("new-review", "new-review=new-model", kind="agent_session")))
    review_custody._ACTIVE.clear()
    review_custody._NO_RESEND.clear()

    def forbidden(*args, **kwargs):
        pytest.fail("recorded collection must not prepare, dispatch, scan history, or create a commit")

    monkeypatch.setattr(git, "_prepare_review_commit_worktree", forbidden)
    monkeypatch.setattr(git, "_preflight_and_tests_gate", forbidden)
    monkeypatch.setattr("ouroboros.review_substrate.run_review_request", forbidden)
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", forbidden)
    monkeypatch.setattr("ouroboros.claudexor_daemon.read_owned_gateway", forbidden)
    return ctx, attempt, ready, unknown, peer, prompt


def _public(ctx, **changes):
    arguments = {**INTENT, **changes}
    message = arguments.pop("commit_message", MESSAGE)
    handler = next(tool.handler for tool in git.get_tools() if tool.name == "commit_reviewed")
    return handler(ctx, commit_message=message, **arguments)


@pytest.mark.parametrize("selectors", [{}, {"reviewers": ["old-triad", "old-peer"]}, {"preflight_reviewer": "removed-preflight"}])
def test_public_commit_collects_original_contract_and_removed_reviewers(paid_commit, selectors):
    ctx, attempt, ready, unknown, peer, _prompt = paid_commit
    before = read_call_payload(ctx.drive_root, task_id=ctx.task_id, call_id="op-old-triad_prompt")
    index, head = _git(ctx, "write-tree"), _git(ctx, "rev-parse", "HEAD")
    assert not getattr(ctx, "_review_reconcile_only", False)
    result = _public(ctx, **selectors)
    rows = {row["slot_id"]: row for row in ctx._last_triad_raw_results}
    assert set(rows) == {"old-triad", "old-unknown", "old-peer"}, result
    assert rows["old-triad"]["status"] == rows["old-peer"]["status"] == "responded"
    assert rows["old-triad"]["raw_text"] == "[]"
    for source in (ready, unknown, peer):
        assert rows[source["slot_id"]]["operation_id"] == source["operation_id"]
        assert rows[source["slot_id"]]["recovery_binding"] == source["recovery_binding"]
    assert rows["old-unknown"]["operation_state"] == "custody_lost"
    assert rows["old-unknown"]["pending_invocation_id"] == unknown["pending_invocation_id"]
    assert rows["old-unknown"]["late_result_pending"] is True
    saved = load_state(ctx.drive_root).attempts
    assert len(saved) == 1 and saved[0].paid and saved[0].late_result_pending
    assert saved[0].review_retry_key == attempt.review_retry_key
    assert saved[0].review_contract_fingerprint == historical._CONTRACT
    assert _git(ctx, "write-tree") == index and _git(ctx, "rev-parse", "HEAD") == head
    assert read_call_payload(ctx.drive_root, task_id=ctx.task_id, call_id="op-old-triad_prompt") == before


@pytest.mark.parametrize("changed", ["candidate", "goal", "review_rebuttal"])
def test_public_commit_rejects_changed_candidate_or_intent_before_collection(paid_commit, monkeypatch, changed):
    ctx, _attempt, *_rest = paid_commit
    if changed == "candidate":
        (ctx.repo_dir / "change.py").write_text("value = 99\n")
        _git(ctx, "add", "change.py")
    changes = {} if changed == "candidate" else {changed: "different intent"}
    monkeypatch.setattr(git, "_run_parallel_review", lambda *a, **k: pytest.fail("foreign retry must not collect"))
    before = asdict(load_state(ctx.drive_root).attempts[0])
    result = _public(ctx, **changes)
    assert "REVIEWED_ATTEMPT_IN_PROGRESS" in result, result
    assert not ctx._last_triad_raw_results
    assert asdict(load_state(ctx.drive_root).attempts[0]) == before


@pytest.mark.parametrize("damage", ["foreign_task", "wrong_token", "missing_prompt"])
def test_public_commit_rejects_foreign_source_but_keeps_independent_paid_answer(paid_commit, damage):
    ctx, _attempt, ready, _unknown, _peer, prompt = paid_commit
    if damage == "missing_prompt":
        ready["operation_id"] = "operation-with-no-prompt"
    elif damage == "wrong_token":
        ready["pending_invocation_id"] = "foreign-token"
    else:
        if damage == "foreign_task":
            prompt["request"]["task_id"] = "another-task"
        elif damage == "foreign_slot":
            prompt["slot"]["slot_id"] = "another-slot"
        else:
            prompt["request"]["reconciliation_identity"]["roster_hash"] = "foreign-roster"
        manifest, _payload, _refs = read_call_payload(ctx.drive_root, task_id=ctx.task_id, call_id="op-old-triad_prompt")
        persist_call(ctx.drive_root, task_id=ctx.task_id, call_id="op-old-triad_prompt", call_type="multi_model_review_prompt",
                     payload=prompt, manifest={"review_operation_binding": manifest["review_operation_binding"]})
    if damage in {"wrong_token", "missing_prompt"}:
        def damage_row(state):
            state.attempts[0].triad_raw_results[0] = ready
        update_state(ctx.drive_root, damage_row)
    result = _public(ctx)
    rows = {row["slot_id"]: row for row in ctx._last_triad_raw_results}
    assert rows, result
    assert rows["old-triad"]["status"] != "responded"
    assert rows["old-triad"]["operation_state"] == "custody_lost"
    assert rows["old-triad"]["pending_invocation_id"] == ready["pending_invocation_id"]
    assert rows["old-peer"]["status"] == "responded"
    assert load_state(ctx.drive_root).attempts[0].late_result_pending


@pytest.mark.parametrize("changes", [
    {"root": "project"}, {"preflight_reviewer": "removed", "skip_advisory_review": True},
    {"preflight_reviewer": "removed", "review_reference": {}, "review_rebuttal": "",
     "author_disposition": {"disposition": "accepted", "rationale": "Known tradeoff."}},
])
def test_public_pending_collection_preserves_argument_refusals(paid_commit, monkeypatch, changes):
    ctx, *_rest = paid_commit
    monkeypatch.setattr("ouroboros.review_state.update_state", lambda *a, **k: pytest.fail("invalid arguments must not touch custody"))
    result = _public(ctx, **changes)
    assert "TOOL_ARG_ERROR" in result and not ctx._last_triad_raw_results


def test_public_commit_missing_recorded_contract_does_not_collect(paid_commit, monkeypatch):
    ctx, *_rest = paid_commit
    def lose_contract(state):
        state.attempts[0].review_contract_fingerprint = ""
    update_state(ctx.drive_root, lose_contract)
    monkeypatch.setattr(git, "_run_parallel_review", lambda *a, **k: pytest.fail("missing contract must not collect"))
    result = _public(ctx)
    assert "REVIEWED_ATTEMPT_IN_PROGRESS" in result and not ctx._last_triad_raw_results
    assert load_state(ctx.drive_root).attempts[0].review_contract_fingerprint == ""


@pytest.mark.parametrize("selectors", [{"reviewers": ["removed-reviewer"]}, {"preflight_reviewer": "removed-preflight"}])
def test_public_fresh_commit_still_refuses_removed_catalog_seat(candidate, monkeypatch, selectors):  # noqa: F811
    set_review_pool(monkeypatch, pool_roster(pool_seat("new-review", "new-review=new-model", kind="agent_session")))
    monkeypatch.setattr(git, "_prepare_review_commit_worktree", lambda *a, **k: pytest.fail("invalid new seat must not stage"))
    result = _public(candidate, **selectors)
    assert "TOOL_ARG_ERROR" in result and "not an enabled catalog row" in result
    assert not load_state(candidate.drive_root).attempts


@pytest.mark.parametrize("attribution_error", ["", "task attribution is unavailable"])
def test_public_other_task_cannot_collect_and_attribution_refusal_keeps_precedence(paid_commit, monkeypatch, attribution_error):
    ctx, attempt, *_rest = paid_commit
    ctx.task_id = "another-task"
    if attribution_error:
        monkeypatch.setattr(git, "_task_attributed_commit_paths", lambda *_a: (None, None, attribution_error, None))
    monkeypatch.setattr(git, "_run_parallel_review", lambda *a, **k: pytest.fail("another task must not collect"))
    result = _public(ctx)
    assert (attribution_error or "REVIEWED_ATTEMPT_IN_PROGRESS") in result
    saved = load_state(ctx.drive_root).attempts
    original = next(row for row in saved if row.task_id == attempt.task_id)
    assert original.paid and original.late_result_pending and original.review_retry_key == attempt.review_retry_key
    refused = next(row for row in saved if row.task_id == "another-task")
    assert refused.block_reason == ("mutation_attribution" if attribution_error else "overlap_guard")


def test_public_commit_collects_under_an_emptied_pool_with_its_saved_history_binding(paid_commit, monkeypatch):
    """An emptied catalog revokes no paid custody, and the collection restores the
    dispute input saved on the paid attempt instead of reading today's history."""
    ctx, attempt, ready, _unknown, peer, _prompt = paid_commit
    saved = {"previous_records": [], "gaps": [{"code": "REVIEW_HISTORY_SOURCE_UNAVAILABLE",
                                               "reason": "saved at the paid preparation", "source": {}}]}

    def bind(state):
        state.attempts[0].review_input = saved
    update_state(ctx.drive_root, bind)
    set_review_pool(monkeypatch, pool_roster())
    result = _public(ctx)
    rows = {row["slot_id"]: row for row in ctx._last_triad_raw_results}
    assert set(rows) == {"old-triad", "old-unknown", "old-peer"}, result
    assert rows["old-triad"]["operation_id"] == ready["operation_id"]
    assert rows["old-peer"]["status"] == "responded"
    assert ctx._review_dispute_input == saved
    from ouroboros.review_ledger import load_record, recent_records
    [emitted] = [load_record(ctx.drive_root, r["record_id"]) for r in recent_records(ctx.drive_root, ctx.task_id)]
    assert emitted["surface"] == "commit_gate" and emitted["brief"]["dispute_input"] == saved
    # The commit gate's own rejoin before its first ledger write records the test execution fact
    # frozen in the paid requests; this call's tests gate is forbidden by the fixture.
    assert emitted["preflight_execution"] == DISPATCHED
    [row] = load_state(ctx.drive_root).attempts
    assert row.review_input == saved and row.review_retry_key == attempt.review_retry_key
