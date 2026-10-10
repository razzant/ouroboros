"""S1 qualification of the current preflight -> review_change custody consumer.

The public preflight handler, review wave, paid-attempt/operation records, physical
attempt accounting, Pause and warm Resume are real. Git subject materialization
and prompt preparation are inert (the existing direct_commit fixture); the fake
provider finishes one bounded two-round review. No Git history or provider is used.
The fixture is deliberately not evidence of Git checkout or OS-worker recovery.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import contextmanager

import pytest

from ouroboros import owner_pause, review_ledger, review_pause
from ouroboros.owner_wait import consume_warm_resume
from ouroboros.review_state import load_state, make_repo_key
from ouroboros.tools import preflight_review, review_change
from ouroboros.tools.registry import ToolRegistry
from ouroboros.tools.review_subject import FrozenSubject
from supervisor.budget_resume import resume_warm_owner_pause_root
from supervisor.owner_pause_control import refresh_owner_pause_tree, request_owner_pause
from tests._budget_pause_exact_helpers import _install_queue
from tests._usage_store_testing import attempt_rows_in_start_order
from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool
from tests.test_pause_review_completion import direct_commit, paused_env  # noqa: F401
from tests.test_review_operation_lifetime import TASK, _parent, env, until  # noqa: F401

pytestmark = pytest.mark.serial
REVIEWER = "commit-reviewer"


@pytest.fixture
def preflight(direct_commit, monkeypatch):  # noqa: F811
    d = direct_commit
    set_review_pool(monkeypatch, pool_roster(pool_seat(REVIEWER, "m", delivery="native", effort="high")))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    d.f.ctx.root_task_id = TASK
    d.f.ctx.system_repo_dir = d.f.ctx.repo_dir
    d.target = d.f.ctx.repo_dir / "candidate.py"
    d.target.write_text("value = 2\n", encoding="utf-8")
    d.subjects, d.retentions = [], []
    monkeypatch.setattr(review_change, "resolve_review_root", lambda ctx, _binding, _request:
                        ("system_repo", ctx.repo_dir))
    monkeypatch.setattr(review_change, "_governance_repo", lambda ctx: ctx.repo_dir)

    @contextmanager
    def subject(ctx, spec, *, retain, token):
        patch = b"--- a/candidate.py\n+++ b/candidate.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
        identity = FrozenSubject(spec, patch.decode(), hashlib.sha256(patch).hexdigest(),
                                 "2" * 40, "1" * 40, name_status=(("M", "candidate.py"),))
        # Only materialization is replaced; the production round/task token and
        # frozen subject feed the actual paid/retry/reuse identities below.
        checkout = ctx.drive_root / "fixture-checkouts" / token(identity)
        checkout.mkdir(parents=True, exist_ok=True)
        from dataclasses import replace
        frozen = replace(identity, checkout=str(checkout))
        d.subjects.append(frozen)
        try:
            yield frozen
        finally:
            d.retentions.append(dict(retain()))

    monkeypatch.setattr(review_change, "isolated_checkout", subject)
    _queue, state, d.workers = _install_queue(d.f.root, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5.0)
    d.workers.RUNNING[TASK] = {
        "task": {"id": TASK, "type": "task", "chat_id": 7, "root_task_id": TASK},
        "worker_id": 0, "attempt": 1,
    }
    # An actual registry consumer gives apply_patch the same author identity.
    d.registry = ToolRegistry(repo_dir=d.f.ctx.repo_dir, drive_root=d.f.root)
    d.registry._ctx.task_id = d.registry._ctx.root_task_id = TASK
    yield d
    d.model.release_first.set()


def _look(d):
    with _parent(d.f):
        return json.loads(preflight_review._handle_preflight_review(
            d.f.ctx, reviewer=REVIEWER, goal="Qualify the candidate", scope="candidate.py"))


def _start(d):
    result = {}

    def call():
        try:
            result["value"] = _look(d)
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=call, name="preflight-author")
    thread.start()
    return thread, result


def _returned(thread, result):
    until(lambda: bool(result))
    thread.join(10)
    assert not thread.is_alive()
    if "error" in result:
        raise result["error"]
    return result["value"]


def _paid(d):
    return [row for row in load_state(d.f.root).attempts
            if row.task_id == TASK and row.tool_name == "review_change"
            and row.repo_key == make_repo_key(d.f.ctx.repo_dir)]


def _pause(d):
    ack = request_owner_pause(TASK, request_id="preflight-pause")
    assert ack["ok"] and ack["state"] == "requested", ack
    return ack


def _park(d, ack):
    # The worker-row seam used by the existing warm-pause tests. No second
    # assignment is made: the one author retains its worker/attempt identity.
    d.workers.RUNNING[TASK]["owner_wait"] = {
        "state": "waiting", "reason": "owner_pause", "wait_id": "preflight-warm",
        "owner_pause": {"fence_id": ack["fence_id"]},
    }
    assert refresh_owner_pause_tree(TASK) == "paused"
    fence = owner_pause.read_fence(d.f.root, TASK)
    assert fence["warm_members"] == [TASK] and TASK in fence["finishing_reviews"]


def _resume(d):
    resumed = resume_warm_owner_pause_root(TASK)
    assert resumed["ok"] and resumed["warm"], resumed
    assert consume_warm_resume(d.f.ctx)
    consumed = owner_pause.read_fence(d.f.root, TASK)["resume_grant"]["consumed_at"]
    assert consumed and consume_warm_resume(d.f.ctx)
    assert owner_pause.read_fence(d.f.root, TASK)["resume_grant"]["consumed_at"] == consumed
    assert list(d.workers.RUNNING) == [TASK]


def _author_effects_refused(d):
    # Call the real final-commit admission and actual apply_patch registry entry.
    # The commit command is inert even if this assertion fails.
    sha, refusal = d.git._commit_reviewed_candidate(d.f.ctx, "must stay paused", time.time(), None)
    assert sha == "" and "OWNER_PAUSE_NOT_STARTED" in refusal, (sha, refusal)
    result = d.registry.execute_result("apply_patch", {
        "root": "system_repo",
        "patch": "*** Begin Patch\n*** Update File: candidate.py\n@@\n-value = 2\n+value = 3\n*** End Patch",
    })
    assert result.meta.get("owner_pause_not_started"), result
    assert d.target.read_text(encoding="utf-8") == "value = 2\n"
    assert not any(row[0] in {"tag", "push"} or row[1:2] == ("commit",) for row in d.effects), d.effects


@pytest.mark.parametrize("resume_before_settlement", [True, False], ids=["rejoin-live", "collect-finished"])
def test_preflight_pause_detaches_and_resume_collects_the_same_paid_operation(preflight, resume_before_settlement):
    d = preflight
    thread, result = _start(d)
    rejoin_thread, rejoin_result = None, None
    try:
        assert d.model.first_sent.wait(10)
        ack = _pause(d)
        first = _returned(thread, result)
        assert first["state"] == "pending" and first["aggregate"] != "PASS", first
        [paid] = _paid(d)
        [actor] = paid.triad_raw_results
        operation_id = actor["operation_id"]
        assert actor["operation_state"] == "in_flight" and actor["late_result_pending"]
        assert review_pause.live_detached_operations(TASK) and len(d.model.rounds) == 1
        assert d.retentions[-1]["seats"] == [REVIEWER]
        _park(d, ack)
        _author_effects_refused(d)
        initial_revision = review_ledger.load_record(d.f.root, first["record_id"])["revision"]

        if resume_before_settlement:
            _resume(d)
            assert review_pause.live_detached_operations(TASK), "Resume did not wait for the critic"
            rejoin_thread, rejoin_result = _start(d)
            until(lambda: getattr(d.f.ctx, "_review_reconcile_only", False))
            assert not rejoin_result and len(d.model.rounds) == 1, "live rejoin must not repurchase"
        d.model.release_first.set()
        until(lambda: not review_pause.live_detached_operations(TASK))
        if not resume_before_settlement:
            # Late PASS has no right to run the old author's commit/apply tail.
            _author_effects_refused(d)
            _resume(d)
            rejoin_thread, rejoin_result = _start(d)
        collected = _returned(rejoin_thread, rejoin_result)
        assert (collected["state"], collected["aggregate"], collected["record_id"]) == (
            "settled", "PASS", first["record_id"]), collected
        [settled] = _paid(d)
        assert settled.attempt == paid.attempt and not settled.late_result_pending
        assert settled.triad_raw_results[0]["operation_id"] == operation_id
        record = review_ledger.load_record(d.f.root, first["record_id"])
        assert record["surface"] == "preflight" and record["revision"] > initial_revision
        assert [label for label, _ in d.model.rounds] == ["first", "follow-up"]
        attempts = attempt_rows_in_start_order(d.f.root)
        assert len(attempts) == 2 and all(row["state"] == "settled" for row in attempts)
        assert sum(row["cost_usd"] for row in attempts) == pytest.approx(0.02)
        replay = _look(d)
        assert replay["reused"] and replay["record_id"] == first["record_id"]
        assert replay["cost"] == {"usd": 0.0, "unknown": False}
        assert attempt_rows_in_start_order(d.f.root) == attempts and len(_paid(d)) == 1
        assert not any(row[0] in {"tag", "push"} or row[1:2] == ("commit",) for row in d.effects)
        assert d.target.read_text(encoding="utf-8") == "value = 2\n"
        print("PREFLIGHT_QUALIFICATION " + json.dumps({
            "resume_before_settlement": resume_before_settlement, "record_id": first["record_id"],
            "operation_id": operation_id, "revision": record["revision"], "paid_review_rows": 1,
            "physical_attempt_ids": [row["attempt_id"] for row in attempts],
            "physical_cost_usd": sum(row["cost_usd"] for row in attempts),
            "author_commit_apply_effects": 0, "replay_usd": replay["cost"]["usd"],
        }, sort_keys=True))
    finally:
        d.model.release_first.set()
        thread.join(10)
        if rejoin_thread is not None:
            rejoin_thread.join(10)


def test_preflight_unsent_preparation_stays_unsent_after_rapid_pause_resume(preflight, monkeypatch):
    d = preflight
    preparing, release = threading.Event(), threading.Event()
    slot_episode = review_pause.slot_episode

    @contextmanager
    def delayed_episode(entry, scope):
        preparing.set()
        assert release.wait(10)
        with slot_episode(entry, scope) as marker:
            yield marker

    monkeypatch.setattr(review_pause, "slot_episode", delayed_episode)
    thread, result = _start(d)
    rejoin_thread = None
    try:
        assert preparing.wait(10)
        ack = _pause(d)
        first = _returned(thread, result)
        assert first["state"] == "pending" and first["aggregate"] != "PASS", first
        [paid] = _paid(d)
        [actor] = paid.triad_raw_results
        assert actor["operation_state"] == "pending_dispatch" and actor["late_result_pending"]
        assert not d.model.rounds
        _park(d, ack)
        _author_effects_refused(d)
        _resume(d)  # release the fence while the old preparation is still parked
        rejoin_thread, rejoin_result = _start(d)
        until(lambda: getattr(d.f.ctx, "_review_reconcile_only", False))
        release.set()
        d.model.release_first.set()
        collected = _returned(rejoin_thread, rejoin_result)
        assert collected["state"] == "settled" and collected["aggregate"] != "PASS", collected
        assert collected["record_id"] == first["record_id"]
        [settled] = _paid(d)
        assert settled.attempt == paid.attempt
        [refused] = settled.triad_raw_results
        assert refused["operation_id"] == actor["operation_id"]
        assert refused["operation_state"] == "not_dispatched" and not refused["late_result_pending"]
        assert not d.model.rounds and not attempt_rows_in_start_order(d.f.root)
        assert not any(row[0] in {"tag", "push"} or row[1:2] == ("commit",) for row in d.effects)
        assert d.target.read_text(encoding="utf-8") == "value = 2\n"
        print("PREFLIGHT_QUALIFICATION " + json.dumps({
            "scenario": "rapid-resume-before-episode-binding", "record_id": first["record_id"],
            "operation_id": actor["operation_id"], "aggregate": collected["aggregate"],
            "physical_sends": len(d.model.rounds), "author_commit_apply_effects": 0,
        }, sort_keys=True))
    finally:
        release.set()
        d.model.release_first.set()
        thread.join(10)
        if rejoin_thread is not None:
            rejoin_thread.join(10)
