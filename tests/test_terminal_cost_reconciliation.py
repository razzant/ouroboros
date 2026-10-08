"""Revisioned projection debt survives failures, changed ownership and receipts."""
import contextlib
import json
import logging
from types import SimpleNamespace

import pytest

from ouroboros import server_maintenance as maintenance
from ouroboros import task_results
from ouroboros import usage_store
from ouroboros import usage_accounting as usage
from supervisor import events_task_done as done
from supervisor import queue
from tests._usage_store_testing import ledger_rows


@pytest.fixture
def env(tmp_path, monkeypatch):
    live = set()
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "task_has_live_ownership", lambda task_id, **_kw: task_id in live)
    return SimpleNamespace(root=tmp_path, live=live)


def task(env, tid="root", **fields):
    return task_results.write_task_result(env.root, tid, "cancelled", result="kept answer", **fields)


def attempt(env, tid="root", *, logical=None, cost=0.4, final=True, provider="openai", non_task=False):
    # ``system:*`` probes/one-shots dispatch under the explicit non-task scope
    # their real producers bind (``update_letter``, ``llm_probe``): no task
    # control owner, so no task-result Pause/sleep admission read.
    scope = (usage.usage_scope(usage.UsageScope(drive_root=env.root, task_id=tid, root_task_id=logical or tid,
                                                non_task_operation=True))
             if non_task else contextlib.nullcontext())
    with scope:
        reservation = usage.reserve_attempt(usage.AttemptRequest(
            model="test", provider=provider, drive_root=env.root, task_id=tid,
            root_task_id=logical or tid, reservation_usd=1.0, global_limit_usd=100.0))
        usage.mark_dispatched(reservation)
    if final:
        usage.settle_attempt(reservation, {"prompt_tokens": 7}, cost_usd=cost, cost_final=True)
    return reservation


def dirty(env):
    with usage_store.read(env.root) as txn:
        return dict(txn.dirty_owners())


def warm(env):
    maintenance._reconcile_abandoned_usage(env.root)
    assert not dirty(env)


def test_quiet_history_has_no_result_or_projection_reads(env, monkeypatch):
    for tid in ("a", "b", "c"):
        task(env, tid)
        attempt(env, tid)
    warm(env)
    monkeypatch.setattr(task_results, "load_task_result", lambda *a, **k: pytest.fail("completed history"))
    monkeypatch.setattr(usage, "usage_breakdown", lambda *a, **k: pytest.fail("completed summaries"))
    maintenance._reconcile_abandoned_usage(env.root)


@pytest.mark.parametrize("failure", ["false", "raise"])
def test_failed_refresh_keeps_dirty_revision(env, monkeypatch, failure):
    task(env)
    attempt(env)
    before = dirty(env)
    refresh = done._refresh_terminal_task_cost
    def failed(*a, **k):
        if failure == "raise":
            raise OSError("unwritable result")
        return False
    monkeypatch.setattr(done, "_refresh_terminal_task_cost", failed)
    maintenance._reconcile_abandoned_usage(env.root)
    assert dirty(env) == before
    monkeypatch.setattr(done, "_refresh_terminal_task_cost", refresh)
    warm(env)


def test_receipt_after_projection_before_ack_keeps_new_revision(env, monkeypatch):
    task(env)
    attempt(env)
    before = dirty(env)["root"]
    refresh = done._refresh_terminal_task_cost
    raced = []
    def racing(*a, **k):
        answer = refresh(*a, **k)
        if not raced:
            raced.append(True)
            attempt(env, cost=0.2)
        return answer
    monkeypatch.setattr(done, "_refresh_terminal_task_cost", racing)
    maintenance._reconcile_abandoned_usage(env.root)
    assert dirty(env)["root"] > before
    assert task_results.load_task_result(env.root, "root")["accounted_upper_bound_usd"] == 0.4
    warm(env)
    assert task_results.load_task_result(env.root, "root")["accounted_upper_bound_usd"] == 0.6


def test_task_and_root_are_refreshed_and_acknowledged(env):
    task(env)
    task(env, "child", parent_task_id="root", root_task_id="root", delegation_role="subagent")
    attempt(env, "child", logical="root")
    assert set(dirty(env)) == {"root", "child"}
    warm(env)
    assert task_results.load_task_result(env.root, "child")["accounted_upper_bound_usd"] == 0.4
    assert task_results.load_task_result(env.root, "root")["accounted_upper_bound_usd_with_children"] == 0.4


@pytest.mark.parametrize("condition", ["missing", "unreadable", "live", "synthesis"])
def test_ineligible_result_keeps_debt(env, condition):
    task(env)
    attempt(env)
    before = dirty(env)
    path = task_results.task_result_path(env.root, "root")
    if condition == "missing":
        path.unlink()
    elif condition == "unreadable":
        path.write_text("{", encoding="utf-8")
    elif condition == "live":
        env.live.add("root")
    else:
        task(env, root_phase_checkpoint={"post_task_synthesis": "running"})
    maintenance._reconcile_abandoned_usage(env.root)
    assert dirty(env) == before


@pytest.mark.parametrize("basis", ["owner_pause_authority_missing", "owner_pause_authority_unreadable"])
def test_missing_authority_reports_once_and_later_projects_same_debt(env, monkeypatch, caplog, basis):
    from ouroboros import terminal_cost_reconciliation as duty
    from supervisor.task_ownership import TaskOwnershipRead

    monkeypatch.setattr(duty, "_LAST_UNRESOLVED", {})
    task(env)
    attempt(env)
    before = dirty(env)
    original = task_results.task_result_path(env.root, "root").read_bytes()
    load = TaskOwnershipRead.load

    def unavailable(self, tid):
        raise ValueError(basis)

    monkeypatch.setattr(TaskOwnershipRead, "load", unavailable)
    with caplog.at_level(logging.WARNING, logger=duty.__name__):
        for _ in range(3):
            maintenance._reconcile_abandoned_usage(env.root)
    assert dirty(env) == before
    assert task_results.task_result_path(env.root, "root").read_bytes() == original
    warnings = [r for r in caplog.records if r.name == duty.__name__ and r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and warnings[0].exc_info is not None

    path = env.root / "logs" / "supervisor.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    observations = [r for r in rows if r.get("subject") == duty.COST_PROJECTIONS]
    assert len(observations) == 1
    assert observations[0]["task_ids"] == ["root"]
    assert observations[0]["by_basis"] == {f"ValueError:{basis}": 1}
    assert "attempt_ids" not in observations[0]
    # The same text in an attempt id is a different obligation namespace.
    duty._publish_unresolved(env.root, {"root": basis})
    monkeypatch.setattr(TaskOwnershipRead, "load", load)
    warm(env)
    stored = task_results.load_task_result(env.root, "root")
    assert stored["cost_final"] and stored["accounted_upper_bound_usd"] == 0.4
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["subject"] == duty.COST_PROJECTIONS and rows[-1]["count"] == 0
    assert any(r.get("attempt_ids") == ["root"] for r in rows)


def test_equal_unknown_cost_can_ack_without_inventing_finality(env):
    task(env)
    reservation = attempt(env, final=False)
    warm(env)
    row = task_results.load_task_result(env.root, "root")
    assert row["cost_final"] is False and row["accounted_upper_bound_usd"] == 1
    usage.settle_attempt(reservation, cost_usd=0.1, cost_final=True)
    warm(env)
    assert task_results.load_task_result(env.root, "root")["cost_final"] is True


def test_foreign_live_owner_never_opens_foreign_money(env, monkeypatch):
    task(env, budget_drive_root=str(env.root / "foreign"))
    attempt(env)
    env.live.add("root")
    monkeypatch.setattr(usage, "usage_breakdown", lambda *a, **k: pytest.fail("live owner"))
    maintenance._reconcile_abandoned_usage(env.root)
    assert dirty(env) and not (env.root / "foreign").exists()


@pytest.mark.parametrize("owner", ["running", "busy", "direct", "postwork", "retry", "malformed_retry", "review", "control_review"])
def test_real_ownership_predicate_still_fences_projection_and_recovery(env, monkeypatch, owner):
    from ouroboros import review_operation
    from supervisor import queue_transitions, workers

    monkeypatch.setattr(queue, "task_has_live_ownership", queue_transitions.task_has_live_ownership)
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "QUEUE_MAX_RETRIES", 3)
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "direct_chat_turn", lambda tid: object() if owner == "direct" else None)
    monkeypatch.setattr(queue_transitions, "post_task_synthesis_in_flight", lambda *_: owner == "postwork")
    task(env)
    reservation = attempt(env, final=False)
    if owner == "running":
        queue.RUNNING["root"] = {"task": {"id": "root"}}
    if owner == "busy":
        workers.WORKERS[1] = SimpleNamespace(busy_task_id="root")
    if owner in {"retry", "malformed_retry"}:
        lineage = dict(root_task_id="root", parent_task_id="", delegation_role="root",
                       original_task_id="root", timeout_retry_from="root", supersedes_task_id="root")
        task(env, "retry", **lineage)
        queue.RUNNING["retry"] = {"task": {"id": "retry", **lineage}}
        # The malformed case is a live retry OUTSIDE the reciprocal chain.
        if owner == "retry":
            task(env, superseded_by="retry", retry_task_id="retry")
    if owner in {"review", "control_review"}:
        primary = {"state": review_operation.OPERATION_DISPATCHED, "controller": {"fixture": "alive"}, "source_ref": {"id": "source"}}
        monkeypatch.setattr(review_operation, "controller_state", lambda _: "alive")
        if owner == "review":
            task(env, **{review_operation.OPERATIONS_FIELD: {"operation": primary}})
        else:
            task(env, "subject", **{review_operation.OPERATIONS_FIELD: {"operation": primary}})
            link = {"control_only": True, "subject_task_id": "subject", "controller": primary["controller"], "source_ref": primary["source_ref"]}
            task(env, **{review_operation.OPERATIONS_FIELD: {"operation": link}})
    assert queue.task_has_live_ownership("root")
    ledger = ledger_rows(env.root)
    original = task_results.task_result_path(env.root, "root").read_bytes()
    maintenance._reconcile_abandoned_usage(env.root)
    assert ledger_rows(env.root) == ledger
    assert task_results.task_result_path(env.root, "root").read_bytes() == original
    usage.settle_attempt(reservation, cost_usd=0.4, cost_final=True)
    maintenance._reconcile_abandoned_usage(env.root)
    assert task_results.task_result_path(env.root, "root").read_bytes() == original


def test_a_live_review_link_changed_elsewhere_is_checked_before_new_money(env, monkeypatch):
    from ouroboros import review_operation
    from supervisor import queue_transitions, workers

    task(env)
    attempt(env)
    primary = {"state": review_operation.OPERATION_CLOSED, "controller": {"fixture": "alive"}, "source_ref": {"id": "s"}}
    task(env, "subject", **{review_operation.OPERATIONS_FIELD: {"operation": primary}})
    link = {"control_only": True, "subject_task_id": "subject", "controller": primary["controller"], "source_ref": primary["source_ref"]}
    task(env, **{review_operation.OPERATIONS_FIELD: {"operation": link}})
    warm(env)
    before = task_results.task_result_path(env.root, "root").read_bytes()
    monkeypatch.setattr(queue, "task_has_live_ownership", queue_transitions.task_has_live_ownership)
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "direct_chat_turn", lambda _: None)
    monkeypatch.setattr(queue_transitions, "post_task_synthesis_in_flight", lambda *_: False)
    monkeypatch.setattr(review_operation, "controller_state", lambda _: "alive")
    task(env, "subject", **{review_operation.OPERATIONS_FIELD: {"operation": {**primary, "state": review_operation.OPERATION_DISPATCHED}}})
    # An acknowledged projection does not claim this owner is dead; a receipt
    # must invoke the exact control-link predicate despite unchanged owner file.
    maintenance._reconcile_abandoned_usage(env.root)
    attempt(env, cost=0.1)
    maintenance._reconcile_abandoned_usage(env.root)
    assert task_results.task_result_path(env.root, "root").read_bytes() == before
    assert dirty(env)



def test_changed_authority_at_write_keeps_debt_and_does_not_publish_old_money(env, monkeypatch):
    task(env)
    attempt(env)
    writer = done.write_task_result
    def moved(root, tid, status, **fields):
        writer(root, tid, status, budget_drive_root=str(root / 'moved'))
        return writer(root, tid, status, **fields)
    monkeypatch.setattr(done, 'write_task_result', moved)
    maintenance._reconcile_abandoned_usage(env.root)
    stored = task_results.load_task_result(env.root, 'root')
    assert 'accounted_upper_bound_usd' not in stored
    assert dirty(env)


def test_logical_root_debt_refreshes_its_reciprocal_terminal_retry(env, monkeypatch):
    monkeypatch.setattr(queue, 'QUEUE_MAX_RETRIES', 3)
    monkeypatch.setattr(queue, 'RUNNING', {})
    monkeypatch.setattr(queue, 'PENDING', [])
    task(env, 'root', superseded_by='retry', retry_task_id='retry')
    task(env, 'retry', root_task_id='root', parent_task_id='', delegation_role='root',
         supersedes_task_id='root', original_task_id='root', timeout_retry_from='root')
    task(env, 'child', root_task_id='root', parent_task_id='root', delegation_role='subagent')
    attempt(env, 'retry', logical='root')
    maintenance._reconcile_abandoned_usage(env.root)
    # The earlier retry projection replaced its result. Its root's cached
    # ownership is now stale, so that debt survives for a fresh next pass.
    assert set(dirty(env)) == {'root'}
    warm(env)
    attempt(env, 'child', logical='root', cost=0.6)
    warm(env)
    for tid in ('root', 'retry'):
        assert task_results.load_task_result(env.root, tid)['accounted_upper_bound_usd_with_children'] == 1.0


@pytest.mark.parametrize("attribution, candidate", [
    ({"review_wave_id": "wave-deep"}, True),
    ({"review_wave_id": "commit:c1", "review_slot_id": "triad-slot"}, False),
    ({"review_skill": "weather", "review_wave_id": "skill-w1", "review_slot_id": "s1"}, False),
])
def test_a_wave_alone_keeps_the_owner_in_the_cost_refresh(env, attribution, candidate):
    """Deep self-review rows carry a wave but no reviewer slot (#1544), and that review is the
    whole task: its owner still needs the terminal cost refresh. Rows of a reviewer slot or a
    skill review stay with the review's own custody and never make the owner a candidate."""
    with usage.usage_scope(usage.UsageScope(drive_root=env.root, task_id="t", root_task_id="t", **attribution)):
        attempt(env, "t")
    assert ("t" in dirty(env)) is candidate
