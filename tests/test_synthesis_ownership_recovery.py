"""One synthesis owner: canonical progress, unsent recovery and publication-only retry."""
from __future__ import annotations

import contextlib
import copy
import threading
from types import SimpleNamespace

import pytest

from ouroboros import agent_task_pipeline as pipeline, post_task_checkpoint as phase
from ouroboros import usage_accounting as money
from ouroboros.task_results import load_task_result, write_task_result
from tests._budget_pause_exact_helpers import _install_queue
from tests.test_delegate_body_collection import _until

pytestmark = pytest.mark.serial


@pytest.fixture
def work(tmp_path, monkeypatch):
    q, _state, _workers = _install_queue(tmp_path, monkeypatch)
    task = {"id": "synthesis", "root_task_id": "synthesis", "task_attempt": 1,
            "budget_drive_root": str(tmp_path), "type": "task"}
    write_task_result(tmp_path, task["id"], "failed", task_attempt=1, result="Preserved solve answer",
                      root_task_id=task["id"], root_phase_checkpoint={"post_task_synthesis": "pending_once"})
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path, drive_path=lambda rel: tmp_path / rel)
    calls = []
    monkeypatch.setattr(pipeline, "_pre_synthesis_usage_snapshot", lambda *a: {})
    monkeypatch.setattr(pipeline, "_record_task_facts", lambda *a: None)
    monkeypatch.setattr(pipeline, "_run_memory_fallback_draft", lambda *a: calls.append("fallback"))
    monkeypatch.setattr("ouroboros.post_task_evolution.maybe_promote", lambda *a: calls.append("promotion"))
    monkeypatch.setattr("ouroboros.llm.LLMClient", lambda: object())
    monkeypatch.setattr("ouroboros.memory.Memory", lambda **k: object())
    yield SimpleNamespace(root=tmp_path, env=env, task=task, calls=calls, q=q)
    _until(lambda: not phase.post_task_synthesis_in_flight(tmp_path, task["id"]))


def run(work, **kwargs):
    return pipeline._run_post_task_processing_async(work.env, work.task, {}, {},
        work.root / "logs", blocking=True, **kwargs)


def recover(work, **kwargs):
    return pipeline.recover_pending_root_post_task_synthesis(work.root, work.root,
        startup=False, owner_liveness=work.q.task_settlement_liveness,
        claim_guard=work.q.task_settlement_interlock, **kwargs)


def checkpoint(work):
    return load_task_result(work.root, work.task["id"], strict=True)["root_phase_checkpoint"]


def contention():
    from ouroboros.store import StoreBusy
    error = StoreBusy("controlled result publication contention", reason="contention")
    error.rollback_confirmed = True
    return error


@pytest.mark.parametrize("canonical", ["running", "paused", "completed"])
def test_child_pending_cannot_regress_canonical_phase_during_actual_copyback(work, canonical):
    from ouroboros.headless import copy_child_task_result
    from ouroboros.task_status import load_effective_task_result
    child = work.root / "child"
    saved = {"post_task_synthesis": canonical, "post_task_pause": {"payload_ref": {"sha256": "kept"}}}
    write_task_result(work.root, work.task["id"], "failed", child_drive_root=str(child),
                      root_phase_checkpoint=saved)
    write_task_result(child, work.task["id"], "failed", task_attempt=1,
        result="Child answer enriched", root_phase_checkpoint={"post_task_synthesis": "pending_once", "status": "pass"})
    viewed = load_effective_task_result(work.root, work.task["id"], materialize_artifacts=False)
    copied = copy_child_task_result(work.root, {**work.task, "drive_root": str(child)})
    for row in (viewed, copied):
        assert row["root_phase_checkpoint"]["post_task_synthesis"] == canonical
    assert copied["result"] == "Child answer enriched"
    assert checkpoint(work)["post_task_pause"] == saved["post_task_pause"]


def test_unconfirmed_running_publication_never_enters_stages_and_recovers_once(work, monkeypatch):
    writer = phase.write_task_result
    def busy(*args, **kwargs):
        if kwargs.get("root_phase_checkpoint", {}).get("post_task_synthesis") == "running":
            raise contention()
        return writer(*args, **kwargs)
    monkeypatch.setattr(phase, "write_task_result", busy)
    with pytest.raises(type(contention())):
        run(work)
    assert work.calls == [] and checkpoint(work)["post_task_synthesis"] == "pending_once"
    assert not phase.post_task_synthesis_in_flight(work.root, work.task["id"])
    monkeypatch.setattr(phase, "write_task_result", writer)
    assert recover(work) == 1
    _until(lambda: checkpoint(work)["post_task_synthesis"] == "completed")
    _until(lambda: not phase.post_task_synthesis_in_flight(work.root, work.task["id"]))
    assert work.calls == ["fallback", "promotion"]
    assert recover(work) == 0


@pytest.mark.parametrize("at", ["snapshot", "thread"])
def test_unstarted_preparation_exception_leaves_no_physical_reservation(work, monkeypatch, at):
    def fail(*a, **k):
        raise RuntimeError("controlled pre-worker failure")
    if at == "snapshot":
        monkeypatch.setattr(pipeline, "_pre_synthesis_usage_snapshot", fail)
    else:
        monkeypatch.setattr(pipeline.threading, "Thread", lambda *a, **k: SimpleNamespace(start=fail))
    with pytest.raises(RuntimeError, match="pre-worker"):
        pipeline._run_post_task_processing_async(work.env, work.task, {}, {}, work.root / "logs")
    assert not phase.post_task_synthesis_in_flight(work.root, work.task["id"])
    assert work.calls == []


def test_final_publication_retries_only_the_computed_result_with_locks_released(work, monkeypatch):
    writer, final_writes, pauses = phase.write_task_result, [], []
    def write(*args, **kwargs):
        if kwargs.get("root_phase_checkpoint", {}).get("post_task_synthesis") == "completed":
            final_writes.append(True)
            if len(final_writes) == 1:
                raise contention()
        return writer(*args, **kwargs)
    def wait(_seconds):
        assert phase.POST_TASK_SYNTHESIS_LOCK.acquire(blocking=False)
        phase.POST_TASK_SYNTHESIS_LOCK.release()
        assert phase._CHECKPOINT_WRITE_LOCK.acquire(blocking=False)
        phase._CHECKPOINT_WRITE_LOCK.release()
        assert phase.post_task_synthesis_in_flight(work.root, work.task["id"])
        pauses.append(True)
    monkeypatch.setattr(phase, "write_task_result", write)
    monkeypatch.setattr(pipeline.time, "sleep", wait)
    run(work)
    assert final_writes == [True, True] and pauses == [True]
    assert work.calls == ["fallback", "promotion"]
    assert checkpoint(work)["post_task_synthesis"] == "completed"
    assert load_task_result(work.root, work.task["id"])["result"] == "Preserved solve answer"


def test_acceptance_enrichment_during_paid_work_does_not_disown_its_final_checkpoint(work, monkeypatch):
    def promote(*a, **k):
        work.calls.append("promotion")
        current = checkpoint(work)
        write_task_result(work.root, work.task["id"], "failed",
            root_phase_checkpoint={**current, "phase": "task_acceptance", "status": "pass", "pass_index": 3})
    monkeypatch.setattr("ouroboros.post_task_evolution.maybe_promote", promote)
    run(work)
    assert checkpoint(work)["post_task_synthesis"] == "completed"
    assert checkpoint(work)["status"] == "pass" and checkpoint(work)["pass_index"] == 3
    assert work.calls.count("promotion") == 1


@pytest.mark.parametrize("ending", ["stop", "unconfirmed_write", "permission"])
def test_final_publication_debt_keeps_paid_results_without_retrying_unknown_writes(work, monkeypatch, ending):
    stopped, attempts = threading.Event(), []
    writer = phase.write_task_result
    def write(*args, **kwargs):
        if kwargs.get("root_phase_checkpoint", {}).get("post_task_synthesis") == "completed":
            attempts.append(True)
            if ending == "permission":
                raise PermissionError("publication unavailable")
            error = contention()
            if ending == "stop":
                stopped.set()
            else:
                error.rollback_confirmed = False
            raise error
        return writer(*args, **kwargs)
    monkeypatch.setattr(phase, "write_task_result", write)
    run(work, stop_requested=stopped.is_set)
    assert attempts == [True] and work.calls == ["fallback", "promotion"]
    assert checkpoint(work)["post_task_synthesis"] == "running"
    assert not phase.post_task_synthesis_in_flight(work.root, work.task["id"])
    assert load_task_result(work.root, work.task["id"])["result"] == "Preserved solve answer"


def test_new_attempt_during_phase_cannot_receive_the_previous_attempts_final(work, monkeypatch):
    from ouroboros.task_store import patch_document
    def promote(*a, **k):
        patch_document(work.root, work.task["id"], lambda current: {**current, "task_attempt": 2})
    monkeypatch.setattr("ouroboros.post_task_evolution.maybe_promote", promote)
    run(work)
    assert checkpoint(work)["post_task_synthesis"] == "running"
    assert not phase.post_task_synthesis_in_flight(work.root, work.task["id"])


def test_stale_original_callback_cannot_start_current_attempts_phase(work):
    from ouroboros.task_store import patch_document
    patch_document(work.root, work.task["id"], lambda current: {**current, "task_attempt": 2})
    with pytest.raises(phase.PostTaskCheckpointChanged):
        run(work)
    assert checkpoint(work)["post_task_synthesis"] == "pending_once" and not work.calls
    assert not phase.post_task_synthesis_in_flight(work.root, work.task["id"])
    # The admitted worker's carrier uses _attempt while its result uses task_attempt.
    work.task.pop("task_attempt")
    work.task["_attempt"] = 2
    run(work)
    assert checkpoint(work)["post_task_synthesis"] == "completed"


@pytest.mark.parametrize("direct", [False, True])
def test_real_agent_start_record_allows_the_original_unstamped_task_to_finish(work, monkeypatch, direct):
    import queue
    from ouroboros.agent import OuroborosAgent

    task = {"id": "real-start", "root_task_id": "real-start", "type": "task", "chat_id": 0,
            "text": "Finish the admitted work", "_is_direct_chat": direct}
    if not direct:
        task["_attempt"] = 2
    actor = SimpleNamespace(env=work.env, _task_started_ts=1_800_000_000.0)
    OuroborosAgent._persist_running_record(actor, task)
    started = load_task_result(work.root, task["id"], strict=True)
    assert started["started_at"] and "started_at" not in task
    assert started["task_attempt"] == (0 if direct else 2)
    monkeypatch.setattr(pipeline, "in_worker_process", lambda: not direct)
    pipeline.emit_task_results(work.env, None, None, [], task, "Actual completed answer", {"rounds": 1},
        {"tool_calls": [], "reasoning_notes": []}, actor._task_started_ts, work.root / "logs", event_queue=queue.Queue())
    stored = load_task_result(work.root, task["id"], strict=True)
    assert stored["root_phase_checkpoint"]["post_task_synthesis"] == "completed"
    assert stored["result"] == "Actual completed answer"
    assert work.calls == ["fallback", "promotion"]
    assert not phase.post_task_synthesis_in_flight(work.root, task["id"])


@pytest.mark.parametrize("source,category", [("post_task_synthesis", "post_task"),
    ("memory_fallback", "consolidation"), ("scratchpad_consolidation", "consolidation")])
@pytest.mark.parametrize("state", ["dispatched", "settled", "unresolved"])
def test_legacy_pending_with_actual_paid_facts_is_never_replayed(work, source, category, state):
    held = money.reserve_attempt(money.AttemptRequest(model="test", provider="test", task_id=work.task["id"],
        root_task_id=work.task["id"], drive_root=work.root, category=category, source=source, reservation_usd=0))
    money.mark_dispatched(held)
    if state == "settled":
        money.settle_attempt(held, {}, cost_usd=0, cost_final=True)
    elif state == "unresolved":
        money.mark_unresolved(held, "transport_outcome_unknown")
    assert recover(work) == 1
    assert work.calls == [] and checkpoint(work)["post_task_synthesis"] == "degraded"
    assert checkpoint(work)["post_task_stop_reason"] == "owner_ended_indeterminate_post_task"
    assert len(money.read_usage_records(work.root)) == 1
    assert recover(work) == 0


def test_a_positively_unrun_release_does_not_remove_synthesis_recovery(work):
    held = money.reserve_attempt(money.AttemptRequest(model="test", provider="test", task_id=work.task["id"],
        root_task_id=work.task["id"], drive_root=work.root, category="post_task", source="post_task_synthesis",
        reservation_usd=0))
    money.release_attempt(held, "not_sent")
    assert recover(work) == 1
    _until(lambda: checkpoint(work)["post_task_synthesis"] == "completed")
    assert work.calls == ["fallback", "promotion"]


def test_unreadable_legacy_dispatch_history_keeps_debt_without_a_paid_retry(work, monkeypatch):
    from ouroboros import usage_store
    from ouroboros.usage_ledger import UsageLedgerCorrupt
    monkeypatch.setattr(usage_store, "read", lambda *args: (_ for _ in ()).throw(UsageLedgerCorrupt("test corrupt")))
    assert recover(work) == 0
    assert checkpoint(work)["post_task_synthesis"] == "pending_once" and work.calls == []


@pytest.mark.parametrize("liveness", [True, None])
def test_live_or_unknown_physical_owner_prevents_recovery_even_after_paid_phase(work, liveness):
    phase.set_root_post_task_checkpoint(work.env, work.task, "running")
    assert pipeline.recover_pending_root_post_task_synthesis(work.root, startup=False,
        owner_liveness=lambda _tid: liveness) == 0
    assert checkpoint(work)["post_task_synthesis"] == "running" and work.calls == []


def test_startup_revival_never_degrades_an_inprocess_owner(work):
    phase.set_root_post_task_checkpoint(work.env, work.task, "running")
    key = phase.reserve_post_task_synthesis(work.root, work.task["id"])
    try:
        assert pipeline.recover_pending_root_post_task_synthesis(work.root) == 0
        assert checkpoint(work)["post_task_synthesis"] == "running"
    finally:
        phase.release_post_task_synthesis(key)


def test_newer_phase_write_wins_exact_phase_cas_and_periodic_recovery_keeps_it(work, monkeypatch):
    writer, changed = phase.write_task_result, []
    stopped = {"post_task_synthesis": "degraded",
               "post_task_stop_reason": "owner_stopped:skipped=memory_fallback_draft,promotion"}
    def race(*args, **kwargs):
        if kwargs.get("root_phase_checkpoint", {}).get("post_task_synthesis") == "running" and not changed:
            changed.append(True)
            writer(work.root, work.task["id"], "failed", root_phase_checkpoint=stopped)
        return writer(*args, **kwargs)
    monkeypatch.setattr(phase, "write_task_result", race)
    with pytest.raises(phase.PostTaskCheckpointChanged):
        run(work)
    before = copy.deepcopy(checkpoint(work))
    assert recover(work) == 0
    assert checkpoint(work) == before and before["post_task_stop_reason"] == stopped["post_task_stop_reason"]
    assert work.calls == []


def test_closed_generation_cannot_claim_a_late_recovery(work):
    stopped = threading.Event()
    @contextlib.contextmanager
    def guard(stop):
        stopped.set()
        yield not stop()
    assert pipeline.recover_pending_root_post_task_synthesis(work.root, startup=False,
        owner_liveness=lambda _tid: False, claim_guard=guard, stop_requested=stopped.is_set) == 0
    assert checkpoint(work)["post_task_synthesis"] == "pending_once" and work.calls == []


def test_blocked_checkpoint_cost_does_not_block_queue_owner_observation(work, monkeypatch):
    from supervisor import state
    entered, release, errors = threading.Event(), threading.Event(), []
    original = state.reconstruct_task_cost
    def cost(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)
    monkeypatch.setattr(state, "reconstruct_task_cost", cost)
    def finish():
        try:
            phase.set_root_post_task_checkpoint(work.env, work.task, "completed")
        except BaseException as exc:
            errors.append(exc)
    writer = threading.Thread(target=finish)
    writer.start()
    observed = threading.Event()
    def control():
        work.q.task_settlement_liveness(work.task["id"])
        with work.q._queue_lock:
            assert not phase.post_task_synthesis_in_flight(work.root, work.task["id"])
            observed.set()
    reader = threading.Thread(target=control)
    try:
        assert entered.wait(3)
        reader.start()
        assert observed.wait(3), "control waited on another phase's final-cost publication"
        assert not release.is_set()
    finally:
        release.set()
        writer.join(10)
        if reader.ident is not None:
            reader.join(10)
    assert not errors and not writer.is_alive() and not reader.is_alive()
