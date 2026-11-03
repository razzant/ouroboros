"""Cold producer publication and successful-absence usage debt do no history work at rest."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ouroboros import maintenance, store, subagent_metrics as metrics, swarm_rollup
from ouroboros import terminal_cost_reconciliation as duty, usage_store
from ouroboros.delegate_custody import RunCustody
from ouroboros.review_records import ReviewActorRecord, ReviewRequest, ReviewSlot
from ouroboros.review_seat_outcome import finalize_seat_outcome, publish_seat_execution
from ouroboros.subagent_history import record_session_execution, record_task_execution
from ouroboros.task_results import write_task_result
from tests._cost_meter import BOUNDARY, CostMeter, HISTORY
from tests.test_subagent_metrics import event
from tests.test_terminal_cost_reconciliation import attempt, dirty, env as env  # noqa: F401


@pytest.mark.parametrize("stale", [False, True])
@pytest.mark.parametrize("producer", ["execution", "review", "api"])
@pytest.mark.parametrize("archived", [False, True])
def test_first_publication_never_folds_a_cold_suffix(tmp_path, monkeypatch, stale, producer, archived):
    from ouroboros import utils
    counts = []
    for scale in (1, 30):
        root = tmp_path / f"root{scale:02}"
        root.mkdir()
        with store.open_store(root, write=True):
            pass  # serving has published its store; only the metrics cursor is cold
        if stale:
            metrics.publish(root, event("prior"))
        path = root / "logs/events.jsonl"
        path.parent.mkdir(exist_ok=True)
        if archived:
            (root / "archive").mkdir()
            if stale:
                path.rename(root / "archive/events_20261021T000000.jsonl")
            for index in range(scale):
                (root / f"archive/events_20261022T{index:06}.jsonl").write_text(
                    '{"type":"old_event"}\n' * 40, encoding="utf-8")
        else:
            with path.open("a", encoding="utf-8") as stream:
                stream.write('{"type":"old_event"}\n' * ((metrics.BATCH_BYTES // 20 + 1) * scale))
        calls = []
        original = swarm_rollup.fold_event_suffix
        with monkeypatch.context() as foreground:
            foreground.setattr(swarm_rollup, "fold_event_suffix", lambda *a, **k: calls.append(1) or original(*a, **k))
            foreground.setattr(utils, "jsonl_archive_segments", lambda *a, **k: pytest.fail("publisher enumerated archives"))
            if producer == "execution":
                record_session_execution(root, RunCustody(task_id="task", run_id="run", route_id="route",
                    model="model", selected_subagent_id="worker"), {"summary": {"state": "succeeded", "answer_received": True}}, {})
            elif producer == "api":
                record_task_execution({"id": "task", "configured_subagent": {"selected_subagent_id": "worker",
                    "route": {"kind": "api_model", "target_id": "test/model"}}}, {"llm_call_refs": [{
                    "llm_call_id": "api", "model": "test/model", "dispatched": True,
                    "model_output_received": True, "usable_solve_response": True}]}, drive_root=root)
            else:
                request = ReviewRequest(surface="result_review", goal="review", task_id="task", policy={"parts": ["result"]})
                slot = ReviewSlot(slot_id="worker", model="test/model", subagent_id="worker")
                actor = ReviewActorRecord(slot_id="worker", model="test/model", status="ok", raw_text="[]", operation_id="op")
                finalize_seat_outcome(request, slot, actor)
                publish_seat_execution(request, slot, actor, SimpleNamespace(drive_root=root, budget_drive_root=root))
        counts.append(len(calls))
        assert metrics.read(root, "worker")["coverage"]["stream"] == "pending"
        assert maintenance.run_job(root, "subagent_metrics")["state"] == "done"
        facts = metrics.read(root, "worker")
        assert facts["coverage"]["stream"] == "complete" and facts["rows"]
        assert maintenance.run_job(root, "subagent_metrics") is None
        utils._CLOSED_SEGMENTS.clear()  # warm cursor in a fresh answering process
        with monkeypatch.context() as foreground:
            foreground.setattr(utils, "jsonl_archive_segments", lambda *a, **k: pytest.fail("warm publisher enumerated archives"))
            metrics.publish(root, event("warm-next"))
        assert metrics.read(root, "worker")["coverage"]["stream"] == "complete"
    assert counts == [0, 0]


def test_warm_metrics_publish_folds_bounded_interleaved_events_without_a_child(tmp_path, monkeypatch):
    from ouroboros.utils import append_jsonl

    metrics.publish(tmp_path, event())
    monkeypatch.setattr(maintenance.child, "request", lambda: pytest.fail("bounded warm publication woke the child"))
    for index in range(3):
        append_jsonl(tmp_path / "logs/events.jsonl", {"type": "ordinary_event", "index": index})
        metrics.publish(tmp_path, event(str(index), purpose="execution", terminal=False))
    rows = metrics.read(tmp_path, "worker")["rows"]
    assert next(row for row in rows if row["purpose"] == "execution")["pending"] == 3
    assert all(row["coverage"]["stream"] == "complete" for row in rows)
    assert maintenance.run_job(tmp_path, "subagent_metrics") is None


def test_fold_and_projection_never_hold_a_store_write_transaction(tmp_path, monkeypatch):
    original_open, original_fold, original_project = store.open_store, swarm_rollup.fold_event_suffix, metrics.fold
    writing, visited = [], []

    @contextmanager
    def opened(*args, **kwargs):
        with original_open(*args, **kwargs) as txn:
            if kwargs.get("write"):
                writing.append(True)
            try:
                yield txn
            finally:
                if kwargs.get("write"):
                    writing.pop()

    def fold(*args, **kwargs):
        assert not writing
        visited.append("suffix")
        return original_fold(*args, **kwargs)

    def project(*args, **kwargs):
        assert not writing
        visited.append("projection")
        return original_project(*args, **kwargs)

    monkeypatch.setattr(store, "open_store", opened)
    monkeypatch.setattr(swarm_rollup, "fold_event_suffix", fold)
    monkeypatch.setattr(metrics, "fold", project)
    metrics.publish(tmp_path, event())
    metrics.catch_up(tmp_path)
    assert {"suffix", "projection"} <= set(visited)
    assert metrics.read(tmp_path, "worker")["rows"][0]["dispatched"] == 1


def test_suffix_commit_uses_its_captured_horizon_and_does_not_erase_a_racing_cursor(tmp_path, monkeypatch):
    from ouroboros.utils import append_jsonl

    metrics.publish(tmp_path, event())
    append_jsonl(tmp_path / "logs/events.jsonl", {"type": metrics.EVENT, **event("two")})
    project = metrics._projection
    raced = []

    def racing(root, events):
        result = project(root, events)
        if not raced:
            raced.append(True)
            append_jsonl(root / "logs/events.jsonl", {"type": metrics.EVENT, **event("three")})
            metrics.catch_up(root)  # another reducer commits first
        return result

    monkeypatch.setattr(metrics, "_projection", racing)
    assert not metrics.catch_up(tmp_path)
    assert metrics.read(tmp_path, "worker")["rows"][0]["dispatched"] == 3
    assert maintenance.run_job(tmp_path, "subagent_metrics")["state"] == "done"
    assert metrics.read(tmp_path, "worker")["rows"][0]["dispatched"] == 3


def test_child_batches_keep_cursor_and_projection_atomic_and_retry_a_repaired_torn_tail(tmp_path, monkeypatch):
    from ouroboros.utils import append_jsonl
    with store.open_store(tmp_path, write=True):
        pass
    path = tmp_path / "logs/events.jsonl"
    for index in range(5):
        append_jsonl(path, event(str(index), type=metrics.EVENT))
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"type":"tail"}')
    monkeypatch.setattr(metrics, "BATCH_BYTES", 600)
    commit, checkpoints = metrics._commit_suffix, []
    def observed(root, cursor, events, through, upper, source):
        result = commit(root, cursor, events, through, upper, source)
        if result:
            with store.open_store(root, write=False) as txn:
                checkpoints.append((txn.meta(metrics.CURSOR)["through"], len(txn.set_rows(metrics.PRODUCERS))))
        return result
    monkeypatch.setattr(metrics, "_commit_suffix", observed)
    assert maintenance.run_job(tmp_path, "subagent_metrics")["state"] == "failed"
    assert len(checkpoints) > 1 and checkpoints[-1][1] == 5
    assert all(a[0] < b[0] and a[1] < b[1] for a, b in zip(checkpoints, checkpoints[1:]) if a != b)
    assert maintenance.run_job(tmp_path, "subagent_metrics") is None
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    assert maintenance.run_job(tmp_path, "subagent_metrics")["state"] == "done"
    assert metrics.read(tmp_path, "worker")["rows"][0]["dispatched"] == 5


@pytest.mark.parametrize("running", [False, True])
@pytest.mark.parametrize("money_write", [False, True])
def test_absent_usage_owners_rest_flat_and_new_facts_wake_them(env, monkeypatch, running, money_write):
    counts = []
    for scale in (1, 30):
        root = env.root / f"root{scale:02}"
        current = SimpleNamespace(root=root, live=env.live)
        for index in range(10 * scale):
            attempt(current, f"missing{index:04}", cost=.04)
        if running:
            write_task_result(root, "running", "running")
            attempt(current, "running")
        # Equal-width physical markers keep the cost comparison about row count.
        with usage_store.hold(root) as txn:
            txn.set_meta("publication_marker", [0, 100000])
        monkeypatch.setattr(duty, "_LAST_UNRESOLVED", {})
        duty.reconcile_abandoned_usage(root)
        debt = dirty(current)
        with store.open_store(root, write=False) as txn:
            rests = txn.set_rows("duty_rest")
        assert len(rests) == 10 * scale and all(row["signature"] is None for row in rests.values())
        if running:
            write_task_result(root, "missing0001", "running")
            duty.reconcile_abandoned_usage(root)
            with store.open_store(root, write=False) as txn:
                assert txn.get_row("duty_rest", "cost_projection:missing0001") is None
        monkeypatch.setattr("ouroboros.utils.utc_now_iso", lambda: "2099-01-01T00:00:00+00:00")
        with store.open_store(root, write=True) as txn:
            txn.set_meta("unrelated_job", {"state": "done"})
        if running:
            write_task_result(root, "running", "running", result="live progress")
            write_task_result(root, "missing0001", "running", result="progress after arrival")
        write_task_result(root, "unrelated", "completed", result="ordinary completion")
        if money_write:
            attempt(current, "system:unrelated", non_task=True)
            debt = dirty(current)
        store.close_connections()
        with CostMeter(root) as meter:
            token = BOUNDARY.set("rest")
            try:
                duty.reconcile_abandoned_usage(root)
            finally:
                BOUNDARY.reset(token)
        observed = meter.snapshot("rest")["counts"]
        assert observed.get("writes", 0) == 0
        assert all(observed.get(f"{kind}:{category}", 0) == 0 for category in HISTORY for kind in ("open", "read_bytes"))
        counts.append({key: observed.get(key, 0) for key in ("vm_steps", "statements", "json_loads", "json_chars")})
        assert dirty(current) == debt
        # A new price for an already absent owner still advances its own rest.
        attempt(current, "missing0000", cost=.02)
        duty.reconcile_abandoned_usage(root)
        with store.open_store(root, write=False) as txn:
            assert txn.get_row("duty_rest", "cost_projection:missing0000")["inputs"] == {
                "usage_revision": dirty(current)["missing0000"]}
        write_task_result(root, "missing0000", "completed", result="late body")
        duty.reconcile_abandoned_usage(root)
        from ouroboros.task_results import load_task_result
        assert load_task_result(root, "missing0000")["accounted_upper_bound_usd"] == .06
        assert "missing0000" not in dirty(current)
        attempt(current, "new-owner")
        duty.reconcile_abandoned_usage(root)
        with store.open_store(root, write=False) as txn:
            assert txn.get_row("duty_rest", "cost_projection:new-owner")["waits_on"] == ["result_absent"]
    assert counts[0] == counts[1]


def test_absence_racing_task_arrival_is_not_latched(env, monkeypatch):
    attempt(env, "arriving")
    present = duty._present_facts
    raced = []

    def arriving(root, ids):
        result = present(root, ids)
        if ids and not raced:
            raced.append(True)
            write_task_result(root, "arriving", "completed", result="landed during observation")
        return result

    monkeypatch.setattr(duty, "_present_facts", arriving)
    duty.reconcile_abandoned_usage(env.root)
    duty.reconcile_abandoned_usage(env.root)
    assert not dirty(env)
    with store.open_store(env.root, write=False) as txn:
        assert txn.get_row("duty_rest", "cost_projection:arriving") is None


def test_money_only_pass_does_not_swallow_another_absent_owners_arrival(env, monkeypatch):
    attempt(env, "waiting")
    attempt(env, "priced")
    duty.reconcile_abandoned_usage(env.root)
    attempt(env, "priced", cost=.2)
    present = duty._present_facts
    selected = []

    def arriving(root, ids):
        ids = set(ids)
        answer = present(root, ids)
        if ids:
            selected.append(ids)
            write_task_result(root, "waiting", "completed")
        return answer

    with monkeypatch.context() as race:
        race.setattr(duty, "_present_facts", arriving)
        duty.reconcile_abandoned_usage(env.root)
    assert selected == [{"priced"}], "only the owner of the newer money is selected"
    assert "waiting" in dirty(env), "the racing arrival retains its projection debt"
    duty.reconcile_abandoned_usage(env.root)
    assert "waiting" not in dirty(env)
    from ouroboros.task_results import load_task_result
    assert load_task_result(env.root, "waiting")["accounted_upper_bound_usd"] == .4


def test_ledger_epoch_change_reselects_absent_owners(env, monkeypatch):
    attempt(env, "waiting")
    duty.reconcile_abandoned_usage(env.root)
    with usage_store.hold(env.root) as txn:
        epoch, sequence = txn.marker()
        txn.set_meta("publication_marker", [epoch + 1, sequence])
    select, calls = usage_store.Txn.dirty_owners, []

    def selected(txn, **kwargs):
        calls.append(kwargs)
        return select(txn, **kwargs)

    monkeypatch.setattr(usage_store.Txn, "dirty_owners", selected)
    duty.reconcile_abandoned_usage(env.root)
    assert calls == [{}], "a new ledger epoch cannot reuse the previous sequence frontier"
    duty.reconcile_abandoned_usage(env.root)
    assert calls == [{}], "the new successful absence rests again"


def test_old_usage_store_gains_the_dirty_revision_index_only_at_boot(env):
    attempt(env, "old")
    with usage_store.hold(env.root) as txn:
        marker = txn.marker()
        txn.conn.execute("DROP INDEX dirty_owners_revision")
    attempt(env, "new")
    with usage_store.read(env.root) as txn:
        assert [owner for owner, _ in txn.dirty_owners(since=marker[1])] == ["new"]
        assert txn.conn.execute("SELECT 1 FROM sqlite_master WHERE name='dirty_owners_revision'").fetchone() is None
    assert usage_store.migrate_from_journal(env.root, initialize_journal=True)["dirty_owner_index"] == "created"
    assert usage_store.migrate_from_journal(env.root, initialize_journal=True)["dirty_owner_index"] == "present"
    with usage_store.read(env.root) as txn:
        assert [owner for owner, _ in txn.dirty_owners(since=marker[1])] == ["new"]
        assert {owner for owner, _ in txn.dirty_owners()} == {"old", "new"}
