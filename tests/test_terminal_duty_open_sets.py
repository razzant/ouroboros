"""The duty follows outstanding custody and projection debt, independent of history."""
import json
from types import SimpleNamespace

import pytest

from ouroboros import usage_accounting as usage, usage_store
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from tests.test_usage_abandoned_reconciliation import (
    env as env, _attempt, _terminal, _final, _remote_request, _remote_receipt,
)
from ouroboros.terminal_cost_reconciliation import reconcile_abandoned_usage


@pytest.mark.parametrize("state", ["reserved", "dispatched", "unresolved", "abandoned"])
def test_candidates_come_only_from_open_set(env, monkeypatch, state):
    _terminal(env)
    reservation = _attempt(env, state="dispatched" if state == "abandoned" else state, provider="claudexor")
    if state == "abandoned":
        usage.terminalize_abandoned_attempt(reservation, reason="fixture")
    _remote_receipt(env, reservation, "operation")
    monkeypatch.setattr(usage, "read_usage_records", lambda *a, **k: pytest.fail("history read"))
    monkeypatch.setattr(usage_store, "read_usage_records", lambda *a, **k: pytest.fail("history read"))
    reconcile_abandoned_usage(env.root)
    with usage_store.read(env.root) as txn:
        row = txn.attempt(reservation.attempt_id)
    assert row["state"] == ("released" if state == "reserved" else "settled")
    if state != "reserved":
        assert row["cost_usd"] == 0.4


def test_operation_gone_stops_http_but_late_receipt_settles(env, monkeypatch):
    _terminal(env)
    reservation = _attempt(env, provider="claudexor")
    _remote_request(env, reservation, "gone")
    calls = []

    def gone(*a, **kw):
        assert env.lock_depth[0] == 0
        calls.append(kw)
        raise ClaudexorUnavailable("not_found", "gone", status_code=404)

    monkeypatch.setattr("ouroboros.claudexor_daemon.read_owned_gateway",
                        lambda: SimpleNamespace(get_model_operation=gone, close=lambda: None))
    reconcile_abandoned_usage(env.root)
    first = _final(env, reservation)
    assert first["recovery"]["outcome"] == "operation_gone"
    assert first["recovery"]["basis"] == {"operation_get_status": 404, "code": "not_found"}
    reconcile_abandoned_usage(env.root)
    assert len(calls) == 1 and calls[0]["timeout_sec"] > 0
    assert _final(env, reservation) == first
    _remote_receipt(env, reservation, "gone")
    reconcile_abandoned_usage(env.root)
    row = _final(env, reservation)
    assert row["settle_reason"] == "late_receipt" and row["cost_usd"] == 0.4


def test_retry_after_is_a_provider_fact_and_unresolved_without_it_is_retried(env, monkeypatch):
    _terminal(env)
    reservation = _attempt(env, provider="claudexor")
    _remote_request(env, reservation, "limited")
    calls = []

    def limited(*a, **kw):
        calls.append(kw)
        error = ClaudexorUnavailable("limited", "retry", status_code=429)
        error.retry_after = "120"
        raise error

    gateway = SimpleNamespace(get_model_operation=limited, close=lambda: None)
    monkeypatch.setattr("ouroboros.claudexor_daemon.read_owned_gateway", lambda: gateway)
    reconcile_abandoned_usage(env.root)
    row = _final(env, reservation)
    assert row["recovery"]["basis"] == {"header": "Retry-After", "value": "120", "code": "limited"}
    assert row["recovery"]["due_at"] > row["recovery"]["at"]
    reconcile_abandoned_usage(env.root)
    assert len(calls) == 1
    with usage_store.hold(env.root) as txn:
        txn.record_recovery(reservation.attempt_id, {**row["recovery"], "due_at": "2000-01-01T00:00:00+00:00"},
                            expected_revision=row["revision"])
    gateway.get_model_operation = lambda *a, **kw: {"state": "running"}
    for _ in range(2):
        reconcile_abandoned_usage(env.root)
    events = [json.loads(line) for line in (env.root / "logs/supervisor.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row['by_basis'] for row in events if row.get('type') == 'duty_unresolved'
            and row['subject'] == 'attempt'] == [{'limited': 1}, {'receipt_or_terminal_custody_unavailable': 1}]


def test_metadata_update_preserves_money_and_receipt_race(env):
    _terminal(env)
    reservation = _attempt(env)
    before = usage.usage_projection(env.root)
    row = _final(env, reservation)
    with usage_store.hold(env.root) as txn:
        assert txn.record_recovery(reservation.attempt_id, {"outcome": "unresolved"}, expected_revision=row["revision"])
    assert usage.usage_projection(env.root) == before
    usage.settle_attempt(reservation, cost_usd=0.6, cost_final=True)
    with usage_store.hold(env.root) as txn:
        assert not txn.record_recovery(reservation.attempt_id, {"outcome": "terminal"}, expected_revision=row["revision"])
    usage.settle_attempt(reservation, cost_usd=0.1, cost_final=True, expected_revision=row["revision"])
    assert _final(env, reservation)["cost_usd"] == 0.6


def test_unresolved_rows_publish_once_without_an_interprocess_lock(tmp_path, monkeypatch):
    from ouroboros.terminal_cost_reconciliation import _publish_unresolved, _unresolved

    events, delivered = {}, []
    for attempt_id in ('first', 'second'):
        _unresolved(events, {'task_id': 'task', 'attempt_id': attempt_id}, 'custody_unknown')
    monkeypatch.setattr('ouroboros.platform_layer.acquire_exclusive_file_lock',
                        lambda *a, **kw: pytest.fail('duty observation acquired a lockfile'))
    monkeypatch.setattr('supervisor.message_bus.try_get_bridge',
                        lambda: SimpleNamespace(push_log=delivered.append))
    _publish_unresolved(tmp_path, events)
    stored = [json.loads(line) for line in
              (tmp_path / 'logs/supervisor.jsonl').read_text(encoding='utf-8').splitlines()]
    assert stored == delivered and len(stored) == 1
    assert stored[0]['count'] == 2 and stored[0]['attempt_ids'] == ['first', 'second']
    assert stored[0]['by_basis'] == {'custody_unknown': 2} and stored[0]['subject'] == 'attempt'
    assert stored[0]['reasons'] == {'first': 'custody_unknown', 'second': 'custody_unknown'}


def test_unresolved_summary_changes_once_and_caps_one_row_for_300_candidates(env, monkeypatch):
    from ouroboros import terminal_cost_reconciliation as duty

    _terminal(env)
    row = _final(env, _attempt(env, provider='claudexor'))
    rows = [{**row, 'attempt_id': f'row-{index:03}'} for index in range(2)]
    monkeypatch.setattr(usage_store.Txn, 'open_attempts', lambda self: list(rows))
    monkeypatch.setattr('ouroboros.llm_claudexor.recover_model_attempt', lambda *a, **kw: None)
    monkeypatch.setattr(duty, '_LAST_UNRESOLVED', {})
    def observed():
        events = (json.loads(line) for line in
                  (env.root / 'logs/supervisor.jsonl').read_text(encoding='utf-8').splitlines())
        return [event for event in events if event.get('type') == 'duty_unresolved' and event['subject'] == 'attempt']
    for _ in range(2):
        reconcile_abandoned_usage(env.root)
    assert [event['count'] for event in observed()] == [2]
    rows.append({**row, 'attempt_id': 'row-002'})
    reconcile_abandoned_usage(env.root)
    assert [event['count'] for event in observed()] == [2, 3]
    rows.extend({**row, 'attempt_id': f'row-{index:03}'} for index in range(3, 300))
    for _ in range(2):
        reconcile_abandoned_usage(env.root)
    events = observed()
    assert [event['count'] for event in events] == [2, 3, 300]
    assert events[-1]['by_basis'] == {'receipt_or_terminal_custody_unavailable': 300}
    assert events[-1]['attempt_ids'] == [f'row-{index:03}' for index in range(50)]
    rows[-1] = {**row, 'attempt_id': 'row-999'}  # Beyond the visible 50, same count and reasons.
    reconcile_abandoned_usage(env.root)
    assert len(observed()) == 4
    duty._LAST_UNRESOLVED.clear()  # A new process reports the current set once again.
    reconcile_abandoned_usage(env.root)
    assert len(observed()) == 5
    rows.clear()
    for _ in range(2):
        reconcile_abandoned_usage(env.root)
    assert [event['count'] for event in observed()] == [2, 3, 300, 300, 300, 0]
    assert observed()[-1]['by_basis'] == {} and observed()[-1]['attempt_ids'] == []


@pytest.mark.parametrize('history', [10, 2000])
def test_completed_history_does_not_add_duty_reads(env, monkeypatch, history):
    from collections import Counter
    from pathlib import Path
    from ouroboros import task_results
    from supervisor import queue, queue_transitions, workers

    exemplar = task_results.write_task_result(env.root, 'history', 'completed', result='Stored')
    for index in range(history):
        (env.root / 'task_results' / f'history-{index}.json').write_text(
            json.dumps({**exemplar, 'task_id': f'history-{index}'}), encoding='utf-8')
    for index in range(5):
        tid = f'open-{index}'
        task_results.write_task_result(env.root, tid, 'completed', result='Stored')
        reservation = usage.reserve_attempt(usage.AttemptRequest(
            model='test', provider='claudexor', drive_root=env.root, task_id=tid,
            root_task_id=tid, reservation_usd=1, global_limit_usd=100))
        usage.mark_dispatched(reservation)
    monkeypatch.setattr(queue, 'task_has_live_ownership', queue_transitions.task_has_live_ownership)
    monkeypatch.setattr(queue, 'RUNNING', {})
    monkeypatch.setattr(queue, 'PENDING', [])
    monkeypatch.setattr(workers, 'WORKERS', {})
    monkeypatch.setattr(workers, 'direct_chat_turn', lambda _: None)
    monkeypatch.setattr('ouroboros.llm_claudexor.recover_model_attempt', lambda *a, **k: None)
    counts = Counter()
    load, decode = task_results.load_task_result, usage_store._decode
    body, bucket, owners = Path.read_text, usage_store._bucket_of, usage_store.Txn.dirty_owners
    def read(root, tid, **kw):
        assert not queue._queue_lock._is_owned()
        counts['ownership_loads'] += 1
        return load(root, tid, **kw)
    def read_body(path, *a, **kw):
        if path.parent == env.root / 'task_results' and path.suffix == '.json':
            assert not queue._queue_lock._is_owned()
            counts['result_parses'] += 1
        return body(path, *a, **kw)
    def decoded(row):
        counts['attempt_rows'] += 1
        return decode(row)
    def summary(row):
        counts['summary_rows'] += 1
        return bucket(row)
    def dirty(txn):
        rows = owners(txn)
        counts['dirty_rows'] += len(rows)
        return rows
    monkeypatch.setattr(Path, 'read_text', read_body)
    monkeypatch.setattr(task_results, 'load_task_result', read)
    monkeypatch.setattr(usage_store, '_decode', decoded)
    monkeypatch.setattr(usage_store, '_bucket_of', summary)
    monkeypatch.setattr(usage_store.Txn, 'dirty_owners', dirty)
    monkeypatch.setattr(usage, 'read_usage_records', lambda *a, **kw: pytest.fail('history fold'))
    reconcile_abandoned_usage(env.root)
    # Five ownership parses plus the existing atomic result writer's five
    # merge reads. No completed-history result or attempt is opened.
    assert counts == {'ownership_loads': 5, 'result_parses': 10, 'attempt_rows': 5,
                      'summary_rows': 10, 'dirty_rows': 5}


def test_one_inflight_probe_per_attempt(env, monkeypatch):
    import threading

    _terminal(env)
    _attempt(env, provider='claudexor')
    entered, release = threading.Event(), threading.Event()
    calls = []
    def recover(*a, **kw):
        calls.append(True)
        entered.set()
        assert release.wait(10), 'test did not release the probe'
        return None
    monkeypatch.setattr('ouroboros.llm_claudexor.recover_model_attempt', recover)
    worker = threading.Thread(target=reconcile_abandoned_usage, args=(env.root,))
    worker.start()
    try:
        assert entered.wait(10), 'probe did not start'
        reconcile_abandoned_usage(env.root)
        assert len(calls) == 1
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive()
