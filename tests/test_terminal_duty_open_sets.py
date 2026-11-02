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


def _reads(monkeypatch):
    """Record every attempt the duty hands to the real offline/HTTP recovery."""
    from ouroboros import llm_claudexor

    reads, recover = [], llm_claudexor.recover_model_attempt
    monkeypatch.setattr(llm_claudexor, "recover_model_attempt",
                        lambda root, row, **kw: reads.append(row["attempt_id"]) or recover(root, row, **kw))
    return reads


def test_operation_gone_ends_the_search_but_keeps_the_late_receipt_right(env, monkeypatch):
    _terminal(env)
    reservation = _attempt(env, provider="claudexor")
    _remote_request(env, reservation, "gone")
    calls, reads = [], _reads(monkeypatch)

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
    # The engine said the call is over: no payload read and no HTTP, even with a retained receipt.
    _remote_receipt(env, reservation, "gone")
    reconcile_abandoned_usage(env.root)
    assert len(calls) == 1 and calls[0]["timeout_sec"] > 0 and reads == [reservation.attempt_id]
    assert _final(env, reservation) == first
    # The ONE late receipt stays an accounting right for an explicit writer.
    usage.settle_attempt(reservation, cost_usd=0.4, cost_final=True)
    row = _final(env, reservation)
    assert row["settle_reason"] == "late_receipt" and row["cost_usd"] == 0.4


def _record(env, reservation, fact):
    with usage_store.hold(env.root) as txn:
        assert txn.record_recovery(reservation.attempt_id, fact,
                                   expected_revision=txn.attempt(reservation.attempt_id)["revision"])


_ABSENT = object()


@pytest.mark.parametrize("outcome", [_ABSENT, None, "terminal", "operation_gone", "unresolved", "other"])
def test_a_recorded_ended_custody_alone_takes_an_abandoned_row_out(env, monkeypatch, outcome):
    """Absent, null and every other outcome keep the row selected; its retained receipt settles offline."""
    _terminal(env)
    reservation = _attempt(env, provider="claudexor")
    usage.terminalize_abandoned_attempt(reservation, reason="fixture")
    if outcome is not _ABSENT:
        _record(env, reservation, {"outcome": outcome})
    _remote_request(env, reservation, "operation")
    _remote_receipt(env, reservation, "operation")
    before, reads = _final(env, reservation), _reads(monkeypatch)
    reconcile_abandoned_usage(env.root)
    row = _final(env, reservation)
    if outcome in {"terminal", "operation_gone"}:
        assert reads == [] and row == before  # Cost unknown; nothing searched or rewritten.
    else:
        assert reads == [reservation.attempt_id]
        assert row["settle_reason"] == "late_receipt" and row["cost_usd"] == 0.4


def _old_filter(row):
    """The duty's per-row filter before the selector (954ef8aff), applied to the full open set."""
    from ouroboros._usage_rows import REVIEW_CUSTODY_KEYS
    from ouroboros.usage_ledger import is_abandoned_settlement

    kind = row.get("kind", "attempt")
    if kind not in {"attempt", "usage_baseline_group"} or any(row.get(key) for key in REVIEW_CUSTODY_KEYS):
        return False
    remote, abandoned = row.get("provider") == "claudexor", is_abandoned_settlement(row)
    return not (kind != "attempt" or (row.get("state") not in {"reserved", "dispatched", "unresolved"}
                                      and not (remote and abandoned)))


def _clone(env, source, variants, tag="copy"):
    """Fixture rows written straight into ``attempts``: copies of one real row with changed columns/extra."""
    with usage_store.hold(env.root) as txn:
        record = dict(txn.conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (source,)).fetchone())
        top = txn.conn.execute("SELECT MAX(seq) FROM attempts").fetchone()[0]
        for index, (columns, edit) in enumerate(variants, 1):
            extra = json.loads(record["extra"])
            edit(extra)
            row = {**record, **columns, "attempt_id": f"{source}-{tag}-{index}", "seq": top + index,
                   "seq_first": top + index, "extra": json.dumps(extra)}
            txn.conn.execute(f"INSERT INTO attempts ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
                             tuple(row.values()))


def _outcome(value):
    def edit(extra):
        if value is not _ABSENT:
            extra["recovery"] = {"outcome": value, "at": "2026-10-10T00:00:00+00:00", "basis": "fixture"}
    return edit


def test_selection_is_the_old_full_scan_minus_ended_custody(env):
    import itertools

    from ouroboros import terminal_cost_reconciliation as duty
    from ouroboros._usage_rows import REVIEW_CUSTODY_KEYS
    from ouroboros.usage_ledger import is_abandoned_settlement

    _terminal(env)
    bases = [_attempt(env, state=state, provider="claudexor") for state in ("reserved", "dispatched", "unresolved")]
    usage.terminalize_abandoned_attempt(abandoned := _attempt(env, provider="claudexor"), reason="fixture")
    usage.settle_attempt(receipt := _attempt(env, provider="claudexor"), cost_usd=0.4, cost_final=True)
    sources = [(row.attempt_id, {}) for row in (*bases, abandoned, receipt)] + [(abandoned.attempt_id, {"cost_usd": "0.4"})]
    kinds = [({"kind": "attempt"}, lambda extra: None), ({"kind": None}, lambda extra: extra.pop("kind", None)),
             ({"kind": None}, lambda extra: extra.setdefault("__none__", []).append("kind")),
             ({"kind": "usage_baseline_group"}, lambda extra: None)]
    reviews = [{}, {"review_skill": "skill"}, {"review_slot_id": "slot"}, {"review_wave_id": "wave"}]
    providers = ["claudexor", "openrouter", None]
    outcomes = [_ABSENT, None, "terminal", "operation_gone", "unresolved", "other"]
    for index, (source, override) in enumerate(sources):
        variants = []
        for (kind, kind_edit), review, provider, outcome in itertools.product(kinds, reviews, providers, outcomes):
            def edit(extra, kind_edit=kind_edit, outcome=outcome, override=override):
                kind_edit(extra)
                if "cost_usd" in override:
                    extra["__none__"] = [key for key in extra.get("__none__", []) if key != "cost_usd"]
                _outcome(outcome)(extra)
            variants.append(({**kind, **review, "provider": provider, **override}, edit))
        _clone(env, source, variants, tag=str(index))
    with usage_store.read(env.root) as txn:
        full = txn.open_attempts()
        selected = txn.attempts(f"{usage_store._OPEN_PREDICATE} AND {duty._RECOVERABLE}")
    old = {row["attempt_id"] for row in full if _old_filter(row)}
    ended = {row["attempt_id"] for row in full if _old_filter(row) and is_abandoned_settlement(row)
             and (row.get("recovery") or {}).get("outcome") in {"terminal", "operation_gone"}}
    assert len(full) > 1000 and ended and old - ended
    assert {row["attempt_id"] for row in selected if duty._candidate(row)} == old - ended
    # Mechanically out: imported aggregates, review custody, abandoned rows outside the engine.
    assert not [row for row in selected if row.get("kind") == "usage_baseline_group"
                or any(row.get(key) for key in REVIEW_CUSTODY_KEYS)
                or row.get("state") == "settled" and row.get("provider") != "claudexor"]


@pytest.mark.parametrize("history", [10, 2000])
def test_ended_custody_adds_no_decode_or_payload_read(env, monkeypatch, history):
    _terminal(env)
    usage.terminalize_abandoned_attempt(ended := _attempt(env, provider="claudexor"), reason="fixture")
    _record(env, ended, {"outcome": "terminal"})
    _clone(env, ended.attempt_id, [({}, lambda extra: None)] * history)
    _remote_request(env, live := _attempt(env, provider="claudexor"), "running")
    gateway = SimpleNamespace(get_model_operation=lambda *a, **kw: {"state": "running"}, close=lambda: None)
    monkeypatch.setattr("ouroboros.claudexor_daemon.read_owned_gateway", lambda: gateway)
    decoded, decode, reads = [], usage_store._decode, _reads(monkeypatch)
    monkeypatch.setattr(usage_store, "_decode", lambda record: decoded.append(record["attempt_id"]) or decode(record))
    reconcile_abandoned_usage(env.root)
    assert reads == [live.attempt_id] and decoded == [live.attempt_id]


def test_ended_custody_selection_is_recomputed_from_rows_after_a_restart(env, monkeypatch):
    from ouroboros import terminal_cost_reconciliation as duty
    from ouroboros.usage_ledger import is_abandoned_settlement

    _terminal(env)
    ended, running = _attempt(env, provider="claudexor"), _attempt(env, provider="claudexor")
    _remote_request(env, ended, "ended")
    _remote_request(env, running, "running")
    calls, reads = [], _reads(monkeypatch)

    def operation(operation_id, **_kw):
        calls.append(operation_id)
        return {"state": "cancelled", "dispatch": {"state": "unknown"}} if operation_id == "ended" else {"state": "running"}

    monkeypatch.setattr("ouroboros.claudexor_daemon.read_owned_gateway",
                        lambda: SimpleNamespace(get_model_operation=operation, close=lambda: None))
    reconcile_abandoned_usage(env.root)
    row = _final(env, ended)
    assert is_abandoned_settlement(row) and row["recovery"]["outcome"] == "terminal"
    assert calls == ["ended", "running"] and reads == [ended.attempt_id, running.attempt_id]
    # A new process remembers nothing: the recorded rows alone decide the selection.
    usage_store.forget(env.root)
    monkeypatch.setattr(duty, "_LAST_UNRESOLVED", {})
    reconcile_abandoned_usage(env.root)
    assert calls == ["ended", "running", "running"] and _final(env, ended) == row
    assert reads == [ended.attempt_id, running.attempt_id, running.attempt_id]


def test_a_receipt_racing_the_terminal_mark_stays_authoritative(env, monkeypatch):
    _terminal(env)
    reservation = _attempt(env, provider="claudexor")
    usage.terminalize_abandoned_attempt(reservation, reason="fixture")

    def recover(root, row, **_kw):
        usage.settle_attempt(reservation, cost_usd=0.6, cost_final=True)  # A late receipt lands mid-pass.
        return "abandoned", {}, None, False

    monkeypatch.setattr("ouroboros.llm_claudexor.recover_model_attempt", recover)
    reconcile_abandoned_usage(env.root)
    row = _final(env, reservation)
    assert row["settle_reason"] == "late_receipt" and row["cost_usd"] == 0.6 and "recovery" not in row


@pytest.mark.parametrize("taken", [True, False])
@pytest.mark.parametrize("operation", ["cancelled", "not_started", "gone"])
def test_ownership_taken_during_a_blocked_probe_refuses_the_settlement(env, monkeypatch, operation, taken):
    """Discovery eligibility is cached; the post-I/O fences re-read ownership before any write."""
    import threading

    _terminal(env)
    reservation = _attempt(env, provider="claudexor")
    _remote_request(env, reservation, "operation")
    entered, release = threading.Event(), threading.Event()

    def blocked(*_a, **_kw):
        entered.set()
        assert release.wait(10), "test did not release the probe"
        if operation == "gone":
            raise ClaudexorUnavailable("not_found", "gone", status_code=404)
        return {"state": "failed", "dispatch": {"state": "not_started"}} if operation == "not_started" else {
            "state": "cancelled", "dispatch": {"state": "unknown"}}

    monkeypatch.setattr("ouroboros.claudexor_daemon.read_owned_gateway",
                        lambda: SimpleNamespace(get_model_operation=blocked, close=lambda: None))
    worker = threading.Thread(target=reconcile_abandoned_usage, args=(env.root,))
    worker.start()
    try:
        assert entered.wait(10), "probe did not start"
        if taken:
            env.live.add("child")  # An owner Continue re-admits the task while the probe waits.
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive()
    row = _final(env, reservation)
    if taken:
        assert row["state"] == "dispatched" and "recovery" not in row
    elif operation == "not_started":
        assert row["state"] == "released"
    else:
        assert row["state"] == "settled" and row["recovery"]["outcome"] == (
            "operation_gone" if operation == "gone" else "terminal")


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
    monkeypatch.setattr(usage_store, 'decode_attempts', lambda captured: list(rows))  # the open set the read captured
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
    from ouroboros import store, task_results
    from supervisor import queue, queue_transitions, workers

    exemplar = task_results.write_task_result(env.root, 'history', 'completed', result='Stored')
    with store.open_store(env.root, write=True) as txn:
        for index in range(history):
            txn.store_task({**exemplar, 'task_id': f'history-{index}'})
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
    body, bucket, owners = store.Txn.body_text, usage_store._bucket_of, usage_store.Txn.dirty_owners
    def read(root, tid, **kw):
        assert not queue._queue_lock._is_owned()
        counts['ownership_loads'] += 1
        return load(root, tid, **kw)
    def read_body(txn, tid):
        assert not queue._queue_lock._is_owned()
        counts['result_parses'] += 1
        return body(txn, tid)
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
    monkeypatch.setattr(store.Txn, 'body_text', read_body)
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
