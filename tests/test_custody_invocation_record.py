"""Invocation lookup must equal the independent chain oracle, including publication order."""
from __future__ import annotations

from copy import deepcopy

import pytest

from ouroboros import delegate_custody as c
from ouroboros import delegate_custody_current as current
from ouroboros import delegate_invocation as inv
from ouroboros import store


def _request(iid="inv", **facts):
    return {"type": c.START_REQUESTED, "invocation_id": iid, "task_id": "task",
            "root_task_id": "task", "surface": "task_acceptance", "operation_id": "operation",
            "slot_id": "slot", "request": {"prompt": "original"},
            "billing_group": {"billing_group_id": "original-group"}, **facts}


def _started(rid, iid="inv"):
    return {"type": c.STARTED, "invocation_id": iid, "run_id": rid, "task_id": "task"}


def _publish(root, row, number):
    event = {"event_id": f"event-{number}", "ts": f"2026-10-16T00:00:{number:02d}Z", **row}
    assert current.publish(root, event)


def _oracle_operation(root, rows):
    token = next((str(row.get("invocation_id") or "") for row in rows
                  if row["type"] == c.START_REQUESTED and row.get("operation_id") == "operation"
                  and row.get("slot_id") == "slot"), "")
    record = c.invocation_record(root, token, rows=rows) if token else None
    return {**record, "invocation_id": token} if record else None


@pytest.mark.parametrize("history", [
    [_request(), _started("healthy")],
    [_request(), {"type": c.START_FAILED, "invocation_id": "inv", "definite": True},
     _request(request={"prompt": "replacement"}, billing_group={"billing_group_id": "replacement-group"}),
     _started("b")],
    [_request(), _started("a"), _started("b")],
    [_request(), _started("z"), _started("a"), _started("z")],
], ids=["healthy", "refused-repeated-start", "last-start", "repeated-run-last-start"])
def test_live_invocation_equals_chain_at_every_prefix(tmp_path, history):
    for number, event in enumerate(history):
        _publish(tmp_path, event, number)
        rows = list(c.custody_rows(tmp_path))
        expected = c.invocation_record(tmp_path, "inv", rows=rows)
        assert c.invocation_record(tmp_path, "inv") == expected
    bound = current.run(tmp_path, expected["run_id"])
    assert bound.billing_group == history[0]["billing_group"]
    assert bound.request_event == rows[0]


@pytest.mark.parametrize("times", [
    ("2026-10-16T02:00:00Z", "2026-10-16T01:00:00Z"),
    ("2026-10-16T01:00:00Z", "2026-10-16T01:00:00Z"),
], ids=["inverted", "tied"])
def test_operation_uses_append_order_after_delayed_outbox_flush(tmp_path, monkeypatch, times):
    # Publish-time clocks tie too; event ids sort AGAINST the actual commit order.
    monkeypatch.setattr(store, "utc_now_iso", lambda: "2026-10-16T03:00:00Z")
    with monkeypatch.context() as paused:
        paused.setattr(current, "_append_durable", lambda *args: False)
        _publish(tmp_path, _request("z-first", ts=times[0], event_id="z-first"), 0)
        _publish(tmp_path, _request("a-second", ts=times[1], event_id="a-second"), 1)
    assert current.flush_outbox(tmp_path) == 2
    rows = list(c.custody_rows(tmp_path))
    assert [row["invocation_id"] for row in rows] == ["z-first", "a-second"]
    assert rows[0]["custody_ordinal"] < rows[1]["custody_ordinal"]
    assert inv.operation_record(tmp_path, "operation", "slot") == _oracle_operation(tmp_path, rows)
    # New transactions/connections must keep the same durable selection.
    _publish(tmp_path, _started("first-run", "z-first"), 2)
    _publish(tmp_path, _started("second-run", "a-second"), 3)
    rows = list(c.custody_rows(tmp_path))
    assert inv.operation_record(tmp_path, "operation", "slot") == _oracle_operation(tmp_path, rows)
    assert inv.operation_record(tmp_path, "operation", "slot")["run_id"] == "first-run"


def test_invocation_reads_return_independent_copies(tmp_path):
    _publish(tmp_path, _request(), 0)
    before = deepcopy(c.invocation_record(tmp_path, "inv"))
    returned = c.invocation_record(tmp_path, "inv")
    returned["request"]["prompt"] = "changed by reader"
    assert c.invocation_record(tmp_path, "inv") == before


def _legacy_rows(root, *, lose_bound_request=False):
    """Keep exactly the rows the previous writer knew how to retain."""
    with store.open_store(root, write=True) as txn:
        for name in (inv.INVOCATIONS, inv.OPERATIONS, inv.EVENTS):
            txn.replace_set(name, {})
        txn.set_meta(inv.COMPLETE, False)
        for rid, facts in txn.custody_rows().items():
            facts.pop("_invocation_started_ordinal", None)
            if lose_bound_request:
                facts.update(request_event={}, billing_group={})
            txn.put_custody(rid, facts, state=current.HELD)


@pytest.mark.parametrize("legacy", [False, True])
def test_refused_original_can_be_recovered_without_chain_reads(tmp_path, monkeypatch, legacy):
    for number, row in enumerate([_request(), {"type": c.START_FAILED, "invocation_id": "inv", "definite": True},
                                  _request(request={"prompt": "later"}), _started("b")]):
        _publish(tmp_path, row, number)
    expected = c.invocation_record(tmp_path, "inv", rows=list(c.custody_rows(tmp_path)))
    if legacy:
        _legacy_rows(tmp_path, lose_bound_request=True)
    monkeypatch.setattr(c, "custody_rows", lambda *a, **kw: pytest.fail("forensic replay in lookup"))
    monkeypatch.setattr(c, "_iter_rows", lambda *a, **kw: pytest.fail("chain replay in lookup"))
    assert c.invocation_record(tmp_path, "inv") == expected


def test_legacy_ambiguous_order_is_explicit_until_an_explicit_rebuild(tmp_path, monkeypatch):
    for number, row in enumerate([_request(), _started("a"), _started("b")]):
        _publish(tmp_path, row, number)
    rows = list(c.custody_rows(tmp_path))
    expected = c.invocation_record(tmp_path, "inv", rows=rows)
    _legacy_rows(tmp_path)
    with monkeypatch.context() as patched:
        patched.setattr(c, "_iter_rows", lambda *a, **kw: pytest.fail("legacy lookup replayed chain"))
        assert c.invocation_record(tmp_path, "inv") == {
            "state": inv.LEGACY_GAP, "run_id": "", "reason": "last_started_order_unknown"}
        assert inv.operation_record(tmp_path, "operation", "slot")["state"] == inv.LEGACY_GAP
        assert c.invocation_record(tmp_path, "absent") is None
    current.rebuild(tmp_path, replace=True)  # Explicit repair, never a reader or periodic fallback.
    monkeypatch.setattr(c, "_iter_rows", lambda *a, **kw: pytest.fail("replayed after rebuild"))
    assert c.invocation_record(tmp_path, "inv") == expected
    assert inv.operation_record(tmp_path, "operation", "slot") == {**expected, "invocation_id": "inv"}


@pytest.mark.parametrize("history", [
    [_started("b"), _request()],
    [_request(), _started("b"), {"type": c.START_FAILED, "invocation_id": "inv", "definite": True}],
    [{"type": c.START_FAILED, "invocation_id": "inv", "definite": True}, _request()],
], ids=["start-before-request", "refusal-after-start", "refusal-before-request"])
def test_live_and_imported_edge_orderings_equal_oracle(tmp_path, history):
    for number, row in enumerate(history):
        _publish(tmp_path, row, number)
        rows = list(c.custody_rows(tmp_path))
        assert c.invocation_record(tmp_path, "inv") == c.invocation_record(tmp_path, "inv", rows=rows)
    legacy_chain = [{k: v for k, v in row.items() if k != "custody_ordinal"} for row in rows]
    with store.open_store(tmp_path, write=True) as txn:
        current.write_fold(txn, current.fold(legacy_chain), replace=True)
    assert c.invocation_record(tmp_path, "inv") == c.invocation_record(tmp_path, "inv", rows=legacy_chain)


def test_duplicate_publication_does_not_change_last_started_or_outbox_order(tmp_path, monkeypatch):
    with monkeypatch.context() as paused:
        paused.setattr(current, "_append_durable", lambda *args: False)
        for number, row in enumerate([_request(), _started("a"), _started("b")]):
            _publish(tmp_path, row, number)
        _publish(tmp_path, _started("a"), 1)  # Same event, delayed after b's start.
        assert c.invocation_record(tmp_path, "inv")["run_id"] == "b"
    assert current.flush_outbox(tmp_path) == 3
    rows = list(c.custody_rows(tmp_path))
    assert [row["custody_ordinal"] for row in rows] == [1, 2, 3]
    assert c.invocation_record(tmp_path, "inv") == c.invocation_record(tmp_path, "inv", rows=rows)


@pytest.mark.parametrize("legacy", [False, True])
def test_addressed_lookup_work_is_independent_of_foreign_history(tmp_path, monkeypatch, legacy):
    _publish(tmp_path, _request(), 0)
    _publish(tmp_path, _started("b"), 1)
    if legacy:
        _legacy_rows(tmp_path)
    expected = c.invocation_record(tmp_path, "inv")
    real_open = store.open_store
    from contextlib import contextmanager

    steps = []

    @contextmanager
    def measured(*args, **kwargs):
        with real_open(*args, **kwargs) as txn:
            txn.conn.set_progress_handler(lambda: steps.append(1) and 0, 1)
            yield txn
            txn.conn.set_progress_handler(None, 0)

    costs = []
    for count in (10, 1000):
        with real_open(tmp_path, write=True) as txn:
            for number in range(count):
                txn.put_custody(f"foreign-{number}", {"invocation_id": f"foreign-{number}"}, state=current.CLOSED)
        with monkeypatch.context() as patched:
            patched.setattr(store, "open_store", measured)
            steps.clear()
            assert c.invocation_record(tmp_path, "inv") == expected
            costs.append(len(steps))
    assert costs[0] > 0 and costs[1] <= costs[0] + 20, costs
