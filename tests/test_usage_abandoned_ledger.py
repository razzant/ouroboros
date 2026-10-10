"""Late receipts keep their exact attempt: the store's write path and the
one-time journal import apply the same transition table (``usage_ledger``)."""

from __future__ import annotations

import hashlib
import json

import pytest

from ouroboros import usage_journal as journal
from ouroboros import usage_ledger as ledger
from ouroboros import usage_store
from tests._legacy_usage_rows_oracle import _summary as oracle_summary


def _row(attempt_id, state, **facts):
    return {
        "kind": "attempt", "attempt_id": attempt_id, "state": state,
        "ts": "2000-01-01T00:00:00+00:00", "model": "test-model",
        "provider": "test", "task_id": "child", "root_task_id": "root",
        "parent_task_id": "root", "category": "task", "source": "test",
        "reservation_upper_bound_usd": 1.25, "cost_usd": None,
        "cost_final": False, **facts,
    }


def _chain(attempt_id, final):
    rows = [_row(attempt_id, "reserved"), _row(attempt_id, "dispatched")]
    if final in {"unresolved", "abandoned"}:
        rows.append(_row(attempt_id, "unresolved", reason="response lost"))
    if final == "abandoned":
        rows.append(_row(attempt_id, "settled", settle_reason="abandoned"))
    elif final == "settled":
        rows.append(_row(attempt_id, "settled", cost_usd=0.5, cost_final=True))
    elif final == "released":
        rows.append(_row(attempt_id, "released", reason="before_dispatch_failed:not_started"))
    return rows


def _numbered(rows, start=1):
    return [{**row, "seq": seq} for seq, row in enumerate(rows, start)]


def _encoded(rows):
    return b"".join((json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows)


def _persist(root, rows):
    path = root / ledger.LEDGER_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_encoded(rows))
    return path


def _write(root, rows):
    """One store transaction: each row replaces its attempt's current row
    through the store's only write path."""
    stored = []
    with usage_store.hold(root) as txn:
        for row in rows:
            row = {key: value for key, value in row.items() if key != "seq"}
            stored.append(txn.write(row, txn.attempt(row["attempt_id"])))
    return stored


def _current(root, attempt_id):
    with usage_store.read(root) as txn:
        return txn.attempt(attempt_id)


@pytest.fixture
def root(tmp_path):
    yield tmp_path
    usage_store.forget(tmp_path)


@pytest.mark.parametrize("prior", ["unresolved", "abandoned"])
@pytest.mark.parametrize("cost", [0.0, 0.7, None])
def test_one_late_receipt_is_accepted_once_by_the_store_and_the_import(root, prior, cost):
    before = _numbered(_chain("late", prior))
    actual = _numbered([_row("late", "settled", settle_reason="late_receipt",
                            cost_usd=cost, cost_final=cost is not None,
                            prompt_tokens=20, completion_tokens=5)], len(before) + 1)
    journal._validate_records(before + actual)
    duplicate = _numbered(actual, len(before) + 2)
    with pytest.raises(ledger.UsageLedgerCorrupt, match="changed after terminal"):
        journal._validate_records(before + actual + duplicate)

    _write(root, before)
    assert ledger.late_receipt_eligible(_current(root, "late"))
    [stored] = _write(root, actual)
    assert stored["state"] == "settled" and not ledger.late_receipt_eligible(stored)
    with pytest.raises(ledger.UsageLedgerCorrupt, match="changed after terminal"):
        _write(root, duplicate)
    assert _current(root, "late") == stored


def test_a_reopened_store_keeps_then_consumes_the_abandonment_fact(root):
    _write(root, _chain("late", "unresolved"))
    [abandoned] = _write(root, [_row("late", "settled", settle_reason="abandoned")])
    assert ledger.is_abandoned_settlement(abandoned)
    # A fresh process reads the same permission from the stored row alone.
    usage_store.forget(root)
    assert ledger.late_receipt_eligible(_current(root, "late"))
    [actual] = _write(root, [_row("late", "settled", settle_reason="late_receipt", cost_usd=.4, cost_final=True)])
    assert not ledger.late_receipt_eligible(actual)
    usage_store.forget(root)
    with pytest.raises(ledger.UsageLedgerCorrupt, match="changed after terminal"):
        _write(root, [_row("late", "settled", settle_reason="late_receipt", cost_usd=.9, cost_final=True)])
    assert _current(root, "late") == actual


@pytest.mark.parametrize("prior", ["unresolved", "abandoned", "settled", "released"])
def test_untyped_settlement_cannot_change_a_terminal_attempt(root, prior):
    rows = _numbered(_chain("terminal", prior))
    update = _numbered([_row("terminal", "settled", cost_usd=.8, cost_final=True)], len(rows) + 1)
    with pytest.raises(ledger.UsageLedgerCorrupt, match="changed after terminal"):
        journal._validate_records(rows + update)
    terminal = _write(root, rows)[-1]
    with pytest.raises(ledger.UsageLedgerCorrupt, match="changed after terminal"):
        _write(root, update)
    assert _current(root, "terminal") == terminal


@pytest.mark.parametrize("prior", ["settled", "released"])
def test_late_marker_cannot_change_an_ordinary_terminal(root, prior):
    rows = _numbered(_chain("terminal", prior))
    update = _numbered([_row("terminal", "settled", settle_reason="late_receipt",
                            cost_usd=.8, cost_final=True)], len(rows) + 1)
    with pytest.raises(ledger.UsageLedgerCorrupt, match="changed after terminal"):
        journal._validate_records(rows + update)
    _write(root, rows)
    with pytest.raises(ledger.UsageLedgerCorrupt, match="changed after terminal"):
        _write(root, update)


@pytest.mark.parametrize("facts", [
    {"cost_usd": 0.0}, {"cost_usd": 1.25}, {"cost_final": True},
    {"cost_final": None}, {"state": "unresolved"}, {"kind": "subscription_session"},
])
def test_abandoned_marker_cannot_claim_a_price_or_another_kind(root, facts):
    base = _numbered(_chain("abandoned", "dispatched"))
    invalid = _row("abandoned", "settled", settle_reason="abandoned")
    invalid.update(facts)
    assert not ledger.is_abandoned_settlement(invalid)
    with pytest.raises(ledger.UsageLedgerCorrupt, match="invalid abandoned settlement"):
        journal._validate_records(base + _numbered([invalid], len(base) + 1))
    dispatched = _write(root, base)[-1]
    with pytest.raises(ledger.UsageLedgerCorrupt, match="invalid abandoned settlement"):
        _write(root, [invalid])
    assert _current(root, "abandoned") == dispatched


@pytest.mark.parametrize("prior", ["unresolved", "abandoned"])
def test_positive_never_started_receipt_can_release_once(root, prior):
    before = _numbered(_chain("late", prior))
    released = _numbered([_row("late", "released", reason="before_dispatch_failed:not_started")], len(before) + 1)
    journal._validate_records(before + released)
    untyped = [{**released[0], "reason": "cancelled"}]
    with pytest.raises(ledger.UsageLedgerCorrupt, match="changed after terminal"):
        journal._validate_records(before + untyped)
    with pytest.raises(ledger.UsageLedgerCorrupt, match="changed after terminal"):
        journal._validate_records(before + released + _numbered(released, len(before) + 2))

    _write(root, before)
    with pytest.raises(ledger.UsageLedgerCorrupt, match="changed after terminal"):
        _write(root, untyped)
    [stored] = _write(root, released)
    assert stored["state"] == "released" and not ledger.late_receipt_eligible(stored)
    with pytest.raises(ledger.UsageLedgerCorrupt, match="changed after terminal"):
        _write(root, released)


def test_legacy_unresolved_cannot_be_retyped_into_a_correctable_attempt(root):
    legacy = _numbered([_row("legacy", "unresolved", kind="legacy_call")])
    update = _numbered([_row("legacy", "settled", settle_reason="late_receipt",
                            cost_usd=.3, cost_final=True)], 2)
    with pytest.raises(ledger.UsageLedgerCorrupt, match="changed after terminal"):
        journal._validate_records(legacy + update)
    _write(root, legacy)
    with pytest.raises(ledger.UsageLedgerCorrupt, match="changed after terminal"):
        _write(root, update)


def test_the_import_keeps_correctable_attempts_until_the_real_receipt(root):
    raw_rows = [row for i in range(24) for row in _chain(f"closed-{i}", "settled")]
    raw_rows += _chain("unresolved", "unresolved") + _chain("abandoned", "abandoned")
    before = _numbered(raw_rows)
    before_bytes = _persist(root, before).read_bytes()
    finals = ledger._final_rows(before)
    with usage_store.read(root) as txn:  # the first access runs the one-time import
        assert txn.bucket("global").render_summary() == oracle_summary(list(finals.values()))
        for attempt_id in ("unresolved", "abandoned"):
            stored = txn.attempt(attempt_id)
            assert {key: stored.get(key) for key in finals[attempt_id] if key != "seq"} == {
                key: value for key, value in finals[attempt_id].items() if key != "seq"}
            assert ledger.late_receipt_eligible(stored)
    assert (root / ledger.LEDGER_REL).read_bytes() == before_bytes  # kept in place, never written again
    [correction] = _write(root, [_row("abandoned", "settled", settle_reason="late_receipt",
                                      cost_usd=.6, cost_final=True)])
    assert not ledger.is_abandoned_settlement(correction)
    assert ledger.late_receipt_eligible(_current(root, "unresolved"))


def test_previously_folded_unknown_groups_import_without_attempt_recreation(root):
    archived = _numbered(_chain("historical-unknown", "unresolved"))
    original_bytes = _encoded(archived)
    archive_rel = "archive/usage_ledger/historical.jsonl"
    segment = root / archive_rel
    segment.parent.mkdir(parents=True)
    segment.write_bytes(original_bytes)
    header = {"kind": "usage_baseline", "attempt_id": "old-baseline", "state": "settled",
              "seq": 1, "baseline_id": "old-baseline", "compaction_epoch": 1,
              "archive_rel": archive_rel, "source_sha256": hashlib.sha256(original_bytes).hexdigest(),
              "source_size_bytes": len(original_bytes), "source_row_count": 3,
              "source_first_seq": 1, "source_last_seq": 3, "folded_row_count": 3,
              "folded_attempt_count": 1, "group_count": 1, "retained_row_count": 0}
    group = _row("old-baseline-group", "unresolved", kind="usage_baseline_group",
                 baseline_id="old-baseline", folded_attempt_count=1,
                 reservation_upper_bound_usd="1.25", seq=2)
    rows = [header, group] + _numbered(_chain("ordinary", "settled"), 3)
    journal._validate_records(rows)
    _persist(root, rows)
    expected = oracle_summary([row for row in ledger._final_rows(rows).values()
                               if row.get("kind") != "usage_baseline"])
    with usage_store.read(root) as txn:
        assert txn.bucket("global").render_summary() == expected
        assert txn.attempt("historical-unknown") is None
        assert txn.attempt("old-baseline-group")["kind"] == "usage_baseline_group"
        assert txn.marker() == [1, 5]
    assert segment.read_bytes() == original_bytes
