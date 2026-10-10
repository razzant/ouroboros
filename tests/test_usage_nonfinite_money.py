"""Nonfinite monetary evidence fails closed without deleting a liability."""
from __future__ import annotations

import contextlib
import json
from decimal import Decimal

import pytest

from ouroboros import usage_accounting as ua
from ouroboros import usage_journal as journal
from ouroboros import usage_ledger as ledger
from ouroboros import usage_store
from ouroboros.usage_journal import IMPORT_REL
from tests._usage_store_testing import request, root as root, seed, write_journal


@contextlib.contextmanager
def nonfinite_failure():
    with pytest.raises(ua.UsageAccountingError, match="non-finite") as error:
        yield
    assert type(error.value).__name__ == "UsageNonFiniteMoney"
    assert not isinstance(error.value, (ledger.UsageLedgerCorrupt, ledger.UsageLockUnavailable))


def archive_bytes(root):
    return {path.relative_to(root): path.read_bytes()
            for path in (root / "archive").rglob("*") if path.is_file()}


_HELD = {"attempt_id": "held", "kind": "attempt", "state": "reserved", "task_id": "child",
         "root_task_id": "dominant", "provider": "local", "model": "stub",
         "reservation_upper_bound_usd": "0.1", "pricing_known": True}


def _held(root):
    return ua.AttemptReservation("held", root, "stub", "local", 0.1, "", "", None)


def _refuses_everywhere(root, held):
    """Every non-display money access re-runs the refused import and fails closed; a display
    reports the store unavailable (never zero) without running the import."""
    sends = []
    with nonfinite_failure():
        ua.usage_projection(root)
    with pytest.raises(ledger.UsageLockUnavailable):
        ua.usage_projection(root, allow_stale=True)
    with nonfinite_failure():
        ua.mark_dispatched(held)
    with nonfinite_failure():
        ua.execute_physical_attempt(
            request(root, reservation_usd=.1, global_limit_usd=.5), lambda: sends.append(1))
    assert sends == []
    assert not (root / usage_store.STORE_REL).exists()
    assert not (root / ledger.QUARANTINE_REL).exists()


@pytest.mark.parametrize("value", ["Infinity", "-Infinity", "NaN", float("inf"), float("nan")])
@pytest.mark.parametrize("source", ["cost", "usage.cost", "usage.total_cost", "state"])
def test_real_legacy_import_refuses_nonfinite_before_projection_or_send(root, value, source):
    path = write_journal(root, [_HELD])
    original = path.read_bytes()
    events_path, state_path = root / "logs/events.jsonl", root / "state/state.json"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    event = {"type": "llm_usage", "task_id": "legacy", "root_task_id": "legacy", "provider": "openai"}
    if source == "state":
        state_path.write_text(json.dumps({"spent_usd": value}))
    elif source == "cost":
        event["cost"] = value
    else:
        event["usage"] = {source.split(".")[1]: value}
    events_path.write_text(json.dumps(event) + "\n")
    sources = {file: file.read_bytes() for file in (events_path, state_path) if file.exists()}
    with nonfinite_failure():
        usage_store.migrate_from_journal(root)
    archived = archive_bytes(root)
    assert sources[events_path] in archived.values()
    if state_path in sources:
        assert sources[state_path] in archived.values()
    _refuses_everywhere(root, _held(root))
    assert path.read_bytes() == original
    assert not (root / IMPORT_REL).exists()
    assert archive_bytes(root) == archived
    assert all(file.read_bytes() == data for file, data in sources.items())


@pytest.mark.parametrize("field", ["cost_usd", "reservation_upper_bound_usd"])
@pytest.mark.parametrize("value", ["Infinity", "-Infinity", "NaN", float("inf"), float("nan")])
def test_a_persisted_nonfinite_journal_row_is_never_quarantined_or_imported(root, field, value):
    seed(root, 12, cost=".01")
    path = root / ledger.LEDGER_REL
    prefix = path.read_bytes()
    count = len(prefix.splitlines())
    held = {**_HELD, "seq": count + 1, "ts": "2020-01-01T00:00:00Z"}
    bad = {"seq": count + 2, "attempt_id": "legacy-nonfinite", "kind": "legacy_usage",
           "root_task_id": "legacy", "state": "settled", "cost_usd": .01,
           "cost_final": True, "reservation_upper_bound_usd": .2, field: value}
    body = prefix + (json.dumps(held) + "\n").encode() + (json.dumps(bad) + "\n").encode()
    path.write_bytes(body)
    with nonfinite_failure():
        journal._validate_records([json.loads(line) for line in body.splitlines()])
    with nonfinite_failure():
        ua._summary([bad])
    _refuses_everywhere(root, _held(root))
    assert path.read_bytes() == body


@pytest.mark.parametrize("field", ["cost_usd", "reservation_upper_bound_usd", "reservation_usd",
                                   "max_budget_usd", "global_limit_usd", "root_limit_usd"])
def test_the_store_never_persists_nonfinite_money(root, field):
    held = ua.reserve_attempt(request(root, reservation_usd=.1))
    with usage_store.read(root) as txn:
        before = txn.attempts()
    bad = {"attempt_id": "legacy-nonfinite", "kind": "legacy_usage", "root_task_id": "legacy",
           "state": "settled", "cost_usd": .01, "cost_final": True, field: "Infinity"}
    with nonfinite_failure():
        with usage_store.hold(root) as txn:
            txn.write(bad)
    current = dict(before[-1])
    with nonfinite_failure():
        with usage_store.hold(root) as txn:
            txn.write({**current, "state": "dispatched", field: float("inf")}, current)
    with usage_store.read(root) as txn:
        assert txn.attempts() == before
    ua.mark_dispatched(held)  # the refused writes left the reservation intact
    assert not (root / ledger.QUARANTINE_REL).exists()


@pytest.mark.parametrize("field", ["reservation_usd", "max_budget_usd", "global_limit_usd", "root_limit_usd"])
def test_other_persisted_money_fields_refuse_without_tail_repair(root, field):
    dispatched = {**_HELD, "state": "dispatched", field: "Infinity"}
    path = write_journal(root, [_HELD, dispatched])
    body = path.read_bytes()
    _refuses_everywhere(root, _held(root))
    assert path.read_bytes() == body


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"),
                                   " Infinity ", "-inf", "NaN", Decimal("sNaN")])
def test_exact_amount_never_turns_nonfinite_into_unknown(value):
    from ouroboros._usage_money import amount

    with nonfinite_failure():
        amount(value)


@pytest.mark.parametrize("value", [.125, "1.25e-1", " .125 "])
def test_finite_legacy_import_projection_admission_and_send_survive(root, value):
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "logs/events.jsonl").write_text(json.dumps({
        "type": "llm_usage", "task_id": "legacy", "root_task_id": "legacy",
        "cost": value, "provider": "openai", "prompt_tokens": 7}) + "\n")
    assert usage_store.migrate_from_journal(root)["status"] == "completed"
    assert json.loads((root / IMPORT_REL).read_text(encoding="utf-8"))["completed"] is True
    projection = ua.usage_projection(root, root_task_id="legacy")
    assert projection["accounted_usd"] == .125
    assert projection["cost_final"] is True
    assert projection["priced_rows"] == 1
    assert projection["attempt_counts"] == {"settled": 1}
    assert ua.usage_breakdown(root)["prompt_tokens"] == 7
    sends, response = [], object()
    assert ua.execute_physical_attempt(
        request(root, root_task_id="legacy", reservation_usd=.1, global_limit_usd=.5),
        lambda: sends.append(1) or response, extractor=lambda _: ({}, .1, True)) is response
    assert sends == [1]
    assert ua.usage_projection(root)["accounted_usd"] == .225
