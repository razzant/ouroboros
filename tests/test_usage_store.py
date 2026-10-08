"""The usage store (docs/USAGE_STORE.md): exact summaries, transition rules,
history independence, the one-time import, the downgrade export, the two lock
tiers and the bulk-reader allowlist."""
from __future__ import annotations

import ast
import json
import os
import pathlib
import random
import subprocess
import sys
from decimal import Decimal

import pytest

from ouroboros import usage_accounting as ua
from ouroboros import usage_ledger as ledger
from ouroboros import usage_store
from ouroboros._usage_money import billing_group_key, monetary_scope_key
from tests import _legacy_usage_rows_oracle as oracle
from tests._usage_store_testing import ledger_rows, request, root as root, write_compacted_journal, write_journal

REPO = pathlib.Path(__file__).resolve().parents[1]
_AXES = ("model", "provider", "category")
_LEGACY_UNATTRIBUTED = {"legacy_metadata", "legacy_delta"}


# ---- summaries equal the frozen fold over the addressed attempts --------------

def _addresses(row):
    """Every ``(scope, key)`` the spec says a row is addressed by (written here
    from the scope list, independently of ``usage_store.summary_keys``)."""
    kind, task = str(row.get("kind") or ""), str(row.get("task_id") or "")
    root_id, group = monetary_scope_key(row), billing_group_key(row)
    legacy = kind in _LEGACY_UNATTRIBUTED
    found = {("global", ""), ("root", root_id), ("group", group), ("kind", kind)}
    if task:
        found.add(("task", task))
    for axis in _AXES:
        value = "" if legacy else str(row.get(axis) or "")
        found.add((axis, value) if value else ("unattributed", axis))
        if root_id:
            found.add((f"root_{axis}", f"{root_id}|{value}"))
        if task:
            found.add((f"task_{axis}", f"{task}|{value}"))
    if legacy or not task:
        found.add(("unattributed", "task"))
    if legacy or not root_id:
        found.add(("unattributed", "root"))
    if root_id:
        found.add(("root_task", f"{root_id}|{'' if legacy else task}"))
    if task:
        found.add(("task_root", f"{task}|{'' if legacy else root_id}"))
    if kind == "subscription_session":
        found.update(item for item in (("root_delegated", root_id), ("task_delegated", task)) if item[1])
    return found


def _min_known(values):
    known = [Decimal(str(value)) for value in values if oracle._number(value) is not None]
    return min(known) if known else None


def _assert_summaries_equal_oracle(root):
    with usage_store.read(root) as txn:
        rows = txn.attempts()
        stored = {(record["scope"], record["key"]) for record in txn.conn.execute(
            "SELECT scope, key FROM summaries WHERE rows > 0")}
        expected: dict = {}
        for row in rows:
            for address in _addresses(row):
                expected.setdefault(address, []).append(row)
        assert stored == set(expected), sorted(stored ^ set(expected))[:10]
        for (scope, key), addressed in expected.items():
            bucket = txn.bucket(scope, key)
            assert bucket.render_breakdown() == oracle._breakdown_bucket(addressed), (scope, key)
            assert bucket.render_summary() == oracle._summary(addressed), (scope, key)
            if scope == "root":
                assert bucket.min_cap() == _min_known(row.get("root_limit_usd") for row in addressed), key
            if scope == "group":
                assert bucket.min_cap() == _min_known(
                    row.get("billing_group_limit_usd", row.get("root_limit_usd")) for row in addressed), key


def _money(rng):
    choice = rng.random()
    if choice < .1:
        return 0
    if choice < .4:
        return rng.choice(["0.1", "1.25", "0.000001", "12345678901234567890.123456789"])
    return round(rng.random() * 3, rng.choice([2, 6, 9]))


def _base_row(rng, index):
    row = {"kind": "attempt", "attempt_id": f"a{index}", "task_id": rng.choice(["t1", "t2", "t3", ""]),
           "root_task_id": rng.choice(["r1", "r2", ""]), "model": rng.choice(["m1", "m2", ""]),
           "provider": rng.choice(["openai", "openrouter", ""]), "category": rng.choice(["task", "review", ""]),
           "source": "property", "reservation_upper_bound_usd": rng.choice([None, _money(rng)]),
           "pricing_known": rng.choice([True, False])}
    if rng.random() < .3:
        row["root_limit_usd"] = rng.choice([None, 5, "2.5", 100.0])
    if rng.random() < .3:
        row["billing_group_id"] = rng.choice(["g1", "r1"])
        row["billing_group_limit_usd"] = rng.choice([None, 7, "3.5"])
    if rng.random() < .2:
        row["review_skill"], row["review_wave_id"], row["review_slot_id"] = "skill", rng.choice(["w1", "w2"]), "s"
    if rng.random() < .2:
        row["prompt_cache_ttl"] = rng.choice(["5m", "1h"])
    return row


def _settled_fields(rng):
    fields = {"cost_usd": rng.choice([None, _money(rng)]), "cost_final": rng.choice([True, False])}
    for name in ("prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens"):
        if rng.random() < .6:
            fields[name] = rng.randrange(0, 5000)
    if rng.random() < .3:
        fields["processing"] = {"observed": rng.choice(["fast", "standard"])}
        fields["cost_evidence"] = {"valuationUsd": _money(rng), "valuationKnowledge": "exact",
                                   "cashUsd": _money(rng), "knowledge": rng.choice(["exact", "unknown"])}
    return fields


def _random_history(rng, attempts):
    """Interleaved legal transitions of ``attempts`` attempts plus one-shot rows.
    A dispatch may move the attempt to another group or root cap, as an owner
    amendment applied at dispatch does."""
    plans = {}
    for index in range(attempts):
        kind = rng.random()
        if kind < .08:
            plans[index] = [{"kind": "subscription_session", "attempt_id": f"s{index}", "state": "settled",
                             "task_id": rng.choice(["t1", "t2"]), "root_task_id": rng.choice(["r1", "r2"]),
                             "provider": "claudexor", "model": rng.choice(["fable", ""]), "category": "task",
                             "subscription_route": rng.choice(["route-a", "route-b"]),
                             "subscription_reset_at": rng.choice(["2026-09-01T00:00:00Z", "2026-10-01T00:00:00Z"]),
                             "cost_usd": rng.choice([None, _money(rng)]), "cost_final": rng.choice([True, False])}]
            continue
        if kind < .12:
            plans[index] = [{"kind": rng.choice(["legacy_usage", "legacy_delta", "legacy_metadata"]),
                             "attempt_id": f"l{index}", "state": "settled", "task_id": rng.choice(["t1", ""]),
                             "root_task_id": rng.choice(["r1", ""]), "provider": "legacy", "category": "legacy",
                             "cost_usd": _money(rng), "cost_final": False}]
            continue
        base = _base_row(rng, index)
        steps = [{**base, "state": "reserved"}]
        if rng.random() < .2:
            steps.append({**base, "state": "released", "reason": "not_dispatched"})
        else:
            dispatched = {**base, "state": "dispatched"}
            if rng.random() < .2:
                dispatched.update(billing_group_id=rng.choice(["g2", "r2"]), billing_group_limit_usd=rng.choice([4, None]))
            steps.append(dispatched)
            ending = rng.random()
            if ending < .55:
                steps.append({**dispatched, "state": "settled", **_settled_fields(rng)})
            elif ending < .65:
                steps.append({**dispatched, "state": "released", "reason": "before_dispatch_failed:not_started"})
            else:
                steps.append({**dispatched, "state": "unresolved", "reason": "response lost"})
                late = rng.random()
                if late < .3:
                    steps.append({**dispatched, "state": "settled", "settle_reason": "abandoned",
                                  "cost_usd": None, "cost_final": False})
                    if rng.random() < .5:
                        steps.append({**dispatched, "state": "settled", "settle_reason": "late_receipt",
                                      **_settled_fields(rng)})
                elif late < .6:
                    steps.append({**dispatched, "state": "settled", "settle_reason": "late_receipt",
                                  **_settled_fields(rng)})
        plans[index] = steps
    order = [index for index, steps in plans.items() for _ in steps]
    rng.shuffle(order)
    cursor = {index: 0 for index in plans}
    for index in order:  # each attempt's own steps stay in order; attempts interleave
        step = plans[index][cursor[index]]
        cursor[index] += 1
        yield step


@pytest.mark.parametrize("seed", range(12))
def test_every_summary_equals_the_frozen_fold_over_its_addressed_attempts(root, seed):
    rng = random.Random(seed)
    # Imported aggregates first: weight multiplies counts, never cash or tokens.
    write_compacted_journal(root, [
        {"attempt_id": f"fold-{index}", "task_id": rng.choice(["t1", "t2"]), "root_task_id": rng.choice(["r1", "r2"]),
         "model": "m1", "provider": "openai", "category": "task", "source": "old", "pricing_known": True,
         "folded_attempt_count": rng.randrange(1, 40), "cost_usd": str(_money(rng)), "cost_final": True,
         "reservation_upper_bound_usd": "1.5", "prompt_tokens": rng.randrange(0, 900),
         **({"root_limit_usd": "9"} if index % 2 else {})} for index in range(3)])
    for step in _random_history(rng, 70):
        with usage_store.hold(root) as txn:
            txn.write(step, txn.attempt(step["attempt_id"]))
    _assert_summaries_equal_oracle(root)
    usage_store.forget(root)  # a fresh process reads the same stored buckets
    _assert_summaries_equal_oracle(root)


def test_a_group_change_at_dispatch_moves_the_money_to_the_new_group(root):
    with ua.usage_scope(ua.UsageScope(drive_root=root, task_id="t", root_task_id="r", billing_group_id="old",
                                      billing_group_limit_usd=10.0)):
        held = ua.reserve_attempt(ua.AttemptRequest(model="m", provider="test", reservation_usd=.5))
    with usage_store.hold(root) as txn:
        current = txn.attempt(held.attempt_id)
        txn.write({**current, "state": "dispatched", "billing_group_id": "new", "billing_group_limit_usd": 3}, current)
    with usage_store.read(root) as txn:
        assert txn.bucket("group", "old").rows == 0 and txn.bucket("group", "new").rows == 1
        assert txn.summary(billing_group_id="new")["accounted_usd"] == .5
        assert txn.bucket("group", "new").min_cap() == 3
    _assert_summaries_equal_oracle(root)


def test_exact_money_beyond_the_default_decimal_precision(root):
    """10**28 + 1 is 29 digits: the ambient 28-digit context would lose the 1."""
    for cost in (1e28, 1.0):
        held = ua.reserve_attempt(request(root, reservation_usd=0.0, global_limit_usd=1e40))
        ua.mark_dispatched(held)
        ua.settle_attempt(held, {}, cost_usd=cost, cost_final=True)
    with usage_store.read(root) as txn:
        record = txn.conn.execute("SELECT settled FROM summaries WHERE scope='global'").fetchone()
    assert Decimal(record["settled"]) == Decimal("10000000000000000000000000001")


# ---- transitions -------------------------------------------------------------

def test_dispatch_rechecks_a_limit_reached_after_the_reservation(root):
    held = ua.reserve_attempt(request(root, reservation_usd=.4, global_limit_usd=1.0))
    # Known spend reaches the limit exactly between the reservation and the dispatch:
    # the reservation's own predicate (#1487, equality refuses) never sends.
    ua.record_subscription_session("late-spend", drive_root=root, route="subscription", task_id="other",
                                   root_task_id="other", spend_usd=1.0)
    with pytest.raises(ua.BudgetExceeded, match="changed before dispatch"):
        ua.mark_dispatched(held)
    [row] = [row for row in ledger_rows(root) if row["attempt_id"] == held.attempt_id]
    assert (row["state"], row["revision"]) == ("reserved", 1)
    ua.release_attempt(held, "before_dispatch_failed:limit")


def test_dispatch_never_counts_its_own_hold_as_spending(root):
    """Known $0.70 + this attempt's own $0.40 hold exceeds $1.00 only as exposure:
    known spend is below the limit, so the reserved call is sent (overshoot accepted)."""
    held = ua.reserve_attempt(request(root, reservation_usd=.4, global_limit_usd=1.0))
    ua.record_subscription_session("late-spend", drive_root=root, route="subscription", task_id="other",
                                   root_task_id="other", spend_usd=.7)
    ua.mark_dispatched(held)
    [row] = [row for row in ledger_rows(root) if row["attempt_id"] == held.attempt_id]
    assert row["state"] == "dispatched"


def test_the_owner_pause_fence_holds_at_dispatch(root):
    from ouroboros import owner_pause
    from ouroboros.llm_attempt import _PhysicalSendNotStarted
    from ouroboros.task_results import write_task_result

    write_task_result(root, "dominant", "running", root_task_id="dominant")
    held = ua.reserve_attempt(request(root))
    owner_pause.install_fence(root, "dominant", request_id="pause-before-dispatch")
    with pytest.raises(_PhysicalSendNotStarted):
        ua.mark_dispatched(held)
    [row] = [row for row in ledger_rows(root) if row["attempt_id"] == held.attempt_id]
    assert row["state"] == "reserved"


def test_a_recovery_that_raced_a_settlement_leaves_it_authoritative(root):
    held = ua.reserve_attempt(request(root))
    ua.mark_dispatched(held)
    observed = ledger_rows(root)[-1]["revision"]
    ua.settle_attempt(held, {}, cost_usd=.2, cost_final=True)
    assert ua.terminalize_abandoned_attempt(held, reason="owner ended", expected_revision=observed) == "settled"
    [row] = ledger_rows(root)
    assert (row["cost_usd"], row.get("settle_reason"), row["revision"]) == (.2, None, 3)


def test_one_shots_compare_identity_not_payload(root):
    first = ua.record_subscription_session("session", drive_root=root, route="r", model="first",
                                           task_id="t", root_task_id="r", spend_usd=1.0)
    before = ledger_rows(root)
    assert ua.record_subscription_session("session", drive_root=root, route="r", model="second",
                                          task_id="t", root_task_id="r", spend_usd=9.0) == first
    assert ledger_rows(root) == before
    with pytest.raises(ua.UsageAccountingError, match="conflicting settled-row identity"):
        ua.record_subscription_session("session", drive_root=root, route="other", task_id="t", root_task_id="r")


# ---- the one-time import -----------------------------------------------------

def _journal_rows():
    """A journal as an upgraded install carries it: settled, open, abandoned
    and one-shot attempts, a root cap and a review wave."""
    row = dict(kind="attempt", task_id="t", root_task_id="r", provider="openai", model="m", category="task",
               source="journal", reservation_upper_bound_usd="0.5", pricing_known=True, root_limit_usd=20.0)
    rows = [{**row, "attempt_id": "settled", "state": state} for state in ("reserved", "dispatched")]
    rows.append({**row, "attempt_id": "settled", "state": "settled", "cost_usd": "0.25", "cost_final": True,
                 "prompt_tokens": 10, "completion_tokens": 2})
    rows.append({**row, "attempt_id": "open", "state": "reserved"})
    rows += [{**row, "attempt_id": "lost", "state": state} for state in ("reserved", "dispatched")]
    rows.append({**row, "attempt_id": "lost", "state": "unresolved", "reason": "lost"})
    rows.append({**row, "attempt_id": "lost", "state": "settled", "settle_reason": "abandoned",
                 "cost_usd": None, "cost_final": False})
    rows.append({**row, "attempt_id": "wave", "state": "reserved", "review_skill": "s", "review_wave_id": "w"})
    rows.append({"kind": "subscription_session", "attempt_id": "sess", "state": "settled", "task_id": "t",
                 "root_task_id": "r", "provider": "claudexor", "subscription_route": "route", "cost_usd": 0.5,
                 "cost_final": True})
    return rows


def _store_snapshot(root):
    with usage_store.read(root) as txn:
        summaries = [tuple(record) for record in txn.conn.execute("SELECT * FROM summaries ORDER BY scope, key")]
        bindings = [tuple(record) for record in txn.conn.execute("SELECT * FROM bindings ORDER BY scope, key")]
        attempts = {row["attempt_id"]: {key: value for key, value in row.items() if key not in {"revision", "seq"}}
                    for row in txn.attempts()}
        return summaries, bindings, attempts, txn.marker()


def _supervisor_rows(root):
    path = root / "logs" / "supervisor.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if json.loads(line).get("type") == "usage_store_migration"] if path.exists() else []


def test_the_import_is_idempotent_and_reports_its_timing_and_counts(root):
    journal = write_journal(root, _journal_rows())
    source = journal.read_bytes()
    report = usage_store.migrate_from_journal(root)
    assert report["status"] == "completed" and report["attempts"] == 5 and report["journal_rows"] == 10
    assert journal.read_bytes() == source  # kept in place: an older release still reads it after a rollback
    [row] = _supervisor_rows(root)
    assert row["phase"] == "completed" and row["duration_seconds"] >= 0 and row["attempts"] == 5
    with usage_store.read(root) as txn:
        provenance = txn.meta("import")
        assert provenance["status"] == "completed" and provenance["source"]["rows"] == 10
        assert provenance["source"]["size"] == len(source)
        assert txn.meta("lock_tier") == usage_store.TIER_ENFORCED
        assert txn.marker() == [0, 10]
        assert [row["attempt_id"] for row in txn.open_attempts()] == ["open", "lost", "wave"]
        assert ledger.late_receipt_eligible(txn.attempt("lost"))
    before = (root / usage_store.STORE_REL).read_bytes(), _store_snapshot(root)
    assert usage_store.migrate_from_journal(root)["status"] == "already_completed"
    assert ((root / usage_store.STORE_REL).read_bytes(), _store_snapshot(root)) == before
    assert [row["phase"] for row in _supervisor_rows(root)] == ["completed", "already_completed"]


def test_a_display_read_never_imports_a_waiting_journal(root):
    """A display read (the providerless /api/state, the loop's status) while the journal awaits the lifecycle
    import reports the store unavailable and leaves the journal alone; the lifecycle job imports it."""
    journal = write_journal(root, _journal_rows())
    source = journal.read_bytes()
    with pytest.raises(ledger.UsageLockUnavailable):
        ua.usage_writer_snapshot(root, allow_stale=True)
    assert journal.read_bytes() == source and not (root / usage_store.STORE_REL).exists()
    assert usage_store.migrate_from_journal(root)["status"] == "completed"  # the lifecycle job
    assert ua.usage_writer_snapshot(root, allow_stale=True)  # now a plain addressed read


def test_a_display_read_never_runs_the_pre_ledger_import(root):
    """No journal, but a pre-ledger event chain never imported: a display still reports the store
    unavailable instead of reading that history; the lifecycle job imports it."""
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "logs" / "events.jsonl").write_text(json.dumps({
        "type": "llm_usage", "task_id": "legacy", "root_task_id": "legacy", "cost": 0.125,
        "provider": "openai", "prompt_tokens": 7}) + "\n", encoding="utf-8")
    with pytest.raises(ledger.UsageLockUnavailable):
        ua.usage_projection(root, allow_stale=True)
    assert not (root / usage_store.STORE_REL).exists()
    assert usage_store.migrate_from_journal(root)["status"] == "completed"
    assert ua.usage_projection(root, allow_stale=True)["accounted_usd"] == 0.125


def test_a_display_read_on_a_fresh_install_creates_the_empty_store(root):
    """No journal means nothing to import: the display read is served (zero spend), never unavailable."""
    assert not (root / ledger.LEDGER_REL).exists()
    with usage_store.read(root, allow_stale=True) as txn:
        assert txn.marker() is not None
    assert (root / usage_store.STORE_REL).exists()


def test_the_server_imports_the_journal_at_lifespan_start_on_every_door():
    """The import runs where the other lifecycle jobs run (before any request, worker or supervisor), so a
    providerless install that never starts the supervisor imports too."""
    import inspect

    import server

    source = inspect.getsource(server.lifespan)
    assert source.index("prepare_startup_state(") < source.index("usage_store.migrate_from_journal(") \
        < source.index("yield")


def test_an_import_interrupted_before_publication_is_redone_from_scratch(root, monkeypatch):
    journal = write_journal(root, _journal_rows())
    stale = root / "state" / "usage.sqlite.import-99999-deadbeef"
    stale.write_bytes(b"a build a crashed process left behind")
    build = usage_store._build

    def crash(tmp, *args, **kwargs):
        build(tmp, *args, **kwargs)  # the whole build, then the process dies before the rename
        raise KeyboardInterrupt("process died")

    monkeypatch.setattr(usage_store, "_build", crash)
    with pytest.raises(KeyboardInterrupt):
        usage_store.migrate_from_journal(root)
    assert not (root / usage_store.STORE_REL).exists() and journal.exists()
    assert not stale.exists(), "an unpublished build is discarded"
    monkeypatch.setattr(usage_store, "_build", build)
    assert usage_store.migrate_from_journal(root)["status"] == "completed"
    assert {row["attempt_id"] for row in ledger_rows(root)} == {"settled", "open", "lost", "wave", "sess"}


def test_a_kept_journal_is_never_reimported_and_an_older_release_append_is_disclosed(root):
    """The import leaves the journal in place (an older release reads it after a rollback); a
    later start never re-imports it, and rows an older release appended are disclosed, not merged."""
    journal = write_journal(root, _journal_rows())
    source = journal.read_bytes()
    assert usage_store.migrate_from_journal(root)["status"] == "completed"
    held = ua.reserve_attempt(request(root))  # the store serves money; the journal is not written
    report = usage_store.migrate_from_journal(root)
    assert report["status"] == "already_completed" and report["journal"] == "kept"
    assert journal.read_bytes() == source
    with journal.open("ab") as handle:  # a rollback: the older release appended one row
        handle.write((json.dumps({"seq": 99, "attempt_id": "older", "state": "reserved"}) + "\n").encode())
    report = usage_store.migrate_from_journal(root)
    assert report["journal"] == "changed_after_import" and report["journal_size"] > len(source)
    assert sorted(row["attempt_id"] for row in ledger_rows(root)) == sorted(
        ["settled", "open", "lost", "wave", "sess", held.attempt_id])


def test_a_pre_ledger_install_imports_its_telemetry_once(root):
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "logs" / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in (
        {"type": "llm_usage", "task_id": "old", "model": "m", "provider": "openai", "cost": 0.5,
         "prompt_tokens": 3, "completion_tokens": 1},
        {"type": "llm_usage", "task_id": "old", "model": "m", "provider": "openai", "cost": 0.25})), encoding="utf-8")
    (root / "state" / "state.json").write_text(json.dumps({"spent_usd": 1.0, "spent_calls": 3}), encoding="utf-8")
    report = usage_store.migrate_from_journal(root)
    assert report["status"] == "completed"
    kinds = sorted(row["kind"] for row in ledger_rows(root))
    assert kinds == ["legacy_delta", "legacy_metadata", "legacy_usage", "legacy_usage"]
    assert ua.usage_projection(root)["accounted_usd"] == pytest.approx(1.0)  # 0.75 events + 0.25 state delta
    from ouroboros.usage_journal import IMPORT_REL

    assert json.loads((root / IMPORT_REL).read_text(encoding="utf-8"))["completed"] is True
    with usage_store.read(root) as txn:
        assert txn.meta("import")["legacy"]["legacy_usage_count"] == 2
    usage_store.forget(root)
    assert usage_store.migrate_from_journal(root)["status"] == "already_completed"
    assert len(ledger_rows(root)) == 4


def test_a_published_store_that_cannot_be_read_is_refused_never_replaced(root):
    journal = write_journal(root, _journal_rows())
    damaged = b"not a database\n" * 64
    (root / usage_store.STORE_REL).write_bytes(damaged)
    with pytest.raises(ledger.UsageLedgerCorrupt):
        usage_store.migrate_from_journal(root)
    with pytest.raises(ledger.UsageLedgerCorrupt):
        ua.usage_projection(root)
    assert (root / usage_store.STORE_REL).read_bytes() == damaged and journal.exists()


def test_the_marker_continues_the_journal_and_never_goes_lower(root):
    write_compacted_journal(root, [{"attempt_id": "fold", "task_id": "t", "root_task_id": "r", "model": "m",
                                    "provider": "p", "category": "task", "source": "old", "folded_attempt_count": 4,
                                    "cost_usd": "1", "cost_final": True}], epoch=3)
    with usage_store.read(root) as txn:
        assert txn.marker() == [3, 2]
    held = ua.reserve_attempt(request(root))
    with usage_store.read(root) as txn:
        assert txn.marker() == [3, 3]
    ua.release_attempt(held)
    assert ua.usage_breakdown(root)["_ledger_high_water_seq"] == [3, 4]


# ---- the downgrade export ----------------------------------------------------

def _varied_store(root):
    """Imported aggregates and journal rows, then live writes of every shape."""
    write_compacted_journal(root, [
        {"attempt_id": "fold-a", "task_id": "t", "root_task_id": "r", "model": "m", "provider": "openai",
         "category": "task", "source": "old", "folded_attempt_count": 7, "cost_usd": "1.75", "cost_final": True,
         "root_limit_usd": "20", "reservation_upper_bound_usd": "2"}], _journal_rows())
    settled = ua.reserve_attempt(request(root, root_task_id="r", root_limit_usd=20.0))
    ua.mark_dispatched(settled)
    ua.settle_attempt(settled, {"prompt_tokens": 5}, cost_usd=0.125, cost_final=True)
    abandoned = ua.reserve_attempt(request(root, root_task_id="r2", reservation_usd=0.75))
    ua.mark_dispatched(abandoned)
    ua.mark_unresolved(abandoned, "lost")
    ua.terminalize_abandoned_attempt(abandoned, reason="owner ended")
    released = ua.reserve_attempt(request(root))
    ua.release_attempt(released)
    dispatched = ua.reserve_attempt(request(root, task_id="open-d"))
    ua.mark_dispatched(dispatched)
    ua.record_unmetered_external_dispatch("ext", drive_root=root, model="x", task_id="t", prompt_tokens=1)
    return abandoned


def test_the_export_passes_the_journal_era_validator_and_reimports_identically(root):
    from tests import _legacy_usage_validator as legacy_validator

    abandoned = _varied_store(root)
    state = root / "state" / "state.json"
    state.write_text(json.dumps({"spent_usd": 1, "usage_ledger_high_water_seq": [0, 30]}), encoding="utf-8")
    summaries, bindings, attempts, _marker = _store_snapshot(root)
    report = usage_store.export_journal(root)
    assert report["status"] == "exported" and not (root / usage_store.STORE_REL).exists()
    journal = (root / ledger.LEDGER_REL).read_text(encoding="utf-8")
    rows = [json.loads(line, parse_float=legacy_validator.LiteralFloat) for line in journal.splitlines()]
    legacy_validator._validate_records(rows)  # the journal-era validator, frozen at 84febbdd3
    assert json.loads(state.read_text(encoding="utf-8"))["usage_ledger_high_water_seq"] == report["marker"]
    with pytest.raises(ua.UsageAccountingError, match="refusing to overwrite"):
        usage_store.export_journal(root)
    # The next access (an upgrade again) imports the exported journal: the same money.
    again = _store_snapshot(root)
    assert again[0] == summaries and again[1] == bindings
    for attempt_id, row in attempts.items():
        assert again[2][attempt_id] == row, (attempt_id, {key: (row.get(key), again[2][attempt_id].get(key))
                                                          for key in set(row) | set(again[2][attempt_id])
                                                          if row.get(key) != again[2][attempt_id].get(key)})
    assert set(again[2]) == set(attempts)
    ua.settle_attempt(abandoned, {}, cost_usd=0.5, cost_final=True)  # the late-receipt right survived
    assert ua.usage_projection(root, root_task_id="r2")["confirmed_usd"] == 0.5


# ---- lock tiers --------------------------------------------------------------

def _dead_pid():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


@pytest.fixture
def name_tier(root, monkeypatch):
    """An installation whose data root takes no kernel file locks (the probe's
    name tier: Drive, FUSE, NFS); the import records the tier once."""
    from ouroboros import platform_layer

    with monkeypatch.context() as probe:
        probe.setattr(platform_layer, "kernel_file_locks_enforced", lambda _path: False)
        usage_store.migrate_from_journal(root)
    usage_store.forget(root)  # every later process trusts the recorded tier, never a new probe
    return root


def test_the_name_tier_serializes_every_access_by_the_name_lock(name_tier, monkeypatch):
    root = name_tier
    assert usage_store._header_tier(root / usage_store.STORE_REL) == usage_store.TIER_NAME
    with usage_store.read(root) as txn:
        assert txn.meta("lock_tier") == usage_store.TIER_NAME
    taken = []
    named = usage_store._named_lock

    def counted(lock_root, name, **kwargs):
        taken.append(name)
        return named(lock_root, name, **kwargs)

    monkeypatch.setattr(usage_store, "_named_lock", counted)
    held = ua.reserve_attempt(request(root))
    ua.mark_dispatched(held)
    ua.settle_attempt(held, {}, cost_usd=.5, cost_final=True)
    writes = len(taken)
    assert writes >= 3 and set(taken) == {ledger.LOCK_REL.name}
    assert ua.usage_projection(root)["confirmed_usd"] == .5
    assert len(taken) > writes, "reads run under the name lock too"
    usage_store.forget(root)  # reopen in a fresh process: the recorded tier, the same money
    assert ua.usage_projection(root)["confirmed_usd"] == .5
    assert not (root / ledger.LOCK_REL).exists()


def test_a_crashed_name_tier_writer_is_reclaimed_through_the_stale_owner_path(name_tier):
    root = name_tier
    lock = root / ledger.LOCK_REL
    lock.write_text(f"pid={_dead_pid()} ts=0\n", encoding="utf-8")  # a writer that died holding it
    held = ua.reserve_attempt(request(root))
    ua.release_attempt(held)
    assert [row["state"] for row in ledger_rows(root)] == ["released"]
    assert not lock.exists()


_HOLDER = """
import sqlite3, sys, time
conn = sqlite3.connect(sys.argv[1], isolation_level=None)
conn.execute("BEGIN IMMEDIATE")
print("held", flush=True)
sys.stdin.readline()
conn.execute("ROLLBACK")
"""


@pytest.mark.serial
def test_enforced_tier_contention_across_processes_honours_stop(root):
    import threading

    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    from ouroboros.model_wait import task_model_wait_scope
    from ouroboros.task_results import write_task_result

    for task_id in ("dominant", "child"):
        write_task_result(root, task_id, "running", root_task_id="dominant")
    held = ua.reserve_attempt(request(root))
    holder = subprocess.Popen([sys.executable, "-c", _HOLDER, str(root / usage_store.STORE_REL)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        stopped = threading.Event()
        timer = threading.Timer(.3, stopped.set)
        timer.start()
        with task_model_wait_scope(task={"id": "child"}, drive_root=root, event_queue=None,
                                   worker_slot_held=False,
                                   owner_control=lambda: "cancelled" if stopped.is_set() else None):
            with pytest.raises(PhysicalDispatchInterrupted) as error:
                ua.mark_dispatched(held)
        timer.join()
        assert error.value.control_reason == "cancelled"
    finally:
        holder.stdin.write("\n")
        holder.stdin.flush()
        holder.wait(10)
    assert [row["state"] for row in ledger_rows(root)] == ["reserved"]
    ua.mark_dispatched(held)  # the other process let go: the same attempt proceeds
    assert [row["state"] for row in ledger_rows(root)] == ["dispatched"]


# ---- no ordinary path reads every attempt -------------------------------------

_BULK_ALLOWED = {"ouroboros/usage_store.py", "ouroboros/usage_accounting.py", "ouroboros/model_send_seal.py"}


def _production_sources():
    for directory in ("ouroboros", "supervisor", "scripts"):
        yield from sorted((REPO / directory).rglob("*.py"))
    yield REPO / "server.py"


def test_bulk_readers_are_called_only_by_the_audit_and_the_export():
    callers = set()
    for path in _production_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            # ``Txn.attempts()`` without a predicate is the same scan of every attempt.
            if name == "read_usage_records" or (name == "attempts" and not node.args and not node.keywords
                                                 and isinstance(node.func, ast.Attribute)):
                callers.add(path.relative_to(REPO).as_posix())
    assert callers <= _BULK_ALLOWED, callers - _BULK_ALLOWED
    assert "ouroboros/model_send_seal.py" in callers  # the explicit audit still reads history


def test_ordinary_money_paths_never_read_every_attempt(root, monkeypatch):
    from ouroboros import consciousness_allowance, server_maintenance
    from supervisor import queue

    held = ua.reserve_attempt(request(root))
    ua.release_attempt(held)
    scan = usage_store.Txn.attempts

    def forbidden_scan(txn, where="1", params=()):
        assert str(where).strip() != "1", "an ordinary path scanned every attempt"
        return scan(txn, where, params)

    monkeypatch.setattr(usage_store, "read_usage_records", lambda *_a, **_k: pytest.fail("bulk read"))
    monkeypatch.setattr(usage_store.Txn, "attempts", forbidden_scan)
    monkeypatch.setattr(queue, "task_has_live_ownership", lambda _task_id, **_kw: False)
    sent = ua.execute_physical_attempt(request(root, reservation_usd=.1), lambda: {"usage": {}},
                                       extractor=lambda _r: ({}, .1, True))
    assert sent == {"usage": {}}
    ua.usage_projection(root, global_limit_usd=10.0)
    ua.usage_projection(root, root_task_id="dominant")
    ua.usage_breakdown(root)
    ua.usage_breakdown(root, task_id="child")
    ua.usage_breakdown(root, root_task_id="dominant")
    ua.usage_writer_snapshot(root)
    consciousness_allowance.allowance_window(root)
    server_maintenance._reconcile_abandoned_usage(root)


# ---- history independence ----------------------------------------------------

def _unrelated_history(root, attempts):
    """``attempts`` settled attempts of other roots, imported as an upgraded
    install carries them (the import is the one full fold, outside the
    measurement)."""
    rows = []
    for index in range(attempts):
        base = dict(kind="attempt", attempt_id=f"old-{index}", task_id=f"old-task-{index % 400}",
                    root_task_id=f"old-root-{index % 100}", model=f"m{index % 4}", provider="openai",
                    category="task", source="history", reservation_upper_bound_usd="0.01", pricing_known=True,
                    ts="2026-01-01T00:00:00+00:00")
        rows += [{**base, "state": "reserved"}, {**base, "state": "dispatched"},
                 {**base, "state": "settled", "cost_usd": "0.001", "cost_final": True, "prompt_tokens": 3}]
    write_journal(root, rows)
    usage_store.migrate_from_journal(root)
    # Finish the import's projection obligations before measuring ordinary
    # work. Missing results are real debt, not unrelated completed history.
    from ouroboros.task_results import write_task_result
    from supervisor.events_task_done import _refresh_terminal_task_cost
    with usage_store.read(root) as txn:
        owners = txn.dirty_owners()
    for owner, revision in owners:
        write_task_result(root, owner, "completed", result="Historical result")
        assert _refresh_terminal_task_cost(root, owner)
        with usage_store.hold(root) as txn:
            assert txn.ack_dirty_owner(owner, revision)
    usage_store.forget(root)


class _Steps:
    """SQLite VM instructions executed by every store connection (in units of 50)."""

    def __init__(self, monkeypatch):
        self.count = 0
        connect = usage_store._connect

        def counted(*args, **kwargs):
            conn = connect(*args, **kwargs)
            conn.set_progress_handler(self._tick, 50)
            return conn

        monkeypatch.setattr(usage_store, "_connect", counted)

    def _tick(self):
        self.count += 1
        return 0

    def measure(self, call):
        self.count = 0
        call()
        return self.count


def _api_state(root, monkeypatch):
    import asyncio
    import types

    from starlette.requests import Request

    from ouroboros.gateway.state import api_state
    from supervisor import queue, state, workers

    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 100.0)
    monkeypatch.setattr(state, "DRIVE_ROOT", str(root))
    monkeypatch.setattr(state, "load_state", lambda: {"current_branch": "ouroboros"})
    for module in (workers, queue):
        monkeypatch.setattr(module, "PENDING", [])
        monkeypatch.setattr(module, "RUNNING", {})
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(queue, "load_state", lambda: {"current_branch": "ouroboros"})
    monkeypatch.setattr(queue, "_read_evolution_campaign", lambda: {})
    request = Request({"type": "http", "method": "GET", "path": "/api/state", "headers": [], "query_string": b"",
                       "scheme": "http", "server": ("test", 80), "client": ("test", 1),
                       "app": types.SimpleNamespace(state=types.SimpleNamespace(drive_root=root, app_start=0.0))})
    return lambda: asyncio.run(api_state(request))


def _ordinary_reads(root, monkeypatch):
    """The ordinary readers of the active root, each measured on its own."""
    import asyncio
    import types

    from ouroboros.context_runtime_facts import _runtime_budget_info
    from ouroboros.gateway.cost_breakdown import make_cost_breakdown_endpoint

    def admission():
        ua.execute_physical_attempt(request(root, root_task_id="active", task_id="active-task", reservation_usd=.01),
                                    lambda: {"usage": {}}, extractor=lambda _r: ({}, .001, True))

    def candidates():
        with usage_store.read(root) as txn:
            txn.open_attempts()
            txn.dirty_owner_ids()

    cost_breakdown = make_cost_breakdown_endpoint(root)
    return {
        "admission": admission,
        "usage_projection(root)": lambda: ua.usage_projection(root, root_task_id="active"),
        "usage_projection(global)": lambda: ua.usage_projection(root, global_limit_usd=100.0),
        "usage_breakdown(task)": lambda: ua.usage_breakdown(root, task_id="active-task"),
        "usage_breakdown(root)": lambda: ua.usage_breakdown(root, root_task_id="active"),
        "/api/state": _api_state(root, monkeypatch),
        "/api/cost-breakdown": lambda: asyncio.run(cost_breakdown(None)),
        "task start facts": lambda: _runtime_budget_info(types.SimpleNamespace(drive_root=root),
                                                         {"budget_drive_root": str(root)}),
        "duty candidates": candidates,
    }


@pytest.mark.serial
def test_ordinary_readers_do_not_grow_with_unrelated_history(tmp_path, monkeypatch):
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    measured = {}
    for size in (100, 10_000):
        root = tmp_path / f"history-{size}"
        (root / "state").mkdir(parents=True)
        monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
        monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
        _unrelated_history(root, size)
        ua.execute_physical_attempt(request(root, root_task_id="active", task_id="active-task", reservation_usd=.01),
                                    lambda: {"usage": {}}, extractor=lambda _r: ({}, .001, True))
        with monkeypatch.context() as patch:
            steps = _Steps(patch)
            readers = _ordinary_reads(root, patch)
            for name, call in readers.items():
                call()  # warm the process (imports, first open)
            measured[size] = {name: steps.measure(call) for name, call in readers.items()}
        usage_store.forget(root)
    (tmp_path / "history_steps.json").write_text(json.dumps(measured, indent=1), encoding="utf-8")  # evidence
    for name, small in measured[100].items():
        large = measured[10_000][name]
        # Index seeks are one instruction at any depth: the count is the addressed
        # scope's, not the history's (a 100x larger store, a few % more steps).
        assert large <= small * 1.25 + 4, (name, small, large)


def test_a_fresh_process_answers_without_importing(root):
    held = ua.reserve_attempt(request(root, root_task_id="active", task_id="active-task"))
    ua.mark_dispatched(held)
    ua.settle_attempt(held, {}, cost_usd=.5, cost_final=True)
    expected = ua.usage_projection(root, root_task_id="active")["accounted_usd"]
    script = (
        "import json, sys\n"
        "from ouroboros import usage_accounting as ua, usage_store\n"
        "usage_store.migrate_from_journal = lambda root: (_ for _ in ()).throw(AssertionError('imported'))\n"
        "print(json.dumps(ua.usage_projection(sys.argv[1], root_task_id='active')['accounted_usd']))\n")
    result = subprocess.run([sys.executable, "-c", script, str(root)], capture_output=True, text=True,
                            cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)}, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(result.stdout.strip().splitlines()[-1]) == expected == .5
