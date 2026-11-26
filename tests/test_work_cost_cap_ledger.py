"""Work-cap ledger decisions through ordinary synthetic admission, export and recovery."""
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ouroboros import usage_accounting as ua
from ouroboros import usage_journal, usage_store
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.usage_admission import group_money_spent, task_billing_fields, task_money_snapshot, work_cap
from ouroboros.usage_ledger import LEDGER_REL, UsageAccountingError, UsageLedgerCorrupt
from tests._usage_store_testing import ledger_rows, write_compacted_journal
from tests.test_work_cost_cap import (  # noqa: F401
    BY,
    MIND,
    OWNER,
    _binding,
    _raise,
    _scope,
    _spend,
    _work,
    data_root,
    register,
    world,
)


def test_frozen_admission_can_be_amended_and_reopened_before_any_attempt(data_root):  # noqa: F811
    task = _work(data_root, "quiet-work")
    initial = load_task_result(data_root, task["id"])["billing_group"]
    grant = _raise(data_root, task["id"], 16.0)
    assert grant["ok"] and grant["previous_limit_usd"] == 10.0 and grant["remaining_known_usd"] == 16.0
    with usage_store.read(data_root) as view:
        cap, marker = work_cap(view, task["id"]), view.marker()
        assert view.binding("group", task["id"]) is None and view.attempts() == []
    report = usage_store.export_journal(data_root)
    assert report["attempts"] == report["journal"]["rows"] == 0
    assert (data_root / LEDGER_REL).read_bytes() == b""
    assert (data_root / report["work_caps_watermark"]).is_file()
    with usage_store.read(data_root) as view:
        assert view.attempts() == [] and view.marker() == marker
        assert work_cap(view, task["id"]) == cap
    assert ua.usage_projection(data_root, billing_group_id=task["id"])["limit_usd"] == 16.0
    assert load_task_result(data_root, task["id"])["billing_group"] == initial
    _spend(_scope(data_root, task["id"], task["id"], task["id"]), 11.0)
    assert task_money_snapshot(data_root, task, task["id"], root_limit=10.0)["remaining_known_usd"] == 5.0


def test_normal_host_admission_freezes_the_group_in_its_receipt_before_the_worker_starts(world, monkeypatch):  # noqa: F811
    from supervisor.task_admission import enqueue_with_admission_receipt

    _ctx, root, _pending = world
    task_id = "quiet-pooled"
    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", "10")
    admitted = enqueue_with_admission_receipt({"id": task_id, "root_task_id": task_id, "type": "task", "text": "test"})
    assert not admitted.get("_admission_blocked")
    before = load_task_result(root, task_id)
    assert before["host_admission"]["status"] == "accepted" and "billing_group" not in before
    assert before["metadata"]["billing_group"]["billing_group_limit_usd"] == 10.0
    # A change to defaults cannot reprice already admitted work before its first call.
    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", "50")
    assert _raise(root, task_id, 14.0)["previous_limit_usd"] == 10.0
    assert load_task_result(root, task_id) == before and ledger_rows(root) == []


def test_export_resumes_after_a_watermark_write_refusal(data_root, monkeypatch):  # noqa: F811
    task = _work(data_root, "quiet-work")
    assert _raise(data_root, task["id"], 13.0)["ok"]

    def not_written(*_args, **_kw):
        raise OSError("watermark write refused")

    with monkeypatch.context() as mp:
        mp.setattr(usage_journal, "write_exported_work_caps", not_written)
        with pytest.raises(OSError, match="watermark write refused"):
            usage_store.export_journal(data_root)
    assert (data_root / usage_store.STORE_REL).is_file()
    with usage_store.read(data_root) as view:
        assert work_cap(view, task["id"])["limit_usd"] == 13.0 and view.attempts() == []
    assert usage_store.export_journal(data_root, resume=True)["status"] == "exported"
    with usage_store.read(data_root) as view:
        assert work_cap(view, task["id"])["limit_usd"] == 13.0 and view.attempts() == []


def test_new_same_amount_confirmation_is_ordered_and_identical_retry_leaves_no_write(data_root):  # noqa: F811
    _work(data_root, "work-a")
    newer = {**OWNER, "source": {"ts": "2026-01-01T03:00:00Z"}}
    assert _raise(data_root, "work-a", 16.0)["ok"]
    assert _raise(data_root, "work-a", 16.0, newer)["confirmed"]
    with usage_store.read(data_root) as view:
        saved = work_cap(view, "work-a")
    assert saved["basis"] == newer
    unchanged = _raise(data_root, "work-a", 16.0, newer)
    assert unchanged["unchanged"] and unchanged["basis"] == "owner_words" and unchanged["revision"] == saved["revision"]
    assert _raise(data_root, "work-a", 18.0, MIND)["error"] == "owner_cap_binds"
    assert _raise(data_root, "work-a", 18.0, {**OWNER, "source": {"ts": "2026-01-01T02:00:00Z"}})[
        "error"] == "owner_cap_needs_newer_decision"
    with usage_store.read(data_root) as view:
        assert work_cap(view, "work-a") == saved and view.attempts() == []


def test_finite_owner_delegation_preserves_source_order_and_current_attribution(data_root):  # noqa: F811
    from ouroboros.usage_admission import set_work_cap

    task = _work(data_root, "bounded")
    owner = {**OWNER, "source": {"ts": "2026-01-02T00:00:00Z", "message_id": "synthetic"}, "delegated_max_usd": 19}
    assert _raise(data_root, "bounded", 15, owner)["ok"]
    for bad in (True, 14, 15, float("inf"), 10**400):
        assert _raise(data_root, "bounded", 15, {**owner, "delegated_max_usd": bad})["error"] == "cost_cap_basis_unverified"
    assert _raise(data_root, "bounded", 15, {**MIND, "delegated_max_usd": 200})["error"] == "cost_cap_basis_unverified"
    by = {**BY, "text": "I am using the agreed allowance"}
    grant = set_work_cap(data_root, task, "bounded", limit_usd=18, by=by, basis=MIND)
    assert grant["ok"] and grant["basis"] == "own_judgement"
    noop = _raise(data_root, "bounded", 18, MIND)
    assert noop["unchanged"] and noop["basis"] == "own_judgement" and noop["revision"] == grant["revision"]
    stale = _raise(data_root, "bounded", 20, OWNER)
    assert stale["error"] == "owner_cap_needs_newer_decision"
    assert _raise(data_root, "bounded", 20, MIND)["error"] == "owner_cap_binds"
    with usage_store.read(data_root) as view:
        before = work_cap(view, "bounded")
    assert before["owner_decision"] == {"limit_usd": 15.0, "basis": owner} and before["by"] == by
    usage_store.export_journal(data_root)
    with usage_store.read(data_root) as view:
        assert work_cap(view, "bounded") == before and view.attempts() == []
    assert _raise(data_root, "bounded", 19, MIND)["ok"]
    now = {**OWNER, "source": {"ts": "2026-01-03T00:00:00Z"}}
    assert _raise(data_root, "bounded", 19, now)["confirmed"]
    assert _raise(data_root, "bounded", 20, MIND)["error"] == "owner_cap_binds"
    snap = task_money_snapshot(data_root, task, "bounded", root_limit=10.0)
    axis = snap["group_axis"]
    assert axis["basis_kind"] == "owner_words" and axis["original_limit_usd"] == 10.0
    assert not ({"basis", "owner_decision", "by"} & axis.keys())


@pytest.mark.parametrize("fault", ["missing", "unreadable", "invalid"])
def test_existing_export_authority_is_not_reimported_from_legacy_spend(data_root, monkeypatch, fault):  # noqa: F811
    work = _work(data_root, "work-a")
    scope = _scope(data_root, "work-a", "work-a", "work-a")
    _spend(scope, 1.0)
    assert _raise(data_root, "work-a", 12.0)["ok"]
    usage_store.export_journal(data_root)
    watermark, ledger = (data_root / rel for rel in (usage_journal.IMPORT_REL, LEDGER_REL))
    saved_stamp, saved_ledger = watermark.read_bytes(), ledger.read_bytes()
    if fault == "missing":
        watermark.unlink()
    elif fault == "invalid":
        watermark.write_bytes(b"{")
    read = type(watermark).read_text

    def denied(path, *args, **kwargs):
        if path == watermark:
            raise PermissionError("read refused")
        return read(path, *args, **kwargs)

    with monkeypatch.context() as mp:
        if fault == "unreadable":
            mp.setattr(type(watermark), "read_text", denied)
        with pytest.raises(UsageAccountingError, match="watermark"):
            usage_store.migrate_from_journal(data_root)
    assert not (data_root / usage_store.STORE_REL).exists() and ledger.read_bytes() == saved_ledger
    watermark.write_bytes(saved_stamp)
    snapshot = task_money_snapshot(data_root, work, "work-a", root_limit=10.0)
    assert snapshot["remaining_known_usd"] == 11.0 and snapshot["group_axis"]["basis_kind"] == "owner_words"


def test_export_checks_for_completion_before_it_freezes_the_store(data_root):  # noqa: F811
    watermark = data_root / usage_journal.IMPORT_REL
    watermark.write_text(json.dumps({"completed": True, "source": "legacy"}), encoding="utf-8")
    _work(data_root, "work-a")
    _spend(_scope(data_root, "work-a", "work-a", "work-a"), 1.0)
    assert _raise(data_root, "work-a", 12.0)["ok"]
    with usage_store.hold(data_root) as txn:
        # An existing SQLite-format import recorded completion but not a portable copy of its watermark.
        txn.set_meta("import", {**txn.meta("import"), "legacy": None})
    stamp = watermark.read_bytes()
    watermark.unlink()
    with pytest.raises(UsageAccountingError, match="completed legacy import watermark"):
        usage_store.export_journal(data_root)
    with usage_store.read(data_root, allow_stale=True) as view:
        assert view.meta("export") is None and work_cap(view, "work-a")["limit_usd"] == 12.0
    _spend(_scope(data_root, "work-a", "work-a", "work-a"), 1.0)
    watermark.write_bytes(stamp)
    assert usage_store.export_journal(data_root)["status"] == "exported"


def test_missing_or_unverified_owner_source_cannot_widen_a_restored_decision(data_root):  # noqa: F811
    _work(data_root, "work-a")
    assert _raise(data_root, "work-a", 12, {**OWNER, "delegated_max_usd": 14})["ok"]
    assert _raise(data_root, "work-a", 13, MIND)["ok"]
    usage_store.export_journal(data_root)
    watermark = data_root / usage_journal.IMPORT_REL
    saved = watermark.read_bytes()
    edited = json.loads(saved)
    del edited[usage_journal.EXPORTED_WORK_CAPS]["caps"]["work-a"]["owner_decision"]["basis"]["source"]
    watermark.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(UsageLedgerCorrupt, match="invalid exported work caps"):
        usage_store.migrate_from_journal(data_root)
    assert not (data_root / usage_store.STORE_REL).exists()
    watermark.write_bytes(saved)
    assert _raise(data_root, "work-a", 15, MIND)["error"] == "owner_cap_binds"


def test_unverified_compaction_keeps_money_and_limits_only_amended_groups(data_root):  # noqa: F811
    from supervisor.budget_resume import _root_money_refusal

    work = _work(data_root, "known")
    known_scope = _scope(data_root, "known", "known", "known")
    _spend(known_scope, 2.0)
    _work(data_root, "without-original")
    for group in ("known", "without-original"):
        assert _raise(data_root, group, 14.0)["ok"]
    usage_store.export_journal(data_root)
    watermark = (data_root / usage_journal.IMPORT_REL).read_bytes()
    # A valid compacted live format, with no archived proof. This fixture exercises the reader,
    # not the old writer's transformation (qualified separately against the historical source).
    write_compacted_journal(data_root, [dict(_binding("known"), attempt_id="folded", root_task_id="known",
                                             folded_attempt_count=1, cost_usd=2.0, cost_final=True)])
    imported = usage_store.migrate_from_journal(data_root)
    assert imported["status"] == "completed" and imported["unverified_work_caps"] == 2
    snapshot = task_money_snapshot(data_root, work, "known", root_limit=10.0)
    axis = snapshot["group_axis"]
    assert axis["limit_usd"] == 10.0 and axis["source"] == "legacy_live" and snapshot["settled_usd"] == 2.0
    assert axis["amendment_state"] == "unverified_export" and "basis_kind" not in axis
    assert _raise(data_root, "known", 15.0)["error"] == "work_cap_export_unverified"
    _spend(known_scope, 8.0)  # through the existing admission and settlement paths, under the known original
    refusal = _root_money_refusal(data_root, work, "known")
    assert (refusal["error"], refusal["action"]) == ("work_cap_export_unverified", "verify_work_cap_export_then_resume")
    # An independent root fence still takes precedence over an unverified exported group amendment.
    from ouroboros.tools.review_helpers import review_wave_binding_fence
    with ua.usage_scope(_scope(data_root, "known", "known", "known", root_limit=10.0, root_limit_source="producer")):
        indep_wave = ua.review_wave_admission(data_root, root_task_id="known", task_id="known", models=[],
                                              session_slots=1, prompt_chars=0, root_limit_usd=10.0,
                                              root_limit_source="producer")
    assert (indep_wave["fits"], indep_wave["binding_axis"], indep_wave["binding_can_use_work_cap"]) == (False, "root", False)
    assert "amendment_state" not in indep_wave
    assert "resolve that limit at its source" in review_wave_binding_fence(indep_wave)[1]
    from ouroboros._usage_rows import work_cap_block
    assert work_cap_block({"root": (5.0, "producer"), "group": (10.0, "legacy_live")},
                          {"root": 0.0, "group": 5.0}, "root", 10.0, unverified=True) == ("independent_cap_exhausted", "root")
    with pytest.raises(ua.BudgetExceeded, match="whole-work budget exhausted for group known"):
        _spend(known_scope, 0.01)  # the previously exported raise is not applied
    with pytest.raises(ua.BillingAuthorityUnavailable, match="work_cap_original_unavailable"):
        _spend(_scope(data_root, "without-original", "without-original", "without-original"), 0.01)
    _work(data_root, "unrelated")
    _spend(_scope(data_root, "unrelated", "unrelated", "unrelated"), 1.0)
    with usage_store.read(data_root) as txn:
        retained = json.loads(watermark)[usage_journal.EXPORTED_WORK_CAPS]["caps"]
        assert txn.binding("work_cap", "known") == {"unverified_export": retained["known"]}
        assert txn.summary()["settled_usd"] == 11.0
    # A subsequent export carries pending unverified amendments separately, never upgrading them.
    assert usage_store.export_refusal(data_root) == ""
    assert usage_store.export_journal(data_root)["status"] == "exported"
    carried = usage_journal.completed_legacy_watermark(data_root)[usage_journal.EXPORTED_WORK_CAPS]
    assert carried["caps"] == {} and carried["unverified_caps"] == retained
    reimported = usage_store.migrate_from_journal(data_root)
    assert reimported["status"] == "completed" and reimported["unverified_work_caps"] == 2
    with usage_store.read(data_root) as txn:
        assert txn.binding("work_cap", "known") == {"unverified_export": retained["known"]}
        assert txn.binding("work_cap", "without-original") == {"unverified_export": retained["without-original"]}
        assert txn.summary()["settled_usd"] == 11.0
    with pytest.raises(ua.BudgetExceeded, match="whole-work budget exhausted for group known"):
        _spend(known_scope, 0.01)
    with pytest.raises(ua.BillingAuthorityUnavailable, match="work_cap_original_unavailable"):
        _spend(_scope(data_root, "without-original", "without-original", "without-original"), 0.01)


def test_waiting_legacy_reservation_keeps_its_uncertain_separate_bound(data_root):  # noqa: F811
    _work(data_root, "work-a")
    scope = _scope(data_root, "work-a", "work-a", "work-a")
    _spend(scope, 9.0)
    with usage_store.hold(data_root) as view:
        view.write({"kind": "attempt", "state": "reserved", "attempt_id": "old-reservation", "task_id": "work-a",
                    "root_task_id": "work-a", "root_limit_usd": 12.0, "provider": "openai", "model": "test",
                    "reservation_upper_bound_usd": 1.0, **_binding("work-a")})
    assert _raise(data_root, "work-a", 15.0)["ok"]
    _spend(scope, 3.0)
    with pytest.raises(ua.BudgetExceeded, match="root model budget changed before dispatch"):
        ua.mark_dispatched(ua.AttemptReservation("old-reservation", data_root, "test", "openai", 1.0))
    saved = next(row for row in ledger_rows(data_root) if row["attempt_id"] == "old-reservation")
    assert saved["state"] == "reserved" and saved["root_limit_usd"] == 12.0


def test_wave_and_continue_each_take_one_budget_snapshot(data_root, monkeypatch):  # noqa: F811
    _work(data_root, "work-a")
    scope = _scope(data_root, "work-a", "work-a", "work-a")
    _spend(scope, 10.0)
    assert _raise(data_root, "work-a", 17.0)["ok"]
    opens, read = [], usage_store.read

    @contextmanager
    def observed(*args, **kwargs):
        opens.append(True)
        with read(*args, **kwargs) as view:
            yield view

    monkeypatch.setattr(usage_store, "read", observed)
    assert not group_money_spent(data_root, _binding("work-a")) and len(opens) == 1
    with ua.usage_scope(scope):
        wave = ua.review_wave_admission(data_root, root_task_id="work-a", task_id="work-a", models=[],
                                        session_slots=1, prompt_chars=0, root_limit_usd=10.0)
    assert len(opens) == 2
    assert wave["fits"] and wave["remaining_usd"] == 7.0 and wave["known_usd"] == 10.0
    assert wave["binding_axis"] == "group" and wave["binding_source"] == "work_cap" and wave["billing_group_id"] == "work-a"


def test_an_independent_root_fence_binds_on_a_tie_even_after_a_group_amendment(data_root):  # noqa: F811
    task = _work(data_root, "work-a")
    scope = _scope(data_root, "work-a", "work-a", "work-a",
                   root_limit_source="producer", root_limit_revision="first")
    _spend(scope, 10.0)
    assert task_money_snapshot(data_root, task, "work-a", root_limit=10.0)["binding_axis"] == "root"
    assert _raise(data_root, "work-a", 12.0)["ok"]
    snap = task_money_snapshot(data_root, task, "work-a", root_limit=12.0)
    assert snap["remaining_known_usd"] == 0.0 and snap["binding_axis"] == "root"
    assert snap["root_axis"]["source"] == "producer" and snap["root_axis"]["revision"] == "first"
    with ua.usage_scope(scope):
        wave = ua.review_wave_admission(data_root, root_task_id="work-a", task_id="work-a", models=[],
                                        session_slots=1, prompt_chars=0, root_limit_usd=12.0)
    assert not wave["fits"] and wave["binding_axis"] == "root" and wave["binding_source"] == "producer"
    with pytest.raises(ua.BudgetExceeded, match="root model budget exhausted"):
        _spend(scope, 1.0)


def test_later_schedule_edit_does_not_change_an_accepted_occurrences_wallet(world):  # noqa: F811
    from supervisor import queue
    from supervisor import queue_schedules as schedules
    from supervisor.worker_assignment import _claim_worker_launch

    _ctx, root, pending = world
    registered = register(world)
    queue.check_scheduled_tasks()
    [task] = pending
    row = load_task_result(root, task["id"])
    assert row["schedule_admission"]["status"] == "accepted"
    bound = row["billing_group"]
    assert _raise(root, task["id"], 14.0)["error"] == "work_cap_unbound"
    with usage_store.read(root) as view:
        assert work_cap(view, bound["billing_group_id"]) is None
    sent = []
    assert _claim_worker_launch(queue, task, SimpleNamespace(in_q=SimpleNamespace(put=sent.append)))
    assert sent == [task] and task["admitted_dispatch"] == "possible"
    # A later schedule uses its own wallet; the frozen accepted occurrence already belongs to the origin.
    schedules._write_scheduled_tasks({"tasks": [{**registered, "followup_relation": {"kind": "independent"}}]}, root)
    assert task_billing_fields(task, task["id"], 10.0, root)["billing_group_id"] == bound["billing_group_id"]
    assert _raise(root, task["id"], 14.0)["billing_group_id"] == bound["billing_group_id"]
    with usage_store.read(root) as view:
        assert work_cap(view, task["id"]) is None and view.attempts() == []
        assert work_cap(view, bound["billing_group_id"])["limit_usd"] == 14.0


def test_unverified_occurrence_origin_is_not_rederived_from_the_template(world):  # noqa: F811
    _, root, pending = world
    register(world)
    from supervisor import queue

    queue.check_scheduled_tasks()
    [task] = pending
    row = load_task_result(root, task["id"])
    write_task_result(root, task["id"], "scheduled", schedule_admission={**row["schedule_admission"], "dispatch": "unknown"})
    assert _raise(root, task["id"], 14.0)["error"] == "work_cap_unbound"
    with usage_store.read(root) as view:
        assert view.bindings("work_cap") == {}


def _fresh_root(tmp_path, monkeypatch, name):
    root = tmp_path / name
    (root / "state").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    ua._reset_task_cache_splits()
    return root


def _intermediate_sqlite_import(root):
    """Simulate an intermediate SQLite release (e.g. 4b8e4221) importing without ``exported_work_caps``."""
    usage_store.forget(root)
    watermark = root / usage_journal.IMPORT_REL
    saved = watermark.read_bytes() if watermark.exists() else None
    if saved is not None:
        doc = json.loads(saved)
        doc.pop(usage_journal.EXPORTED_WORK_CAPS, None)
        watermark.write_text(json.dumps(doc), encoding="utf-8")
    try:
        assert usage_store.migrate_from_journal(root)["status"] == "completed"
    finally:
        if saved is not None:
            watermark.write_bytes(saved)
    with usage_store.hold(root) as txn:
        txn.conn.execute("DELETE FROM bindings WHERE scope='work_cap'")
        meta = dict(txn.meta("import"))
        meta.pop("work_caps_export", None)
        meta["legacy"] = None
        txn.set_meta("import", meta)
    usage_store.forget(root)


def _append_older_attempt(root, group, attempt_id, cost):
    path = root / LEDGER_REL
    seq = len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0
    base = {"kind": "attempt", "attempt_id": attempt_id, "task_id": group, "root_task_id": group, "provider": "openai",
            "model": "m", "category": "task", "source": "older", "root_limit_usd": 10.0,
            "reservation_upper_bound_usd": cost, "pricing_known": True, "ts": "2026-01-01T00:05:00Z", **_binding(group)}
    rows = [{**base, "state": "reserved", "seq": seq + 1}, {**base, "state": "dispatched", "seq": seq + 2},
            {**base, "state": "settled", "cost_usd": cost, "cost_final": True, "seq": seq + 3}]
    with path.open("ab") as handle:
        handle.write(usage_journal.journal_bytes(rows))


def test_intermediate_import_restores_caps_with_lifespan_boot_spend_and_idempotence(data_root, monkeypatch):  # noqa: F811
    from ouroboros.usage_admission import set_work_cap

    work, quiet = _work(data_root, "work-a"), _work(data_root, "quiet-work")
    _work(data_root, "unrelated")
    scope_a, scope_q = _scope(data_root, "work-a", "work-a", "work-a"), _scope(data_root, "quiet-work", "quiet-work", "quiet-work")
    _spend(scope_a, 8.0)
    owner = {**OWNER, "source": {"ts": "2026-01-02T00:00:00Z"}, "delegated_max_usd": 19}
    assert _raise(data_root, "work-a", 15.0, owner)["ok"]
    assert set_work_cap(data_root, work, "work-a", limit_usd=18.0, by=BY, basis=MIND)["ok"]
    assert _raise(data_root, "quiet-work", 16.0)["ok"]
    with usage_store.read(data_root) as view:
        expected_a, expected_q = work_cap(view, "work-a"), work_cap(view, "quiet-work")

    assert usage_store.export_journal(data_root)["status"] == "exported"
    _intermediate_sqlite_import(data_root)
    _spend(scope_a, 2.0)
    _spend(scope_q, 3.0)
    _spend(_scope(data_root, "unrelated", "unrelated", "unrelated"), 1.5)
    with pytest.raises(ua.BudgetExceeded, match="whole-work budget exhausted for group work-a"):
        _spend(scope_a, 0.5)

    repaired = usage_store.migrate_from_journal(data_root, initialize_journal=True)
    assert repaired["status"] == "already_completed" and repaired["work_caps_restored"] == 2
    assert "unverified_work_caps" not in repaired
    with usage_store.read(data_root) as view:
        assert work_cap(view, "work-a") == expected_a and work_cap(view, "quiet-work") == expected_q
        assert view.summary(billing_group_id="work-a")["settled_usd"] == 10.0
        assert view.summary(billing_group_id="quiet-work")["settled_usd"] == 3.0
        assert view.meta("import")["work_caps_export"]["status"] == "verified"

    usage_store.forget(data_root)
    store_bytes = (data_root / usage_store.STORE_REL).read_bytes()
    with monkeypatch.context() as mp:
        mp.setattr(usage_journal, "completed_legacy_watermark",
                   lambda _root: pytest.fail("watermark must not be read after reconciliation"))
        again = usage_store.migrate_from_journal(data_root, initialize_journal=True)
        assert not ({"work_caps_restored", "unverified_work_caps", "work_caps_preserved"} & again.keys())
        usage_store.forget(data_root)
        assert (data_root / usage_store.STORE_REL).read_bytes() == store_bytes
        assert task_money_snapshot(data_root, quiet, "quiet-work", root_limit=10.0)["remaining_known_usd"] == 13.0

    _spend(scope_a, 7.0)
    assert _raise(data_root, "work-a", 19.0, MIND)["ok"]
    assert _raise(data_root, "work-a", 20.0, MIND)["error"] == "owner_cap_binds"


def test_intermediate_lineage_empty_appended_packed_and_bare_zero_prefix(tmp_path, monkeypatch):
    import hashlib
    import lzma

    # 1. Exact 0-byte export matches even if the empty journal is later removed; a nonempty changed journal does not.
    for changed in (False, True):
        root = _fresh_root(tmp_path, monkeypatch, f"empty-{changed}")
        quiet = _work(root, "quiet-work")
        assert _raise(root, "quiet-work", 16.0)["ok"]
        assert usage_store.export_journal(root)["journal"]["size"] == 0
        if changed:
            _append_older_attempt(root, "quiet-work", "older-1", 2.0)
            # Cold import also refuses to establish lineage on a bare 0-byte prefix.
            cold = usage_store.migrate_from_journal(root)
            assert cold["unverified_work_caps"] == 1
            (root / usage_store.STORE_REL).unlink()
        _intermediate_sqlite_import(root)
        if not changed:
            _spend(_scope(root, "quiet-work", "quiet-work", "quiet-work"), 4.0)
            (root / LEDGER_REL).unlink()
        report = usage_store.migrate_from_journal(root)
        if changed:
            assert report["unverified_work_caps"] == 1 and "work_caps_restored" not in report
            assert task_money_snapshot(root, quiet, "quiet-work", root_limit=10.0)["group_axis"]["amendment_state"] == "unverified_export"
        else:
            assert report["work_caps_restored"] == 1
            assert task_money_snapshot(root, quiet, "quiet-work", root_limit=10.0)["remaining_known_usd"] == 12.0

    # 2. Appended nonempty journal before intermediate SQLite import verifies via kept-journal prefix.
    appended = _fresh_root(tmp_path, monkeypatch, "appended")
    work_a = _work(appended, "work-a")
    scope_a = _scope(appended, "work-a", "work-a", "work-a")
    _spend(scope_a, 3.0)
    assert _raise(appended, "work-a", 15.0)["ok"]
    assert usage_store.export_journal(appended)["status"] == "exported"
    _append_older_attempt(appended, "work-a", "old-append", 2.0)
    _intermediate_sqlite_import(appended)
    _spend(scope_a, 1.0)
    assert usage_store.migrate_from_journal(appended)["work_caps_restored"] == 1
    snap = task_money_snapshot(appended, work_a, "work-a", root_limit=10.0)
    assert snap["settled_usd"] == 6.0 and snap["remaining_known_usd"] == 9.0

    # 3. Packed .xz compacted archive verifies via meta.import.header even after the live journal is removed.
    compacted = _fresh_root(tmp_path, monkeypatch, "compacted")
    work_c, scope_c = _work(compacted, "work-c"), _scope(compacted, "work-c", "work-c", "work-c")
    _spend(scope_c, 3.0)
    assert _raise(compacted, "work-c", 15.0)["ok"] and usage_store.export_journal(compacted)["status"] == "exported"
    exported = (compacted / LEDGER_REL).read_bytes()
    archive_rel, n_rows = "archive/usage_ledger/seg-1.jsonl", len(exported.splitlines())
    packed = compacted / (archive_rel + ".xz")
    packed.parent.mkdir(parents=True, exist_ok=True)
    packed.write_bytes(lzma.compress(exported))
    (compacted / LEDGER_REL).write_bytes(usage_journal.journal_bytes([
        {"kind": "usage_baseline", "attempt_id": "b1", "baseline_id": "b1", "state": "settled", "compaction_epoch": 1,
         "archive_rel": archive_rel, "source_size_bytes": len(exported), "source_sha256": hashlib.sha256(exported).hexdigest(),
         "folded_attempt_count": 1, "group_count": 1, "source_row_count": n_rows, "folded_row_count": n_rows,
         "retained_row_count": 0, "source_first_seq": 1, "source_last_seq": n_rows, "ts": "2026-01-01T00:06:00Z", "seq": 1},
        {"kind": "usage_baseline_group", "attempt_id": "fold-1", "baseline_id": "b1", "state": "settled", "task_id": "work-c",
         "root_task_id": "work-c", "model": "m", "provider": "openai", "category": "task", "source": "old",
         "folded_attempt_count": 1, "cost_usd": "3.0", "cost_final": True, "root_limit_usd": 10.0, **_binding("work-c"),
         "ts": "2026-01-01T00:06:00Z", "seq": 2},
    ]))
    _intermediate_sqlite_import(compacted)
    _spend(scope_c, 2.0)
    (compacted / LEDGER_REL).unlink()
    assert usage_store.migrate_from_journal(compacted)["work_caps_restored"] == 1
    snap_c = task_money_snapshot(compacted, work_c, "work-c", root_limit=10.0)
    assert snap_c["settled_usd"] == 5.0 and snap_c["remaining_known_usd"] == 10.0


def test_unverified_export_carries_across_reexport_and_intermediate_boot_while_newer_decision_wins(data_root):  # noqa: F811
    work_known, work_active = _work(data_root, "known"), _work(data_root, "active")
    _work(data_root, "without-original")
    _work(data_root, "unrelated")
    known_scope = _scope(data_root, "known", "known", "known")
    _spend(known_scope, 2.0)
    for group in ("known", "without-original"):
        assert _raise(data_root, group, 14.0)["ok"]
    assert usage_store.export_journal(data_root)["status"] == "exported"
    orig_caps = json.loads((data_root / usage_journal.IMPORT_REL).read_text(encoding="utf-8"))[
        usage_journal.EXPORTED_WORK_CAPS]["caps"]
    write_compacted_journal(data_root, [dict(_binding("known"), attempt_id="folded", root_task_id="known",
                                             folded_attempt_count=1, cost_usd=2.0, cost_final=True)])
    _intermediate_sqlite_import(data_root)
    assert usage_store.migrate_from_journal(data_root)["unverified_work_caps"] == 2
    assert _raise(data_root, "active", 18.0)["ok"]

    # Export carries pending unverified amendments separately alongside active caps.
    assert usage_store.export_refusal(data_root) == ""
    assert usage_store.export_journal(data_root)["status"] == "exported"
    carried = usage_journal.completed_legacy_watermark(data_root)[usage_journal.EXPORTED_WORK_CAPS]
    assert set(carried["caps"]) == {"active"} and carried["unverified_caps"] == orig_caps

    # Through an intermediate SQLite import, a newer pre-upgrade SQLite decision on ``known`` wins,
    # ``active`` restores verified, and ``without-original`` stays inactive (never upgraded).
    _intermediate_sqlite_import(data_root)
    newer = {**OWNER, "source": {"ts": "2026-01-05T00:00:00Z"}}
    newer_record = {"billing_group_id": "known", "task_id": "known", "limit_usd": 22.0,
                    "original_limit_usd": 10.0, "previous_limit_usd": 10.0,
                    "revision": "2026-01-05T00:00:01Z", "by": dict(BY), "basis": newer}
    with usage_store.hold(data_root) as txn:
        txn.conn.execute("INSERT INTO bindings (scope, key, binding) VALUES ('work_cap', 'known', ?)",
                         (json.dumps(newer_record),))
    repaired = usage_store.migrate_from_journal(data_root, initialize_journal=True)
    assert repaired["work_caps_restored"] == 1 and repaired["unverified_work_caps"] == 1 and repaired["work_caps_preserved"] == 1
    with usage_store.read(data_root) as view:
        assert work_cap(view, "known") == newer_record
        assert work_cap(view, "active")["limit_usd"] == 18.0
        assert view.binding("work_cap", "without-original") == {"unverified_export": orig_caps["without-original"]}
        assert view.meta("import")["work_caps_export"]["status"] == "partial"
    assert task_money_snapshot(data_root, work_known, "known", root_limit=10.0)["remaining_known_usd"] == 20.0
    assert task_money_snapshot(data_root, work_active, "active", root_limit=10.0)["remaining_known_usd"] == 18.0
    with pytest.raises(ua.BillingAuthorityUnavailable, match="work_cap_original_unavailable"):
        _spend(_scope(data_root, "without-original", "without-original", "without-original"), 0.01)


def test_export_refusal_and_export_journal_share_prerequisites_with_tx_recheck(tmp_path, monkeypatch):
    root = _fresh_root(tmp_path, monkeypatch, "prereq")
    _work(root, "work-a")
    _spend(_scope(root, "work-a", "work-a", "work-a"), 1.0)
    assert _raise(root, "work-a", 14.0)["ok"]
    assert usage_store.export_journal(root)["status"] == "exported"
    _intermediate_sqlite_import(root)

    # 1. Unreconciled store missing prior export decisions refuses in BOTH preview (even without live journal) and export.
    (root / LEDGER_REL).unlink()
    assert "not held in the active store" in usage_store.export_refusal(root)
    with pytest.raises(UsageLedgerCorrupt, match="not held in the active store"):
        usage_store.export_journal(root)
    with usage_store.read(root) as view:
        assert view.meta("export") is None

    # 2. Missing legacy watermark refuses in both preview and export before freezing.
    assert usage_store.migrate_from_journal(root)["work_caps_restored"] == 1
    watermark = root / usage_journal.IMPORT_REL
    saved_wm = watermark.read_bytes()
    watermark.unlink()
    assert "completed legacy import watermark" in usage_store.export_refusal(root)
    with pytest.raises(UsageAccountingError, match="completed legacy import watermark"):
        usage_store.export_journal(root)
    watermark.write_bytes(b"{")
    assert "unreadable legacy usage watermark" in usage_store.export_refusal(root)
    with pytest.raises(UsageLedgerCorrupt, match="unreadable legacy usage watermark"):
        usage_store.export_journal(root)
    watermark.write_bytes(saved_wm)

    # 3. Malformed work_cap binding in SQLite refuses in both preview and export before freezing.
    with usage_store.hold(root) as txn:
        valid_binding = txn.binding("work_cap", "work-a")
        txn.conn.execute("UPDATE bindings SET binding=? WHERE scope='work_cap' AND key='work-a'", (json.dumps({"bad": 1}),))
    assert "invalid exported work caps" in usage_store.export_refusal(root)
    with pytest.raises(UsageLedgerCorrupt, match="invalid exported work caps"):
        usage_store.export_journal(root)
    with usage_store.hold(root) as txn:
        txn.set_work_cap("work-a", valid_binding)
        assert txn.meta("export") is None

    # 4. Transaction-time recheck catches watermark damage occurring after preview, without freezing or moving files.
    real_error = usage_store._export_error

    def race_after_preview(r, *, resume=False):
        assert real_error(r, resume=resume) is None
        watermark.write_bytes(b"{")
        return None

    with monkeypatch.context() as mp:
        mp.setattr(usage_store, "_export_error", race_after_preview)
        with pytest.raises(UsageLedgerCorrupt, match="unreadable legacy usage watermark"):
            usage_store.export_journal(root)
    with usage_store.read(root) as view:
        assert view.meta("export") is None
    assert not (root / LEDGER_REL).exists()


def test_completed_store_legacy_watermark_failure_preserves_money_and_tier_locks(tmp_path, monkeypatch):
    from ouroboros import platform_layer

    # 1. Ordinary already-complete store with unreadable or corrupt legacy watermark keeps established money working.
    ordinary = _fresh_root(tmp_path, monkeypatch, "ordinary")
    _work(ordinary, "work-a")
    _spend(_scope(ordinary, "work-a", "work-a", "work-a"), 2.0)
    with usage_store.hold(ordinary) as txn:
        meta = dict(txn.meta("import"))
        meta.pop("work_caps_export", None)
        txn.set_meta("import", meta)
    wm_ord = ordinary / usage_journal.IMPORT_REL
    real_read_text = type(wm_ord).read_text
    with monkeypatch.context() as mp:
        mp.setattr(type(wm_ord), "read_text", lambda p, *a, **k: (_ for _ in ()).throw(PermissionError("denied")) if p == wm_ord else real_read_text(p, *a, **k))
        assert usage_store.migrate_from_journal(ordinary, initialize_journal=True)["work_caps_reconcile"] == "watermark_unreadable"
    wm_ord.write_bytes(b"{")
    report = usage_store.migrate_from_journal(ordinary, initialize_journal=True)
    assert report["status"] == "already_completed" and report["work_caps_reconcile"] == "watermark_unreadable"
    _spend(_scope(ordinary, "work-a", "work-a", "work-a"), 1.0)
    with usage_store.read(ordinary) as view:
        assert view.summary(billing_group_id="work-a")["settled_usd"] == 3.0

    # 2. Post-export intermediate store: missing watermark discloses unknown without crashing; corrupt cap stamp fails closed;
    # frozen exporting store is skipped; TIER_NAME takes the money lock; post-reconcile watermark damage is ignored.
    root = _fresh_root(tmp_path, monkeypatch, "exported-name-tier")
    _work(root, "work-a")
    _spend(_scope(root, "work-a", "work-a", "work-a"), 1.0)
    assert _raise(root, "work-a", 12, {**OWNER, "delegated_max_usd": 14})["ok"]
    assert _raise(root, "work-a", 13, MIND)["ok"]
    assert usage_store.export_journal(root)["status"] == "exported"
    with monkeypatch.context() as mp:
        mp.setattr(platform_layer, "kernel_file_locks_enforced", lambda _path: False)
        _intermediate_sqlite_import(root)
    assert usage_store._header_tier(root / usage_store.STORE_REL) == usage_store.TIER_NAME

    watermark = root / usage_journal.IMPORT_REL
    saved = watermark.read_bytes()
    watermark.unlink()
    missing_report = usage_store.migrate_from_journal(root)
    assert missing_report["work_caps_reconcile"] == "watermark_missing_after_export"
    _spend(_scope(root, "work-a", "work-a", "work-a"), 1.0)

    damaged = json.loads(saved)
    del damaged[usage_journal.EXPORTED_WORK_CAPS]["caps"]["work-a"]["owner_decision"]["basis"]["source"]
    watermark.write_text(json.dumps(damaged), encoding="utf-8")
    with pytest.raises(UsageLedgerCorrupt, match="invalid exported work caps"):
        usage_store.migrate_from_journal(root)

    watermark.write_bytes(saved)
    with usage_store.hold(root, exporting=True) as txn:
        txn.set_meta("export", {"status": "writing", "journal": {"size": 0, "sha256": "0" * 64, "rows": 0}})
    assert "work_caps_restored" not in usage_store.migrate_from_journal(root)
    with usage_store.hold(root, exporting=True) as txn:
        txn.set_meta("export", None)

    taken, named = [], usage_store._named_lock
    with monkeypatch.context() as mp:
        mp.setattr(usage_store, "_named_lock", lambda r, name, **kw: taken.append(name) or named(r, name, **kw))
        assert usage_store.migrate_from_journal(root)["work_caps_restored"] == 1
    assert usage_store.LOCK_REL.name in taken and _raise(root, "work-a", 15, MIND)["error"] == "owner_cap_binds"
    watermark.write_bytes(b"{")
    assert usage_store.migrate_from_journal(root)["status"] == "already_completed"


def test_pending_reconcile_guards_new_caps_keeps_spending_zero_scan_and_recovers(tmp_path, monkeypatch):
    from ouroboros.usage_admission import work_cap_target

    # 1. Post-export intermediate SQLite install: reading target or requesting an amendment never triggers recovery
    # or scans export files (before boot or after watermark returns); migrate_from_journal is the sole recovery owner.
    root = _fresh_root(tmp_path, monkeypatch, "pending-migrate-only")
    work_a, work_q = _work(root, "work-a"), _work(root, "quiet")
    _work(root, "local")
    scope_a = _scope(root, "work-a", "work-a", "work-a")
    _spend(scope_a, 1.0)
    assert _raise(root, "work-a", 12.0, {**OWNER, "delegated_max_usd": 14.0})["ok"]
    assert _raise(root, "work-a", 13.0, MIND)["ok"]
    assert _raise(root, "quiet", 16.0)["ok"]
    assert usage_store.export_journal(root)["status"] == "exported"
    _intermediate_sqlite_import(root)

    # Simulate a pre-upgrade local SQLite work_cap decision on ``local`` AND a populated meta.import.legacy.
    local_record = {"billing_group_id": "local", "task_id": "local", "limit_usd": 15.0,
                    "original_limit_usd": 10.0, "previous_limit_usd": 10.0,
                    "revision": "2026-01-04T00:00:00Z", "by": dict(BY), "basis": dict(MIND)}
    watermark = root / usage_journal.IMPORT_REL
    saved_wm = watermark.read_bytes()
    with usage_store.hold(root) as txn:
        txn.conn.execute("INSERT INTO bindings (scope, key, binding) VALUES ('work_cap', 'local', ?)",
                         (json.dumps(local_record),))
        txn.set_meta("import", {**txn.meta("import"), "legacy": usage_journal.import_completion(json.loads(saved_wm))})

    # Before migrate_from_journal runs (even with readable watermark on disk), target read and cap write stay read-only/refused.
    with monkeypatch.context() as mp:
        mp.setattr(usage_journal, "completed_legacy_watermark",
                   lambda _r: pytest.fail("cap paths must not read watermark before migration"))
        mp.setattr(usage_store, "_had_usage_export",
                   lambda _r: pytest.fail("cap paths must not scan export files before migration"))
        assert work_cap_target(root, load_task_result(root, "work-a"), "work-a") == {
            "error": "work_cap_reconcile_pending", "billing_group_id": "work-a", "reason": "reconcile_not_run"}
        assert _raise(root, "work-a", 15.0, MIND) == {
            "ok": False, "error": "work_cap_reconcile_pending", "billing_group_id": "work-a", "reason": "reconcile_not_run"}
    with usage_store.read(root) as view:
        assert "work_caps_export" not in view.meta("import")
        assert set(view.bindings("work_cap")) == {"local"}

    # Boot with unreadable watermark records pending status; repeated boot skips redundant SQLite writes.
    real_read_text = type(watermark).read_text
    with monkeypatch.context() as mp:
        mp.setattr(type(watermark), "read_text",
                   lambda p, *a, **k: (_ for _ in ()).throw(PermissionError("denied")) if p == watermark else real_read_text(p, *a, **k))
        boot = usage_store.migrate_from_journal(root, initialize_journal=True)
    assert boot["work_caps_reconcile"] == "watermark_unreadable_after_export"
    with usage_store.read(root) as view:
        assert view.meta("import")["work_caps_export"] == {"status": "pending", "reason": "watermark_unreadable_after_export"}
    watermark.write_bytes(b"{")
    usage_store.forget(root)
    before_bytes = (root / usage_store.STORE_REL).read_bytes()
    assert usage_store.migrate_from_journal(root, initialize_journal=True)["work_caps_reconcile"] == "watermark_unreadable_after_export"
    usage_store.forget(root)
    assert (root / usage_store.STORE_REL).read_bytes() == before_bytes

    # Regular usage, projections, and cap-guard checks stay under proved original limits and NEVER scan files.
    with monkeypatch.context() as mp:
        mp.setattr(usage_journal, "completed_legacy_watermark",
                   lambda _r: pytest.fail("ordinary usage and cap checks must not read watermark"))
        mp.setattr(usage_store, "_had_usage_export",
                   lambda _r: pytest.fail("ordinary usage and cap checks must not scan export files"))
        _spend(scope_a, 8.0)
        snap = task_money_snapshot(root, work_a, "work-a", root_limit=10.0)
        assert snap["settled_usd"] == 9.0 and snap["remaining_known_usd"] == 1.0
        assert snap["group_axis"]["amendment_state"] == "reconcile_pending"
        assert ua.usage_projection(root, billing_group_id="work-a")["settled_usd"] == 9.0
        assert not group_money_spent(root, load_task_result(root, "work-a")["billing_group"])
        wave = ua.review_wave_admission(root, root_task_id="work-a", models=[], prompt_chars=0, session_slots=1)
        assert wave["fits"] is True and wave["binding_can_use_work_cap"] is False
        assert wave["amendment_state"] == "reconcile_pending"
        _spend(scope_a, 1.0)
        with pytest.raises(ua.BudgetExceeded, match="whole-work budget exhausted for group work-a"):
            _spend(scope_a, 0.5)
        assert group_money_spent(root, load_task_result(root, "work-a")["billing_group"])
        # Readers agree with the writer while reconciliation is pending: Resume and review wave report pending
        # reconciliation (never telling the owner to call set_task_cost_cap), while a local SQLite cap stays amendable.
        from ouroboros.consciousness_view import _resume_room
        from ouroboros.tools.review_helpers import review_wave_binding_fence
        from supervisor.budget_resume import _root_money_refusal

        spent_snap = task_money_snapshot(root, work_a, "work-a", root_limit=10.0)
        assert spent_snap["work_cap_block"] == {"error": "work_cap_reconcile_pending", "axis": "group"}
        resume_refusal = _root_money_refusal(root, work_a, "work-a")
        assert (resume_refusal["error"], resume_refusal["binding_axis"], resume_refusal["action"]) == (
            "work_cap_reconcile_pending", "group", "reconcile_work_caps_then_resume")
        assert _resume_room(root, "work-a", {}, {}) == {"resume_additional_usd": 0.0}
        spent_wave = ua.review_wave_admission(root, root_task_id="work-a", models=[], prompt_chars=0, session_slots=1)
        assert spent_wave["fits"] is False and spent_wave["binding_can_use_work_cap"] is False
        assert spent_wave["amendment_state"] == "reconcile_pending"
        assert "reconciliation" in review_wave_binding_fence(spent_wave)[1]
        assert "raise this work's cap conversationally" not in review_wave_binding_fence(spent_wave)[1]
        # Independent root fence takes precedence over pending reconciliation on the same work.
        indep_wave = ua.review_wave_admission(root, root_task_id="work-a", models=[], prompt_chars=0, session_slots=1,
                                              root_limit_usd=10.0, root_limit_source="producer")
        assert indep_wave["fits"] is False and indep_wave["binding_axis"] == "root"
        assert "amendment_state" not in indep_wave
        assert "resolve that limit at its source" in review_wave_binding_fence(indep_wave)[1]
        local_snap = task_money_snapshot(root, {"id": "local", "root_task_id": "local"}, "local", root_limit=10.0)
        assert "amendment_state" not in local_snap["group_axis"]

        # Cap target, set_work_cap, and Txn.set_work_cap refuse unpopulated groups while local SQLite cap remains amendable.
        assert work_cap_target(root, load_task_result(root, "work-a"), "work-a") == {
            "error": "work_cap_reconcile_pending", "billing_group_id": "work-a", "reason": "watermark_unreadable_after_export"}
        assert _raise(root, "work-a", 15.0, MIND) == {
            "ok": False, "error": "work_cap_reconcile_pending", "billing_group_id": "work-a",
            "reason": "watermark_unreadable_after_export"}
        with usage_store.hold(root) as txn:
            assert txn.binding("work_cap", "work-a") is None
            with pytest.raises(UsageAccountingError, match="reconciliation is pending"):
                txn.set_work_cap("work-a", {**local_record, "billing_group_id": "work-a", "task_id": "work-a"})
        assert not work_cap_target(root, load_task_result(root, "local"), "local").get("error")
        assert _raise(root, "local", 17.0, MIND)["ok"]

    # Missing post-export watermark updates pending reason and refuses export even when meta.import.legacy is populated.
    watermark.unlink()
    assert usage_store.migrate_from_journal(root)["work_caps_reconcile"] == "watermark_missing_after_export"
    assert "missing after an export" in usage_store.export_refusal(root)
    with pytest.raises(UsageLedgerCorrupt, match="missing after an export"):
        usage_store.export_journal(root)
    with usage_store.read(root) as view:
        assert view.meta("export") is None
        pending_before = dict(view.meta("import")["work_caps_export"])
        bindings_before = view.bindings("work_cap")

    # Even AFTER the readable watermark file returns on disk, reading target or requesting an amendment BEFORE
    # migrate_from_journal must not scan export/archive files or alter the pending marker or prior bindings.
    watermark.write_bytes(saved_wm)
    usage_store.forget(root)
    store_bytes_before = (root / usage_store.STORE_REL).read_bytes()
    with monkeypatch.context() as mp:
        mp.setattr(usage_journal, "completed_legacy_watermark",
                   lambda _r: pytest.fail("target/cap paths must not read returned watermark"))
        mp.setattr(usage_journal, "imported_work_caps",
                   lambda *_a, **_kw: pytest.fail("target/cap paths must not reconcile caps"))
        mp.setattr(usage_journal, "_archived_export_prefix",
                   lambda *_a, **_kw: pytest.fail("target/cap paths must not inspect archives"))
        mp.setattr(usage_store, "_had_usage_export",
                   lambda _r: pytest.fail("target/cap paths must not scan export files"))
        assert work_cap_target(root, load_task_result(root, "work-a"), "work-a") == {
            "error": "work_cap_reconcile_pending", "billing_group_id": "work-a", "reason": "watermark_missing_after_export"}
        assert _raise(root, "work-a", 15.0, MIND) == {
            "ok": False, "error": "work_cap_reconcile_pending", "billing_group_id": "work-a",
            "reason": "watermark_missing_after_export"}
    usage_store.forget(root)
    assert (root / usage_store.STORE_REL).read_bytes() == store_bytes_before
    with usage_store.read(root) as view:
        assert view.meta("import")["work_caps_export"] == pending_before
        assert view.bindings("work_cap") == bindings_before

    # Running normal migrate_from_journal reconciles once, clears pending, restores exported caps, and preserves spend/local cap.
    recovered = usage_store.migrate_from_journal(root, initialize_journal=True)
    assert recovered["work_caps_restored"] == 2 and "work_caps_reconcile" not in recovered
    with usage_store.read(root) as view:
        assert view.meta("import")["work_caps_export"]["status"] == "verified"
        assert work_cap(view, "work-a")["limit_usd"] == 13.0
        assert work_cap(view, "quiet")["limit_usd"] == 16.0
        assert work_cap(view, "local")["limit_usd"] == 17.0
        assert view.summary(billing_group_id="work-a")["settled_usd"] == 10.0

    # Subsequent spending and amendments proceed against restored authority without reading the watermark again.
    with monkeypatch.context() as mp:
        mp.setattr(usage_journal, "completed_legacy_watermark",
                   lambda _r: pytest.fail("watermark must not be read once reconciled"))
        assert _raise(root, "work-a", 15.0, MIND)["error"] == "owner_cap_binds"
        assert _raise(root, "work-a", 14.0, MIND)["ok"]
        _spend(scope_a, 3.0)
        assert task_money_snapshot(root, work_a, "work-a", root_limit=10.0)["settled_usd"] == 13.0
        assert task_money_snapshot(root, work_q, "quiet", root_limit=10.0)["remaining_known_usd"] == 16.0
        assert _raise(root, "quiet", 18.0)["error"] == "owner_cap_needs_newer_decision"
        assert _raise(root, "quiet", 18.0, {**OWNER, "source": {"ts": "2026-01-06T00:00:00Z"}})["ok"]
        assert usage_store.migrate_from_journal(root, initialize_journal=True)["status"] == "already_completed"

    # 2. Corrupt work_caps_export metadata in SQLite fails closed on target read, amendment, and export.
    bad_root = _fresh_root(tmp_path, monkeypatch, "bad-export-meta")
    _work(bad_root, "work-a")
    _spend(_scope(bad_root, "work-a", "work-a", "work-a"), 1.0)
    with usage_store.hold(bad_root) as txn:
        txn.set_meta("import", {**txn.meta("import"), "work_caps_export": "bogus"})
    with pytest.raises(ua.BillingAuthorityUnavailable, match="work_cap_export_invalid"):
        work_cap_target(bad_root, load_task_result(bad_root, "work-a"), "work-a")
    bad_raise = _raise(bad_root, "work-a", 14.0, MIND)
    assert bad_raise["error"] == "billing_authority_unavailable" and bad_raise["cause"] == "work_cap_export_invalid"
    with pytest.raises(ua.BillingAuthorityUnavailable, match="work_cap_export_invalid"):
        task_money_snapshot(bad_root, {"id": "work-a", "root_task_id": "work-a"}, "work-a", root_limit=10.0)
    from supervisor.budget_resume import _root_money_refusal
    bad_resume = _root_money_refusal(bad_root, {"id": "work-a", "root_task_id": "work-a"}, "work-a")
    assert bad_resume["error"] == "billing_authority_unavailable" and bad_resume["cause"] == "work_cap_export_invalid"
    bad_wave = ua.review_wave_admission(bad_root, root_task_id="work-a", models=[], prompt_chars=0, session_slots=1)
    assert bad_wave["fits"] is False and bad_wave["reason"] == "billing_authority_unavailable"
    assert "invalid work_caps_export metadata" in usage_store.export_refusal(bad_root)
    with pytest.raises(UsageLedgerCorrupt, match="invalid work_caps_export metadata"):
        usage_store.export_journal(bad_root)

    # 3. Ordinary upgraded store (no export) settles work_caps_export to None on first boot and stops reading watermark.
    ordinary = _fresh_root(tmp_path, monkeypatch, "ordinary-settle")
    _work(ordinary, "work-a")
    _spend(_scope(ordinary, "work-a", "work-a", "work-a"), 1.0)
    with usage_store.hold(ordinary) as txn:
        meta = dict(txn.meta("import"))
        meta.pop("work_caps_export", None)
        txn.set_meta("import", meta)
    assert usage_store.migrate_from_journal(ordinary, initialize_journal=True)["status"] == "already_completed"
    with usage_store.read(ordinary) as view:
        assert "work_caps_export" in view.meta("import") and view.meta("import")["work_caps_export"] is None
    (ordinary / usage_journal.IMPORT_REL).write_bytes(b"{")
    with monkeypatch.context() as mp:
        mp.setattr(usage_journal, "completed_legacy_watermark",
                   lambda _r: pytest.fail("settled ordinary store must not read watermark"))
        assert usage_store.migrate_from_journal(ordinary, initialize_journal=True)["status"] == "already_completed"
        assert _raise(ordinary, "work-a", 14.0)["ok"]


def test_zero_byte_export_against_compacted_archive_stays_unverified_without_opening_archive(tmp_path, monkeypatch):
    root = _fresh_root(tmp_path, monkeypatch, "zero-compacted")
    quiet = _work(root, "quiet-work")
    assert _raise(root, "quiet-work", 16.0)["ok"]
    assert usage_store.export_journal(root)["journal"]["size"] == 0
    write_compacted_journal(root, [dict(_binding("quiet-work"), attempt_id="folded", root_task_id="quiet-work",
                                        folded_attempt_count=1, cost_usd=2.0, cost_final=True)])
    with monkeypatch.context() as mp:
        mp.setattr(usage_journal, "_archived_export_prefix",
                   lambda *_a, **_kw: pytest.fail("0-byte export must never inspect compacted archives"))
        cold = usage_store.migrate_from_journal(root)
        assert cold["unverified_work_caps"] == 1
        (root / usage_store.STORE_REL).unlink()
        _intermediate_sqlite_import(root)
        reconciled = usage_store.migrate_from_journal(root)
        assert reconciled["unverified_work_caps"] == 1 and "work_caps_restored" not in reconciled
    assert task_money_snapshot(root, quiet, "quiet-work", root_limit=10.0)["group_axis"]["amendment_state"] == "unverified_export"
