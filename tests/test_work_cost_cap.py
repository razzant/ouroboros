"""Raising a started work's whole cap in conversation (``usage_admission.set_work_cap``), through the REAL ledger.

One fictional work spends against a $10 cap. Its cap, raised to $15, is read by every admission in its own money
transaction: a new request, a reservation saved before the raise at its dispatch, a helper still carrying the
original cap, the review wave, the Resume and Continue money reads and the per-round money facts. The original
binding and the attempt rows stay as written; the amount is the work's total, never money added; the operation only
raises. A narrower cap whose provenance is not the work's own, the shared wallet and an independent work beside it
keep binding; the downgrade export carries the cap apart from physical rows. End to end: my human's words go through
the real tool, its dispatcher and the ordinary Safety check (only the checker's physical model call is replaced), the
host writer, and a helper that was already running; a related follow-up and a Continue fired afterwards
spend under the raised cap.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ouroboros import usage_accounting as ua
from ouroboros.usage_admission import set_work_cap, task_money_snapshot
from tests._safety_fixtures import supervisor  # noqa: F401
from tests._usage_store_testing import ledger_rows
from tests.test_g1_followup_policy import BINDING, ORIGIN, register, world  # noqa: F401

WORDS = "Raise work-a to $15 in total so it can finish its last check"
OWNER = {"kind": "owner_words", "source": {"ts": "2026-01-01T00:00:00Z"}}
MIND = {"kind": "own_judgement"}
BY = {"actor": "ouroboros", "task_id": "main-a", "text": "my human asked to let the work finish"}


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    (root / "state").mkdir(parents=True)
    ua._reset_task_cache_splits()
    return root


def _binding(group, limit=10.0):
    return {"billing_group_id": group, "billing_group_limit_usd": limit,
            "billing_group_limit_source": "initial_task_admission", "billing_group_limit_revision": "2026-01-01"}


def _work(root, work_id, limit=10.0, *, chat_id=None):
    """A started root, pinned as its own group under the configured per-task cap it started with."""
    from ouroboros.task_results import write_task_result

    write_task_result(root, work_id, "running", root_task_id=work_id, billing_group=_binding(work_id, limit),
                      **({"chat_id": chat_id} if chat_id is not None else {}))
    return {"id": work_id, "root_task_id": work_id}


def _scope(root, task_id, root_id, group, *, root_limit=10.0, group_limit=10.0, **extra):
    """What an actor of the work carries: the configured cap at its own start, the group's original binding."""
    return ua.UsageScope(drive_root=root, task_id=task_id, root_task_id=root_id, source="test",
                         root_limit_usd=root_limit, **_binding(group, group_limit), **extra)


def _spend(scope, cost, *, request_root_limit=None):
    with ua.usage_scope(scope):
        reservation = ua.reserve_attempt(ua.AttemptRequest(model="openai/gpt-5.2", provider="openai",
                                                           reservation_usd=cost, root_limit_usd=request_root_limit))
        ua.mark_dispatched(reservation)
        ua.settle_attempt(reservation, {"prompt_tokens": 1, "completion_tokens": 1}, cost_usd=cost, cost_final=True)
    return reservation


def _raise(root, work_id, limit, basis=OWNER):
    return set_work_cap(root, {"id": work_id, "root_task_id": work_id}, work_id, limit_usd=limit, by=BY, basis=basis)


def test_a_raised_cap_is_the_works_total_and_admits_the_next_request(data_root):
    from ouroboros import usage_store
    from ouroboros.usage_admission import original_group_limit, work_cap

    _work(data_root, "work-a")
    scope = _scope(data_root, "work-a", "work-a", "work-a")
    _spend(scope, 9.0)
    raised = _raise(data_root, "work-a", 15.0)
    # $9 of $10 spent, the whole cap is now $15: $6 left by known spend, nothing added or reset.
    assert raised["ok"] and raised["limit_usd"] == 15.0 and raised["previous_limit_usd"] == 10.0
    assert raised["known_usd"] == pytest.approx(9.0) and raised["remaining_known_usd"] == pytest.approx(6.0)
    with usage_store.read(data_root) as view:
        record = work_cap(view, "work-a")
    # The actor that executed it and the basis it rests on, never a new decision of my human beyond their words.
    assert record["by"]["actor"] == "ouroboros" and record["basis"] == OWNER and record["original_limit_usd"] == 10.0
    _spend(scope, 5.5)  # past the original $10: admitted on both axes the work started under
    snapshot = task_money_snapshot(data_root, {"id": "work-a"}, "work-a", root_limit=10.0)
    assert snapshot["root_limit_usd"] == pytest.approx(15.0) and snapshot["remaining_known_usd"] == pytest.approx(0.5)
    assert snapshot["binding_axis"] == "group"
    assert snapshot["group_axis"]["source"] == snapshot["root_axis"]["source"] == "work_cap"
    assert ua.usage_projection(data_root, root_task_id="work-a")["limit_usd"] == 15.0
    assert ua.usage_projection(data_root, billing_group_id="work-a")["limit_usd"] == 15.0
    _spend(scope, 0.5)
    with pytest.raises(ua.BudgetExceeded, match=r"whole-work budget exhausted for group work-a.*limit=\$15\.000000"):
        _spend(scope, 0.01)  # known spend reached the raised cap
    assert original_group_limit(data_root, "work-a")["limit_usd"] == 10.0
    assert {(row["billing_group_limit_usd"], row["root_limit_usd"]) for row in ledger_rows(data_root)} == {(10.0, 10.0)}


def test_only_up_the_same_amount_reads_back_and_my_humans_words_stay_theirs(data_root):
    from ouroboros import usage_store
    from ouroboros.usage_admission import work_cap

    assert _raise(data_root, "work-a", 15.0)["error"] == "work_cap_unbound"  # no admitted task or recorded binding
    _work(data_root, "work-a")
    _spend(_scope(data_root, "work-a", "work-a", "work-a"), 1.0)
    assert _raise(data_root, "work-a", 12.0, MIND)["ok"]  # Ouroboros's own judgement (the writer is told Cyber Pro)
    assert _raise(data_root, "work-a", 11.0, MIND)["error"] == "cost_cap_not_raised"  # never down
    assert _raise(data_root, "work-a", 12.0, MIND).get("unchanged") is True
    confirmed = _raise(data_root, "work-a", 12.0)  # my human's words confirm the same amount: now theirs
    assert confirmed.get("confirmed") is True and confirmed["limit_usd"] == 12.0
    with usage_store.read(data_root) as view:
        revision = work_cap(view, "work-a")["revision"]
    assert _raise(data_root, "work-a", 12, OWNER).get("unchanged") is True
    with usage_store.read(data_root) as view:
        assert work_cap(view, "work-a")["revision"] == revision and work_cap(view, "work-a")["basis"] == OWNER
    refused = _raise(data_root, "work-a", 20.0, MIND)
    assert refused == {"ok": False, "error": "owner_cap_binds", "billing_group_id": "work-a", "limit_usd": 12.0}
    assert _raise(data_root, "work-a", 20.0)["error"] == "owner_cap_needs_newer_decision"
    assert _raise(data_root, "work-a", 20.0, {**OWNER, "source": {"ts": "2026-01-02T00:00:00Z"}})["previous_limit_usd"] == 12.0
    for bad in (0, -1, float("inf"), True, None):
        assert _raise(data_root, "work-a", bad)["error"] == "invalid_limit"
    _work(data_root, "open-a", limit=None)  # a group bound with no cap at all: nothing finite to raise
    _spend(_scope(data_root, "open-a", "open-a", "open-a", root_limit=None, group_limit=None), 1.0)
    assert _raise(data_root, "open-a", 50.0)["error"] == "work_cap_unlimited"


def test_my_human_can_confirm_the_original_amount_without_raising_it(data_root):
    from ouroboros import usage_store
    from ouroboros.usage_admission import work_cap

    _work(data_root, "work-a")
    _spend(_scope(data_root, "work-a", "work-a", "work-a"), 1.0)
    assert _raise(data_root, "work-a", 10.0, MIND)["unchanged"]
    assert _raise(data_root, "work-a", 10.0)["confirmed"]
    with usage_store.read(data_root) as view:
        stored = work_cap(view, "work-a")
    assert stored["basis"] == OWNER and stored["original_limit_usd"] == stored["limit_usd"] == 10.0
    assert _raise(data_root, "work-a", 10.0)["unchanged"]
    assert _raise(data_root, "work-a", 11.0, MIND)["error"] == "owner_cap_binds"
    with usage_store.read(data_root) as view:
        assert work_cap(view, "work-a") == stored


def test_money_writer_refuses_a_stale_expected_group(data_root):
    from ouroboros import usage_store
    from ouroboros.usage_admission import work_cap

    work = _work(data_root, "work-a")
    result = set_work_cap(data_root, work, "work-a", limit_usd=15, by=BY, basis=OWNER,
                          expected_group="earlier-a")
    assert result == {"ok": False, "error": "cost_cap_target_changed"}
    with usage_store.read(data_root) as view:
        assert work_cap(view, "work-a") is None


@pytest.mark.parametrize("raised,carried,source,explicit,expected", [
    (None, 10, "", 12, 10), (15, 10, "", 12, 12), (15, 10, "", 10, 10),
    (15, 10, "", 8, 8), (15, 10, "", 18, 15), (15, 10, "", None, 15),
    (15, 10, "producer", 12, 10), (15, 8, "", 12, 8), (15, 10, "", 0, 0),
    (15, None, "", 12, 12), (15, None, "", None, None),
])
def test_request_maximum_intersects_the_current_root_cap(raised, carried, source, explicit, expected):
    from ouroboros.usage_admission import _attempt_root_cap

    cap = {"original_limit_usd": 10, "limit_usd": raised} if raised is not None else None
    assert _attempt_root_cap(cap, carried, source, explicit) == expected


def test_reserved_request_reloads_its_own_maximum_beside_the_current_work_cap(data_root):
    _work(data_root, "work-a")
    scope = _scope(data_root, "work-a", "work-a", "work-a")
    _spend(scope, 9.0)
    request = ua.AttemptRequest(model="m", provider="openai", reservation_usd=1.0, root_limit_usd=12.0)
    with ua.usage_scope(scope):
        first, second = (ua.reserve_attempt(request) for _ in range(2))
    assert _raise(data_root, "work-a", 15.0)["ok"]
    _spend(scope, 2.0)
    # Dispatch reads the current rows, not a request or scope left in memory by reserve.
    first, second = replace(first, scope=None), replace(second, scope=None)
    ua.mark_dispatched(first)
    ua.settle_attempt(first, {"prompt_tokens": 1}, cost_usd=1.0, cost_final=True)
    with pytest.raises(ua.BudgetExceeded, match="root model budget changed before dispatch"):
        ua.mark_dispatched(second)
    with ua.usage_scope(scope), pytest.raises(ua.BudgetExceeded, match=r"limit=\$12\.000000"):
        ua.reserve_attempt(request)
    rows = [row for row in ledger_rows(data_root) if row["attempt_id"] in {first.attempt_id, second.attempt_id}]
    assert [(r["state"], r["root_limit_usd"], r["billing_group_limit_usd"], r["request_root_limit_usd"])
            for r in rows] == [("reserved", 10.0, 10.0, 12.0), ("settled", 10.0, 10.0, 12.0)]
    _spend(scope, 1.0)  # a different request still has the work's full room


def test_a_saved_reservation_and_a_running_helper_read_the_cap_at_their_own_admission(data_root):
    _work(data_root, "work-a")
    owner_scope = _scope(data_root, "work-a", "work-a", "work-a")
    helper = _scope(data_root, "helper-a", "work-a", "work-a", parent_task_id="work-a")  # born before the raise
    _spend(owner_scope, 8.0)
    with ua.usage_scope(owner_scope):
        saved = ua.reserve_attempt(ua.AttemptRequest(model="m", provider="openai", reservation_usd=1.0))
    _spend(helper, 2.0)  # known spend reaches the original $10 while the reservation waits
    with pytest.raises(ua.BudgetExceeded, match="budget changed before dispatch"):
        ua.mark_dispatched(saved)
    assert _raise(data_root, "work-a", 15.0)["ok"]
    ua.mark_dispatched(saved)  # the same reservation, read again under the cap at its dispatch, before any send
    ua.settle_attempt(saved, {"prompt_tokens": 1}, cost_usd=1.0, cost_final=True)
    _spend(helper, 3.0)  # the helper's scope still carries $10; the work's cap governs
    assert ua.usage_projection(data_root, billing_group_id="work-a")["settled_usd"] == pytest.approx(14.0)


def test_narrower_caps_not_the_works_own_and_the_shared_wallet_keep_binding(data_root, monkeypatch):
    from supervisor.budget_resume import _root_money_refusal

    work = _work(data_root, "work-a")
    scope = _scope(data_root, "work-a", "work-a", "work-a")
    _spend(scope, 6.0)
    # A narrower cap recorded on a saved request BEFORE the raise, its provenance unknown (no source: an older
    # request's own as much as a lowered setting): it stays at that request's dispatch.
    with ua.usage_scope(_scope(data_root, "helper-a", "work-a", "work-a", root_limit=6.5)):
        older = ua.reserve_attempt(ua.AttemptRequest(model="m", provider="openai", reservation_usd=0.25))
    _spend(scope, 1.0)  # $7 known: past the older request's own $6.5
    # Resume and unfenced review-wave reads use the work's own applicable caps ($10), not helper-a's $6.5 row.
    assert _root_money_refusal(data_root, work, "work-a") is None
    assert _raise(data_root, "work-a", 15.0)["ok"]
    assert _root_money_refusal(data_root, work, "work-a") is None
    wave = ua.review_wave_admission(data_root, root_task_id="work-a", task_id="work-a", models=[],
                                    session_slots=1, prompt_chars=0)
    assert wave["fits"] is True and wave["limit_usd"] == 15.0 and wave["remaining_usd"] == pytest.approx(8.0)
    with pytest.raises(ua.BudgetExceeded, match="root model budget changed before dispatch"):
        ua.mark_dispatched(older)
    with pytest.raises(ua.BudgetExceeded, match=r"limit=\$6\.900000"):  # a request's own explicit cap, kept
        _spend(scope, 0.25, request_root_limit=6.9)
    _spend(scope, 0.25)  # the work's own cap is raised
    monkeypatch.setenv("TOTAL_BUDGET", "7.25")
    with pytest.raises(ua.BudgetExceeded, match="global model budget exhausted") as refused:
        _spend(scope, 0.25)
    assert refused.value.limit_scope == "global"


def test_bare_request_maximum_does_not_become_a_works_persistent_resume_ceiling(data_root):
    from supervisor.budget_resume import _root_money_refusal

    work = _work(data_root, "work-req")
    # First row recorded without an active UsageScope carries an attempt-only maximum (attempt_request).
    held = ua.reserve_attempt(ua.AttemptRequest(model="openai/gpt-5.2", provider="openai", drive_root=data_root,
                                                task_id="helper-req", root_task_id="work-req",
                                                reservation_usd=6.0, root_limit_usd=6.5))
    ua.mark_dispatched(held)
    ua.settle_attempt(held, {"prompt_tokens": 1, "completion_tokens": 1}, cost_usd=6.0, cost_final=True)
    _spend(_scope(data_root, "work-req", "work-req", "work-req"), 1.0)  # $7 known: above the $6.5 request cap
    snap = task_money_snapshot(data_root, work, "work-req")
    assert snap["root_axis"]["limit_usd"] is None and snap["group_axis"]["limit_usd"] == 10.0
    assert snap["remaining_known_usd"] == pytest.approx(3.0)
    assert _root_money_refusal(data_root, work, "work-req") is None
    assert _raise(data_root, "work-req", 15.0)["ok"]
    assert _root_money_refusal(data_root, work, "work-req") is None
    wave = ua.review_wave_admission(data_root, root_task_id="work-req", task_id="work-req", models=[],
                                    session_slots=1, prompt_chars=0)
    assert wave["fits"] is True and wave["limit_usd"] == 15.0 and wave["remaining_usd"] == pytest.approx(8.0)


@pytest.mark.parametrize("session_first,group_limit,root_source", [
    (False, None, ""), (False, 10.0, ""), (True, None, ""), (False, None, "initial_task_admission"),
])
def test_explicitly_unlimited_root_does_not_inherit_later_request_minimum_in_unfenced_review(
    data_root, monkeypatch, session_first, group_limit, root_source,
):
    from supervisor.budget_resume import _root_money_refusal

    work_id = f"open-{int(session_first)}-{group_limit or 'none'}-{root_source or 'unsourced'}"
    work = _work(data_root, work_id, limit=group_limit)
    unlimited = _scope(data_root, work_id, work_id, work_id, root_limit=None, group_limit=group_limit,
                       root_limit_source=root_source)
    if session_first:
        with ua.usage_scope(unlimited):
            ua.record_subscription_session(
                model="claudexor::codex=gpt-6-astra", route="codex", session_id=f"sess-{work_id}",
                spend_usd=0.0, task_id=work_id, root_task_id=work_id, drive_root=data_root)
    _spend(unlimited, 2.0)  # First root binding explicitly records root_limit_usd=None.
    helper_scope = _scope(data_root, f"{work_id}-helper", work_id, work_id, root_limit=6.5, group_limit=group_limit)
    with ua.usage_scope(helper_scope):
        held = ua.reserve_attempt(ua.AttemptRequest(model="openai/gpt-5.2", provider="openai", reservation_usd=0.25))
    if root_source:
        # Even a later sourced request on the same root never replaces a provenanced unlimited root binding.
        _spend(_scope(data_root, work_id, work_id, work_id, root_limit=6.5, group_limit=group_limit,
                      root_limit_source="test"), 1.0)
        _spend(unlimited, 4.0)
    else:
        _spend(unlimited, 5.0, request_root_limit=6.5)  # $7.0 known: past $6.5, still within any $10 group.

    assert _root_money_refusal(data_root, work, work_id) is None
    wave = ua.review_wave_admission(data_root, root_task_id=work_id, task_id=work_id, models=[],
                                    session_slots=1, prompt_chars=0)
    assert wave["fits"] is True and wave["limit_usd"] == group_limit
    if group_limit is None:
        assert wave["binding_axis"] == "global" and wave["remaining_usd"] == pytest.approx(993.0)
        monkeypatch.setenv("TOTAL_BUDGET", "0")
        unbounded = ua.review_wave_admission(data_root, root_task_id=work_id, task_id=work_id, models=[],
                                             session_slots=1, prompt_chars=0)
        assert unbounded["fits"] is True and unbounded["limit_usd"] is None and unbounded["binding_axis"] is None
    else:
        assert wave["binding_axis"] == "group" and wave["remaining_usd"] == pytest.approx(3.0)
    # The individual helper/request cap stays effective on its own request and its own fenced review.
    with pytest.raises(ua.BudgetExceeded, match="root model budget changed before dispatch"):
        ua.mark_dispatched(held)
    with pytest.raises(ua.BudgetExceeded, match=r"limit=\$6\.500000"):
        _spend(unlimited, 0.25, request_root_limit=6.5)
    fenced = ua.review_wave_admission(data_root, root_task_id=work_id, task_id=f"{work_id}-helper", models=[],
                                      session_slots=1, prompt_chars=0, root_limit_usd=6.5)
    assert fenced["fits"] is False and fenced["binding_axis"] == "root" and fenced["limit_usd"] == 6.5


def _trace_usage_reads(monkeypatch, fn):
    import contextlib
    from ouroboros import usage_store

    statements = []
    real_read = usage_store.read

    @contextlib.contextmanager
    def traced(root, *a, **k):
        with real_read(root, *a, **k) as txn:
            txn.conn.set_trace_callback(statements.append)
            try:
                yield txn
            finally:
                txn.conn.set_trace_callback(None)

    with monkeypatch.context() as ctx:
        ctx.setattr(usage_store, "read", traced)
        return fn(), statements


def test_addressed_resume_and_review_agree_without_scanning_attempts(data_root, monkeypatch):
    """Both Resume and review wave resolve root caps from addressed bindings only (zero ``attempts`` reads):
    a sourced root cap after a group-only session binds identically at both readers, provenanced unlimited
    and helper-only rows never invent a ceiling, and a caller's own sourced limit skips extra lookups."""
    from ouroboros import usage_store
    from supervisor.budget_resume import _root_money_refusal

    trace_sql = lambda fn: _trace_usage_reads(monkeypatch, fn)  # noqa: E731

    # 1. Group-only subscription session first, then root's own sourced $1 cap -> Resume & review agree.
    work_capped = _work(data_root, "sess-capped", limit=100.0)
    base_capped = _scope(data_root, "sess-capped", "sess-capped", "sess-capped", root_limit=None, group_limit=100.0)
    with ua.usage_scope(base_capped):
        ua.record_subscription_session(
            model="claudexor::codex=gpt-6-astra", route="codex", session_id="sess-capped-1",
            spend_usd=0.0, task_id="sess-capped", root_task_id="sess-capped", drive_root=data_root)
    _spend(_scope(data_root, "sess-capped", "sess-capped", "sess-capped",
                  root_limit=1.0, group_limit=100.0, root_limit_source="test"), 1.0)

    refusal, resume_sql = trace_sql(lambda: _root_money_refusal(data_root, work_capped, "sess-capped"))
    wave, wave_sql = trace_sql(lambda: ua.review_wave_admission(
        data_root, root_task_id="sess-capped", task_id="sess-capped", models=[], session_slots=1, prompt_chars=0))
    assert refusal == {"ok": False, "error": "independent_cap_exhausted", "binding_axis": "root",
                       "binding_source": "test", "action": "resolve_limit_source_or_cancel"}
    assert (wave["fits"], wave["binding_axis"], wave["binding_source"], wave["limit_usd"]) == (False, "root", "test", 1.0)
    assert not any("FROM attempts" in sql for sql in (*resume_sql, *wave_sql))

    # Caller with an independently confirmed current limit pays neither an attempts scan nor a root_cap lookup.
    authored, authored_sql = trace_sql(lambda: ua.review_wave_admission(
        data_root, root_task_id="sess-capped", task_id="sess-capped", models=[], session_slots=1, prompt_chars=0,
        root_limit_usd=5.0, root_limit_source="authored_root_ceiling"))
    assert authored["fits"] is True and authored["limit_usd"] == 5.0 and authored["binding_source"] == "authored_root_ceiling"
    assert not any("FROM attempts" in sql for sql in authored_sql)
    assert not any("root_cap" in sql for sql in authored_sql)

    # 2. Group-only session, then provenanced unlimited, then later sourced $1 cap -> stays unlimited at both.
    work_unlim = _work(data_root, "sess-unlim", limit=100.0)
    with ua.usage_scope(_scope(data_root, "sess-unlim", "sess-unlim", "sess-unlim", root_limit=None, group_limit=100.0)):
        ua.record_subscription_session(
            model="claudexor::codex=gpt-6-astra", route="codex", session_id="sess-unlim-1",
            spend_usd=0.0, task_id="sess-unlim", root_task_id="sess-unlim", drive_root=data_root)
    _spend(_scope(data_root, "sess-unlim", "sess-unlim", "sess-unlim",
                  root_limit=None, group_limit=100.0, root_limit_source="initial_task_admission"), 0.5)
    _spend(_scope(data_root, "sess-unlim", "sess-unlim", "sess-unlim",
                  root_limit=1.0, group_limit=100.0, root_limit_source="test"), 1.0)
    assert _root_money_refusal(data_root, work_unlim, "sess-unlim") is None
    unlim_wave, unlim_sql = trace_sql(lambda: ua.review_wave_admission(
        data_root, root_task_id="sess-unlim", task_id="sess-unlim", models=[], session_slots=1, prompt_chars=0))
    assert (unlim_wave["fits"], unlim_wave["binding_axis"], unlim_wave["limit_usd"]) == (True, "group", 100.0)
    assert not any("FROM attempts" in sql for sql in unlim_sql)

    # 3. Group-only session, then helper-only (even sourced) & bare attempt_request -> never binds the root.
    work_helper = _work(data_root, "sess-helper", limit=100.0)
    with ua.usage_scope(_scope(data_root, "sess-helper", "sess-helper", "sess-helper", root_limit=None, group_limit=100.0)):
        ua.record_subscription_session(
            model="claudexor::codex=gpt-6-astra", route="codex", session_id="sess-helper-1",
            spend_usd=0.0, task_id="sess-helper", root_task_id="sess-helper", drive_root=data_root)
    _spend(_scope(data_root, "sess-helper-sub", "sess-helper", "sess-helper",
                  root_limit=1.0, group_limit=100.0, root_limit_source="producer"), 1.0)
    bare = ua.reserve_attempt(ua.AttemptRequest(model="openai/gpt-5.2", provider="openai", drive_root=data_root,
                                                task_id="sess-helper", root_task_id="sess-helper",
                                                reservation_usd=1.0, root_limit_usd=1.5))
    ua.mark_dispatched(bare)
    ua.settle_attempt(bare, {"prompt_tokens": 1}, cost_usd=1.0, cost_final=True)
    assert _root_money_refusal(data_root, work_helper, "sess-helper") is None
    helper_wave, helper_sql = trace_sql(lambda: ua.review_wave_admission(
        data_root, root_task_id="sess-helper", task_id="sess-helper", models=[], session_slots=1, prompt_chars=0))
    assert (helper_wave["fits"], helper_wave["binding_axis"], helper_wave["limit_usd"]) == (True, "group", 100.0)
    assert not any("FROM attempts" in sql for sql in helper_sql)

    # 4. Export -> re-import preserves the materialized root_cap fact identically.
    assert usage_store.export_journal(data_root)["status"] == "exported"
    assert _root_money_refusal(data_root, work_capped, "sess-capped") == refusal
    re_wave = ua.review_wave_admission(data_root, root_task_id="sess-capped", task_id="sess-capped",
                                       models=[], session_slots=1, prompt_chars=0)
    assert (re_wave["fits"], re_wave["binding_axis"], re_wave["binding_source"], re_wave["limit_usd"]) == (
        False, "root", "test", 1.0)
    assert _root_money_refusal(data_root, work_unlim, "sess-unlim") is None
    assert _root_money_refusal(data_root, work_helper, "sess-helper") is None


def _seed_prechange_root_cap_db(data_root):
    from ouroboros import usage_store

    def session(work_id):
        with ua.usage_scope(_scope(data_root, work_id, work_id, work_id, root_limit=None, group_limit=100.0)):
            ua.record_subscription_session(
                model="claudexor::codex=gpt-6-astra", route="codex", session_id=f"pre-{work_id}-1",
                spend_usd=0.0, task_id=work_id, root_task_id=work_id, drive_root=data_root)

    works = {name: _work(data_root, name, limit=10.0 if name == "pre-neighbor" else 100.0) for name in (
        "pre-capped", "pre-neighbor", "pre-unlim", "pre-absent", "pre-sess-only", "pre-ambig")}
    for name in ("pre-capped", "pre-unlim", "pre-absent", "pre-sess-only", "pre-ambig"):
        session(name)
    _spend(_scope(data_root, "pre-capped", "pre-capped", "pre-capped",
                  root_limit=1.0, group_limit=100.0, root_limit_source="test"), 1.0)
    neighbor_scope = _scope(data_root, "pre-neighbor", "pre-neighbor", "pre-neighbor", root_limit=10.0, group_limit=10.0)
    _spend(neighbor_scope, 2.5)
    with ua.usage_scope(neighbor_scope):
        unknown_held = ua.reserve_attempt(ua.AttemptRequest(model="unknown/unpriced", provider="openai",
                                                            drive_root=data_root, force_unknown_reservation=True))
    ua.mark_dispatched(unknown_held)
    ua.settle_attempt(unknown_held, {}, cost_usd=None, cost_final=False)
    _spend(_scope(data_root, "pre-unlim", "pre-unlim", "pre-unlim",
                  root_limit=None, group_limit=100.0, root_limit_source="initial_task_admission"), 0.5)
    _spend(_scope(data_root, "pre-unlim", "pre-unlim", "pre-unlim",
                  root_limit=1.0, group_limit=100.0, root_limit_source="test"), 1.0)
    _spend(_scope(data_root, "pre-absent-sub", "pre-absent", "pre-absent",
                  root_limit=1.0, group_limit=100.0, root_limit_source="producer"), 1.0)
    bare = ua.reserve_attempt(ua.AttemptRequest(model="openai/gpt-5.2", provider="openai", drive_root=data_root,
                                                task_id="pre-absent", root_task_id="pre-absent",
                                                reservation_usd=1.0, root_limit_usd=1.5))
    ua.mark_dispatched(bare)
    ua.settle_attempt(bare, {"prompt_tokens": 1}, cost_usd=1.0, cost_final=True)
    with ua.usage_scope(_scope(data_root, "pre-absent", "pre-absent", "pre-absent", root_limit=None, group_limit=100.0)):
        req_held = ua.reserve_attempt(ua.AttemptRequest(model="openai/gpt-5.2", provider="openai",
                                                        drive_root=data_root, reservation_usd=0.25, root_limit_usd=2.25))
    _spend(_scope(data_root, "pre-absent", "pre-absent", "pre-absent", root_limit=None, group_limit=100.0),
           0.5, request_root_limit=2.5)
    with ua.usage_scope(_scope(data_root, "pre-ambig", "pre-ambig", "pre-ambig", root_limit=1.0, group_limit=100.0)):
        ambig_held = ua.reserve_attempt(ua.AttemptRequest(model="openai/gpt-5.2", provider="openai",
                                                          drive_root=data_root, reservation_usd=0.5))
    ua.mark_dispatched(ambig_held)
    ua.settle_attempt(ambig_held, {"prompt_tokens": 1}, cost_usd=0.5, cost_final=True)
    _spend(_scope(data_root, "pre-ambig", "pre-ambig", "pre-ambig",
                  root_limit=5.0, group_limit=100.0, root_limit_source="later"), 0.25)
    with usage_store.hold(data_root) as txn:
        txn.conn.execute("DELETE FROM bindings WHERE scope='root_cap'")
        meta = dict(txn.meta("import"))
        meta.pop("root_caps", None)
        txn.set_meta("import", meta)
        extra = json.loads(txn.conn.execute("SELECT extra FROM attempts WHERE attempt_id=?",
                                            (ambig_held.attempt_id,)).fetchone()["extra"])
        for key in ("request_root_limit_usd", "root_limit_source"):
            extra.pop(key, None)
        if "__none__" in extra:
            extra["__none__"] = [k for k in extra["__none__"] if k != "request_root_limit_usd"]
        txn.conn.execute("UPDATE attempts SET extra=? WHERE attempt_id=?",
                         (json.dumps(extra, sort_keys=True, separators=(",", ":")), ambig_held.attempt_id))
    return works, req_held


def test_prechange_sqlite_startup_backfills_root_cap_safely(data_root, monkeypatch):
    """Existing pre-change SQLite stores recover ``('root_cap', root)`` once at startup without widening
    ambiguous older roots, without preemption by later writes, and without scanning ``attempts`` on reads."""
    from ouroboros import usage_store
    from supervisor.budget_resume import _root_money_refusal

    trace_sql = lambda fn: _trace_usage_reads(monkeypatch, fn)  # noqa: E731
    works, req_held = _seed_prechange_root_cap_db(data_root)
    wave_of = lambda root_id, **kw: ua.review_wave_admission(  # noqa: E731
        data_root, root_task_id=root_id, task_id=root_id, models=[], session_slots=1, prompt_chars=0, **kw)

    # Before startup upgrade: missing evidence holds pre-capped without scanning attempts; known-good neighbors
    # and confirmed caller limits keep serving; a later write cannot preempt the earlier $1 "test" source.
    pre_refusal, pre_r_sql = trace_sql(lambda: _root_money_refusal(data_root, works["pre-capped"], "pre-capped"))
    pre_wave, pre_w_sql = trace_sql(lambda: wave_of("pre-capped"))
    assert (pre_refusal["error"], pre_refusal["cause"]) == ("billing_authority_unavailable", "root_cap_backfill_pending")
    assert (pre_wave["fits"], pre_wave["reason"], pre_wave["billing_authority"]) == (
        False, "billing_authority_unavailable", "unreadable")
    assert not any("FROM attempts" in sql for sql in (*pre_r_sql, *pre_w_sql))
    assert _root_money_refusal(data_root, works["pre-neighbor"], "pre-neighbor") is None
    assert _root_money_refusal(data_root, works["pre-sess-only"], "pre-sess-only") is None
    pre_auth, pre_a_sql = trace_sql(lambda: wave_of("pre-capped", root_limit_usd=5.0,
                                                    root_limit_source="authored_root_ceiling"))
    assert pre_auth["fits"] is True and pre_auth["limit_usd"] == 5.0
    assert not any("FROM attempts" in sql or "root_cap" in sql for sql in pre_a_sql)

    _spend(_scope(data_root, "pre-capped", "pre-capped", "pre-capped",
                  root_limit=None, group_limit=100.0, root_limit_source="initial_task_admission"), 0.25)
    with usage_store.read(data_root) as txn:
        assert txn.bindings("root_cap") == {}
        before_summaries = [tuple(r) for r in txn.conn.execute("SELECT * FROM summaries ORDER BY scope, key")]
        before_roots, before_groups = txn.bindings("root"), txn.bindings("group")
    before_nb_proj = ua.usage_projection(data_root, root_task_id="pre-neighbor")
    assert (before_nb_proj["settled_usd"], before_nb_proj["unknown_unmetered"], before_nb_proj["limit_usd"]) == (2.5, 1, 10.0)

    # Supported once-only startup migration recovers confirmed facts and marks ambiguous older roots idempotently.
    report = usage_store.migrate_from_journal(data_root, initialize_journal=True)
    assert report["status"] == "already_completed"
    assert (report["root_caps_backfilled"], report["root_caps_ambiguous"]) == (2, 1)
    usage_store.forget(data_root)
    upgraded_bytes = (data_root / usage_store.STORE_REL).read_bytes()
    again = usage_store.migrate_from_journal(data_root, initialize_journal=True)
    assert again["status"] == "already_completed" and "root_caps_backfilled" not in again
    usage_store.forget(data_root)
    assert (data_root / usage_store.STORE_REL).read_bytes() == upgraded_bytes

    with usage_store.read(data_root) as txn:
        assert [tuple(r) for r in txn.conn.execute("SELECT * FROM summaries ORDER BY scope, key")] == before_summaries
        assert txn.bindings("root") == before_roots and txn.bindings("group") == before_groups
        assert txn.meta("import")["root_caps"] == {"status": "completed", "backfilled": 2, "ambiguous": 1}
        assert all(txn.binding("root_cap", k) is None for k in ("pre-absent", "pre-sess-only", "pre-neighbor"))
    assert ua.usage_projection(data_root, root_task_id="pre-neighbor") == before_nb_proj

    # Repeated cheap addressed reads agree with zero attempts scans; earlier $1 "test" survived the later write.
    for _ in range(2):
        refusal, r_sql = trace_sql(lambda: _root_money_refusal(data_root, works["pre-capped"], "pre-capped"))
        wave, w_sql = trace_sql(lambda: wave_of("pre-capped"))
        authored, a_sql = trace_sql(lambda: wave_of("pre-capped", root_limit_usd=5.0,
                                                    root_limit_source="authored_root_ceiling"))
        assert refusal == {"ok": False, "error": "independent_cap_exhausted", "binding_axis": "root",
                           "binding_source": "test", "action": "resolve_limit_source_or_cancel"}
        assert (wave["fits"], wave["binding_axis"], wave["binding_source"], wave["limit_usd"]) == (
            False, "root", "test", 1.0)
        assert (authored["fits"], authored["binding_axis"], authored["binding_source"], authored["limit_usd"]) == (
            True, "root", "authored_root_ceiling", 5.0)
        assert not any("FROM attempts" in sql for sql in (*r_sql, *w_sql, *a_sql))
        assert not any("root_cap" in sql for sql in a_sql)

    # Unaffected neighbor, proven unlimited, accounted absence + one-request max, and ambiguous older root.
    nb_wave, nb_sql = trace_sql(lambda: wave_of("pre-neighbor"))
    assert _root_money_refusal(data_root, works["pre-neighbor"], "pre-neighbor") is None
    assert (nb_wave["fits"], nb_wave["limit_usd"], nb_wave["remaining_usd"]) == (True, 10.0, pytest.approx(7.5))
    assert not any("FROM attempts" in sql or "root_cap" in sql for sql in nb_sql)

    unlim_snap = task_money_snapshot(data_root, works["pre-unlim"], "pre-unlim")
    assert unlim_snap["root_axis"]["limit_usd"] is None and unlim_snap["root_axis"]["source"] == "initial_task_admission"
    unlim_wave, unlim_sql = trace_sql(lambda: wave_of("pre-unlim"))
    assert _root_money_refusal(data_root, works["pre-unlim"], "pre-unlim") is None
    assert (unlim_wave["fits"], unlim_wave["binding_axis"], unlim_wave["limit_usd"]) == (True, "group", 100.0)
    assert not any("FROM attempts" in sql for sql in unlim_sql)

    absent_snap = task_money_snapshot(data_root, works["pre-absent"], "pre-absent")
    assert absent_snap["root_axis"]["limit_usd"] is None and absent_snap["root_axis"]["source"] is None
    absent_wave, absent_sql = trace_sql(lambda: wave_of("pre-absent"))
    assert _root_money_refusal(data_root, works["pre-absent"], "pre-absent") is None
    assert (absent_wave["fits"], absent_wave["binding_axis"], absent_wave["limit_usd"]) == (True, "group", 100.0)
    assert not any("FROM attempts" in sql for sql in absent_sql)
    with pytest.raises(ua.BudgetExceeded, match="root model budget changed before dispatch"):
        ua.mark_dispatched(req_held)
    with pytest.raises(ua.BudgetExceeded, match=r"limit=\$2\.500000"):
        _spend(_scope(data_root, "pre-absent", "pre-absent", "pre-absent", root_limit=None, group_limit=100.0),
               0.1, request_root_limit=2.5)

    ambig_ref, ambig_r_sql = trace_sql(lambda: _root_money_refusal(data_root, works["pre-ambig"], "pre-ambig"))
    ambig_wave, ambig_w_sql = trace_sql(lambda: wave_of("pre-ambig"))
    assert (ambig_ref["error"], ambig_ref["cause"]) == ("billing_authority_unavailable", "root_cap_source_ambiguous")
    assert (ambig_wave["fits"], ambig_wave["reason"], ambig_wave["billing_authority"]) == (
        False, "billing_authority_unavailable", "unreadable")
    assert not any("FROM attempts" in sql for sql in (*ambig_r_sql, *ambig_w_sql))
    assert wave_of("pre-ambig", root_limit_usd=5.0, root_limit_source="authored_root_ceiling")["fits"] is True

    # Export -> re-import preserves confirmed, proven-unlimited, accounted-absence and ambiguous root_cap facts.
    assert usage_store.export_journal(data_root)["status"] == "exported"
    assert _root_money_refusal(data_root, works["pre-capped"], "pre-capped") == refusal
    assert _root_money_refusal(data_root, works["pre-unlim"], "pre-unlim") is None
    assert _root_money_refusal(data_root, works["pre-absent"], "pre-absent") is None
    assert _root_money_refusal(data_root, works["pre-ambig"], "pre-ambig")["cause"] == "root_cap_source_ambiguous"


def test_commit_gate_review_refuses_unverified_root_cap_authority_without_dispatch(data_root, tmp_path, monkeypatch):
    from ouroboros import review_continuation, usage_store
    from ouroboros.tools import parallel_review, review
    from tests.test_commit_gate_wave_admission import BOUNDS, COUPLING_MODEL, PACKET_MODELS, LedgerLLM, _prepared

    _seed_prechange_root_cap_db(data_root)
    monkeypatch.setattr(ua, "_reservation_cost", lambda req: BOUNDS[req.model])
    monkeypatch.setattr(parallel_review, "run_cmd", lambda *_a, **_k: "staged diff")
    monkeypatch.setattr(review, "_prepare_unified_review", lambda *_a, **_k: (_prepared(tmp_path), None, False))

    def run_gate(root_id, *, root_limit=None, root_source="", group_limit=100.0):
        llm = LedgerLLM()
        monkeypatch.setattr(review, "LLMClient", lambda: llm)
        ctx = SimpleNamespace(
            repo_dir=tmp_path, drive_root=data_root, task_id=root_id,
            task_metadata={"root_task_id": root_id, "budget_drive_root": str(data_root)}, pending_events=[],
            _review_history=[], _review_advisory=[], _coupling_review_history={}, _review_iteration_count=0,
            _last_review_critical_findings=[], _review_degraded_reasons=[])
        with ua.usage_scope(_scope(data_root, root_id, root_id, root_id, root_limit=root_limit,
                                   group_limit=group_limit, root_limit_source=root_source)):
            return ctx, llm, parallel_review.run_parallel_review(ctx, "prechange review gate")

    ctx_pre, llm_pre, (err_pre, coup_pre, reason_pre, _) = run_gate("pre-capped")
    assert (err_pre, llm_pre.calls, coup_pre.status, reason_pre) == (None, [], "not_dispatched", "review_wave_billing_authority_unavailable")
    ev_pre = [e for e in ctx_pre.pending_events if e.get("type") == "review_wave_billing_authority_unavailable"]
    assert len(ev_pre) == 1 and ev_pre[0]["cause"] == "root_cap_backfill_pending" and ev_pre[0]["limit_usd"] is None
    assert not [e for e in ctx_pre.pending_events if e.get("type") == "review_wave_budget_insufficient"]
    adv_pre = "\n".join(map(str, ctx_pre._review_advisory))
    assert "root_cap_backfill_pending" in adv_pre and "Known spend has reached" not in adv_pre

    _, llm_auth, (err_a, coup_a, _, _) = run_gate("pre-capped", root_limit=5.0, root_source="authored_root_ceiling")
    _, llm_nb, (err_n, coup_n, _, _) = run_gate("pre-neighbor", root_limit=10.0, group_limit=10.0)
    want = sorted([COUPLING_MODEL, *PACKET_MODELS])
    assert err_a is None and coup_a.status == "responded" and sorted(llm_auth.calls) == want
    assert err_n is None and coup_n.status == "responded" and sorted(llm_nb.calls) == want

    assert usage_store.migrate_from_journal(data_root, initialize_journal=True)["root_caps_ambiguous"] == 1
    ctx_amb, llm_amb, (err_amb, coup_amb, reason_amb, _) = run_gate("pre-ambig")
    assert (err_amb, llm_amb.calls, coup_amb.status, reason_amb) == (None, [], "not_dispatched", "review_wave_billing_authority_unavailable")
    ev_amb = [e for e in ctx_amb.pending_events if e.get("type") == "review_wave_billing_authority_unavailable"]
    assert len(ev_amb) == 1 and ev_amb[0]["cause"] == "root_cap_source_ambiguous" and ev_amb[0]["limit_usd"] is None
    assert "root_cap_source_ambiguous" in "\n".join(map(str, ctx_amb._review_advisory))
    s_amb = SimpleNamespace(task_id="pre-ambig", root_task_id="pre-ambig", drive_root=data_root, pending_events=[])
    slot = [("slot_1", {"surface": "commit_gate", "task_id": "pre-ambig", "root_task_id": "pre-ambig"}, "triad/a", 1000, 100)]
    with ua.usage_scope(_scope(data_root, "pre-ambig", "pre-ambig", "pre-ambig", root_limit=None, group_limit=100.0)):
        with pytest.raises(review_continuation.ContinuationRefused) as refused:
            review_continuation._admit_new_generation(s_amb, data_root, s_amb, slot, [])
    assert refused.value.reason == "billing_authority_unavailable" and "root_cap_source_ambiguous" in str(refused.value)
    assert "known spend has reached" not in str(refused.value).lower()


def test_a_live_call_check_reads_the_cap_and_a_damaged_record_never_widens_it(data_root):
    from ouroboros import usage_store
    from ouroboros.live_usage import check_live_budget

    _work(data_root, "work-a")
    scope = _scope(data_root, "work-a", "work-a", "work-a")
    _spend(scope, 10.0)
    with ua.usage_scope(scope):
        with pytest.raises(ua.BudgetExceeded):
            check_live_budget(engine="openai", model="m", task_id="work-a", drive_root=data_root)
        assert _raise(data_root, "work-a", 15.0)["ok"]
        check_live_budget(engine="openai", model="m", task_id="work-a", drive_root=data_root)
        open_row = ua.reserve_attempt(ua.AttemptRequest(model="m", provider="openai", reservation_usd=1.0))
        ua.mark_dispatched(open_row)
    with usage_store.hold(data_root) as txn:  # damage: a record no writer of this version produces
        txn.conn.execute("UPDATE bindings SET binding=? WHERE scope='work_cap' AND key='work-a'",
                         (json.dumps({"limit_usd": "a lot"}),))
    with pytest.raises(ua.BillingAuthorityUnavailable, match="malformed_work_cap"):
        _spend(scope, 0.5)  # an unreadable cap admits nothing new
    ua.settle_attempt(open_row, {"prompt_tokens": 1}, cost_usd=1.0, cost_final=True)  # the receipt still lands
    assert ua.usage_projection(data_root, billing_group_id="work-a")["limit_usd"] == 10.0  # the recorded cap shows


def test_review_wave_resume_and_continue_money_reads_see_the_cap(data_root):
    from supervisor.budget_resume import _root_money_refusal
    from supervisor.continuation_admission import _group_money_spent

    work = _work(data_root, "work-a")
    scope = _scope(data_root, "work-a", "work-a", "work-a")
    _spend(scope, 10.0)
    queue = SimpleNamespace(DRIVE_ROOT=data_root)
    refusal = _root_money_refusal(data_root, work, "work-a")
    assert refusal["error"] == "root_hard_cap_exhausted"  # the work's own cap: the honest remedy is its raise
    # Equal ordinary root/group caps bind on the work's group axis, matching snapshot and wave admission.
    assert refusal["binding_axis"] == "group" and refusal["action"] == "raise_work_cap_then_resume"
    assert _group_money_spent(queue, _binding("work-a")) is True
    assert _raise(data_root, "work-a", 15.0)["ok"]
    assert _root_money_refusal(data_root, work, "work-a") is None  # Resume may proceed; nothing resumed it here
    assert _group_money_spent(queue, _binding("work-a")) is False
    with ua.usage_scope(scope):
        wave = ua.review_wave_admission(data_root, root_task_id="work-a", models=["openai/gpt-5.2"],
                                        prompt_chars=4000, max_completion_tokens=1000, task_id="work-a",
                                        root_limit_usd=10.0)
    assert wave["fits"] is True and wave["remaining_usd"] == pytest.approx(5.0) and wave["limit_usd"] == 15.0


def test_resume_refusal_names_an_independent_fence_not_the_work_cap(data_root, monkeypatch):
    """A producer's own root fence is not the work's conversational cap: the refusal says so, and
    raising the ordinary whole cap — which succeeds — does not lift it."""
    from ouroboros.consciousness_view import _resume_room
    from supervisor.budget_resume import _root_money_refusal

    work = _work(data_root, "work-a")
    scope = _scope(data_root, "work-a", "work-a", "work-a",
                   root_limit_source="producer", root_limit_revision="first")
    _spend(scope, 10.0)
    refusal = _root_money_refusal(data_root, work, "work-a")
    assert refusal["error"] == "independent_cap_exhausted" and refusal["binding_axis"] == "root"
    assert refusal["binding_source"] == "producer"
    assert refusal["action"] == "resolve_limit_source_or_cancel"  # never the conversational raise
    assert _raise(data_root, "work-a", 12.0)["ok"]  # the ordinary whole cap rises...
    still = _root_money_refusal(data_root, work, "work-a")  # ...and the independent fence still binds
    assert still["error"] == "independent_cap_exhausted" and still["binding_source"] == "producer"
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_DAILY_USD", "25")
    assert _resume_room(data_root, "work-a", {}, {}) == {"resume_additional_usd": 0.0}


@pytest.mark.parametrize("source", [None, "producer"])
def test_resume_sees_the_exhausted_successor_root_even_when_the_group_is_tighter(data_root, source):
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.review_helpers import review_wave_binding_fence
    from supervisor.budget_resume import _root_money_refusal

    _work(data_root, "origin")
    _spend(_scope(data_root, "origin", "origin", "origin"), 8.0)
    write_task_result(data_root, "successor", "running", root_task_id="successor", billing_group=_binding("origin"))
    task = {"id": "successor", "root_task_id": "successor"}
    scope = _scope(data_root, "successor", "successor", "origin", root_limit=5.0, root_limit_source=source)
    _spend(scope, 5.0)

    assert task_money_snapshot(data_root, task, "successor")["binding_axis"] == "group"
    refused = _root_money_refusal(data_root, task, "successor")
    assert (refused["error"], refused["binding_axis"], refused["action"]) == (
        "independent_cap_exhausted", "root", "resolve_limit_source_or_cancel")
    assert refused.get("binding_source") == source
    with ua.usage_scope(scope):
        wave = ua.review_wave_admission(data_root, root_task_id="successor", task_id="successor",
                                        models=[], session_slots=1, prompt_chars=0, root_limit_usd=5.0,
                                        root_limit_source=source)
    assert (wave["fits"], wave["binding_axis"], wave["limit_usd"]) == (False, "root", 5.0)
    assert "resolve that limit at its source" in review_wave_binding_fence(wave)[1]


def test_resume_names_an_independent_group_tied_with_an_ordinary_root(data_root):
    from ouroboros.task_results import write_task_result
    from supervisor.budget_resume import _root_money_refusal

    group = {**_binding("work-a", 5.0), "billing_group_limit_source": "review_envelope"}
    write_task_result(data_root, "work-a", "running", root_task_id="work-a", billing_group=group)
    task = {"id": "work-a", "root_task_id": "work-a"}
    scope = replace(_scope(data_root, "work-a", "work-a", "work-a", root_limit=5.0, group_limit=5.0),
                    billing_group_limit_source="review_envelope")
    _spend(scope, 5.0)

    assert task_money_snapshot(data_root, task, "work-a")["binding_axis"] == "root"
    assert _root_money_refusal(data_root, task, "work-a") == {
        "ok": False, "error": "independent_cap_exhausted", "binding_axis": "group",
        "binding_source": "review_envelope", "action": "resolve_limit_source_or_cancel"}


def test_finite_local_cap_in_originally_unlimited_group_is_reported_as_independent(data_root):
    from ouroboros.tools.review_helpers import review_wave_binding_fence
    from supervisor.budget_resume import _root_money_refusal

    work = _work(data_root, "open-local", limit=None)
    scope = _scope(data_root, "open-local", "open-local", "open-local", root_limit=5.0, group_limit=None)
    _spend(scope, 5.0)
    assert _raise(data_root, "open-local", 10.0)["error"] == "work_cap_unlimited"
    snap = task_money_snapshot(data_root, work, "open-local")
    assert snap["work_cap_block"] == {"error": "independent_cap_exhausted", "axis": "root"}
    refusal = _root_money_refusal(data_root, work, "open-local")
    assert refusal == {"ok": False, "error": "independent_cap_exhausted", "binding_axis": "root",
                       "action": "resolve_limit_source_or_cancel"}
    with ua.usage_scope(scope):
        wave = ua.review_wave_admission(data_root, root_task_id="open-local", task_id="open-local",
                                        models=[], session_slots=1, prompt_chars=0, root_limit_usd=5.0)
    assert (wave["fits"], wave["binding_axis"], wave["binding_can_use_work_cap"]) == (False, "root", False)
    assert "resolve that limit at its source" in review_wave_binding_fence(wave)[1]


def test_work_cap_precision_survives_into_resume_and_authored_stop(data_root):
    from ouroboros.loop_budget import authored_completion_budget_exhausted
    from supervisor.budget_resume import _root_money_refusal

    task = _work(data_root, "work-a")
    scope = _scope(data_root, "work-a", "work-a", "work-a")
    _spend(scope, 10.0)
    raised = 10.0000004
    receipt = _raise(data_root, "work-a", raised)
    assert receipt["ok"] and receipt["remaining_known_usd"] == 0.0000004

    assert task_money_snapshot(data_root, task, "work-a")["root_limit_usd"] == raised
    assert task_money_snapshot(data_root, task, "work-a")["remaining_known_usd"] == 0.0000004
    assert ua.usage_projection(data_root, root_task_id="work-a")["limit_usd"] == raised
    assert _root_money_refusal(data_root, task, "work-a") is None
    with ua.usage_scope(scope):
        wave = ua.review_wave_admission(data_root, root_task_id="work-a", task_id="work-a", models=[],
                                        session_slots=1, prompt_chars=0, root_limit_usd=10.0)
    assert wave["fits"] is True and wave["remaining_usd"] == 0.0000004
    ctx = SimpleNamespace(active_use_local=True, active_model="local-fixture", accumulated_usage={"cost": 10.0})
    with ua.usage_scope(scope):
        assert not authored_completion_budget_exhausted(ctx, 100.0, SimpleNamespace(root_cap_usd=10.0))


def test_the_downgrade_export_carries_the_cap_separately_and_preserves_costs(data_root):
    from ouroboros import usage_journal, usage_store
    from ouroboros import usage_ledger as ledger
    from tests import _legacy_usage_validator as legacy_validator

    _work(data_root, "work-a")
    _spend(_scope(data_root, "work-a", "work-a", "work-a"), 10.0)
    _spend(_scope(data_root, "other-a", "other-a", "other-a"), 1.0)  # the journal's last rows: another work's
    assert _raise(data_root, "work-a", 15.0)["ok"]
    with usage_store.read(data_root) as view:
        before = view.binding("work_cap", "work-a")
    assert usage_store.export_journal(data_root)["status"] == "exported"  # before any further spend
    rows = [json.loads(line, parse_float=legacy_validator.LiteralFloat)
            for line in (data_root / ledger.LEDGER_REL).read_text(encoding="utf-8").splitlines()]
    legacy_validator._validate_records(rows)  # the journal-era validator accepts the exported attempts
    assert all("work_cap" not in row for row in rows)
    stamp = usage_journal.completed_legacy_watermark(data_root)[usage_journal.EXPORTED_WORK_CAPS]
    assert stamp["caps"] == {"work-a": before} and stamp["rows"] == len(rows)
    with usage_store.read(data_root) as view:  # the next access imports the exported journal
        assert view.binding("work_cap", "work-a") == before
        assert view.summary(billing_group_id="work-a")["settled_usd"] == 10.0
        assert view.summary(billing_group_id="other-a")["settled_usd"] == 1.0
    assert all("work_cap" not in row for row in ledger_rows(data_root))
    _spend(_scope(data_root, "work-a", "work-a", "work-a"), 4.0)
    assert ua.usage_projection(data_root, billing_group_id="work-a")["remaining_known_usd"] == pytest.approx(1.0)


def test_root_projection_reads_each_binding_axis_once(data_root):
    from ouroboros import usage_store
    from ouroboros._usage_rows import projection_view

    for task_id in ("work-a", "other-a", "other-b"):
        _work(data_root, task_id)
        _spend(_scope(data_root, task_id, task_id, task_id), 1)
    assert _raise(data_root, "work-a", 15.0)["ok"]
    statements = []
    with usage_store.read(data_root) as view:
        view.conn.set_trace_callback(statements.append)
        try:
            roots = projection_view(view, include_roots=True)["by_root"]
        finally:
            view.conn.set_trace_callback(None)
    assert {key: value["limit_usd"] for key, value in roots.items()} == {
        "work-a": 15.0, "other-a": 10.0, "other-b": 10.0}
    assert len([sql for sql in statements if "FROM bindings" in sql]) == 2


def _decision_context(tmp_path, repo, monkeypatch, *, words=WORDS, chat_id=1, task_id="main-a", project_id=""):
    from ouroboros.tools.registry import ToolContext
    from supervisor import message_bus

    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    monkeypatch.setattr(message_bus, "load_state", lambda: {})
    bridge = message_bus.LocalChatBridge({})
    bridge.ui_send(words, client_message_id="owner-decision", chat_id=chat_id, project_id=project_id)
    [update] = bridge.get_updates(offset=0, timeout=0)
    inbound = update["message"]
    assert inbound["accepted_source_row"]["ingress_accepted"] is True
    ctx = ToolContext(repo_dir=repo, drive_root=tmp_path, task_id=task_id, current_chat_id=chat_id, is_direct_chat=True,
                      task_metadata={"work_kind": "owner", "origin_message_ref": inbound["accepted_source_ref"]})
    return ctx


@pytest.fixture
def listed_rooms(data_root):
    from ouroboros.projects_registry import create_project
    from ouroboros.utils import utc_now_iso
    from tests._shared import write_queue_document

    keys = ("work-a", "other-a")
    rooms = {key: create_project(data_root, f"project-{key}") for key in keys}
    stamp = utc_now_iso()
    write_queue_document(data_root / "state/queue_snapshot.json", {
        "ts": stamp, "pending": [], "running": [{"id": key, "attempt": 1, "task": {
            "id": key, "chat_id": room["chat_id"], "project_id": room["id"], "title": key,
        }} for key, room in rooms.items()],
    })
    write_queue_document(data_root / "state/queue_dispatch.json", {"ts": stamp, "budget_root_fences": [], "members": {}})
    return rooms


def test_owner_words_through_dispatcher_and_safety_reach_a_running_helper(
    data_root, listed_rooms, supervisor, monkeypatch,  # noqa: F811
):
    from ouroboros import peer_roster, usage_store
    from ouroboros.tool_call_log import invocation_scope
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.usage_admission import work_cap

    root = data_root
    _work(root, "work-a", chat_id=listed_rooms["work-a"]["chat_id"])
    _work(root, "other-a", chat_id=listed_rooms["other-a"]["chat_id"])
    room = listed_rooms["work-a"]
    listed = peer_roster.host_listed_independent_root(root, "work-a")
    assert listed["chat_id"] == room["chat_id"] and listed["project_name"] == room["name"]
    original = _scope(root, "work-a", "work-a", "work-a")
    helper = _scope(root, "helper-a", "work-a", "work-a", parent_task_id="work-a")
    neighbor = _scope(root, "other-a", "other-a", "other-a")
    _spend(original, 8.0)
    _spend(helper, 1.0)
    _spend(neighbor, 10.0)
    repo = root / "repo"
    repo.mkdir()
    ctx = _decision_context(root, repo, monkeypatch, chat_id=room["chat_id"], project_id=room["id"])
    # The ordinary owner corpus also held this opening. The checker should cite the same source once.
    ref = ctx.task_metadata["origin_message_ref"]
    ctx._owner_directives = [{"source": "origin_message", "content": WORDS, "row_sha256": ref["row_sha256"]}]
    registry = ToolRegistry(repo_dir=repo, drive_root=root)
    registry.set_context(ctx)
    with invocation_scope({"tool_call_id": "call-1"}):
        result = registry.execute_result("set_task_cost_cap", {
            "task_id": "work-a", "limit_usd": 15, "reason": "Let the work finish"})
    assert result.status == "ok", result.text
    outcome = json.loads(result.text)
    assert outcome["ok"] and outcome["limit_usd"] == 15.0
    assert outcome["known_usd"] == pytest.approx(9.0) and outcome["remaining_known_usd"] == pytest.approx(6.0)
    assert not ctx.pending_events
    [prompt] = [call["user_prompt"] for call in supervisor.calls]
    assert all(part in prompt for part in ('"work-a"', "15", WORDS))
    assert prompt.count(WORDS) == 1
    with usage_store.read(root) as view:
        stored = work_cap(view, "work-a")
    assert stored["basis"]["source"] == ref
    assert stored["basis"]["decision_kind"] == "owner_words"
    assert stored["task_id"] == "work-a" and stored["limit_usd"] == 15.0
    _spend(helper, 5.5)
    with pytest.raises(ua.BudgetExceeded, match="other-a"):
        _spend(neighbor, 0.5)


def test_unavailable_source_and_independent_boundaries_leave_money_unchanged(data_root, listed_rooms, monkeypatch):
    from ouroboros import usage_store
    from ouroboros.tool_call_log import invocation_scope
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.usage_admission import work_cap

    root = data_root
    _work(root, "work-a", chat_id=listed_rooms["work-a"]["chat_id"])
    _spend(_scope(root, "work-a", "work-a", "work-a"), 4.0)
    ctx = _decision_context(root, root, monkeypatch)
    registry = ToolRegistry(repo_dir=root, drive_root=root)
    registry.set_context(ctx)
    monkeypatch.setattr("ouroboros.tools.control_models.cyber_pro", lambda _ctx: False)

    def call(target="work-a", **kw):
        with invocation_scope({"tool_call_id": "call-denied"}):
            result = registry.execute_result("set_task_cost_cap", {"task_id": target, "limit_usd": 15.0,
                                                                  "reason": "Test decision", **kw})
            assert result.status == "error"
            return json.loads(result.text)

    assert call("missing")["error"] == "task_not_live"
    assert call(owner_source="absent")["error"] == "cost_cap_basis_unverified"
    assert call(owner_source="own_judgement")["error"] == "cost_cap_basis_unverified"
    ctx.current_chat_id = listed_rooms["other-a"]["chat_id"]
    assert call()["error"] == "work_out_of_scope"
    with usage_store.read(root) as view:
        assert work_cap(view, "work-a") is None
    assert ua.usage_projection(root, billing_group_id="work-a")["settled_usd"] == 4.0


def test_a_related_followup_fired_after_the_raise_spends_under_it(world, monkeypatch):  # noqa: F811
    from ouroboros.usage_admission import task_billing_fields
    from supervisor import queue

    _ctx, root, pending = world
    origin = ua.UsageScope(drive_root=root, task_id=ORIGIN, root_task_id=ORIGIN, root_limit_usd=10.0, source="test",
                           **BINDING)
    _spend(origin, 10.0)
    assert set_work_cap(root, {"id": ORIGIN}, ORIGIN, limit_usd=14.0, by=BY, basis=OWNER)["ok"]
    register(world)  # the published schedule_followup tool, relation related
    queue.check_scheduled_tasks()  # the schedule lifecycle fires it
    [task] = pending
    fields = task_billing_fields(task, task["id"], 10.0, root)  # its own admission's read, the configured $10
    assert fields["billing_group_id"] == ORIGIN and fields["billing_group_limit_usd"] == 10.0
    followup = ua.UsageScope(drive_root=root, task_id=task["id"], root_task_id=task["id"], source="test", **fields)
    _spend(followup, 4.0)  # past the original $10 of the whole work
    with pytest.raises(ua.BudgetExceeded, match=f"whole-work budget exhausted for group {ORIGIN}"):
        _spend(followup, 0.5)


def test_a_continue_admitted_after_the_raise_spends_under_it(tmp_path, monkeypatch):
    from ouroboros.usage_admission import task_billing_fields
    from supervisor.continuation_admission import admit_continuation
    from tests._budget_pause_exact_helpers import _install_queue
    from tests._owner_continue_helpers import NONCE, _interrupted

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    _spend(_scope(tmp_path, "pred-1", "pred-1", "pred-1", root_limit=20.0, group_limit=20.0), 20.0)
    assert set_work_cap(tmp_path, {"id": "pred-1"}, "pred-1", limit_usd=26.0, by=BY, basis=OWNER)["ok"]
    _interrupted(tmp_path)
    ack = admit_continuation("pred-1", action_nonce=NONCE)
    row = next(task for task in workers.PENDING if task["id"] == ack["successor_task_id"])
    fields = task_billing_fields(row, row["id"], 20.0, tmp_path)
    assert fields["billing_group_id"] == "pred-1" and fields["billing_group_limit_usd"] == 20.0
    successor = ua.UsageScope(drive_root=tmp_path, task_id=row["id"], root_task_id=row["id"], source="test",
                              **fields)
    _spend(successor, 5.5)
    _spend(successor, 0.5)
    with pytest.raises(ua.BudgetExceeded, match="whole-work budget exhausted for group pred-1"):
        _spend(successor, 1.0)
