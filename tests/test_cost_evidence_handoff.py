"""Typed review cash and scoped priced-zero evidence survive consumer handoffs."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import usage_accounting as accounting
from ouroboros.cost_projection import (
    carry_cost_meta,
    cost_display,
    cost_projection,
    live_root_cost_projection,
    normalize_task_result_cost_planes,
    with_cost_aliases,
)
from supervisor.state import reconstruct_task_cost
from tests.test_review_operation_collection import (
    PASS, _collect, _ctx, _observe, _session_run, _started,
    forbid_effects as forbid_effects,  # noqa: F401
)

def _attempt(amount, knowledge="exact"):
    return {"usageCost": {"cashUsd": amount, "cashKnowledge": knowledge}}


@pytest.mark.parametrize("attempts,summary,amount,estimated,partial", [
    pytest.param([_attempt(2), _attempt(None, "unknown")], {"spendUsd": 2}, 2, False, True, id="partial-exact"),
    pytest.param([_attempt(None, "unknown")], {"spendUsd": 0}, None, False, True, id="unknown"),
    pytest.param([_attempt(2)], {"spendUsd": 99}, 2, False, False, id="typed-exact-wins"),
    pytest.param([_attempt(2, "estimated")], {"spendUsd": 99}, 2, True, False, id="estimate"),
    pytest.param([_attempt(2, "estimated"), _attempt(None, "unknown")], {"spendUsd": 2}, 2, True, True, id="partial-estimate"),
    pytest.param([_attempt(True)], {"spendUsd": 99}, None, False, True, id="invalid-cash"),
    pytest.param([None], {"spendUsd": 99}, None, False, True, id="malformed-attempt"),
    pytest.param(None, {"spendUsd": 2}, 2, False, False, id="legacy-exact"),
    pytest.param(None, {"spendUsd": 2, "spendEstimated": True}, 2, True, False, id="legacy-estimate"),
    pytest.param(None, {"spendUsd": True}, None, False, False, id="legacy-invalid"),
])
@pytest.mark.parametrize("state,output_failure", [("succeeded", False), ("failed", False), ("succeeded", True)])
def test_attach_only_cash_reaches_actor_persisted_event_and_live_frame_once(
    tmp_path, monkeypatch, forbid_effects, attempts, summary, amount, estimated, partial, state, output_failure,
):
    from ouroboros import review_execution, review_operation
    from supervisor.events_budget import _handle_llm_usage

    monkeypatch.setattr("ouroboros.delegate_custody_usage._EMITTED_SESSION_USAGE", {})
    monkeypatch.setattr(accounting, "record_subscription_session",
                        lambda *a, **kw: pytest.fail("attach-only collection tried to write money"))
    detail = {"summary": {"state": state, "runId": "run-cash", **summary},
              "primaryOutput": {"text": PASS, "truncated": False}}
    if attempts is not None:
        detail["attemptExecution"] = attempts
    calls = _observe(monkeypatch, detail)
    _started(tmp_path, run_id="run-cash")
    run, ctx = _session_run(), _ctx(tmp_path)
    if output_failure:
        original = review_execution._full_session_text
        def fail_output(*a, **kw):
            raise OSError("output temporarily unreadable")
        monkeypatch.setattr(review_execution, "_full_session_text", fail_output)
        assert _collect(run, tmp_path, ctx)[1] == review_operation.DEFERRED
        assert ctx.event_queue.qsize() == 1  # terminal usage precedes the fallible output read
        monkeypatch.setattr(review_execution, "_full_session_text", original)
    actor, fact = _collect(run, tmp_path, ctx)
    assert fact == review_operation.COLLECTED
    assert actor["status"] == ("ok" if state == "succeeded" else "error")
    final = amount is not None and not estimated and not partial
    facts = actor["usage"]
    assert {key: facts[key] for key in ("cost_disclosed_usd", "cost_estimated", "cost_partial", "cost_final")} == {
        "cost_disclosed_usd": amount, "cost_estimated": estimated, "cost_partial": partial, "cost_final": final,
    }
    if attempts is not None:
        assert facts["attempt_execution"] == attempts
    assert facts["claudexor"]["run_id"] == "run-cash"
    assert facts["claudexor"]["requested_route"] == "codex"
    # A changed daemon disclosure cannot reprice a retained observation or emit it twice.
    before = list(calls)
    detail["summary"]["spendUsd"] = 1000
    replay, replay_fact = _collect(run, tmp_path, ctx)
    assert replay_fact == fact and replay["usage"] == facts and calls == before
    event, = list(ctx.event_queue.queue)
    captured = []
    _handle_llm_usage(event, SimpleNamespace(DRIVE_ROOT=tmp_path,
        bridge=SimpleNamespace(push_log=captured.append)))
    persisted = [json.loads(line) for line in (tmp_path / "logs/events.jsonl").read_text().splitlines()]
    stored, = [row for row in persisted if row.get("type") == "llm_usage"]
    assert captured == [stored]
    assert (stored["cost"], stored["cost_known"], stored["cost_final"]) == (amount if final else None, final, final)
    assert stored["task_id"] == stored["root_task_id"] == "collect-root"
    assert stored["claudexor"] == facts["claudexor"]


@pytest.mark.parametrize("cash", [0, 2, None], ids=["priced-zero", "partial-positive", "unknown-only"])
@pytest.mark.parametrize("custody_unavailable", [False, True])
def test_reconstructed_amount_survives_normalizers_and_text_reader(tmp_path, monkeypatch, cash, custody_unavailable):
    from ouroboros import delegate_custody_current, store

    accounting.record_subscription_session("paid", drive_root=tmp_path, route="test",
        task_id="task", root_task_id="task", spend_usd=cash)
    assert custody.record_started(tmp_path, custody.RunCustody(
        run_id="live", task_id="task", root_task_id="task", route_id="test"))
    if custody_unavailable:
        def unavailable(*a, **kw):
            raise store.StoreBusy("custody unavailable for this read", reason="contention")
        monkeypatch.setattr(delegate_custody_current, "task_snapshot", unavailable)
    bucket = accounting.usage_breakdown(tmp_path, task_id="task", include_custody=True)
    assert bucket["priced_rows"] == int(cash is not None)
    assert bucket["unknown_unmetered"] == int(cash is None)
    fields = reconstruct_task_cost("task", fields=True, drive_root=tmp_path)
    assert fields["cost_presentation"]["tracked_amount"] == cash
    for normalize in (with_cost_aliases, carry_cost_meta, cost_projection, normalize_task_result_cost_planes):
        fields = normalize(fields)
        assert fields["accounted_upper_bound_usd"] == cash
        assert fields["cost_presentation"]["tracked_amount"] == cash
        assert fields["cost_final"] is False
        assert fields["cost_presentation"]["accounting_open"] is True
    assert cost_display(fields) == (
        "unknown (accounting unavailable or not settled)" if cash is None
        else f"Tracked: ${cash:.2f} (total unknown)")
    if cash == 0:
        # An own-scope zero cannot establish a tree amount, nor replace explicit null.
        assert with_cost_aliases({**fields, "accounted_upper_bound_usd_with_children": 0})[
            "accounted_upper_bound_usd_with_children"] is None
        assert with_cost_aliases({**fields, "accounted_upper_bound_usd": None})["accounted_upper_bound_usd"] is None


def test_root_priced_zero_and_sibling_live_run_keep_separate_own_scopes(tmp_path):
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "root", "running", root_task_id="root")
    accounting.record_subscription_session("paid-child", drive_root=tmp_path, route="test",
        task_id="child", root_task_id="root", parent_task_id="root", spend_usd=0)
    assert custody.record_started(tmp_path, custody.RunCustody(
        run_id="live-sibling", task_id="sibling", root_task_id="root", parent_task_id="root", route_id="test"))
    child = reconstruct_task_cost("child", fields=True, drive_root=tmp_path)
    sibling = reconstruct_task_cost("sibling", fields=True, drive_root=tmp_path)
    assert child["accounted_upper_bound_usd"] == 0 and child["cost_final"] is True
    assert sibling["accounted_upper_bound_usd"] is None and sibling["cost_final"] is False
    tree = live_root_cost_projection("root", {"root_task_id": "root"}, {}, tmp_path)
    assert tree["cost_presentation"]["scope"] == "root_tree"
    assert tree["cost_presentation"]["tracked_amount"] == 0
    for normalize in (with_cost_aliases, carry_cost_meta, cost_projection, normalize_task_result_cost_planes):
        tree = normalize(tree)
        assert tree["accounted_upper_bound_usd_with_children"] == 0
        assert tree["cost_final"] is False
    # The priced child cannot establish the root's own zero through a tree carrier.
    assert with_cost_aliases({**tree, "accounted_upper_bound_usd": 0})["accounted_upper_bound_usd"] is None


def test_unavailable_money_does_not_fabricate_priced_zero(tmp_path, monkeypatch):
    def unavailable(*a, **kw):
        raise accounting.UsageAccountingError("ledger unavailable for this read")
    monkeypatch.setattr(accounting, "usage_breakdown", unavailable)
    fields = reconstruct_task_cost("task", fields=True, drive_root=tmp_path)
    assert fields["cost_accounting_status"] == "unavailable"
    assert fields["cost_presentation"] is None
    assert with_cost_aliases(fields)["accounted_upper_bound_usd"] is None
    assert "unknown" in cost_display(fields) and "$0" not in cost_display(fields)


def test_priced_zero_carrier_preserves_the_existing_unpriced_scalar_rule(tmp_path):
    from ouroboros.cost_projection import honest_accounted_amount

    for session, price in (("priced", 0), ("unpriced", None)):
        accounting.record_subscription_session(session, drive_root=tmp_path, route="test",
            task_id="task", root_task_id="task", spend_usd=price)
    bucket = accounting.usage_breakdown(tmp_path, task_id="task", include_custody=True)
    fields = reconstruct_task_cost("task", fields=True, drive_root=tmp_path)
    assert bucket["priced_rows"] == bucket["unknown_unmetered"] == 1
    assert honest_accounted_amount(bucket) is None
    assert fields["cost_presentation"]["tracked_amount"] == 0
    assert fields["cost_presentation"]["has_unpriced"] is True
    for normalize in (with_cost_aliases, carry_cost_meta, cost_projection, normalize_task_result_cost_planes):
        fields = normalize({**fields, "accounted_upper_bound_usd": 0})
        assert fields["accounted_upper_bound_usd"] is None
        assert fields["cost_final"] is False
