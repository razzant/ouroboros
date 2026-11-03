"""#1748 / #1129: the per-round facts line is the one every-round carrier of time,
money, nanny burn, tool counts and the answer-protocol latch — facts with their bounds
named, unknown kept unknown, and one money observation shared with the round's
threshold notices (no second ledger read, no forecast, no resource-JSON dump)."""

from __future__ import annotations

import datetime
from types import SimpleNamespace

import pytest

from ouroboros import loop, task_pacing
from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
from ouroboros.context_runtime_facts import round_clock_fact
from ouroboros.loop_messages import CONTEXT_FACTS_NAME, append_context_facts
from ouroboros.loop_nudges import _inject_round_checkpoints
from tests.test_loop_compaction import _ctx, _fit, _measure_cycle

T = datetime.datetime(2026, 10, 13, 1, 0, tzinfo=datetime.timezone.utc).timestamp()


def _iso(epoch: float) -> str:
    return datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).isoformat()


def test_time_segment_names_wall_execution_limit_deadline_and_finalization(monkeypatch):
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")
    ctx = SimpleNamespace(task_id="t", task_started_at=T, _budget_paused_sec=300.0,
                          task_metadata={"deadline_at": _iso(T + 3600), "execution_timeout_sec": 1800})
    fact = round_clock_fact(ctx, now=T + 600)
    assert fact.split("; ") == [
        "time sampled 2026-10-13T01:10:00Z",
        "wall 10.0 min since start 2026-10-13T01:00:00Z",
        "execution 5.0 min (quota waits and pauses excluded)",
        "task execution budget 30.0 min, 25.0 min left",
        "deadline 2026-10-13T02:00:00Z, 50.0 min left",
        # the nearer explicit bound is the execution budget (25 min), minus the 2 min grace
        "host requests one tool-less final answer at ~2026-10-13T01:33:00Z (2.0 min before the nearer explicit bound)",
    ]


def test_time_segment_without_start_or_bounds_says_unknown_and_none_never_zero():
    fact = round_clock_fact(SimpleNamespace(task_id="t", task_metadata={}), now=T)
    assert "wall elapsed unknown (no recorded start)" in fact and "execution elapsed unknown" in fact
    assert "no execution limit" in fact and "deadline" not in fact and "final answer" not in fact
    assert " 0.0 min" not in fact


def test_a_global_ceiling_alone_has_no_local_finalization_claim(monkeypatch):
    monkeypatch.setenv("OUROBOROS_TASK_ABS_CEILING_SEC", "3600")
    fact = round_clock_fact(SimpleNamespace(task_id="t", task_started_at=T, task_metadata={}), now=T + 600)
    assert "global execution ceiling 60.0 min, 50.0 min left" in fact
    assert "final answer" not in fact  # the loop-local finalize fires only on explicit bounds


def _facts_ctx(tmp_path, monkeypatch, **tool_attrs):
    context = _ctx(tmp_path)
    for key, value in tool_attrs.items():
        setattr(context.tools._ctx, key, value)
    monkeypatch.setattr(loop, "_measure_round_main_fit", _measure_cycle([_fit()]))
    return context


def test_the_round_notices_and_the_facts_line_share_one_money_observation(tmp_path, monkeypatch):
    """#1129: a neighbour drained the wallet; the round's wallet crossing and its facts
    line render the SAME observation, read once (no added per-round ledger read)."""
    reads = {"wallet": 0, "tree": 0}

    def wallet():
        reads["wallet"] += 1
        return {"status": "known", "remaining_usd": 19.0, "limit_usd": 100.0, "observed_at": "2026-10-13T01:10:00Z"}

    def tree(**_kw):
        reads["tree"] += 1
        return {"settled_usd": 3.0, "accounted_usd": 4.5, "root_limit_usd": 50.0, "age_sec": 7.0}

    monkeypatch.setattr(task_pacing, "shared_wallet_observation", wallet)
    monkeypatch.setattr(loop, "_loop_tree_accounting", tree)
    checkpoints: list = []
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda _q, _id, _logs, payload: checkpoints.append(payload))
    ceiling = task_pacing.CostCeiling(state="disabled", root_cap_usd=50.0, basis="no_default_cost_stop")
    context = _facts_ctx(tmp_path, monkeypatch, _cost_ceiling=ceiling, task_contract={})
    context.accumulated_usage["cost"] = 0.4
    assert _inject_round_checkpoints(
        round_idx=1, messages=context.messages, accumulated_usage=context.accumulated_usage,
        tools=context.tools, event_queue=None, task_id="task", drive_logs=None,
        budget_remaining_usd=40.0, cost_ceiling=ceiling) is True
    assert [row.get("budget_axis") for row in checkpoints] == ["shared_wallet"]
    assert "shared wallet 50% remaining crossed" in context.messages[-1]["content"]
    assert append_context_facts(context) is True
    line = context.messages[-1]["content"]
    assert context.messages[-1][HOST_CONTEXT_KIND_KEY] == CONTEXT_FACTS_NAME
    assert reads == {"wallet": 1, "tree": 1}
    for part in ("shared wallet $19.00 known headroom of $100.00 limit (unreserved; shared with other tasks)",
                 "Task tree spend: ~$3.00 of $50.00 hard tree cap (known, subagents included; +$1.50 open holds "
                 "not counted; ceiling is unreserved); ledger snapshot 7 s old",
                 "own recorded cost $0.40"):
        assert part in line, part
    assert "in-task cost ceiling" not in line  # disabled: no early stop exists to name
    assert line.count("$50.00 hard tree cap") == 1 and "hard tree cap $50.00" not in line  # named once
    assert "{" not in line  # no resource JSON on the every-round carrier


def test_wallet_status_unknown_and_unlimited_are_named(tmp_path, monkeypatch):
    for observed, expected in (({"status": "unknown", "reason": "ledger integrity degraded"},
                                "shared wallet unknown (ledger integrity degraded)"),
                               ({"status": "no_global_limit"}, "shared wallet: no global limit")):
        monkeypatch.setattr(task_pacing, "shared_wallet_observation", lambda observed=observed: dict(observed))
        monkeypatch.setattr(loop, "_loop_tree_accounting", lambda **_kw: None)
        context = _facts_ctx(tmp_path, monkeypatch)
        assert append_context_facts(context) is True
        assert expected in context.messages[-1]["content"]


def test_a_nanny_facts_line_carries_its_burn_and_a_protocol_task_its_latch(tmp_path, monkeypatch):
    from ouroboros.loop import _note_nanny_delegate_activity

    monkeypatch.setattr(task_pacing, "shared_wallet_observation", lambda: {"status": "no_global_limit"})
    monkeypatch.setattr(loop, "_loop_tree_accounting", lambda **_kw: None)
    context = _facts_ctx(tmp_path, monkeypatch, _nanny_route_dispatched=True,
                         task_contract={"answer_protocol": "final_answer_line"},
                         _execution_trace={"tool_calls": [], "best_valid_final_answer": "42"})
    tool_ctx = context.tools._ctx
    _note_nanny_delegate_activity(tool_ctx, 1, {"cost": 0.1}, [
        {"id": "c", "type": "function", "function": {"name": "delegate_start", "arguments": "{}"}}])
    _note_nanny_delegate_activity(tool_ctx, 4, {"cost": 0.7}, [])
    assert append_context_facts(context) is True
    line = context.messages[-1]["content"]
    assert "nanny: 3 own rounds since the last delegate verb call (round 1)" in line
    assert "own recorded cost $0.60 since the last delegate_start/schedule_subagent call (round 1)" in line
    assert "FINAL ANSWER line latched: yes" in line


def test_tool_counts_come_from_the_loop_usage_carrier(tmp_path, monkeypatch):
    monkeypatch.setattr(task_pacing, "shared_wallet_observation", lambda: {"status": "no_global_limit"})
    monkeypatch.setattr(loop, "_loop_tree_accounting", lambda **_kw: None)
    context = _facts_ctx(tmp_path, monkeypatch)
    context.tools._ctx._accumulated_usage = context.accumulated_usage
    task_pacing.record_tool_activity(context.tools._ctx, {"fn_name": "read_file", "is_error": False})
    task_pacing.record_tool_activity(context.tools._ctx, {"fn_name": "run_command", "is_error": True})
    assert append_context_facts(context) is True
    assert "observed tool results 2 (1 errored)" in context.messages[-1]["content"]


def test_an_unreadable_clock_is_said_and_never_takes_the_round_down(tmp_path, monkeypatch):
    import ouroboros.context_runtime_facts as runtime_facts

    def broken(_ctx, now=None):
        raise RuntimeError("clock owner unavailable")

    monkeypatch.setattr(runtime_facts, "round_clock_fact", broken)
    monkeypatch.setattr(task_pacing, "shared_wallet_observation", lambda: {"status": "no_global_limit"})
    monkeypatch.setattr(loop, "_loop_tree_accounting", lambda **_kw: None)
    context = _facts_ctx(tmp_path, monkeypatch)
    assert append_context_facts(context) is True
    assert "time and nanny facts unavailable" in context.messages[-1]["content"]


def test_a_continue_successor_sees_its_whole_work_group_headroom(tmp_path, monkeypatch):
    """A billing group (``usage_admission.task_money_snapshot``) keeps its room, binding
    axis and both axes on the facts line — never relabelled as a hard tree cap — from the
    same one reading the round's notices use."""
    snapshot = {"settled_usd": 30.0, "accounted_usd": 31.0, "root_limit_usd": 40.0, "remaining_known_usd": 10.0,
                "root_axis": {"settled_usd": 12.0, "accounted_usd": 12.0, "limit_usd": 50.0, "source": "x"},
                "group_axis": {"settled_usd": 30.0, "accounted_usd": 31.0, "limit_usd": 40.0, "source": "y"},
                "accounting_basis": "billing_group", "binding_axis": "group", "integrity_degraded": False,
                "age_sec": 0.0}
    monkeypatch.setattr(task_pacing, "shared_wallet_observation", lambda: {"status": "no_global_limit"})
    monkeypatch.setattr(loop, "_loop_tree_accounting", lambda **_kw: dict(snapshot))
    ceiling = task_pacing.CostCeiling(state="disabled", root_cap_usd=40.0)
    context = _facts_ctx(tmp_path, monkeypatch, _cost_ceiling=ceiling, task_contract={})
    assert append_context_facts(context) is True
    line = context.messages[-1]["content"]
    assert ("whole-work group spend ~$30.00 known (+$1.00 open holds not counted), $10.00 left "
            "(group cap binds); root ~$12.00 of $50.00, group ~$30.00 of $40.00") in line
    assert "hard tree cap" not in line
    note = task_pacing.build_cost_budget_note(
        SimpleNamespace(), cost_ceiling_usd=None, task_cost=1.0, tree_cost_usd=30.0, root_cap_usd=40.0,
        whole_work_group=True)
    assert note is not None and "whole-work allowance" in note.text and "hard tree cap" not in note.text
    assert note.checkpoint["budget_axis"] == "whole_work_allowance"


@pytest.mark.parametrize("later_calls", [0, 1, 2])
def test_answer_fact_distinguishes_a_saved_line_from_its_current_currency(later_calls):
    from ouroboros.loop_messages import _final_answer_fact

    trace = {"best_valid_final_answer": "42", "best_valid_final_answer_tools": 1,
             "tool_calls": [{}] * (1 + later_calls)}
    ctx = SimpleNamespace(task_contract={"answer_protocol": "final_answer_line"}, _execution_trace=trace)
    fact = _final_answer_fact(ctx)
    assert ("not current" in fact) is bool(later_calls), fact
    if later_calls:
        assert f"{later_calls} later tool call" in fact
    else:
        assert "latched: yes" in fact


def test_changed_route_reprepare_refreshes_money_without_rereading_unchanged_facts(tmp_path, monkeypatch):
    reads = []
    def wallet():
        reads.append(True)
        return {"status": "known", "remaining_usd": 90.0 if len(reads) == 1 else 10.0,
                "limit_usd": 100.0, "observed_at": _iso(T + 3600 * (len(reads) - 1))}
    monkeypatch.setattr(task_pacing, "shared_wallet_observation", wallet)
    monkeypatch.setattr(loop, "_loop_tree_accounting", lambda **_kw: None)
    context = _facts_ctx(tmp_path, monkeypatch)
    monkeypatch.setattr(loop, "_measure_round_main_fit", lambda *_a, **_kw: _fit())
    assert append_context_facts(context)
    assert "shared wallet $90.00" in context.messages[-1]["content"]
    assert not append_context_facts(context)
    assert len(reads) == 1
    context.active_model = "test-provider/route-after-wait"
    assert append_context_facts(context)
    assert "shared wallet $10.00" in context.messages[-1]["content"]
    assert len(reads) == 2
    assert not append_context_facts(context)
    assert len(reads) == 2


def test_an_owner_turn_names_its_own_wait_so_far_and_a_wake_has_none(monkeypatch):
    """The round line reads the turn's own execution from the process-local registry: accepted at T,
    this request N s later so far with the slowest phase; after the first observed dispatch, when it
    went out and on which basis. A wake (nobody awaits it) carries no such segment."""
    from collections import deque

    from supervisor import active_activity
    from supervisor.active_activity import get_direct_activity_registry

    monkeypatch.setattr(active_activity, "_DIRECT_TURNS", deque(maxlen=100))
    monkeypatch.setattr("supervisor.state.append_jsonl", lambda path, row, *a, **k: True)
    registry = get_direct_activity_registry()
    registry.clear()
    accepted = _iso(T - 20)
    try:
        owner = registry.register("owner-turn", 1, reply_expected=True, origin_message_ref={
            "chat_id": 1, "client_message_id": "m-1", "ts": accepted, "text_sha256": "0" * 64})
        owner.execution.entered_at = T  # the handler's thread began twenty seconds after acceptance
        owner.execution.advance("started")
        fact = round_clock_fact(SimpleNamespace(task_id="owner-turn", task_metadata={}), now=T + 600)
        segment = fact.split("no execution limit; ", 1)[1]
        assert segment.startswith(f"my human's message accepted at {accepted}; this request ")
        assert " s later; slowest phase so far " in segment and "went out" not in segment
        assert float(segment.split("this request ")[1].split(" s later")[0]) >= 20.0
        owner.execution.observe("main_llm_call_state", {"llm_call_id": "c", "phase": "started"})
        owner.execution.observe("main_llm_call_state", {"llm_call_id": "c", "phase": "observed",
                                                        "model_activity": {"phase": "dispatching"}})
        fact = round_clock_fact(SimpleNamespace(task_id="owner-turn", task_metadata={}), now=T + 600)
        assert "; my first request went out " in fact and " s later (observed_send); slowest phase so far " in fact

        registry.register("wake", 1, reply_expected=False)
        fact = round_clock_fact(SimpleNamespace(task_id="wake", task_metadata={}), now=T + 600)
        assert "my human" not in fact and "went out" not in fact and fact.endswith("no execution limit")
        assert "my human" not in round_clock_fact(SimpleNamespace(task_id="pooled", task_metadata={}), now=T)
    finally:
        registry.clear()
