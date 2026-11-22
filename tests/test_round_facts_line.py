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


@pytest.mark.parametrize("source", ["owner", None])
def test_model_segment_names_an_owner_choice_and_leaves_the_follower_plain(tmp_path, monkeypatch, source):
    monkeypatch.setattr(task_pacing, "shared_wallet_observation", lambda: {"status": "no_global_limit"})
    monkeypatch.setattr(loop, "_loop_tree_accounting", lambda **_kw: None)
    context = _facts_ctx(tmp_path, monkeypatch, task_model_override="openai::chosen" if source else None,
                         task_route_source=source)
    assert append_context_facts(context)
    line = context.messages[-1]["content"]
    assert f"model {context.active_model}" in line
    assert ("set by my human for this work" in line) is bool(source)


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


# ---- the subscription census part (C09; C07 consensus C1) ------------------------------------------------------------

_HEAD = ("quota (subscription accounts, engine verdicts; counts unchanged since 21.11 13:05:21Z; shares read 21.11 "
         "13:05Z): ")


def _census_ctx(tmp_path, monkeypatch, **tool_attrs):
    from ouroboros import config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(task_pacing, "shared_wallet_observation", lambda: {"status": "no_global_limit"})
    monkeypatch.setattr(loop, "_loop_tree_accounting", lambda **_kw: None)
    return _facts_ctx(tmp_path, monkeypatch, **tool_attrs)


def _store_census(root, share=0.5, *, families=None, spent=False, **fields):
    """The census file as the duty writes it: one codex account with room at ``share``, or at its limit when
    ``spent``, unless ``families`` is given."""
    from ouroboros import subscription_census as census
    from tests.test_subscription_census import NOW, WEEK, _payload, _profile, _row

    if families is None:
        row = (_row("codex", "a", state="exhausted", resets_at="2026-11-22T12:00:00Z", windows=[(1.0, WEEK)]) if spent
               else _row("codex", "a", windows=[(share, WEEK)]))
        families = census.build_census(_payload([_profile("codex", "a")], [row]), NOW)
    census._write({"type": census.CENSUS_TYPE, "read": "ok", "code": None, "families": families,
                   "since": "2026-11-21T13:05:21+00:00", "read_at": "2026-11-21T13:05:21+00:00", **fields}, root)


def _quota_of(row) -> str:
    """The quota part of one facts row, "" without one: the row's whole text is searched (an earlier part may span
    lines)."""
    content = row.get("content") if isinstance(row.get("content"), str) else ""  # a system row may carry blocks
    return next((part.split("\n", 1)[0] for part in content.split(" | ") if part.startswith("quota (")), "")


def _next_round(context) -> str:
    context.round_idx += 1
    assert append_context_facts(context) is True
    return _quota_of(context.messages[-1])


def test_the_census_part_is_printed_when_its_counts_change_and_never_repeated(tmp_path, monkeypatch):
    """Change-only against the latest part in an intact facts row in view, every facts row searched: rows without
    the part between do not make it due again, shares alone do not either (they show the next time it is due), and
    A -> B -> A shows B, then A."""
    context = _census_ctx(tmp_path, monkeypatch)
    assert _next_round(context) == ""  # no census file, nothing in view: no part
    _store_census(tmp_path, 0.5)
    assert _next_round(context) == _HEAD + "codex 1 with room (7d 50% used, 1 of 1 measured)"
    assert [_next_round(context) for _ in range(3)] == ["", "", ""]  # the part stands in an older facts row
    _store_census(tmp_path, 0.7, read_at="2026-11-21T13:15:21+00:00")
    assert _next_round(context) == ""  # the shares alone moved
    _store_census(tmp_path, spent=True, since="2026-11-21T13:25:21+00:00", read_at="2026-11-21T13:25:21+00:00")
    assert _next_round(context).endswith("codex 1 at limit (earliest reset 22.11 12:00Z)")
    _store_census(tmp_path, 0.7, since="2026-11-21T13:35:21+00:00", read_at="2026-11-21T13:35:21+00:00")
    assert _next_round(context) == ("quota (subscription accounts, engine verdicts; counts unchanged since 21.11 "
                                    "13:35:21Z; shares read 21.11 13:35Z): codex 1 with room (7d 70% used, 1 of 1 measured)")
    _store_census(tmp_path, 0.7, since="2026-11-21T13:35:21+00:00", read_at="2026-11-21T15:00:00+00:00")
    assert _next_round(context) == ""  # the read time is shown, never compared


def test_a_return_to_earlier_counts_is_shown_after_the_newer_row_left_the_view(tmp_path, monkeypatch):
    """C07's case: A's row stays in view and B's row left in the same round as the census returned to A. The part
    carries its ``since``, which moves only with the counts: a new one says the return, the same one is no change."""
    context = _census_ctx(tmp_path, monkeypatch)
    _store_census(tmp_path, 0.5)
    assert _next_round(context)
    _store_census(tmp_path, spent=True, since="2026-11-21T13:25:21+00:00")
    assert _next_round(context)
    context.messages[:] = [row for row in context.messages if "at limit" not in str(row.get("content"))]
    _store_census(tmp_path, 0.5, since="2026-11-21T13:35:21+00:00")
    assert "counts unchanged since 21.11 13:35:21Z" in _next_round(context)
    other = _census_ctx(tmp_path, monkeypatch)
    _store_census(tmp_path, 0.5)
    assert _next_round(other)
    _store_census(tmp_path, 0.6)  # the same counts and since: nothing returned
    assert _next_round(other) == ""
    # A, B and A again inside one minute (each pass 16 s after the last): the epoch to the second tells the return
    same_minute = _census_ctx(tmp_path, monkeypatch)
    _store_census(tmp_path, 0.5)
    assert _next_round(same_minute)
    _store_census(tmp_path, spent=True, since="2026-11-21T13:05:37+00:00")
    assert _next_round(same_minute)
    same_minute.messages[:] = [row for row in same_minute.messages if "at limit" not in str(row.get("content"))]
    _store_census(tmp_path, 0.5, since="2026-11-21T13:05:53+00:00")
    assert "counts unchanged since 21.11 13:05:53Z" in _next_round(same_minute)


def test_the_census_part_returns_once_its_rows_left_the_view_or_changed(tmp_path, monkeypatch):
    context = _census_ctx(tmp_path, monkeypatch)
    _store_census(tmp_path, 0.5)
    assert _next_round(context).startswith("quota (subscription accounts, engine verdicts")
    assert _next_round(context) == ""
    printed = next(row for row in context.messages if _quota_of(row))
    printed["content"] += " (edited)"  # no longer exactly what the host wrote: it carries nothing
    assert _next_round(context).startswith("quota (subscription accounts, engine verdicts")
    context.messages[:] = [row for row in context.messages if row.get(HOST_CONTEXT_KIND_KEY) != CONTEXT_FACTS_NAME]
    assert _next_round(context).startswith("quota (subscription accounts, engine verdicts")  # a rewrite retired them
    assert _next_round(context) == ""


def test_a_multiline_lead_before_the_part_keeps_it_change_only(tmp_path, monkeypatch):
    """A lead the owner's words chose carries their quote verbatim, line breaks included, in a part before quota."""
    from ouroboros import subscription_census as census

    lead = {"source": "owner_words", "cited": "use this model\nfor the whole room"}
    context = _census_ctx(tmp_path, monkeypatch, task_model_override="same-model", task_route_source="owner_words",
                          task_metadata={"lead": lead})
    _store_census(tmp_path, 0.5)
    assert "codex 1 with room" in _next_round(context)
    before, _sep, _rest = context.messages[-1]["content"].partition(" | quota ")
    assert "«use this model\nfor the whole room»" in before  # the quota part sits on the quote's last line
    assert [_next_round(context) for _ in range(2)] == ["", ""]
    _store_census(tmp_path, families={})
    assert _next_round(context) == "quota " + census.RETRACTION
    (tmp_path / census.CENSUS_FILE).write_text("{torn", encoding="utf-8")
    assert _next_round(context) == "quota " + census.UNCONFIRMED
    assert _next_round(context) == ""


def test_a_failed_read_is_a_change_and_its_code_alone_is_not(tmp_path, monkeypatch):
    context = _census_ctx(tmp_path, monkeypatch)
    _store_census(tmp_path, 0.5)
    _next_round(context)
    _store_census(tmp_path, 0.5, read="failed", code="daemon_unreachable", confirmed_at="2026-11-21T14:09:00+00:00")
    assert _next_round(context) == ("quota (subscription accounts, engine not read since 21.11 14:09Z (daemon_unreachable); "
                                    "counts unchanged since 21.11 13:05:21Z; shares read 21.11 13:05Z): codex 1 with room "
                                    "(7d 50% used, 1 of 1 measured)")
    _store_census(tmp_path, 0.5, read="failed", code="observation_read_timeout", confirmed_at="2026-11-21T14:09:00+00:00")
    assert _next_round(context) == ""
    _store_census(tmp_path, 0.5)
    assert _next_round(context).startswith("quota (subscription accounts, engine verdicts")  # the recovery


def test_none_retracts_a_part_in_view_once_and_a_lost_file_says_so_once(tmp_path, monkeypatch):
    from ouroboros import subscription_census as census

    context = _census_ctx(tmp_path, monkeypatch)
    _store_census(tmp_path, families={})
    assert _next_round(context) == ""  # «none» with nothing in view: nothing to retract
    _store_census(tmp_path, 0.5)
    _next_round(context)
    _store_census(tmp_path, families={})
    assert _next_round(context) == "quota " + census.RETRACTION
    assert _next_round(context) == ""
    _store_census(tmp_path, 0.5)
    _next_round(context)
    (tmp_path / census.CENSUS_FILE).write_text("{torn", encoding="utf-8")
    assert _next_round(context) == "quota " + census.UNCONFIRMED
    (tmp_path / census.CENSUS_FILE).unlink()
    assert _next_round(context) == ""  # said once: the latest part in view already says so


def test_a_failed_read_of_an_empty_roster_says_so_once_where_a_part_is_in_view(tmp_path, monkeypatch):
    """Only a successful empty read retracts; a failed one says the engine was not read, once, and only to a view
    holding an earlier part."""
    from ouroboros import subscription_census as census

    failed = {"read": "failed", "code": "daemon_unreachable", "confirmed_at": "2026-11-21T14:09:00+00:00"}
    fresh = _census_ctx(tmp_path, monkeypatch)
    _store_census(tmp_path, families={}, **failed)
    assert _next_round(fresh) == ""
    context = _census_ctx(tmp_path, monkeypatch)
    _store_census(tmp_path, 0.5)
    _next_round(context)
    _store_census(tmp_path, families={})  # an empty roster read, no round between
    _store_census(tmp_path, families={}, **failed)  # then an account was enabled and the reads failed
    assert _next_round(context) == ("quota (subscription accounts, engine not read since 21.11 14:09Z "
                                    "(daemon_unreachable); counts unchanged since 21.11 13:05:21Z; shares read 21.11 "
                                    "13:05Z): none")
    assert _next_round(context) == ""
    _store_census(tmp_path, families={})  # the read lands: no subscription account
    assert _next_round(context) == "quota " + census.RETRACTION
    _store_census(tmp_path, families={}, **failed)
    assert "engine not read since" in _next_round(context)  # the failed form, never a second retraction
    _store_census(tmp_path, 0.5, since="2026-11-21T15:00:00+00:00")
    assert "codex 1 with room" in _next_round(context)


def test_a_same_round_reprepare_sends_a_part_that_became_due_and_reuses_the_row_otherwise(tmp_path, monkeypatch):
    """The reuse key carries the census signature, which a pending mark does not change: a row prepared while the
    census was pending is reused while it stays pending, and rebuilt once its read lands with the same counts; a failed
    verdict in view and then an account change sends «no longer confirmed» in the same round."""
    from ouroboros import subscription_census as census

    context = _census_ctx(tmp_path, monkeypatch)
    _store_census(tmp_path, 0.5, pending=True)  # a confirmed paid-key change awaits its read
    assert append_context_facts(context) is True and _quota_of(context.messages[-1]) == ""
    assert append_context_facts(context) is False  # still pending: the same row
    _store_census(tmp_path, 0.5)  # the read lands, the counts unchanged: due for this fresh view
    assert append_context_facts(context) is True
    assert _quota_of(context.messages[-1]).startswith(_HEAD)
    assert append_context_facts(context) is False  # shown: reused again
    failed = {"read": "failed", "code": "daemon_unreachable", "confirmed_at": "2026-11-21T14:09:00+00:00"}
    _store_census(tmp_path, 0.5, **failed)
    assert append_context_facts(context) is True and "engine not read since" in _quota_of(context.messages[-1])
    _store_census(tmp_path, 0.5, pending=True, **failed)  # an account disabled while the engine is unread
    assert append_context_facts(context) is True
    assert _quota_of(context.messages[-1]) == "quota " + census.UNCONFIRMED
    assert append_context_facts(context) is False


def test_a_pending_census_keeps_what_is_in_view_also_across_a_same_round_reprepare(tmp_path, monkeypatch):
    from ouroboros import subscription_census as census

    context = _census_ctx(tmp_path, monkeypatch)
    _store_census(tmp_path, 0.5)
    assert append_context_facts(context) is True
    census.invalidate("account_disabled")  # a confirmed disable: pending until the duty's read lands
    assert census.read_file()["pending"] is True
    rows = len(context.messages)
    assert append_context_facts(context) is False and len(context.messages) == rows  # the same row: no new clock
    assert _next_round(context) == ""
    fresh = _census_ctx(tmp_path, monkeypatch)  # a work with nothing in view is not told the roster that changed
    assert _next_round(fresh) == ""
    _store_census(tmp_path, spent=True, since="2026-11-21T13:25:21+00:00")
    assert "at limit" in _next_round(context)
    assert "at limit" in _next_round(fresh)


def test_a_pending_census_whose_read_failed_says_once_the_summary_is_no_longer_confirmed(tmp_path, monkeypatch):
    from ouroboros import subscription_census as census

    context = _census_ctx(tmp_path, monkeypatch)
    _store_census(tmp_path, 0.5)
    _next_round(context)
    _store_census(tmp_path, 0.5, pending=True, read="failed", code="daemon_unreachable",
                  confirmed_at="2026-11-21T14:09:00+00:00")
    assert _next_round(context) == "quota " + census.UNCONFIRMED
    assert _next_round(context) == ""
    fresh = _census_ctx(tmp_path, monkeypatch)
    assert _next_round(fresh) == ""  # nothing in view to withdraw


def test_a_same_round_reprepare_reevaluates_a_new_census(tmp_path, monkeypatch):
    context = _census_ctx(tmp_path, monkeypatch)
    _store_census(tmp_path, 0.5)
    assert append_context_facts(context) is True
    assert append_context_facts(context) is False  # unchanged: the row is reused
    _store_census(tmp_path, 0.8, read_at="2026-11-21T13:15:21+00:00")
    assert append_context_facts(context) is False  # the shares alone: the row is reused
    _store_census(tmp_path, spent=True, since="2026-11-21T13:25:21+00:00")  # a quota wait in the same round, then a limit
    assert append_context_facts(context) is True
    assert "codex 1 at limit" in _quota_of(context.messages[-1])


def test_building_the_row_opens_no_connection(tmp_path, monkeypatch):
    from ouroboros import claudexor_daemon
    from ouroboros.gateway import claudexor_passive
    from ouroboros.gateways import claudexor as wire

    def connection(*_args, **_kwargs):
        pytest.fail("the facts row reads only the census file")

    monkeypatch.setattr(claudexor_passive, "_quota_payload", connection)
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", connection)
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", connection)
    monkeypatch.setattr(wire.ClaudexorGateway, "__init__", connection)
    context = _census_ctx(tmp_path, monkeypatch)
    _store_census(tmp_path, 0.5)
    assert "codex 1 with room" in _next_round(context)


@pytest.mark.parametrize("lane", ["claudexor", "openrouter-anthropic"])
def test_two_quota_values_reach_the_wire_in_their_conversation_row_alone(tmp_path, monkeypatch, lane):
    """The part rides in the user-role facts row: with the round's own system blocks and tool schemas passed to the
    lane's own request builder, two different values leave the leading system text and the transmitted tools
    byte-identical, each value only in its own conversation row."""
    import json

    from ouroboros.llm import LLMClient
    from ouroboros.llm_claudexor import _request as claudexor_request
    from tests.test_message_cache_boundary import _OPENROUTER, _system
    from tests.test_review_prompt_caching import _openrouter_target

    schemas = [{"type": "function", "function": {"name": name, "description": f"the {name} tool",
                                                  "parameters": {"type": "object", "properties": {}}}}
               for name in ("read_file", "schedule_subagent")]

    def wire(context):
        if lane == "claudexor":
            return claudexor_request({"source": "codex", "resolved_model": "model"}, context.messages,
                                     context.tool_schemas, {"model_account_override": ""})
        return LLMClient(api_key="unused")._build_remote_kwargs(
            _openrouter_target(_OPENROUTER[lane]), context.messages, "high", 512, "auto", None, context.tool_schemas,
            skip_capability_fetch=True)

    def leading(payload):
        return (payload.get("system"), payload.get("instructions"), payload.get("tools"),
                [message for message in payload.get("messages") or [] if message.get("role") in ("system", "developer")])

    sent = []
    for census_state in ({"share": 0.5}, {"spent": True, "since": "2026-11-21T13:25:21+00:00"}):
        context = _census_ctx(tmp_path, monkeypatch)
        context.messages[0] = _system(False)  # governance, identity/story and runtime blocks, as a round carries them
        context.tool_schemas = schemas
        _store_census(tmp_path, **census_state)
        assert _next_round(context)
        assert next(row for row in context.messages if _quota_of(row))["role"] == "user"
        sent.append(wire(context))
    assert all("schedule_subagent" in json.dumps(payload.get("tools")) for payload in sent)  # the tools were sent
    assert json.dumps(leading(sent[0])) == json.dumps(leading(sent[1]))
    for payload, words in zip(sent, ("codex 1 with room", "codex 1 at limit")):
        holders = [message for message in payload["messages"] if words in json.dumps(message, ensure_ascii=False)]
        assert [message["role"] for message in holders] == ["user"], holders


def _child(tmp_path, *, acting=False, budget=None, profile=None, denied=()):
    from ouroboros.contracts.task_constraint import TaskConstraint

    metadata = {"delegation_role": "subagent", **({"tool_profile": profile} if profile else {})}
    constraint = (TaskConstraint(mode="acting_subagent", surface="self_worktree", write_root=str(tmp_path)) if acting
                  else TaskConstraint(mode="local_readonly_subagent"))
    return {"task_metadata": metadata, "task_constraint": constraint,
            "task_contract": {"delegation_budget": budget if budget is not None else {"may_delegate": False},
                              **({"disabled_tools": list(denied)} if denied else {})}}


@pytest.mark.parametrize("kind,room_mode,expected", [
    ("integrating", "auto", True),
    ("acting child without the delegation right", "auto", True),  # review_change is callable for its profile
    ("acting child whose review_change is withheld", "auto", False),
    ("read-only child without the delegation right", "auto", False),
    ("read-only child with the delegation right", "auto", True),
    ("read-only child with the right and schedule_subagent withheld", "auto", True),  # the direct start is left
    ("read-only child with the right and both delegation entries withheld", "auto", False),
    ("read-only child with the right left implicit", "auto", True),  # an omitted may_delegate: admission's default
    ("read-only child with an unreadable depth", "auto", True),  # unknown, not spent
    ("read-only child with the right, subagents Off in its room", "off", False),
    ("read-only child whose depth is spent", "auto", False),
])
def test_the_part_goes_to_the_foci_that_may_choose_an_executor(tmp_path, monkeypatch, kind, room_mode, expected):
    from ouroboros import settings_integrity

    right = {"may_delegate": True, "depth_remaining": 2}
    attrs = {"integrating": {},
             "acting child without the delegation right": _child(tmp_path, acting=True),
             "acting child whose review_change is withheld": _child(tmp_path, acting=True, denied=("review_change",)),
             "read-only child without the delegation right": _child(tmp_path),
             "read-only child with the delegation right": _child(tmp_path, budget=right),
             "read-only child with the right and schedule_subagent withheld": _child(
                 tmp_path, budget=right, denied=("schedule_subagent",)),
             "read-only child with the right and both delegation entries withheld": _child(
                 tmp_path, budget=right, denied=("schedule_subagent", "delegate_start")),
             "read-only child with the right left implicit": _child(tmp_path, budget={"depth_remaining": 2}),
             "read-only child with an unreadable depth": _child(tmp_path, budget={"may_delegate": True,
                                                                                  "depth_remaining": "two"}),
             "read-only child with the right, subagents Off in its room": _child(tmp_path, budget=right),
             "read-only child whose depth is spent": _child(tmp_path, budget={"may_delegate": True,
                                                                              "depth_remaining": 0})}[kind]
    monkeypatch.setattr(settings_integrity, "effective_subagents_mode", lambda **_kw: room_mode)
    context = _census_ctx(tmp_path, monkeypatch, **attrs)
    _store_census(tmp_path, 0.5)
    assert bool(_next_round(context)) is expected, kind
    from ouroboros import subscription_census as census

    assert (census.quota_part(context)[1] == "") is (not expected)  # a non-recipient's reuse term is constant


@pytest.mark.parametrize("grants,denied,expected", [
    (None, (), True),  # an unrestricted Presence task
    (None, ("schedule_subagent", "delegate_start", "start_task", "review_change", "commit_reviewed",
            "vcs_commit_reviewed", "preflight_review", "plan_task"), False),
    (("web_search", "send_message"), (), False),
    (("schedule_subagent",), (), True),
    (("schedule_subagent",), ("schedule_subagent",), False),
    (("review_change",), (), True),
    (("vcs_commit_reviewed",), (), True),  # the alias a sealed grant may name alone
    (("preflight_review",), (), True),
    (("preflight_review",), ("preflight_review",), False),
    (("advisory_review",), (), True),  # the registered legacy spelling of the same reviewer
    (("advisory_review",), ("advisory_review",), False),
])
def test_a_presence_task_receives_the_part_by_its_real_ceiling_and_denials(tmp_path, monkeypatch, grants, denied,
                                                                          expected):
    from ouroboros.presence_authority import PresenceCapabilityCeiling, PresenceToolGrant, presence_ceiling_payload

    contract = {"disabled_tools": list(denied)} if denied else {}
    if grants is not None:
        contract["capability_ceiling"] = presence_ceiling_payload(PresenceCapabilityCeiling(
            skill_name="helper", skill_content_hash="a" * 64, profile_fingerprint="b" * 64, state_fingerprint="c" * 64,
            selection_fingerprint="d" * 64, model_slot="main", inline_max_rounds=10,
            tool_grants=tuple(PresenceToolGrant(name) for name in grants), resource_grants=(), digest="e" * 64))
    context = _census_ctx(tmp_path, monkeypatch, task_metadata={"presence": {"place_id": "place-1"}},
                          task_contract=contract)
    _store_census(tmp_path, 0.5)
    assert bool(_next_round(context)) is expected
