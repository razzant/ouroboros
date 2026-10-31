"""The external (Codex) audit of the planning organ, 2026-10-15: each point verified against the
code and pinned in both directions through the real ToolContext, the real plan record, the
real ledger and the organ's own delivery builders (the review substrate is the fake of
``tests/test_plan_task_tool.py``). Point 1 (OpenRouter's field-less parameter filter) lives in
``tests/test_request_wire_parameter_filter.py``; the pure record rules of points 2 and 5 in
``tests/test_plan_record.py`` and ``tests/test_plan_gate.py``."""
from __future__ import annotations

from ouroboros import owner_hurry as oh
from ouroboros import plan_record
from ouroboros.review_records import ReviewActorRecord
from ouroboros.tools import plan_ask, plan_review as pr
from tests._store_shared import seed_result
from tests.test_plan_gate import _v2
from tests.test_plan_task_tool import CLEAN, _call, _record, _reply, _slots, _user_text, _write, harness as _harness

harness = _harness  # noqa: F811 - pytest fixture re-export

DEAD = {sid: {"failure_code": "credential_pool_exhausted", "reset_at": "2030-01-01T00:00:00+00:00"} for sid in ("s1", "s2", "s3")}


def _content(sub, call, seat):
    return _user_text(sub.calls[call]["request"].slot_messages[seat][-1]["content"])


# ---------------------------------------------------------------- point 2: asked, but no route

def test_an_ask_every_route_refuses_leaves_a_typed_attempt_instead_of_holding(harness):
    harness.monkeypatch.setattr("ouroboros.tools.plan_review_runtime.plan_panel_health_snapshot", lambda _slots: dict(DEAD))
    sub = harness.install({})
    ctx = harness.make_ctx(force_plan=True)
    text, result = _write(ctx, ask="council")  # write + ask in one call, as the mind does
    assert "PLAN_ASK_NOTHING_SENDABLE" in text and result.code == "CAPABILITY_UNAVAILABLE" and sub.calls == []
    assert "Revision 1 was recorded" in text and "asked, no route" in text and "Nothing was written" not in text
    record = _record(harness)
    assert [r["n"] for r in record["revisions"]] == [1] and record["asks"] == []
    (attempt,) = record["attempts"]
    assert attempt["kind"] == "no_route" and attempt["revision"] == 1 and attempt["ask"] == "council" and attempt["ts"]
    assert set(attempt["not_sendable"]) == {"s1", "s2", "s3"} and "credential_pool_exhausted" in attempt["not_sendable"]["s1"]
    decision = oh.plan_gate_decision(ctx, {}, enforcement="blocking")
    assert decision["status"] == "clear" and decision["allow"] is True and decision["holding"] == []
    assert decision["asks_unroutable"] == 1 and decision["asks_dispatched"] == 0
    assert oh.plan_review_disclosure(decision) == ""
    read, _ = _call(ctx)
    assert "attempt at" in read and "no route" in read and "nobody was called" in read and "Nothing holds" in read
    assert any("reached nobody" in line for line in harness.progress)
    # The other direction: a plan the mind never asked about holds nothing; a reading before any plan
    # that no route could take leaves the same typed attempt, with no revision, and holds nothing.
    unasked = harness.make_ctx(task_id="task-2", force_plan=True)
    _write(unasked)
    assert oh.plan_gate_decision(unasked, {}, enforcement="blocking")["allow"] is True
    bare = harness.make_ctx(task_id="task-3", force_plan=True)
    text, result = _call(bare, ask="council")
    assert "PLAN_ASK_NOTHING_SENDABLE" in text and result.code == "CAPABILITY_UNAVAILABLE" and len(sub.calls) == 0
    bare_record = _record(harness, "task-3")
    assert bare_record["revisions"] == [] and bare_record["asks"] == [] and bare_record["attempts"][0]["revision"] is None
    assert oh.plan_gate_decision(bare, {}, enforcement="blocking")["allow"] is True
    read, _ = _call(bare)
    assert "attempt at" in read and "on the reading of the request: no route" in read and "Nothing holds" in read


# ---------------------------------------------------------------- point 3: imported objections reach their raiser

def test_an_imported_objection_is_listed_to_its_raiser_and_only_its_withdrawal_releases_the_hold(harness):
    seed_result(harness.drive, "task-1", {"task_id": "task-1", "status": "running",
                                           "plan_review_state": _v2(closed=False, blocking=("s1",))})
    sub = harness.install({"s1": CLEAN, "s2": _reply("Not mine to withdraw.", withdrawn=[("s1:f1", "looks fine to me")])})
    ctx = harness.make_ctx()
    text, result = _write(ctx)
    assert result.meta["plan_holding"] == 1 and "HELD" in text and "imported from the previous plan review" in text
    _call(ctx, ask=["s2"], message="anything to add?")
    assert "You have no open objection on this plan." in _content(sub, 0, "s2")
    text, result = _call(ctx)
    assert result.meta["plan_holding"] == 1 and "ignored withdrawals" in text and "s2/s1:f1" in text, "a foreign withdrawal changes nothing"
    _call(ctx, ask=["s1"], message="still valid?")
    own = _content(sub, 1, "s1").split("## YOUR OPEN OBJECTIONS")[1].split("## WHAT OUROBOROS ASKS YOU NOW")[0]
    assert "- s1:f1 (your blocking finding in this task's previous plan review, imported" in own and "wrong shape" in own
    assert "You have no open objection" not in own and "Adjudicate these FIRST" in own
    text, result = _call(ctx)
    assert result.meta["plan_holding"] == 1 and "NOT ADJUDICATED" in text, "a reply that mentioned nothing keeps it open, flagged"
    harness.install({"s1": _reply("Fixed by revision 1.", withdrawn=[("s1:f1", "the shape is right now")])})
    _call(ctx, ask=["s1"], message="withdraw?")
    text, result = _call(ctx)
    assert result.meta["plan_holding"] == 0 and "Nothing holds your final answer." in text and "withdrawn: s1/s1:f1" in text
    assert oh.plan_gate_decision(ctx, {}, enforcement="blocking")["status"] == "clear"


# ---------------------------------------------------------------- point 4: moved-out prose reaches a late critic

def test_a_late_critic_reads_moved_out_prose_from_its_retained_source_or_is_told_what_is_missing(harness):
    harness.monkeypatch.setattr(plan_record, "_PLAN_RECORD_MAX_BYTES", 10_000)
    harness.state["slots"] = _slots(("s1", "m/a"), ("s2", "m/b"), ("s3", "m/c"), ("s4", "m/d"))
    said = []

    def long_answer(_request):
        said.append(f"{'Slide five carries the argument. ' * 280}#{len(said) + 1}")
        return _reply(said[-1])

    sub = harness.install({"s1": long_answer, "s2": CLEAN, "s3": CLEAN, "s4": CLEAN})
    ctx = harness.make_ctx()
    _write(ctx, ask=["s1", "s2"])
    _call(ctx, ask=["s1", "s2"], message="once more")  # the second ask pushes the index past its bound: ask 1's prose moves out
    first = next(p for p in _record(harness)["asks"][0]["participants"] if p["seat_id"] == "s1")
    assert first["reply"]["prose_inline"] is False and first["reply"]["prose"] == "" and first["reply_ref"]["status"] == "retained"
    _call(ctx, ask=["s3"], message="late critic")
    content = _content(sub, 2, "s3")
    assert said[0] in content and said[1] in content, "ask 1's prose came back from the retained source; ask 2's is inline"
    assert "prose not delivered" not in content and "prose not inline" not in content
    assert next(p for p in _record(harness)["asks"][0]["participants"] if p["seat_id"] == "s1")["reply"]["prose"] == "", \
        "delivery never writes the prose back into the index"
    harness.monkeypatch.setattr("ouroboros.tools.plan_evidence.EVIDENCE_PER_ITEM_BYTES", 1_000)
    _call(ctx, ask=["s4"], message="another late critic")
    content = _content(sub, 3, "s4")
    assert said[0] not in content and "s1 (m/a) answered:\n  (prose not delivered:" in content
    assert "larger than one attachment" in content and f"retained at {first['reply_ref']['ref']['path']}" in content
    assert "answered:\n-" not in content and "answered:\n\n" not in content and not content.rstrip().endswith("answered:"), \
        "never silent emptiness"


# ---------------------------------------------------------------- point 5: the hook settled the seat before the barrier rows were written

def test_a_seat_the_custody_hook_settled_before_the_barrier_rows_were_written_keeps_its_answer(harness):
    ctx = harness.make_ctx()
    pending = {"status": "pending", "raw_text": "", "operation_state": "pending_dispatch", "late_result_pending": True}

    def settled_by_the_hook_then_reported_pending(request):
        slot = next(s for s in harness.state["slots"] if s.slot_id == "s1")
        actor = ReviewActorRecord(slot_id="s1", model="m/a", status="ok", raw_text=_reply("Early.", objections=[("o1", "too thin")]),
                                  operation_id="op-s1-early", operation_state="late_settled", usage={"cost": 0.01, "resolved_model": "m/a"})
        plan_ask.settle_released_reply(ctx, request=request, task_id="task-1", actor=actor, slot=slot, settled_wave={}, roster_size=3)
        return dict(pending)  # the barrier's row still says pending: the dispatcher's copy of the record is stale

    harness.install({"s1": settled_by_the_hook_then_reported_pending, "s2": pending})
    _write(ctx, ask="council")
    rows = {p["seat_id"]: p for p in _record(harness)["asks"][0]["participants"]}
    assert rows["s1"]["status"] == "answered" and rows["s1"]["reply"]["objections"] == [{"id": "o1", "text": "too thin"}]
    assert rows["s1"].get("late_result_pending") is not True and rows["s2"]["status"] == "pending" and rows["s3"]["status"] == "answered"
    assert pr._decision(ctx, _record(harness))["holding"] == ["s1:o1"], "the early objection holds; the echo re-opened nothing"
    _text, result = _call(ctx)
    assert result.meta["plan_holding"] == 1 and result.meta["plan_pending_asks"] == 1, "s2 is still owed, s1 is not"
