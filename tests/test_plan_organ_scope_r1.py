"""Scope review R1 (Astra F1–F3, Opus C1–C3, Fable advisories) — fix batch 1, pinned in both
directions through the real ToolContext, the real plan record, the real ledger and the organ's
own delivery builders (the review substrate is the fake of ``tests/test_plan_task_tool.py``).
Pure record rules live in ``tests/test_plan_record.py``; the web panel in
``web/tests/review_plan_record.test.js``; the gateway projection in ``tests/test_plan_record_gateway.py``."""
from __future__ import annotations

from ouroboros import owner_hurry as oh
from ouroboros import plan_record
from ouroboros.review_records import ReviewActorRecord
from ouroboros.tools import plan_ask, plan_review as pr
from tests._store_shared import seed_result
from tests.test_plan_gate import _v2
from tests.test_plan_task_tool import CLEAN, _call, _record, _reply, _slots, _user_text, _write, harness as _harness

harness = _harness  # noqa: F811 - pytest fixture re-export

REFUSED = {"status": "not_dispatched", "raw_text": "", "error": "delegated review route subscription window is exhausted (resets 2030-01-01)",
           "failure_code": "subscription_window_exhausted", "reset_at": "2030-01-01T00:00:00+00:00", "operation_state": "not_dispatched"}
PENDING = {"status": "pending", "raw_text": "", "operation_state": "pending_dispatch", "late_result_pending": True}


def _content(sub, call, seat):
    return _user_text(sub.calls[call]["request"].slot_messages[seat][-1]["content"])


def _settle(ctx, harness, request, seat, text, *, roster=2):
    slot = next(s for s in harness.state["slots"] if s.slot_id == seat)
    actor = ReviewActorRecord(slot_id=seat, model=slot.model, status="ok", raw_text=text, operation_id=f"op-{seat}-late",
                              operation_state="late_settled", usage={"cost": 0.01, "resolved_model": slot.model})
    plan_ask.settle_released_reply(ctx, request=request, task_id="task-1", actor=actor, slot=slot, settled_wave={}, roster_size=roster)


# ---------------------------------------------------------------- C1: a quoted object in a preamble is not the reply

def test_a_reply_behind_a_quoted_done_when_row_keeps_its_objection_and_holds(harness):
    preamble = ('I re-read the bar:\n```json\n{"id": "c1", "claim": "exactly 5 slides", "support": "count them"}\n```\n'
                + _reply("Too thin.", objections=[("o1", "no story arc")]))
    harness.install({"s1": preamble, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    text, result = _write(ctx, ask="council")
    assert result.meta["plan_holding"] == 1 and "s1 / o1" in text and "HELD" in text
    s1 = next(p for p in _record(harness)["asks"][0]["participants"] if p["seat_id"] == "s1")
    assert s1["status"] == "answered" and s1["reply"]["prose"] == "Too thin." and s1["reply"]["objections"] == [{"id": "o1", "text": "no story arc"}]
    assert any("s1 answered" in line or "m/a answered — 1 objection" in line for line in harness.progress)


# ---------------------------------------------------------------- F1: a $0 pre-dispatch refusal never counts as a reached reviewer

def test_a_seat_refused_before_dispatch_is_not_reached_and_an_all_refused_ask_is_attempted_not_held(harness):
    harness.state["slots"] = _slots(("s1", "m/a"), ("s2", "m/b"))
    harness.install({"s1": dict(REFUSED), "s2": dict(REFUSED)})
    ctx = harness.make_ctx(force_plan=True)
    text, _result = _write(ctx, ask="council")
    rows = {p["seat_id"]: p for p in _record(harness)["asks"][0]["participants"]}
    assert all(p["status"] == "not_dispatched" and p["dispatched"] is False for p in rows.values())
    assert rows["s1"]["failure_code"] == "subscription_window_exhausted" and "not sent" in text
    decision = oh.plan_gate_decision(ctx, {}, enforcement="blocking")
    assert decision["asks_dispatched"] == 0 and decision["asks_unroutable"] == 1
    assert decision["status"] == "clear" and decision["allow"] is True and decision["holding"] == []
    # The other direction: one seat refused and one reached is a reached ask; the refused seat never saw the plan.
    other = harness.make_ctx(task_id="task-2", force_plan=True)
    harness.install({"s1": dict(REFUSED), "s2": CLEAN})
    _write(other, ask="council")
    met = oh.plan_gate_decision(other, {}, enforcement="blocking")
    assert met["asks_dispatched"] == 1 and met["asks_unroutable"] == 0
    views = plan_record.participant_views(_record(harness, "task-2"))
    assert views["s1"]["last_seen_revision"] is None and views["s2"]["last_seen_revision"] == 1


def test_a_late_pre_dispatch_refusal_through_the_custody_hook_withdraws_the_dispatch_flag_too(harness):
    harness.state["slots"] = _slots(("s1", "m/a"), ("s2", "m/b"))
    sub = harness.install({"s1": dict(PENDING), "s2": dict(PENDING)})
    ctx = harness.make_ctx(force_plan=True)
    _write(ctx, ask="council")
    assert oh.plan_gate_decision(ctx, {}, enforcement="blocking")["asks_dispatched"] == 1, "pending seats were sent"
    request = sub.calls[0]["request"]
    for seat in ("s1", "s2"):
        slot = next(s for s in harness.state["slots"] if s.slot_id == seat)
        actor = ReviewActorRecord(slot_id=seat, model=slot.model, status="not_dispatched", raw_text="", error=REFUSED["error"],
                                  operation_id=f"op-{seat}-refused", operation_state="not_dispatched",
                                  failure_code="subscription_window_exhausted", reset_at=REFUSED["reset_at"])
        plan_ask.settle_released_reply(ctx, request=request, task_id="task-1", actor=actor, slot=slot, settled_wave={}, roster_size=2)
        decision = oh.plan_gate_decision(ctx, {}, enforcement="blocking")
        row = next(p for p in _record(harness)["asks"][0]["participants"] if p["seat_id"] == seat)
        assert row["status"] == "not_dispatched" and row["dispatched"] is False and row["reset_at"] == REFUSED["reset_at"]
    assert decision["asks_dispatched"] == 0 and decision["asks_unroutable"] == 1 and decision["allow"] is True
    assert decision["status"] == "clear"


def test_an_exception_after_the_hand_off_is_unknown_never_nobody_reached(harness):
    """Coordinator correction (03 recovery §2 D5, probe P4): the substrate already held the seats, so an
    exception after the fan-out proves neither "no send" nor "$0". The seats stay pending and unresolved,
    the barrier's dispatched fact stands (force_plan met, nothing holds under Blocking), nothing is counted
    unroutable, the read view / progress line / disclosure say "outcome unknown — a new ask may pay twice",
    and a late reply still settles the seat as answered through the custody hook."""
    import ouroboros.review_substrate as rs

    captured = {}

    def boom(request, **_kw):
        captured["request"] = request
        raise RuntimeError("substrate down after hand-off")

    harness.monkeypatch.setattr(rs, "run_review_request", boom)
    ctx = harness.make_ctx(force_plan=True)
    text, _result = _write(ctx, ask="council")
    rows = _record(harness)["asks"][0]["participants"]
    assert rows and all(p["status"] == "pending" and p["unresolved"] is True and p["dispatched"] is True
                        and p["error"] == "dispatch outcome unknown: substrate down after hand-off" for p in rows)
    decision = oh.plan_gate_decision(ctx, {}, enforcement="blocking")
    assert decision["asks_unroutable"] == 0 and decision["asks_dispatched"] == 1
    assert decision["allow"] is True and decision["status"] == "awaiting" and decision["unknown_outcome"] == ["a1"]
    assert "Dispatch outcome unknown for s1, s2, s3" in text and "pay twice" in text and "reached nobody" not in text
    assert "pending — outcome unknown (dispatch outcome unknown: substrate down after hand-off)" in text
    assert any("dispatch outcome unknown (RuntimeError)" in line and "pay twice" in line for line in harness.progress)
    assert "the dispatch outcome of ask(s) a1 is unknown — a new ask may pay twice" in oh.plan_review_disclosure(decision)
    # A late reply still settles the unknown seat as answered through the custody hook — and holds.
    _settle(ctx, harness, captured["request"], "s1", _reply("Late, but here.", objections=[("o1", "too thin")]), roster=3)
    s1 = next(p for p in _record(harness)["asks"][0]["participants"] if p["seat_id"] == "s1")
    assert s1["status"] == "answered" and s1["reply"]["objections"][0]["id"] == "o1" and s1["unresolved"] is False
    assert pr._decision(ctx, _record(harness))["holding"] == ["s1:o1"]
    # The other direction stays: every seat POSITIVELY refused before dispatch is attempted, disclosed, not held.
    other = harness.make_ctx(task_id="task-2", force_plan=True)
    harness.install({"s1": dict(REFUSED), "s2": dict(REFUSED), "s3": dict(REFUSED)})
    _write(other, ask="council")
    refused = oh.plan_gate_decision(other, {}, enforcement="blocking")
    assert refused["asks_unroutable"] == 1 and refused["status"] == "clear" and refused["allow"] is True


# ---------------------------------------------------------------- F2: a late peer reply to an earlier ask reaches the next delivery

def test_a_peer_reply_to_an_earlier_ask_that_settles_late_reaches_the_recipients_next_delivery(harness):
    """Astra F2: a1 → A, B, C with B pending; A answers; a2 → A, A answers; B's a1 reply settles;
    a3 → A carries B's a1 reply (attributed to a1, marked late); C's a1 reply, which A saw before
    a2 was built, is not repeated; A's own words never are."""
    harness.state["slots"] = _slots(("A", "m/a"), ("B", "m/b"), ("C", "m/c"))
    first = harness.install({"A": _reply("A on a1."), "B": dict(PENDING), "C": _reply("C on a1, seen in time.")})
    ctx = harness.make_ctx()
    _write(ctx, ask="council")
    harness.install({"A": _reply("A on a2.")})
    _call(ctx, ask=["A"], message="second look")
    _settle(ctx, harness, first.calls[0]["request"], "B", _reply("B on a1, late."), roster=3)
    third = harness.install({"A": CLEAN})
    _call(ctx, ask=["A"], message="third look")
    content = _content(third, 0, "A")
    council = content.split("## THE COUNCIL SO FAR")[1].split("## YOUR OPEN OBJECTIONS")[0]
    assert "B on a1, late." in council and "### Ask a1" in council and "answers that arrived after your last delivery" in council
    assert council.index("### Ask a1") < council.index("### Ask a2"), "the late reply is told first, under its own ask"
    assert "C on a1, seen in time." not in council, "a peer reply A already saw is not repeated"
    assert "A on a1." not in council and "A on a2." not in council, "its own words are never repeated to it"
    # The other direction: A answered a3 after B's late reply arrived, so a4 does not repeat it.
    fourth = harness.install({"A": CLEAN})
    _call(ctx, ask=["A"], message="fourth look")
    council = _content(fourth, 0, "A").split("## THE COUNCIL SO FAR")[1].split("## YOUR OPEN OBJECTIONS")[0]
    assert "B on a1, late." not in council and "### Ask a3" in council and "### Ask a1" not in council


# ---------------------------------------------------------------- Fable 8 / Grok: the delta diffs the retained text, not a blank

def test_a_repeat_participants_delta_diffs_the_retained_text_when_the_index_moved_the_prose_out(harness):
    harness.monkeypatch.setattr(plan_record, "_PLAN_RECORD_MAX_BYTES", 6_000)
    harness.state["slots"] = _slots(("s1", "m/a"))
    sub = harness.install({"s1": CLEAN})
    ctx = harness.make_ctx()
    big = "Slide five carries the argument. " * 150
    _write(ctx, plan=big + "\nEnd A.", ask="council")  # revision 1 and ask a1
    _write(ctx, plan=big + "\nEnd B.")                   # revision 2: the index is past its bound, revision 1's prose moves out
    first = _record(harness)["revisions"][0]
    assert first["plan_inline"] is False and first["plan"] == "" and first["text_ref"]["path"]
    _call(ctx, ask=["s1"], message="again")              # a2: s1 answered about revision 1, so the delta is against its retained text
    delta = _content(sub, 1, "s1").split("## THE PLAN")[1].split("The current text follows whole.")[0]
    assert "-End A." in delta and "+End B." in delta and "+Slide five carries the argument." not in delta
    # The other direction: an unreadable retained source is named, never diffed as a blank.
    _write(ctx, plan=big + "\nEnd C.")                   # revision 3: revision 2's prose moves out
    second = _record(harness)["revisions"][1]
    assert second["plan_inline"] is False

    def corrupt(record):
        record["revisions"][1]["text_ref"]["sha256"] = "0" * 64  # the retained source no longer verifies on read
        return record

    plan_record._update_plan_record(harness.drive, "task-1", corrupt)
    _call(ctx, ask=["s1"], message="once more")          # a3: s1 answered a2 about revision 2
    delta = _content(sub, 2, "s1").split("## THE PLAN")[1].split("The current text follows whole.")[0]
    assert "retained source could not be read" in delta and second["text_ref"]["path"] in delta and "```diff" not in delta


# ---------------------------------------------------------------- Fable 3 / D2: the tool's read and the gate agree on an unreadable earlier state

def test_the_tools_read_and_the_gate_agree_that_an_unreadable_earlier_state_holds(harness):
    seed_result(harness.drive, "task-1", {"task_id": "task-1", "status": "running", "plan_review_state": {"schema_version": 99}})
    harness.install({})
    ctx = harness.make_ctx()
    text, _result = _write(ctx)
    assert "HELD: this task's earlier plan-review state cannot be read" in text and "PLAN_REVIEW_STATE_INVALID" in text
    assert "Nothing holds your final answer." not in text
    assert oh.plan_gate_decision(ctx, {}, enforcement="blocking")["status"] == "unreadable"
    # The other direction: a readable, closed earlier state holds nothing, in the tool and at the gate alike.
    seed_result(harness.drive, "task-2", {"task_id": "task-2", "status": "running", "plan_review_state": _v2(closed=True)})
    other = harness.make_ctx(task_id="task-2")
    text, _result = _write(other)
    assert "Nothing holds your final answer." in text and oh.plan_gate_decision(other, {}, enforcement="blocking")["status"] == "clear"


# ---------------------------------------------------------------- A3 / Fable 9: the owner's progress line speaks typed facts

def test_a_failed_seats_progress_line_carries_its_typed_tag_not_the_raw_provider_error(harness):
    raw = '{"nextActions": ["Retry the run", "Inspect the run"], "safeMessage": "harness_error", "detail": "' + "x" * 500 + '"}'
    harness.install({"s1": {"status": "error", "error": f"delegated review session run-e23 ended failed: {raw}", "failure_code": "run_failed",
                            "reported_cause": "the harness could not start a session for this profile",
                            "reset_at": "2026-10-16T00:00:00+00:00", "usage": {"cost": 0.0}}, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    text, _result = _write(ctx, ask="council")
    [line] = [row for row in harness.progress if "didn't answer" in row]
    assert line == ("📐 Plan: m/a didn't answer: failed[run_failed] — reported cause: "
                    '"the harness could not start a session for this profile" (reset at 2026-10-16T00:00:00+00:00).')
    assert "nextActions" not in line and "x" * 40 not in line
    assert "s1 (m/a): failed[run_failed] — reported cause:" in text, "the read view speaks the same typed tag"
    # The other direction: a failure that reported nothing typed keeps a bounded first line, still no retry coach.
    harness.install({"s1": {"status": "error", "error": f"transport died: {raw}", "usage": {"cost": 0.0}}, "s2": CLEAN, "s3": CLEAN})
    _call(ctx, ask="council", message="again")
    bare = [row for row in harness.progress if "didn't answer" in row][-1]
    assert bare.startswith("📐 Plan: m/a didn't answer: failed — transport died:") and len(bare) < 260
    assert "Under Blocking, outside Cyber Pro, an open objection of an owner-marked reviewer holds" in pr._SCHEMA["description"]
