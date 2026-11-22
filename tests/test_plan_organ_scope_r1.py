"""Scope review R1 (Astra F1–F3, Opus C1–C3, Fable advisories) — fix batch 1, pinned in both
directions through the real ToolContext, the real plan record, the real ledger and the organ's
own delivery builders (the review substrate is the fake of ``tests/test_plan_task_tool.py``).
Pure record rules live in ``tests/test_plan_record.py``; the web panel in
``web/tests/review_plan_record.test.js``; the gateway projection in ``tests/test_plan_record_gateway.py``."""
from __future__ import annotations

from ouroboros import owner_hurry as oh
from ouroboros import plan_record
from ouroboros.review_records import ReviewActorRecord
from ouroboros.tools import plan_ask
from tests._store_shared import seed_result
from tests.test_plan_gate import _v2
from tests.test_plan_task_tool import CLEAN, _call, _record, _reply, _slots, _user_text, _write, harness as _harness

harness = _harness  # noqa: F811 - pytest fixture re-export
PENDING = {"status": "pending", "raw_text": "", "operation_state": "pending_dispatch", "late_result_pending": True}


def _content(sub, call, seat):
    return _user_text(sub.calls[call]["request"].slot_messages[seat][-1]["content"])


def _settle(ctx, harness, request, seat, text, *, roster=2):
    slot = next(s for s in harness.state["slots"] if s.slot_id == seat)
    actor = ReviewActorRecord(slot_id=seat, model=slot.model, status="ok", raw_text=text, operation_id=f"op-{seat}-late",
                              operation_state="late_settled", usage={"cost": 0.01, "resolved_model": slot.model})
    plan_ask.settle_released_reply(ctx, request=request, task_id="task-1", actor=actor, slot=slot, settled_wave={}, roster_size=roster)


# ---------------------------------------------------------------- C1: a quoted object in a preamble is not the reply

def test_a_reply_behind_a_quoted_done_when_row_keeps_its_objection(harness):
    preamble = ('I re-read the bar:\n```json\n{"id": "c1", "claim": "exactly 5 slides", "support": "count them"}\n```\n'
                + _reply("Too thin.", objections=[("o1", "no story arc")]))
    harness.install({"s1": preamble, "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    text, result = _write(ctx, ask="council")
    assert result.meta["plan_open_objections"] == 1 and "s1 / o1" in text and "nothing holds it" in text
    s1 = next(p for p in _record(harness)["asks"][0]["participants"] if p["seat_id"] == "s1")
    assert s1["status"] == "answered" and s1["reply"]["prose"] == "Too thin." and s1["reply"]["objections"] == [{"id": "o1", "text": "no story arc"}]
    assert any("s1 answered" in line or "m/a answered — 1 objection" in line for line in harness.progress)


# ---------------------------------------------------------------- F1: a $0 pre-dispatch refusal never counts as a reached reviewer


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
    assert oh.plan_gate_decision(ctx, {})["status"] == "unreadable"
    # The other direction: a readable, closed earlier state holds nothing, in the tool and at the gate alike.
    seed_result(harness.drive, "task-2", {"task_id": "task-2", "status": "running", "plan_review_state": _v2(closed=True)})
    other = harness.make_ctx(task_id="task-2")
    text, _result = _write(other)
    assert "Nothing holds your final answer." in text and oh.plan_gate_decision(other, {})["status"] == "clear"


# ---------------------------------------------------------------- A3 / Fable 9: the owner's progress line speaks typed facts
