"""A Presence turn is never held for its review, so the review's whole feedback reaches its room later (#1738).

ARCHITECTURE §6 "Result review": Presence speech is never held; its owed review settles as a late mark. Through the
real Host, loop, review coordinator, terminal pipeline and maintenance pass (``replay_owed_late_reviews``, the server's
off-loop sweep), with only the model and the reviewer's answer scripted, the consumer here is the room's next turn: the
memory view it reads names the mark and its reader, and that reader returns every finding whole. The turn that spoke
stays as it ended: no further model call, no second review and no reopening. A review already settled when the turn
spoke, with no seat pending, owes the same duty when its outcome never reached its author whole; one its own review
call returned whole owes none. A call that does so after the duty was written, in a round a place event reopened, ends
the duty once a saved state holds its whole receipt (``review_waiting.acknowledge_delivery``), whatever then ends the
turn; a receipt short of whole or of saved ends nothing. A review the spent allowance refused is bound to the terminal
as recorded, and a refusal is no reviewer feedback.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import loop_tool_execution, review_ledger, review_substrate, review_waiting, store, working_checkpoint
from ouroboros.chronicle_store import ChronicleStore, is_late_review_mark
from ouroboros.late_review_marks import OWED_SET, replay_owed_late_reviews
from ouroboros.review_execution import ReviewAttemptResult
from ouroboros.review_wave import ReviewWave
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.test_environment import isolated_environment
from ouroboros.tools import review_result
from ouroboros.tools.chronicle import _memory_read
from ouroboros.tools.registry import ToolContext
from tests._memory_view_context import room_text
from tests._store_shared import as_install
from tests.test_presence_contact_wait import start as start_focus
from tests.test_presence_continuation import ANSWER, NEW_WORDS, author, deliver, parked, review, wait_for
from tests.test_presence_continuation import harness as harness
from tests.test_presence_continuation_bootstrap import finish, install_bootstrap_harness, tool_call
from tests.test_presence_continuation_host import _turn_id, poll, turn
from tests.test_presence_continuation_host import host as host
from tests.test_presence_owner_contact import contact_runtime as _contact_runtime, focus  # noqa: F401
from tests.test_presence_owner_stop import FOLLOW_UP, _follow_up_arrives, _holds
from tests.test_presence_owner_stop import owner_stop as owner_stop

contact_runtime = _contact_runtime

SECTION = "## Late review feedback (host)"
WORDS = [f"LATE-FINDING-{index:02d}" for index in range(15)]  # more than the twelve an earlier view cut
ANSWERED = json.dumps([{"item": f"item_{index}", "verdict": "FAIL", "severity": "critical" if index == 0 else "advisory",
                        "reason": word} for index, word in enumerate(WORDS)])


def reviewer_answers(monkeypatch):
    """The harness's held seat, answering with these findings."""
    held = review_substrate._review_route_executor

    def route(assignment, **kwargs):
        executor = held(assignment, **kwargs)
        execute = executor.execute

        def answered():
            return ReviewAttemptResult(message={"content": ANSWERED}, raw_text=ANSWERED, usage=execute().usage)

        executor.execute = answered
        return executor

    monkeypatch.setattr(review_substrate, "_review_route_executor", route)


def room_view(h, ref):
    """What the next turn of ``ref``'s room reads of it: the real view, under that turn's own metadata."""
    row = load_task_result(h.data, ref)
    return room_text(h.data, {"id": "next-turn", "chat_id": row["chat_id"], "metadata": row["metadata"]})


def assert_the_room_reads_every_finding(h, ref):
    wait_for(lambda: not replay_owed_late_reviews(h.data)["waiting"] and SECTION in room_view(h, ref),
             what="the late mark in its room")
    section = room_view(h, ref).split(SECTION, 1)[1]
    assert f"Late result review of task {ref}" in section
    row = load_task_result(h.data, ref)
    reader = ToolContext(repo_dir=h.repo, drive_root=h.data, task_id="next-turn", current_chat_id=row["chat_id"],
                         task_metadata=row["metadata"])
    shown = re.search(r"memory_read\(source_ref=(\{.*?\})\)", section).group(1)
    source = _memory_read(reader, source_ref=json.loads(shown))  # the call exactly as the view prints it
    assert [word for word in WORDS if word not in source] == []
    late = [mark for mark in ChronicleStore(h.data).active_marks(None) if is_late_review_mark(mark)]
    assert [mark["review"]["task_id"] for mark in late] == [ref]  # one mark, in this room, never a copy
    assert not any(word in room_text(h.data, {"id": "elsewhere", "chat_id": 1}) for word in WORDS)
    assert not replay_owed_late_reviews(h.data)["waiting"]  # discharged, and a further sweep adds nothing


def assert_the_spoken_turn_stayed(h, ref, calls):
    assert (poll(h, ref).json()["status"], poll(h, ref).json()["text"]) == ("completed", ANSWER)
    assert load_task_result(h.data, ref)["status"] == "completed"
    assert h.calls == calls and len(h.reviews) == 1  # no reopening and nothing bought
    assert "[Result review]" not in str(h.inputs)  # never held for it: this feedback is the room's


def test_a_required_review_still_running_when_the_turn_spoke_reaches_its_room_whole(host, monkeypatch):
    h = host
    reviewer_answers(monkeypatch)
    h.addressed, h.script = "", lambda _messages: deliver()
    spoken = turn(h, 42).json()
    ref = _turn_id(h, 42)
    assert (spoken["status"], spoken["outcome"], spoken["text"]) == ("completed", "message", ANSWER)
    wait_for(h.entered.is_set, what="the owed review's seat")
    wait_for(lambda: poll(h, ref).status_code == 200, what="the terminal")
    assert replay_owed_late_reviews(h.data)["waiting"] == 1 and SECTION not in room_view(h, ref)  # its seat still runs
    h.release.set()
    assert_the_room_reads_every_finding(h, ref)
    assert_the_spoken_turn_stayed(h, ref, 1)


def test_a_left_review_still_running_when_the_woken_author_spoke_reaches_its_room_whole(host, monkeypatch):
    h = host
    reviewer_answers(monkeypatch)
    h.script = author(deliver)
    ref = turn(h, 42).json()["continuation_ref"]
    wait_for(lambda: parked(h, ref), what="the review park")
    h.quick.add("telegram:bot-1:43")  # armed, never run: the follow-up wakes this author, which speaks
    turn(h, 43, NEW_WORDS)
    wait_for(lambda: poll(h, ref).status_code == 200, what="the woken author's speech")
    assert SECTION not in room_view(h, ref)  # the reviewer has not answered yet
    h.release.set()
    assert_the_room_reads_every_finding(h, ref)
    assert_the_spoken_turn_stayed(h, ref, 3)


def test_a_left_review_settled_before_its_author_spoke_reaches_its_room_whole(host, monkeypatch):
    h = host
    reviewer_answers(monkeypatch)
    h.script = author(deliver)
    ref = turn(h, 42).json()["continuation_ref"]
    wait_for(lambda: parked(h, ref), what="the review park")
    h.release.set()
    wait_for(lambda: poll(h, ref).status_code == 200, what="the author's speech")
    assert_the_room_reads_every_finding(h, ref)
    assert_the_spoken_turn_stayed(h, ref, 3)


def seats_answer_before_the_drain_leaves(monkeypatch):
    """The departed drain's first slice comes after its seats answered (the barrier only fixes that order)."""
    observe = ReviewWave.observe_detach

    def after_the_seat_answered(self, operation, observer):
        for entry in list(self.entries.values()):
            assert entry.event.wait(60)
        return observe(self, operation, observer)

    monkeypatch.setattr(ReviewWave, "observe_detach", after_the_seat_answered)


def test_a_required_review_settled_before_its_drain_left_reaches_its_room_whole(host, monkeypatch):
    """Its seat answered before the departed drain's first slice (the barrier only fixes that order): no seat is
    pending, the settled record is bound, and its outcome never reached its author, so it owes the same duty."""
    h = host
    reviewer_answers(monkeypatch)
    h.addressed, h.script = "", lambda _messages: deliver()
    h.release.set()
    seats_answer_before_the_drain_leaves(monkeypatch)
    spoken = turn(h, 42).json()
    ref = _turn_id(h, 42)
    assert (spoken["status"], spoken["outcome"], spoken["text"]) == ("completed", "message", ANSWER)
    wait_for(lambda: poll(h, ref).status_code == 200, what="the terminal")
    assert load_task_result(h.data, ref)["outcome_axes"]["review"]["aggregate_signals"] == ["FAIL"]  # settled, bound
    assert_the_room_reads_every_finding(h, ref)
    section = room_view(h, ref).split(SECTION, 1)[1]
    assert "that never reached its author" in section and "addressed message" not in section
    assert_the_spoken_turn_stayed(h, ref, 1)


def test_an_outcome_its_author_received_whole_keeps_no_mark(host, monkeypatch):
    """The falsifying side: its own review call returned every finding into its transcript, so the room owes nothing."""
    h = host
    reviewer_answers(monkeypatch)
    h.addressed = ""
    h.release.set()
    h.script = [review(), deliver()]
    spoken = turn(h, 42).json()
    ref = _turn_id(h, 42)
    assert (spoken["status"], spoken["text"]) == ("completed", ANSWER)
    wait_for(lambda: poll(h, ref).status_code == 200, what="the terminal")
    assert all(word in str(h.inputs[1]) for word in WORDS) and h.calls == 2 and len(h.reviews) == 1
    assert replay_owed_late_reviews(h.data) == {"settled": 0, "waiting": 0, "retired": 0, "failed": 0}
    assert SECTION not in room_view(h, ref)


DRAFT = "The status report draft: two of the three checks have run."
REVISED = "The status report is complete: all three checks passed, and the backup window is named."
FORCED = "The round limit's best answer: the report is ready, its findings are unaddressed."


def test_speech_revised_after_its_own_review_is_the_authors_and_buys_nothing_at_a_spent_allowance(host, monkeypatch):
    """Required with one paid result cycle per root: the author's own review of its draft spent it and returned
    whole, and the answer it then speaks is that review's revision, its author's to accept (``_revised``): no host
    review is bought or refused at $0, nothing holds that speech, the terminal's review axis keeps the earlier PASS
    as the accepted earlier revision's rather than reading ``skipped``, the ledger holds that one review, and
    nothing is owed as a mark."""
    h = host
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    h.addressed = ""  # nothing to leave the draft's review for: its own call returns that outcome whole
    h.release.set()
    h.script = [review(answer=DRAFT), deliver()]
    spoken = turn(h, 42).json()
    ref = _turn_id(h, 42)
    assert (spoken["status"], spoken["outcome"], spoken["text"]) == ("completed", "message", ANSWER)
    wait_for(lambda: poll(h, ref).status_code == 200, what="the terminal")
    axis = load_task_result(h.data, ref)["outcome_axes"]["review"]
    assert (axis["status"], axis["source"], axis["aggregate_signals"], axis["author_finish"]) == (
        "pass", "result_review", ["PASS"], "previous_revision_accepted")
    rows = review_ledger.recent_records(h.data, ref, limit=50, surface="result")
    assert [(row.get("verdict") or {}).get("aggregate") for row in rows] == ["PASS"], "no refusal, no second wave"
    assert replay_owed_late_reviews(h.data) == {"settled": 0, "waiting": 0, "retired": 0, "failed": 0}
    assert SECTION not in room_view(h, ref)
    assert_the_spoken_turn_stayed(h, ref, 2)


def host_owed_record(root, task_id):
    """The id of the review the host owed this turn's first answer (``chosen_by="host"``)."""
    rows = review_ledger.recent_records(root, task_id, limit=50, surface="result")
    [record_id] = [row["record_id"] for row in rows
                   if (review_ledger.load_record(root, row["record_id"])["panel"] or {}).get("chosen_by") == "host"]
    return record_id


def unread_duties(root):
    """The record ids this root owes an unread duty for."""
    with store.open_store(root, write=False) as txn:
        rows = txn.set_rows(OWED_SET).values()
    return [row["left_review"]["record_id"] for row in rows if (row.get("left_review") or {}).get("left") == "unread"]


def reopened_turn(monkeypatch, r, steps):
    """A Place focus turn through the production agent bootstrap, loop, admission and final drain (only inference and
    the reviewer's transport are scripted). The host's review of its answer settles before the departed drain leaves,
    so the delivery binds it unread and owes its mark; a later event of the conversation arrives while that answer's
    call is out, so the final drain reopens the unsent answer (CREATING_SKILLS "Places"). ``steps`` answer the calls of
    the reopened round, in order. ``h.deliveries``: the call behind each delivery step; ``h.owed_at_call``: the unread
    duties each model call found."""
    h = install_bootstrap_harness(monkeypatch, r.root, review_mode="required")
    reviewer_answers(monkeypatch)
    h.addressed = ""
    h.release.set()
    seats_answer_before_the_drain_leaves(monkeypatch)
    h.deliveries, h.owed_at_call, h.arrivals = [], {}, []
    deliver = review_result.result_review_at_delivery

    def counted(ctx, content, trace, hold):
        h.deliveries.append(h.calls)
        return deliver(ctx, content, trace, hold)

    monkeypatch.setattr(review_result, "result_review_at_delivery", counted)

    def model(messages):
        from ouroboros.artifacts import store_task_artifact_bytes
        from ouroboros.llm_attempt import require_physical_dispatch_window

        require_physical_dispatch_window()  # the production gate before any physical send
        h.owed_at_call[h.calls] = unread_duties(r.root)
        if h.calls > 1:
            return steps[h.calls - 2](messages)
        # The report the turn retained is the result the host reviews (#1731), as in the Host harness.
        ctx = h.agents[0].tools._ctx
        store_task_artifact_bytes(ctx.drive_root, ctx.task_id, "report.txt", ANSWER.encode())
        h.arrivals.append(_follow_up_arrives(r))  # while this answer's call is out
        return {"content": ANSWER, "tool_calls": []}

    h.script = model
    return h


def re_review(_messages):
    """The author re-issues its review of the same answer: its call returns the host's settled record whole at $0."""
    return review(answer=ANSWER)


def select(text):
    return lambda _messages: finish("final", message=text)  # a reopened answer is selected explicitly


def a_further_tool(_messages):
    return {"content": None, "tool_calls": [tool_call("knowledge_list", {}, "look")]}


def budget_refuses(_messages):
    """That call's reservation is refused before a byte is sent (a zero global limit, as in the owner-Stop suite):
    a priced call, since a reached global limit refuses paid admission only (owner 2A)."""
    from ouroboros.usage_accounting import AttemptRequest, reserve_attempt

    reserve_attempt(AttemptRequest(model="budget-refusal", provider="openai", reservation_usd=0.01,
                                   global_limit_usd=0, allow_live_fetch=False))
    raise AssertionError("a zero global limit refuses this reservation")


def forced_final(_messages):
    return {"content": FORCED, "tool_calls": []}


def assert_the_review_call_returned_it_whole(h, call):
    """In the tool row of the review call itself: the host's decision rows carry findings of their own."""
    [*_, returned] = [str(row.get("content")) for row in h.inputs[call - 1] if row.get("role") == "tool"]
    assert [word for word in WORDS if word not in returned] == []


def assert_no_mark_and_the_record_kept(h, r, tid, first):
    assert unread_duties(r.root) == []
    assert replay_owed_late_reviews(r.root) == {"settled": 0, "waiting": 0, "retired": 0, "failed": 0}
    assert SECTION not in room_view(h, tid)
    assert [mark for mark in ChronicleStore(r.root).active_marks(None) if is_late_review_mark(mark)] == []
    kept = review_ledger.load_record(r.root, first)  # the duty ended; the review's own evidence did not
    assert (kept["state"], kept["verdict"]["aggregate"]) == ("settled", "FAIL")
    verdict = kept["verdict"]
    assert [word for word in WORDS if word not in str(verdict["critical_findings"] + verdict["advisory_findings"])] == []
    assert all(row.get("source_refs") for row in kept["rows"]) and kept["waiting"]["seen_outcomes"]


def assert_the_mark_stayed(h, r, tid, first):
    assert unread_duties(r.root) == [first]
    assert_the_room_reads_every_finding(h, tid)
    section = room_view(h, tid).split(SECTION, 1)[1]
    assert "that never reached its author" in section and first in section
    [mark] = [mark for mark in ChronicleStore(r.root).active_marks(None) if is_late_review_mark(mark)]
    assert mark["review"]["subject"]["record_id"] == first
    assert unread_duties(r.root) == []  # the mark discharged it


@pytest.mark.parametrize("reopened", ["received", "no_review_call", "another_record_received"])
def test_an_unread_duty_follows_what_its_author_received_after_a_place_event_reopened_its_answer(
        contact_runtime, monkeypatch, reopened):
    """In the reopened round (``reopened_turn``) the author

    - ``received``: re-issues its review of the same answer; the saved state holding that call's whole receipt ends
      the duty, before anything is delivered again, so no mark claims the review never reached its author;
    - ``no_review_call``: keeps its answer without a review call. The host's decision rows are in its input, but they
      are no certified delivery of the record, so the duty stays and the room reads every finding;
    - ``another_record_received``: reviews a revised answer (a second paid wave) and receives that record whole:
      that receipt is not the first record's, so the first record's duty stays and its room reads every finding."""
    r = contact_runtime
    final = REVISED if reopened == "another_record_received" else ANSWER
    h = reopened_turn(monkeypatch, r, {"received": [re_review, select(ANSWER)], "no_review_call": [select(ANSWER)],
                                       "another_record_received": [lambda _m: review(answer=REVISED), select(REVISED)]
                                       }[reopened])
    ended = start_focus(r, h).result.result(timeout=90)
    tid = h.agents[0].tools._ctx.task_id
    assert (ended.status, ended.text) == ("completed", final)
    assert [holder for _outcome, holder in h.arrivals] == [tid]
    assert _holds(r.root, tid) == [(1, "prose", "place_mail")]  # the one hold: the arrival, never the review
    assert str(h.inputs[1]).count(FOLLOW_UP) == 1 and "[Result review]" not in str(h.inputs)
    first = host_owed_record(r.root, tid)
    assert h.owed_at_call[2] == [first] and h.deliveries[0] == 1
    if reopened == "received":
        assert h.calls == 3 and len(h.reviews) == 1 and h.deliveries == [1, 3]  # the re-issued call reused it at $0
        assert_the_review_call_returned_it_whole(h, 3)
        assert h.owed_at_call[3] == []  # retired by the saved receipt, before the next delivery ran
        assert_no_mark_and_the_record_kept(h, r, tid, first)
        return
    assert (h.calls, len(h.reviews)) == ((2, 1) if reopened == "no_review_call" else (3, 2))
    assert_the_mark_stayed(h, r, tid, first)


@pytest.mark.parametrize("ending", ["stop_at_the_next_dispatch", "budget_after_a_further_call", "round_limit_final"])
def test_a_saved_whole_receipt_ends_the_duty_whatever_then_ends_the_turn(owner_stop, monkeypatch, ending):
    """No delivery step follows the receipt on any of these exits (owner Stop, Wrap up, a budget refusal, the forced
    final of the round limit), so the duty has to end at the acknowledgement's own checkpoint: the ready save before the
    next dispatch, or the loop's terminal save. "Received" is the existing exposure of ``_outcome_received``: the whole
    result entered a saved transcript. ``*_at_the_next_*`` pin that it holds even when no model call read that
    transcript afterwards (a refused dispatch): the record and its sources keep the outcome, the room gets no mark."""
    r, press = owner_stop
    stops = []

    def stop_during(step):
        def stopping(messages):
            stops.append(press(h.agents[0].tools._ctx.task_id, cascade=True))
            return step(messages)
        return stopping

    if ending in {"stop_at_the_next_dispatch", "wrap_up_final"}:  # pressed as the review's whole result is returned
        publish = review_waiting.publish_delivery
        policy = "finalize_then_cancel" if ending == "wrap_up_final" else "immediate"

        def publishing(ctx, text, record, rendered_rows=None):
            returned = publish(ctx, text, record, rendered_rows)
            stops.append(press(ctx.task_id, cascade=True, policy=policy))
            return returned

        monkeypatch.setattr(review_waiting, "publish_delivery", publishing)
    if ending == "round_limit_final":
        monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "2")  # the reopened review is the last ordinary round
    h = reopened_turn(monkeypatch, r, [re_review, *{
        "stop_at_the_next_dispatch": [],
        "stop_after_a_further_call": [stop_during(a_further_tool)],
        "wrap_up_final": [forced_final],
        "budget_at_the_next_call": [budget_refuses],
        "budget_after_a_further_call": [a_further_tool, budget_refuses],
        "round_limit_final": [forced_final]}[ending]])
    start_focus(r, h).result.exception(timeout=90)  # a Stop ends the turn raising; the row says how it ended
    tid = h.agents[0].tools._ctx.task_id
    row = load_task_result(r.root, tid)
    assert (row["status"], row["reason_code"]) == {
        "stop": ("cancelled", "owner_requested_finalization"), "wrap": ("completed", "owner_requested_finalization"),
        "budget": ("completed", "budget_exhausted"), "round": ("completed", "round_limit")}[ending.split("_")[0]]
    first = host_owed_record(r.root, tid)
    assert h.owed_at_call[2] == [first] and len(h.reviews) == 1
    assert h.deliveries == [1], "no delivery step ran after the receipt"
    assert [stop.status_code for stop in stops] == ([202] if ending.split("_")[0] in {"stop", "wrap"} else [])
    if ending != "wrap_up_final":  # its one tool-less call reshapes the transcript; the saved cursor below still names it
        assert_the_review_call_returned_it_whole(h, 3)
    assert_no_mark_and_the_record_kept(h, r, tid, first)


@pytest.mark.parametrize("gap", ["partial_view", "failed_saves", "failure_exit"])
def test_a_receipt_short_of_whole_or_of_saved_keeps_the_duty_and_its_mark(contact_runtime, monkeypatch, gap):
    """The author re-issues its review, but no saved whole receipt results: the transcript shortened the result
    (``result_partial``), the publication carrying its receipt was lost, every save after it failed (the existing
    "a failed save leaves the mail unread"), or the loop left by a failure before its checkpoint. The acknowledgement
    never ran, so the duty stays and the room reads every finding once the task ended."""
    r = contact_runtime
    steps = [re_review, select(ANSWER)]
    if gap == "partial_view":
        deliver = loop_tool_execution._deliver_tool_results

        def shortened(ctx, results, *args, **kwargs):
            views = deliver(ctx, results, *args, **kwargs)
            for result, view in zip(results, views):
                if result["fn_name"] == "review_change":
                    view.update(result=str(view["result"])[:120], result_partial=True)
            return views

        monkeypatch.setattr(loop_tool_execution, "_deliver_tool_results", shortened)
    if gap == "lost_publication":
        monkeypatch.setattr(review_waiting, "publish_delivery", lambda _ctx, text, _record, rendered_rows=None: text)
    if gap == "failed_saves":
        monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "2")  # the forced final takes no tool batch, whose save would hold it
        steps = [re_review, forced_final]
        publish, write, published = review_waiting.publish_delivery, working_checkpoint._write, []

        def publishing(ctx, text, record, rendered_rows=None):
            published.append(record["record_id"])
            return publish(ctx, text, record, rendered_rows)

        def failing(*args, **kwargs):
            if published:
                raise OSError("no space left on device")
            return write(*args, **kwargs)

        monkeypatch.setattr(review_waiting, "publish_delivery", publishing)
        monkeypatch.setattr(working_checkpoint, "_write", failing)
    h = reopened_turn(monkeypatch, r, steps)
    if gap == "failure_exit":
        save_ready = working_checkpoint.save_ready

        def dying(limit_ctx):
            if h.calls == 2:
                assert limit_ctx.tools._ctx._pending_content_acks  # its acknowledgement is staged, nothing saved it
                raise RuntimeError("the worker fails before this checkpoint")
            return save_ready(limit_ctx)

        monkeypatch.setattr(working_checkpoint, "save_ready", dying)
    start_focus(r, h).result.exception(timeout=90)
    tid = h.agents[0].tools._ctx.task_id
    first = host_owed_record(r.root, tid)
    assert h.owed_at_call[2] == [first] and len(h.reviews) == 1
    assert load_task_result(r.root, tid)["status"] == ("failed" if gap == "failure_exit" else "completed")
    if gap == "partial_view":  # the review call's own row: the host's decision rows carry findings of their own
        assert not [word for word in WORDS if word in [str(row.get("content")) for row in h.inputs[2]
                                                       if row.get("role") == "tool"][-1]]
    assert_the_mark_stayed(h, r, tid, first)


def die_before_its_checkpoint(root):
    """Child process of the next test: ``reopened_turn``, whose process ends without any teardown once the review's
    whole result entered the transcript and before the ready save that would acknowledge it."""
    from ouroboros import places
    from ouroboros.presence_admission import admit_presence_turn
    from ouroboros.presence_runner import PresenceTurnEvent

    patch = pytest.MonkeyPatch()
    place = places.connect_place(root, connector="bridge", provider="chat", account_id="a", conversation_id="r",
                                 wakes=["mention"])
    r = SimpleNamespace(root=root, place=place)
    h = reopened_turn(patch, r, [re_review])
    r.admission = admit_presence_turn(drive_root=root, authenticated_transport_skill="bridge", provider="chat",
                                      account_id="a", conversation_id="r")
    r.event = PresenceTurnEvent("event1", "chat", "a", "r", "", "chat:a:r", {"platform_actor_id": "person"},
                                {"title": "A shared thread"}, {"url": "https://example.test/room/thread#comment-7"},
                                "hello", facts=("mention",))
    save_ready = working_checkpoint.save_ready

    def dying(limit_ctx):
        if h.calls == 2:
            assert limit_ctx.tools._ctx._pending_content_acks
            (root.parent / "died.json").write_text(json.dumps({
                "task_id": limit_ctx.tools._ctx.task_id, "owed": unread_duties(root)}), encoding="utf-8")
            os._exit(137)
        return save_ready(limit_ctx)

    patch.setattr(working_checkpoint, "save_ready", dying)
    start_focus(r, h).result.exception(timeout=120)
    os._exit(1)  # never reached: the process dies at the ready save


def test_a_process_that_dies_before_the_checkpoint_keeps_the_duty_and_its_mark(tmp_path):
    """A real process death between the transcript insertion and the ready save: nothing acknowledged the receipt and
    no model call read it. The owed row and the record are what the crash left: the duty waits for the task's end,
    then the room reads every finding (``working_checkpoint``: unsaved content "was never acknowledged")."""
    repo = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    code = ("import pathlib, sys; from tests.test_presence_late_review_delivery import die_before_its_checkpoint as die; "
            "die(pathlib.Path(sys.argv[1]))")
    env = isolated_environment(tmp_path, repo)
    assert Path(env["OUROBOROS_DATA_DIR"]) == data
    done = subprocess.run([sys.executable, "-c", code, str(data)], cwd=repo, env=env, capture_output=True, text=True, timeout=240)
    (tmp_path / "child.log").write_text(done.stdout + done.stderr, encoding="utf-8")
    assert done.returncode == 137, done.stdout[-2000:] + done.stderr[-2000:]
    died = json.loads((tmp_path / "died.json").read_text(encoding="utf-8"))
    tid = died["task_id"]
    with as_install(data):
        [first] = died["owed"]
        assert unread_duties(data) == [first]
        assert not review_ledger.load_record(data, first).get("waiting")  # no acknowledgement ran
        assert replay_owed_late_reviews(data)["waiting"] == 1  # its task never ended: owed, nothing lost
        # The host's orphan reconciler marks the lost stack; nothing restarts it.
        write_task_result(data, tid, "failed", reason_code="orphaned_running_after_worker_restart",
                          status_reconciled_from="running")
        assert_the_mark_stayed(SimpleNamespace(data=data, repo=repo), SimpleNamespace(root=data), tid, first)


def test_retiring_an_unread_duty_takes_that_records_row_alone(tmp_path):
    """The exact side of that retirement, over the real owed store: only this task's unread row of the received
    record goes. An unwaited wave's duty (which ``ended_left_review`` settles on its own terms), another record's
    unread duty and another task's unread duty of the same record id stay owed; a second retirement finds nothing."""
    from ouroboros import store
    from ouroboros.late_review_marks import OWED_SET
    from ouroboros.review_pause import owe_unwaited_review, retire_unread_review

    def ctx(task_id):
        return ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=task_id)

    for task_id, record_id, left in (("turn", "rl-a", "unread"), ("turn", "rl-b", "unread"),
                                     ("turn", "rl-c", "unwaited"), ("other", "rl-a", "unread")):
        assert owe_unwaited_review(ctx(task_id), surface="result", retry_key=f"key-{record_id}", record_id=record_id,
                                   left=left)

    def owed():
        with store.open_store(tmp_path, write=False) as txn:
            return sorted(tuple(row["left_review"][key] for key in ("task_id", "record_id", "left"))
                          for row in txn.set_rows(OWED_SET).values())

    assert retire_unread_review(ctx("turn"), surface="result", record_id="rl-c") is False  # unwaited: not this duty
    assert retire_unread_review(ctx("turn"), surface="result", record_id="rl-a") is True
    assert retire_unread_review(ctx("turn"), surface="result", record_id="rl-a") is False  # already retired
    assert owed() == [("other", "rl-a", "unread"), ("turn", "rl-b", "unread"), ("turn", "rl-c", "unwaited")]


@pytest.mark.parametrize("delivered", ["none", "pending_then_partial",
                                      "mail_left_after_unwaited"])
def test_saved_wait_cursors_do_not_prove_receipt_of_a_settled_presence_outcome(tmp_path, monkeypatch, delivered):
    """Cursors prove no whole receipt. A partial result keeps the wave's duty; a mail-left duty stays
    independent even after whole feedback, while a new unread-only duty can retire at its checkpoint."""
    from ouroboros import review_pause
    from ouroboros.review_pause import ended_left_review, owe_unwaited_review
    from ouroboros.working_checkpoint import flush_content_acks
    from tests.test_review_ledger import BOTH, _raw
    from tests.test_review_waiting import _author, _duties, _receipt, _result_record

    monkeypatch.setattr(review_result, "_blocking_enforcement", lambda: False)
    ctx = _author(tmp_path)
    record = _result_record(tmp_path, settled=("a",) if delivered in {
        "pending_then_partial", "mail_left_then_paused_rejoin"} else ("a", "b"))
    record = review_ledger.revise_record(tmp_path, record["record_id"], lambda row: {
        **row, "fingerprints": {**row.get("fingerprints", {}), "retry_key": "key-" + row["record_id"]}})
    trace = {"tool_calls": []}
    independent = None
    if delivered in {"mail_left", "mail_left_then_paused_rejoin", "mail_left_after_unwaited"}:
        retry_key = record["fingerprints"]["retry_key"]
        if delivered == "mail_left_after_unwaited":
            assert owe_unwaited_review(ctx, surface="result", retry_key=retry_key, record_id=record["record_id"])
        ctx._review_detached_operations = [{"retry_key": retry_key, "reason": review_pause.ADDRESSED_EVENT}]
        monkeypatch.setattr(review_pause, "notify_when_settled", lambda *_a, **_kw: True)
        assert review_pause.watch_left_review(ctx, surface="result", retry_key=retry_key,
            record_id=record["record_id"], reissue="review_change(review_reference=...)")
        with store.open_store(tmp_path, write=False) as txn:
            [independent] = list(txn.set_rows(OWED_SET).values())
        assert independent["left_review"].get("left") is None  # mail strengthens an earlier convenience duty
        if delivered == "mail_left_then_paused_rejoin":
            # The full-argument review_change pending-rejoin path uses these same two
            # writers after owner Pause: no new mail-left watcher, then an unwaited duty.
            assert record["state"] == "pending"
            ctx._review_detached_operations = [{"retry_key": retry_key, "reason": "owner_pause"}]
            assert review_pause.watch_left_review(ctx, surface="result", retry_key=retry_key,
                record_id=record["record_id"], reissue="review_change(review_reference=...)") is None
            assert owe_unwaited_review(ctx, surface="result", retry_key=retry_key,
                                       record_id=record["record_id"])
            record = review_ledger.settle_from_collected(tmp_path, record["record_id"],
                [_raw("b", "b", parts=BOTH, operation_id="b")])
    if delivered == "pending_then_partial":
        assert owe_unwaited_review(ctx, surface="result", retry_key=record["fingerprints"]["retry_key"],
                                   record_id=record["record_id"])
    if delivered not in {"none", "mail_left", "mail_left_then_paused_rejoin", "mail_left_after_unwaited"}:
        receipt = _receipt(ctx, record, record["rows"])
        trace["tool_calls"].append({"tool_result_meta": {"review_delivery": receipt}})
        review_waiting.acknowledge_delivery(ctx, receipt)
        flush_content_acks(ctx)
        if delivered == "earlier_outcome":
            record = review_ledger.revise_record(tmp_path, record["record_id"], lambda row: {
                **row, "rows": [{**row["rows"][0], "status": "continued"}, *row["rows"][1:]]})
        elif delivered == "pending_then_partial":
            record = review_ledger.settle_from_collected(tmp_path, record["record_id"],
                [_raw("b", "b", parts=BOTH, operation_id="b")])
            trace["tool_calls"].append({"result_partial": True,
                                      "tool_result_meta": {"review_delivery": _receipt(ctx, record, record["rows"])}})
        else:
            record = review_ledger.load_record(tmp_path, record["record_id"])
        assert record["waiting"]  # both a complete receipt and mere cursor information make this truthy
    assert record["state"] == "settled"
    assert review_result._outcome_received(ctx, trace, record) is (delivered == "whole")
    assert review_result._presence_result_review(ctx, "answer", trace,
        SimpleNamespace(diff_sha=record["subject"]["diff_sha"]), record, owed=False) is False
    if delivered == "whole":
        assert _duties(tmp_path) == []
    elif delivered in {"mail_left", "mail_left_then_paused_rejoin", "mail_left_after_unwaited"}:
        # The delivery attempt must not replace the existing mail-left room duty with unread-only feedback.
        with store.open_store(tmp_path, write=False) as txn:
            assert list(txn.set_rows(OWED_SET).values()) == [independent]
        review_waiting.acknowledge_delivery(ctx, _receipt(ctx, record, record["rows"]))
        flush_content_acks(ctx)
        with store.open_store(tmp_path, write=False) as txn:
            [remaining] = list(txn.set_rows(OWED_SET).values())
        assert remaining["left_review"].get("left") is None
        assert remaining["left_review"]["record_id"] == record["record_id"]
        write_task_result(tmp_path, ctx.task_id, "completed", chat_id=1)
        # This ledger-only fixture has no live controller/attempt. Exercise its crash collector;
        # keeping pytest's actual PID alive would correctly leave the incomplete roster waiting.
        from ouroboros import review_operation
        monkeypatch.setattr(review_operation, "controller_state", lambda _controller: "dead")
        assert replay_owed_late_reviews(tmp_path)["settled"] == 1
        [mark] = [mark for mark in ChronicleStore(tmp_path).active_marks(None) if is_late_review_mark(mark)]
        assert mark["review"]["subject"]["record_id"] == record["record_id"]
    else:
        expected_left = "unwaited" if delivered == "pending_then_partial" else "unread"
        assert _duties(tmp_path) == [(record["record_id"], expected_left)]
        with store.open_store(tmp_path, write=False) as txn:
            [duty] = list(txn.set_rows(OWED_SET).values())
        assert ended_left_review(dict(duty["left_review"])) != "collected"
