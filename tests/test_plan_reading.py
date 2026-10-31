"""The council reads the request before any plan: ``plan_task(ask=...)`` with no revision is a
reading. It travels the ordinary ask — participants, the intention ledger, the mailbox — carrying
the objective with its author, the owner's words and the room but no plan; its replies are advice
that move no objection in any mode, while the same participant objecting to a stored revision holds
as before. A reading of declared own-body work hears every marked reviewer under Blocking without
spending the first body revision's ask. Each rule is pinned both ways through the real ToolContext,
the real plan record and the fake review substrate of ``tests/test_plan_task_tool.py``; the last two
tests pin the cached prefix of a participant's packet."""
from __future__ import annotations

from ouroboros import owner_hurry as oh
from ouroboros import plan_record
from ouroboros.tools import plan_packet
from tests.test_plan_task_tool import CLEAN, _call, _record, _reply, _user_text, _write
from tests.test_plan_task_tool import harness as _harness

harness = _harness  # noqa: F811 - pytest fixture re-export


def _sent(sub, call, seat):
    return _user_text(sub.calls[call]["request"].slot_messages[seat][-1]["content"])


def test_a_reading_on_an_empty_record_asks_the_council_and_records_one_ask_without_a_revision(harness):
    sub = harness.install({"s1": _reply("It reads as a five-slide story."), "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()
    text, result = _call(ctx, ask="council", message="How do you read the request?")
    assert (result.status, result.meta["plan_effect"], result.meta["plan_revision"]) == ("ok", "ask", 0)
    assert [s.slot_id for s in sub.calls[0]["slots"]] == ["s1", "s2", "s3"]
    record = _record(harness)
    assert record["revisions"] == [] and [a["ask_id"] for a in record["asks"]] == ["a1"]
    assert "revision" in record["asks"][0] and record["asks"][0]["revision"] is None
    assert plan_record.validate_record(record) == record
    content = _sent(sub, 0, "s1")
    head, marker, tail = content.partition("## THE PLAN")
    assert marker and "## TASK OBJECTIVE\n\nDeliver the thing\n\nThe objective above: " in head
    assert tail.startswith("\n\nNo plan yet: Ouroboros asks how you read the request before it writes one.")
    for absent in ("### Goal", "### done_when", "### affected_paths", "Revision "):
        assert absent not in content, absent
    assert "How do you read the request?" in content
    assert "Ask a1 on the reading of the request: sent to s1" in text
    assert any("ask a1 on the reading of the request — sending to 3 participants" in line for line in harness.progress)
    from ouroboros.review_ledger import load_record

    ledger = load_record(harness.drive, record["asks"][0]["record_id"])
    assert ledger["subject"]["plan_revision"] is None and ledger["brief"]["scope"] == "the reading of the request"
    # plan_task() with no arguments stays the free read: it shows the reading and calls nobody.
    read, read_result = _call(ctx)
    assert read_result.meta["plan_effect"] == "read" and len(sub.calls) == 1
    assert read.startswith("No plan revision is recorded.") and "- ask a1 on the reading of the request" in read
    assert "s1: saw the reading of the request, last answered in ask a1" in read and "It reads as a five-slide story." in read
    # On a record with nothing in it, the read is the lead alone, as before.
    empty, _ = _call(harness.make_ctx(task_id="task-2"))
    assert empty.startswith("No plan revision is recorded.") and "## Council" not in empty
    assert _record(harness, "task-2") == plan_record.empty_record()


def test_a_readings_objection_never_holds_and_the_same_objection_on_a_revision_holds(harness):
    sub = harness.install({"s1": _reply("Wrong direction.", objections=[("o1", "the owner asked for a story, not slides")]),
                           "s2": CLEAN, "s3": CLEAN})
    ctx = harness.make_ctx()  # Blocking (the harness's enforcement)
    text, result = _call(ctx, ask="council")
    assert result.meta["plan_holding"] == 0 and result.meta["plan_open_objections"] == 0
    assert "listed as objections in this reading — advice, never a standing objection: s1/o1" in text
    assert "Nothing holds your final answer." in text
    decision = oh.plan_gate_decision(ctx, {}, enforcement="blocking")
    assert decision["allow"] is True and decision["holding"] == [] and decision["open_objections"] == []
    # The same seat and the same objection, stated about a stored revision, holds under Blocking.
    text, result = _write(ctx, ask=["s1"], message="Here is the plan.")
    assert result.meta["plan_holding"] == 1 and "HELD" in text
    [objection] = plan_record.open_objections(_record(harness))
    assert (objection["id"], objection["revision"], objection["holds"]) == ("o1", 1, True)
    assert oh.plan_gate_decision(ctx, {}, enforcement="blocking")["allow"] is False
    assert len(sub.calls) == 2


def test_after_a_reading_the_first_plan_ask_sends_the_plan_whole_and_continues_the_seat(harness):
    sub = harness.install({"s1": _reply("A story, five slides."), "s2": _reply("Agreed with s1."), "s3": CLEAN})
    ctx = harness.make_ctx()
    _call(ctx, ask="council", message="Read it first.")
    _write(ctx, ask=["s1"], message="Now the plan.")
    transcript = sub.calls[1]["request"].slot_messages["s1"]
    assert [m["role"] for m in transcript] == ["system", "user", "assistant", "user"]  # its reading continues
    assert "A story, five slides." in transcript[2]["content"]
    user = _user_text(transcript[-1]["content"])
    assert "You see this plan for the first time; it follows whole." in user and "exactly 5 slides" in user
    assert "Changes since revision" not in user and "No plan yet" not in user
    assert "### Ask a1 (about the reading of the request," in user and "Agreed with s1." in user
    assert _record(harness)["asks"][1]["revision"] == 1


def test_a_declared_body_reading_hears_every_marked_reviewer_and_spends_no_first_body_ask(harness):
    sub = harness.install({})
    ctx = harness.make_ctx(active_workspace=False)
    ctx.task_contract["context_requires_self_body_docs"] = True
    text, _ = _call(ctx, ask=["s2"], message="Read it first.")
    assert [s.slot_id for s in sub.calls[0]["slots"]] == ["s2", "s1", "s3"]
    ask = _record(harness)["asks"][0]
    assert ask["revision"] is None and ask["body_first_ask"] is True and ask["host_added"] == ["s1", "s3"]
    assert "This reading concerns Ouroboros's own body" in text
    system_prompt = _user_text(sub.calls[0]["request"].slot_messages["s2"][0]["content"])
    assert "## BIBLE.md (constitution" in system_prompt and "## ARCHITECTURE.md" in system_prompt
    # A later reading names its participants like any later ask: the first reading heard everyone.
    _call(ctx, ask=["s2"], message="One more question about the request.")
    assert [s.slot_id for s in sub.calls[1]["slots"]] == ["s2"] and _record(harness)["asks"][1]["body_first_ask"] is False
    # The first ask about a body revision still widens: the readings spent nothing.
    _write(ctx, affected_paths=["ouroboros/loop.py"], ask=["s2"])
    assert [s.slot_id for s in sub.calls[2]["slots"]] == ["s2", "s1", "s3"] and _record(harness)["asks"][2]["body_first_ask"] is True
    # The other direction: without the declaration a reading carries no pack and widens nothing.
    _call(harness.make_ctx(active_workspace=False, task_id="task-2"), ask=["s2"])
    assert [s.slot_id for s in sub.calls[3]["slots"]] == ["s2"] and _record(harness, "task-2")["asks"][0]["body_first_ask"] is False
    assert "## BIBLE.md" not in _user_text(sub.calls[3]["request"].slot_messages["s2"][0]["content"])


# ---------------------------------------------------------------- the cached prefix of a participant's packet

# ``build_participant_content`` of an ask about revision 1, byte for byte as the code before the reading wrote it.
_BEFORE = (
    "## TASK OBJECTIVE\n\nDeliver the thing\n\n## OWNER WORDS\n\nmake it good\n\n## OWN ROOM DIALOGUE\n\nrows\n\n## THE PLAN\n\n"
    "You see this plan for the first time; it follows whole.\n\nRevision 1 (recorded 2026-10-15T01:00:00+00:00; changes "
    "Ouroboros's own body: no)\n\n### Goal\n\nShip the deck\n\n### done_when (what acceptance will judge; id · claim · how it "
    "will be shown)\n\n```json\n[\n  {\n    \"claim\": \"five slides\",\n    \"id\": \"c1\",\n    \"support\": \"count them\"\n  }\n]"
    "\n```\n\n### affected_paths (files the work will change; [] = none)\n\n```json\n[]\n```\n\n### Plan\n\nOutline first.\n\n"
    "## THE COUNCIL SO FAR (since the ask you last answered in)\n\nNothing yet: this is the first ask on this plan and every "
    "first answer is blind.\n\n## YOUR OPEN OBJECTIONS\n\nYou have no open objection on this plan.\n\n## WHAT OUROBOROS ASKS "
    "YOU NOW\n\nFirst look.\n\n## ATTACHED EVIDENCE\n\n(nothing attached)\n"
)
_REVISION = {"n": 1, "ts": "2026-10-15T01:00:00+00:00", "goal": "Ship the deck", "plan": "Outline first.",
             "done_when": [{"id": "c1", "claim": "five slides", "support": "count them"}], "affected_paths": [], "body": False}


def _packet(revision=_REVISION, **over):
    fields = dict(objective="Deliver the thing", owner_words="## OWNER WORDS\n\nmake it good",
                  dialogue_section="## OWN ROOM DIALOGUE\n\nrows\n", revision=revision, delta=None, cross_talk="",
                  own_objections=[], message="First look.", evidence_section="(nothing attached)\n", first_time=True)
    fields.update(over)
    return plan_packet.build_participant_content(**fields)


def test_an_ask_about_a_revision_keeps_its_packet_bytes_and_gains_only_the_author_line_in_its_stable_prefix():
    assert _packet() == _BEFORE  # no author fact given: the bytes the code wrote before
    author = "drafted by task root-0, not spoken by my human"
    with_author = _packet(objective_author=author)
    line = f"\n\nThe objective above: {author}."
    assert with_author == _BEFORE.replace("Deliver the thing\n", "Deliver the thing" + line + "\n", 1)
    boundary, before = plan_packet.plan_user_stable_len(with_author), plan_packet.plan_user_stable_len(_BEFORE)
    assert with_author.index(line) < boundary and with_author[boundary:] == _BEFORE[before:]  # the plan part moves not a byte


def test_two_asks_of_one_task_share_the_stable_prefix_and_the_reading_text_rides_after_it(harness):
    sub = harness.install({})
    ctx = harness.make_ctx()
    ctx.task_metadata["objective_author"] = {"kind": "task", "task_id": "root-0"}
    _call(ctx, ask=["s1"], message="Read it first.")
    _write(ctx, ask=["s1"], message="Now the plan.")
    _call(ctx, ask=["s1"], message="And again.")
    sent = [_sent(sub, n, "s1") for n in range(3)]
    prefixes = [text[:plan_packet.plan_user_stable_len(text)] for text in sent]
    assert prefixes[0] and prefixes[0] == prefixes[1] == prefixes[2]
    assert "The objective above: drafted by task root-0, not spoken by my human." in prefixes[0]
    assert "No plan yet" not in prefixes[0] and "No plan yet" in sent[0] and "No plan yet" not in sent[1]
    # A task the host recorded no origin for says so in the same place, never "drafted".
    plain = harness.make_ctx(task_id="task-2")
    _call(plain, ask=["s1"])
    unrecorded = _sent(sub, 3, "s1")
    assert "The objective above: its author is not recorded by the host." in unrecorded and "drafted by" not in unrecorded
