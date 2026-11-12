"""The message my human replied to travels beside their words through every transfer of them: a drafted start, the
terminal evidence, Continue, mail into running work and a relay. It stays typed, never counts as my human's words and
owes no reply; when its words cannot be read its address stays, and a place connector's platform id says honestly
that no host address reads it. One message is one corpus row by its identity, and an empty id names none.

Built on the review's reproductions (RV1W Astra F1, Grok F1, Opus F7).
"""
from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_store
from ouroboros.chat_chain import format_address, row_address
from ouroboros.loop_messages import _initialize_owner_directives, _record_owner_directive, owner_input_revision
from ouroboros.owner_words import FIELD, REPLY_TARGET, SCHEDULE_TEXT, governing_words_for_schedule

PROPOSAL = "The exact proposed specification, step by step."


def _proposal(root):
    dialogue_store.append(root, "chat", {"ts": "2026-11-04T09:00:00+00:00", "direction": "out", "chat_id": 1,
                                         "text": PROPOSAL, "client_message_id": "proposal"})
    stored = next(entry.row for entry in dialogue_store.iter_probes(root, "chat", [("client_message_id", "proposal")]))
    return {"address": format_address(row_address(stored))}


def _door_run(tmp_path):
    from tests.test_owner_words_carried import _owner_root

    ctx = _owner_root(tmp_path, {"origin_message_ref": {"chat_id": 1, "client_message_id": "replying-owner"},
                                 "origin_message_text": "Do this."})
    ctx.task_metadata["reply_reference"] = _proposal(ctx.drive_root)
    _initialize_owner_directives(ctx, [{"role": "user", "content": "Do this."}])
    return ctx


def _targets(ctx):
    return [row["text"] for row in governing_words_for_schedule(ctx)[FIELD] if row["source"] == REPLY_TARGET]


@pytest.mark.parametrize("routed", [False, True])
def test_a_reply_target_survives_a_drafted_start_into_its_helper(tmp_path, routed):
    from ouroboros.tools.control_routing import _attach_drafted_objective, _attach_origin_from_metadata

    ctx = _door_run(tmp_path)
    assert ctx._owner_directives[0]["reply_to"]["text"] == PROPOSAL
    if routed:  # start_task from this turn: the door's row becomes the start's origin message
        event = {}
        _attach_origin_from_metadata(ctx, event)
        _attach_drafted_objective(ctx, event)
        [origin] = [row for row in event["owner_corpus"] if row["source"] == "origin_message"]
        assert origin["client_message_id"] == "replying-owner" and origin["reply_to"]["text"] == PROPOSAL
        ctx = SimpleNamespace(task_id="new-root", task_metadata=event)
        _initialize_owner_directives(ctx, [{"role": "user", "content": "Implement the plan."}])
        assert owner_input_revision(ctx) == 0, "carried words owe no reply"
    assert _targets(ctx) == [PROPOSAL], "the helper reads the message my human replied to, under its author"


def test_a_reply_target_rides_the_requirements_into_a_continue_successor(tmp_path):
    from ouroboros.owner_continue import owner_corpus_rows, owner_sources
    from ouroboros.review_evidence_sections import _accept_owner_directives
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result, write_task_result

    ctx = _door_run(tmp_path)
    later = {"address": "row:1@2026-11-04T09:30:00+00:00#0123456789ab", "author": "Sasha", "text": "Sasha's draft."}
    _record_owner_directive(ctx, source="owner_mailbox", content="Keep the old API too.", msg_id="mail-1",
                            origin={"client_message_id": "owner-2"}, reply_to=later)
    requirements = _accept_owner_directives(ctx, None, "root-1")
    assert [row.get("reply_to", {}).get("text") for row in requirements] == [PROPOSAL, "Sasha's draft."]
    assert requirements[1]["client_message_id"] == "owner-2"
    root = ctx.drive_root
    write_task_result(root, "root-1", STATUS_CANCELLED, chat_id=1, root_task_id="root-1", result="interrupted",
                      origin_message_text="Do this.", origin_message_ref={"chat_id": 1, "client_message_id": "replying-owner"},
                      review_evidence={"task_inputs": {"owner_requirements_and_decisions": requirements}},
                      owner_mailbox={"rows": [], "read_complete": True})
    sources = owner_sources(root, load_task_result(root, "root-1"), "root-1")
    successor = SimpleNamespace(task_id="successor", task_metadata={
        "objective_author": {"kind": "continuation", "predecessor_task_id": "root-1"},
        "owner_corpus": owner_corpus_rows(sources)})
    _initialize_owner_directives(successor, [{"role": "user", "content": "[CONTINUE] work order"}])
    assert _targets(successor) == [PROPOSAL, "Sasha's draft."]
    assert [row.get("client_message_id") for row in successor._owner_directives] == [None, "owner-2"]


@pytest.mark.parametrize("source", ["initial_user", "owner_mailbox"])
def test_a_scheduled_run_continues_with_the_words_and_reply_target_it_carried(tmp_path, source):
    """RV2W Astra R1: a follow-up root has no door of its own, so its carried door rows are my human's words, not
    its first text: the real registration, fired run, terminal evidence and Continue seed keep them and their target."""
    from ouroboros.owner_continue import owner_corpus_rows, owner_sources, work_order_text
    from ouroboros.review_evidence_sections import _accept_owner_directives
    from ouroboros.tools.followup import _handle_schedule_followup
    from supervisor import queue as schedules
    from tests.test_schedule_continue_basis import receipt

    ctx = _door_run(tmp_path)
    if source == "owner_mailbox":
        reply = ctx._owner_directives[0]["reply_to"]
        ctx._owner_directives = []
        _record_owner_directive(ctx, source=source, content="Do this.", msg_id="mail-1", reply_to=reply)
    assert _handle_schedule_followup(ctx, relation="independent", objective="Check the plan weekly.",
                                     cron="0 9 * * 1").startswith("FOLLOWUP_SCHEDULED")
    [registered] = schedules.list_scheduled_tasks(ctx.drive_root)["tasks"]
    fired = schedules._task_from_schedule(registered)
    live = SimpleNamespace(task_id="root", task_metadata=fired["metadata"])
    _initialize_owner_directives(live, [{"role": "user", "content": fired["text"]}])
    assert _targets(live) == [PROPOSAL] and owner_input_revision(live) == 0
    record = receipt()  # the scheduler's accepted receipt shape, holding the generated task and its metadata
    scheduled = record["schedule_admission"]["task"]
    scheduled.update(text=fired["text"])
    scheduled["metadata"].update(fired["metadata"])
    record.update(metadata=scheduled["metadata"], owner_mailbox={"rows": [], "read_complete": True},
                  review_evidence={"task_inputs": {"owner_requirements_and_decisions":
                                                  _accept_owner_directives(live, None, "root")}})
    sources = owner_sources(ctx.drive_root, record, "root")
    assert sources["gaps"] == [] and [row["content"] for row in sources["later"]] == ["Do this."]
    assert "Do this." in work_order_text("root", "interrupted_retry_lost", sources)
    successor = SimpleNamespace(task_id="successor", task_metadata={
        "objective_author": {"kind": "continuation", "predecessor_task_id": "root"},
        "owner_corpus": owner_corpus_rows(sources)})
    _initialize_owner_directives(successor, [{"role": "user", "content": "[CONTINUE] work order"}])
    assert _targets(successor) == [PROPOSAL], "the successor reads the message my human replied to"
    assert [row["content"] for row in successor._owner_directives if row["source"] != SCHEDULE_TEXT] == ["Do this."]
    # Beside it the schedule's own text rides as the assignment, a separate source, never my human's words.
    assert [row["content"] for row in successor._owner_directives
            if row["source"] == SCHEDULE_TEXT] == ["Check the plan weekly."]


def test_mail_into_running_work_keeps_the_message_it_replies_to(tmp_path):
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import _mailbox_path
    from ouroboros.projects_registry import create_project
    from ouroboros.server_owner_routing import _route_project_chat_to_running_task
    from tests.test_project_routing_v664 import _ctx

    chat_id = int(create_project(tmp_path, "reply-room")["chat_id"])
    task = {"id": "work-root", "chat_id": chat_id, "root_task_id": "work-root", "delegation_role": "root",
            "drive_root": str(tmp_path)}  # its mailbox lives on this drive
    reference = _proposal(tmp_path)
    routed = _route_project_chat_to_running_task(_ctx(tmp_path, running={"work-root": {"task": task}}), chat_id,
                                                 "Yes, but skip step two.", "owner-reply",
                                                 task_metadata={"reply_reference": reference})
    assert routed == "work-root"
    assert _mailbox_path(tmp_path, "work-root").read_text(encoding="utf-8").count(reference["address"]) == 1
    reader = SimpleNamespace(task_id="work-root", drive_root=tmp_path, budget_drive_root=tmp_path, current_chat_id=chat_id,
                             task_metadata={}, pending_events=[], event_queue=None, drive_logs=lambda: tmp_path / "logs")
    _drain_incoming_messages([], queue.Queue(), tmp_path, "work-root", None, set(), owner_ctx=reader)
    [row] = reader._owner_directives
    assert row["content"].startswith("Yes, but skip step two.") and row["reply_to"]["text"] == PROPOSAL
    assert owner_input_revision(reader) == 1, "my human's own words still owe their reply; the target owes none"


def test_a_relayed_quote_keeps_the_message_it_replies_to(tmp_path):
    from ouroboros.owner_relay import inject_relay
    from tests.test_owner_relay import context

    reference = _proposal(tmp_path)
    quoted = {"chat_id": 1, "direction": "in", "ts": "2026-11-04T10:00:00Z", "text": "Do it as proposed.",
              "client_message_id": "owner-quoted", "reply_to": reference}
    receiver = context(tmp_path)
    receiver.task_id = "receiver"
    inject_relay(receiver, {"owner_rows": [quoted], "source_task_id": "teller", "text": "note",
                            "relay": {"relay_id": "r1"}}, [])
    [row] = receiver._owner_directives
    assert row["carried"] is True and row["reply_to"]["text"] == PROPOSAL


def test_a_place_reply_names_honestly_what_the_host_cannot_read():
    from ouroboros.owner_words import UNREAD_PLATFORM, UNREAD_TARGET, _row, render_owner_words

    place = _row("", source=REPLY_TARGET, carrier="ctx", ref="platform message 42 in this room", author="")
    chat = _row("", source=REPLY_TARGET, carrier="ctx", ref="row:1@2026-11-04T09:00:00+00:00#0123456789ab", author="")
    text = render_owner_words([place, chat], audience="child")
    assert UNREAD_PLATFORM in text and text.count(UNREAD_TARGET) == 1, "only a host address claims to read the words"
    assert "author not recorded: the message my human replied to, not my human's words" in text


@pytest.mark.parametrize("empty", ["", None])
def test_an_empty_message_id_never_merges_two_messages(empty):
    ctx = SimpleNamespace(task_id="run")
    for index in (1, 2):
        _record_owner_directive(ctx, source="owner_mailbox", content=f"message {index}", msg_id=empty or "",
                                origin={"client_message_id": empty})
    _record_owner_directive(ctx, source="owner_mailbox", content="the same message again", msg_id="",
                            origin={"client_message_id": "c-3"})
    _record_owner_directive(ctx, source="owner_mailbox", content="the same message again", msg_id="",
                            origin={"client_message_id": "c-3"})
    assert [row["content"] for row in ctx._owner_directives] == ["message 1", "message 2", "the same message again"]
