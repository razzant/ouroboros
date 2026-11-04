"""My human's words reach the work they caused, by value: from an addressed place turn, through a follow-up
(once and cron) to its root and that root's helpers, with the message they replied to under its own author.

Every guard is checked both ways: the words arrive where my human addressed the work, and stay out where
they did not (an unaddressed remark, a participant, a run with no owner words).
"""
from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_store
from ouroboros.chat_chain import format_address, row_address
from ouroboros.loop_messages import _initialize_owner_directives, _record_owner_directive
from ouroboros.owner_words import (FIELD, OWNER_SOURCES, REPLY_TARGET, _deduped, governing_words_for_schedule,
                                   owner_corpus_rows, render_owner_words)
from ouroboros.tools.registry import ToolContext

LONG = ("Sasha's estimate is fine, but before you start: keep the old export path working for the finance team, "
        "write the migration as a reversible step, and only then switch the default. ") * 3 + "Do NOT ship on Friday."


def _place_turn(tmp_path, *, addressed=True, confirmed=True, reply_to_message_id=""):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, current_task_type="presence")
    ctx.task_id = "place-turn"
    message = {"reply_to_message_id": reply_to_message_id} if reply_to_message_id else {}
    ctx.task_metadata = {
        "focus": {"place_id": "p1", "room_chat_id": 2 ** 41 + 3, "addressed": addressed,
                  "actor": {"platform_actor_id": "anton", "owner_confirmed": confirmed}},
        "presence": {"observed_text": LONG, "event": {"source_event_id": "ev-9", "message": message}}}
    _initialize_owner_directives(ctx, [{"role": "user", "content": "[FOCUS] an event in Team chat …"}])
    return ctx


def test_an_addressed_place_turn_from_my_human_carries_their_exact_words(tmp_path):
    ctx = _place_turn(tmp_path)
    place, frame = ctx._owner_directives
    assert place == {"source": "place_owner", "content": LONG, "msg_id": f"chat {2 ** 41 + 3} / ev-9", "carried": True}
    assert frame["source"] == "initial_text", "the event frame stays the first text Safety reads"
    words = governing_words_for_schedule(ctx)[FIELD]
    assert [row["text"] for row in words] == [LONG], "a helper gets the whole words, not a cut"


@pytest.mark.parametrize("addressed,confirmed", [(False, True), (True, False)])
def test_an_unaddressed_remark_or_a_participant_is_not_work(tmp_path, addressed, confirmed):
    ctx = _place_turn(tmp_path, addressed=addressed, confirmed=confirmed)
    assert [row["source"] for row in ctx._owner_directives] == ["initial_text"]
    assert FIELD not in governing_words_for_schedule(ctx)


def test_a_place_reply_names_its_target_by_address_when_the_connector_gave_no_words(tmp_path):
    rows = governing_words_for_schedule(_place_turn(tmp_path, reply_to_message_id="m-41"))[FIELD]
    assert rows[1] == {"text": "", "source": REPLY_TARGET, "carrier": "ctx", "task_id": "place-turn", "ts": "",
                       "ref": "platform message m-41 in this room", "author": "author not recorded"}
    text = render_owner_words(rows, audience="child")
    assert "author not recorded: the message my human replied to, not my human's words" in text
    assert "(its words are not carried here; this address reads them)" in text


def _owner_root(tmp_path, metadata=None):
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data", task_id="root-1",
                      task_metadata={"root_task_id": "root-1", "delegation_role": "root", **(metadata or {})},
                      task_contract={"objective": "x", "delegation_role": "root"})
    (tmp_path / "repo").mkdir(exist_ok=True)
    (tmp_path / "data").mkdir(exist_ok=True)
    return ctx


@pytest.mark.parametrize("trigger", [{"run_at": "2030-01-01T00:00:00+00:00"}, {"cron": "0 9 * * 1"}])
def test_a_follow_up_carries_the_words_to_its_root_and_the_roots_helpers(tmp_path, trigger):
    from ouroboros.main_context_authority import project_main_task_authority
    from ouroboros.tools.followup import _handle_schedule_followup
    from supervisor import queue as schedules

    ctx = _owner_root(tmp_path, {"origin_message_ref": {"chat_id": 1, "client_message_id": "owner-7"}})
    _initialize_owner_directives(ctx, [{"role": "user", "content": LONG}])
    _record_owner_directive(ctx, source="owner_mailbox", content="And keep the finance export.", msg_id="mail-1")
    out = _handle_schedule_followup(ctx, relation="independent", objective="Check the migration.", **trigger)
    assert out.startswith("FOLLOWUP_SCHEDULED"), out
    [record] = schedules.list_scheduled_tasks(tmp_path / "data")["tasks"]
    carried = record["task"]["metadata"]["owner_corpus"]
    assert [row["content"] for row in carried] == [LONG, "And keep the finance export."]
    successor = schedules._task_from_schedule(record)
    live = SimpleNamespace(task_id="fu-1", task_metadata=successor["metadata"], current_task_type="task")
    _initialize_owner_directives(live, [{"role": "user", "content": successor["text"]}])
    helper = governing_words_for_schedule(live)[FIELD]
    assert [row["text"] for row in helper] == [LONG, "And keep the finance export."], "the second generation too"
    shown = project_main_task_authority(successor)["task_owner_words"]
    assert shown["carried_by"] == "root-1" and [row["content"] for row in shown["rows"]] == [
        LONG, "And keep the finance export."], "the follow-up's own mind sees the words, not only its helpers"


def test_a_follow_up_of_a_run_without_owner_words_claims_none(tmp_path):
    from ouroboros.main_context_authority import project_main_task_authority
    from ouroboros.tools.followup import _handle_schedule_followup
    from supervisor import queue as schedules

    ctx = _owner_root(tmp_path, {"initiator": "consciousness"})
    _initialize_owner_directives(ctx, [{"role": "user", "content": "[Wake-up] my own time"}])
    assert _handle_schedule_followup(ctx, relation="independent", objective="Look again.",
                                     run_at="2030-01-01T00:00:00+00:00").startswith("FOLLOWUP_SCHEDULED")
    [record] = schedules.list_scheduled_tasks(tmp_path / "data")["tasks"]
    assert "owner_corpus" not in record["task"]["metadata"]
    assert "task_owner_words" not in project_main_task_authority(schedules._task_from_schedule(record))


def test_one_list_names_my_humans_corpus_labels_for_every_reader(tmp_path):
    from ouroboros.owner_continue import CORPUS_INITIAL_SOURCES, CORPUS_OWNER_SOURCES
    from ouroboros.tools.control_routing import _attach_drafted_objective

    assert "place_owner" in OWNER_SOURCES and CORPUS_OWNER_SOURCES == OWNER_SOURCES - CORPUS_INITIAL_SOURCES
    ctx = _place_turn(tmp_path)
    evt: dict = {}
    _attach_drafted_objective(ctx, evt)
    assert [row["source"] for row in evt["owner_corpus"]] == ["place_owner"] == [
        row["source"] for row in owner_corpus_rows(ctx)]
    seeded = SimpleNamespace(task_metadata={"objective_author": {"kind": "task", "task_id": "place-turn"},
                                            "owner_corpus": evt["owner_corpus"]})
    _initialize_owner_directives(seeded, [{"role": "user", "content": "drafted"}])
    assert [row["content"] for row in seeded._owner_directives] == [LONG]


def test_words_repeat_only_by_identity_never_by_likeness():
    def row(text, ref):
        return {"text": text, "source": "owner_mailbox", "carrier": "ctx", "task_id": "t", "ts": "", "ref": ref}

    assert len(_deduped([row("да", "msg a"), row("да", "msg b")])) == 2, "two yeses are two messages"
    assert len(_deduped([row("да", "msg a"), row("да", "msg a")])) == 1
    assert len(_deduped([row("да", ""), row("да", "")])) == 1, "without an identity the text is the only key"


def test_the_message_my_human_replied_to_rides_whole_under_its_author(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    spec = {"ts": "2026-11-04T09:00:00+00:00", "direction": "in", "chat_id": 1, "text": LONG + " — Sasha",
            "sender_label": "Sasha", "client_message_id": "sasha-1"}
    dialogue_store.append(root, "chat", spec)
    stored = next(entry.row for entry in dialogue_store.iter_rows(root, "chat"))
    address = format_address(row_address(stored))
    ctx = _owner_root(tmp_path, {"origin_message_ref": {"chat_id": 1, "client_message_id": "owner-8"},
                                 "reply_reference": {"address": address, "author_label": "Sasha"}})
    ctx.event_queue = queue.Queue()
    _initialize_owner_directives(ctx, [{"role": "user", "content": "Do this."}])
    [owner] = ctx._owner_directives
    assert owner["reply_to"] == {"address": address, "author": "Sasha", "text": LONG + " — Sasha"}
    rows = governing_words_for_schedule(ctx)[FIELD]
    assert [(row["source"], row["text"]) for row in rows] == [("initial_user", "Do this."),
                                                             (REPLY_TARGET, LONG + " — Sasha")]
    text = render_owner_words(rows, audience="child")
    assert "Sasha: the message my human replied to, not my human's words · " + address in text
    from ouroboros.loop_messages import owner_input_revision
    assert owner_input_revision(ctx) == 1, "the target is context, never a second owner message"
    lost = _owner_root(tmp_path, {"origin_message_ref": {"chat_id": 1, "client_message_id": "owner-9"},
                                  "reply_reference": {"address": "row:1@2026-11-04T09:00:00+00:00#000000000000"}})
    _initialize_owner_directives(lost, [{"role": "user", "content": "Do that."}])
    assert lost._owner_directives[0]["reply_to"]["text"] == "", "an unreadable target keeps its address alone"


def test_a_root_hands_on_the_words_it_carries_beside_its_own(tmp_path):
    carried = {"text": "The original ask.", "source": "origin_message", "carrier": "ctx", "task_id": "first",
               "ts": "", "ref": "chat 1 / owner-1"}
    ctx = _owner_root(tmp_path, {FIELD: [carried], "origin_message_ref": {"chat_id": 1, "client_message_id": "o2"}})
    _initialize_owner_directives(ctx, [{"role": "user", "content": "A newer word."}])
    assert [row["text"] for row in governing_words_for_schedule(ctx)[FIELD]] == ["The original ask.", "A newer word."]
