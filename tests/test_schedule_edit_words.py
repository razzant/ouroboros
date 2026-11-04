"""A schedule edit that changes the assignment keeps the words its definition already carries and adds the editing
run's own by identity, in the schedule's own write; an edit of its time alone leaves them as they were.

Built on the review's reproduction (RV1W Astra F2): the next run and the helper it schedules read the words that
changed its assignment.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros.loop_messages import _initialize_owner_directives, _record_owner_directive
from ouroboros.owner_words import FIELD, governing_words_for_schedule, merged_corpus

OLD, NEW = "Check the staging deployment.", "Use the production deployment, but never restart it."


def _scheduled(tmp_path):
    from ouroboros.tools.followup import _handle_schedule_followup
    from supervisor import queue as schedules
    from tests.test_owner_words_carried import _owner_root

    ctx = _owner_root(tmp_path)
    _record_owner_directive(ctx, source="owner_mailbox", content=OLD, msg_id="old")
    assert _handle_schedule_followup(ctx, relation="independent", objective="Check staging.",
                                     cron="0 9 * * 1").startswith("FOLLOWUP_SCHEDULED")
    [record] = schedules.list_scheduled_tasks(ctx.drive_root)["tasks"]
    return ctx, record


def _next_run_words(ctx):
    from supervisor import queue as schedules

    [updated] = schedules.list_scheduled_tasks(ctx.drive_root)["tasks"]
    successor = schedules._task_from_schedule(updated)
    live = SimpleNamespace(task_id="successor", task_metadata=successor["metadata"])
    _initialize_owner_directives(live, [{"role": "user", "content": successor["text"]}])
    return [row["text"] for row in governing_words_for_schedule(live)[FIELD]]  # what it hands its helper


@pytest.mark.parametrize("edit", [{"objective": "Check production."}, {"context": "Production only."},
                                  {"cron": "0 10 * * 1"}, {"name": "Production check"}])
def test_an_edit_of_the_assignment_carries_the_words_that_changed_it(tmp_path, edit):
    from ouroboros.tools.followup import _manage_schedules

    ctx, record = _scheduled(tmp_path)
    _record_owner_directive(ctx, source="owner_mailbox", content=NEW, msg_id="new")
    response = json.loads(_manage_schedules(ctx, action="update", schedule_id=record["id"],
                                            reason="The owner changed the instruction.", **edit))
    assert response["ok"] and response["changed"]
    assignment = bool(edit.keys() & {"objective", "context"})
    texts = _next_run_words(ctx)
    assert texts.count(OLD) == 1, "the words already carried stay, once"
    assert (NEW in texts) is assignment, "an edit of the time or the name leaves the words as they were"
    assert ("owner_corpus" in response["changed_fields"]) is assignment


def test_the_words_join_the_stored_definition_under_the_write_itself(tmp_path):
    """The patch owner reads the carried words inside its own transaction: words a concurrent writer stored
    after the editing run looked are kept, never overwritten by that run's older view."""
    from supervisor import queue as schedules
    from supervisor.schedule_lifecycle import _write_scheduled_tasks, patch_scheduled_task, schedule_transaction

    ctx, record = _scheduled(tmp_path)
    with schedule_transaction(ctx.drive_root):
        data = schedules.load_schedule_store(ctx.drive_root)
        data["tasks"][0]["task"]["metadata"]["owner_corpus"].append(
            {"source": "owner_mailbox", "content": "Stored meanwhile.", "msg_id": "meanwhile"})
        _write_scheduled_tasks(data, ctx.drive_root)
    words = [{"source": "owner_mailbox", "content": OLD, "msg_id": "old"},
             {"source": "owner_mailbox", "content": NEW, "msg_id": "new"}]
    outcome = patch_scheduled_task(record["id"], {"objective": "Check production.", "owner_corpus": words},
                                   reason="edit", actor="agent", drive_root=ctx.drive_root)
    assert outcome["status"] == "updated"
    assert [row["content"] for row in outcome["task"]["metadata"]["owner_corpus"]] == [OLD, "Stored meanwhile.", NEW]


def test_one_source_is_one_carried_row_and_an_empty_id_names_none():
    kept = [{"source": "owner_mailbox", "content": "yes", "msg_id": "m1"}]
    added = [{"source": "owner_mailbox", "content": "yes", "msg_id": "m1"},  # the same message again
             {"source": "owner_mailbox", "content": "yes", "msg_id": "m2"},  # another message, the same words
             {"source": "owner_mailbox", "content": "no", "msg_id": "", "client_message_id": ""},
             {"source": "owner_mailbox", "content": "maybe", "msg_id": None, "client_message_id": None},
             {"source": "owner_mailbox", "content": "yes", "client_message_id": "c1"},
             {"source": "owner_mailbox", "content": "later", "msg_id": "m3", "client_message_id": "c1"}]
    assert [row["content"] for row in merged_corpus(kept, added)] == ["yes", "yes", "no", "maybe", "yes"]
