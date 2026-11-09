"""A scheduled run's carried owner message through chained Continues: one message stays one row.

The first Continue is pinned in ``test_reply_target_carried.py``; these follow the successor's own
``owner_corpus`` and ``continuation.owner_sources`` into a second and third Continue on the real producers
(schedule registration, firing, directive intake, review evidence, ``owner_sources``). A door row carries a
``client_message_id`` and no ``msg_id``; a mailbox row carries a ``msg_id``. Both must keep one row per message
(``owner_continue.row_identity``), and the work order must print the message once.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def _successor_record(sources, predecessor, task_id, ctx_successor):
    from ouroboros.owner_continue import owner_corpus_rows
    from ouroboros.review_evidence_sections import _accept_owner_directives

    metadata = {
        "objective_author": {"kind": "continuation", "predecessor_task_id": predecessor},
        "owner_corpus": owner_corpus_rows(sources),
        "continuation": {
            "predecessor_task_id": predecessor,
            "owner_sources": {"original": sources.get("original"), "schedule": sources.get("schedule"),
                              "later": sources.get("later") or [],
                              "held_owner_messages": sources.get("held_owner_messages") or {},
                              "open_relays": sources.get("open_relays") or {}},
            "peer_context": sources.get("peer_context") or [],
        },
    }
    return {"id": task_id, "metadata": metadata, "owner_mailbox": {"rows": [], "read_complete": True},
            "review_evidence": {"task_inputs": {"owner_requirements_and_decisions":
                                                _accept_owner_directives(ctx_successor, None, task_id)}}}


@pytest.mark.parametrize("source", ["initial_user", "owner_mailbox"])
def test_a_second_continue_of_a_scheduled_run_keeps_one_row_per_message(tmp_path, source):
    from ouroboros.loop_messages import _initialize_owner_directives, _record_owner_directive
    from ouroboros.owner_continue import owner_corpus_rows, owner_sources, work_order_text
    from ouroboros.review_evidence_sections import _accept_owner_directives
    from ouroboros.tools.followup import _handle_schedule_followup
    from supervisor import queue as schedules
    from tests.test_reply_target_carried import _door_run
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
    record = receipt()
    scheduled = record["schedule_admission"]["task"]
    scheduled.update(text=fired["text"])
    scheduled["metadata"].update(fired["metadata"])
    record.update(metadata=scheduled["metadata"], owner_mailbox={"rows": [], "read_complete": True},
                  review_evidence={"task_inputs": {"owner_requirements_and_decisions":
                                                  _accept_owner_directives(live, None, "root")}})
    first = owner_sources(ctx.drive_root, record, "root")
    assert first["gaps"] == [] and [row["content"] for row in first["later"]] == ["Do this."]
    # Successor S1 runs (seeded from its owner_corpus), is interrupted, and is Continued again.
    s1 = SimpleNamespace(task_id="s1", task_metadata={
        "objective_author": {"kind": "continuation", "predecessor_task_id": "root"},
        "owner_corpus": owner_corpus_rows(first)})
    _initialize_owner_directives(s1, [{"role": "user", "content": "[CONTINUE] work order"}])
    s1_record = _successor_record(first, "root", "s1", s1)
    second = owner_sources(ctx.drive_root, s1_record, "s1")
    assert second["gaps"] == []
    assert work_order_text("s1", "interrupted_retry_lost", second).count("Do this.") == 1
    assert [row["content"] for row in second["later"]] == ["Do this."], "one message, one row, on a chained Continue"


def test_every_further_continue_keeps_the_carried_message_once(tmp_path):
    from ouroboros.loop_messages import _initialize_owner_directives
    from ouroboros.owner_continue import owner_corpus_rows, owner_sources
    from ouroboros.owner_words import SCHEDULE_TEXT
    from ouroboros.review_evidence_sections import _accept_owner_directives
    from ouroboros.tools.followup import _handle_schedule_followup
    from supervisor import queue as schedules
    from tests.test_reply_target_carried import _door_run
    from tests.test_schedule_continue_basis import receipt

    ctx = _door_run(tmp_path)
    assert _handle_schedule_followup(ctx, relation="independent", objective="Check the plan weekly.",
                                     cron="0 9 * * 1").startswith("FOLLOWUP_SCHEDULED")
    [registered] = schedules.list_scheduled_tasks(ctx.drive_root)["tasks"]
    fired = schedules._task_from_schedule(registered)
    live = SimpleNamespace(task_id="root", task_metadata=fired["metadata"])
    _initialize_owner_directives(live, [{"role": "user", "content": fired["text"]}])
    record = receipt()
    scheduled = record["schedule_admission"]["task"]
    scheduled.update(text=fired["text"])
    scheduled["metadata"].update(fired["metadata"])
    record.update(metadata=scheduled["metadata"], owner_mailbox={"rows": [], "read_complete": True},
                  review_evidence={"task_inputs": {"owner_requirements_and_decisions":
                                                  _accept_owner_directives(live, None, "root")}})
    sources, predecessor, counts, schedule_text = owner_sources(ctx.drive_root, record, "root"), "root", [], []
    for index in range(1, 4):
        successor = SimpleNamespace(task_id=f"s{index}", task_metadata={
            "objective_author": {"kind": "continuation", "predecessor_task_id": predecessor},
            "owner_corpus": owner_corpus_rows(sources)})
        _initialize_owner_directives(successor, [{"role": "user", "content": "[CONTINUE] work order"}])
        successor_record = _successor_record(sources, predecessor, f"s{index}", successor)
        corpus = successor_record["metadata"]["owner_corpus"]
        counts.append(len([row for row in corpus if row["source"] != SCHEDULE_TEXT]))
        schedule_text.append([(row.get("field"), row["content"]) for row in corpus if row["source"] == SCHEDULE_TEXT])
        sources, predecessor = owner_sources(ctx.drive_root, successor_record, f"s{index}"), f"s{index}"
    counts.append(len(sources["later"]))
    assert counts == [1, 1, 1, 1]
    # Beside it the schedule's own text rides once per field on every hop, as the assignment, never their message.
    assert schedule_text == [[("objective", "Check the plan weekly.")]] * 3
