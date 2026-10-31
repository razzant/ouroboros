"""Deferred follow-ups (B2b, W=A): the ``once`` trigger's pure selection
logic, the supervisor queue firing/mark-done semantics, and the agent-facing
``schedule_followup`` tool (one-shot or existing cron, authority guard, typed cap).

The scheduler is the EXISTING one (``supervisor/queue.py`` scheduled-tasks table);
these tests pin that no second scheduler was built: the tool only writes the table
the supervisor already consumes, and one-shot support is the smallest addition to
``check_scheduled_tasks`` (fires at/after ``run_at``, then marked done, never
re-fired).
"""

from __future__ import annotations

import pytest

import datetime
import pathlib

from ouroboros.tools.registry import ToolContext

UTC = datetime.timezone.utc


# ------------------------------------------------------------------- once_due


def test_once_due_selection_logic_with_a_fake_clock():
    from supervisor.schedule_time import once_due

    trigger = {"type": "once", "run_at": "2030-01-01T00:00:00+00:00"}
    before = datetime.datetime(2029, 12, 31, 23, 59, tzinfo=UTC)
    exactly = datetime.datetime(2030, 1, 1, 0, 0, tzinfo=UTC)
    after = datetime.datetime(2030, 1, 1, 12, 0, tzinfo=UTC)
    assert once_due(trigger, UTC, before) == (False, "")
    assert once_due(trigger, UTC, exactly) == (True, "")
    assert once_due(trigger, UTC, after) == (True, "")  # at/after, never only-at
    # typed record errors, never silent skips
    assert once_due({"type": "once"}, UTC, after)[1]
    assert once_due({"type": "once", "run_at": "not-a-time"}, UTC, after)[1]
    assert once_due({}, UTC, after)[1]


# ------------------------------------------------------- queue one-shot firing


def _queue(tmp_path):
    from supervisor import queue

    queue.init(tmp_path)
    pending: list = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    return queue, pending


def test_once_schedule_fires_exactly_once_and_is_marked_done(tmp_path):
    queue, pending = _queue(tmp_path)
    queue.upsert_scheduled_task({
        "id": "fu-due", "name": "Follow-up", "enabled": True, "source": "task_followup",
        "trigger": {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"},
        "task": {"type": "task", "text": "resume the blocked plan after the window resets", "chat_id": 1,
                 "metadata": {"origin_task_id": "t-origin", "resource_intent": {"kind": "system_repo"}}},
    })
    queue.upsert_scheduled_task({
        "id": "fu-future", "name": "Later", "enabled": True, "source": "task_followup",
        "trigger": {"type": "once", "run_at": "2999-01-01T00:00:00+00:00"},
        "task": {"type": "task", "text": "far future"},
    })
    queue.check_scheduled_tasks()
    queue.check_scheduled_tasks()  # a consumed one-shot never re-fires
    assert len(pending) == 1
    assert pending[0]["text"] == "resume the blocked plan after the window resets"
    assert pending[0]["delegation_role"] == "root"  # ordinary queued root, normal admission
    assert pending[0]["actor_id"] == "scheduler"
    assert pending[0]["metadata"]["schedule_id"] == "fu-due"
    records = {r["id"]: r for r in queue.list_scheduled_tasks(tmp_path)["tasks"]}
    done = records["fu-due"]
    assert done["enabled"] is False and done["completed_at"]  # durable receipt, not deletion
    assert done["last_task_id"] == pending[0]["id"] and done["next_run_at"] == ""
    future = records["fu-future"]
    assert future["enabled"] is True and not future.get("last_run_at")


def test_once_schedule_survives_a_refused_admission_and_retries(tmp_path, monkeypatch):
    """Review fix 5 + #1315: a refused admission consumes nothing and mints no failed
    root: the row WAITS with a typed reason, keeps ONE occurrence (the same task id)
    and retries it when the wait ends; admission then consumes the one-shot."""
    from ouroboros.task_results import load_task_result

    import ouroboros.config as config

    queue, pending = _queue(tmp_path)
    monkeypatch.setattr(config, "get_bg_wakeup_min_sec", lambda: 0)  # the wait ends at once in this test
    queue.upsert_scheduled_task({
        "id": "fu-blocked", "name": "Follow-up", "enabled": True, "source": "task_followup",
        "trigger": {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"},
        "task": {"type": "task", "text": "resume after the window resets", "chat_id": 1,
                 "metadata": {"resource_intent": {"kind": "system_repo"}}},
    })
    real_enqueue = queue.enqueue_task
    attempts: list = []
    monkeypatch.setattr(
        queue, "enqueue_task",
        lambda task, **_kw: attempts.append(task["id"]) or {**task, "_admission_blocked": "worker_pool_unavailable"})
    queue.check_scheduled_tasks()
    queue.check_scheduled_tasks()
    assert pending == [] and len(attempts) == 2 and len(set(attempts)) == 1  # one occurrence, retried
    record = queue.list_scheduled_tasks(tmp_path)["tasks"][0]
    assert record["enabled"] is True and not record.get("completed_at")
    assert record["hold"]["reason"] == "worker_pool_unavailable"
    assert not record.get("failure_count") and load_task_result(tmp_path, attempts[0]) is None  # no failed root
    # Admission heals: the next tick admits the SAME occurrence and consumes the record.
    monkeypatch.setattr(queue, "enqueue_task", real_enqueue)
    queue.check_scheduled_tasks()
    assert len(pending) == 1 and pending[0]["id"] == attempts[0]
    record = queue.list_scheduled_tasks(tmp_path)["tasks"][0]
    assert record["enabled"] is False and record["completed_at"] and "hold" not in record
    assert record.get("last_error") == ""


def test_once_schedule_refused_by_the_consciousness_door_waits_on_the_row(tmp_path, monkeypatch):
    """A one-shot a wake scheduled and the consciousness door refuses (allowance, concurrency —
    a refusal that can last hours) must not re-fire on every supervisor pass: the row waits
    with the typed reason until the alarm floor passes; no failed root, run point unchanged."""
    import datetime

    queue, pending = _queue(tmp_path)
    monkeypatch.setenv("OUROBOROS_BG_WAKEUP_MIN", "900")
    queue.upsert_scheduled_task({
        "id": "fu-conscious", "name": "Follow-up", "enabled": True, "source": "task_followup",
        "trigger": {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"},
        "task": {"type": "task", "text": "resume later", "chat_id": 1,
                 "metadata": {"initiator": "consciousness", "usage_category": "consciousness_task",
                              "resource_intent": {"kind": "system_repo"}}},
    })
    fires: list = []
    monkeypatch.setattr(
        queue, "enqueue_task",
        lambda task, **_kw: fires.append(task["id"]) or {
            **task, "_admission_blocked": "consciousness_allowance_exhausted",
            "_admission_detail": "$20.00 of $20.00 spent in the last 24 h"})
    before = datetime.datetime.now(datetime.timezone.utc)
    queue.check_scheduled_tasks()
    queue.check_scheduled_tasks()  # the very next pass: still waiting
    assert len(fires) == 1 and pending == []
    record = queue.list_scheduled_tasks(tmp_path)["tasks"][0]
    assert record["enabled"] is True and not record.get("completed_at")
    assert record["trigger"]["run_at"] == "2000-01-01T00:00:00+00:00"
    assert record["hold"]["reason"] == "consciousness_allowance_exhausted"
    assert "$20.00" in record["hold"]["detail"]
    assert datetime.datetime.fromisoformat(record["hold"]["retry_after"]) >= before + datetime.timedelta(seconds=890)


def test_re_enabled_completed_once_never_refires(tmp_path):
    """Round-3 exactly-once: a consumed one-shot (non-empty completed_at) must not
    fire again even when the owner flips enabled back on from the UI — re-arming
    goes through the gateway upsert with a fresh run_at, never a bare toggle."""
    queue, pending = _queue(tmp_path)
    fired = datetime.datetime(2020, 1, 1, tzinfo=UTC).isoformat()
    queue.upsert_scheduled_task({
        "id": "fu-rearmed", "name": "Follow-up", "enabled": True,  # UI re-enable
        "completed_at": fired, "last_task_id": "t-already-ran",
        "trigger": {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"},  # long due
        "task": {"type": "task", "text": "must not run twice"},
    })
    queue.check_scheduled_tasks()
    queue.check_scheduled_tasks()
    assert pending == []
    record = queue.list_scheduled_tasks(tmp_path)["tasks"][0]
    assert record["completed_at"] == fired  # receipt untouched
    assert record["last_task_id"] == "t-already-ran"


def test_once_schedule_with_invalid_run_at_records_a_typed_error(tmp_path):
    queue, pending = _queue(tmp_path)
    queue.upsert_scheduled_task({
        "id": "fu-bad", "enabled": True,
        "trigger": {"type": "once", "run_at": "not-a-time"},
        "task": {"type": "task", "text": "never fires"},
    })
    queue.check_scheduled_tasks()
    assert pending == []
    record = queue.list_scheduled_tasks(tmp_path)["tasks"][0]
    assert "run_at" in str(record.get("last_error") or "")
    assert record["enabled"] is True  # a typed error is visible, not a silent consume


# --------------------------------------------------------- schedule_followup


def _ctx(tmp_path, *, task_id="root-1", role="root"):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    (tmp_path / "data").mkdir(exist_ok=True)
    return ToolContext(
        repo_dir=repo, drive_root=tmp_path / "data", task_id=task_id,
        task_metadata={"root_task_id": task_id, "delegation_role": role},
        task_contract={"objective": "x", "delegation_role": role},
    )


def _followup(ctx, **kw):
    from ouroboros.tools.followup import _handle_schedule_followup

    params = {"relation": "independent", "run_at": "2030-01-01T00:00:00+00:00",
              "objective": "Re-run the plan panel once the reviewer window resets."}
    if "cron" in kw and "run_at" not in kw:
        params.pop("run_at")
    params.update(kw)
    return _handle_schedule_followup(ctx, **params)


def test_schedule_followup_registers_a_one_shot_entry(tmp_path):
    ctx = _ctx(tmp_path)
    out = _followup(ctx, name="Review window", context="plan review for root-1 was quorum-unreachable")
    assert out.startswith("FOLLOWUP_SCHEDULED")
    from supervisor.queue import list_scheduled_tasks

    root = pathlib.Path(tmp_path / "data").resolve()
    records = list_scheduled_tasks(root)["tasks"]
    assert len(records) == 1
    record = records[0]
    assert record["source"] == "task_followup" and record["enabled"] is True
    assert record["trigger"] == {"type": "once", "run_at": "2030-01-01T00:00:00+00:00"}
    # the agent's own words ride verbatim — no host template
    assert record["task"]["text"] == "Re-run the plan panel once the reviewer window resets."
    assert record["task"]["context"] == "plan review for root-1 was quorum-unreachable"
    assert record["name"] == "Review window"
    assert "description" not in record and "description" not in record["task"]
    assert "Next due: 2030-01-01T00:00:00+00:00." in out  # a one-shot names its own instant


@pytest.mark.parametrize("recurring", [False, True])
def test_schedule_followup_notify_leaves_a_note_in_the_one_table(tmp_path, recurring):
    """notify=true writes a ``kind: "notify"`` row: the agent's words and when they were
    written, independent work, counted in the same cap; no future task text is stored."""
    from supervisor import queue

    ctx = _ctx(tmp_path)
    out = _followup(ctx, notify=True, relation="related", objective="Call mother", name="Family call",
                   **({"cron": "0 9 * * *", "timezone": "Europe/Paris"} if recurring else {}))
    assert out.startswith("FOLLOWUP_SCHEDULED: note "), out
    assert "relation='related' ignored" in out, "a note carries no money or deadline"
    [record] = queue.list_scheduled_tasks(tmp_path / "data")["tasks"]
    assert record["kind"] == "notify" and record["source"] == "task_followup"
    assert record["notification"]["text"] == "Call mother" and record["notification"]["set_at"]
    assert record["name"] == "Family call" and "description" not in record
    assert f"Next due: {record['next_run_at'] or record['trigger']['run_at']}." in out
    assert "not determined" not in out
    assert record["followup_relation"]["kind"] == "independent"
    assert record["followup_origin"]["task_id"] == "root-1"
    assert "text" not in record["task"], "a note has no future task objective"
    assert _followup(ctx, notify=True, objective="Second").startswith("FOLLOWUP_SCHEDULED")
    third = _followup(ctx, objective="Third")
    assert third.startswith("FOLLOWUP_SCHEDULED") and "for this task: 3." in third, "notes and wake-ups share the count"


@pytest.mark.parametrize(("metadata", "contract", "chat_id"), [
    ({"presence": {"binding_id": "b" * 32}}, None, 1),
    ({"presence_binding_authority": {"binding_id": "mail-room"}}, None, 1),
    ({}, {"capability_ceiling": {"tools": ["schedule_followup"]}}, 1),  # a lost carrier under a ceiling
    ({}, None, -1001),  # agent-to-agent
], ids=["presence-speaker", "presence-work", "lost-carrier", "a2a"])
def test_notes_are_refused_to_callers_that_do_not_speak_for_the_owner(tmp_path, metadata, contract, chat_id):
    from supervisor.queue import list_scheduled_tasks

    ctx = _ctx(tmp_path)
    ctx.task_metadata.update(metadata)
    if contract is not None:
        ctx.task_contract.update(contract)
    ctx.current_chat_id = chat_id
    assert "FOLLOWUP_NOTIFY_REFUSED" in _followup(ctx, notify=True, objective="Wake owner")
    assert list_scheduled_tasks(ctx.drive_root)["tasks"] == []


def test_a_note_from_a_consciousness_root_in_main_is_a_legitimate_contact(tmp_path):
    """No room check: a root consciousness started (neither a Project nor owner ingress) may leave one."""
    ctx = _ctx(tmp_path)
    ctx.current_chat_id = 1
    ctx.task_metadata["initiator"] = "consciousness"
    assert _followup(ctx, notify=True, objective="The kettle is on.").startswith("FOLLOWUP_SCHEDULED: note ")


def test_a_delegated_subagent_cannot_leave_a_note(tmp_path):
    ctx = _ctx(tmp_path, role="subagent")
    assert "FOLLOWUP_SUBAGENT_REFUSED" in _followup(ctx, notify=True, objective="Wake owner")


def test_schedule_followup_notify_refuses_context_and_non_boolean_mode(tmp_path):
    ctx = _ctx(tmp_path)
    assert "FOLLOWUP_NOTIFY_CONTEXT" in _followup(ctx, notify=True, context="hidden")
    assert "FOLLOWUP_NOTIFY_INVALID" in _followup(ctx, notify="yes")
    assert "FOLLOWUP_TEXT_TOO_LONG" in _followup(ctx, notify=True, objective="x" * 4001)
    from supervisor import queue

    assert queue.list_scheduled_tasks(tmp_path / "data")["tasks"] == []
    assert _followup(ctx, notify=None).startswith("FOLLOWUP_SCHEDULED: follow-up "), "null is the default wake-up"


def test_a_timer_follow_up_is_framed_as_task_authored_and_its_note_never_enters_the_owner_corpus(tmp_path):
    """TZ-2 B3: the successor of a timer follow-up reads the same ``objective_author``
    stamp a promote / route_to_project carries, so its first turn is framed as drafted
    by the scheduling task and the note text never becomes an owner directive. The
    owner door's stamp (``origin_message_ref``) is still not carried (#1271)."""
    from types import SimpleNamespace

    from ouroboros.context import build_user_content
    from ouroboros.dialogue_provenance import run_origin
    from ouroboros.loop_messages import _initialize_owner_directives
    from supervisor import queue

    ctx = _ctx(tmp_path)
    ctx.task_metadata["origin_message_ref"] = {"chat_id": 1, "message_id": "owner-7"}
    note = "Re-check the reviewer window and resume the plan."
    assert _followup(ctx, objective=note).startswith("FOLLOWUP_SCHEDULED")
    [record] = queue.list_scheduled_tasks(tmp_path / "data")["tasks"]
    author = {"kind": "task", "task_id": "root-1"}
    assert record["task"]["metadata"]["objective_author"] == author
    assert "origin_message_ref" not in record["task"]["metadata"]
    successor = queue._task_from_schedule(record)
    assert successor["metadata"]["objective_author"] == author
    origin = run_origin(successor)
    assert (origin["owner_ingress"], origin["objective_author"]) == (False, author)
    content = build_user_content(successor)
    assert content.startswith("[OBJECTIVE_AUTHOR] The objective below was drafted by task root-1, "), content
    assert content.endswith("[/OBJECTIVE_AUTHOR]\n\n" + note), content
    live = SimpleNamespace(task_metadata=successor["metadata"])
    _initialize_owner_directives(live, [{"role": "user", "content": content}])
    assert getattr(live, "_owner_directives", []) == []  # the note is the task's, not the owner's


def test_presence_followup_preserves_ceiling_and_return_context(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.task_metadata["presence"] = {"binding_id": "b" * 32}
    ctx.task_contract = {"capability_ceiling": {"digest": "a" * 64}}
    assert _followup(ctx).startswith("FOLLOWUP_SCHEDULED")
    from supervisor import queue

    record = queue.list_scheduled_tasks(tmp_path / "data")["tasks"][0]
    assert record["task"]["metadata"]["presence"] == {"binding_id": "b" * 32}
    assert record["task"]["task_contract"] == ctx.task_contract
    assert record["followup_origin"]["task_id"] == "root-1"


def test_presence_recurring_followup_uses_existing_cron_and_preserves_authority(tmp_path):
    from ouroboros.presence_authority import (
        PresenceCapabilityCeiling,
        PresenceToolGrant,
        presence_ceiling_payload,
    )

    ctx = _ctx(tmp_path)
    ctx.task_metadata["presence"] = {
        "binding_id": "b" * 32,
        "conversation_key": "telegram:account:chat:0",
    }
    ceiling = PresenceCapabilityCeiling(
        skill_name="community-helper",
        skill_content_hash="a" * 64,
        profile_fingerprint="b" * 64,
        state_fingerprint="c" * 64,
        selection_fingerprint="d" * 64,
        model_slot="main",
        inline_max_rounds=10,
        tool_grants=(PresenceToolGrant("telegram_send"),),
        resource_grants=(),
        digest="0" * 64,
    )
    payload = presence_ceiling_payload(ceiling)
    ctx.task_contract = {"objective": "presence turn", "capability_ceiling": payload}

    out = _followup(ctx, cron="15 9 * * 1-5", timezone="Europe/Moscow")
    assert out.startswith("FOLLOWUP_SCHEDULED")
    assert "recurring cron 15 9 * * 1-5 (Europe/Moscow)" in out
    from supervisor import queue

    record = queue.list_scheduled_tasks(tmp_path / "data")["tasks"][0]
    assert record["trigger"] == {"type": "cron", "expr": "15 9 * * 1-5"}
    assert record["timezone"] == "Europe/Moscow"
    assert record["next_run_at"]
    assert f"Next due: {record['next_run_at']}" in out
    assert record["task"]["metadata"]["presence"] == ctx.task_metadata["presence"]
    assert record["task"]["task_contract"] == {"capability_ceiling": payload}  # the turn's objective is not the follow-up's
    scheduled = queue._task_from_schedule(record)
    assert scheduled["metadata"]["presence"] == ctx.task_metadata["presence"]
    assert scheduled["task_contract"]["capability_ceiling"] == ctx.task_contract["capability_ceiling"]
    assert scheduled["task_contract"]["objective"] == "Re-run the plan panel once the reviewer window resets."


def test_a_bindings_root_follow_up_runs_its_own_objective_under_the_inherited_authority(tmp_path):
    from ouroboros.presence_authority import presence_ceiling_payload
    from supervisor import queue
    from tests.test_presence_own_work import _ceiling

    ctx = _ctx(tmp_path)
    # A root a delegated descendant promoted: binding authority only, a full contract of its own.
    ctx.task_metadata["presence_binding_authority"] = {"binding_id": "b" * 32}
    ceiling = presence_ceiling_payload(_ceiling())
    ctx.task_contract = {"objective": "Compile the full audit", "context": "old audit ids 1-9",
                         "expected_output": "Nine figures", "acceptance_claims": ["Q1 is closed"],
                         "success_criteria": ["All nine figures reviewed"],
                         "allowed_resources": {"network": True}, "capability_ceiling": ceiling}

    assert _followup(ctx, objective="Revisit the Q2 figures", context="Q2 closes Friday").startswith(
        "FOLLOWUP_SCHEDULED")
    assert _followup(ctx, objective="Check the Q3 draft").startswith("FOLLOWUP_SCHEDULED")

    records = queue.list_scheduled_tasks(tmp_path / "data")["tasks"]
    scheduled = {row["task_contract"]["objective"]: row for row in map(queue._task_from_schedule, records)}
    assert set(scheduled) == {"Revisit the Q2 figures", "Check the Q3 draft"}
    assert scheduled["Revisit the Q2 figures"]["task_contract"]["context"] == "Q2 closes Friday"
    assert scheduled["Check the Q3 draft"]["task_contract"]["context"] == ""  # no inherited context
    for task in scheduled.values():
        assert task["task_contract"]["expected_output"] == ""
        assert task["task_contract"]["acceptance_claims"] == []
        assert task["task_contract"]["success_criteria"] == []
        assert task["metadata"]["presence_binding_authority"] == {"binding_id": "b" * 32}
        assert task["task_contract"]["capability_ceiling"] == ceiling
        assert task["task_contract"]["allowed_resources"] == {"network": True}


def test_schedule_followup_requires_exactly_one_valid_trigger(tmp_path):
    ctx = _ctx(tmp_path)
    assert _followup(ctx, run_at="", cron="").startswith("ERROR: FOLLOWUP_TRIGGER_REQUIRED")
    both = _followup(ctx, cron="0 9 * * *", run_at="2030-01-01T00:00:00+00:00")
    assert both.startswith("ERROR: FOLLOWUP_TRIGGER_REQUIRED")
    assert _followup(ctx, cron="hourly").startswith("ERROR: FOLLOWUP_CRON_INVALID")
    bad_zone = _followup(ctx, cron="0 9 * * *", timezone="Mars/Olympus")
    assert bad_zone.startswith("ERROR: FOLLOWUP_TIMEZONE_INVALID")
    from supervisor.queue import list_scheduled_tasks

    assert list_scheduled_tasks(tmp_path / "data")["tasks"] == []
    # A zone beside an absolute one-shot instant asks for nothing (models fill every
    # schema key): the follow-up is scheduled, the receipt says the zone was ignored,
    # and the ignored zone is not stored with the record.
    run_at_zone = _followup(ctx, timezone="Europe/Moscow")
    assert run_at_zone.startswith("FOLLOWUP_SCHEDULED")
    assert "timezone='Europe/Moscow' ignored: it applies only to recurring cron follow-ups" in run_at_zone
    [stored] = list_scheduled_tasks(tmp_path / "data")["tasks"]
    assert stored["trigger"]["type"] == "once" and not stored.get("timezone")
    # A zone beside a run_at WITHOUT an offset is not a no-op: the naive time would be read
    # as UTC, hours off. Still refused, and the refusal names the repair.
    naive = _followup(ctx, run_at="2030-01-01T09:00:00", timezone="Europe/Moscow")
    assert naive.startswith("ERROR: FOLLOWUP_TIMEZONE_WITH_RUN_AT") and "+03:00" in naive
    assert len(list_scheduled_tasks(tmp_path / "data")["tasks"]) == 1


def test_schedule_followup_count_is_reported_never_a_cap(tmp_path):
    """How many follow-ups a task registers is the agent's judgment: each one lands and the
    reply states this task's pending total (Station: twelve meal posts registered up front)."""
    ctx = _ctx(tmp_path)
    replies = [_followup(ctx, run_at=f"2030-{month:02d}-01T00:00:00+00:00") for month in range(1, 13)]
    assert all(reply.startswith("FOLLOWUP_SCHEDULED") for reply in replies), replies
    assert "for this task: 12." in replies[-1]
    # another task's count is its own
    other = _followup(_ctx(tmp_path, task_id="root-2"))
    assert other.startswith("FOLLOWUP_SCHEDULED") and "for this task: 1." in other


def test_schedule_followup_counts_inside_the_write_it_is_about_to_make(tmp_path, monkeypatch):
    """The pending count and the write share ONE hold on the schedule table, so the
    total a reply states is the table it wrote."""
    from supervisor import queue_schedules

    ctx = _ctx(tmp_path)
    acquisitions: list[str] = []
    real = queue_schedules.acquire_exclusive_file_lock
    monkeypatch.setattr(
        queue_schedules, "acquire_exclusive_file_lock",
        lambda path, **kw: (acquisitions.append(str(path)), real(path, **kw))[1])
    assert _followup(ctx).startswith("FOLLOWUP_SCHEDULED")
    assert len(acquisitions) == 1, acquisitions
    assert acquisitions[0].endswith("scheduled_tasks.json.lock")
    # Every later registration takes the same single hold for its count and its write.
    acquisitions.clear()
    second = _followup(ctx, run_at="2030-02-01T00:00:00+00:00")
    assert second.startswith("FOLLOWUP_SCHEDULED") and "for this task: 2." in second
    assert len(acquisitions) == 1
    from supervisor.queue import list_scheduled_tasks

    assert len(list_scheduled_tasks(tmp_path / "data")["tasks"]) == 2


def test_schedule_followup_reports_a_missed_table_lock_as_a_typed_refusal(tmp_path, monkeypatch):
    """A lock the transaction could not take is a typed unavailable result in the
    text ABI, not a generic TOOL_ERROR wrapping a bare TimeoutError; nothing lands."""
    from supervisor import queue_schedules
    from supervisor.queue import list_scheduled_tasks

    ctx = _ctx(tmp_path)
    monkeypatch.setattr(queue_schedules, "acquire_exclusive_file_lock", lambda *_a, **_k: None)
    text = _followup(ctx)
    assert text.startswith("⚠️ CAPABILITY_UNAVAILABLE: FOLLOWUP_STORE_UNAVAILABLE")
    assert "schedule lock" in text
    assert list_scheduled_tasks(tmp_path / "data")["tasks"] == []


def test_schedule_followup_surfaces_a_corrupt_store_as_the_typed_refusal(tmp_path, monkeypatch):
    """The pending count refuses an unreadable table before attempting a write."""
    from supervisor import queue

    ctx = _ctx(tmp_path)
    state = tmp_path / "data" / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "scheduled_tasks.json").write_text("{not json", encoding="utf-8")
    writes = []
    upsert = queue.upsert_scheduled_task
    monkeypatch.setattr(queue, "upsert_scheduled_task", lambda row, **kw: (writes.append(row), upsert(row, **kw))[1])
    text = _followup(ctx)
    assert text.startswith("⚠️ CAPABILITY_UNAVAILABLE: FOLLOWUP_STORE_UNAVAILABLE")
    assert "PERSIST_FAILED" not in text
    assert (state / "scheduled_tasks.json").read_text(encoding="utf-8") == "{not json"
    assert writes == []
    (state / "scheduled_tasks.json").write_text('{"tasks":[]}', encoding="utf-8")
    assert _followup(ctx).startswith("FOLLOWUP_SCHEDULED") and len(writes) == 1


def test_schedule_followup_discloses_a_lost_audit_outcome_on_its_success_line(tmp_path, monkeypatch):
    """The write seam returns what its AUDIT achieved; the success text says so.

    A durable record whose outcome fact never reached logs/events.jsonl is not a
    clean registration — reporting only "registered" would hide the one thing
    that makes the change accountable, and the agent would never know to re-check.
    """
    from ouroboros import utils as ouro_utils

    ctx = _ctx(tmp_path)
    clean = _followup(ctx)
    assert clean.startswith("FOLLOWUP_SCHEDULED") and "AUDIT_INCOMPLETE" not in clean

    calls = {"n": 0}
    real = ouro_utils.append_jsonl

    def _intent_only(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs) if calls["n"] == 1 else False

    monkeypatch.setattr(ouro_utils, "append_jsonl", _intent_only)
    disclosed = _followup(_ctx(tmp_path, task_id="root-2"), run_at="2030-02-01T00:00:00+00:00")
    assert disclosed.startswith("FOLLOWUP_SCHEDULED")
    assert "AUDIT_INCOMPLETE" in disclosed and "audit=incomplete" in disclosed
    # Disclosed, never rolled back: an automatic undo would be a second
    # equally unaudited mutation.
    from supervisor.queue import list_scheduled_tasks

    assert len(list_scheduled_tasks(tmp_path / "data")["tasks"]) == 2


def test_schedule_followup_overlong_text_is_a_typed_refusal_never_truncated(tmp_path):
    """Review fix 7: the objective/context ride VERBATIM into the future task, so an
    over-limit text is a typed FOLLOWUP_TEXT_TOO_LONG refusal naming the limit —
    never a silent cut that changes what the future task is."""
    from ouroboros.tools.followup import _MAX_CONTEXT_CHARS, _MAX_OBJECTIVE_CHARS

    ctx = _ctx(tmp_path)
    long_objective = _followup(ctx, objective="x" * (_MAX_OBJECTIVE_CHARS + 1))
    assert long_objective.startswith("ERROR: FOLLOWUP_TEXT_TOO_LONG")
    assert str(_MAX_OBJECTIVE_CHARS) in long_objective
    long_context = _followup(ctx, context="y" * (_MAX_CONTEXT_CHARS + 1))
    assert long_context.startswith("ERROR: FOLLOWUP_TEXT_TOO_LONG")
    assert str(_MAX_CONTEXT_CHARS) in long_context
    from supervisor.queue import list_scheduled_tasks

    assert list_scheduled_tasks(pathlib.Path(tmp_path / "data").resolve())["tasks"] == []
    # At-limit text is accepted whole, byte-for-byte.
    ok = _followup(ctx, objective="z" * _MAX_OBJECTIVE_CHARS)
    assert ok.startswith("FOLLOWUP_SCHEDULED")
    record = list_scheduled_tasks(pathlib.Path(tmp_path / "data").resolve())["tasks"][0]
    assert record["task"]["text"] == "z" * _MAX_OBJECTIVE_CHARS


def test_schedule_followup_guards_authority_and_inputs(tmp_path):
    # narrower-than-parent: a delegated subagent may not mint future root tasks
    sub = _followup(_ctx(tmp_path, role="subagent"))
    assert sub.startswith("ERROR: FOLLOWUP_SUBAGENT_REFUSED")
    # a real task id is required for the durable per-task cap
    no_task = _followup(_ctx(tmp_path, task_id=""))
    assert no_task.startswith("ERROR: FOLLOWUP_TASK_ID_REQUIRED")
    ctx = _ctx(tmp_path)
    assert _followup(ctx, run_at="soon").startswith("ERROR: FOLLOWUP_RUN_AT_INVALID")
    assert _followup(ctx, objective="  ").startswith("ERROR: FOLLOWUP_OBJECTIVE_REQUIRED")
    from supervisor.queue import list_scheduled_tasks

    assert list_scheduled_tasks(pathlib.Path(tmp_path / "data").resolve())["tasks"] == []


def test_schedule_followup_root_id_falls_back_to_task_id_never_the_string_none(tmp_path):
    """Review fix 9: metadata WITHOUT root_task_id must fall back to task_id —
    `str(None)` used to persist the literal string "None" as the origin root."""
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    (tmp_path / "data").mkdir(exist_ok=True)
    ctx = ToolContext(
        repo_dir=repo, drive_root=tmp_path / "data", task_id="root-3",
        task_metadata={"delegation_role": "root"},  # no root_task_id key
        task_contract={"objective": "x", "delegation_role": "root"},
    )
    assert _followup(ctx).startswith("FOLLOWUP_SCHEDULED")
    from supervisor.queue import list_scheduled_tasks

    record = list_scheduled_tasks(pathlib.Path(tmp_path / "data").resolve())["tasks"][0]
    assert record["followup_origin"]["root_task_id"] == "root-3"


def test_schedule_followup_preserves_source_project_and_chat(tmp_path):
    ctx = _ctx(tmp_path, task_id="project-task")
    ctx.project_id = "memory-atlas"
    ctx.current_chat_id = 233966548

    assert _followup(ctx).startswith("FOLLOWUP_SCHEDULED")
    from supervisor.queue import list_scheduled_tasks

    record = list_scheduled_tasks(pathlib.Path(tmp_path / "data").resolve())["tasks"][0]
    assert record["task"]["chat_id"] == 233966548
    assert record["task"]["project_id"] == "memory-atlas"
    from supervisor.queue_schedules import _task_from_schedule
    from ouroboros.project_facts import resolve_project_id

    queued = _task_from_schedule(record)
    assert queued["project_id"] == "memory-atlas"
    assert resolve_project_id(queued) == "memory-atlas"
    assert queued["chat_id"] == 233966548


def test_schedule_followup_of_an_unscoped_task_invents_no_project_address(tmp_path):
    """Preserving a source address must not become a new addressing policy: an
    unscoped task's follow-up keeps the existing owner-chat default."""
    from supervisor.queue import list_scheduled_tasks
    from supervisor.queue_schedules import _task_from_schedule
    from supervisor import state
    from ouroboros.project_facts import resolve_project_id

    # The initial missing binding is unknown, never the hidden chat-0 partition.
    # A real owner-bound installation supplies the positive destination.
    state.init(tmp_path / "data")
    state.init_state()
    state.update_state(lambda st: st.update(owner_chat_id=17))
    assert _followup(_ctx(tmp_path, task_id="plain-task")).startswith("FOLLOWUP_SCHEDULED")
    record = list_scheduled_tasks(pathlib.Path(tmp_path / "data").resolve())["tasks"][0]
    assert "project_id" not in record["task"]
    assert "chat_id" not in record["task"]

    queued = _task_from_schedule(record)
    assert resolve_project_id(queued) == ""
    assert queued["chat_id"] == 17


# ------------------------------------------------- gateway + digest + queue GC


def test_schedules_gateway_accepts_and_validates_once_triggers(tmp_path):
    """Review fix 9: the Schedules upsert accepts `{"type":"once","run_at":ISO}`
    (the owner's enable/disable toggle round-trips a followup record through this
    endpoint), validates run_at, and still rejects unknown trigger types."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.schedules import api_schedules_list, api_schedules_upsert
    from supervisor import queue

    queue.init(tmp_path)
    app = Starlette(routes=[
        Route("/api/schedules", endpoint=api_schedules_list, methods=["GET"]),
        Route("/api/schedules", endpoint=api_schedules_upsert, methods=["POST"]),
    ])
    app.state.drive_root = tmp_path
    client = TestClient(app)

    ok = client.post("/api/schedules", json={
        "id": "fu-1", "name": "Follow-up",
        "trigger": {"type": "once", "run_at": "2030-01-01T00:00:00+00:00"},
        "task": {"type": "task", "text": "resume"},
    })
    assert ok.status_code == 200
    stored = client.get("/api/schedules").json()["tasks"][0]
    assert stored["trigger"] == {"type": "once", "run_at": "2030-01-01T00:00:00+00:00"}
    bad_run_at = client.post("/api/schedules", json={
        "id": "fu-2", "trigger": {"type": "once", "run_at": "soon"},
        "task": {"type": "task", "text": "x"},
    })
    assert bad_run_at.status_code == 400 and "run_at" in bad_run_at.json()["error"]
    unknown = client.post("/api/schedules", json={
        "id": "fu-3", "trigger": {"type": "interval", "run_at": "2030-01-01T00:00:00+00:00"},
        "task": {"type": "task", "text": "x"},
    })
    assert unknown.status_code == 400


def test_gateway_rearm_of_completed_once_requires_a_fresh_run_at(tmp_path):
    """Round-3 exactly-once vs re-enable: re-enabling a CONSUMED one-shot through
    the Schedules upsert without a NEW run_at is a 400; supplying a fresh run_at
    re-arms it (completed_at cleared) and it fires exactly once; a disable that
    keeps the same run_at carries the receipt forward for GC."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.schedules import api_schedules_upsert
    from supervisor import queue

    queue.init(tmp_path)
    pending: list = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    fired = datetime.datetime(2020, 1, 1, tzinfo=UTC).isoformat()
    queue.upsert_scheduled_task({
        "id": "fu-done", "name": "Follow-up", "enabled": False, "completed_at": fired,
        "trigger": {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"},
        "task": {"type": "task", "text": "resume", "chat_id": 1},
    })
    app = Starlette(routes=[Route("/api/schedules", endpoint=api_schedules_upsert, methods=["POST"])])
    app.state.drive_root = tmp_path
    client = TestClient(app)

    # Bare re-enable with the SAME run_at: refused with a clear re-arm message.
    refused = client.post("/api/schedules", json={
        "id": "fu-done", "enabled": True,
        "trigger": {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"},
        "task": {"type": "task", "text": "resume", "chat_id": 1},
    })
    assert refused.status_code == 400 and "run_at" in refused.json()["error"]
    # Disable/edit keeping the same run_at: allowed, receipt carried forward.
    kept = client.post("/api/schedules", json={
        "id": "fu-done", "enabled": False,
        "trigger": {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"},
        "task": {"type": "task", "text": "resume", "chat_id": 1},
    })
    assert kept.status_code == 200
    assert kept.json()["schedule"]["completed_at"] == fired
    # A fresh run_at re-arms: completed_at cleared, and the record fires ONCE.
    rearmed = client.post("/api/schedules", json={
        "id": "fu-done", "enabled": True,
        "trigger": {"type": "once", "run_at": "2000-02-01T00:00:00+00:00"},
        "task": {"type": "task", "text": "resume", "chat_id": 1,
                 "metadata": {"resource_intent": {"kind": "system_repo"}}},
    })
    assert rearmed.status_code == 200
    assert "completed_at" not in rearmed.json()["schedule"]
    queue.check_scheduled_tasks()
    queue.check_scheduled_tasks()
    assert len(pending) == 1
    record = queue.list_scheduled_tasks(tmp_path)["tasks"][0]
    assert record["enabled"] is False and record["completed_at"]


def test_scheduled_tasks_digest_projects_run_at_for_once_records(tmp_path):
    """Review fix 9: the context digest shows a one-shot's fire instant (run_at)
    instead of an empty-string cron; cron records keep their cron field."""
    from types import SimpleNamespace

    from ouroboros.context import _scheduled_tasks_digest
    from supervisor import queue

    queue.init(tmp_path)
    queue.upsert_scheduled_task({
        "id": "fu", "name": "Follow-up", "enabled": True,
        "trigger": {"type": "once", "run_at": "2030-01-01T00:00:00+00:00"},
        "task": {"type": "task", "text": "resume"},
    })
    queue.upsert_scheduled_task({
        "id": "cr", "name": "Nightly", "enabled": True,
        "trigger": {"type": "cron", "expr": "0 3 * * *"},
        "task": {"type": "task", "text": "sweep"},
    })
    env = SimpleNamespace(drive_path=lambda rel: tmp_path / rel)
    digest = _scheduled_tasks_digest(env)
    rows = {row["id"]: row for row in digest["standing"] + digest["one_shots"]}
    assert [row["id"] for row in digest["standing"]] == ["cr"]
    assert [row["id"] for row in digest["one_shots"]] == ["fu"]
    assert rows["fu"]["run_at"] == "2030-01-01T00:00:00+00:00"
    assert "cron" not in rows["fu"]
    assert rows["cr"]["cron"] == "0 3 * * *"
    assert "run_at" not in rows["cr"]


def test_consumed_once_records_are_pruned_past_gc_retention(tmp_path):
    """Review fix 10b: consumed one-shot receipts (enabled=False + completed_at)
    older than the unified GC retention are pruned during the scheduler's write
    cycle; fresh consumed receipts and ENABLED records are always kept."""
    queue, pending = _queue(tmp_path)
    now = datetime.datetime.now(UTC)
    old = (now - datetime.timedelta(days=400)).isoformat()
    queue.upsert_scheduled_task({
        "id": "consumed-old", "enabled": False, "completed_at": old, "last_task_id": "settled-old",
        "trigger": {"type": "once", "run_at": old},
        "task": {"type": "task", "text": "done long ago"},
    })
    from ouroboros.task_results import write_task_result
    write_task_result(tmp_path, "settled-old", "completed", result="settled")
    queue.upsert_scheduled_task({
        "id": "consumed-unidentified", "enabled": False, "completed_at": old,
        "trigger": {"type": "once", "run_at": old}, "task": {"type": "task", "text": "unknown custody"}})
    queue.upsert_scheduled_task({
        "id": "consumed-fresh", "enabled": False, "completed_at": now.isoformat(),
        "trigger": {"type": "once", "run_at": now.isoformat()},
        "task": {"type": "task", "text": "done just now"},
    })
    queue.upsert_scheduled_task({
        "id": "enabled-future", "enabled": True,
        "trigger": {"type": "once", "run_at": "2999-01-01T00:00:00+00:00"},
        "task": {"type": "task", "text": "still standing"},
    })
    queue.upsert_scheduled_task({  # owner-disabled cron: no completed_at, never pruned
        "id": "disabled-cron", "enabled": False,
        "trigger": {"type": "cron", "expr": "0 3 * * *"},
        "task": {"type": "task", "text": "paused"},
    })
    queue.upsert_scheduled_task({  # round-3: disabled CRON with a stray old completed_at
        "id": "disabled-cron-stamped", "enabled": False, "completed_at": old,
        "trigger": {"type": "cron", "expr": "0 4 * * *"},
        "task": {"type": "task", "text": "paused, once ran"},
    })
    queue.check_scheduled_tasks()
    ids = {r["id"] for r in queue.list_scheduled_tasks(tmp_path)["tasks"]}
    # Only the aged-out CONSUMED ONE-SHOTS are pruned (an owner schedule's receipt
    # ages out with or without a recorded task id); a disabled cron row is a
    # standing schedule the owner may re-enable, even when it carries completed_at.
    assert ids == {"consumed-fresh", "enabled-future", "disabled-cron", "disabled-cron-stamped"}
    assert pending == []


def test_schema_version_is_authored_on_write(tmp_path):
    """CPL4-C7: the stamp is written to the durable file, not just defaulted on
    read — a legacy unversioned table gains it on its next write cycle."""
    import json

    queue, _pending = _queue(tmp_path)
    path = tmp_path / "state" / "scheduled_tasks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tasks": []}), encoding="utf-8")  # legacy: no version

    queue.upsert_scheduled_task({
        "id": "st", "enabled": True,
        "trigger": {"type": "once", "run_at": "2999-01-01T00:00:00+00:00"},
        "task": {"type": "task", "text": "stamped"},
    })
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == 1


def test_identical_last_error_does_not_rewrite_the_table_every_tick(tmp_path, monkeypatch):
    """Review fix 10a: a permanently invalid record (bad once run_at AND a cron
    row with no expression) writes its typed last_error ONCE; later ticks with the
    identical error text do not rewrite the table."""
    queue, pending = _queue(tmp_path)
    queue.upsert_scheduled_task({
        "id": "bad-once", "enabled": True,
        "trigger": {"type": "once", "run_at": "not-a-time"},
        "task": {"type": "task", "text": "never fires"},
    })
    queue.upsert_scheduled_task({
        "id": "bad-cron", "enabled": True,
        "trigger": {"type": "cron", "expr": ""},
        "task": {"type": "task", "text": "never fires either"},
    })
    writes = []
    real_write = queue._write_scheduled_tasks
    from supervisor import queue_schedules
    monkeypatch.setattr(queue_schedules, "_write_scheduled_tasks",
                        lambda data, drive_root=None: (writes.append(1), real_write(data, drive_root))[1])
    queue.check_scheduled_tasks()
    assert len(writes) == 1  # first tick records both typed errors
    records = {r["id"]: r for r in queue.list_scheduled_tasks(tmp_path)["tasks"]}
    assert "run_at" in records["bad-once"]["last_error"]
    assert "cron" in records["bad-cron"]["last_error"]
    queue.check_scheduled_tasks()
    queue.check_scheduled_tasks()
    assert len(writes) == 1, "identical error text must not rewrite the table per tick"
    assert pending == []


def test_schedule_followup_registration_surfaces():
    """DEVELOPMENT checklist: ToolEntry + explicit TOOL_POLICY + capability class."""
    from ouroboros.tools.followup import get_tools

    entries = get_tools()
    # Both schedule-table tools live here, so the module's own surfaces are pinned
    # together; `manage_schedules` keeps its separate authority pins in
    # tests/test_consciousness_observe_dispatch.py.
    assert [e.name for e in entries] == ["schedule_followup", "manage_schedules"]
    schema = entries[0].schema["parameters"]
    assert set(schema["required"]) == {"objective", "relation"}
    assert {"run_at", "cron"} <= set(schema["properties"])
    assert not ({"anyOf", "oneOf", "allOf"} & set(schema))
    from ouroboros.safety import POLICY_SKIP, POLICY_CHECK, TOOL_POLICY

    assert TOOL_POLICY["schedule_followup"] == POLICY_CHECK
    from ouroboros.tool_capabilities import (
        ACTING_SUBAGENT_TOOL_NAMES,
        CORE_TOOL_NAMES,
        LOCAL_READONLY_SUBAGENT_TOOL_NAMES,
    )

    assert "schedule_followup" in CORE_TOOL_NAMES
    assert "schedule_followup" not in LOCAL_READONLY_SUBAGENT_TOOL_NAMES
    assert "schedule_followup" not in ACTING_SUBAGENT_TOOL_NAMES
    from ouroboros.tools.registry import ToolRegistry

    assert "followup" in ToolRegistry._FROZEN_TOOL_MODULES
    assert TOOL_POLICY["manage_schedules"] == POLICY_SKIP
    assert "manage_schedules" in CORE_TOOL_NAMES
    # Reading the table is research; changing it is refused inside the tool.
    assert "manage_schedules" in LOCAL_READONLY_SUBAGENT_TOOL_NAMES
    assert "manage_schedules" not in ACTING_SUBAGENT_TOOL_NAMES
