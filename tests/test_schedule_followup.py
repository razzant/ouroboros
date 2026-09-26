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
        "task": {"type": "task", "text": "resume the blocked plan after the window resets",
                 "metadata": {"origin_task_id": "t-origin"}},
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
    """Review fix 5: the once-trigger is consumed ONLY when admission succeeded.
    A refused admission (worker pool down, duplicate id, routing fence) leaves the
    record enabled with last_error, so the next scheduler tick retries it."""
    queue, pending = _queue(tmp_path)
    queue.upsert_scheduled_task({
        "id": "fu-blocked", "name": "Follow-up", "enabled": True, "source": "task_followup",
        "trigger": {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"},
        "task": {"type": "task", "text": "resume after the window resets"},
    })
    real_enqueue = queue.enqueue_task
    monkeypatch.setattr(
        queue, "enqueue_task",
        lambda task: {**task, "_admission_blocked": "worker_pool_unavailable"})
    queue.check_scheduled_tasks()
    assert pending == []
    record = queue.list_scheduled_tasks(tmp_path)["tasks"][0]
    assert record["enabled"] is True and not record.get("completed_at")
    assert "worker_pool_unavailable" in str(record.get("last_error") or "")
    # Admission heals: the very next tick fires and consumes the record.
    monkeypatch.setattr(queue, "enqueue_task", real_enqueue)
    queue.check_scheduled_tasks()
    assert len(pending) == 1
    record = queue.list_scheduled_tasks(tmp_path)["tasks"][0]
    assert record["enabled"] is False and record["completed_at"]
    assert record.get("last_error") == ""


def test_once_schedule_refused_by_the_consciousness_door_defers_by_the_alarm_floor(tmp_path, monkeypatch):
    """A one-shot a wake scheduled and the consciousness door refuses (allowance, concurrency —
    a refusal that can last hours) must not re-fire on every supervisor pass: the record stays
    armed, its run point moves forward by the alarm floor (opus round 4)."""
    import datetime

    queue, pending = _queue(tmp_path)
    monkeypatch.setenv("OUROBOROS_BG_WAKEUP_MIN", "900")
    queue.upsert_scheduled_task({
        "id": "fu-conscious", "name": "Follow-up", "enabled": True, "source": "task_followup",
        "trigger": {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"},
        "task": {"type": "task", "text": "resume later",
                 "metadata": {"initiator": "consciousness", "usage_category": "consciousness_task"}},
    })
    fires: list = []
    monkeypatch.setattr(
        queue, "enqueue_task",
        lambda task: fires.append(task["id"]) or {**task, "_admission_blocked": "consciousness_allowance_exhausted",
                                                 "_admission_detail": "$20.00 of $20.00 spent in the last 24 h"})
    before = datetime.datetime.now(datetime.timezone.utc)
    queue.check_scheduled_tasks()
    queue.check_scheduled_tasks()  # the very next pass: NOT due again
    assert len(fires) == 1 and pending == []
    record = queue.list_scheduled_tasks(tmp_path)["tasks"][0]
    assert record["enabled"] is True and not record.get("completed_at")
    assert "consciousness_allowance_exhausted" in str(record.get("last_error") or "")
    run_at = datetime.datetime.fromisoformat(record["trigger"]["run_at"])
    assert run_at >= before + datetime.timedelta(seconds=890)


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

    params = {"run_at": "2030-01-01T00:00:00+00:00",
              "objective": "Re-run the plan panel once the reviewer window resets."}
    if "cron" in kw and "run_at" not in kw:
        params.pop("run_at")
    params.update(kw)
    return _handle_schedule_followup(ctx, **params)


def test_schedule_followup_registers_a_one_shot_entry(tmp_path):
    ctx = _ctx(tmp_path)
    out = _followup(ctx, context="plan review for root-1 was quorum-unreachable")
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


def test_presence_followup_preserves_ceiling_and_return_context(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.task_metadata["presence"] = {"binding_id": "b" * 32}
    ctx.task_contract = {"capability_ceiling": {"digest": "a" * 64}}
    assert _followup(ctx).startswith("FOLLOWUP_SCHEDULED")
    from supervisor import queue

    record = queue.list_scheduled_tasks(tmp_path / "data")["tasks"][0]
    assert record["task"]["metadata"]["presence"] == {"binding_id": "b" * 32}
    assert record["task"]["task_contract"] == ctx.task_contract
    assert record["task"]["metadata"]["origin_task_id"] == "root-1"


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
    assert record["task"]["metadata"]["presence"] == ctx.task_metadata["presence"]
    assert record["task"]["task_contract"] == ctx.task_contract
    scheduled = queue._task_from_schedule(record)
    assert scheduled["metadata"]["presence"] == ctx.task_metadata["presence"]
    assert scheduled["task_contract"]["capability_ceiling"] == ctx.task_contract["capability_ceiling"]


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


def test_schedule_followup_cap_refusal_is_typed_and_disclosed(tmp_path):
    ctx = _ctx(tmp_path)
    assert _followup(ctx).startswith("FOLLOWUP_SCHEDULED")
    assert _followup(ctx, run_at="2030-02-01T00:00:00+00:00").startswith("FOLLOWUP_SCHEDULED")
    third = _followup(ctx, run_at="2030-03-01T00:00:00+00:00")
    assert third.startswith("ERROR: FOLLOWUP_CAP_REACHED")
    assert "2 pending" in third  # discloses the pending records, never silent
    # another task keeps its own budget
    assert _followup(_ctx(tmp_path, task_id="root-2")).startswith("FOLLOWUP_SCHEDULED")


def test_schedule_followup_counts_the_cap_inside_the_write_it_is_about_to_make(tmp_path, monkeypatch):
    """The cap read and the write share ONE hold on the schedule table.

    Two tasks registering at the same instant would otherwise each read the same
    under-cap count and both land, so the cap would be advisory rather than real.
    """
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
    # The refused call takes the same single hold and writes nothing.
    acquisitions.clear()
    assert _followup(ctx, run_at="2030-02-01T00:00:00+00:00").startswith("FOLLOWUP_SCHEDULED")
    acquisitions.clear()
    assert _followup(ctx, run_at="2030-03-01T00:00:00+00:00").startswith("ERROR: FOLLOWUP_CAP_REACHED")
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


def test_schedule_followup_surfaces_a_corrupt_store_as_the_typed_refusal(tmp_path):
    """The lenient cap read tolerates a corrupt table, the strict upsert refuses
    it; that refusal must reach the caller typed, not as FOLLOWUP_PERSIST_FAILED."""
    from supervisor.queue import list_scheduled_tasks

    ctx = _ctx(tmp_path)
    state = tmp_path / "data" / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "scheduled_tasks.json").write_text("{not json", encoding="utf-8")
    text = _followup(ctx)
    assert text.startswith("⚠️ CAPABILITY_UNAVAILABLE: FOLLOWUP_STORE_UNAVAILABLE")
    assert "PERSIST_FAILED" not in text
    assert (state / "scheduled_tasks.json").read_text(encoding="utf-8") == "{not json"
    assert list_scheduled_tasks(tmp_path / "data")["tasks"] == []


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
    assert record["task"]["metadata"]["origin_root_task_id"] == "root-3"


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
    from ouroboros.project_facts import resolve_project_id

    assert _followup(_ctx(tmp_path, task_id="plain-task")).startswith("FOLLOWUP_SCHEDULED")
    record = list_scheduled_tasks(pathlib.Path(tmp_path / "data").resolve())["tasks"][0]
    assert "project_id" not in record["task"]
    assert "chat_id" not in record["task"]

    queued = _task_from_schedule(record)
    assert resolve_project_id(queued) == ""
    assert queued["chat_id"] == 0  # the existing owner_chat_id default, unchanged


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
        "task": {"type": "task", "text": "resume"},
    })
    app = Starlette(routes=[Route("/api/schedules", endpoint=api_schedules_upsert, methods=["POST"])])
    app.state.drive_root = tmp_path
    client = TestClient(app)

    # Bare re-enable with the SAME run_at: refused with a clear re-arm message.
    refused = client.post("/api/schedules", json={
        "id": "fu-done", "enabled": True,
        "trigger": {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"},
        "task": {"type": "task", "text": "resume"},
    })
    assert refused.status_code == 400 and "run_at" in refused.json()["error"]
    # Disable/edit keeping the same run_at: allowed, receipt carried forward.
    kept = client.post("/api/schedules", json={
        "id": "fu-done", "enabled": False,
        "trigger": {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"},
        "task": {"type": "task", "text": "resume"},
    })
    assert kept.status_code == 200
    assert kept.json()["schedule"]["completed_at"] == fired
    # A fresh run_at re-arms: completed_at cleared, and the record fires ONCE.
    rearmed = client.post("/api/schedules", json={
        "id": "fu-done", "enabled": True,
        "trigger": {"type": "once", "run_at": "2000-02-01T00:00:00+00:00"},
        "task": {"type": "task", "text": "resume"},
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
    rows = {row["id"]: row for row in digest["active"]}
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
        "id": "consumed-old", "enabled": False, "completed_at": old,
        "trigger": {"type": "once", "run_at": old},
        "task": {"type": "task", "text": "done long ago"},
    })
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
    # Only the aged-out CONSUMED ONE-SHOT is pruned; a disabled cron row is a
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
    assert set(schema["required"]) == {"objective"}
    assert {"run_at", "cron"} <= set(schema["properties"])
    assert not ({"anyOf", "oneOf", "allOf"} & set(schema))
    from ouroboros.safety import POLICY_SKIP, TOOL_POLICY

    assert TOOL_POLICY["schedule_followup"] == POLICY_SKIP
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


# ------------------------------------------------- kind: "notify" rows (5B)


def _events(tmp_path):
    import json

    path = tmp_path / "logs" / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _notify_row(schedule_id, *, source="task_followup", trigger=None, key="k1", text="Call mother"):
    return {
        "id": schedule_id, "name": f"Reminder ({key})", "kind": "notify", "enabled": True, "source": source,
        "trigger": trigger or {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"},
        "notification": {"text": text, "key": key},
    }


def test_notify_once_row_emits_one_owner_notification_and_is_consumed(tmp_path):
    """The second dispatch verb of the one table: a due ``kind: "notify"`` row
    becomes an ``owner_notification`` events row (its live frame rides the log
    sink) plus a topic publish AFTER the table lock — never a queued task."""
    from ouroboros import event_bus

    queue, pending = _queue(tmp_path)
    bus = event_bus.init_global_event_bus()
    seen: list = []
    bus.subscribe("telegram", event_bus.OWNER_NOTIFICATION, seen.append)
    queue.upsert_scheduled_task(_notify_row("n-due"))
    queue.upsert_scheduled_task(_notify_row("n-future", trigger={"type": "once", "run_at": "2999-01-01T00:00:00+00:00"}))
    queue.check_scheduled_tasks()
    queue.check_scheduled_tasks()  # a consumed one-shot never re-fires
    assert pending == [], "a notify row admits no task"
    notices = [row for row in _events(tmp_path) if row.get("type") == "owner_notification"]
    assert len(notices) == 1
    notice = notices[0]
    assert notice["text"] == "Call mother" and notice["source"] == "task_followup"
    assert notice["chat_id"] == 1 and "task_id" not in notice
    assert notice["scheduled_for"] == "2000-01-01T00:00:00+00:00"
    # The frame key carries the due instant: a re-armed or recurring reminder
    # rings on every occurrence, a crash replay of the same one collapses.
    assert notice["key"] == f"k1@{notice['scheduled_for']}"
    assert len(seen) == 1 and seen[0]["key"] == notice["key"]
    records = {r["id"]: r for r in queue.list_scheduled_tasks(tmp_path)["tasks"]}
    done = records["n-due"]
    assert done["enabled"] is False and done["completed_at"] and done["last_run_at"]
    assert done.get("last_task_id") is None and done["next_run_at"] == ""
    assert records["n-future"]["enabled"] is True and not records["n-future"].get("last_run_at")
    event_bus.init_global_event_bus()


def test_notify_cron_row_fires_and_advances_like_a_task_row(tmp_path):
    queue, pending = _queue(tmp_path)
    queue.upsert_scheduled_task(_notify_row(
        "n-cron", trigger={"type": "cron", "expr": "* * * * *"}, key="daily"))
    # Force the first occurrence into the past, as an offline gap would.
    store = queue.load_schedule_store(tmp_path)
    store["tasks"][0]["next_run_at"] = "2000-01-01T00:00:00+00:00"
    from supervisor import queue_schedules

    queue_schedules._write_scheduled_tasks(store, tmp_path)
    queue.check_scheduled_tasks()
    queue.check_scheduled_tasks()  # the next occurrence is in the future now
    assert pending == []
    notices = [row for row in _events(tmp_path) if row.get("type") == "owner_notification"]
    assert len(notices) == 1 and notices[0]["key"] == "daily@2000-01-01T00:00:00+00:00"
    record = queue.list_scheduled_tasks(tmp_path)["tasks"][0]
    assert record["enabled"] is True and not record.get("completed_at")
    assert record["next_run_at"] > "2000-01-02"


def test_notify_row_of_a_disabled_or_missing_skill_stays_silent(tmp_path):
    """The resync never touches ``skill:`` rows, so the tick itself must not
    ring for a skill the owner switched off or removed."""
    from ouroboros.skill_loader import save_enabled

    queue, pending = _queue(tmp_path)
    skill_dir = tmp_path / "skills" / "external" / "cal"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: cal\ndescription: calendar\nversion: 0.1\ntype: extension\nentry: plugin.py\n"
        "permissions: [notify_owner]\n---\n# cal\n", encoding="utf-8")
    (skill_dir / "plugin.py").write_text("def register(api): pass\n", encoding="utf-8")
    queue.upsert_scheduled_task(_notify_row("n-ghost", source="skill:ghost", key="g"))
    queue.upsert_scheduled_task(_notify_row("n-cal", source="skill:cal", key="c"))
    save_enabled(tmp_path, "cal", False)
    queue.check_scheduled_tasks()
    assert [row for row in _events(tmp_path) if row.get("type") == "owner_notification"] == []
    save_enabled(tmp_path, "cal", True)
    queue.check_scheduled_tasks()
    notices = [row for row in _events(tmp_path) if row.get("type") == "owner_notification"]
    assert [n["source"] for n in notices] == ["skill:cal"]
    records = {r["id"]: r for r in queue.list_scheduled_tasks(tmp_path)["tasks"]}
    assert records["n-ghost"]["enabled"] is True and not records["n-ghost"].get("completed_at")
    assert records["n-cal"]["enabled"] is False and records["n-cal"]["completed_at"]


def test_notify_row_survives_a_failed_append_and_retries(tmp_path, monkeypatch):
    from ouroboros import utils

    queue, pending = _queue(tmp_path)
    queue.upsert_scheduled_task(_notify_row("n-retry"))
    real_append = utils.append_jsonl
    monkeypatch.setattr(utils, "append_jsonl", lambda path, obj, **kw: False if path.name == "events.jsonl" and obj.get("type") == "owner_notification" else real_append(path, obj, **kw))
    queue.check_scheduled_tasks()
    record = queue.list_scheduled_tasks(tmp_path)["tasks"][0]
    assert record["enabled"] is True and not record.get("completed_at")
    assert "notification log write failed" in str(record.get("last_error") or "")
    monkeypatch.setattr(utils, "append_jsonl", real_append)
    queue.check_scheduled_tasks()
    record = queue.list_scheduled_tasks(tmp_path)["tasks"][0]
    assert record["enabled"] is False and record["completed_at"] and record.get("last_error") == ""
    assert len([row for row in _events(tmp_path) if row.get("type") == "owner_notification"]) == 1


def test_notify_row_projects_its_text_as_the_model_preview_and_audits_its_kind(tmp_path):
    queue, _pending = _queue(tmp_path)
    queue.upsert_scheduled_task(_notify_row("n-view", text="Dentist at 9"))
    from supervisor.queue_schedules import _schedule_projection_row

    row = _schedule_projection_row(queue.list_scheduled_tasks(tmp_path)["tasks"][0])
    assert row["kind"] == "notify" and row["objective_preview"] == "Dentist at 9"
    audits = [row for row in _events(tmp_path) if row.get("type") == "schedule_mutation"]
    assert audits and all("Dentist" not in json_dumps(a) for a in audits), "the sentence never enters the audit"
    assert any((a.get("after") or {}).get("kind") == "notify" for a in audits)


def json_dumps(value):
    import json

    return json.dumps(value, ensure_ascii=False)


def test_suppressed_notify_row_never_fires_until_the_owner_restores_it(tmp_path):
    queue, pending = _queue(tmp_path)
    queue.upsert_scheduled_task(_notify_row("n-off"))
    off = queue.mutate_scheduled_task("delete", "n-off", reason="owner: stop", actor="owner:gateway", drive_root=tmp_path)
    assert off["status"] == "suppressed"
    queue.check_scheduled_tasks()
    assert [row for row in _events(tmp_path) if row.get("type") == "owner_notification"] == []
    record = queue.list_scheduled_tasks(tmp_path)["tasks"][0]
    assert queue.schedule_lifecycle_status(record) == "suppressed" and record["enabled"] is False
    # An upsert of the same row (the skill re-posting) cannot re-arm it either.
    queue.upsert_scheduled_task({**_notify_row("n-off"), "trigger": {"type": "once", "run_at": "2000-01-02T00:00:00+00:00"}})
    queue.check_scheduled_tasks()
    assert [row for row in _events(tmp_path) if row.get("type") == "owner_notification"] == []
    assert queue.mutate_scheduled_task("restore", "n-off", reason="owner: back", actor="owner:gateway", drive_root=tmp_path)["ok"] is True
    queue.check_scheduled_tasks()
    assert len([row for row in _events(tmp_path) if row.get("type") == "owner_notification"]) == 1


def test_unknown_schedule_kind_is_left_untouched_with_a_typed_error(tmp_path):
    queue, pending = _queue(tmp_path)
    queue.upsert_scheduled_task({**_notify_row("n-weird"), "kind": "future_action"})
    queue.check_scheduled_tasks()
    assert pending == [], "an unknown verb never dispatches a model task"
    record = queue.list_scheduled_tasks(tmp_path)["tasks"][0]
    assert record["enabled"] is True and not record.get("completed_at") and not record.get("last_run_at")
    assert "unsupported schedule kind: future_action" in record["last_error"]
    assert [row for row in _events(tmp_path) if row.get("type") == "owner_notification"] == []


def test_notify_row_with_the_longest_producer_key_still_fires(tmp_path):
    """The route accepts 128-character keys; the occurrence key must fit the
    same cap, so a long key rides as its digest instead of never firing."""
    queue, pending = _queue(tmp_path)
    long_key = "k" * 128
    queue.upsert_scheduled_task(_notify_row("n-long", key=long_key))
    queue.check_scheduled_tasks()
    notices = [row for row in _events(tmp_path) if row.get("type") == "owner_notification"]
    assert len(notices) == 1 and len(notices[0]["key"]) <= 128 and notices[0]["key"].endswith("@2000-01-01T00:00:00+00:00")
    record = queue.list_scheduled_tasks(tmp_path)["tasks"][0]
    assert record["enabled"] is False and record["completed_at"] and record.get("last_error") == ""


def test_notify_row_that_the_emitter_rejects_records_why(tmp_path):
    queue, pending = _queue(tmp_path)
    queue.upsert_scheduled_task({**_notify_row("n-empty"), "notification": {"text": "   ", "key": "e"}})
    queue.check_scheduled_tasks()
    record = queue.list_scheduled_tasks(tmp_path)["tasks"][0]
    assert record["enabled"] is True and not record.get("completed_at")
    assert "invalid notification" in str(record.get("last_error") or ""), "the durable row says why it never rings"
    assert [row for row in _events(tmp_path) if row.get("type") == "owner_notification"] == []
