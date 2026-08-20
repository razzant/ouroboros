"""One-shot deferred follow-ups (B2b, W=A): the ``once`` trigger's pure selection
logic, the supervisor queue firing/mark-done semantics, and the agent-facing
``schedule_followup`` tool (registration surfaces, authority guard, typed cap).

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
from ouroboros.tools.tool_result import (
    LegacyTextResultAdapter,
    ToolResult,
    _install_tool_result_sidecar,
    _published_tool_result,
    _restore_tool_result_sidecar,
)

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

    queue.init(tmp_path)  # v7 retired the three timeout parameters (D04)
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
    """One registration, returning the NATIVE result the producer published.

    ``registry_core`` installs a per-invocation sentinel and accepts a published
    result only when its text is exactly the string the handler returned, so the
    helper pins both halves: the string ABI the model sees is unchanged, and the
    typed answer beside it is the producer's own. Callers read ``.text`` for every
    assertion they made before.
    """
    from ouroboros.tools.followup import _handle_schedule_followup

    params = {"run_at": "2030-01-01T00:00:00+00:00",
              "objective": "Re-run the plan panel once the reviewer window resets."}
    params.update(kw)
    sentinel = object()
    token = _install_tool_result_sidecar(ctx, sentinel)
    try:
        text = _handle_schedule_followup(ctx, **params)
        published = _published_tool_result(ctx, sentinel)
    finally:
        _restore_tool_result_sidecar(token)
    assert isinstance(published, ToolResult), "schedule_followup published no typed result"
    assert published.text == text, "published text is not the returned text"
    # Owner item A.22 (owner decision 2026-08-19, "B"): EVERY sentence this tool
    # writes is markerless, so the single adapter answers `ok` for all of them and
    # a refused follow-up used to read exactly like a registered one. Asserted here
    # rather than restated per call site: the divergence the owner approved has to
    # be real at every terminal, and a marker quietly added to one of these texts
    # would make the adapter agree by accident and hide the producer's own answer.
    assert LegacyTextResultAdapter.from_text("schedule_followup", text).code == "OK"
    return published


def test_schedule_followup_registers_a_one_shot_entry(tmp_path):
    ctx = _ctx(tmp_path)
    out = _followup(ctx, context="plan review for root-1 was quorum-unreachable")
    assert out.text.startswith("FOLLOWUP_SCHEDULED")
    assert (out.code, out.status) == ("OK", "ok")
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
    assert record["task"]["metadata"]["origin_task_id"] == "root-1"


def test_schedule_followup_cap_refusal_is_typed_and_disclosed(tmp_path):
    ctx = _ctx(tmp_path)
    assert _followup(ctx).text.startswith("FOLLOWUP_SCHEDULED")
    assert _followup(ctx, run_at="2030-02-01T00:00:00+00:00").text.startswith("FOLLOWUP_SCHEDULED")
    third = _followup(ctx, run_at="2030-03-01T00:00:00+00:00")
    assert third.text.startswith("ERROR: FOLLOWUP_CAP_REACHED")
    # The per-task budget refused to mint the future task: the same code the
    # subtask depth limit publishes, because it is the same kind of refusal.
    assert (third.code, third.status) == ("RESOURCE_CONSTRAINT_BLOCKED", "blocked")
    assert "2 pending" in third.text  # discloses the pending records, never silent
    # another task keeps its own budget
    assert _followup(_ctx(tmp_path, task_id="root-2")).text.startswith("FOLLOWUP_SCHEDULED")


def test_schedule_followup_overlong_text_is_a_typed_refusal_never_truncated(tmp_path):
    """Review fix 7: the objective/context ride VERBATIM into the future task, so an
    over-limit text is a typed FOLLOWUP_TEXT_TOO_LONG refusal naming the limit —
    never a silent cut that changes what the future task is."""
    from ouroboros.tools.followup import _MAX_CONTEXT_CHARS, _MAX_OBJECTIVE_CHARS

    ctx = _ctx(tmp_path)
    long_objective = _followup(ctx, objective="x" * (_MAX_OBJECTIVE_CHARS + 1))
    assert long_objective.text.startswith("ERROR: FOLLOWUP_TEXT_TOO_LONG")
    assert str(_MAX_OBJECTIVE_CHARS) in long_objective.text
    assert (long_objective.code, long_objective.status) == ("TOOL_ARG_ERROR", "error")
    long_context = _followup(ctx, context="y" * (_MAX_CONTEXT_CHARS + 1))
    assert long_context.text.startswith("ERROR: FOLLOWUP_TEXT_TOO_LONG")
    assert str(_MAX_CONTEXT_CHARS) in long_context.text
    assert (long_context.code, long_context.status) == ("TOOL_ARG_ERROR", "error")
    from supervisor.queue import list_scheduled_tasks

    assert list_scheduled_tasks(pathlib.Path(tmp_path / "data").resolve())["tasks"] == []
    # At-limit text is accepted whole, byte-for-byte.
    ok = _followup(ctx, objective="z" * _MAX_OBJECTIVE_CHARS)
    assert ok.text.startswith("FOLLOWUP_SCHEDULED")
    assert (ok.code, ok.status) == ("OK", "ok")
    record = list_scheduled_tasks(pathlib.Path(tmp_path / "data").resolve())["tasks"][0]
    assert record["task"]["text"] == "z" * _MAX_OBJECTIVE_CHARS


def test_schedule_followup_guards_authority_and_inputs(tmp_path):
    # narrower-than-parent: a delegated subagent may not mint future root tasks.
    # An authority denial, like the acting-child guards in control_scheduling.
    sub = _followup(_ctx(tmp_path, role="subagent"))
    assert sub.text.startswith("ERROR: FOLLOWUP_SUBAGENT_REFUSED")
    assert (sub.code, sub.status) == ("ACCESS_BLOCKED", "blocked")
    # a real task id is required for the durable per-task cap. The agent cannot
    # supply one, so this is the substrate saying no, not a malformed call.
    no_task = _followup(_ctx(tmp_path, task_id=""))
    assert no_task.text.startswith("ERROR: FOLLOWUP_TASK_ID_REQUIRED")
    assert (no_task.code, no_task.status) == ("LEGACY_UNAVAILABLE", "unavailable")
    # The agent's own malformed call stays a degrading argument error.
    ctx = _ctx(tmp_path)
    bad_run_at = _followup(ctx, run_at="soon")
    assert bad_run_at.text.startswith("ERROR: FOLLOWUP_RUN_AT_INVALID")
    assert (bad_run_at.code, bad_run_at.status) == ("TOOL_ARG_ERROR", "error")
    no_objective = _followup(ctx, objective="  ")
    assert no_objective.text.startswith("ERROR: FOLLOWUP_OBJECTIVE_REQUIRED")
    assert (no_objective.code, no_objective.status) == ("TOOL_ARG_ERROR", "error")
    from supervisor.queue import list_scheduled_tasks

    assert list_scheduled_tasks(pathlib.Path(tmp_path / "data").resolve())["tasks"] == []


def test_schedule_followup_host_failures_are_typed_and_register_nothing(tmp_path, monkeypatch):
    """The two host failures name themselves instead of riding out as ok text.

    A drive root that would not resolve and a table that refused the write are
    both `nothing was registered` — the agent must not go on waiting for an
    instant no record will ever fire at."""
    import ouroboros.tool_access as tool_access
    from supervisor import queue

    ctx = _ctx(tmp_path)

    def _no_root(_ctx):
        raise ValueError("task drive root is not under the owner data root")

    monkeypatch.setattr(tool_access, "canonical_data_root", _no_root)
    unresolved = _followup(ctx)
    assert unresolved.text.startswith("ERROR: FOLLOWUP_DATA_ROOT_UNRESOLVED")
    assert (unresolved.code, unresolved.status) == ("TOOL_ERROR", "error")

    monkeypatch.undo()

    def _no_space(_record, drive_root=None):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(queue, "upsert_scheduled_task", _no_space)
    persist = _followup(ctx)
    assert persist.text.startswith("ERROR: FOLLOWUP_PERSIST_FAILED")
    assert (persist.code, persist.status) == ("TOOL_ERROR", "error")
    monkeypatch.undo()
    assert queue.list_scheduled_tasks(pathlib.Path(tmp_path / "data").resolve())["tasks"] == []


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
    assert _followup(ctx).text.startswith("FOLLOWUP_SCHEDULED")
    from supervisor.queue import list_scheduled_tasks

    record = list_scheduled_tasks(pathlib.Path(tmp_path / "data").resolve())["tasks"][0]
    assert record["task"]["metadata"]["origin_root_task_id"] == "root-3"


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

    queue.init(tmp_path)  # v7 retired the three timeout parameters (D04)
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

    queue.init(tmp_path)  # v7 retired the three timeout parameters (D04)
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

    queue.init(tmp_path)  # v7 retired the three timeout parameters (D04)
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
    # v7 split: check_scheduled_tasks and the durable writer both live in
    # supervisor/queue_schedules.py (queue re-exports the writer), so the tick is
    # intercepted at its OWNER — patching the facade name would never be called.
    from supervisor import queue_schedules

    real_write = queue_schedules._write_scheduled_tasks
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
    assert [e.name for e in entries] == ["schedule_followup"]
    schema = entries[0].schema["parameters"]
    assert set(schema["required"]) == {"run_at", "objective"}
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
    # v7 derives the frozen module list by AST scan instead of carrying a literal
    # (ouroboros/tool_module_inventory.py), and ToolRegistry only caches it once a
    # registry loads its catalog — so the inventory itself is what to assert.
    import pathlib as _pathlib

    from ouroboros.tool_module_inventory import tool_modules_for_runtime
    from ouroboros.tools.registry_core import _FROZEN_TOOL_MANIFEST_PATH

    modules, inventory_errors = tool_modules_for_runtime(
        _pathlib.Path(__file__).resolve().parents[1] / "ouroboros" / "tools",
        _FROZEN_TOOL_MANIFEST_PATH,
    )
    assert not inventory_errors
    assert "followup" in modules
