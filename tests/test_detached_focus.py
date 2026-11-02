"""A direct root keeps its own focus after handing the same work to the pool (#2494).

``is_direct_chat`` stays the historical origin and ``_execution_owner`` names the current carrier.
The seam runs the real arm, checkpoint, park, wake, assignment and worker materialization, then the
real tool dispatch; only the model and the worker process are absent.
"""

import copy
import json
import queue as queue_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import budget_pause as bp
from ouroboros.task_carrier import execution_kind
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_project_hold_recovery import worker
from tests.test_swarm_host_admission import host as host

pytestmark = pytest.mark.serial

TASK_ID = "head"
RECENT = {"reader": "recent_tasks"}


def _actor(host, monkeypatch, events):
    """One carrier's real agent with no model: the same class a worker or the chat lane builds."""
    from ouroboros.agent import OuroborosAgent, make_agent
    from supervisor import workers

    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr("ouroboros.agent.build_llm_messages", lambda **_kw: ([], {}))
    return make_agent(str(workers.REPO_DIR), str(host.root), event_queue=events)


def _materialize(actor, task):
    """The start path both carriers share: turn binding, the RUNNING record and the ToolContext."""
    actor._bind_turn(task)
    ctx, _messages, _caps = actor._prepare_task_context(task)
    return ctx


def _publish(actor, text, source=RECENT):
    """The model's tool call through the loop's own dispatch, guards included."""
    from ouroboros.loop_tool_execution import _execute_single_tool

    call = {"id": "focus-call", "function": {"name": "update_focus",
                                             "arguments": json.dumps({"text": text, "source_ref": source})}}
    tools = actor.tools
    return _execute_single_tool(tools, call, Path(tools._ctx.drive_root) / "logs", tools._ctx.task_id)["result"]


def _focus(root):
    return (load_task_result(root, TASK_ID) or {}).get("focus")


def _focus_events(events):
    """The focus events this carrier emitted since the last look (its other live events aside)."""
    drained = []
    while not events.empty():
        drained.append(events.get_nowait())
    return [event for event in drained if event.get("type") == "task_focus_updated"]


@pytest.fixture
def direct(host, monkeypatch):
    """A live direct root on the real chat-lane carrier, past its pre-effect checkpoint."""
    from ouroboros import model_detach, working_checkpoint
    from ouroboros.owner_wait import direct_owner_wait
    from supervisor import active_activity, queue

    sent = worker(host, monkeypatch)
    monkeypatch.setattr(host.ctx.bridge, "push_log", lambda _event: None, raising=False)
    monkeypatch.setattr(model_detach, "pool_available", lambda: {"available": True})
    registry = active_activity.DirectActivityRegistry()
    monkeypatch.setattr(active_activity, "_DIRECT_ACTIVITY_REGISTRY", registry)
    events = queue_module.Queue()
    actor = _actor(host, monkeypatch, events)
    actor.owner_wait_callback = direct_owner_wait
    task = {"id": TASK_ID, "type": "task", "root_task_id": TASK_ID, "chat_id": 1, "text": "work",
            "_is_direct_chat": True, "_attempt": 1, "tool_profile": "operator_control"}
    ctx = _materialize(actor, task)
    registry.register(TASK_ID, 1, actor=actor)
    # What the loop binds around a model turn; the fields it carries while no model call is made.
    ctx.task_started_at = actor._task_started_ts or 1.0
    ctx.active_model, ctx.active_effort, ctx.active_use_local = "m", "high", False
    ctx.active_context_mode, ctx.context_fit_plan, ctx.model_wait_context = "max", None, None
    ctx._cost_ceiling, ctx._owner_directives, ctx._delivery_candidate = None, [], None
    ctx._accumulated_usage = {}
    limit = SimpleNamespace(
        tools=actor.tools, accumulated_usage={}, llm_trace={"tool_calls": []}, owner_msg_seen=set(),
        round_idx=2, tool_schemas=[{"type": "function", "function": {"name": "update_focus"}}],
        messages=[{"role": "user", "content": "work"},
                  {"role": "assistant", "content": "", "tool_calls": [
                      {"id": "call_b", "type": "function", "function": {"name": "y", "arguments": "{}"}}]},
                  {"role": "tool", "tool_call_id": "call_b", "content": "done b"}])
    assert working_checkpoint.save_round(limit, "pre_effect")
    assert queue.persist_queue_snapshot(reason="direct-before-detach")
    return SimpleNamespace(actor=actor, ctx=ctx, task=task, events=events, limit=limit, registry=registry, sent=sent)


def _handed_to_pool(direct):
    """``continue_in_background`` to the row a worker receives, through the real owners only."""
    from ouroboros import working_checkpoint
    from ouroboros.tools.control_mind import _continue_in_background
    from supervisor import queue, workers
    from supervisor.direct_pause import park_direct_pause
    from supervisor.sleep_wake import wake_ready_sleepers

    armed = json.loads(_continue_in_background(direct.ctx, "continue the same work"))
    assert armed["status"] == "detach_armed", armed
    assert working_checkpoint.save_round(direct.limit, "post_batch")
    with pytest.raises(bp.BudgetPauseRequested) as stopped:
        bp.enter_cold_sleep(direct.limit)
    assert park_direct_pause(direct.task, bp.pause_event(direct.task, stopped.value.pause))
    direct.registry.unregister(TASK_ID)
    assert any(outcome.get("ok") for outcome in wake_ready_sleepers(queue))
    workers.assign_tasks()
    [handed] = direct.sent
    return handed


def _started_in_pool(direct, host, monkeypatch):
    """The same work after handoff, on the pooled worker's own start and the loop's Resume consumption."""
    from ouroboros import owner_wait

    handed = _handed_to_pool(direct)
    events = queue_module.Queue()
    actor = _actor(host, monkeypatch, events)
    ctx = _materialize(actor, copy.deepcopy(handed))
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a: (None, "max"))  # no model route here
    bp.resume_paused_loop(actor.tools, bp.load_budget_pause(ctx), [], {}, {}, set(), budget_remaining_usd=5.0)
    return SimpleNamespace(actor=actor, ctx=ctx, handed=handed, events=events, direct=direct)


@pytest.fixture
def pooled(direct, host, monkeypatch):
    return _started_in_pool(direct, host, monkeypatch)


def test_handed_off_root_publishes_through_every_real_consumer(pooled, host):
    from ouroboros.focus import compact_focus
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from ouroboros.gateway.work_projection import works_snapshot
    from ouroboros.live_works import live_works_present
    from ouroboros.peer_roster import independent_roots, render_roster_note
    from supervisor.events_worker_reports import _handle_task_focus_updated

    # The whole transition, as its owners recorded it: the origin stays, the carrier moved.
    assert pooled.handed["_is_direct_chat"] is True and pooled.handed["_execution_owner"]["kind"] == "pooled"
    assert pooled.ctx.is_direct_chat is True and execution_kind(pooled.ctx) == "pooled"
    stored = load_task_result(host.root, TASK_ID)
    assert stored["status"] == "running" and stored["_is_direct_chat"] is True
    assert stored["execution_owner"] == pooled.ctx.task_metadata["_execution_owner"]

    text = "Continuing the same work in the background"
    answer = _publish(pooled.actor, text)
    assert answer.startswith(f"OK: focus[{TASK_ID}]"), answer
    focus = _focus(host.root)
    assert focus["text"] == text and focus["author_task_id"] == TASK_ID and focus["source_handle"]

    # The supervisor's projection of the event this carrier emitted...
    [event] = _focus_events(pooled.events)
    assert event == {"type": "task_focus_updated", "task_id": TASK_ID, "focus": focus}
    _handle_task_focus_updated(event, host.ctx)
    assert compact_focus(host.running[TASK_ID]["task"]["focus"]) == compact_focus(focus)
    # ...and what the other live roots and the owner read from the queue's own records.
    [row] = [row for row in independent_roots(host.root)["roots"] if row["task_id"] == TASK_ID]
    assert row["status"] == "running" and row["direct_chat"] is False and row["focus"]["text"] == text
    assert text in render_roster_note(independent_roots(host.root))
    [work] = [work for work in live_works_present(host.root)["works"] if work["task_id"] == TASK_ID]
    assert work["last_line"] == text
    # The owner's card, composed as /api/state composes it.
    owners, facts = [], {}
    _chat_activities_snapshot_safe(host.root, decision_owners=owners, work_facts=facts)
    [card] = [work for work in works_snapshot(host.root, owners, facts)["works"] if TASK_ID in work["members"]["ids"]]
    assert card["focus"]["text"] == text


def test_reported_path_project_move_handoff_then_a_journal_source(direct, host, monkeypatch):
    """#2494 as observed: an owner turn moved to a Project, continued in the pool, wrote its journal."""
    from ouroboros.loop_tool_execution import _execute_single_tool
    from ouroboros.project_placement import move_task_to_project, placement_basis
    from ouroboros.projects_registry import project_id_for_task

    move_task_to_project(host.root, TASK_ID, "alpha", name="Alpha", origin={"absent": "mid_task_no_origin"},
                         basis=placement_basis(host.root, TASK_ID, "alpha"), running={}, pending=[])
    assert project_id_for_task(host.root, TASK_ID, strict=True) == "alpha"
    pooled = _started_in_pool(direct, host, monkeypatch)
    assert pooled.ctx.project_id == "alpha" and execution_kind(pooled.ctx) == "pooled"
    call = {"id": "journal-call", "function": {"name": "journal_write", "arguments": json.dumps(
        {"kind": "note", "text": "Milestone after the handoff"})}}
    written = _execute_single_tool(pooled.actor.tools, call, Path(pooled.ctx.drive_root) / "logs", TASK_ID)
    assert not written["is_error"], written
    answer = _publish(pooled.actor, "Reading the Project's own journal", {"reader": "journal_read"})
    assert answer.startswith(f"OK: focus[{TASK_ID}]"), answer
    focus = _focus(host.root)
    assert focus["source_ref"] == {"reader": "journal_read", "project_id": "alpha"}
    retained = host.root / "task_results" / "artifacts" / TASK_ID / focus["source_handle"]["path"]
    assert "Milestone after the handoff" in retained.read_text(encoding="utf-8")
    assert len(_focus_events(pooled.events)) == 1


def test_focus_published_before_the_handoff_survives_it_and_is_revised_after(direct, host, monkeypatch):
    assert _publish(direct.actor, "Before the handoff").startswith("OK:")
    before = _focus(host.root)
    pooled = _started_in_pool(direct, host, monkeypatch)
    assert _focus(host.root) == before
    assert _publish(pooled.actor, "After the handoff").startswith("OK:")
    after = _focus(host.root)
    assert after["text"] == "After the handoff" and after["authored_at"] > before["authored_at"]


def test_superseded_direct_actor_cannot_publish_beside_its_pooled_successor(pooled, host):
    """The registry's absence ends the OLD carrier; it never turns that carrier into a pooled one."""
    assert _publish(pooled.actor, "From the current attempt").startswith("OK:")
    published = _focus(host.root)
    refused = _publish(pooled.direct.actor, "From the actor that already left")
    assert "FOCUS_TASK_NOT_LIVE" in refused
    assert _focus(host.root) == published and execution_kind(pooled.direct.ctx) == "direct"
    assert _focus_events(pooled.direct.events) == []


def _superseding_attempt(pooled, host, monkeypatch):
    """The queue hands the same id to a new attempt: its assignment, then its native start."""
    from supervisor import worker_assignment

    retry = copy.deepcopy(pooled.handed)
    retry["_attempt"] = 2
    for key in ("_budget_pause_resume", "_execution_owner", "admitted_dispatch"):
        retry.pop(key, None)
    worker_assignment._mirror_assigned_running_status(retry)
    assert retry["_execution_owner"]["kind"] == "pooled" and retry["_execution_owner"]["task_attempt"] == 2
    actor = _actor(host, monkeypatch, queue_module.Queue())
    ctx = _materialize(actor, retry)
    assert load_task_result(host.root, TASK_ID)["execution_owner"] == ctx.task_metadata["_execution_owner"]
    return actor


def _end(pooled, host, monkeypatch):
    write_task_result(host.root, TASK_ID, "completed", result="done")


@pytest.mark.parametrize("change", [_end, _superseding_attempt], ids=["ended", "superseded"])
@pytest.mark.parametrize("moment", ["before_the_call", "while_retaining_the_source", "before_the_locked_write"])
def test_ended_or_superseded_pooled_attempt_writes_nothing(pooled, host, monkeypatch, change, moment):
    from ouroboros import task_results
    from ouroboros.tools import project_journal

    changed = []
    if moment == "before_the_call":
        changed.append(change(pooled, host, monkeypatch))
    elif moment == "while_retaining_the_source":  # past the first liveness read
        real_retain = project_journal._retain_focus_source

        def retain(*args, **kwargs):
            changed.append(change(pooled, host, monkeypatch))
            return real_retain(*args, **kwargs)

        monkeypatch.setattr(project_journal, "_retain_focus_source", retain)
    else:  # past the second read too: only the projector, under the row's own lock, still decides
        real_write = task_results.write_task_result

        def write(root, task_id, status, **fields):
            if "focus" in fields and not changed:
                changed.append(change(pooled, host, monkeypatch))
            return real_write(root, task_id, status, **fields)

        monkeypatch.setattr(task_results, "write_task_result", write)
    refused = _publish(pooled.actor, "Too late")
    assert changed, f"the attempt was refused before it reached this moment: {refused}"
    assert "FOCUS_TASK_NOT_LIVE" in refused and "OK:" not in refused
    assert _focus(host.root) is None and _focus_events(pooled.events) == []
    if changed[0] is not None:  # the surviving path: the attempt the row names still speaks
        assert _publish(changed[0], "From the current attempt").startswith("OK:")
        assert _focus(host.root)["text"] == "From the current attempt"


def test_source_and_newer_focus_checks_still_bind_after_handoff(pooled, host):
    from ouroboros.focus import normalize_focus

    missing = _publish(pooled.actor, "Pointing at a ghost", {"reader": "get_task_result", "task_id": "missing1"})
    assert "FOCUS_SOURCE_UNRESOLVED" in missing and _focus(host.root) is None and _focus_events(pooled.events) == []
    newer = normalize_focus("Already newer", RECENT, task_id=TASK_ID, authored_at="2999-01-01T00:00:00+00:00")
    write_task_result(host.root, TASK_ID, "running", focus=newer)
    stale = _publish(pooled.actor, "Older than the record")
    assert "FOCUS_STALE" in stale and _focus(host.root) == newer and _focus_events(pooled.events) == []


@pytest.mark.parametrize("ending", ["turn_released", "row_settled"])
def test_live_direct_root_keeps_its_actor_owned_publication(direct, host, ending):
    from supervisor.workers import direct_chat_turn

    assert execution_kind(direct.ctx) == "direct"
    answer = _publish(direct.actor, "Working in this chat")
    assert answer.startswith("OK:"), answer
    focus = _focus(host.root)
    assert focus["text"] == "Working in this chat" and direct_chat_turn(TASK_ID)["focus"] == focus
    assert len(_focus_events(direct.events)) == 1
    if ending == "turn_released":  # still registered, with a RUNNING row: only the actor ended
        direct.actor._release_turn()
    else:  # the actor still bound, the durable row already settled
        write_task_result(host.root, TASK_ID, "completed", result="done")
    assert "FOCUS_TASK_NOT_LIVE" in _publish(direct.actor, "After the end")
    assert _focus(host.root) == focus and _focus_events(direct.events) == []


def test_a_live_direct_actor_is_not_held_to_the_rows_owner_record(direct, host):
    """Only a pooled attempt answers to the row's owner; the live direct actor to its own registration."""
    recorded = load_task_result(host.root, TASK_ID)["execution_owner"]
    write_task_result(host.root, TASK_ID, "running", execution_owner={**recorded, "process": "an-earlier-process"})
    assert _publish(direct.actor, "Still this turn").startswith("OK:")


@pytest.mark.parametrize("owner", [None, {"kind": "unrecognized"}, "direct"], ids=["legacy", "malformed", "not_a_record"])
@pytest.mark.parametrize("origin_direct,expected", [(True, "FOCUS_TASK_NOT_LIVE"), (False, "OK:")],
                         ids=["direct_origin", "queued_origin"])
def test_absent_or_malformed_owner_is_no_fact_and_the_origin_decides(tmp_path, owner, origin_direct, expected):
    from ouroboros.task_status import execution_owner_record
    from ouroboros.tools.project_journal import _update_focus

    recorded = execution_owner_record(tmp_path, {"id": TASK_ID, "_attempt": 1}, "pooled")
    write_task_result(tmp_path, TASK_ID, "running", _is_direct_chat=origin_direct, task_attempt=1, execution_owner=recorded)
    ctx = SimpleNamespace(task_id=TASK_ID, drive_root=tmp_path, project_id="", is_direct_chat=origin_direct,
                          task_metadata={"root_task_id": TASK_ID, "_execution_owner": owner})
    assert expected in _update_focus(ctx, "Inferred nothing", RECENT)
    assert bool(_focus(tmp_path)) is (expected == "OK:")


@pytest.mark.parametrize("kind,metadata,flags", [
    ("pooled", {"delegation_role": "subagent", "parent_task_id": "head", "root_task_id": "head"}, {}),
    ("presence", {"source": "presence", "presence": {"binding_id": "b"}}, {"_presence_turn": True}),
    ("pooled", {"source": "presence_promote", "presence": {"binding_id": "b"}}, {}),
    ("pooled", {"presence_binding_authority": {"binding_id": "b"}, "root_task_id": TASK_ID}, {}),
], ids=["child", "presence_turn", "presence_origin_on_the_pool", "acting_for_a_binding"])
def test_the_carrier_grants_no_child_or_presence_authority(tmp_path, kind, metadata, flags):
    from ouroboros.task_status import execution_owner_record
    from ouroboros.tools.project_journal import _update_focus

    owner = execution_owner_record(tmp_path, {"id": TASK_ID, "_attempt": 1}, kind)
    write_task_result(tmp_path, TASK_ID, "running", _is_direct_chat=True, task_attempt=1, execution_owner=owner)
    ctx = SimpleNamespace(task_id=TASK_ID, drive_root=tmp_path, project_id="", is_direct_chat=True,
                          task_metadata={**metadata, "_execution_owner": owner}, **flags)
    assert "TOOL_FORBIDDEN" in _update_focus(ctx, "Not mine to publish", RECENT)
    assert _focus(tmp_path) is None
