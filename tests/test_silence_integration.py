"""Focused #1376 consumer probes; no provider, real process or real control effects."""
import json
from types import SimpleNamespace

import pytest

from ouroboros.task_results import load_task_result, write_task_result
from supervisor import queue
from tests.test_silence_diagnostic_boundaries import (
    attention as attention, host as host,
    no_external_effects as no_external_effects, _agent, _mint,
)
from tests.test_silence_attention import observation


def revoke(attention, monkeypatch, change):
    if change == "progress":
        fact = {**observation(attention), "state": "progress_observed"}
        write_task_result(attention.root, "root", "running", silence_attention=fact)
    elif change == "memory_progress":
        attention.meta["last_progress_at"] = attention.clock[0]
    elif change == "attempt":
        write_task_result(attention.root, "root", "running", task_attempt=2)
    elif change == "memory_attempt":
        attention.meta["attempt"] = 2
    elif change == "memory_pause":
        attention.meta["task"]["_budget_pause_hold"] = {"reason": "owner"}
    elif change == "stop":
        from ouroboros.cancel_intents import request_cancel
        request_cancel(attention.root, "root", reason="fixture owner Stop", source="owner", scope="single")
    elif change == "panic":
        monkeypatch.setattr("ouroboros.installation_stop.installation_panic", lambda *a, **k: {"active": True})
    elif change == "missing":
        real = load_task_result
        monkeypatch.setattr("ouroboros.task_results.load_task_result",
                            lambda root, tid, **kw: None if tid == "root" else real(root, tid, **kw))
    elif change == "unreadable":
        from ouroboros.store import StoreError
        real = load_task_result
        def read(root, tid, **kw):
            if tid == "root":
                raise StoreError("fixture source unreadable")
            return real(root, tid, **kw)
        monkeypatch.setattr("ouroboros.task_results.load_task_result", read)


@pytest.mark.parametrize("change", ["progress", "attempt", "stop", "panic", "missing", "unreadable"])
def test_control_request_rechecks_after_model_response(attention, monkeypatch, change):
    from ouroboros.tool_call_log import invocation_scope
    child = _mint(attention)
    registry, ctx = _agent(attention, monkeypatch, child)
    ctx.pending_events, ctx.event_queue = [], None
    revoke(attention, monkeypatch, change)
    with invocation_scope({"tool_call_id": "returned-model-response"}):
        result = registry.execute_result("pause_task", {"task_id": "root", "reason": "inspection decision"})
    assert result.status == "blocked", result.text
    assert ctx.pending_events == []


@pytest.mark.parametrize("action", ["pause_task", "resume_task", "continue_task"])
@pytest.mark.parametrize("change", ["progress", "memory_progress", "attempt", "memory_attempt", "memory_pause", "stop", "panic", "missing", "unreadable"])
def test_deferred_effect_rechecks_current_source(attention, monkeypatch, action, change):
    from ouroboros.tools.control_mind import _request_control
    from ouroboros.tool_call_log import invocation_scope
    from supervisor.events_mind_controls import _apply
    child = _mint(attention)
    registry, ctx = _agent(attention, monkeypatch, child)
    ctx.pending_events, ctx.event_queue = [], None
    effects = []
    for path in ("supervisor.owner_pause_control.request_owner_pause",
                 "supervisor.queue_transitions.resume_budget_paused_task",
                 "supervisor.continuation_admission.admit_continuation"):
        monkeypatch.setattr(path, lambda *a, **kw: effects.append(a) or {"ok": True})
    with invocation_scope({"tool_call_id": "deferred-control"}):
        assert json.loads(_request_control(ctx, action, "root", "inspection decision"))["ok"]
    event = ctx.pending_events.pop()
    revoke(attention, monkeypatch, change)
    result = _apply(event)
    assert not result["ok"], result
    assert effects == []


def test_ordinary_mind_control_reads_only_its_host_binding(attention, monkeypatch):
    from supervisor.events_mind_controls import _apply
    reads = []
    def read(root, task_id, **kwargs):
        reads.append(task_id)
        assert task_id == "ordinary-mind"
        return load_task_result(root, task_id, **kwargs)
    write_task_result(attention.root, "ordinary-mind", "running")
    monkeypatch.setattr("ouroboros.task_results.load_task_result", read)
    monkeypatch.setattr("supervisor.owner_pause_control.request_owner_pause", lambda *a, **kw: {"ok": True})
    assert _apply({"action": "pause_task", "task_id": "root", "actor_task_id": "ordinary-mind",
                   "action_nonce": "ordinary-control"})["ok"]
    assert reads == ["ordinary-mind"]


def test_host_diagnostic_binding_survives_missing_context_and_unprefixed_actor(attention):
    from ouroboros.tool_call_log import invocation_scope
    from ouroboros.tools.control_mind import _request_control
    from supervisor.events_mind_controls import _apply

    child = _mint(attention)
    # Neither the local context nor an event's spelling classifies the actor.
    actor = "unprefixed-actor"
    write_task_result(attention.root, actor, "running", metadata=child["metadata"])
    ctx = SimpleNamespace(task_id=actor, drive_root=attention.root, task_metadata={},
                          pending_events=[], event_queue=None)
    with invocation_scope({"tool_call_id": "host-binding"}):
        result = json.loads(_request_control(ctx, "pause_task", "unrelated", "inspection"))
    assert result["error"] == "diagnostic_control_target" and not ctx.pending_events
    assert _apply({"action": "pause_task", "task_id": "unrelated", "actor_task_id": actor,
                   "action_nonce": "host-binding"})["error"] == "diagnostic_control_target"


@pytest.mark.parametrize("change", ["quiet", "progress", "episode", "attempt"])
@pytest.mark.parametrize("boundary", ["send", "launch"])
def test_unpublished_source_progress_reaches_unsent_boundaries(attention, monkeypatch, change, boundary):
    from ouroboros.model_wait import TaskModelWait, dispatch_control_reason
    from ouroboros.task_status import _load_queue_snapshot, queue_snapshot_observation
    from supervisor import worker_assignment, workers
    from tests.worker_pool_support import affirm_ready

    child = _mint(attention)
    if change == "progress":
        attention.meta["last_progress_at"] = attention.clock[0]
    elif change == "episode":
        attention.meta["silence_attention"]["state"] = "progress_observed"
    elif change == "attempt":
        attention.meta["attempt"] += 1
    # The recovery lane has not published its result; the supervisor has published
    # its real projection. A raw RUNNING dict masquerading as a snapshot misses this bug.
    assert load_task_result(attention.root, "root")["status"] == "running"
    assert observation(attention)["state"] == "pending_attention"
    assert queue.persist_queue_snapshot(reason="source_currentness_probe")
    snapshot = _load_queue_snapshot(attention.root)
    assert queue_snapshot_observation(snapshot)["fresh"]
    owner = TaskModelWait(task=child, drive_root=attention.root, event_queue=None, worker_slot_held=True)
    assert dispatch_control_reason(owner, handed_off=True) is None  # sent work still settles
    if boundary == "send":
        assert dispatch_control_reason(owner) == (None if change == "quiet" else "cancelled")
    else:
        sent = []
        monkeypatch.setattr(workers, "REPO_DIR", attention.root)
        worker = SimpleNamespace(wid=1, proc=SimpleNamespace(pid=77, is_alive=lambda: True),
            busy_task_id=None, reaping=False, active_capacity=True, in_q=SimpleNamespace(put=sent.append))
        affirm_ready(worker)
        workers.WORKERS[1] = worker
        with queue._queue_lock:
            assert worker_assignment._claim_worker_launch(queue, child, worker) is None
        attention.flush()
        assert [task["id"] for task in sent] == ([child["id"]] if change == "quiet" else [])
        worker_assignment._forget_departed_launches(queue.PENDING)
    assert "root" in attention.running and queue._reap_queue.empty()


@pytest.mark.parametrize("reason", ["task_exception", "provider_failure", "llm_api_error", "worker_crash_owner_wait"])
def test_terminal_source_cannot_use_old_episode_for_continue(attention, monkeypatch, reason):
    from ouroboros.owner_continue import continuation_eligibility
    from ouroboros.work_diagnostics import control_reason
    from ouroboros.utils import utc_now_iso
    from supervisor.events_mind_controls import _apply
    child = _mint(attention)
    write_task_result(attention.root, "root", "failed", reason_code=reason)
    assert continuation_eligibility(load_task_result(attention.root, "root"), "root")["eligible"]
    attention.running.pop("root")
    monkeypatch.setattr("ouroboros.task_status._load_queue_snapshot", lambda *a: {"generation": "fixture", "_queue_alive": {"generation": "fixture", "ts": utc_now_iso()}, "running": [], "pending": []})
    monkeypatch.setattr("supervisor.continuation_admission.admit_continuation", attention.forbidden)
    assert control_reason(child, attention.root, before_handoff=True)
    result = _apply({"action": "continue_task", "task_id": "root", "actor_task_id": child["id"],
                     "action_nonce": "terminal-root"})
    assert not result["ok"]


@pytest.mark.parametrize("witness", ["live", "missing", "stale"])
def test_answer_bytes_do_not_assert_source_writers_exited(attention, monkeypatch, witness):
    from ouroboros.work_diagnostics import control_reason
    from ouroboros.task_status import _load_queue_snapshot
    child = _mint(attention)
    write_task_result(attention.root, "root", "completed", result="answer bytes")
    assert queue.persist_queue_snapshot(reason="source_answer_bytes")
    snapshot = _load_queue_snapshot(attention.root)
    if witness == "missing":
        snapshot = {"_snapshot_missing": True}
    elif witness == "stale":
        snapshot["_queue_alive"]["ts"] = "2000-01-01T00:00:00Z"
    monkeypatch.setattr("ouroboros.task_status._load_queue_snapshot", lambda *a: snapshot)
    reason = control_reason(child, attention.root, before_handoff=True)
    assert (reason is None) is (witness == "live")
    assert attention.running["root"] is attention.meta and queue._reap_queue.empty()


@pytest.mark.parametrize("mode", ["warm", "cold", "slot"])
@pytest.mark.parametrize("selected", ["root", "middle"])
def test_late_waiter_receives_one_attention_without_readmission(attention, monkeypatch, mode, selected):
    from ouroboros import model_sleep
    from ouroboros.owner_mailbox import drain_owner_entries
    from tests.test_model_sleep import _ctx
    from supervisor import silence_attention
    task = attention.meta["task"]
    task["parent_task_id"] = "parent" if selected == "root" else "middle"
    if selected == "middle":
        attention.running["middle"] = {"task": {"id": "middle", "parent_task_id": "parent"},
                                       "attempt": 1, "started_at": 4999, "last_progress_at": 4999}
        write_task_result(attention.root, "middle", "running")
    parent = {"task": {"id": "parent"}, "attempt": 1, "started_at": 4999, "last_progress_at": 4999}
    attention.running["parent"] = parent
    write_task_result(attention.root, "parent", "running")
    diagnostic = _mint(attention)
    ctx = _ctx(attention.root, "parent", root="parent")
    chosen = model_sleep.selectors(ctx, tasks=[selected])
    checkpoint = {"reason": "sleep", "state": "waiting", "task_attempt": 1, "sleep": chosen}
    if mode == "cold":
        attention.running.pop("parent")
        queue.PENDING.append({"id": "parent", "_budget_pause": checkpoint})
    elif mode == "warm":
        parent["owner_wait"] = checkpoint
    else:
        parent["active_operation_leases"] = {"wait": {"kind": "in_slot_wait", "until_ts": 9000,
                                                     "wait_for_tasks": [selected]}}
    attention.tick()
    attention.flush()
    assert model_sleep.wake_reason(ctx, chosen) == "work_attention:root"
    mail = drain_owner_entries(attention.root, "parent")
    assert len(mail) == 1 and mail[0]["work_attention"]["waiting_for_task_id"] == selected
    assert sum(t["id"] == diagnostic["id"] for t in queue.PENDING) == 1
    monkeypatch.setattr(silence_attention, "_publish", attention.forbidden)
    for _ in range(3):
        attention.tick()
        attention.flush()
    # Draining exposes mail; until acknowledged the existing row stays readable.
    assert drain_owner_entries(attention.root, "parent") == mail


def test_diagnostic_external_schema_does_not_discover(attention, monkeypatch):
    registry, _ctx = _agent(attention, monkeypatch, _mint(attention))
    discoveries = []
    monkeypatch.setattr("ouroboros.mcp_client.ensure_configured_from_settings", lambda: discoveries.append("mcp"))
    monkeypatch.setattr("ouroboros.extension_loader.parse_extension_surface_name", lambda n: discoveries.append("extension"))
    assert registry.get_schema_by_name("mcp__not_authorized__tool") is None
    assert not discoveries


def test_uncertain_append_preserves_snapshot_duty(attention, monkeypatch):
    original = queue.enqueue_task
    snapshots = []
    def enqueue(task):
        result = original(task)
        return {**result, "_admission_blocked": "contention", "_admission_never_admitted": True}
    monkeypatch.setattr(queue, "enqueue_task", enqueue)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **kw: snapshots.append(kw) or True)
    _mint(attention)
    assert {"reason": "schedule_subagent_admitted"} in snapshots
    assert len(queue.PENDING) == 1


def test_presence_diagnostic_inherits_restrictions_without_speaker(attention, monkeypatch):
    from ouroboros.contracts.task_contract import attach_task_contract
    from ouroboros.presence_authority import presence_ceiling_payload
    from tests.frozen_presence_authority import build_presence_capability_ceiling
    from tests.frozen_presence_capabilities import PresenceToolTarget, PresenceResourceTarget
    from tests.test_presence_authority import _resolution
    source = attention.meta["task"]
    source["metadata"]["presence"] = {"binding_id": "own-room", "reply_route": {"private": "speaker-only"}}
    ceiling = presence_ceiling_payload(build_presence_capability_ceiling(
        skill_name="moderator", skill_content_hash="c" * 64, state_fingerprint="d" * 64,
        resolution=_resolution(PresenceToolTarget("builtin", "read_file"),
                               PresenceResourceTarget("active_workspace", ("read",), "shared"))))
    source["task_contract"] = {"capability_ceiling": ceiling, "disabled_tools": ["list_files"],
                               "allowed_resources": {"network": False}, "input_sources": "declared",
                               "context": "Selected source only"}
    attach_task_contract(source)
    child = _mint(attention)
    assert child["task_contract"]["capability_ceiling"] == ceiling
    assert child["task_contract"]["input_sources"] == "declared"
    assert child["task_contract"]["allowed_resources"] == {"network": False}
    assert child["metadata"]["presence_binding_authority"] == {"binding_id": "own-room"}
    assert "presence" not in child["metadata"]
    registry, ctx = _agent(attention, monkeypatch, child)
    repo = ctx.active_repo_dir()
    (repo / "shared").mkdir(exist_ok=True)
    (repo / "shared" / "allowed.txt").write_text("permitted source")
    (repo / "private.txt").write_text("PRIVATE SENTINEL")
    allowed = registry.execute_result("read_file", {"root": "active_workspace", "path": "shared/allowed.txt"})
    assert allowed.status == "ok" and "permitted source" in allowed.text, allowed.text
    denied = registry.execute_result("read_file", {"root": "active_workspace", "path": "private.txt"})
    assert denied.status == "blocked" and "PRIVATE SENTINEL" not in denied.text, denied.text
    for name in ("pause_task", "list_files", "write_file", "schedule_subagent"):
        assert registry.get_schema_by_name(name) is None
        assert registry.execute_result(name, {}).status == "blocked"
    for tid, binding in (("same-room", "own-room"), ("foreign-room", "elsewhere")):
        write_task_result(attention.root, tid, "completed", root_task_id=tid, delegation_role="root",
                          metadata={"presence_binding_authority": {"binding_id": binding}}, result=tid + " result")
    own = registry.execute_result("get_task_result", {"task_id": "same-room"})
    foreign = registry.execute_result("get_task_result", {"task_id": "foreign-room"})
    assert "same-room result" in own.text, own.text
    assert "foreign-room result" not in foreign.text, foreign.text


@pytest.mark.parametrize("mode", ["warm", "cold", "slot"])
def test_real_late_wait_consumers_wake(attention, monkeypatch, mode):
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.tools import control_task_results
    from ouroboros import budget_pause
    from ouroboros.owner_wait import checkpoint_owner_wait, direct_owner_wait, _sleep_wake
    from tests.test_model_sleep import _ctx, _cold_park
    from supervisor import state, workers
    from supervisor.sleep_wake import wake_ready_sleepers
    from supervisor.worker_owner_wait import handle_owner_wait

    attention.meta["task"]["parent_task_id"] = "parent"
    write_task_result(attention.root, "parent", "running")
    parent = {"task": {"id": "parent", "root_task_id": "parent"}, "worker_id": 1,
              "attempt": 1, "started_at": 4900.0, "last_progress_at": 4999.0}
    attention.running["parent"] = parent
    child = _mint(attention)
    ctx = _ctx(attention.root, "parent", root="parent")
    ctx.event_queue = attention.bus
    if mode == "cold":
        monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 5.0)
        monkeypatch.setattr(budget_pause, "observe_task_runs", lambda *a, **kw: {"custody_read": "ok", "runs": []})
        _cold_park(attention.root, monkeypatch, workers, task_id="parent", tasks=["root"])
        parked = next(t for t in queue.PENDING if t["id"] == "parent")
        assert wake_ready_sleepers(queue) == []
        attention.tick()
        attention.flush()
        outcome = wake_ready_sleepers(queue)
        assert outcome[0]["ok"] and outcome[0]["ready"]["reason"] == "work_attention:root"
        assert parked["_budget_pause_resume"]["selected_by"] == "sleep_wake"
    elif mode == "warm":
        assert '"sleep_armed"' in control_task_results._await_messages(ctx, mode="warm", tasks=["root"])
        checkpoint = checkpoint_owner_wait(ctx, [{"role": "user", "content": "await helper"}], {}, {}, 1, [], set())
        workers.WORKERS[1] = SimpleNamespace(wid=1, proc=SimpleNamespace(pid=77, is_alive=lambda: True),
            busy_task_id="parent", reaping=False, active_capacity=True, in_q=SimpleNamespace(put=lambda x: None))
        handle_owner_wait({"task_id": "parent", "wait_id": checkpoint["wait_id"], "task_attempt": 1,
            "worker_id": 1, "pid": 77, "phase": "park", "checkpoint": checkpoint},
            SimpleNamespace(RUNNING=attention.running, WORKERS=workers.WORKERS, DRIVE_ROOT=attention.root))
        assert _sleep_wake(ctx, checkpoint) == ""
        attention.tick()
        attention.flush()
        assert direct_owner_wait(ctx, checkpoint) == "work_attention:root"
    else:
        def wait(root, ids, **kwargs):
            attention.drain()
            assert kwargs["on_poll"]({}, {}) is None
            attention.tick()
            attention.flush()
            early = kwargs["on_poll"]({}, {})
            assert early["reason"] == "owner_mailbox_pending"
            return {"early_return": early, "tasks": {}, "elapsed_sec": 0}
        monkeypatch.setattr(control_task_results, "wait_for_effective_tasks", wait)
        assert "interrupted by an unread message" in control_task_results._wait_for_task(ctx, "root", 30)
    assert len(drain_owner_entries(attention.root, "parent")) == 1
    assert sum(t["id"].startswith("silence-") for t in queue.PENDING) == 1
    assert child in queue.PENDING


@pytest.mark.parametrize("unavailable", ["missing", "owner", "other_task", "held"])
def test_no_republication_for_unavailable_late_waiter(attention, monkeypatch, unavailable):
    from supervisor import silence_attention
    attention.meta["task"]["parent_task_id"] = "parent"
    _mint(attention)
    if unavailable != "missing":
        parent = {"task": {"id": "parent"}, "attempt": 1, "started_at": 4999, "last_progress_at": 4999,
                  "active_operation_leases": {"wait": {"kind": "in_slot_wait", "until_ts": 9000,
                    "wait_for_tasks": ["other" if unavailable == "other_task" else "root"]}}}
        if unavailable == "owner":
            parent["owner_wait"] = {"state": "waiting", "task_attempt": 1, "reason": "owner"}
        elif unavailable == "held":
            parent["task"]["_budget_pause_hold"] = {"reason": "owner"}
        attention.running["parent"] = parent
    monkeypatch.setattr(silence_attention, "_publish", attention.forbidden)
    attention.tick()
    attention.flush()


def test_definite_refusal_is_recorded_without_retrying_episode(attention, monkeypatch):
    from ouroboros.config import BODY_MAINTENANCE_INTERVAL_SEC
    from supervisor import silence_attention
    puts = []
    def refuse(task):
        puts.append(task["id"])
        return {**task, "_admission_blocked": "fixture_project_refused", "_admission_never_admitted": True}
    monkeypatch.setattr(queue, "enqueue_task", refuse)
    for _ in range(3):
        attention.tick()
        attention.flush()
        attention.clock[0] += BODY_MAINTENANCE_INTERVAL_SEC
    assert len(puts) == 1 and queue.PENDING == []
    fact = observation(attention)
    assert fact["attention_wait"] == "diagnostic_recorded" and fact["diagnostic_status"] == "failed"
    monkeypatch.setattr(silence_attention, "_publish", attention.forbidden)
    attention.tick()
    attention.flush()


@pytest.mark.parametrize("carrier", ["presence", "presence_binding_authority"])
def test_binding_only_source_preserves_existing_scoped_read_authority(attention, monkeypatch, carrier):
    source = attention.meta["task"]
    source["metadata"][carrier] = {"binding_id": "one-binding"}
    child = _mint(attention)
    assert child["metadata"]["presence_binding_authority"] == {"binding_id": "one-binding"}
    assert "presence" not in child["metadata"]
    registry, _ctx = _agent(attention, monkeypatch, child)
    write_task_result(attention.root, "foreign", "completed", root_task_id="foreign", delegation_role="root",
        metadata={"presence_binding_authority": {"binding_id": "other-binding"}}, result="FOREIGN SECRET")
    source_registry, _source_ctx = _agent(attention, monkeypatch, source)
    # Without a retained ceiling the existing ordinary reader is unscoped for
    # both source and child; carrying binding authority does not mint a ceiling.
    for reader in (source_registry, registry):
        assert "FOREIGN SECRET" in reader.execute_result("get_task_result", {"task_id": "foreign"}).text
        result = reader.execute_result("get_task_result", {"task_id": "foreign", "presence_scope": "own_binding"})
        assert result.status == "blocked" and "FOREIGN SECRET" not in result.text


@pytest.mark.parametrize("barrier", ["parent", "pause", "billing", "store"])
def test_unchanged_attention_avoids_tick_reads_and_recovers(attention, monkeypatch, barrier):
    from ouroboros.config import BODY_MAINTENANCE_INTERVAL_SEC
    from ouroboros.store import StoreError
    from supervisor import silence_attention

    available = [False]
    if barrier == "parent":
        attention.meta["task"]["parent_task_id"] = "parent"
        attention.running["parent"] = {"task": {"id": "parent"}, "attempt": 1,
            "started_at": 4999, "last_progress_at": 4999, "owner_wait": {
                "state": "waiting", "task_attempt": 1, "reason": "sleep", "sleep": {"tasks": ["root"]}}}
        write_task_result(attention.root, "parent", "running")
    elif barrier == "pause":
        write_task_result(attention.root, "root", "running", owner_pause={"state": "requested", "fence_id": "pause"})
    elif barrier == "billing":
        from ouroboros.usage_admission import task_billing_fields
        monkeypatch.setattr("ouroboros.usage_admission.task_billing_fields",
            lambda *a, **kw: task_billing_fields(*a, **kw) if available[0] else {})
    publishes, reads = [], []
    publish = silence_attention._publish
    def counted_publish(*args):
        publishes.append(attention.clock[0])
        return publish(*args)
    def read(root, task_id, **kwargs):
        reads.append(task_id)
        if barrier == "store" and task_id == "root" and not available[0]:
            raise StoreError("temporary source unavailability")
        return load_task_result(root, task_id, **kwargs)
    monkeypatch.setattr(silence_attention, "_publish", counted_publish)
    monkeypatch.setattr("ouroboros.task_results.load_task_result", read)
    attention.tick()
    attention.flush()
    initial_reads = len(reads)
    for _ in range(9):
        attention.clock[0] += 0.5
        attention.tick()
        attention.flush()
    print("ATTENTION_REPEATS", barrier, "publishes", len(publishes), "source_reads", len(reads))
    assert len(publishes) == 1 and len(reads) == initial_reads
    assert queue.PENDING == []
    if barrier == "parent":
        attention.running.pop("parent")  # losing the responsible parent is noticed without a timer
    elif barrier == "pause":
        write_task_result(attention.root, "root", "running", owner_pause=None)
    available[0] = True
    if barrier != "parent":
        attention.clock[0] += BODY_MAINTENANCE_INTERVAL_SEC
    attention.tick()
    attention.flush()
    assert len(publishes) == 2 and len(queue.PENDING) == 1
    assert observation(attention)["state"] == "pending_attention"


def _finish_source(attention, monkeypatch):
    """The recovered source's own loop answers once; only a held finish buys round two."""
    import queue as stdqueue
    import ouroboros.loop as loop_mod
    from ouroboros.tools.registry import ToolRegistry
    from tests.test_completion_selection import finish

    calls, progress = [], []

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def model(_llm, request_messages, *_args, **_kwargs):
        calls.append([dict(m) for m in request_messages])
        if len(calls) == 1:
            return finish("Source answer."), 0.0
        return finish("Source answer; its helper is undecided.", action="stop",
                      rationale="An ordinary child still needs an author disposition."), 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", model)
    result, usage, trace = loop_mod.run_llm_loop(
        messages=[{"role": "user", "content": "work"}],
        tools=ToolRegistry(repo_dir=attention.root, drive_root=attention.root), llm=FakeLLM(),
        drive_logs=attention.root, emit_progress=lambda text, **_meta: progress.append(text),
        incoming_messages=stdqueue.Queue(), task_id="root", drive_root=attention.root)
    return result, usage, trace, calls, progress


@pytest.mark.parametrize("kind", ["diagnostic", "ordinary"])
@pytest.mark.parametrize("state", ["pending", "queued_only", "running", "completed", "failed"])
def test_recovered_source_finish_owes_its_diagnostic_no_disposition(attention, monkeypatch, state, kind):
    """The source's real handoff and absorption gates: a host diagnostic stays a listed,
    accounted child but needs no author disposition; the same lineage without the host
    binding is an ordinary child that still holds the finish."""
    from ouroboros.loop_forced_finalization import _load_direct_child_results
    from ouroboros.task_store import delete_document

    child = _mint(attention)
    if kind == "ordinary":
        child["metadata"].pop("work_diagnostic")
        write_task_result(attention.root, child["id"], "scheduled", metadata=child["metadata"])
    if state == "queued_only":  # only the queue overlay knows the row
        assert delete_document(attention.root, child["id"])
    elif state == "running":
        queue.PENDING.remove(child)
        attention.running[child["id"]] = {"task": child, "attempt": 1, "worker_id": 1,
                                          "started_at": attention.clock[0], "last_progress_at": attention.clock[0]}
        write_task_result(attention.root, child["id"], "running")
    elif state in {"completed", "failed"}:  # a later writer need not repeat the binding
        queue.PENDING.remove(child)
        write_task_result(attention.root, child["id"], state, result="Inspection: the source waits on a held lock.")
    assert queue.persist_queue_snapshot(reason="source_finish_probe")
    assert [row["task_id"] for row in _load_direct_child_results(attention.root, "root", "root")] == [child["id"]]
    result, usage, trace, calls, progress = _finish_source(attention, monkeypatch)
    held = json.dumps(calls[1:]) + json.dumps(progress)
    print("SOURCE_FINISH", json.dumps({"state": state, "kind": kind, "model_calls": len(calls),
          "completion": (trace.get("task_completion") or {}).get("action"), "result": result,
          "notice": usage.get("terminal_host_notice", "")}, default=str))
    if kind == "diagnostic":
        # A plain finish records no explicit stop selection.
        assert len(calls) == 1 and result == "Source answer." and "task_completion" not in trace
        assert "CHILD_ABSORPTION_REQUIRED" not in held and "handoff" not in held
        assert child["id"] not in str(usage.get("terminal_host_notice") or "")
    else:
        assert len(calls) == 2 and trace["task_completion"]["action"] == "stop"
        assert "Subagent handoff status refreshed" in held
    assert "root" in attention.running and queue._reap_queue.empty()


def _review_while_child_settles(attention, monkeypatch, child, outcome):
    """A scripted result review (no provider) that passes the answer while ``child``
    settles as its owners settle it: its own answer, or the single-scope cancellation its
    intent requested (the host withdraws a diagnostic whose source progressed)."""
    from ouroboros.cancel_intents import active_intent, request_cancel
    from ouroboros.work_diagnostics import diagnostic_ref, withdraw_revoked
    from supervisor.cancel_publication import _intent_outcome_fields

    reviewed = []

    def review(_ctx, content, _trace, _hold):
        reviewed.append(content)
        if len(reviewed) == 1:
            fields = {"result": "Inspection: the source waits on a held lock."}
            if outcome == "cancelled":
                if diagnostic_ref(child):
                    withdraw_revoked(child, attention.root)
                else:
                    request_cancel(attention.root, child["id"], reason="fixture owner Stop of the helper",
                                   source="owner", requested_by="owner", scope="single")
                fields = {"result": "Task cancelled.",
                          **_intent_outcome_fields(active_intent(attention.root, child["id"]))}
            attention.running.pop(child["id"])
            write_task_result(attention.root, child["id"], outcome, **fields)
            assert queue.persist_queue_snapshot(reason="child_settled_during_review")
        return False  # the review passes; nothing is held

    monkeypatch.setattr("ouroboros.tools.review_result.result_review_at_delivery", review)
    return reviewed


@pytest.mark.parametrize("kind", ["diagnostic", "adopted", "ordinary"])
@pytest.mark.parametrize("outcome", ["completed", "cancelled"])
def test_child_settling_under_result_review_rearms_only_claimed_evidence(attention, monkeypatch, outcome, kind):
    """The source answers while its child runs, and the child settles while that answer
    is under result review. An unclaimed host diagnostic is not evidence of the answer:
    one model round delivers it, and the diagnostic stays listed. A diagnostic the author
    adopted by exact hash, or an ordinary child, still invalidates the reviewed answer."""
    from ouroboros.loop_forced_finalization import _load_direct_child_results
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256
    from ouroboros.tools.task_tree import _tree_note

    child = _mint(attention)
    if kind == "ordinary":
        child["metadata"].pop("work_diagnostic")
        write_task_result(attention.root, child["id"], "scheduled", metadata=child["metadata"])
    queue.PENDING.remove(child)
    attention.running[child["id"]] = {"task": child, "attempt": 1, "worker_id": 1,
                                      "started_at": attention.clock[0], "last_progress_at": attention.clock[0]}
    write_task_result(attention.root, child["id"], "running")
    assert queue.persist_queue_snapshot(reason="source_answer_probe")
    if kind != "diagnostic":  # the author's exact-hash claim lets its answer past the gates
        author = SimpleNamespace(drive_root=str(attention.root), budget_drive_root=str(attention.root),
                                 task_metadata={"budget_drive_root": str(attention.root), "root_task_id": "root"},
                                 task_id="root", role="orchestrator")
        digest = _child_result_sha256(load_effective_task_result(attention.root, child["id"]))
        assert _tree_note(author, "decision", "adopted the helper's current result", payload={
            "type": "child_result_disposition", "child_task_id": child["id"],
            "disposition": "integrated", "child_result_sha256": digest}).startswith("OK:")
    reviewed = _review_while_child_settles(attention, monkeypatch, child, outcome)
    result, usage, trace, calls, progress = _finish_source(attention, monkeypatch)
    rearmed = ("Delivery evidence changed after the answer was retained; a complete replacement answer "
               "is required.") in trace["reasoning_notes"]
    settled = load_task_result(attention.root, child["id"]) or {}
    print("SETTLED_UNDER_REVIEW", json.dumps({"kind": kind, "outcome": outcome, "model_calls": len(calls),
          "rearmed": rearmed, "result": result, "child_status": settled.get("status"),
          "parent_decision": settled.get("parent_decision"),
          "cancel_source": (settled.get("cancel_origin") or {}).get("source")}, default=str))
    assert reviewed == ["Source answer."] and settled["status"] == outcome
    assert [row["task_id"] for row in _load_direct_child_results(attention.root, "root", "root")] == [child["id"]]
    if kind == "diagnostic":
        assert len(calls) == 1 and result == "Source answer." and not rearmed
        assert "task_completion" not in trace
    else:
        assert len(calls) == 2 and rearmed and trace["task_completion"]["action"] == "stop"
        assert "[DELIVERY_FINALIZATION_CONTROL]" in json.dumps(calls[1])
    assert "root" in attention.running and queue._reap_queue.empty()


def test_ordinary_progress_keeps_the_certificate_and_source_progress_revokes(attention, monkeypatch):
    """Only an open episode projects source progress into the certified queue; that
    projection still revokes a diagnostic's control, launch and send before the source
    result is written."""
    from ouroboros.model_wait import TaskModelWait, dispatch_control_reason
    from ouroboros.task_status import _load_queue_snapshot
    from ouroboros.tool_call_log import invocation_scope
    from supervisor import worker_assignment, workers
    from tests.worker_pool_support import affirm_ready

    busy = {"task": {"id": "busy", "root_task_id": "busy"}, "attempt": 1, "worker_id": 2,
            "started_at": 4900.0, "last_progress_at": 4950.0}
    attention.running["busy"] = busy
    child = _mint(attention)
    registry, ctx = _agent(attention, monkeypatch, child)
    ctx.pending_events, ctx.event_queue = [], None

    def running(task_id):
        return next(row for row in _load_queue_snapshot(attention.root)["running"] if row["id"] == task_id)

    assert queue.persist_queue_snapshot(reason="certificate_probe")
    generation = queue.QUEUE_PUBLICATION[2]
    assert not {"last_progress_at", "silence_attention"} & set(running("busy"))
    assert running("root")["silence_attention"]["state"] == "pending_attention"
    for stamp in (4960.0, 4970.0):
        busy["last_progress_at"] = stamp
        assert queue.persist_queue_snapshot(reason="certificate_probe")
        assert queue.QUEUE_PUBLICATION[2] == generation
    attention.meta["last_progress_at"] = attention.clock[0]  # not yet in the durable source row
    assert queue.persist_queue_snapshot(reason="certificate_probe")
    assert queue.QUEUE_PUBLICATION[2] != generation
    assert running("root")["last_progress_at"] == attention.clock[0]
    assert observation(attention)["state"] == "pending_attention"
    with invocation_scope({"tool_call_id": "after-source-progress"}):
        refused = registry.execute_result("pause_task", {"task_id": "root", "reason": "inspection decision"})
    assert refused.status == "blocked" and ctx.pending_events == []
    sent = []
    monkeypatch.setattr(workers, "REPO_DIR", attention.root)
    worker = SimpleNamespace(wid=1, proc=SimpleNamespace(pid=77, is_alive=lambda: True), busy_task_id=None,
                             reaping=False, active_capacity=True, in_q=SimpleNamespace(put=sent.append))
    affirm_ready(worker)
    workers.WORKERS[1] = worker
    with queue._queue_lock:
        assert worker_assignment._claim_worker_launch(queue, child, worker) is None
    attention.flush()
    assert sent == []
    worker_assignment._forget_departed_launches(queue.PENDING)
    owner = TaskModelWait(task=child, drive_root=attention.root, event_queue=None, worker_slot_held=True)
    assert dispatch_control_reason(owner) == "cancelled"
    assert "root" in attention.running and queue._reap_queue.empty()


def _helper_source(attention):
    """The silent work is a helper of a live root with a live sibling."""
    helper = attention.meta["task"]
    helper.update(parent_task_id="parent", root_task_id="parent", delegation_role="subagent")
    billing = load_task_result(attention.root, "root")["billing_group"]
    for task_id, fields in (("parent", {}), ("sibling", {"parent_task_id": "parent", "delegation_role": "subagent"}),
                            ("root", {"parent_task_id": "parent", "delegation_role": "subagent"})):
        write_task_result(attention.root, task_id, "running", root_task_id="parent", task_attempt=1,
                          billing_group={**billing, "billing_group_id": "parent"}, **fields)
    for task_id in ("parent", "sibling"):
        attention.running[task_id] = {"task": {"id": task_id, "root_task_id": "parent",
                                               **({"parent_task_id": "parent", "delegation_role": "subagent"}
                                                  if task_id == "sibling" else {})},
                                      "attempt": 1, "worker_id": 2, "started_at": 4999.0, "last_progress_at": 4999.0}
    return _mint(attention)


@pytest.mark.parametrize("action", ["pause_task", "resume_task", "continue_task"])
@pytest.mark.parametrize("target", ["parent", "sibling"])
def test_child_source_diagnostic_cannot_control_root_or_sibling(attention, monkeypatch, action, target):
    from ouroboros.tool_call_log import invocation_scope
    from supervisor.events_mind_controls import _apply

    child = _helper_source(attention)
    assert child["metadata"]["work_diagnostic"]["root_task_id"] == "parent"
    registry, ctx = _agent(attention, monkeypatch, child)
    ctx.pending_events, ctx.event_queue = [], None
    for path in ("supervisor.owner_pause_control.request_owner_pause",
                 "supervisor.queue_transitions.resume_budget_paused_task",
                 "supervisor.continuation_admission.admit_continuation"):
        monkeypatch.setattr(path, attention.forbidden)
    with invocation_scope({"tool_call_id": "widen-target"}):
        requested = registry.execute_result(action, {"task_id": target, "reason": "inspection decision"})
    assert requested.status == "blocked" and "diagnostic_control_target" in requested.text
    assert ctx.pending_events == []
    assert _apply({"action": action, "task_id": target, "actor_task_id": child["id"],
                   "action_nonce": "widen-target"}) == {"ok": False, "error": "diagnostic_control_target"}


def test_child_source_pause_reaches_its_owner_and_is_refused_honestly(attention, monkeypatch):
    """The request passes the diagnostic gate; the ordinary Pause owner refuses a
    non-root source and changes nothing, rather than widening to its root."""
    from ouroboros.owner_pause import read_fence
    from ouroboros.tool_call_log import invocation_scope
    from supervisor.events_mind_controls import _apply

    child = _helper_source(attention)
    registry, ctx = _agent(attention, monkeypatch, child)
    ctx.pending_events, ctx.event_queue = [], None
    with invocation_scope({"tool_call_id": "pause-helper-source"}):
        requested = registry.execute_result("pause_task", {"task_id": "root", "reason": "inspection decision"})
    assert requested.status == "ok", requested.text
    outcome = _apply(ctx.pending_events.pop())
    assert outcome["ok"] is False and outcome["error"] == "not_a_root_task" and outcome["root_task_id"] == "parent"
    assert not read_fence(attention.root, "parent") and not read_fence(attention.root, "root")
    assert {"root", "parent", "sibling"} <= set(attention.running) and queue._reap_queue.empty()
