"""Unknown live silence is a fact for a mind, never a timeout verdict."""

from types import SimpleNamespace

import pytest

from tests.test_idle_observation import host as _host, _controls
from ouroboros.owner_mailbox import drain_owner_entries
from ouroboros.task_results import load_task_result, write_task_result
from supervisor import queue, workers

host = _host


@pytest.fixture
def attention(host, monkeypatch):
    from supervisor import queue_snapshot, silence_attention, workspace_preparation as preparation

    jobs = []
    monkeypatch.setattr(preparation, "_process_owner", lambda: {"pid": 11, "birth": 1})
    generation = preparation.PreparationGeneration()
    monkeypatch.setattr(generation, "submit", lambda job, lane: jobs.append(job) or True)
    monkeypatch.setattr(generation, "current_lane", lambda: preparation.ADMISSION_LANE)
    monkeypatch.setattr("supervisor.workspace_preparation.current_generation", lambda: generation)
    monkeypatch.setattr(silence_attention, "time", SimpleNamespace(time=lambda: host.clock[0]))
    for name in ("BUDGET_ROOT_FENCES", "ACCEPTANCE_FENCES", "ADMISSION_RESERVATIONS", "RETURNS_NAMED"):
        monkeypatch.setattr(queue, name, {})
    monkeypatch.setattr(workers, "_worker_pool_execution_state", lambda: {"available": True, "disabled_reason": ""})
    monkeypatch.setattr(queue, "QUEUE_SEQ_COUNTER_REF", {"value": 0})
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", host.root / "state" / "queue_snapshot.json")
    monkeypatch.setattr(queue, "QUEUE_PUBLICATION", None)
    monkeypatch.setattr(queue, "persist_queue_snapshot", queue_snapshot.persist_queue_snapshot)
    monkeypatch.setattr(workers, "PENDING", queue.PENDING)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(host.root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(host.root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    monkeypatch.setattr("ouroboros.platform_layer.process_start_time", lambda pid: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("No process death or real worker launch is permitted in this test")

    monkeypatch.setattr("supervisor.worker_pool_lifecycle.kill_worker_tree", forbidden)
    monkeypatch.setattr("supervisor.task_reaper._kill_and_confirm_worker_dead", forbidden)
    host.forbidden = forbidden
    write_task_result(host.root, "root", "running", root_task_id="root", task_attempt=1,
                      billing_group={"billing_group_id": "root", "billing_group_limit_usd": 10,
                        "billing_group_limit_source": "initial_task_admission", "billing_group_limit_revision": "original"})
    host.meta["task"]["metadata"] = {"work_kind": "owner"}

    def flush():
        while jobs:
            jobs.pop(0)()

    host.flush = flush
    host.generation = generation
    return host


def observation(host):
    return (load_task_result(host.root, "root") or {}).get("silence_attention", {})


def test_heartbeat_only_never_finalizes_or_reaps(host):
    host.bus.put({"type": "task_heartbeat", "task_id": "root"})
    host.drain()
    host.tick()
    host.clock[0] += 10000
    host.tick()
    assert _controls(host) == []
    assert queue._reap_queue.empty()
    assert host.running["root"] is host.meta


def test_root_background_off_gets_one_readonly_related_diagnostic(attention, monkeypatch):
    from ouroboros.work_diagnostics import diagnostic_ref

    monkeypatch.setattr(queue, "load_state", lambda: {"bg_consciousness_enabled": False})
    for _ in range(3):
        attention.tick()
        attention.flush()
    assert len(queue.PENDING) == 1
    child = queue.PENDING[0]
    assert child["parent_task_id"] == child["root_task_id"] == "root"
    assert child["task_constraint"]["mode"] == "local_readonly_subagent"
    assert child["metadata"]["billing_group"]["billing_group_limit_revision"] == "original"
    assert diagnostic_ref(child)["episode_id"] == observation(attention)["episode_id"]
    assert observation(attention)["diagnostic_id"] == child["id"]
    assert load_task_result(attention.root, child["id"])["delegation_admission"]["status"] == "accepted"
    mail = drain_owner_entries(attention.root, "root")
    assert len(mail) == 1 and mail[0]["kind"] == "task_message"
    assert "not a stop request" in mail[0]["text"]
    assert queue._reap_queue.empty() and attention.running["root"] is attention.meta


@pytest.mark.parametrize("mode", ["off", "auto", "swarm"])
def test_diagnostic_admission_obeys_live_room_mode(attention, monkeypatch, mode):
    monkeypatch.setenv("OUROBOROS_SUBAGENTS_MODE", mode)
    attention.tick()
    attention.flush()
    if mode == "off":
        assert not queue.PENDING
        fact = observation(attention)
        assert fact["attention_wait"] == "subagents_off_in_room"
        assert load_task_result(attention.root, fact["diagnostic_id"])["reason_code"] == "subagents_off_in_room"
    else:
        assert len(queue.PENDING) == 1
        assert observation(attention)["attention_wait"] == "diagnostic_queued"
    assert attention.running["root"] is attention.meta and queue._reap_queue.empty()


@pytest.mark.parametrize("phase", ["before", "preparation"])
def test_deleted_project_refuses_diagnostic_without_stopping_source(attention, monkeypatch, phase):
    from ouroboros import projects_registry as registry
    from supervisor import workspace_preparation

    registry.create_project(attention.root, "room")
    attention.meta["task"]["project_id"] = "room"
    if phase == "before":
        registry.begin_project_deletion(attention.root, "room")
    else:
        original_accept = workspace_preparation.accept
        def accept(*args, admission, **kwargs):
            def deleted(preparation):
                registry.begin_project_deletion(attention.root, "room")
                return admission(preparation)
            return original_accept(*args, admission=deleted, **kwargs)
        monkeypatch.setattr(workspace_preparation, "accept", accept)
    attention.tick()
    attention.flush()
    fact = observation(attention)
    assert not queue.PENDING
    assert fact["attention_wait"].startswith("project_routing_fence")
    assert load_task_result(attention.root, fact["diagnostic_id"])["reason_code"] == fact["attention_wait"]
    assert attention.running["root"] is attention.meta and queue._reap_queue.empty()


def test_diagnostic_rechecks_room_after_preparation_without_reading_under_queue_lock(attention, monkeypatch):
    from ouroboros import settings_integrity
    from supervisor import workspace_preparation

    monkeypatch.setenv("OUROBOROS_SUBAGENTS_MODE", "swarm")
    original_read, original_accept = settings_integrity.live_subagents_mode, workspace_preparation.accept
    reads = []

    def read(**kwargs):
        assert not queue._queue_lock._is_owned()
        value = original_read(**kwargs)
        reads.append(value)
        return value

    def accept(*args, admission, **kwargs):
        def switched(preparation):
            monkeypatch.setenv("OUROBOROS_SUBAGENTS_MODE", "off")
            return admission(preparation)
        return original_accept(*args, admission=switched, **kwargs)

    monkeypatch.setattr(settings_integrity, "live_subagents_mode", read)
    monkeypatch.setattr(workspace_preparation, "accept", accept)
    attention.tick()
    attention.flush()
    assert reads[0] == "swarm" and reads[-1] == "off"
    assert not queue.PENDING and observation(attention)["attention_wait"] == "subagents_off_in_room"


@pytest.mark.parametrize("origin", ["evolution", "evolution-descendant", "evolution-transaction", "consciousness"])
def test_autonomous_diagnostics_keep_their_own_controls(attention, monkeypatch, origin):
    from ouroboros.settings_integrity import live_subagents_mode

    monkeypatch.setenv("OUROBOROS_SUBAGENTS_MODE", "off")
    if origin == "evolution":
        attention.meta["task"]["type"] = "evolution"
    elif origin == "evolution-descendant":
        attention.meta["task"]["metadata"]["evolution_root_task_id"] = "evolution-root"
    elif origin == "evolution-transaction":
        attention.meta["task"]["metadata"]["evolution_transaction"] = {"id": "transaction"}
    else:
        attention.meta["task"]["metadata"]["work_kind"] = "consciousness"
    attention.tick()
    attention.flush()
    assert len(queue.PENDING) == 1
    assert live_subagents_mode(task=queue.PENDING[0], drive_root=attention.root) == "auto"
    assert "evolution_transaction" not in queue.PENDING[0]["metadata"]


def test_waiting_parent_receives_one_attention_through_actual_wait_consumer(attention, monkeypatch):
    from ouroboros.tools import control_task_results
    from ouroboros.tools.registry import ToolContext

    attention.meta["task"]["parent_task_id"] = "parent"
    attention.running["parent"] = {
        "task": {"id": "parent", "root_task_id": "parent"}, "attempt": 1,
        "started_at": 4000.0, "last_progress_at": 4999.0,
    }
    ctx = ToolContext(repo_dir=attention.root, drive_root=attention.root, task_id="parent",
                      event_queue=attention.bus)
    ctx.task_attempt = 1
    ctx.root_task_id = "parent"

    def wait(root, ids, **kwargs):
        assert ids == ["root"]
        attention.drain()  # actual in_slot_wait producer -> cognitive_operations reducer
        assert kwargs["on_poll"]({}, {}) is None
        for _ in range(2):
            attention.tick()
            attention.flush()
        early = kwargs["on_poll"]({}, {})
        assert early["reason"] == "owner_mailbox_pending"
        return {"early_return": early, "tasks": {}, "elapsed_sec": 0}

    monkeypatch.setattr(control_task_results, "wait_for_effective_tasks", wait)
    result = control_task_results._wait_for_task(ctx, "root", 30)
    assert "interrupted by an unread message" in result
    mail = drain_owner_entries(attention.root, "parent")
    assert len(mail) == 1 and mail[0]["source_task_id"] == "root"
    assert observation(attention)["recipient_task_id"] == "parent"
    assert queue.PENDING == []


def test_real_admission_generation_publishes_off_queue_lock(attention, monkeypatch):
    from supervisor import workspace_preparation as preparation

    monkeypatch.setattr(preparation, "_process_owner", lambda: {"pid": 11, "birth": 1})
    generation = preparation.PreparationGeneration()
    monkeypatch.setattr(preparation, "current_generation", lambda: generation)
    try:
        attention.tick()
        assert generation.lane_idle(preparation.ADMISSION_LANE, 15)
        assert len(queue.PENDING) == 1
        assert observation(attention)["self_delivered"]
    finally:
        generation.close()
        assert generation.lane_idle(preparation.ADMISSION_LANE, 15)


@pytest.mark.parametrize("selected", ["root", "middle"])
def test_warm_parent_waiting_for_helper_terminal_wakes_on_one_attention(attention, monkeypatch, selected):
    from ouroboros.owner_wait import checkpoint_owner_wait, direct_owner_wait, _sleep_wake
    from ouroboros.tools.control_task_results import _await_messages
    from supervisor.worker_owner_wait import handle_owner_wait
    from tests.test_model_sleep import _ctx

    attention.meta["task"]["parent_task_id"] = "parent" if selected == "root" else "middle"
    if selected == "middle":
        write_task_result(attention.root, "middle", "running")
        attention.running["middle"] = {"task": {"id": "middle", "parent_task_id": "parent"},
                                        "attempt": 1, "started_at": 4990.0, "last_progress_at": 4999.0}
    write_task_result(attention.root, "parent", "running")
    ctx = _ctx(attention.root, "parent", root="parent")
    assert '"sleep_armed"' in _await_messages(ctx, mode="warm", tasks=[selected])
    checkpoint = checkpoint_owner_wait(ctx, [{"role": "user", "content": "await helper"}], {}, {}, 1, [], set())
    commands = []
    worker = SimpleNamespace(wid=1, proc=SimpleNamespace(pid=77, is_alive=lambda: True),
                             busy_task_id="parent", reaping=False, active_capacity=True,
                             in_q=SimpleNamespace(put=commands.append))
    workers.WORKERS[1] = worker
    parent = {"task": {"id": "parent"}, "worker_id": 1, "attempt": 1,
              "started_at": 4900.0, "last_progress_at": 4999.0}
    attention.running["parent"] = parent
    handle_owner_wait({"task_id": "parent", "wait_id": checkpoint["wait_id"], "task_attempt": 1,
                       "worker_id": 1, "pid": 77, "phase": "park", "checkpoint": checkpoint},
                      SimpleNamespace(RUNNING=attention.running, WORKERS=workers.WORKERS, DRIVE_ROOT=attention.root))
    assert parent["sleep_parked_at"] and not worker.active_capacity
    assert _sleep_wake(ctx, checkpoint) == ""
    for _ in range(2):
        attention.tick()
        attention.flush()
    assert _sleep_wake(ctx, checkpoint) == "work_attention:root"
    assert direct_owner_wait(ctx, checkpoint) == "work_attention:root"
    mail = drain_owner_entries(attention.root, "parent")
    assert len(mail) == 1 and mail[0]["provenance"] == "system"
    assert mail[0]["work_attention"]["waiting_for_task_id"] == selected
    assert observation(attention)["parent_delivered"] and queue.PENDING == []


@pytest.mark.parametrize("paused", [False, True])
def test_cold_parent_waiting_for_helper_terminal_uses_existing_wake_grant(attention, monkeypatch, paused):
    from ouroboros import budget_pause, owner_pause
    from supervisor import state
    from supervisor.sleep_wake import wake_ready_sleepers
    from tests.test_model_sleep import _cold_park

    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    monkeypatch.setattr(budget_pause, "observe_task_runs", lambda *_a, **_k: {"custody_read": "ok", "runs": []})
    attention.meta["task"]["parent_task_id"] = "parent"
    _cold_park(attention.root, monkeypatch, workers, task_id="parent", tasks=["root"])
    parked = next(row for row in queue.PENDING if row["id"] == "parent")
    assert "parent" not in queue.RUNNING
    assert wake_ready_sleepers(queue) == []
    if paused:
        owner_pause.install_fence(attention.root, "parent", request_id="owner-pause")
    for _ in range(2):
        attention.tick()
        attention.flush()
    outcome = wake_ready_sleepers(queue)
    assert outcome[0]["ready"]["reason"] == "work_attention:root"
    if paused:
        assert outcome[0]["veto"] == "owner_pause"
        assert "_budget_pause_resume" not in parked
    else:
        assert outcome[0]["ok"]
        assert parked["_budget_pause_resume"]["selected_by"] == "sleep_wake"
    assert len(queue.PENDING) == 1 and queue.PENDING[0] is parked
    assert observation(attention)["parent_delivered"]
    assert len(drain_owner_entries(attention.root, "parent")) == 1


def test_real_owner_answer_wait_is_unavailable_even_with_old_helper_wait_lease(attention):
    attention.meta["task"]["parent_task_id"] = "parent"
    attention.running["parent"] = {
        "task": {"id": "parent"}, "attempt": 1, "started_at": 4900.0, "last_progress_at": 4999.0,
        "owner_wait": {"state": "waiting", "task_attempt": 1, "reason": "owner", "source_ref": {"x": 1}},
        "active_operation_leases": {"old": {"kind": "in_slot_wait", "until_ts": 9000, "wait_for_tasks": ["root"]}},
    }
    attention.tick()
    attention.flush()
    assert len(queue.PENDING) == 1 and not observation(attention).get("parent_delivered")


def test_sleep_attention_is_bound_to_the_selected_task_and_system_fact(attention):
    from ouroboros import model_sleep
    from ouroboros.owner_mailbox import write_task_message
    from tests.test_model_sleep import _ctx

    write_task_result(attention.root, "parent", "running")
    write_task_result(attention.root, "other", "running")
    ctx = _ctx(attention.root, "parent", root="parent")
    chosen = model_sleep.selectors(ctx, tasks=["other"])
    attention.meta["task"]["parent_task_id"] = "parent"
    attention.running["parent"] = {
        "task": {"id": "parent"}, "attempt": 1, "started_at": 4990.0,
        "owner_wait": {"state": "waiting", "task_attempt": 1, "reason": "sleep", "sleep": chosen},
    }
    attention.tick()
    attention.flush()
    assert len(queue.PENDING) == 1 and not observation(attention).get("parent_delivered")
    assert model_sleep.wake_reason(ctx, chosen) == ""
    write_task_message(attention.root, "ordinary system mail about selected task", "parent",
                       source_task_id="other", provenance="system")
    assert model_sleep.wake_reason(ctx, chosen) == ""
    write_task_message(attention.root, "an unrelated silence", "parent", source_task_id="root", provenance="system",
                       work_attention={"kind": "silence", "task_id": "root", "episode_id": "e",
                                       "waiting_for_task_id": "root"})
    assert model_sleep.wake_reason(ctx, chosen) == ""


@pytest.mark.parametrize("wait", [{"wait_for_tasks": ["unrelated"]}, {"kind": "owner_wait"}, {}])
def test_unrelated_or_unavailable_parent_does_not_take_responsibility(attention, wait):
    attention.meta["task"]["parent_task_id"] = "parent"
    attention.running["parent"] = {
        "task": {"id": "parent"}, "attempt": 1, "started_at": 4990.0, "last_progress_at": 4999.0,
        "active_operation_leases": {"wait": {"kind": "in_slot_wait", "until_ts": 9000, **wait}},
    }
    attention.tick()
    attention.flush()
    assert len(queue.PENDING) == 1
    assert not observation(attention).get("recipient_task_id")
    assert drain_owner_entries(attention.root, "parent") == []


def test_no_capacity_preserves_pending_diagnostic_and_original_custody(attention):
    attention.tick()
    attention.flush()
    assert workers.WORKERS == {} and len(queue.PENDING) == 1
    assert observation(attention)["attention_wait"] == "diagnostic_queued"
    assert queue._reap_queue.empty() and "root" in attention.running
    # Queue admission is demand, not a worker handoff or promise of recovery.
    assert queue.PENDING[0]["admitted_dispatch"] == "none"


def test_progress_withdraws_only_unsent_diagnostic_through_pending_cancellation(attention, monkeypatch):
    from ouroboros.work_diagnostics import control_reason

    attention.tick()
    attention.flush()
    first = observation(attention)["episode_id"]
    child = queue.PENDING[0]
    attention.meta["last_progress_at"] = attention.clock[0]
    attention.tick()
    attention.flush()
    assert observation(attention)["state"] == "progress_observed"
    assert control_reason(child, attention.root, before_handoff=True) == "cancelled"
    monkeypatch.setattr(workers, "_emit_task_done_terminal", lambda *a, **k: True)
    with queue._queue_lock:
        assert workers._drop_cancelled_pending()
    assert queue.PENDING == []
    assert load_task_result(attention.root, child["id"])["status"] == "cancelled"
    attention.clock[0] += 4000
    attention.tick()
    attention.flush()
    assert observation(attention)["episode_id"] != first
    assert len(queue.PENDING) == 1
    assert queue.PENDING[0]["id"] != child["id"]


def test_recovery_does_not_release_claimed_diagnostic_or_allow_overlap(attention):
    from ouroboros.cancel_intents import cancel_pending

    attention.tick()
    attention.flush()
    child = queue.PENDING[0]
    child["_launch_claim"] = {"token": "in-flight-preparation"}
    attention.meta["last_progress_at"] = attention.clock[0]
    attention.tick()
    attention.flush()
    assert not cancel_pending(attention.root, child["id"], strict=True)
    attention.clock[0] += 4000
    attention.tick()
    attention.flush()
    assert queue.PENDING == [child]
    assert observation(attention)["attention_wait"] == "diagnostic_already_owned"


def test_terminal_file_does_not_release_or_kill_source(attention):
    worker = SimpleNamespace(busy_task_id="root", proc=SimpleNamespace(
        is_alive=lambda: True, terminate=attention.forbidden, kill=attention.forbidden))
    workers.WORKERS[0] = worker
    write_task_result(attention.root, "root", "completed", result="An answer is not process death.")
    attention.tick()
    attention.flush()
    assert load_task_result(attention.root, "root")["status"] == "completed"
    assert "root" in attention.running and queue._reap_queue.empty()
    assert len(queue.PENDING) == 1
    assert workers.WORKERS[0] is worker and worker.busy_task_id == "root"


@pytest.mark.parametrize("barrier", ["panic", "stop", "pause", "deadline", "billing"])
def test_source_barriers_prevent_diagnostic_admission(attention, monkeypatch, barrier):
    if barrier == "panic":
        monkeypatch.setattr("ouroboros.installation_stop.installation_panic", lambda *a, **k: True)
    elif barrier == "stop":
        from ouroboros.cancel_intents import request_cancel
        request_cancel(attention.root, "root", reason="owner stop", source="owner", requested_by="owner")
    elif barrier == "pause":
        write_task_result(attention.root, "root", "running", owner_pause={"state": "requested", "fence_id": "pause"})
    elif barrier == "deadline":
        write_task_result(attention.root, "root", "running", deadline_at="2000-01-01T00:00:00Z")
    else:
        monkeypatch.setattr("ouroboros.usage_admission.task_billing_fields", lambda *a, **k: {})
    attention.tick()
    attention.flush()
    assert queue.PENDING == []
    assert observation(attention)["attention_wait"]
    assert queue._reap_queue.empty() and "root" in attention.running


def test_diagnostic_controls_rechecked_before_model_send(attention, monkeypatch):
    from ouroboros.model_wait import TaskModelWait, dispatch_control_reason

    monkeypatch.setattr("ouroboros.platform_layer.process_start_time", lambda pid: None)
    attention.tick()
    attention.flush()
    owner = TaskModelWait(task=queue.PENDING[0], drive_root=attention.root,
                          event_queue=None, worker_slot_held=True)
    assert dispatch_control_reason(owner, before_task_start=True) is None
    write_task_result(attention.root, "root", "running", owner_pause={"state": "requested", "fence_id": "pause"})
    assert dispatch_control_reason(owner, handed_off=True) is None
    assert dispatch_control_reason(owner) == "owner_pause"
    assert dispatch_control_reason(owner, before_task_start=True) == "owner_pause"
    from ouroboros.budget_pause import prestart_pause_event
    error = SimpleNamespace(before_task_start=True, control_reason="owner_pause")
    event = prestart_pause_event(owner.task, error, owner)
    assert event["resource_limit"]["fence_id"] == "pause"
    assert event["resource_limit"]["physical_calls"] == 0


def test_diagnostic_never_recursively_diagnosed(attention):
    attention.tick()
    attention.flush()
    child = queue.PENDING.pop()
    attention.running[child["id"]] = {"task": child, "started_at": 1000.0,
        "last_progress_at": 1000.0, "attempt": 1, "worker_id": 1}
    for _ in range(2):
        attention.tick()
        attention.flush()
    fact = load_task_result(attention.root, child["id"])["silence_attention"]
    assert fact["attention_wait"] == "diagnostic_not_recursed"
    assert queue.PENDING == [] and queue._reap_queue.empty()


def test_diagnostic_registry_exposes_controls_but_blocks_product_and_external_tools(attention, monkeypatch):
    from ouroboros.agent import Env, OuroborosAgent
    from ouroboros.tool_access_roots import active_tool_profile

    attention.tick()
    attention.flush()
    task = queue.PENDING[0]
    repo = attention.root / "repo"
    repo.mkdir()
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr("ouroboros.agent.build_llm_messages", lambda **kwargs: ([], {}))
    agent = OuroborosAgent(Env(repo_dir=repo, drive_root=attention.root))
    monkeypatch.setattr(agent.llm, "chat", attention.forbidden)
    monkeypatch.setattr(agent, "_run_delegate_preflight", lambda logs, task, dispatch: (dispatch, False))
    ctx, _messages, _caps = agent._prepare_task_context(task)
    assert ctx.task_metadata["work_diagnostic"] == task["metadata"]["work_diagnostic"]
    registry = agent.tools
    monkeypatch.setattr("ouroboros.mcp_client.ensure_configured_from_settings", attention.forbidden)
    assert active_tool_profile(ctx) == "local_readonly_subagent"
    assert registry.get_schema_by_name("pause_task") is not None
    assert registry.get_schema_by_name("get_task_result") is not None
    for name in ("write_file", "run_command", "delegate_start", "schedule_subagent", "configure_presence"):
        assert registry.get_schema_by_name(name) is None
        assert registry.execute_result(name, {}).status == "blocked"
    assert not registry._extension_rows(record=False) and not registry._mcp_rows(record=False)


def test_diagnostic_can_pass_writer_lease_but_keeps_capacity_and_budget_gates(attention):
    from supervisor.worker_assignment import _candidate_ineligible

    attention.meta["task"]["project_id"] = "same-project"
    attention.tick()
    attention.flush()
    child = queue.PENDING[0]
    running = list(attention.running.values())
    assert not _candidate_ineligible(queue, child, 1000, running)
    assert _candidate_ineligible(queue, child, 0, running)
    queue.BUDGET_ROOT_FENCES["root"] = {"fence_id": "money"}
    assert _candidate_ineligible(queue, child, 1000, running)


@pytest.mark.parametrize("source_stopped", [False, True])
def test_diagnostic_producer_to_worker_put_revalidates_source_off_queue_lock(attention, monkeypatch, source_stopped):
    from ouroboros import work_diagnostics
    from ouroboros.cancel_intents import request_cancel
    from supervisor import worker_assignment, workspace_preparation as preparation
    from tests.worker_pool_support import affirm_ready

    attention.tick()
    attention.flush()
    child = queue.PENDING[0]
    if source_stopped:
        request_cancel(attention.root, "root", reason="owner stop", source="owner", requested_by="owner")
    checks, sent = [], []
    check = work_diagnostics.control_reason

    def off_lock(*args, **kwargs):
        assert not queue._queue_lock._is_owned()
        checks.append(True)
        return check(*args, **kwargs)

    monkeypatch.setattr(work_diagnostics, "control_reason", off_lock)
    monkeypatch.setattr(preparation, "_process_owner", lambda: {"pid": 11, "birth": 1})
    monkeypatch.setattr(workers, "REPO_DIR", attention.root)
    worker = SimpleNamespace(wid=1, proc=SimpleNamespace(pid=77, is_alive=lambda: True),
                             busy_task_id=None, reaping=False, active_capacity=True,
                             in_q=SimpleNamespace(put=sent.append))
    affirm_ready(worker)
    workers.WORKERS[1] = worker
    generation = preparation.PreparationGeneration()
    monkeypatch.setattr(preparation, "current_generation", lambda: generation)
    try:
        with queue._queue_lock:
            assert worker_assignment._claim_worker_launch(queue, child, worker) is None
        assert generation.lane_idle(preparation.ADMISSION_LANE, 15)
        assert checks
        if source_stopped:
            assert not sent and child in queue.PENDING and not worker.busy_task_id
        else:
            assert len(sent) == 1 and sent[0]["metadata"]["work_diagnostic"] == child["metadata"]["work_diagnostic"]
            assert sent[0]["task_constraint"]["mode"] == "local_readonly_subagent"
            assert child["id"] in attention.running and child not in queue.PENDING
        assert "root" in attention.running and queue._reap_queue.empty()
    finally:
        generation.close()
        assert generation.lane_idle(preparation.ADMISSION_LANE, 15)


@pytest.mark.parametrize("source_change", ["pause", "new_episode"])
@pytest.mark.parametrize("root_control", ["stop", "deadline"])
def test_sent_diagnostic_source_change_cannot_hide_root_hard_control(monkeypatch, source_change, root_control):
    from ouroboros.work_diagnostics import control_reason

    task = {"id": "silence-episode", "root_task_id": "root", "parent_task_id": "helper",
            "delegation_role": "subagent", "task_constraint": {"mode": "local_readonly_subagent"},
            "metadata": {"work_diagnostic": {"kind": "silence", "task_id": "helper", "root_task_id": "root",
                                               "task_attempt": 1, "episode_id": "episode"}}}
    source = {"root_task_id": "root", "task_attempt": 1, "silence_attention": {
        "task_id": "helper", "task_attempt": 1, "episode_id": "new" if source_change == "new_episode" else "episode"}}
    if source_change == "pause":
        source["owner_pause"] = {"state": "requested", "fence_id": "pause"}
    rows = {"helper": source, "root": {"deadline_at": "2000-01-01T00:00:00Z"} if root_control == "deadline" else {"id": "root"}}
    monkeypatch.setattr("ouroboros.installation_stop.installation_panic", lambda *_a, **_k: False)
    monkeypatch.setattr("ouroboros.cancel_intents.cancel_pending", lambda _root, tid, **_k: root_control == "stop" and tid == "root")
    monkeypatch.setattr("ouroboros.task_results.load_task_result", lambda _root, tid, **_k: rows[tid])
    assert control_reason(task, "unused", before_handoff=False) == ("cancelled" if root_control == "stop" else "deadline")


def test_diagnostic_physical_reservation_cannot_reset_original_money(attention):
    from ouroboros import usage_accounting as ua
    from ouroboros.usage_admission import task_billing_fields
    from tests.test_billing_group import _scope, _spend
    from tests._usage_store_testing import ledger_rows

    attention.tick()
    attention.flush()
    child = queue.PENDING[0]
    fields = task_billing_fields(child, "root", None, attention.root)
    assert fields["billing_group_id"] == "root" and fields["billing_group_limit_usd"] == 10
    _spend(attention.root, _scope(attention.root, "root", "root", group="root", group_limit=10), 10)
    scope = ua.UsageScope(drive_root=attention.root, task_id=child["id"], root_task_id="root", **fields)
    with ua.usage_scope(scope), pytest.raises(ua.BudgetExceeded):
        ua.reserve_attempt(ua.AttemptRequest(model="openai/gpt-5.2", provider="openai", reservation_usd=0.01))
    assert {row["task_id"] for row in ledger_rows(attention.root)} == {"root"}


def test_diagnostic_deadline_is_original_remaining_time_not_fresh_lifetime(attention):
    from ouroboros.deadline_utils import parse_deadline_ts

    attention.meta["task"].update(execution_timeout_sec=4100, deadline_at="2030-01-01T00:00:00Z")
    attention.tick()
    attention.flush()
    child = queue.PENDING[0]
    assert child["execution_timeout_sec"] == 4100 and child["_execution_consumed_sec"] == 4000
    assert parse_deadline_ts(child["deadline_at"]).timestamp() == 5100
    assert attention.meta["task"]["deadline_at"] == "2030-01-01T00:00:00Z"


def test_arbitrary_parent_or_unbound_episode_has_no_diagnostic_admission(attention):
    from ouroboros.work_diagnostics import control_reason

    attention.tick()
    attention.flush()
    child = queue.PENDING[0]
    assert control_reason(child, attention.root, before_handoff=True) is None
    assert "binding_mismatch" in control_reason({**child, "parent_task_id": "unrelated"}, attention.root, before_handoff=True)
    write_task_result(attention.root, "root", "running", silence_attention={})
    assert control_reason(child, attention.root, before_handoff=True) == "cancelled"


def test_delivery_failure_retries_same_episode_without_duplicate_consumption(attention, monkeypatch):
    from ouroboros.config import BODY_MAINTENANCE_INTERVAL_SEC
    from supervisor import silence_attention

    save, calls = silence_attention._save, []

    def interrupted(*args):
        calls.append(True)
        if len(calls) == 2:
            raise OSError("lost acknowledgement after delivery and admission")
        return save(*args)

    monkeypatch.setattr(silence_attention, "_save", interrupted)
    for _ in range(3):
        attention.tick()
        attention.flush()
        attention.clock[0] += BODY_MAINTENANCE_INTERVAL_SEC
    assert len(queue.PENDING) == 1
    assert len(drain_owner_entries(attention.root, "root")) == 1
    assert observation(attention)["diagnostic_id"] == queue.PENDING[0]["id"]


def test_unavailable_diagnostic_model_remains_one_owned_wait(attention):
    from ouroboros.model_wait import TaskModelWait
    from supervisor.task_model_wait import handle_task_model_wait

    attention.tick()
    attention.flush()
    child = queue.PENDING.pop()
    attention.running[child["id"]] = {"task": child, "started_at": 1000.0,
        "last_progress_at": 1000.0, "attempt": 1, "worker_id": 1}
    owner = TaskModelWait(task=child, drive_root=attention.root, event_queue=attention.bus,
                          worker_slot_held=True)
    owner._publish({"wait_id": "route-unavailable", "task_attempt": 1,
                    "state": "waiting", "reason": "unavailable", "auto_continue": False})
    ctx = SimpleNamespace(RUNNING=attention.running, DRIVE_ROOT=attention.root,
                          bridge=SimpleNamespace(push_log=lambda event: None))
    handle_task_model_wait(attention.bus.get_nowait(), ctx)
    attention.clock[0] += 4000
    attention.tick()
    attention.flush()
    row = load_task_result(attention.root, child["id"])
    assert row["model_waits"]["route-unavailable"]["state"] == "waiting"
    assert row["model_waits"]["route-unavailable"]["reason"] == "unavailable"
    assert queue.PENDING == [] and queue._reap_queue.empty()
    assert child["id"] in attention.running and "root" in attention.running


@pytest.mark.parametrize("fields,label", [
    ({"title": "Quarterly cost reconciliation", "text": "Reconcile\nthe ledger"}, "Quarterly cost reconciliation"),
    ({"text": "\nRebuild the search index\nwith all shards"}, "Rebuild the search index"),
    ({}, "untitled work"),
])
def test_diagnostic_card_names_the_source_by_its_human_title(attention, fields, label):
    attention.meta["task"].update(fields)
    attention.tick()
    attention.flush()
    child = queue.PENDING[0]
    assert child["description"] == f"Silence check: {label}"
    assert load_task_result(attention.root, child["id"])["description"] == child["description"]
    # The opaque identity stays in the details the diagnostic reads.
    assert "work root (" in child["text"] and child["metadata"]["work_diagnostic"]["task_id"] == "root"
