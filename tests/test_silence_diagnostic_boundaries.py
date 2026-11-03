"""A silence diagnostic retains its source's progress, money and admission owners."""

import json
from types import SimpleNamespace

import pytest

from ouroboros.task_results import load_task_result, write_task_result
from supervisor import queue, workers
from tests.test_silence_attention import attention as _attention, host as _host, observation

attention = _attention
host = _host


@pytest.fixture(autouse=True)
def no_external_effects(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("This test permits no network, subprocess or process signal")

    for path in ("socket.create_connection", "socket.socket.connect", "socket.socket.connect_ex",
                 "subprocess.Popen", "os.kill", "os.killpg"):
        monkeypatch.setattr(path, forbidden, raising=False)


def test_missing_platform_signal_symbol_is_still_forbidden(monkeypatch):
    import os

    monkeypatch.delattr(os, "killpg", raising=False)  # Windows has no such symbol.
    no_external_effects.__wrapped__(monkeypatch)
    with pytest.raises(AssertionError, match="no network, subprocess or process signal"):
        os.killpg(0, 0)


def _mint(attention):
    attention.tick()
    attention.flush()
    return queue.PENDING[0]


def _running(attention, task):
    if task in queue.PENDING:
        queue.PENDING.remove(task)
    attention.running[task["id"]] = {
        "task": task, "attempt": 1, "worker_id": 1,
        "started_at": 1000.0, "last_progress_at": 1000.0,
    }


@pytest.mark.parametrize("diagnostic", [True, False])
@pytest.mark.parametrize("retired", [True, False])
def test_child_settlement_only_rearms_source_for_real_work(attention, monkeypatch, diagnostic, retired):
    from supervisor.events_task_done import _handle_task_done
    from ouroboros.utils import append_jsonl

    child = _mint(attention)
    episode = observation(attention)["episode_id"]
    if not diagnostic:
        child["metadata"].pop("work_diagnostic")
    _running(attention, child)
    write_task_result(attention.root, child["id"], "completed", metadata=child["metadata"],
                      parent_task_id="root", root_task_id="root", delegation_role="subagent",
                      result="Inspection finished; source custody remains.")
    if retired:
        attention.running.pop(child["id"])
    delivered = []
    ctx = SimpleNamespace(RUNNING=attention.running, WORKERS=workers.WORKERS, DRIVE_ROOT=attention.root,
                          append_jsonl=append_jsonl, send_with_budget=lambda *a, **k: True,
                          persist_queue_snapshot=queue.persist_queue_snapshot,
                          bridge=SimpleNamespace(push_log=delivered.append))
    monkeypatch.setattr("ouroboros.headless.terminal_task_files_ready", lambda *a, **k: True)
    _handle_task_done({"type": "task_done", "task_id": child["id"], "worker_id": 1,
                       "status": "completed", "task_type": "task", "_files_prepared_attempt": 1}, ctx)
    assert child["id"] not in attention.running
    assert any(row.get("type") == "task_done" and row["task_id"] == child["id"] for row in delivered)
    assert attention.meta["last_progress_at"] == (1000.0 if diagnostic else attention.clock[0])
    attention.tick()
    attention.flush()
    attention.clock[0] += 4000
    attention.tick()
    attention.flush()
    assert (observation(attention)["episode_id"] == episode) is diagnostic
    assert len(queue.PENDING) == (0 if diagnostic else 1)


@pytest.mark.parametrize("diagnostic", [True, False])
def test_pending_and_folded_child_activity_keep_their_own_progress(attention, monkeypatch, diagnostic):
    from supervisor import cognitive_operations, events_budget, events_chat_delivery

    child = _mint(attention)
    if not diagnostic:
        child["metadata"].pop("work_diagnostic")
    _running(attention, child)
    # No prior observation: the outstanding child's event must not hide new source silence.
    attention.meta.pop("silence_attention")
    attention.bus.put({"type": "send_message", "task_id": child["id"], "chat_id": 7,
                       "text": "Inspection progress", "is_progress": True})
    attention.tick()
    assert bool(attention.meta.get("silence_attention")) is diagnostic
    for module in (cognitive_operations, events_budget, events_chat_delivery):
        monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: attention.clock[0]))
    attention.drain()
    assert attention.running[child["id"]]["last_progress_at"] == attention.clock[0]
    assert attention.meta["last_progress_at"] == 1000.0
    assert queue._subtree_progressing("root", attention.clock[0], 900) is (not diagnostic)


@pytest.mark.parametrize("nested", [False, True])
def test_diagnostic_scope_resolves_original_producer_allowance(attention, nested):
    from ouroboros import task_pacing, usage_accounting as ua
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.usage_admission import task_billing_fields
    from ouroboros.usage_origin import task_usage_scope

    source = attention.meta["task"]
    source["metadata"].update(work_kind="consciousness", root_cost_ceiling_usd=2.0)
    root_id = "ancestor" if nested else "root"
    source["root_task_id"] = root_id
    write_task_result(attention.root, root_id, "running", root_task_id=root_id,
                      metadata={"work_kind": "consciousness", "root_cost_ceiling_usd": 0.75},
                      task_contract=build_task_contract({"id": root_id, "text": "Original work"}),
                      billing_group={"billing_group_id": root_id, "billing_group_limit_usd": 10,
                                     "billing_group_limit_source": "initial_task_admission"})
    if nested:
        write_task_result(attention.root, "root", "running", root_task_id=root_id)
    child = _mint(attention)
    scope = task_usage_scope(child, drive_root=attention.root, root_task_id=root_id, parent_task_id="root",
                             billing=task_billing_fields(child, root_id, None, attention.root))
    ctx = SimpleNamespace(task_contract=child["task_contract"])
    with ua.usage_scope(scope):
        ceiling = task_pacing.resolve_task_cost_ceiling(ctx, 1000)
        assert task_pacing.cost_stop_authority(ctx) == task_pacing.COST_STOP_PRODUCER
    assert ceiling.ceiling_usd == 0.75
    assert ceiling.root_cap_usd == 10


def _agent(attention, monkeypatch, task):
    from ouroboros.agent import Env, OuroborosAgent

    repo = attention.root / "repo"
    repo.mkdir(exist_ok=True)
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr("ouroboros.agent.build_llm_messages", lambda **kwargs: ([], {}))
    agent = OuroborosAgent(Env(repo_dir=repo, drive_root=attention.root))
    monkeypatch.setattr(agent.llm, "chat", attention.forbidden)
    monkeypatch.setattr(agent, "_run_delegate_preflight", lambda logs, task, dispatch: (dispatch, False))
    ctx, _messages, _caps = agent._prepare_task_context(task)
    monkeypatch.setattr("ouroboros.mcp_client.ensure_configured_from_settings", attention.forbidden)
    return agent.tools, ctx


def test_nano_diagnostic_can_load_only_permitted_schemas(attention, monkeypatch):
    from ouroboros.tool_policy import initial_tool_schemas, request_tool_schema_selection
    from ouroboros.tools.tool_discovery import bind_resident_schemas

    registry, ctx = _agent(attention, monkeypatch, _mint(attention))
    ctx.active_context_mode = "nano"
    resident = initial_tool_schemas(registry, context_mode="nano")
    names = lambda schemas: {s["function"]["name"] for s in schemas}
    assert {"enable_tools", "list_available_tools", "compact_context"} <= names(resident)
    assert "get_task_result" not in names(resident)
    bind_resident_schemas(registry, resident)
    result = registry.execute_result("enable_tools", {"tools": "get_task_result,write_file,run_command,browse_page"})
    assert result.status == "ok", result.text
    assert "get_task_result" in names(resident)
    requested = request_tool_schema_selection(ctx, registry, ["query_data", "write_file"],
                                              context_mode="nano", current_schemas=resident)
    assert "query_data" in requested.selection.chosen
    assert "write_file" not in requested.selection.chosen
    for name in ("write_file", "run_command", "browse_page", "schedule_subagent", "delegate_start"):
        assert name not in names(resident)
        assert registry.get_schema_by_name(name) is None
        assert registry.execute_result(name, {}).status == "blocked"
    ctx.task_contract["disabled_tools"] = ["enable_tools"]
    assert registry.get_schema_by_name("enable_tools") is None
    assert registry.execute_result("enable_tools", {"tools": "read_file"}).status == "blocked"


@pytest.mark.parametrize("gap", ["readback", "enqueue", "source_recovered"])
def test_receipt_gap_keeps_one_admission_owner(attention, monkeypatch, gap):
    from ouroboros.store import StoreBusy
    from supervisor import task_admission, workspace_preparation as preparation

    monkeypatch.setattr(preparation, "_process_owner", lambda: {"pid": 11, "birth": 1})
    generation = preparation.PreparationGeneration()
    monkeypatch.setattr(preparation, "current_generation", lambda: generation)
    original_read, original_write, original_enqueue = (
        task_admission.load_task_result, task_admission.write_task_result, queue.enqueue_task)
    receipts, gaps, puts = [], [], []

    def write(root, tid, status, **fields):
        result = original_write(root, tid, status, **fields)
        if status == "scheduled" and tid.startswith("silence-"):
            receipts.append(tid)
            if gap == "source_recovered":
                attention.meta["last_progress_at"] = attention.clock[0]
                attention.meta["silence_attention"].update(state="progress_observed", recovered_at=attention.clock[0])
        return result

    def read(root, tid, **kwargs):
        if gap == "readback" and tid in receipts and not gaps:
            gaps.append(tid)
            raise StoreBusy("injected receipt readback contention", reason="contention")
        return original_read(root, tid, **kwargs)

    def enqueue(task):
        if gap == "enqueue" and not gaps:
            gaps.append(task["id"])
            return {**task, "_admission_blocked": "contention", "_admission_never_admitted": True}
        result = original_enqueue(task)
        puts.append(result)
        return result

    monkeypatch.setattr(task_admission, "write_task_result", write)
    monkeypatch.setattr(task_admission, "load_task_result", read)
    monkeypatch.setattr(queue, "enqueue_task", enqueue)
    try:
        for _ in range(3):
            attention.tick()
            assert generation.lane_idle(preparation.ADMISSION_LANE, 10)
        assert len(receipts) == 1
        if gap == "source_recovered":
            assert queue.PENDING == [] and puts == []
            assert load_task_result(attention.root, receipts[0])["status"] == "failed"
            assert load_task_result(attention.root, receipts[0])["admission_outcome"] == "never_admitted"
        else:
            assert gaps == receipts
            assert len(queue.PENDING) == len(puts) == 1
            row = load_task_result(attention.root, receipts[0])
            assert row["workspace_preparation"]["phase"] == "admitted"
            assert row["workspace_preparation"]["token"] == queue.PENDING[0]["_admission_owner_token"]
        assert not queue.ADMISSION_RESERVATIONS
    finally:
        generation.close()
        assert generation.lane_idle(preparation.ADMISSION_LANE, 10)


@pytest.mark.parametrize("dispatch", ["none", "committed_unsent", "possible"])
def test_recovery_between_launch_claim_and_refusal_settles_only_unsent(attention, monkeypatch, dispatch):
    from ouroboros.cancel_intents import cancel_pending
    from supervisor import worker_assignment, workspace_preparation as preparation
    from tests.worker_pool_support import affirm_ready

    child = _mint(attention)
    episode = observation(attention)["episode_id"]
    monkeypatch.setattr(preparation, "_process_owner", lambda: {"pid": 11, "birth": 1})
    generation = preparation.PreparationGeneration()
    jobs = []
    monkeypatch.setattr(generation, "submit", lambda job, lane: jobs.append(job) or True)
    monkeypatch.setattr(preparation, "current_generation", lambda: generation)
    monkeypatch.setattr(workers, "REPO_DIR", attention.root)
    sent = []
    worker = SimpleNamespace(wid=1, proc=SimpleNamespace(pid=77, is_alive=lambda: True),
                             busy_task_id=None, reaping=False, active_capacity=True,
                             in_q=SimpleNamespace(put=sent.append))
    affirm_ready(worker)
    workers.WORKERS[1] = worker
    attention.meta["last_progress_at"] = attention.clock[0]
    attention.tick()  # recovery publication is queued before the launch job
    if dispatch == "possible":
        child["admitted_dispatch"] = "possible"
    with queue._queue_lock:
        assert worker_assignment._claim_worker_launch(queue, child, worker) is None
    assert child["_launch_claim"]
    if dispatch == "committed_unsent":
        assert worker_assignment._commit_launch(queue, child) is not None
        assert child["admitted_dispatch"] == "possible"
        assert worker_assignment._unsent(child) is not None
    jobs.pop(0)()  # cannot withdraw somebody else's live claim
    assert not cancel_pending(attention.root, child["id"], strict=True)
    jobs.pop(0)()  # refusal and release by the launch's own owner
    assert not sent and not child.get("_launch_claim") and worker.busy_task_id is None
    assert cancel_pending(attention.root, child["id"], strict=True) is (dispatch != "possible")
    if dispatch != "possible":
        monkeypatch.setattr(workers, "_emit_task_done_terminal", lambda *a, **k: True)
        with queue._queue_lock:
            assert workers._drop_cancelled_pending()
        assert queue.PENDING == []
        assert load_task_result(attention.root, child["id"])["status"] == "cancelled"
        attention.clock[0] += 4000
        attention.tick()
        while jobs:
            jobs.pop(0)()
        assert observation(attention)["episode_id"] != episode
        assert len(queue.PENDING) == 1 and queue.PENDING[0]["id"] != child["id"]
    worker_assignment._forget_departed_launches(queue.PENDING)


@pytest.mark.parametrize("busy", [True, False])
def test_physical_control_keeps_contention_distinct_from_unreadable(attention, monkeypatch, busy):
    from ouroboros.model_wait import TaskModelWait, dispatch_control_reason
    from ouroboros.store import StoreBusy, StoreError

    child = _mint(attention)
    owner = TaskModelWait(task=child, drive_root=attention.root, event_queue=None, worker_slot_held=True)
    error = StoreBusy("injected source contention", reason="contention") if busy else StoreError("unreadable source")
    monkeypatch.setattr("ouroboros.task_results.load_task_result", lambda *a, **k: (_ for _ in ()).throw(error))
    if busy:
        with pytest.raises(StoreBusy):
            dispatch_control_reason(owner, before_task_start=True)
    else:
        assert "diagnostic_source_unreadable" in dispatch_control_reason(owner, before_task_start=True)


@pytest.mark.parametrize("action", ["pause_task", "resume_task", "continue_task"])
def test_diagnostic_controls_cannot_target_unrelated_work(attention, monkeypatch, action):
    from ouroboros.tool_call_log import invocation_scope

    child = _mint(attention)
    registry, ctx = _agent(attention, monkeypatch, child)
    ctx.pending_events, ctx.event_queue = [], None
    with invocation_scope({"tool_call_id": "control-call"}):
        outcome = registry.execute_result(action, {"task_id": "unrelated", "reason": "inspection decision"})
    assert not ctx.pending_events
    assert outcome.status == "blocked", outcome.text
    assert "diagnostic_control_target" in outcome.text


@pytest.mark.parametrize("action", ["pause_task", "resume_task", "continue_task"])
def test_supervisor_rechecks_diagnostic_target_from_host_record(attention, monkeypatch, action):
    from supervisor.events_mind_controls import _apply

    child = _mint(attention)
    for path in ("supervisor.owner_pause_control.request_owner_pause",
                 "supervisor.queue_transitions.resume_budget_paused_task",
                 "supervisor.continuation_admission.admit_continuation"):
        monkeypatch.setattr(path, attention.forbidden)
    result = _apply({"action": action, "task_id": "unrelated", "actor_task_id": child["id"],
                     "action_nonce": "direct-event", "reason": "inspection decision"})
    assert result == {"ok": False, "error": "diagnostic_control_target"}


def test_diagnostic_source_pause_uses_owner_mail_and_binds_the_diagnostic(attention, monkeypatch):
    from ouroboros.owner_mailbox import drain_owner_entries
    from ouroboros.owner_pause import read_fence
    from ouroboros.tool_call_log import invocation_scope
    from supervisor.events_mind_controls import handle_mind_task_control

    child = _mint(attention)
    registry, ctx = _agent(attention, monkeypatch, child)
    # A queued source lets the real Pause/Resume owners settle without a process.
    source = attention.running.pop("root")["task"]
    source["admitted_dispatch"] = "none"
    queue.PENDING.append(source)
    write_task_result(attention.root, "root", "scheduled", root_task_id="root")
    ctx.pending_events, ctx.event_queue = [], None
    delivered = []
    host_ctx = SimpleNamespace(DRIVE_ROOT=attention.root, bridge=SimpleNamespace(push_log=delivered.append))
    with invocation_scope({"tool_call_id": "pause-source"}):
        requested = registry.execute_result("pause_task", {"task_id": "root", "reason": "source inspection"})
    assert requested.status == "ok", requested.text
    handle_mind_task_control(ctx.pending_events.pop(), host_ctx)
    mail = drain_owner_entries(attention.root, child["id"])
    assert len(mail) == 1
    assert json.loads(mail[0]["text"])["outcome"]["ok"], mail
    assert delivered[-1]["outcome"]["ok"]
    assert read_fence(attention.root, "root")["by"]["task_id"] == child["id"]
    # The same tree's Pause also binds the diagnostic; its tool catalog is no exemption.
    with invocation_scope({"tool_call_id": "resume-source"}):
        refused = registry.execute_result("resume_task", {"task_id": "root", "reason": "source inspection"})
    assert refused.status == "blocked" and "OWNER_PAUSE_NOT_STARTED" in refused.text
    assert ctx.pending_events == []


def test_delivered_episode_stops_republishing_but_unavailable_admission_recovers(attention, monkeypatch):
    from ouroboros.config import BODY_MAINTENANCE_INTERVAL_SEC
    from supervisor import silence_attention
    from ouroboros import usage_admission

    billing = usage_admission.task_billing_fields
    monkeypatch.setattr(usage_admission, "task_billing_fields", lambda *a, **k: {})
    attention.tick()
    attention.flush()
    assert not queue.PENDING
    monkeypatch.setattr(usage_admission, "task_billing_fields", billing)
    attention.clock[0] += BODY_MAINTENANCE_INTERVAL_SEC
    _mint(attention)
    published = observation(attention)
    monkeypatch.setattr(silence_attention, "_publish", attention.forbidden)
    for _ in range(3):
        attention.clock[0] += 4000
        attention.tick()
        attention.flush()
    assert observation(attention) == published
    assert len(queue.PENDING) == 1


def test_receipt_owner_never_replays_an_uncertain_queue_append(attention, monkeypatch):
    enqueue = queue.enqueue_task
    puts = []

    def entered(task):
        result = enqueue(task)
        puts.append(result)
        return {**result, "_admission_blocked": "contention", "_admission_never_admitted": True}

    monkeypatch.setattr(queue, "enqueue_task", entered)
    for _ in range(3):
        attention.tick()
        attention.flush()
    assert len(puts) == len(queue.PENDING) == 1
    row = load_task_result(attention.root, puts[0]["id"])
    assert row["status"] == "scheduled"
    assert row["workspace_preparation"]["token"] == queue.PENDING[0]["_admission_owner_token"]


def _handoff_diagnostic(attention, monkeypatch):
    """Use the real chooser/launch lane; only the worker and its inbox are doubles."""
    from supervisor import worker_assignment
    from supervisor import workspace_preparation as preparation
    from tests.worker_pool_support import affirm_ready

    sent = []
    monkeypatch.setattr(workers, "REPO_DIR", attention.root)
    worker = SimpleNamespace(wid=1, proc=SimpleNamespace(pid=77, is_alive=lambda: True),
                             busy_task_id=None, reaping=False, active_capacity=True,
                             in_q=SimpleNamespace(put=sent.append))
    affirm_ready(worker)
    workers.WORKERS[1] = worker
    generation = preparation.PreparationGeneration()
    with monkeypatch.context() as patch:
        patch.setattr(preparation, "current_generation", lambda: generation)
        try:
            with queue._queue_lock:
                index, _ = worker_assignment._choose_candidate(
                    queue, 1000, set(), {row["id"] for row in queue.PENDING
                                        if not row["id"].startswith("silence-")},
                    list(attention.running.values()), set())
                if index is not None:
                    assert worker_assignment._claim_worker_launch(queue, queue.PENDING[index], worker) is None
            assert generation.lane_idle(preparation.ADMISSION_LANE, 15)
        finally:
            generation.close()
            assert generation.lane_idle(preparation.ADMISSION_LANE, 15)
    return sent


@pytest.mark.parametrize("budget,depth,reason", [
    ({"may_delegate": False}, 0, "delegation_rights_may_delegate"),
    ({"depth_remaining": 0}, 0, "delegation_rights_depth_exhausted"),
    ({}, 3, "subtask_depth_limit"),
    ({"may_delegate": True, "depth_remaining": 1}, 0, ""),
], ids=["no-delegation", "depth-exhausted", "configured-depth", "permitted-control"])
def test_diagnostic_respects_source_delegation_at_worker_handoff(attention, monkeypatch, budget, depth, reason):
    from ouroboros.contracts.task_contract import attach_task_contract

    source = attention.meta["task"]
    source.update(depth=depth, delegation_budget=budget)
    attach_task_contract(source)
    monkeypatch.setattr("supervisor.events.get_max_subagent_depth", lambda: 3)
    monkeypatch.setattr(queue, "load_state", lambda: {"bg_consciousness_enabled": False})
    attention.tick()
    attention.flush()
    fact = observation(attention)
    sent = _handoff_diagnostic(attention, monkeypatch)
    print("DELEGATION_HANDOFF " + json.dumps({"budget": budget, "source_depth": depth,
          "expected_refusal": reason, "observation": fact, "sent": [task["id"] for task in sent]}))
    assert len(sent) == (0 if reason else 1)
    if reason:
        assert fact["attention_wait"] == reason
        _assert_delegation_refusal_stays_settled(attention, monkeypatch, reason)
    else:
        child = sent[0]
        assert child["task_constraint"]["mode"] == "local_readonly_subagent"
        rights = child["task_contract"]["delegation_budget"]
        assert rights["may_delegate"] is rights["may_mutate"] is rights["may_fan_out"] is False
        assert child["metadata"]["billing_group"]["billing_group_limit_revision"] == "original"
    assert attention.running["root"] is attention.meta and queue._reap_queue.empty()
    assert "finalization_requested_at" not in attention.meta


def _assert_delegation_refusal_stays_settled(attention, monkeypatch, reason):
    from ouroboros.config import BODY_MAINTENANCE_INTERVAL_SEC
    from ouroboros.owner_mailbox import drain_owner_entries
    from supervisor import silence_attention

    fact = observation(attention)
    refused = load_task_result(attention.root, fact["diagnostic_id"])
    assert refused["status"] == "failed" and refused["admission_outcome"] == "never_admitted"
    assert (refused.get("delegation_admission") or {}).get("status") != "accepted"
    assert refused["reason_code"] == reason
    assert refused["accounted_upper_bound_usd"] == 0
    assert not any(row["id"] == fact["diagnostic_id"] for row in queue.PENDING)
    attention.clock[0] += BODY_MAINTENANCE_INTERVAL_SEC
    attention.tick()
    attention.flush()
    fact = observation(attention)
    assert fact["diagnostic_status"] == "failed" and fact["diagnostic_reason"] == reason
    with monkeypatch.context() as patch:
        patch.setattr(silence_attention, "_publish", attention.forbidden)
        for _ in range(3):
            attention.clock[0] += BODY_MAINTENANCE_INTERVAL_SEC
            attention.tick()
            attention.flush()
    assert load_task_result(attention.root, fact["diagnostic_id"]) == refused
    mail = drain_owner_entries(attention.root, "root")
    assert len(mail) == 1 and mail[0]["kind"] == "task_message"
    assert attention.running["root"] is attention.meta and queue._reap_queue.empty()


@pytest.mark.parametrize("budget,reason", [
    ({"max_children": 1}, "delegation_rights_max_children"),
    ({"may_fan_out": False}, "delegation_rights_may_fan_out"),
])
@pytest.mark.parametrize("durable_only", [False, True])
def test_diagnostic_inherits_frozen_child_limits(attention, monkeypatch, budget, reason, durable_only):
    from ouroboros.contracts.task_contract import attach_task_contract

    source = attention.meta["task"]
    source["delegation_budget"] = budget
    attach_task_contract(source)
    if durable_only:
        contract = source.pop("task_contract")
        source["metadata"].pop("task_contract", None)
        write_task_result(attention.root, "root", "running", task_contract=contract)
    source["delegation_budget"] = {}  # A loose projection cannot replace the admitted contract.
    write_task_result(attention.root, "earlier-child", "completed", parent_task_id="root",
                      root_task_id="root", delegation_role="subagent")
    attention.tick()
    attention.flush()
    assert not _handoff_diagnostic(attention, monkeypatch)
    assert observation(attention)["attention_wait"] == reason
    _assert_delegation_refusal_stays_settled(attention, monkeypatch, reason)


@pytest.mark.parametrize("permitted,live,depth,allowed", [
    (4, 2, 3, True), (2, 4, 2, False), (4, 0, 2, False),
    (None, 3, 2, True), (None, 2, 2, False),
])
def test_diagnostic_depth_keeps_frozen_permission_and_current_global_zero(
        attention, monkeypatch, permitted, live, depth, allowed):
    from ouroboros.contracts.task_contract import attach_task_contract

    source = attention.meta["task"]
    source["depth"] = depth
    if permitted is not None:
        source["delegation_budget"] = {"depth_provenance": {"permitted_depth": permitted}}
    attach_task_contract(source)
    monkeypatch.setattr("supervisor.events.get_max_subagent_depth", lambda: live)
    attention.tick()
    attention.flush()
    sent = _handoff_diagnostic(attention, monkeypatch)
    assert bool(sent) is allowed
    if allowed:
        child = sent[0]
        provenance = child["task_contract"]["delegation_budget"]["depth_provenance"]
        assert provenance["permitted_depth"] == (live if permitted is None else permitted)
        assert provenance["attempted_depth"] == child["depth"] == depth + 1
    else:
        _assert_delegation_refusal_stays_settled(attention, monkeypatch, "subtask_depth_limit")


@pytest.mark.parametrize("early,final,limited", [(None, 0, True), (None, None, True),
                                                (0, None, True), (None, None, False)])
def test_diagnostic_unknown_child_count_is_decided_by_interlocked_recheck(
        attention, monkeypatch, early, final, limited):
    from ouroboros.contracts.task_contract import attach_task_contract
    from supervisor import events_schedule_task as delegation

    attention.meta["task"]["delegation_budget"] = {"max_children": 1} if limited else {}
    attach_task_contract(attention.meta["task"])
    reads = []

    def count(*args, **kwargs):
        assert not queue._queue_lock._is_owned()
        reads.append(bool(kwargs.get("propagate_contention")))
        return final if reads[-1] else early

    monkeypatch.setattr(delegation, "durable_direct_child_count", count)
    attention.tick()
    attention.flush()
    sent = _handoff_diagnostic(attention, monkeypatch)
    assert reads == [False, True]
    assert bool(sent) is (final == 0 or not limited)
    if final is None and limited:
        _assert_delegation_refusal_stays_settled(
            attention, monkeypatch, "delegation_rights_child_count_unknown")
    else:
        assert load_task_result(attention.root, sent[0]["id"])["delegation_admission"]["direct_child_count"] == final


@pytest.mark.parametrize("change,reason", [
    ("child", "delegation_rights_max_children"),
    ("legacy_depth", "subtask_depth_limit"),
    ("frozen_zero", "subtask_depth_limit"),
    ("frozen_nonzero", ""),
    ("hard_cap", "active_subagent_hard_cap"),
])
def test_diagnostic_rechecks_between_preparation_receipt_and_queue_put(attention, monkeypatch, change, reason):
    from ouroboros.contracts.task_contract import attach_task_contract
    from supervisor import events_schedule_task as delegation
    from supervisor import task_admission
    from supervisor import workspace_preparation as preparation

    source = attention.meta["task"]
    source.update(depth=2, delegation_budget={"max_children": 1})
    if change.startswith("frozen"):
        source["delegation_budget"]["depth_provenance"] = {"permitted_depth": 3}
    attach_task_contract(source)
    cap = [3]
    monkeypatch.setattr("supervisor.events.get_max_subagent_depth", lambda: cap[0])
    monkeypatch.setattr(delegation, "get_max_active_subagents_per_root", lambda: 1)
    real_accept, real_write = preparation.accept, task_admission.write_task_result
    boundaries = []

    def before_admission(*args, **kwargs):
        if change == "child":
            write_task_result(attention.root, "racing-child", "completed", parent_task_id="root",
                              root_task_id="root", delegation_role="subagent")
            boundaries.append("preparation")
        return real_accept(*args, **kwargs)

    def after_receipt(*args, **kwargs):
        result = real_write(*args, **kwargs)
        if args[2] == "scheduled" and args[1].startswith("silence-"):
            assert not queue._queue_lock._is_owned()
            boundaries.append("receipt")
            if change in {"legacy_depth", "frozen_zero", "frozen_nonzero"}:
                cap[0] = 0 if change == "frozen_zero" else 1
            if change == "hard_cap":
                with queue._queue_lock:
                    queue.PENDING.extend({"id": f"sibling-{n}", "parent_task_id": "root",
                                          "root_task_id": "root", "delegation_role": "subagent"}
                                         for n in range(delegation.MAX_ACTIVE_SUBAGENTS_HARD_CAP))
        return result

    monkeypatch.setattr(preparation, "accept", before_admission)
    monkeypatch.setattr(task_admission, "write_task_result", after_receipt)
    attention.tick()
    attention.flush()
    sent = _handoff_diagnostic(attention, monkeypatch)
    assert boundaries == (["preparation"] if change == "child" else ["receipt"])
    assert bool(sent) is (not reason)
    if reason:
        assert observation(attention)["attention_wait"] == reason
        if change == "hard_cap":
            queue.PENDING.clear()  # End sibling demand before observing the still-silent source again.
        _assert_delegation_refusal_stays_settled(attention, monkeypatch, reason)


@pytest.mark.parametrize("hard", [False, True])
def test_diagnostic_uses_existing_root_active_cap_without_new_capacity(attention, monkeypatch, hard):
    from supervisor import events_schedule_task as delegation

    monkeypatch.setattr(delegation, "get_max_active_subagents_per_root", lambda: 1)
    monkeypatch.setattr("ouroboros.config.get_max_active_subagents_per_root", lambda: 1)
    attention.tick()  # Observe silence before the competing children become active.
    siblings = [f"sibling-{n}" for n in range(delegation.MAX_ACTIVE_SUBAGENTS_HARD_CAP if hard else 1)]
    for task_id in siblings:
        attention.running[task_id] = {"task": {"id": task_id, "root_task_id": "root",
            "parent_task_id": "root", "delegation_role": "subagent"}, "attempt": 1,
            "started_at": attention.clock[0], "last_progress_at": attention.clock[0]}
    attention.flush()
    assert not _handoff_diagnostic(attention, monkeypatch)
    if hard:
        for task_id in siblings:
            attention.running.pop(task_id)
        _assert_delegation_refusal_stays_settled(attention, monkeypatch, "active_subagent_hard_cap")
    else:
        child = queue.PENDING[0]
        assert observation(attention)["attention_wait"] == "diagnostic_queued"
        attention.running.pop(siblings[0])
        sent = _handoff_diagnostic(attention, monkeypatch)
        assert len(sent) == 1 and sent[0]["id"] == child["id"]


def test_diagnostic_child_count_contention_defers_same_claim_on_admission_lane(attention, monkeypatch):
    from ouroboros.contracts.task_contract import attach_task_contract
    from ouroboros.store import StoreBusy
    from supervisor import events_schedule_task as delegation
    from supervisor import workspace_preparation as preparation

    attention.meta["task"]["delegation_budget"] = {"may_fan_out": False}
    attach_task_contract(attention.meta["task"])
    real_count = delegation.durable_direct_child_count
    tokens = []

    def count(*args, **kwargs):
        assert not queue._queue_lock._is_owned()
        if kwargs.get("propagate_contention"):
            child_id = kwargs["exclude_task_id"]
            row = load_task_result(attention.root, child_id)
            tokens.append(row["workspace_preparation"]["token"])
            if len(tokens) == 1:
                raise StoreBusy("controlled count contention", reason="contention")
        return real_count(*args, **kwargs)

    monkeypatch.setattr(delegation, "durable_direct_child_count", count)
    generation = preparation.PreparationGeneration()
    with monkeypatch.context() as patch:
        patch.setattr(preparation, "current_generation", lambda: generation)
        try:
            attention.tick()
            assert generation.lane_idle(preparation.ADMISSION_LANE, 15)
        finally:
            generation.close()
            assert generation.lane_idle(preparation.ADMISSION_LANE, 15)
    assert len(tokens) == 2 and tokens[0] == tokens[1]
    sent = _handoff_diagnostic(attention, monkeypatch)
    assert len(sent) == 1 and queue.PENDING == []


def _durable_only_budget(attention, budget, earlier):
    """The live source carries no contract; its canonical rights live in its durable row."""
    from ouroboros.contracts.task_contract import attach_task_contract

    source = attention.meta["task"]
    source["delegation_budget"] = budget
    attach_task_contract(source)
    contract = source.pop("task_contract")
    source["metadata"].pop("task_contract", None)
    source["delegation_budget"] = {}
    write_task_result(attention.root, "root", "running", task_contract=contract)
    for n in range(earlier):
        write_task_result(attention.root, f"earlier-{n}", "completed", parent_task_id="root",
                          root_task_id="root", delegation_role="subagent")


@pytest.mark.parametrize("error", ["contention", "unreadable"])
@pytest.mark.parametrize("budget,earlier,reason", [
    ({"may_delegate": False}, 0, "delegation_rights_may_delegate"),
    ({"max_children": 1}, 1, "delegation_rights_max_children"),
    ({"max_children": 2}, 1, ""),
], ids=["no-delegation", "children-exhausted", "permitted-control"])
def test_unreadable_parent_budget_defers_the_same_episode_until_it_reads(
        attention, monkeypatch, error, budget, earlier, reason):
    """A failed canonical read is not the omitted legacy budget: nothing is frozen or
    recorded, and the existing cadence rereads the same unsent episode's authority."""
    from ouroboros.config import BODY_MAINTENANCE_INTERVAL_SEC
    from ouroboros.store import StoreBusy, StoreError
    from supervisor import events

    _durable_only_budget(attention, budget, earlier)
    real_read, reads = events.load_task_result, []

    def gap(root, task_id, *args, **kwargs):
        reads.append((str(task_id), kwargs.get("strict")))
        if str(task_id) == "root" and len(reads) == 1:
            raise (StoreBusy("one parent-budget read contention", reason="contention") if error == "contention"
                   else StoreError("parent budget row unreadable"))
        return real_read(root, task_id, *args, **kwargs)

    monkeypatch.setattr(events, "load_task_result", gap)
    attention.tick()
    attention.flush()
    fact = observation(attention)
    assert fact["attention_wait"] == "delegation_authority_unavailable" and fact["state"] == "pending_attention"
    assert reads == [("root", True)] and queue.PENDING == []
    assert load_task_result(attention.root, fact["diagnostic_id"]) is None  # no receipt, refusal or rights
    assert not _handoff_diagnostic(attention, monkeypatch)
    attention.clock[0] += BODY_MAINTENANCE_INTERVAL_SEC
    attention.tick()
    attention.flush()
    assert observation(attention)["episode_id"] == fact["episode_id"] and ("root", True) in reads[1:]
    sent = _handoff_diagnostic(attention, monkeypatch)
    print("PARENT_BUDGET_RECOVERY " + json.dumps({"error": error, "budget": budget, "reads": reads,
          "observation": observation(attention), "sent": [row["id"] for row in sent]}))
    if reason:
        assert not sent and observation(attention)["attention_wait"] == reason
        _assert_delegation_refusal_stays_settled(attention, monkeypatch, reason)
    else:
        rights = sent[0]["task_contract"]["delegation_budget"]
        assert len(sent) == 1 and rights["may_delegate"] is rights["may_fan_out"] is False


def test_parent_budget_reader_keeps_fail_soft_callers_and_strict_admission(attention, monkeypatch):
    from ouroboros.contracts.task_contract import attach_task_contract
    from ouroboros.store import StoreBusy
    from supervisor import events

    ctx = SimpleNamespace(RUNNING={}, DRIVE_ROOT=attention.root)
    parent = attach_task_contract({"id": "parent", "text": "work", "delegation_budget": {"may_delegate": False}})
    write_task_result(attention.root, "parent", "running", task_contract=parent["task_contract"])
    assert events._parent_delegation_budget(ctx, "parent", attention.root, strict=True)["may_delegate"] is False
    assert events._parent_delegation_budget(ctx, "absent", attention.root, strict=True) == {}  # confirmed absence
    monkeypatch.setattr(events, "load_task_result",
                        lambda *a, **k: (_ for _ in ()).throw(StoreBusy("busy", reason="contention")))
    assert events._parent_delegation_budget(ctx, "parent", attention.root) == {}  # schedule_subagent's own read
    with pytest.raises(StoreBusy):
        events._parent_delegation_budget(ctx, "parent", attention.root, strict=True)


@pytest.mark.parametrize("moment", ["before_first_send", "after_source_progress"])
def test_revoked_diagnostic_ends_through_single_scope_cancellation(attention, monkeypatch, moment):
    """The real worker boundary: no model call, no task exception, and only the
    diagnostic's own single-scope intent; the source keeps every control."""
    from ouroboros.agent import Env, OuroborosAgent
    from ouroboros.cancel_intents import active_intent

    child = _mint(attention)
    repo = attention.root / "repo"
    repo.mkdir(exist_ok=True)
    monkeypatch.setattr("ouroboros.utils.in_worker_process", lambda: True)  # a pooled worker holds this slot
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr("ouroboros.mcp_client.ensure_configured_from_settings", attention.forbidden)
    agent = OuroborosAgent(Env(repo_dir=repo, drive_root=attention.root))

    def provider(*_args, **_kwargs):
        # Every provider client admits the physical window before request bytes
        # (``llm_attempt._execute_candidate``); this double then forbids the send.
        from ouroboros._usage_wait import retry_presend
        from ouroboros.llm_attempt import require_physical_dispatch_window

        retry_presend(require_physical_dispatch_window, check=False)
        return attention.forbidden()

    monkeypatch.setattr(agent.llm, "chat", provider)
    monkeypatch.setattr(agent, "_run_delegate_preflight", lambda logs, task, dispatch: (dispatch, False))

    def progressed():
        attention.meta["last_progress_at"] = attention.clock[0]
        assert queue.persist_queue_snapshot(reason="source_progress_before_send")

    def messages(**kwargs):
        if moment == "after_source_progress":
            progressed()  # after the start check passed, before the first send
        return [{"role": "user", "content": "Inspect the silent source."}], {}

    monkeypatch.setattr("ouroboros.agent.build_llm_messages", messages)
    if moment == "before_first_send":
        progressed()
    events = agent.handle_task(dict(child))
    intent = active_intent(attention.root, child["id"])
    print("REVOKED_DIAGNOSTIC " + json.dumps({"moment": moment, "events": events, "intent": intent}, default=str))
    assert events == []
    assert intent["source"] == "silence_resolved" and intent["scope"] == "single"
    assert active_intent(attention.root, "root") is None
    errors = (attention.root / "logs" / "events.jsonl")
    assert not errors.exists() or "task_error" not in errors.read_text(encoding="utf-8")
    assert (load_task_result(attention.root, child["id"]) or {}).get("status") != "failed"
    assert attention.running["root"] is attention.meta and queue._reap_queue.empty()
