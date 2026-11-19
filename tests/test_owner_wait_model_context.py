"""A cold owner wait keeps existing model choices and execution-clock meaning."""

import copy
from dataclasses import replace
import json
import queue as stdqueue
import threading
import time
from types import SimpleNamespace

from tests.worker_pool_support import affirm_ready

import pytest

from ouroboros import agent as agent_module, context, model_wait, owner_wait, task_pacing
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.task_results import write_task_result
from ouroboros.tools.registry import ToolRegistry
from supervisor import queue, task_model_wait, workers
from tests._context_shared import _make_health_env
from tests.test_context_fit_v664 import _plan
from tests.test_llm_claudexor import MODEL
from tests.test_loop_transport_wait import _loop_kwargs


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Block provider/catalog calls without interfering with local event-loop IPC."""
    attempted = []

    def refuse(*args, **kwargs):
        attempted.append((args, kwargs))
        raise AssertionError("owner-wait model-state fixture attempted a provider/catalog call")

    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", refuse)
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat_async", refuse)
    monkeypatch.setattr("ouroboros.pricing._fetch_live_rows", refuse)
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_TASK_ABS_CEILING_SEC", "21600")
    yield
    assert not attempted


@pytest.fixture
def saved_wait(tmp_path, monkeypatch):
    now = time.time()
    clock = {"mono": 1000.0, "wall": now - 22000}
    monkeypatch.setattr(model_wait, "time", SimpleNamespace(
        monotonic=lambda: clock["mono"], time=lambda: clock["wall"], sleep=time.sleep))
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id, ctx.task_attempt, ctx.task_started_at = "t-wait", 1, clock["wall"]
    ctx.active_model, ctx.active_effort = MODEL, "high"
    ctx.active_use_local, ctx.active_context_mode = False, "max"
    ctx._cost_ceiling = task_pacing.CostCeiling(state="disabled")
    ctx._owner_wait_requested = "question"
    ctx.context_fit_plan = replace(_plan(window=500000, known=True), model=MODEL,
                                   model_role="fallback:0", model_route={"credentialProfileId": "observed-only"})
    task = {"id": ctx.task_id, "type": "task", "chat_id": 1, "_attempt": 1, "text": "continue"}
    write_task_result(tmp_path, ctx.task_id, "running")
    events = stdqueue.Queue()
    with model_wait.task_model_wait_scope(task=task, drive_root=tmp_path, event_queue=events,
                                          worker_slot_held=True) as controller:
        ctx.model_wait_context = controller
        controller.tool_context = ctx
        controller.overrides = {
            "fallback:0": {"model": MODEL, "use_local": False, "model_account_override": "selected"},
            "vision": {"model": "vision-selected", "use_local": True, "model_account_override": ""},
        }
        controller.auto_continue = {"fallback:0": False, "vision": True}
        row = {"wait_id": "quota", "task_attempt": 1, "role": "main", "state": "waiting",
               "reason": "quota", "worker_slot_held": True}
        controller.quota_enter("quota", "slot-a")
        controller._publish(row)
        clock.update(mono=19000, wall=now - 4000)
        controller.quota_leave("quota", "slot-a")
        row["state"] = "resolved"
        controller._publish(row)
        clock.update(mono=23000, wall=now)
        original_state = controller.continuation_state()
        wait = owner_wait.checkpoint_owner_wait(ctx, ctx.context_fit_plan.messages_for("max"),
                                                {"tool_calls": []}, {"cost": 2.0}, 1, [], set())
        owner_wait.set_owner_wait(tmp_path, ctx.task_id, {**wait, "state": "waiting"})
        task["_owner_wait_resume"] = {**wait, "restart_transaction_id": "tx"}
        from ouroboros.delegate_recovery import _write_restart_transaction

        _write_restart_transaction(tmp_path, {"transaction_id": "tx", "status": "normal_exit_acknowledged",
                                             "task_ids": [ctx.task_id]})
        yield SimpleNamespace(root=tmp_path, task=task, ctx=ctx, wait=wait, state=original_state,
                              controller=controller, clock=clock, events=events)


def test_clock_state_preserves_quota_union_origin_and_calendar_deadline(saved_wait):
    case = saved_wait
    assert owner_wait.restore_owner_wait_allowed(case.root, case.task)
    assert case.wait["model_wait_quota_clock"]["elapsed_sec"] == 18000
    assert model_wait.quota_waited_seconds is task_model_wait.quota_waited_seconds
    # Time spent waiting for the owner (or restarting) is still execution time.
    case.clock["mono"] += 120
    case.clock["wall"] += 120
    with model_wait.task_model_wait_scope(task=case.task, drive_root=case.root, event_queue=None,
                                          worker_slot_held=True) as fresh:
        fresh.restore_continuation(case.state, started_at=case.ctx.task_started_at)
        assert fresh.paused_seconds() == 18000
        assert fresh.execution_window_remaining() == pytest.approx(17480)
        assert fresh.execution_window_remaining() == case.controller.execution_window_remaining()
        assert fresh.worker_slot_held is True and fresh.control_reason() is None
        assert fresh.overrides == case.controller.overrides
        assert fresh.auto_continue == case.controller.auto_continue
        with model_wait.calendar_scope("2000-01-01T00:00:00+00:00"):
            assert fresh.control_reason() == "deadline"


def test_actual_assignment_and_later_quota_publication_keep_one_clock(saved_wait, monkeypatch):
    from supervisor import evolution_lifecycle, state

    case = saved_wait
    pending, running = [case.task], {}
    for module in (workers, queue):
        monkeypatch.setattr(module, "DRIVE_ROOT", case.root)
        monkeypatch.setattr(module, "PENDING", pending)
        monkeypatch.setattr(module, "RUNNING", running)
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", case.root / "state/queue_snapshot.json")
    monkeypatch.setattr(queue, "ACCEPTANCE_FENCES", {})
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr(workers, "load_state", lambda: {})
    monkeypatch.setattr(workers, "repo_writer_task_allowed", lambda task: True)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")
    worker = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False, active_capacity=True,
                             in_q=stdqueue.Queue()))
    monkeypatch.setattr(workers, "WORKERS", {0: worker})
    workers.assign_tasks()
    meta = running["t-wait"]
    assert task_model_wait.quota_waited_seconds(meta, time.time()) == 18000
    assert meta["started_at"] == case.ctx.task_started_at
    reaps = stdqueue.Queue()
    monkeypatch.setattr(queue, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(queue, "_reap_queue", reaps)
    queue._enforce_task_timeouts_locked(workers, time.time(), 0, {})
    assert reaps.empty() and "t-wait" in running
    events = stdqueue.Queue()
    host = SimpleNamespace(RUNNING=running, DRIVE_ROOT=case.root,
                           append_jsonl=lambda *a: None, bridge=SimpleNamespace(push_log=lambda row: None))
    with model_wait.task_model_wait_scope(task=case.task, drive_root=case.root, event_queue=events,
                                          worker_slot_held=True) as fresh:
        fresh.restore_continuation(case.state, started_at=case.ctx.task_started_at)
        fresh.quota_enter("new-quota", "slot-new")
        row = {"wait_id": "new-quota", "task_attempt": 1, "role": "main", "state": "waiting"}
        fresh._publish(row)
        task_model_wait.handle_task_model_wait(events.get_nowait(), host)
        case.clock["mono"] += 10
        case.clock["wall"] += 10
        fresh.quota_leave("new-quota", "slot-new")
        row["state"] = "resolved"
        fresh._publish(row)
        task_model_wait.handle_task_model_wait(events.get_nowait(), host)
        assert meta["model_wait_quota_clock"]["revision"] == case.state["quota_clock"]["revision"] + 2
        assert task_model_wait.quota_waited_seconds(meta, case.clock["wall"]) == 18010


@pytest.mark.parametrize("pin", ["selected", ""])
def test_agent_restores_once_before_runtime_and_cold_dispatch_keeps_role_and_pin(saved_wait, monkeypatch, pin):
    from ouroboros import loop

    case = saved_wait
    case.controller.overrides["fallback:0"]["model_account_override"] = pin
    wait = owner_wait.checkpoint_owner_wait(case.ctx, case.ctx.context_fit_plan.messages_for("max"),
                                            {"tool_calls": []}, {"cost": 2.0}, 1, [], set())
    owner_wait.set_owner_wait(case.root, "t-wait", {**wait, "state": "waiting"})
    task = {**case.task, "_owner_wait_resume": {**wait, "restart_transaction_id": "tx"}}
    env = _make_health_env(case.root)
    env.branch_dev, env.budget_drive_root = "ouroboros", case.root
    registry = ToolRegistry(repo_dir=env.repo_dir, drive_root=case.root)
    agent = object.__new__(agent_module.OuroborosAgent)
    agent.__dict__.update(env=env, tools=registry, memory=None, _event_queue=None,
                          _current_chat_id=1, _current_task_type="task", _pending_events=[],
                          _task_started_ts=case.ctx.task_started_at, _owner_message_admission_lock=threading.Lock(),
                          owner_wait_callback=lambda *_: None)
    for name in ("_emit_live_log", "_emit_typing_start", "_emit_progress", "_capture_mutation_baseline"):
        monkeypatch.setattr(agent, name, lambda *a, **kw: None)
    monkeypatch.setattr(agent, "_run_delegate_preflight", lambda _logs, _task, dispatch: (dispatch, False))
    rebinds, dispatches = [], []

    def route(task, **kwargs):
        rebinds.append(copy.deepcopy(task))
        assert task["model_role"] == "fallback:0" and task["credential_profile_id"] == pin
        assert task["model_route"] == {}  # Startup/observed accounts are never new pins.
        return {"model": MODEL, "provider": "claudexor"}, SimpleNamespace(
            status="confirmed", stale=False, window_tokens=500000, route_fp="fresh-route",
            source_id="codex", credential_profile_id=pin or "fresh-auto-account",
            account_fingerprint="fresh-fingerprint", source="fixture")

    monkeypatch.setattr(context, "_context_fit_route", route)

    def runtime_messages(env, task, ctx, **kwargs):
        controller = ctx.model_wait_context
        assert controller.overrides == case.controller.overrides
        assert controller.auto_continue["fallback:0"] is False
        assert controller.paused_seconds() == 18000
        runtime = context.build_runtime_section(env, task, ctx=ctx)
        startup = replace(_plan(window=500000, known=True), model=MODEL, model_role="main",
                          model_route={"source": "codex", "model": "exact-model",
                                       "credentialProfileId": "wrong-startup-account", "accountFingerprint": "wrong"})
        projection = replace(startup.max_projection, system_content_json=json.dumps(runtime))
        ctx.context_fit_plan = replace(startup, max_projection=projection, low_projection=projection)
        return ctx.context_fit_plan.messages_for("max"), {}

    monkeypatch.setattr(agent_module, "build_llm_messages", runtime_messages)

    def response(*args, **kwargs):
        dispatches.append(kwargs)
        assert kwargs["model_role"] == "fallback:0" and kwargs["model_account_override"] == pin
        assert model_wait.current_model_wait().auto_continue["fallback:0"] is True
        assert model_wait.current_model_wait().overrides["vision"]["model"] == "changed-after-runtime"
        return {"role": "assistant", "content": "continued", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", response)
    with model_wait.task_model_wait_scope(task=task, drive_root=case.root, event_queue=None,
                                          worker_slot_held=True) as fresh:
        ctx, messages, caps = agent._prepare_task_context(task)
        assert fresh.tool_context is ctx and ctx.model_wait_context is fresh
        fresh.auto_continue["fallback:0"] = True
        fresh.overrides["vision"]["model"] = "changed-after-runtime"
        result, _, _ = loop.run_llm_loop(**{**_loop_kwargs(case.root, registry, []),
            "messages": messages, "drive_logs": case.root / "logs", "budget_remaining_usd": caps["budget_remaining"]})
        assert result == "continued" and len(dispatches) == 1 and rebinds
        assert ctx.context_fit_plan.model_role == "fallback:0"
        assert fresh.paused_seconds() == 18000
    saved = json.loads(read_actor_source_bytes(case.root, "t-wait", wait["source_ref"]))
    assert saved["model_wait"]["auto_continue"]["fallback:0"] is False
    assert saved["model_wait"]["overrides"]["vision"]["model"] == "vision-selected"


_LADDER_TOOLS = [{"type": "function", "function": {"name": f"tool_{i}", "description": "Read one fixture.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "A path."}},
                   "required": ["path"]}}} for i in range(60)]  # direct OpenAI reasoning sends a longer catalog
_WHOLE_EDGE: dict = {}  # the searched window by its exact capture; every case proves it again in its own world


@pytest.mark.parametrize("carrier,acting,pending", [
    (carrier, acting, pending)
    for carrier in ("owner", "exact_tool", "exact_no_tool", "working_post_batch", "working_ready",
                    "working_resumed_tool", "working_resumed_no_tool")
    for acting, pending in (("high", "none"), ("none", "high"))
] + [("owner", "none", None), ("owner", "high", None), ("legacy", "high", None), ("fresh", "high", None)])
def test_a_cold_continuation_fits_its_start_in_the_effort_it_resumes_on(tmp_path, monkeypatch, carrier, acting, pending):
    """A restored task fits the next ordinary send's selected effort, including a pending override.
    Budget tails keep the old acting route until that send; memory is not folded for a different request shape."""
    from ouroboros import budget_pause, capability_evidence as ce, context_fit as cf, loop, loop_routes, memory_view as mv
    from ouroboros import working_checkpoint as wc
    from ouroboros.delegate_recovery import _write_restart_transaction
    from tests import _memory_view_synthetic as syn

    sol, window = "openai::gpt-6-sol", [0]

    def route(*_a, **_kw):  # Main is direct OpenAI: its reasoning sends the custom ladder
        return {"provider": "openai", "model": sol}, ce.CapabilityEvidence(window[0], "confirmed", "metadata", "resume")

    monkeypatch.setenv("OUROBOROS_MODEL", sol)
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path / "evidence")
    monkeypatch.setattr(cf, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    monkeypatch.setattr(context, "_context_fit_route", route)
    snap = syn.snapshot(room_facts=syn.room("1"), story=syn.pages(1, 300),
                        conversations=[syn.conversation(i, words=10, word_chars=1_000) for i in range(10)])
    book = "# Architecture\n\n" + "Whole source book. " * 2_000
    core = cf.ContextCore("Core", "BIBLE", book, "", "Identity", "Facts", '"Assignment"', False,
                          memory_view_json=mv.snapshot_json(snap))
    task = {"id": "t-wait", "type": "task", "chat_id": 1, "_attempt": 1, "text": "continue", "reasoning_effort": "high"}

    def fit(effort, resolver=route):  # the start's fit over this one capture
        return cf.build_context_fit_plan(None, core, task, preferred_mode="max", tool_schemas=_LADDER_TOOLS,
                                         reasoning_effort=effort, route_resolver=resolver)

    def whole(plan):
        return plan.initial_mode == "max" and not any(plan.max_projection.memory_facts["floor"]["steps"].values())

    def holds(tokens):
        window[0] = tokens
        return whole(fit("none"))

    key = json.dumps([vars(core), task, _LADDER_TOOLS, sol])  # exact fit inputs; the route and none effort are fixed
    edge = _WHOLE_EDGE.get(key)
    if edge is None or holds(edge - 1) or not holds(edge):  # a known edge is proved again in this case's own world
        low, high = 50_000, 400_000  # the first window holding the whole memory in the effort it resumes on
        while low < high:
            middle = (low + high) // 2
            low, high = (low, middle) if holds(middle) else (middle + 1, high)
        _WHOLE_EDGE[key] = edge = high
    assert not holds(edge - 1) and holds(edge)  # also prove a newly searched edge in this case's own world
    window[0] = edge
    ideal, started = fit("none"), fit("high")
    assert whole(ideal) and not whole(started)  # its start effort's longer ladder folds memory in this window
    before = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)._ctx  # the process that moved high -> none
    before.task_id, before.task_attempt, before.task_started_at = "t-wait", 1, time.time()
    before.active_model, before.active_effort, before.active_use_local = sol, acting, False
    if pending:
        from ouroboros.tools.control_runtime import _switch_model
        monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
        monkeypatch.setattr("ouroboros.llm.LLMClient.available_models", lambda _self: [sol])
        answer = _switch_model(before, effort=pending)
        assert before.active_effort == acting and before.active_effort_override == pending, answer
    before.active_context_mode, before.context_fit_plan = "max", started
    before._cost_ceiling = task_pacing.CostCeiling(state="disabled")
    write_task_result(tmp_path, "t-wait", "running")
    wait = owner_wait.checkpoint_owner_wait(before, started.messages_for("max"), {"tool_calls": []}, {"cost": 0.0}, 3,
                                            _LADDER_TOOLS, set())
    state = json.loads(read_actor_source_bytes(tmp_path, "t-wait", wait["source_ref"]))
    if carrier == "legacy":
        state["route"].pop("active_effort_override")
        wait["source_ref"] = owner_wait.store_continuation_source(before, state, "legacy")
    if carrier in {"owner", "legacy"}:
        owner_wait.set_owner_wait(tmp_path, "t-wait", {**wait, "state": "waiting"})
        _write_restart_transaction(tmp_path, {"transaction_id": "tx", "status": "normal_exit_acknowledged",
                                              "task_ids": ["t-wait"]})
        task["_owner_wait_resume"] = {**wait, "restart_transaction_id": "tx"}
    elif carrier.startswith("exact_"):
        state.update(pause_id="pause", resume_point={"budget_tail": carrier.removeprefix("exact_")})
        source = owner_wait.store_continuation_source(before, state, "exact")
        grant = {"grant_id": "grant", "generation": 1, "granted_at_ts": time.time()}
        row = {"pause_id": "pause", "pause_generation": 1, "task_attempt": 1, "source_ref": source,
               "state": budget_pause.STATE_RESUME_GRANTED, "grant": grant, "rail": "global",
               "resume_point": state["resume_point"]}
        budget_pause.set_budget_pause(tmp_path, "t-wait", row)
        task["_budget_pause_resume"] = {**row, "grant_id": "grant", "grant_generation": 1,
                                        "started_at": before.task_started_at}
    elif carrier.startswith("working_"):
        boundary = "resumed" if "resumed" in carrier else carrier.removeprefix("working_")
        state["working"] = {"boundary": boundary, "budget_tail": "no_tool" if carrier.endswith("no_tool") else "tool"}
        source = owner_wait.store_continuation_source(before, state, "working")
        task["_working_recovery"] = {"source_ref": source, "source_task_id": "t-wait", "from_attempt": 1,
                                     "source_kind": wc.SOURCE_WORKING, "boundary": boundary,
                                     "started_at": before.task_started_at}
    env = _make_health_env(tmp_path)
    env.branch_dev, env.budget_drive_root = "ouroboros", tmp_path
    registry = ToolRegistry(repo_dir=env.repo_dir, drive_root=tmp_path)
    agent = object.__new__(agent_module.OuroborosAgent)
    agent.__dict__.update(env=env, tools=registry, memory=None, _event_queue=None, _current_chat_id=1,
                          _current_task_type="task", _pending_events=[], _task_started_ts=before.task_started_at,
                          _owner_message_admission_lock=threading.Lock(), owner_wait_callback=lambda *_: None)
    for name in ("_emit_live_log", "_emit_typing_start", "_emit_progress", "_capture_mutation_baseline"):
        monkeypatch.setattr(agent, name, lambda *a, **kw: None)
    monkeypatch.setattr(agent, "_run_delegate_preflight", lambda _logs, _task, dispatch: (dispatch, False))
    fitted, rebound, tails, sent, ordinary_rebinds = [], [], [], [], []
    rebind, finish_tail = loop._rebind_context_fit_plan, loop._finish_tool_round_budget
    ordinary = loop_routes._apply_round_route_overrides

    def restored_fit(*args, **kwargs):
        rebound.append(kwargs.get("reasoning_effort", getattr(args[1]._ctx, "active_effort", "")))
        return rebind(*args, **kwargs)

    def ordinary_fit(*args, **kwargs):
        before = len(rebound)
        result = ordinary(*args, **kwargs)
        ordinary_rebinds.append(len(rebound) - before)
        return result

    def tool_tail(call, *args, **kwargs):
        tails.append((call.active_effort, call.tools._ctx.active_effort_override))
        return finish_tail(call, *args, **kwargs)

    def start_fit(*, ctx, reasoning_effort, route_resolver, **_kw):  # the build over this capture, its effort given
        fitted.append(reasoning_effort)
        ctx.context_fit_plan = fit(reasoning_effort, route_resolver)
        return ctx.context_fit_plan.messages_for(ctx.context_fit_plan.initial_mode), {}

    def dispatch(*args, **_kw):
        sent.append((copy.deepcopy(args[1]), args[4]))  # the request's messages and effort
        return {"role": "assistant", "content": "continued", "tool_calls": []}, 0.0

    monkeypatch.setattr(agent_module, "build_llm_messages", start_fit)
    monkeypatch.setattr(loop, "call_llm_with_retry", dispatch)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", restored_fit)
    monkeypatch.setattr(loop_routes, "_apply_round_route_overrides", ordinary_fit)
    monkeypatch.setattr(loop, "_finish_tool_round_budget", tool_tail)
    main = SimpleNamespace(default_model=lambda: sol)  # the route the restored task follows stays its acting one
    with model_wait.task_model_wait_scope(task=task, drive_root=tmp_path, event_queue=None, worker_slot_held=True):
        ctx, messages, caps = agent._prepare_task_context(task)
        loaded = agent_module._load_start_continuation(ctx, task)
        assert (loaded.get("route") or {}).get("active_effort_override") == pending
        kwargs = {**_loop_kwargs(tmp_path, registry, [], main), "messages": messages, "drive_logs": tmp_path / "logs",
                  "initial_effort": agent_module._initial_effort_for(task, "task")}
        result, _, _ = loop.run_llm_loop(**kwargs, budget_remaining_usd=caps["budget_remaining"])
    [(request, effort)] = sent  # fitted in the effort it resumes on, never the one the task started on: memory whole
    expected = pending or acting
    print("COLD_EFFORT", json.dumps({"carrier": carrier, "acting": acting, "pending": pending, "fitted": fitted,
        "rebound": rebound, "ordinary_rebinds": ordinary_rebinds, "tails": tails, "sent": effort,
        "steps": ctx.context_fit_plan.max_projection.memory_facts["floor"]["steps"]}))
    assert (result, fitted, effort) == ("continued", [expected], expected)
    assert ordinary_rebinds and not any(ordinary_rebinds)  # the pending shape was already fitted on restore
    if carrier != "fresh":
        assert rebound and all(value == expected for value in rebound)
    if carrier in {"owner", "legacy", "exact_tool", "working_post_batch", "working_resumed_tool"}:
        assert tails[0] == (acting, pending)
    else:
        assert not tails
    assert ctx.active_effort_override is None
    if expected == "none":
        assert whole(ctx.context_fit_plan)
        assert request[0]["content"] == ideal.projection("max").system_message()["content"]


def test_the_active_turn_token_never_enters_a_cold_owner_wait_checkpoint(saved_wait):
    """A live transport turn is process state; a cold restart opens a new one."""
    from ouroboros.llm_claudexor import ModelTurnState

    case = saved_wait
    token = "opaque-turn-in-flight"
    case.ctx.model_turn_state = ModelTurnState(
        {"route": {}, "format": "codex.turn.v1", "payload": {"turnState": token}})
    case.ctx._owner_wait_requested = "question"
    wait = owner_wait.checkpoint_owner_wait(
        case.ctx, case.ctx.context_fit_plan.messages_for("max"), {"tool_calls": []}, {"cost": 2.0}, 1, [], set())
    stored = read_actor_source_bytes(case.root, case.ctx.task_id, wait["source_ref"]).decode("utf-8")
    assert token not in json.dumps(wait) and token not in stored
    assert "model_turn_state" not in stored


@pytest.mark.parametrize('decision', ['saved', 'pending', 'new_lead', 'new_return', 'pending_forced',
                                      'pending_local_forced', 'pending_priced'])
def test_saved_model_first_fit_and_real_send(tmp_path, monkeypatch, decision):
    """A wider saved route fits its own window; a later owner lead/return still wins. A forced final in the restored
    tail (``pending_forced``: the deadline rail) sends on the acting route, measured and fitted to it; the switch stays
    pending until an ordinary round consumes it. ``pending_local_forced``: the tail acts on the local server of the
    very model whose cloud route the switch chose, so the same id is fitted again to the local window.
    ``pending_priced``: the tail fitted the acting route to price its wrap-up, then admitted ordinary work; that
    round still spends the switch and shows the wider route's whole view."""
    from ouroboros import capability_evidence as ce, context_fit as cf, loop, memory_view as mv
    from ouroboros.delegate_recovery import _write_restart_transaction
    from ouroboros.tools.control_runtime import _switch_model
    from tests import _memory_view_synthetic as syn

    small, wide, latest = 'openai::gpt-6-sol', 'openai::gpt-6-astra', 'openai::gpt-6-terra'
    limits = {small: 100000, wide: 500000, latest: 750000}
    local = decision == 'pending_local_forced'  # that model's local server answers in 100k
    acting = wide if local else small
    calls = []

    monkeypatch.setattr('ouroboros.llm.LLMClient.available_models', lambda _: [small, wide, latest])
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'cyber_pro')
    monkeypatch.setenv('OUROBOROS_MODEL', small)
    monkeypatch.setenv('OUROBOROS_CONTEXT_MODE', 'max')
    monkeypatch.setenv('OUROBOROS_TASK_REVIEW_MODE', 'off')
    monkeypatch.setattr(ce, 'canonical_evidence_root', lambda: tmp_path/'evidence')
    monkeypatch.setattr(cf, '_route_calibration_ratio', lambda *_a, **_kw: 1.0)

    def route(task, **kw):
        chosen, _ = cf._context_route(task)
        window = 100000 if chosen['use_local'] else limits[chosen['model']]
        calls.append({'model': chosen['model'], 'role': chosen['model_role'], 'window': window})
        return chosen, ce.CapabilityEvidence(window, 'confirmed', 'metadata',
                                             chosen['model'] + (' (local)' if chosen['use_local'] else ''))

    monkeypatch.setattr(context, '_context_fit_route', route)
    snapshot = syn.snapshot(room_facts=syn.room('1'), story=syn.pages(1, 300),
        conversations=[syn.conversation(i, words=10, word_chars=1000) for i in range(10)])
    core = cf.ContextCore('Core', 'BIBLE', '# Architecture\n' + 'Whole source book. '*2000,
        '', 'Identity', 'Facts', '"Assignment"', False, memory_view_json=mv.snapshot_json(snapshot))
    from datetime import datetime, timezone

    task = {'id': 'model-witness', 'type': 'task', 'chat_id': 1, '_attempt': 1, 'text': 'continue',
            'reasoning_effort': 'none', **({'deadline_at': datetime.fromtimestamp(time.time() + 60, timezone.utc)
                                            .isoformat()} if decision.endswith('_forced') else {})}  # in the grace

    def fit(task, effort='none', resolver=route):
        return cf.build_context_fit_plan(None, core, task, preferred_mode='max', tool_schemas=[],
            reasoning_effort=effort, route_resolver=resolver)

    def view(plan):
        return {'mode': plan.initial_mode, 'steps': plan.projection(plan.initial_mode).memory_facts['floor']['steps']}

    old = fit({**task, 'model': acting, 'use_local_model': True} if local else task)
    expected_model = latest if decision == 'new_lead' else small if decision == 'new_return' else wide
    ideal = fit({**task, 'model': expected_model})
    before = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)._ctx
    before.task_id, before.task_attempt, before.task_started_at = task['id'], 1, time.time()
    before.active_model, before.active_effort, before.active_use_local = acting, 'none', local
    before.primary_route = {'model': acting, 'use_local': local, 'role': 'main'}
    before.route_choice = {'kind': 'following', 'role': 'main'}
    answer = _switch_model(before, model=wide)
    assert (before.active_model_override, before.active_use_local_override) == (wide, False), answer
    if not decision.startswith('pending'):
        before.active_model, before.active_use_local, before.active_effort = loop._apply_runtime_overrides(
            before, small, False, 'none')
    before.active_context_mode, before.context_fit_plan = old.initial_mode, old
    before._cost_ceiling = task_pacing.CostCeiling(state='disabled')
    write_task_result(tmp_path, task['id'], 'running')
    wait = owner_wait.checkpoint_owner_wait(before, old.messages_for(old.initial_mode), {'tool_calls': []},
        {'cost': 0.0}, 3, [], set())
    owner_wait.set_owner_wait(tmp_path, task['id'], {**wait, 'state': 'waiting'})
    _write_restart_transaction(tmp_path, {'transaction_id':'tx', 'status':'normal_exit_acknowledged', 'task_ids':[task['id']]})
    task['_owner_wait_resume'] = {**wait, 'restart_transaction_id':'tx'}
    if decision.startswith('new_'):
        write_task_result(tmp_path, task['id'], 'running',
            metadata={'lead': {'source':'owner', 'model':latest, 'use_local':False} if decision == 'new_lead' else {'source':'main'}},
            owner_model={'request_id':'later', 'action':'set' if decision == 'new_lead' else 'return'})
    env = _make_health_env(tmp_path)
    env.branch_dev, env.budget_drive_root = 'ouroboros', tmp_path
    registry = ToolRegistry(repo_dir=env.repo_dir, drive_root=tmp_path)
    agent = object.__new__(agent_module.OuroborosAgent)
    agent.__dict__.update(env=env, tools=registry, memory=None, _event_queue=None, _current_chat_id=1,
        _current_task_type='task', _pending_events=[], _task_started_ts=before.task_started_at,
        _owner_message_admission_lock=threading.Lock(), owner_wait_callback=lambda *_:None,
        _accepting_owner_messages=True)  # a forced rail closes this intake
    for name in ('_emit_live_log','_emit_typing_start','_emit_progress','_capture_mutation_baseline'):
        monkeypatch.setattr(agent, name, lambda *a, **kw:None)
    monkeypatch.setattr(agent, '_run_delegate_preflight', lambda _l,_t,d:(d,False))
    initial, sends = [], []

    def build(*, ctx, task, reasoning_effort, route_resolver, **kw):
        ctx.context_fit_plan = fit(task, reasoning_effort, route_resolver)
        initial.append({'model':ctx.context_fit_plan.model, 'provider':ctx.context_fit_plan.provider,
                        **view(ctx.context_fit_plan)})
        return ctx.context_fit_plan.messages_for(ctx.context_fit_plan.initial_mode), {}

    def send(*args, **kw):
        sends.append({'model':args[2], 'effort':args[4], 'role':kw.get('model_role'),
            'measured': kw.get('physical_context') is not None, 'plan': registry._ctx.context_fit_plan.model,
            'tokens': cf.estimate_context_prompt_tokens(args[1], args[3]), 'pending': registry._ctx.active_model_override,
            'admitted': registry._ctx.context_fit_plan.window_record.admitted, 'local': kw.get('use_local'),
            'provider': registry._ctx.context_fit_plan.provider, 'pending_local': registry._ctx.active_use_local_override})
        return {'role':'assistant','content':'continued','tool_calls':[]}, 0.0

    monkeypatch.setattr(agent_module, 'build_llm_messages', build)
    monkeypatch.setattr(loop, 'call_llm_with_retry', send)
    bound = []
    if decision == 'pending_priced':  # the budget tail's acting fit (``loop_budget._check_budget_limits``), then no stop
        from ouroboros.loop_forced_finalization import _bind_forced_route
        monkeypatch.setattr(loop, '_finish_tool_round_budget', lambda limit, *_a: bound.append(
            (_bind_forced_route(limit), registry._ctx.context_fit_plan.model)))
    with model_wait.task_model_wait_scope(task=task, drive_root=tmp_path, event_queue=None, worker_slot_held=True):
        ctx, messages, caps = agent._prepare_task_context(task)
        result, _, trace = loop.run_llm_loop(messages=messages, tools=registry, llm=SimpleNamespace(default_model=lambda:small),
            drive_logs=tmp_path/'logs', emit_progress=lambda *a,**kw:None, incoming_messages=stdqueue.Queue(),
            task_id=task['id'], drive_root=tmp_path, initial_effort='none', budget_remaining_usd=caps['budget_remaining'])
    record = {'decision': decision, 'queries':calls, 'initial':initial, 'sent':sends,
        'actual':view(ctx.context_fit_plan), 'ideal':view(ideal), 'expected_model':expected_model, 'result':result}
    print('SAVED_MODEL_WITNESS', json.dumps(record, sort_keys=True))
    if decision.endswith('_forced'):  # the start fits the next ordinary send; the rail's one call is the acting route's
        [forced] = sends
        assert (result, forced['model'], forced['measured'], forced['plan'], forced['pending']) == (
            'continued', acting, True, acting, wide) and forced['tokens'] <= (100000 if local else limits[small])
        assert (forced['local'], forced['provider'] == 'local', forced['pending_local']) == (local, local, False)
        assert (forced['role'], forced['effort']) == ('main', 'none')
        assert forced['tokens'] <= forced['admitted']
        assert trace['forced_finalization']['reason_code'] == 'deadline_local'
        assert (initial[0]['model'], initial[0]['provider']) == (expected_model, 'openai')
        return
    assert bound == ([(True, small)] if decision == 'pending_priced' else [])
    assert result == 'continued' and len(sends) == 1 and sends[0]['model'] == expected_model
    assert sends[0]['pending'] is None
    if decision != 'new_return':
        assert view(ctx.context_fit_plan) == view(ideal), 'the selected wider route must keep its full view'
    assert initial[0]['model'] == expected_model
