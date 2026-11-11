"""The durable work primary wins over mail order, stale workers and saved routes."""
from __future__ import annotations

import copy
import json
import queue
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.gateway.task_model import api_task_model
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros import lead_fact, loop, owner_wait, route_choice
from ouroboros.tools.registry import ToolRegistry
from tests._budget_pause_exact_helpers import _install_queue
from tests._task_settings_helpers import isolated_runtime, save_owner  # noqa: F401

A, B, C = "openai::a", "openai::b", "openai::c"


@pytest.fixture
def host(isolated_runtime, monkeypatch):  # noqa: F811
    q, _state, workers = _install_queue(isolated_runtime, monkeypatch)
    save_owner({"OUROBOROS_MODEL": A})
    app = Starlette(routes=[Route("/api/tasks/{task_id}/model", api_task_model, methods=["POST"])])
    app.state.drive_root = isolated_runtime
    with TestClient(app) as client:
        yield isolated_runtime, q, workers, client


def queued(host, **fields):
    root, _q, workers, _client = host
    task = {"id": "work", "type": "task", "chat_id": 7, "root_task_id": "work",
            "metadata": {"lead": {"source": "main"}}, **fields}
    workers.PENDING.append(task)
    write_task_result(root, "work", "scheduled", **{key: value for key, value in task.items() if key != "id"})
    return task


def act(host, action="set", *, request_id="press", task_id="work", **fields):
    return host[3].post(f"/api/tasks/{task_id}/model", json={"request_id": request_id, "action": action, **fields})


def test_queued_set_persists_the_primary_and_exact_retry_receipt(host):
    task = queued(host)
    response = act(host, model=B, use_local=True)
    assert response.status_code == 200, response.text
    expected = {"model": B, "use_local": True, "basis": "owner"}
    assert response.json() == {"ok": True, "task_id": "work", "applied": "at_start", "lead": expected}
    assert task["metadata"]["lead"] == load_task_result(host[0], "work")["metadata"]["lead"] == {
        "source": "owner", "model": B, "use_local": True}
    saved = copy.deepcopy(task)
    assert act(host, model=C).json() == response.json()
    assert task == saved
    preferences = json.loads((host[0] / "state/ui_preferences.json").read_text(encoding="utf-8"))
    assert preferences["model_recents"][0] == {"model": B, "use_local": True}


def test_queued_follow_keeps_a_reference_and_an_unlisted_model_is_accepted(host):
    task = queued(host)
    assert act(host, model="vendor/new-unlisted-model").status_code == 200
    followed = act(host, "follow", request_id="follow")
    assert followed.status_code == 200
    assert task["metadata"]["lead"] == {"source": "main"}
    from ouroboros.lead_fact import project_lead
    save_owner({"OUROBOROS_MODEL": C})
    assert project_lead(task=task, queued=True) == {"model": C, "use_local": False, "basis": "main"}


def test_an_older_retry_cannot_undo_a_later_press(host):
    task = queued(host)
    first = act(host, model=B, request_id="first")
    assert act(host, model=C, request_id="second").status_code == 200
    saved = copy.deepcopy(task)
    assert act(host, model=A, request_id="first").json() == first.json()
    assert task == saved and task["metadata"]["lead"]["model"] == C


@pytest.mark.parametrize("body", [
    {"model": ""}, {"model": "a\nb"}, {"model": B, "use_local": "yes"},
    {"model": "claudexor::codex=unlisted", "use_local": True}, {"model": B, "text": "hi"},
])
def test_invalid_model_writes_nothing(host, body):
    task = queued(host)
    before = copy.deepcopy(load_task_result(host[0], "work"))
    response = act(host, **body)
    assert response.status_code == 400 and response.json()["reason_code"] == "task_model_invalid"
    assert load_task_result(host[0], "work") == before
    assert task["metadata"]["lead"] == {"source": "main"}


@pytest.mark.parametrize("action", ["return", "wait", "keep"])
def test_fallback_exits_refuse_a_queued_work(host, action):
    queued(host)
    response = act(host, action)
    assert response.status_code == 409 and response.json()["reason_code"] == "model_action_state"


@pytest.mark.parametrize("fields,code", [
    ({"parent_task_id": "parent", "root_task_id": "parent", "delegation_role": "subagent"}, "not_a_root_task"),
    ({"_presence_turn": True}, "presence_model_unsupported"),
])
def test_non_roots_and_presence_are_refused(host, fields, code):
    queued(host, **fields)
    response = act(host, model=B)
    assert response.status_code == 409 and response.json()["reason_code"] == code


def test_unknown_finished_and_non_durable_turns_are_refused(host):
    assert act(host, model=B).json()["reason_code"] == "task_not_live"
    task = queued(host)
    write_task_result(host[0], "work", "completed")
    assert act(host, model=B).json()["reason_code"] == "task_not_live"
    task["id"] = "direct"
    task["root_task_id"] = "direct"
    assert act(host, task_id="direct", model=B).json()["reason_code"] == "model_control_unavailable"


def running(host):
    task = queued(host)
    host[2].PENDING.remove(task)
    host[2].RUNNING["work"] = {"task": task, "attempt": 1, "lead": {"model": A, "use_local": False, "basis": "main"}}
    write_task_result(host[0], "work", "running")
    return task


def turn(host, monkeypatch, metadata=None):
    tools = ToolRegistry(repo_dir=host[0], drive_root=host[0])
    ctx = tools._ctx
    ctx.task_id, ctx.task_attempt, ctx.task_metadata = "work", 1, copy.deepcopy(metadata or {"lead": {"source": "main"}})
    ctx.budget_drive_root, ctx.event_queue = host[0], queue.Queue()
    route_choice.bind_task_route(ctx, ctx.task_metadata, presence=False)
    loop._reset_turn_state(ctx)
    ctx.active_model, ctx.active_use_local, ctx.active_effort = A, False, "medium"
    ctx.primary_route, ctx.route_choice = {"model": A, "use_local": False, "role": "main"}, route_choice.following("main")
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda plan, *_args, **_kwargs: (plan, "max"))
    return tools, ctx


def drain_and_apply(host, tools, ctx, seen=None):
    messages = []
    controls = loop._drain_incoming_messages(messages, queue.Queue(), host[0], "work", ctx.event_queue,
                                            set() if seen is None else seen, owner_ctx=ctx)
    from ouroboros.loop_routes import after_mail
    limit = SimpleNamespace(tools=tools, messages=messages, active_model=ctx.active_model,
        active_use_local=ctx.active_use_local, active_effort=ctx.active_effort, tool_schemas=[], llm=None)
    after_mail(limit, None, "max", "max", controls)
    assert not messages  # typed control never enters the owner's words or a dialogue row
    return controls


def test_tool_tail_and_mail_without_an_act_keep_the_pending_route_until_a_fresh_round(host, monkeypatch):
    from ouroboros.loop_routes import after_mail, bind_round_route

    tools, ctx = turn(host, monkeypatch)
    ctx.active_model_override = B
    ctx.active_use_local_override = True
    ctx.active_effort_override = "high"
    plan = SimpleNamespace(model_role="lead")
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *_args, **_kwargs: (plan, "low"))
    old_route = (A, False, "medium")
    assert bind_round_route(tools, [], old_route, None, "max", "max", [], pending_tool_budget=True) == (
        *old_route, None, "max")
    limit = SimpleNamespace(tools=tools, active_model=ctx.active_model,
                            active_use_local=ctx.active_use_local, active_effort=ctx.active_effort)
    assert after_mail(limit, None, "max", "max", {}) == (*old_route, None, "max")
    assert ctx.active_model_override == B and ctx.active_effort_override == "high"
    assert ctx.event_queue.empty()

    assert bind_round_route(tools, [], old_route, None, "max", "max", []) == (B, True, "high", plan, "low")
    assert (ctx.active_model, ctx.active_use_local, ctx.active_effort) == (B, True, "high")
    assert ctx.context_fit_plan is plan and ctx.active_context_mode == "low"
    assert ctx.active_model_override is None and ctx.active_effort_override is None
    events = list(ctx.event_queue.queue)
    assert len(events) == 1 and events[0]["type"] == "work_lead"
    assert events[0]["lead"]["model"] == B and events[0]["lead"]["use_local"] is True


def test_reversed_mail_applies_the_later_durable_decision_once(host, monkeypatch):
    from ouroboros import owner_mailbox
    running(host)
    tools, ctx = turn(host, monkeypatch)
    write, delayed = owner_mailbox.write_owner_message, []
    monkeypatch.setattr(owner_mailbox, "write_owner_message", lambda *args, **kw: delayed.append((args, kw)) or True)
    first = act(host, model=B, request_id="first")
    second = act(host, model=C, request_id="second")
    assert first.status_code == second.status_code == 200
    assert second.json()["lead"] == {"model": A, "use_local": False, "basis": "main",
                                     "pending": {"model": C, "use_local": False}}
    assert ctx.active_model == A  # admission never touches the running call
    for args, kwargs in reversed(delayed):
        assert write(*args, **kwargs)
    drain_and_apply(host, tools, ctx)
    assert ctx.active_model == ctx.primary_route["model"] == ctx.task_model_override == C
    assert ctx.route_choice == route_choice.explicit("owner") and ctx._owner_model_applied_id == "second"
    assert len([e for e in ctx.event_queue.queue if e["type"] == "work_lead"]) == 1
    assert not drain_and_apply(host, tools, ctx).get("owner_model")


@pytest.mark.parametrize("at_start", [False, True])
def test_the_next_call_uses_the_act_after_a_drain_or_from_a_stale_saved_queue(host, monkeypatch, at_start):
    from tests.test_unknown_fallback_first import _RouteLLM
    from tests.test_completion_selection import finish
    from ouroboros import usage_accounting as ua

    task = queued(host) if at_start else running(host)
    tools, ctx = turn(host, monkeypatch)
    llm = _RouteLLM(host[0])
    original_chat, original_drain, presses = llm.chat, loop._drain_incoming_messages, []
    if at_start:
        presses.append(act(host, model=B))
        assert presses[0].json()["applied"] == "at_start"
        host[2].PENDING.remove(task)
        host[2].RUNNING["work"] = {"task": task, "attempt": 1}
        write_task_result(host[0], "work", "running")
    def draining(*args, **kwargs):
        if not presses:
            presses.append(act(host, model=B))
        return original_drain(*args, **kwargs)
    def chat(**kwargs):
        _message, usage = original_chat(**kwargs)
        return finish("done"), usage
    monkeypatch.setattr(loop, "_drain_incoming_messages", draining)
    llm.chat = chat
    with ua.usage_scope(ua.UsageScope(drive_root=host[0], task_id="work", root_task_id="work", global_limit_usd=100)):
        text, _usage, _trace = loop.run_llm_loop(
            messages=[{"role": "user", "content": "go"}], tools=tools, llm=llm, drive_logs=host[0],
            emit_progress=lambda *a, **kw: None, incoming_messages=queue.Queue(),
            task_id="work", drive_root=host[0], event_queue=ctx.event_queue)
    assert presses[0].status_code == 200
    assert text == "done" and [model for model, _messages in llm.sent] == [B]
    assert ctx.primary_route == {"model": B, "use_local": False, "role": "lead"}
    assert "My human" in str(llm.sent[0][1])


@pytest.mark.parametrize("checkpoint_after", [False, True])
def test_resume_after_ack_uses_the_durable_primary_with_or_without_a_new_checkpoint(host, monkeypatch, checkpoint_after):
    running(host)
    tools, ctx = turn(host, monkeypatch)
    def save():
        return json.loads(json.dumps(owner_wait.continuation_state(ctx, [], {}, {}, 3, [], set())))
    old = save()
    assert act(host, model=B).status_code == 200
    drain_and_apply(host, tools, ctx)
    state = save() if checkpoint_after else old
    fresh_tools, fresh = turn(host, monkeypatch)
    owner_wait.restore_continuation_state(fresh_tools, state, [], {}, {}, set())
    from ouroboros.loop_routes import _apply_round_route_overrides
    model, local, *_ = _apply_round_route_overrides(fresh, fresh_tools, [],
        (fresh.active_model, fresh.active_use_local, "medium"), None, "max", "max", [])
    assert (model, local) == (B, False)
    assert fresh.primary_route["model"] == B and fresh.task_model_override == B
    assert fresh.route_choice == route_choice.explicit("owner")
    assert not drain_and_apply(host, fresh_tools, fresh).get("owner_model")  # ACK already survived


def test_stale_worker_results_keep_the_host_primary_and_follow_choice(host):
    task = queued(host, metadata={"lead": {"source": "room", "model": A, "use_local": False}, "model_role": "consciousness"})
    stale = copy.deepcopy(task)
    assert act(host, model=B).status_code == 200
    write_task_result(host[0], "work", "running", metadata=stale["metadata"], owner_model={"request_id": "stale"})
    assert load_task_result(host[0], "work")["metadata"]["lead"]["model"] == B
    assert act(host, "follow", request_id="follow").status_code == 200
    write_task_result(host[0], "work", "completed", metadata=stale["metadata"])
    saved = load_task_result(host[0], "work")
    assert saved["metadata"]["lead"] == {"source": "main"} and saved["metadata"]["model_role"] == "main"
    assert saved["owner_model"]["request_id"] == "follow"


def test_keep_changes_the_primary_and_follow_rejoins_main(host, monkeypatch):
    from ouroboros.loop_model_call import _route_candidates
    running(host)
    tools, ctx = turn(host, monkeypatch)
    ctx.active_model = C
    lead_fact.adopt_fallback(ctx, False, {"failure_code": "quota"}, "fallback:0")
    host[2].RUNNING["work"]["lead"] = lead_fact.lead_fact(ctx)
    ctx.model_wait_context = SimpleNamespace(overrides={"lead": {"model": "obsolete"}})
    assert act(host, "keep").status_code == 200
    drain_and_apply(host, tools, ctx)
    assert ctx.primary_route["model"] == ctx.active_model == C
    assert "lead" not in ctx.model_wait_context.overrides
    assert lead_fact.lead_fact(ctx) == {"model": C, "use_local": False, "basis": "owner"}
    save_owner({"OUROBOROS_MODEL": B})
    ctx.active_model = A
    lead_fact.adopt_fallback(ctx, False, {}, "fallback:0")
    candidates = _route_candidates(ctx, A, False, "fallback:0", None)
    assert any(row[0] == C and row[3] for row in candidates)
    assert all(row[0] != B for row in candidates)
    assert act(host, "follow", request_id="follow").status_code == 200
    drain_and_apply(host, tools, ctx)
    assert ctx.active_model == B and ctx.route_choice == route_choice.following("main")
    save_owner({"OUROBOROS_MODEL": A})
    assert route_choice.round_route(ctx, None, (B, False)) == (A, False)


@pytest.mark.parametrize("action", ["return", "wait"])
def test_owner_fallback_exits_use_the_same_return_and_wait_as_the_tool(host, monkeypatch, action):
    from ouroboros.tools.control_runtime import _switch_model

    running(host)
    tools, ctx = turn(host, monkeypatch)
    ctx.active_model = B
    lead_fact.adopt_fallback(ctx, False, {}, "fallback:0")
    host[2].RUNNING["work"]["lead"] = lead_fact.lead_fact(ctx)
    _other_tools, other = turn(host, monkeypatch)
    other.active_model = B
    assert "primary route" in _switch_model(other, primary=action)
    assert act(host, action).status_code == 200
    drain_and_apply(host, tools, ctx)
    assert ctx.active_model == other.active_model_override == A
    assert ctx.route_wait_on_primary is (action == "wait")
    assert ctx.route_choice == other.route_choice
    assert lead_fact.lead_fact(ctx)["basis"] == "fallback"  # the primary has not answered yet
    from ouroboros.primary_route_observation import record_round_route_result
    call = SimpleNamespace(tools=tools, active_model=A, active_use_local=False, accumulated_usage={})
    record_round_route_result(call, "main", accepted=True)
    assert lead_fact.lead_fact(ctx)["basis"] == "main"


def test_return_preserves_the_role_of_a_carrierless_primary(host, monkeypatch):
    task = running(host)
    task["metadata"] = {"model_role": "consciousness"}
    from ouroboros.task_store import patch_document
    patch_document(host[0], "work", lambda row: {**row, "metadata": task["metadata"]})
    save_owner({"OUROBOROS_MODEL": A, "OUROBOROS_MODEL_CONSCIOUSNESS": B})
    tools, ctx = turn(host, monkeypatch, task["metadata"])
    ctx.primary_route = {"model": B, "use_local": False, "role": "consciousness"}
    ctx.active_model = C
    lead_fact.adopt_fallback(ctx, False, {}, "fallback:0")
    host[2].RUNNING["work"]["lead"] = lead_fact.lead_fact(ctx)
    response = act(host, "return")
    assert response.status_code == 200
    assert response.json()["lead"]["pending"]["model"] == B
    drain_and_apply(host, tools, ctx)
    assert ctx.active_model == B and ctx.primary_route["role"] == "consciousness"
    assert ctx.route_choice == route_choice.following("consciousness")


def test_a_recent_model_write_failure_can_be_retried_without_changing_the_primary(host, monkeypatch):
    from ouroboros.gateway import ui_preferences

    task = queued(host)
    write = ui_preferences.record_model_recent
    def unavailable(*args, **kwargs):
        raise OSError("test preferences unavailable")
    monkeypatch.setattr(ui_preferences, "record_model_recent", unavailable)
    assert act(host, model=B).json()["reason_code"] == "model_recents_pending"
    monkeypatch.setattr(ui_preferences, "record_model_recent", write)
    assert act(host, model=C).status_code == 200
    assert task["metadata"]["lead"]["model"] == B
    prefs = json.loads((host[0] / "state/ui_preferences.json").read_text(encoding="utf-8"))
    assert prefs["model_recents"][0]["model"] == B


@pytest.mark.parametrize("action", ["return", "wait", "keep"])
def test_fallback_exits_refuse_a_running_primary(host, action):
    running(host)
    response = act(host, action)
    assert response.status_code == 409 and response.json()["reason_code"] == "model_action_state"


@pytest.mark.parametrize("newer_act", [False, True])
def test_only_the_decision_holds_the_queue_lock_and_recents_ack_its_own_act(host, monkeypatch, newer_act):
    from ouroboros import owner_model, task_store
    from ouroboros.gateway import ui_preferences

    queued(host)
    q = host[1]
    patch, record = task_store.patch_document, ui_preferences.record_model_recent
    locks = []
    def patched(*args, **kwargs):
        locks.append(q._queue_lock._is_owned())
        return patch(*args, **kwargs)
    def recent(root, model, local):
        assert not q._queue_lock._is_owned()
        if model == C:
            raise OSError("second suggestion still owed")
        if newer_act:
            with pytest.raises(owner_model.ModelActRefused, match="model recents pending"):
                owner_model.admit("work", {"request_id": "second", "action": "set", "model": C})
        return record(root, model, local)
    monkeypatch.setattr(task_store, "patch_document", patched)
    monkeypatch.setattr(ui_preferences, "record_model_recent", recent)
    first = act(host, model=B)
    assert first.status_code == 200
    assert locks == ([True, True, False] if newer_act else [True, False])
    stored = load_task_result(host[0], "work")
    assert stored["owner_model"]["recent_recorded"] is (not newer_act)
    assert stored["metadata"]["lead"]["model"] == (C if newer_act else B)
    monkeypatch.setattr(ui_preferences, "record_model_recent", record)
    assert act(host, model=A).json() == first.json()
    stored = load_task_result(host[0], "work")
    assert stored["owner_model"]["recent_recorded"] is True
    assert stored["metadata"]["lead"]["model"] == (C if newer_act else B)


def test_continue_and_the_finishing_pipeline_keep_the_new_carrier(host, monkeypatch):
    from ouroboros.agent_task_pipeline import _store_task_result
    from tests.test_continue_work_identity import _successor

    task = running(host)
    stale = copy.deepcopy(task)
    assert act(host, model=B).status_code == 200
    env = SimpleNamespace(drive_root=host[0], repo_dir=host[0], drive_path=lambda rel: host[0] / rel)
    _store_task_result(env, stale, "Done.", {"rounds": 1, "cost": 0}, {"tool_calls": [], "reasoning_notes": []})
    stored = load_task_result(host[0], "work")
    assert stored["status"] == "completed" and stored["metadata"]["lead"]["model"] == B
    successor = _successor(host[0], "work")
    assert successor["metadata"]["lead"] == stored["metadata"]["lead"]


@pytest.mark.parametrize("path", ["/api/tasks/work/model", "/api/owner/chat-model", "/api/tasks/work/%6dodel"])
def test_the_agent_browser_cannot_press_owner_model_controls(host, monkeypatch, path):
    from ouroboros import browser_policy

    monkeypatch.setattr(browser_policy, "browser_url_block_reason", lambda *a, **k: "")
    monkeypatch.setattr(browser_policy, "runtime_service_kind", lambda *a: "main")
    request = SimpleNamespace(method="POST", url="http://example.invalid" + path)
    assert "BROWSER_OWNER_CONTROL_BLOCKED" in browser_policy.browser_request_block_reason(
        request, None, restricted=False, runtime_mode="standard")
    assert browser_policy.browser_request_block_reason(request, None, restricted=False, runtime_mode="cyber_pro") == ""
    request.method = "GET"
    assert browser_policy.browser_request_block_reason(request, None, restricted=False, runtime_mode="standard") == ""
    # The owner's HTTP ingress is separate from the agent browser guard.
    queued(host)
    assert act(host, model=B).status_code == 200


def test_model_control_and_pending_share_the_census_and_work_projection(host, monkeypatch):
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from ouroboros.gateway.work_projection import works_snapshot
    from supervisor.events import _handle_work_lead
    from ouroboros.server_routing_context import addressable_roots

    running(host)
    tools, ctx = turn(host, monkeypatch)
    def snapshot():
        raw, owners = {}, []
        [activity] = _chat_activities_snapshot_safe(host[0], work_facts=raw, decision_owners=owners)
        [work] = works_snapshot(host[0], owners, raw)["works"]
        [manifest] = addressable_roots(host[1]).rows()
        assert activity["lead"] == work["lead"] == manifest["lead"]
        return work
    assert {"action": "model", "target": "work"} in snapshot()["controls"]
    assert act(host, model=B).status_code == 200
    assert snapshot()["lead"]["pending"] == {"model": B, "use_local": False}
    drain_and_apply(host, tools, ctx)
    for event in list(ctx.event_queue.queue):
        _handle_work_lead(event, host[1])
    assert snapshot()["lead"] == {"model": B, "use_local": False, "basis": "owner"}
    ctx.active_model = C
    lead_fact.adopt_fallback(ctx, False, {}, "fallback:0")
    _handle_work_lead(list(ctx.event_queue.queue)[-1], host[1])
    assert snapshot()["lead"]["basis"] == "fallback" and "pending" not in snapshot()["lead"]


def test_a_lost_wake_is_healed_by_the_same_id_without_changing_its_decision(host, monkeypatch):
    from ouroboros import owner_mailbox

    running(host)
    tools, ctx = turn(host, monkeypatch)
    real_write = owner_mailbox.write_owner_message
    monkeypatch.setattr(owner_mailbox, "write_owner_message", lambda *a, **k: False)
    response = act(host, model=B)
    assert response.status_code == 503 and response.json()["reason_code"] == "model_wake_failed"
    assert load_task_result(host[0], "work")["metadata"]["lead"]["model"] == B
    monkeypatch.setattr(owner_mailbox, "write_owner_message", real_write)
    assert act(host, model=C).status_code == 200
    drain_and_apply(host, tools, ctx)
    assert ctx.active_model == B
    assert json.loads((host[0] / "state/ui_preferences.json").read_text(encoding="utf-8"))["model_recents"][0]["model"] == B


def test_owner_model_wakes_a_refused_call_wait_but_never_interrupts_a_sent_call(host, monkeypatch):
    from ouroboros.model_wait import ModelWaitInterrupted, TaskModelWait
    from ouroboros.owner_model import waiting_wake

    task = running(host)
    tools, ctx = turn(host, monkeypatch)
    waiter = TaskModelWait(task=task, drive_root=host[0], event_queue=queue.Queue(),
                           worker_slot_held=True, owner_control=lambda: None)
    waiter.tool_context = ctx
    assert waiting_wake(waiter) is None
    assert act(host, model=B).status_code == 200
    assert waiter.control_reason() is None  # the physical sender's polling reader stays quiet
    with pytest.raises(ModelWaitInterrupted) as interrupted:
        waiter.wait(None, RuntimeError("resource unavailable"), {"model": A, "model_role": "main",
                    "model_poll_control": lambda: "test_wait_limit"})
    assert interrupted.value.control_reason == "owner_model"
    assert loop._handle_model_wait_control(SimpleNamespace(), interrupted.value) is None
    drain_and_apply(host, tools, ctx)
    assert ctx.active_model == B and waiting_wake(waiter) is None


@pytest.mark.parametrize("kind", ["presence", "direct"])
def test_model_control_is_absent_when_the_owner_act_cannot_hold(host, monkeypatch, kind):
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from ouroboros.gateway.work_projection import works_snapshot
    from supervisor import active_activity

    if kind == "presence":
        queued(host, _presence_turn=True, execution_owner={"kind": "presence"})
    else:
        registry = active_activity.DirectActivityRegistry()
        monkeypatch.setattr(active_activity, "get_direct_activity_registry", lambda: registry)
        _tools, ctx = turn(host, monkeypatch)
        ctx.model_wait_context = SimpleNamespace(task_id="work", attempt=1, closed=False, overrides={})
        registry.register("work", 7, actor=SimpleNamespace(tools=_tools, drive_root=host[0]))
    raw, owners = {}, []
    _chat_activities_snapshot_safe(host[0], work_facts=raw, decision_owners=owners)
    works = works_snapshot(host[0], owners, raw)["works"]
    assert works
    assert all(control["action"] != "model" for work in works for control in work["controls"])
    if kind == "direct":
        registry.unregister("work")


def test_cited_work_switch_uses_supervisor_primary_and_returns_there_after_fallback(host, monkeypatch):
    from ouroboros.tools.control_runtime import _switch_model
    from supervisor.events_mind_controls import handle_mind_task_control

    task = running(host)
    tools, ctx = turn(host, monkeypatch)
    ctx._owner_directives = [{"source": "owner_mailbox", "content": "Use the other route for this work."}]
    class Bus:
        def put_nowait(self, event):
            assert event["type"] == "mind_task_control"
            handle_mind_task_control(event, SimpleNamespace(DRIVE_ROOT=host[0], bridge=SimpleNamespace(push_log=lambda row: None)))
    ctx.event_queue = Bus()
    result = _switch_model(ctx, B, owner_words="Use the other route")
    assert '"ok": true' in result
    assert task["metadata"]["lead"] == {"source": "owner_words", "model": B, "use_local": False}
    assert ctx.primary_route == {"model": B, "use_local": False, "role": "lead"}
    ctx.active_model, ctx.route_choice = C, route_choice.explicit("fallback")
    _switch_model(ctx, primary="return")
    assert ctx.active_model_override == B and ctx.active_role_override == "lead"
    restored = turn(host, monkeypatch)[1]
    from ouroboros.owner_model import restore_current
    restore_current(restored)
    assert restored.primary_route == ctx.primary_route


@pytest.mark.parametrize("scope", ["room", "main"])
def test_cited_settings_scopes_use_owner_writers_and_uncited_attempts_refuse(host, monkeypatch, scope):
    from ouroboros.tools.control_runtime import _switch_model
    from ouroboros.projects_registry import create_project
    from ouroboros.composer_cognition import read_model_choice
    from ouroboros import config

    root = host[0]
    create_project(root, "room", origin="owner_ui")
    _tools, ctx = turn(host, monkeypatch)
    ctx.project_id = "room"
    ctx._owner_directives = [{"source": "initial_user", "content": "Set that model for it."}]
    for mode in ("pro", "cyber_pro"):
        monkeypatch.setattr(config, "get_runtime_mode", lambda: mode)
        refused = _switch_model(ctx, B, scope=scope)
        assert "requires verified owner_words" in refused
        assert read_model_choice(root, "room") == {"kind": "main"}
        assert json.loads(config.SETTINGS_PATH.read_text(encoding="utf-8"))["OUROBOROS_MODEL"] == A
    result = _switch_model(ctx, B, owner_words="Set that model", scope=scope)
    assert '"ok": true' in result
    if scope == "room":
        assert read_model_choice(root, "room")["model"] == B
        _switch_model(ctx, "main", owner_words="Set that model", scope=scope, project_id="room")
        assert read_model_choice(root, "room") == {"kind": "main"}
    else:
        assert '"account": "none"' in result and '"window_reset": false' in result
        assert json.loads(config.SETTINGS_PATH.read_text(encoding="utf-8"))["OUROBOROS_MODEL"] == B
        rows = (root / "logs/events.jsonl").read_text(encoding="utf-8")
        assert '"surface": "tool"' in rows and '"actor_task_id": "work"' in rows
    assert json.loads((root / "state/ui_preferences.json").read_text(encoding="utf-8"))["model_recents"][0]["model"] == B


def test_cited_work_switch_reports_a_failed_wake_even_after_the_primary_was_saved(host, monkeypatch):
    from ouroboros import owner_mailbox
    from ouroboros.tools.control_runtime import _switch_model
    from supervisor.events_mind_controls import handle_mind_task_control

    running(host)
    _tools, ctx = turn(host, monkeypatch)
    ctx._owner_directives = [{"source": "owner_mailbox", "content": "Use the other route."}]
    monkeypatch.setattr(owner_mailbox, "write_owner_message", lambda *a, **kw: False)
    class Bus:
        def put_nowait(self, event):
            handle_mind_task_control(event, SimpleNamespace(DRIVE_ROOT=host[0], bridge=SimpleNamespace(push_log=lambda row: None)))
    ctx.event_queue = Bus()
    result = _switch_model(ctx, B, owner_words="Use the other route")
    assert json.loads(result) == {"ok": False, "error": "model_wake_failed"}
    assert load_task_result(host[0], "work")["metadata"]["lead"]["model"] == B
    assert ctx.primary_route["model"] == A


def test_cited_main_write_discloses_a_failed_recent_suggestion_without_claiming_no_effect(host, monkeypatch):
    from ouroboros.tools.control_runtime import _switch_model
    from ouroboros.gateway import ui_preferences
    from ouroboros import config

    _tools, ctx = turn(host, monkeypatch)
    ctx._owner_directives = [{"source": "initial_user", "content": "Use the other route."}]
    def unavailable(*args):
        raise OSError("preferences unavailable")
    monkeypatch.setattr(ui_preferences, "record_model_recent", unavailable)
    result = _switch_model(ctx, B, owner_words="Use the other route", scope="main")
    assert "Model choice saved; recent suggestion could not be recorded" in result
    assert json.loads(config.SETTINGS_PATH.read_text(encoding="utf-8"))["OUROBOROS_MODEL"] == B


@pytest.mark.parametrize("action", ["return", "wait"])
def test_a_card_returns_an_uncited_cyber_start_to_its_own_primary(host, monkeypatch, action):
    task = queued(host, metadata={"model_role": "lead", "model": B, "use_local_model": False})
    host[2].PENDING.remove(task)
    host[2].RUNNING["work"] = {"task": task, "attempt": 1, "lead": {
        "model": C, "use_local": False, "basis": "fallback",
        "primary": {"model": B, "use_local": False, "basis": "mind"}}}
    write_task_result(host[0], "work", "running")
    assert act(host, action).status_code == 200
    _tools, ctx = turn(host, monkeypatch)
    from ouroboros.owner_model import restore_current
    restore_current(ctx)
    assert ctx.primary_route == {"model": B, "use_local": False, "role": "lead"}
    assert ctx.route_wait_on_primary is (action == "wait")
    assert lead_fact.primary_basis(ctx) == "mind"
