"""Live model waits retain their owner, accepted controls and terminal rails."""

import asyncio
from copy import deepcopy
import json
import pathlib
import queue
from types import SimpleNamespace

import pytest

from ouroboros import cancel_intents, loop, model_wait, owner_mailbox, usage_accounting as ua
from ouroboros.model_slots import MODEL_ACCOUNTS_KEY
from ouroboros.task_results import load_task_result
from tests.test_llm_claudexor import MODEL, result, setup as gateway_fixture
from tests.test_model_wait import live_wait as wait_fixture
from tests.test_subscription_main_wait import main_call as main_fixture

setup = gateway_fixture
live_wait = wait_fixture
main_call = main_fixture


def _accepted_switch(live_wait):
    root, _gateway, _client, owner, _events, decide = live_wait
    row = {"wait_id": "wait-one", "revision": 1, "task_attempt": 1, "state": "waiting", "role": "light"}
    model_wait.mutate_wait(root, "task-one", "wait-one", lambda _: row)
    owner.waits["wait-one"] = dict(row)
    body = {"request_id": "switch-one", "decision_id": "model_wait:task-one:wait-one", "revision": 1,
            "action": "switch", "model": MODEL, "credential_profile_id": "account-b", "use_local": False}
    assert decide(body).status_code == 202
    return root, owner, body, decide


def test_failed_mailbox_read_retries_accepted_switch_without_another_post(live_wait, monkeypatch):
    root, owner, _body, _decide = _accepted_switch(live_wait)
    mailbox = owner_mailbox._mailbox_path(root, "task-one")
    read = pathlib.Path.read_text
    failed = False

    def fail_once(path, *args, **kwargs):
        nonlocal failed
        if path == mailbox and not failed:
            failed = True
            raise OSError("one failed read")
        return read(path, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", fail_once)
    owner._drain_controls()
    assert not owner.seen_controls and owner.mailbox_stamp is None
    owner._drain_controls()
    assert owner.waits["wait-one"]["_action"]["credential_profile_id"] == "account-b"


def test_failed_canonical_read_does_not_consume_accepted_control(live_wait, monkeypatch):
    _root, owner, _body, _decide = _accepted_switch(live_wait)
    reader = owner.read_rows
    monkeypatch.setattr(owner, "read_rows", lambda: (_ for _ in ()).throw(ValueError("unreadable authority")))
    with pytest.raises(ValueError, match="unreadable authority"):
        owner._drain_controls()
    assert not owner.seen_controls and owner.mailbox_stamp is None
    monkeypatch.setattr(owner, "read_rows", reader)
    owner._drain_controls()
    assert owner.waits["wait-one"]["_action"]["request_id"] == "switch-one"


def test_torn_mailbox_read_cannot_apply_or_consume_a_parsed_control(live_wait):
    root, owner, _body, _decide = _accepted_switch(live_wait)
    path = owner_mailbox._mailbox_path(root, "task-one")
    complete = path.read_bytes()
    path.write_bytes(complete.rstrip(b"\r\n"))
    owner._drain_controls()
    assert not owner.seen_controls and owner.mailbox_stamp is None
    assert "_action" not in owner.waits["wait-one"]
    path.write_bytes(complete)
    owner._drain_controls()
    assert owner.waits["wait-one"]["_action"]["request_id"] == "switch-one"


@pytest.mark.parametrize("attempt,state,expected", [(1, "waiting", False), (2, "waiting", True), (2, "resolved", False)])
def test_supervisor_wait_belongs_to_current_retry_attempt(attempt, state, expected):
    from supervisor.task_model_wait import model_waiting

    task = {"_attempt": 1, "model_waits": {"wait": {"state": state, "task_attempt": attempt}}}
    retried = dict(task)
    retried["_attempt"] = 2
    assert model_waiting({"task": retried, "attempt": 2}) is expected
    assert task["model_waits"] == retried["model_waits"]  # History is retained.


def test_plan_executor_carries_wait_and_admitted_wallet_without_main_capture(tmp_path, monkeypatch):
    from ouroboros import review_substrate
    from ouroboros.tools import plan_review_runtime

    observed = []
    wallet = ua.UsageScope(drive_root=tmp_path, task_id="plan", root_task_id="root", root_limit_usd=17)
    physical = ua.PhysicalAttemptContext("owner_max", "max", "cold_estimate", "main-route", "round", None, 872000, False, False)

    def review(*args, **kwargs):
        observed.append((model_wait.current_model_wait(), ua.current_usage_scope(),
                         ua.current_physical_attempt_context(), ua.current_physical_attempt_predicate()))
        return SimpleNamespace(actors=[])

    monkeypatch.setattr(review_substrate, "run_review_request", review)
    with ua.usage_scope(wallet), ua.bind_physical_attempt_context(physical, lambda _candidate: False):
        with model_wait.task_model_wait_scope(task={"id": "plan"}, drive_root=tmp_path,
                                              event_queue=None, worker_slot_held=True) as owner:
            asyncio.run(plan_review_runtime.run_plan_review_slots(
                SimpleNamespace(drive_root=tmp_path, task_id="plan"), [], system_prompt="Review", user_content="Plan"))
    assert observed == [(owner, wallet, None, None)]


def test_pinned_wait_rejects_other_catalog_account_before_new_generation(live_wait, monkeypatch):
    _root, gateway, client, _owner, _events, _decide = live_wait
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"light": "account-a"}))
    monkeypatch.setattr(model_wait.time, "sleep", lambda _seconds: None)
    from ouroboros import config
    monkeypatch.setattr(config, "NETWORK_WAIT_BACKOFF_START_SEC", 0)
    gateway.results = [result(outcome="failed", problem={"code": "subscription_window_exhausted", "message": "quota"}), result()]
    gateway.dispatch = ["not_started", "response_received"]
    polls = []

    def catalog(source, profile=None, **kwargs):
        assert profile == "account-a" and len(gateway.accepted_operations) == 1
        polls.append(profile)
        return {"source": source, "credentialProfileId": "account-b" if len(polls) == 1 else "account-a",
                "models": [{"id": "exact-model"}]}

    monkeypatch.setattr(client, "claudexor_model_catalog", catalog)
    client.chat([], MODEL, model_role="light")
    assert len(polls) == 2 and len(gateway.accepted_operations) == 2
    assert all(payload["account"] == {"mode": "pin", "profileId": "account-a"} for payload, _key in gateway.uploads)


def _loop_tools(ctx, owner):
    from ouroboros.tools.registry import ToolRegistry

    tools = ToolRegistry(repo_dir=ctx.drive_root, drive_root=ctx.drive_root)
    tools._ctx.context_fit_plan = ctx.context_fit_plan
    tools._ctx.task_model_override = MODEL
    tools._ctx.task_attempt = 1
    tools._ctx.task_metadata = {}
    owner.tool_context = tools._ctx
    return tools


def test_main_quota_calls_configured_api_fallback_before_any_owner_wait(main_call, monkeypatch):
    """Owner order: Auto rotation, then the configured fallback, and only then the owner question."""
    from ouroboros.llm_attempt import _attempt_request, _candidate_before_dispatch

    ctx, gateway, owner, events, _decide, _observations = main_call
    tools = _loop_tools(ctx, owner)
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", "openai::alternate")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    gateway.results = [result(outcome="failed", problem={"code": "subscription_window_exhausted", "message": "quota"})]
    gateway.dispatch = ["not_started"]
    api_calls = []
    remote = ctx.llm._chat_remote

    def send(target, messages, schemas, *args, **kwargs):
        if target["provider"] == "claudexor":
            return remote(target, messages, schemas, *args, **kwargs)
        api_calls.append(target["provider"])
        request_body = {"messages": deepcopy(messages), "tools": schemas or [], "model": "alternate"}
        request = _attempt_request(target, request_body)
        usage = {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.1, "provider": "openai",
                 "resolved_model": "alternate", "cost_final": True}
        return ua.execute_physical_attempt(request, lambda: ({"role": "assistant", "content": "Finished"}, usage),
                                           extractor=lambda value: (value[1], 0.1, True),
                                           before_dispatch=_candidate_before_dispatch(request_body, request))

    def catalog(*args, **kwargs):
        pytest.fail("the owner is asked only after every configured route of the round failed")

    monkeypatch.setattr(ctx.llm, "_chat_remote", send)
    monkeypatch.setattr(ctx.llm, "claudexor_model_catalog", catalog)
    text, usage, _trace = loop.run_llm_loop(
        ctx.messages, tools, ctx.llm, ctx.drive_logs, lambda *_args, **_kwargs: None, queue.Queue(),
        task_id="task-one", drive_root=ctx.drive_root, event_queue=events)
    assert text == "Finished" and api_calls == ["openai"]
    assert not [event for event in list(events.queue) if event.get("type") == "task_model_wait"]
    assert usage["_model_route"] == {} and len(gateway.accepted_operations) == 1
    assert any(message.get("content") == "verified read A" for message in ctx.messages)
    assert any(message.get("content") == "completed review B" for message in ctx.messages)


def test_graceful_intent_waits_for_current_control_and_never_looks_like_hard_cancel(live_wait):
    from supervisor.owner_stop import owner_stop_control_id
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION

    root, _gateway, _client, owner, _events, _decide = live_wait
    intent = cancel_intents.request_cancel(root, "task-one", requested_stop_policy=cancel_intents.STOP_POLICY_FINALIZE)
    assert owner.control_reason() is None
    owner_mailbox.write_owner_message(root, REASON_OWNER_REQUESTED_FINALIZATION, "task-one",
                                     msg_id=owner_stop_control_id(intent), kind=owner_mailbox.KIND_FINALIZE_NOW)
    assert owner.control_reason() == "finalize_requested"
    owner.tool_context = SimpleNamespace(_loop_mailbox_seen_ids={owner_stop_control_id(intent)})
    assert owner.control_reason() is None
    cancel_intents.request_cancel(root, "task-one", requested_stop_policy=cancel_intents.STOP_POLICY_IMMEDIATE)
    assert owner.control_reason() == "cancelled"


def test_pre_call_graceful_control_after_round_drain_runs_one_final_turn(main_call, monkeypatch):
    """Wrap up arriving between the round drain and model admission keeps its final turn."""
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
    from supervisor.owner_stop import owner_stop_control_id

    ctx, gateway, owner, events, _decide, _observations = main_call
    tools = _loop_tools(ctx, owner)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    final = result()
    final["message"] = {"role": "assistant", "content": "Verified work summarized after owner stop"}
    gateway.results = [final]
    gateway.dispatch = ["response_received"]
    call = loop._call_round_model
    injected = []

    def after_drain(model_call):
        assert not injected
        injected.append(True)
        intent = cancel_intents.request_cancel(ctx.drive_root, "task-one",
            requested_stop_policy=cancel_intents.STOP_POLICY_FINALIZE)
        assert owner_mailbox.write_owner_message(ctx.drive_root,
            REASON_OWNER_REQUESTED_FINALIZATION, "task-one",
            msg_id=owner_stop_control_id(intent), kind=owner_mailbox.KIND_FINALIZE_NOW)
        return call(model_call)

    monkeypatch.setattr(loop, "_call_round_model", after_drain)
    text, usage, trace = loop.run_llm_loop(ctx.messages, tools, ctx.llm, ctx.drive_logs,
        lambda *_args, **_kwargs: None, queue.Queue(), task_id="task-one",
        drive_root=ctx.drive_root, event_queue=events)
    assert text == final["message"]["content"] and len(gateway.creates) == 1
    assert usage["reason_code"] == REASON_OWNER_REQUESTED_FINALIZATION
    assert usage["terminal_origin"] == "model_final"
    assert trace["forced_finalization"]["source"] == "model"
    assert owner.control_reason() is None  # The final call did not re-read its own stop control.
    assert any(message.get("content") == "verified read A" for message in gateway.uploads[0][0]["messages"])


def test_pre_call_graceful_control_after_settled_provider_hiccup_keeps_final_turn(main_call, monkeypatch):
    """A SETTLED earlier provider hiccup leaves the mutable ``_last_llm_error_kind``
    projection behind; only OPEN custody (the round's ``_transport_deaths`` record)
    may deny the owner's pre-call Wrap up its one final turn (E10 CI signature)."""
    from ouroboros.loop_llm_call import TRANSPORT_DEATHS_KEY
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
    from supervisor.owner_stop import owner_stop_control_id

    ctx, gateway, owner, events, _decide, _observations = main_call
    tools = _loop_tools(ctx, owner)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    mid_task = next(message for message in ctx.messages if message.get("role") == "assistant")
    mid_task["content"] = "still working"
    final = result()
    final["message"] = {"role": "assistant", "content": "Verified work summarized after the settled hiccup"}
    gateway.results = [final]
    gateway.dispatch = ["response_received"]
    call = loop._call_round_model
    injected = []

    def after_drain(model_call):
        assert not injected
        injected.append(True)
        # The projection of an attempt that already SETTLED; no open custody record exists.
        model_call.accumulated_usage["_last_llm_error_kind"] = "provider_outcome_unknown"
        assert TRANSPORT_DEATHS_KEY not in model_call.accumulated_usage
        intent = cancel_intents.request_cancel(ctx.drive_root, "task-one",
            requested_stop_policy=cancel_intents.STOP_POLICY_FINALIZE)
        assert owner_mailbox.write_owner_message(ctx.drive_root,
            REASON_OWNER_REQUESTED_FINALIZATION, "task-one",
            msg_id=owner_stop_control_id(intent), kind=owner_mailbox.KIND_FINALIZE_NOW)
        return call(model_call)

    monkeypatch.setattr(loop, "_call_round_model", after_drain)
    text, usage, trace = loop.run_llm_loop(ctx.messages, tools, ctx.llm, ctx.drive_logs,
        lambda *_args, **_kwargs: None, queue.Queue(), task_id="task-one",
        drive_root=ctx.drive_root, event_queue=events)
    assert text == final["message"]["content"] and text != "still working" and len(gateway.creates) == 1
    assert usage["reason_code"] == REASON_OWNER_REQUESTED_FINALIZATION
    assert usage["terminal_origin"] == "model_final"
    assert usage.get("_best_effort_extracted") is True  # outcomes.py lifts this to best-effort, not failed
    assert trace["forced_finalization"]["source"] == "model"


def test_pre_call_wrap_keeps_the_existing_transport_episode_no_call(main_call, monkeypatch):
    from ouroboros import loop_transport
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
    from supervisor.owner_stop import owner_stop_control_id

    ctx, gateway, owner, events, _decide, _observations = main_call
    tools = _loop_tools(ctx, owner)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda *_args: False)
    call, reconcile = loop._call_round_model, loop._reconcile_transport_wait
    passes, episodes = [], []

    def observe_episode(*args, **kwargs):
        episode = reconcile(*args, **kwargs)
        if episode is not None:
            episodes.append(episode)
        return episode

    def before_redial(model_call):
        passes.append(True)
        if len(passes) == 1:
            model_call.accumulated_usage["_last_llm_error_kind"] = "transport_unavailable"
            model_call.accumulated_usage["_last_llm_error"] = "Controlled connection refusal"
            return None, 0.0, model_call.active_context_mode
        assert len(passes) == 2 and isinstance(episodes[0], loop_transport.TransportWaitEpisode)
        intent = cancel_intents.request_cancel(ctx.drive_root, "task-one",
            requested_stop_policy=cancel_intents.STOP_POLICY_FINALIZE)
        assert owner_mailbox.write_owner_message(ctx.drive_root,
            REASON_OWNER_REQUESTED_FINALIZATION, "task-one",
            msg_id=owner_stop_control_id(intent), kind=owner_mailbox.KIND_FINALIZE_NOW)
        return call(model_call)

    monkeypatch.setattr(loop, "_reconcile_transport_wait", observe_episode)
    monkeypatch.setattr(loop, "_call_round_model", before_redial)
    _text, _usage, trace = loop.run_llm_loop(ctx.messages, tools, ctx.llm, ctx.drive_logs,
        lambda *_args, **_kwargs: None, queue.Queue(), task_id="task-one",
        drive_root=ctx.drive_root, event_queue=events)
    assert not gateway.creates
    assert trace["forced_finalization"]["source"] == "transport_unavailable_no_resend"


@pytest.mark.parametrize("interactive", [False, True])
def test_outage_wrap_keeps_older_wire_death_custody_without_summary(tmp_path, monkeypatch, interactive):
    import httpx
    from ouroboros import loop_llm_call, loop_transport
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
    from supervisor.owner_stop import owner_stop_control_id
    from tests.test_transport_death_retry import _LedgerLLM, _ledger, _loop_kwargs, _no_chain

    llm = _LedgerLLM(tmp_path, lambda: httpx.ReadError("controlled wire death"))

    class ControlledLLM:
        def default_model(self):
            return llm.default_model()

        @model_wait.model_waitable
        def chat(self, **kwargs):
            return llm.chat(**kwargs)

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setattr(loop, "_run_cross_model_fallback_chain", _no_chain)
    posted = []

    def release_wait_after_control_check(_seconds, wake_check):
        assert not wake_check() and not posted
        posted.append(True)
        intent = cancel_intents.request_cancel(tmp_path, "t-death",
            requested_stop_policy=cancel_intents.STOP_POLICY_FINALIZE)
        assert owner_mailbox.write_owner_message(tmp_path,
            REASON_OWNER_REQUESTED_FINALIZATION, "t-death",
            msg_id=owner_stop_control_id(intent), kind=owner_mailbox.KIND_FINALIZE_NOW)
        return True

    if interactive:
        monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline",
                            lambda seconds, deadline, **kw: release_wait_after_control_check(seconds, kw["wake_check"]))
    else:
        monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", release_wait_after_control_check)
    kwargs = _loop_kwargs(tmp_path, ControlledLLM(), [])
    kwargs["tools"]._ctx.is_direct_chat = interactive
    with model_wait.task_model_wait_scope(task={"id": "t-death"}, drive_root=tmp_path,
            event_queue=None, worker_slot_held=not interactive) as owner:
        owner.tool_context = kwargs["tools"]._ctx
        _text, usage, trace = loop.run_llm_loop(**kwargs)
    assert posted and llm.calls == 1
    assert [row["state"] for row in _ledger(tmp_path)] == ["reserved", "dispatched", "unresolved"]
    assert loop_llm_call.provider_no_call_source(usage, False)[0] == "provider_outcome_unknown_no_resend"
    if interactive:
        assert trace["forced_finalization"]["control_reason"] == "finalize_requested"
    else:
        assert "owner requested Wrap up" in _text
        assert trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"


@pytest.mark.parametrize("stop,expected_reason", [
    ("wrap", "owner_requested_finalization"), ("deadline", "deadline_local"),
    ("ceiling", "finalization_grace"), ("wrap_unknown", "owner_requested_finalization"),
])
def test_real_main_control_preserves_candidate_without_new_summary(main_call, monkeypatch, stop, expected_reason):
    from ouroboros import config
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
    from supervisor.owner_stop import owner_stop_control_id

    ctx, gateway, owner, events, _decide, _observations = main_call
    tools = _loop_tools(ctx, owner)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    completed = result()
    completed["message"] = {"role": "assistant", "content": "Verified answer retained before the wait"}
    refused = result(outcome="failed", problem={"code": "subscription_window_exhausted", "message": "quota"})
    gateway.results = [completed, refused]
    gateway.dispatch = ["response_received", "not_started"]
    held = []

    def hold(content, limit, trace, actual_tools, *_args):
        held.append(loop._replace_delivery_candidate(actual_tools, limit, trace, content, control="hold_for_verification"))
        if stop == "wrap_unknown":
            gateway.pending = True
        return None

    def request_stop():
        if stop == "deadline":
            owner.task["deadline_at"] = "2000-01-01T00:00:00Z"
            tools._ctx.task_metadata["deadline_at"] = owner.task["deadline_at"]
        elif stop == "ceiling":
            monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: 0)
        else:
            intent = cancel_intents.request_cancel(ctx.drive_root, "task-one", requested_stop_policy=cancel_intents.STOP_POLICY_FINALIZE)
            owner_mailbox.write_owner_message(ctx.drive_root, REASON_OWNER_REQUESTED_FINALIZATION, "task-one",
                                             msg_id=owner_stop_control_id(intent), kind=owner_mailbox.KIND_FINALIZE_NOW)

    def catalog(*args, **kwargs):
        request_stop()
        raise ClaudexorUnavailable("subscription_window_exhausted", "still waiting")

    read = gateway.get_model_operation

    def pending_read(*args, **kwargs):
        request_stop()
        return read(*args, **kwargs)

    monkeypatch.setattr(loop, "_no_tool_final_answer", hold)
    monkeypatch.setattr(ctx.llm, "claudexor_model_catalog", catalog)
    if stop == "wrap_unknown":
        monkeypatch.setattr(gateway, "get_model_operation", pending_read)
    text, usage, trace = loop.run_llm_loop(
        ctx.messages, tools, ctx.llm, ctx.drive_logs, lambda *_args, **_kwargs: None, queue.Queue(),
        task_id="task-one", drive_root=ctx.drive_root, event_queue=events)
    assert len(held) == 1 and text == completed["message"]["content"]
    assert usage["reason_code"] == trace["forced_finalization"]["reason_code"] == expected_reason
    assert len(gateway.accepted_operations) == 2  # Paid answer + interrupted call, never a summary retry.
    assert trace["forced_finalization"]["source"].startswith("model_wait_retained_candidate")
    if stop == "wrap_unknown":
        assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"
        assert trace["forced_finalization"]["physical_attempt_state"] == "unresolved"
        assert len(gateway.cancels) == 1
    else:
        assert trace["forced_finalization"]["physical_attempt_state"] == "released"


def test_hard_cancel_returns_empty_events_to_real_worker_loop_and_keeps_queue_owner(main_call, monkeypatch):
    import sys
    from ouroboros import agent as agent_module, config, extension_loader, platform_layer, process_custody, subagent_runtime, utils
    from supervisor import queue as task_queue, worker_process
    from tests.test_llm_claudexor import ledger

    ctx, gateway, _owner, events, _decide, _observations = main_call
    monkeypatch.setenv("OUROBOROS_IN_WORKER", "1")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setattr(agent_module.OuroborosAgent, "_log_worker_boot_once", lambda *_args: None)
    actual = agent_module.OuroborosAgent(agent_module.Env(ctx.drive_root, ctx.drive_root), event_queue=events)
    actual.llm = ctx.llm
    actual.tools._ctx.context_fit_plan = ctx.context_fit_plan
    actual.tools._ctx.task_model_override = MODEL
    actual.tools._ctx.task_attempt = 1
    actual.tools._ctx.task_metadata = {}

    def prepare(task, _refusal):
        model_wait.current_model_wait().tool_context = actual.tools._ctx
        actual._persist_running_record(task)
        return actual.tools._ctx, ctx.messages, {"budget_remaining": 100}

    monkeypatch.setattr(actual, "_prepare_task_context", prepare)
    monkeypatch.setattr(actual, "_start_task_heartbeat_loop", lambda *_args: None)
    monkeypatch.setattr(subagent_runtime, "apply_task_start_settings_or_disclose", lambda *_args: None)
    monkeypatch.setattr(agent_module, "make_agent", lambda **_kwargs: actual)
    monkeypatch.setattr(worker_process, "_bind_worker_repo_root", lambda *_args: None)
    monkeypatch.setattr(worker_process, "_prepare_worker_task_runtime", lambda: None)
    monkeypatch.setattr(worker_process, "_adopt_published_extensions", lambda *_args: None)
    monkeypatch.setattr(platform_layer, "create_new_session", lambda: None)
    monkeypatch.setattr(process_custody, "start_parent_lifeline", lambda **_kwargs: None)
    monkeypatch.setattr(extension_loader, "reload_all", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(config, "initialize_runtime_mode_baseline", lambda: None)
    monkeypatch.setattr(utils, "set_log_sink", lambda *_args: None)
    monkeypatch.setattr(utils, "get_git_info", lambda *_args: ("fixture", "fixture"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    crashes = []
    monkeypatch.setattr(worker_process, "_log_worker_crash", lambda *args: crashes.append(args))
    gateway.pending = True
    read = gateway.get_model_operation

    def cancel_during_read(*args, **kwargs):
        cancel_intents.request_cancel(ctx.drive_root, "task-one", requested_stop_policy=cancel_intents.STOP_POLICY_IMMEDIATE)
        return read(*args, **kwargs)

    monkeypatch.setattr(gateway, "get_model_operation", cancel_during_read)
    task = task_queue.RUNNING["task-one"]["task"]
    task.update(type="task", text="Continue verified work")
    reads = []

    class Input:
        def get(self):
            reads.append(True)
            if len(reads) == 1:
                return task
            assert "task-one" in task_queue.RUNNING
            assert load_task_result(ctx.drive_root, "task-one")["status"] == "running"
            assert cancel_intents.cancel_pending(ctx.drive_root, "task-one")
            assert not any(row.get("type") == "task_done" for row in list(events.queue))
            assert not crashes
            return None  # End the test's worker only after verifying retained ownership.

    worker_process.worker_main(1, Input(), events, str(ctx.drive_root), str(ctx.drive_root))
    assert len(reads) == 2 and len(gateway.accepted_operations) == 1 and not crashes
    assert ledger(ctx.drive_root)[-1]["state"] == "unresolved"


@pytest.mark.parametrize("stop,deadline,transport,expected", [
    (True, True, False, ["stop"]), (False, True, False, ["deadline"]),
    (False, False, False, ["deadline", "cost"]), (False, True, True, ["cost"]),
])
def test_pre_round_terminal_order_keeps_cost_after_stop_and_deadline(monkeypatch, stop, deadline, transport, expected):
    from ouroboros import loop_round_limits

    calls = []
    result_value = ("finished", {}, {})
    monkeypatch.setattr(loop, "_handle_forced_finalization", lambda *_args: calls.append("stop") or result_value)
    monkeypatch.setattr(loop_round_limits, "_maybe_deadline_local_finalize",
                        lambda *_args: calls.append("deadline") or (result_value if deadline else None))
    monkeypatch.setattr(loop, "_soft_land_exhausted_ceiling", lambda *_args: calls.append("cost") or result_value)
    result_value_actual = loop_round_limits._maybe_early_finalize(
        SimpleNamespace(), SimpleNamespace(_ctx=SimpleNamespace()),
        {"finalize_now": "deadline"} if stop else {},
        cost_ceiling=object(), transport_episode=object() if transport else None)
    assert result_value_actual == result_value and calls == expected


@pytest.mark.serial
@pytest.mark.parametrize("first", ["web", "host"])
@pytest.mark.parametrize("persist_role", [False, True])
def test_web_and_host_wait_decisions_share_saved_replay_and_attempt_fences(live_wait, monkeypatch, first, persist_role):
    from ouroboros import config
    from supervisor import queue as task_queue
    from tests.test_model_wait import _decision_clients

    root, _gateway, _client, owner, _events, _decide = live_wait
    monkeypatch.setattr(config, "SETTINGS_PATH", root / "settings.json")
    row = {"wait_id": "wait-one", "revision": 1, "task_attempt": 1, "state": "waiting", "role": "light"}
    model_wait.mutate_wait(root, "task-one", "wait-one", lambda _: row)
    owner.waits["wait-one"] = dict(row)
    owner.revision = 1
    body = {"request_id": "cross-surface", "decision_id": "model_wait:task-one:wait-one", "revision": 1,
            "action": "switch", "model": MODEL, "credential_profile_id": "account-b",
            "use_local": False, "persist_role": persist_role}
    with _decision_clients(root) as clients:
        response = clients[first](body)
        assert response.status_code == 202 and response.json()["saved"] is persist_role
        path = root / "settings.json"
        saved_bytes = path.read_bytes() if path.exists() else None
        saved_stamp = path.stat().st_mtime_ns if path.exists() else None
        other = "host" if first == "web" else "web"
        replay = clients[other](body)
        assert replay.status_code == 200 and replay.json()["duplicate"] is True
        assert replay.json()["saved"] is persist_role
        assert (path.read_bytes() if path.exists() else None) == saved_bytes
        assert (path.stat().st_mtime_ns if path.exists() else None) == saved_stamp
        owner._drain_controls()
        assert owner.waits["wait-one"]["_action"]["credential_profile_id"] == "account-b"
        task_queue.RUNNING["task-one"]["attempt"] = 2
        stale = clients[other]({**body, "request_id": "stale-attempt"})
        assert stale.status_code == 409 and stale.json()["reason_code"] == "stale_model_wait"
        if persist_role:
            settings = json.loads(saved_bytes)
            assert settings["OUROBOROS_MODEL_LIGHT"] == MODEL
            assert json.loads(settings[MODEL_ACCOUNTS_KEY])["light"] == "account-b"
        else:
            assert saved_bytes is None


@pytest.mark.serial
@pytest.mark.parametrize("surface", ["web", "host"])
def test_wait_decision_preserves_existing_settings_timeout_receipt(live_wait, monkeypatch, surface):
    import threading
    from ouroboros import config
    from ouroboros.gateway import task_model_wait as gateway
    from tests.test_model_wait import _decision_clients

    root, _gateway, _client, _owner, _events, _decide = live_wait
    monkeypatch.setattr(config, "SETTINGS_PATH", root / "settings.json")
    monkeypatch.setattr(config, "get_settings_document_lock_timeout_sec", lambda: 0.02)
    row = {"wait_id": "wait-one", "revision": 1, "task_attempt": 1, "state": "waiting", "role": "light"}
    model_wait.mutate_wait(root, "task-one", "wait-one", lambda _: row)
    release, completed = threading.Event(), threading.Event()
    decide = gateway._decide

    def held(*args, **kwargs):
        try:
            response = decide(*args, **kwargs)
            assert release.wait(5)
            return response
        finally:
            completed.set()

    monkeypatch.setattr(gateway, "_decide", held)
    body = {"request_id": "slow-save", "decision_id": "model_wait:task-one:wait-one", "revision": 1,
            "action": "switch", "model": MODEL, "credential_profile_id": "account-b",
            "use_local": False, "persist_role": True}
    with _decision_clients(root) as clients:
        try:
            response = clients[surface](body)
            assert response.status_code == 503
            assert response.json()["code"] == "settings_save_timeout"
            assert response.json()["saved"] is None
        finally:
            release.set()
            assert completed.wait(5)
