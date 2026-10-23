"""Production Presence question parks: real loop, leases, bus, HTTP and controls.

Only inference/review transport, Safety and focus allowance are deterministic
fixtures. No provider calls, live server, or installed data are used.
"""
import contextvars
import datetime as dt
import functools
import json
import queue
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros.cancel_intents import request_cancel
from ouroboros.owner_quiz import quiz_states
from ouroboros.presence_runner import PresenceTurnError, presence_event_identity, presence_turn_task_id, run_presence_turn
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_presence_continuation_bootstrap import finish, install_bootstrap_harness, tool_call, wait_for
from tests.test_presence_owner_contact import _answer, _ask, _run, contact_runtime as _contact_runtime, focus  # noqa: F401

contact_runtime = _contact_runtime
pytestmark = pytest.mark.serial


@pytest.mark.parametrize("request_id", ["same-question", ""], ids=["explicit-key", "implicit-presence-key"])
def test_ended_wait_reask_refuses_same_identity_and_new_key_can_wait(contact_runtime, request_id):
    from ouroboros.gateway.history import _assemble_history_response
    from ouroboros.model_wait import TaskModelWait
    from ouroboros.owner_wait import wait_after_tools

    r = contact_runtime

    def during(ctx, _task, _agent):
        args = {"question": "May I take on the requested work?", "options": [{"label": "Research only"}],
                "wait_for_answer": True, **({"request_id": request_id} if request_id else {})}
        def ask(**changes):
            return ctx._consumer_tools.execute_result("escalate", {**args, **changes})

        assert ask().status == "ok"
        qid = ctx._owner_wait_requested
        before = quiz_states(r.root, ctx.task_id)
        assert ask().status == "ok", "identical retry before the first wait remains valid"
        assert quiz_states(r.root, ctx.task_id) == before
        first_frame = next(f for f in r.frames if f.get("type") == "quiz")
        ctx.task_attempt, ctx.task_started_at = 1, time.time()
        ctx.model_wait_context = TaskModelWait(task=_task, drive_root=r.root, event_queue=None, worker_slot_held=False)
        ctx.model_wait_context.tool_context = ctx
        ctx._owner_wait_deadline_at = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)).isoformat()
        wait_after_tools(ctx, [], {}, {}, 1, [], set())
        r.pump()
        ended = quiz_states(r.root, ctx.task_id)[qid]
        assert ended["state"] == "open" and ended["wait_ended_at"] and not ended.get("wait_for_answer")
        wait = load_task_result(r.root, ctx.task_id)["owner_wait"]
        assert wait["state"] == "resumed" and wait["resume_reason"] == "timeout"
        end_frame = next(f for f in r.frames if f.get("type") == "quiz_state" and f.get("wait_for_answer") is False)
        frames, events = list(r.frames), list(ctx.pending_events)
        refused = ask()
        assert refused.code == "TOOL_ARG_ERROR", refused
        assert refused.status == "error" and "QUIZ_REQUEST_CONFLICT" in refused.text
        assert "use a new request_id to wait again" in refused.text
        assert quiz_states(r.root, ctx.task_id)[qid] == ended
        assert r.frames == frames and ctx.pending_events == events and r.bus.empty()
        assert not ctx._owner_wait_requested and not ctx._owner_wait_deadline_at
        wait_after_tools(ctx, [], {}, {}, 2, [], set())
        assert load_task_result(r.root, ctx.task_id)["owner_wait"] == wait
        assert ask(request_id="another-wait").status == "ok"
        new_id = ctx._owner_wait_requested
        blocks = quiz_states(r.root, ctx.task_id)
        assert new_id != qid and blocks[qid] == ended
        assert blocks[new_id]["wait_for_answer"] and not blocks[new_id].get("wait_ended_at")
        assert blocks[new_id]["published_at"]

        def observe_new_wait():
            try:
                wait_for(lambda: (load_task_result(r.root, ctx.task_id).get("owner_wait") or {}).get("state") == "waiting",
                         timeout=10, what="new request parked")
                assert load_task_result(r.root, ctx.task_id)["owner_wait"]["quiz_id"] == new_id
                history = json.loads(_assemble_history_response(r.root, 1, 20, 0))
                run = subprocess.run([shutil.which("node"), "web/tests/presence_contact_consumer.mjs"],
                    input=json.dumps({"reask": {"first": first_frame, "ended": end_frame,
                        "fresh": next(f for f in r.frames if f.get("quiz_id") == new_id), "history": history}}),
                    text=True, capture_output=True, timeout=15, cwd=Path(__file__).resolve().parents[1])
                assert run.returncode == 0, run.stdout + run.stderr
            finally:
                # The original question still accepts an owner answer while the new one waits.
                assert _answer(r, ctx.task_id, qid).status_code == 200

        with ThreadPoolExecutor(max_workers=1) as observer:
            presentation = observer.submit(observe_new_wait)
            wait_after_tools(ctx, [], {}, {}, 3, [], set())
            presentation.result(timeout=20)
        assert quiz_states(r.root, ctx.task_id)[qid]["state"] == "answered"
        assert not ctx._owner_wait_requested
        assert load_task_result(r.root, ctx.task_id)["owner_wait"]["quiz_id"] == new_id
        new_question = quiz_states(r.root, ctx.task_id)[new_id]
        assert new_question["wait_ended_at"] and new_question["state"] == "open"
        assert "answered_index" not in new_question

    _run(r, during)


def test_presence_live_and_history_rows_validate_against_published_contract(contact_runtime):
    from copy import deepcopy

    from jsonschema import Draft202012Validator
    from ouroboros.gateway.history import _assemble_history_response
    from ouroboros.gateway.schema import contract_document
    from ouroboros.owner_quiz import mark_wait_ended
    from ouroboros.presence_owner_contact import flush_owner_events
    from ouroboros.tools.control_runtime import _send_user_message

    r = contact_runtime
    schemas = contract_document()["schemas"]

    def check(row, schema, *, ended=False):
        validator = Draft202012Validator(schema)
        validator.validate(row)
        assert row["presence_contact"]["url"] == r.event.message["url"]
        malformed = deepcopy(row)
        malformed["presence_contact"]["task_id"] = 7
        assert not validator.is_valid(malformed), "the consumer must reject a malformed source-work identity"
        malformed["presence_contact"] = "not an object"
        assert not validator.is_valid(malformed)
        if ended:
            assert row["wait_ended_at"]
            assert not validator.is_valid({**row, "wait_ended_at": False})
        legacy = {key: value for key, value in row.items() if key not in {"presence_contact", "wait_ended_at"}}
        validator.validate(legacy)  # both additions remain optional

    def during(ctx, _task, _agent):
        assert _send_user_message(ctx, "Private source report", destination="main").startswith("OK:")
        assert _ask(ctx, wait_for_answer=True).startswith("OK:")
        qid = ctx._owner_wait_requested
        # Publication after an ended wait is a real producer of both additive fields.
        assert mark_wait_ended(r.root, ctx.task_id, qid)
        flush_owner_events(r.root, ctx.pending_events)
        r.pump()
        check(next(f for f in r.frames if f.get("type") == "chat"), schemas["ChatOutbound"])
        check(next(f for f in r.frames if f.get("type") == "quiz"), schemas["QuizOutbound"], ended=True)
        history = json.loads(_assemble_history_response(r.root, 1, 20, 0))
        schema = schemas["ChatHistoryResponse"]
        Draft202012Validator(schema).validate(history)
        row_schema = schema["properties"]["messages"]["items"]
        check(next(m for m in history["messages"] if m.get("presence_contact")), row_schema)
        question = next(m for m in history["messages"] if m.get("quiz"))
        check(question["quiz"], row_schema["properties"]["quiz"], ended=True)
        malformed = deepcopy(history)
        next(m for m in malformed["messages"] if m.get("quiz"))["quiz"]["presence_contact"]["binding_id"] = 3
        assert not Draft202012Validator(schema).is_valid(malformed)

    _run(r, during, live=False)


def start(r, h):
    event = replace(r.event, continuation_version=1)
    return h.executions.start_thread(
        presence_turn_task_id(r.admission.binding_id, event.source_event_id),
        identity=presence_event_identity(r.admission.binding_id, event),
        admit=lambda: h.gate.acquire(event.conversation_key),
        run=lambda lease: run_presence_turn(admission=r.admission, event=event, repo_dir=h.repo,
            drive_root=r.root, agent_factory=h.factory, admitted=lease), context=contextvars.copy_context())[0]


@pytest.mark.parametrize("wake", ["answer", "timeout", "stop"])
def test_required_question_lends_and_reacquires_in_production_loop(contact_runtime, monkeypatch, wake):
    r = contact_runtime
    h = install_bootstrap_harness(monkeypatch, r.root, review_mode="off")
    h.script = [
        {"content": None, "tool_calls": [tool_call("escalate", {
            "question": "May I research this?", "options": [{"label": "Research only"}],
            "wait_for_answer": True, "request_id": "required"}, "ask")]},
        finish("done", message="The private decision was considered."),
    ]
    # A real absolute wait bound, advanced by the fixture without waiting a minute.
    if wake == "timeout":
        from ouroboros import owner_wait
        checkpoint = owner_wait.checkpoint_owner_wait

        def bounded(*args, **kwargs):
            row = checkpoint(*args, **kwargs)
            row["wait_deadline_at"] = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)).isoformat()
            return row

        monkeypatch.setattr(owner_wait, "checkpoint_owner_wait", bounded)
    execution = start(r, h)
    try:
        initial = execution.initial.result(timeout=40)
        assert initial.status == "continuing" and not initial.text
        tid = initial.task_id
        qid, = quiz_states(r.root, tid)
        ctx = h.agents[0].tools._ctx
        lease = h.agents[0].review_wait_callback.args[0].lease
        assert ctx.owner_wait_callback is None
        if wake != "timeout":
            wait_for(lambda: (load_task_result(r.root, tid).get("owner_wait") or {}).get("state") == "waiting",
                     timeout=15, what="private owner question parked")
            r.pump()
            assert quiz_states(r.root, tid)[qid]["publication_chat_id"] == 1
            assert not lease.held() and h.calls == 1
            # The single active slot AND this exact conversation are available to another turn.
            until = time.monotonic() + 2
            borrowed = h.gate.acquire_resources(r.event.conversation_key, lambda: time.monotonic() > until)
            assert borrowed is not None
            borrowed.close()
            if wake == "answer":
                response = _answer(r, tid, qid)
                assert response.status_code == 200, response.text
                assert not response.json().get("forwarded")
            else:
                request_cancel(r.root, tid, reason="owner stop", source="test")
        try:
            final = execution.result.result(timeout=40)
        except PresenceTurnError as exc:
            assert wake == "stop" and exc.code == "presence_turn_cancelled"
            final = SimpleNamespace(text="")
        r.pump()
        row = load_task_result(r.root, tid)
        block = quiz_states(r.root, tid)[qid]
        assert row["owner_wait"]["state"] != "waiting"
        if wake == "answer":
            assert block["state"] == "answered" and h.calls == 2
            assert "PRESENCE CONVERSATION RESUMED" in str(h.inputs[-1])
            assert "Read the public context only" in str(h.inputs[-1])
            assert block["delivery"]["state"] == "delivered"
        else:
            assert block["state"] == "expired_terminal" and "answered_index" not in block
            assert not block.get("wait_for_answer")
            assert any(f.get("state") == "expired_terminal" and f.get("chat_id") == 1 for f in r.frames)
            if wake == "stop":
                assert h.calls == 1 and not final.text
            else:
                assert h.calls == 2 and "PRESENCE CONVERSATION RESUMED" in str(h.inputs[-1])
                assert "not consent" in str(h.inputs[-1])
                assert any(f.get("state") == "open" and f.get("wait_for_answer") is False
                           and f.get("chat_id") == 1 for f in r.frames)
    finally:
        if h.agents and not execution.result.done():
            request_cancel(r.root, h.agents[0]._current_task_id, reason="fixture teardown", source="test")
        try:
            execution.result.result(timeout=40)
        except PresenceTurnError as exc:
            assert exc.code == "presence_turn_cancelled"
        assert not h.executions.live()


def test_inline_controls_keep_their_existing_capability_boundary(contact_runtime):
    from ouroboros.budget_pause import request_pause
    from ouroboros.model_detach import arm_detach
    from ouroboros.tools.control_task_results import _await_as_sleep

    def during(ctx, _task, _agent):
        assert ctx.owner_wait_callback is None and callable(ctx.review_wait_callback)
        usage = {}
        request_pause(SimpleNamespace(tools=SimpleNamespace(_ctx=ctx), accumulated_usage=usage),
                      rail="dispatch_refused", scope="root", reason_text="allowance spent")
        assert usage["exact_pause_unavailable"] == "no_continuation_owner"
        assert not load_task_result(ctx.drive_root, ctx.task_id).get("budget_pause")
        assert arm_detach(ctx, "continue elsewhere")["error"] == "no_continuation_owner"
        for mode in ("warm", "cold"):
            assert "no continuation owner" in _await_as_sleep(ctx, mode, wake_after_sec=5)
        assert not getattr(ctx, "_model_sleep", None)

    _run(contact_runtime, during)


def test_focus_allowance_refusal_ends_without_budget_pause(contact_runtime, monkeypatch):
    from ouroboros import loop, usage_accounting as ua

    r = contact_runtime
    h = install_bootstrap_harness(monkeypatch, r.root, review_mode="off")
    scopes = []

    def refused_transport(*_args, **_kwargs):
        scope = ua.current_usage_scope()
        scopes.append(scope)
        # Settle deterministic prior usage against the real wallet, then ask
        # its reservation owner for the next dispatch. No provider is invoked.
        prior = ua.reserve_attempt(ua.AttemptRequest(model="fixture/prior", provider="fixture",
                                                    reservation_usd=scope.root_limit_usd))
        ua.mark_dispatched(prior)
        ua.settle_attempt(prior, {}, cost_usd=scope.root_limit_usd, cost_final=True)
        ua.reserve_attempt(ua.AttemptRequest(model="fixture/main", provider="fixture", reservation_usd=.01))
        pytest.fail("an exhausted wallet admitted another model dispatch")

    monkeypatch.setattr(loop, "call_llm_with_retry", refused_transport)
    execution = start(r, h)
    final = execution.result.result(timeout=40)
    row = load_task_result(r.root, final.task_id)
    assert len(scopes) == 1 and h.calls == 0
    assert row["outcome_axes"]["execution"]["reason_code"] == "budget_exhausted"
    assert row["status"] == "failed" and not row.get("budget_pause")
    assert row["metadata"]["focus"] and not final.text
    assert not h.executions.live()
    until = time.monotonic() + 2
    released = h.gate.acquire_resources(r.event.conversation_key, lambda: time.monotonic() > until)
    assert released is not None
    released.close()


def test_pooled_source_question_keeps_worker_wait_and_owner_answer(contact_runtime):
    from ouroboros.owner_wait import wait_after_tools, worker_owner_wait
    from ouroboros.presence_runner import _build_task
    from ouroboros.tools.core_artifacts import _escalate
    from ouroboros.tools.registry import ToolContext
    from supervisor import queue as task_queue

    r = contact_runtime
    task = _build_task(r.admission, r.event, drive_root=r.root, staged_files=(), physical_task_id="pooled")
    write_task_result(r.root, "pooled", "running", metadata=task["metadata"], chat_id=task["chat_id"])
    task_queue.RUNNING["pooled"] = {"task": task, "worker_id": 2}
    ctx = ToolContext(repo_dir=r.root, drive_root=r.root, task_id="pooled", task_metadata=task["metadata"],
                      current_chat_id=task["chat_id"])
    ctx.task_attempt, ctx.task_started_at, ctx._loop_mailbox_seen_ids = 1, time.time(), set()
    commands, events, done = queue.Queue(), queue.Queue(), threading.Event()
    ctx.owner_wait_callback = functools.partial(worker_owner_wait, 2, commands, events)
    assert _escalate(ctx, question="May I investigate?", options=[{"label": "Research only"}],
                     wait_for_answer=True).startswith("OK:")
    qid, = quiz_states(r.root, "pooled")
    thread = threading.Thread(target=lambda: (wait_after_tools(ctx, [], {}, {}, 1, [], set()), done.set()))
    thread.start()
    identity = {}
    try:
        r.bus.put(events.get(timeout=5))  # real worker flushes the deferred quiz before parking
        r.pump()
        park = events.get(timeout=5)
        assert park["phase"] == "park" and quiz_states(r.root, "pooled")[qid]["publication_chat_id"] == 1
        identity = {key: park[key] for key in ("type", "task_id", "task_attempt", "wait_id")}
        commands.put({**identity, "phase": "parked"})
        assert _answer(r, "pooled", qid).status_code == 200
        resume = events.get(timeout=5)
        assert resume["phase"] == "resume" and resume["resume_reason"] == "answer"
        assert not done.is_set(), "answer alone cannot grant pooled capacity"
        commands.put({**identity, "phase": "resume_granted"})
        thread.join(timeout=5)
        assert done.is_set() and quiz_states(r.root, "pooled")[qid]["state"] == "answered"
    finally:
        if identity:
            commands.put({**identity, "phase": "resume_granted"})
        request_cancel(r.root, "pooled", reason="fixture teardown", source="test")
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_answer_storage_failure_is_unknown_and_never_holds_queue_lock(contact_runtime, monkeypatch):
    from ouroboros import task_results
    from supervisor import queue as q

    r = contact_runtime

    def during(ctx, _task, _agent):
        assert _ask(ctx).startswith("OK:")
        qid, = quiz_states(r.root, ctx.task_id)
        original = task_results.load_task_result
        probes = []

        def unavailable(*args, **kwargs):
            probes.append(q._queue_lock._is_owned())
            assert not probes[-1], "Presence task storage was probed under Q"
            raise OSError("fixture store temporarily unavailable")

        with monkeypatch.context() as scoped:
            scoped.setattr(task_results, "load_task_result", unavailable)
            response = _answer(r, ctx.task_id, qid)
        assert probes and response.status_code == 503
        assert response.json()["reason_code"] == "quiz_answer_unconfirmed"
        assert response.json()["state"] == "open"
        assert quiz_states(r.root, ctx.task_id)[qid]["state"] == "open"
        assert not any(f.get("state") == "answered" for f in r.frames)
        assert r.bridge._inbox.empty()
        assert original(r.root, ctx.task_id)["status"] == "running"
        assert _answer(r, ctx.task_id, qid).status_code == 200

    _run(r, during)


@pytest.mark.parametrize("carrier", [{"presence_binding_authority": {"binding_id": "related"}}, {"presence": {}}])
def test_unattributed_root_keeps_ordinary_question_and_no_private_report(contact_runtime, carrier):
    from ouroboros.tools.core_artifacts import _escalate
    from ouroboros.tools.control_runtime import _send_user_message
    from ouroboros.tools.registry import ToolContext

    r = contact_runtime
    write_task_result(r.root, "related-root", "running", metadata=carrier, chat_id=1)
    ctx = ToolContext(repo_dir=r.root, drive_root=r.root, task_id="related-root", current_chat_id=1,
                      task_metadata=carrier, event_queue=r.bus)
    assert _escalate(ctx, question="Proceed?", assumption="Continue reading").startswith("OK:")
    r.pump()
    block, = quiz_states(r.root, ctx.task_id).values()
    assert block["published_at"] and not block.get("presence_contact")
    assert "MAIN_NOTICE_BLOCKED" in _send_user_message(ctx, "No attributable origin", destination="main")


def test_unsettled_presence_attempt_does_not_expire_question(contact_runtime):
    r, saved = contact_runtime, {}

    def during(ctx, _task, _agent):
        assert _ask(ctx).startswith("OK:")
        saved.update(tid=ctx.task_id, qid=next(iter(quiz_states(r.root, ctx.task_id))))
        raise RuntimeError("fixture ends without a terminal write")

    with pytest.raises(RuntimeError, match="without a terminal"):
        _run(r, during)
    assert load_task_result(r.root, saved["tid"])["status"] == "running"
    assert quiz_states(r.root, saved["tid"])[saved["qid"]]["state"] == "open"
