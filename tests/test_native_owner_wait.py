"""Ordinary chat keeps the existing owner-wait capability without a pool slot."""

import queue
from types import SimpleNamespace

import pytest

from ouroboros.owner_mailbox import drain_owner_entries, write_owner_message
from ouroboros.model_wait import TaskModelWait
from ouroboros.owner_wait import direct_owner_wait, wait_after_tools
from ouroboros.task_results import load_task_result
from tests.test_owner_wait import context


def native_context(tmp_path):
    ctx = context(tmp_path)
    ctx.owner_wait_callback = direct_owner_wait
    ctx.event_queue = queue.Queue()
    ctx.model_wait_context = TaskModelWait(
        task={"id": ctx.task_id}, drive_root=tmp_path,
        event_queue=ctx.event_queue, worker_slot_held=False,
    )
    ctx.model_wait_context.tool_context = ctx
    return ctx


def test_native_wait_retains_source_and_leaves_answer_delivery_to_loop(tmp_path, monkeypatch):
    ctx = native_context(tmp_path)
    question = {"type": "send_quiz", "task_id": ctx.task_id, "quiz_id": "q1"}
    ctx.pending_events = [question]
    messages = [{"role": "tool", "tool_call_id": "saved", "content": "Saved form"}]
    waits = []

    def answer(_seconds):
        waits.append(load_task_result(tmp_path, ctx.task_id)["owner_wait"])
        assert ctx.event_queue.get_nowait() == question
        assert write_owner_message(tmp_path, "Continue with that form", ctx.task_id, msg_id="answer")

    monkeypatch.setattr("ouroboros.owner_wait.time.sleep", answer)
    wait_after_tools(ctx, messages, {}, {}, 4, [], set())
    assert len(waits) == 1 and waits[0]["state"] == "waiting"
    after = load_task_result(tmp_path, ctx.task_id)["owner_wait"]
    assert after == {**waits[0], "state": "resumed", "resume_reason": "owner_text"}
    assert after["source_ref"] and "restart_transaction_id" not in after
    assert ctx.pending_events == [] and ctx._owner_wait_requested == ""
    assert ctx._loop_mailbox_seen_ids == set()
    assert messages == [{"role": "tool", "tool_call_id": "saved", "content": "Saved form"}]
    assert drain_owner_entries(tmp_path, ctx.task_id, set())[0]["text"] == "Continue with that form"


def test_peer_mail_wakes_but_does_not_answer_an_open_question(tmp_path, monkeypatch):
    from ouroboros.owner_mailbox import write_task_message
    from ouroboros.owner_quiz import quiz_states, record_asked
    import supervisor.message_bus as mb

    ctx = native_context(tmp_path)
    ctx.current_chat_id = 1
    record_asked(tmp_path, ctx.task_id, quiz_id="q1", question="Which path?",
                 options=[], wait_for_answer=True, chat_id=1)
    frames = []
    bridge = mb.LocalChatBridge()
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(mb, "get_bridge", lambda: bridge)

    def wake(_seconds):
        assert write_task_message(tmp_path, "Here is context", task_id=ctx.task_id,
                                  source_task_id="peer-1", provenance="independent_task")

    monkeypatch.setattr("ouroboros.owner_wait.time.sleep", wake)
    messages = []
    wait_after_tools(ctx, messages, {}, {}, 1, [], set())
    assert load_task_result(tmp_path, ctx.task_id)["owner_wait"]["resume_reason"] == "mail:peer-1"
    assert quiz_states(tmp_path, ctx.task_id)["q1"]["state"] == "open"
    assert "wait_for_answer" not in quiz_states(tmp_path, ctx.task_id)["q1"]
    assert len([frame for frame in frames if frame.get("type") == "quiz_state"]) == 1
    assert len(messages) == 1 and "peer-1" in messages[0]["content"]
    assert "not a confirmed owner answer" in messages[0]["content"]


def test_answer_racing_a_wait_end_never_reopens_a_settled_card(tmp_path, monkeypatch):
    from ouroboros.owner_quiz import quiz_states, record_asked, record_answered
    from ouroboros.owner_wait import announce_wait_ended
    import supervisor.message_bus as mb

    record_asked(tmp_path, "root-1", quiz_id="q1", question="Which path?", options=[],
                 wait_for_answer=True, chat_id=1)
    assert record_answered(tmp_path, "root-1", quiz_id="q1", option_index=None,
                           request_id="r1", comment="Proceed")['ok']
    frames = []
    bridge = mb.LocalChatBridge()
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(mb, "get_bridge", lambda: bridge)
    announce_wait_ended(tmp_path, "root-1", "q1", 1)
    assert quiz_states(tmp_path, "root-1")["q1"]["state"] == "answered"
    assert not [frame for frame in frames if frame.get("type") == "quiz_state"]


def test_answer_before_capacity_grant_replaces_a_stale_timeout(tmp_path):
    from ouroboros.owner_mailbox import KIND_QUIZ_ANSWER
    from ouroboros.owner_wait import _fresh_wake, classify_wake

    ctx = native_context(tmp_path)
    assert classify_wake([{"kind": "hurry", "msg_id": "h"}], "q1") == "hurry"
    assert classify_wake([{"kind": "task_message", "provenance": "ancestor_task"}], "q1") == "mail:unknown"
    assert classify_wake([{"kind": KIND_QUIZ_ANSWER, "msg_id": "quiz_answer:other"}], "q1") == "owner_text"
    assert write_owner_message(tmp_path, "Owner answered", ctx.task_id,
                               msg_id="quiz_answer:q1", kind=KIND_QUIZ_ANSWER)
    assert _fresh_wake(ctx, "q1", "timeout") == "answer"


def _spent_bound(ctx, minutes=5):
    """An absolute stamp already in the past — the bound of a wait that ended."""
    import datetime

    from ouroboros.deadline_utils import utc_now

    ctx._owner_wait_deadline_at = (utc_now() - datetime.timedelta(seconds=1)).isoformat()
    ctx._owner_wait_max_minutes = minutes
    return ctx._owner_wait_deadline_at


def test_bounded_wait_resumes_with_a_notice_that_names_the_recorded_assumption(tmp_path, monkeypatch):
    """escalate(wait_for_answer=True, max_wait_minutes=N): the task parks, then
    continues on its own bound with a HOST notice (never owner-marked content)
    that keeps the card open and says what to do about the unanswered fork."""
    from ouroboros.owner_quiz import record_asked

    ctx = native_context(tmp_path)
    record_asked(tmp_path, ctx.task_id, quiz_id="q1", question="Proceed?",
                 options=["Yes", "No"], wait_for_answer=True, chat_id=1,
                 assumption="Keep the current layout", max_wait_minutes=5)
    deadline = _spent_bound(ctx)
    monkeypatch.setattr("ouroboros.owner_wait.time.sleep",
                        lambda _seconds: pytest.fail("a spent bound must not keep sleeping"))
    import supervisor.message_bus as mb

    frames: list = []
    bridge = mb.LocalChatBridge()
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(mb, "get_bridge", lambda: bridge)
    ctx.current_chat_id = 1
    messages = []
    wait_after_tools(ctx, messages, {}, {}, 2, [], set())

    row = load_task_result(tmp_path, ctx.task_id)["owner_wait"]
    # No new wait state: the row RESUMES, with the reason as an additive field.
    assert row["state"] == "resumed" and row["resume_reason"] == "timeout"
    # The DIRECT lane announces the closed bound too (opus round 4): the card's projection
    # stops saying "waiting" and the live card gets the additive frame; it stays open.
    from ouroboros.owner_quiz import quiz_states

    block = quiz_states(tmp_path, ctx.task_id)["q1"]
    assert block["state"] == "open" and "wait_for_answer" not in block and block["wait_ended_at"]
    [frame] = [f for f in frames if f.get("type") == "quiz_state"]
    assert frame["quiz_id"] == "q1" and frame["state"] == "open" and frame["wait_for_answer"] is False
    assert frame["chat_id"] == 1  # the card's chat, exactly as the pool's grant names it
    assert row["wait_deadline_at"] == deadline and row["wait_max_minutes"] == 5
    [notice] = messages
    assert notice["role"] == "user" and notice["content"].startswith("[SYSTEM NOTICE]")
    assert "within 5 minutes" in notice["content"]
    assert "The question card stays open" in notice["content"]
    assert "Proceed under your stated assumption: Keep the current layout" in notice["content"]
    assert not drain_owner_entries(tmp_path, ctx.task_id, set())  # no fabricated answer
    assert ctx._owner_wait_requested == "" and ctx._owner_wait_deadline_at == ""
    assert ctx._owner_wait_max_minutes == 0


def test_bounded_wait_without_an_assumption_says_silence_is_not_consent(tmp_path, monkeypatch):
    ctx = native_context(tmp_path)
    _spent_bound(ctx, minutes=15)
    monkeypatch.setattr("ouroboros.owner_wait.time.sleep", lambda _seconds: None)
    messages = []
    wait_after_tools(ctx, messages, {}, {}, 1, [], set())
    assert "within 15 minutes" in messages[0]["content"]
    assert "No assumption was recorded; no answer is not consent" in messages[0]["content"]


def test_a_spent_bound_travels_in_the_durable_checkpoint_not_the_live_context(tmp_path, monkeypatch):
    """The bound is an ABSOLUTE stamp in the saved wait, so a cold continuation
    after a planned restart resumes the SAME bound instead of granting it again."""
    from types import SimpleNamespace

    from ouroboros.owner_wait import checkpoint_owner_wait, direct_owner_wait

    ctx = native_context(tmp_path)
    deadline = _spent_bound(ctx, minutes=7)
    checkpoint = checkpoint_owner_wait(ctx, [], {}, {}, 1, [], set())
    assert checkpoint["wait_deadline_at"] == deadline and checkpoint["wait_max_minutes"] == 7

    successor = native_context(tmp_path)  # a fresh process: no ctx wait attributes
    assert not getattr(successor, "_owner_wait_deadline_at", "")
    monkeypatch.setattr("ouroboros.owner_wait.time.sleep",
                        lambda _seconds: pytest.fail("the restored bound was already spent"))
    assert direct_owner_wait(successor, dict(checkpoint)) == "timeout"
    assert load_task_result(tmp_path, ctx.task_id)["owner_wait"]["resume_reason"] == "timeout"
    assert isinstance(successor, SimpleNamespace)


def test_an_answer_before_the_bound_resumes_without_a_timeout_notice(tmp_path, monkeypatch):
    import datetime

    from ouroboros.deadline_utils import utc_now

    ctx = native_context(tmp_path)
    ctx._owner_wait_deadline_at = (utc_now() + datetime.timedelta(minutes=30)).isoformat()
    ctx._owner_wait_max_minutes = 30

    def answer(_seconds):
        assert write_owner_message(tmp_path, "Use the second option", ctx.task_id, msg_id="a1")

    monkeypatch.setattr("ouroboros.owner_wait.time.sleep", answer)
    messages = []
    wait_after_tools(ctx, messages, {}, {}, 1, [], set())
    assert messages == []  # the owner answered; the loop delivers it as usual
    assert load_task_result(tmp_path, ctx.task_id)["owner_wait"]["resume_reason"] == "owner_text"


@pytest.mark.parametrize("reason", ["cancelled", "finalize_requested", "deadline", "absolute_ceiling"])
def test_a_control_reason_outranks_a_spent_bound(tmp_path, monkeypatch, reason):
    """Ordering is load-bearing: Stop, the task deadline and the absolute ceiling
    are consulted BEFORE the soft bound, so a timeout notice can never displace
    them (the loop acts on the control reason instead)."""
    ctx = native_context(tmp_path)
    _spent_bound(ctx)
    monkeypatch.setattr(ctx.model_wait_context, "control_reason", lambda: reason)
    messages = []
    wait_after_tools(ctx, messages, {}, {}, 1, [], set())
    row = load_task_result(tmp_path, ctx.task_id)["owner_wait"]
    assert row["state"] == "resumed" and row["resume_reason"] == f"control:{reason}"
    assert messages == []


@pytest.mark.parametrize("reason", ["cancelled", "finalize_requested", "deadline", "absolute_ceiling"])
def test_native_wait_rejoins_existing_control_without_waiting_for_answer(tmp_path, monkeypatch, reason):
    ctx = native_context(tmp_path)
    monkeypatch.setattr(ctx.model_wait_context, "control_reason", lambda: reason)
    monkeypatch.setattr("ouroboros.owner_wait.time.sleep", lambda _: pytest.fail("control did not release wait"))
    wait_after_tools(ctx, [], {}, {}, 1, [], set())
    assert load_task_result(tmp_path, ctx.task_id)["owner_wait"]["state"] == "resumed"
    assert not drain_owner_entries(tmp_path, ctx.task_id, set())  # no fabricated owner answer


def test_native_agent_factory_binds_existing_wait_callback(tmp_path, monkeypatch):
    from supervisor import workers

    actor = SimpleNamespace()
    monkeypatch.setattr("ouroboros.agent.make_agent", lambda **_: actor)
    monkeypatch.setattr(workers, "REPO_DIR", tmp_path)
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "get_event_q", queue.Queue)
    assert workers._get_chat_agent() is actor
    assert actor.owner_wait_callback is direct_owner_wait


def test_direct_root_can_request_wait_through_the_existing_quiz(tmp_path):
    from ouroboros.tools.core_artifacts import _escalate
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="native",
                      is_direct_chat=True, current_chat_id=1, event_queue=queue.Queue())
    ctx.owner_wait_callback = direct_owner_wait
    result = _escalate(ctx, question="Continue?", options=[{"label": "Yes"}, {"label": "No"}],
                       wait_for_answer=True)
    assert result.startswith("OK:"), result
    event = ctx.event_queue.get_nowait()
    assert event["type"] == "send_quiz" and event["wait_for_answer"] is True
    assert ctx._owner_wait_requested == event["quiz_id"]
    assert load_task_result(tmp_path, "native")["owner_quiz"][event["quiz_id"]]["wait_for_answer"] is True


def test_escalate_records_the_bound_and_says_what_the_wait_promises(tmp_path):
    """The receipt is the asker's only view of what it agreed to: a bounded wait
    says so, and an optional card no longer claims it expires with the task."""
    from ouroboros.tools.core_artifacts import _escalate
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="native",
                      is_direct_chat=True, current_chat_id=1, event_queue=queue.Queue())
    ctx.owner_wait_callback = direct_owner_wait
    result = _escalate(ctx, question="Continue?", options=[{"label": "Yes"}, {"label": "No"}],
                       wait_for_answer=True, max_wait_minutes=20)
    assert "waits up to 20 minutes after this tool batch, then continues with a notice" in result
    quiz_id = ctx.event_queue.get_nowait()["quiz_id"]
    block = load_task_result(tmp_path, "native")["owner_quiz"][quiz_id]
    assert block["max_wait_minutes"] == 20 and block["chat_id"] == 1
    assert ctx._owner_wait_max_minutes == 20 and ctx._owner_wait_deadline_at
    assert ctx._owner_wait_requested == quiz_id

    bad = _escalate(ctx, question="Continue?", options=[{"label": "Yes"}, {"label": "No"}],
                    wait_for_answer=True, max_wait_minutes=-3)
    # The refusal names the repair, and the effect clause says the quiz was not sent.
    assert bad.startswith("⚠️ QUIZ_WAIT_BOUND_INVALID") and "omit it for an unbounded wait" in bad
    assert "The quiz was not sent." in bad

    optional = _escalate(ctx, question="Which one?", options=["a", "b"], assumption="a meanwhile")
    assert "the card stays answerable after this task ends" in optional
    assert "a later answer reaches this chat as an ordinary owner message" in optional
    assert "max_wait_minutes ignored" not in optional


def test_optional_question_with_a_habit_filled_bound_is_asked_and_says_so(tmp_path):
    """A schema-filling model names max_wait_minutes on a question it does not wait for. That
    is the documented default spelled out, not a different request: the card is asked, no wait
    starts, and the receipt discloses the ignored argument."""
    from ouroboros.tools.core_artifacts import _escalate
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="native",
                      is_direct_chat=True, current_chat_id=1, event_queue=queue.Queue())
    ctx.owner_wait_callback = direct_owner_wait
    for named_default in (0, 1, 60):
        result = _escalate(ctx, question="Which one?", options=["a", "b"],
                           assumption="a meanwhile", max_wait_minutes=named_default)
        assert result.startswith("OK: quiz "), result
        assert "max_wait_minutes ignored: it applies only to wait_for_answer=true." in result
        event = ctx.event_queue.get_nowait()
        assert event["type"] == "send_quiz" and "wait_for_answer" not in event
        block = load_task_result(tmp_path, "native")["owner_quiz"][event["quiz_id"]]
        assert "max_wait_minutes" not in block
    assert not getattr(ctx, "_owner_wait_requested", "")


def test_a_bounded_wait_does_not_lend_its_bound_to_the_next_question_of_the_batch(tmp_path):
    """One tool batch shares one wait, named after its LAST waiting question. A bound the
    earlier question asked for must not survive into a wait that asked for none."""
    from ouroboros.tools.core_artifacts import _escalate
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="native",
                      is_direct_chat=True, current_chat_id=1, event_queue=queue.Queue())
    ctx.owner_wait_callback = direct_owner_wait
    _escalate(ctx, question="First?", options=["a", "b"], wait_for_answer=True, max_wait_minutes=20)
    assert ctx._owner_wait_max_minutes == 20 and ctx._owner_wait_deadline_at
    second = _escalate(ctx, question="Second?", options=["a", "b"], wait_for_answer=True)
    assert "the task waits after this tool batch" in second and "up to" not in second
    assert ctx._owner_wait_max_minutes == 0 and ctx._owner_wait_deadline_at == ""
    from ouroboros.owner_wait import _wait_bound_fields

    assert _wait_bound_fields(ctx) == {}, "the parked wait carries no bound either"
    assert ctx._owner_wait_requested == list(ctx.event_queue.queue)[-1]["quiz_id"]


@pytest.mark.parametrize("required", [False, True])
def test_answer_frame_describes_only_work_that_continued(required):
    from ouroboros.gateway.task_decision import _quiz_answer_frame

    block = {"quiz_id": "q", "question": "Proceed?", "options": ["Yes", "No"],
             "assumption": "Keep the current layout", "wait_for_answer": required}
    frame = _quiz_answer_frame(block, 0, "Proceed with the saved form")
    assert "Proceed with the saved form" in frame
    assert ("You continued under the assumption" in frame) is not required


@pytest.mark.parametrize("cold", [False, True])
def test_expired_wait_clock_precedes_post_tool_budget(tmp_path, monkeypatch, cold):
    """A saved tail may have spent its budget, but an expired clock keeps its cause."""
    import json
    import time
    from ouroboros import loop, model_wait, owner_wait, task_pacing
    from tests.test_owner_wait_cold_loop import cold_registry
    from tests.test_loop_transport_wait import _loop_kwargs

    def no_network(*_args, **_kwargs):
        pytest.fail("synthetic wait must never contact a provider")
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", no_network)
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat_async", no_network)
    monkeypatch.setattr("ouroboros.pricing._fetch_live_rows", no_network)
    monkeypatch.setenv("OUROBOROS_TASK_ABS_CEILING_SEC", "21600")
    registry = cold_registry(tmp_path, monkeypatch, task_pacing.CostCeiling(state="active", ceiling_usd=9.0))
    ctx = registry._ctx
    ctx.is_direct_chat, ctx.current_chat_id, ctx.event_queue = True, 1, queue.Queue()
    calls = []

    with model_wait.task_model_wait_scope(task={"id": ctx.task_id, "_is_direct_chat": True},
            drive_root=tmp_path, event_queue=ctx.event_queue, worker_slot_held=False) as controller:
        ctx.model_wait_context, controller.tool_context = controller, ctx

        def expire_and_wait(context, checkpoint):
            controller.started_monotonic = time.monotonic() - 21601
            assert controller.control_reason() == "absolute_ceiling"
            owner_wait.direct_owner_wait(context, checkpoint)
        ctx.owner_wait_callback = expire_and_wait
        if not cold:
            ctx.owner_wait_resume, ctx._owner_wait_requested = None, ""

        def dispatch(call, _disposition, **_kwargs):
            calls.append("ordinary_send")
            call.accumulated_usage["cost"] = 10.0
            return {"role": "assistant", "content": "", "tool_calls": [{
                "id": "ask", "type": "function", "function": {
                    "name": "escalate", "arguments": json.dumps({"question": "Continue?",
                        "options": [{"label": "Yes"}, {"label": "No"}], "wait_for_answer": True})}}]}, 0.0
        monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
        monkeypatch.setattr(loop, "_call_forced_model_once", lambda *_a, **_k: pytest.fail("expired clock entered a paid finalizer"))
        monkeypatch.setattr(loop, "_finish_tool_round_budget", lambda *_a, **_k: pytest.fail("expired clock entered budget tail"))
        _text, usage, trace = loop.run_llm_loop(**{
            **_loop_kwargs(tmp_path, registry, []), "event_queue": ctx.event_queue})

    assert calls == ([] if cold else ["ordinary_send"])
    assert usage["reason_code"] != "budget_exhausted"
    assert trace["forced_finalization"]["control_reason"] == "absolute_ceiling"
