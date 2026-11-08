"""A Wrap up that keeps the round's next tool from starting still gets its one tool-less final turn (E10).

``finalize_then_cancel`` owes exactly one tool-less final turn (ARCHITECTURE §5 Cancellation). When the
Wrap up lands while a model call is on the wire, the call completes and the next tool's admission
(``owner_pause.run_tool_handler``) refuses to start it with ``PhysicalDispatchInterrupted`` — positive
evidence that nothing was sent. The loop's outer exit hands that refusal to the model-wait rails, which
denied the final turn because the interruption had a cause; a pre-send refusal interrupts no provider
wait, so the bounded finalizer runs. A hard Stop at the same admission keeps its own no-call rail
(``tests/test_loop_tool_control.py``), and a refusal carrying a sent capture keeps the no-resend rail.
"""

import queue


from ouroboros import cancel_intents, loop, owner_mailbox, owner_pause
from ouroboros.llm_attempt import PhysicalDispatchInterrupted
from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
from supervisor.owner_stop import owner_stop_control_id
from tests.test_llm_claudexor import result
from tests.test_llm_claudexor import setup as gateway_fixture
from tests.test_model_wait import live_wait as wait_fixture
from tests.test_model_wait_controls import _loop_tools
from tests.test_subscription_main_wait import main_call as main_fixture


setup = gateway_fixture
live_wait = wait_fixture
main_call = main_fixture


def test_wrap_up_refusing_the_next_tool_start_still_runs_one_final_turn(main_call, monkeypatch):
    ctx, gateway, owner, events, _decide, _observations = main_call
    tools = _loop_tools(ctx, owner)
    tools._ctx.task_metadata = {"root_task_id": "task-one"}
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    effects, refusals = [], []
    tools.override_handler("knowledge_read", lambda *_a, **_k: effects.append(True) or "read once")
    tool_round = result()
    tool_round["message"] = {"role": "assistant", "content": "still working", "tool_calls": [
        {"id": "kr-1", "type": "function", "function": {"name": "knowledge_read", "arguments": '{"topic": "fixture"}'}}]}
    final = result()
    final["message"] = {"role": "assistant", "content": "Verified work summarized after the Wrap up"}
    gateway.results = [tool_round, final]
    gateway.dispatch = ["response_received", "response_received"]
    actual = owner_pause.run_tool_handler

    def wrap_up_at_admission(*args, **kwargs):
        # The owner's Wrap up, recorded while the round's answer was on its way: the tool's
        # admission is the first boundary to read it.
        intent = cancel_intents.request_cancel(ctx.drive_root, "task-one",
            requested_stop_policy=cancel_intents.STOP_POLICY_FINALIZE)
        assert owner_mailbox.write_owner_message(ctx.drive_root, REASON_OWNER_REQUESTED_FINALIZATION, "task-one",
            msg_id=owner_stop_control_id(intent), kind=owner_mailbox.KIND_FINALIZE_NOW)
        try:
            return actual(*args, **kwargs)
        except PhysicalDispatchInterrupted as error:
            refusals.append(error)
            raise

    monkeypatch.setattr(owner_pause, "run_tool_handler", wrap_up_at_admission)
    text, usage, trace = loop.run_llm_loop(ctx.messages, tools, ctx.llm, ctx.drive_logs,
        lambda *_args, **_kwargs: None, queue.Queue(), task_id="task-one",
        drive_root=ctx.drive_root, event_queue=events)

    # The premise: the Wrap up refused the tool's start; nothing of it ran.
    assert [error.control_reason for error in refusals] == ["finalize_requested"] and effects == []
    # The contract: the round's call, then exactly ONE final turn, answered by the model.
    assert text == final["message"]["content"] and len(gateway.creates) == 2
    assert usage["reason_code"] == REASON_OWNER_REQUESTED_FINALIZATION
    assert usage["terminal_origin"] == "model_final"
    assert trace["forced_finalization"]["source"] == "model"


def test_a_pre_send_refusal_carrying_a_sent_capture_keeps_the_no_call_rail(tmp_path, monkeypatch):
    """The repair widens only the unsent case. The production ladder (``llm_fallback._send``), here with
    its real primitives: the round's first send dies on the wire (a real attempt left ``unresolved``),
    the owner's Wrap up lands, the recovery send's real dispatch window refuses it before any bytes, and
    ``preserve_prior_dispatch`` raises that refusal carrying the SENT capture at top level. Work on the
    wire with an unknown outcome keeps the no-resend rail through the real round handler: no finalizer,
    no further model request or attempt, the same unresolved attempt and the Wrap up's own reason.
    (The recovery send is refused at its reservation — ``send_acquisition`` checks the window before any
    row exists — so the inner refusal has no capture of its own; the unknown rides the top-level one.)"""
    import httpx

    from ouroboros import loop_llm_call, model_wait
    from ouroboros import usage_accounting as ua
    from ouroboros.llm_attempt import (
        preserve_prior_dispatch,
        require_physical_dispatch_window,
        strongest_dispatch_capture,
    )
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.transport_custody import outcome_unknown_on_chain
    from ouroboros.usage_ledger import UsageAccountingError
    from tests._shared import saved_settings
    from tests.test_transport_death_retry import _ledger, _loop_kwargs, _no_chain

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    saved_settings(monkeypatch, tmp_path, OUROBOROS_MODEL_FALLBACKS="")  # a saved blank: no chain
    monkeypatch.setattr(loop, "_run_cross_model_fallback_chain", _no_chain)
    # The managed task's Pause authority is its lifecycle row (agent.py publishes it before round one).
    write_task_result(tmp_path, "t-death", STATUS_RUNNING, root_task_id="t-death")

    def extract(_response):
        return {"prompt_tokens": 1, "completion_tokens": 1}, 0.01, True

    def request():
        return ua.AttemptRequest(model="test-model", provider="openrouter", reservation_usd=1.0,
                                 drive_root=tmp_path, task_id="t-death", root_task_id="t-death", source="test.ladder")

    class LadderLLM:
        def __init__(self):
            self.chats, self.sends, self.dead, self.refusals = 0, 0, [], []

        def default_model(self):
            return "test-model"

        @model_wait.model_waitable
        def chat(self, **_kwargs):
            self.chats += 1

            def dying():
                self.sends += 1
                raise RuntimeError("Connection error.") from httpx.ReadError("controlled wire death")

            prior = None
            try:
                ua.execute_physical_attempt(request(), dying, extractor=extract)
            except Exception as exc:  # the ladder keeps the sent capture and tries to recover
                self.dead.append(exc)
                prior = strongest_dispatch_capture(prior, getattr(exc, "physical_attempt_capture", None))
            intent = cancel_intents.request_cancel(tmp_path, "t-death",
                requested_stop_policy=cancel_intents.STOP_POLICY_FINALIZE)
            assert owner_mailbox.write_owner_message(tmp_path, REASON_OWNER_REQUESTED_FINALIZATION, "t-death",
                msg_id=owner_stop_control_id(intent), kind=owner_mailbox.KIND_FINALIZE_NOW)

            def recovery():
                require_physical_dispatch_window()  # the real window: the Wrap up refuses it before bytes
                self.sends += 1
                return {"content": "a recovery send that must never leave"}

            try:
                ua.execute_physical_attempt(request(), recovery, extractor=extract)
            except UsageAccountingError as exc:  # llm_fallback._send's own clause
                self.refusals.append(exc)
                preserve_prior_dispatch(exc, prior)
                raise
            raise AssertionError("the recovery send passed the owner's Wrap up")

    llm = LadderLLM()
    kwargs = _loop_kwargs(tmp_path, llm, [])
    with model_wait.task_model_wait_scope(task={"id": "t-death"}, drive_root=tmp_path, event_queue=None,
                                          worker_slot_held=True) as owner:
        owner.tool_context = kwargs["tools"]._ctx
        text, usage, trace = loop.run_llm_loop(**kwargs)

    # The premise, as the real producers built it: the sent attempt is unresolved and rides the refusal at
    # TOP level (the handler's ``capture``); the inner refusal was refused at its reservation, before any
    # row, so it carries no capture and nothing unknown of its own.
    sent = llm.dead[0].physical_attempt_capture
    assert sent.state == "unresolved" and llm.sends == 1
    (inner,) = llm.refusals
    assert inner.control_reason == "finalize_requested"
    assert getattr(inner, "physical_attempt_capture", None) is None and not outcome_unknown_on_chain(inner)
    # The contract: no finalizer, no further request or attempt; the ledger keeps the one sent attempt.
    assert llm.chats == 1, "a sent-capture refusal bought a finalization request"
    assert [(row["attempt_id"], row["state"]) for row in _ledger(tmp_path)] == [(sent.attempt_id, "unresolved")]
    # Identity and outcome: the same unresolved attempt, unknown, the Wrap up's reason, never a model final.
    assert sent.attempt_id in usage["ledger_attempt_ids"]
    assert trace["forced_finalization"]["physical_attempt_state"] == "unresolved"
    assert trace["forced_finalization"]["control_reason"] == "finalize_requested"
    assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"
    assert loop_llm_call.provider_no_call_source(usage, False)[0] == "provider_outcome_unknown_no_resend"
    assert usage["reason_code"] == REASON_OWNER_REQUESTED_FINALIZATION
    assert usage.get("terminal_origin") != "model_final" and trace["forced_finalization"]["source"] != "model"
    assert "a recovery send that must never leave" not in text


def test_a_wrap_up_after_the_attempt_reservation_still_runs_one_final_turn(tmp_path, monkeypatch):
    """The same Wrap up can land after the round's first attempt is reserved but before any bytes (E10 on a
    fast runner). The production send seam (``llm_attempt._execute_candidate`` under ``llm_fallback._send``'s
    clause) refuses at its dispatch window, ``_pre_dispatch_failure`` releases the row, and the refusal carries
    that RELEASED capture. Released is no sent capture: the bounded finalizer runs one tool-less turn,
    answered by the model, and the first send never leaves (the sent-capture case above keeps its rail)."""
    from ouroboros import model_wait
    from ouroboros import usage_accounting as ua
    from ouroboros.llm_attempt import _execute_candidate, preserve_prior_dispatch
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.usage_ledger import UsageAccountingError
    from tests._shared import saved_settings
    from tests.test_transport_death_retry import _ledger, _loop_kwargs, _no_chain

    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    saved_settings(monkeypatch, tmp_path, OUROBOROS_MODEL_FALLBACKS="")  # a saved blank: no chain
    monkeypatch.setattr(loop, "_run_cross_model_fallback_chain", _no_chain)
    write_task_result(tmp_path, "t-death", STATUS_RUNNING, root_task_id="t-death")
    final_text = "Verified work summarized after the Wrap up"

    class ReservedLLM:
        def __init__(self):
            self.chats, self.sends, self.refusals = 0, 0, []

        def default_model(self):
            return "test-model"

        @model_wait.model_waitable
        def chat(self, **_kwargs):
            self.chats += 1
            if self.chats > 1:  # the promised final turn
                return {"role": "assistant", "content": final_text}, {"prompt_tokens": 1, "completion_tokens": 1}

            def wrap_up_while_reserved(_reservation):
                # The owner's Wrap up, recorded after this attempt's row exists and before its send.
                intent = cancel_intents.request_cancel(tmp_path, "t-death",
                    requested_stop_policy=cancel_intents.STOP_POLICY_FINALIZE)
                assert owner_mailbox.write_owner_message(tmp_path, REASON_OWNER_REQUESTED_FINALIZATION, "t-death",
                    msg_id=owner_stop_control_id(intent), kind=owner_mailbox.KIND_FINALIZE_NOW)

            def send():
                self.sends += 1
                return {"content": "a first send that must never leave"}

            request = ua.AttemptRequest(model="test-model", provider="openrouter", reservation_usd=1.0,
                                        drive_root=tmp_path, task_id="t-death", root_task_id="t-death",
                                        source="test.reserved")
            try:
                _execute_candidate(request, send, wrap_up_while_reserved)
            except UsageAccountingError as exc:  # llm_fallback._send's own clause
                self.refusals.append(exc)
                preserve_prior_dispatch(exc, None)
                raise
            raise AssertionError("the first send passed the owner's Wrap up")

    llm = ReservedLLM()
    kwargs = _loop_kwargs(tmp_path, llm, [])
    with model_wait.task_model_wait_scope(task={"id": "t-death"}, drive_root=tmp_path, event_queue=None,
                                          worker_slot_held=True) as owner:
        owner.tool_context = kwargs["tools"]._ctx
        text, usage, trace = loop.run_llm_loop(**kwargs)

    # The premise, as the real producers built it: refused at the dispatch window after the reservation,
    # nothing sent, and the refusal carries the RELEASED row it reserved.
    (refusal,) = llm.refusals
    assert isinstance(refusal, PhysicalDispatchInterrupted) and refusal.control_reason == "finalize_requested"
    assert refusal.physical_attempt_capture.state == "released" and llm.sends == 0
    assert [row["state"] for row in _ledger(tmp_path)] == ["released"]
    # The contract: exactly ONE final turn after the refused call, answered by the model.
    assert llm.chats == 2 and text == final_text
    assert usage["reason_code"] == REASON_OWNER_REQUESTED_FINALIZATION
    assert usage["terminal_origin"] == "model_final"
    assert trace["forced_finalization"]["source"] == "model"
