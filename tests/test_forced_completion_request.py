"""A forced boundary keeps the completion the mind selected: one request, native or serialized.

On a forced rail (round limit, Wrap up, deadline, budget, provider death) the one forced call is
sent with the round's tool schemas. A reply whose only tool call is ``finish_task`` is the same
completion request as that request serialized in its text: both go through the ordinary
``finish_task`` argument contract and byte selection, purely, and nothing is executed, staged
or retried. The accepted request keeps its silent choice, its one sentence for the human and its
stop rationale; the rail stays the recorded cause (best effort, never PASS). Mixed or doubled
calls, malformed arguments, silence while a reply is owed or on a run that neither started itself
nor answers its own place, and late owner words keep today's refusal, retained answer or host
fallback, with the typed reason on the trace. A provider length cut after a whole request is
typed and disclosed beside the answer, never a refusal (owner 30 Oct, A). Stop now keeps the
answer on the row, unsent.

Scenarios run the real loop and terminal pipeline (scripted inference only); the Place cases run
the real Place producer, agent bootstrap and runner (``install_bootstrap_harness``).
"""
from __future__ import annotations

import hashlib
import json

import pytest

import ouroboros.agent_task_pipeline as pipeline
import ouroboros.loop as loop
from ouroboros.task_results import load_task_result
from tests.test_delivery_forced_finalization import _arm_latch_with_candidate, _forced_test_context
from tests.test_finish_silent_wake import INTERNAL, _spoken, call, emit, sends, wake as _silent_wake
from tests.test_presence_continuation_bootstrap import install_bootstrap_harness, tool_call
from tests.test_presence_contact_wait import start
from tests.test_presence_owner_contact import contact_runtime as _contact_runtime, focus  # noqa: F401
from tests.test_presence_owner_stop import _kept_unsent, _outbound, owner_stop as _owner_stop

contact_runtime, owner_stop, wake = _contact_runtime, _owner_stop, _silent_wake
pytestmark = pytest.mark.serial

SUMMARY = "Sent the card and the HTML page; both deliveries are confirmed."
ANSWER = "Wrap-up: the first probe is green; the second one is still unread."
PARTIAL = "Partial: one probe read."
RATIONALE = "The second probe remains unread."
LIMIT_TEXT = "Task exceeded MAX_ROUNDS"


def native(arguments, call_id="final"):
    return call("finish_task", arguments, call_id)


def serialized(arguments):
    return {"role": "assistant", "content": json.dumps(arguments)}


FORMS = {"native": native, "text": serialized}


@pytest.fixture(autouse=True)
def _no_post_task_model(monkeypatch):
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *a, **k: None)


@pytest.fixture
def at_limit(wake, monkeypatch):
    """One ordinary tool round, then the production round limit's one forced call."""
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")
    wake.registry.override_handler("chat_history", lambda _ctx, **_kw: "read result")
    return wake


def _request_fact(trace):
    return trace.get("forced_completion_request") or {}


def _observed_source(messages):
    """The owner-source selector the forced prompt showed (``[ACCEPTANCE_SUBJECT_OBSERVATION]``)."""
    content = next(str(row["content"]) for row in reversed(messages)
                   if row.get("role") == "user" and "[ACCEPTANCE_SUBJECT_OBSERVATION]" in str(row.get("content")))
    return json.loads(content.split("[ACCEPTANCE_SUBJECT_OBSERVATION]\n", 1)[1].splitlines()[0])["owner_source_sha256"]


# --- the incident shape: a Place focus at its round limit -------------------------------------

@pytest.mark.parametrize("subject", [False, True], ids=["no-subject", "observed-subject"])
@pytest.mark.parametrize("form", ["native", "text"])
def test_place_round_limit_keeps_the_silent_finish_its_focus_selected(contact_runtime, monkeypatch, form, subject):
    """content=null plus a sole finish_task(silent, summary, answer) on a real Place: the answer is the internal
    record, nothing is spoken in the room, the summary is kept, and the round limit stays the cause."""
    r = contact_runtime
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")
    h = install_bootstrap_harness(monkeypatch, r.root, review_mode="off")

    def forced(messages):
        arguments = {"action": "finish", "silent": True, "summary": SUMMARY, "answer": INTERNAL}
        if subject:
            arguments["acceptance_subject"] = {"owner_source_sha256": _observed_source(messages)}
        return FORMS[form](arguments)

    h.script = [{"content": None, "tool_calls": [tool_call("knowledge_list", {}, "look")]}, forced]
    result = start(r, h).result.result(timeout=120)

    row = load_task_result(r.root, result.task_id)
    assert h.calls == 2 and "[ROUND_LIMIT]" in json.dumps(h.inputs[1])
    assert (result.outcome, result.text) == ("silent", "")
    assert _outbound(r.root, result.task_id) == [], "no reply, send or delivery receipt in the room"
    assert (row["result"], row.get("wake_silent"), row.get("wake_summary")) == (INTERNAL, True, SUMMARY)
    assert (row["reason_code"], row["terminal_origin"]) == ("round_limit", "model_final")
    assert row["outcome_axes"]["execution"]["status"] == "best_effort"
    assert row["outcome_axes"]["objective"]["status"] != "pass"
    assert LIMIT_TEXT not in json.dumps(row.get("result"))


def test_stop_now_during_the_forced_call_keeps_a_native_answer_unsent(owner_stop, monkeypatch):
    """Stop now pressed while the forced call is out: its native finish answer stays on the row, never sent."""
    r, press = owner_stop
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")
    h = install_bootstrap_harness(monkeypatch, r.root, review_mode="off")
    stops = []

    def forced(_messages):
        stops.append(press(h.agents[0].tools._ctx.task_id, cascade=True))
        return native({"action": "finish", "answer": ANSWER})

    h.script = [{"content": None, "tool_calls": [tool_call("knowledge_list", {}, "look")]}, forced]
    stopped = start(r, h).result.exception(timeout=90)
    assert h.calls == 2
    [stop] = stops
    _kept_unsent(r, h, stopped, stop, ANSWER)


def test_owner_wrap_up_accepts_the_native_finish_of_its_one_final_turn(owner_stop, monkeypatch):
    """Wrap up's one final turn answers with finish_task(answer=...): that answer is the reply, never the
    host's "no final answer" notice, and the owner's finalization stays the recorded cause."""
    from tests.test_presence_owner_stop import SUMMARY as WRAP_SUMMARY, _wrap_up

    r, press = owner_stop
    h, generated, inputs, _presses, _listed = _wrap_up(monkeypatch, r, press)
    scripted = h.script

    def model(messages):
        reply = scripted(messages)
        return native({"action": "finish", "answer": reply["content"]}) if len(generated) == 2 else reply

    h.script = model
    result = start(r, h).result.result(timeout=90)
    assert generated == [1, 2] and "[OWNER_STOP]" in json.dumps(inputs[1], ensure_ascii=False)
    assert result.text == WRAP_SUMMARY
    row = load_task_result(r.root, result.task_id)
    assert (row["status"], row["reason_code"], row["terminal_origin"]) == (
        "completed", "owner_requested_finalization", "model_final")
    assert [(out["direction"], out["text"]) for out in _outbound(r.root, result.task_id)] == [("out", WRAP_SUMMARY)]


# --- one request, two forms: the same result through the real loop and pipeline -------------

CASES = {
    "answer": {"action": "finish", "answer": ANSWER},
    "silent-empty": {"action": "finish", "silent": True},
    "silent-summary": {"action": "finish", "silent": True, "summary": SUMMARY, "answer": INTERNAL},
    "stop": {"action": "stop", "rationale": RATIONALE, "answer": PARTIAL},
    # A strict tool schema fills the fields the model did not choose with "": they name nothing.
    "strict-schema-fill": {"action": "finish", "answer": ANSWER, "answer_sha256": "", "rationale": "",
                           "pending_review": "wait", "silent": False, "summary": "", "swarm_comparison": ""},
}


@pytest.mark.parametrize("form", ["native", "text"])
@pytest.mark.parametrize("case", list(CASES))
def test_forced_native_and_text_completion_select_the_same_result(at_limit, form, case):
    turn = at_limit
    text, usage, trace, bus = turn.run([call("chat_history", {}), FORMS[form](CASES[case])])
    assert len(turn.calls) == 2, "one forced call; no repair or retry"
    assert trace["forced_finalization"]["source"] == "model"
    # The scripted transport reports no provider call count; the completion-only fact stands apart from it.
    assert _request_fact(trace) == {"form": form, "status": "selected", "tool_call_count": None}
    pending, stored = emit(turn, text, usage, trace, bus=bus)
    spoken = _spoken(turn, pending, bus)
    selected = {"answer": ANSWER, "silent-empty": "", "silent-summary": INTERNAL, "stop": PARTIAL,
                "strict-schema-fill": ANSWER}[case]
    silent = case.startswith("silent")
    assert (text, stored["result"], usage["terminal_origin"]) == (selected, selected, "model_final")
    assert (stored["reason_code"], stored["outcome_axes"]["execution"]["status"]) == ("round_limit", "best_effort")
    assert stored.get("wake_silent") is (True if silent else None)
    assert stored.get("wake_summary") == (SUMMARY if case == "silent-summary" else None)
    if silent:
        [notice] = sends(pending)
        assert notice["terminal_origin"] == "host_notice"
        assert notice["text"].startswith("⚠️ This run chose to finish without a message, but it ended best_effort")
        assert not selected or selected not in spoken, "internal words are never spoken"
    else:
        assert selected in spoken
    if case == "stop":
        assert trace["task_completion"] == {"action": "stop", "rationale": RATIONALE, "source": "forced_completion",
                                            "answer_sha256": hashlib.sha256(PARTIAL.encode()).hexdigest()}
        assert stored["outcome_axes"]["objective"]["reason"] == "author_stop"
    else:
        assert "task_completion" not in trace
    assert LIMIT_TEXT not in spoken and LIMIT_TEXT not in stored["result"]


def test_an_earlier_live_send_stays_the_only_message_and_is_never_resent(at_limit):
    """The forced silent finish adds no speech: the run's earlier live reply stays the one message."""
    from tests.test_finish_silent_wake import drained

    turn = at_limit
    note = call("send_user_message", {"text": "Found the regression source."})
    text, usage, trace, bus = turn.run([note, native({"action": "finish", "silent": True, "answer": INTERNAL})])
    assert trace["forced_finalization"]["source"] == "model" and usage.get("wake_silent") is True
    assert [e["text"] for e in sends(drained(bus))] == ["Found the regression source."]
    pending, stored = emit(turn, text, usage, trace, bus=bus)
    spoken = _spoken(turn, pending, bus)
    assert INTERNAL not in spoken and "Found the regression source." not in spoken, "nothing is sent twice"
    assert stored["result"] == INTERNAL and stored.get("wake_silent") is True


# --- refusals: nothing executes, the existing degrade stands, the reason is typed ------------

def _host_fallback(text, usage, trace):
    assert text.startswith("⚠️ " + LIMIT_TEXT) and usage.get("wake_silent") is not True
    assert trace["forced_finalization"]["source"] == "host_fallback"
    assert usage["terminal_origin"] == "host_notice" and not usage.get("_best_effort_extracted")


@pytest.mark.parametrize("form", ["native", "text"])
def test_silence_while_a_reply_is_owed_is_refused_on_both_forms(at_limit, form):
    turn = at_limit
    asked = turn.owner_writes("Are you there? Please answer me.", call("chat_history", {}), "owner-1")
    text, usage, trace, bus = turn.run([asked, FORMS[form]({"action": "finish", "silent": True, "answer": INTERNAL})])
    _host_fallback(text, usage, trace)
    assert _request_fact(trace)["status"] == "refused"
    assert _request_fact(trace)["reason"].startswith("owner_reply_owed")
    pending, stored = emit(turn, text, usage, trace, bus=bus)
    assert INTERNAL not in _spoken(turn, pending, bus) and stored["outcome_axes"]["execution"]["status"] == "failed"


@pytest.mark.parametrize("arguments", [{"action": "finish", "silent": True, "answer": INTERNAL},
                                       {"action": "finish", "summary": SUMMARY, "answer": ANSWER}],
                         ids=["silent", "summary"])
@pytest.mark.parametrize("form", ["native", "text"])
@pytest.mark.parametrize("metadata", [{}, {"presence_binding_authority": {"binding_id": "b1"}}],
                         ids=["ordinary", "binding-authority-only"])
def test_a_run_that_did_not_start_itself_cannot_finish_silently_or_leave_a_sentence(at_limit, form, arguments, metadata):
    turn = at_limit
    text, usage, trace, _bus = turn.run([call("chat_history", {}), FORMS[form](arguments)], metadata=metadata)
    _host_fallback(text, usage, trace)
    assert _request_fact(trace)["reason"].startswith("invalid_completion_request: silent and summary are available")


@pytest.mark.parametrize("calls", [
    [("finish_task", {"action": "finish", "answer": ANSWER}), ("chat_history", {})],
    [("update_scratchpad", {"content": "Next: the docs"}), ("finish_task", {"action": "finish", "answer": ANSWER})],
    [("finish_task", {"action": "finish", "answer": ANSWER}), ("finish_task", {"action": "finish", "answer": PARTIAL})],
], ids=["finish-beside-read", "receipt-write-beside-finish", "two-finishes"])
def test_a_completion_beside_other_calls_executes_nothing_and_is_not_adopted(at_limit, calls):
    turn = at_limit
    executed = []
    for name in ("chat_history", "update_scratchpad", "finish_task"):
        turn.registry.override_handler(name, lambda _ctx, _name=name, **_kw: executed.append(_name) or "ok")
    reply = {"role": "assistant", "content": None, "tool_calls": [
        tool_call(name, arguments, f"call-{index}") for index, (name, arguments) in enumerate(calls)]}
    text, usage, trace, _bus = turn.run([call("chat_history", {}), reply])
    assert executed == ["chat_history"], "only the ordinary round's own call ran; the forced reply ran nothing"
    _host_fallback(text, usage, trace)
    assert _request_fact(trace) == {"form": "native", "status": "not_adopted", "reason": "completion_beside_other_calls",
                                    "tool_call_count": None}


@pytest.mark.parametrize("arguments", ['{"action": "finish", "answer": "Truncated', '', '[]', '{"answer": "No verb"}',
                                       '{"action": "finish", "answer": "One", "answer": "Two"}'])
def test_malformed_native_arguments_degrade_without_publishing_them(at_limit, arguments):
    turn = at_limit
    reply = {"role": "assistant", "content": None, "tool_calls": [{
        "id": "final", "type": "function", "function": {"name": "finish_task", "arguments": arguments}}]}
    text, usage, trace, _bus = turn.run([call("chat_history", {}), reply])
    _host_fallback(text, usage, trace)
    assert arguments not in text or not arguments
    assert _request_fact(trace)["status"] == "refused"
    assert _request_fact(trace)["reason"].startswith("invalid_completion_request")


# Native arguments reach the rail parsed, never checked against the schema: each field is read at finish_task's own
# schema type before any truth test or membership, so a value of another type is a refused request, never a coercion.
WRONG_TYPES = {  # name: (the request, its typed refusal)
    "action-list": ({"action": [], "answer": ANSWER}, "action must be finish or stop"),
    "action-object": ({"action": {"verb": "finish"}, "answer": ANSWER}, "action must be finish or stop"),
    "pending-review-object": ({"action": "finish", "answer": ANSWER, "pending_review": {}},
                              "pending_review must be wait or finish"),
    "pending-review-list": ({"action": "finish", "answer": ANSWER, "pending_review": ["wait"]},
                            "pending_review must be wait or finish"),
    "silent-string": ({"action": "finish", "silent": "false", "answer": INTERNAL}, "silent must be true or false"),
    "silent-number": ({"action": "finish", "silent": 1, "answer": INTERNAL}, "silent must be true or false"),
    "summary-object": ({"action": "finish", "silent": True, "summary": {"text": SUMMARY}, "answer": INTERNAL},
                       "summary must be text"),
    "rationale-object": ({"action": "stop", "rationale": {"why": RATIONALE}, "answer": PARTIAL}, "rationale must be text"),
}


@pytest.mark.parametrize("case", list(WRONG_TYPES))
@pytest.mark.parametrize("form", ["native", "text"])
def test_a_field_of_another_type_is_refused_and_the_retained_answer_stands(tmp_path, form, case):
    """Astra R1, real transport, a self-started run: ``action=[]`` and ``pending_review={}`` raised TypeError out of
    the rail (the run ended in task_exception instead of its retained answer) and ``silent="false"`` finished
    silently. Both forms now refuse with the typed reason; nothing is selected, stopped, silenced or summarized."""
    request, reason = WRONG_TYPES[case]
    text, usage, returned, registry, kept = _real_transport(
        tmp_path, FORMS[form](request), "stop", retained=True, before=_self_started)
    assert _request_fact(returned) == {"form": form, "status": "refused", "tool_call_count": 1 if form == "native" else 0,
                                       "reason": "invalid_completion_request: " + reason}
    assert (text, usage["terminal_origin"]) == (kept.full_text, "model_final")
    assert registry._ctx._delivery_candidate.full_text == kept.full_text
    assert registry._ctx._delivery_control_required is True, "a refused request consumes nothing"
    assert "task_completion" not in returned and usage.get("wake_silent") is not True and "wake_summary" not in usage


NULL_FIELDS = {"action": "finish", "answer": ANSWER, "answer_sha256": None, "rationale": None, "acceptance_subject": None,
               "pending_review": None, "silent": None, "summary": None, "swarm_comparison": None}


@pytest.mark.parametrize("request_kind", ["null-fields", "silent-false"])
@pytest.mark.parametrize("form", ["native", "text"])
def test_null_fields_name_nothing_and_false_keeps_a_spoken_finish(tmp_path, form, request_kind):
    """The controls of the same reading: a null optional field is absent, as the handler's own staging drops it (a
    null subject acknowledges or degrades nothing), and silent=false is an ordinary spoken finish."""
    request = NULL_FIELDS if request_kind == "null-fields" else {"action": "finish", "silent": False, "answer": ANSWER}
    text, usage, returned, _registry, _kept = _real_transport(
        tmp_path, FORMS[form](request), "stop", retained=True, before=_self_started)
    assert _request_fact(returned) == {"form": form, "status": "selected", "tool_call_count": 1 if form == "native" else 0}
    assert (text, usage["terminal_origin"]) == (ANSWER, "model_final")
    assert "forced_acceptance_subject" not in returned, "a null subject is no subject"
    assert usage.get("wake_silent") is not True and "wake_summary" not in usage and "task_completion" not in returned


@pytest.mark.parametrize("case", ["action-list", "pending-review-object", "silent-string", "summary-object"])
def test_the_handler_refuses_the_same_types_and_stages_nothing(wake, case):
    """The ordinary finish_task call reads the same contract: a typed argument error, nothing staged."""
    request, reason = WRONG_TYPES[case]
    answer = wake.registry.execute("finish_task", request)
    assert answer == "ERROR: COMPLETION_ARGUMENT: " + reason, answer
    assert getattr(wake.ctx, "_completion_request", None) is None


def test_a_sole_legacy_presence_finish_call_stays_a_preamble(at_limit):
    from tests.presence_legacy_helpers import register_legacy_completion

    turn = at_limit
    register_legacy_completion(turn.registry)
    reply = call("presence_finish", {"outcome": "message", "message": ANSWER})
    text, usage, trace, _bus = turn.run([call("chat_history", {}), reply])
    _host_fallback(text, usage, trace)
    assert ANSWER not in text and not _request_fact(trace)


def test_late_owner_words_during_the_forced_call_are_answered_before_any_completion(at_limit):
    turn = at_limit
    late = turn.owner_writes("Also mention the backup window.", native({"action": "finish", "answer": ANSWER}), "late-1")
    current = {"role": "assistant", "content": "Current answer: both probes, and the backup window is unchecked."}
    text, usage, trace, _bus = turn.run([call("chat_history", {}), late, current])
    assert len(turn.calls) == 3 and "[FORCED_OWNER_REFRESH]" in json.dumps(turn.calls[2])
    assert text == current["content"] and ANSWER not in text
    assert usage["terminal_origin"] == "model_final" and not _request_fact(trace)


def test_a_refused_subject_keeps_the_selected_bytes_and_their_silence(at_limit):
    """A stale owner-source selector degrades the binding, never the author's silent act."""
    turn = at_limit
    request = {"action": "finish", "silent": True, "answer": INTERNAL,
               "acceptance_subject": {"owner_source_sha256": "not-the-observed-source"}}
    text, usage, trace, bus = turn.run([call("chat_history", {}), native(request)])
    assert trace["forced_acceptance_subject"]["applied"] is False
    assert (text, usage.get("wake_silent"), usage["terminal_origin"]) == (INTERNAL, True, "model_final")
    assert trace["delivery_candidate"]["degraded_reason"] == loop.REASON_DELIVERY_CONTROL_DEGRADED
    assert trace["delivery_candidate"]["acceptance_binding"]["authoritative"] is False
    pending, _stored = emit(turn, text, usage, trace, bus=bus)
    assert INTERNAL not in _spoken(turn, pending, bus)


# --- the instruction the model receives at a forced boundary ----------------------------------

def test_every_forced_rail_names_the_accepted_completion_forms(at_limit):
    """No retained answer exists: the forced request still says what is accepted and that no tool runs."""
    turn = at_limit
    turn.run([call("chat_history", {}), native({"action": "finish", "answer": ANSWER})])
    forced_prompt = next(str(row["content"]) for row in reversed(turn.calls[1]) if row.get("role") == "user")
    assert "[ROUND_LIMIT]" in forced_prompt
    assert "No tool runs at this boundary" in forced_prompt and "finish_task" in forced_prompt
    assert "answer_sha256" not in forced_prompt, "no retained selector is offered when no answer is retained"


# --- byte selection and the real transport ----------------------------------------------------

@pytest.mark.parametrize("selection", ["answer", "retained_hash", "held_hash"])
def test_a_native_request_selects_bytes_exactly_like_its_text(tmp_path, monkeypatch, selection):
    from ouroboros.loop_delivery import hold_completion_response

    loop_module, registry, ctx, trace = _forced_test_context(tmp_path)
    retained = _arm_latch_with_candidate(loop_module, registry, ctx, trace)
    held = "A short correction."
    hold_completion_response(held, registry, ctx, trace)
    expected = retained.full_text if selection == "retained_hash" else held
    request = {"action": "finish"}
    request.update({"answer": held} if selection == "answer" else {
        "answer_sha256": retained.content_sha256 if selection == "retained_hash" else hashlib.sha256(held.encode()).hexdigest()})
    ctx.tool_schemas = [registry.get_schema_by_name("finish_task")]
    calls = []
    monkeypatch.setattr(loop_module, "call_llm_with_retry", lambda *_a, **_k: (calls.append(1) or native(request), 0.0))
    text, usage, returned = loop_module._handle_round_limit(ctx)
    assert text == expected and calls == [1] and trace["tool_calls"] == []
    assert usage["reason_code"] == "round_limit" and usage["terminal_origin"] == "model_final"
    assert registry._ctx._delivery_candidate.full_text == expected
    assert _request_fact(returned)["status"] == "selected"


ABSENT = object()


def _real_transport(tmp_path, reply, finish, *, retained=False, schemas=None, before=None, prepare=None):
    """``_handle_round_limit`` through the real ``call_llm_with_retry``; only the provider client is a double.
    ``finish`` is its ``response_finish_reason`` (``ABSENT`` omits the key; ``None`` is a present null); ``reply``
    may read the messages the provider received; ``prepare(registry)`` runs before an answer is retained and
    ``before(registry, candidate)`` just before the call."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    loop_module, registry, ctx, trace = _forced_test_context(tmp_path)
    (tmp_path / "logs").mkdir(exist_ok=True)
    ctx.tool_schemas = [registry.get_schema_by_name(name) for name in (schemas or ("finish_task",))]
    if prepare is not None:
        prepare(registry)
    kept = _arm_latch_with_candidate(loop_module, registry, ctx, trace) if retained else None
    if before is not None:
        before(registry, kept)

    class ScriptedProvider:
        def chat(self, **kwargs):
            usage = {"provider": "fake", "resolved_model": "fake/model"}
            if finish is not ABSENT:
                usage["response_finish_reason"] = finish
            return dict(reply(kwargs.get("messages") or []) if callable(reply) else reply), usage

    ctx.llm = ScriptedProvider()
    text, usage, returned = loop_module._handle_round_limit(ctx)
    return text, usage, returned, registry, kept


@pytest.mark.parametrize("finish", ["tool_calls", "tool_use", "function_call", "stop", ABSENT],
                         ids=["tool_calls", "tool_use", "function_call", "stop", "absent"])
def test_the_real_transport_carries_the_call_and_keeps_the_actual_call_count(tmp_path, finish):
    """Through the real ``call_llm_with_retry``: the response facts keep tool_call_count=1, and a reply that
    ended only because it asked for its one completion call is complete."""
    text, usage, returned, _registry, _kept = _real_transport(
        tmp_path, native({"action": "finish", "answer": ANSWER}), finish)
    assert (text, usage["terminal_origin"], returned["forced_finalization"]["source"]) == (ANSWER, "model_final", "model")
    assert _request_fact(returned) == {"form": "native", "status": "selected", "tool_call_count": 1}


INCOMPLETE_FINISHES = {"context_overflow": "context_length_exceeded", "present_null": None}
CUT_FINISHES = ("length", "max_tokens")
CUT_REQUESTS = {"answer": {"action": "finish", "answer": PARTIAL},
                "stop": {"action": "stop", "rationale": RATIONALE, "answer": PARTIAL},
                "silent": {"action": "finish", "silent": True, "answer": PARTIAL}}


def _self_started(registry, _kept):
    """A consciousness wake's own turn: the run whose request may choose silence."""
    from tests.test_finish_silent_wake import WAKE

    registry._ctx.is_direct_chat = True
    registry._ctx.task_metadata = {**registry._ctx.task_metadata, **WAKE}


@pytest.mark.parametrize("retained", [False, True], ids=["no-retained-answer", "retained-answer"])
@pytest.mark.parametrize("request_kind", list(CUT_REQUESTS))
@pytest.mark.parametrize("form", ["native", "text"])
@pytest.mark.parametrize("finish", list(INCOMPLETE_FINISHES))
def test_an_overflowed_or_null_finished_completion_is_not_adopted_in_either_form(
        tmp_path, finish, form, request_kind, retained):
    """A whole finish_task request in a response the provider overflowed or ended with a null finish is a partial
    reply in either form. Its eligibility is read before the request is: it is not adopted and consumes nothing
    (no selection, subject acknowledgement, rebinding, stop record, silence, summary or latch change), the actual
    tool_call_count stays the response's, and the rail degrades to the retained answer or the host's own words.
    The serialized form used to be selected first and judged after, so a silent one finished silently (R4)."""
    from ouroboros.outcomes import derive_loop_outcome

    seen = {}

    def reply(messages):
        request = dict(CUT_REQUESTS[request_kind])
        if retained:  # the observed owner source, so an acknowledgement would be real
            request["acceptance_subject"] = {"owner_source_sha256": _observed_source(messages)}
        return FORMS[form](request)

    def before(registry, kept):
        _self_started(registry, kept)  # silence would be this run's to choose in a complete response
        if kept is not None:
            seen.update(binding=_binding(kept), ack=getattr(registry._ctx, "_acceptance_ack_source_sha256", ""))

    text, usage, returned, registry, kept = _real_transport(
        tmp_path, reply, INCOMPLETE_FINISHES[finish], retained=retained, before=before)
    ctx = registry._ctx
    assert _request_fact(returned) == {"form": form, "status": "not_adopted", "reason": "completion_response_incomplete",
                                       "tool_call_count": 1 if form == "native" else 0}
    assert PARTIAL not in text and ctx._delivery_candidate.full_text != PARTIAL
    assert "task_completion" not in returned and "forced_acceptance_subject" not in returned
    assert usage.get("wake_silent") is not True and "wake_summary" not in usage
    assert getattr(ctx, "_completion_selected", None) is None
    if retained:
        assert (text, usage["terminal_origin"]) == (kept.full_text, "model_final")
        assert ctx._delivery_control_required is True, "the delivery-control latch stays armed"
        assert ctx._delivery_candidate is kept and _binding(kept) == seen["binding"], "the retained answer is not rebound"
        assert getattr(ctx, "_acceptance_ack_source_sha256", "") == seen["ack"], "no owner-source acknowledgement"
    else:
        assert text.startswith("⚠️ " + LIMIT_TEXT) and usage["terminal_origin"] == "host_notice"
    assert derive_loop_outcome(text, usage, returned)["outcome_axes"]["execution"]["status"] == (
        "best_effort" if retained else "failed")


@pytest.mark.parametrize("retained", [False, True], ids=["no-retained-answer", "retained-answer"])
@pytest.mark.parametrize("request_kind", list(CUT_REQUESTS))
@pytest.mark.parametrize("form", ["native", "text"])
@pytest.mark.parametrize("finish", CUT_FINISHES)
def test_a_whole_completion_at_a_length_cut_is_selected_alike_in_both_forms(tmp_path, finish, form, request_kind,
                                                                              retained):
    """Owner 30 Oct (A): a whole, unambiguous request in a response the provider cut at its output limit is the
    author's act in either form. Its bytes replace a retained answer, its stop or silence stands, the rail stays
    best effort, and the cut is typed on the trace and stated beside the answer: a whole object does not prove
    the model meant to end there."""
    from ouroboros.outcomes import derive_loop_outcome

    text, usage, returned, registry, _kept = _real_transport(
        tmp_path, FORMS[form](CUT_REQUESTS[request_kind]), finish, retained=retained, before=_self_started)
    assert _request_fact(returned) == {"form": form, "status": "selected", "response_cut": finish,
                                       "tool_call_count": 1 if form == "native" else 0}
    assert (text, usage["terminal_origin"]) == (PARTIAL, "model_final")
    assert registry._ctx._delivery_candidate.full_text == PARTIAL, "the new choice replaces a retained answer"
    assert f"output limit ({finish})" in usage["terminal_host_notice"] and PARTIAL not in usage["terminal_host_notice"]
    assert ("task_completion" in returned) is (request_kind == "stop")
    assert (usage.get("wake_silent") is True) is (request_kind == "silent")
    assert derive_loop_outcome(text, usage, returned)["outcome_axes"]["execution"]["status"] == "best_effort"


CUT_REFUSALS = {  # name: (the reply's request body, the typed refusal it earns; None = no request is read)
    "cut-inside": ('{"action": "finish", "answer": "Part', None),
    "unknown-answer": (json.dumps({"action": "finish", "answer_sha256": "0" * 64}), "selection_unavailable"),
    "silence-owed": (json.dumps({"action": "finish", "silent": True, "answer": PARTIAL}), "owner_reply_owed"),
}


@pytest.mark.parametrize("form", ["native", "text"])
@pytest.mark.parametrize("refusal", list(CUT_REFUSALS))
def test_a_cut_request_that_is_broken_unknown_or_owes_a_reply_is_refused_without_effects(tmp_path, form, refusal):
    """At a length cut too: a cut inside the request, a selector naming no offered answer and silence while the
    owner's words are unanswered are not accepted. The retained answer stands; no stop, silence, summary or
    subject applies and no cut is disclosed for bytes the reply did not select. A reply is owed the way a run
    owes it: the owner wrote, the answer was drafted after reading it and nothing was sent live; the cut then
    changes nothing, so the same request without a cut ends the same way."""
    body, reason = CUT_REFUSALS[refusal]
    reply = ({"role": "assistant", "content": None, "tool_calls": [
        {"id": "final", "type": "function", "function": {"name": "finish_task", "arguments": body}}]}
             if form == "native" else {"role": "assistant", "content": body})

    def owed(registry):
        _self_started(registry, None)
        registry._ctx._owner_directives = [{"source": "owner_mailbox", "content": "Are you there?", "msg_id": "o1"}]

    setup = {"prepare": owed} if refusal == "silence-owed" else {"before": _self_started}
    text, usage, returned, registry, kept = _real_transport(tmp_path / "cut", reply, "length", retained=True, **setup)
    if refusal == "silence-owed":
        uncut = _real_transport(tmp_path / "uncut", reply, "stop", retained=True, **setup)
        assert text == kept.full_text and uncut[0] == uncut[4].full_text, "the retained answer stands, never silence"
        assert (usage["terminal_origin"], returned["forced_finalization"]["source"]) == (
            uncut[1]["terminal_origin"], uncut[2]["forced_finalization"]["source"])
        assert {**_request_fact(returned), "response_cut": None} == {**_request_fact(uncut[2]), "response_cut": None}
    else:
        assert (text, usage["terminal_origin"]) == (kept.full_text, "model_final")
    assert "task_completion" not in returned and "forced_acceptance_subject" not in returned
    assert usage.get("wake_silent") is not True and "wake_summary" not in usage
    assert "output limit" not in str(usage.get("terminal_host_notice") or "")
    assert registry._ctx._delivery_control_required is True, "a rejected request consumes nothing, in either form"
    fact = _request_fact(returned)
    if reason is None and form == "text":
        assert fact == {}, "a broken body is no request: the cut text stays an incomplete draft"
    else:
        assert (fact["status"], fact["response_cut"]) == ("refused", "length")
        assert fact["reason"].startswith(reason or "invalid_completion_request")


BROKEN_CALLS = {"arguments-absent": {"name": "finish_task"},
                "arguments-null": {"name": "finish_task", "arguments": None}}
SALVAGE = "Salvaged words beside a broken completion call."


@pytest.mark.parametrize("content", ["serialized-request", "prose"])
@pytest.mark.parametrize("broken", list(BROKEN_CALLS))
def test_a_native_call_without_arguments_is_refused_and_only_plain_words_beside_it_remain(at_limit, broken, content):
    """The call, not its arguments, marks the reply's completion request: absent or null arguments are a refused
    request, and a serialized request in the reply's text is never read as a second, authored completion. Plain
    prose beside the refused call is the reply's own incomplete draft: with no retained answer it ends the rail
    on the existing forced_model_incomplete fallback, never as the selected final (R4)."""
    turn = at_limit
    body = json.dumps({"action": "finish", "answer": SALVAGE}) if content == "serialized-request" else SALVAGE
    reply = {"role": "assistant", "content": body,
             "tool_calls": [{"id": "final", "type": "function", "function": dict(BROKEN_CALLS[broken])}]}
    text, usage, trace, bus = turn.run([call("chat_history", {}), reply])
    assert _request_fact(trace) == {"form": "native", "status": "refused", "tool_call_count": None,
                                    "reason": "invalid_completion_request: one exact JSON object of finish_task arguments"}
    assert "task_completion" not in trace and usage.get("wake_silent") is not True and "wake_summary" not in usage
    if content == "serialized-request":
        _host_fallback(text, usage, trace)
        assert SALVAGE not in text
    else:
        assert (text, trace["forced_finalization"]["source"]) == (SALVAGE, "forced_model_incomplete")
        assert usage.get("terminal_origin") != "model_final"
    pending, stored = emit(turn, text, usage, trace, bus=bus)
    if content == "serialized-request":
        assert SALVAGE not in stored["result"] and SALVAGE not in _spoken(turn, pending, bus)
    else:
        assert SALVAGE in stored["result"]
    assert stored["outcome_axes"]["execution"]["status"] == "failed"


REFUSED_SOLE_CALLS = {  # the R4 probe shapes: one finish_task whose arguments are refused (None: no arguments key)
    "no-selection": '{"action": "finish"}',
    "unknown-selector": json.dumps({"action": "finish", "answer_sha256": "0" * 64}),
    "cut-arguments": '{"action": "finish", "answer": "Cut',
    "arguments-absent": None,
}


@pytest.mark.parametrize("refusal", list(REFUSED_SOLE_CALLS))
def test_plain_words_beside_a_refused_sole_request_end_as_beside_an_unadoptable_call(tmp_path, refusal):
    """Real transport, no retained answer: prose beside one finish_task whose request is refused ends exactly as the
    same prose beside a call this rail cannot adopt, on forced_model_incomplete with the reply's words; nothing
    executes and the refused request selects, stops, silences or summarizes nothing."""
    executed = []

    def before(registry, _kept):
        for name in ("chat_history", "finish_task"):
            registry.override_handler(name, lambda _ctx, _name=name, **_kw: executed.append(_name) or "ok")

    arguments = REFUSED_SOLE_CALLS[refusal]
    function = {"name": "finish_task", **({"arguments": arguments} if arguments is not None else {})}
    refused = _real_transport(tmp_path / "refused", {"role": "assistant", "content": SALVAGE, "tool_calls": [
        {"id": "final", "type": "function", "function": function}]}, "tool_calls", before=before)
    beside = _real_transport(tmp_path / "beside", {"role": "assistant", "content": SALVAGE, "tool_calls": [
        tool_call("chat_history", {}, "call-0")]}, "tool_calls", before=before)
    (text, usage, returned, _registry, _kept), (other, other_usage, other_returned, _r, _k) = refused, beside
    assert executed == [], "nothing dispatches"
    assert (text, returned["forced_finalization"]["source"]) == (SALVAGE, "forced_model_incomplete")
    assert (text, usage.get("terminal_origin"), returned["forced_finalization"]["source"]) == (
        other, other_usage.get("terminal_origin"), other_returned["forced_finalization"]["source"])
    assert _request_fact(returned)["status"] == "refused" and _request_fact(other_returned) == {}
    assert "task_completion" not in returned and usage.get("wake_silent") is not True and "wake_summary" not in usage


def test_a_broken_call_through_the_real_transport_keeps_its_count_and_salvages_nothing(tmp_path):
    """The same through the real transport, whose response facts carry the actual tool_call_count."""
    reply = {"role": "assistant", "content": json.dumps({"action": "finish", "answer": SALVAGE}),
             "tool_calls": [{"id": "final", "type": "function", "function": {"name": "finish_task"}}]}
    text, usage, returned, _registry, _kept = _real_transport(tmp_path, reply, "tool_calls")
    assert SALVAGE not in text and text.startswith("⚠️ " + LIMIT_TEXT)
    assert usage["terminal_origin"] == "host_notice" and not usage.get("_best_effort_extracted")
    assert _request_fact(returned) == {"form": "native", "status": "refused", "tool_call_count": 1,
                                       "reason": "invalid_completion_request: one exact JSON object of finish_task arguments"}


# --- a reply with native calls this rail cannot adopt consumes nothing -----------------------
# Its text is only classified (a pure parse) so a protocol attempt stays private; no request in it is consumed.

CONTENT_ANSWER = "Bytes of a completion written beside native calls."
CRITERIA = "Criteria carried by a reply that also asked for tools."
NON_ADOPTABLE = {  # name: (native calls, provider finish, offered tools, native unadoption reason)
    "other-call": ([("chat_history", {})], "tool_calls", ("finish_task",), ""),
    "finish-beside-other": ([("finish_task", {"action": "finish", "answer": ANSWER}), ("chat_history", {})],
                            "tool_calls", ("finish_task",), "completion_beside_other_calls"),
    "two-finishes": ([("finish_task", {"action": "finish", "answer": ANSWER}),
                      ("finish_task", {"action": "finish", "answer": PARTIAL})],
                     "tool_calls", ("finish_task",), "completion_beside_other_calls"),
    "overflowed-finish": ([("finish_task", {"action": "finish", "answer": ANSWER})], "context_length_exceeded",
                          ("finish_task",), "completion_response_incomplete"),
    "cut-finish-beside-other": ([("finish_task", {"action": "finish", "answer": ANSWER}), ("chat_history", {})],
                                "length", ("finish_task",), "completion_beside_other_calls"),
    "unoffered-finish": ([("finish_task", {"action": "finish", "answer": ANSWER})], "tool_calls", ("chat_history",),
                         "completion_tool_not_offered"),
}


def _content(kind, messages):
    """The reply's text: a serialized completion (with a subject naming the observed source), an old control, or prose."""
    subject = {"owner_source_sha256": _observed_source(messages), "effective_criteria": CRITERIA}
    return {"serialized-finish": lambda: json.dumps({"action": "finish", "answer": CONTENT_ANSWER, "acceptance_subject": subject}),
            "serialized-stop": lambda: json.dumps({"action": "stop", "rationale": RATIONALE, "answer": CONTENT_ANSWER,
                                                   "acceptance_subject": subject}),
            "historical-keep": lambda: json.dumps({"delivery_control": "keep", "acceptance_subject": subject}),
            "prose": lambda: CONTENT_ANSWER}[kind]()


def _binding(candidate):
    return (candidate.full_text, candidate.effective_criteria, candidate.material_tool_indices,
            candidate.owner_source_sha256, candidate.evidence_revision, candidate.evidence_fingerprint,
            json.dumps(candidate.acceptance_binding, sort_keys=True))


@pytest.mark.parametrize("content", ["serialized-finish", "serialized-stop", "historical-keep", "prose"])
@pytest.mark.parametrize("shape", list(NON_ADOPTABLE))
def test_a_reply_with_calls_this_rail_cannot_adopt_consumes_nothing(tmp_path, shape, content):
    """Real transport, a retained answer and an armed latch: whatever the reply's text holds, a reply whose native
    calls cannot be adopted acknowledges no subject, rebinds nothing on the retained answer, records no stop,
    sets no silence or summary and keeps the delivery-control latch; nothing dispatches and the retained answer
    stands. The typed fact says not_adopted whenever a completion request was present, never selected."""
    calls, finish, schemas, native_reason = NON_ADOPTABLE[shape]
    seen, executed = {}, []

    def reply(messages):
        return {"role": "assistant", "content": _content(content, messages), "tool_calls": [
            tool_call(name, arguments, f"call-{index}") for index, (name, arguments) in enumerate(calls)]}

    def before(registry, kept):
        for name in ("chat_history", "finish_task"):
            registry.override_handler(name, lambda _ctx, _name=name, **_kw: executed.append(_name) or "ok")
        seen.update(binding=_binding(kept), ack=getattr(registry._ctx, "_acceptance_ack_source_sha256", ""))

    text, usage, returned, registry, kept = _real_transport(
        tmp_path, reply, finish, retained=True, schemas=schemas, before=before)
    ctx = registry._ctx
    assert executed == [], "nothing dispatches"
    assert ctx._delivery_control_required is True, "the delivery-control latch stays armed"
    assert "forced_acceptance_subject" not in returned
    assert getattr(ctx, "_acceptance_ack_source_sha256", "") == seen["ack"], "no owner-source acknowledgement"
    assert ctx._delivery_candidate is kept and _binding(kept) == seen["binding"], "the retained answer is not rebound"
    assert "task_completion" not in returned and getattr(ctx, "_completion_selected", None) is None
    assert usage.get("wake_silent") is not True and "wake_summary" not in usage
    assert text == kept.full_text and CONTENT_ANSWER not in text
    request_present = content.startswith("serialized") or bool(native_reason)
    expected = {"form": "native" if native_reason else "text", "status": "not_adopted", "tool_call_count": len(calls),
                "reason": native_reason or "completion_beside_other_calls",
                **({"response_cut": finish} if finish in CUT_FINISHES else {})} if request_present else {}
    assert _request_fact(returned) == expected


@pytest.mark.parametrize("request_body", [
    {"action": "finish", "silent": True, "summary": SUMMARY, "answer": INTERNAL},
    {"action": "stop", "rationale": RATIONALE, "answer": INTERNAL},
], ids=["silent-summary", "stop"])
@pytest.mark.parametrize("calls", [
    [("chat_history", {})],
    [("finish_task", {"action": "finish", "answer": ANSWER}), ("chat_history", {})],
    [("finish_task", {"action": "finish", "answer": ANSWER}), ("finish_task", {"action": "finish", "answer": PARTIAL})],
], ids=["other-call", "finish-beside-other", "two-finishes"])
def test_a_serialized_completion_beside_native_calls_is_never_the_final_through_the_pipeline(at_limit, calls, request_body):
    """Scripted transport (no reported call count) and the real terminal pipeline: a serialized completion in the
    text of a reply that also asked for tools is never the authored final — no silence, summary or stop, the
    host's own words end the run, and the selected bytes are neither the stored result nor spoken."""
    turn = at_limit
    executed = []
    for name in ("chat_history", "finish_task"):
        turn.registry.override_handler(name, lambda _ctx, _name=name, **_kw: executed.append(_name) or "ok")
    reply = {"role": "assistant", "content": json.dumps(request_body), "tool_calls": [
        tool_call(name, arguments, f"call-{index}") for index, (name, arguments) in enumerate(calls)]}
    text, usage, trace, bus = turn.run([call("chat_history", {}), reply])
    assert executed == ["chat_history"], "only the ordinary round's own call ran"
    # A reply that asks for tools is a preamble (the existing label); its contained request leaves the host's words.
    assert text.startswith("⚠️ " + LIMIT_TEXT) and trace["forced_finalization"]["source"] == "forced_model_incomplete"
    assert usage["terminal_origin"] == "host_notice" and not usage.get("_best_effort_extracted")
    assert usage.get("wake_silent") is not True
    assert "task_completion" not in trace and "wake_summary" not in usage
    assert _request_fact(trace)["status"] == "not_adopted"
    pending, stored = emit(turn, text, usage, trace, bus=bus)
    assert INTERNAL not in stored["result"] and INTERNAL not in _spoken(turn, pending, bus)
    assert stored.get("wake_silent") is None and stored.get("wake_summary") is None
    assert stored["outcome_axes"]["execution"]["status"] == "failed"


def test_an_adopted_native_completion_never_reads_a_second_request_in_its_text(tmp_path):
    """Guard: a complete reply whose only call is finish_task is adopted; a serialized request in its text is ignored."""
    reply = {"role": "assistant", "content": json.dumps({"action": "finish", "answer": CONTENT_ANSWER}),
             "tool_calls": [tool_call("finish_task", {"action": "finish", "answer": ANSWER}, "final")]}
    text, usage, returned, _registry, _kept = _real_transport(tmp_path, reply, "tool_calls")
    assert (text, usage["terminal_origin"]) == (ANSWER, "model_final") and CONTENT_ANSWER not in text
    assert _request_fact(returned) == {"form": "native", "status": "selected", "tool_call_count": 1}
