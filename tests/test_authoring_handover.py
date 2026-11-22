"""Typed host-driven model handover and fallback-finalization regressions."""

from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest

from ouroboros.loop_delivery import _no_tool_final_answer
from ouroboros.loop_model_call import _adopt_fallback_route, observe_authoring_response
from ouroboros.loop_nudges import _maybe_inject_finalization_nudges, accept_handover_selection, observe_handover_work
from ouroboros.loop_round_limits import _RoundLimitContext
from ouroboros.outcomes import EXECUTION_DEGRADED, derive_loop_outcome
from ouroboros.owner_wait import continuation_state, restore_continuation_state
from ouroboros.tools.registry import ToolRegistry


def _ctx(tmp_path):
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    trace = {"tool_calls": [{"tool": "vcs_read", "status": "ok"}], "reasoning_notes": []}
    registry._ctx._execution_trace = trace
    registry._ctx._accumulated_usage = {}
    return registry, trace


def test_fallback_records_authorship_only_when_it_answers(tmp_path):
    registry, trace = _ctx(tmp_path)
    registry._ctx.active_model = "codex=gpt-6-astra"
    registry._ctx.active_use_local = False
    usage = registry._ctx._accumulated_usage
    usage["accepted_routes"] = [{"to_binding": ["codex=gpt-6-astra", False, ""]}]
    registry._ctx._pending_handover_reason = "invalid_continuation"
    messages = [{"role": "user", "content": "work"}]
    fallback_messages = list(messages)

    _adopt_fallback_route(
        registry._ctx,
        registry,
        "claude-fable-5-1",
        False,
        messages,
        fallback_messages,
        None,
        "max",
        [],
        usage,
    )
    assert "authoring_handovers" not in usage
    state = json.loads(json.dumps(continuation_state(registry._ctx, messages, trace, usage, 2, [], set())))
    registry, trace = _ctx(tmp_path)
    usage = registry._ctx._accumulated_usage
    restore_continuation_state(registry, state, [], trace, usage, set())
    observe_authoring_response(registry._ctx, "claude-fable-5-1")
    assert registry._ctx._authoring_handover["from_model"] == "codex=gpt-6-astra"
    assert registry._ctx._authoring_handover["to_model"] == "claude-fable-5-1"
    assert usage["authoring_handovers"][0]["reason"] == "invalid_continuation"
    assert trace["route_handovers"][0]["tool_calls_at_handover"] == 1


def test_first_toolless_response_after_handover_gets_one_recovery_round(tmp_path):
    registry, trace = _ctx(tmp_path)
    registry._ctx._authoring_handover = {
        "from_model": "codex=gpt-6-astra",
        "to_model": "claude-fable-5-1",
        "reason": "invalid_continuation",
        "tool_calls_at_handover": 1,
        "recovery_prompted": False,
    }
    messages = []
    progress = []

    injected = _maybe_inject_finalization_nudges(
        registry, tmp_path, "handover-1", trace, "the worktree is clean", messages,
        progress.append,
    )

    assert injected is True
    assert registry._ctx._authoring_handover["recovery_prompted"] is True
    assert registry._ctx._accumulated_usage == {}
    assert any("model handover" in str(row.get("content") or "") for row in messages)
    assert any("handover recovery" in str(item).lower() for item in progress)


def test_toolless_final_after_the_reminder_that_is_neither_work_nor_selection_is_degraded(tmp_path):
    """The live loop holds such prose for a selection before this door; a final that reaches it anyway keeps
    the degraded record, the mark a restored execution can still carry."""
    registry, trace = _ctx(tmp_path)
    registry._ctx._authoring_handover = {
        "from_model": "codex=gpt-6-astra",
        "to_model": "claude-fable-5-1",
        "reason": "invalid_continuation",
        "tool_calls_at_handover": 1,
        "recovery_prompted": True,
    }

    injected = _maybe_inject_finalization_nudges(
        registry, tmp_path, "handover-2", trace, "the worktree is still clean", [],
        lambda _text: None,
    )

    assert injected is False
    assert registry._ctx._authoring_handover is trace["authoring_handover_incomplete"]
    assert trace["authoring_handover_incomplete"]["status"] == "incomplete"
    assert registry._ctx._accumulated_usage["execution_status"] == EXECUTION_DEGRADED
    assert registry._ctx._accumulated_usage["reason_code"] == "authoring_handover_incomplete"
    outcome = derive_loop_outcome(
        "the worktree is still clean",
        registry._ctx._accumulated_usage,
        trace,
    )
    assert outcome["outcome_axes"]["execution"]["status"] == EXECUTION_DEGRADED
    assert outcome["outcome_axes"]["execution"]["reason_code"] == "authoring_handover_incomplete"


@pytest.mark.parametrize(("between", "routes", "credited"), [
    pytest.param(None, (), True, id="next-round"),
    pytest.param("refused", (), True, id="refused-round"),
    pytest.param("unsent", ("a", "b"), True, id="unsent-round"),
    pytest.param("unsent", ("a", "c", "b"), False, id="another-route-answered"),
    pytest.param("unsent", (), False, id="routes-unrecorded"),
    pytest.param("not-a-solve", ("a", "b"), False, id="reply-not-a-solve"),
])
def test_an_older_wait_source_credits_the_successors_recorded_work(tmp_path, between, routes, credited):
    """A parked wait's source names no working boundary. The previous body left the row pending after the
    successor's successful batch; the execution's receipts still name the author of that round, also past a
    round that ended without a reply: one whose attempts all failed, or one that sent nothing while the accepted
    routes show no other author. A round the records do not account for leaves the pair as it was written."""
    registry, trace = _ctx(tmp_path)
    ctx, usage, execution = registry._ctx, registry._ctx._accumulated_usage, "exec-wait"
    ctx.task_id, ctx.task_attempt = "waited", 1
    last = 2 if between is None else 3  # the successor's round
    receipts = [
        {"llm_call_id": f"llm-{number}", "execution_id": execution, "round_id": f"{execution}:round:{number}",
         "round": number, "attempt": 1, "model": model, "usable_solve_response": True}
        for number, model in ((1, "a"), (last, "b"))]
    if between == "refused":  # the provider refused the request: a failed attempt's row names no execution
        receipts.insert(1, {"llm_call_id": "llm-refused", "round": 2, "model": "a", "failure_code": "context_overflow"})
    elif between == "not-a-solve":  # a reply was received there and nothing accepted it as a solve
        receipts.insert(1, {**receipts[0], "llm_call_id": "llm-other", "round": 2, "model": "c",
                            "round_id": f"{execution}:round:2", "usable_solve_response": False})
    usage.update(_task_attempt=1, execution_id=execution, llm_call_refs=receipts)
    if routes:  # the dispatcher's record of each accepted reply on a changed route
        usage["accepted_routes"] = [
            {"from_binding": [before, False, ""] if before else None, "to_binding": [model, False, ""],
             "status": "accepted_response", "at": "2026-01-01T00:00:00+00:00", "model_route": {}}
            for before, model in zip((None, *routes), routes)]
    trace["tool_calls"] = [{"tool": "read_file", "status": "ok", "is_error": False,
                            "round_id": f"{execution}:round:{number}"} for number in (1, last)]
    pair = {"from_model": "a", "to_model": "b", "reason": "fallback", "tool_calls_at_handover": 1,
            "recovery_prompted": False, "status": "pending"}  # as the previous body wrote it
    trace["route_handovers"], usage["authoring_handovers"] = [pair], [dict(pair)]
    state = json.loads(json.dumps(continuation_state(ctx, [], trace, usage, last, [], set())))
    assert "working" not in state and "_completion_observation" not in state["delivery"]

    registry, trace = _ctx(tmp_path)
    registry._ctx.task_id = "waited"
    usage = registry._ctx._accumulated_usage
    restore_continuation_state(registry, state, [], trace, usage, set())
    [row] = usage["authoring_handovers"]
    ending = ("recovered", True, True) if credited else ("pending", None, None)
    assert (row["status"], row.get("work_observed"), row.get("reply_observed")) == ending
    assert trace["route_handovers"] == [row] and getattr(registry._ctx, "_authoring_handover", None) is None
    assert registry._ctx._completion_observation["model"] == "b"
    assert not _maybe_inject_finalization_nudges(registry, tmp_path, "waited", trace, "Report", [], lambda _text: None)
    assert derive_loop_outcome("Report", usage, trace)["outcome_axes"]["execution"]["status"] == "ok"


def test_text_after_the_unbound_reminder_leaves_the_row_incomplete_and_names_no_author(tmp_path):
    """A restored pair that no saved fact confirms is asked once, unbound. Text that is neither a choice nor work
    is no confirmation after that reminder: the row is marked incomplete and still names no author. Its target's
    real reply attaches it then, and that model's own selection withdraws the mark."""
    registry, trace = _ctx(tmp_path)
    ctx, usage = registry._ctx, registry._ctx._accumulated_usage
    ctx.task_id = "legacy"
    pair = {"from_model": "a", "to_model": "b", "reason": "fallback", "tool_calls_at_handover": 1,
            "recovery_prompted": False, "status": "pending"}  # as the previous body wrote it; no receipt, no author
    trace["route_handovers"], usage["authoring_handovers"] = [pair], [dict(pair)]
    state = json.loads(json.dumps(continuation_state(ctx, [], trace, usage, 2, [], set())))

    registry, trace = _ctx(tmp_path)
    ctx, usage = registry._ctx, registry._ctx._accumulated_usage
    ctx.task_id = "legacy"
    restore_continuation_state(registry, state, [], trace, usage, set())
    [row] = usage["authoring_handovers"]
    assert trace["route_handovers"][-1] is row and getattr(ctx, "_authoring_handover", None) is None
    messages = []
    assert _maybe_inject_finalization_nudges(registry, tmp_path, "legacy", trace, "Draft", messages,
                                             lambda _text: None, restored=True)
    assert "does not confirm its author" in messages[-1]["content"]
    assert (row["status"], row["recovery_prompted"]) == ("pending", True) and "execution_status" not in usage

    assert not _maybe_inject_finalization_nudges(registry, tmp_path, "legacy", trace, "Still a draft", [],
                                                 lambda _text: None)
    assert (row["status"], row.get("incomplete_observed")) == ("incomplete", True)
    assert trace["authoring_handover_incomplete"] is row and getattr(ctx, "_authoring_handover", None) is None
    assert not {"reply_observed", "work_observed", "selection"} & set(row)  # the mark claims no successor act
    execution = derive_loop_outcome("Still a draft", usage, trace)["outcome_axes"]["execution"]
    assert (execution["status"], execution["reason_code"]) == (EXECUTION_DEGRADED, "authoring_handover_incomplete")

    observe_authoring_response(ctx, "b")
    assert ctx._authoring_handover is row and (row["status"], row["reply_observed"]) == ("incomplete", True)
    ctx._completion_selected = {"action": "finish", "answer": "Report", "observation": {"model": "b"}}
    assert not _maybe_inject_finalization_nudges(registry, tmp_path, "legacy", trace, "Report", [], lambda _text: None)
    accept_handover_selection(ctx, trace)
    assert (row["status"], row["selection"], row["incomplete_observed"]) == ("selected", "answer", True)
    assert "authoring_handover_incomplete" not in trace and "execution_status" not in usage and "reason_code" not in usage


@pytest.mark.parametrize(("status", "needs_reminder"), [("ok", False), ("git_error", True)])
def test_handover_counts_successful_tool_work(tmp_path, status, needs_reminder):
    registry, trace = _ctx(tmp_path)
    registry._ctx._authoring_handover = {
        "from_model": "codex=gpt-6-astra",
        "to_model": "claude-fable-5-1",
        "reason": "fallback",
        "tool_calls_at_handover": 1,
        "recovery_prompted": False,
    }
    # A vcs_read refusal has an informative result with is_error=False.
    trace["tool_calls"].append({"tool": "vcs_read", "status": status, "is_error": False})

    observe_handover_work(registry._ctx, trace, trace["tool_calls"][-1:])
    injected = _maybe_inject_finalization_nudges(
        registry, tmp_path, "handover-3", trace, "done", [], lambda _text: None,
    )

    assert injected is needs_reminder
    assert (registry._ctx._authoring_handover is not None) is needs_reminder
    assert getattr(registry._ctx, "_authoring_handover_incomplete", False) is False


def test_ordinary_toolless_final_without_handover_is_unchanged(tmp_path):
    registry, trace = _ctx(tmp_path)
    injected = _maybe_inject_finalization_nudges(
        registry, tmp_path, "handover-4", trace, "ordinary answer", [], lambda _text: None,
    )
    assert injected is False
    assert registry._ctx._accumulated_usage == {}


def test_production_finalizer_requires_selection_during_handover_recovery(tmp_path):
    from copy import deepcopy
    from ouroboros.loop_delivery import completion_observation, consume_completion_request
    from ouroboros.tools.control_runtime import _finish_task
    registry, trace = _ctx(tmp_path)
    registry._ctx._authoring_handover = row = {
        "from_model": "codex=gpt-6-astra",
        "to_model": "claude-fable-5-1",
        "reason": "invalid_continuation",
        "tool_calls_at_handover": 1,
        "recovery_prompted": False,
    }
    messages = []
    usage = registry._ctx._accumulated_usage
    limit = _RoundLimitContext(
        messages=messages,
        llm=SimpleNamespace(),
        active_model="claude-fable-5-1",
        active_effort="medium",
        max_retries=3,
        drive_logs=tmp_path,
        task_id="handover-finalizer",
        round_idx=1,
        event_queue=None,
        accumulated_usage=usage,
        task_type="task",
        active_use_local=False,
        max_rounds=100,
        drive_root=tmp_path,
        root_task_id="handover-finalizer",
        tools=registry,
        llm_trace=trace,
        incoming_messages=queue.Queue(),
        owner_msg_seen=set(),
        tool_schemas=[],
    )
    # Match run_llm_loop's binding before calling its finalizer in isolation.
    registry._ctx.task_id, registry._ctx.messages = limit.task_id, messages

    first = _no_tool_final_answer(
        "the worktree is clean", limit, trace, registry, queue.Queue(), set(), lambda _text: None,
    )

    assert first is None
    candidate = registry._ctx._delivery_candidate
    assert candidate.finalization_control == "authoring_handover_recovery_required"
    assert registry._ctx._delivery_control_required is True
    assert not any("DELIVERY_FINALIZATION_CONTROL" in str(row.get("content") or "") for row in messages)

    before_second = deepcopy(messages)
    second = _no_tool_final_answer(
        "I will continue the requested work", limit, trace, registry, queue.Queue(), set(), lambda _text: None,
    )

    assert second is None
    assert candidate.full_text == "the worktree is clean"
    assert messages[:len(before_second)] == before_second
    assert messages[len(before_second)] == {"role": "assistant", "content": "I will continue the requested work"}
    assert all(row["role"] == "user" for row in messages[len(before_second) + 1:])
    assert messages[-1]["role"] == "user" and "No completion selection" in messages[-1]["content"]
    registry._ctx._completion_observation = {**completion_observation(registry._ctx, trace), "model": row["to_model"]}
    _finish_task(registry._ctx, action="finish", answer="No further work was performed.")
    assert consume_completion_request(registry, limit, trace)
    third = _no_tool_final_answer("No further work was performed.", limit, trace, registry,
                                 queue.Queue(), set(), lambda _text: None, explicit_candidate=True)
    assert third is not None and third[0] == "No further work was performed."
    # The successor's own selection after the reminder ends the handover: no extra call, no warning of its own.
    assert usage.get("reason_code") != "authoring_handover_incomplete"
    assert third[1].get("execution_status") != "degraded" and "authoring_handover_incomplete" not in trace
    assert (row["status"], row["selection"]) == ("selected", "answer")
    assert derive_loop_outcome(third[0], third[1], trace)["outcome_axes"]["execution"]["status"] == "ok"


def test_handover_incomplete_trace_survives_usage_round_cleanup(tmp_path):
    registry, trace = _ctx(tmp_path)
    trace["authoring_handover_incomplete"] = {
        "from_model": "codex=gpt-6-astra",
        "to_model": "claude-fable-5-1",
    }
    outcome = derive_loop_outcome(
        "still incomplete",
        {"rounds": 2},
        trace,
    )
    assert outcome["outcome_axes"]["execution"]["status"] == EXECUTION_DEGRADED
    assert outcome["outcome_axes"]["execution"]["reason_code"] == "authoring_handover_incomplete"


@pytest.mark.parametrize("author", ["model-b", None], ids=["other-author", "unknown-author"])
def test_an_ordinary_selection_belongs_to_the_model_that_returned_it(tmp_path, author):
    registry, trace = _ctx(tmp_path)
    ctx = registry._ctx
    row = ctx._authoring_handover = {"to_model": "model-c", "status": "pending", "recovery_prompted": True}
    ctx._completion_selected = {"action": "finish", "answer": "Report", "observation": {"model": author}}
    accept_handover_selection(ctx, trace)
    assert row["status"] == "pending" and "selection" not in row


@pytest.mark.parametrize(("first", "second"), [
    ({"answer_sha256": "c" * 64}, {"answer": "", "answer_sha256": "c" * 64}),
    ({"answer": "Report"}, {"answer": "Report", "answer_sha256": ""}),
    ({"answer": "Report"}, {"answer": "Report", "rationale": "", "silent": False, "summary": "", "fold_read": False}),
    ({"answer": "Report", "action": "stop", "rationale": "Unfinished."},
     {"answer": "Report", "action": "stop", "rationale": "Unfinished.", "pending_review": None}),
    ({"answer": "Report"}, {"answer": "Report", "rationale": None}),
    ({"answer": "Report"}, {"answer": "Report", "rationale": "Accepted over the findings."}),
    ({"answer": "Report"}, {"answer": "Another report"}),
    ({"answer": "Report"}, {"answer": "Report", "action": "stop", "rationale": "Unfinished."}),
    ({"answer_sha256": "c" * 64}, {"answer_sha256": "d" * 64}),
], ids=["empty-unused-answer", "empty-unused-selector", "named-defaults", "null-optional", "null-rationale",
        "other-rationale", "other-answer", "other-action", "other-selector"])
def test_recorded_calls_are_one_request_exactly_when_the_staging_admits_the_repeat(tmp_path, first, second):
    """A restore reads two recorded calls as one request through the staging's own normalization, so it agrees
    with what the live batch did in both directions: an admitted repeat is one identity, a conflict is two."""
    from ouroboros.tools.control_runtime import _finish_task, staged_request_identity

    registry, _trace = _ctx(tmp_path)
    ctx = registry._ctx
    ctx._completion_observation, ctx._completion_request, ctx._completion_conflict = {"tool_count": 1}, None, False
    first, second = {"action": "finish", **first}, {"action": "finish", **second}
    for arguments in (first, second):
        _finish_task(ctx, **arguments)
    assert ctx._completion_request is not None  # the first call was staged; the second repeated or contradicted it
    admitted = not ctx._completion_conflict
    same = staged_request_identity(ctx, "finish_task", first) == staged_request_identity(ctx, "finish_task", second)
    assert same is admitted


@pytest.mark.parametrize(("tool", "arguments"), [
    ("finish_task", {"answer": "Report"}),
    ("finish_task", {"action": "finish", "answer": "Report", "outcome": "message"}),
    ("finish_task", {"action": ["finish"], "answer": "Report"}),
    ("finish_task", {"action": "finish", "answer": {"text": "Report"}}),
    ("finish_task", {"action": "finish", "answer": "Report", "silent": "false"}),
    ("finish_task", {"_error": "sanitization_failed"}),
    ("finish_task", None),
    ("presence_finish", {"action": "finish", "answer": "Report"}),
], ids=["omitted-action", "foreign-field", "unhashable-action", "object-answer", "text-for-a-flag", "unsanitized",
        "no-arguments", "another-command"])
def test_a_recorded_call_the_contract_does_not_read_is_identical_only_to_itself(tmp_path, tool, arguments):
    """Saved arguments name a request only as ``finish_task``'s own contract reads them. A default the handler
    never took, a malformed or foreign shape and another command's arguments are read as nothing: no request is
    inferred from them, so such a call is one act only beside the very same call."""
    from ouroboros.tools.control_runtime import staged_request_identity

    ctx = _ctx(tmp_path)[0]._ctx
    identity = staged_request_identity(ctx, tool, arguments)
    assert identity == (tool, None, arguments) == staged_request_identity(ctx, tool, json.loads(json.dumps(arguments)))
    assert identity != staged_request_identity(ctx, "finish_task", {"action": "finish", "answer": "Report"})


@pytest.mark.parametrize(("respelled", "credited"), [
    ({"answer": ""}, True), ({"rationale": "", "silent": False}, True), ({"rationale": "Another reason."}, False),
], ids=["empty-unused-answer", "named-defaults", "another-request"])
def test_an_older_wait_source_reads_a_respelled_repeat_as_the_one_selection_it_saved(tmp_path, respelled, credited):
    """A parked wait's source holds the selection an older body accepted without its author. The successor's round
    sent that choice twice; a repeat the staging admits as the same request leaves one act to attribute, and a
    different request beside it leaves the batch saying two things, so nothing is attributed."""
    registry, trace = _ctx(tmp_path)
    ctx, usage, execution, selector = registry._ctx, registry._ctx._accumulated_usage, "exec-wait", "c" * 64
    ctx.task_id, ctx.task_attempt = "waited", 1
    usage.update(_task_attempt=1, execution_id=execution, llm_call_refs=[
        {"llm_call_id": f"llm-{number}", "execution_id": execution, "round_id": f"{execution}:round:{number}",
         "round": number, "attempt": 1, "model": model, "usable_solve_response": True}
        for number, model in ((1, "a"), (2, "b"))])
    chosen = {"action": "finish", "answer_sha256": selector}
    control = {"tool": "finish_task", "completion_control": True, "status": "ok", "is_error": False,
               "round_id": f"{execution}:round:2"}
    trace["tool_calls"] = [{"tool": "read_file", "status": "ok", "is_error": False, "round_id": f"{execution}:round:1"},
                           {**control, "args": chosen}, {**control, "args": {**chosen, **respelled}}]
    pair = {"from_model": "a", "to_model": "b", "reason": "fallback", "tool_calls_at_handover": 1,
            "recovery_prompted": True, "status": "pending"}  # as the previous body wrote it, after its reminder
    trace["route_handovers"], usage["authoring_handovers"] = [pair], [dict(pair)]
    ctx._completion_observation = {"subject": {}, "tool_count": 1, "owner_source_sha256": "0" * 64}
    ctx._completion_selected = {**chosen, "rationale": "", "allow_empty": False, "source": "finish_task",
                                "reply_later": False, "observation": dict(ctx._completion_observation)}
    state = json.loads(json.dumps(continuation_state(ctx, [], trace, usage, 2, [], set())))

    registry, trace = _ctx(tmp_path)
    registry._ctx.task_id = "waited"
    usage = registry._ctx._accumulated_usage
    restore_continuation_state(registry, state, [], trace, usage, set())
    [row] = usage["authoring_handovers"]
    assert row["reply_observed"] is True and registry._ctx._completion_observation["model"] == "b"
    assert registry._ctx._completion_selected["observation"].get("model") == ("b" if credited else None)
    accept_handover_selection(registry._ctx, trace)
    assert (row["status"], row.get("selection")) == (("selected", "answer_sha256") if credited else ("pending", None))


@pytest.mark.parametrize("case", ["own-response", "own-response-restored", "later-response", "earlier-work",
                                  "another-round", "other-author", "forced-reply", "history-only"])
def test_only_the_response_whose_work_ended_the_row_adds_its_accepted_choice(tmp_path, case):
    """A request names its response by the tool count it was staged under. Its accepted choice joins the row that
    response's own work ended, also from a saved source, and no other: not a later reply's, not after an earlier
    batch already worked, not work of another round, another author's or a forced reply's, and not a row another
    execution left as history."""
    from ouroboros.loop_model_call import _record_authoring_handover

    registry, trace = _ctx(tmp_path)
    ctx, usage = registry._ctx, registry._ctx._accumulated_usage
    ctx.task_id = "worked"
    _record_authoring_handover(ctx, from_model="a", to_model="b", reason="fallback", tool_calls_at_handover=1)
    row = ctx._authoring_handover

    def batch(round_id, *names):
        calls = [{"tool": name, "status": "ok", "is_error": False, "round_id": round_id,
                  **({"completion_control": True} if name == "finish_task" else {})} for name in names]
        trace["tool_calls"] += calls
        observe_handover_work(ctx, trace, calls)

    if case == "earlier-work":
        batch("round-2", "read_file")
    start = len(trace["tool_calls"])  # where the stopping response's batch begins
    if case == "another-round":  # the stop's own response did nothing else; a later round's read ended the row
        batch("round-3", "finish_task")
        batch("round-4", "read_file")
    else:
        batch("round-3", "read_file", "finish_task")
    assert (row["status"], row["work_observed"]) == ("recovered", True) and ctx._authoring_handover is None
    stop = {"action": "stop", "rationale": "The second file is unread."}
    observation = {"model": "c" if case == "other-author" else "b",
                   "tool_count": len(trace["tool_calls"]) if case == "later-response" else start}
    if case == "history-only":
        usage.pop("authoring_handovers")
    elif case == "own-response-restored":  # the same facts read back from a saved source by the same task
        state = json.loads(json.dumps(continuation_state(ctx, [], trace, usage, 3, [], set())))
        registry, trace = _ctx(tmp_path)
        ctx, usage = registry._ctx, registry._ctx._accumulated_usage
        ctx.task_id = "worked"
        restore_continuation_state(registry, state, [], trace, usage, set())
        [row] = usage["authoring_handovers"]
    if case == "forced-reply":
        accept_handover_selection(ctx, trace, stop, forced_model="b")
    else:
        accept_handover_selection(ctx, trace, {**stop, "observation": observation})
    if case.startswith("own-response"):
        assert (row["status"], row["selection"], row["work_observed"]) == ("selected", "stop", True)
        assert usage["authoring_handovers"] == trace["route_handovers"] == [row]
    else:
        assert (row["status"], row["work_observed"]) == ("recovered", True) and "selection" not in row


def test_legacy_handover_target_without_an_answer_receipt_cannot_name_a_predecessor(tmp_path):
    registry, trace = _ctx(tmp_path)
    registry._ctx._authoring_handover = {"to_model": "planned", "status": "pending"}
    observe_authoring_response(registry._ctx, "answering")
    assert "route_handovers" not in trace and "authoring_handovers" not in registry._ctx._accumulated_usage


def test_voluntary_switch_after_an_unsettled_handover_does_not_open_another(tmp_path):
    from ouroboros.loop_model_call import _record_authoring_handover

    registry, trace = _ctx(tmp_path)
    ctx = registry._ctx
    _record_authoring_handover(ctx, from_model="a", to_model="b", reason="fallback", tool_calls_at_handover=1)
    row = ctx._authoring_handover
    ctx._completion_observation = {"model": "b"}
    ctx._pending_handover_reason = "mind_switch"
    observe_authoring_response(ctx, "c")
    assert ctx._authoring_handover is None and trace["route_handovers"] == [row]
    assert row["status"] == "pending" and "selection" not in row


def test_later_tools_heal_only_handover_warning_and_keep_history(tmp_path):
    registry, trace = _ctx(tmp_path)
    from ouroboros.loop_model_call import _record_authoring_handover
    _record_authoring_handover(registry._ctx, from_model="a", to_model="b",
                              reason="fallback", tool_calls_at_handover=1)
    row = registry._ctx._authoring_handover
    for _ in range(2):
        _maybe_inject_finalization_nudges(registry, tmp_path, "healing", trace,
                                         "status", [], lambda _: None)
    usage = registry._ctx._accumulated_usage
    assert derive_loop_outcome("status", usage, trace)["reason_code"] == "authoring_handover_incomplete"
    assert row["incomplete_observed"] and row["recovery_prompted"]
    trace["tool_calls"].append({"tool": "read_file", "status": "ok", "is_error": False})
    observe_handover_work(registry._ctx, trace, trace["tool_calls"][-1:])
    assert "authoring_handover_incomplete" not in trace
    assert "execution_status" not in usage
    assert derive_loop_outcome("finished", usage, trace)["outcome_axes"]["execution"]["status"] == "ok"
    assert row["status"] == "recovered" and row["incomplete_observed"]
    from ouroboros.task_finalization import terminal_result_fields
    assert terminal_result_fields(usage)["authoring_handovers"][-1] == row


@pytest.mark.parametrize("independent_failure", [False, True])
def test_selection_withdraws_an_incomplete_mark_and_keeps_its_history(tmp_path, independent_failure):
    from ouroboros.loop_model_call import _record_authoring_handover
    registry, trace = _ctx(tmp_path)
    registry._ctx.task_id = "report"
    usage = registry._ctx._accumulated_usage
    if independent_failure:
        usage.update(execution_status="infra_failed", reason_code="task_exception")
    _record_authoring_handover(registry._ctx, from_model="a", to_model="b",
                              reason="fallback", tool_calls_at_handover=1)
    for _ in range(2):  # the reminder, then a tool-less final that is neither work nor a selection
        _maybe_inject_finalization_nudges(registry, tmp_path, "report", trace, "status", [], lambda _: None)
    state = json.loads(json.dumps(continuation_state(registry._ctx, [], trace, usage, 2, [], set())))
    registry, trace = _ctx(tmp_path)
    registry._ctx.task_id = state["task_id"]
    usage = registry._ctx._accumulated_usage
    restore_continuation_state(registry, state, [], trace, usage, set())
    row = trace["route_handovers"][-1]
    assert row["status"] == "incomplete"
    registry._ctx._completion_selected = {"action": "finish", "answer": "Report", "observation": {"model": "b"}}
    assert not _maybe_inject_finalization_nudges(registry, tmp_path, "report", trace, "Report", [], lambda _: None)
    assert row["status"] == "incomplete"  # passing the reminder alone has settled nothing
    accept_handover_selection(registry._ctx, trace)
    assert "authoring_handover_incomplete" not in trace and usage["authoring_handovers"] == [row]
    assert (row["status"], row["selection"], row["incomplete_observed"]) == ("selected", "answer", True)
    if independent_failure:
        execution = derive_loop_outcome("Report", usage, trace)["outcome_axes"]["execution"]
        assert (execution["status"], execution["reason_code"]) == ("infra_failed", "task_exception")
    else:
        assert "execution_status" not in usage and "reason_code" not in usage


@pytest.mark.parametrize(("act", "ending"), [
    ("selection", {"status": "selected", "selection": "answer_sha256"}),
    ("tool_work", {"status": "recovered", "work_observed": True}),
])
def test_a_later_handover_ends_only_its_own_row_beside_an_earlier_incomplete_one(tmp_path, act, ending):
    """A→B is marked incomplete, then B→C switches: C's act ends B→C and withdraws the execution's degraded mark;
    the A→B row keeps its own history, since B made no such choice."""
    from ouroboros.loop_model_call import _record_authoring_handover
    registry, trace = _ctx(tmp_path)
    registry._ctx.task_id = "two-rows"
    _record_authoring_handover(registry._ctx, from_model="a", to_model="b",
                              reason="fallback", tool_calls_at_handover=1)
    earlier = registry._ctx._authoring_handover
    for _ in range(2):  # B's reminder, then a tool-less final that is neither work nor a selection
        _maybe_inject_finalization_nudges(registry, tmp_path, "two-rows", trace, "status", [], lambda _: None)
    assert trace["authoring_handover_incomplete"] is earlier and registry._ctx._authoring_handover is earlier
    _record_authoring_handover(registry._ctx, from_model="b", to_model="c",
                              reason="fallback", tool_calls_at_handover=1)
    assert _maybe_inject_finalization_nudges(registry, tmp_path, "two-rows", trace, "C status", [], lambda _: None)
    usage = registry._ctx._accumulated_usage
    state = json.loads(json.dumps(continuation_state(registry._ctx, [], trace, usage, 3, [], set())))
    registry, trace = _ctx(tmp_path)
    registry._ctx.task_id = state["task_id"]
    usage = registry._ctx._accumulated_usage
    restore_continuation_state(registry, state, [], trace, usage, set())
    earlier, later = trace["route_handovers"]
    if act == "selection":
        registry._ctx._completion_selected = {"action": "finish", "answer_sha256": "c" * 64, "observation": {"model": "c"}}
    else:
        trace["tool_calls"].append({"tool": "read_file", "status": "ok", "is_error": False})
        observe_handover_work(registry._ctx, trace, trace["tool_calls"][-1:])
    accept_handover_selection(registry._ctx, trace)
    assert {key: later.get(key) for key in ending} == ending and later["recovery_prompted"] is True
    assert (earlier["status"], earlier["incomplete_observed"], earlier.get("selection")) == ("incomplete", True, None)
    assert usage["authoring_handovers"] == trace["route_handovers"] == [earlier, later]
    assert "authoring_handover_incomplete" not in trace and registry._ctx._authoring_handover is None
    assert "execution_status" not in usage and "reason_code" not in usage
    assert derive_loop_outcome("Report", usage, trace)["outcome_axes"]["execution"]["status"] == "ok"


def test_repeated_status_does_not_skip_existing_skill_nudge(tmp_path, monkeypatch):
    from ouroboros import loop
    registry, trace = _ctx(tmp_path)
    registry._ctx._authoring_handover = {
        "from_model": "a", "to_model": "b", "tool_calls_at_handover": 1,
        "recovery_prompted": True,
    }
    monkeypatch.setattr(loop, "_skill_finalization_message", lambda *_: "skill still needs work")
    messages = []
    assert _maybe_inject_finalization_nudges(registry, tmp_path, "skill", trace,
                                           "status", messages, lambda _: None)
    assert "skill still needs work" in messages[-1]["content"]
    assert trace["authoring_handover_incomplete"]["status"] == "incomplete"
