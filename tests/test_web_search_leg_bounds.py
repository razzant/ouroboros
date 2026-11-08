"""#572: each provider web-search leg keeps its own per-attempt bound at the actual send.

A leg's bound is ``OUROBOROS_WEBSEARCH_TIMEOUT_SEC`` (480 s by default) narrowed by the task deadline
minus the finalization reserve and by the tool's execution window; a bound spent before the send sends
nothing. Real ``tools.search`` legs, recovery driver and accounting; the OpenAI SDK runs over an
in-memory transport, the Anthropic SDK (not a locked dependency) is the repo's usual stand-in, and the
clock is controlled.
"""
from __future__ import annotations

import json
import queue
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest

# The loop modules load before any fixture replaces model_wait.monotonic_now: they bind it at import, and one
# imported first under a fixture would keep that test's clock for every later test in this process.
from ouroboros import (  # noqa: F401
    deadline_utils,
    loop,
    loop_tool_execution,
    model_wait,
    net_transport,
    observability,
    review_custody,
    review_substrate,
)
from ouroboros import usage_accounting as ua
from ouroboros.llm_attempt import _PhysicalSendNotStarted
from ouroboros.model_wait import ModelWaitInterrupted
from ouroboros.tools import search
from tests._usage_store_testing import ledger_rows
from tests._cases import chosen_rows

PHASES = ("connect", "read", "write", "pool")


@pytest.fixture
def legs(tmp_path, monkeypatch):
    import ouroboros.request_wire_contract as wire
    from ouroboros import pricing

    now, seen = [100.0], []
    monkeypatch.setattr(deadline_utils, "utc_now", lambda: datetime.fromtimestamp(now[0], timezone.utc))
    monkeypatch.setattr(deadline_utils, "time", SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(model_wait, "monotonic_now", lambda slot=None: now[0])
    for key, value in {"OUROBOROS_DATA_DIR": str(tmp_path), "TOTAL_BUDGET": "100",
                       "OPENROUTER_API_KEY": "fixture", "ANTHROPIC_API_KEY": "fixture", "OPENAI_API_KEY": "fixture"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(wire, "canonical_wire_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(pricing, "estimate_cost_optional", lambda *a, **k: None)
    monkeypatch.setattr(ua, "_reservation_cost", lambda request: 1.0)

    def respond(request):  # the OpenAI SDK's actual HTTP request
        seen.append(dict(request.extensions["timeout"]))
        if request.url.path.endswith("/responses"):
            completed = {"type": "response.completed", "response": {
                "id": "resp", "object": "response", "created_at": 1, "model": "search-model", "status": "completed",
                "output": [], "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}}}
            body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in (
                {"type": "response.output_text.delta", "delta": "answer"}, completed))
            return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={
            "id": "gen", "object": "chat.completion", "created": 1, "model": "search-model",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "answer"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "cost": 0.01}})

    monkeypatch.setattr(net_transport, "remote_httpx_transport", lambda *a, **k: httpx.MockTransport(respond))

    def anthropic_client(**client_kwargs):  # stand-in: construction timeout, then any per-request override
        def create(**kwargs):
            seen.append(dict.fromkeys(PHASES, float(kwargs.get("timeout", client_kwargs.get("timeout")))))
            return {"content": [{"type": "text", "text": "answer"}], "usage": {"input_tokens": 3, "output_tokens": 2}}
        return SimpleNamespace(messages=SimpleNamespace(create=create))

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=anthropic_client))
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="search", root_task_id="search")):
        yield SimpleNamespace(root=tmp_path, now=now, seen=seen)


def run_leg(env, leg, *, deadline=None, tool_window=None, progress=None):
    metadata = {"deadline_at": datetime.fromtimestamp(deadline, timezone.utc).isoformat()} if deadline else {}
    ctx = SimpleNamespace(task_id="search", task_metadata=metadata, drive_root=env.root, emit_progress_fn=progress)
    send = {"openrouter": search._web_search_openrouter, "anthropic": search._web_search_anthropic,
            "openai": search._web_search_openai}[leg]
    if tool_window is None:
        return send(ctx, "query")
    with model_wait.execution_deadline_scope(env.now[0] + tool_window):  # the web_search tool body's window
        return send(ctx, "query")


def spend_while_recording(env, monkeypatch, seconds):
    original = observability.persist_physical_candidate

    def persist(*args, **kwargs):
        result = original(*args, **kwargs)
        env.now[0] += seconds
        return result

    monkeypatch.setattr(observability, "persist_physical_candidate", persist)


@pytest.mark.parametrize(("deadline", "window", "expected", "leg"), [
    (None, None, 480.0, "openrouter"),  # no deadline, no tool window: the configured socket bound, unchanged
    (None, 1500.0, 480.0, "openai"),  # the 1500 s web_search window used to replace the leg's 480 s
    (None, 40.0, 30.0, "openai"),  # a narrower tool window: what is left of it at the send
    (400.0, None, 170.0, "openrouter"),  # deadline − 120 s reserve − now 100, less the 10 s spent recording
    (400.0, 1500.0, 170.0, "anthropic"),
    (400.0, 180.0, 170.0, "anthropic"),  # the loop's own clamp to the same deadline − reserve: one reserve, not two
    (400.0, 40.0, 30.0, "openai"),
])
def test_each_search_leg_sends_with_what_is_left_of_its_bounds(legs, monkeypatch, leg, deadline, window, expected):
    """Each leg projected its socket bound before reserving and recording, so 10 s of that preparation kept
    the whole 180 s and rode into the finalization reserve; the direct leg also ignored a narrower window."""
    spend_while_recording(legs, monkeypatch, 10)
    assert "answer" in run_leg(legs, leg, deadline=deadline, tool_window=window)
    assert legs.seen == [dict.fromkeys(PHASES, expected)]


@pytest.mark.parametrize(("window", "leg"), [
    (None, "openrouter"),
    (1500.0, "openrouter"),
    (1500.0, "anthropic"),
    (40.0, "anthropic"),
])
def test_a_deadline_spent_while_a_recovery_leg_records_sends_nothing(legs, monkeypatch, leg, window):
    """181 s of recording passes the 180 s that deadline − reserve left. Without a tool window, or with a
    wider one, the leg still sent with the socket bound it projected first. Now it sends nothing, releases the
    attempt and returns its own deadline_exhausted result, as an exhausted leg already does before admission."""
    spend_while_recording(legs, monkeypatch, 181)
    assert json.loads(run_leg(legs, leg, deadline=400.0, tool_window=window))["reason_code"] == "deadline_exhausted"
    assert legs.seen == []
    assert [row["state"] for row in ledger_rows(legs.root)] == ["released"]


@pytest.mark.parametrize(("stage", "clock", "deadline", "window", "spent"), [
    ("recording", "deadline", 400.0, None, 181),  # 400 − 120 s reserve − now 100 leaves 180 s
    ("recording", "tool_window", None, 40.0, 41),
    ("progress", "deadline", 400.0, None, 181),  # 400 − 120 s reserve − now 100 leaves 180 s
])
def test_a_bound_spent_before_the_direct_send_sends_nothing(legs, monkeypatch, stage, clock, deadline, window, spent):
    """Recording the candidate, and after the dispatch claim its progress row, spend the send's own clock.
    The deadline once sent a 0.001 s request; a spent bound now sends no bytes and releases the attempt."""
    def spend_in_progress(_text):  # the "Trying OpenAI web search" row, written after the claim
        legs.now[0] += spent

    if stage == "recording":
        spend_while_recording(legs, monkeypatch, spent)

    def search_once():
        return run_leg(legs, "openai", deadline=deadline, tool_window=window,
                       progress=spend_in_progress if stage == "progress" else None)

    if clock == "deadline":  # the leg's own window, before or after its dispatch claim: its own result
        assert json.loads(search_once())["reason_code"] == "deadline_exhausted"
    else:  # a caller's tool window at the claim or send, with no deadline: typed, unsent
        with pytest.raises(_PhysicalSendNotStarted) as refused:
            search_once()
        assert refused.value.physical_attempt_capture.state == "released"
    assert legs.seen == []
    assert [row["state"] for row in ledger_rows(legs.root)] == ["released"]



def at(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def search_round(env, monkeypatch, backend, *, deadline=400.0, stage=None, spent=0.0, effect=None,
                 dies_after_send=False):
    """One real ``run_llm_loop`` round whose model calls ``web_search`` under a real wait owner, then a scripted answer.

    ``stage`` spends ``spent`` seconds, then runs ``effect``, just before the leg reserves ("reservation"), while it
    records its candidate ("recording") or right after its dispatch claim ("claim"); ``dies_after_send`` spends them
    while a written request dies. Only the two model rounds and the loop's own graceful final answer (observed at
    its seam) are scripted."""
    from ouroboros.task_results import write_task_result
    from tests.test_billing_authority_unavailable import _loop_driver

    tid = "search"
    write_task_result(env.root, tid, "running", root_task_id=tid)
    _same_loop, registry, _ = _loop_driver(env.root, monkeypatch, tid)
    monkeypatch.setattr(loop, "utc_now", lambda: datetime.fromtimestamp(env.now[0], timezone.utc))
    # Both hold their own import-time reference: the tool window must open on this test's clock, not a former one.
    monkeypatch.setattr(loop_tool_execution, "monotonic_now", lambda slot=None: env.now[0])
    monkeypatch.setenv("OUROBOROS_WEBSEARCH_BACKEND", backend)
    registry._ctx.is_direct_chat = True
    bound = {"deadline_at": at(deadline)} if deadline else {}
    registry._ctx.task_metadata = {"root_task_id": tid, **bound}

    def spend():
        env.now[0] += spent
        if effect is not None:
            effect()

    if stage == "reservation":
        for module in (ua, search):  # the recovery driver's reservation, then the direct leg's own
            def reserve(*args, _reserve=module.reserve_attempt, **kwargs):
                spend()
                return _reserve(*args, **kwargs)

            monkeypatch.setattr(module, "reserve_attempt", reserve)
    elif stage == "recording":
        recorded = observability.persist_physical_candidate

        def persist(*args, **kwargs):
            result = recorded(*args, **kwargs)
            spend()
            return result

        monkeypatch.setattr(observability, "persist_physical_candidate", persist)
    elif stage == "claim":
        for module in (ua, search):  # the recovery driver's claim, then the direct leg's own
            def claim(*args, _claim=module.mark_dispatched, **kwargs):
                _claim(*args, **kwargs)
                spend()

            monkeypatch.setattr(module, "mark_dispatched", claim)
    if dies_after_send:
        def died(request):
            env.seen.append(dict(request.extensions["timeout"]))
            spend()
            raise httpx.ReadError("connection reset after the request was written", request=request)

        def anthropic_died(**client_kwargs):
            def create(**kwargs):
                env.seen.append(dict.fromkeys(PHASES, float(kwargs.get("timeout", client_kwargs.get("timeout")))))
                spend()
                raise ConnectionResetError("connection reset after the request was written")
            return SimpleNamespace(messages=SimpleNamespace(create=create))

        monkeypatch.setattr(net_transport, "remote_httpx_transport", lambda *a, **k: httpx.MockTransport(died))
        monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=anthropic_died))
    rounds, finals = [], []

    def model(call):
        rounds.append([dict(row) for row in call.messages])
        if len(rounds) == 1:
            calls = [{"id": "search-1", "type": "function",
                      "function": {"name": "web_search", "arguments": json.dumps({"query": "pixel"})}}]
            return {"role": "assistant", "content": None, "tool_calls": calls}, 0.01, call.active_context_mode
        return {"role": "assistant", "content": "Answered after the search."}, 0.01, call.active_context_mode

    def graceful(ctx, **kwargs):  # the loop's own deadline finalization: one tool-less final answer call
        finals.append({"reason_code": kwargs.get("reason_code"),
                       "tool_rows": [str(row.get("content")) for row in ctx.messages if row.get("role") == "tool"]})
        return "Graceful final answer.", {"reason_code": kwargs.get("reason_code")}, {}

    monkeypatch.setattr(loop, "_call_round_model", model)
    monkeypatch.setattr(loop, "_forced_final_answer", graceful)
    task = {"id": tid, "root_task_id": tid, "_attempt": 1, "_is_direct_chat": True, **bound}
    run = SimpleNamespace(text=None, usage={}, trace={}, raised=None, rounds=rounds, finals=finals)
    try:
        with model_wait.task_model_wait_scope(task=task, drive_root=env.root, event_queue=None, worker_slot_held=False):
            run.text, run.usage, run.trace = loop.run_llm_loop(
                messages=[{"role": "user", "content": "Search for the pixel"}], tools=registry,
                llm=SimpleNamespace(default_model=lambda: "fixture"), drive_logs=env.root / "logs",
                emit_progress=lambda *_a, **_k: None, incoming_messages=queue.Queue(),
                task_id=tid, drive_root=env.root, task_type="presence")
    except ModelWaitInterrupted as exc:
        run.raised = exc.control_reason
    return run


LOOP_LEGS = ["openrouter", "anthropic", "openai"]


def states(env):
    return [row["state"] for row in ledger_rows(env.root)]


@pytest.mark.parametrize("backend", chosen_rows(LOOP_LEGS, 0, 2))
@pytest.mark.parametrize("stage,ledger", [("reservation", []), ("recording", ["released"]), ("claim", ["released"])])
def test_a_search_window_spent_before_the_send_leaves_the_loop_its_graceful_finish(legs, monkeypatch, backend,
                                                                                  stage, ledger):
    """The leg's own deadline − reserve window ends before it reserves, while it records its candidate or right
    after its claim; the loop's tool window, clamped to the same deadline, ends with it. Nothing is sent and the
    tool result is deadline_exhausted, so the loop's own deadline finalization buys its one graceful final answer;
    a task-level no-call stop used to replace that answer. No release is invented where nothing was reserved."""
    run = search_round(legs, monkeypatch, backend, stage=stage, spent=181)
    assert (run.text, run.raised, len(run.rounds)) == ("Graceful final answer.", None, 1)
    [final] = run.finals
    assert final["reason_code"] == "deadline_local"
    assert [json.loads(row)["reason_code"] for row in final["tool_rows"]] == ["deadline_exhausted"]
    assert legs.seen == [] and states(legs) == ledger


@pytest.mark.parametrize("backend", chosen_rows(LOOP_LEGS, 0, 2))
@pytest.mark.parametrize("deadline,read", [(400.0, 170.0), (None, 480.0)], ids=["timely", "unbounded"])
def test_a_search_with_time_left_or_no_deadline_still_answers(legs, monkeypatch, backend, deadline, read):
    """10 s of recording: the send gets what is left of the window, or with no deadline the plain 480 s socket."""
    run = search_round(legs, monkeypatch, backend, deadline=deadline, stage="recording", spent=10)
    assert (run.text, run.raised, run.finals, len(run.rounds)) == ("Answered after the search.", None, [], 2)
    [tool_row] = [row["content"] for row in run.rounds[1] if row.get("role") == "tool"]
    assert "answer" in json.loads(tool_row)
    assert [timeout["read"] for timeout in legs.seen] == [read] and states(legs) == ["settled"]


@pytest.mark.parametrize("backend", LOOP_LEGS)
def test_the_tasks_own_stop_keeps_its_rail(legs, monkeypatch, backend):
    """Stopped while it records: this direct chat turn takes the owner's Stop rail, not a search result."""
    from ouroboros import cancel_intents
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION

    run = search_round(legs, monkeypatch, backend, stage="recording",
                       effect=lambda: cancel_intents.request_cancel(legs.root, "search", source="search-test"))
    assert run.text == "⏹ The owner stopped this chat turn; no further work was done." and run.raised is None
    assert run.usage["reason_code"] == REASON_OWNER_REQUESTED_FINALIZATION
    assert run.trace["forced_finalization"]["source"] == "owner_stopped_direct_turn"
    assert run.finals == [] and len(run.rounds) == 1 and legs.seen == [] and states(legs) == ["released"]


@pytest.mark.parametrize("backend", LOOP_LEGS)
def test_the_owners_pause_keeps_its_rail(legs, monkeypatch, backend):
    """Only the fence read is replaced: a parked member would otherwise wait here for its Resume."""
    from ouroboros import budget_pause, owner_pause

    paused, entered = [], []
    member_fence = owner_pause.member_fence
    monkeypatch.setattr(owner_pause, "member_fence", lambda source, **kwargs: (
        {"state": owner_pause.FENCE_PAUSED, "reason": "owner_pause", "fence_id": "fence-1"} if paused
        else member_fence(source, **kwargs)))
    monkeypatch.setattr(budget_pause, "enter_owner_pause", lambda _limit_ctx: entered.append(True) if paused else None)
    run = search_round(legs, monkeypatch, backend, stage="recording", effect=lambda: paused.append(True))
    assert run.raised == "owner_pause" and entered == [True] and run.finals == [] and len(run.rounds) == 1
    assert legs.seen == [] and states(legs) == ["released"]


@pytest.mark.parametrize("backend", LOOP_LEGS)
def test_the_tasks_own_deadline_keeps_its_rail(legs, monkeypatch, backend):
    """301 s after the claim the task's deadline itself has passed: the leg's window is no longer the cause."""
    run = search_round(legs, monkeypatch, backend, stage="claim", spent=301)
    assert run.raised is None and run.finals == [] and len(run.rounds) == 1
    assert run.usage["reason_code"] == "deadline_local"
    assert run.trace["forced_finalization"]["source"] == "model_wait_control"
    assert legs.seen == [] and states(legs) == ["released"]


@pytest.mark.parametrize("backend", LOOP_LEGS)
def test_a_send_that_died_after_its_bytes_stays_unknown_and_is_not_retried(legs, monkeypatch, backend):
    """The window also runs out while the written request dies: still an unknown outcome, one request, no resend."""
    run = search_round(legs, monkeypatch, backend, spent=181, dies_after_send=True)
    assert (run.text, run.raised, len(run.rounds)) == ("Graceful final answer.", None, 1)
    [final] = run.finals
    assert [json.loads(row)["reason_code"] for row in final["tool_rows"]] == ["provider_outcome_unknown"]
    assert len(legs.seen) == 1 and states(legs) == ["unresolved"]



@pytest.mark.parametrize("state,reason,spent,unreadable,expected", [
    ("released", "deadline", True, False, True),
    (None, "execution_deadline", True, False, True),
    ("dispatched", "deadline", True, False, False),   # a possibly sent attempt is never relabelled
    ("unresolved", "deadline", True, False, False),
    ("reserved", "deadline", True, False, False),     # an open reservation is no unsent proof
    ("released", "cancelled", True, False, False),
    ("released", "owner_pause", True, False, False),
    ("released", "deadline", False, False, False),    # the caller's own window still has time
    ("released", "deadline", True, True, False),      # an unreadable control is no clear one
], ids=["released", "no_capture", "dispatched", "unresolved", "reserved", "stop", "pause", "time_left", "unreadable"])
def test_only_a_positively_unsent_refusal_by_the_callers_own_window_is_its_own_result(monkeypatch, state, reason,
                                                                                       spent, unreadable, expected):
    from ouroboros import llm_attempt
    from ouroboros.usage_accounting import PhysicalAttemptCapture

    refusal = llm_attempt._PhysicalSendNotStarted(reason)
    if state is not None:
        refusal.physical_attempt_capture = PhysicalAttemptCapture("attempt", "model", "openai", state, "opaque")
    if unreadable:
        from ouroboros.store import StoreBusy

        def busy():
            raise StoreBusy("controls unreadable")

        monkeypatch.setattr(model_wait, "owner_control_reason", busy)
    assert llm_attempt.refused_by_own_window(refusal, own_window_spent=spent) is expected
