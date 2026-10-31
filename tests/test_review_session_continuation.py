"""A stopped review continues inside its original seat and durable custody."""

import copy

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.gateways.claudexor import ClaudexorGateway, ClaudexorUnavailable, DaemonEndpoint
from ouroboros.review_execution import SessionInvocation, run_delegated_review_session
from ouroboros.review_session_custody import validate_producer_invocation
from tests._review_session_route_shared import (
    FakeGateway, _terminal_detail,
    _owned_gateway_uses_each_test_transport, fake_route,  # noqa: F401
)


def _engine(monkeypatch, *, resumable=True, supported=True, refusal=False, lose_response=False,
            stops=1, final_failed=False):
    accepted, posts = {}, []

    def capabilities(self):
        return {"harnesses": [dict(FakeGateway.catalog_entry)],
                "runControlKeys": ["continueFrom"] if supported else []}

    def continued(self, run_id, *, idempotency_key=""):
        posts.append((run_id, idempotency_key))
        if refusal:
            raise ClaudexorUnavailable("continuation_refused", "Cannot continue here", status_code=409)
        handle = accepted.setdefault(idempotency_key, {"runId": f"run-{len(accepted) + 2}"})
        if lose_response and len(posts) == 1:
            raise OSError("Continuation accepted, response lost")
        return handle

    def detail(self, run_id, **_kw):
        self.run_gets.append(run_id)
        n = int(run_id.removeprefix("run-"))
        failed = n <= stops or final_failed
        result = _terminal_detail("partial" if failed else "[]",
                                  state="failed" if failed else "succeeded", conformance="passed")
        result["summary"]["spendUsd"] = float(n)
        if failed:
            result["summary"]["failure"] = {"code": "credential_pool_exhausted",
                "safeMessage": "Account limit reached during review", "resetsAt": "2030-01-01T00:00:00Z"}
            if resumable and n <= stops:
                result["summary"]["resumable"] = {"cause": "pool_exhausted", "carriers": ["native", "packet"]}
        return copy.deepcopy(result)

    monkeypatch.setattr(FakeGateway, "agent_capabilities", capabilities)
    first_start = FakeGateway.start_run

    def start(self, request, *, idempotency_key=""):
        if "continueFrom" in request:  # a continuation is a start naming the stopped run
            return continued(self, request["continueFrom"], idempotency_key=idempotency_key)
        return first_start(self, request, idempotency_key=idempotency_key)

    monkeypatch.setattr(FakeGateway, "start_run", start)
    monkeypatch.setattr(FakeGateway, "get_run", detail)
    return accepted, posts


def _run(tmp_path, state, checkpoints, stamps, *, reconcile=False):
    return run_delegated_review_session(prompt="Review this frozen subject", root=str(tmp_path),
        custody_drive=tmp_path, invocation=SessionInvocation(task_id="t-review", surface="multi_model_review",
            slot_id="seat", operation_id="op-review", timeout_sec=60, retry_state=state,
            reconcile_only=reconcile, pending_invocation_checkpoint=checkpoints.append,
            dispatch_stamp=lambda: stamps.append("paid")))


@pytest.mark.parametrize("stops", [1, 2])
def test_continues_each_new_stop_and_collects_in_same_seat(tmp_path, monkeypatch, fake_route, stops):  # noqa: F811
    accepted, posts = _engine(monkeypatch, stops=stops)
    state, checkpoints, stamps = {}, [], []
    facts = _run(tmp_path, state, checkpoints, stamps)
    assert facts["text"] == "[]" and facts["run_id"] == f"run-{stops + 1}"
    assert len(accepted) == len(posts) == stops and stamps == ["paid"]
    assert len(checkpoints) == 1  # Original slot binding is the recovery anchor.
    assert facts["spend"] == sum(float(n) for n in range(1, stops + 2))
    assert [r["from_run_id"] for r in facts["review_continuations"]] == [f"run-{n}" for n in range(1, stops + 1)]
    record = custody.invocation_record(tmp_path, checkpoints[0])
    assert record["run_id"] == "run-1" and record["operation_id"] == "op-review"
    expected = {k: record[k] for k in ("task_id", "root_task_id", "surface", "slot_id", "operation_id")}
    validate_producer_invocation(tmp_path, checkpoints[0], expected,
        {"usage": {"delegated_run_id": facts["run_id"]}}, {"status": "ok"}, error_source=False)
    assert state == {}


@pytest.mark.parametrize("resumable,supported,refusal", [(False, True, False), (True, False, False), (True, True, True)])
def test_unavailable_continuation_keeps_original_typed_error(tmp_path, monkeypatch, fake_route,  # noqa: F811
                                                            resumable, supported, refusal):
    _accepted, posts = _engine(monkeypatch, resumable=resumable, supported=supported, refusal=refusal)
    checkpoints, stamps = [], []
    with pytest.raises(ClaudexorUnavailable) as error:
        _run(tmp_path, {}, checkpoints, stamps)
    assert error.value.code == "credential_pool_exhausted"
    assert error.value.reported_cause == "Account limit reached during review"
    assert len(posts) == int(refusal)
    with pytest.raises(ClaudexorUnavailable) as replay:
        _run(tmp_path, {"pending_invocation_id": checkpoints[0]}, checkpoints, stamps, reconcile=True)
    assert replay.value.code == error.value.code and len(posts) == int(refusal)
    assert stamps == ["paid"]


@pytest.mark.parametrize("crash", ["response_lost", "after_binding"])
def test_restart_rejoins_successor_from_original_slot_token(tmp_path, monkeypatch, fake_route, crash):  # noqa: F811
    accepted, posts = _engine(monkeypatch, lose_response=crash == "response_lost")
    from ouroboros import review_execution

    poll = review_execution._poll_session_terminal
    crashed = []

    def crash_once(gateway, custody_module, drive, entry, run_id, seconds):
        if crash == "after_binding" and run_id == "run-2" and not crashed:
            crashed.append(True)
            raise OSError("Process stopped before collecting the successor")
        return poll(gateway, custody_module, drive, entry, run_id, seconds)

    monkeypatch.setattr(review_execution, "_poll_session_terminal", crash_once)
    state, checkpoints, stamps = {}, [], []
    with pytest.raises(OSError):
        _run(tmp_path, state, checkpoints, stamps)
    anchor = checkpoints[0]
    custody._CUSTODY.clear()
    # A restarted caller knows only what its durable slot checkpoint retained.
    facts = _run(tmp_path, {"pending_invocation_id": anchor}, checkpoints, stamps, reconcile=True)
    assert facts["text"] == "[]" and facts["run_id"] == "run-2"
    assert len(accepted) == 1 and stamps == ["paid"] and checkpoints == [anchor]
    assert len(posts) == (2 if crash == "response_lost" else 1)
    assert len({key for _run_id, key in posts}) == 1


def test_gateway_continuation_uses_existing_start_protocol(monkeypatch):
    gateway = ClaudexorGateway(DaemonEndpoint(host="127.0.0.1", port=1, token="fixture-token"))
    seen = []
    monkeypatch.setattr(gateway, "_request", lambda method, path, **kw: seen.append((method, path, kw)) or {"runId": "next"})
    try:
        assert gateway.start_run({"continueFrom": "stopped"}, idempotency_key="once")["runId"] == "next"
        assert seen == [("POST", "/v2/runs", {"json_body": {"continueFrom": "stopped"},
                                               "headers": {"Idempotency-Key": "once"}})]
    finally:
        gateway.close()


@pytest.mark.parametrize("final_failed", [False, True])
def test_completed_producer_replays_its_continued_answer(tmp_path, monkeypatch, fake_route, final_failed):  # noqa: F811
    from ouroboros import review_operation, review_substrate
    from tests._review_session_route_shared import _agent_request, _agent_slot, FakeLLM

    accepted, posts = _engine(monkeypatch, final_failed=final_failed)
    request = _agent_request(session_root=str(tmp_path), task_id="t-review", retry_key="wave-one")
    slot = _agent_slot()
    result = review_substrate.run_review_request(request, slots=[slot], drive_root=tmp_path, llm=FakeLLM())
    actor = result.actors[0]
    assert actor["status"] == ("error" if final_failed else "ok")
    if final_failed:
        assert actor["failure_code"] == "credential_pool_exhausted"
        assert actor["reported_cause"] == "Account limit reached during review"
    assert actor["usage"]["review_continuations"][0]["run_id"] == "run-2"
    original = custody.lookup(tmp_path, "t-review", "run-1")[1]
    successor = custody.lookup(tmp_path, "t-review", "run-2")[1]
    assert original.review_owned and successor.review_owned and original.review_wave_id
    for field in ("root_task_id", "parent_task_id", "source", "category", "review_skill", "review_wave_id", "review_slot_id"):
        assert getattr(successor, field) == getattr(original, field)
    stored_request, stored_slot = review_operation.recorded_review_assignment(tmp_path,
        task_id="t-review", surface=request.surface, row=actor, retry_key="wave-one")
    custody._CUSTODY.clear()
    recovered = review_operation.recover_review_producer(tmp_path, stored_request, stored_slot, actor)
    assert recovered.status == actor["status"] and recovered.raw_text == actor["raw_text"]
    assert len(accepted) == len(posts) == 1


@pytest.mark.parametrize("bound", [False, True])
def test_continued_thread_collects_answer_with_only_observed_thread_binding(tmp_path, monkeypatch, fake_route, bound):  # noqa: F811
    _engine(monkeypatch)
    monkeypatch.setattr(FakeGateway, "start_thread_turn",
        lambda *_a, **_k: {"runId": "run-1", "turnId": "turn-1"}, raising=False)
    turns = [{"id": "turn-1", "runId": "run-1"}]
    if bound:
        turns.append({"id": "turn-2", "runId": "run-2"})
    monkeypatch.setattr(FakeGateway, "get_thread", lambda *_a: {"turns": turns}, raising=False)
    facts = run_delegated_review_session(prompt="Review the plan", root=str(tmp_path), custody_drive=tmp_path,
        invocation=SessionInvocation(task_id="t-review", surface="plan_review", slot_id="seat",
            operation_id="op-thread", timeout_sec=60, use_thread=True, thread_id="thread-1"))
    assert facts["text"] == "[]" and facts["run_id"] == "run-2"
    assert facts["thread_id"] == ("thread-1" if bound else "")
    assert facts["turn_id"] == ("turn-2" if bound else "")
    if not bound:
        assert facts["thread_receipt"]["prior_thread_id"] == "thread-1"
        assert facts["thread_receipt"]["status"] == "continuation_not_in_thread"
