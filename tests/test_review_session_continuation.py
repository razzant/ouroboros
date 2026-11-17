"""A stopped review session creates no successor of its own; only its author's explicit choice does.

The transport keeps the stopped run's typed stop facts (the engine's ``resumable`` and
``outcomeFacts``) and its whole partial output as evidence, and joins only a successor an older
host already recorded automatically under the seat's anchor (``review_continuation`` owns new ones).
"""

import copy

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.gateways.claudexor import ClaudexorGateway, ClaudexorUnavailable, DaemonEndpoint
from ouroboros.review_execution import SessionInvocation, run_delegated_review_session
from tests._review_session_route_shared import (
    FakeGateway, _terminal_detail,
    _owned_gateway_uses_each_test_transport, fake_route,  # noqa: F401
)

RESUMABLE = {"cause": "pool_exhausted", "carriers": ["native", "packet"], "resetsAt": None, "session": None,
             "workspace": {"kind": "none", "root": None}}


def _engine(monkeypatch, *, resumable=True, stops=1, lose_response=False):
    """``lose_response``: the engine accepts the first continuation POST (its run exists) and the reply is lost."""
    posts, accepted = [], {}

    def capabilities(self):
        return {"harnesses": [dict(FakeGateway.catalog_entry)], "runControlKeys": ["continueFrom"]}

    def detail(self, run_id, **_kw):
        self.run_gets.append(run_id)
        n = int(run_id.removeprefix("run-"))
        failed = n <= stops
        result = _terminal_detail("PARTIAL REVIEW NOTES" if failed else "[]",
                                  state="failed" if failed else "succeeded", conformance="passed")
        result["summary"]["spendUsd"] = float(n)
        if failed:
            result["summary"]["failure"] = {"code": "credential_pool_exhausted",
                                            "safeMessage": "Account limit reached during review"}
            result["summary"]["outcomeFacts"] = {"work_state": "unfinished"}
            if resumable:
                result["summary"]["resumable"] = copy.deepcopy(RESUMABLE)
        return result

    first_start = FakeGateway.start_run

    def start(self, request, *, idempotency_key=""):
        posts.append((dict(request), idempotency_key))
        if "continueFrom" in request:  # one run per idempotency key, like the engine
            self.start_requests.append(dict(request))
            run_id = accepted.setdefault(idempotency_key, f"run-{len(accepted) + 2}")
            if lose_response and sum("continueFrom" in sent for sent, _key in posts) == 1:
                raise OSError("Continuation accepted, response lost")
            return {"runId": run_id, "storeId": "store-r"}  # the admitting store, as an opted-in receipt names it
        return first_start(self, request, idempotency_key=idempotency_key)

    monkeypatch.setattr(FakeGateway, "agent_capabilities", capabilities)
    monkeypatch.setattr(FakeGateway, "start_run", start)
    monkeypatch.setattr(FakeGateway, "get_run", detail)
    return posts


def _run(tmp_path, state, checkpoints, stamps, *, reconcile=False, **extra):
    return run_delegated_review_session(prompt="Review this frozen subject", root=str(tmp_path),
        custody_drive=tmp_path, invocation=SessionInvocation(task_id="t-review", surface="multi_model_review",
            slot_id="seat", operation_id="op-review", timeout_sec=60, retry_state=state,
            reconcile_only=reconcile, pending_invocation_checkpoint=checkpoints.append,
            dispatch_stamp=lambda: stamps.append("paid"), **extra))


@pytest.mark.parametrize("resumable", [True, False])
def test_a_stopped_run_starts_no_successor_and_keeps_its_stop_facts_and_partial_output(tmp_path, monkeypatch, fake_route,  # noqa: F811
                                                                                       resumable):
    posts = _engine(monkeypatch, resumable=resumable)
    checkpoints, stamps = [], []
    with pytest.raises(ClaudexorUnavailable) as error:
        _run(tmp_path, {}, checkpoints, stamps)
    assert [request.get("continueFrom") for request, _key in posts] == [None]  # nothing continued it
    assert error.value.code == "credential_pool_exhausted"
    assert error.value.reported_cause == "Account limit reached during review"
    assert error.value.review_partial_output == "PARTIAL REVIEW NOTES"  # whole output, evidence only
    details = error.value.failure_details
    assert details["outcomeFacts"] == {"work_state": "unfinished"}
    assert (details.get("resumable") or {}).get("cause") == ("pool_exhausted" if resumable else None)
    # A later collection with the seat's token replays nothing and keeps the same typed failure.
    with pytest.raises(ClaudexorUnavailable) as replay:
        _run(tmp_path, {"pending_invocation_id": checkpoints[0]}, checkpoints, stamps, reconcile=True)
    assert replay.value.code == error.value.code and len(posts) == 1 and stamps == ["paid"]


def legacy_successor(tmp_path, checkpoints, *, started=False):
    """What an older host left behind: the stopped run-1 and the automatic successor it recorded for it."""
    with pytest.raises(ClaudexorUnavailable):
        _run(tmp_path, {}, checkpoints, [])
    return record_legacy_successor(tmp_path, checkpoints[0], started=started)


_LEGACY_KEYS = ("root_task_id", "parent_task_id", "category", "source", "review_skill", "review_wave_id",
                "review_slot_id", "authority_source", "execution_root", "target_root")


def record_legacy_successor(root, anchor, *, started=False):
    """The automatic successor an older host recorded for the stopped run of seat token ``anchor``, as its
    continuation step wrote it: the START naming the stopped run under ``("review_continue", invocation, run)``
    and, once bound, the successor ``run-2``'s STARTED row. The current host only joins it."""
    seat = custody.invocation_record(root, anchor)
    stopped = custody.lookup(root, seat["task_id"], seat["run_id"])[1]
    token = custody.idempotency_key("review_continue", stopped.invocation_id, stopped.run_id)
    assert custody.record_start_requested(root, run_id="", task_id=stopped.task_id, invocation_id=token,
        idempotency_key=token, operation_id=seat["operation_id"], request={"continueFrom": stopped.run_id},
        continuation_of=stopped.run_id, surface=seat["surface"], slot_id=seat["slot_id"], route=stopped.route_id,
        max_seconds=stopped.max_seconds, project_id=stopped.project_id, project_owned=stopped.project_owned,
        **{key: getattr(stopped, key) for key in _LEGACY_KEYS})
    if started:
        successor = custody.RunCustody(run_id="run-2", invocation_id=token, idempotency_key=token,
            continuation_of=stopped.run_id, ledger_root=str(root), **{key: getattr(stopped, key) for key in (
                "task_id", "route_id", "model", "profile_id", "project_id", "project_owned", "project_persistent",
                *_LEGACY_KEYS)})
        assert custody.record_started(root, successor, shape={
            "access": stopped.access, "mode": stopped.mode, "isolation": stopped.isolation,
            "delegated": stopped.delegated, "max_seconds": stopped.max_seconds,
            "surface": seat["surface"], "slot_id": seat["slot_id"]})
    custody._CUSTODY.clear()
    return token


@pytest.mark.parametrize("legacy_state", ["started", "pending"])
def test_a_legacy_successor_recorded_by_an_older_host_is_joined_never_recreated(tmp_path, monkeypatch, fake_route,  # noqa: F811
                                                                              legacy_state):
    posts = _engine(monkeypatch, stops=1)
    checkpoints, stamps = [], []
    token = legacy_successor(tmp_path, checkpoints, started=legacy_state == "started")
    facts = _run(tmp_path, {"pending_invocation_id": checkpoints[0]}, checkpoints, stamps, reconcile=True)
    assert facts["text"] == "[]" and facts["run_id"] == "run-2"
    replayed = [key for request, key in posts if "continueFrom" in request]
    assert replayed == ([] if legacy_state == "started" else [token])  # the recorded POST, its own key and bytes
    custody._CUSTODY.clear()  # a joined successor binds only the store its own receipt named
    assert custody.replay(tmp_path)["run-2"].daemon_store_id == ("" if legacy_state == "started" else "store-r")


def test_gateway_continuation_uses_existing_start_protocol(monkeypatch):
    gateway = ClaudexorGateway(DaemonEndpoint(host="127.0.0.1", port=1, token="fixture-token"))
    seen = []
    monkeypatch.setattr(gateway, "_request", lambda method, path, **kw: seen.append((method, path, kw)) or {"runId": "next"})
    try:
        assert gateway.start_run({"continueFrom": "stopped"}, idempotency_key="once")["runId"] == "next"
        assert seen == [("POST", "/v2/runs", {"json_body": {"continueFrom": "stopped"},
                                               "headers": {"Idempotency-Key": "once", "X-Claudexor-Store-Identity": "1"}})]
    finally:
        gateway.close()


def test_a_stopped_producer_retains_its_whole_partial_output_as_evidence_not_a_verdict(tmp_path, monkeypatch,
                                                                                       fake_route):  # noqa: F811
    from ouroboros import review_operation, review_substrate
    from ouroboros.observability import read_call_payload
    from tests._review_session_route_shared import _agent_request, _agent_slot, FakeLLM

    posts = _engine(monkeypatch)
    request = _agent_request(session_root=str(tmp_path), task_id="t-review", retry_key="wave-one")
    result = review_substrate.run_review_request(request, slots=[_agent_slot()], drive_root=tmp_path, llm=FakeLLM())
    actor = result.actors[0]
    assert actor["status"] == "error" and actor["failure_code"] == "credential_pool_exhausted"
    assert actor["failure_details"]["resumable"]["cause"] == "pool_exhausted"
    assert actor["outcome"]["usage_ref"]["delegated_run_id"] == "run-1"
    payload = read_call_payload(tmp_path, task_id="t-review", call_id=f"{actor['operation_id']}_error")[1]
    assert payload["message"] == {"content": "", "session_transcript": "PARTIAL REVIEW NOTES"}
    custody._CUSTODY.clear()
    stored_request, stored_slot = review_operation.recorded_review_assignment(
        tmp_path, task_id="t-review", surface=request.surface, row=actor, retry_key="wave-one")
    recovered = review_operation.recover_review_producer(tmp_path, stored_request, stored_slot, actor)
    assert recovered.status == "error" and recovered.raw_text == ""  # never parsed into an answer
    assert len(posts) == 1


def test_a_stopped_thread_turn_is_never_continued_from_its_run(tmp_path, monkeypatch, fake_route):  # noqa: F811
    posts = _engine(monkeypatch)
    turns = []
    monkeypatch.setattr(FakeGateway, "start_thread_turn",
                        lambda self, thread_id, request, **_k: turns.append(dict(request)) or {"runId": "run-1", "turnId": "t1"},
                        raising=False)
    monkeypatch.setattr(FakeGateway, "get_thread", lambda *_a: {"turns": [{"id": "t1", "runId": "run-1"}]}, raising=False)
    with pytest.raises(ClaudexorUnavailable) as error:
        run_delegated_review_session(prompt="Review the plan", root=str(tmp_path), custody_drive=tmp_path,
            invocation=SessionInvocation(task_id="t-review", surface="plan_review", slot_id="seat",
                operation_id="op-thread", timeout_sec=60, use_thread=True, thread_id="thread-1"))
    assert len(turns) == 1 and "continueFrom" not in turns[0] and not posts
    assert error.value.failure_details["resumable"]["cause"] == "pool_exhausted"
