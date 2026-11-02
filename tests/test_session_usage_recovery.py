"""#2136: terminal usage survives fallible collection, without changing the failure."""

import copy
import json

import pytest

from ouroboros import delegate_custody as custody, delegate_custody_usage as dcu
from ouroboros import review_execution as rx
from ouroboros.review_session_usage import session_usage_facts
from tests._review_session_route_shared import (
    FakeGateway, _agent_request, _agent_slot, _terminal_detail,
    _owned_gateway_uses_each_test_transport, fake_route as fake_route,  # noqa: F401
)
from tests.test_session_input_events import (
    SPLIT, collect, entry, events, persist_review, _synthetic_root_lineage,  # noqa: F401
)
from tests.test_review_session_continuation import _engine
from tests._usage_store_testing import ledger_rows

pytestmark = pytest.mark.serial


@pytest.fixture(autouse=True)
def _isolated_observation(monkeypatch):
    monkeypatch.setattr(dcu, "_EMITTED_SESSION_USAGE", {})


def _executor(root):
    result = rx.AgentSessionReviewExecutor(rx.ReviewAssignment(
        _agent_request(session_root=str(root), task_id="t-review"), _agent_slot(),
        custody_root=root, call_id="usage-recovery"))
    result.usage_observer = [].append
    return result


def _with_split(monkeypatch):
    original = FakeGateway.get_run
    def get_run(self, run_id, **kwargs):
        detail = original(self, run_id, **kwargs)
        detail["summary"].update(runId=run_id, inputTokenUsage=copy.deepcopy(SPLIT),
                                 inputTokens=4, cachedInputTokens=20484, outputTokens=142)
        final = self.run_dir / "final"
        final.mkdir(parents=True, exist_ok=True)
        (final / "telemetry.yaml").write_text(json.dumps({"run_id": run_id, "final_attempt_id": "a01",
            "attempts": [{"attempt_id": "a01", "harness_id": "fake-review", "observed_model": "fake-small"}]}))
        detail["summary"]["runDir"] = str(self.run_dir)
        return detail
    monkeypatch.setattr(FakeGateway, "get_run", get_run)


def test_lost_continuation_keeps_read_facts_and_same_executor_observes_successor(tmp_path, monkeypatch, fake_route):
    accepted, posts = _engine(monkeypatch, lose_response=True)
    _with_split(monkeypatch)
    executor = _executor(tmp_path)
    observed = []
    executor.usage_observer = observed.append
    from ouroboros.usage_accounting import UsageScope, usage_scope
    with usage_scope(UsageScope(task_id="t-review", root_task_id="root", category="review",
                                source="review_substrate", review_slot_id="scope_slot_1")):
        with pytest.raises(OSError, match="response lost") as failure:
            executor.execute()
        failed_usage = failure.value.delegated_usage
        assert failed_usage["input_token_usage"] == SPLIT
        assert failed_usage["observed_attempt"]["model"] == "fake-small"
        assert failed_usage["delegated_run_id"] == "run-1"
        assert len(observed) == 1 and observed[0]["cost"] is None
        # Replay the original pending token, then collect the already accepted successor.
        assert executor.execute().raw_text == "[]"
    assert [row["claudexor"]["run_id"] for row in observed] == ["run-1", "run-2"]
    assert all(row["input_token_usage"] == SPLIT for row in observed)
    assert len(accepted) == 1 and len(posts) == 2 and posts[0][1] == posts[1][1]
    for row in observed:
        persist_review(tmp_path, row)
    assert [row["input_token_usage"] for row in events(tmp_path) if row["type"] == "llm_usage"] == [SPLIT, SPLIT]
    assert len([row for row in ledger_rows(tmp_path) if row["kind"] == "subscription_session"]) == 2


@pytest.mark.parametrize("waiting", [False, True])
def test_verified_cancel_usage_reaches_failure_projection(tmp_path, monkeypatch, fake_route, waiting):
    from ouroboros.delegate_collection import record_terminal_money
    fake_route.detail = _terminal_detail("", state="cancelled")
    _with_split(monkeypatch)
    running = {"summary": {"state": "running", "waitingOnUser": waiting}, "lastSeq": 1}
    monkeypatch.setattr(rx, "_observe_session", lambda *_: (running, None))
    cancellations = []
    def cancel(root, gateway, held, reason):
        terminal = gateway.get_run(held.run_id)
        assert record_terminal_money(root, held, terminal)["settled"]
        cancellations.append(reason)
        return {"outcome": "confirmed", "state": "cancelled", "terminal_detail": terminal}
    monkeypatch.setattr(custody, "cancel_and_verify", cancel)
    error_type = rx.ReviewSessionWaitingOnUser if waiting else TimeoutError
    with pytest.raises(error_type) as failure:
        rx.run_delegated_review_session(prompt="review", root=str(tmp_path), custody_drive=tmp_path,
            invocation=rx.SessionInvocation(task_id="t-review", surface="scope_review", slot_id="s",
                                            timeout_sec=0))
    usage = failure.value.delegated_usage
    assert usage["input_token_usage"] == SPLIT
    assert usage["observed_attempt"]["attempt_id"] == "a01"
    assert "host-cancelled with a verified terminal receipt" in str(failure.value)
    assert cancellations == ["review_session_waiting_on_user" if waiting else "review_slot_timeout"]
    assert ledger_rows(tmp_path)[0]["input_token_usage"] == SPLIT
    assert persist_review(tmp_path, usage)["input_token_usage"] == SPLIT


@pytest.mark.parametrize("carried", ["same", "job_alias", "foreign", "live", "unnamed", "none"])
def test_successor_failure_never_borrows_prior_run_or_unbound_exception_detail(tmp_path, monkeypatch, fake_route, carried):
    _engine(monkeypatch)
    _with_split(monkeypatch)
    poll = rx._poll_session_terminal
    error = OSError("successor observation failed")
    detail = {"summary": {"state": "running" if carried == "live" else "cancelled",
                           "inputTokenUsage": {**SPLIT, "total_tokens": None}}}
    if carried == "job_alias":
        detail["summary"].update(runId="canonical-run-2", jobId="run-2")
    elif carried != "unnamed":
        detail["summary"]["runId"] = "other-run" if carried == "foreign" else "run-2"
    if carried != "none":
        error.checking_terminal_detail = detail
    def fail_successor(gateway, module, root, held, run_id, seconds):
        if run_id == "run-2":
            raise error
        return poll(gateway, module, root, held, run_id, seconds)
    monkeypatch.setattr(rx, "_poll_session_terminal", fail_successor)
    executor = _executor(tmp_path)
    observed = []
    executor.usage_observer = observed.append
    with pytest.raises(OSError) as failure:
        executor.execute()
    assert failure.value is error
    assert [row["delegated_run_id"] for row in observed] == ["run-1", "run-2"]
    assert observed[0]["input_token_usage"] == SPLIT
    last = observed[-1]
    if carried in {"same", "job_alias"}:
        assert last["input_token_usage"] == detail["summary"]["inputTokenUsage"]
    else:
        assert "input_token_usage" not in last and last["observed_attempt"] == {}


@pytest.mark.parametrize("state,split", [("succeeded", SPLIT), ("failed", SPLIT),
                                         ("succeeded", dict.fromkeys(SPLIT)), ("failed", None)])
def test_attach_only_usage_uses_recorded_attribution_without_new_effects(tmp_path, monkeypatch, state, split):
    from ouroboros import review_operation
    from ouroboros.usage_accounting import UsageScope, usage_scope
    from tests.test_review_operation_collection import (
        TASK, SURFACE, PASS, _session_run, _collect, _ctx, _observe,
    )
    custody.emit(tmp_path, custody.START_REQUESTED, {
        "invocation_id": "inv-op-session", "task_id": TASK, "operation_id": "op-session", "slot_id": "s",
        "surface": SURFACE, "root_task_id": TASK, "parent_task_id": "recorded-parent",
        "review_skill": "original-skill", "review_wave_id": "original-wave", "review_slot_id": "s",
        "request": {"prompt": "review", "harnesses": ["codex"]}})
    custody.emit(tmp_path, custody.STARTED, {"run_id": "run-cold", "invocation_id": "inv-op-session",
                                           "task_id": TASK, "surface": SURFACE})
    detail = {"summary": {"state": state, "runId": "run-cold", "inputTokenUsage": split},
              "primaryOutput": {"text": PASS, "truncated": False}}
    calls = _observe(monkeypatch, detail)
    def forbidden(*_a, **_kw):
        pytest.fail("attach-only collection reached accounting or generation")
    monkeypatch.setattr("ouroboros.review_substrate.ReviewCoordinator.run", forbidden)
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", forbidden)
    monkeypatch.setattr("ouroboros.usage_accounting.record_subscription_session", forbidden)
    run, ctx = _session_run(), _ctx(tmp_path)
    # A caller without a delivery endpoint must not consume the process-local emission opportunity.
    assert _collect(run, tmp_path, None)[1] == review_operation.COLLECTED
    with usage_scope(UsageScope(task_id="collector", root_task_id="wrong-root", parent_task_id="wrong-parent",
                                review_skill="wrong-skill", review_wave_id="wrong-wave", review_slot_id="wrong-slot")):
        actor, fact = _collect(run, tmp_path, ctx)
        assert _collect(run, tmp_path, ctx)[1] == fact
    assert fact == review_operation.COLLECTED
    assert actor["status"] == ("ok" if state == "succeeded" else "error")
    rows = list(ctx.event_queue.queue)
    assert len(rows) == 1
    row = rows[0]
    assert row["task_id"] == TASK and row["root_task_id"] == TASK
    assert row["parent_task_id"] == "recorded-parent" and row["delegation_role"] == "subagent"
    assert row["review_wave_id"] == "original-wave" and row["review_skill"] == "original-skill"
    assert row["review_slot_id"] == "s" and row["source"] == "review_substrate:scope_review"
    assert row["usage"]["claudexor"]["run_id"] == "run-cold"
    if split is not None:
        assert row["usage"]["input_token_usage"] == split
    else:
        assert "input_token_usage" not in row["usage"]
    # The first collection retains the complete observation; later collections reuse that
    # receipt (collection by reference) and never read the engine again.
    assert calls == [("handshake",), ("get_run", "run-cold"), ("close",)]


@pytest.mark.parametrize("parent,root", [("parent", "root"), ("", "")])
def test_ordinary_usage_has_every_passport_field(tmp_path, parent, root):
    from ouroboros.contracts.record_contract import ANCHOR_BY_TYPE
    run = entry(tmp_path)
    run.parent_task_id, run.root_task_id = parent, root
    collect(tmp_path, run)
    row = next(row for row in events(tmp_path) if row["type"] == "llm_usage")
    assert ANCHOR_BY_TYPE["llm_usage"].guaranteed <= row.keys()
    assert row["root_task_id"] == (root or run.task_id)
    assert row["delegation_role"] == ("subagent" if parent else "")


@pytest.mark.parametrize("fails_again", [False, True])
def test_executor_failure_does_not_latch_out_a_new_run(tmp_path, monkeypatch, fails_again):
    executor, rows = _executor(tmp_path), []
    executor.usage_observer = rows.append
    first = OSError("lost reply")
    first.delegated_run_started, first.delegated_run_id = True, "first"
    first.delegated_usage = session_usage_facts("first", summary={"inputTokenUsage": SPLIT})
    second = OSError("next run collection failed")
    second.delegated_run_started, second.delegated_run_id = True, "next"
    second.delegated_usage = session_usage_facts("next", summary={"inputTokenUsage": SPLIT})
    calls = []
    def collect_next():
        calls.append(True)
        if len(calls) == 1:
            executor._retry_state["pending_invocation_id"] = "retained-token"
            raise first
        executor._retry_state.clear()
        if fails_again:
            raise second
        executor._run_id = "next"
        executor._session_usage = session_usage_facts("next", summary={"inputTokenUsage": SPLIT})
    monkeypatch.setattr(executor, "_run_session", collect_next)
    monkeypatch.setattr(executor, "_verdict_result", lambda **_: "verdict")
    with pytest.raises(OSError) as failure:
        executor.execute()
    assert failure.value is first
    if fails_again:
        for _ in range(2):
            with pytest.raises(OSError) as later:
                executor.execute()
            assert later.value is second
        assert executor.failure_custody()["delegated_run_id"] == "next"
    else:
        assert executor.execute() == executor.execute() == "verdict"
    assert [row["delegated_run_id"] for row in rows] == ["first", "next"]


def test_attach_only_emits_terminal_facts_even_when_output_is_temporarily_unreadable(tmp_path, monkeypatch):
    from ouroboros import review_operation
    from tests.test_review_operation_collection import (
        PASS, _started, _session_run, _collect, _ctx, _observe,
    )
    _started(tmp_path, run_id="run-output")
    detail = {"summary": {"state": "succeeded", "runId": "run-output", "inputTokenUsage": SPLIT},
              "primaryOutput": {"text": PASS, "truncated": False}}
    _observe(monkeypatch, detail)
    original = rx._full_session_text
    def unreadable(*_a, **_kw):
        raise OSError("output temporarily unreadable")
    monkeypatch.setattr(rx, "_full_session_text", unreadable)
    ctx, run = _ctx(tmp_path), _session_run()
    assert _collect(run, tmp_path, ctx)[1] == review_operation.DEFERRED
    assert list(ctx.event_queue.queue)[0]["usage"]["input_token_usage"] == SPLIT
    monkeypatch.setattr(rx, "_full_session_text", original)
    assert _collect(run, tmp_path, ctx)[1] == review_operation.COLLECTED
    assert ctx.event_queue.qsize() == 1
