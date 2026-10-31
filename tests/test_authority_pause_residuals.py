"""Authority preservation never grants execution or displaces a real control rail."""
import copy
import queue
import time

import pytest

from ouroboros import budget_pause as bp, sleep_custody as sc, usage_accounting as ua
from tests.test_authority_pause_composition import LiveGateway

pytestmark = pytest.mark.serial


def _prepared_transfer(tmp_path, monkeypatch, *, sleep=False):
    from ouroboros import delegate_collection as body, model_sleep
    from tests.test_cold_external_custody import _setup

    env = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(sc, "_runtime", body.body_runtime)
    env.runtime = body.start_body_runtime(tmp_path, gateway_factory=LiveGateway, max_workers=2)
    original = sc.confirm_release

    class Published(BaseException):
        pass

    def capture(ctx, prepared, seed, **_kwargs):
        # The real source and transfer are published; stop this test actor before
        # the independent body's first acceptance pass, without inventing a receipt.
        def published(_ctx):
            assert bp.budget_pause_row(tmp_path, "sleeper")["external_custody"]["state"] == "requested"
            raise Published()
        return original(ctx, prepared, seed, control=published)

    try:
        with monkeypatch.context() as preparation:
            preparation.setattr(sc, "confirm_release", capture)
            with pytest.raises(Published):
                if sleep:
                    model_sleep.request_sleep(env.ctx, model_sleep.selectors(env.ctx, runs=["run-one"]), "cold")
                    bp.enter_cold_sleep(env.limit)
                else:
                    bp.request_pause(env.limit, rail=bp.RAIL_CONTROL_AUTHORITY, scope="task",
                                     reason_text="owner fence unreadable")
        return env
    except BaseException:
        env.runtime.close(wait=True)
        raise
    finally:
        bp.end_dispatch_fence("sleeper")


@pytest.mark.parametrize("fence", ["unknown", "owner_pause"])
@pytest.mark.parametrize("sleep", [False, True])
def test_preservation_accepts_authority_only_under_fence_and_real_pause_still_reaches_run(
        tmp_path, monkeypatch, fence, sleep):
    from ouroboros import delegate_custody_current as current, owner_pause
    from ouroboros.gateways import claudexor
    from ouroboros.task_store import patch_document
    from supervisor.events_budget import install_exact_budget_pause
    from supervisor.owner_pause_control import request_owner_pause, _record_settled_member
    from tests._budget_pause_exact_helpers import _supervisor_ctx

    env = _prepared_transfer(tmp_path, monkeypatch, sleep=sleep)
    controls = []

    class Gateway(LiveGateway):
        def handshake(self):
            pass

        def cancel_run(self, run_id, reason=""):
            controls.append(run_id)
            return {"accepted": True, "status": "accepted"}

    monkeypatch.setattr(claudexor, "ClaudexorGateway", Gateway)
    try:
        if fence == "unknown":
            patch_document(tmp_path, "sleeper", lambda row: {**row, "owner_pause": {"state": "corrupt"}})
            assert owner_pause.member_fence(env.ctx)["state"] == "unknown"
        else:
            assert request_owner_pause("sleeper", request_id="pause-before-custody")["ok"]
            assert owner_pause.fence_closed(owner_pause.member_fence(env.ctx))
        retained = sc.reconcile(env.q)
        row = bp.budget_pause_row(tmp_path, "sleeper")
        assert bool(sc.accepted_receipt("sleeper", row)) is not sleep
        assert retained == (set() if sleep else {"sleeper"})
        assert not current.run(tmp_path, "run-one").settled
        if not sleep:
            install_exact_budget_pause(_supervisor_ctx(tmp_path, env.workers, env.q, [], []), "sleeper",
                                       bp.exact_pause_marker(row)["checkpoint"], evt=bp.pause_event(env.task, row))
            assert not env.q.resume_budget_paused_task("sleeper")["ok"]
            assert "_budget_pause_resume" not in env.workers.PENDING[0]
            if fence == "owner_pause":
                for _ in range(2):
                    _record_settled_member(env.q, env.workers.PENDING[0])
                assert controls == ["run-one"]
            else:
                assert controls == []
    finally:
        env.runtime.close(wait=True)
        assert not env.runtime.thread.is_alive()


@pytest.mark.parametrize("fault", ["source", "binding", "foreign_run", "invocation", "cancel", "stop_requested"])
def test_authority_preservation_requires_source_binding_runs_and_no_stop(tmp_path, monkeypatch, fault):
    from ouroboros import cancel_intents
    env = _prepared_transfer(tmp_path, monkeypatch)
    try:
        row = copy.deepcopy(bp.budget_pause_row(tmp_path, "sleeper"))
        if fault == "source":
            row["source_ref"]["sha256"] = "0" * 64
            row["external_custody"]["binding"]["source_ref"] = copy.deepcopy(row["source_ref"])
        elif fault == "binding":
            row["task_attempt"] += 1
        elif fault in {"foreign_run", "invocation"}:
            row["external_custody"]["runs"][0][0 if fault == "foreign_run" else 1] = "foreign"
        elif fault == "cancel":
            cancel_intents.request_cancel(tmp_path, "sleeper", source="preservation-negative")
        bp.set_budget_pause(tmp_path, "sleeper", row, expected_pause_id=row["pause_id"])
        assert sc.reconcile(env.q, stop_requested=lambda: fault == "stop_requested") == set()
        assert not sc.accepted_receipt("sleeper", bp.budget_pause_row(tmp_path, "sleeper"))
    finally:
        env.runtime.close(wait=True)
        assert not env.runtime.thread.is_alive()


@pytest.mark.parametrize("control", ["panic", "cancelled", "deadline"])
def test_authority_receipt_does_not_release_actor_past_control(tmp_path, monkeypatch, control):
    from ouroboros.model_wait import ModelWaitInterrupted
    env = _prepared_transfer(tmp_path, monkeypatch)
    try:
        assert sc.reconcile(env.q) == {"sleeper"}
        row = bp.budget_pause_row(tmp_path, "sleeper")
        with pytest.raises(ModelWaitInterrupted) as caught:
            sc.confirm_release(env.ctx, row, row, control=lambda _ctx: control)
        assert caught.value.control_reason == control
    finally:
        env.runtime.close(wait=True)


@pytest.mark.parametrize("kind", ["unreadable", "refused", "control"])
def test_forced_authority_error_preserves_typed_identity_and_late_directive_without_retry(tmp_path, monkeypatch, kind):
    from tests.test_delivery_forced_finalization import _forced_test_context
    incoming = queue.Queue()
    loop, registry, ctx, trace = _forced_test_context(tmp_path, incoming=incoming)
    error = (ua.ControlAuthorityUnavailable("fence_read") if kind == "control" else
             ua.BillingAuthorityUnavailable("parent1", kind=kind, cause="original_group_read"))
    calls = []

    def refused(_ctx):
        calls.append(True)
        incoming.put("retain the new owner criterion")
        raise error

    monkeypatch.setattr(loop, "_call_forced_model_once", refused)
    with pytest.raises(type(error)) as caught:
        loop._handle_round_limit(ctx)
    assert caught.value is error and calls == [True]
    assert len(registry._ctx._owner_directives) == 1
    assert "new owner criterion" in str(ctx.messages)
    assert "forced_finalization" not in trace


@pytest.mark.parametrize("rail", ["budget_exhausted", "owner_requested_finalization", "finalization_grace", "deadline_local"])
def test_forced_authority_failure_keeps_preceding_terminal_rail_without_second_send(tmp_path, monkeypatch, rail):
    from tests.test_delivery_forced_finalization import _forced_test_context
    incoming = queue.Queue()
    loop, registry, ctx, _trace = _forced_test_context(tmp_path, incoming=incoming)
    calls = []

    def refused(_ctx):
        calls.append(True)
        incoming.put("keep this late owner directive")
        raise ua.BillingAuthorityUnavailable("parent1", kind="unreadable", cause="original_group_read")

    monkeypatch.setattr(loop, "_call_forced_model_once", refused)
    text, usage, trace = loop._forced_final_answer(ctx, prompt="finish", fallback_text="original terminal reason",
                                                 reason_code=rail)
    assert calls == [True] and usage["reason_code"] == rail
    assert text == "original terminal reason"
    assert trace["forced_finalization"]["source"] == "forced_authority_unavailable"
    assert len(registry._ctx._owner_directives) == 1
    assert bp.budget_pause_row(tmp_path, "parent1") == {}


def test_elapsed_deadline_never_attempts_forced_model_or_technical_pause(tmp_path, monkeypatch):
    from tests.test_delivery_forced_finalization import _forced_test_context
    loop, _registry, ctx, _trace = _forced_test_context(tmp_path)
    ctx.deadline_ts = time.time() - 1
    monkeypatch.setattr(loop, "_call_forced_model_once", lambda _ctx: pytest.fail("send after deadline"))
    _text, usage, _trace = loop._forced_final_answer(ctx, prompt="finish", fallback_text="deadline",
                                                   reason_code="deadline_local")
    assert usage["reason_code"] == "deadline_local" and bp.budget_pause_row(tmp_path, "parent1") == {}


def test_control_authority_error_is_on_public_accounting_export_surface():
    namespace = {}
    exec("from ouroboros.usage_accounting import *", namespace)
    assert namespace["ControlAuthorityUnavailable"] is ua.ControlAuthorityUnavailable


@pytest.mark.parametrize("kind,exact", [("unreadable", True), ("unreadable", False), ("refused", True),
                                       ("control", True), ("control", False)])
def test_real_loop_forced_admission_routes_authority_and_keeps_late_owner_directives(tmp_path, monkeypatch, kind, exact):
    from types import SimpleNamespace
    from ouroboros.task_results import write_task_result
    from tests.test_billing_authority_unavailable import _loop_driver, _task_scope
    from tests._usage_store_testing import ledger_rows

    tid = "forced-authority"
    binding = {"billing_group_id": tid} if kind == "unreadable" else {
        "billing_group_id": "unavailable:old", "billing_group_limit_usd": 0.0}
    if kind == "control":
        from ouroboros import queue_observation
        binding = {"billing_group_id": tid, "billing_group_limit_usd": 20.0}
        (tmp_path / "state").mkdir(exist_ok=True)
        (tmp_path / "state/queue_dispatch.json").write_text("{}")
        monkeypatch.setattr(queue_observation, "dispatch_probe", lambda *_a, **_k: None)
    write_task_result(tmp_path, tid, "running", billing_group=binding)
    loop, registry, ordinary = _loop_driver(tmp_path, monkeypatch, tid, exact_owner=exact)
    registry._ctx.inline_max_rounds = 0  # real round-limit caller before its first paid boundary
    incoming, calls, sends = queue.Queue(), [], []

    def forced(_ctx):
        calls.append(True)
        incoming.put("also retain this criterion")
        return ua.execute_physical_attempt(
            ua.AttemptRequest(model="fixture", provider="fixture", reservation_usd=0.1),
            lambda: sends.append(True))

    monkeypatch.setattr(loop, "_call_forced_model_once", forced)
    with ua.usage_scope(_task_scope(tmp_path, tid, tid)):
        def run():
            return loop.run_llm_loop(messages=[{"role": "user", "content": "preserve my revision"}],
                tools=registry, llm=SimpleNamespace(default_model=lambda: "fixture"), drive_logs=tmp_path / "logs",
                emit_progress=lambda *_a, **_k: None, incoming_messages=incoming, task_id=tid, drive_root=tmp_path)
        if kind in {"unreadable", "control"} and exact:
            with pytest.raises(bp.BudgetPauseRequested) as caught:
                run()
            assert caught.value.pause["reason"] == "authority"
            bp.end_dispatch_fence(tid)
        else:
            text, usage, _trace = run()
            assert usage["reason_code"] == ("control_authority_unreadable" if kind == "control" else "billing_authority_" + kind)
            assert usage["execution_status"] == "infra_failed"
            assert "not a budget limit" in text
    assert calls == [True] and sends == ordinary == []
    assert [row["content"] for row in registry._ctx._owner_directives] == [
        "preserve my revision", "also retain this criterion"]
    assert not [row for row in ledger_rows(tmp_path) if row.get("kind") == "attempt"]


def test_explicit_host_non_task_start_keeps_its_admitted_uncapped_group(tmp_path):
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import task_snapshot, tree_billing_group
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, root_task_id="host-operation", non_task_operation=True)):
        assert custody.record_start_requested(tmp_path, task_id="host-operation", root_task_id="host-operation",
            invocation_id="host-start", request={"prompt": "host operation"}, idempotency_key="host-start")
    assert len(task_snapshot(tmp_path, "host-operation")["pending"]) == 1
    assert tree_billing_group(tmp_path, "host-operation") == {
        "billing_group_id": "host-operation", "billing_group_limit_usd": None}


@pytest.mark.parametrize("rail", ["budget", "owner_stop", "deadline_grace"])
def test_existing_terminal_callers_keep_their_fact_when_wrapup_authority_fails(tmp_path, monkeypatch, rail):
    from tests.test_delivery_forced_finalization import _forced_test_context
    loop, _registry, ctx, _trace = _forced_test_context(tmp_path)
    calls = []

    def refused(_ctx):
        calls.append(True)
        raise ua.ControlAuthorityUnavailable("dispatch fence unreadable")

    monkeypatch.setattr(loop, "_call_forced_model_once", refused)
    if rail == "budget":
        ctx.accumulated_usage["rounds"] = 1  # real work already happened; global remainder really is zero
        text, usage, _trace = loop._check_budget_limits(ctx, 0.0)
        assert usage["reason_code"] == "budget_exhausted" and "Total budget exhausted" in text
    else:
        text, usage, _trace = loop._handle_forced_finalization(
            ctx, "owner_requested_finalization" if rail == "owner_stop" else "deadline")
        assert usage["reason_code"] == ("owner_requested_finalization" if rail == "owner_stop" else "finalization_grace")
    assert calls == [True] and bp.budget_pause_row(tmp_path, "parent1") == {}
