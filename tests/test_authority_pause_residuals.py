"""Authority preservation never grants execution or displaces a real control rail."""
import copy

import pytest

from ouroboros import budget_pause as bp, sleep_custody as sc, usage_accounting as ua
from tests.test_authority_pause_composition import LiveGateway



def _prepared_transfer(tmp_path, monkeypatch, *, sleep=False):
    from ouroboros import delegate_collection as body, model_sleep
    from tests.test_cold_external_custody import _setup

    env = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(sc, "_runtime", body.body_runtime)
    env.runtime = body.start_body_runtime(tmp_path, gateway_factory=LiveGateway)
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
