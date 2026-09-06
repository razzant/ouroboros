"""Cancellation and settlement claim only what they verified.

Split verbatim out of ``tests/test_delegated_subagent_transport.py`` by theme. This
module owns the cancel receipt vocabulary, the loud durable incident an unverifiable
cancel leaves behind, and the atomicity of the settlement that follows it.
"""

from __future__ import annotations

import json
import pytest

from tests._delegated_transport_shared import (  # noqa: F401  (autouse fixture applies on import)
    _LiveRunStub,
    _health_invariants,
    _nanny_ctx,
    _owned_gateway_uses_each_test_transport,
)


@pytest.mark.parametrize("accepted,state,expected,may_be_live", [
    (True, "cancelled", "confirmed", False),
    (True, "running", "requested", True),
    (False, "running", "failed", True),
])
def test_cancel_never_claims_more_than_a_terminal_receipt_proves(
    tmp_path, monkeypatch, accepted, state, expected, may_be_live,
):
    """`status: cancelled` used to be returned for all of these — including a daemon
    that REFUSED the control while the run kept mutating."""
    import ouroboros.delegate_custody as dc
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    class _Stub(_LiveRunStub):
        def cancel_run(self, rid, reason=""):
            return {"accepted": accepted, "status": "accepted" if accepted else "rejected"}
        def get_run(self, rid, **_kw):
            return {"lastSeq": 3, "summary": {"state": state, "spendUsd": 0.0}}

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    delegate._CUSTODY.clear()
    delegate._CUSTODY["run-1"] = delegate._RunCustody(
        run_id="run-1", task_id="t-a", route_id="r", model="m", project_id="p", project_owned=False)
    out = json.loads(delegate._delegate_cancel(_nanny_ctx(tmp_path), "run-1", reason="stuck"))
    delegate._CUSTODY.clear()
    assert out["status"] == expected, out
    assert out["run_may_still_be_live"] is may_be_live, out
    faults = dc.open_containment_faults(tmp_path)
    assert bool(faults) is (expected == "failed"), (expected, faults)


def test_cancel_and_verify_carries_the_verify_reads_terminal_detail(tmp_path):
    """BR2-1, purely additive: when the verify read discovers a terminal state,
    the already-read run detail rides the result as the OPTIONAL `terminal_detail`
    key, so a caller consuming a discovered natural terminal (completion wins)
    never depends on a second fetch after settlement. The key is ABSENT on every
    other outcome — the historical six-key shape is untouched — and it never
    rides the emitted cancel-outcome event."""
    import ouroboros.delegate_custody as dc

    detail = {"lastSeq": 9, "summary": {"state": "succeeded", "spendUsd": 0.0,
                                        "inputTokens": 1, "outputTokens": 1}}

    class _Finished:
        def cancel_run(self, rid, reason=""):
            return {"accepted": True, "status": "accepted"}
        def get_run(self, rid, **_kw):
            return detail

    entry = dc.RunCustody(run_id="run-td", task_id="t-a", route_id="r", model="m",
                          project_id="p", project_owned=False, root_task_id="t-a",
                          ledger_root=str(tmp_path))
    dc.record_started(tmp_path, entry)
    out = dc.cancel_and_verify(tmp_path, _Finished(), entry, "test")
    assert out["outcome"] == "confirmed" and out["state"] == "succeeded"
    assert out["terminal_detail"] == detail

    class _Live:
        def cancel_run(self, rid, reason=""):
            return {"accepted": True, "status": "accepted"}
        def get_run(self, rid, **_kw):
            return {"lastSeq": 3, "summary": {"state": "running"}}

    entry2 = dc.RunCustody(run_id="run-td2", task_id="t-a", route_id="r", model="m",
                           project_id="p", project_owned=False, root_task_id="t-a",
                           ledger_root=str(tmp_path))
    dc.record_started(tmp_path, entry2)
    out2 = dc.cancel_and_verify(tmp_path, _Live(), entry2, "test")
    assert out2["outcome"] == "requested"
    assert set(out2) == {"outcome", "accepted", "control_status", "state",
                         "fault_reason", "detail"}, out2

    rows = [json.loads(line) for line in
            (tmp_path / "logs" / "events.jsonl").read_text().splitlines()]
    outcomes = [r for r in rows if r.get("type") == "delegate_run_cancel_outcome"]
    assert outcomes and all("terminal_detail" not in r for r in outcomes)


def test_an_unverifiable_cancel_is_a_loud_durable_incident(tmp_path, monkeypatch):
    """A cancel that never reached the daemon left a typed refusal and nothing else: an
    overpowered mutating run stayed live with no durable trace and no owner-visible
    signal. It is now a containment fault that rides the health invariants until a
    terminal receipt clears it."""
    import ouroboros.delegate_custody as dc
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    class _Deaf(_LiveRunStub):
        def cancel_run(self, rid, reason=""):
            raise gw.ClaudexorUnavailable("daemon_unreachable", "connection refused")

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Deaf())
    delegate._CUSTODY.clear()
    delegate._CUSTODY["run-1"] = delegate._RunCustody(
        run_id="run-1", task_id="t-a", route_id="r", model="m", project_id="p", project_owned=False)
    out = json.loads(delegate._delegate_cancel(_nanny_ctx(tmp_path), "run-1"))
    assert out["status"] == "containment_fault_run_may_still_be_live", out
    assert out["run_may_still_be_live"] is True
    faults = dc.open_containment_faults(tmp_path)
    assert [f["run_id"] for f in faults] == ["run-1"], faults

    invariants = _health_invariants(tmp_path)
    assert "DELEGATED RUN MAY STILL BE LIVE" in invariants, invariants
    assert "run-1" in invariants

    # A later VERIFIED terminal receipt clears the incident — the fault is a live
    # condition, not a permanent scar.
    class _Stopped(_LiveRunStub):
        def cancel_run(self, rid, reason=""): return {"accepted": True, "status": "accepted"}
        def get_run(self, rid, **_kw):
            return {"lastSeq": 4, "summary": {"state": "cancelled", "spendUsd": 0.0}}

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stopped())
    again = json.loads(delegate._delegate_cancel(_nanny_ctx(tmp_path), "run-1"))
    delegate._CUSTODY.clear()
    assert again["status"] == "confirmed", again
    assert dc.open_containment_faults(tmp_path) == []
    assert "DELEGATED RUN MAY STILL BE LIVE" not in _health_invariants(tmp_path)


def test_cancelling_a_run_this_module_already_settled_is_not_an_incident(tmp_path, monkeypatch):
    """`settle_run` short-circuits on `custody.settled`; its twin `cancel_and_verify` never
    consulted it, and its `cancel_run` failure branch declared a containment fault WITHOUT
    reading the run — with the read three lines below, unused. So an ordinary cancel of an
    already-settled run (the daemon answers 409 `run_already_terminal`) manufactured a
    permanent CRITICAL against a run this very module had recorded as closed."""
    import ouroboros.delegate_custody as dc
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    class _Finished(_LiveRunStub):
        def get_run(self, rid, **_kw):
            return {"lastSeq": 9, "summary": {"state": "succeeded", "spendUsd": 0.0,
                                              "inputTokens": 1, "outputTokens": 1}}
        def cancel_run(self, rid, reason=""):
            raise gw.ClaudexorUnavailable("run_already_terminal", "conflict", status_code=409)

    class _Deaf(_LiveRunStub):
        def cancel_run(self, rid, reason=""):
            raise gw.ClaudexorUnavailable("daemon_unreachable", "connection refused")
        def get_run(self, rid, **_kw):
            raise gw.ClaudexorUnavailable("daemon_unreachable", "connection refused")

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Finished())
    delegate._CUSTODY.clear()
    entry = delegate._RunCustody(run_id="run-1", task_id="t-a", route_id="r", model="m",
                                 project_id="p", project_owned=False, root_task_id="t-a",
                                 ledger_root=str(tmp_path))
    dc.record_started(tmp_path, entry)
    ctx = _nanny_ctx(tmp_path)
    assert json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=1))["settlement"]["settled"] is True

    # The daemon then goes away entirely — the common shape, since a finished run is often
    # the last thing it did. Nothing can be read back, so only the durable settlement this
    # module already wrote can answer, and it does.
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Deaf())
    after_settlement = json.loads(delegate._delegate_cancel(ctx, "run-1", reason="ordinary"))
    assert after_settlement["status"] == "confirmed", after_settlement
    assert after_settlement["run_may_still_be_live"] is False
    assert dc.open_containment_faults(tmp_path) == []
    assert "DELEGATED RUN MAY STILL BE LIVE" not in _health_invariants(tmp_path)

    # The other half of the same defect, on a run with NO settlement to short-circuit on:
    # the refused control is not a verdict about the RUN, so the state read decides, and a
    # run that has already stopped is confirmed rather than faulted.
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Finished())
    dc.record_started(tmp_path, delegate._RunCustody(
        run_id="run-2", task_id="t-a", route_id="r", model="m", project_id="p",
        project_owned=False, root_task_id="t-a", ledger_root=str(tmp_path)))
    unsettled = json.loads(delegate._delegate_cancel(ctx, "run-2", reason="stuck"))
    delegate._CUSTODY.clear()
    assert unsettled["status"] == "confirmed", unsettled
    assert dc.open_containment_faults(tmp_path) == []
    assert dc.replay(tmp_path)["run-2"].settled is True, "the read that confirmed it also settles it"


def test_a_retirement_that_landed_is_not_replayed_as_still_owned(tmp_path, monkeypatch):
    """Settlement's two obligations can fail independently. When the RETIREMENT landed
    and the ledger write did not, the durable replay must know the registration is gone
    — otherwise a restart retries `remove_project` on an already-removed project and the
    settlement can never complete."""
    import ouroboros.delegate_custody as dc
    import ouroboros.tools.delegate as delegate
    import ouroboros.usage_accounting as ua
    from ouroboros.gateways import claudexor as gw

    removed = []

    class _Stub(_LiveRunStub):
        def get_run(self, rid, **_kw):
            return {"lastSeq": 9, "summary": {"state": "succeeded", "spendUsd": 0.0}}
        def remove_project(self, pid): removed.append(pid)

    def _boom(*a, **k):
        raise ua.UsageAccountingError("usage accounting lock unavailable")

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    monkeypatch.setattr(ua, "record_subscription_session", _boom)
    delegate._CUSTODY.clear()
    dc.record_started(tmp_path, delegate._RunCustody(
        run_id="run-1", task_id="t-a", route_id="r", model="m",
        project_id="prj-ours", project_owned=True, root_task_id="t-a", ledger_root=str(tmp_path)))
    json.loads(delegate._delegate_wait(_nanny_ctx(tmp_path), "run-1", wait_sec=1))
    delegate._CUSTODY.clear()          # the worker restarts

    replayed = dc.replay(tmp_path)["run-1"]
    assert removed == ["prj-ours"]
    assert replayed.project_owned is False, "a retirement that landed must replay as landed"
    assert replayed.ledger_recorded is False and replayed.settled is False
