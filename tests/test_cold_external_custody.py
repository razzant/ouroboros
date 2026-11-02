"""Cold transfer through real pause, queue, wake and continuation consumers."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _fast_hold, _install_queue, _loop_ctx, _supervisor_ctx

pytestmark = pytest.mark.serial  # the installed queue fixture binds module state


class Runtime:
    def __init__(self, identity="first"):
        self.active, self.calls, self.name = True, [], identity

    def ensure_runs(self, pairs, *, task_id=""):
        self.calls.append((copy.deepcopy(pairs), task_id))
        return self.active and pairs == [["run-one", "invoke-one"]] and task_id == "sleeper"

    def identity(self):
        return {"runtime_id": self.name, "active": self.active}


def _setup(tmp_path, monkeypatch, *, direct=False, access="workspace_write"):
    from ouroboros import budget_pause as bp, delegate_custody as custody, sleep_custody as sc
    from ouroboros.task_results import write_task_result

    q, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    monkeypatch.setattr(q, "persist_queue_snapshot", lambda **_k: True)
    ctx, limit = _loop_ctx(tmp_path, "sleeper", direct=direct)
    ctx.project_id = "project"
    ctx.task_metadata = {"root_task_id": "sleeper", "project_id": "project"}
    limit.messages = [{"role": "user", "content": "original intent"}]
    task = {"id": "sleeper", "type": "task", "chat_id": 0, "root_task_id": "sleeper", "_attempt": 1,
            "project_id": "project", "budget_drive_root": str(tmp_path), "_is_direct_chat": direct}
    write_task_result(tmp_path, "sleeper", "running", chat_id=0, project_id="project", root_task_id="sleeper")
    assert custody.emit(tmp_path, custody.STARTED, {"run_id": "run-one", "invocation_id": "invoke-one",
        "task_id": "sleeper", "root_task_id": "sleeper", "route": "route", "source": "delegate_start",
        "access": access})
    if not direct:
        workers.RUNNING["sleeper"] = {"task": task, "worker_id": 0, "attempt": 1}
        workers.WORKERS[0] = SimpleNamespace(busy_task_id="sleeper")
    runtime = Runtime()
    monkeypatch.setattr(sc, "_runtime", lambda _root: runtime)
    _fast_hold(monkeypatch, bp)
    return SimpleNamespace(q=q, state=state, workers=workers, ctx=ctx, limit=limit, task=task, runtime=runtime)


def _park(env, tmp_path, monkeypatch, *, direct=False, selected=None):
    from ouroboros import budget_pause as bp, model_sleep, sleep_custody as sc
    from supervisor.events_budget import install_exact_budget_pause

    chosen = model_sleep.selectors(env.ctx, **({"runs": ["run-one"]} if selected is None else selected))
    model_sleep.request_sleep(env.ctx, chosen, "cold")
    observed_before_release = []

    def host_tick(_ctx):
        sc.reconcile(env.q, direct=[env.task] if direct else (), runtime=env.runtime)
        row = bp.budget_pause_row(tmp_path, "sleeper")
        if row.get("external_custody", {}).get("state") == "accepted":
            observed_before_release.append(row)
        return ""

    monkeypatch.setattr(bp, "_hold_control_reason", host_tick)
    with pytest.raises(bp.BudgetPauseRequested) as raised:
        bp.enter_cold_sleep(env.limit)
    row = raised.value.pause
    assert observed_before_release and sc.accepted_receipt("sleeper", row)
    assert "settlement" not in row
    sup = _supervisor_ctx(tmp_path, env.workers, env.q, [], [])
    install_exact_budget_pause(sup, "sleeper", bp.exact_pause_marker(row)["checkpoint"],
                               evt=bp.pause_event(env.task, row))
    bp.end_dispatch_fence("sleeper")
    return bp.budget_pause_row(tmp_path, "sleeper")


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("terminal", [False, True])
def test_exact_cold_transfer_park_wake_and_same_author_restore(tmp_path, monkeypatch, direct, terminal):
    from ouroboros import budget_pause as bp, delegate_custody_current as current, owner_wait
    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.project_lease import cold_project_holders, candidate_is_leasable
    from supervisor.sleep_wake import wake_ready_sleepers

    env = _setup(tmp_path, monkeypatch, direct=direct)
    pause = _park(env, tmp_path, monkeypatch, direct=direct)
    assert pause["state"] == "paused" and env.workers.RUNNING == {}
    task = env.workers.PENDING[0]
    holders = cold_project_holders(env.workers.PENDING)
    assert holders == {"project": {"sleeper"}}
    assert not candidate_is_leasable({"id": "foreign", "project_id": "project"}, set(), cold_holders=holders)
    assert candidate_is_leasable(task, set(), cold_holders=holders)
    run = current.run(tmp_path, "run-one")
    assert not run.settled
    if terminal:
        # Exercise the additive current field before collection code is integrated.
        original = current._decode
        monkeypatch.setattr(current, "_decode", lambda raw: _terminal(original(raw)))
    else:
        assert write_owner_message(tmp_path, "come back", "sleeper", msg_id="owner-wake")
    out = wake_ready_sleepers(env.q)
    assert out[0]["ok"], out
    assert out[0]["ready"]["reason"] == ("run:run-one" if terminal else "owner_text")
    handoff = task["_budget_pause_resume"]
    fresh_ctx, _ = _loop_ctx(tmp_path, "sleeper", direct=direct)
    fresh_ctx.budget_pause_resume = handoff
    saved = bp.load_budget_pause(fresh_ctx)
    assert saved["task_id"] == "sleeper" and saved["task_attempt"] == 1
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a: (None, "max"))
    messages, usage = [], {}
    bp.resume_paused_loop(SimpleNamespace(_ctx=fresh_ctx), saved, messages, {}, usage, set(),
                          budget_remaining_usd=5.0)
    assert messages[0]["content"] == "original intent"
    assert fresh_ctx.task_id == run.task_id == "sleeper"
    assert fresh_ctx.active_model == "m" and fresh_ctx.active_effort == "high"
    assert bp.budget_pause_row(tmp_path, "sleeper")["grant"]["consumed_at"]
    assert "body retained your exact external runs" in messages[-1]["content"]
    assert not current.run(tmp_path, "run-one").settled, "return never invents collection or stop"


def _terminal(run):
    run.terminal_state = "succeeded"
    return run


def test_lost_hint_and_runtime_replacement_rebind_the_existing_receipt(tmp_path, monkeypatch):
    from ouroboros import budget_pause as bp, sleep_custody as sc
    env = _setup(tmp_path, monkeypatch, direct=True)
    row = _park(env, tmp_path, monkeypatch, direct=True)
    old = copy.deepcopy(row["external_custody"])
    replacement = Runtime("successor")
    assert sc.reconcile(env.q, runtime=replacement) == {"sleeper"}
    assert replacement.calls == [([["run-one", "invoke-one"]], "sleeper")]
    assert bp.budget_pause_row(tmp_path, "sleeper")["external_custody"] == old
    assert sc.carrier_receipt(env.workers.PENDING[0]), "runtime id is provenance, not validity"
    replacement.active = False
    assert sc.reconcile(env.q, runtime=replacement) == {"sleeper"}, "outage must not invent an orphan"


def test_question_wakes_an_active_transferred_run(tmp_path, monkeypatch):
    from ouroboros import budget_pause as bp, delegate_custody_current as current, owner_wait, sleep_custody as sc
    from ouroboros.model_sleep import wake_reason
    from supervisor.sleep_wake import wake_ready_sleepers
    env = _setup(tmp_path, monkeypatch)
    _park(env, tmp_path, monkeypatch)
    real = current._decode

    def question(raw):
        run = real(raw)
        run.pending_interaction_ids = ["question-1"]
        run.interaction_source = {"artifact": "question-source", "sha256": "source-hash"}
        return run

    monkeypatch.setattr(current, "_decode", question)
    result = wake_ready_sleepers(env.q)
    assert result[0]["ok"] and result[0]["ready"]["reason"] == "run_question:run-one"
    assert current.run(tmp_path, "run-one").terminal_state == ""
    ctx, _limit = _loop_ctx(tmp_path, "sleeper")
    ctx.budget_pause_resume = env.workers.PENDING[0]["_budget_pause_resume"]
    saved = bp.load_budget_pause(ctx)
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a: (None, "max"))
    messages, usage = [], {}
    bp.resume_paused_loop(SimpleNamespace(_ctx=ctx), saved, messages, {}, usage, set(), budget_remaining_usd=5.0)
    assert ctx._sleep_interaction_seen == {"run-one": ["question-1"]}
    retained = owner_wait.continuation_state(ctx, messages, {}, usage, 4, [], set())
    restored, _ = _loop_ctx(tmp_path, "sleeper")
    owner_wait.restore_continuation_state(SimpleNamespace(_ctx=restored), retained, [], {}, {}, set())
    chosen = {"body_runs": [["run-one", "invoke-one"]], "any_mail": False,
              "interaction_seen": sc.seen_interactions(restored, [["run-one", "invoke-one"]])}
    assert wake_reason(restored, chosen) == "", "the delivered question notice does not buy another wake"
    restored._sleep_interaction_seen = {"other-run": ["question-1"]}
    chosen["interaction_seen"] = {}
    assert wake_reason(restored, chosen) == "run_question:run-one", "ids are scoped by run"


def test_confirmation_expiry_withdraws_exact_unaccepted_seed(tmp_path, monkeypatch):
    from ouroboros import budget_pause as bp, model_sleep, routing_wait, sleep_custody as sc
    env = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(routing_wait, "_confirm_wait_sec", lambda _bound: 0)
    model_sleep.request_sleep(env.ctx, model_sleep.selectors(env.ctx, runs=["run-one"]), "cold")
    bp.enter_cold_sleep(env.limit)
    row = bp.budget_pause_row(tmp_path, "sleeper")
    assert row["state"] == "abandoned" and not bp.dispatch_fenced("sleeper")
    assert env.ctx._model_sleep is None
    assert sc.reconcile(env.q, runtime=env.runtime) == set()
    assert bp.budget_pause_row(tmp_path, "sleeper")["state"] == "abandoned"
    assert "not confirmed" in env.limit.messages[-1]["content"]


def test_acceptance_racing_withdrawal_wins_without_returning_awake(tmp_path, monkeypatch):
    from ouroboros import budget_pause as bp, model_sleep, routing_wait, sleep_custody as sc
    env = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(routing_wait, "_confirm_wait_sec", lambda _bound: 0)
    real = sc._replace
    raced = []

    def replace(root, tid, expected, replacement):
        if replacement.get("state") == "abandoned" and not raced:
            raced.append(True)
            sc.reconcile(env.q, runtime=env.runtime)
        return real(root, tid, expected, replacement)

    monkeypatch.setattr(sc, "_replace", replace)
    model_sleep.request_sleep(env.ctx, model_sleep.selectors(env.ctx, runs=["run-one"]), "cold")
    with pytest.raises(bp.BudgetPauseRequested) as result:
        bp.enter_cold_sleep(env.limit)
    assert raced and sc.accepted_receipt("sleeper", result.value.pause)
    bp.end_dispatch_fence("sleeper")


def test_owner_pause_routes_real_stop_for_a_cold_receipt_without_sleep_settlement_flag(tmp_path, monkeypatch):
    from ouroboros import budget_pause as bp, owner_pause, delegate_custody as custody
    from ouroboros.gateways import claudexor
    from supervisor.owner_pause_control import _record_settled_member
    env = _setup(tmp_path, monkeypatch)
    _park(env, tmp_path, monkeypatch)
    owner_pause.install_fence(tmp_path, "sleeper", request_id="owner-pause")
    calls, reads = [], []

    class Gateway:
        def handshake(self):
            pass

        def get_run(self, rid):
            reads.append(rid)
            return {"summary": {"state": "running"}}

        def close(self):
            pass

    def cancel(root, gateway, run, reason):
        calls.append((run.run_id, reason))
        return {"outcome": custody.CANCEL_REQUESTED}

    monkeypatch.setattr(claudexor, "ClaudexorGateway", Gateway)
    monkeypatch.setattr(custody, "cancel_and_verify", cancel)
    _record_settled_member(env.q, env.workers.PENDING[0])
    _record_settled_member(env.q, env.workers.PENDING[0])
    assert calls == [("run-one", "owner_pause_settlement_check")]
    assert reads == ["run-one"], "repeated Pause observation must not resend the control"
    row = bp.budget_pause_row(tmp_path, "sleeper")
    assert row["external_runs"]["runs"][0]["state"] == "stop_requested"
    assert "settlement" not in row

    monkeypatch.setattr(owner_pause, "member_fence", lambda _ctx: {"state": "unknown"})
    _record_settled_member(env.q, env.workers.PENDING[0])
    assert len(reads) == 1, "unknown Pause authority cannot authorize another control"


def test_an_owner_pause_over_a_cold_sleep_is_named_when_the_sleep_wakes(tmp_path, monkeypatch):
    """R1 #1562: the owner's Pause stopped a cold sleeper's own run. The sleep keeps its reason and
    its own notices, and its wake also names what became of that run: the Pause's participants,
    associated by the record the Pause wrote, not by the checkpoint's reason."""
    from ouroboros import budget_pause as bp, owner_pause, owner_wait
    from ouroboros.gateways import claudexor
    from supervisor.owner_pause_control import _record_settled_member
    from supervisor.sleep_wake import wake_ready_sleepers

    env = _setup(tmp_path, monkeypatch)
    _park(env, tmp_path, monkeypatch)
    fence = owner_pause.install_fence(tmp_path, "sleeper", request_id="owner-pause")

    class Daemon:
        def handshake(self, **_kw):
            return {"compatible": True}

        def cancel_run(self, run_id, reason=""):
            return {"accepted": True, "status": "accepted"}

        def get_run(self, run_id, **_kw):
            return {"lastSeq": 4, "summary": {"state": "cancelled", "spendUsd": 0.0,
                                              "outcomeFacts": {"reason": "host_cancelled"},
                                              "resumable": {"cause": "cancelled"}}}

        def close(self):
            return None

    monkeypatch.setattr(claudexor, "ClaudexorGateway", lambda *a, **k: Daemon())
    _record_settled_member(env.q, env.workers.PENDING[0])
    row = bp.budget_pause_row(tmp_path, "sleeper")
    assert row["reason"] == "sleep", "the sleep keeps its own reason"
    assert [(item["run_id"], item["stop_outcome"]) for item in row["episode_runs"]] == [("run-one", "confirmed")]
    assert "episode_runs_partial" not in row, "this Pause's own first observation starts its list"
    owner_pause.release_fence(tmp_path, "sleeper", reason="owner_resume")
    assert not owner_pause.fence_closed(owner_pause.read_fence(tmp_path, "sleeper")), fence
    out = wake_ready_sleepers(env.q)
    assert out and out[0]["ok"], out
    ctx, _ = _loop_ctx(tmp_path, "sleeper")
    ctx.budget_pause_resume = env.workers.PENDING[0]["_budget_pause_resume"]
    saved = bp.load_budget_pause(ctx)
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a: (None, "max"))
    messages: list = []
    bp.resume_paused_loop(SimpleNamespace(_ctx=ctx), saved, messages, {}, {}, set(), budget_remaining_usd=5.0)
    texts = [str(message.get("content") or "") for message in messages]
    assert any("body retained your exact external runs" in text for text in texts), "sleep notices stay"
    named = [line for text in texts for line in text.splitlines() if line.startswith("- run run-one")]
    assert len(named) == 1, f"the wake names the run the owner's Pause stopped, once: {texts}"
    line = named[0]
    assert "this Pause requested its stop (stop outcome: confirmed)" in line
    assert "engine terminal: cancelled (host_cancelled)" in line


def test_retained_lane_dependency_blocks_cold_but_unencumbered_cold_survives(tmp_path, monkeypatch):
    from ouroboros import model_sleep, delegate_custody as custody
    from ouroboros.task_results import write_task_result
    env = _setup(tmp_path, monkeypatch)
    write_task_result(tmp_path, "other-root", "scheduled", project_id="project", root_task_id="other-root")
    selected = model_sleep.selectors(env.ctx, tasks=["other-root"])
    with pytest.raises(ValueError, match="retains the project lane"):
        model_sleep.request_sleep(env.ctx, selected, "cold")
    custody.emit(tmp_path, custody.SETTLED, {"run_id": "run-one", "task_id": "sleeper", "state": "succeeded"})
    assert model_sleep.request_sleep(env.ctx, selected, "cold")["reason"] == "sleep_armed"


def test_unaccepted_request_never_releases_worker_and_new_scope_never_borrows_receipt(tmp_path, monkeypatch):
    from ouroboros import budget_pause as bp, sleep_custody as sc
    from supervisor.events_budget import install_exact_budget_pause
    env = _setup(tmp_path, monkeypatch)
    row = _park(env, tmp_path, monkeypatch)
    accepted = copy.deepcopy(row)
    assert sc.accepted_receipt("sleeper", accepted)
    for field, changed in [("task_attempt", 2), ("pause_generation", 999), ("source_ref", {"sha256": "other"})]:
        assert not sc.accepted_receipt("sleeper", {**accepted, field: changed})
    assert not sc.accepted_receipt("foreign", accepted)
    env.workers.PENDING.clear()
    env.workers.RUNNING["sleeper"] = {"task": env.task, "worker_id": 0, "attempt": 1}
    bp.set_budget_pause(tmp_path, "sleeper", {**row, "state": "pausing",
                        "external_custody": {**row["external_custody"], "state": "requested"}})
    with pytest.raises(ValueError, match="no exact body acceptance"):
        install_exact_budget_pause(_supervisor_ctx(tmp_path, env.workers, env.q, [], []), "sleeper",
                                   bp.exact_pause_marker(row)["checkpoint"])
    assert "sleeper" in env.workers.RUNNING and not env.workers.PENDING


def test_readonly_transfer_keeps_the_capability_without_reserving_a_writer_lane(tmp_path, monkeypatch):
    from ouroboros.project_lease import cold_project_holders
    env = _setup(tmp_path, monkeypatch, access="readonly")
    row = _park(env, tmp_path, monkeypatch)
    assert row["external_custody"]["state"] == "accepted"
    assert not row["external_custody"]["retains_project_lane"]
    assert cold_project_holders(env.workers.PENDING) == {}


def test_accepted_owner_projection_is_passive_and_does_not_protect_all_pending(tmp_path, monkeypatch):
    from ouroboros import budget_pause as bp, sleep_custody as sc
    env = _setup(tmp_path, monkeypatch)
    _park(env, tmp_path, monkeypatch)
    env.workers.PENDING.append({"id": "unrelated", "_budget_pause": {"reason": "sleep"}})
    monkeypatch.setattr(bp, "budget_pause_row", lambda *_a: pytest.fail("passive census opened a document"))
    monkeypatch.setattr(sc, "_runtime", lambda *_a: pytest.fail("passive census touched runtime"))
    assert sc.accepted_owner_ids(env.q) == {"sleeper"}


def test_cold_rebind_reads_only_active_and_accepted_pending_carriers(tmp_path, monkeypatch):
    from ouroboros import budget_pause as bp, sleep_custody as sc
    env = _setup(tmp_path, monkeypatch)
    _park(env, tmp_path, monkeypatch)
    env.workers.PENDING.extend({"id": f"queued-{i}"} for i in range(100))
    original, read = bp.budget_pause_row, []
    def exact(root, task_id):
        read.append(task_id)
        return original(root, task_id)
    monkeypatch.setattr(bp, "budget_pause_row", exact)
    assert sc.reconcile(env.q, runtime=env.runtime) == {"sleeper"}
    assert read == ["sleeper"]


def test_body_acceptance_keeps_a_late_local_writer_as_its_actual_hold(tmp_path, monkeypatch):
    from ouroboros import budget_pause as bp, sleep_custody as sc
    from ouroboros.task_results import write_task_result

    env = _setup(tmp_path, monkeypatch)
    pause = _park(env, tmp_path, monkeypatch)
    write_task_result(tmp_path, "child", "scheduled", root_task_id="sleeper",
                      launch_handoffs={"local": {"tool": "opaque"}})
    with pytest.raises(bp.ColdSleepWritersUnsettled, match="local custody"):
        sc.confirm_release(env.ctx, pause, pause, control=lambda _ctx: "")
    assert sc.accepted_receipt("sleeper", bp.budget_pause_row(tmp_path, "sleeper"))
    write_task_result(tmp_path, "child", "scheduled", launch_handoffs={})
    assert sc.confirm_release(env.ctx, pause, pause, control=lambda _ctx: "") == pause
