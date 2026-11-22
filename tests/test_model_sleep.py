"""Owner Batch4 (6B): the model's own warm/cold sleep, through its REAL consumers.

``await_messages`` (its default in-slot wait untouched), the owner-wait park
loops, the execution clocks, the supervisor park/grant, the cold park at the
round boundary, the sleep-wake policy with its vetoes, and the Restart/Panic
holds are driven directly.
"""

from __future__ import annotations

import datetime
import json
import time
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _fast_hold, _install_queue, _loop_ctx, _quiet_external


def _ctx(tmp_path, task_id="sleeper", *, root="sleeper"):
    from ouroboros.model_wait import TaskModelWait

    ctx = SimpleNamespace(task_id=task_id, root_task_id=root, drive_root=tmp_path, budget_drive_root=tmp_path,
                          task_attempt=1, _loop_mailbox_seen_ids=set(), task_metadata={},
                          owner_wait_callback=lambda *_a, **_k: "unknown", pending_events=[],
                          event_queue=SimpleNamespace(put=lambda _e: None), current_chat_id=0,
                          _budget_paused_sec=0.0)
    ctx.model_wait_context = TaskModelWait(task={"id": task_id, "root_task_id": root,
                                                 "budget_drive_root": str(tmp_path)},
                                           drive_root=tmp_path, event_queue=None, worker_slot_held=False)
    ctx.model_wait_context.tool_context = ctx
    return ctx


def _result(tmp_path, task_id, status="running", **fields):
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, task_id, status, chat_id=0, **fields)


def _mail(tmp_path, task_id, text, *, sender="", kind="task_message", msg_id=None):
    from ouroboros.owner_mailbox import write_owner_message

    assert write_owner_message(tmp_path, text, task_id, msg_id=msg_id or f"m-{time.time_ns()}", kind=kind)
    if sender:  # write_owner_message has no sender field: stamp it as the task-message writers do
        from ouroboros.owner_mailbox import _mailbox_path

        path = _mailbox_path(tmp_path, task_id)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[-1]["source_task_id"] = sender
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _await_tool(ctx, **args):
    """The registered tool through the executor's public dispatch (``ToolRegistry.execute``), whose
    guards read the ``repo_dir`` every ``ToolContext`` carries."""
    from ouroboros.tools.registry import ToolRegistry

    ctx.repo_dir = getattr(ctx, "repo_dir", None) or ctx.drive_root
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    return registry.execute("await_messages", args)


def test_an_unnamed_mode_sleeps_and_a_named_in_slot_wait_is_unchanged(tmp_path):
    """Owner Q2=A: with no mode named, an omitted or positive timeout_sec or a selector sleeps; a named
    in_slot and a zero window hold the slot as before, and in_slot never takes a selector."""
    from ouroboros.tools.control_task_results import _await_messages

    _result(tmp_path, "peer")
    ctx = _ctx(tmp_path)
    out = json.loads(_await_messages(ctx, 1, mode="in_slot"))
    assert out["reason"] == "timeout" and out["slot"] == "held"
    assert json.loads(_await_messages(ctx, 0))["slot"] == "held", "a zero window is a peek in the slot"
    refused = _await_messages(ctx, 1, mode="in_slot", senders=["peer"])
    assert "in_slot takes only timeout_sec" in refused and not getattr(ctx, "_model_sleep", None)
    armed = json.loads(_await_messages(ctx, 30))
    assert armed["reason"] == "sleep_armed" and ctx._model_sleep["wake_at"] and ctx._model_sleep["any_mail"]
    assert ctx._model_sleep["mode"] == armed["mode"] in {"cold", "warm"}
    ctx._model_sleep, ctx._owner_wait_requested = None, ""
    assert json.loads(_await_messages(ctx, senders=["peer"]))["reason"] == "sleep_armed"
    assert ctx._model_sleep["senders"] == ["peer"] and not ctx._model_sleep["any_mail"]
    ctx._model_sleep, ctx._owner_wait_requested = None, ""
    bare = json.loads(_await_messages(ctx))
    assert bare["reason"] == "sleep_armed" and ctx._model_sleep["any_mail"] and not ctx._model_sleep["wake_at"]
    from ouroboros.model_sleep import wake_reason

    assert not wake_reason(ctx, ctx._model_sleep)
    _mail(tmp_path, ctx.task_id, "from an unselected peer", sender="peer")
    assert wake_reason(ctx, ctx._model_sleep)
    presence = _ctx(tmp_path, task_id="presence")
    presence.owner_wait_callback = None
    assert json.loads(_await_messages(presence))["slot"] == "held"
    assert not getattr(presence, "_model_sleep", None)


def test_timeout_sec_is_the_wake_time_of_any_sleep_and_an_explicit_time_still_outranks_it(tmp_path):
    """#3332: timeout_sec is a sleep's wake time whether the mode is omitted or named warm/cold, beside
    the selected sources. An explicit wake_after_sec/wake_at still outranks it, a zero or omitted one arms
    no time (events only), and a negative or non-integer one is refused as an unnamed sleep refuses it."""
    _result(tmp_path, "sleeper")
    _result(tmp_path, "child")
    ctx = _ctx(tmp_path)

    def arm(**args):
        ctx._model_sleep, ctx._owner_wait_requested, ctx._owner_wait_deadline_at = None, "", ""
        out = _await_tool(ctx, **args)
        assert '"reason": "sleep_armed"' in out, (args, out)
        assert ctx._model_sleep["mode"] == args.get("mode", ctx._model_sleep["mode"]), (args, out)
        return ctx._model_sleep

    for mode in ({}, {"mode": "warm"}, {"mode": "cold"}):
        before = datetime.datetime.now(datetime.timezone.utc)
        sleep = arm(timeout_sec=60, tasks=["child"], **mode)
        after = datetime.datetime.now(datetime.timezone.utc)
        assert sleep["wake_at"] and ctx._owner_wait_deadline_at == sleep["wake_at"], mode
        wake = datetime.datetime.fromisoformat(sleep["wake_at"])
        assert before + datetime.timedelta(seconds=60) <= wake <= after + datetime.timedelta(seconds=60)
        assert sleep["tasks"] == ["child"] and sleep["any_mail"] is False
    later = "2099-01-01T00:00:00+00:00"
    assert arm(mode="warm", timeout_sec=60, wake_at=later)["wake_at"] == later
    soon = datetime.datetime.fromisoformat(arm(mode="cold", timeout_sec=3600, wake_after_sec=5)["wake_at"])
    assert soon <= datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=5)
    assert arm(mode="warm")["wake_at"] == arm(mode="cold", timeout_sec=0)["wake_at"] == ""
    for mode in ({}, {"mode": "warm"}, {"mode": "cold"}):
        for bad in (-5, "soon"):
            ctx._model_sleep = None
            refused = _await_tool(ctx, timeout_sec=bad, **mode)
            assert refused.startswith("⚠️ TOOL_ARG_ERROR (await_messages)") and not ctx._model_sleep, refused


def test_selectors_are_validated_and_a_ready_source_answers_at_once(tmp_path):
    from ouroboros import model_sleep
    from ouroboros.tools.control_task_results import _await_messages

    _result(tmp_path, "sleeper")
    _result(tmp_path, "peer-a")
    _result(tmp_path, "child-1")
    ctx = _ctx(tmp_path)
    assert "task ghost is unknown" in _await_messages(ctx, mode="warm", tasks=["ghost"])
    assert "not delegated runs of this task" in _await_messages(ctx, mode="warm", runs=["run-x"])
    assert "not both" in _await_messages(ctx, mode="warm", wake_at="2099-01-01T00:00:00+00:00", wake_after_sec=5)
    chosen = model_sleep.selectors(ctx, senders=["peer-a"], wake_after_sec=60)
    assert chosen["any_mail"] is False and chosen["wake_at"]
    # Already-terminal selected task: answered at once, nothing armed.
    _result(tmp_path, "child-1", "completed")
    ready = json.loads(_await_messages(ctx, mode="warm", tasks=["child-1"]))
    assert ready == {"reason": "ready", "woke_by": "task:child-1:completed", "slept": False, "mode": "warm"}
    assert not getattr(ctx, "_model_sleep", None)
    # A passed wake time and pending owner words also answer at once.
    past = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=5)).isoformat()
    assert json.loads(_await_messages(ctx, mode="warm", wake_at=past))["woke_by"] == "timeout"
    _mail(tmp_path, "sleeper", "please stop and summarise", kind="owner_text")
    assert json.loads(_await_messages(ctx, mode="warm", senders=["peer-a"]))["woke_by"] == "owner_text"


def test_a_warm_sleep_wakes_only_on_its_selected_source_and_never_loses_one_landing_before_the_park(tmp_path):
    """Check → install → recheck: selected mail written after the tool's check but
    before the park wakes the very first poll; unselected mail never wakes it."""
    from ouroboros.owner_wait import checkpoint_owner_wait, direct_owner_wait
    from ouroboros.tools.control_task_results import _await_messages

    _result(tmp_path, "sleeper")
    _result(tmp_path, "peer-a")
    _result(tmp_path, "peer-b")
    ctx = _ctx(tmp_path)
    armed = json.loads(_await_messages(ctx, mode="warm", senders=["peer-a"]))
    assert armed["reason"] == "sleep_armed" and ctx._owner_wait_requested.startswith("sleep:")
    checkpoint = checkpoint_owner_wait(ctx, [{"role": "user", "content": "x"}], {}, {}, 1, [], set())
    assert checkpoint["reason"] == "sleep" and checkpoint["quiz_id"] == ""
    assert checkpoint["sleep"]["senders"] == ["peer-a"]
    _mail(tmp_path, "sleeper", "noise", sender="peer-b")          # unselected: stays unread, no wake
    _mail(tmp_path, "sleeper", "tests done", sender="peer-a")     # lands before the park
    assert direct_owner_wait(ctx, checkpoint) == "mail:peer-a"
    from ouroboros.owner_mailbox import OwnerMailboxPeek

    assert OwnerMailboxPeek().pending(tmp_path, "sleeper", set(), 1), "nothing was acknowledged by the wake"


def test_a_named_warm_sleep_saves_its_timeout_sec_and_that_time_wakes_the_park(tmp_path):
    """#3332: the saved continuation carries the wake time beside the selected task; the time wakes it."""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.owner_wait import checkpoint_owner_wait, direct_owner_wait

    _result(tmp_path, "sleeper")
    _result(tmp_path, "child")
    ctx = _ctx(tmp_path)
    armed = _await_tool(ctx, mode="warm", tasks=["child"], timeout_sec=1)
    assert '"reason": "sleep_armed"' in armed, armed
    wake_at = ctx._model_sleep["wake_at"]
    assert wake_at, "the named warm sleep kept no wake time"
    checkpoint = checkpoint_owner_wait(ctx, [{"role": "user", "content": "x"}], {}, {}, 1, [], set())
    saved = json.loads(read_actor_source_bytes(tmp_path, "sleeper", checkpoint["source_ref"]))
    assert saved["sleep"]["wake_at"] == saved["wait_deadline_at"] == wake_at
    assert saved["sleep"]["tasks"] == ["child"] and saved["sleep"]["mode"] == "warm"
    assert direct_owner_wait(ctx, checkpoint) == "timeout", "the selected child still runs: only the time wakes it"


def test_an_owner_pause_or_stop_control_is_never_filtered(tmp_path):
    from ouroboros import model_sleep
    from ouroboros.owner_mailbox import KIND_OWNER_PAUSE

    _result(tmp_path, "sleeper")
    _result(tmp_path, "peer-a")
    ctx = _ctx(tmp_path)
    chosen = model_sleep.selectors(ctx, senders=["peer-a"])
    assert model_sleep.wake_reason(ctx, chosen) == ""
    _mail(tmp_path, "sleeper", "owner_pause", kind=KIND_OWNER_PAUSE)
    assert model_sleep.wake_reason(ctx, chosen) == "control:owner_pause"


def test_sleep_is_excluded_from_execution_once_and_the_deadline_does_not_move(tmp_path):
    from ouroboros import model_sleep
    from ouroboros.model_wait import execution_elapsed_seconds

    ctx = _ctx(tmp_path)
    waiter = ctx.model_wait_context
    waiter.started_monotonic -= 150.0                       # 150 s of wall time: 100 s working...
    model_sleep.begin(ctx)
    waiter.sleep_started_monotonic -= 50.0                  # ...then the last 50 s asleep
    ctx._model_sleep_started -= 50.0
    assert 99.0 <= waiter.executed_seconds() <= 101.0
    slept = model_sleep.end(ctx)
    assert 49.0 <= slept <= 51.0 and 49.0 <= waiter.budget_paused_sec <= 51.0
    assert 99.0 <= waiter.executed_seconds() <= 101.0, "folded once, never subtracted twice"
    # The supervisor's row: live exclusion while parked, folded at the grant.
    now = time.time()
    meta = {"started_at": now - 300, "sleep_parked_at": now - 120}
    assert 179.0 <= execution_elapsed_seconds(meta, now) <= 181.0


def test_the_pooled_park_excludes_the_sleep_live_and_the_grant_folds_it(tmp_path, monkeypatch):
    from supervisor import worker_owner_wait
    from supervisor.worker_owner_wait import handle_owner_wait

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _result(tmp_path, "sleeper")
    proc = SimpleNamespace(pid=77, is_alive=lambda: True)
    commands: list = []
    worker = SimpleNamespace(wid=0, proc=proc, busy_task_id="sleeper", reaping=False, active_capacity=True,
                             in_q=SimpleNamespace(put=commands.append))
    workers.WORKERS[0] = worker
    meta = {"task": {"id": "sleeper"}, "worker_id": 0, "attempt": 1, "started_at": time.time() - 10}
    workers.RUNNING["sleeper"] = meta
    ctx = SimpleNamespace(RUNNING=workers.RUNNING, WORKERS=workers.WORKERS, DRIVE_ROOT=tmp_path)
    checkpoint = {"wait_id": "w1", "task_attempt": 1, "source_ref": {"x": 1}, "reason": "sleep",
                  "budget_paused_sec": 0.0, "sleep_started_at": time.time(),
                  "sleep": {"senders": ["peer-a"], "any_mail": False}}
    handle_owner_wait({"task_id": "sleeper", "wait_id": "w1", "task_attempt": 1, "worker_id": 0, "pid": 77,
                       "phase": "park", "checkpoint": checkpoint}, ctx)
    assert isinstance(meta.get("sleep_parked_at"), float) and worker.active_capacity is False
    meta["sleep_parked_at"] -= 30.0
    parked_at = meta["sleep_parked_at"]
    meta["owner_wait_resume_requested"] = True
    monkeypatch.setattr(worker_owner_wait, "_resume_memory_allowed", lambda *_a: True)
    monkeypatch.setattr(worker_owner_wait, "_resume_authority_allowed", lambda *_a: True)
    before_grant = time.time()
    assert worker_owner_wait._grant_resume("sleeper", meta, worker) is True
    after_grant = time.time()
    assert "sleep_parked_at" not in meta
    assert before_grant - parked_at <= meta["budget_paused_sec"] <= after_grant - parked_at


def test_a_cold_sleep_is_refused_over_its_own_live_writers(tmp_path, monkeypatch):
    from ouroboros import model_sleep
    from ouroboros.tools import services

    _result(tmp_path, "sleeper")
    ctx = _ctx(tmp_path)
    record = SimpleNamespace(task_id="sleeper", name="dev-server", proc=SimpleNamespace(poll=lambda: None))
    monkeypatch.setitem(services._SERVICES, "sleeper:dev-server", record)
    with pytest.raises(ValueError, match="service dev-server"):
        model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, wake_after_sec=60), model_sleep.MODE_COLD)
    services._SERVICES.pop("sleeper:dev-server", None)
    armed = model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, wake_after_sec=60), model_sleep.MODE_COLD)
    assert armed["reason"] == "sleep_armed" and ctx._model_sleep["mode"] == "cold"


def _cold_park(tmp_path, monkeypatch, workers, task_id="sleeper", arm=None, barrier=None, **selected):
    """``arm(ctx)`` arms the sleep through the tool instead of these ``selected`` sources."""
    from ouroboros import budget_pause, model_sleep
    from supervisor.events_budget import install_exact_budget_pause

    _result(tmp_path, task_id)
    ctx, limit_ctx = _loop_ctx(tmp_path, task_id)
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    monkeypatch.setattr(model_sleep, "cold_blockers", lambda _ctx, **_kw: [])
    if arm is None:
        ctx._model_sleep = {"sleep_id": "s1", "mode": "cold", **model_sleep.selectors(ctx, **selected)}
    else:
        arm(ctx)
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.enter_cold_sleep(limit_ctx)
    budget_pause.end_dispatch_fence(task_id)
    row = raised.value.pause
    workers.RUNNING[task_id] = {"task": {"id": task_id, "type": "task", "chat_id": 0, "root_task_id": task_id},
                                "worker_id": 0, "attempt": 1}
    sup = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                          WORKERS=workers.WORKERS, sort_pending=lambda: None,
                          persist_queue_snapshot=barrier or (lambda reason="": True), bridge=None)
    install_exact_budget_pause(sup, task_id, budget_pause.exact_pause_marker(row)["checkpoint"])
    return row


def test_a_cold_sleep_parks_exactly_and_only_its_own_readiness_wakes_it(tmp_path, monkeypatch):
    from ouroboros.budget_pause import budget_pause_row
    from supervisor.sleep_wake import wake_ready_sleepers

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    _result(tmp_path, "peer-a")
    row = _cold_park(tmp_path, monkeypatch, workers, senders=["peer-a"])
    assert row["reason"] == "sleep" and row["sleep"]["senders"] == ["peer-a"] and "sleep_seen" in row
    parked = workers.PENDING[0]
    assert parked["_budget_pause"]["reason"] == "sleep" and parked["_budget_pause"]["scope"] == "task"
    assert wake_ready_sleepers(queue) == [], "nothing selected has happened yet"
    _mail(tmp_path, "sleeper", "unrelated", sender="someone-else")
    assert wake_ready_sleepers(queue) == []
    _mail(tmp_path, "sleeper", "done", sender="peer-a")
    outcome = wake_ready_sleepers(queue)
    assert outcome[0]["ok"] is True and outcome[0]["ready"]["reason"] == "mail:peer-a"
    handoff = workers.PENDING[0]["_budget_pause_resume"]
    assert handoff["selected_by"] == "sleep_wake"
    assert budget_pause_row(tmp_path, "sleeper")["sleep_ready"]["reason"] == "mail:peer-a"


def test_readiness_is_recorded_but_never_overrides_an_owner_pause_or_a_restart_hold(tmp_path, monkeypatch):
    from ouroboros import owner_pause
    from ouroboros.budget_pause import budget_pause_row
    from supervisor.sleep_wake import wake_ready_sleepers

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    past = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=1)).isoformat()
    _cold_park(tmp_path, monkeypatch, workers, wake_at=past)
    time.sleep(1.2)
    owner_pause.install_fence(tmp_path, "sleeper", request_id="p")
    vetoed = wake_ready_sleepers(queue)
    assert vetoed[0]["error"] == "sleep_wake_vetoed" and vetoed[0]["veto"] == "owner_pause"
    assert budget_pause_row(tmp_path, "sleeper")["sleep_ready"]["reason"] == "timeout", "readiness kept"
    owner_pause.release_fence(tmp_path, "sleeper", reason="test")
    # The owner's Restart now holds it: the kept readiness still does not wake it.
    workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Owner restart",
                         hold_never_started=True)
    held = wake_ready_sleepers(queue)
    assert held[0]["veto"] == "owner_restart_hold"
    assert workers.PENDING[0]["_budget_pause"]["reason"] == "sleep", "the saved sleep itself is retained"


def test_a_panic_boot_holds_a_cold_sleeper_before_its_marker_is_consumed(tmp_path, monkeypatch):
    from supervisor.events_budget import HOLD_PANIC, budget_hold_fact

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _cold_park(tmp_path, monkeypatch, workers, wake_after_sec=1)
    queue.persist_queue_snapshot(reason="before_panic")
    workers.PENDING[:] = []
    (tmp_path / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")
    assert queue.restore_pending_from_snapshot() == 1
    assert budget_hold_fact(workers.PENDING[0])["reason"] == HOLD_PANIC


def test_an_ordinary_restart_restores_a_cold_sleep_that_wakes_itself(tmp_path, monkeypatch):
    from supervisor.events_budget import budget_hold_fact
    from supervisor.sleep_wake import wake_ready_sleepers

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    _cold_park(tmp_path, monkeypatch, workers, wake_after_sec=1)
    # An update or crash recovery: neither the owner's Restart flag nor a Panic flag.
    assert queue.persist_queue_snapshot(reason="before_update")
    workers.PENDING[:] = []
    assert queue.restore_pending_from_snapshot() == 1
    assert budget_hold_fact(workers.PENDING[0]) is None
    time.sleep(1.2)
    outcome = wake_ready_sleepers(queue)
    assert outcome[0]["ok"] is True and outcome[0]["ready"]["reason"] == "timeout"


def test_a_named_cold_sleep_parks_its_timeout_sec_and_that_time_wakes_it(tmp_path, monkeypatch):
    """#3332: the saved pause row carries the wake time beside the selected task; the time wakes it."""
    from supervisor.sleep_wake import wake_ready_sleepers

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    _result(tmp_path, "child")

    def arm(ctx):
        armed = _await_tool(ctx, mode="cold", tasks=["child"], timeout_sec=1)
        assert '"reason": "sleep_armed"' in armed, armed

    row = _cold_park(tmp_path, monkeypatch, workers, arm=arm)
    assert row["sleep"]["mode"] == "cold" and row["sleep"]["tasks"] == ["child"]
    assert row["sleep"]["wake_at"], "the named cold sleep kept no wake time"
    time.sleep(1.2)
    outcome = wake_ready_sleepers(queue)
    assert outcome[0]["ok"] is True and outcome[0]["ready"]["reason"] == "timeout", outcome


@pytest.mark.parametrize("source", ["mail", "task", "runs"])
def test_read_errors_do_not_hide_a_selected_time_or_become_a_wake(tmp_path, monkeypatch, source):
    from ouroboros import delegate_custody_current, model_sleep, owner_wait, task_store
    from supervisor.sleep_wake import _readiness

    ctx = _ctx(tmp_path)
    failure = OSError(f"unreadable {source}")

    def broken(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(owner_wait, "_wait_entries", lambda _ctx: [])
    monkeypatch.setattr(model_sleep, "_owned_runs", lambda *_a, **_k: {})
    monkeypatch.setattr(delegate_custody_current, "run", lambda *_a: None)
    monkeypatch.setattr(model_sleep, "_service_wake", lambda _pin: "")
    # The typed task reader propagates store-open failures.
    # The service reader instead turns its own read failure into an unknown wake.
    reader = {"mail": (owner_wait, "_wait_entries"), "task": (task_store, "task_versions"),
              "runs": (model_sleep, "_owned_runs")}[source]
    monkeypatch.setattr(*reader, broken)
    chosen = {"tasks": ["selected"], "runs": ["run-one"], "body_runs": [["run-one", "inv"]],
              "services": [{"name": "selected-service"}], "wake_at": "2000-01-01T00:00:00+00:00"}
    assert owner_wait._sleep_wake(ctx, {"sleep": chosen}) == "timeout"
    assert _readiness(SimpleNamespace(DRIVE_ROOT=tmp_path), {"id": "sleeper"},
                      {"sleep": chosen}) == "timeout"
    assert model_sleep.request_sleep(ctx, chosen, "warm")["woke_by"] == "timeout"
    assert not getattr(ctx, "_model_sleep", None), "an already-ready source never parks"
    chosen["wake_at"] = "2099-01-01T00:00:00+00:00"
    with pytest.raises(OSError) as raised:
        model_sleep.wake_reason(ctx, chosen)
    assert raised.value is failure, "no ready source: preserve the original read failure"
    assert owner_wait._sleep_wake(ctx, {"sleep": chosen}) == ""


def test_failed_selected_task_does_not_hide_another_task_or_an_exited_service(tmp_path, monkeypatch):
    from ouroboros import model_sleep, owner_wait, task_results

    _result(tmp_path, "ready", "completed")
    ctx = _ctx(tmp_path)
    load = task_results.read_document

    def task_read(root, task_id, **kwargs):
        if task_id == "broken":
            raise OSError("broken selected task")
        return load(root, task_id, **kwargs)

    monkeypatch.setattr(task_results, "read_document", task_read)
    chosen = {"tasks": ["broken", "ready"]}
    assert model_sleep.wake_reason(ctx, chosen) == "task:ready:completed"
    monkeypatch.setattr(model_sleep, "_service_wake", lambda _pin: "service:server:exited:0")
    chosen = {"tasks": ["broken"], "services": [{"name": "server"}]}
    assert owner_wait._sleep_wake(ctx, {"sleep": chosen}) == "service:server:exited:0"


@pytest.mark.parametrize("source", ["runs", "mail"])
def test_failed_read_preserves_direct_wait_and_owner_control_priority(tmp_path, monkeypatch, source):
    from ouroboros import model_sleep, owner_wait

    _result(tmp_path, "sleeper")
    ctx = _ctx(tmp_path)
    def broken(*_a, **_k):
        raise OSError("unreadable selected run")
    reader = (model_sleep, "_owned_runs") if source == "runs" else (owner_wait, "_wait_entries")
    with monkeypatch.context() as fault:
        fault.setattr(*reader, broken)
        chosen = {"runs": ["run-one"], "wake_at": "2000-01-01T00:00:00+00:00"}
        checkpoint = {"wait_id": "wait", "reason": "sleep", "sleep": chosen, "quiz_id": "",
                      "wait_deadline_at": chosen["wake_at"]}
        assert owner_wait.direct_owner_wait(ctx, checkpoint) == "timeout"
    _mail(tmp_path, "sleeper", "pause", kind="owner_pause")
    assert model_sleep.wake_reason(ctx, chosen) == "control:owner_pause"
    assert owner_wait._fresh_wake(ctx, "", "timeout") == "control:owner_pause"


@pytest.mark.parametrize("source", ["runs", "mail"])
def test_pooled_sleep_requests_resume_after_read_failure_when_its_time_is_due(tmp_path, monkeypatch, source):
    import queue
    from ouroboros import model_sleep, owner_wait

    _result(tmp_path, "sleeper")
    ctx = _ctx(tmp_path)
    events = []
    def broken(*_a, **_k):
        raise OSError("unreadable selected run")
    reader = (model_sleep, "_owned_runs") if source == "runs" else (owner_wait, "_wait_entries")
    monkeypatch.setattr(*reader, broken)
    chosen = {"runs": ["run-one"], "wake_at": "2000-01-01T00:00:00+00:00"}
    checkpoint = {"wait_id": "wait", "sleep": chosen, "wait_deadline_at": chosen["wake_at"]}
    class Commands:
        calls = 0
        def get(self, **_kwargs):
            self.calls += 1
            assert self.calls <= 3, "the ready clock did not request resume after parking"
            if self.calls > 1 and not any(e.get("phase") == "resume" for e in events):
                raise queue.Empty()
            return {"type": "owner_wait", "task_id": "sleeper", "task_attempt": 1,
                    "wait_id": "wait", "phase": "parked" if self.calls == 1 else "resume_granted"}
    assert owner_wait.worker_owner_wait(0, Commands(), SimpleNamespace(put=events.append),
                                        ctx, checkpoint) == "timeout"
    assert [e["phase"] for e in events] == ["park", "resume"]


def test_failed_wake_refresh_cannot_invent_a_reason(tmp_path, monkeypatch):
    from ouroboros import owner_wait
    failure = OSError("mailbox unavailable")
    def broken(_ctx):
        raise failure
    monkeypatch.setattr(owner_wait, "_wait_entries", broken)
    with pytest.raises(OSError) as raised:
        owner_wait._fresh_wake(_ctx(tmp_path), "", "unknown")
    assert raised.value is failure


def test_cold_clock_is_recorded_after_read_failure_but_owner_pause_still_vetoes(tmp_path, monkeypatch):
    from ouroboros import deadline_utils, owner_pause, owner_wait
    from ouroboros.budget_pause import budget_pause_row
    from supervisor.sleep_wake import wake_ready_sleepers

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    _cold_park(tmp_path, monkeypatch, workers, wake_at="2099-01-01T00:00:00+00:00")
    owner_pause.install_fence(tmp_path, "sleeper", request_id="p")
    monkeypatch.setattr(deadline_utils, "utc_now", lambda: datetime.datetime(2100, 1, 1,
                                                                          tzinfo=datetime.timezone.utc))
    def unreadable(_ctx):
        raise OSError("mailbox unavailable")
    monkeypatch.setattr(owner_wait, "_wait_entries", unreadable)
    outcome = wake_ready_sleepers(queue)
    assert outcome[0]["ready"]["reason"] == "timeout"
    assert outcome[0]["veto"] == "owner_pause" and not outcome[0]["ok"]
    row = budget_pause_row(tmp_path, "sleeper")
    assert row["sleep_ready"]["reason"] == "timeout" and not row.get("grant")


@pytest.mark.parametrize("status,expected", [("running", ""), ("completed", "task:selected:completed")])
def test_selected_status_poll_never_reads_task_bodies(tmp_path, monkeypatch, status, expected):
    import sqlite3
    from contextlib import contextmanager
    from ouroboros import model_sleep, owner_wait, task_store
    from supervisor.sleep_wake import _readiness

    _result(tmp_path, "selected", status, result="large evidence" * 10000)
    _result(tmp_path, "other", "running")
    original = task_store.open_store
    reads = []

    @contextmanager
    def observed(*args, **kwargs):
        with original(*args, **kwargs) as txn:
            def authorize(action, table, column, _db, _trigger):
                if action == sqlite3.SQLITE_READ:
                    assert table != "task_bodies", "readiness needs status, never evidence"
                    reads.append((table, column))
                return sqlite3.SQLITE_OK
            txn.conn.set_authorizer(authorize)
            try:
                yield txn
            finally:
                txn.conn.set_authorizer(lambda *_args: sqlite3.SQLITE_OK)

    monkeypatch.setattr(task_store, "open_store", observed)
    chosen = {"tasks": ["other", "selected"]}
    ctx = _ctx(tmp_path)
    assert model_sleep.wake_reason(ctx, chosen) == expected
    assert owner_wait._sleep_wake(ctx, {"sleep": chosen}) == expected
    assert _readiness(SimpleNamespace(DRIVE_ROOT=tmp_path), {"id": "sleeper"}, {"sleep": chosen}) == expected
    assert ("tasks", "status") in reads


def test_busy_sleep_poll_logs_one_traceback_then_counts_and_resets(tmp_path, monkeypatch, caplog):
    from ouroboros import owner_wait, task_store
    from ouroboros.store import StoreBusy

    ctx = _ctx(tmp_path)
    checkpoint = {"wait_id": "one", "sleep": {"tasks": ["selected"]}}
    busy = True

    def read(*_args):
        if busy:
            raise StoreBusy("locked")
        return {"selected": (1, "running", False)}

    monkeypatch.setattr(task_store, "task_versions", read)
    for _ in range(3):
        assert owner_wait._sleep_wake(ctx, checkpoint) == ""
    records = [r for r in caplog.records if r.name == "ouroboros.owner_wait"]
    assert [bool(r.exc_info) for r in records] == [True, False, False]
    assert [r.args[-1] for r in records] == [1, 2, 3]
    busy = False
    assert owner_wait._sleep_wake(ctx, checkpoint) == ""
    busy = True
    assert owner_wait._sleep_wake(ctx, checkpoint) == ""
    assert caplog.records[-1].exc_info and caplog.records[-1].args[-1] == 1
    checkpoint = {**checkpoint, "wait_id": "two"}
    assert owner_wait._sleep_wake(ctx, checkpoint) == ""
    assert caplog.records[-1].exc_info and caplog.records[-1].args[-1] == 1


def test_wake_grant_has_no_durable_permission_until_its_snapshot(tmp_path, monkeypatch):
    from ouroboros.budget_pause import budget_pause_row
    from supervisor.sleep_wake import wake_ready_sleepers

    q, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 5.0)
    _result(tmp_path, "peer")
    original = _cold_park(tmp_path, monkeypatch, workers, senders=["peer"], barrier=q.publish_queue_snapshot)
    _mail(tmp_path, "sleeper", "ready", sender="peer")
    before = q.QUEUE_SNAPSHOT_PATH.read_bytes()

    class Crash(BaseException):
        pass

    def crash(reason):
        assert reason == "budget_exact_resume_granted"
        assert not q._queue_lock._is_owned()
        assert budget_pause_row(tmp_path, "sleeper")["state"] == "paused"
        raise Crash()

    with monkeypatch.context() as fault, pytest.raises(Crash):
        fault.setattr(q, "publish_queue_snapshot", crash)
        wake_ready_sleepers(q)
    assert q.QUEUE_SNAPSHOT_PATH.read_bytes() == before
    workers.PENDING.clear()
    assert q.restore_pending_from_snapshot() == 1
    assert workers.PENDING[0]["_budget_pause"]["checkpoint"]["source_ref"] == original["source_ref"]
    assert wake_ready_sleepers(q)[0]["ok"]
    assert wake_ready_sleepers(q) == []


def test_one_publication_for_all_ready_pauses_and_none_for_unready(tmp_path, monkeypatch):
    from ouroboros import budget_pause as bp
    from ouroboros.utils import atomic_write_json
    from supervisor.sleep_wake import wake_ready_sleepers
    from tests._budget_pause_exact_helpers import _install_queue, _parked

    q, state, _ = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    for i in range(3):
        task, row = _parked(tmp_path, monkeypatch, task_id=f"tech-{i}")
        bp.end_dispatch_fence(task["id"])
        row.update(
            reason="authority", rail=bp.RAIL_CONTROL_AUTHORITY, cause="dispatch_fence_unreadable", state="paused"
        )
        bp.set_budget_pause(tmp_path, task["id"], row)
        task["_budget_pause"] = bp.exact_pause_marker(row)
    atomic_write_json(
        tmp_path / "state/queue_dispatch.json", {"generation": "g", "budget_root_fences": [], "members": {}}
    )
    barriers, writes = [], []
    publish, write = q.publish_queue_snapshot, q.atomic_write_text

    def counted(reason=""):
        barriers.append(reason)
        return publish(reason=reason)

    def counted_write(path, data):
        writes.append((path.name, len(data)))
        return write(path, data)

    monkeypatch.setattr(q, "publish_queue_snapshot", counted)
    monkeypatch.setattr(q, "atomic_write_text", counted_write)
    with monkeypatch.context() as blocked:
        def refuse(reason):
            barriers.append(reason)
            return False
        blocked.setattr(q, "publish_queue_snapshot", refuse)
        refused = wake_ready_sleepers(q)
    assert len(barriers) == 1 and len(refused) == 3
    assert all(reply["error"] == "snapshot_not_persisted" for reply in refused)
    assert all("_budget_pause_resume" not in t and bp.budget_pause_row(tmp_path, t["id"])["state"] == "paused"
               for t in q.PENDING)
    barriers.clear()
    outcomes = wake_ready_sleepers(q)
    woken = [o for o in outcomes if o.get("ok")]
    snapshot_writes = [w for w in writes if w[0] == "queue_snapshot.json"]
    assert len(woken) == 3 and len(snapshot_writes) == len(barriers) == 1
    assert not wake_ready_sleepers(q) and len(barriers) == 1


def _sleep_after_consumed_resume(tmp_path, monkeypatch, *, fold_first=True, **second_arm):
    import copy
    from ouroboros import budget_pause, model_sleep, owner_wait
    from supervisor.events_budget import _handle_budget_pause
    from tests.test_budget_pause_lineage_restore import _install, _sup, _workers

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    monkeypatch.setattr(model_sleep, "cold_blockers", lambda _ctx, **_kw: [])
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_k: (None, "max"))
    for tid in ("sleeper", "peer-a"):
        _result(tmp_path, tid)
    sent, events, sup = _workers(workers), [], _sup(tmp_path, queue, workers)
    workers.WORKERS[0].busy_task_id, workers.WORKERS[0].proc = "sleeper", SimpleNamespace(pid=77, is_alive=lambda: True)
    workers.RUNNING["sleeper"] = {"task": {"id": "sleeper", "type": "task", "chat_id": 0, "root_task_id": "sleeper"},
                                  "worker_id": 0, "attempt": 1, "started_at": time.time() - 60.0}
    ctx, limit_ctx = _loop_ctx(tmp_path, "sleeper")
    ctx.repo_dir, ctx._loop_mailbox_seen_ids, ctx._budget_paused_sec = tmp_path, set(), 0.0
    ctx.event_queue = SimpleNamespace(put=events.append)

    def park_cold(**arm):
        ctx._model_sleep, ctx._owner_wait_requested, ctx._owner_wait_deadline_at = None, "", ""
        assert '"reason": "sleep_armed"' in _await_tool(ctx, mode="cold", **arm)
        with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
            budget_pause.enter_cold_sleep(limit_ctx)
        budget_pause.end_dispatch_fence("sleeper")
        _install(sup, "sleeper", raised.value.pause)
        return raised.value.pause

    first_pause = park_cold(wake_after_sec=3600)
    first_grant = queue.resume_budget_paused_task("sleeper")
    assert first_grant["ok"] is True
    workers.assign_tasks()
    restored_messages: list[dict] = []
    _, _, _, _, restored_round, _ = budget_pause.resume_paused_loop(
        limit_ctx.tools, budget_pause.load_budget_pause(ctx, sent[0]["_budget_pause_resume"]),
        restored_messages, {}, {}, set(), budget_remaining_usd=5.0,
    )
    consumed_evt = next(evt for evt in events if evt.get("phase") == "consumed")
    if fold_first:
        _handle_budget_pause(consumed_evt, sup)
    carrier = copy.deepcopy(workers.RUNNING["sleeper"]["task"]["_budget_pause_resume"])
    assert carrier["pause_id"] == first_pause["pause_id"] and carrier["authority"] == "explicit_resume"

    restored_messages.extend([
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_step", "type": "function", "function": {"name": "step", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_step", "content": "completed_once"},
    ])
    limit_ctx.messages, limit_ctx.round_idx = restored_messages, restored_round + 1
    sent.clear()
    events.clear()
    return queue, workers, sup, ctx, limit_ctx, sent, events, first_pause, first_grant, consumed_evt, carrier, park_cold(**second_arm)


@pytest.mark.parametrize("source", ["time", "mail"])
def test_a_second_cold_sleep_after_a_consumed_explicit_resume_wakes_once_and_resumes_without_replay(
    tmp_path, monkeypatch, source
):
    from ouroboros import budget_pause, deadline_utils, owner_pause
    from supervisor.sleep_wake import wake_ready_sleepers

    arm = {"wake_at": "2099-06-01T00:00:00+00:00"} if source == "time" else {"senders": ["peer-a"]}
    queue, workers, _sup, ctx, limit_ctx, sent, _events, first_pause, first_grant, _cevt, _carrier, second_pause = (
        _sleep_after_consumed_resume(tmp_path, monkeypatch, **arm)
    )
    assert second_pause["pause_id"] != first_pause["pause_id"] and second_pause["pause_generation"] == 2
    assert wake_ready_sleepers(queue) == []
    if source == "time":
        monkeypatch.setattr(deadline_utils, "utc_now",
                            lambda: datetime.datetime(2099, 6, 1, 0, 0, 5, tzinfo=datetime.timezone.utc))
        expected = "timeout"
    else:
        _mail(tmp_path, "sleeper", "peer reply ready", sender="peer-a")
        expected = "mail:peer-a"

    owner_pause.install_fence(tmp_path, "sleeper", request_id="hold-check")
    assert wake_ready_sleepers(queue)[0]["veto"] == "owner_pause"
    owner_pause.release_fence(tmp_path, "sleeper", reason="test")

    outcomes = wake_ready_sleepers(queue)
    assert len(outcomes) == 1 and outcomes[0].get("ok") is True and outcomes[0]["ready"]["reason"] == expected, outcomes
    assert wake_ready_sleepers(queue) == []
    workers.assign_tasks()
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["sleeper"]
    handoff = sent[0]["_budget_pause_resume"]
    assert (handoff["pause_id"], handoff["pause_generation"], handoff["authority"]) == (
        second_pause["pause_id"], 2, "sleep_readiness"
    ) and handoff["grant_id"] != first_grant["grant_id"]

    woken_messages: list[dict] = []
    _, _, _, _, round_after_wake, _ = budget_pause.resume_paused_loop(
        limit_ctx.tools, budget_pause.load_budget_pause(ctx, handoff), woken_messages, {}, {}, set(),
        budget_remaining_usd=5.0,
    )
    assert round_after_wake == limit_ctx.round_idx
    assert [m["content"] for m in woken_messages if m.get("tool_call_id") == "call_step"] == ["completed_once"]
    assert len([m for m in woken_messages if m.get("tool_call_id") == "call_b"]) == 1
    assert ("the wake time you chose" if source == "time" else "mail from peer-a") in woken_messages[-1]["content"]
    with pytest.raises(ValueError):
        budget_pause.load_budget_pause(ctx, handoff)


def test_late_first_consumption_and_prefix_snapshot_restore_keep_clocks_and_current_grant(tmp_path, monkeypatch):
    from ouroboros import budget_pause
    from supervisor.events_budget import _handle_budget_pause, budget_hold_fact
    from supervisor.sleep_wake import wake_ready_sleepers

    queue, workers, sup, ctx, limit_ctx, sent, events, _p1, _g1, late_consumed, prefix_p1_carrier, second_pause = (
        _sleep_after_consumed_resume(tmp_path, monkeypatch, fold_first=False, senders=["peer-a"])
    )
    folded_first = second_pause["paused_duration_sec"]
    assert folded_first > 0 and "_budget_pause_resume" not in workers.PENDING[0]
    _handle_budget_pause(late_consumed, sup)
    assert "_budget_pause_resume" not in workers.PENDING[0]
    assert budget_pause.budget_pause_row(tmp_path, "sleeper")["paused_duration_sec"] == pytest.approx(folded_first)

    snap = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    snap["pending"][0]["task"]["_budget_pause_resume"] = prefix_p1_carrier
    queue.QUEUE_SNAPSHOT_PATH.write_text(json.dumps(snap), encoding="utf-8")
    workers.PENDING.clear()
    assert queue.restore_pending_from_snapshot() == 1
    restored = workers.PENDING[0]
    assert budget_hold_fact(restored) is None and "_budget_pause_resume" not in restored
    assert restored["_budget_pause"]["checkpoint"]["pause_id"] == second_pause["pause_id"]

    _mail(tmp_path, "sleeper", "done", sender="peer-a")
    woke = wake_ready_sleepers(queue)
    assert len(woke) == 1 and woke[0].get("ok") is True and woke[0]["grant_id"] != _g1["grant_id"]
    assert wake_ready_sleepers(queue) == []

    workers.assign_tasks()
    meta = workers.RUNNING["sleeper"]
    before_late = (meta["sleep_parked_at"], meta["budget_paused_sec"], meta["task"]["_budget_pause_resume"]["grant_id"])
    _handle_budget_pause(late_consumed, sup)
    assert (meta["sleep_parked_at"], meta["budget_paused_sec"], meta["task"]["_budget_pause_resume"]["grant_id"]) == before_late

    handoff = sent[0]["_budget_pause_resume"]
    budget_pause.resume_paused_loop(
        limit_ctx.tools, budget_pause.load_budget_pause(ctx, handoff), [], {}, {}, set(), budget_remaining_usd=5.0
    )
    _handle_budget_pause(next(evt for evt in events if evt.get("phase") == "consumed"), sup)
    assert "sleep_parked_at" not in meta and meta["budget_paused_sec"] >= folded_first
    assert meta["budget_paused_sec"] == pytest.approx(budget_pause.budget_pause_row(tmp_path, "sleeper")["paused_duration_sec"])
    with pytest.raises(ValueError):
        budget_pause.load_budget_pause(ctx, handoff)


@pytest.mark.parametrize(
    ("case_id", "kind", "row_state", "grant_fields", "task", "lends"),
    [
        ("auto_sleep_granted", "exact", "resume_granted",
         {"authority": "sleep_readiness", "selected_by": "sleep_wake"},
         {"id": "r", "_budget_pause_resume": {"grant_id": "g1"}}, False),
        ("auto_sleep_resumed_handoff", "exact", "resumed",
         {"authority": "sleep_readiness", "consumed_at": 1.0},
         {"id": "r", "_budget_pause_resume": {"grant_id": "g1"}}, False),
        ("auto_authority_granted", "exact", "resume_granted",
         {"authority": "authority_readable", "selected_by": "authority_readable"},
         {"id": "r", "_budget_pause_resume": {"grant_id": "g1"}}, False),
        ("unknown_exact_not_repaired_by_handoff", "exact", "resume_granted",
         {"authority": "unexpected", "selected_by": "owner"},
         {"id": "r", "_budget_pause_resume": {"grant_id": "g1", "authority": "explicit_resume", "selected_by": "owner"}},
         False),
        ("null_exact_not_repaired_by_selection", "exact", "resumed",
         {"authority": None, "selected_by": "owner", "consumed_at": 1.0},
         {"id": "r", "_resume_selection": {"selected_by": "owner", "grant_id": "g1", "grant_generation": 1}},
         False),
        ("unknown_or_null_warm", "warm", "released", {"authority": None, "selected_by": "owner"}, {"id": "r"}, False),
        ("absent_exact_without_proof", "exact", "resume_granted", {},
         {"id": "r", "_budget_pause_resume": {"grant_id": "g1"}}, False),
        ("absent_exact_non_owner_selected_by", "exact", "resume_granted", {"selected_by": "child-1"},
         {"id": "r", "_budget_pause_resume": {"grant_id": "g1"}}, False),
        ("absent_warm_without_proof", "warm", "released", {}, {"id": "r"}, False),
        ("legacy_owner_exact_unconsumed", "exact", "resume_granted", {"selected_by": "owner"}, {"id": "r"}, True),
        ("legacy_owner_exact_handoff", "exact", "resumed", {"selected_by": "owner", "consumed_at": 1.0},
         {"id": "r", "_budget_pause_resume": {"grant_id": "g1"}}, True),
        ("legacy_owner_exact_retired_durable_only", "exact", "resumed",
         {"selected_by": "owner", "consumed_at": 1.0}, {}, False),
        ("legacy_owner_warm_live", "warm", "released", {"selected_by": "owner"}, {"id": "r"}, True),
        ("legacy_owner_warm_retired_durable_only", "warm", "released", {"selected_by": "owner"}, {}, False),
        ("modern_explicit_exact_unconsumed", "exact", "resume_granted",
         {"authority": "explicit_resume", "selected_by": "owner"}, {"id": "r"}, True),
        ("modern_explicit_exact_retained_selection", "exact", "resumed",
         {"authority": "explicit_resume", "consumed_at": 1.0},
         {"id": "r", "_resume_selection": {"selected_by": "owner", "grant_id": "g1", "grant_generation": 1}}, True),
        ("modern_explicit_warm_live", "warm", "released", {"authority": "explicit_resume"}, {"id": "r"}, True),
        ("modern_explicit_warm_retired_durable_only", "warm", "released", {"authority": "explicit_resume"}, {}, False),
    ],
    ids=lambda val: val if isinstance(val, str) else None,
)
def test_root_resume_grants_synthetic_provenance_table(case_id, kind, row_state, grant_fields, task, lends):
    from supervisor.events_budget import _root_resume_grants, resume_basis

    grant = {"grant_id": "g1", "generation": 1, **grant_fields}
    pause = ({"state": row_state, "pause_id": "p1", "resume_generation": 1, "grant": grant}
             if kind == "exact" else {})
    owner = ({"state": row_state, "fence_id": "f1", "generation": 1, "resume_grant": grant}
             if kind == "warm" else {})
    durable_id = {"grant_id": "g1", "generation": 1, "pause_id": "p1" if kind == "exact" else ""}
    state = {"task": task, "pause": pause, "owner_fence": owner, "latch": None,
             "basis": resume_basis("r", pause, owner)}
    assert _root_resume_grants(state) == (durable_id if lends else {}, durable_id), case_id


def test_automatic_sleep_wake_grants_no_new_child_selection_while_preserving_explicit_choice(tmp_path, monkeypatch):
    from ouroboros import budget_pause
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result
    from supervisor import events_budget as eb
    from supervisor.sleep_wake import wake_ready_sleepers
    from tests.test_restart_retention import _queued

    queue, workers, sup, ctx, limit_ctx, sent, events, _p1, first_grant, _cevt, _carrier, _second_pause = (
        _sleep_after_consumed_resume(tmp_path, monkeypatch, senders=["peer-a"])
    )
    for cid in ("prior-selected", "held-model", "held-owner"):
        write_task_result(tmp_path, cid, STATUS_SCHEDULED, chat_id=0, root_task_id="sleeper", parent_task_id="sleeper")
        row = _queued(cid, root_task_id="sleeper", parent_task_id="sleeper")
        eb.hold_budget_row(row, reason=eb.HOLD_SAVED_WORK, result_root=tmp_path)
        workers.PENDING.append(row)
    prior_child, held_model, held_owner = (next(r for r in workers.PENDING if r["id"] == cid)
                                           for cid in ("prior-selected", "held-model", "held-owner"))
    prior_child.pop(eb.BUDGET_HOLD_KEY)
    prior_child[eb.SELECTION_KEY] = {
        "selected_by": "sleeper", "selected_at": "2026-11-21T00:00:00+00:00", "by": {"actor": "ouroboros"},
        "root_grant_id": first_grant["grant_id"], "root_resume_generation": 1,
    }

    _mail(tmp_path, "sleeper", "ready", sender="peer-a")
    assert wake_ready_sleepers(queue)[0].get("ok") is True
    assert eb.live_root_resume_grant(queue, "sleeper", tmp_path) == {}
    assert queue.resume_budget_paused_task("held-model", selected_by="sleeper")["error"] == "root_resume_grant_missing"
    assert eb.budget_resume_dispatch_allowed(queue, prior_child) is True

    workers.assign_tasks()
    assert [t["id"] for t in sent] == ["sleeper"]
    handoff = sent[0]["_budget_pause_resume"]
    budget_pause.resume_paused_loop(
        limit_ctx.tools, budget_pause.load_budget_pause(ctx, handoff), [], {}, {},
        ctx._loop_mailbox_seen_ids, budget_remaining_usd=5.0,
    )
    assert eb.live_root_resume_grant(queue, "sleeper", tmp_path) == {}
    assert queue.resume_budget_paused_task("held-model", selected_by="sleeper")["error"] == "root_resume_grant_missing"
    _handle_consumed = next(evt for evt in events if evt.get("phase") == "consumed")
    eb._handle_budget_pause(_handle_consumed, sup)
    assert eb._root_resume_grants(eb._root_resume_state(queue, "sleeper", tmp_path)) == (
        {}, {"grant_id": handoff["grant_id"], "generation": 1, "pause_id": _second_pause["pause_id"]}
    )
    assert queue.resume_budget_paused_task("held-model", selected_by="sleeper")["error"] == "root_resume_grant_missing"
    assert eb.budget_hold_fact(held_model) is not None

    owner_choice = queue.resume_budget_paused_task("held-owner")
    assert owner_choice["ok"] is True and not held_owner[eb.SELECTION_KEY].get("root_grant_id")
    assert eb.budget_resume_dispatch_allowed(queue, held_owner) is True

    _result(tmp_path, "peer-b")
    ctx._model_sleep, ctx._owner_wait_requested, ctx._owner_wait_deadline_at = None, "", ""
    assert '"reason": "sleep_armed"' in _await_tool(ctx, mode="cold", senders=["peer-b"])
    with pytest.raises(budget_pause.BudgetPauseRequested) as third_sleep:
        budget_pause.enter_cold_sleep(limit_ctx)
    budget_pause.end_dispatch_fence("sleeper")
    eb.install_exact_budget_pause(sup, "sleeper", budget_pause.exact_pause_marker(third_sleep.value.pause)["checkpoint"])
    explicit = queue.resume_budget_paused_task("sleeper")
    assert explicit["ok"] is True
    assert eb.live_root_resume_grant(queue, "sleeper", tmp_path)["grant_id"] == explicit["grant_id"]
    assert queue.resume_budget_paused_task("held-model", selected_by="sleeper")["ok"] is True
