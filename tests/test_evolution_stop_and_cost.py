"""Adversarial state-machine coverage for `/evolve stop` and abnormal-termination
cost reconstruction.

Covers:
  - reconstruct_task_cost using the durable physical-attempt ledger by task_id;
  - stop_evolution_tasks cancelling only evolution tasks (pending AND running)
    through the durable-intent + typed-custody ingress;
  - the hard-timeout retry gate: a killed evolution task is NOT re-enqueued when
    the campaign is stopped, IS re-enqueued when still enabled, and either way
    records reconstructed cost/rounds (never zeros).
"""

import json
from types import SimpleNamespace

import pytest


def _write_events(tmp_path, rows):
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "events.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def test_reconstruct_task_cost_sums_llm_usage(tmp_path, monkeypatch):
    import supervisor.state as state

    monkeypatch.setattr(state, "DRIVE_ROOT", tmp_path)
    _write_events(tmp_path, [
        {"type": "llm_usage", "task_id": "t1", "cost": 0.5, "prompt_tokens": 100, "completion_tokens": 20},
        {"type": "llm_usage", "task_id": "t1", "cost": 0.25, "prompt_tokens": 50, "completion_tokens": 10},
        {"type": "llm_usage", "task_id": "other", "cost": 9.0, "prompt_tokens": 1, "completion_tokens": 1},
        {"type": "llm_round", "task_id": "t1", "cost_usd": 99},  # not llm_usage -> ignored
    ])

    cost, rounds, prompt, completion = state.reconstruct_task_cost("t1")
    assert round(cost, 6) == 0.75
    assert rounds == 2
    assert prompt == 150
    assert completion == 30
    # Unknown / empty task ids reconstruct to zeros, never raise.
    assert state.reconstruct_task_cost("missing") == (0.0, 0, 0, 0)
    assert state.reconstruct_task_cost("") == (0.0, 0, 0, 0)


def test_reconstruct_task_cost_never_fabricates_zero_when_ledger_unavailable(
    tmp_path, monkeypatch,
):
    import ouroboros.usage_accounting as accounting
    import supervisor.state as state

    monkeypatch.setattr(state, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(
        accounting, "ensure_legacy_imported",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(accounting.UsageLedgerCorrupt("bad")),
    )
    fields = state.reconstruct_task_cost("paid-task", fields=True)
    assert fields["cost_accounting_status"] == "unavailable"
    assert fields["cost_final"] is False
    assert fields["cost_accounting_error"] == "ledger_unavailable"
    assert fields["accounted_upper_bound_usd"] is None
    assert fields["total_rounds"] is None
    assert fields["ledger_integrity_degraded"] is True
    with pytest.raises(accounting.UsageAccountingError):
        state.reconstruct_task_cost("paid-task")


def test_stop_evolution_tasks_cancels_only_evolution_pending_and_running(monkeypatch):
    """GR2-13: PENDING evolution tasks go through the same durable-intent +
    typed-custody ingress as running ones (never an in-place prune), and only
    evolution tasks are touched."""
    import supervisor.queue as q

    monkeypatch.setattr(q, "RUNNING", {
        "evo1": {"task": {"type": "evolution"}},
        "task2": {"task": {"type": "task"}},
        "evo3": {"task": {"type": "evolution"}},
    })
    monkeypatch.setattr(q, "PENDING", [
        {"id": "evo-pending", "type": "evolution"},
        {"id": "plain-pending", "type": "chat"},
    ])
    intents: list = []
    monkeypatch.setattr(
        "ouroboros.cancel_intents.request_cancel",
        lambda root, tid, **kw: intents.append((tid, kw.get("source"))) or {},
    )
    custody: list = []
    monkeypatch.setattr(q, "cancel_task_custody",
                        lambda tid, **_kw: custody.append(tid) or q.CANCEL_CANCELLED)

    out = q.stop_evolution_tasks("test stop")

    assert sorted(out["cancelled"]) == ["evo-pending", "evo1", "evo3"]
    assert sorted(custody) == ["evo-pending", "evo1", "evo3"]
    assert sorted(tid for tid, _src in intents) == ["evo-pending", "evo1", "evo3"]
    assert all(src == "evolution_stop" for _tid, src in intents)
    assert not any(bucket for key, bucket in out.items() if key != "cancelled")


def test_stop_evolution_tasks_reports_settled_and_failed_honestly(monkeypatch):
    """GR2-13: 'cancelled' names only CANCEL_CANCELLED outcomes; already-settled
    and failed teardowns are separate buckets, and the report composer marks the
    stop incomplete only for still-live tasks."""
    import supervisor.queue as q

    monkeypatch.setattr(q, "RUNNING", {
        "evo-done": {"task": {"type": "evolution"}},
        "evo-stuck": {"task": {"type": "evolution"}},
    })
    monkeypatch.setattr(q, "PENDING", [])
    monkeypatch.setattr("ouroboros.cancel_intents.request_cancel",
                        lambda root, tid, **kw: {})
    outcomes = {"evo-done": q.CANCEL_ALREADY_SETTLED, "evo-stuck": q.CANCEL_FAILED}
    monkeypatch.setattr(q, "cancel_task_custody", lambda tid, **_kw: outcomes[tid])

    out = q.stop_evolution_tasks("test stop")

    assert out["cancelled"] == []
    assert out["already_settled"] == ["evo-done"]
    assert out["failed"] == ["evo-stuck"]
    lines, incomplete = q.evolution_stop_report(out)
    assert incomplete is True
    assert any("Already settled" in line and "evo-done" in line for line in lines)
    assert any("INCOMPLETE" in line and "evo-stuck" in line for line in lines)
    assert not any(line.startswith("🛑 Cancelled") for line in lines)


def _drive_hard_timeout(tmp_path, monkeypatch, *, evolution_enabled):
    """Drive enforce_task_timeouts for a single overdue evolution task and return
    (enqueued, emitted_events, written_result)."""
    import time

    import supervisor.queue as q
    import supervisor.state as state
    from supervisor import evolution_lifecycle
    from supervisor import workers as workers_mod
    import ouroboros.tools.services as services_mod

    state.init(tmp_path)
    q.init(tmp_path)
    campaign = evolution_lifecycle.start_evolution_campaign("Improve", source="test")
    state.update_state(lambda live: live.update(
        owner_chat_id=7,
        evolution_mode_enabled=True,
        evolution_owner_stopped=False,
    ))
    transaction = evolution_lifecycle.begin_evolution_transaction(
        "evo1", cycle=1, campaign=campaign,
    )
    if not evolution_enabled:
        state.update_state(lambda live: live.update(evolution_mode_enabled=False))
    monkeypatch.setattr(q, "FINALIZATION_GRACE_SEC", 0)
    # Activity model: drive an IDLE kill (no progress for the idle window), not a flat
    # wall-clock/ceiling kill — small idle/ceiling getters + a recent started_at keep
    # terminal_reason == "idle_timeout" so the evolution retry path is exercised.
    monkeypatch.setattr(q, "get_task_idle_timeout_sec", lambda: 1)
    monkeypatch.setattr(q, "get_per_call_timeout_ceiling_sec", lambda: 1)
    monkeypatch.setattr(q, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(q, "_reap_queue", q._stdqueue.Queue())
    # Three distinct legacy usage rows totalling $1.50 are imported into the
    # physical-attempt ledger before the kill.  Distinct timestamps make these
    # three calls rather than duplicated telemetry for one call.
    _write_events(tmp_path, [
        {
            "type": "llm_usage", "task_id": "evo1", "cost": 0.5,
            "prompt_tokens": 10, "completion_tokens": 2,
            "ts": f"2026-01-01T00:00:0{index}Z",
        }
        for index in range(3)
    ])

    task = {
        "id": "evo1",
        "type": "evolution",
        "chat_id": 7,
        "_attempt": 1,
        "metadata": {"evolution_transaction": transaction},
    }
    monkeypatch.setattr(q, "RUNNING", {
        "evo1": {"task": task, "started_at": time.time() - 1000, "worker_id": 0, "attempt": 1},
    })

    class _FakeProc:
        pid = 0
        def is_alive(self):
            return False
        def join(self, timeout=None):
            return None

    fake_worker = SimpleNamespace(busy_task_id="evo1", proc=_FakeProc(), wid=0)
    monkeypatch.setattr(workers_mod, "WORKERS", {0: fake_worker})
    monkeypatch.setattr(workers_mod, "respawn_worker", lambda wid: None)

    emitted = []
    monkeypatch.setattr(workers_mod, "get_event_q", lambda: SimpleNamespace(put=lambda evt: emitted.append(evt)))
    monkeypatch.setattr(services_mod, "archive_task_service_logs", lambda *a, **k: None)

    enqueued = []
    monkeypatch.setattr(q, "enqueue_task", lambda t, front=False: enqueued.append(t))
    monkeypatch.setattr(q, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(q, "send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(q, "load_state", lambda: {"evolution_mode_enabled": evolution_enabled, "owner_chat_id": 0})

    q.enforce_task_timeouts()
    # Variant A: terminal write + retry happen in the off-loop reaper; drain it here.
    while not q._reap_queue.empty():
        q._reap_timed_out_task(q._reap_queue.get_nowait())

    from ouroboros.task_results import load_task_result
    written = load_task_result(tmp_path, "evo1") or {}
    return enqueued, emitted, written


def test_hard_timeout_cleans_owner_mailbox_at_terminal_dispatch(tmp_path, monkeypatch):
    """A hard kill preserves authority until its durable terminal is dispatched."""
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, _mailbox_path, write_owner_message
    from supervisor import events as events_mod

    write_owner_message(tmp_path, "hard_timeout", "evo1", kind=KIND_FINALIZE_NOW)
    assert _mailbox_path(tmp_path, "evo1").exists()

    _enqueued, emitted, written = _drive_hard_timeout(
        tmp_path, monkeypatch, evolution_enabled=False,
    )
    assert _mailbox_path(tmp_path, "evo1").exists()
    done = next(evt for evt in emitted if evt.get("type") == "task_done")
    task = {"id": "evo1", "type": "evolution", "chat_id": 7, "_attempt": 1}
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING={}, PENDING=[], WORKERS={},
        send_with_budget=lambda *_a, **_k: None,
        persist_queue_snapshot=lambda **_k: True,
        bridge=SimpleNamespace(push_log=lambda _evt: None),
    )
    events_mod._finish_task_done_dispatch(
        {}, ctx, task_id="evo1", worker_id=0, task=task,
        final_task_result=written, task_done_event=done,
    )

    assert not _mailbox_path(tmp_path, "evo1").exists()


def test_hard_timeout_evolution_stopped_no_requeue_records_cost(tmp_path, monkeypatch):
    enqueued, emitted, written = _drive_hard_timeout(tmp_path, monkeypatch, evolution_enabled=False)

    # Campaign stopped: the killed evolution task must NOT be re-enqueued.
    assert enqueued == []
    # Terminal status records reconstructed cost/rounds, not zeros.
    assert written.get("status") == "failed"
    assert round(float(written.get("accounted_upper_bound_usd") or 0), 6) == 1.5
    assert int(written.get("total_rounds") or 0) == 3
    # The terminal task_done carries the reconstructed cost for the campaign tally.
    done = [e for e in emitted if e.get("type") == "task_done"]
    assert done and round(float(done[0].get("accounted_upper_bound_usd") or 0), 6) == 1.5
    assert int(done[0].get("total_rounds") or 0) == 3


def test_handle_task_done_reconstructs_cost_from_physical_ledger(tmp_path):
    """A zeroed terminal event and stale result cannot override the ledger."""
    from ouroboros import usage_accounting as accounting
    from supervisor import evolution_lifecycle as lifecycle, queue
    from supervisor import state as supervisor_state
    from supervisor.events import _handle_task_done
    from ouroboros.task_results import STATUS_CANCELLED, write_task_result

    supervisor_state.init(tmp_path)
    queue.init(tmp_path)
    campaign = queue.start_evolution_campaign("Improve", source="test")
    supervisor_state.update_state(lambda live: live.update(
        owner_chat_id=1,
        evolution_mode_enabled=True,
        evolution_owner_stopped=False,
    ))
    transaction = lifecycle.begin_evolution_transaction(
        "evo-zero", cycle=1, campaign=campaign,
    )

    reservation = accounting.reserve_attempt(accounting.AttemptRequest(
        model="openai/gpt-5.2", provider="openai", reservation_usd=2.5,
        drive_root=tmp_path, task_id="evo-zero", root_task_id="evo-zero",
        category="evolution", global_limit_usd=10.0,
    ))
    accounting.mark_dispatched(reservation)
    accounting.settle_attempt(
        reservation, {"prompt_tokens": 100, "completion_tokens": 50},
        cost_usd=2.5, cost_final=True,
    )
    write_task_result(
        tmp_path, "evo-zero", STATUS_CANCELLED,
        cost_usd=99.0, total_rounds=99, prompt_tokens=999, completion_tokens=999,
    )

    broadcast = []
    ctx = SimpleNamespace(
        RUNNING={}, WORKERS={}, DRIVE_ROOT=tmp_path, REPO_DIR=tmp_path,
        load_state=supervisor_state.load_state, save_state=supervisor_state.save_state,
        append_jsonl=supervisor_state.append_jsonl,
        persist_queue_snapshot=lambda reason="": None,
        bridge=SimpleNamespace(push_log=lambda event: broadcast.append(event)),
    )

    _handle_task_done(
        {
            "type": "task_done", "task_id": "evo-zero", "task_type": "evolution",
            "result_status": "cancelled", "cost_usd": 0, "total_rounds": 0,
            "metadata": {"evolution_transaction": transaction},
        },
        ctx,
    )

    # The broadcast task_done carries the reconstructed cost, not the zeroed event value.
    done = [e for e in broadcast if e.get("type") == "task_done"]
    assert done and float(done[0].get("accounted_upper_bound_usd") or 0) == 2.5
    assert int(done[0].get("total_rounds") or 0) == 1
    # Reconstructed cost/rounds are preserved, but a cancelled task is still an
    # axis-level failure. Cost no longer proves evolution success.
    assert int(supervisor_state.load_state().get("evolution_consecutive_failures") or 0) == 1


def test_hard_timeout_evolution_enabled_requeues(tmp_path, monkeypatch):
    enqueued, emitted, written = _drive_hard_timeout(tmp_path, monkeypatch, evolution_enabled=True)

    # Campaign still enabled: the timed-out task is re-enqueued (one retry).
    assert len(enqueued) == 1
    assert enqueued[0].get("type") == "evolution"
    assert enqueued[0].get("_attempt") == 2
    # A retry keeps the live card active, so no terminal task_done is emitted.
    assert [e for e in emitted if e.get("type") == "task_done"] == []
    # The interrupted rollup still records reconstructed cost (not zeros).
    assert round(float(written.get("accounted_upper_bound_usd") or 0), 6) == 1.5
