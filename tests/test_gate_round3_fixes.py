"""GATE ROUND-3 (v6.98.0 phase A) — probe-backed regression tests.

GR3-1  cascade intent is postcondition-owned ALWAYS (replay summary included,
       widened-mid-flight survival, dead-descendants defer);
GR3-2  a live claim is never stolen by age + the minimal pre-write claim fence;
GR3-3  evolution closure gated on completeness (settle-time backstop);
GR3-4  outbox registration failure is a real failure;
GR3-5  the persist→register crash window is closed by ORDERING;
GR3-6  an unowned lifecycle fault rides the normal terminal dispatch;
GR3-7  audit failure is typed UNKNOWN, never clean;
GR3-8  stale sweep failures do not veto a converged tree;
GR3-9  strict registry reads refuse a corrupt projection;
GR3-11 import coherence (the fail_tasks half was retired with the
       function itself in the 7.0 ABI window, owner Q10=A).
"""

from __future__ import annotations

import json
import pathlib
import time
import types

import pytest

from ouroboros import cancel_intents as ci
from ouroboros.task_results import (
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_RUNNING,
    load_task_result,
    write_task_result,
)


class _CaptureQueue:
    def __init__(self):
        self.events = []

    def put(self, evt):
        self.events.append(evt)


class _DeadProc:
    """A worker proc surface that is already dead (no real subprocess)."""

    pid = 0

    def is_alive(self):
        return False

    def terminate(self):
        return None

    def join(self, timeout=None):
        return None


def _trail_rows(drive, log_name="supervisor.jsonl"):
    path = pathlib.Path(drive) / "logs" / log_name
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture()
def qenv(tmp_path, monkeypatch):
    import supervisor.queue as q
    from supervisor import task_lifecycle, workers

    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(q, "PENDING", [])
    monkeypatch.setattr(q, "RUNNING", {}, raising=False)
    monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
    monkeypatch.setattr(workers, "respawn_worker", lambda wid: None, raising=False)
    monkeypatch.setattr(q, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(task_lifecycle, "CANCELLED_ROOT_FENCES", {}, raising=False)
    monkeypatch.setattr(task_lifecycle, "_ACTIVE_CASCADE_FENCES", {}, raising=False)
    return types.SimpleNamespace(q=q, tl=task_lifecycle, workers=workers, drive=tmp_path)


# --------------------------------------------------------------------------
# GR3-1 — cascade intent is postcondition-owned, ALWAYS
# --------------------------------------------------------------------------


def test_cascade_replay_after_crash_delivers_exactly_one_summary(qenv, monkeypatch):
    """GR3-1c: 'killed the children, died before the summary' replays through
    the already-down branch and must still deliver the tree's ONE message —
    owed BEFORE the settle, under the deterministic per-intent delivery id
    (GR4-2). A LATER separate cancel request (new request_id) legitimately
    delivers its own summary; the SAME-intent replay dedup is pinned in
    test_gate_round4_fixes."""
    from supervisor import terminal_delivery as td
    from supervisor import workers

    queue = _CaptureQueue()
    monkeypatch.setattr(workers, "get_event_q", lambda: queue, raising=False)
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *a, **kw: None)
    # The crash shape: every task of the tree is already durably settled, but
    # the root's cascade intent is still open (crash before summary + settle).
    write_task_result(qenv.drive, "cr1", STATUS_CANCELLED, chat_id=6, result="killed")
    minted = ci.request_cancel(
        qenv.drive, "cr1", scope=ci.SCOPE_CASCADE, allow_settled_target=True,
    )

    assert qenv.tl.cancel_task_by_id("cr1", cascade=True) is True

    sends = [e for e in queue.events if e.get("type") == "send_message"]
    assert len(sends) == 1, "the replay/already-down path still owes the summary"
    assert sends[0]["chat_id"] == 6
    assert sends[0]["delivery_id"] == f"cascade:cr1:{minted['request_id']}", (
        "GR4-2: the summary's identity is the INTENT, not the message content"
    )
    assert ci.active_intent(qenv.drive, "cr1") is None, "settled only AFTER the summary was owed"
    assert [r["delivery_id"] for r in td.pending_deliveries(qenv.drive)] == [
        sends[0]["delivery_id"],
    ], "the summary is durably OWED — a crash between summary and settle replays it"

    # Confirmed send → a LATER separate cancel request mints a new intent
    # (new request_id) and delivers its OWN summary (GR4-2 distinctness).
    td.register_delivery(qenv.drive, sends[0]["delivery_id"])
    queue.events.clear()
    second = ci.request_cancel(
        qenv.drive, "cr1", scope=ci.SCOPE_CASCADE, allow_settled_target=True,
    )
    assert second["request_id"] != minted["request_id"]
    assert qenv.tl.cancel_task_by_id("cr1", cascade=True) is True
    later = [e for e in queue.events if e.get("type") == "send_message"]
    assert len(later) == 1
    assert later[0]["delivery_id"] == f"cascade:cr1:{second['request_id']}"


def test_per_task_custody_defers_a_cascade_root_with_dead_descendants(qenv, monkeypatch):
    """GR3-1a: even with EVERY descendant already dead, per-task custody never
    settles a scope=cascade intent — the postcondition owns the settle (and the
    summary that must precede it). The deferring custody's claim is released in
    the same write, so the watchdog replays the cascade to convergence."""
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *a, **kw: None)
    qenv.q.PENDING[:] = [{"id": "dd1", "chat_id": 3}]
    write_task_result(qenv.drive, "dd1", "scheduled", chat_id=3)
    ci.request_cancel(qenv.drive, "dd1", scope=ci.SCOPE_CASCADE)

    outcome = qenv.tl.cancel_task_custody("dd1")

    assert outcome == qenv.tl.CANCEL_CANCELLED
    assert load_task_result(qenv.drive, "dd1")["status"] == STATUS_CANCELLED
    row = ci.active_intent(qenv.drive, "dd1")
    assert row is not None and row["scope"] == ci.SCOPE_CASCADE, (
        "the cascade intent is kept for the postcondition"
    )
    assert row["state"] == ci.INTENT_REQUESTED, "the deferring claim was auto-released"

    # The watchdog replays it as a CASCADE, whose postcondition settles it.
    monkeypatch.setattr(
        "supervisor.terminal_delivery.deliver_unreviewed_salvage",
        lambda *a, **kw: True,
    )
    swept = qenv.tl.sweep_cancel_intents(now=time.time() + 30)
    assert swept == {"dd1": qenv.tl.CANCEL_CANCELLED}
    assert ci.active_intent(qenv.drive, "dd1") is None


def test_widened_mid_flight_intent_survives_a_stale_claim_settle(tmp_path):
    """GR3-1b: the scope is re-read ATOMICALLY at settle time — a claim snapshot
    taken while the intent was still scope=single cannot settle the row after a
    cascade ingress widened it; the refused claimant's claim is auto-released
    so the watchdog can replay the cascade."""
    ci.request_cancel(tmp_path, "w1")
    claim = ci.claim_intent(tmp_path, "w1", owner="pending_drop")
    assert str(claim.get("scope") or "") != ci.SCOPE_CASCADE  # the stale snapshot
    assert ci.mark_intent_scope(tmp_path, "w1", ci.SCOPE_CASCADE) is True

    settled = ci.settle_intent(
        tmp_path, "w1", outcome="cancelled",
        expected_generation=claim.get("generation"),
        request_id=str(claim.get("request_id") or ""),
    )

    assert settled is None
    row = ci.active_intent(tmp_path, "w1")
    assert row is not None and row["scope"] == ci.SCOPE_CASCADE
    assert row["state"] == ci.INTENT_REQUESTED, "claim auto-released for the watchdog"
    assert any(r.get("event") == "cascade_settle_deferred" for r in _trail_rows(tmp_path))

    # Only the cascade postcondition settles it, with the explicit override.
    fresh = ci.active_intent(tmp_path, "w1")
    assert ci.settle_intent(
        tmp_path, "w1", outcome="cancelled",
        expected_generation=fresh.get("generation"),
        request_id=str(fresh.get("request_id") or ""),
        allow_cascade_scope=True,
    ) is not None
    assert ci.active_intent(tmp_path, "w1") is None


# --------------------------------------------------------------------------
# GR3-2 — minimal write fence on the custody kill path
# --------------------------------------------------------------------------


def test_lost_claim_before_the_terminal_write_aborts_publication(qenv):
    """GR3-2 write fence: the claim is re-verified between the kill/join window
    and the durable terminal write; a stolen claim aborts (CANCEL_FAILED,
    custody restored, no terminal write) instead of publishing over the new
    owner's teardown."""
    stolen = {"done": False}
    store = qenv.drive / "state" / "cancel_intents.json"

    class _StealingProc(_DeadProc):
        def join(self, timeout=None):
            if not stolen["done"]:
                data = json.loads(store.read_text(encoding="utf-8"))
                row = data["intents"]["steal1"]
                row["generation"] = int(row["generation"]) + 1
                row["claim_pid"] = 2 ** 22
                row["claim_owner"] = "other-custody"
                data["intents"]["steal1"] = row
                store.write_text(json.dumps(data), encoding="utf-8")
                stolen["done"] = True

    task = {"id": "steal1", "chat_id": 2}
    worker = types.SimpleNamespace(
        wid=0, proc=_StealingProc(), busy_task_id="steal1", reaping=False,
    )
    qenv.workers.WORKERS[0] = worker
    qenv.q.RUNNING["steal1"] = {"task": task, "worker_id": 0}
    write_task_result(qenv.drive, "steal1", STATUS_RUNNING, result="working")
    ci.request_cancel(qenv.drive, "steal1", reason="stop")

    outcome = qenv.tl.cancel_task_custody("steal1")

    assert outcome == qenv.tl.CANCEL_FAILED
    assert load_task_result(qenv.drive, "steal1")["status"] == STATUS_RUNNING, (
        "no terminal write over a lost claim"
    )
    assert "steal1" in qenv.q.RUNNING, "custody restored"
    assert worker.reaping is False


# --------------------------------------------------------------------------
# GR3-3 — evolution closure gated on completeness
# --------------------------------------------------------------------------


def _stop_outcomes(*, failed=()):
    return {
        "cancelled": [], "already_settled": [], "not_found": [],
        "failed": list(failed), "intent_write_failed": [],
    }


def test_incomplete_evolution_stop_leaves_the_campaign_open_until_settle(tmp_path, monkeypatch):
    import supervisor.queue as q
    import supervisor.state as state
    from supervisor import events as events_mod
    from supervisor import evolution_lifecycle as el

    state.init(tmp_path)
    q.init(tmp_path)
    assert el.start_evolution_campaign("Improve", source="test").get("status") == "active"
    state.update_state(lambda live: live.update(
        owner_chat_id=7, evolution_mode_enabled=True, evolution_owner_stopped=False,
    ))
    sent: list = []
    ctx = types.SimpleNamespace(
        sort_pending=lambda: None,
        persist_queue_snapshot=lambda reason="": None,
        send_with_budget=lambda chat_id, text, **kw: sent.append(text),
        load_state=state.load_state,
    )
    monkeypatch.setattr(
        q, "stop_evolution_tasks", lambda reason: _stop_outcomes(failed=["evo-live"]),
    )

    events_mod._handle_toggle_evolution({"enabled": False}, ctx)

    assert bool(state.load_state().get("evolution_owner_stopped")) is True
    assert el._read_evolution_campaign().get("status") == "active", (
        "an INCOMPLETE stop must not close the campaign"
    )
    assert any("INCOMPLETE" in text for text in sent)

    # The settle-time backstop closes it once the live task settles — invoked
    # from the evolution terminal path even for a rejected terminal (finally).
    events_mod._handle_evolution_task_done(
        ctx, evt={}, task_id="evo-live", task={},
        task_done_event={"status": "failed"}, outcome_axes={}, cost=None, rounds=0,
    )
    closed = el._read_evolution_campaign()
    assert closed.get("status") == "stopped"
    assert "owner stop" in str(closed.get("completion_reason") or "")


def test_owner_evolution_stop_gates_closure_on_completeness(tmp_path, monkeypatch):
    import server as srv
    import supervisor.queue as q
    import supervisor.state as state
    from supervisor import evolution_lifecycle as el

    state.init(tmp_path)
    q.init(tmp_path)
    assert el.start_evolution_campaign("Improve", source="test").get("status") == "active"
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        sort_pending=lambda: None,
        persist_queue_snapshot=lambda reason="": None,
        send_with_budget=lambda chat_id, text, **kw: None,
    )
    monkeypatch.setattr(
        q, "stop_evolution_tasks", lambda reason: _stop_outcomes(failed=["evo-live"]),
    )
    wording = srv._owner_evolution_stop(ctx, 7)
    assert "INCOMPLETE" in wording
    assert el._read_evolution_campaign().get("status") == "active", "left open"

    # A later CLEAN pass closes the campaign terminally.
    monkeypatch.setattr(q, "stop_evolution_tasks", lambda reason: _stop_outcomes())
    wording = srv._owner_evolution_stop(ctx, 7)
    assert "INCOMPLETE" not in wording
    assert el._read_evolution_campaign().get("status") == "stopped"


# --------------------------------------------------------------------------
# GR3-4 — outbox registration failure is a real failure
# --------------------------------------------------------------------------


def test_failed_owed_registration_keeps_the_cancel_intent_open(qenv, monkeypatch):
    """GR3-4 cancel path: an answer that could not be durably owed must NOT be
    settled over — the intent stays open (claim released) for the watchdog,
    whose retry finds the settled result and re-delivers on the miss lane."""
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *a, **kw: None)
    monkeypatch.setattr(
        "supervisor.terminal_delivery.register_pending_delivery",
        lambda *a, **kw: False,
    )
    task = {"id": "owed1", "chat_id": 4}
    worker = types.SimpleNamespace(wid=0, proc=_DeadProc(), busy_task_id="owed1", reaping=False)
    qenv.workers.WORKERS[0] = worker
    qenv.q.RUNNING["owed1"] = {"task": task, "worker_id": 0}
    write_task_result(qenv.drive, "owed1", STATUS_RUNNING, result="working")
    ci.request_cancel(qenv.drive, "owed1", reason="stop")

    outcome = qenv.tl.cancel_task_custody("owed1")

    assert outcome == qenv.tl.CANCEL_CANCELLED, "the teardown itself is not gated"
    assert load_task_result(qenv.drive, "owed1")["status"] == STATUS_CANCELLED
    row = ci.active_intent(qenv.drive, "owed1")
    assert row is not None and row["state"] == ci.INTENT_REQUESTED, (
        "the intent stays OPEN (claim released) over the unowed answer"
    )


def test_corrupt_delivery_registry_refuses_mutation_and_discloses(tmp_path):
    """GR3-9 + GR3-4 normal path: a malformed terminal_deliveries.json refuses
    the mutation (never a {}-collapse overwrite that loses every owed answer)
    and the failed owed-registration is a typed, durable disclosure."""
    from supervisor import terminal_delivery as td

    store = tmp_path / "state" / "terminal_deliveries.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("[1, 2, 3]", encoding="utf-8")  # non-object JSON

    event = {"type": "send_message", "chat_id": 1, "task_id": "c9",
             "text": "answer", "delivery_id": td.delivery_id_for("c9", "answer")}
    assert td.register_pending_delivery(tmp_path, event) is False
    assert store.read_text(encoding="utf-8") == "[1, 2, 3]", "no overwrite"
    rows = _trail_rows(tmp_path, "events.jsonl")
    assert any(r.get("type") == "terminal_delivery_registry_corrupt" for r in rows)
    assert any(r.get("type") == "terminal_delivery_unregistered" for r in rows)
    # The delivered-side registration fails open toward delivery — still no overwrite.
    assert td.register_delivery(tmp_path, "final:x:abc") is True
    assert store.read_text(encoding="utf-8") == "[1, 2, 3]"


def test_corrupt_intent_projection_refuses_the_ingress_loudly(tmp_path):
    """GR3-9: a corrupt cancel_intents.json refuses the mutation with a typed
    error (the ingress fails closed) instead of silently dropping every active
    intent in one overwrite. The forensic trail records the refusal."""
    ci.request_cancel(tmp_path, "keepme", reason="live intent")
    store = tmp_path / "state" / "cancel_intents.json"
    store.write_text('"not an object"', encoding="utf-8")

    with pytest.raises(ci.CancelIntentProjectionCorrupt):
        ci.request_cancel(tmp_path, "newone", reason="must refuse")

    assert store.read_text(encoding="utf-8") == '"not an object"', "no overwrite"
    assert any(r.get("event") == "projection_corrupt_refused" for r in _trail_rows(tmp_path))


# --------------------------------------------------------------------------
# GR3-5 — the persist→register crash window is closed by ordering
# --------------------------------------------------------------------------


def test_final_answer_is_owed_before_the_durable_result_write(tmp_path):
    """GR3-5: the owed row embeds the payload, so registered-then-crashed leaves
    a row boot replay delivers (projection-over-replay; no boot scan of
    task_results). The ordering is pinned at the pipeline source."""
    from ouroboros.task_finalization import register_final_answer_owed
    from supervisor import terminal_delivery as td

    task = {"id": "ord1", "chat_id": 8}
    send_event = {"type": "send_message", "chat_id": 8, "task_id": "ord1",
                  "text": "the answer"}
    register_final_answer_owed(task, send_event, env_drive_root=tmp_path)
    # Crash HERE — before _store_task_result: the owed row already carries the
    # full payload and the boot/tick replay delivers it exactly once.
    owed = td.pending_deliveries(tmp_path)
    assert [r["task_id"] for r in owed] == ["ord1"]
    assert owed[0]["text"] == "the answer"
    # Replay behavior is covered by the outbox suite; here the ORDERING is
    # pinned at the pipeline source (owed registration before the store):
    source = (
        pathlib.Path(__file__).resolve().parents[1]
        / "ouroboros" / "agent_task_pipeline.py"
    ).read_text(encoding="utf-8")
    register_at = source.index(
        "register_final_answer_owed(task, send_event, env_drive_root=env.drive_root)"
    )
    store_call_at = source.index("        _store_task_result(")
    assert register_at < store_call_at, (
        "GR3-5: the owed registration must precede the durable result write"
    )


def test_reaper_already_terminal_recovery_registers_the_answer_owed(qenv, monkeypatch):
    """GR3-5 reaper half: a worker that self-finalized and died before delivery
    routes its recovery through the same owed-registration delivery seam —
    owed BEFORE enqueued, deduped by the shared delivery id."""
    import ouroboros.tools.services as services_mod
    from supervisor import task_reaper
    from supervisor import terminal_delivery as td
    from supervisor import workers

    queue = _CaptureQueue()
    monkeypatch.setattr(workers, "get_event_q", lambda: queue, raising=False)
    monkeypatch.setattr(services_mod, "archive_task_service_logs", lambda *a, **k: None)
    qenv.workers.WORKERS[0] = types.SimpleNamespace(
        wid=0, proc=_DeadProc(), busy_task_id=None, reaping=True,
    )
    write_task_result(
        qenv.drive, "sf1", STATUS_COMPLETED, chat_id=5, result="the finished answer",
    )

    task_reaper.reap_timed_out_task({
        "worker_id": 0, "proc": None, "task_id": "sf1",
        "task": {"id": "sf1", "chat_id": 5}, "task_type": "task",
        "terminal_reason": "idle_timeout", "attempt": 1,
        "owner_chat_id": 0, "runtime_sec": 10.0, "will_retry": False,
    })

    did = td.delivery_id_for("sf1", "the finished answer")
    assert [r["delivery_id"] for r in td.pending_deliveries(qenv.drive)] == [did], (
        "the completed answer is durably OWED, not just task_done'd"
    )
    sends = [e for e in queue.events if e.get("type") == "send_message"]
    assert sends and sends[0]["delivery_id"] == did
    assert [e for e in queue.events if e.get("type") == "task_done"], (
        "the idempotent task_done still resolves the card"
    )


# --------------------------------------------------------------------------
# GR3-6 — unowned lifecycle fault uses normal terminal dispatch
# --------------------------------------------------------------------------


def test_lifecycle_fault_produces_a_terminal_frame_and_clears_fences(tmp_path, monkeypatch):
    import supervisor.queue as q
    from ouroboros.utils import append_jsonl
    from supervisor.events import _handle_task_done

    frames: list = []
    cleared: list = []
    monkeypatch.setattr(q, "clear_acceptance_fence_for_root", lambda tid: cleared.append(tid))
    running = {"f1": {"task": {"id": "f1", "chat_id": 12}}}
    slot = types.SimpleNamespace(busy_task_id="f1", reaping=False)
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING=running, WORKERS={3: slot},
        append_jsonl=append_jsonl,
        persist_queue_snapshot=lambda reason="": None,
        bridge=types.SimpleNamespace(push_log=lambda evt: frames.append(evt)),
    )
    write_task_result(tmp_path, "f1", STATUS_RUNNING, result="working")

    _handle_task_done({"task_id": "f1", "status": "running", "worker_id": 3}, ctx)

    assert load_task_result(tmp_path, "f1")["status"] == "failed"
    (frame,) = [e for e in frames if e.get("type") == "task_done"]
    assert frame["task_id"] == "f1" and frame["status"] == "failed"
    assert frame["reason_code"] == "task_done_lifecycle_fault"
    assert frame["chat_id"] == 12
    assert cleared == ["f1"], "acceptance fence cleared through the normal seam"
    assert "f1" not in running and slot.busy_task_id is None


def test_lifecycle_fault_persistence_failure_retains_ownership(tmp_path, monkeypatch):
    import ouroboros.task_results as tr
    from ouroboros.utils import append_jsonl
    from supervisor.events import _handle_task_done

    def _boom(*a, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(tr, "write_task_result", _boom)
    running = {"f2": {"task": {"id": "f2"}}}
    slot = types.SimpleNamespace(busy_task_id="f2", reaping=False)
    frames: list = []
    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING=running, WORKERS={0: slot},
        append_jsonl=append_jsonl,
        persist_queue_snapshot=lambda reason="": None,
        bridge=types.SimpleNamespace(push_log=lambda evt: frames.append(evt)),
    )

    _handle_task_done({"task_id": "f2", "status": "running", "worker_id": 0}, ctx)

    assert "f2" in running and slot.busy_task_id == "f2", (
        "GR3-6: failed persistence RETAINS lifecycle ownership (no slot release)"
    )
    assert [e for e in frames if e.get("type") == "task_done"] == []


# --------------------------------------------------------------------------
# GR3-7 — audit failure is typed UNKNOWN, never clean
# --------------------------------------------------------------------------


def test_delegated_audit_failure_is_typed_unknown_never_clean(qenv, monkeypatch):
    import ouroboros.delegate_custody as dc
    from supervisor.terminal_delivery import (
        RUN_STATE_UNKNOWN_PREFIX,
        build_unreviewed_salvage_event,
    )

    monkeypatch.setattr(dc, "reconcile_task_runs", lambda *a, **kw: None)

    def _boom(*a, **kw):
        raise RuntimeError("custody rows unreadable")

    monkeypatch.setattr(dc, "open_runs", _boom)
    still = qenv.tl._reconcile_delegated_runs_on_kill(qenv.q, "au1")
    assert still == [f"{RUN_STATE_UNKNOWN_PREFIX}:audit_failed"]
    rows = [
        r for r in _trail_rows(qenv.drive, "events.jsonl")
        if r.get("type") == "delegated_runs_unreconciled"
    ]
    assert rows and rows[0]["flavor"] == "audit_failed" and rows[0]["run_ids"] == still

    # The marker rides the delivery note as an honest UNKNOWN sentence.
    event = build_unreviewed_salvage_event(
        qenv.drive, {"id": "au1", "chat_id": 3}, "au1",
        outcome="cancelled", salvaged_text="partial", unreconciled_runs=still,
        settled_status="cancelled",
    )
    assert "DELEGATED RUN STATE UNKNOWN" in event["text"]

    # The pending-invocation audit failure surfaces the same way.
    monkeypatch.setattr(dc, "open_runs", lambda *a, **kw: [])

    def _boom2(*a, **kw):
        raise RuntimeError("pending rows unreadable")

    monkeypatch.setattr(dc, "pending_invocations", _boom2)
    still2 = qenv.tl._reconcile_delegated_runs_on_kill(qenv.q, "au2")
    assert still2 == [f"{RUN_STATE_UNKNOWN_PREFIX}:pending_invocation_audit_failed"]


# --------------------------------------------------------------------------
# GR3-8 — stale sweep failures do not veto a converged tree
# --------------------------------------------------------------------------


def test_stale_sweep_failures_do_not_veto_a_converged_tree(qenv, monkeypatch):
    """GR3-8: a child whose custody REFUSED in sweep N (a concurrent cascade
    claimed it) but which that cascade settled afterwards must not turn a
    settled tree into a skipped summary and a 503 — the postcondition re-judges
    each failed id against the CURRENT durable status."""
    delivered: list = []
    monkeypatch.setattr(
        "supervisor.terminal_delivery.deliver_unreviewed_salvage",
        lambda drive, task, tid, **kw: delivered.append(tid) or True,
    )
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *a, **kw: None)
    qenv.q.PENDING[:] = [
        {"id": "st-root", "chat_id": 5},
        {"id": "st-kid", "chat_id": 5, "parent_task_id": "st-root",
         "root_task_id": "st-root"},
    ]
    write_task_result(qenv.drive, "st-root", "scheduled", chat_id=5)
    write_task_result(qenv.drive, "st-kid", "scheduled")
    ci.request_cancel(qenv.drive, "st-root", scope=ci.SCOPE_CASCADE)

    real_custody = qenv.tl.cancel_task_custody

    def _concurrently_settled(tid, **kw):
        if tid == "st-kid":
            # The CONCURRENT cascade owns this child: our custody is refused,
            # but the child settles durably and leaves the live queue.
            write_task_result(qenv.drive, tid, STATUS_CANCELLED, result="settled elsewhere")
            qenv.q.PENDING[:] = [t for t in qenv.q.PENDING if t.get("id") != tid]
            return qenv.q.CANCEL_FAILED
        return real_custody(tid, **kw)

    monkeypatch.setattr(qenv.q, "cancel_task_custody", _concurrently_settled)

    assert qenv.tl.cancel_task_by_id("st-root", cascade=True) is True
    assert delivered == ["st-root"], "one summary; no 503 over a settled tree"
    assert ci.active_intent(qenv.drive, "st-root") is None


# --------------------------------------------------------------------------
# GR3-11a — import coherence
# --------------------------------------------------------------------------


def test_task_subtree_is_live_is_re_exported_through_the_queue():
    import supervisor.queue as q
    from supervisor import task_lifecycle as tl

    assert q.task_subtree_is_live is tl.task_subtree_is_live
