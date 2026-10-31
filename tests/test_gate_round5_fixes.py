"""GATE ROUND-5 (v6.98.0 phase A) — owner-audit-driven micro-fix regression tests.

GR5-1  a FAILED ``/evolve start`` / ``toggle_evolution(True)`` restores the
       CAPTURED prior ``evolution_owner_stopped`` value (both ingresses) —
       the GR4-6 pre-mint clear must not outlive a start that never happened,
       or the post-task promotion pipeline can re-arm evolution the owner
       believes is off;
GR5-2  the timeout reaper reconciles the killed task's open DELEGATED runs
       (same custody seam as the cancel kill path) and discloses what stayed
       open on the reap outcome;
GR5-3  the fast already-settled and finalize-on-miss cancel paths run the
       same open-run audit and thread ``unreconciled_runs`` into the
       miss-lane delivery — a cancel over a dead task with live delegated
       runs must not read as a clean completion;
GR5-4  cascade digest MEMBERSHIP comes from the durable tree merged with the
       sweep outcomes — a watchdog replay after the children terminalized
       (empty outcomes) still lists them;
GR5-5  project deletion cascades only the lineage ROOTS of the live set —
       descendants fall with their trees (one cascade, one summary per
       tree); an orphan child without a live root still gets its own;
GR5-6  nested registry strictness, cheap form: a present-but-non-dict
       ``pending`` / ``intents`` under a valid top-level dict refuses the
       mutation exactly like top-level corruption (no overwrite), and the
       read paths distinguish "file absent" (empty, silent) from
       "unreadable/malformed" (loud typed disclosure, then empty).
"""

from __future__ import annotations

import json
import logging
import pathlib
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

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class _CaptureQueue:
    def __init__(self):
        self.events = []

    def put(self, evt):
        self.events.append(evt)


def _event_rows(drive, log_name="events.jsonl"):
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


def _patch_open_delegated_run(monkeypatch, task_id: str, run_id: str = "run-open"):
    """Custody rows say one delegated run is still open for ``task_id``."""
    from ouroboros.delegate_custody import RunCustody

    calls: list = []
    monkeypatch.setattr(
        "ouroboros.delegate_custody.reconcile_task_runs",
        lambda root, tid, **kw: calls.append(str(tid)) or [],
    )
    # `state` is part of these projections' contract: the terminal audit shares
    # one custody replay across them, so a double must accept the keyword. The
    # row is a REAL RunCustody: the audit reads its owner kind (`review_owned`)
    # as well as its ids, and a partial double answers for neither.
    monkeypatch.setattr(
        "ouroboros.delegate_custody.open_runs",
        lambda root, state=None: [RunCustody(task_id=task_id, run_id=run_id)],
    )
    monkeypatch.setattr(
        "ouroboros.delegate_custody.pending_invocations", lambda root, rows=None: [],
    )
    return calls


# --------------------------------------------------------------------------
# GR5-1 — a failed evolution start restores the captured owner-stop flag
# --------------------------------------------------------------------------


def _toggle_ctx(state, sent):
    return types.SimpleNamespace(
        load_state=state.load_state,
        send_with_budget=lambda cid, text, **kw: sent.append(text),
    )


def test_gr5_1_toggle_start_failure_restores_a_prior_owner_stop(tmp_path, monkeypatch):
    """Prior flag True + failed start → the flag is True again afterwards, so
    ``apply_pending_request`` keeps refusing the autonomous re-arm."""
    import supervisor.state as state
    from supervisor import events as events_mod
    from supervisor import evolution_lifecycle as el

    state.init(tmp_path)
    state.save_state({})  # an initialized install: only explicit init creates state (#1307)
    state.update_state(lambda live: live.update(
        owner_chat_id=7, evolution_owner_stopped=True,
    ))
    monkeypatch.setattr(el, "evolution_block_reason", lambda: "")
    monkeypatch.setattr(
        el, "start_evolution_campaign",
        lambda objective, source="": (_ for _ in ()).throw(OSError("campaign store io")),
    )
    sent: list = []

    # An OWNER start reaches the mint (В12: the agent tool is refused outright
    # while the flag stands — pinned in test_post_task_evolution).
    events_mod._handle_toggle_evolution(
        {"enabled": True, "objective": "x", "source": "owner_chat"}, _toggle_ctx(state, sent),
    )

    assert bool(state.load_state().get("evolution_owner_stopped")) is True, (
        "GR5-1: a failed start must restore the captured owner-stop flag"
    )
    assert any("stayed OFF" in text for text in sent)


def test_gr5_1_toggle_start_failure_with_no_prior_stop_stays_false(tmp_path, monkeypatch):
    """Prior flag False + refused start ("Evolution stayed OFF" branch) → the
    flag stays False: an unconditional True would invent an owner stop."""
    import supervisor.state as state
    from supervisor import events as events_mod
    from supervisor import evolution_lifecycle as el

    state.init(tmp_path)
    state.save_state({})  # an initialized install: only explicit init creates state (#1307)
    state.update_state(lambda live: live.update(
        owner_chat_id=7, evolution_owner_stopped=False,
    ))
    monkeypatch.setattr(el, "evolution_block_reason", lambda: "")
    monkeypatch.setattr(el, "start_evolution_campaign", lambda objective, source="": {})
    sent: list = []

    events_mod._handle_toggle_evolution({"enabled": True, "objective": "x"}, _toggle_ctx(state, sent))

    assert bool(state.load_state().get("evolution_owner_stopped")) is False, (
        "GR5-1: no owner stop preceded — the restore must not fabricate one"
    )
    assert any("stayed OFF" in text for text in sent)


def test_gr5_1_server_evolve_start_failure_restores_the_captured_flag():
    """The owner-chat `/evolve` ingress runs the shared owner start transaction; its
    capture -> clear -> (failure) restore ordering is pinned at the source, and the
    restore puts back an UNKNOWN prior as unknown (#1307)."""
    import inspect

    from supervisor.events_runtime_controls import owner_evolution_start

    src = (REPO_ROOT / "server.py").read_text(encoding="utf-8")
    assert 'owner_evolution_start(objective, source="owner_chat")' in src
    body = inspect.getsource(owner_evolution_start)
    capture_at = body.index('prior["known"], prior["value"] = control_value(live, "evolution_owner_stopped")')
    clear_at = body.index("update_state(_clear_owner_stop")
    start_at = body.index("start_evolution_campaign(")
    restore_at = body.index("update_state(_restore)")
    stayed_off_at = body.index("Evolution stayed OFF: campaign state could not be created.")
    assert capture_at < clear_at < start_at < restore_at < stayed_off_at
    assert 'mark_unconfirmed(live, "evolution_owner_stopped")' in body


def test_gr5_2_reap_reconciles_delegated_runs_and_discloses(qenv, monkeypatch):
    import ouroboros.task_results as task_results_mod
    from supervisor import task_reaper, workers

    queue = _CaptureQueue()
    monkeypatch.setattr(workers, "get_event_q", lambda *, create=True: queue, raising=False)
    monkeypatch.setattr(task_reaper, "send_with_budget", lambda *a, **kw: None)
    calls = _patch_open_delegated_run(monkeypatch, "rp5")
    disclosure_writes: list = []
    real_write = task_results_mod.write_task_result

    def _counting_write(root, tid, status, **kwargs):
        if str(tid) == "rp5" and (
            "delegated_runs_unreconciled" in kwargs
            or "delegate_terminal_reconciliation" in kwargs
        ):
            disclosure_writes.append(dict(kwargs))
        return real_write(root, tid, status, **kwargs)

    monkeypatch.setattr(task_results_mod, "write_task_result", _counting_write)

    task_reaper.reap_timed_out_task({
        "worker_id": 1, "proc": None, "task_id": "rp5",
        "task": {"id": "rp5", "chat_id": 3}, "task_type": "chat",
        "terminal_reason": "deadline", "attempt": 3, "owner_chat_id": 3,
        "runtime_sec": 100.0, "deadline_reached": True, "ceiling_reached": False,
    })

    assert calls == ["rp5"], (
        "GR5-2: the reaper must call the delegate-custody reconcile seam"
    )
    row = load_task_result(qenv.drive, "rp5")
    assert row["status"] == "failed"
    assert row["delegated_runs_unreconciled"] == ["run-open"], (
        "GR5-2: the reap outcome discloses the still-open runs"
    )
    # R2: the audit envelope rides the SAME single terminal write as the flat
    # list, stamped with the reaper's own trigger.
    envelope = row["delegate_terminal_reconciliation"]
    assert envelope["trigger"] == "reaper_deadline"
    assert envelope["open_run_ids"] == ["run-open"]
    (only_write,) = disclosure_writes
    assert only_write["delegated_runs_unreconciled"] == ["run-open"]
    assert only_write["delegate_terminal_reconciliation"]["trigger"] == "reaper_deadline"
    assert any(
        r.get("type") == "delegated_runs_unreconciled" and r.get("task_id") == "rp5"
        for r in _event_rows(qenv.drive)
    ), "the same typed event vocabulary as the cancel kill path"
    sends = [e for e in queue.events if e.get("type") == "send_message"]
    assert sends and "DELEGATED RUNS NOT RECONCILED" in sends[0]["text"]
    assert "run-open" in sends[0]["text"]


# --------------------------------------------------------------------------
# GR5-3 — miss/fast cancel paths audit + disclose delegated runs
# --------------------------------------------------------------------------


def test_gr5_3_already_settled_fast_path_discloses_open_delegated_runs(qenv, monkeypatch):
    """A cancel of a task that already settled (leaving open delegated runs)
    must not deliver a clean completion: the fast path audits custody and the
    owner message carries the same disclosure as the kill path."""
    from supervisor import workers

    queue = _CaptureQueue()
    monkeypatch.setattr(workers, "get_event_q", lambda *, create=True: queue, raising=False)
    _patch_open_delegated_run(monkeypatch, "fs5")
    write_task_result(qenv.drive, "fs5", STATUS_RUNNING, chat_id=4, result="working")
    ci.request_cancel(qenv.drive, "fs5", reason="stop")
    # An earlier actor settled the task before this custody arrived.
    write_task_result(qenv.drive, "fs5", STATUS_CANCELLED, chat_id=4, result="killed elsewhere")

    assert qenv.tl.cancel_task_custody("fs5") == qenv.tl.CANCEL_ALREADY_SETTLED

    sends = [e for e in queue.events if e.get("type") == "send_message"]
    assert sends, "the settled answer is still delivered on the miss lane"
    assert "DELEGATED RUNS NOT RECONCILED" in sends[0]["text"], (
        "GR5-3: the fast already-settled path discloses the open delegated runs"
    )
    assert "run-open" in sends[0]["text"]
    assert any(
        r.get("type") == "delegated_runs_unreconciled" and r.get("task_id") == "fs5"
        for r in _event_rows(qenv.drive)
    )


def test_gr5_3_finalize_on_miss_discloses_open_delegated_runs(qenv, monkeypatch):
    """The finalize-on-miss lane (neither queued nor running) runs the same
    audit and threads the disclosure into its delivery."""
    from supervisor import workers

    queue = _CaptureQueue()
    monkeypatch.setattr(workers, "get_event_q", lambda *, create=True: queue, raising=False)
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *a, **kw: None)
    _patch_open_delegated_run(monkeypatch, "ml5")
    write_task_result(qenv.drive, "ml5", STATUS_RUNNING, chat_id=5, result="was working")
    ci.request_cancel(qenv.drive, "ml5", reason="stop")

    assert qenv.tl.cancel_task_custody("ml5") == qenv.tl.CANCEL_CANCELLED

    assert load_task_result(qenv.drive, "ml5")["status"] == STATUS_CANCELLED
    sends = [e for e in queue.events if e.get("type") == "send_message"]
    assert sends and "DELEGATED RUNS NOT RECONCILED" in sends[0]["text"], (
        "GR5-3: the finalize-on-miss delivery carries the disclosure"
    )
    assert "run-open" in sends[0]["text"]


# --------------------------------------------------------------------------
# GR5-4 — replay-shape digest membership comes from the durable tree
# --------------------------------------------------------------------------


def test_gr5_4_replay_digest_lists_children_from_the_durable_tree(tmp_path, monkeypatch):
    """A watchdog replay after the children already terminalized arrives with
    EMPTY sweep outcomes — the digest must still list every durable
    descendant with its CURRENT status (GR4-4 rule per line)."""
    from supervisor import terminal_delivery as td
    from supervisor import workers

    queue = _CaptureQueue()
    monkeypatch.setattr(workers, "get_event_q", lambda *, create=True: queue, raising=False)
    write_task_result(tmp_path, "r5", STATUS_CANCELLED, chat_id=9, result="root down")
    write_task_result(
        tmp_path, "c5a", STATUS_CANCELLED, result="child down",
        parent_task_id="r5", root_task_id="r5", delegation_role="subagent",
    )
    write_task_result(
        tmp_path, "c5b", STATUS_COMPLETED, result="child done",
        parent_task_id="c5a", root_task_id="r5", delegation_role="subagent",
    )

    owed = td.deliver_cascade_summary(tmp_path, "r5", {"id": "r5", "chat_id": 9}, {})

    assert owed is True
    (event,) = [e for e in queue.events if e.get("type") == "send_message"]
    assert "2 descendant task(s) were settled with it" in event["text"]
    outcomes = {
        row["task_id"]: row["outcome"]
        for row in load_task_result(tmp_path, "r5")["cancel_receipt"]["children"]
    }
    assert outcomes["c5a"] == "cancelled", (
        "GR5-4: the replay digest enumerates the durable tree, not only the "
        "(empty) sweep outcomes"
    )
    assert outcomes["c5b"] == "completed"


def test_gr5_4_sweep_outcome_still_wins_for_a_child_with_no_durable_row(tmp_path, monkeypatch):
    from supervisor import terminal_delivery as td
    from supervisor import workers

    queue = _CaptureQueue()
    monkeypatch.setattr(workers, "get_event_q", lambda *, create=True: queue, raising=False)
    write_task_result(tmp_path, "r5w", STATUS_CANCELLED, chat_id=9, result="root down")

    owed = td.deliver_cascade_summary(
        tmp_path, "r5w", {"id": "r5w", "chat_id": 9}, {"c5w": "failed"},
    )

    assert owed is True
    (event,) = [e for e in queue.events if e.get("type") == "send_message"]
    outcomes = {
        row["task_id"]: row["outcome"]
        for row in load_task_result(tmp_path, "r5w")["cancel_receipt"]["children"]
    }
    assert outcomes["c5w"] == "failed", (
        "a child with no durable row yet keeps the sweep's honest outcome"
    )


# --------------------------------------------------------------------------
# GR5-5 — project deletion cascades lineage roots only
# --------------------------------------------------------------------------


def test_gr5_5_live_project_ids_roots_only_filters_covered_descendants(tmp_path, monkeypatch):
    import supervisor.queue as q
    from supervisor import queue_transitions as qt

    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(q, "PENDING", [
        {"id": "root-a", "root_task_id": "root-a", "project_id": "p5"},
        {"id": "child-a", "parent_task_id": "root-a", "root_task_id": "root-a"},
        # Orphan: its recorded root/parent is NOT live any more.
        {"id": "orphan-b", "parent_task_id": "gone-root", "root_task_id": "gone-root",
         "project_id": "p5"},
    ])
    monkeypatch.setattr(q, "RUNNING", {
        "grand-a": {"task": {
            "id": "grand-a", "parent_task_id": "child-a", "root_task_id": "root-a",
        }},
    }, raising=False)

    assert set(qt._live_project_task_ids(tmp_path, "p5")) == {
        "root-a", "child-a", "grand-a", "orphan-b",
    }
    assert set(qt._live_project_task_ids(tmp_path, "p5", roots_only=True)) == {
        "root-a", "orphan-b",
    }, (
        "GR5-5: descendants covered by a live root fall with its cascade; an "
        "orphan without a live ancestor keeps its own"
    )


def test_gr5_5_mid_tree_live_ancestor_covers_its_branch(tmp_path, monkeypatch):
    """The recorded root already settled, but a mid-tree parent is live: the
    parent-chain walk must still cover the grandchild (one cascade for the
    live branch, rooted at the mid-tree ancestor)."""
    import supervisor.queue as q
    from supervisor import queue_transitions as qt

    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(q, "PENDING", [
        {"id": "mid", "parent_task_id": "settled-root", "root_task_id": "settled-root",
         "project_id": "p5m"},
        {"id": "leaf", "parent_task_id": "mid", "root_task_id": "settled-root"},
    ])
    monkeypatch.setattr(q, "RUNNING", {}, raising=False)

    assert set(qt._live_project_task_ids(tmp_path, "p5m", roots_only=True)) == {"mid"}


# --------------------------------------------------------------------------
# GR5-6 — nested registry strictness (cheap form)
# --------------------------------------------------------------------------


def test_gr5_6_nested_corrupt_pending_refuses_the_mutation(tmp_path):
    """A valid top-level dict whose ``pending`` is not an object used to be
    coerced to {} — the next mutation overwrote every owed row silently. The
    mutators now refuse it exactly like top-level corruption: no overwrite,
    typed disclosure."""
    from supervisor import terminal_delivery as td

    store = tmp_path / "state" / "terminal_deliveries.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(
        json.dumps({"schema_version": 2, "delivered": ["final:old:aa"], "pending": [1, 2]}),
        encoding="utf-8",
    )
    before = store.read_text(encoding="utf-8")

    assert td.register_pending_delivery(tmp_path, {
        "type": "send_message", "chat_id": 1, "task_id": "n6",
        "text": "x", "delivery_id": "final:n6:abc",
    }) is False
    assert store.read_text(encoding="utf-8") == before, "no overwrite"
    assert td.register_delivery(tmp_path, "final:n6:abc") is True, (
        "register_delivery keeps its fail-open contract on a refused mutation"
    )
    assert store.read_text(encoding="utf-8") == before, "no overwrite"
    corrupt_events = [
        r for r in _event_rows(tmp_path)
        if r.get("type") == "terminal_delivery_registry_corrupt"
    ]
    assert len(corrupt_events) >= 2, "each refused mutation is a loud typed event"


def test_gr5_6_nested_corrupt_intents_refuses_the_mint(tmp_path):
    path = tmp_path / "state" / "cancel_intents.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "intents": ["not-a-dict"]}), encoding="utf-8",
    )
    before = path.read_text(encoding="utf-8")

    with pytest.raises(ci.CancelIntentProjectionCorrupt):
        ci.request_cancel(tmp_path, "n6i", reason="stop")

    assert path.read_text(encoding="utf-8") == before, "no overwrite"
    rows = _event_rows(tmp_path, log_name="supervisor.jsonl")
    assert any(
        r.get("event") == "projection_corrupt_refused" and r.get("task_id") == "n6i"
        for r in rows
    )


def test_gr5_6_corrupt_intent_projection_read_is_loud_and_empty(tmp_path, caplog):
    path = tmp_path / "state" / "cancel_intents.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger="ouroboros.cancel_intents"):
        assert ci.active_intents(tmp_path) == {}
    assert any(
        "unreadable/malformed" in record.message for record in caplog.records
    ), "GR5-6: a malformed projection read is a loud log.error, never silent"

    # The watchdog's enforcement read additionally emits the typed forensic row.
    assert ci.active_intents(tmp_path, disclose_corruption=True) == {}
    rows = _event_rows(tmp_path, log_name="supervisor.jsonl")
    assert any(
        r.get("event") == "projection_corrupt_refused" and r.get("op") == "active_intents"
        for r in rows
    ), "enforcement degradation is owner-visible"

    # Nested corruption under a valid top-level dict discloses the same way.
    path.write_text(json.dumps({"intents": ["bad"]}), encoding="utf-8")
    assert ci.active_intents(tmp_path) == {}


def test_gr5_6_corrupt_outbox_read_is_loud_and_empty(tmp_path, caplog):
    from supervisor import terminal_delivery as td

    store = tmp_path / "state" / "terminal_deliveries.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("[1, 2, 3]", encoding="utf-8")

    with caplog.at_level(logging.ERROR, logger="supervisor.terminal_delivery"):
        assert td.pending_deliveries(tmp_path) == []
    assert any(
        "unreadable/malformed" in record.message for record in caplog.records
    )

    # The replay (watchdog) read discloses through the existing typed event.
    assert td.replay_pending_deliveries(tmp_path, event_queue=_CaptureQueue()) == []
    assert any(
        r.get("type") == "terminal_delivery_registry_corrupt"
        and r.get("op") == "pending_deliveries_read"
        for r in _event_rows(tmp_path)
    )


def test_gr5_6_absent_registries_read_empty_and_silent(tmp_path, caplog):
    from supervisor import terminal_delivery as td

    with caplog.at_level(logging.ERROR):
        assert ci.active_intents(tmp_path) == {}
        assert td.pending_deliveries(tmp_path) == []
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], (
        "GR5-6: an ABSENT file is an ordinary empty projection, not corruption"
    )
