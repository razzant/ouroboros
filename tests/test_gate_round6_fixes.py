"""GATE ROUND-6 (v6.98.0 phase A) — final 6-lane-gate regression tests.

GR6-1  CLASS closure: "settled RESULT does not mean dead WORKER". The durable
       terminal result is persisted BEFORE post-task cognition ends, so
       ``already_settled`` is a terminal answer ONLY when no live physical
       ownership remains (no RUNNING row / busy worker). One predicate
       (`task_has_live_ownership` supervisor-side, the queue-snapshot twin
       worker-side); every cancel ingress passes ``allow_settled_target``
       when ownership is live; custody kills a settled-but-live worker while
       completion-wins preserves the stored result; project deletion mints
       the cascade intent over a settled-but-live root and treats a settled
       root winding down as pending wind-down, not instant failure;
GR6-2  the cascade digest enumerates descendants by ANCESTRY rooted at the
       cancelled node (mid-tree grandchildren and non-subagent descendants
       included), never by root_task_id equality;
GR6-3  registry strictness validates ROWS, not just containers: a malformed
       pending/delivered/intent row refuses the mutation (typed corruption,
       bytes kept) and the enforcement reads disclose loudly once, then
       quarantine (skip, never silently drop);
GR6-4  an EXISTING-but-unreadable custody log audits as typed UNKNOWN
       (``delegated_run_state_unknown:custody_log_unreadable``), never as
       "cleanly reconciled"; an absent log stays a clean empty state;
GR6-5  (a) the unreconciled-runs disclosure line is outcome-INDEPENDENT —
       completed and failed deliveries carry it too; (b) the retry/effective
       projection preserves ``delegated_runs_unreconciled``.
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
import types

import pytest

from ouroboros import cancel_intents as ci
from ouroboros.task_results import (
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_INTERRUPTED,
    STATUS_RUNNING,
    load_task_result,
    write_task_result,
)
from tests._shared import write_queue_document
from tests._store_shared import release_store_handles, result_signature

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


def _install_live_worker(qenv, monkeypatch, task_id, *, wid=0):
    """A worker double whose LIVE process dies only through kill_pid_tree."""
    state = {"alive": True}
    proc = types.SimpleNamespace(
        pid=4242,
        is_alive=lambda: state["alive"],
        terminate=lambda: state.__setitem__("alive", False),
        join=lambda timeout=None: None,
    )
    qenv.workers.WORKERS[wid] = types.SimpleNamespace(
        wid=wid, proc=proc, busy_task_id=task_id, reaping=False,
    )
    monkeypatch.setattr(
        "ouroboros.platform_layer.kill_pid_tree",
        lambda *a, **kw: state.__setitem__("alive", False),
    )
    return state


# --------------------------------------------------------------------------
# GR6-1 — settled RESULT does not mean dead WORKER (the class closure)
# --------------------------------------------------------------------------


def test_gr6_1_cancel_kills_a_settled_tasks_live_worker_and_keeps_the_result(
    qenv, monkeypatch,
):
    """The incident shape: the result settled ``completed`` while the worker
    kept burning post-task LLM calls. The cancel (intent minted with the
    live-ownership fact) must KILL the worker, preserve the stored completed
    result verbatim, and settle the intent ``already_settled`` only after the
    confirmed death.

    GR7-2 fixture completion (sol): the task carries a REAL ``drive_root`` —
    the shared-drive shape — so the settled row is exposed to the kill path's
    child-copy/finalize mutations and the byte-identity assertion actually
    catches them (a drive-less row made the mutating block a silent no-op)."""
    from supervisor import workers

    task_id = "burn1"
    queue = _CaptureQueue()
    monkeypatch.setattr(workers, "get_event_q", lambda *, create=True: queue, raising=False)
    emitted: list = []
    monkeypatch.setattr(
        qenv.q, "_emit_cancel_task_done", lambda *a, **kw: emitted.append(kw),
    )
    state = _install_live_worker(qenv, monkeypatch, task_id)
    qenv.q.RUNNING[task_id] = {
        "task": {"id": task_id, "chat_id": 6, "drive_root": str(qenv.drive)},
        "worker_id": 0,
    }
    write_task_result(qenv.drive, task_id, STATUS_COMPLETED, chat_id=6,
                      result="the finished answer")
    row_bytes_before = result_signature(qenv.drive, task_id)

    assert qenv.tl.task_has_live_ownership(task_id) is True
    minted = ci.request_cancel(qenv.drive, task_id, reason="stop spending",
                               allow_settled_target=True)
    assert minted.get("request_id"), (
        "GR6-1a: live ownership means the mint must NOT no-op as already_settled"
    )

    assert qenv.tl.cancel_task_custody(task_id) == qenv.tl.CANCEL_ALREADY_SETTLED
    assert not state["alive"], "GR6-1b: the settled-but-live worker is killed"
    stored = load_task_result(qenv.drive, task_id)
    assert stored["status"] == STATUS_COMPLETED, "completion wins: no overwrite"
    assert stored["result"] == "the finished answer"
    assert result_signature(qenv.drive, task_id) == row_bytes_before, (
        "GR7-2: the settled terminal row survives the kill BYTE-IDENTICAL"
    )
    assert task_id not in qenv.q.RUNNING
    assert ci.active_intent(qenv.drive, task_id) is None, (
        "the intent settles already_settled after the confirmed death"
    )
    trail = (qenv.drive / "logs" / "supervisor.jsonl").read_text(encoding="utf-8")
    assert '"outcome": "already_settled"' in trail
    assert emitted, "the card resolves through the stored terminal truth"


def test_gr6_1_live_ownership_predicates(qenv, tmp_path):
    """Both halves of the ONE predicate: supervisor live maps and the
    worker-side queue-snapshot twin."""
    from ouroboros.task_status import task_has_live_queue_ownership

    assert qenv.tl.task_has_live_ownership("nobody") is False
    qenv.q.RUNNING["own-r"] = {"task": {"id": "own-r"}}
    assert qenv.tl.task_has_live_ownership("own-r") is True
    qenv.workers.WORKERS[3] = types.SimpleNamespace(
        wid=3, busy_task_id="own-w", reaping=False, proc=None,
    )
    assert qenv.tl.task_has_live_ownership("own-w") is True

    from ouroboros.utils import utc_now_iso

    snap = tmp_path / "state" / "queue_snapshot.json"
    snap.parent.mkdir(parents=True, exist_ok=True)
    # A FRESH snapshot (GR7-1a): only a fresh one may prove a dead worker.
    write_queue_document(snap, {
        "ts": utc_now_iso(),
        "running": [{"id": "own-s", "task": {"id": "own-s"}}],
        "pending": [{"id": "own-p", "task": {"id": "own-p"}}],
    })
    assert task_has_live_queue_ownership(tmp_path, "own-s") is True
    assert task_has_live_queue_ownership(tmp_path, "own-p") is False, (
        "a queued row is not physical ownership (no process to kill)"
    )
    assert task_has_live_queue_ownership(tmp_path, "own-x") is False


def test_gr6_1_every_ingress_passes_the_live_ownership_fact():
    """Source pins for the ingress wiring (the custody behavior itself is
    functionally tested above): agent tool, HTTP single, evolution stop,
    project delete, cascade descendants."""
    jl = (REPO_ROOT / "ouroboros" / "tools" / "join_ledger.py").read_text(encoding="utf-8")
    assert "task_has_live_queue_ownership(status_drive_root, tid)" in jl
    assert "allow_settled_target=live_ownership" in jl

    gw = (REPO_ROOT / "ouroboros" / "gateway" / "tasks.py").read_text(encoding="utf-8")
    assert "task_has_live_ownership as _live_ownership" in gw
    assert "allow_settled_target=bool(cascade or live_own)" in gw  # graceful ingress
    # HTTP Stop: a target the intent request found settled is re-requested with
    # the live-ownership fact (cascade: also its live subtree).
    assert "return task_has_live_ownership(task_id) or (cascade and task_subtree_is_live(task_id))" in gw
    assert ('if intent.get("already_settled") and still_owned():\n'
            '        intent = record(allow_settled_target=True)') in gw

    qt = (REPO_ROOT / "supervisor" / "queue_transitions.py").read_text(encoding="utf-8")
    assert qt.count("allow_settled_target=task_has_live_ownership(task_id)") == 2, (
        "evolution stop AND project delete both pass the live-ownership fact"
    )

    tl = (REPO_ROOT / "supervisor" / "task_lifecycle.py").read_text(encoding="utf-8")
    assert 'source="cascade_descendant", requested_by=task_id,\n                allow_settled_target=True' in tl


def _patch_project_registry(monkeypatch):
    import ouroboros.projects_registry as pr

    done: dict = {"complete": [], "fail": []}
    monkeypatch.setattr(pr, "project_task_bindings", lambda root: {})
    monkeypatch.setattr(
        pr, "complete_project_deletion",
        lambda root, pid: done["complete"].append(str(pid)),
    )
    monkeypatch.setattr(
        pr, "fail_project_deletion",
        lambda root, pid, detail: done["fail"].append((str(pid), str(detail))),
    )
    return done


def test_gr6_1_project_delete_over_settled_live_root_mints_and_converges(
    qenv, monkeypatch,
):
    """The GR6-1c probe: a durably-completed root still in RUNNING (worker
    winding down / burning) plus a pending child. The deletion mints the
    cascade coordination intent (allow_settled_target from live ownership),
    the cascade kills the live worker, and the deletion CONVERGES instead of
    failing "did not quiesce"."""
    from supervisor import queue_transitions as qt
    from supervisor import workers

    done = _patch_project_registry(monkeypatch)
    queue = _CaptureQueue()
    monkeypatch.setattr(workers, "get_event_q", lambda *, create=True: queue, raising=False)
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *a, **kw: None)

    root_id = "pd-root"
    state = _install_live_worker(qenv, monkeypatch, root_id)
    qenv.q.RUNNING[root_id] = {
        "task": {"id": root_id, "chat_id": 8, "project_id": "p6",
                 "root_task_id": root_id},
        "worker_id": 0,
    }
    qenv.q.PENDING.append(
        {"id": "pd-kid", "parent_task_id": root_id, "root_task_id": root_id},
    )
    write_task_result(qenv.drive, root_id, STATUS_COMPLETED, chat_id=8,
                      result="root finished, worker still up")

    mints: list = []
    real_mint = ci.request_cancel

    def _recording_mint(drive, task_id, **kw):
        mints.append({"task_id": str(task_id), **kw})
        return real_mint(drive, task_id, **kw)

    monkeypatch.setattr(ci, "request_cancel", _recording_mint)

    qt.run_project_deletion(qenv.drive, "p6", 8)

    assert done["complete"] == ["p6"] and not done["fail"], (
        "GR6-1c: the deletion converges instead of 'did not quiesce'"
    )
    root_mints = [m for m in mints if m["task_id"] == root_id
                  and m.get("source") == "project_delete"]
    assert root_mints and root_mints[0].get("allow_settled_target") is True, (
        "the cascade coordination intent IS minted over the settled-but-live root"
    )
    assert root_mints[0].get("scope") == ci.SCOPE_CASCADE
    assert not state["alive"], "the winding-down worker is killed by the cascade"
    assert load_task_result(qenv.drive, root_id)["status"] == STATUS_COMPLETED
    assert load_task_result(qenv.drive, "pd-kid")["status"] == STATUS_CANCELLED
    assert ci.active_intents(qenv.drive) == {}, "everything settled"


def test_gr6_1_crash_mid_delete_leaves_a_replayable_cascade_intent(
    qenv, monkeypatch,
):
    """A teardown crash after the mint must leave the durable cascade intent
    ACTIVE (the watchdog's replay trigger) and fail the deletion visibly."""
    from supervisor import queue_transitions as qt

    done = _patch_project_registry(monkeypatch)
    root_id = "pd-crash"
    qenv.q.RUNNING[root_id] = {
        "task": {"id": root_id, "chat_id": 8, "project_id": "p6c",
                 "root_task_id": root_id},
        "worker_id": 0,
    }
    monkeypatch.setattr(
        qenv.q, "cancel_task_by_id",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("teardown crashed")),
    )

    qt.run_project_deletion(qenv.drive, "p6c", 8)

    assert done["fail"] and done["fail"][0][0] == "p6c"
    intent = ci.active_intent(qenv.drive, root_id)
    assert intent is not None and intent.get("scope") == ci.SCOPE_CASCADE, (
        "the durable cascade intent survives the crash for the watchdog replay"
    )


def test_gr6_1_quiescence_defers_for_a_settled_root_winding_down(
    qenv, monkeypatch,
):
    """A settled root whose RUNNING row lingers (finalizer mid-removal, no
    live process) is 'pending wind-down': the quiescence check defers and
    re-checks instead of raising instantly."""
    from supervisor import queue_transitions as qt
    from supervisor import workers

    done = _patch_project_registry(monkeypatch)
    queue = _CaptureQueue()
    monkeypatch.setattr(workers, "get_event_q", lambda *, create=True: queue, raising=False)
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *a, **kw: None)

    root_id = "pd-wind"
    qenv.q.RUNNING[root_id] = {
        "task": {"id": root_id, "chat_id": 8, "project_id": "p6w",
                 "root_task_id": root_id},
        "worker_id": 0,
    }
    write_task_result(qenv.drive, root_id, STATUS_COMPLETED, chat_id=8,
                      result="done; finalizer still owns the row")

    sleeps: list = []

    def _finalizer_finishes(seconds):
        sleeps.append(seconds)
        qenv.q.RUNNING.pop(root_id, None)

    monkeypatch.setattr(time, "sleep", _finalizer_finishes)

    qt.run_project_deletion(qenv.drive, "p6w", 8)

    assert sleeps, "the wind-down defer path ran instead of an instant failure"
    assert done["complete"] == ["p6w"] and not done["fail"], (
        "GR6-1c: a settled root winding down converges once its row leaves"
    )


# --------------------------------------------------------------------------
# GR6-2 — cascade digest membership by ANCESTRY rooted at the cancelled node
# --------------------------------------------------------------------------


def test_gr6_2_mid_tree_cascade_digest_lists_grandchildren(tmp_path, monkeypatch):
    """A mid-tree target's deeper descendants keep the ORIGINAL tree's
    root_task_id — equality matching lost them. The ancestry walk lists the
    grandchild (and a non-subagent child) while excluding the target's own
    ancestors; sweep outcomes were empty (the watchdog-replay shape)."""
    from supervisor import terminal_delivery as td
    from supervisor import workers

    queue = _CaptureQueue()
    monkeypatch.setattr(workers, "get_event_q", lambda *, create=True: queue, raising=False)
    write_task_result(tmp_path, "orig-root", STATUS_RUNNING, chat_id=9, result="alive")
    write_task_result(tmp_path, "mid6", STATUS_CANCELLED, chat_id=9, result="mid down",
                      parent_task_id="orig-root", root_task_id="orig-root")
    # Non-subagent child (grok's adjacent residual): no delegation_role.
    write_task_result(tmp_path, "kid6", STATUS_CANCELLED, result="kid down",
                      parent_task_id="mid6", root_task_id="orig-root")
    write_task_result(tmp_path, "gkid6", STATUS_COMPLETED, result="gkid done",
                      parent_task_id="kid6", root_task_id="orig-root",
                      delegation_role="subagent")

    owed = td.deliver_cascade_summary(
        tmp_path, "mid6", {"id": "mid6", "chat_id": 9,
                           "parent_task_id": "orig-root",
                           "root_task_id": "orig-root"}, {},
    )

    assert owed is True
    (event,) = [e for e in queue.events if e.get("type") == "send_message"]
    assert "2 descendant task(s) were settled with it" in event["text"]
    outcomes = {
        row["task_id"]: row["outcome"]
        for row in load_task_result(tmp_path, "mid6")["cancel_receipt"]["children"]
    }
    assert outcomes.get("kid6") == "cancelled", (
        "GR6-2: a non-subagent direct child is in the digest"
    )
    assert outcomes.get("gkid6") == "completed", (
        "GR6-2: the mid-tree grandchild is enumerated by ancestry"
    )
    assert "orig-root" not in outcomes, (
        "the target's own ANCESTOR is not a descendant"
    )


# --------------------------------------------------------------------------
# GR6-3 — registry strictness validates ROWS, not just containers
# --------------------------------------------------------------------------


def test_gr6_3_malformed_pending_row_refuses_the_mutation_and_replay_discloses(
    tmp_path, caplog,
):
    from supervisor import terminal_delivery as td

    store = tmp_path / "state" / "terminal_deliveries.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps({
        "schema_version": 2, "delivered": [],
        "pending": {"final:bad:aa": "bad-row"},
    }), encoding="utf-8")
    before = store.read_text(encoding="utf-8")

    # The probe: the same id used to be answered "already durably owed".
    assert td.register_pending_delivery(tmp_path, {
        "type": "send_message", "chat_id": 1, "task_id": "bad",
        "text": "x", "delivery_id": "final:bad:aa",
    }) is False
    assert store.read_text(encoding="utf-8") == before, "no overwrite, bytes kept"
    assert any(
        r.get("type") == "terminal_delivery_registry_corrupt"
        for r in _event_rows(tmp_path)
    )

    # Clearing the same malformed row as "delivered" is refused too.
    assert td.register_delivery(tmp_path, "final:bad:aa") is True  # fail-open contract
    assert store.read_text(encoding="utf-8") == before

    # The enforcement read quarantines loudly instead of silently dropping.
    with caplog.at_level(logging.ERROR, logger="supervisor.terminal_delivery"):
        assert td.pending_deliveries(tmp_path, disclose_corruption=True) == []
    assert any("malformed owed row" in r.message for r in caplog.records)
    assert any(
        r.get("type") == "terminal_delivery_registry_corrupt"
        and r.get("op") == "pending_row_malformed"
        for r in _event_rows(tmp_path)
    )
    assert store.read_text(encoding="utf-8") == before, "reads never rewrite the file"


def test_gr6_3_malformed_delivered_container_or_entry_refuses_the_mutation(tmp_path):
    from supervisor import terminal_delivery as td

    store = tmp_path / "state" / "terminal_deliveries.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    # Present-but-non-list delivered.
    store.write_text(json.dumps({
        "schema_version": 2, "delivered": {"oops": 1}, "pending": {},
    }), encoding="utf-8")
    before = store.read_text(encoding="utf-8")
    assert td.register_pending_delivery(tmp_path, {
        "type": "send_message", "chat_id": 1, "task_id": "d6",
        "text": "x", "delivery_id": "final:d6:aa",
    }) is False
    assert store.read_text(encoding="utf-8") == before

    # Valid list, non-string ENTRY.
    store.write_text(json.dumps({
        "schema_version": 2, "delivered": [{"not": "a-string"}], "pending": {},
    }), encoding="utf-8")
    before = store.read_text(encoding="utf-8")
    assert td.register_delivery(tmp_path, "final:d6:bb") is True  # fail-open
    assert store.read_text(encoding="utf-8") == before, (
        "GR6-3: a non-string delivered entry is never silently coerced/overwritten"
    )
    corrupt = [
        r for r in _event_rows(tmp_path)
        if r.get("type") == "terminal_delivery_registry_corrupt"
    ]
    assert len(corrupt) >= 2


def test_gr6_3_malformed_intent_row_refuses_the_mint_and_read_discloses(
    tmp_path, caplog,
):
    path = tmp_path / "state" / "cancel_intents.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 1,
        "intents": {"bad6": "not-a-dict",
                    "good6": {"request_id": "ci_ok", "task_id": "good6",
                              "state": "requested", "generation": 0}},
    }), encoding="utf-8")
    before = path.read_text(encoding="utf-8")

    with pytest.raises(ci.CancelIntentProjectionCorrupt):
        ci.request_cancel(tmp_path, "bad6", reason="stop")
    assert path.read_text(encoding="utf-8") == before, "no overwrite, bytes kept"

    with caplog.at_level(logging.ERROR, logger="ouroboros.cancel_intents"):
        active = ci.active_intents(tmp_path, disclose_corruption=True)
    assert "good6" in active and "bad6" not in active, (
        "GR6-3: the malformed row is quarantined, the valid one still enforced"
    )
    assert any("malformed row" in r.message for r in caplog.records)
    rows = _event_rows(tmp_path, log_name="supervisor.jsonl")
    assert any(
        r.get("event") == "projection_corrupt_refused"
        and r.get("op") == "active_intents_row"
        for r in rows
    )
    assert path.read_text(encoding="utf-8") == before


# --------------------------------------------------------------------------
# GR6-4 — unreadable custody log audits as typed UNKNOWN, never clean
# --------------------------------------------------------------------------


def test_gr6_4_unreadable_custody_log_is_typed_unknown_not_clean(qenv, monkeypatch):
    from ouroboros import delegate_custody as dc
    from supervisor.terminal_delivery import RUN_STATE_UNKNOWN_PREFIX

    # ABSENT log: positively-established clean state.
    assert dc.custody_log_unreadable(qenv.drive) is False
    assert qenv.tl._reconcile_delegated_runs_on_kill(qenv.q, "au6") == []

    # EXISTING but unreadable: a directory where the custody store should be makes
    # the store raise its typed unreadable on every platform (the injected unreadability).
    from ouroboros.store import STORE_REL

    unreadable = qenv.drive / STORE_REL
    if unreadable.exists():
        release_store_handles(unreadable)  # this process's cached handle (Windows refuses the delete)
        unreadable.unlink()
    unreadable.mkdir(parents=True)
    assert dc.custody_log_unreadable(qenv.drive) is True

    still = qenv.tl._reconcile_delegated_runs_on_kill(qenv.q, "au6")
    assert still == [f"{RUN_STATE_UNKNOWN_PREFIX}:custody_log_unreadable"], (
        "GR6-4: unreadable evidence is UNKNOWN, never 'cleanly reconciled'"
    )
    disclosures = [
        r for r in _event_rows(qenv.drive)
        if r.get("type") == "delegated_runs_unreconciled" and r.get("task_id") == "au6"
    ]
    assert disclosures and disclosures[0].get("flavor") == "audit_failed"


# --------------------------------------------------------------------------
# GR6-5 — outcome-independent disclosure + projection preservation
# --------------------------------------------------------------------------


def test_gr6_5a_completed_miss_lane_delivery_carries_the_disclosure(tmp_path):
    from supervisor import terminal_delivery as td

    queue = _CaptureQueue()
    row = {"task_id": "c6", "chat_id": 4, "result": "the finished answer"}
    assert td.deliver_miss_lane_outcome(
        tmp_path, tmp_path, row, "c6", "completed",
        event_queue=queue, unreconciled_runs=["run-x", "run-y"],
    ) is True
    (event,) = [e for e in queue.events if e.get("type") == "send_message"]
    assert event["text"] == "the finished answer", "the answer keeps the model's words alone"
    assert "2 delegated run(s) may still be live: run-x, run-y" in event["terminal_custody_notice"], (
        "the completed outcome carries the disclosure on its own custody row (#1006)"
    )


def test_gr6_5a_failed_outcome_still_discloses_open_runs(tmp_path):
    from supervisor import terminal_delivery as td

    queue = _CaptureQueue()
    row = {"task_id": "f6", "chat_id": 4, "result": "died"}
    assert td.deliver_miss_lane_outcome(
        tmp_path, tmp_path, row, "f6", "failed",
        event_queue=queue, unreconciled_runs=["run-z"],
    ) is True
    (event,) = [e for e in queue.events if e.get("type") == "send_message"]
    assert "1 delegated run(s) may still be live: run-z" in event["text"], (
        "GR6-5a: a failed outcome must not swallow the open-run fact"
    )

    # And with NOTHING open, a failed outcome still delivers nothing here
    # (its own path owns the failure message).
    quiet = _CaptureQueue()
    assert td.deliver_miss_lane_outcome(
        tmp_path, tmp_path, row, "f6", "failed", event_queue=quiet,
    ) is True
    assert quiet.events == []


def test_gr6_5a_owed_and_publish_halves_mint_one_delivery_id(tmp_path):
    """The disclosure is part of the TEXT before the id is minted, so the
    owed registration and the publish half (same list) dedupe as one."""
    from supervisor import terminal_delivery as td

    stored = {"status": "completed", "result": "answer"}
    task = {"id": "one6", "chat_id": 4}
    owed_event = td.build_completed_result_event(
        tmp_path, task, "one6", stored, unreconciled_runs=["run-a"],
    )
    assert td.register_pending_delivery(tmp_path, owed_event) is True
    queue = _CaptureQueue()
    td.deliver_completed_result(
        tmp_path, task, "one6", stored, event_queue=queue,
        unreconciled_runs=["run-a"],
    )
    (event,) = queue.events
    assert event["delivery_id"] == owed_event["delivery_id"]
    assert td.register_pending_delivery(tmp_path, event) is True, "already owed"
    assert len(td.pending_deliveries(tmp_path)) == 1, "one owed row, not two"


def test_gr6_5b_retry_projection_preserves_unreconciled_runs(tmp_path):
    from ouroboros.task_status import effective_task_result

    write_task_result(
        tmp_path, "orig6", STATUS_INTERRUPTED, chat_id=2,
        reason_code="idle_timeout_retry", retry_task_id="retry6",
        superseded_by="retry6", delegated_runs_unreconciled=["run-r"],
        result="killed; retrying",
    )
    write_task_result(tmp_path, "retry6", STATUS_COMPLETED, chat_id=2,
                      result="retry finished")

    effective = effective_task_result(
        tmp_path, load_task_result(tmp_path, "orig6"), materialize_artifacts=False,
    )
    assert effective["status"] == STATUS_COMPLETED
    assert effective["delegated_runs_unreconciled"] == ["run-r"], (
        "GR6-5b: the retry projection keeps the raw interrupted row's disclosure"
    )
    raw = load_task_result(tmp_path, "orig6")
    assert raw["delegated_runs_unreconciled"] == ["run-r"], "raw row untouched"
