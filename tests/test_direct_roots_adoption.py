"""Boot adoption of orphaned direct-turn results (hard-crash class, 2026-10-09).

A machine that dies mid-turn can leave ``state/direct_roots.json`` not naming a
live direct turn whose durable result still says ``running``: the roster fragment
is rewritten by the main loop's tick, and that tick never ran again. The pooled
rows have their own boot sweep; direct rows had none, so the row outlived every
registry that named it (observed live: task ``1f7eef5f`` on 2026-10-09, progress
to 22:03, machine dead by 22:05, the 22:05 boot restored nothing).

These tests run the REAL seam: a durable result written the way the live writer
does, ``queue.init`` taking the fragment, and ``restore_pending_from_snapshot``
fencing what the handover names.
"""
from __future__ import annotations

import json

import pytest


def _write_running_direct_result(root, task_id, *, kind="direct"):
    from ouroboros.task_results import write_task_result

    return write_task_result(
        root, task_id, "running",
        execution_owner={"kind": kind, "data_root": str(root), "task_id": task_id},
        _is_direct_chat=True, chat_id=1,
    )


def _write_roster_fragment(root, task_ids):
    from supervisor import direct_roots

    direct_roots.publish_direct_roots(root)  # ensure parent dirs exist
    direct_roots.atomic_write_json(
        direct_roots._fragment_path(root),
        {"ts": direct_roots.utc_now_iso(), "roots": [
            {"task_id": task_id} for task_id in task_ids
        ], "incomplete": False},
    )


def _isolated_queue(monkeypatch, tmp_path):
    from supervisor import queue as queue_mod

    pending, running = [], {}
    queue_mod.init_queue_refs(pending, running, {"value": 0})
    queue_mod.ACCEPTANCE_FENCES.clear()
    monkeypatch.setattr(queue_mod, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue_mod, "QUEUE_SNAPSHOT_PATH", tmp_path / "state" / "queue_snapshot.json")
    return queue_mod, pending


def _write_restore_snapshot(tmp_path, tasks, running_rows=(), fences=()):
    from ouroboros.utils import utc_now_iso

    path = tmp_path / "state" / "queue_snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "ts": utc_now_iso(),
            "pending": [{"task": task} for task in tasks],
            "running": list(running_rows),
            "acceptance_fences": list(fences),
        }),
        encoding="utf-8",
    )


def test_boot_adopts_orphaned_direct_running_result(monkeypatch, tmp_path):
    queue_mod, _ = _isolated_queue(monkeypatch, tmp_path)
    stored = _write_running_direct_result(tmp_path, "dead-direct-1")
    assert stored["status"] == "running"
    _write_roster_fragment(tmp_path, [])  # the crash lost the turn's roster row
    _write_restore_snapshot(tmp_path, [])

    queue_mod.init(tmp_path)  # boot: takes the fragment and adopts the orphan

    adopted = dict(queue_mod.PRIOR_DIRECT_ROOTS)
    assert adopted["task_ids"] == ["dead-direct-1"]
    events = [json.loads(line) for line in
              (tmp_path / "logs" / "supervisor.jsonl").read_text().splitlines()]
    assert any(e.get("type") == "direct_roots_adopted_orphans"
               and e.get("task_ids") == ["dead-direct-1"] for e in events)

    fenced = []
    queue_mod.restore_pending_from_snapshot(terminalized=fenced)
    assert fenced == ["dead-direct-1"]
    from ouroboros.task_results import load_task_result
    from ouroboros.cancel_intents import has_active_intent
    assert has_active_intent(tmp_path, "dead-direct-1")
    row = load_task_result(tmp_path, "dead-direct-1")
    assert row["status"] in ("cancel_requested", "running")  # custody owns the settle


def test_boot_does_not_adopt_named_roster_or_snapshot_rows(monkeypatch, tmp_path):
    queue_mod, _ = _isolated_queue(monkeypatch, tmp_path)
    _write_running_direct_result(tmp_path, "roster-named")
    _write_running_direct_result(tmp_path, "snapshot-named")
    _write_running_direct_result(tmp_path, "truly-orphaned")
    _write_roster_fragment(tmp_path, ["roster-named"])
    _write_restore_snapshot(
        tmp_path, [], running_rows=[{"id": "snapshot-named", "task": {"id": "snapshot-named"}}],
    )

    queue_mod.init(tmp_path)

    adopted = dict(queue_mod.PRIOR_DIRECT_ROOTS)
    # The roster's own handover is preserved; only the orphan is adopted beside
    # it, and the snapshot-named row stays with the snapshot's own fence path.
    assert adopted["task_ids"] == ["roster-named", "truly-orphaned"]


def test_boot_does_not_adopt_pooled_or_terminal_rows(monkeypatch, tmp_path):
    queue_mod, _ = _isolated_queue(monkeypatch, tmp_path)
    from ouroboros.task_results import write_task_result

    write_task_result(  # pooled ownership: the snapshot fence owns this class
        tmp_path, "pooled-row", "running",
        execution_owner={"kind": "pooled", "data_root": str(tmp_path), "task_id": "pooled-row"},
    )
    write_task_result(  # terminal: nothing to settle
        tmp_path, "done-row", "completed",
        execution_owner={"kind": "direct", "data_root": str(tmp_path), "task_id": "done-row"},
    )
    write_task_result(  # no execution_owner: legacy/unknown, never guessed
        tmp_path, "ownerless-row", "running",
    )
    _write_roster_fragment(tmp_path, [])
    _write_restore_snapshot(tmp_path, [])

    queue_mod.init(tmp_path)

    assert dict(queue_mod.PRIOR_DIRECT_ROOTS)["task_ids"] == []


def test_boot_adoption_failure_keeps_row_unsettled(monkeypatch, tmp_path):
    queue_mod, _ = _isolated_queue(monkeypatch, tmp_path)
    _write_running_direct_result(tmp_path, "orphan-adopt-fail")
    _write_roster_fragment(tmp_path, [])
    _write_restore_snapshot(tmp_path, [])

    def _boom(*_a, **_k):
        raise RuntimeError("scan failed")

    from supervisor import direct_roots
    monkeypatch.setattr(direct_roots, "adopt_orphaned_direct_results", _boom)
    with pytest.raises(RuntimeError):
        queue_mod.init(tmp_path)
    # The durable row is untouched: unknown, never a fabricated terminal.
    from ouroboros.task_results import load_task_result
    assert load_task_result(tmp_path, "orphan-adopt-fail")["status"] == "running"


def test_boot_never_adopts_a_live_turn_of_this_process(monkeypatch, tmp_path):
    """An in-process revival re-runs queue.init beside live turns (N3)."""
    queue_mod, _ = _isolated_queue(monkeypatch, tmp_path)
    _write_running_direct_result(tmp_path, "live-turn")
    _write_roster_fragment(tmp_path, [])
    _write_restore_snapshot(tmp_path, [])

    class _Registry:
        @staticmethod
        def snapshot():
            return [{"activity_id": "live-turn"}]

    import supervisor.active_activity as active_activity
    monkeypatch.setattr(active_activity, "get_direct_activity_registry", lambda: _Registry)

    queue_mod.init(tmp_path)

    assert dict(queue_mod.PRIOR_DIRECT_ROOTS)["task_ids"] == []
    events_path = tmp_path / "logs" / "supervisor.jsonl"
    events = ([json.loads(line) for line in events_path.read_text().splitlines()]
              if events_path.exists() else [])
    assert not [e for e in events if e.get("type") == "direct_roots_adopted_orphans"]
