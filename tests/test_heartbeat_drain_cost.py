"""The heartbeat on the drain: an in-place liveness stamp and a display read, nothing more.

Two guards of the same handler, pinned in both directions: a heartbeat never
writes a RUNNING entry back (a cross-thread cancel that popped the row between
the get and the write would be undone), and the root cost figure it carries is
the stale-allowed display projection, which returns the stored summary as is —
no owner-amendment overlay, so no read of the current task-result store for
every running root on every heartbeat.
"""
from __future__ import annotations

import time
import types


from tests._usage_store_testing import root  # noqa: F401  (fixture)


class _PoppedBetweenGetAndWrite(dict):
    """RUNNING as a cross-thread cancel leaves it: the row is gone once read."""

    def get(self, key, default=None):
        value = super().get(key, default)
        super().pop(key, None)
        return value


def _ctx(tmp_path, running):
    pushed = []
    return types.SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=running,
                                 bridge=types.SimpleNamespace(push_log=pushed.append)), pushed


def test_a_heartbeat_after_a_concurrent_pop_does_not_resurrect_the_task(tmp_path, monkeypatch):
    from ouroboros import cost_projection
    from supervisor.events_worker_reports import _handle_task_heartbeat

    monkeypatch.setattr(cost_projection, "live_root_cost_projection", lambda *a, **k: {})
    meta = {"task": {"id": "root-hb", "type": "task", "root_task_id": "root-hb"}, "started_at": time.time()}
    ctx, pushed = _ctx(tmp_path, _PoppedBetweenGetAndWrite({"root-hb": meta}))
    _handle_task_heartbeat({"task_id": "root-hb", "phase": "running"}, ctx)
    assert "root-hb" not in ctx.RUNNING, "no write-back: the popped row stays popped"
    assert meta["last_heartbeat_at"] > 0 and meta["heartbeat_phase"] == "running", "the stamp is in place"
    assert len(pushed) == 1 and pushed[0]["type"] == "task_heartbeat"

    # The ordinary row is the SAME object before and after: stamped, never replaced.
    live = {"task": {"id": "child-hb", "type": "task"}, "started_at": time.time()}
    ctx, pushed = _ctx(tmp_path, {"child-hb": live})
    _handle_task_heartbeat({"task_id": "child-hb", "phase": "running"}, ctx)
    assert ctx.RUNNING["child-hb"] is live and live["heartbeat_phase"] == "running" and len(pushed) == 1
    # A task that already left RUNNING, or a malformed entry, is a no-op.
    ctx, pushed = _ctx(tmp_path, {"odd": "not a dict"})
    _handle_task_heartbeat({"task_id": "gone", "phase": "running"}, ctx)
    _handle_task_heartbeat({"task_id": "odd", "phase": "running"}, ctx)
    assert pushed == [] and ctx.RUNNING == {"odd": "not a dict"}


def test_a_display_projection_never_reads_the_task_result_store(root, monkeypatch):  # noqa: F811
    from ouroboros import task_results, usage_admission

    reads = []
    monkeypatch.setattr(task_results, "load_task_result",
                        lambda *a, **k: reads.append((a, k)) or None)

    stale = usage_admission.current_usage_projection(root, root_task_id="root-hb", allow_stale=True)
    strict = usage_admission.current_usage_projection(root, root_task_id="root-hb")
    # The money read's owner-amendment overlay (acceptance root-cap amendments) left with the
    # review fold: both reads are the store's own figure.
    assert isinstance(stale, dict) and stale == strict and reads == []


def test_a_root_heartbeat_reads_no_task_result_on_the_drain(root, monkeypatch):  # noqa: F811
    from ouroboros import task_results
    from supervisor.events_worker_reports import _handle_task_heartbeat

    reads = []
    monkeypatch.setattr(task_results, "load_task_result", lambda *a, **k: reads.append(a) or None)
    meta = {"task": {"id": "root-hb", "type": "task", "root_task_id": "root-hb", "budget_drive_root": str(root)},
            "started_at": time.time()}
    ctx, pushed = _ctx(root, {"root-hb": meta})
    _handle_task_heartbeat({"task_id": "root-hb", "phase": "running"}, ctx)
    assert len(pushed) == 1 and pushed[0]["task_id"] == "root-hb"
    assert pushed[0].get("cost_accounting_status") != "unavailable"
    assert reads == [], "the heartbeat's cost figure is the stale display read, never current.sqlite"
