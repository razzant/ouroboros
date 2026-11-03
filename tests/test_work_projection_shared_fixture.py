"""The projection fixture owns its private data root: the store handles its real reads cache go back before it goes.

The projection reads a paused tree's model recipients from its data root's usage store through the process's cached
connections. POSIX removes a folder under an open file; Windows does not (``WinError 32`` on ``state/usage.sqlite``),
so ``work_fields`` hands its root's handles back first: its root's only, never another root's (the installation's
included). These assert the handles, not the removal, so they fail without the release on every OS, as
``tests/test_consciousness_client_fixture.py`` does for the wake-fixture generator.
"""
from __future__ import annotations

from ouroboros import store
from ouroboros.gateway import work_projection as wp
from tests._work_projection_shared import ROOT_PREFIX, work_fields
from tests.test_store import publish

# A paused tree: the projection asks its root's usage store which model recipients the Pause still waits for.
PAUSED = {"activity_id": "t", "phase": "budget_paused", "kind": "managed_task", "chat_id": 42, "task_attempt": 1,
          "started_at": 100.0, "pause_cause": "owner", "can_resume": False}


def _handles(fragment):
    """Cached store handles (every thread's) on the files whose path holds FRAGMENT."""
    with store._CONNECTION_GUARD:
        return sorted(name for pool in list(store._CONNECTION_POOLS) for name in pool.entries if fragment in name)


def test_the_fixture_hands_its_roots_stores_back_before_the_root_goes(monkeypatch):
    inside = []
    real = wp.works_snapshot

    def watched(*args, **kwargs):  # observation only: the real projection, then what it left in the connection cache
        snapshot = real(*args, **kwargs)
        inside.extend(_handles(ROOT_PREFIX))
        return snapshot

    monkeypatch.setattr(wp, "works_snapshot", watched)
    fields = work_fields([PAUSED])
    assert fields["works_complete"] is True and [work["activity"] for work in fields["works"]] == ["paused"]
    assert [name for name in inside if name.endswith("usage.sqlite")], "control: the paused tree's read caches a handle"
    assert _handles(ROOT_PREFIX) == [], "every handle under the fixture's root is back before the root goes"


def test_the_fixture_closes_only_its_own_roots_handles(tmp_path):
    other = tmp_path / "other"
    publish(other)
    try:
        with store.open_store(other, write=False) as txn:
            held = txn.conn
        assert _handles(str(other))
        work_fields([PAUSED])
        assert _handles(str(other)), "another root's cached handle stays"
        assert held.execute("SELECT 1").fetchone()[0] == 1
    finally:
        store.close_connections_under(other)
    assert _handles(str(other)) == []
