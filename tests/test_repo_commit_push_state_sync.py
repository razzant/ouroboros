"""Regression test: `_repo_commit_push` must write `state.current_sha`
synchronously and atomically after a successful commit.

`supervisor/worker_pool_lifecycle.py`'s `_watch_booting_slots` reads the
persisted `current_sha` as `expected_sha` and compares it against a booting
worker's actual `git_sha` on every respawn — not only after a managed
update, but on any crash-triggered or manually-requested restart. Before
this fix, `current_sha` in `state.json` was only ever refreshed by the
managed-update checkout (`git_ops_reset.py`) and rollback
(`update_recovery.py`) paths; an ordinary `commit_reviewed` commit never
touched it. So a worker crash-respawn any time after a plain commit would
compare the worker's real (new) `git_sha` against a stale `current_sha`,
producing a false "Worker SHA mismatch after spawn" owner-facing alert.

The naive fix would be ``st = load_state(); st["current_sha"] = commit_sha;
save_state(st)`` — three separate lock acquisitions, so a concurrent writer's
save could land between our load and our save and be silently dropped (or
drop ours). The actual fix uses ``update_state(mutator)``, which holds
STATE_LOCK for the whole read-modify-write as one critical section.

This test pins three things so the write cannot regress silently:

1. ``update_state`` actually persists the new ``current_sha`` to disk.
2. Two concurrent ``update_state`` writers writing different keys both
   land — the class-fix test for the lost-update race.
3. ``ouroboros/tools/git.py`` no longer imports the racy
   ``load_state, save_state`` pair together for ``current_sha`` writes,
   and it DOES import ``update_state`` (so the synchronous write is
   actually present).

A full end-to-end test through ``_repo_commit_push`` itself would need to
mock the entire advisory/triad review pipeline just to reach the commit —
disproportionate for pinning a single post-commit write — so this test
targets the write mechanism directly plus a static guard that it is wired
into ``git.py``.
"""
from __future__ import annotations

import inspect
import json
import threading


def _setup_state_root(tmp_path):
    """Redirect supervisor.state at tmp_path so the test is hermetic.

    init() resets the module-level globals (DRIVE_ROOT, STATE_PATH,
    STATE_LOCK_PATH) so the production data root never sees test
    writes. ensure_state_defaults() (called inside _load_state_unlocked)
    initialises ``current_sha`` to None; that is the starting state we
    then overwrite through update_state().
    """
    drive = tmp_path / "drive"
    from supervisor import state as state_mod
    state_mod.init(drive)
    return drive / "state" / "state.json"


def test_update_state_writes_current_sha_atomically(tmp_path):
    """update_state persists the new current_sha to disk."""
    state_path = _setup_state_root(tmp_path)
    new_sha = "deadbeef" * 4

    from supervisor import state as state_mod
    state_mod.update_state(lambda st: st.__setitem__("current_sha", new_sha))

    final = json.loads(state_path.read_text())
    assert final["current_sha"] == new_sha


def test_update_state_preserves_concurrent_writes(tmp_path):
    """Two concurrent update_state writers on different keys both land.

    Under a racy load-then-save pattern, a writer could lose its change if
    another writer's save interleaved between its load and save. With
    update_state holding STATE_LOCK for the whole operation, both writes
    are serialised and neither is lost.
    """
    state_path = _setup_state_root(tmp_path)
    from supervisor import state as state_mod

    barrier = threading.Barrier(2)
    completed = []

    def writer_current_sha():
        barrier.wait()
        state_mod.update_state(
            lambda st: st.__setitem__("current_sha", "writer1_sha")
        )
        completed.append("current_sha")

    def writer_other():
        barrier.wait()
        state_mod.update_state(
            lambda st: st.__setitem__("other", "writer2_value")
        )
        completed.append("other")

    t1 = threading.Thread(target=writer_current_sha)
    t2 = threading.Thread(target=writer_other)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert sorted(completed) == ["current_sha", "other"], (
        f"concurrent writers did not both finish: {completed}"
    )

    final = json.loads(state_path.read_text())
    assert final["current_sha"] == "writer1_sha", (
        f"current_sha write lost to concurrent writer: {final}"
    )
    assert final["other"] == "writer2_value", (
        f"other write lost to concurrent writer: {final}"
    )


def test_git_py_uses_update_state_not_racy_load_state_save_state():
    """Static guard: the racy load/save pattern must not return to git.py.

    The racy pattern is a paired import
    ``from supervisor.state import load_state, save_state``. If that
    exact import reappears anywhere in ouroboros/tools/git.py, the
    current_sha write is no longer atomic. The fix imports
    ``update_state`` (alone) so the synchronous post-commit write is
    actually present.
    """
    from ouroboros.tools import git as git_module
    src = inspect.getsource(git_module)
    assert "from supervisor.state import load_state, save_state" not in src, (
        "git.py uses the racy load_state/save_state pattern for "
        "current_sha. Use update_state(mutator) instead."
    )
    assert "from supervisor.state import update_state" in src, (
        "git.py no longer imports update_state — the post-commit "
        "synchronous current_sha write is missing entirely."
    )
