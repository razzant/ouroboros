"""S6 C5 — owner-stop descendant fences against a concurrent cascade's prune.

A graceful owner stop over a cascade intent hard-settles the live descendants
first (``owner_stop._settle_descendants_hard``) and fences every id it captured
in ``CANCELLED_ROOT_FENCES``, so a schedule event still draining cannot admit a
new child under a root that is finalizing. It calls the shared subtree sweep
WITHOUT a cascade token, so those ids never join ``_ACTIVE_CASCADE_FENCES`` —
the protected set an ordinary cascade holds for as long as it runs.

The question this module answers with a test rather than an argument: can a
concurrent, unrelated cascade's ``_prune_cancellation_fences`` evict them?
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import supervisor.owner_stop as ostop
import supervisor.task_lifecycle as tl


def _isolate(monkeypatch, tmp_path, *, pending=(), running=None):
    """A queue holding a live tree, with custody stubbed to a queue removal."""
    from supervisor import queue as q
    from supervisor import workers

    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(q, "PENDING", [dict(t) for t in pending])
    monkeypatch.setattr(q, "RUNNING", dict(running or {}))
    monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
    monkeypatch.setattr(q, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(tl, "CANCELLED_ROOT_FENCES", {}, raising=False)
    monkeypatch.setattr(tl, "_ACTIVE_CASCADE_FENCES", {}, raising=False)

    def _custody(task_id, **_kw):
        for index, item in enumerate(list(q.PENDING)):
            if str(item.get("id")) == task_id:
                q.PENDING.pop(index)
                return q.CANCEL_CANCELLED
        q.RUNNING.pop(task_id, None)
        return q.CANCEL_CANCELLED

    monkeypatch.setattr(q, "cancel_task_custody", _custody)
    monkeypatch.setattr(q, "append_jsonl", lambda *a, **k: None)
    return q


def _stale(seconds: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


def _fill_quiescent(count: int) -> None:
    for index in range(count):
        tl.CANCELLED_ROOT_FENCES[f"old{index}"] = "2026-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# C5 — the concurrency the reviewer hypothesis names
# ---------------------------------------------------------------------------


def test_c5_a_concurrent_cascade_prune_does_not_evict_owner_stop_fences(
    tmp_path, monkeypatch,
):
    """C5: the eviction does NOT reproduce.

    Owner stop A fences its root and its live grandchild; cascade B — an
    unrelated tree, its own protected set — then prunes a registry over the
    cap. A's ids survive because a fence is evictable only after the recency
    GRACE window, and A's were planted a moment ago. Absence from
    ``_ACTIVE_CASCADE_FENCES`` is not by itself an exposure.
    """
    q = _isolate(
        monkeypatch, tmp_path,
        running={"A": {"task": {"id": "A", "chat_id": 0, "root_task_id": "A"}}},
        pending=[
            {"id": "A-child", "root_task_id": "A", "parent_task_id": "A", "depth": 1},
            {"id": "A-grand", "root_task_id": "A", "parent_task_id": "A-child", "depth": 2},
        ],
    )
    monkeypatch.setattr(tl, "_CANCELLED_ROOT_FENCE_CAP", 32, raising=False)

    ostop._settle_descendants_hard(q, "A")
    planted = {"A", "A-child", "A-grand"}
    assert planted <= set(tl.CANCELLED_ROOT_FENCES), tl.CANCELLED_ROOT_FENCES
    _fill_quiescent(64)

    # Cascade B prunes on behalf of ITS tree only.
    tl._prune_cancellation_fences(protected={"B", "B-child"})

    assert planted <= set(tl.CANCELLED_ROOT_FENCES), (
        "a concurrent cascade's prune evicted the owner-stop episode's fences"
    )
    assert sum(1 for key in tl.CANCELLED_ROOT_FENCES if key.startswith("old")) < 64


def test_c5_the_recency_grace_window_is_what_protects_them(tmp_path, monkeypatch):
    """C5, the mechanism: it is the grace window, not a protected set.

    Aged past the window and with the registry over cap, the very same ids ARE
    evicted — so the protection is temporal. This matters for the remedy: a
    cascade token would only hold for the duration of the sweep call, a window
    in which the fences are young and already protected, so it would not change
    this outcome. The durable answer for a LONG episode is the re-stamp below.
    """
    q = _isolate(
        monkeypatch, tmp_path,
        running={"A": {"task": {"id": "A", "chat_id": 0, "root_task_id": "A"}}},
        pending=[{"id": "A-child", "root_task_id": "A", "parent_task_id": "A", "depth": 1}],
    )
    monkeypatch.setattr(tl, "_CANCELLED_ROOT_FENCE_CAP", 32, raising=False)

    ostop._settle_descendants_hard(q, "A")
    aged = _stale(tl._CANCELLED_ROOT_FENCE_GRACE_SEC + 60)
    for task_id in ("A", "A-child"):
        tl.CANCELLED_ROOT_FENCES[task_id] = aged
    _fill_quiescent(64)

    tl._prune_cancellation_fences(protected={"B"})

    assert "A-child" not in tl.CANCELLED_ROOT_FENCES, (
        "an aged fence outside every protected set is evictable by design"
    )


def test_c5_each_hold_tick_restamps_the_root_fence(tmp_path, monkeypatch):
    """C5, why a long episode still refuses late admission: every hold tick
    re-stamps the ROOT's fence, so it never ages out while the episode runs —
    and a task scheduled under that root is refused by the root's own entry
    (its ancestry walk reaches `A` directly, or matches its `root_task_id`).
    """
    q = _isolate(
        monkeypatch, tmp_path,
        running={"A": {"task": {"id": "A", "chat_id": 0, "root_task_id": "A"}}},
        pending=[{"id": "A-child", "root_task_id": "A", "parent_task_id": "A", "depth": 1}],
    )

    ostop._settle_descendants_hard(q, "A")
    tl.CANCELLED_ROOT_FENCES["A"] = _stale(tl._CANCELLED_ROOT_FENCE_GRACE_SEC + 60)

    ostop._settle_descendants_hard(q, "A")  # the next sweep tick of the same episode

    assert tl.CANCELLED_ROOT_FENCES["A"] != _stale(0), "sanity: stamps are timestamps"
    fresh = datetime.fromisoformat(str(tl.CANCELLED_ROOT_FENCES["A"]).replace("Z", "+00:00"))
    age = datetime.now(timezone.utc).timestamp() - fresh.timestamp()
    assert age < tl._CANCELLED_ROOT_FENCE_GRACE_SEC, "the root fence is re-stamped"
    late = {"id": "A-late", "root_task_id": "A", "parent_task_id": "A"}
    assert tl.root_cancellation_fenced(late) is True


def test_c5_a_prune_racing_the_sweep_itself_cannot_see_a_half_fenced_tree(
    tmp_path, monkeypatch,
):
    """C5, the narrow window made explicit: the sweep plants every fence and
    calls the prune INSIDE one hold of the queue lock, so a concurrent prune
    can only run before or after the whole set exists — never between the
    fences of one tree. That, plus the grace window, is the exclusion the
    cascade token provides for an ordinary cascade.
    """
    q = _isolate(
        monkeypatch, tmp_path,
        running={"A": {"task": {"id": "A", "chat_id": 0, "root_task_id": "A"}}},
        pending=[
            {"id": "A-child", "root_task_id": "A", "parent_task_id": "A", "depth": 1},
            {"id": "A-grand", "root_task_id": "A", "parent_task_id": "A-child", "depth": 2},
        ],
    )
    monkeypatch.setattr(tl, "_CANCELLED_ROOT_FENCE_CAP", 32, raising=False)
    _fill_quiescent(64)
    seen: list[set] = []
    stop = threading.Event()

    def _pruner():
        while not stop.is_set():
            with q._queue_lock:
                seen.append({
                    key for key in tl.CANCELLED_ROOT_FENCES if key.startswith("A")
                })
                tl._prune_cancellation_fences(protected={"B"})
            time.sleep(0.001)  # yield the lock; this is a shared host

    thread = threading.Thread(target=_pruner, name="cascade-B", daemon=True)
    thread.start()
    try:
        ostop._settle_descendants_hard(q, "A")
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not thread.is_alive()

    assert {"A", "A-child", "A-grand"} <= set(tl.CANCELLED_ROOT_FENCES)
    partial = [snapshot for snapshot in seen if snapshot and snapshot != {"A", "A-child", "A-grand"}]
    assert partial in ([], [{"A"}]), (
        f"a prune observed a partially fenced tree: {partial}"
    )


def test_c5_the_sweep_is_still_token_less(tmp_path, monkeypatch):
    """C5, stated as the durable fact behind the disclosure: the owner-stop
    sweep passes no cascade token, so its ids never enter the protected set.
    If a future change gives long episodes a protected set, this is the
    assertion that has to be updated with it."""
    q = _isolate(
        monkeypatch, tmp_path,
        running={"A": {"task": {"id": "A", "chat_id": 0, "root_task_id": "A"}}},
        pending=[{"id": "A-child", "root_task_id": "A", "parent_task_id": "A", "depth": 1}],
    )
    captured: list = []
    real_sweep = tl._cancel_subtree_sweep

    def _spy(queue_mod, task_id, already, cascade_token=""):
        captured.append(cascade_token)
        return real_sweep(queue_mod, task_id, already, cascade_token)

    monkeypatch.setattr(tl, "_cancel_subtree_sweep", _spy)

    ostop._settle_descendants_hard(q, "A")

    assert captured == [""], "the owner-stop descendant sweep runs token-less"
    assert tl._ACTIVE_CASCADE_FENCES == {}


def test_c5_an_ordinary_cascade_keeps_its_protected_set(tmp_path, monkeypatch):
    """The control: a real cascade DOES register a token, and its ids are
    protected even when they age past the grace window while it runs."""
    q = _isolate(
        monkeypatch, tmp_path,
        pending=[
            {"id": "C", "root_task_id": "C"},
            {"id": "C-child", "root_task_id": "C", "parent_task_id": "C", "depth": 1},
        ],
    )
    monkeypatch.setattr(tl, "_CANCELLED_ROOT_FENCE_CAP", 32, raising=False)
    token = tl._next_cascade_token("C")

    tl._cancel_subtree_sweep(q, "C", set(), token)

    assert tl._ACTIVE_CASCADE_FENCES[token] == {"C", "C-child"}
    aged = _stale(tl._CANCELLED_ROOT_FENCE_GRACE_SEC + 60)
    for task_id in ("C", "C-child"):
        tl.CANCELLED_ROOT_FENCES[task_id] = aged
    _fill_quiescent(64)

    tl._prune_cancellation_fences(protected={"B"})

    assert {"C", "C-child"} <= set(tl.CANCELLED_ROOT_FENCES), (
        "a live cascade's fences survive on the token, not on recency"
    )
