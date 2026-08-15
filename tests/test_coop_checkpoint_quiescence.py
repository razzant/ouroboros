"""v6.91 coop checkpoint at tree quiescence (supervisor/events.py).

A root-scope budget death always terminalizes the ROOT before its children, so
the root-done checkpoint call saw live tree tasks and never re-ran — a wave's
coop tree stayed an uncommitted pile (live evidence: coop_bf850dfa6b00 held
only its genesis commit two days after the death). These pin the closed class:
the LAST live subtree member's terminal event re-triggers the checkpoint, the
detect runs after the finishing child left RUNNING, and the git chain runs OFF
the supervisor event-drain thread with an in-flight latch per root.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
from types import SimpleNamespace

import pytest


def _init_git_repo(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=str(path), check=True)
    (path / "README.md").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(path), check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@local", "commit", "-qm", "init"],
        cwd=str(path), check=True,
    )


def _ctx(data: pathlib.Path, *, pending=None, running=None) -> SimpleNamespace:
    return SimpleNamespace(
        DRIVE_ROOT=data,
        PENDING=list(pending or []),
        RUNNING=dict(running or {}),
    )


def _subagent_task(task_id: str, root_id: str) -> dict:
    return {
        "id": task_id,
        "root_task_id": root_id,
        "parent_task_id": root_id,
        "delegation_role": "subagent",
    }


# --- detection (no git; pure fake ctx) --------------------------------------


def test_quiescence_detect_fires_for_settled_root_and_empty_tree(tmp_path, monkeypatch):
    from supervisor import events
    from ouroboros.task_results import write_task_result

    data = tmp_path / "data"
    data.mkdir()
    write_task_result(data, "root1", "failed", reason_code="budget_exhausted", title="Sunken city")
    spawned = []
    monkeypatch.setattr(
        events, "_spawn_coop_checkpoint",
        lambda ctx, root_tid, *, title, trigger: spawned.append((root_tid, title, trigger)),
    )
    events._maybe_checkpoint_coop_on_tree_quiescence(
        _ctx(data), _subagent_task("child1", "root1"), "child1",
    )
    assert spawned == [("root1", "Sunken city", "tree_quiescence")]


def test_quiescence_detect_skips_while_siblings_live(tmp_path, monkeypatch):
    from supervisor import events
    from ouroboros.task_results import write_task_result

    data = tmp_path / "data"
    data.mkdir()
    write_task_result(data, "root1", "failed", reason_code="budget_exhausted")
    spawned = []
    monkeypatch.setattr(
        events, "_spawn_coop_checkpoint",
        lambda *a, **k: spawned.append(a),
    )
    # A sibling subagent still RUNNING under the same root.
    running = {"sib": {"task": _subagent_task("sib", "root1")}}
    events._maybe_checkpoint_coop_on_tree_quiescence(
        _ctx(data, running=running), _subagent_task("child1", "root1"), "child1",
    )
    assert spawned == []


def test_quiescence_detect_skips_while_root_still_running_or_pending(tmp_path, monkeypatch):
    from supervisor import events
    from ouroboros.task_results import write_task_result

    data = tmp_path / "data"
    data.mkdir()
    write_task_result(data, "root1", "failed")
    spawned = []
    monkeypatch.setattr(events, "_spawn_coop_checkpoint", lambda *a, **k: spawned.append(a))
    events._maybe_checkpoint_coop_on_tree_quiescence(
        _ctx(data, running={"root1": {"task": {"id": "root1"}}}),
        _subagent_task("child1", "root1"), "child1",
    )
    events._maybe_checkpoint_coop_on_tree_quiescence(
        _ctx(data, pending=[{"id": "root1", "task": {"id": "root1"}}]),
        _subagent_task("child1", "root1"), "child1",
    )
    assert spawned == []


def test_quiescence_detect_requires_truly_settled_root(tmp_path, monkeypatch):
    """cancel_requested is NOT settled: its cancellation custody is still in
    flight and the root's own terminal event re-triggers later."""
    from supervisor import events
    from ouroboros.task_results import write_task_result

    data = tmp_path / "data"
    data.mkdir()
    write_task_result(data, "root1", "cancel_requested")
    spawned = []
    monkeypatch.setattr(events, "_spawn_coop_checkpoint", lambda *a, **k: spawned.append(a))
    events._maybe_checkpoint_coop_on_tree_quiescence(
        _ctx(data), _subagent_task("child1", "root1"), "child1",
    )
    assert spawned == []
    # Missing root result (never registered) is also not proof of a settled root.
    events._maybe_checkpoint_coop_on_tree_quiescence(
        _ctx(data), _subagent_task("child2", "ghost-root"), "child2",
    )
    assert spawned == []


# --- off-loop execution (real git; spawns subprocesses) ---------------------


@pytest.mark.serial
def test_spawned_checkpoint_commits_budget_dead_tree_off_loop(tmp_path, monkeypatch):
    """End-to-end wave shape: root died budget_exhausted, last child terminal,
    tree dirty — the off-loop run commits it and appends a triggered receipt."""
    from supervisor import events
    from ouroboros.task_results import write_task_result

    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(projects_root))
    data = tmp_path / "data"
    (data / "logs").mkdir(parents=True)

    tree = projects_root / "coop_root1"
    _init_git_repo(tree)
    write_task_result(
        data, "root1", "failed", reason_code="budget_exhausted", title="Sunken city",
    )
    write_task_result(
        data, "child1", "failed",
        delegation_role="subagent", parent_task_id="root1", root_task_id="root1",
        task_constraint={"mode": "acting_subagent", "surface": "external_workspace", "write_root": str(tree)},
    )
    (tree / "paid-work.txt").write_text("do not lose me\n", encoding="utf-8")

    ctx = _ctx(data)
    events._maybe_checkpoint_coop_on_tree_quiescence(
        ctx, _subagent_task("child1", "root1"), "child1",
    )
    # The detect spawned an off-loop thread; join it via the latch drain.
    import time

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        with events._COOP_CHECKPOINT_LOCK:
            busy = "root1" in events._COOP_CHECKPOINT_INFLIGHT
        if not busy:
            break
        time.sleep(0.05)
    assert not busy, "off-loop checkpoint did not finish"

    log_out = subprocess.run(
        ["git", "log", "-1", "--format=%s"], cwd=str(tree),
        capture_output=True, text=True, encoding="utf-8",
    ).stdout
    assert "ouroboros: checkpoint after task root1" in log_out
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(tree), capture_output=True, text=True,
    ).stdout.strip()
    assert status == ""  # the paid work is committed, not a dirty pile
    rows = [
        json.loads(line)
        for line in (data / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    receipts = [r for r in rows if r.get("type") == "coop_checkpoint_commit"]
    assert receipts and receipts[-1]["trigger"] == "tree_quiescence"
    assert receipts[-1]["committed"] is True


@pytest.mark.serial
def test_spawned_checkpoint_revalidates_quiescence_before_git(tmp_path, monkeypatch):
    """A tree member admitted between detect and run wins: the off-loop body
    re-reads liveness and the helper skips entirely."""
    from supervisor import events

    data = tmp_path / "data"
    (data / "logs").mkdir(parents=True)
    ctx = _ctx(data)
    thread = events._spawn_coop_checkpoint(ctx, "rootX", title="", trigger="tree_quiescence")
    assert thread is not None
    thread.join(timeout=30)
    # Now with a racing live member the helper must be invoked with live=True.
    calls = {}

    def _fake_commit(drive_root, root_tid, *, title="", has_live_tree_tasks=False):
        calls["live"] = has_live_tree_tasks
        return []

    import ouroboros.coop_checkpoint as coop

    monkeypatch.setattr(coop, "checkpoint_commit_coop_roots", _fake_commit)
    ctx_live = _ctx(data, running={"sib": {"task": _subagent_task("sib", "rootY")}})
    thread = events._spawn_coop_checkpoint(ctx_live, "rootY", title="", trigger="tree_quiescence")
    assert thread is not None
    thread.join(timeout=30)
    assert calls["live"] is True


@pytest.mark.serial
def test_spawned_checkpoint_revalidation_survives_a_racing_running_pop(tmp_path, monkeypatch):
    """The off-loop re-validation must read PENDING/RUNNING under the queue lock.

    Off the drain thread those are LIVE containers that the drain's own
    ``ctx.RUNNING.pop`` (supervisor/events.py, under ``_queue_lock``), queue
    admission and the worker reaper all mutate. Counting them unlocked raises
    ``RuntimeError: dictionary changed size during iteration``, which the body's
    broad ``except`` converts into a loud-fail receipt and returns WITHOUT
    committing — and since this trigger fires on the LAST live tree member it is
    the last trigger there is, so the coop pile stays uncommitted: exactly the
    defect the off-loop move exists to close.
    """
    import ouroboros.coop_checkpoint as coop
    from supervisor import events
    from supervisor.queue import _queue_lock

    data = tmp_path / "data"
    (data / "logs").mkdir(parents=True)
    running = {f"sib{i}": {"task": _subagent_task(f"sib{i}", "rootR")} for i in range(6)}
    ctx = _ctx(data, running=running)
    observed: dict = {}
    # The predicate and its caller both live in `supervisor.subagent_admission` now,
    # so patch it THERE: a name re-exported through `events` binds at import time and
    # the real caller would never see the substitute.
    from supervisor import subagent_admission

    real_predicate = subagent_admission._is_active_subagent_task

    def _racing_predicate(task, root_task_id):
        # Model a racing pop landing mid-count, as the drain thread does.
        observed.setdefault("lock_held", _queue_lock._is_owned())
        if ctx.RUNNING:
            ctx.RUNNING.pop(next(iter(ctx.RUNNING)), None)
        return real_predicate(task, root_task_id)

    def _fake_commit(drive_root, root_tid, *, title="", has_live_tree_tasks=False):
        observed["live"] = has_live_tree_tasks
        return [{"committed": True, "root": str(root_tid)}]

    monkeypatch.setattr(subagent_admission, "_is_active_subagent_task", _racing_predicate)
    monkeypatch.setattr(coop, "checkpoint_commit_coop_roots", _fake_commit)
    thread = events._spawn_coop_checkpoint(ctx, "rootR", title="", trigger="tree_quiescence")
    assert thread is not None
    thread.join(timeout=30)

    # The convention every other RUNNING reader in supervisor/events.py follows.
    assert observed.get("lock_held") is True, "re-validation must hold supervisor.queue._queue_lock"
    rows = [
        json.loads(line)
        for line in (data / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    receipts = [r for r in rows if r.get("type") == "coop_checkpoint_commit"]
    assert receipts, "the racing pop aborted the run before any commit"
    assert receipts[-1].get("committed") is True
    assert not receipts[-1].get("error")  # not the loud-fail receipt


@pytest.mark.serial
def test_inflight_latch_defers_duplicate_trigger_for_replay(tmp_path):
    """A trigger hitting the latch mid-flight is not lost: it is remembered
    (per root, last-wins) for a single replay after the run completes."""
    from supervisor import events

    data = tmp_path / "data"
    (data / "logs").mkdir(parents=True)
    ctx = _ctx(data)
    with events._COOP_CHECKPOINT_LOCK:
        events._COOP_CHECKPOINT_INFLIGHT.add("rootZ")
    try:
        assert events._spawn_coop_checkpoint(ctx, "rootZ", title="t", trigger="root_done") is None
        with events._COOP_CHECKPOINT_LOCK:
            assert events._COOP_CHECKPOINT_DROPPED.get("rootZ") == {
                "title": "t", "trigger": "root_done",
            }
    finally:
        with events._COOP_CHECKPOINT_LOCK:
            events._COOP_CHECKPOINT_INFLIGHT.discard("rootZ")
            events._COOP_CHECKPOINT_DROPPED.pop("rootZ", None)


@pytest.mark.serial
def test_dropped_quiescence_trigger_is_replayed_after_latch_clear(tmp_path, monkeypatch):
    """G4-4 regression — the lost-trigger interleaving:

    1. an earlier trigger spawns the off-loop worker; it samples liveness and
       sees the LAST child still live, so the helper skips the commit;
    2. that child terminalizes and its ``tree_quiescence`` trigger fires while
       the latch is still held — before the fix it was dropped outright;
    3. the worker clears the latch having committed nothing, and no further
       tree event exists: the quiescence commit never happened.

    The fix replays the dropped trigger once after the latch clears; the
    replayed run re-validates liveness (now zero) and commits."""
    import threading as _threading
    import time

    import ouroboros.coop_checkpoint as coop
    from supervisor import events

    data = tmp_path / "data"
    (data / "logs").mkdir(parents=True)
    # ONE shared ctx, as in production: the drain thread mutates RUNNING live.
    ctx = _ctx(data, running={"lastchild": {"task": _subagent_task("lastchild", "rootQ")}})

    first_call_entered = _threading.Event()
    release_first_call = _threading.Event()
    calls = []

    def _fake_commit(drive_root, root_tid, *, title="", has_live_tree_tasks=False):
        calls.append(has_live_tree_tasks)
        if len(calls) == 1:
            first_call_entered.set()
            release_first_call.wait(timeout=30)
        return []

    monkeypatch.setattr(coop, "checkpoint_commit_coop_roots", _fake_commit)

    t1 = events._spawn_coop_checkpoint(ctx, "rootQ", title="wave", trigger="root_done")
    assert t1 is not None
    assert first_call_entered.wait(timeout=30)
    assert calls == [True]  # worker sampled the last child while it was live

    # The last child terminalizes NOW: the drain thread pops it from RUNNING and
    # its quiescence trigger hits the still-held latch.
    ctx.RUNNING.clear()
    assert events._spawn_coop_checkpoint(
        ctx, "rootQ", title="wave", trigger="tree_quiescence",
    ) is None

    release_first_call.set()
    t1.join(timeout=30)

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        with events._COOP_CHECKPOINT_LOCK:
            busy = (
                "rootQ" in events._COOP_CHECKPOINT_INFLIGHT
                # getattr: on pre-fix code the memo does not exist — let the
                # semantic assert below report the lost trigger instead.
                or "rootQ" in getattr(events, "_COOP_CHECKPOINT_DROPPED", {})
            )
        if not busy and len(calls) >= 2:
            break
        time.sleep(0.02)
    assert calls == [True, False], (
        "dropped tree_quiescence trigger was not replayed after the latch cleared"
    )


def test_root_done_path_defers_to_quiescence_when_children_live(tmp_path, monkeypatch):
    """The root-done handler must NOT permanently skip a live tree — the skip
    is now a deferral to the quiescence trigger."""
    from supervisor import events

    data = tmp_path / "data"
    data.mkdir()
    spawned = []
    monkeypatch.setattr(events, "_spawn_coop_checkpoint", lambda *a, **k: spawned.append(a))
    running = {"child": {"task": _subagent_task("child", "root1")}}
    events._checkpoint_coop_roots_on_root_done(
        _ctx(data, running=running), {"id": "root1", "root_task_id": "root1"}, "root1",
    )
    assert spawned == []  # deferred, not executed
    events._checkpoint_coop_roots_on_root_done(
        _ctx(data), {"id": "root1", "root_task_id": "root1"}, "root1",
    )
    assert [s[1] for s in spawned] == ["root1"]
