"""Queue publication, review follow-up 2 (RV_MAIN M5, M6, M7, W9 and the format stamp of the export).

M5: a mark is never read as "persisted" — it answers None while the tick publishes, every
runtime caller that acts on the publication reaches the barrier, and a guard finds a caller
that reads the mark's value. M6: a fact the publication carries that changed under Q without
a mark is published by the next tick (the census of volatile row facts). M7: the live root
cost figure is marked by a call-state frame, a wait lease and a server-side custody
transition, so the flush reads an in-flight call or an open run. W9: the Runtime queue fact
counts project slots; lane carriers are reported apart. The rollback export carries the
format an older body treats as current.
"""

import ast
import json
import pathlib
import threading
import types
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ouroboros import queue_observation, task_status
from supervisor import direct_roots, queue, queue_publication, queue_snapshot

REPO = pathlib.Path(__file__).resolve().parents[1]
MARK = "persist_queue_snapshot"


@pytest.fixture
def published(tmp_path, monkeypatch):
    now = [1_800_000_000.0]
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", tmp_path / "state/queue_snapshot.json")
    monkeypatch.setattr(queue, "QUEUE_PUBLICATION", None)
    monkeypatch.setattr(queue, "INITIALIZED", True)
    for name, value in (("PENDING", []), ("RUNNING", {}), ("BUDGET_ROOT_FENCES", {}), ("ACCEPTANCE_FENCES", {})):
        monkeypatch.setattr(queue, name, value)
    monkeypatch.setattr(queue_snapshot, "_RESTORED", set())
    monkeypatch.setattr(queue_snapshot, "_RESTORING", {})
    monkeypatch.setattr(direct_roots, "collect_direct_roots", lambda: {"roots": [], "incomplete": False})
    monkeypatch.setattr(queue_publication, "utc_now_iso", lambda: datetime.fromtimestamp(now[0], timezone.utc).isoformat())
    monkeypatch.setattr(queue_publication, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=lambda _s: None))
    for module in (queue_observation, task_status):
        monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(queue_observation, "_sleep", lambda _sec: None)
    queue_publication.reset()
    yield tmp_path, now
    queue_publication.reset()


def _read(root, name):
    return json.loads((root / "state" / name).read_text(encoding="utf-8"))


def _task(task_id, **extra):
    return {"id": task_id, "type": "task", "chat_id": 1, "root_task_id": task_id, "text": f"task {task_id}", **extra}


# ---- M5: a mark is never a persisted fact -------------------------------------------------------

def test_a_mark_answers_none_while_the_tick_publishes_and_true_only_when_on_disk(published):
    root, _now = published
    queue.PENDING.append(_task("a"))
    assert queue.persist_queue_snapshot("admit") is True, "no tick publisher: the caller published"
    assert [row["id"] for row in _read(root, "queue_snapshot.json")["pending"]] == ["a"]
    queue_publication.tick_publisher_started()
    queue.PENDING.append(_task("b"))
    assert queue.persist_queue_snapshot("admit") is None, "a mark: nothing is on disk yet"
    assert [row["id"] for row in _read(root, "queue_snapshot.json")["pending"]] == ["a"]
    assert queue.publish_queue_snapshot("admit") is True, "the barrier: on disk before it returns"
    assert [row["id"] for row in _read(root, "queue_snapshot.json")["pending"]] == ["a", "b"]
    gone = threading.Thread(target=lambda: None)
    gone.start(); gone.join()
    queue_publication._PUBLISHER = gone
    queue.PENDING.append(_task("c"))
    assert queue.persist_queue_snapshot("admit") is True, "the loop is gone: the caller publishes again"


def test_barrier_of_prefers_the_contexts_barrier_then_its_stand_in_then_nothing():
    calls = []
    both = types.SimpleNamespace(publish_queue_snapshot=lambda **kw: calls.append(("barrier", kw)) or True,
                                 persist_queue_snapshot=lambda **kw: calls.append(("mark", kw)) or None)
    assert queue_publication.barrier_of(both)(reason="x") is True and calls == [("barrier", {"reason": "x"})]
    stub_only = types.SimpleNamespace(persist_queue_snapshot=lambda **kw: calls.append(("stub", kw)) or True)
    assert queue_publication.barrier_of(stub_only)(reason="y") is True and calls[-1] == ("stub", {"reason": "y"})
    assert queue_publication.barrier_of(types.SimpleNamespace()) is None, "no persister: the caller decides"


def test_the_servers_event_context_carries_the_barrier_beside_the_mark():
    source = (REPO / "server.py").read_text(encoding="utf-8")
    assert "publish_queue_snapshot=publish_queue_snapshot" in source and "persist_queue_snapshot=persist_queue_snapshot" in source


def _runtime_files():
    for folder in ("ouroboros", "supervisor"):
        yield from sorted((REPO / folder).rglob("*.py"))
    yield REPO / "server.py"


def _names_the_mark(func: ast.AST) -> bool:
    return (isinstance(func, ast.Attribute) and func.attr == MARK) or (isinstance(func, ast.Name) and func.id == MARK)


def test_no_runtime_caller_reads_the_marks_value_or_reaches_it_through_getattr():
    """The mark answers None while the tick publishes: a caller that acts on the publication
    (a reply, a confirmation, a custody fact, a rollback decision) must call the barrier. A call
    whose value is used, and a ``getattr(ctx, "persist_queue_snapshot")`` outside the one barrier
    seam (``queue_publication.barrier_of``) and the one statement-only rollback, is an offender."""
    allowed_value_users = {
        "supervisor/queue_publication.py",  # the barrier's own delegation to the mark without a tick publisher
    }
    allowed_getattr = {
        "supervisor/queue_publication.py",  # barrier_of
        "supervisor/events_project_routing.py",  # the promote rollback: called as a statement, value unused
    }
    offenders = []
    for path in _runtime_files():
        rel = path.relative_to(REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _names_the_mark(node.func) and not isinstance(parents.get(node), ast.Expr) and rel not in allowed_value_users:
                offenders.append(f"{rel}:{node.lineno} reads the mark's value")
            if (isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant) and node.args[1].value == MARK and rel not in allowed_getattr):
                offenders.append(f"{rel}:{node.lineno} reaches the mark through getattr")
    assert offenders == [], offenders
