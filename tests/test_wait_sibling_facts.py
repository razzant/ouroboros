"""The sibling advisory reads indexed, stored facts, not unrelated task bodies."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ouroboros import store, task_store, task_tree_ledger
from ouroboros.task_results import (
    STATUS_CANCELLED, STATUS_COMPLETED, STATUS_FAILED, STATUS_INTERRUPTED,
    STATUS_REJECTED_DUPLICATE, STATUS_REQUESTED, STATUS_RUNNING, STATUS_SCHEDULED,
    write_task_result,
)
from ouroboros.tools import control_task_results as mod
from tests.test_hot_facts_tree import _measure

PARENT, OTHER, CHILD = "wsf-parent", "wsf-other", "wsf-child"
UNRELATED = "UNRELATED_BODY" * 100
ADVISORY = "[ADVISORY] {} other child(ren) still running/scheduled"


@pytest.fixture(autouse=True)
def _tree_ledger_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(task_tree_ledger, "DATA_DIR", str(tmp_path))


def _ctx(drive, **fields):
    return SimpleNamespace(drive_root=drive, **{"task_id": PARENT, **fields})


def _row(drive, task_id, status, **fields):
    return write_task_result(drive, task_id, status, **{"result": UNRELATED, **fields})


def _family(drive):
    write_task_result(drive, CHILD, STATUS_COMPLETED, parent_task_id=PARENT,
                      delegation_role="subagent", result="child answer")
    _row(drive, "wsf-run", STATUS_RUNNING, parent_task_id=PARENT, delegation_role="subagent")
    _row(drive, "wsf-sched", STATUS_SCHEDULED, parent_task_id=PARENT)


def _wait(drive):
    return mod._wait_for_task(_ctx(drive), CHILD, timeout_sec=0)


def _count(drive, excluded=CHILD, ctx=None):
    return mod._count_live_sibling_children(ctx or _ctx(drive), drive, exclude_task_id=excluded)


def test_the_real_wait_decodes_no_unrelated_body_from_20_to_2000_foreign_rows(tmp_path, monkeypatch):
    roots = [tmp_path / "small", tmp_path / "large"]
    for root, foreign in zip(roots, (20, 2000)):
        root.mkdir()
        _family(root)
        exemplar = _row(root, "wsf-foreign-0000", STATUS_RUNNING,
                        parent_task_id=OTHER, delegation_role="subagent")
        with store.open_store(root, write=True) as txn:
            for index in range(1, foreign):
                txn.store_task({**exemplar, "task_id": f"wsf-foreign-{index:04}"})
    for root in roots:
        text, counts = _measure(root, monkeypatch, _wait)
        assert text.startswith("Task wait completed") and "child answer" in text
        assert ADVISORY.format(2) in text
        # The real wait may read its selected child, never its siblings or foreign bodies.
        assert counts["unrelated_bytes"] == 0, counts


def test_an_unreadable_or_missing_body_no_longer_hides_the_advisory(tmp_path):
    _family(tmp_path)
    _row(tmp_path, "wsf-req", STATUS_REQUESTED, parent_task_id=PARENT)
    _row(tmp_path, "wsf-foreign", STATUS_RUNNING, parent_task_id=OTHER)
    with store.open_store(tmp_path, write=True) as txn:
        txn.conn.executemany("UPDATE task_bodies SET body='{not json' WHERE task_id=?",
                             [("wsf-foreign",), ("wsf-run",)])
        txn.conn.execute("DELETE FROM task_bodies WHERE task_id=?", ("wsf-req",))
    assert ADVISORY.format(3) in _wait(tmp_path)


def test_each_wait_reports_the_siblings_live_at_its_own_read(tmp_path):
    write_task_result(tmp_path, CHILD, STATUS_COMPLETED, parent_task_id=PARENT, result="child answer")
    _row(tmp_path, "wsf-foreign", STATUS_RUNNING, parent_task_id=OTHER)
    assert "[ADVISORY]" not in _wait(tmp_path)
    _row(tmp_path, "wsf-run", STATUS_RUNNING, parent_task_id=PARENT)
    assert ADVISORY.format(1) in _wait(tmp_path)
    write_task_result(tmp_path, "wsf-run", STATUS_COMPLETED)
    assert "[ADVISORY]" not in _wait(tmp_path)


def test_this_parents_rows_count_by_stored_status_in_any_role(tmp_path, monkeypatch):
    _row(tmp_path, "wsf-run", STATUS_RUNNING, parent_task_id=PARENT, delegation_role="subagent")
    _row(tmp_path, "wsf-sched", STATUS_SCHEDULED, parent_task_id=PARENT)
    _row(tmp_path, "wsf-req", STATUS_REQUESTED, parent_task_id=PARENT, delegation_role="subagent")
    # Pause does not rewrite the stored running status; it is not this advice's authority.
    _row(tmp_path, "wsf-paused", STATUS_RUNNING, parent_task_id=PARENT, budget_pause={"state": "paused"})
    for status in (STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED, STATUS_INTERRUPTED, STATUS_REJECTED_DUPLICATE):
        _row(tmp_path, f"wsf-{status}", status, parent_task_id=PARENT)
    _row(tmp_path, PARENT, STATUS_RUNNING)
    _row(tmp_path, "wsf-grandchild", STATUS_RUNNING, parent_task_id="wsf-run", root_task_id=PARENT)
    _row(tmp_path, "wsf-foreign", STATUS_RUNNING, parent_task_id=OTHER, delegation_role="subagent")
    _row(tmp_path, "wsf-root", STATUS_SCHEDULED)
    total, counts = _measure(tmp_path, monkeypatch, lambda drive: _count(drive, "wsf-none"))
    assert total == 4 and counts["unrelated_bytes"] == 0, counts
    assert _count(tmp_path, "wsf-paused") == _count(tmp_path, "wsf-req") == 3
    assert _count(tmp_path, "wsf-completed") == _count(tmp_path, PARENT) == 4


def test_a_padded_stored_parent_counts_like_its_typed_column(tmp_path):
    # Explicit change: the old body comparison missed this canonical parent identity.
    _row(tmp_path, "wsf-padded", STATUS_RUNNING, parent_task_id=f" {PARENT} ")
    assert task_store.read_document(tmp_path, "wsf-padded")["parent_task_id"] == f" {PARENT} "
    assert task_store.task_facts(tmp_path)["wsf-padded"]["parent_task_id"] == PARENT
    assert _count(tmp_path) == 1


def test_no_parent_or_unavailable_facts_count_zero_without_raising(tmp_path, monkeypatch):
    _row(tmp_path, "wsf-run", STATUS_RUNNING, parent_task_id=PARENT)

    def busy(*_args, **_kwargs):
        raise store.StoreBusy("held by another writer")

    with monkeypatch.context() as patch:
        patch.setattr(task_store, "open_store", lambda *_a, **_k: pytest.fail("a parentless count read the store"))
        assert _count(tmp_path, ctx=_ctx(tmp_path, task_id="")) == 0
        assert _count(tmp_path, ctx=SimpleNamespace(drive_root=tmp_path)) == 0
        patch.setattr(task_store, "open_store", busy)
        assert _count(tmp_path) == 0
    assert _count(tmp_path) == 1
