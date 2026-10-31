"""Every Continue episode carries its work: the first root's id and its name.

Admission stamps ``metadata.continuation.work_id`` and ``work_name``. A stamped
predecessor names the work; an older chain is walked back over its recorded
``predecessor_task_id`` edges to the first root, ending at the last episode read
when an edge names a record that cannot be read (as the work projection resolves
the same records). The name is the one its card shows (coined, else authored)
and the successor's title carries it under exactly one ``Continue: `` prefix.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ouroboros.task_results import load_task_result
from tests._budget_pause_exact_helpers import _install_queue
from tests._store_shared import delete_result, seed_result, unreadable_result
from tests.test_owner_continue import NONCE, _interrupted

SOURCES = {"original": {"content": "Write the Friday report"}, "later": []}


def _successor(root, predecessor, successor="c-next"):
    """The task a Continue of ``predecessor`` would admit, built from its STORED record."""
    from supervisor.continuation_admission import _successor_task

    return _successor_task(SimpleNamespace(DRIVE_ROOT=root), predecessor, load_task_result(root, predecessor) or {},
                           {"successor_task_id": successor, "chat_id": 7}, {"cause": "context_overflow"}, SOURCES, {})


def _work(task):
    continuation = task["metadata"]["continuation"]
    return continuation["work_id"], continuation["work_name"]


def _root(root, task_id="root-1"):
    seed_result(root, task_id, {"status": "cancelled", "title": "Write the Friday report",
                                "suggested_name": "Weekly numbers", "root_task_id": task_id})


def _legacy_episode(root, task_id, predecessor, name="Continue: Continue: Weekly numbers"):
    """A Continue recorded before the stamp: only its predecessor edge, its title stacked."""
    seed_result(root, task_id, {"status": "failed", "title": name, "suggested_name": name, "root_task_id": task_id,
                                "metadata": {"continuation": {"predecessor_task_id": predecessor,
                                                              "cause": "context_overflow"}}})


def test_admitted_continues_stamp_the_first_root_and_the_stamp_outlives_its_record(tmp_path, monkeypatch):
    from supervisor.continuation_admission import admit_continuation

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, "pred-1", suggested_name="Weekly numbers")
    ack = admit_continuation("pred-1", action_nonce=NONCE)
    assert ack["ok"] is True, ack
    first = next(task for task in workers.PENDING if task["id"] == ack["successor_task_id"])
    stored = load_task_result(tmp_path, first["id"])
    # The card's coined name, not the authored title; one prefix.
    assert _work(first) == _work(stored) == ("pred-1", "Weekly numbers")
    assert first["title"] == first["suggested_name"] == stored["title"] == "Continue: Weekly numbers"

    second = _successor(tmp_path, first["id"])
    assert _work(second) == ("pred-1", "Weekly numbers")
    assert second["title"] == second["suggested_name"] == "Continue: Weekly numbers"
    assert second["metadata"]["continuation"]["predecessor_task_id"] == first["id"]
    # The stamp is the authority: the first root's record is no longer needed.
    assert delete_result(tmp_path, "pred-1")
    assert _work(_successor(tmp_path, first["id"])) == ("pred-1", "Weekly numbers")


def test_a_legacy_chain_resolves_over_its_recorded_edges_and_never_stacks_prefixes(tmp_path):
    _root(tmp_path)
    _legacy_episode(tmp_path, "c1", "root-1")
    _legacy_episode(tmp_path, "c2", "c1", name="Continue: Weekly numbers")
    for predecessor in ("c1", "c2"):
        task = _successor(tmp_path, predecessor)
        assert _work(task) == ("root-1", "Weekly numbers")
        assert task["title"] == task["suggested_name"] == "Continue: Weekly numbers"
    # A plain root predecessor is its own work.
    assert _work(_successor(tmp_path, "root-1")) == ("root-1", "Weekly numbers")


@pytest.mark.parametrize("gap", ["missing", "unreadable"])
def test_an_edge_to_a_record_that_cannot_be_read_ends_at_the_last_episode_read(tmp_path, monkeypatch, gap):
    _root(tmp_path)
    _legacy_episode(tmp_path, "c1", "root-1")
    _legacy_episode(tmp_path, "c2", "c1")
    if gap == "missing":
        assert delete_result(tmp_path, "root-1")
    else:
        unreadable_result(monkeypatch, "root-1", root=tmp_path)
    task = _successor(tmp_path, "c2")
    assert _work(task) == ("c1", "Weekly numbers")
    assert task["title"] == "Continue: Weekly numbers"


def test_a_corrupted_cycle_ends_at_the_predecessor(tmp_path):
    _legacy_episode(tmp_path, "c1", "c0")
    _legacy_episode(tmp_path, "c0", "c1", name="Continue: Other")
    assert _work(_successor(tmp_path, "c1")) == ("c1", "Weekly numbers")


def test_a_name_less_chain_keeps_the_older_title_fallbacks(tmp_path):
    seed_result(tmp_path, "root-1", {"status": "cancelled", "root_task_id": "root-1"})
    _legacy_episode(tmp_path, "c1", "root-1", name="Continue: Continue: ")
    assert _work(_successor(tmp_path, "c1")) == ("root-1", "c1")
    assert _successor(tmp_path, "c1")["title"] == "Continue: c1"
