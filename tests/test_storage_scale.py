"""Real-reader counts for filtered storage reads; no wall-clock gates."""

import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ouroboros import store, swarm_rollup, task_status, task_store
from ouroboros.task_finalization import build_swarm_efficiency
from ouroboros.task_results import list_task_results, write_task_result
from ouroboros.task_status import find_child_tasks
from ouroboros.utils import append_jsonl
from tests._store_journal_shared import held_open, tree_state

pytestmark = pytest.mark.scale


def _decoded_rows(monkeypatch):
    """Count the documents each ``task_store._decode`` call decodes (its input is captured pages)."""
    decoded, real = [], task_store._decode

    def decode(pages):
        documents = real(pages)
        decoded.append(len(documents))
        return documents

    monkeypatch.setattr(task_store, "_decode", decode)
    return decoded


@pytest.mark.parametrize("count", [1, 64, 256])
def test_listing_decodes_only_the_rows_the_status_prefilter_selects(tmp_path, monkeypatch, count):
    for index in range(count):
        write_task_result(tmp_path, f"task-{index}", "completed")
    decoded = _decoded_rows(monkeypatch)
    assert list_task_results(tmp_path, statuses=["running"]) == []
    assert decoded == [0]
    assert len(list_task_results(tmp_path)) == count
    assert decoded == [0, count]


def test_child_lookup_selects_only_the_lineage_rows_on_every_call(tmp_path, monkeypatch):
    for index in range(32):
        write_task_result(tmp_path, f"task-{index}", "completed", delegation_role="subagent",
                          parent_task_id="parent" if index == 0 else "other", root_task_id="parent" if index == 0 else "other")
    decoded = _decoded_rows(monkeypatch)
    for _ in range(2):
        assert [row["task_id"] for row in find_child_tasks(tmp_path, parent_task_id="parent", materialize_artifacts=False)] == ["task-0"]
    assert decoded == [1, 1]


def test_swarm_rollup_reads_one_event_and_resolves_depth_once(tmp_path, monkeypatch):
    append_jsonl(tmp_path / "logs/events.jsonl", {"type": "swarm_fanout", "parent_task_id": "parent", "task_ids": ["child"]})
    decode = Mock(wraps=json.loads)
    children = Mock(wraps=task_status.find_child_tasks)
    monkeypatch.setattr(swarm_rollup, "json", SimpleNamespace(loads=decode))
    monkeypatch.setattr(task_status, "find_child_tasks", children)
    rollup = build_swarm_efficiency(SimpleNamespace(drive_root=tmp_path), {"id": "parent"})
    assert rollup["subagent_count"] == 1
    assert decode.call_count == 1
    children.assert_called_once_with(tmp_path, parent_task_id="parent", root_task_id="parent",
                                     scope="subtree", materialize_artifacts=False)


def test_reader_failure_propagates_and_next_call_succeeds(tmp_path, monkeypatch):
    from ouroboros import task_results

    write_task_result(tmp_path, "task", "completed")
    with monkeypatch.context() as broken:
        def refuse(*args, **kwargs):
            raise OSError("fixture unreadable")
        broken.setattr(task_results, "list_documents", refuse)
        with pytest.raises(OSError, match="fixture unreadable"):
            list_task_results(tmp_path)
    assert len(list_task_results(tmp_path)) == 1


@pytest.mark.parametrize("existing_journal", [False, True])
@pytest.mark.parametrize("another_root", [False, True])
def test_listing_does_not_write_to_runtime_or_inspected_root(tmp_path, monkeypatch, existing_journal, another_root):
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    inspected = tmp_path / "inspected" if another_root else tmp_path
    write_task_result(inspected, "task", "completed")
    if existing_journal:
        append_jsonl(inspected / "logs/events.jsonl", {"type": "original_evidence"})
    with held_open(store.store_path(inspected)):  # SQLite's journal set-up falls outside the window
        before = tree_state(tmp_path, mtimes=True, directories=False)
        assert len(list_task_results(inspected)) == 1
        assert tree_state(tmp_path, mtimes=True, directories=False) == before


def test_a_listing_of_three_hundred_thousand_entries_is_walked_completely(tmp_path, monkeypatch):
    """A1 at the scale the old entry bound (250 000) refused: the blob folder of a real install.
    The walker's own loop meets 300 000 regular-file entries of one real folder (the listing is
    served to it, its stats the shape of one real one-byte file) and counts every one: no entry,
    time or depth bound. Seconds, not a disk full of files; the real-disk listing path is proved
    by the dashboard tests' deep-and-wide tree."""
    from ouroboros import storage

    data = tmp_path / "data"
    blobs = data / "observability" / "blobs"
    blobs.mkdir(parents=True)
    sample = blobs / "000000.gz"
    sample.write_bytes(b"g")
    shape = sample.lstat()
    identity = (blobs.stat().st_dev, blobs.stat().st_ino)
    real_scandir = os.scandir

    class Entry:
        __slots__ = ("name",)

        def __init__(self, name):
            self.name = name

        def stat(self, follow_symlinks=True):
            return shape

    def served(target=None):
        handle = "." if target is None else target
        try:
            own = os.fstat(handle) if isinstance(handle, int) else os.stat(handle)
        except OSError:
            return real_scandir(handle)
        if (own.st_dev, own.st_ino) != identity:
            return real_scandir(handle)
        return (Entry(f"{index:06x}.gz") for index in range(300_000))  # next() and close(), as the walk uses them

    monkeypatch.setattr(os, "scandir", served)
    monkeypatch.setattr(storage, "_listing_stat", lambda entry, _fd: entry.stat(follow_symlinks=False))
    table = [{"id": "model_calls", "group": "records", "roots": [(str(data), ("observability",))]}]
    record = storage.measure(data, places=table, sleep=lambda _s: None)
    [place] = record["places"]
    assert (place["files"], place["logical_bytes"], place["unreadable"]) == (300_000, 300_000, 0)
    assert place["parts"] == [{"name": "blobs", "bytes": place["bytes"], "files": 300_000}]
    assert record["files"] == 300_000
