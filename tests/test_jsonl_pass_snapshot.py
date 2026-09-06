"""A logical source pass reads chain metadata once and closes every read batch."""

import io
import json
import os
import pathlib

import pytest

from ouroboros import utils
from ouroboros.gateway.task_events import _TaskEventCursorFollower
from tests.test_task_event_cursor import append, content, seed


@pytest.mark.parametrize("archives,records", [(40, 1), (100, 1), (20, 4)])
def test_cold_metadata_reads_are_linear_in_archives_and_not_multiplied_by_batches(tmp_path, monkeypatch, archives, records):
    root = seed(tmp_path / "root")
    directory = root / "archive"
    directory.mkdir()
    payload = "row" if records == 1 else "x" * 40000
    raw = (json.dumps({"task_id": "root", "content": payload}) + "\n").encode()
    for index in range(archives):
        (directory / f"progress_20260101T{index:06d}.jsonl").write_bytes(raw * records)
    append(root, "live")
    counts = {"stat": 0, "enumerate": 0}
    real_stat, real_segments = pathlib.Path.stat, utils.jsonl_archive_segments
    def stat(path, *args, **kwargs):
        if path.parent == directory:
            counts["stat"] += 1
        return real_stat(path, *args, **kwargs)
    def segments(path, **kwargs):
        if path.name == "progress.jsonl":
            counts["enumerate"] += 1
        return real_segments(path, **kwargs)
    monkeypatch.setattr(pathlib.Path, "stat", stat)
    monkeypatch.setattr(utils, "jsonl_archive_segments", segments)
    follower = _TaskEventCursorFollower(root, "root", {"v": 2, "seq": 0, "view": "", "positions": {}})
    follower.refresh_view()
    assert len(list(follower.read_events())) == archives * records + 1
    assert counts == {"stat": archives, "enumerate": 1}


def test_original_live_rebinds_once_while_newer_generations_keep_rotating(tmp_path, monkeypatch):
    root = seed(tmp_path / "root")
    directory = root / "archive"
    directory.mkdir()
    for index in range(6):
        append(root, str(index) * 40000)
    append(root, "tool", source="tools")
    calls = []
    real = utils.jsonl_archive_segments
    def segments(path, **kwargs):
        if path.name == "progress.jsonl":
            calls.append(path)
        return real(path, **kwargs)
    monkeypatch.setattr(utils, "jsonl_archive_segments", segments)
    follower = _TaskEventCursorFollower(root, "root", {"v": 2, "seq": 0, "view": "", "positions": {}})
    follower.refresh_view()
    observed = []
    for index, row in enumerate(follower.read_events()):
        observed.append(row)
        if row["source"] == "progress":
            live = root / "logs" / "progress.jsonl"
            os.replace(live, directory / f"progress_20260101T{index:06d}.jsonl")
            live.touch()
            append(root, "next-pass")
    assert content(observed) == [str(index) * 40000 for index in range(6)] + ["tool"]
    assert len(calls) == 2  # one snapshot and one original-generation rebind
    assert content(list(follower.read_events())) == ["next-pass"] * 6


@pytest.mark.parametrize("change", ["delete", "shrink"])
def test_required_unread_segment_failure_never_advances_the_cursor(tmp_path, change):
    root = seed(tmp_path / "root")
    directory = root / "archive"
    directory.mkdir()
    raw = b'{"task_id":"root","content":"archived"}\n'
    first, second = [directory / f"progress_20260101T00000{index}.jsonl" for index in (1, 2)]
    first.write_bytes(raw)
    second.write_bytes(raw)
    follower = _TaskEventCursorFollower(root, "root", {"v": 2, "seq": 0, "view": "", "positions": {}})
    follower.refresh_view()
    rows = follower.read_events()
    next(rows)
    checkpoint = follower.checkpoint()
    second.unlink() if change == "delete" else second.write_bytes(b"x")
    with pytest.raises(utils.JsonlChainUnreadable):
        next(rows)
    assert follower.checkpoint() == checkpoint


def test_snapshot_is_bound_to_one_source(tmp_path):
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    first.write_bytes(b"{}\n")
    second.write_bytes(b"{}\n")
    state = {}
    with utils.jsonl_chain_handles(first, strict=True, start_offset=0, snapshot=state) as handles:
        assert len(handles) == 1
    with pytest.raises(ValueError, match="different source"):
        with utils.jsonl_chain_handles(second, strict=True, start_offset=0, snapshot=state):
            pass


def test_shared_parser_preserves_borrowed_handle_ownership_and_read_errors(tmp_path):
    handle = io.BytesIO(b'{"ok":1}\ninvalid\n[1]\n{"text":"\xff"}\n')
    gaps = set()
    rows = list(utils.iter_jsonl_objects(tmp_path / "unused", _handle=handle, gap_reasons=gaps))
    assert rows == [{"ok": 1}, {"text": "\ufffd"}]
    assert gaps == {"malformed_jsonl", "invalid_jsonl_row", "unreadable_bytes"}
    assert not handle.closed
    handle.close()
    class LostStream:
        def __iter__(self):
            raise FileNotFoundError("borrowed source lost")
    with pytest.raises(FileNotFoundError, match="borrowed source lost"):
        list(utils.iter_jsonl_objects(tmp_path / "unused", _handle=LostStream()))
