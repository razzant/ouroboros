"""Whole-chain custody reads: every rotated segment or a typed refusal, never part of the history.

The importer's fold, the legacy catch-up, the explicit rebuild and the forensic replay read every
``archive/events_*.jsonl`` segment and the live file. They hold the live handle pinned at entry and
one archive at a time, so a long chain under a low descriptor limit is read whole; a required
segment that cannot be opened or read raises before the rebuild replaces any retained row.
"""
from __future__ import annotations

import errno
import json
import os
import pathlib
import subprocess
import sys
from contextlib import contextmanager

import pytest

from ouroboros import delegate_custody as c
from ouroboros import delegate_custody_current as current
from ouroboros import delegate_invocation as inv
from ouroboros import store, utils
from ouroboros.store_import import ensure_current_store

REPO = pathlib.Path(__file__).resolve().parents[1]


def _row(run_id, kind=c.STARTED, **facts):
    return {"ts": "2026-01-01T00:00:00+00:00", "type": kind, "event_id": f"event-{run_id}",
            "run_id": run_id, "task_id": f"task-{run_id}", **facts}


def _line(row):
    return (json.dumps(row) + "\n").encode()


def _chain(root, archives, *, live=("run-live",), empty=()):
    """``archives`` rotated segments holding one STARTED row each (``empty`` indexes hold none)
    and a live file with ``live``'s rows (``None``: no live file). Returns the live path and the
    run ids in chain order."""
    path, expected = c.event_log_path(root), []
    for index in range(archives):
        segment = root / "archive" / f"events_20260101T{index:06d}.jsonl"
        segment.parent.mkdir(parents=True, exist_ok=True)
        segment.write_bytes(b"" if index in empty else _line(_row(f"run-{index:03d}")))
        expected += [] if index in empty else [f"run-{index:03d}"]
    if live is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"".join(_line(_row(run_id)) for run_id in live))
        expected += list(live)
    return path, expected


def _store_facts(root):
    with store.open_store(root, write=False) as txn:
        return {table: txn.conn.execute(f"SELECT * FROM {table} ORDER BY 1, 2").fetchall()
                for table in ("custody_runs", "obligations", "meta")}


def _fail_open(monkeypatch, lost):
    real = pathlib.Path.open

    def failing(self, *args, **kwargs):
        if self == lost:
            raise OSError(errno.EIO, "Input/output error")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", failing)


class _Unreadable:
    def __iter__(self):
        raise OSError(errno.EIO, "Input/output error")


def _fail_read(monkeypatch, lost):
    real = utils.jsonl_chain_handles

    @contextmanager
    def failing(chain, **kwargs):
        with real(chain, **kwargs) as handles:
            yield ((segment, _Unreadable() if segment == lost else handle) for segment, handle in handles)

    monkeypatch.setattr(utils, "jsonl_chain_handles", failing)


_LOW_LIMIT_CHILD = """
import json, resource, sys
from pathlib import Path
from ouroboros import delegate_custody as c, delegate_custody_current as current, maintenance_jobs, store_import

root, consumer = Path(sys.argv[1]), sys.argv[2]
resource.setrlimit(resource.RLIMIT_NOFILE, (int(sys.argv[3]), resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
if consumer == "custody_rows":
    print(json.dumps([row["run_id"] for row in c.custody_rows(root)]))
    raise SystemExit
if consumer == "rebuild":
    current.rebuild(root, replace=True)
elif consumer == "import":
    store_import.ensure_current_store(root)
else:  # the legacy catch-up a maintenance pass runs over an old body's appends
    maintenance_jobs.record(root, "legacy_reconcile", state="pending", cursor=None, done=0, total=None)
    store_import.run_legacy_reconcile(root, should_stop=lambda: False)
print(json.dumps(sorted(c.replay(root))))
"""


@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_NOFILE is a POSIX process limit")
@pytest.mark.parametrize("consumer", ["rebuild", "custody_rows", "import", "catch_up"])
def test_every_whole_chain_consumer_reads_eighty_archives_and_live_under_a_64_descriptor_limit(tmp_path, consumer):
    """Only the child's own soft limit is lowered; this process and the host keep theirs."""
    from tests.test_store_import import result, write_result

    root = tmp_path / "data"
    root.mkdir()
    if consumer == "catch_up":
        ensure_current_store(root)  # published before an old body's appends: the boundary is 0
        write_result(root, result("fresh"))  # the old body's result change that makes the pass reconcile
    _path, expected = _chain(root, 80)
    if consumer == "rebuild":
        ensure_current_store(root)  # the retained rows an explicit rebuild replaces
        assert sorted(c.replay(root)) == expected
    child = subprocess.run([sys.executable, "-c", _LOW_LIMIT_CHILD, str(root), consumer, "64"], cwd=REPO,
                           env={**os.environ, "PYTHONPATH": str(REPO)}, capture_output=True, text=True,
                           timeout=300)
    assert child.returncode == 0, child.stderr[-4000:]
    assert json.loads(child.stdout.splitlines()[-1]) == expected


def test_a_whole_custody_read_holds_only_the_pinned_live_handle_and_one_archive(tmp_path, monkeypatch):
    path, expected = _chain(tmp_path, 6)
    opened = []
    real = pathlib.Path.open

    def tracked(self, *args, **kwargs):
        handle = real(self, *args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(pathlib.Path, "open", tracked)
    seen, peak = [], 0
    for row in c._iter_rows(path, strict=True):
        seen.append(row["run_id"])
        peak = max(peak, sum(not handle.closed for handle in opened))
    assert seen == expected and len(opened) == len(expected)  # each segment opened once
    assert peak == 2 and all(handle.closed for handle in opened)
    reader = c._iter_rows(path, strict=True)
    assert [next(reader)["run_id"], next(reader)["run_id"]] == expected[:2]
    reader.close()  # an abandoned pass releases the live handle and its archive
    assert all(handle.closed for handle in opened)


@pytest.mark.parametrize("failure", ["open", "read"])
def test_rebuild_refuses_a_segment_failing_after_the_probe_and_keeps_every_retained_fact(tmp_path, monkeypatch, failure):
    from ouroboros.obligations import ObligationsUnavailable

    path, expected = _chain(tmp_path, 4, live=())
    path.write_bytes(_line(_row("run-live", invocation_id="inv-live"))
                     + _line({**_row("inv-pending", c.START_REQUESTED, request={"prompt": "p"}),
                              "run_id": "", "invocation_id": "inv-pending"}))
    ensure_current_store(tmp_path)
    retained = _store_facts(tmp_path)
    assert sorted(c.replay(tmp_path)) == [*expected, "run-live"]
    assert [row["invocation_id"] for row in c.pending_invocations(tmp_path)] == ["inv-pending"]
    with store.open_store(tmp_path, write=False) as txn:
        assert txn.meta(inv.COMPLETE) is True and len(txn.set_rows(inv.EVENTS)) == 6
    lost = sorted((tmp_path / "archive").glob("events_*.jsonl"))[1]
    monkeypatch.setattr(c, "custody_chain_unreadable", lambda root: False)  # the probe saw it readable
    (_fail_open if failure == "open" else _fail_read)(monkeypatch, lost)
    with pytest.raises(ObligationsUnavailable, match="custody event chain is unreadable"):
        current.rebuild(tmp_path, replace=True)
    assert _store_facts(tmp_path) == retained


def test_the_lenient_reader_skips_only_an_unopenable_middle_segment(tmp_path, monkeypatch):
    path, expected = _chain(tmp_path, 4)
    _fail_open(monkeypatch, sorted((tmp_path / "archive").glob("events_*.jsonl"))[1])
    assert [row["run_id"] for row in c._iter_rows(path)] == [run for run in expected if run != "run-001"]
    with pytest.raises(utils.JsonlChainUnreadable):
        list(c._iter_rows(path, strict=True))


@pytest.mark.parametrize("live", [("run-live",), ()])
def test_the_pinned_live_file_is_read_to_its_eof_and_released_after_the_pass(tmp_path, live):
    path, expected = _chain(tmp_path, 2, live=live)
    reader = c._iter_rows(path, strict=True)
    seen = [next(reader)["run_id"]]  # the pass began: the live file is pinned and the chain listed
    with path.open("ab") as handle:  # appended before its EOF is read
        handle.write(_line(_row("run-late")))
    seen += [row["run_id"] for row in reader]
    assert seen == [*expected, "run-late"]
    os.replace(path, tmp_path / "archive" / "events_20261231T000000.jsonl")  # nothing holds it now


# Windows opens files without delete sharing: a pinned live file cannot be renamed there.
@pytest.mark.skipif(sys.platform == "win32", reason="renaming a file another handle holds open is POSIX behaviour")
@pytest.mark.parametrize("live", [("run-live",), ()])
def test_the_pinned_live_generation_is_read_to_its_eof_across_a_rotation(tmp_path, live):
    path, expected = _chain(tmp_path, 2, live=live)
    reader = c._iter_rows(path, strict=True)
    seen = [next(reader)["run_id"]]  # the pass began: the live file is pinned and the chain listed
    rotated = tmp_path / "archive" / "events_20261231T000000.jsonl"
    os.replace(path, rotated)
    path.write_bytes(_line(_row("run-new-generation")))
    with rotated.open("ab") as handle:  # appended to the pinned generation before its EOF is read
        handle.write(_line(_row("run-late")))
    seen += [row["run_id"] for row in reader]
    assert seen == [*expected, "run-late"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows refuses to rename a file another handle holds open")
def test_a_rotation_during_the_pass_is_refused_on_windows_and_the_chain_still_reads_whole(tmp_path):
    path, expected = _chain(tmp_path, 2)
    reader = c._iter_rows(path, strict=True)
    seen = [next(reader)["run_id"]]
    rotated = tmp_path / "archive" / "events_20261231T000000.jsonl"
    with pytest.raises(PermissionError):  # the rotator contains this and retries on a later tick
        os.replace(path, rotated)
    seen += [row["run_id"] for row in reader]
    assert seen == expected
    os.replace(path, rotated)  # the pass released its handle


def test_offsets_read_the_chain_suffix_and_one_beyond_the_chain_is_refused(tmp_path):
    path, expected = _chain(tmp_path, 3, live=("run-live", "run-last"))
    total, first = c.chain_length(path), (tmp_path / "archive" / "events_20260101T000000.jsonl").stat().st_size

    def after(offset):
        rows, length = c.rows_after(path, offset)
        return [row["run_id"] for row in rows], length

    assert after(0) == (expected, total)
    assert after(first) == (expected[1:], total)  # a segment boundary
    assert after(total - len(_line(_row("run-last")))) == (["run-last"], total)  # inside the live segment
    assert after(total) == ([], total)  # the chain's end
    assert after(total + 1) == (expected, total)  # a chain shorter than the boundary is read whole
    for strict in (True, False):
        with pytest.raises(utils.JsonlChainUnreadable):
            list(c._iter_rows(path, strict=strict, start_offset=total + 1))


@pytest.mark.parametrize("shape", ["short", "archive_only", "empty_middle_archive", "empty_live", "absent"])
def test_short_archive_only_and_empty_chains_read_and_rebuild_whole(tmp_path, shape):
    archives, live, empty = {"short": (1, ("run-live",), ()), "archive_only": (2, None, ()),
                             "empty_middle_archive": (3, ("run-live",), (1,)), "empty_live": (0, (), ()),
                             "absent": (0, None, ())}[shape]
    _path, expected = _chain(tmp_path, archives, live=live, empty=empty)
    assert [row["run_id"] for row in c.custody_rows(tmp_path)] == expected
    ensure_current_store(tmp_path)
    assert current.rebuild(tmp_path, replace=True) == len(expected)
    assert sorted(c.replay(tmp_path)) == expected
