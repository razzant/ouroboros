"""A pass over a long rotated chain does not stat every closed archive again.

Closed segments never change after their rename, so their stats are reused while the live
file keeps its inode and ``archive/`` its mtime; the newest archive is re-stat'ed (an unlocked
append can still grow it), and a rotation lists the chain afresh.
"""

from __future__ import annotations

import os
import pathlib

from ouroboros import utils


def _chain(root, archives):
    (root / "logs").mkdir(parents=True)
    (root / "archive").mkdir()
    for index in range(archives):
        (root / "archive" / f"chat_2026101{index}T000000.jsonl").write_text(f'{{"n": {index}}}\n')
    live = root / "logs" / "chat.jsonl"
    live.write_text('{"n": "live"}\n')
    return live


def _segments(live):
    with utils.jsonl_chain_handles(live) as handles:
        return [path.name for path, _handle in handles]


def test_closed_segments_are_stat_once_until_the_chain_rotates(tmp_path, monkeypatch):
    live = _chain(tmp_path, 20)
    first = _segments(live)
    assert len(first) == 21 and first[-1] == "chat.jsonl"
    stats = []
    real = pathlib.Path.stat

    def counted(self, *args, **kwargs):
        stats.append(self.as_posix())  # one separator, so a native Windows path is counted too
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "stat", counted)
    assert _segments(live) == first
    archive_stats = [path for path in stats if "/archive/chat_" in path]
    assert len(archive_stats) <= 1, archive_stats  # only the newest, which a late append can grow
    # A rotation renames the live file into the archive: the chain is listed again.
    live.rename(tmp_path / "archive" / "chat_20261020T000000.jsonl")
    live.write_text('{"n": "new live"}\n')
    stats.clear()
    after = _segments(live)
    assert after[-2] == "chat_20261020T000000.jsonl" and after[-1] == "chat.jsonl" and len(after) == 22
    assert len([path for path in stats if "/archive/chat_" in path]) >= 21


def test_a_grown_newest_archive_and_a_removed_archive_are_seen(tmp_path):
    live = _chain(tmp_path, 3)
    _segments(live)
    newest = tmp_path / "archive" / "chat_20261012T000000.jsonl"
    with newest.open("a") as handle:
        handle.write('{"n": "late append"}\n')
    with utils.jsonl_chain_handles(live) as handles:
        sizes = {path.name: os.fstat(handle.fileno()).st_size for path, handle in handles}
    assert sizes[newest.name] == newest.stat().st_size
    (tmp_path / "archive" / "chat_20261010T000000.jsonl").unlink()  # the directory's mtime moves
    assert "chat_20261010T000000.jsonl" not in _segments(live)


def test_a_sequential_pass_yields_the_eager_pairs_one_archive_at_a_time(tmp_path):
    live = _chain(tmp_path, 5)
    for live_present in (True, False):  # without the live file the last pair is an archive
        if not live_present:
            live.unlink()
        with utils.jsonl_chain_handles(live) as handles:
            eager = [(path.name, handle.read()) for path, handle in handles]
        lazy, read = [], []
        with utils.jsonl_chain_handles(live, sequential=True) as handles:
            for path, handle in handles:
                assert all(earlier.closed for earlier in read)  # the previous archive closed first
                lazy.append((path.name, handle.read()))
                read.append(handle)
        assert lazy == eager and all(handle.closed for handle in read)
        assert len(lazy) == 5 + live_present and (lazy[-1][0] == "chat.jsonl") is live_present
    with utils.jsonl_chain_handles(live, sequential=True) as handles:
        _path, first = next(iter(handles))
    assert first.closed  # an abandoned pass is closed with its context
