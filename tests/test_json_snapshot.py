"""One parse per file identity for rename-written JSON state (``utils.json_snapshot``).

Each case removes one guard: the memo, the inode or change time in the key, the
clock-tick rule, the remembered validity, or the refusal to remember a failed read.
"""

from __future__ import annotations

import json
import os

import pytest

from ouroboros import utils


@pytest.fixture
def parses(monkeypatch):
    """The documents ``json_snapshot`` parses, in order."""
    seen, real = [], json.loads
    monkeypatch.setattr(utils.json, "loads", lambda text, *args, **kwargs: seen.append(text) or real(text, *args, **kwargs))
    return seen


def _settled(monkeypatch):
    monkeypatch.setattr(utils, "_RACY_NS", 0)  # trust a just-written file at once (the tick rule has its own case)


def test_two_reads_share_one_parse_and_one_object(tmp_path, parses, monkeypatch):
    _settled(monkeypatch)
    path = tmp_path / "state.json"
    utils.atomic_write_json(path, {"bindings": {"t": 1}})

    first, second = utils.json_snapshot(path), utils.json_snapshot(path, strict=True)

    assert first == {"bindings": {"t": 1}} and second is first
    assert len(parses) == 1


def test_a_rename_replace_with_the_old_size_and_mtime_is_read_again(tmp_path, parses, monkeypatch):
    _settled(monkeypatch)
    path = tmp_path / "state.json"
    utils.atomic_write_json(path, {"v": 1})
    before = path.stat()
    assert utils.json_snapshot(path) == {"v": 1}

    utils.atomic_write_json(path, {"v": 2})  # the writer's rename: same size, a new inode
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (before.st_size, before.st_mtime_ns)

    assert utils.json_snapshot(path) == {"v": 2}
    assert len(parses) == 2


@pytest.mark.skipif(os.name == "nt", reason="POSIX change time; Windows reports creation time")
def test_an_in_place_same_size_rewrite_with_mtime_restored_is_read_again(tmp_path, parses, monkeypatch):
    _settled(monkeypatch)
    path = tmp_path / "state.json"
    path.write_text('{"v": 1}', encoding="utf-8")
    before = path.stat()
    assert utils.json_snapshot(path) == {"v": 1}

    path.write_text('{"v": 2}', encoding="utf-8")  # a restore's copy: same inode, same size
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (before.st_ino, before.st_size, before.st_mtime_ns)

    assert utils.json_snapshot(path) == {"v": 2}


def test_a_file_changed_within_a_clock_tick_is_compared_by_its_bytes(tmp_path, parses, monkeypatch):
    """A coarse file clock gives the next in-place write this very identity: within the tick
    window the bytes, not the stamp, decide; unchanged bytes are never parsed twice."""
    monkeypatch.setattr(utils, "file_identity", lambda st: (st.st_dev, st.st_ino, st.st_size))  # a clock that did not move
    path = tmp_path / "state.json"
    path.write_text('{"v": 1}', encoding="utf-8")
    assert utils.json_snapshot(path) == {"v": 1}

    path.write_text('{"v": 2}', encoding="utf-8")  # same inode, same size, the same coarse stamp
    assert utils.json_snapshot(path) == utils.json_snapshot(path) == {"v": 2}
    assert len(parses) == 2


def test_strict_refuses_a_malformed_document_a_lenient_read_remembered(tmp_path, parses, monkeypatch):
    _settled(monkeypatch)
    torn, listed = tmp_path / "torn.json", tmp_path / "listed.json"
    torn.write_text("{torn", encoding="utf-8")
    listed.write_text("[1]", encoding="utf-8")

    assert utils.json_snapshot(torn) is None and utils.json_snapshot(listed) is None
    for path in (torn, listed):
        with pytest.raises(ValueError):
            utils.json_snapshot(path, strict=True)
    assert len(parses) == 2  # the failure was remembered with its validity, not parsed again


def test_an_unreadable_file_is_never_remembered(tmp_path, monkeypatch):
    _settled(monkeypatch)
    path = tmp_path / "state.json"
    utils.atomic_write_json(path, {"v": 1})

    def refused(*_args, **_kwargs):
        raise PermissionError("refused")

    with monkeypatch.context() as patch:  # the file and its identity stay exactly as they were
        patch.setattr(utils, "open", refused, raising=False)
        with pytest.raises(PermissionError):
            utils.json_snapshot(path, strict=True)
        assert utils.json_snapshot(path) is None

    assert utils.json_snapshot(path, strict=True) == {"v": 1}


def test_an_absent_file_is_none_on_both_paths(tmp_path):
    assert utils.json_snapshot(tmp_path / "missing.json") is None
    assert utils.json_snapshot(tmp_path / "missing.json", strict=True) is None
