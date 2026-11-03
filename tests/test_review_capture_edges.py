"""Regressions from independent capture review: refusal must match missing data."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from ouroboros import review_checking as rc
from tests._review_session_route_shared import fake_route  # noqa: F401
from tests.test_review_checking import author  # noqa: F401
from tests.test_review_thread_checking import _run, thread_gateway  # noqa: F401


@pytest.mark.skipif(os.name == "nt", reason="symlink privileges vary on Windows")
def test_ignore_warning_does_not_disable_next_thread_turn(author, tmp_path, thread_gateway):  # noqa: F811
    drive = tmp_path / "drive"
    source = rc.freeze_checking_source(author, data_root=drive, subject=rc.SUBJECT_INDEX)
    def replace_ignore(area):
        (area / ".gitignore").unlink()
        (area / ".gitignore").symlink_to("f.txt")

    thread_gateway.before_turn = replace_ignore
    first = _run(author, drive, source)
    receipt = first["checking_evidence"]
    assert receipt["status"] == "retained"
    assert ".gitignore" in receipt["listing_warning"]
    second = _run(author, drive, source, operation="op-2", thread_id=first["thread_id"])
    assert second["checking_evidence"]["status"] == "retained"
    assert len(thread_gateway.turn_requests) == 2


def test_trace_environment_is_not_missing_inventory(author, tmp_path, monkeypatch):  # noqa: F811
    drive = tmp_path / "drive"
    source = rc.freeze_checking_source(author, data_root=drive, subject=rc.SUBJECT_INDEX)
    rc.prepare_seat_area(drive, "task", "op", source)
    monkeypatch.setenv("GIT_TRACE", "1")
    monkeypatch.setenv("GIT_TRACE_SETUP", "1")
    result = rc.capture_seat_evidence(drive, "task", "op")
    assert result["status"] == "retained" and not result["listing_warning"]


@pytest.mark.parametrize("kind", ["git", "directory"])
@pytest.mark.parametrize("output", ["file", "fifo"])
def test_named_root_or_special_is_one_gap_and_releases_the_copy(author, tmp_path, kind, output):  # noqa: F811
    if output == "fifo" and not hasattr(os, "mkfifo"):
        pytest.skip("FIFO requires POSIX")
    drive = tmp_path / "drive"
    if kind == "git":
        source = rc.freeze_checking_source(author, data_root=drive, subject=rc.SUBJECT_INDEX)
    else:
        folder = tmp_path / "folder"
        folder.mkdir()
        (folder / "input.txt").write_bytes(b"original")
        source = rc.freeze_checking_source(folder, data_root=drive)
    area = Path(rc.prepare_seat_area(drive, "task", "op", source))
    named = area / rc.EVIDENCE_DIRNAME
    if output == "file":
        named.write_bytes(b"a command redirected its log here\n")
        expected = {"path": rc.EVIDENCE_DIRNAME, "reason": "evidence_root_not_directory"}
    else:
        named.mkdir()
        os.mkfifo(named / "pipe")
        expected = {"path": f"{rc.EVIDENCE_DIRNAME}/pipe", "reason": "unsupported_entry_type"}
    result = rc.capture_seat_evidence(drive, "task", "op")
    assert result["status"] == "partial" and result["gaps"] == [expected]
    assert not area.exists(), "a receipt that names its gap releases the copy"


@pytest.mark.skipif(os.name == "nt", reason="POSIX can open directories as raw descriptors")
def test_open_regular_closes_descriptor_when_entry_is_a_directory(tmp_path, monkeypatch):
    original = os.open
    opened = []

    def observe(path, flags):
        fd = original(path, flags)
        opened.append(fd)
        return fd

    monkeypatch.setattr(os, "open", observe)
    with pytest.raises(rc.CheckingAreaUnavailable):
        rc._open_regular(tmp_path)
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


def test_open_regular_closes_descriptor_when_wrapper_fails(tmp_path, monkeypatch):
    path = tmp_path / "file"
    path.write_bytes(b"data")
    seen = []

    def reject(fd, mode):
        seen.append(fd)
        raise OSError("fixture wrapper failure")

    monkeypatch.setattr(os, "fdopen", reject)
    with pytest.raises(OSError, match="fixture wrapper failure"):
        rc._open_regular(path)
    with pytest.raises(OSError):
        os.fstat(seen[0])


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO requires POSIX")
def test_open_regular_refuses_fifo_without_waiting_for_a_writer(tmp_path, monkeypatch):
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    original = os.open

    def require_nonblocking(path, flags):
        assert flags & os.O_NONBLOCK, "capture would block before it can validate the file type"
        return original(path, flags)

    monkeypatch.setattr(os, "open", require_nonblocking)
    with pytest.raises(rc.CheckingAreaUnavailable):
        rc._open_regular(fifo)
