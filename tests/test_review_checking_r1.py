"""R1 consumer regressions: bucket-aware inputs and bounded evidence capture."""
from __future__ import annotations

import json
import os
import pathlib
import stat

import pytest

from ouroboros import review_checking as rc
from tests.test_review_checking import _author_state, _git, author  # noqa: F401 — shared real-Git fixture
from tests.test_review_thread_checking import _run, fake_route, thread_gateway  # noqa: F401 — offline transport


def _readonly_git_object(area):
    """A read-only file the reviewer leaves in its copy (a linked worktree keeps no objects of its own)."""
    probe = area / "readonly-probe.txt"
    probe.write_bytes(b"reviewer-created read-only output\n")
    probe.chmod(0o444)
    return probe


def _windows_readonly_unlink(monkeypatch, target):
    """Inject only the OS denial; real rmtree/onerror must chmod and retry."""
    events = []
    real_unlink, real_chmod = os.unlink, os.chmod

    def unlink(path, *, dir_fd=None):
        # POSIX rmtree uses a basename + dir_fd; Windows uses the full path.
        watched = pathlib.Path(os.fsdecode(path)).name == target.name
        if watched:
            if not os.stat(path, dir_fd=dir_fd).st_mode & stat.S_IWRITE:
                events.append("denied")
                raise PermissionError(13, "simulated Windows read-only Git object", os.fspath(path))
            events.append("retried")
        return real_unlink(path, dir_fd=dir_fd)

    def chmod(path, mode, *args, **kwargs):
        if pathlib.Path(path) == target:
            events.append("chmod")
            assert mode & stat.S_IWRITE
        return real_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    monkeypatch.setattr(os, "chmod", chmod)
    return events


def test_interrupted_materialization_retries_after_readonly_cleanup(author, tmp_path, monkeypatch):  # noqa: F811
    data = tmp_path / "data"
    source = rc.freeze_checking_source(author, data_root=data, subject=rc.SUBJECT_INDEX)
    before = _author_state(author)
    home = rc.seat_dir(data, "task", "interrupted-readonly")
    materialize = rc._materialize_git
    targets = []

    def interrupted(area, frozen):
        materialize(area, frozen)
        targets.append(_readonly_git_object(area))
        raise OSError("interrupted after materialization, before marker")

    monkeypatch.setattr(rc, "_materialize_git", interrupted)
    with pytest.raises(rc.CheckingAreaUnavailable, match="interrupted after materialization"):
        rc.prepare_seat_area(data, "task", "interrupted-readonly", source)
    # The marker is written first and names the copy as unfinished, so it is never adopted.
    assert json.loads((home / "checking.json").read_text(encoding="utf8"))["materializing"] is True
    monkeypatch.setattr(rc, "_materialize_git", materialize)
    events = _windows_readonly_unlink(monkeypatch, targets[0])
    area = pathlib.Path(rc.prepare_seat_area(data, "task", "interrupted-readonly", source))
    assert events == ["denied", "chmod", "retried"]
    assert area == home / "area.noindex"
    assert "materializing" not in json.loads((home / "checking.json").read_text(encoding="utf8"))
    assert _git(area, "write-tree") == source["tree"] and not (area / "readonly-probe.txt").exists()
    assert _author_state(author) == before
