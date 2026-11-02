"""Real writable reviewer consumers retain raw bytes without executing filters."""
from __future__ import annotations

import hashlib
import os
import pathlib
import shlex
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros import review_checking as rc
from ouroboros.config import CLAUDEXOR_DELEGATED_WORKSPACE_ROOT_MIN_VERSION
from ouroboros.subagents import delegated_run_shape
from tests.test_review_checking import _author_state, _git, author, old_clone_seat  # noqa: F401


def _bytes_git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout


def _replay_patch(tmp_path, source, evidence):
    replay = tmp_path / "replay"
    rc._materialize_git(replay, source)
    patch = pathlib.Path(evidence["evidence_dir"]) / "delta.patch"
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == evidence["patch_sha256"]
    _git(replay, "apply", "--cached", "--binary", str(patch))
    return replay


@pytest.mark.skipif(os.name == "nt", reason="configured filter probe uses a POSIX shell")
@pytest.mark.parametrize("filter_kind", ["clean", "process"])
def test_workspace_write_capture_preserves_raw_bytes_without_running_filters(author, tmp_path, filter_kind):  # noqa: F811
    data = tmp_path / "data"
    source = rc.freeze_checking_source(author, data_root=data, subject=rc.SUBJECT_INDEX)
    before = _author_state(author)
    # Exercise the actual workspace_write consumer binding, not an invented
    # capture-only directory. No engine or paid reviewer is started by this test.
    binding = rc.session_checking(
        SimpleNamespace(engine_version=CLAUDEXOR_DELEGATED_WORKSPACE_ROOT_MIN_VERSION),
        custody_drive=data, task_id="task", operation_id="raw", root=str(author),
        source=source, shape=delegated_run_shape(False, "workspace_write"))
    assert binding["execution"]["delegated"] is True
    area = pathlib.Path(binding["execution"]["workspaceRoot"])
    marker, hook = tmp_path / "filter-ran", tmp_path / "filter.sh"
    hook.write_text("#!/bin/sh\nprintf invoked >> " + shlex.quote(str(marker)) + "\n"
                    + ("exit 1\n" if filter_kind == "process" else "printf normalized\n"))
    hook.chmod(0o755)
    _git(area, "config", f"filter.probe.{filter_kind}", "sh " + shlex.quote(str(hook)))
    _git(area, "config", "filter.probe.required", "true")
    (area / ".gitattributes").write_bytes(b"raw.bin filter=probe\nraw.txt text eol=lf\n")
    raw_binary, raw_text = b"\x00reviewer\r\n\xffbinary", b"first\r\nsecond\r\n"
    (area / "raw.bin").write_bytes(raw_binary)
    (area / "raw.txt").write_bytes(raw_text)
    named = area / rc.EVIDENCE_DIRNAME / "coverage.xml"
    named.parent.mkdir()
    named.write_bytes(b"<coverage>\r\nbyte exact\x00</coverage>")

    evidence = rc.capture_seat_evidence(data, "task", "raw")

    assert evidence["status"] == "retained", evidence
    assert not marker.exists()
    assert not area.exists()
    assert _author_state(author) == before
    replay = _replay_patch(tmp_path, source, evidence)
    assert _bytes_git(replay, "show", ":raw.bin") == raw_binary
    assert _bytes_git(replay, "show", ":raw.txt") == raw_text
    assert _bytes_git(replay, "show", ":f.txt") == b"A staged\n"
    retained = pathlib.Path(evidence["evidence_dir"]) / "files" / rc.EVIDENCE_DIRNAME / "coverage.xml"
    assert retained.read_bytes() == b"<coverage>\r\nbyte exact\x00</coverage>"


@pytest.mark.skipif(os.name == "nt", reason="symlink privileges vary on Windows")
def test_linked_source_and_linked_reviewer_index_leave_source_controls_unchanged(author, tmp_path):  # noqa: F811
    linked, data = tmp_path / "linked", tmp_path / "data"
    _git(author, "worktree", "add", "--detach", str(linked), "HEAD")
    (linked / "f.txt").write_bytes(b"linked staged\n")
    _git(linked, "add", "f.txt")
    (linked / "f.txt").write_bytes(b"linked unstaged\n")
    source = rc.freeze_checking_source(linked, data_root=data, subject=rc.SUBJECT_INDEX)
    before, linked_before = _author_state(author), _author_state(linked)
    area = pathlib.Path(rc.prepare_seat_area(data, "task", "linked", source))
    linked_index = pathlib.Path(_git(linked, "rev-parse", "--path-format=absolute", "--git-path", "index"))
    control_before = linked_index.read_bytes()
    # The reviewer index may even point to source controls: it is never copied,
    # read or written by host capture. Its worktree bytes are the evidence.
    area_index = pathlib.Path(_git(area, "rev-parse", "--path-format=absolute", "--git-path", "index"))
    area_index.unlink()
    area_index.symlink_to(linked_index)
    raw = b"linked checking bytes\r\n\x00"
    (area / "f.txt").write_bytes(raw)
    evidence = rc.capture_seat_evidence(data, "task", "linked")

    assert evidence["status"] == "retained", evidence
    assert linked_index.read_bytes() == control_before
    assert _author_state(author) == before
    assert _author_state(linked) == linked_before
    replay = _replay_patch(tmp_path, source, evidence)
    assert _bytes_git(replay, "show", ":f.txt") == raw


@pytest.mark.skipif(os.name == "nt", reason="symlink privileges vary on Windows")
@pytest.mark.parametrize("control", ["config", "objects"])
def test_an_older_clone_seats_linked_control_is_refused_without_touching_source(author, tmp_path, control):  # noqa: F811
    """A seat made before linked worktrees keeps its own Git controls; one the reviewer replaced
    with a link is refused (its capture would read through it), and the copy stays recoverable."""
    data = tmp_path / "data"
    source = rc.freeze_checking_source(author, data_root=data, subject=rc.SUBJECT_INDEX)
    area = old_clone_seat(data, author, source, "linked-control")
    before = _author_state(author)
    config_before = (author / ".git" / "config").read_bytes()
    target = area / ".git" / control
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    target.symlink_to(author / ".git" / control, target_is_directory=control == "objects")
    (area / "f.txt").write_bytes(b"must remain recoverable\r\n")

    evidence = rc.capture_seat_evidence(data, "task", "linked-control")

    assert evidence["status"] == "capture_failed", evidence
    assert f".git/{control}" in evidence["error"]
    assert area.is_dir() and (area / "f.txt").read_bytes() == b"must remain recoverable\r\n"
    assert not (rc.seat_dir(data, "task", "linked-control") / "evidence" / "delta.patch").exists()
    assert _author_state(author) == before
    assert (author / ".git" / "config").read_bytes() == config_before


def test_reviewer_index_flags_cannot_hide_changes_or_invent_raw_evidence(author, tmp_path):  # noqa: F811
    data = tmp_path / "data"
    source = rc.freeze_checking_source(author, data_root=data, subject=rc.SUBJECT_INDEX)
    area = pathlib.Path(rc.prepare_seat_area(data, "task", "index-flags", source))
    _git(area, "update-index", "--assume-unchanged", "f.txt")
    (area / "f.txt").write_bytes(b"actual final bytes\r\n")
    (area / "index-only").write_bytes(b"forged index content")
    _git(area, "add", "index-only")
    (area / "index-only").unlink()
    (area / ".gitignore").unlink()
    unusual_name = "odd name.txt" if os.name == "nt" else "odd\nname.txt"
    (area / unusual_name).write_bytes(b"raw filename\r\n")
    evidence = rc.capture_seat_evidence(data, "task", "index-flags")

    assert evidence["status"] == "retained", evidence
    replay = _replay_patch(tmp_path, source, evidence)
    assert _bytes_git(replay, "show", ":f.txt") == b"actual final bytes\r\n"
    assert _bytes_git(replay, "show", ":" + unusual_name) == b"raw filename\r\n"
    names = _bytes_git(replay, "ls-files", "-z").split(b"\0")
    assert b"index-only" not in names and b".gitignore" not in names


@pytest.mark.skipif(os.name == "nt", reason="symlink privileges vary on Windows")
def test_directory_replaced_by_symlink_records_link_without_reading_target(author, tmp_path):  # noqa: F811
    nested = author / "nested"
    nested.mkdir()
    (nested / "tracked").write_bytes(b"baseline")
    _git(author, "add", "nested/tracked")
    data = tmp_path / "data"
    source = rc.freeze_checking_source(author, data_root=data, subject=rc.SUBJECT_INDEX)
    area = pathlib.Path(rc.prepare_seat_area(data, "task", "link", source))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "tracked").write_bytes(b"unrelated bytes must not be captured")
    shutil.rmtree(area / "nested")
    (area / "nested").symlink_to(outside, target_is_directory=True)

    evidence = rc.capture_seat_evidence(data, "task", "link")

    assert evidence["status"] == "retained", evidence
    replay = _replay_patch(tmp_path, source, evidence)
    assert _bytes_git(replay, "show", ":nested") == os.fsencode(str(outside))
    assert _git(replay, "ls-files", "--stage", "nested").startswith("120000 ")
    assert b"unrelated bytes" not in pathlib.Path(evidence["evidence_dir"], "delta.patch").read_bytes()


@pytest.mark.parametrize("populated", [False, True])
def test_baseline_gitlink_is_preserved_and_populated_nested_worktree_is_a_named_gap(author, tmp_path, populated):  # noqa: F811
    commit = _git(author, "rev-parse", "HEAD")
    _git(author, "update-index", "--add", "--cacheinfo", f"160000,{commit},nested")
    data = tmp_path / "data"
    source = rc.freeze_checking_source(author, data_root=data, subject=rc.SUBJECT_INDEX)
    area = pathlib.Path(rc.prepare_seat_area(data, "task", "gitlink", source))
    (area / "nested").mkdir(exist_ok=True)
    if populated:
        (area / "nested" / "probe.txt").write_bytes(b"nested reviewer output")

    evidence = rc.capture_seat_evidence(data, "task", "gitlink")

    # No nested Git command or child config is needed to keep the immutable
    # baseline gitlink. Files beneath it are disclosed, never silently dropped;
    # a receipt that names its gap releases the copy.
    assert evidence["after_tree"] == source["tree"], evidence
    assert evidence["changed"] == []
    assert not area.exists()
    if populated:
        assert evidence["status"] == "partial"
        assert evidence["gaps"] == [{"path": "nested", "reason": "nested_git_worktree_not_captured"}]
    else:
        assert evidence["status"] == "retained" and evidence["gaps"] == []


@pytest.mark.skipif(os.name == "nt", reason="symlink privileges vary on Windows")
def test_linked_checking_root_is_not_captured_or_reused_as_author(author, tmp_path):  # noqa: F811
    data = tmp_path / "data"
    source = rc.freeze_checking_source(author, data_root=data, subject=rc.SUBJECT_INDEX)
    area = pathlib.Path(rc.prepare_seat_area(data, "task", "linked-root", source))
    before = _author_state(author)
    original = area.with_name("original-area")
    area.rename(original)
    area.symlink_to(author, target_is_directory=True)
    evidence = rc.capture_seat_evidence(data, "task", "linked-root")
    assert evidence["status"] == "capture_failed" and "target was not read" in evidence["error"]
    with pytest.raises(rc.CheckingAreaUnavailable, match="symlink"):
        rc.prepare_seat_area(data, "task", "linked-root", source)
    assert area.is_symlink() and original.is_dir()
    assert _author_state(author) == before
