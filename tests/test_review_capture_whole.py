"""Reviewer captures keep whole outputs, and a full volume never loses one.

A reviewer's copy is released only after its delta and named outputs are retained
whole: no byte bound applies (the 64 MiB evidence bound is gone for every run), so a
large output is saved, never the reason a larger copy stays. Real consumers run with
a synthetic engine transport.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

import pytest

from ouroboros import review_checking as rc
from tests._review_session_route_shared import (  # noqa: F401 - transport fixtures
    FakeGateway,
    _owned_gateway_uses_each_test_transport,
    fake_route,
)
from tests.test_review_checking import author  # noqa: F401
from tests.test_review_checking_completion import run_one_shot
from tests.test_review_directory_capture import _run as run_directory, _source, _subject
from tests.test_review_session_rights import writable_engine  # noqa: F401

LARGE = 2 << 20  # well past a pipe buffer; a size no bound reaches


def _write(path, size, *, seed=None):
    """``size`` bytes in 1 MiB blocks; seeded random bytes do not compress in a Git patch."""
    rng = random.Random(seed) if seed is not None else None
    block = bytes(range(256)) * 4096
    digest = hashlib.sha256()
    with path.open("wb") as out:
        for start in range(0, size, len(block)):
            chunk = (rng.randbytes(len(block)) if rng else block)[: size - start]
            out.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_reviewer_large_named_output_is_retained_whole_before_release(
    author, tmp_path, writable_engine, monkeypatch,  # noqa: F811
):
    drive, written = tmp_path / "drive", {}
    source = rc.freeze_checking_source(author, data_root=drive, subject=rc.SUBJECT_INDEX)

    def edit(area):
        (area / rc.EVIDENCE_DIRNAME).mkdir()
        written["sha"] = _write(area / rc.EVIDENCE_DIRNAME / "trace.bin", LARGE)

    receipt = run_one_shot(author, drive, source, monkeypatch, edit)["checking_evidence"]
    assert receipt["status"] == "retained" and receipt["gaps"] == [], {k: receipt.get(k) for k in ("status", "gaps")}
    [row] = [row for row in receipt["named_evidence"] if row["path"] == f"{rc.EVIDENCE_DIRNAME}/trace.bin"]
    assert row["size"] == LARGE and row["sha256"] == written["sha"]
    assert _sha(Path(receipt["evidence_dir"]) / "files" / row["path"]) == written["sha"]
    assert not (rc.seat_dir(drive, "task-1", "op-1") / "area.noindex").exists()


def test_reviewer_large_binary_patch_is_retained_and_reproduces_the_tree(
    author, tmp_path, writable_engine, monkeypatch,  # noqa: F811
):
    drive = tmp_path / "drive"
    source = rc.freeze_checking_source(author, data_root=drive, subject=rc.SUBJECT_INDEX)
    receipt = run_one_shot(author, drive, source, monkeypatch,
                           lambda area: _write(area / "fixture.bin", LARGE, seed=1609))["checking_evidence"]
    facts = {k: receipt.get(k) for k in ("status", "gaps", "patch_bytes", "patch_retained")}
    assert receipt["status"] == "retained" and receipt["patch_retained"], facts
    assert receipt["patch_bytes"] > LARGE and not receipt["gaps"]
    assert not (rc.seat_dir(drive, "task-1", "op-1") / "area.noindex").exists()
    # The retained patch alone rebuilds the reviewer's tree over the frozen subject;
    # reads borrow the author's objects and writes stay in a scratch object store.
    objects = tmp_path / "apply-objects"
    objects.mkdir()
    env = {**os.environ, "GIT_INDEX_FILE": str(tmp_path / "apply.index"), "GIT_OBJECT_DIRECTORY": str(objects),
           "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(author / ".git" / "objects")}

    def git(*args):
        return subprocess.run(["git", *args], cwd=author, env=env, check=True,
                              capture_output=True).stdout.decode().strip()

    git("read-tree", source["tree"])
    git("apply", "--cached", str(Path(receipt["evidence_dir"]) / "delta.patch"))
    assert git("write-tree") == receipt["after_tree"]


def test_reviewer_large_ordinary_file_is_retained_whole_before_release(
    tmp_path, writable_engine, monkeypatch,  # noqa: F811
):
    root, drive, written = _subject(tmp_path, "skill_review"), tmp_path / "drive", {}  # a copied subject

    def edit(area):
        written["area"] = area
        written["sha"] = _write(area / "dataset.bin", LARGE)

    actor = run_directory(root, drive, "skill_review", _source(root, drive, "skill_review"), monkeypatch, edit)
    assert actor["status"] == "ok", actor
    receipt = actor["usage"]["checking_evidence"]
    assert receipt["status"] == "retained" and receipt["gaps"] == [], {k: receipt.get(k) for k in ("status", "gaps")}
    [row] = [row for row in receipt["files"] if row["path"] == "dataset.bin"]
    assert row["size"] == LARGE and row["sha256"] == written["sha"]
    assert _sha(Path(receipt["evidence_dir"]) / "files" / "dataset.bin") == written["sha"]
    assert receipt["entries"]["dataset.bin"]["after"]["size"] == LARGE
    assert not written["area"].exists() and not (root / "dataset.bin").exists()


def _no_space(*_args, **_kwargs):
    raise OSError(errno.ENOSPC, "No space left on device")


@pytest.mark.parametrize("boundary", ["patch", "named_output", "summary"])
def test_reviewer_capture_out_of_space_keeps_copy_without_complete_receipt(
    author, tmp_path, writable_engine, monkeypatch, boundary,  # noqa: F811
):
    drive = tmp_path / "drive"
    source = rc.freeze_checking_source(author, data_root=drive, subject=rc.SUBJECT_INDEX)
    real_git, real_copy, real_write = rc._git, rc.shutil.copyfileobj, rc.atomic_write_json

    def git(cwd, *args, **kwargs):
        if args[:1] == ("diff",) and "--binary" in args:
            kwargs["stdout"].write(b"diff --git a/probe_test.py")  # the volume filled mid-stream
            _no_space()
        return real_git(cwd, *args, **kwargs)

    def copy(src, dst, *args, **kwargs):
        if Path(dst.name).parent.name == rc.EVIDENCE_DIRNAME:
            dst.write(b"<cover")
            _no_space()
        return real_copy(src, dst, *args, **kwargs)

    def write(path, value, *args, **kwargs):
        if Path(path).name == "summary.json":
            _no_space()
        return real_write(path, value, *args, **kwargs)

    def edit(area):
        (area / "probe_test.py").write_bytes(b"assert True\n")
        (area / rc.EVIDENCE_DIRNAME).mkdir()
        (area / rc.EVIDENCE_DIRNAME / "coverage.xml").write_bytes(b"<coverage/>\n")
        target, name, fault = {"patch": (rc, "_git", git), "named_output": (rc.shutil, "copyfileobj", copy),
                               "summary": (rc, "atomic_write_json", write)}[boundary]
        monkeypatch.setattr(target, name, fault)  # only capture writes after materialization fail

    receipt = run_one_shot(author, drive, source, monkeypatch, edit)["checking_evidence"]
    assert receipt["status"] == "capture_failed" and "No space left on device" in receipt["error"], receipt
    area = rc.seat_dir(drive, "task-1", "op-1") / "area.noindex"
    assert (area / "probe_test.py").read_bytes() == b"assert True\n"
    assert (area / rc.EVIDENCE_DIRNAME / "coverage.xml").read_bytes() == b"<coverage/>\n"
    saved = rc.seat_dir(drive, "task-1", "op-1") / "evidence" / "summary.json"
    assert not saved.exists() or json.loads(saved.read_text())["status"] == "capture_failed"


@pytest.mark.parametrize("boundary", ["summary", "marker"])
def test_out_of_space_retry_keeps_the_earlier_failed_generation_byte_exact(
    author, tmp_path, monkeypatch, boundary,  # noqa: F811
):
    drive = tmp_path / "drive"
    source = rc.freeze_checking_source(author, data_root=drive, subject=rc.SUBJECT_INDEX)
    area = Path(rc.prepare_seat_area(drive, "task", "op", source))
    (area / rc.EVIDENCE_DIRNAME).mkdir()
    (area / rc.EVIDENCE_DIRNAME / "early.log").write_bytes(b"early\n")
    _write(area / "fixture.bin", 1 << 16, seed=7)
    with monkeypatch.context() as patch:
        patch.setattr(rc, "_git_delta", _no_space)
        first = rc.capture_seat_evidence(drive, "task", "op")
    assert first["status"] == "capture_failed" and area.is_dir()
    earlier = Path(first["evidence_dir"])

    def generation():
        return {path.relative_to(earlier).as_posix(): _sha(path) for path in earlier.rglob("*") if path.is_file()}

    readback = generation()
    assert {"summary.json", f"files/{rc.EVIDENCE_DIRNAME}/early.log"} <= set(readback)
    marker = (area.parent / "checking.json").read_bytes()

    (area / rc.EVIDENCE_DIRNAME / "final.log").write_bytes(b"settled\n")
    real_write = rc.atomic_write_json
    name = "summary.json" if boundary == "summary" else "checking.json"
    monkeypatch.setattr(rc, "atomic_write_json", lambda path, value, *a, **k: _no_space() if Path(path).name == name
                        else real_write(path, value, *a, **k))
    failed = rc.capture_seat_evidence(drive, "task", "op")
    assert failed["status"] == "capture_failed" and area.is_dir()
    assert (area.parent / "checking.json").read_bytes() == marker  # the canonical receipt is still the earlier one
    assert generation() == readback

    monkeypatch.setattr(rc, "atomic_write_json", real_write)
    final = rc.capture_seat_evidence(drive, "task", "op")
    assert final["status"] == "retained" and final["patch_retained"] and not area.exists()
    assert final["evidence_dir"] != first["evidence_dir"] and generation() == readback
    assert json.loads((earlier / "summary.json").read_text()) == first
