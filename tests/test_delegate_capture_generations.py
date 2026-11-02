"""An unchanged partial receipt is debt, not permission for another filesystem pass.

Drive the real tick, keep its driver alive but parked, and join every submitted future.
Count source stats, scandirs and copies, including worker work; a per-tick source lexists or
witness walk would violate this law even if the saved summary were never decoded.

A writer's checkout now goes once its capture and named evidence are saved, so every copy here
is one that stays: a Git repository the run made inside it cannot be saved in the result, which
keeps the checkout until apply or reject (and every later evidence generation on it).
"""
from __future__ import annotations

import errno
import os
import pathlib
import shutil
import stat

import pytest

from tests._delegate_capture_child import capture_child  # noqa: F401

from ouroboros import delegate_collection as body, delegate_custody as c, delegate_custody_current as current
from ouroboros import delegate_named_evidence as named, review_checking as checking
from ouroboros.delegate_capture_attempt import collection_due
from tests.test_delegate_body_collection import terminal
from tests.test_delegate_collection_recovery import driven_runtime  # noqa: F401
from tests.test_recovered_capture import EVIDENCE, prepared


def _prepared(root):
    import subprocess

    material = root.with_name(root.name + "-material")
    material.mkdir()
    entry, area, source = prepared(material, root, "ordinary")
    kept = area / "vendor" / "kept"  # what the result cannot hold keeps this checkout
    kept.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=kept, check=True, capture_output=True)
    return entry, area, source


def _captured(tmp_path, monkeypatch, *, partial=False):
    entry, area, _ = _prepared(tmp_path)
    output = area / checking.EVIDENCE_DIRNAME / "result.bin"
    output.chmod(0o444)
    if partial:
        (output.parent / "gap").symlink_to(output)
    result = body.collect_terminal(tmp_path, entry.run_id, detail=terminal(entry.run_id))
    assert result["workspace_capture"]["named_evidence"]["status"] == ("partial" if partial else "retained")
    return current.run(tmp_path, entry.run_id), area


def test_failed_generations_publish_receipts_without_unlisted_bytes(tmp_path, monkeypatch):
    entry, area = _captured(tmp_path, monkeypatch)
    original, receipt = body.saved_named_evidence(tmp_path, entry)
    original_bytes = original.read_bytes()
    copy = shutil.copyfileobj
    def interrupted(src, dst, *args):
        dst.write(src.read(7))
        raise OSError(errno.EBUSY, "capture interrupted after writing bytes")
    with monkeypatch.context() as broken:
        broken.setattr(shutil, "copyfileobj", interrupted)
        for _ in range(2):
            failed = named.run_named_evidence(tmp_path, entry, recapture=True)
            home = pathlib.Path(failed["evidence_dir"])
            assert failed["status"] == "capture_failed" and failed["retry"] == "transient"
            assert failed["files"] == [] and sorted(p.name for p in home.iterdir()) == ["summary.json"]
            assert original.read_bytes() == original_bytes
    assert shutil.copyfileobj is copy
    healed = named.run_named_evidence(tmp_path, entry, recapture=True)
    assert healed["status"] == "retained" and healed["files"] == receipt["files"]
    assert (pathlib.Path(healed["evidence_dir"]) / "files" / checking.EVIDENCE_DIRNAME / "result.bin").read_bytes() == EVIDENCE
    assert area.is_dir() and original.read_bytes() == original_bytes


@pytest.mark.parametrize("destination", ["readonly", "hardlink", "symlink", "parent_symlink"])
def test_changed_evidence_never_writes_through_old_destinations(tmp_path, monkeypatch, destination):
    entry, area = _captured(tmp_path, monkeypatch)
    old_path, old = body.saved_named_evidence(tmp_path, entry)
    old_file = old_path.parent / "files" / checking.EVIDENCE_DIRNAME / "result.bin"
    assert next(row for row in old["files"] if row["path"].endswith("result.bin"))["mode"] == 0o444
    old_bytes = old_path.read_bytes()
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "result.bin"
    victim.write_bytes(EVIDENCE)
    if destination == "readonly":
        old_file.chmod(0o444)
    elif destination == "hardlink":
        old_file.unlink()
        os.link(victim, old_file)
    elif destination == "symlink":
        old_file.unlink()
        old_file.symlink_to(victim)
    else:
        old_file.unlink()
        old_file.parent.rmdir()
        old_file.parent.symlink_to(outside, target_is_directory=True)
    before = victim.stat()
    source = area / checking.EVIDENCE_DIRNAME / "result.bin"
    source.chmod(0o644)  # Windows also denies unlinking a read-only source fixture.
    source.unlink()
    source.write_bytes(b"new content")
    with monkeypatch.context() as patch:
        patch.setattr(os, "chmod", lambda *a, **k: pytest.fail("capture must not chmod"))
        latest = named.run_named_evidence(tmp_path, entry)
    assert latest["status"] == "retained" and latest["evidence_dir"] != old["evidence_dir"]
    assert old_path.read_bytes() == old_bytes and old_file.read_bytes() == EVIDENCE
    assert victim.read_bytes() == EVIDENCE and victim.stat().st_ino == before.st_ino
    assert stat.S_IMODE(victim.stat().st_mode) == stat.S_IMODE(before.st_mode)
    assert (pathlib.Path(latest["evidence_dir"]) / "files" / checking.EVIDENCE_DIRNAME / "result.bin").read_bytes() == b"new content"
    # Reuse is quiet, and the published manifest preserves the source's mode.
    with monkeypatch.context() as patch:
        patch.setattr(shutil, "copyfileobj", lambda *a, **k: pytest.fail("unchanged evidence copied"))
        assert named.run_named_evidence(tmp_path, entry) == latest


@pytest.mark.parametrize("damage", ["missing", "unreadable", "wrong_type", "bad_address"])
def test_a_missing_or_unreadable_receipt_is_a_new_fact(tmp_path, monkeypatch, damage, driven_runtime):  # noqa: F811
    runtime, clock, tick = driven_runtime
    entry, area = _captured(tmp_path, monkeypatch)
    runtime._collect(entry.run_id, recapture=True)
    tick()  # Exercise loss after a child result has already been accepted, too.
    entry = current.run(tmp_path, entry.run_id)
    assert entry.capture_job["state"] == "consumed"
    receipt, _ = body.saved_named_evidence(tmp_path, entry)
    if damage == "missing":
        receipt.unlink()
    elif damage == "bad_address":
        entry.named_evidence["directory"] = "../outside"
        c.emit(tmp_path, c.NAMED_EVIDENCE_RETAINED,
               {"run_id": entry.run_id, "task_id": entry.task_id, "outcome": entry.named_evidence})
    else:
        receipt.write_text("[1]" if damage == "wrong_type" else "{broken", encoding="utf-8")
    assert not body.saved_named_evidence(tmp_path, entry)[1]
    assert collection_due(current.run(tmp_path, entry.run_id), generation=runtime.runtime_id)
    tick()
    fresh = current.run(tmp_path, entry.run_id)
    assert body.saved_named_evidence(tmp_path, fresh)[1]["status"] == "retained"
    assert not collection_due(fresh, generation=runtime.runtime_id)
    assert area.is_dir() and not fresh.patch_disposed


def test_transient_capture_retries_once_in_the_next_generation(tmp_path, monkeypatch):
    entry, area, _ = _prepared(tmp_path)
    copy = shutil.copyfileobj
    monkeypatch.setattr(shutil, "copyfileobj", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EBUSY, "locked file")))
    first = body.collect_terminal(tmp_path, entry.run_id, detail=terminal(entry.run_id), generation="one")
    assert first["workspace_capture"]["named_evidence"]["status"] == "capture_failed"
    fresh = current.run(tmp_path, entry.run_id)
    assert not collection_due(fresh, generation="one") and collection_due(fresh, generation="two")
    monkeypatch.setattr(shutil, "copyfileobj", copy)
    body.collect_terminal(tmp_path, entry.run_id, generation="two")
    fresh = current.run(tmp_path, entry.run_id)
    assert fresh.named_evidence["status"] == "retained" and not current.collection_pending(fresh)
    assert not collection_due(fresh, generation="three") and area.is_dir()


def test_permission_failure_rests_until_named_source_changes(tmp_path, monkeypatch):
    entry, area, _ = _prepared(tmp_path)
    attempts, copy = [], shutil.copyfileobj
    def denied(*args, **kwargs):
        attempts.append(True)
        raise PermissionError(errno.EACCES, "source is unreadable")
    monkeypatch.setattr(shutil, "copyfileobj", denied)
    body.collect_terminal(tmp_path, entry.run_id, detail=terminal(entry.run_id), generation="one")
    failed = current.run(tmp_path, entry.run_id)
    receipt, saved = body.saved_named_evidence(tmp_path, failed)
    assert saved["status"] == "capture_failed" and saved["retry"] == "changed_input"
    contents = receipt.read_bytes()
    for generation in ("two", "three"):
        assert not collection_due(failed, generation=generation)
        body.collect_terminal(tmp_path, entry.run_id, generation=generation)
    assert attempts == [True] and receipt.read_bytes() == contents
    assert list(receipt.parent.parent.glob("evidence*")) == [receipt.parent]
    monkeypatch.setattr(shutil, "copyfileobj", copy)
    source = area / checking.EVIDENCE_DIRNAME / "result.bin"
    source.write_bytes(b"changed readable source")
    latest = named.run_named_evidence(tmp_path, failed)
    assert latest["status"] == "retained" and latest["evidence_dir"] != saved["evidence_dir"]
    assert receipt.read_bytes() == contents
    assert (pathlib.Path(latest["evidence_dir"]) / "files" / checking.EVIDENCE_DIRNAME / "result.bin").read_bytes() == source.read_bytes()


@pytest.mark.parametrize("staging", ["directory", "linked_receipt"])
def test_unpublished_staging_has_a_remover_but_published_receipts_survive(tmp_path, monkeypatch, staging):
    entry, _ = _captured(tmp_path, monkeypatch)
    receipt, _ = body.saved_named_evidence(tmp_path, entry)
    abandoned = receipt.parent.parent / ".evidence-interrupted.pending"
    if staging == "directory":
        abandoned.mkdir()
        (abandoned / "bytes").write_bytes(b"incomplete")
    else:
        abandoned.symlink_to(receipt.parent, target_is_directory=True)
    named.run_named_evidence(tmp_path, entry, recapture=True)
    assert not abandoned.exists() and receipt.is_file()
