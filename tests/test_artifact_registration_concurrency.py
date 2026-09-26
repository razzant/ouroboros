"""Distinct artifact producers merge registrations without losing each other."""

from concurrent.futures import ThreadPoolExecutor
import json
import threading
from types import SimpleNamespace

from ouroboros import artifacts


def test_review_file_and_directory_registration_preserve_concurrent_records(tmp_path, monkeypatch):
    ctx = SimpleNamespace(drive_root=tmp_path / "data", task_id="registration")
    source = tmp_path / "answer.txt"
    source.write_text("the requested answer")
    directory = tmp_path / "results"
    directory.mkdir()
    (directory / "part.txt").write_text("directory member")
    ready = threading.Barrier(3, timeout=10)
    local = threading.local()
    original = artifacts._register_task_artifact_records

    def prepared(*args, **kwargs):
        if not getattr(local, "prepared", False):
            local.prepared = True
            ready.wait()  # all producers have finished bytes before registration
        return original(*args, **kwargs)

    monkeypatch.setattr(artifacts, "_register_task_artifact_records", prepared)
    with ThreadPoolExecutor(max_workers=3) as workers:
        review = workers.submit(
            artifacts.store_task_artifact_bytes, ctx.drive_root, ctx.task_id,
            "review.json", b'{"verdict":"PASS"}', kind="task_acceptance_review",
        )
        file = workers.submit(artifacts.copy_file_to_task_artifacts, ctx, source)
        bundle = workers.submit(artifacts.copy_directory_to_task_artifacts, ctx, directory)
        review_ref, file_record, directory_records = review.result(), file.result(), bundle.result()
    root = artifacts.task_artifact_dir_path(ctx.drive_root, ctx.task_id, create=False)
    manifest = json.loads((root / artifacts._ARTIFACT_MANIFEST).read_text())["artifacts"]
    expected = {review_ref["path"], file_record["name"], *(r["name"] for r in directory_records)}
    assert set(manifest) == expected
    assert manifest["review.json"]["kind"] == "task_acceptance_review"
    assert manifest[file_record["name"]]["source_path"] == str(source.resolve())


def test_live_manifest_lock_is_private_but_user_lock_named_file_is_an_artifact(tmp_path):
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    artifacts.store_task_artifact_bytes(tmp_path, "registration", "notes.lock", b"requested file")
    root = artifacts.task_artifact_dir_path(tmp_path, "registration", create=False)
    lock_path = root / (artifacts._ARTIFACT_MANIFEST + ".lock")
    fd = acquire_exclusive_file_lock(lock_path)
    assert fd is not None
    try:
        rows = artifacts.collect_task_artifact_records(tmp_path, "registration")
        assert [row["name"] for row in rows] == ["notes.lock"]
    finally:
        release_exclusive_file_lock(lock_path, fd)


def test_identical_registration_leaves_the_manifest_untouched_and_a_change_rewrites(tmp_path):
    """Re-registering an unchanged record is a no-op on disk (issue #1230); a
    changed record still rewrites the manifest."""
    import os

    root = tmp_path / "artifacts"
    root.mkdir()
    record = {"kind": "child_artifact", "name": "a.txt", "path": str(root / "a.txt"),
              "size": 3, "sha256": "aa" * 32, "status": "ready", "errors": [], "source_path": "/src/a.txt"}
    artifacts._register_task_artifact_records(root, [record])
    manifest = root / artifacts._ARTIFACT_MANIFEST
    before = manifest.read_bytes()
    stat_before = os.stat(manifest)

    artifacts._register_task_artifact_records(root, [dict(record)])

    stat_after = os.stat(manifest)
    assert manifest.read_bytes() == before
    assert (stat_after.st_mtime_ns, stat_after.st_ino) == (stat_before.st_mtime_ns, stat_before.st_ino)

    artifacts._register_task_artifact_records(root, [{**record, "sha256": "bb" * 32, "size": 4}])

    changed = json.loads(manifest.read_text(encoding="utf-8"))["artifacts"]["a.txt"]
    assert changed["sha256"] == "bb" * 32 and changed["size"] == 4
    assert manifest.read_bytes() != before
