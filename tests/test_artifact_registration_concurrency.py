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
    original = artifacts.artifact_record

    def prepared(*args, **kwargs):
        record = original(*args, **kwargs)
        if not getattr(local, "prepared", False):
            local.prepared = True
            ready.wait()  # all producers have finished bytes before registration
        return record

    monkeypatch.setattr(artifacts, "artifact_record", prepared)
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
