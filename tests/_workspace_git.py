"""Git repositories for native workspace capture and integration tests."""
import json
from pathlib import Path
import subprocess


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout


def repository(root, *, unborn=False):
    root.mkdir()
    git(root, "init")
    if not unborn:
        git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
            "commit", "--allow-empty", "-m", "base")
    return root


def seeded_repository(path, files):
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.name", "Fixture")
    git(path, "config", "user.email", "fixture@example.invalid")
    for name, content in files.items():
        (path / name).write_bytes(content)
    git(path, "add", "-A")
    git(path, "commit", "-qm", "fixture")
    return path


def commit(root):
    git(root, "add", ".")
    git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "files")
    return git(root, "rev-parse", "HEAD").decode().strip()


def capture_manifest_fields(monkeypatch, *, omit=(), **fields):
    """Choose the fixture's manifest fields before the Git writer measures its published bytes."""
    from ouroboros import workspace_patch_capture

    write_json = workspace_patch_capture.atomic_write_json

    def write_manifest(path, manifest, **options):
        for key in omit:
            manifest.pop(key, None)
        manifest.update(fields)
        return write_json(path, manifest, **options)

    monkeypatch.setattr(workspace_patch_capture, "atomic_write_json", write_manifest)


def publish_capture(drive, task):
    """Finish the child and read the manifest at the address its finalizer published."""
    from ouroboros.headless import finalize_task_artifacts
    from ouroboros.task_results import write_task_result

    write_task_result(drive, task["id"], "completed", **{k: v for k, v in task.items() if k != "id"})
    records = finalize_task_artifacts(drive, task)
    [path] = [Path(row["path"]) for row in records if row["kind"] == "workspace_patch_manifest"]
    return path.parent, json.loads(path.read_bytes())


def retire_published_capture(parent, child, task_id, *, drive_owner=None):
    """Copy back an already finalized capture, then retire its real child store."""
    from ouroboros.headless import copy_child_task_result, retry_child_task_refs
    from ouroboros.task_custody import settle_child_drive
    from ouroboros.task_results import load_task_result, write_task_result

    row = load_task_result(child, task_id)
    write_task_result(parent, task_id, "completed", child_drive_root=str(child),
                      artifact_status=row["artifact_status"], artifact_finalized_at=row["artifact_finalized_at"])
    copy_child_task_result(parent, {"id": task_id, "drive_root": str(child), "workspace_root": row["workspace_root"]})
    retry_child_task_refs(parent, child, task_id)
    settled = settle_child_drive(parent, drive_owner or task_id, child, live=lambda _: False)
    assert settled["status"] == "removed", settled
    assert not child.exists()
