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


def publish_capture(drive, task):
    """Finish a Git child and read the manifest at the address its finalizer published."""
    from ouroboros.headless import finalize_task_artifacts
    from ouroboros.task_results import write_task_result

    write_task_result(drive, task["id"], "completed", **{k: v for k, v in task.items() if k != "id"})
    records = finalize_task_artifacts(drive, task)
    [path] = [Path(row["path"]) for row in records if row["kind"] == "workspace_patch_manifest"]
    return path.parent, json.loads(path.read_bytes())
