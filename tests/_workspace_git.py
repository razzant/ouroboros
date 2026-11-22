"""Git repositories for native workspace capture and integration tests, copied from the run's Git worlds.

The plain Git repository a test starts from, unborn or with one empty commit, is a world of its own
(tests/_worlds.py): ``git_repository(tmp_path / name)`` copies it, where ``git init`` and a commit cost two processes.
A repository whose files are fixed is a world too: its build lays it with ``git_repository`` and ``world_git``, and
each test copies it with ``clone_repository``.
"""
import functools
import json
import os
import subprocess
from pathlib import Path

from tests._worlds import clone, world

FIXTURE_AUTHOR = ("user.name=Fixture", "user.email=fixture@example.invalid")


def git(root, *args, env=None):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, env=env).stdout


def _git_environment() -> dict[str, str]:
    """The run's environment without a test's own Git state (``GIT_DIR``, an author, a date): the first consumer's
    fixtures set it before the build. Configuration stays what tests/conftest.py gives every test: none."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    return env


def _lay_git_repository(folder: Path, *, unborn: bool, identity: bool) -> None:
    env = _git_environment()
    # no sample hooks: a test that wants hooks brings its own template
    git(folder, "init", "-q", "--template=", "-b", "master", env=env)
    if identity:
        for setting in FIXTURE_AUTHOR:
            git(folder, "config", *setting.split("=", 1), env=env)
    if not unborn:
        git(folder, "-c", FIXTURE_AUTHOR[0], "-c", FIXTURE_AUTHOR[1], "-c", "commit.gpgsign=false",
            "commit", "-q", "--allow-empty", "-m", "base", env=env)
    # The world serves the whole run: a stub of subprocess.run in place at the first build fails here, in its own test.
    if not (folder / ".git" / "HEAD").is_file():
        raise RuntimeError("no repository was laid: something stood in for git during the build")


def git_repository(destination: Path, *, unborn: bool = False, identity: bool = False) -> Path:
    """A private Git repository at ``destination``, which must not exist: one empty commit ``base`` by the fixture's
    author on ``master``, or ``unborn``. ``identity`` keeps that author in its configuration for the test's own
    commits. ``master`` is what ``git init`` names a branch where no configuration says otherwise, as in tests.
    Its index holds no entry, so a copy is clean as it lands and needs no Git index refresh. The copy itself is
    ``clone``'s: clonefile on macOS, one ``cp --reflink=auto`` process on Linux, copytree on Windows."""
    key = f"git-{'unborn' if unborn else 'base'}{'-identity' if identity else ''}"
    build = functools.partial(_lay_git_repository, unborn=unborn, identity=identity)
    return clone(world(key, build), destination)


def _subcommand(args):
    """The git subcommand of ``args``: the first word that is no option (``-c``/``-C`` take a value)."""
    words = iter(args)
    for word in words:
        if word in ("-c", "-C"):
            next(words, None)
        elif not word.startswith("-"):
            return word
    return None


def world_git(root, *args):
    """``git`` in a world's build: the first consumer's own Git state (``GIT_*``, system and global
    configuration) stays out of these build commands; HOME, XDG and TZ are inherited from the caller.
    The world serves the whole run, so a ``commit`` after which HEAD names no commit, because something stood in
    for git at the first build, fails in that test instead of publishing an unborn tree."""
    env = _git_environment()
    out = git(root, *args, env=env)
    if _subcommand(args) == "commit":
        head = git(root, "rev-parse", "--verify", "-q", "HEAD^{commit}", env=env).strip()
        if len(head) not in (40, 64) or head.strip(b"0123456789abcdef"):
            raise RuntimeError("no commit was laid: something stood in for git during the build")
    return out


def repository(root, *, unborn=False):
    """One empty commit ``base`` by Fixture, or ``unborn``."""
    return git_repository(root, unborn=unborn)


def seeded_repository(path, files):
    """One root commit ``fixture`` holding ``files``, over the unborn world that keeps Fixture as the author the
    tests' own later commits read: one world for every file set."""
    git_repository(path, unborn=True, identity=True)
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
