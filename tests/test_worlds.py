"""Shared worlds (tests/_worlds.py): built once per run, cloned private, refused when they know their own path. Beside
them, the whole-tree scans' parse (tests/_source_scan.py).
"""
from __future__ import annotations

import ast
import gc
import hashlib
import os
import pathlib
import stat
import subprocess
import sys
import textwrap

import pytest

from ouroboros.test_environment import SCRATCH_MARKER
from tests import _contributor_packet_shared as shared
from tests import _worlds
from tests._live_world import _own_installation, installation_template
from tests._source_scan import parse
from tests._worlds import clone, run_folder, world

REPO = pathlib.Path(__file__).resolve().parents[1]
RACERS = 3


@pytest.fixture
def run_root(tmp_path, monkeypatch):
    """A run of its own: what a test builds here must not reach the real run's templates."""
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setenv("OUROBOROS_TEST_TEMP_ROOT", str(root))
    return root


def _lay_out(folder: pathlib.Path) -> None:
    (folder / "nested").mkdir()
    (folder / "nested" / "data.txt").write_text("the template's bytes\n", encoding="utf-8")
    (folder / "locked.txt").write_text("read only\n", encoding="utf-8")
    os.chmod(folder / "locked.txt", stat.S_IREAD)
    os.symlink(os.path.join("nested", "data.txt"), folder / "link")


def _tree(root: pathlib.Path) -> dict:
    """Every path beneath ``root``: a file by its SHA-256, a link by its target, a folder by that fact."""
    entries = {}
    for directory, folders, files in os.walk(root):
        for name in (*folders, *files):
            path = pathlib.Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                entries[relative] = "-> " + os.readlink(path)
            elif path.is_file():
                entries[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                entries[relative] = "folder"
    return entries


def test_a_clone_is_the_template_and_nothing_it_does_reaches_the_template_or_another_clone(run_root, tmp_path):
    template = world("files", _lay_out)
    first, second = clone(template, tmp_path / "first"), clone(template, tmp_path / "second")
    before = _tree(template)
    assert _tree(first) == _tree(second) == before and len(before) == 4
    assert (first / "link").is_symlink() and not (first / "locked.txt").stat().st_mode & stat.S_IWUSR
    for timed in ("nested", "nested/data.txt"):
        assert (first / timed).stat().st_mtime_ns == (template / timed).stat().st_mtime_ns, timed

    (first / "nested" / "data.txt").write_text("this test's own bytes\n", encoding="utf-8")
    (first / "added.txt").write_text("new\n", encoding="utf-8")
    (first / "link").unlink()
    (second / "nested" / "data.txt").unlink()
    assert _tree(template) == before, "a write into a clone reached its template"
    assert (second / "link").is_symlink() and (first / "added.txt").exists() and not (second / "added.txt").exists()
    third = clone(template, tmp_path / "third")
    assert _tree(third) == before, "a later clone carried an earlier clone's change"
    linked = [path for path in (third / "nested" / "data.txt", third / "locked.txt") if path.stat().st_nlink != 1]
    assert not linked, f"hardlinks: {linked}"
    with pytest.raises(FileExistsError):
        clone(template, tmp_path / "first")


def test_a_build_runs_once_and_a_failed_build_publishes_nothing(run_root):
    calls = []

    def fails_first(folder):
        calls.append(folder)
        (folder / "half.txt").write_text("half a world\n", encoding="utf-8")
        if len(calls) == 1:
            raise RuntimeError("the build broke")
        (folder / "rest.txt").write_text("the rest\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="the build broke"):
        world("flaky", fails_first)
    assert not (run_folder("worlds") / "flaky").exists() and not calls[0].exists()
    template = world("flaky", fails_first)
    assert world("flaky", fails_first) == template and len(calls) == 2
    assert sorted(path.name for path in template.iterdir()) == ["half.txt", "rest.txt"]
    assert world("another", _lay_out) != template


@pytest.mark.parametrize("where", ["file", "link", "folder-link"])
def test_a_world_that_knows_its_own_path_is_refused_and_a_relative_one_is_not(run_root, where):
    def knows(folder):
        if where == "file":
            (folder / "record.json").write_text(f'{{"root": "{folder}"}}', encoding="utf-8")
        elif where == "link":
            os.symlink(folder / "elsewhere", folder / "link")
        else:  # a link to a folder: every clone would write through it into the template
            (folder / "shared").mkdir()
            os.symlink(folder / "shared", folder / "link", target_is_directory=True)

    with pytest.raises(ValueError, match="holds the path it was built at"):
        world("knows-its-path", knows)
    assert not (run_folder("worlds") / "knows-its-path").exists()
    assert (world("relative", _lay_out) / "link").is_symlink()


_RACER = textwrap.dedent("""
    import os, sys, time
    sys.path.insert(0, sys.argv[1])
    from tests._worlds import world

    def build(folder):
        with open(sys.argv[2], "a", encoding="utf-8") as stream:
            stream.write(f"{os.getpid()}\\n")
        time.sleep(0.5)  # the other racers reach the lock meanwhile
        (folder / "built.txt").write_text(str(os.getpid()), encoding="utf-8")

    print(world("raced", build), flush=True)
""")
_KILLED_BUILDER = textwrap.dedent("""
    import os, sys, time
    sys.path.insert(0, sys.argv[1])
    from ouroboros.platform_layer import file_lock_exclusive
    file_lock_exclusive(os.open(sys.argv[2], os.O_CREAT | os.O_RDWR))
    print("holding", flush=True)
    time.sleep(120)
""")


def test_the_processes_of_a_run_share_one_build_and_a_killed_builder_holds_nobody_up(run_root, tmp_path):
    log = tmp_path / "builds.txt"
    racers = [subprocess.Popen([sys.executable, "-c", _RACER, str(REPO), str(log)], stdout=subprocess.PIPE, text=True)
              for _ in range(RACERS)]
    answers = [racer.communicate(timeout=120)[0].strip() for racer in racers]
    assert [racer.returncode for racer in racers] == [0] * RACERS and len(set(answers)) == 1
    assert len(log.read_text(encoding="utf-8").splitlines()) == 1, "two processes built one world"

    lock = run_folder("worlds") / "killed.lock"
    holder = subprocess.Popen([sys.executable, "-c", _KILLED_BUILDER, str(REPO), str(lock)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "holding"
    finally:
        holder.kill()
        holder.communicate(timeout=60)
    assert (world("killed", _lay_out) / "nested" / "data.txt").is_file()


def test_the_copy_is_the_same_where_nothing_copies_on_write(run_root, tmp_path, monkeypatch):
    template = world("files", _lay_out)
    monkeypatch.setattr(_worlds, "_copy_on_write", lambda _template, _destination: False)
    assert _tree(clone(template, tmp_path / "copied")) == _tree(template)

    def stopped_half_way(_template, destination):
        destination.mkdir()
        (destination / "half.txt").write_text("half a copy\n", encoding="utf-8")
        return False

    monkeypatch.setattr(_worlds, "_copy_on_write", stopped_half_way)
    assert _tree(clone(template, tmp_path / "again")) == _tree(template)


def test_the_owners_release_takes_the_declared_folder_and_leaves_what_it_does_not_know(run_root):
    from ouroboros.safe_test_supervisor import release_scratch

    world("files", _lay_out)
    assert (run_folder("worlds") / SCRATCH_MARKER).is_file()
    unknown = run_root / "command-data"
    unknown.mkdir()
    release_scratch(run_root.resolve())
    assert not (run_root / "worlds").exists() and unknown.is_dir()


def test_bare_pytest_keeps_its_own_root_until_the_process_exits(monkeypatch):
    monkeypatch.delenv("OUROBOROS_TEST_TEMP_ROOT", raising=False)
    monkeypatch.setattr(_worlds, "_bare_root", None)
    template = world("files", _lay_out)
    private = pathlib.Path(_worlds._bare_root.name).resolve()
    assert template == private / "worlds" / "files" and world("files", _lay_out) == template
    _worlds._bare_root.cleanup()
    assert not private.exists()


def test_a_cloned_checkout_is_stat_clean_and_its_own_repository(run_root, tmp_path):
    fixture = shared.init_installed_body(tmp_path / "one")
    other = shared.init_installed_body(tmp_path / "two")
    repo, sibling = pathlib.Path(fixture["repo"]), pathlib.Path(other["repo"])
    template = run_folder("worlds") / "installed-body" / "installed"
    assert {key: fixture[key] for key in fixture if key != "repo"} == {key: other[key] for key in other if key != "repo"}
    for spelling, sha in (("HEAD", fixture["base_sha"]), ("proposal", fixture["head_sha"]),
                          ("HEAD^{tree}", shared.git(template, "rev-parse", "base^{tree}"))):
        assert shared.git(repo, "rev-parse", spelling) == shared.git(template, "rev-parse", spelling) == sha
    # Plumbing reads the index's stat data: a clone that skipped its refresh reports every file changed.
    shared.git(repo, "diff-files", "--quiet")
    shared.git(repo, "diff-index", "--quiet", "HEAD")

    (repo / "README.md").write_text("edited in one clone\n", encoding="utf-8")
    shared.git(repo, "add", "-A")
    shared.git(repo, "commit", "-q", "-m", "only here")
    shared.git(repo, "branch", "only-here")
    assert shared.git(repo, "rev-parse", "HEAD") != fixture["base_sha"]
    assert shared.git(template, "rev-parse", "HEAD") == shared.git(sibling, "rev-parse", "HEAD") == fixture["base_sha"]
    assert shared.git(template, "branch", "--list", "only-here") == shared.git(sibling, "branch", "--list", "only-here") == ""
    assert shared.git(sibling, "status", "--porcelain") == ""


def test_the_installation_a_host_starts_from_is_a_private_copy(run_root, tmp_path):
    env, memory, rooms = _own_installation(tmp_path / "one", books=False)
    other, _memory, same = _own_installation(tmp_path / "two", books=False)
    template = installation_template(books=False) / "world"
    before = _tree(template)
    assert rooms == same and set(rooms) == {"alpha", "beta"} and env.drive_root != other.drive_root
    memory.scratchpad_path().write_text("this test's scratchpad", encoding="utf-8")
    assert (other.drive_root / "memory" / "scratchpad.md").read_text(encoding="utf-8") == "scratch"
    assert _tree(template) == before


def test_a_parse_pauses_the_collector_and_leaves_it_as_it_was_found():
    assert ast.dump(parse("x = 1\n")) == ast.dump(ast.parse("x = 1\n"))
    assert isinstance(parse("1 + 1", mode="eval").body, ast.BinOp)
    collections = []

    def noted(phase, _info):
        collections.append(phase)

    gc.callbacks.append(noted)
    try:
        parse("\n".join(f"name_{index} = [{index}]" for index in range(5000)))
    finally:
        gc.callbacks.remove(noted)
    assert not collections, "the collector ran during a parse"
    assert gc.isenabled()
    with pytest.raises(SyntaxError):
        parse("def broken(:\n")
    assert gc.isenabled()
    gc.disable()
    try:
        parse("x = 1\n")
        assert not gc.isenabled(), "a parse switched on a collector its caller had off"
    finally:
        gc.enable()
