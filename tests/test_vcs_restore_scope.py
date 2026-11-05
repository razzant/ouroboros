"""Real Git through the registered ``vcs_restore``: the promised HEAD state, inside the scope only.

#2433: a restore must match HEAD in the index and the files of its scope, staged additions and
untracked files included, whatever mix of changes the scope holds; ignored files, symlink targets
and everything outside the scope keep their bytes and index entries. The preview names the same
scope, staged changes and additions included, and an actual Git failure is reported as such.
A file ignored before the restore stays even when the restore removes the rule that ignored it,
and a workspace folder inside an enclosing repository is the whole scope. A path that changed
between file and folder returns in an order that never deletes what the restore keeps.
"""
from __future__ import annotations

import os
import pathlib
import subprocess

import pytest


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout.decode("utf-8")


def _status(repo: pathlib.Path, *paths: str) -> list:
    rows = _git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames",
                *(["--", *paths] if paths else [])).split("\0")
    return sorted(row for row in rows if row)


@pytest.fixture(autouse=True)
def _allow_safety(monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))


@pytest.fixture
def project(tmp_path):
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    repo, data, system = tmp_path / "project", tmp_path / "data", tmp_path / "system"
    for path in (repo, data, system):
        path.mkdir()
    for root in (repo, system):
        for args in (("init", "-q"), ("config", "user.name", "Test"), ("config", "user.email", "test@example.invalid"),
                     ("config", "core.autocrlf", "false")):
            _git(root, *args)
    (system / "system.txt").write_text("system\n", encoding="utf-8")
    _git(system, "add", "-A")
    _git(system, "commit", "-qm", "system")
    (repo / "dir").mkdir()
    for name in ("tracked.txt", "deleted.txt", "old.txt", "neighbor.txt", "dir/a.txt"):
        (repo / name).write_text(f"{name} base\n", encoding="utf-8")
    (repo / ".gitignore").write_text("*.ignored\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    registry = ToolRegistry(system, data)
    registry.set_context(ToolContext(repo_dir=system, system_repo_dir=system, drive_root=data,
                                     workspace_root=repo, workspace_mode="external", task_id="restore-test"))
    return registry, repo


def _restore(registry, **args) -> str:
    return registry.execute("vcs_restore", {"confirm": True, **args})


def _outside(repo: pathlib.Path) -> tuple:
    """Every byte and index entry a scoped restore of ``work``/``dir`` must leave alone."""
    (repo / "neighbor.txt").write_text("unrelated working change\n", encoding="utf-8")
    (repo / "staged-neighbor.txt").write_text("unrelated staged addition\n", encoding="utf-8")
    _git(repo, "add", "staged-neighbor.txt")
    (repo / "keep.ignored").write_text("ignored bytes\n", encoding="utf-8")
    return lambda: ((repo / "neighbor.txt").read_text(encoding="utf-8"),
                    (repo / "staged-neighbor.txt").read_text(encoding="utf-8"),
                    (repo / "keep.ignored").read_text(encoding="utf-8"),
                    _git(repo, "ls-files", "-s", "--", "neighbor.txt", "staged-neighbor.txt"))


def _changes(repo: pathlib.Path, form: str) -> list:
    """One scope's uncommitted change of the named form; returns the paths to restore."""
    if form == "staged_addition":
        (repo / "new.txt").write_text("staged new\n", encoding="utf-8")
        _git(repo, "add", "new.txt")
        return ["new.txt"]
    if form == "untracked_only":
        (repo / "untracked.txt").write_text("untracked\n", encoding="utf-8")
        return ["untracked.txt"]
    if form == "modified":
        (repo / "tracked.txt").write_text("staged edit\n", encoding="utf-8")
        _git(repo, "add", "tracked.txt")
        (repo / "tracked.txt").write_text("staged edit then working edit\n", encoding="utf-8")
        return ["tracked.txt"]
    if form == "deleted":
        _git(repo, "rm", "-q", "deleted.txt")
        return ["deleted.txt"]
    if form == "rename":
        _git(repo, "mv", "old.txt", "renamed.txt")
        return ["old.txt", "renamed.txt"]
    if form == "directory":
        (repo / "dir" / "a.txt").write_text("edited\n", encoding="utf-8")
        (repo / "dir" / "staged.txt").write_text("staged new\n", encoding="utf-8")
        _git(repo, "add", "dir/staged.txt")
        (repo / "dir" / "nested").mkdir()
        (repo / "dir" / "nested" / "loose.txt").write_text("untracked\n", encoding="utf-8")
        return ["dir"]
    assert form == "mixed"
    return [*_changes(repo, "staged_addition"), *_changes(repo, "untracked_only"), *_changes(repo, "modified")]


@pytest.mark.parametrize("form", ["staged_addition", "rename",
                                  "mixed"])
def test_scoped_restore_matches_head_inside_and_keeps_everything_outside(project, form):
    registry, repo = project
    outside = _outside(repo)
    paths = _changes(repo, form)
    before = outside()

    result = _restore(registry, paths=paths)

    assert "Restored" in result and "RESTORE_" not in result, result
    assert _status(repo, *paths) == []
    assert not (repo / "new.txt").exists() and not (repo / "untracked.txt").exists()
    assert not (repo / "dir" / "staged.txt").exists() and not (repo / "dir" / "nested").exists()
    assert outside() == before


def test_whole_restore_claims_head_only_when_it_holds(project):
    registry, repo = project
    _changes(repo, "mixed")
    _changes(repo, "directory")
    (repo / "keep.ignored").write_text("ignored bytes\n", encoding="utf-8")

    result = _restore(registry)

    assert "Working directory matches HEAD" in result, result
    assert _status(repo) == [] and _git(repo, "diff", "--cached", "--name-only") == ""
    assert (repo / "keep.ignored").read_text(encoding="utf-8") == "ignored bytes\n"


@pytest.mark.skipif(os.name == "nt", reason="symlinks need a privilege Windows runners lack")
def test_restore_never_follows_a_symlink_into_its_target(project, tmp_path):
    registry, repo = project
    target = tmp_path / "outside-target"
    target.mkdir()
    (target / "precious.txt").write_text("outside the repository\n", encoding="utf-8")
    (repo / "link").symlink_to(target, target_is_directory=True)
    (repo / "tracked-link").symlink_to("tracked.txt")
    _git(repo, "add", "tracked-link")
    _git(repo, "commit", "-qm", "tracked link")
    (repo / "tracked-link").unlink()
    (repo / "tracked-link").symlink_to("neighbor.txt")

    result = _restore(registry)

    assert "Working directory matches HEAD" in result, result
    assert not os.path.lexists(repo / "link") and os.readlink(repo / "tracked-link") == "tracked.txt"
    assert (target / "precious.txt").read_text(encoding="utf-8") == "outside the repository\n"


@pytest.mark.parametrize("name,scope,neighbor", [
    ("[ab].txt", "*].txt", "a.txt"),  # as a pattern the name would also match a.txt
    pytest.param(":!x.txt", "*x.txt", "neighbor.txt",  # as magic: everything except x.txt
                 marks=pytest.mark.skipif(os.name == "nt", reason="':' is not a Windows file name character")),
])
def test_changed_names_are_restored_literally(project, name, scope, neighbor):
    """The restore acts on the exact changed names it read: a glob or magic-looking name never
    widens to a file the requested scope does not cover."""
    registry, repo = project
    (repo / name).write_text("base\n", encoding="utf-8")
    (repo / "a.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "names")
    (repo / name).write_text("changed\n", encoding="utf-8")
    (repo / neighbor).write_text("neighbor's own change\n", encoding="utf-8")

    result = _restore(registry, paths=[scope])

    assert "Restored 1 path(s)" in result, result
    assert (repo / name).read_text(encoding="utf-8") == "base\n"
    assert (repo / neighbor).read_text(encoding="utf-8") == "neighbor's own change\n"


def test_an_untracked_nested_repository_is_kept_and_reported(project):
    """``git clean -fd`` deliberately skips a nested repository, whose history may exist nowhere
    else: the restore keeps it and says the scope still differs instead of claiming HEAD."""
    registry, repo = project
    (repo / "vendor").mkdir()
    _git(repo / "vendor", "init", "-q")
    (repo / "vendor" / "unique.txt").write_text("history only here\n", encoding="utf-8")
    _changes(repo, "modified")

    result = _restore(registry)

    assert "RESTORE_PARTIAL" in result and "vendor/" in result and "matches HEAD" not in result, result
    assert (repo / "vendor" / "unique.txt").read_text(encoding="utf-8") == "history only here\n"
    assert _status(repo) == ["?? vendor/"]


@pytest.mark.parametrize("paths", [None, ["new-folder"]])
def test_a_file_ignored_before_the_restore_survives_its_rule_being_restored_away(project, paths):
    """Removing a staged ``.gitignore`` makes the file it ignored untracked; the cleanup acts only on the
    untracked names it read before, so that file stays and the restore reports it instead of HEAD."""
    registry, repo = project
    (repo / "new-folder").mkdir()
    (repo / "new-folder" / ".gitignore").write_text("private.txt\n", encoding="utf-8")
    _git(repo, "add", "new-folder/.gitignore")
    (repo / "new-folder" / "private.txt").write_text("ignored before the restore\n", encoding="utf-8")
    (repo / "new-folder" / "scratch.txt").write_text("untracked\n", encoding="utf-8")

    result = _restore(registry, **({"paths": paths} if paths else {}))

    assert "RESTORE_PARTIAL" in result and "new-folder/private.txt" in result, result
    assert "matches HEAD" not in result and "Restored" not in result
    assert (repo / "new-folder" / "private.txt").read_text(encoding="utf-8") == "ignored before the restore\n"
    assert not (repo / "new-folder" / "scratch.txt").exists() and not (repo / "new-folder" / ".gitignore").exists()
    assert _git(repo, "ls-files", "--", "new-folder") == ""
    assert _status(repo) == ["?? new-folder/private.txt"]


@pytest.mark.parametrize("paths", [None, ["a.txt"]])
def test_a_folder_inside_an_enclosing_repository_is_the_whole_scope(project, paths):
    """Status names rows from the repository's top level while a requested path resolves from the
    folder: both scopes restore, and nothing outside the folder is previewed or touched."""
    from ouroboros.tools.registry import ToolContext

    registry, repo = project
    folder, system = repo / "dir", repo.parent / "system"
    registry.set_context(ToolContext(repo_dir=system, system_repo_dir=system, drive_root=repo.parent / "data",
                                     workspace_root=folder, workspace_mode="external", task_id="restore-test"))
    outside = _outside(repo)
    _changes(repo, "directory")  # dir/a.txt edited, dir/staged.txt staged, dir/nested/loose.txt untracked
    before = outside()

    preview = registry.execute("vcs_restore", {"confirm": False, **({"paths": paths} if paths else {})})
    result = _restore(registry, **({"paths": paths} if paths else {}))

    assert "dir/a.txt" in preview and "neighbor.txt" not in preview, preview
    assert ("Restored 1 path(s)" if paths else "matches HEAD") in result and "RESTORE_" not in result, result
    assert (folder / "a.txt").read_text(encoding="utf-8") == "dir/a.txt base\n"
    assert outside() == before
    if paths:  # the rest of the folder keeps its changes
        assert _status(repo, "dir") == ["?? dir/nested/loose.txt", "A  dir/staged.txt"]
    else:
        assert _status(repo, "dir") == [] and not (folder / "nested").exists()


@pytest.mark.parametrize("paths", [None, ["dir"]])
def test_original_untracked_names_are_removed_even_when_restored_rules_ignore_them(project, paths):
    """Checkout may hide a cleanup target; cleanup still uses the original exact names.
    Initially ignored neighbors must survive that same operation."""
    registry, repo = project
    rule = repo / "dir" / ".gitignore"
    rule.write_text("scratch.txt\n", encoding="utf-8")
    _git(repo, "add", "dir/.gitignore")
    _git(repo, "commit", "-qm", "ignore the scratch file")
    rule.write_text("", encoding="utf-8")
    scratch, ignored = repo / "dir" / "scratch.txt", repo / "dir" / "keep.ignored"
    scratch.write_text("untracked before checkout\n", encoding="utf-8")
    ignored.write_text("ignored before checkout\n", encoding="utf-8")
    args = {"paths": paths} if paths else {}
    preview = registry.execute("vcs_restore", {"confirm": False, **args})
    assert "dir/scratch.txt" in preview and "keep.ignored" not in preview, preview

    result = _restore(registry, **args)

    assert not scratch.exists(), result
    assert ignored.read_text(encoding="utf-8") == "ignored before checkout\n"
    assert rule.read_text(encoding="utf-8") == "scratch.txt\n"
    assert "RESTORE_" not in result and _status(repo, "dir") == [], result


@pytest.mark.parametrize(("neighbor", "paths"), [
    ("loose.txt", None),
    pytest.param("keep.ignored", ["tracked.txt"], id='keep.ignored-paths1'),
])
def test_a_file_turned_into_a_folder_returns_without_deleting_kept_files(project, paths, neighbor):
    """HEAD's file returns once the staged child and the original untracked names are gone. Git writes
    it by deleting the folder in its way, so a folder still holding an ignored file is kept and reported."""
    registry, repo = project
    _git(repo, "rm", "-q", "tracked.txt")
    (repo / "tracked.txt").mkdir()
    (repo / "tracked.txt" / "new.txt").write_text("staged child\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt/new.txt")
    (repo / "tracked.txt" / neighbor).write_text("neighbor bytes\n", encoding="utf-8")

    result = _restore(registry, **({"paths": paths} if paths else {}))

    if neighbor == "keep.ignored":
        assert "RESTORE_PARTIAL" in result and "tracked.txt" in result, result
        assert (repo / "tracked.txt" / neighbor).read_text(encoding="utf-8") == "neighbor bytes\n"
        assert _status(repo, "tracked.txt") == ["D  tracked.txt"] and not (repo / "tracked.txt" / "new.txt").exists()
    else:
        assert ("Restored 3 path(s)" if paths else "matches HEAD") in result and "RESTORE_" not in result, result
        assert (repo / "tracked.txt").read_text(encoding="utf-8") == "tracked.txt base\n"
        assert _status(repo, "tracked.txt") == []


@pytest.mark.parametrize("paths,missing", [(None, False), (["dir"], False), (["dir/a.txt"], False),
                                            (["dir/a.txt"], True)])
def test_a_folder_turned_into_a_file_returns_inside_its_scope_only(project, paths, missing):
    """The reverse change returns HEAD's folder, untracked and ignored neighbors as their scope says;
    restoring only a child would delete the staged file outside that scope, so it is reported instead."""
    registry, repo = project
    _git(repo, "rm", "-rq", "dir")
    (repo / "dir").write_text("staged file\n", encoding="utf-8", newline="\n")  # LF: Git's bytes are compared below
    _git(repo, "add", "dir")
    if missing:
        (repo / "dir").unlink()  # the unique outside-scope bytes now live only in the index
    entries = _git(repo, "ls-files", "-s", "-z")
    (repo / "keep.ignored").write_text("ignored neighbor\n", encoding="utf-8")
    (repo / "loose.txt").write_text("untracked neighbor\n", encoding="utf-8")

    result = _restore(registry, **({"paths": paths} if paths else {}))

    assert (repo / "keep.ignored").read_text(encoding="utf-8") == "ignored neighbor\n"
    if paths == ["dir/a.txt"]:
        assert "RESTORE_PARTIAL" in result and "dir/a.txt" in result, result
        assert _git(repo, "ls-files", "-s", "-z") == entries
        assert _git(repo, "show", ":dir") == "staged file\n"
        if missing:
            assert not (repo / "dir").exists()
        else:
            assert (repo / "dir").read_text(encoding="utf-8") == "staged file\n"
        assert _status(repo, "dir") == ["AD dir" if missing else "A  dir", "D  dir/a.txt"]
    else:
        assert ("Restored 2 path(s)" if paths else "matches HEAD") in result and "RESTORE_" not in result, result
        assert (repo / "dir" / "a.txt").read_text(encoding="utf-8") == "dir/a.txt base\n"
        assert (repo / "loose.txt").exists() is bool(paths)  # the untracked neighbor goes with the whole scope


def test_preview_names_the_scope_with_staged_changes_and_additions(project):
    registry, repo = project
    _outside(repo)
    _changes(repo, "mixed")
    before = _status(repo)

    preview = registry.execute("vcs_restore", {"confirm": False, "paths": ["new.txt", "untracked.txt", "tracked.txt"]})

    assert "new.txt" in preview and "untracked.txt" in preview and "tracked.txt" in preview, preview
    assert "neighbor.txt" not in preview and "staged-neighbor.txt" not in preview
    assert "confirm=true" in preview and _status(repo) == before


def test_nothing_in_scope_is_an_honest_no_op(project):
    registry, repo = project
    _outside(repo)
    before = _status(repo)

    result = _restore(registry, paths=["tracked.txt"])

    assert "Nothing to restore" in result and "Restored" not in result, result
    assert _status(repo) == before


def test_index_lock_failure_is_reported_and_changes_nothing(project):
    registry, repo = project
    paths = _changes(repo, "mixed")
    (repo / ".git" / "index.lock").write_text("", encoding="utf-8")
    before = _status(repo)

    result = _restore(registry, paths=paths)

    assert "RESTORE_ERROR" in result and "index.lock" in result, result
    assert _status(repo) == before
