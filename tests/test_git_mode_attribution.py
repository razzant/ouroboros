"""Exact predecessor custody follows ordinary Git modes, not raw permissions."""
import copy
import hashlib
import os
import subprocess

import pytest

from ouroboros import mutation_attribution as attribution
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_mutation_attribution import _git, _repo
from tests.test_predecessor_mutation_handoff import _previous, _source, _start


@pytest.mark.parametrize("filemode", [True, False])
def test_foreign_git_mode_is_not_adopted(tmp_path, filemode):
    if filemode and os.name == "nt":
        pytest.skip("filesystem executable-bit probe; index-mode case covers Windows")
    root, data = _previous(tmp_path)
    _git(root, "config", "core.filemode", str(filemode).lower())
    if filemode:
        (root / "clean.txt").chmod(0o755)
    else:
        _git(root, "update-index", "--chmod=+x", "clean.txt")
    index = (root / ".git/index").read_bytes()
    _start(data, root, "successor", _source())
    selected, evidence, error = attribution.resolve_attributed_git_paths(data, "successor", root, None)
    assert selected == ["new.txt"] and not error
    assert evidence["excluded_preexisting_dirty"] == ["clean.txt", "dirty.txt"]
    assert (root / ".git/index").read_bytes() == index, "observation must preserve staged work"
    if filemode:
        _git(root, "add", "--", *selected)
        assert "mode change" not in _git(root, "diff", "--cached", "--summary")


@pytest.mark.parametrize("filemode", [True, False])
def test_predecessor_executable_correction_keeps_its_mode(tmp_path, filemode):
    if filemode and os.name == "nt":
        pytest.skip("filesystem executable-bit probe; index-mode case covers Windows")
    root, data = _previous(tmp_path)
    _git(root, "config", "core.filemode", str(filemode).lower())
    if filemode:
        (root / "clean.txt").chmod(0o755)
    else:
        _git(root, "update-index", "--chmod=+x", "clean.txt")
    terminal = attribution.record_terminal_mutation_candidates(data, "previous")
    assert terminal["terminal_candidate_snapshot"]["surfaces"][0]["candidate_fingerprints"]["clean.txt"]["git_mode"] == "100755"
    _start(data, root, "successor", _source())
    selected, _, error = attribution.resolve_attributed_git_paths(data, "successor", root, None)
    assert selected == ["clean.txt", "new.txt"] and not error
    _git(root, "add", "--", *selected)
    assert "mode change 100644 => 100755 clean.txt" in _git(root, "diff", "--cached", "--summary")


@pytest.mark.skipif(os.name == "nt", reason="requires meaningful POSIX chmod")
@pytest.mark.parametrize("filemode,mode", [(True, 0o654), (True, 0o645), (False, 0o755)])
def test_git_ignored_permission_changes_preserve_handoff(tmp_path, filemode, mode):
    root, data = _previous(tmp_path)
    _git(root, "config", "core.filemode", str(filemode).lower())
    (root / "clean.txt").chmod(mode)
    _start(data, root, "successor", _source())
    selected, _, error = attribution.resolve_attributed_git_paths(data, "successor", root, None)
    assert selected == ["clean.txt", "new.txt"] and not error
    _git(root, "add", "--", *selected)
    assert "mode change" not in _git(root, "diff", "--cached", "--summary")


def test_git_mode_capture_is_batched_and_read_only(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    _git(root, "config", "core.filemode", "false")
    paths = ["clean.txt", "dirty.txt"] + [f"new-{i}.txt" for i in range(20)]
    for path in paths[2:]:
        (root / path).write_text("new\n", encoding="utf-8")
    index_before = hashlib.sha256((root / ".git/index").read_bytes()).hexdigest()
    calls, run = [], subprocess.run

    def observe(argv, **kwargs):
        calls.append(argv)
        return run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", observe)
    fingerprints = attribution._git_path_fingerprints(root, paths)
    assert len(calls) == 2, "one config and one index read, independent of path count"
    assert all(row["sha256"] and row["git_mode"] == "100644" for row in fingerprints.values())
    assert hashlib.sha256((root / ".git/index").read_bytes()).hexdigest() == index_before


def test_indexed_symlink_mode_matches_git_on_a_regular_file(tmp_path):
    root = _repo(tmp_path)
    _git(root, "config", "core.symlinks", "false")
    link = root / "link"
    link.write_text("clean.txt", encoding="utf-8")
    blob = _git(root, "hash-object", "-w", "link")
    _git(root, "update-index", "--add", "--cacheinfo", f"120000,{blob},link")
    link.write_text("dirty.txt", encoding="utf-8")
    fingerprint = attribution._git_path_fingerprints(root, ["link"])["link"]
    assert fingerprint["kind"] == "file" and fingerprint["git_mode"] == "120000"
    _git(root, "add", "--", "link")
    assert _git(root, "ls-files", "--stage", "link").startswith("120000 ")


@pytest.mark.skipif(os.name == "nt", reason="requires meaningful POSIX chmod")
def test_predecessor_executable_addition_is_retained(tmp_path):
    root, data = _previous(tmp_path)
    _git(root, "config", "core.filemode", "true")
    (root / "new.txt").chmod(0o744)
    attribution.record_terminal_mutation_candidates(data, "previous")
    _start(data, root, "successor", _source())
    selected, _, error = attribution.resolve_attributed_git_paths(data, "successor", root, None)
    assert selected == ["clean.txt", "new.txt"] and not error
    _git(root, "add", "--", *selected)
    assert "create mode 100755 new.txt" in _git(root, "diff", "--cached", "--summary")


def test_new_files_deletions_and_symlinks_keep_exact_handoff(tmp_path):
    root, data = _previous(tmp_path)
    (root / "clean.txt").unlink()
    try:
        (root / "link").symlink_to("new.txt")
    except OSError:
        pytest.skip("native symlink creation unavailable")
    attribution.record_terminal_mutation_candidates(data, "previous")
    _start(data, root, "successor", _source())
    selected, _, error = attribution.resolve_attributed_git_paths(data, "successor", root, None)
    assert selected == ["clean.txt", "link", "new.txt"] and not error
    _git(root, "add", "--", *selected)
    summary = _git(root, "diff", "--cached", "--summary")
    assert "delete mode 100644 clean.txt" in summary
    assert "create mode 120000 link" in summary and "create mode 100644 new.txt" in summary


def test_legacy_mode_is_not_invented_for_exact_handoff(tmp_path):
    root, data = _previous(tmp_path)
    old = copy.deepcopy(load_task_result(data, "previous")["mutation_evidence"])
    for row in old["terminal_candidate_snapshot"]["surfaces"][0]["candidate_fingerprints"].values():
        row.pop("git_mode", None)
    write_task_result(data, "previous", "completed", mutation_evidence=old)
    _start(data, root, "successor", _source())
    assert attribution.attributed_git_candidates(data, "successor", root)["candidates"] == []
    assert load_task_result(data, "previous")["mutation_evidence"] == old


def test_legacy_excluded_wip_does_not_block_other_paths_or_epoch_advance(tmp_path):
    root, data = _previous(tmp_path)
    evidence = _start(data, root, "independent")
    for row in evidence["baseline"]["surfaces"][0]["git"]["dirty_fingerprints"].values():
        row.pop("git_mode", None)
    write_task_result(data, "independent", "running", mutation_evidence=evidence)
    (root / "mine.txt").write_text("independent work\n", encoding="utf-8")
    selected, _, error = attribution.resolve_attributed_git_paths(data, "independent", root, None)
    assert selected == ["mine.txt"] and not error
    _git(root, "add", "--", *selected)
    _git(root, "commit", "-qm", "fixture independent commit")
    advanced = attribution.advance_mutation_baseline(data, "independent", root)
    dirty = advanced["baseline"]["surfaces"][0]["git"]["dirty_fingerprints"]
    assert all(row["git_mode"] == "100644" for row in dirty.values())


@pytest.mark.skipif(os.name == "nt", reason="requires meaningful POSIX chmod")
def test_foreign_mode_change_is_visible_at_current_and_terminal_comparison(tmp_path):
    root, data = _previous(tmp_path)
    _git(root, "config", "core.filemode", "true")
    _start(data, root, "independent")
    (root / "dirty.txt").chmod(0o755)
    assert "preexisting_dirty_changed" in attribution.attributed_git_candidates(data, "independent", root)["blockers"]
    terminal = attribution.record_terminal_mutation_candidates(data, "independent")
    assert "preexisting_dirty_changed" in terminal["terminal_candidate_snapshot"]["surfaces"][0]["blockers"]


def test_unknown_mode_cannot_authorize_a_regular_file_transfer(tmp_path, monkeypatch):
    root, data = _previous(tmp_path)
    run_git = attribution._run_git

    def unreadable_index(path, *args):
        if args[0] == "ls-files":
            raise RuntimeError("index unavailable")
        return run_git(path, *args)

    monkeypatch.setattr(attribution, "_run_git", unreadable_index)
    terminal = attribution.record_terminal_mutation_candidates(data, "previous")
    assert terminal["terminal_candidate_snapshot"]["surfaces"][0]["candidate_fingerprints"]["clean.txt"]["git_mode"] is None
    monkeypatch.setattr(attribution, "_run_git", run_git)
    _start(data, root, "successor", _source())
    assert attribution.attributed_git_candidates(data, "successor", root)["candidates"] == []
