"""Review checking copies live only as long as their review (owner decision 2026-10-12).

A terminal seat's receipt is written first; its one-shot copy is then deleted, whatever
process census the engine reports. A thread keeps one copy while its task is live, and a
new turn captures an unobserved previous turn instead of refusing. ``prune_review_checkouts``
(``tests/test_review_checking_prune.py``) retires what a crashed or older host left behind.
The copies, receipts, locks and deletions here are real.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import threading
from pathlib import Path

import pytest

from ouroboros import review_checking as rc
from ouroboros import subagent_worktrees as sw

TASK = "task-lifecycle"
EDIT = b"# reviewer reproduction\n"
LOG = b"pytest output\n"
STATES = ("succeeded", "failed", "cancelled", "interrupted")
# The engine's compact process-scope projections; none of them changes what is released.
ABSENT = object()  # the key itself is missing, which is not the same fact as ``null``
SETTLED = {"settled": True, "coverage": "marker+capture", "completeness": "complete",
           "census": "ok", "survivors": 0, "unreadable": 0}
UNSETTLED = {**SETTLED, "settled": False, "survivors": 1}
NO_CENSUS = {"settled": False, "coverage": "tree_only", "completeness": "not_provable",
             "census": "unsupported", "survivors": 0, "unreadable": 0}
SCOPES = {"absent": ABSENT, "null": None, "settled": SETTLED, "unsettled": UNSETTLED, "no-census": NO_CENSUS}
LATER = 400 * 86400  # past any retention period


def _work(area: Path) -> None:
    (area / "check.py").write_bytes(EDIT)
    (area / rc.EVIDENCE_DIRNAME).mkdir(exist_ok=True)
    (area / rc.EVIDENCE_DIRNAME / "out.log").write_bytes(LOG)


def directory_source(folder: Path, data_root: Path) -> dict:
    """A retained folder view: what a writable skill seat copies (an ordinary folder is read in place)."""
    return {"kind": "directory", "root": str(folder), "view": str(rc._retained_view(folder, data_root))}


def _frozen(tmp_path: Path):
    root, subject = tmp_path / "data", tmp_path / "subject"
    subject.mkdir()
    (subject / "check.py").write_bytes(b"# candidate\n")
    return root, directory_source(subject, root)


def _seat(root: Path, source, operation: str, task: str = TASK) -> Path:
    area = Path(rc.prepare_seat_area(root, task, operation, source))
    _work(area)
    return area


def _boom(*_args, **_kwargs):
    raise OSError(5, "Input/output error")


@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("scope", list(SCOPES.values()), ids=list(SCOPES))
def test_every_terminal_releases_its_copy_after_the_receipt(tmp_path, state, scope):
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-1")
    summary = {"state": state, **({} if scope is ABSENT else {"processScope": scope})}
    receipt = rc.terminal_seat_evidence(root, TASK, "op-1", {"summary": summary})
    assert receipt["status"] == "retained" and not area.exists()
    evidence = Path(receipt["evidence_dir"])
    assert json.loads((evidence / "summary.json").read_text(encoding="utf-8")) == receipt
    assert (evidence / "files" / "check.py").read_bytes() == EDIT
    assert (evidence / "files" / rc.EVIDENCE_DIRNAME / "out.log").read_bytes() == LOG
    assert not {"area", "refresh_safe", "retention_reason", "release_basis"} & set(receipt)
    assert rc.terminal_seat_evidence(root, TASK, "op-1", {"summary": summary}) == receipt


@pytest.mark.parametrize("detail", [{"summary": {"state": "running"}}, {"summary": {"state": "queued"}},
                                    {"summary": None}, {}, None],
                         ids=["running", "queued", "no-summary", "empty", "none"])
def test_a_run_not_seen_terminal_keeps_its_copy(tmp_path, detail):
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-1")
    assert rc.terminal_seat_evidence(root, TASK, "op-1", detail) == {}
    assert (area / "check.py").read_bytes() == EDIT


def test_no_copy_goes_without_a_durable_receipt(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-1")
    real = rc.atomic_write_json

    def full_volume(path, *args, **kwargs):
        if Path(path).name == "summary.json":
            raise OSError(28, "No space left on device")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(rc, "atomic_write_json", full_volume)
    failed = rc.capture_seat_evidence(root, TASK, "op-1")
    assert failed["status"] == "capture_failed" and failed["area"] == str(area)
    assert (area / "check.py").read_bytes() == EDIT
    monkeypatch.setattr(rc, "atomic_write_json", real)
    retried = rc.capture_seat_evidence(root, TASK, "op-1")
    assert retried["status"] == "retained" and not area.exists()
    assert (Path(retried["evidence_dir"]) / "files" / "check.py").read_bytes() == EDIT


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_a_capture_with_named_gaps_still_releases(tmp_path):
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-1")
    (area / rc.EVIDENCE_DIRNAME / "link").symlink_to(area / "check.py")
    receipt = rc.capture_seat_evidence(root, TASK, "op-1")
    assert receipt["status"] == "partial" and not area.exists()
    assert {"path": f"{rc.EVIDENCE_DIRNAME}/link", "reason": "evidence_symlink"} in receipt["gaps"]


def _thread(root: Path, source, operation: str) -> Path:
    return Path(rc.thread_seat_area(root, TASK, "plan_review:critic", source, operation_id=operation))


def test_a_new_thread_turn_captures_an_unobserved_previous_turn(tmp_path):
    root, subject = tmp_path / "data", tmp_path / "subject"
    subject.mkdir()
    (subject / "check.py").write_bytes(b"# v1\n")
    area = _thread(root, directory_source(subject, root), "turn-1")
    _work(area)
    (subject / "check.py").write_bytes(b"# v2\n")
    assert _thread(root, directory_source(subject, root), "turn-2") == area
    assert (area / "check.py").read_bytes() == b"# v2\n" and not (area / rc.EVIDENCE_DIRNAME).exists()
    receipt = rc._receipt(root, TASK, "turn-1")
    assert receipt["status"] == "retained"
    assert (Path(receipt["evidence_dir"]) / "files" / "check.py").read_bytes() == EDIT


def test_a_failed_previous_capture_blocks_a_thread_only_without_any_receipt(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    area = _thread(root, source, "turn-1")
    _work(area)
    monkeypatch.setattr(rc, "_directory_delta", _boom)  # the failure is written as the turn's receipt
    assert _thread(root, source, "turn-2") == area
    assert rc._receipt(root, TASK, "turn-1")["status"] == "capture_failed"
    _work(area)
    real = rc.atomic_write_json
    monkeypatch.setattr(rc, "atomic_write_json", lambda path, *args, **kwargs: _boom() if Path(
        path).name == "summary.json" else real(path, *args, **kwargs))
    with pytest.raises(rc.CheckingAreaUnavailable, match="could not be captured"):
        _thread(root, source, "turn-3")
    assert (area / "check.py").read_bytes() == EDIT


def test_a_copy_made_again_for_a_receipted_operation_is_captured_again(tmp_path):
    """A receipt stands for one copy: a copy made again under the same operation is a new generation."""
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-1")
    first = rc.capture_seat_evidence(root, TASK, "op-1")
    assert first["status"] == "retained" and not area.exists()
    again = Path(rc.prepare_seat_area(root, TASK, "op-1", source))
    (again / "check.py").write_bytes(b"# second copy\n")
    second = rc.capture_seat_evidence(root, TASK, "op-1")
    assert second["status"] == "retained" and second["evidence_dir"] != first["evidence_dir"]
    assert (Path(second["evidence_dir"]) / "files" / "check.py").read_bytes() == b"# second copy\n"
    assert (Path(first["evidence_dir"]) / "files" / "check.py").read_bytes() == EDIT
    assert not again.exists()


def test_a_capture_attempt_without_a_receipt_leaves_no_generation(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-1")
    real = rc.atomic_write_json
    monkeypatch.setattr(rc, "atomic_write_json", lambda path, *args, **kwargs: _boom() if Path(
        path).name == "summary.json" else real(path, *args, **kwargs))
    for _attempt in range(3):
        assert rc.capture_seat_evidence(root, TASK, "op-1")["status"] == "capture_failed"
    assert area.is_dir() and not list(area.parent.glob("evidence*"))


def test_an_unreadable_marker_is_not_absence(tmp_path):
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-1")
    (area.parent / "checking.json").write_text("{torn", encoding="utf-8")
    receipt = rc.capture_seat_evidence(root, TASK, "op-1")
    assert receipt["status"] == "capture_failed" and "unreadable" in receipt["error"] and area.is_dir()
    thread = _thread(root, source, "turn-1")
    (thread.parent / "checking.json").write_text("{torn", encoding="utf-8")
    with pytest.raises(rc.CheckingAreaUnavailable, match="unreadable"):
        _thread(root, source, "turn-2")
    assert thread.is_dir()


def test_a_missing_operation_marker_is_recovered_from_the_thread_marker(tmp_path):
    """A refresh interrupted between the two markers still captures the turn before replacing it."""
    root, source = _frozen(tmp_path)
    area = _thread(root, source, "turn-1")
    _work(area)
    (rc.seat_dir(root, TASK, "turn-1") / "checking.json").unlink()
    assert _thread(root, source, "turn-2") == area and (area / "check.py").read_bytes() == b"# candidate\n"
    receipt = rc._receipt(root, TASK, "turn-1")
    assert receipt["status"] == "retained"
    assert (Path(receipt["evidence_dir"]) / "files" / "check.py").read_bytes() == EDIT


def test_a_late_capture_of_an_old_turn_never_touches_the_newer_turn(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    area = _thread(root, source, "turn-1")
    _work(area)
    with monkeypatch.context() as patch:
        patch.setattr(rc, "_directory_delta", _boom)
        assert _thread(root, source, "turn-2") == area  # turn 1's failed copy moved under its own seat
    (area / "turn-2.log").write_bytes(b"newer turn\n")
    late = rc.capture_seat_evidence(root, TASK, "turn-1")
    assert late["status"] == "retained"
    assert (area / "turn-2.log").read_bytes() == b"newer turn\n"
    assert not (rc.seat_dir(root, TASK, "turn-1") / "area.noindex").exists()
    assert (Path(late["evidence_dir"]) / "files" / "check.py").read_bytes() == EDIT


def test_captures_of_one_seat_wait_for_its_lock(tmp_path):
    """Workers, attach-only collection and the prune capture one copy one at a time."""
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-1")
    results = []
    with rc._seat_locked(area.parent):
        worker = threading.Thread(target=lambda: results.append(rc.capture_seat_evidence(root, TASK, "op-1")))
        worker.start()
        worker.join(timeout=0.5)
        assert worker.is_alive() and area.is_dir() and not results
    worker.join(timeout=30)
    assert results[0]["status"] == "retained" and not area.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
def test_deletion_never_changes_a_mode_through_a_link_and_removes_unlistable_folders(tmp_path, monkeypatch):
    outside = tmp_path / "owner.txt"
    outside.write_bytes(b"owner file\n")
    outside.chmod(0o644)
    tree = tmp_path / "copy"
    (tree / "locked").mkdir(parents=True)
    (tree / "locked" / "owner-link").symlink_to(outside)
    (tree / "unlistable").mkdir()
    (tree / "unlistable" / "only-copy.log").write_bytes(b"x")
    (tree / "locked").chmod(0o555)  # unlinking the link needs its folder writable
    (tree / "unlistable").chmod(0)
    monkeypatch.chdir(tmp_path)
    try:
        sw._force_rmtree(Path("copy"))  # a relative tree reports relative paths to the hook
    finally:
        for folder in (tree / "locked", tree / "unlistable"):
            if folder.exists():
                folder.chmod(0o755)
    assert not tree.exists()
    assert stat.S_IMODE(outside.stat().st_mode) == 0o644 and outside.read_bytes() == b"owner file\n"


def test_an_interrupted_rematerialization_is_never_adopted(tmp_path, monkeypatch):
    """The new generation is durable before the copy exists, so a half-made copy is remade."""
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-1")
    first = rc.capture_seat_evidence(root, TASK, "op-1")
    real_copytree = rc.shutil.copytree

    def dies_midway(src, dst, *args, **kwargs):
        Path(dst).mkdir(parents=True)
        (Path(dst) / "half.txt").write_bytes(b"half\n")
        raise OSError(5, "process died during the copy")

    with monkeypatch.context() as patch:
        patch.setattr(rc.shutil, "copytree", dies_midway)
        with pytest.raises(rc.CheckingAreaUnavailable):
            rc.prepare_seat_area(root, TASK, "op-1", source)
    assert json.loads((area.parent / "checking.json").read_text(encoding="utf-8"))["materializing"] is True
    again = Path(rc.prepare_seat_area(root, TASK, "op-1", source))
    assert not (again / "half.txt").exists() and rc.shutil.copytree is real_copytree
    (again / "second.log").write_bytes(b"second run\n")
    second = rc.capture_seat_evidence(root, TASK, "op-1")
    assert second["status"] == "retained" and second["evidence_dir"] != first["evidence_dir"]
    assert (Path(second["evidence_dir"]) / "files" / "second.log").read_bytes() == b"second run\n"


def test_a_copy_its_receipt_already_stands_for_is_not_reused(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-1")
    with monkeypatch.context() as patch:
        patch.setattr(sw, "_force_rmtree", lambda _path: None)
        first = rc.capture_seat_evidence(root, TASK, "op-1")
    assert first["cleanup_error"] and area.is_dir()  # the release left the copy behind
    again = Path(rc.prepare_seat_area(root, TASK, "op-1", source))
    assert (again / "check.py").read_bytes() == b"# candidate\n"  # remade, not the first run's leftovers
    (again / "second.log").write_bytes(b"second run\n")
    second = rc.capture_seat_evidence(root, TASK, "op-1")
    assert second["evidence_dir"] != first["evidence_dir"] and not again.exists()
    assert (Path(second["evidence_dir"]) / "files" / "second.log").read_bytes() == b"second run\n"


def test_a_copy_kept_for_its_failed_capture_is_reused_and_captured_anew(tmp_path, monkeypatch):
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-1")
    with monkeypatch.context() as patch:
        patch.setattr(rc, "_directory_delta", _boom)
        failed = rc.capture_seat_evidence(root, TASK, "op-1")
    assert failed["status"] == "capture_failed" and area.is_dir()
    assert Path(rc.prepare_seat_area(root, TASK, "op-1", source)) == area
    assert rc._receipt(root, TASK, "op-1") == {}  # the failed receipt no longer describes the copy
    (area / "second.log").write_bytes(b"second run\n")
    second = rc.capture_seat_evidence(root, TASK, "op-1")
    assert second["status"] == "retained" and second["evidence_dir"] != failed["evidence_dir"]
    files = Path(second["evidence_dir"]) / "files"
    assert (files / "check.py").read_bytes() == EDIT and (files / "second.log").read_bytes() == b"second run\n"
    assert not area.exists()


def test_a_stale_rejection_never_retires_a_concurrent_repair(tmp_path):
    """Freeze A's check of a damaged inventory view ends after freeze B repaired and published it."""
    root, folder = tmp_path / "data", tmp_path / "skill"
    folder.mkdir()
    (folder / "SKILL.md").write_bytes(b"skill\n")

    def good(view):
        return (Path(view) / "SKILL.md").read_bytes() == b"skill\n"

    def freeze(accept):
        return rc._retained_view(folder, root, payload=True, name="inventory-x", copy=shutil.copytree, accept=accept)

    view = freeze(good)
    checks = []

    def stale(checked):
        checks.append(checked)
        if len(checks) > 1:
            return good(checked)
        (view / "SKILL.md").write_bytes(b"damaged\n")
        assert freeze(good) == view  # B discards the damaged view and publishes a repair
        (view.parent.parent / "published-by-b").write_bytes(b"")
        return False  # A's verdict about the damaged bytes

    assert freeze(stale) == view and (view.parent.parent / "published-by-b").exists() and good(view)


def test_an_unreadable_marker_keeps_the_copy_at_preparation(tmp_path):
    root, source = _frozen(tmp_path)
    area = _seat(root, source, "op-1")
    (area.parent / "checking.json").write_text("{torn", encoding="utf-8")
    with pytest.raises(rc.CheckingAreaUnavailable, match="unreadable"):
        rc.prepare_seat_area(root, TASK, "op-1", source)
    assert (area / "check.py").read_bytes() == EDIT


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
def test_deletion_changes_no_mode_above_the_tree(tmp_path):
    above = tmp_path / "projects"
    tree = above / "copy"
    (tree / "sub").mkdir(parents=True)
    (tree / "sub" / "file.txt").write_bytes(b"x")
    above.chmod(0o555)  # the tree's own folder cannot be removed from here
    try:
        sw._force_rmtree(tree)
        assert stat.S_IMODE(above.stat().st_mode) == 0o555 and not (tree / "sub").exists()
    finally:
        above.chmod(0o755)


def test_a_workspace_thread_turn_starts_empty_and_keeps_the_previous_turns_evidence(tmp_path):
    root, subject = tmp_path / "data", tmp_path / "subject"
    subject.mkdir()
    (subject / "notes.txt").write_bytes(b"the live folder\n")
    source = rc.freeze_checking_source(subject, data_root=root)
    area = _thread(root, source, "turn-1")
    assert source == {"kind": "workspace", "root": str(subject)} and not any(area.iterdir())
    _work(area)
    assert _thread(root, source, "turn-2") == area and not any(area.iterdir())
    receipt = rc._receipt(root, TASK, "turn-1")
    kept = Path(receipt["evidence_dir"]) / "files"
    assert receipt["status"] == "retained" and (kept / rc.EVIDENCE_DIRNAME / "out.log").read_bytes() == LOG
    assert not (kept / "check.py").exists() and (subject / "notes.txt").read_bytes() == b"the live folder\n"
    assert not (root / "state" / "review_checkouts" / "views").exists()
