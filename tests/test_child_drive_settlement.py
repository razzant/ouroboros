"""TZ-1 A: ``task_custody.settle_child_drive`` is the one owner that deletes a task's own
execution drive. It deletes only after the canonical row durably holds the drive's
deliverables (recorded identity first), unread mail and receipts, re-read after the write;
anything missing, unreadable or mismatched keeps the drive, and a restore converges."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import artifacts, headless, owner_mailbox, task_custody
from ouroboros.outcome_receipt_store import read_verification_receipts, verification_receipts_path
from ouroboros.outcomes import append_verification_receipt
from ouroboros.task_results import load_task_result, write_task_result
from tests._store_shared import unreadable_result

TASK = "settle1"


def not_live(_task: str) -> bool:
    return False


def _drive(data: Path, kind: str, task: str = TASK) -> Path:
    if kind == "headless":
        return headless.prepare_task_drive(data, task, "empty")
    drive = data / "task_drives" / task
    drive.mkdir(parents=True)
    return drive


def _capture(tmp_path: Path, drive: Path, text: str = "captured report", task: str = TASK) -> dict:
    source = tmp_path / f"source-{len(list(tmp_path.iterdir()))}.txt"
    source.write_text(text, encoding="utf-8")
    return artifacts.copy_file_to_task_artifacts(SimpleNamespace(drive_root=drive, task_id=task), source,
                                                 immutable=True)


def _cancelled_with_capture(tmp_path: Path, kind: str = "headless"):
    """A queued-then-cancelled split task: its child recorded an immutable capture that
    copy-back never published (a settled cancellation blocks the child RESULT, never its files)."""
    data = tmp_path / "data"
    drive = _drive(data, kind)
    record = _capture(tmp_path, drive)
    write_task_result(drive, TASK, "running", artifacts=[record])
    write_task_result(data, TASK, "cancelled", result="Cancelled.", child_drive_root=str(drive))
    return data, drive, record


def _canonical(data: Path, name: str, task: str = TASK) -> Path:
    return artifacts.task_artifact_dir_path(data, task) / name


def _settle(data, drive, **kwargs):
    return task_custody.settle_child_drive(data, TASK, drive, live=kwargs.pop("live", not_live), **kwargs)


@pytest.mark.parametrize("kind", ["headless", "task_drives"])
def test_missing_immutable_source_retains_each_drive_and_a_restore_converges(tmp_path, kind):
    data, drive, record = _cancelled_with_capture(tmp_path, kind)
    Path(record["path"]).unlink()

    assert _settle(data, drive) == {"status": "retained", "reason": "artifact_source_missing", "published": 0}
    assert drive.is_dir() and not _canonical(data, record["name"]).exists()

    Path(record["path"]).write_text("changed bytes", encoding="utf-8")
    assert _settle(data, drive)["reason"] == "artifact_source_mismatch"  # never a downgrade to current bytes
    Path(record["path"]).write_text("captured report", encoding="utf-8")
    assert _settle(data, drive) == {"status": "removed", "reason": "", "published": 1}
    assert not drive.exists()
    assert _canonical(data, record["name"]).read_text(encoding="utf-8") == "captured report"
    row = next(item for item in load_task_result(data, TASK)["artifacts"] if item["name"] == record["name"])
    assert (row["sha256"], row["size"], row["immutable"]) == (record["sha256"], record["size"], True)
    assert row["path"] == str(_canonical(data, record["name"]).resolve())


def test_a_foreign_canonical_copy_is_never_replaced_and_the_capture_publishes_beside_it(tmp_path):
    data, drive, record = _cancelled_with_capture(tmp_path)
    target = _canonical(data, record["name"])
    target.parent.mkdir(parents=True)
    target.write_text("someone else's bytes", encoding="utf-8")

    assert _settle(data, drive) == {"status": "removed", "reason": "", "published": 1}
    assert target.read_text(encoding="utf-8") == "someone else's bytes"  # never replaced, never trusted
    row = next(item for item in load_task_result(data, TASK)["artifacts"] if item["sha256"] == record["sha256"])
    stem, suffix = record["name"].rsplit(".", 1)
    assert row["name"] == f"{stem}.{record['sha256'][:8]}.{suffix}" and row["immutable"] is True
    assert Path(row["path"]).read_text(encoding="utf-8") == "captured report"

    data, drive, record = _cancelled_with_capture(tmp_path / "again")
    target = _canonical(data, record["name"])
    target.parent.mkdir(parents=True)
    target.write_text("captured report", encoding="utf-8")  # the exact capture: held, nothing to copy
    assert _settle(data, drive) == {"status": "removed", "reason": "", "published": 0}
    assert record["name"] in {row["name"] for row in load_task_result(data, TASK)["artifacts"]}


def test_a_recorded_capture_whose_source_is_gone_stays_held_by_its_verified_canonical_copy(tmp_path):
    data, drive, record = _cancelled_with_capture(tmp_path)
    target = _canonical(data, record["name"])
    target.parent.mkdir(parents=True)
    target.write_text("captured report", encoding="utf-8")
    Path(record["path"]).unlink()
    # A stat-only listing of the canonical copy never outranks the recorded capture.
    from ouroboros.task_status import load_effective_task_result

    listed = next(row for row in load_effective_task_result(data, TASK)["artifacts"] if row["name"] == record["name"])
    assert (listed["sha256"], listed.get("immutable")) == (record["sha256"], True)
    assert _settle(data, drive)["status"] == "removed"
    row = next(item for item in load_task_result(data, TASK)["artifacts"] if item["name"] == record["name"])
    assert (row["sha256"], row["immutable"]) == (record["sha256"], True)


@pytest.mark.parametrize("material", ["child_result", "registration", "file", "directory", "mailbox", "receipts"])
def test_unreadable_material_retains_the_drive(tmp_path, monkeypatch, material):
    data, drive, record = _cancelled_with_capture(tmp_path)
    store = artifacts.task_artifact_dir_path(drive, TASK)
    if material == "child_result":
        unreadable_result(monkeypatch, TASK)
        expected = "task_result_unreadable"
    elif material == "registration":
        (store / ".artifact_manifest.json").write_text("{torn", encoding="utf-8")
        expected = "artifact_registration_unreadable"
    elif material == "file":
        extra = store / "notes.txt"
        extra.write_text("mutable note", encoding="utf-8")
        original_open = Path.open

        def unreadable_open(path, *args, **kwargs):
            if path == extra:
                raise PermissionError("file read denied")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", unreadable_open)
        expected = "child_artifact_store_unreadable"
    elif material == "directory":
        (store / "tree").mkdir()
        (store / "tree" / "leaf.txt").write_text("leaf", encoding="utf-8")
        original_scandir = os.scandir

        def unreadable_scandir(path):
            if Path(path) == store / "tree":
                raise PermissionError("directory read denied")
            return original_scandir(path)

        monkeypatch.setattr(os, "scandir", unreadable_scandir)
        expected = "child_artifact_store_unreadable"
    elif material == "mailbox":
        mailbox = owner_mailbox._mailbox_path(drive, TASK)
        mailbox.parent.mkdir(parents=True, exist_ok=True)
        mailbox.write_text('{"msg_id": "m1", "kind": "owner_text", "text": "hel', encoding="utf-8")
        expected = "unread_mailbox_unreadable"
    else:
        verification_receipts_path(drive, TASK, create=True).write_text("{torn\n", encoding="utf-8")
        expected = "verification_receipts_uncustodied"
    # chmod(0) does not deny Windows reads and is bypassed by POSIX root. Inject
    # the actual read failure so retention and recovery are tested on every host.
    outcome = _settle(data, drive)
    assert outcome["status"] == "retained" and outcome["reason"] == expected, outcome
    assert drive.is_dir() and not _canonical(data, record["name"]).exists()
    if material in {"file", "directory"}:
        if material == "file":
            monkeypatch.setattr(Path, "open", original_open)
        else:
            monkeypatch.setattr(os, "scandir", original_scandir)
        assert _settle(data, drive)["status"] == "removed"
        assert not drive.exists()
        extra_name = "notes.txt" if material == "file" else "tree/leaf.txt"
        assert _canonical(data, extra_name).read_text(encoding="utf-8") == (
            "mutable note" if material == "file" else "leaf"
        )


def test_a_write_that_does_not_land_or_read_back_deletes_and_seals_nothing(tmp_path, monkeypatch):
    data, drive, record = _cancelled_with_capture(tmp_path)
    owner_mailbox.write_owner_message(drive, "late words", TASK, msg_id="late1")
    monkeypatch.setattr(task_custody, "write_task_result", lambda *args, **kwargs: {})

    outcome = _settle(data, drive)

    assert outcome["status"] == "retained" and outcome["reason"] == "canonical_listing_missing"
    assert owner_mailbox._mailbox_path(drive, TASK).is_file() and drive.is_dir()
    assert "unread_mailbox" not in load_task_result(data, TASK)
    monkeypatch.undo()
    assert _settle(data, drive)["status"] == "removed"
    assert json.loads(load_task_result(data, TASK)["unread_mailbox"]["rows"][0])["text"] == "late words"


def test_a_crash_between_publication_and_the_row_write_converges_without_duplicates(tmp_path, monkeypatch):
    data, drive, record = _cancelled_with_capture(tmp_path)
    real = task_custody._write_custody_fields
    calls = []

    def crash_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("process died after the rename")
        return real(*args, **kwargs)

    monkeypatch.setattr(task_custody, "_write_custody_fields", crash_once)
    assert _settle(data, drive)["reason"].startswith("custody_error")
    assert _canonical(data, record["name"]).is_file() and drive.is_dir()  # unlisted, never served as a row
    assert _settle(data, drive)["status"] == "removed"
    rows = [row for row in load_task_result(data, TASK)["artifacts"] if row["name"] == record["name"]]
    assert len(rows) == 1 and rows[0]["sha256"] == record["sha256"]
    assert not (data / "task_results" / "artifact_versions").exists()


def test_a_settled_drive_leaves_only_unserved_trash_for_the_maintenance_child(tmp_path):
    data, drive, _record = _cancelled_with_capture(tmp_path)
    assert _settle(data, drive)["status"] == "removed"
    trash = data / "state" / "custody_trash"
    assert not drive.exists() and len(list(trash.iterdir())) == 1  # moved, not deleted, by the server
    task_custody.sweep_custody_leftovers(data, min_age_sec=0)
    assert len(list(trash.iterdir())) == 1, "the staging sweep never touches the moved drives"
    staging = data / "state" / "custody_staging" / "crashed-settlement"
    staging.mkdir(parents=True)
    task_custody.sweep_custody_leftovers(data, min_age_sec=3600)
    assert staging.exists()  # a young entry may belong to a running settlement
    task_custody.sweep_custody_leftovers(data, min_age_sec=0)
    assert not staging.exists()
    assert task_custody.remove_custody_trash(data) == {"removed": 1, "stuck": {}}
    assert list(trash.iterdir()) == []


def test_read_only_folders_in_custody_leftovers_never_strand_them(tmp_path):
    """A moved drive or a crashed settlement's staging carrying read-only folders and files (a Git
    object store, a tool's own modes) is removed whole by the permission-granting removal, so no
    trash entry waits forever for its identity to change; a staging entry younger than an hour (a
    settlement may still be running) is still left alone."""
    import stat

    data = tmp_path / "data"
    sealed = []

    def read_only_tree(entry: Path) -> Path:
        objects = entry / "data" / ".git" / "objects" / "ab"
        objects.mkdir(parents=True)
        (objects / "cdef").write_bytes(b"object")
        os.chmod(objects / "cdef", stat.S_IREAD)
        os.chmod(objects, stat.S_IREAD | stat.S_IEXEC)
        sealed.append(objects)
        return entry

    moved = read_only_tree(data / "state" / "custody_trash" / f"{TASK}-0123456789ab")
    staging = read_only_tree(data / "state" / "custody_staging" / f"{TASK}-fedcba987654")
    try:
        task_custody.sweep_custody_leftovers(data, min_age_sec=3600)
        assert staging.exists(), "a young staging entry may belong to a running settlement"
        task_custody.sweep_custody_leftovers(data, min_age_sec=0)
        assert not staging.exists()
        assert task_custody.remove_custody_trash(data) == {"removed": 1, "stuck": {}}
        assert not moved.exists()
    finally:
        for objects in sealed:
            if objects.exists():
                os.chmod(objects, stat.S_IRWXU)


@pytest.mark.parametrize("probe", [None, lambda _task: True, lambda _task: None])
def test_unknown_or_live_ownership_deletes_nothing(tmp_path, probe):
    data, drive, record = _cancelled_with_capture(tmp_path)
    outcome = task_custody.settle_child_drive(data, TASK, drive, live=probe)
    assert outcome["status"] == "retained" and outcome["reason"] in {"liveness_unknown", "task_live"}
    assert drive.is_dir() and not _canonical(data, record["name"]).exists()


def test_reassignment_after_preparation_serves_no_prepared_bytes_or_rows(tmp_path, monkeypatch):
    data, drive, record = _cancelled_with_capture(tmp_path)
    looks = []

    def becomes_live(_task):
        looks.append(1)
        return len(looks) > 1  # absent while staging, owned again before the locked phase

    assert _settle(data, drive, live=becomes_live)["reason"] == "task_live"
    assert not artifacts.task_artifact_dir_path(data, TASK).exists()
    assert list((data / "state" / "custody_staging").iterdir()) == []
    assert "artifacts" not in load_task_result(data, TASK)

    real_plan = task_custody._child_store_plan

    def plan_then_new_attempt(*args, **kwargs):
        plan = real_plan(*args, **kwargs)
        # Reassignment changes custody's drive witness; ordinary terminal
        # enrichment cannot invent a new attempt by replacing started_at.
        write_task_result(data, TASK, "cancelled", child_drive_root=str(tmp_path / "reassigned"))
        return plan

    monkeypatch.setattr(task_custody, "_child_store_plan", plan_then_new_attempt)
    assert _settle(data, drive)["reason"] == "canonical_changed"
    assert not artifacts.task_artifact_dir_path(data, TASK).exists() and drive.is_dir()


def test_an_unadopted_terminal_child_result_waits_for_copy_back(tmp_path):
    data = tmp_path / "data"
    drive = _drive(data, "headless")
    record = _capture(tmp_path, drive)
    write_task_result(drive, TASK, "completed", result="child answer", artifacts=[record], artifact_status="ready")
    write_task_result(data, TASK, "failed", result="orphan reconciled", child_drive_root=str(drive))

    assert _settle(data, drive)["reason"] == "child_result_unadopted"
    report = headless.prepare_terminal_task_files(data, {"id": TASK, "drive_root": str(drive)})
    assert not report["error"]
    settled = load_task_result(data, TASK)
    # A settled canonical row keeps its own outcome; the child's is recorded beside it.
    assert (settled["status"], settled["result"], settled["child_status"]) == ("failed", "orphan reconciled", "completed")
    assert _settle(data, drive)["reason"] == "child_refs_pending"
    headless.retry_child_task_refs(data, drive, TASK)
    assert _settle(data, drive)["status"] == "removed"


def test_a_timeout_retry_occupying_the_original_drive_keeps_it_until_its_own_custody(tmp_path):
    data, drive, _record = _cancelled_with_capture(tmp_path)
    retry = "settle1r"
    write_task_result(drive, retry, "completed", result="retry answer")
    write_task_result(data, retry, "running", child_drive_root=str(drive))

    # The retry's canonical row is still running: only its DURABLE settlement counts, and then
    # its finished child result, never copied back, owes custody first.
    assert _settle(data, drive)["reason"] == "canonical_not_settled"
    write_task_result(data, retry, "completed", result="retry answer")
    assert _settle(data, drive)["reason"] == "child_result_unadopted"
    headless.copy_child_task_result(data, {"id": retry, "drive_root": str(drive)})
    assert _settle(data, drive)["reason"] == "child_refs_pending"
    headless.retry_child_task_refs(data, drive, retry)
    assert _settle(data, drive)["status"] == "removed"


def test_a_drive_that_is_not_the_tasks_own_is_never_touched(tmp_path):
    data, drive, _record = _cancelled_with_capture(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    assert _settle(data, elsewhere)["reason"] == "not_own_drive" and elsewhere.is_dir()
    linked_base = data / "state" / "headless_tasks" / "linked1"
    linked_base.mkdir(parents=True)
    try:
        (linked_base / "data").symlink_to(drive, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    write_task_result(data, "linked1", "cancelled", result="cancelled")
    outcome = task_custody.settle_child_drive(data, "linked1", linked_base / "data", live=not_live)
    assert outcome["reason"] == "not_own_drive" and drive.is_dir()


def test_child_verification_receipts_are_unioned_before_the_drive_goes(tmp_path):
    data, drive, _record = _cancelled_with_capture(tmp_path)
    assert append_verification_receipt(drive, TASK, {"status": "pass", "criterion_id": "child-check"})
    assert _settle(data, drive)["status"] == "removed"
    assert [row.get("criterion_id") for row in read_verification_receipts(data, TASK)] == ["child-check"]


def test_unrecorded_files_are_carried_at_their_relpaths_without_invented_rows_and_a_collision_keeps_both(tmp_path):
    from ouroboros.task_status import load_effective_task_result

    data, drive, record = _cancelled_with_capture(tmp_path)
    store = artifacts.task_artifact_dir_path(drive, TASK)
    (store / "reports" / "a").mkdir(parents=True)
    (store / "reports" / "a" / "summary.txt").write_text("alpha", encoding="utf-8")
    (store / "notes.txt").write_text("child notes", encoding="utf-8")
    canonical_notes = _canonical(data, "notes.txt")
    canonical_notes.parent.mkdir(parents=True)
    canonical_notes.write_text("canonical notes", encoding="utf-8")
    write_task_result(data, TASK, "cancelled", artifacts=[artifacts.artifact_record(canonical_notes)])

    assert _settle(data, drive) == {"status": "removed", "reason": "", "published": 3}

    # Only recorded obligations become rows; a file nobody recorded keeps its bytes, not a row.
    recorded = {row.get("relpath") or row["name"] for row in load_task_result(data, TASK)["artifacts"]}
    assert recorded == {"notes.txt", record["name"]}
    assert _canonical(data, "reports/a/summary.txt").read_text(encoding="utf-8") == "alpha"
    assert canonical_notes.read_text(encoding="utf-8") == "canonical notes"  # the listed canonical file wins
    listed = {row.get("relpath") or row["name"]: row for row in load_effective_task_result(data, TASK)["artifacts"]}
    assert listed["reports/a/summary.txt"]["measured"] is False  # listed from the canonical store now
    assert listed["reports/a/summary.txt"]["path"] == str(_canonical(data, "reports/a/summary.txt").resolve())
    kept = [row for name, row in listed.items() if name.startswith("notes.") and name != "notes.txt"]
    assert len(kept) == 1 and Path(kept[0]["path"]).read_text(encoding="utf-8") == "child notes"


def test_a_file_named_like_the_index_of_a_link_to_a_database_is_stored(tmp_path):
    """The child's folder holds actual.sqlite, a link report -> actual.sqlite and the user's own
    report-shm. SQLite's index of actual.sqlite would be actual.sqlite-shm, beside the real file, so
    report-shm is an ordinary artifact the settlement stores before the drive goes (taken for an index
    through the link, it was left out and removed with the drive, never stored)."""
    data, drive, _record = _cancelled_with_capture(tmp_path)
    store = artifacts.task_artifact_dir_path(drive, TASK)
    conn = sqlite3.connect(store / "actual.sqlite")
    conn.execute("CREATE TABLE t(v)")
    conn.close()
    try:
        (store / "report").symlink_to("actual.sqlite")
    except OSError:
        pytest.skip("symbolic links need a privilege here")
    (store / "report-shm").write_text("the user's report\n", encoding="utf-8")

    assert _settle(data, drive)["status"] == "removed"
    assert _canonical(data, "report-shm").read_text(encoding="utf-8") == "the user's report\n"
    assert _canonical(data, "actual.sqlite").is_file()


def test_admission_rollback_removes_only_a_never_started_drive(tmp_path):
    data = tmp_path / "data"
    drive = _drive(data, "headless")
    assert not headless.remove_subagent_task_drive(data, TASK, live=not_live)  # no settled row: custody owed
    assert headless.remove_subagent_task_drive(data, TASK, live=not_live, admission_rollback=True)
    assert not drive.exists()

    drive = _drive(data, "headless")
    owner_mailbox.write_task_message(drive, "early words", TASK, source_task_id="parent1")
    assert not headless.remove_subagent_task_drive(data, TASK, live=not_live, admission_rollback=True)
    assert drive.is_dir()


def test_late_mail_before_terminal_between_terminal_and_gc_and_after_gc(tmp_path):
    data = tmp_path / "data"
    drive = _drive(data, "headless")
    write_task_result(data, TASK, "scheduled", child_drive_root=str(drive))
    owner_mailbox.write_task_message(drive, "before terminal", TASK, source_task_id="parent1", msg_id="m1")
    write_task_result(data, TASK, "cancelled", result="Cancelled before start.")
    held = load_task_result(data, TASK)["unread_mailbox"]
    assert [json.loads(row)["msg_id"] for row in held["rows"]] == ["m1"] and held["read_complete"] is True
    assert not owner_mailbox._ack_path(drive, TASK).exists()  # the capture acknowledges nothing

    owner_mailbox.write_task_message(drive, "after terminal", TASK, source_task_id="parent1", msg_id="m2")
    before = load_task_result(data, TASK)["updated_at"]
    assert _settle(data, drive)["status"] == "removed"
    assert [json.loads(row)["msg_id"] for row in load_task_result(data, TASK)["unread_mailbox"]["rows"]] == ["m1", "m2"]
    assert load_task_result(data, TASK)["updated_at"] != before

    # A sender after the drive went recreates only its mailbox; the next settlement holds it.
    assert owner_mailbox.write_task_message(drive, "after gc", TASK, source_task_id="parent1", msg_id="m3")
    assert drive.is_dir()
    assert _settle(data, drive)["status"] == "removed"
    rows = load_task_result(data, TASK)["unread_mailbox"]["rows"]
    assert [json.loads(row)["msg_id"] for row in rows] == ["m1", "m2", "m3"]
    stamp = load_task_result(data, TASK)["updated_at"]
    assert task_custody.settle_task_mailbox(data, TASK, data)  # nothing owed: no rewrite
    assert load_task_result(data, TASK)["updated_at"] == stamp


def test_a_sender_racing_the_settlement_is_held_or_waits_never_lost(tmp_path, monkeypatch):
    """A row appended while the publication phase runs (the mail lock is free then) is seen by
    the final check and keeps the drive; one appended while the final check holds the mail
    lock waits and lands in a recreated mailbox; each is held by the next settlement."""
    data, drive, _record = _cancelled_with_capture(tmp_path)
    real_publish, real_recheck = task_custody._publish, task_custody._recheck_occupant

    def send(text, msg_id):
        landed = threading.Event()
        threading.Thread(target=lambda: owner_mailbox.write_task_message(
            drive, text, TASK, source_task_id="parent1", msg_id=msg_id) and landed.set(), daemon=True).start()
        return landed

    def publish_while_a_sender_lands(*args, **kwargs):
        # After the publication phase read the mailbox and before its row write: no lock stops the sender.
        assert send("during publication", "a").wait(5), "the publication phase never blocks a sender"
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(task_custody, "_publish", publish_while_a_sender_lands)
    assert _settle(data, drive)["reason"] == "unread_mailbox_uncustodied"
    assert drive.is_dir()
    monkeypatch.undo()
    waiting = []

    def recheck_while_a_sender_waits(*args, **kwargs):
        landed = send("during the move", "b")
        assert not landed.wait(0.3), "an append landed inside the mail lock"
        waiting.append(landed)
        return real_recheck(*args, **kwargs)

    monkeypatch.setattr(task_custody, "_recheck_occupant", recheck_while_a_sender_waits)
    assert _settle(data, drive)["status"] == "removed"
    assert waiting and waiting[0].wait(5)
    rows, complete = task_custody.unread_mail_rows(drive, TASK)
    assert complete and [json.loads(row)["msg_id"] for row in rows] == ["b"]  # recreated: only the late row
    monkeypatch.undo()
    assert _settle(data, drive)["status"] == "removed"
    assert [json.loads(row)["msg_id"] for row in load_task_result(data, TASK)["unread_mailbox"]["rows"]] == ["a", "b"]


def _nested_child(tmp_path):
    data = tmp_path / "data"
    drive = _drive(data, "headless")
    store = artifacts.task_artifact_dir_path(drive, TASK, create=True)
    for sub, text in (("a", "alpha"), ("b", "beta")):
        (store / "reports" / sub).mkdir(parents=True)
        (store / "reports" / sub / "summary.txt").write_text(text, encoding="utf-8")
    write_task_result(drive, TASK, "completed", result="done", artifacts=artifacts.collect_task_artifact_records(drive, TASK))
    write_task_result(data, TASK, "running", child_drive_root=str(drive))
    return data, drive, store


def test_copy_back_keeps_nested_relpaths_and_versions_a_changed_mutable_copy(tmp_path):
    data, drive, store = _nested_child(tmp_path)
    task = {"id": TASK, "drive_root": str(drive)}

    copied = headless.copy_child_task_result(data, task)

    canonical = artifacts.task_artifact_dir_path(data, TASK)
    assert (canonical / "reports/a/summary.txt").read_text(encoding="utf-8") == "alpha"
    assert (canonical / "reports/b/summary.txt").read_text(encoding="utf-8") == "beta"
    assert sorted(row["relpath"] for row in copied["artifacts"]) == ["reports/a/summary.txt", "reports/b/summary.txt"]
    (store / "reports/a/summary.txt").write_text("alpha v2", encoding="utf-8")
    again = headless.copy_child_task_result(data, task)
    assert (canonical / "reports/a/summary.txt").read_text(encoding="utf-8") == "alpha v2"
    row = next(item for item in again["artifacts"] if item["relpath"] == "reports/a/summary.txt")
    assert row["sha256"] == artifacts.stream_artifact_file(canonical / "reports/a/summary.txt")["sha256"]
    versions = data / "task_results" / "artifact_versions" / TASK / "summary.txt"
    assert [path.read_text(encoding="utf-8") for path in versions.iterdir()] == ["alpha"]


def test_relpath_is_result_identity_but_the_physical_store_is_not(tmp_path):
    from ouroboros.tools.join_ledger import _child_result_sha256

    data, drive, store = _nested_child(tmp_path)
    row = next(item for item in artifacts.collect_task_artifact_records(drive, TASK)
               if item.get("relpath") == "reports/a/summary.txt")
    relocated = {**row, "path": str(artifacts.task_artifact_dir_path(data, TASK) / row["relpath"])}
    renamed = {**relocated, "relpath": "reports/b/summary.txt"}
    changed = {**relocated, "sha256": "0" * 64}

    def result(artifact):
        return {"status": "completed", "result": "done", "artifacts": [artifact]}

    assert _child_result_sha256(result(row)) == _child_result_sha256(result(relocated))  # publication moves nothing
    assert _child_result_sha256(result(row)) != _child_result_sha256(result(renamed))
    assert _child_result_sha256(result(row)) != _child_result_sha256(result(changed))  # content is identity
    (store / "flat.txt").write_text("flat", encoding="utf-8")
    flat = next(item for item in artifacts.collect_task_artifact_records(drive, TASK) if item["name"] == "flat.txt")
    assert "relpath" not in flat  # a top-level legacy row keeps its historical identity


# ----------------------------------------------------------------------------- structural probes
# One probe per review finding: the class of failure, not the local example.


@pytest.mark.parametrize("kind", ["headless", "task_drives"])
@pytest.mark.parametrize("recorded_by", ["child_result", "canonical_row", "registration"])
def test_obligations_are_derived_before_the_store_is_looked_at(tmp_path, kind, recorded_by):
    """F1: a WHOLE child store that is gone is a missing source for every recorded row, not
    permission: no row is dropped, the drive stays, a restore converges (both layouts)."""
    import shutil

    data, drive, record = _cancelled_with_capture(tmp_path, kind)
    store = artifacts.task_artifact_dir_path(drive, TASK)
    if recorded_by == "canonical_row":
        write_task_result(drive, TASK, "running", artifacts=[])
        write_task_result(data, TASK, "cancelled", artifacts=[record])  # a legacy row naming the child path
    elif recorded_by == "registration":
        write_task_result(drive, TASK, "running", artifacts=[])  # only the store's own registration remains
        registration = store / ".artifact_manifest.json"
        assert json.loads(registration.read_text(encoding="utf-8"))["artifacts"][record["name"]]["immutable"] is True
    canonical_before = load_task_result(data, TASK).get("artifacts")
    if recorded_by == "registration":
        Path(record["path"]).unlink()  # the file is gone, the registration stays
    else:
        shutil.rmtree(store)

    outcome = _settle(data, drive)

    assert outcome == {"status": "retained", "reason": "artifact_source_missing", "published": 0}, outcome
    assert drive.is_dir() and not _canonical(data, record["name"]).exists()
    assert load_task_result(data, TASK).get("artifacts") == canonical_before  # nothing dropped
    store.mkdir(parents=True, exist_ok=True)
    Path(record["path"]).write_text("captured report", encoding="utf-8")
    if recorded_by == "registration":
        pass  # the registration still names the capture
    assert _settle(data, drive) == {"status": "removed", "reason": "", "published": 1}
    row = next(item for item in load_task_result(data, TASK)["artifacts"] if item["name"] == record["name"])
    assert (row["sha256"], row["immutable"], row["path"]) == (record["sha256"], True, str(_canonical(data, record["name"]).resolve()))


def test_a_recorded_mutable_file_whose_source_is_gone_retains_unless_a_canonical_row_holds_it(tmp_path):
    """F1: a mutable obligation without its source is not permission either; only a canonical
    row recording the same relpath at a verified canonical copy supersedes it."""
    data, drive, _record = _cancelled_with_capture(tmp_path)
    store = artifacts.task_artifact_dir_path(drive, TASK)
    notes = store / "notes.txt"
    notes.write_text("child notes", encoding="utf-8")
    mutable = artifacts.artifact_record(notes)
    write_task_result(drive, TASK, "running", artifacts=[mutable])
    notes.unlink()

    assert _settle(data, drive)["reason"] == "artifact_source_missing" and drive.is_dir()

    canonical_notes = _canonical(data, "notes.txt")
    canonical_notes.parent.mkdir(parents=True)
    canonical_notes.write_text("child notes", encoding="utf-8")
    write_task_result(data, TASK, "cancelled", artifacts=[artifacts.artifact_record(canonical_notes)])
    assert _settle(data, drive)["status"] == "removed"
    canonical_notes.write_text("someone changed it", encoding="utf-8")  # an unverified copy proves nothing


def test_a_raw_nonterminal_row_is_never_settled_on_an_effective_projection(tmp_path):
    """F2: the durable row alone settles; a status-only projection (an orphan the reconciler
    has not persisted) is not custody, however terminal it reads."""
    from ouroboros.task_status import load_effective_task_result

    data = tmp_path / "data"
    drive = _drive(data, "headless")
    write_task_result(drive, TASK, "running")
    write_task_result(data, TASK, "running", child_drive_root=str(drive), result_status="infra_failed",
                      reason_code="provider_failure", result="provider failed")
    (data / "state" / "queue_snapshot.json").write_text('{"pending": [], "running": []}', encoding="utf-8")
    assert load_effective_task_result(data, TASK)["status"] == "failed"

    assert _settle(data, drive)["reason"] == "canonical_not_settled" and drive.is_dir()


@pytest.mark.parametrize("event", ["new_occupant", "live_again", "interlock_closed", "generation_closed"])
def test_the_move_happens_only_under_the_interlock_with_a_fresh_census_and_probe(tmp_path, event):
    """F2: the drive moves inside the caller's ownership interlock (the supervisor's queue
    lock): an occupant that landed after the last probe, ownership that came back, a closed
    interlock or a closed generation each keep the drive; the next open pass converges."""
    import contextlib

    data, drive, record = _cancelled_with_capture(tmp_path)
    looks = []

    @contextlib.contextmanager
    def guard():
        if event == "new_occupant":  # what a timeout retry's mailbox copy does under the queue lock
            owner_mailbox.write_task_message(drive, "retry words", "settle1r", source_task_id="parent1")
        yield event != "interlock_closed"

    def probe(_task):
        looks.append(1)
        return event == "live_again" and len(looks) > 2  # absent while preparing and publishing, owned at the move

    outcome = _settle(data, drive, live=probe, guard=guard,
                      stop=(lambda: True) if event == "generation_closed" else None)
    assert outcome["reason"] == {"new_occupant": "drive_occupant_changed", "live_again": "task_live",
                                 "interlock_closed": "generation_closed", "generation_closed": "generation_closed"}[event]
    assert drive.is_dir() and Path(record["path"]).read_text(encoding="utf-8") == "captured report"
    if event == "generation_closed":
        assert not artifacts.task_artifact_dir_path(data, TASK).exists(), "closed before any publication"
    else:
        assert _canonical(data, record["name"]).is_file()  # the publication phase already secured the bytes
        assert record["sha256"] in {row.get("sha256") for row in load_task_result(data, TASK)["artifacts"]}
    if event == "new_occupant":
        write_task_result(data, "settle1r", "cancelled", result="never ran")
    assert _settle(data, drive)["status"] == "removed"


@pytest.mark.parametrize("kind", ["headless", "task_drives"])
def test_the_move_releases_the_store_handles_the_pass_itself_opened_inside_the_drive(tmp_path, monkeypatch, kind):
    """Windows moves no folder while a file below it is open, and the pass itself reads the
    drive's store through the per-thread connection cache: the move releases those handles
    first. A lease still active inside the drive keeps it, as a transient retention the
    next pass attempts again."""
    from ouroboros import store

    data, drive, _record = _cancelled_with_capture(tmp_path, kind)
    real_replace = os.replace

    def windows_replace(source, target):  # the Windows rule, applied to the store cache
        prefix = os.path.join(os.path.realpath(source), "")
        held = [name for pool in list(store._CONNECTION_POOLS) for name in pool.entries
                if os.path.realpath(name).startswith(prefix)]
        if held:
            raise PermissionError(5, "Access is denied", str(source))
        return real_replace(source, target)

    monkeypatch.setattr(task_custody.os, "replace", windows_replace)
    leased, release = threading.Event(), threading.Event()

    def hold():
        with store.open_store(drive, write=False):
            leased.set()
            release.wait()  # the finally below releases it; join bounds the test

    worker = threading.Thread(target=hold)
    worker.start()
    try:
        assert leased.wait(5)
        outcome = _settle(data, drive)
        assert outcome["status"] == "retained" and outcome["reason"] == "store_busy" and drive.is_dir()
        rested = {}
        task_custody.record_rest(rested, kind, data, TASK, tmp_path, drive, outcome)
        assert rested == {}, "a busy store is attempted again on the next pass"
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive()
    assert _settle(data, drive)["status"] == "removed" and not drive.exists()



def test_the_worker_that_ran_the_task_releases_its_own_handles_on_the_drive(tmp_path):
    """The settlement can close only the server's cached handles: the pool worker that ran the
    task read the drive's store in its own process and releases those once the task ends. A
    late thread still leasing one keeps it, and that thread's end closes it; every idle
    handle goes at once, so the drive never waits on a cache nothing will use again."""
    from ouroboros import store
    from supervisor.worker_process import _release_drive_stores

    data, drive, _record = _cancelled_with_capture(tmp_path)
    with store.open_store(drive, write=False) as txn:
        idle = txn.conn
    leased, release, late = threading.Event(), threading.Event(), []

    def late_thread():
        with store.open_store(drive, write=False) as txn:
            late.append(txn.conn)
            leased.set()
            release.wait()  # the finally below releases it; join bounds the test

    worker = threading.Thread(target=late_thread)
    worker.start()
    try:
        assert leased.wait(5)
        _release_drive_stores(drive)
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            idle.execute("SELECT 1")
        assert late[0].execute("SELECT 1").fetchone()[0] == 1, "a late lease is never closed under it"
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        late[0].execute("SELECT 1")  # the late thread's end disposed of its own handle

def test_a_copy_back_between_preparation_and_publication_is_seen_by_the_revision(tmp_path, monkeypatch):
    """F3: a publisher that changed the row's custody fields after the plan was prepared
    (same status, same attempt) aborts the publication; the next pass converges."""
    data, drive, store = _nested_child(tmp_path)
    task = {"id": TASK, "drive_root": str(drive)}
    headless.copy_child_task_result(data, task)
    assert load_task_result(data, TASK)["status"] == "completed"
    headless.retry_child_task_refs(data, drive, TASK)
    real = task_custody._child_store_plan

    def plan_then_republish(*args, **kwargs):
        plan = real(*args, **kwargs)
        (store / "reports/a/summary.txt").write_text("alpha v2", encoding="utf-8")
        headless.copy_child_task_result(data, task)  # a same-status enrichment of the artifact rows
        return plan

    monkeypatch.setattr(task_custody, "_child_store_plan", plan_then_republish)
    assert _settle(data, drive)["reason"] == "canonical_changed" and drive.is_dir()
    monkeypatch.undo()
    headless.retry_child_task_refs(data, drive, TASK)
    assert _settle(data, drive)["status"] == "removed"
    row = next(item for item in load_task_result(data, TASK)["artifacts"] if item.get("relpath") == "reports/a/summary.txt")
    assert Path(row["path"]).read_text(encoding="utf-8") == "alpha v2"


def test_a_destination_created_after_preparation_is_never_overwritten(tmp_path, monkeypatch):
    """F3: destination identity is checked before every effect; different bytes that appeared
    meanwhile keep their name and the capture publishes beside them on the next pass."""
    data, drive, record = _cancelled_with_capture(tmp_path)
    target = _canonical(data, record["name"])
    real = task_custody._child_store_plan

    def plan_then_intruder(*args, **kwargs):
        plan = real(*args, **kwargs)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("intruder", encoding="utf-8")
        return plan

    monkeypatch.setattr(task_custody, "_child_store_plan", plan_then_intruder)
    assert _settle(data, drive)["reason"] == "destination_changed"
    assert target.read_text(encoding="utf-8") == "intruder" and drive.is_dir()
    assert list((data / "state" / "custody_staging").iterdir()) == []
    monkeypatch.undo()
    assert _settle(data, drive)["status"] == "removed"
    assert target.read_text(encoding="utf-8") == "intruder"
    beside = next(item for item in load_task_result(data, TASK)["artifacts"] if item["sha256"] == record["sha256"])
    assert Path(beside["path"]).read_text(encoding="utf-8") == "captured report" and beside["name"] != record["name"]


def test_a_row_change_after_the_publication_readback_keeps_the_drive(tmp_path, monkeypatch):
    """F3: the final check under the interlock re-reads the row against the revision the
    publication phase left; a concurrent same-status enrichment after that readback
    retains, and the next pass merges without dropping either row."""
    data, drive, record = _cancelled_with_capture(tmp_path)
    late = data / "late.txt"
    late.write_text("late", encoding="utf-8")
    real = task_custody._recheck_occupant

    def enrich_then_recheck(*args, **kwargs):
        current = load_task_result(data, TASK)
        write_task_result(data, TASK, "cancelled", artifacts=[*current["artifacts"], artifacts.artifact_record(late)])
        return real(*args, **kwargs)

    monkeypatch.setattr(task_custody, "_recheck_occupant", enrich_then_recheck)
    assert _settle(data, drive)["reason"] == "canonical_changed" and drive.is_dir()
    monkeypatch.undo()
    assert _settle(data, drive)["status"] == "removed"
    names = {row["name"] for row in load_task_result(data, TASK)["artifacts"]}
    assert {record["name"], "late.txt"} <= names


def test_copy_back_and_settlement_share_one_publication_lock(tmp_path, monkeypatch):
    """F3: copy-back publishes under the same custody lock a settlement holds, answers
    ``CustodyBusy`` while it is held (no artifact failure is stamped), and publishes
    nothing for a drive a settlement already carried."""
    monkeypatch.setattr(headless, "PUBLICATION_LOCK_SEC", 0.2)
    data, drive, store = _nested_child(tmp_path)
    task = {"id": TASK, "drive_root": str(drive)}
    with task_custody.task_custody_lock(data, TASK) as locked:
        assert locked
        with pytest.raises(task_custody.CustodyBusy):
            headless.copy_child_task_result(data, task)
        report = headless.prepare_terminal_task_files(data, task)
    assert report["error"].startswith("CustodyBusy") and report["terminal_source_present"] is True
    assert load_task_result(data, TASK).get("artifact_status") != "failed"
    assert not artifacts.task_artifact_dir_path(data, TASK).exists()

    headless.copy_child_task_result(data, task)
    assert _settle(data, drive)["reason"] == "child_refs_pending"
    headless.retry_child_task_refs(data, drive, TASK)
    assert _settle(data, drive)["status"] == "removed"
    settled = load_task_result(data, TASK)
    assert headless.copy_child_task_result(data, task) is None  # the drive is gone: nothing to publish
    assert load_task_result(data, TASK) == settled
    assert headless.retry_child_task_refs(data, drive, TASK) == settled


def _attached(data, drive, count, *, msg_id):
    """Owner mail with COUNT staged attachments in the recipient's own store."""
    sources = []
    for index in range(count):
        source = data.parent / f"input-{msg_id}-{index}.txt"
        source.write_text(f"input {index} of {msg_id}", encoding="utf-8")
        sources.append(str(source))
    manifest = artifacts.stage_task_attachments(drive, TASK, sources)
    assert all(row["status"] == "staged" for row in manifest), manifest
    assert owner_mailbox.write_owner_message(drive, "see the files", TASK, msg_id=msg_id, attachment_manifest=manifest)
    return manifest


@pytest.mark.parametrize("count", [2, 30])
def test_unread_mail_inputs_are_carried_with_a_canonical_resolvable_closure(tmp_path, count):
    """F4: the exact row text is kept AND its attachments (inline, or the >25 manifest source)
    reach the canonical store with their captured identity, projected per row."""
    data = tmp_path / "data"
    drive = _drive(data, "headless")
    write_task_result(data, TASK, "scheduled", child_drive_root=str(drive))
    _attached(data, drive, count, msg_id="m1")
    write_task_result(data, TASK, "cancelled", result="Cancelled before start.")
    row = load_task_result(data, TASK)["unread_mailbox"]["rows"][0]
    entry = json.loads(row)
    assert (count > 25) == ("attachment_manifest_ref" in entry)

    assert _settle(data, drive)["status"] == "removed"

    custody = load_task_result(data, TASK)["unread_mailbox"]
    assert custody["rows"] == [row]  # the exact line, byte for byte
    projection = custody["inputs"][task_custody._row_key(row)]
    resolved = artifacts.resolve_attachment_manifest(data, TASK, projection)
    assert len(resolved) == count
    canonical_store = artifacts.task_artifact_dir_path(data, TASK).resolve()
    for item in resolved:
        assert Path(item["abs_path"]).is_relative_to(canonical_store)
        artifacts.stream_artifact_file(Path(item["abs_path"]), expected=item)
    assert not drive.exists()


def test_inputs_that_cannot_be_carried_keep_the_drive(tmp_path, monkeypatch):
    """F4: a copy, write or read-back failure of an input closure retains; late mail
    attachments after a complete copy-back are carried by the settlement, not by the
    promotion mark."""
    data, drive, store = _nested_child(tmp_path)
    headless.copy_child_task_result(data, {"id": TASK, "drive_root": str(drive)})
    headless.retry_child_task_refs(data, drive, TASK)
    assert load_task_result(data, TASK)["child_ref_promotion"]["status"] == "complete"
    manifest = _attached(data, drive, 1, msg_id="late")  # after the promotion completed
    staged = Path(manifest[0]["abs_path"])
    original = artifacts.copy_artifact_file

    def refuse_inputs(source, destination, **kwargs):
        if "attachments" in str(destination):
            raise OSError("input copy failed")
        return original(source, destination, **kwargs)

    monkeypatch.setattr(artifacts, "copy_artifact_file", refuse_inputs)
    assert _settle(data, drive)["reason"] == "inputs_uncustodied" and drive.is_dir()
    monkeypatch.undo()
    staged.unlink()
    assert _settle(data, drive)["reason"] == "inputs_uncustodied"  # a missing input is no permission either
    staged.write_text("input 0 of late", encoding="utf-8")
    assert _settle(data, drive)["status"] == "removed"
    projection = next(iter(load_task_result(data, TASK)["unread_mailbox"]["inputs"].values()))
    assert artifacts.resolve_attachment_manifest(data, TASK, projection)[0]["abs_path"].startswith(
        str(artifacts.task_artifact_dir_path(data, TASK).resolve()))


def test_the_view_ranks_identity_apart_from_location_and_refuses_a_conflict(tmp_path):
    """F5: the highest-ranked identity claim wins wherever it was recorded (both drive orders),
    a registration whose file is gone still shows its identity, and a canonical measured row
    that names different bytes than the child's immutable capture is a refusal, never a
    downgrade."""
    from ouroboros.task_status import load_effective_task_result

    data = tmp_path / "data"
    headless_drive = _drive(data, "headless")
    direct_drive = _drive(data, "task_drives")
    capture = _capture(tmp_path, headless_drive)  # an immutable registration in one drive
    twin = artifacts.task_artifact_dir_path(direct_drive, TASK, create=True) / capture["name"]
    twin.write_text("captured report", encoding="utf-8")  # the same relpath listed unmeasured in the other
    write_task_result(data, TASK, "cancelled", result="x")
    for order in (True, False):
        rows = task_custody.store_artifact_view(data, TASK, [], {} if order else None)
        row = next(item for item in rows if item["name"] == capture["name"])
        assert (row["sha256"], row["immutable"], row.get("measured")) == (capture["sha256"], True, None)

    Path(capture["path"]).unlink()  # registration only, file gone
    twin.unlink()
    row = next(item for item in load_effective_task_result(data, TASK)["artifacts"] if item["name"] == capture["name"])
    assert (row["sha256"], row["immutable"]) == (capture["sha256"], True)

    canonical_copy = _canonical(data, capture["name"])
    canonical_copy.parent.mkdir(parents=True)
    canonical_copy.write_text("someone else's bytes", encoding="utf-8")
    write_task_result(data, TASK, "cancelled", artifacts=[artifacts.artifact_record(canonical_copy)])
    row = next(item for item in load_effective_task_result(data, TASK)["artifacts"] if item["name"] == capture["name"])
    assert row["status"] == "failed" and "conflict" in row["errors"][-1]
    assert (row["sha256"], row["immutable"], row["path"]) == (capture["sha256"], True, str(canonical_copy))


def test_exact_rows_never_collapse_on_a_shared_msg_id(tmp_path):
    """F6: two different payloads under one msg_id are two unread rows; an acknowledgement
    reads only the first (delivered) one; the mailbox goes only under the post-work predicate."""
    data = tmp_path / "data"
    drive = _drive(data, "headless")
    write_task_result(data, TASK, "scheduled", child_drive_root=str(drive))
    owner_mailbox.write_task_message(drive, "first payload", TASK, source_task_id="parent1", msg_id="dup")
    owner_mailbox.write_task_message(drive, "second payload", TASK, source_task_id="parent1", msg_id="dup")
    rows, complete = task_custody.unread_mail_rows(drive, TASK)
    assert complete and [json.loads(row)["text"] for row in rows] == ["first payload", "second payload"]
    assert owner_mailbox.acknowledge_task_messages(drive, TASK, ["dup"], wake_id="w1")
    rows, complete = task_custody.unread_mail_rows(drive, TASK)
    assert complete and [json.loads(row)["text"] for row in rows] == ["second payload"]
    write_task_result(data, TASK, "cancelled", result="x")
    held = load_task_result(data, TASK)["unread_mailbox"]
    assert [json.loads(row)["text"] for row in held["rows"]] == ["second payload"]
    assert task_custody.merge_unread_mail(held, {"rows": rows, "read_complete": True, "captured_at": "z"})["total"] == 1

    write_task_result(data, TASK, "cancelled", root_phase_checkpoint={"post_task_synthesis": "pending_once"})
    assert not owner_mailbox.cleanup_task_mailbox(drive, TASK) and owner_mailbox._mailbox_path(drive, TASK).is_file()
    assert _settle(data, drive)["reason"] == "post_work_open"
    empty = tmp_path / "empty"
    assert not task_custody.settle_task_mailbox(empty, "nobody1", empty)  # no canonical result: nothing goes
    write_task_result(data, TASK, "cancelled", root_phase_checkpoint={"post_task_synthesis": "completed"})
    assert owner_mailbox.cleanup_task_mailbox(drive, TASK) and not owner_mailbox._mailbox_path(drive, TASK).exists()


def test_a_closed_generation_stops_each_walk_before_its_next_commit(tmp_path, monkeypatch):
    """F7: the ref-promotion retry and the prune re-ask the generation predicate per item and
    at each publication's commit, so a first item held past the close publishes nothing more."""
    from ouroboros.observability import retry_pending_child_ref_promotions

    data = tmp_path / "data"
    pending = []
    for name in ("first1", "second1"):
        drive = headless.prepare_task_drive(data, name, "empty")
        source = drive / "report.txt"
        source.write_text(f"report of {name}", encoding="utf-8")
        record = artifacts.copy_file_to_task_artifacts(SimpleNamespace(drive_root=drive, task_id=name), source, immutable=True)
        write_task_result(drive, name, "completed", result="done", artifacts=[record], artifact_status="ready")
        write_task_result(data, name, "running", headless_child_drive_root=str(drive))
        Path(record["path"]).write_text("changed", encoding="utf-8")
        copied = headless.copy_child_task_result(data, {"id": name, "drive_root": str(drive)})
        assert copied["child_ref_promotion"]["status"] == "incomplete"
        Path(record["path"]).write_text(f"report of {name}", encoding="utf-8")
        pending.append(name)
    owner_mailbox.write_task_message(headless.task_state_dir(data, "first1") / "data", "late", "first1",
                                     source_task_id="parent1", msg_id="late")
    copies = []
    real_copy = artifacts.copy_artifact_file
    monkeypatch.setattr(artifacts, "copy_artifact_file", lambda *a, **k: copies.append(1) or real_copy(*a, **k))
    closed = []
    report = retry_pending_child_ref_promotions(data, stop=lambda: bool(closed) or closed.append(1))
    assert report["deferred"] == ["second1"] and report["retried"] == ["first1"]
    assert load_task_result(data, "first1")["child_ref_promotion"]["status"] == "incomplete", \
        "the generation closed at the first commit: nothing was published"
    assert load_task_result(data, "second1")["child_ref_promotion"]["status"] == "incomplete"
    assert copies == [], "a walk observed the close before it promoted any file"
    first_mailbox = owner_mailbox._mailbox_path(headless.task_state_dir(data, "first1") / "data", "first1")
    assert first_mailbox.is_file(), "a declined publication cleans no mailbox"
    monkeypatch.undo()
    assert retry_pending_child_ref_promotions(data)["completed"] == ["first1", "second1"]
    assert not first_mailbox.exists()
    assert [json.loads(row)["msg_id"] for row in load_task_result(data, "first1")["unread_mailbox"]["rows"]] == ["late"]

    later = 4_000_000_000
    report = headless.prune_headless_task_drives(data, retention_days=1, now=later, live=not_live, stop=lambda: True)
    assert report["deferred"] == ["first1", "second1"] and not report["pruned"]
    report = headless.prune_headless_task_drives(data, retention_days=1, now=later, live=not_live, budget=1)
    assert [row["task_id"] for row in report["pruned"]] == ["first1"] and report["deferred"] == ["second1"]
    assert report["cursor"] == "first1"
    report = headless.prune_headless_task_drives(data, retention_days=1, now=later, live=not_live, budget=1, after="first1")
    assert [row["task_id"] for row in report["pruned"]] == ["second1"]


def test_pure_custody_relocation_keeps_the_child_result_identity(tmp_path):
    """The parent's decision hash covers recorded identities, never the physical store: a
    settlement that only relocates recorded rows keeps it, a listing nobody recorded
    never enters it, and a content change still invalidates."""
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256

    data, drive, store = _nested_child(tmp_path)
    (store / "unrecorded.txt").write_text("nobody recorded me", encoding="utf-8")
    write_task_result(data, TASK, "completed", result="done", child_drive_root=str(drive))
    headless.copy_child_task_result(data, {"id": TASK, "drive_root": str(drive)})
    before = _child_result_sha256(load_effective_task_result(data, TASK))
    assert _settle(data, drive)["reason"] == "child_refs_pending"
    headless.retry_child_task_refs(data, drive, TASK)
    assert _child_result_sha256(load_effective_task_result(data, TASK)) == before
    assert _settle(data, drive)["status"] == "removed"
    after_view = load_effective_task_result(data, TASK)
    assert _child_result_sha256(after_view) == before
    assert {row["name"] for row in after_view["artifacts"]} == {"summary.txt", "unrecorded.txt"}
    changed = next(row for row in after_view["artifacts"] if row.get("relpath") == "reports/a/summary.txt")
    Path(changed["path"]).write_text("alpha v3", encoding="utf-8")
    write_task_result(data, TASK, "completed", artifacts=[
        {**row, **artifacts.stream_artifact_file(row["path"])} if row is changed else row for row in after_view["artifacts"]
        if row.get("measured") is not False])
    assert _child_result_sha256(load_effective_task_result(data, TASK)) != before


# ------------------------------------------------------------------ exact-review repair probes
# One probe per finding of the whole-diff review: the class of failure, never the local example.


def test_a_mutable_child_edit_after_copy_back_is_published_not_lost(tmp_path):
    """P1: copy-back recorded A for a mutable file, so the child row and the canonical row AGREE;
    the child then rewrote it to same-length B without re-recording. Settlement compares the
    child's CURRENT bytes, never agreeing metadata: B publishes beside A, then the drive goes."""
    data, drive, store = _nested_child(tmp_path)
    headless.copy_child_task_result(data, {"id": TASK, "drive_root": str(drive)})
    headless.retry_child_task_refs(data, drive, TASK)
    canonical = artifacts.task_artifact_dir_path(data, TASK)
    (store / "reports/a/summary.txt").write_text("ALPHA", encoding="utf-8")  # same length, no re-registration

    outcome = _settle(data, drive)

    assert outcome == {"status": "removed", "reason": "", "published": 1}
    assert (canonical / "reports/a/summary.txt").read_text(encoding="utf-8") == "alpha"  # never replaced
    current = next(row for row in load_task_result(data, TASK)["artifacts"]
                   if row["sha256"] == sha256(b"ALPHA").hexdigest())
    assert Path(current["path"]).read_text(encoding="utf-8") == "ALPHA"
    assert current["relpath"].startswith("reports/a/summary.") and current["name"] != "summary.txt"
    assert not drive.exists()


@pytest.mark.parametrize("claim", ["stale_child_claim", "digestless_canonical_row"])
def test_a_missing_mutable_source_is_held_only_by_agreeing_recorded_digests(tmp_path, claim):
    """P1: with the source gone, the canonical copy proves custody only when the canonical row
    records a digest it verifies against AND no child-side claim names other bytes; a stale or
    digest-less canonical row is convenient metadata, not preserved proof."""
    data, drive, store = _nested_child(tmp_path)
    headless.copy_child_task_result(data, {"id": TASK, "drive_root": str(drive)})
    headless.retry_child_task_refs(data, drive, TASK)
    rel = "reports/a/summary.txt"
    child_rows = load_task_result(drive, TASK)["artifacts"]
    if claim == "stale_child_claim":
        write_task_result(drive, TASK, "completed", artifacts=[
            {**row, "sha256": "f" * 64} if row.get("relpath") == rel else row for row in child_rows])
    else:
        canonical_rows = load_task_result(data, TASK)["artifacts"]
        write_task_result(data, TASK, "completed", artifacts=[
            {key: value for key, value in row.items() if key != "sha256"} if row.get("relpath") == rel else row
            for row in canonical_rows])
    (store / rel).unlink()

    assert _settle(data, drive)["reason"] == "artifact_source_missing" and drive.is_dir()

    if claim == "stale_child_claim":
        write_task_result(drive, TASK, "completed", artifacts=child_rows)  # the claims agree again
    else:
        (store / rel).write_text("alpha", encoding="utf-8")  # the source is back: current bytes decide
    assert _settle(data, drive)["status"] == "removed"


@pytest.mark.parametrize("count", [2, 30])
def test_a_mailbox_cleanup_before_the_settlement_never_strands_unread_attachments(tmp_path, count):
    """P1: the loop thread's cheap cleanup keeps a mailbox whose unread rows carry inputs that
    are not in verified canonical custody; the off-loop cleanup carries them (one verified
    canonical projection per exact row) before it unlinks; the later settlement re-verifies the
    held closure and the drive goes with nothing stranded."""
    data = tmp_path / "data"
    drive = _drive(data, "headless")
    write_task_result(data, TASK, "scheduled", child_drive_root=str(drive))
    _attached(data, drive, count, msg_id="m1")
    write_task_result(data, TASK, "cancelled", result="Cancelled before start.")
    row = load_task_result(data, TASK)["unread_mailbox"]["rows"][0]

    assert owner_mailbox.cleanup_task_mailbox(drive, TASK, canonical_root=data, carry_inputs=False) is False
    assert owner_mailbox._mailbox_path(drive, TASK).is_file()
    assert "inputs" not in load_task_result(data, TASK)["unread_mailbox"]

    assert owner_mailbox.cleanup_task_mailbox(drive, TASK, canonical_root=data) is True
    assert not owner_mailbox._mailbox_path(drive, TASK).exists()
    custody = load_task_result(data, TASK)["unread_mailbox"]
    assert custody["rows"] == [row]
    resolved = artifacts.resolve_attachment_manifest(data, TASK, custody["inputs"][task_custody._row_key(row)])
    canonical_store = artifacts.task_artifact_dir_path(data, TASK).resolve()
    assert len(resolved) == count and all(Path(item["abs_path"]).is_relative_to(canonical_store) for item in resolved)

    assert _settle(data, drive)["status"] == "removed"
    for item in resolved:
        artifacts.stream_artifact_file(Path(item["abs_path"]), expected=item)
    assert load_task_result(data, TASK)["unread_mailbox"]["rows"] == [row]


def test_a_held_projection_is_re_verified_not_trusted_by_its_key(tmp_path):
    """P1: a recorded closure whose canonical bytes are gone is not custody; the settlement
    re-carries it from the drive (and retains when the drive cannot supply it either)."""
    data = tmp_path / "data"
    drive = _drive(data, "headless")
    write_task_result(data, TASK, "scheduled", child_drive_root=str(drive))
    manifest = _attached(data, drive, 1, msg_id="m1")
    write_task_result(data, TASK, "cancelled", result="Cancelled before start.")
    assert owner_mailbox.cleanup_task_mailbox(drive, TASK, canonical_root=data) is True
    row = load_task_result(data, TASK)["unread_mailbox"]["rows"][0]
    projection = load_task_result(data, TASK)["unread_mailbox"]["inputs"][task_custody._row_key(row)]
    canonical_copy = Path(artifacts.resolve_attachment_manifest(data, TASK, projection)[0]["abs_path"])
    canonical_copy.unlink()  # the key is still there; the bytes are not
    source = Path(manifest[0]["abs_path"])
    source.rename(source.with_name("hidden"))

    assert _settle(data, drive)["reason"] == "inputs_uncustodied" and drive.is_dir()

    source.with_name("hidden").rename(source)
    assert _settle(data, drive)["status"] == "removed"
    artifacts.stream_artifact_file(canonical_copy, expected=projection["attachment_manifest"][0])


@pytest.mark.parametrize("late", ["attachments", "text"])
def test_a_row_landing_between_preparation_and_publication_is_carried_in_the_same_pass(tmp_path, monkeypatch, late):
    """P1: the input closure is derived under the custody lock from the CURRENT mailbox and the
    durable rows, so mail that lands after preparation is carried, never marked complete on
    the strength of the rows prepared earlier."""
    data, drive, _record = _cancelled_with_capture(tmp_path)
    real = task_custody._child_store_plan

    def plan_then_late_mail(*args, **kwargs):
        plan = real(*args, **kwargs)
        if late == "attachments":
            _attached(data, drive, 1, msg_id="late")
        else:
            owner_mailbox.write_owner_message(drive, "late words", TASK, msg_id="late")
        return plan

    monkeypatch.setattr(task_custody, "_child_store_plan", plan_then_late_mail)
    assert _settle(data, drive)["status"] == "removed"
    custody = load_task_result(data, TASK)["unread_mailbox"]
    row = next(item for item in custody["rows"] if json.loads(item)["msg_id"] == "late")
    if late == "attachments":
        resolved = artifacts.resolve_attachment_manifest(data, TASK, custody["inputs"][task_custody._row_key(row)])
        artifacts.stream_artifact_file(Path(resolved[0]["abs_path"]), expected=resolved[0])
        assert Path(resolved[0]["abs_path"]).is_relative_to(artifacts.task_artifact_dir_path(data, TASK).resolve())
    else:
        assert "inputs" not in custody


def test_publication_never_overwrites_bytes_that_appear_between_the_check_and_the_placement(tmp_path, monkeypatch):
    """F3: the identity check and the placement are two steps for the OS, so the placement
    itself is create-only: an intruder landing between them keeps its bytes, the pass retains,
    and the capture publishes beside it on the next pass."""
    data, drive, record = _cancelled_with_capture(tmp_path)
    real = task_custody._identity
    intruded = []

    def identity_then_intruder(path):
        observed = real(path)
        if path.name == record["name"] and path.parent.name == TASK and observed is None and not intruded:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("intruder", encoding="utf-8")
            intruded.append(path)
        return observed

    monkeypatch.setattr(task_custody, "_identity", identity_then_intruder)
    assert _settle(data, drive)["reason"] == "destination_changed"
    monkeypatch.undo()
    assert intruded and intruded[0].read_text(encoding="utf-8") == "intruder" and drive.is_dir()
    assert list((data / "state" / "custody_staging").iterdir()) == []
    assert _settle(data, drive)["status"] == "removed"
    assert intruded[0].read_text(encoding="utf-8") == "intruder"
    beside = next(item for item in load_task_result(data, TASK)["artifacts"] if item["sha256"] == record["sha256"])
    assert Path(beside["path"]).read_text(encoding="utf-8") == "captured report" and beside["name"] != record["name"]


def test_every_canonical_store_publisher_takes_the_one_custody_lock(tmp_path):
    """F3: host artifact finalization publishes into the same canonical store as copy-back and
    settlement, so it takes the same custody lock and answers ``CustodyBusy`` while it is held."""
    data, drive, _store = _nested_child(tmp_path)
    task = {"id": TASK, "drive_root": str(drive)}
    lock_budget = 0.01  # Exercise real contention without waiting out the production budget.
    with task_custody.task_custody_lock(data, TASK) as locked:
        assert locked
        with pytest.raises(task_custody.CustodyBusy):
            headless.finalize_task_artifacts(data, task, lock_timeout_sec=lock_budget)
    assert not (artifacts.task_artifact_dir_path(data, TASK) / "memory_export.json").exists()
    assert headless.finalize_task_artifacts(data, task, lock_timeout_sec=lock_budget)  # released: the same work completes


def test_artifact_publication_keeps_the_production_lock_budget(tmp_path, monkeypatch):
    from contextlib import nullcontext

    budgets = []

    def unavailable(_root, _task_id, *, timeout_sec):
        budgets.append(timeout_sec)
        return nullcontext(False)

    monkeypatch.setattr(task_custody, "task_custody_lock", unavailable)
    with pytest.raises(task_custody.CustodyBusy):
        headless.finalize_task_artifacts(tmp_path, {"id": TASK})
    assert budgets == [30.0]  # Production publication may wait for a competing writer.


@pytest.mark.parametrize("when", ["before_the_custody_lock", "before_the_row_lock"])
def test_a_cancel_after_the_copy_back_read_declines_every_child_field(tmp_path, monkeypatch, when):
    """P2: cancellation is re-asked under the custody lock and again in the row projector; a
    row cancelled after copy-back's first read adopts nothing from the child, while the child's
    files still reach canonical custody through the settlement."""
    data = tmp_path / "data"
    drive = _drive(data, "headless")
    record = _capture(tmp_path, drive)
    write_task_result(drive, TASK, "completed", result="child done", artifacts=[record], artifact_status="ready")
    write_task_result(data, TASK, "running", headless_child_drive_root=str(drive))
    cancelled = []

    def cancel_once():
        if not cancelled:
            cancelled.append(1)
            write_task_result(data, TASK, "cancelled", result="Cancelled.")

    if when == "before_the_custody_lock":
        real_lock = task_custody.task_custody_lock

        def lock_after_a_cancel(*args, **kwargs):
            cancel_once()
            return real_lock(*args, **kwargs)
        monkeypatch.setattr(task_custody, "task_custody_lock", lock_after_a_cancel)
    else:
        real_project = headless.project_replica_task_result_fields

        def project_after_a_cancel(*args, **kwargs):
            cancel_once()
            return real_project(*args, **kwargs)
        monkeypatch.setattr(headless, "project_replica_task_result_fields", project_after_a_cancel)

    returned = headless.copy_child_task_result(data, {"id": TASK, "drive_root": str(drive)})
    monkeypatch.undo()

    row = load_task_result(data, TASK)
    assert returned["status"] == row["status"] == "cancelled" and row["result"] == "Cancelled."
    assert not any(key in row for key in ("artifacts", "child_ref_promotion", "child_status", "artifact_bundle"))
    # The files are separate custody: placed by the copy that observed the cancel only at its
    # row commit, or by the settlement; either way the cancelled row lists the capture afterwards.
    assert _settle(data, drive)["status"] == "removed"
    assert _canonical(data, record["name"]).read_text(encoding="utf-8") == "captured report"
    settled = load_task_result(data, TASK)
    assert settled["status"] == "cancelled" and [item["sha256"] for item in settled["artifacts"]] == [record["sha256"]]


@pytest.mark.parametrize("boundary", ["lock_wait", "before_files", "row_commit"])
def test_a_generation_closed_during_the_lock_wait_or_preparation_starts_no_effect(tmp_path, monkeypatch, boundary):
    """F7: the generation is re-asked at the real publication boundaries - after the locks are
    taken, before any file or manifest moves, and at the row commit - so a close observed
    during the lock wait or preparation leaves nothing behind but, at worst, one create-only
    file the next pass lists without duplicating."""
    data, drive, record = _cancelled_with_capture(tmp_path)
    _attached(data, drive, 1, msg_id="m1")
    assert append_verification_receipt(drive, TASK, {"status": "pass", "criterion_id": "child-check"})
    closed = []

    def stop():
        return bool(closed)

    if boundary == "lock_wait":
        real_lock = task_custody.task_custody_lock

        def lock_then_close(*args, **kwargs):
            closed.append(1)
            return real_lock(*args, **kwargs)
        monkeypatch.setattr(task_custody, "task_custody_lock", lock_then_close)
    elif boundary == "before_files":
        from ouroboros import outcome_receipt_store

        def union_then_close(*args, **kwargs):
            closed.append(1)
            return True
        monkeypatch.setattr(outcome_receipt_store, "publish_verification_receipt_union", union_then_close)
    else:
        real_publish = task_custody._publish

        def publish_then_close(*args, **kwargs):
            published = real_publish(*args, **kwargs)
            closed.append(1)
            return published
        monkeypatch.setattr(task_custody, "_publish", publish_then_close)

    assert _settle(data, drive, stop=stop)["reason"] == "generation_closed" and drive.is_dir()
    monkeypatch.undo()
    row = load_task_result(data, TASK)
    assert "artifacts" not in row and "inputs" not in (row.get("unread_mailbox") or {})
    if boundary != "row_commit":
        assert not _canonical(data, record["name"]).exists()
        assert not (artifacts.task_artifact_dir_path(data, TASK) / "attachments").exists()
        assert read_verification_receipts(data, TASK) == [] if boundary == "lock_wait" else True
    closed.clear()
    assert _settle(data, drive, stop=stop)["status"] == "removed"
    rows = [item for item in load_task_result(data, TASK)["artifacts"] if item["sha256"] == record["sha256"]]
    assert len(rows) == 1 and Path(rows[0]["path"]).read_text(encoding="utf-8") == "captured report"
    assert [item.get("criterion_id") for item in read_verification_receipts(data, TASK)] == ["child-check"]


def test_the_liveness_probe_is_never_asked_under_a_custody_lock(tmp_path, monkeypatch):
    """F7: the probe takes the supervisor's queue lock and Phase B takes custody locks under
    that queue lock, so asking the probe while holding a custody lock would invert the order."""
    import contextlib

    data, drive, _record = _cancelled_with_capture(tmp_path)
    held = []
    real = task_custody.task_custody_lock

    @contextlib.contextmanager
    def tracked(*args, **kwargs):
        with real(*args, **kwargs) as locked:
            held.append(1)
            try:
                yield locked
            finally:
                held.pop()

    monkeypatch.setattr(task_custody, "task_custody_lock", tracked)
    under_lock = []

    def probe(_task):
        if held:
            under_lock.append(1)
        return False

    assert _settle(data, drive, live=probe)["status"] == "removed"
    assert not under_lock


@pytest.mark.parametrize("state", ["child_healthy", "child_missing", "canonical_foreign"])
def test_a_canonical_manifest_missing_a_member_is_carried_whole_from_the_child(tmp_path, state):
    """R2: a >25-row input closure whose manifest already resolves in the canonical store but lacks
    one member is no closure: it is carried whole from the child's intact store. With the child's
    member gone too the drive is retained (and a restore converges); canonical bytes that differ
    from the captured identity are never replaced and never claimed complete."""
    data = tmp_path / "data"
    drive = _drive(data, "headless")
    write_task_result(data, TASK, "scheduled", child_drive_root=str(drive))
    manifest = _attached(data, drive, 30, msg_id="m1")
    write_task_result(data, TASK, "cancelled", result="Cancelled before start.")
    assert owner_mailbox.cleanup_task_mailbox(drive, TASK, canonical_root=data) is True  # manifest now canonical
    row = load_task_result(data, TASK)["unread_mailbox"]["rows"][0]
    member = artifacts.task_artifact_dir_path(data, TASK) / manifest[7]["relpath"]
    child_member = Path(manifest[7]["abs_path"])
    member.unlink()

    if state == "child_missing":
        child_member.rename(child_member.with_name("hidden"))
        assert _settle(data, drive)["reason"] == "inputs_uncustodied" and drive.is_dir() and not member.exists()
        child_member.with_name("hidden").rename(child_member)
    elif state == "canonical_foreign":
        member.write_text("someone else's bytes", encoding="utf-8")
        assert _settle(data, drive)["reason"] == "inputs_uncustodied" and drive.is_dir()
        assert member.read_text(encoding="utf-8") == "someone else's bytes", "a canonical file was replaced"
        assert "inputs" in load_task_result(data, TASK)["unread_mailbox"]  # the old projection, never re-claimed
        return
    assert _settle(data, drive)["status"] == "removed"
    projection = load_task_result(data, TASK)["unread_mailbox"]["inputs"][task_custody._row_key(row)]
    resolved = artifacts.resolve_attachment_manifest(data, TASK, projection)
    assert len(resolved) == 30
    for item in resolved:
        assert Path(item["abs_path"]).is_relative_to(artifacts.task_artifact_dir_path(data, TASK).resolve())
        artifacts.stream_artifact_file(Path(item["abs_path"]), expected=item)
    assert member.read_text(encoding="utf-8") == "input 7 of m1"


def _close_after_placed_copies(monkeypatch, store: Path, count: int = 1):
    """Close the generation after COUNT copies placed into STORE; ``written_after`` collects any
    copy that still wrote into it afterwards (an attempt refused by the fence never returns)."""
    state: dict = {"closed": False, "placed": 0, "written_after": [], "attempted_after": 0}
    real = artifacts.copy_artifact_file

    def copy(source, destination, **kwargs):
        placed = Path(destination).resolve().is_relative_to(store.resolve()) \
            and Path(source).resolve() != Path(destination).resolve()
        was_closed = state["closed"]
        state["attempted_after"] += int(placed and was_closed)
        measured = real(source, destination, **kwargs)
        if placed:
            if was_closed:
                state["written_after"].append(str(destination))
            state["placed"] += 1
            state["closed"] = state["closed"] or state["placed"] >= count
        return measured

    monkeypatch.setattr(artifacts, "copy_artifact_file", copy)
    return state


def test_a_close_inside_the_input_closure_places_no_further_input(tmp_path, monkeypatch):
    """R3: the settlement fences every copy of its input closure, not only its phase boundaries:
    a generation closed after the first of 30 unread inputs landed places none of the others,
    writes no row and keeps the drive and its mailbox; the next generation converges."""
    data = tmp_path / "data"
    drive = _drive(data, "headless")
    write_task_result(data, TASK, "scheduled", child_drive_root=str(drive))
    manifest = _attached(data, drive, 30, msg_id="m1")
    write_task_result(data, TASK, "cancelled", result="Cancelled before start.")
    before = load_task_result(data, TASK)
    state = _close_after_placed_copies(monkeypatch, artifacts.task_artifact_dir_path(data, TASK))

    assert _settle(data, drive, stop=lambda: state["closed"])["reason"] == "generation_closed"
    assert state["placed"] == 1 and state["attempted_after"] >= 1 and state["written_after"] == []
    assert load_task_result(data, TASK) == before and owner_mailbox._mailbox_path(drive, TASK).is_file()
    monkeypatch.undo()
    assert _settle(data, drive)["status"] == "removed"
    store = artifacts.task_artifact_dir_path(data, TASK)
    for row in manifest:
        artifacts.stream_artifact_file(store / row["relpath"], expected=row)


def test_a_close_between_placements_places_nothing_more(tmp_path, monkeypatch):
    """R3: each create-only placement asks the generation first: closed after the first of two
    captures was placed, the second is never placed and no row lists either; the next pass
    recognizes the placed bytes and publishes each capture exactly once."""
    data = tmp_path / "data"
    drive = _drive(data, "headless")
    records = [_capture(tmp_path, drive, text) for text in ("first report", "second report")]
    write_task_result(drive, TASK, "running", artifacts=records)
    write_task_result(data, TASK, "cancelled", result="Cancelled.", child_drive_root=str(drive))
    closed, real_link = [], os.link

    def link_then_close(source, destination, *args, **kwargs):
        real_link(source, destination, *args, **kwargs)
        closed.append(str(destination))
    monkeypatch.setattr(os, "link", link_then_close)

    assert _settle(data, drive, stop=lambda: bool(closed))["reason"] == "generation_closed" and drive.is_dir()
    assert len(closed) == 1, "a second capture was placed after the close"
    placed = [record for record in records if _canonical(data, record["name"]).exists()]
    assert len(placed) == 1 and "artifacts" not in load_task_result(data, TASK)
    monkeypatch.undo()
    assert _settle(data, drive)["status"] == "removed"
    rows = load_task_result(data, TASK)["artifacts"]
    assert sorted(row["sha256"] for row in rows) == sorted(record["sha256"] for record in records)
    for record in records:
        artifacts.stream_artifact_file(_canonical(data, record["name"]), expected=record)


def test_a_close_inside_the_ref_retry_walk_promotes_nothing_more_and_keeps_its_evidence(tmp_path, monkeypatch):
    """R3: the pending-ref retry threads the generation into every file it promotes, not only
    before its walk and at its commit: closed after the first pending artifact was copied, the
    rest are refused, the row keeps its exact pending refs, and the next retry completes without
    duplicating the copy that landed."""
    data = tmp_path / "data"
    drive = headless.prepare_task_drive(data, TASK, "empty")
    records = []
    for name in ("one", "two", "three"):
        source = drive / f"{name}.txt"
        source.write_text(f"report {name}", encoding="utf-8")
        records.append(artifacts.copy_file_to_task_artifacts(SimpleNamespace(drive_root=drive, task_id=TASK),
                                                             source, immutable=True))
    write_task_result(drive, TASK, "completed", result="done", artifacts=records, artifact_status="ready")
    write_task_result(data, TASK, "running", headless_child_drive_root=str(drive))
    for record in records:
        Path(record["path"]).write_text("changed", encoding="utf-8")
    copied = headless.copy_child_task_result(data, {"id": TASK, "drive_root": str(drive)})
    pending = copied["child_ref_promotion"]["pending_refs"]
    assert len([ref for ref in pending if ref.get("kind") != "history_retention_deferred"]) == 3
    assert any(ref.get("kind") == "history_retention_deferred" for ref in pending)
    for record, name in zip(records, ("one", "two", "three")):
        Path(record["path"]).write_text(f"report {name}", encoding="utf-8")
    before = load_task_result(data, TASK)
    state = _close_after_placed_copies(monkeypatch, artifacts.task_artifact_dir_path(data, TASK))

    assert headless.retry_child_task_refs(data, drive, TASK, stop=lambda: state["closed"]) == before
    assert state["placed"] == 1 and state["attempted_after"] == 2 and state["written_after"] == []
    assert load_task_result(data, TASK) == before, "the pending refs are the retry evidence"
    monkeypatch.undo()
    settled = headless.retry_child_task_refs(data, drive, TASK)
    assert settled["child_ref_promotion"]["status"] == "complete"
    assert sorted(row["sha256"] for row in settled["artifacts"]) == sorted(record["sha256"] for record in records)
    for record in records:
        artifacts.stream_artifact_file(_canonical(data, record["name"]), expected=record)


def test_a_close_after_the_version_backup_deletes_no_retained_version(tmp_path, monkeypatch):
    """R3: the version retention is a fenced unlink too: a generation closed after the backup of
    the current copy landed deletes none of the five retained versions, replaces nothing and keeps
    the pending ref; the next generation publishes the restored bytes and bounds the history."""
    data = tmp_path / "data"
    drive = headless.prepare_task_drive(data, TASK, "empty")
    source = artifacts.task_artifact_dir_path(drive, TASK, create=True) / "report.txt"
    source.write_text("v0", encoding="utf-8")
    write_task_result(drive, TASK, "completed", result="done", artifacts=[artifacts.artifact_record(source)],
                      artifact_status="ready")
    write_task_result(data, TASK, "running", headless_child_drive_root=str(drive))
    task = {"id": TASK, "drive_root": str(drive)}
    for version in range(6):
        source.write_text(f"v{version}", encoding="utf-8")
        headless.copy_child_task_result(data, task)
    versions = data / "task_results" / "artifact_versions" / TASK / "report.txt"
    history = sorted(versions.iterdir())
    assert [path.read_text(encoding="utf-8") for path in history] == ["v0", "v1", "v2", "v3", "v4"]
    source.unlink()
    assert headless.copy_child_task_result(data, task)["child_ref_promotion"]["pending_refs"]
    source.write_text("v6", encoding="utf-8")  # restored: the retry has a newer copy to publish
    before = load_task_result(data, TASK)
    state = _close_after_placed_copies(monkeypatch, versions)

    assert headless.retry_child_task_refs(data, drive, TASK, stop=lambda: state["closed"]) == before
    assert state["placed"] == 1 and state["written_after"] == []
    assert [path.read_text(encoding="utf-8") for path in history] == ["v0", "v1", "v2", "v3", "v4"]
    assert sorted(path.read_text(encoding="utf-8") for path in versions.iterdir()) == ["v0", "v1", "v2", "v3", "v4", "v5"]
    assert _canonical(data, "report.txt").read_text(encoding="utf-8") == "v5"
    assert load_task_result(data, TASK) == before, "the pending ref is the retry evidence"
    monkeypatch.undo()
    settled = headless.retry_child_task_refs(data, drive, TASK)
    assert settled["child_ref_promotion"]["status"] == "complete"
    assert _canonical(data, "report.txt").read_text(encoding="utf-8") == "v6"
    row = next(item for item in settled["artifacts"] if item["name"] == "report.txt")
    assert row["sha256"] == artifacts.stream_artifact_file(_canonical(data, "report.txt"))["sha256"]
    assert len(list(versions.iterdir())) == 5  # the open generation applies the bounded retention
