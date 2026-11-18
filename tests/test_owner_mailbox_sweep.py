"""CPL4-C18 pins: the drive-custody pass sweeps mailboxes whose task settled off-path.

A mailbox (and its acks file) goes only when the task's durable result is
SETTLED; no result, a non-terminal result, or an unclassifiable filename
keeps it — an undelivered owner directive must survive any ambiguity.
"""

from __future__ import annotations

import json

import pytest

from ouroboros.owner_mailbox import (
    _ack_path,
    _mailbox_path,
    sweep_settled_owner_mailboxes,
    write_owner_message,
)
from ouroboros.task_result_schema import stamp_task_result_schema
from ouroboros import maintenance_jobs, server_maintenance, store_import
from tests._store_shared import seed_result


def _settle(root, task_id, status="completed"):
    results = root / "task_results"
    results.mkdir(parents=True, exist_ok=True)
    row = stamp_task_result_schema({"task_id": task_id, "status": status})
    seed_result(root, task_id, row)


def test_settled_task_mailbox_swept_others_kept(tmp_path):
    write_owner_message(tmp_path, "goodbye", "t-done")
    _settle(tmp_path, "t-done")
    write_owner_message(tmp_path, "still live", "t-running")
    _settle(tmp_path, "t-running", status="running")
    write_owner_message(tmp_path, "orphan but unproven", "t-noresult")

    report = sweep_settled_owner_mailboxes(tmp_path)

    assert report["removed"] == ["t-done"]
    assert not _mailbox_path(tmp_path, "t-done").exists()
    assert not _ack_path(tmp_path, "t-done").exists()
    assert _mailbox_path(tmp_path, "t-running").exists()
    assert _mailbox_path(tmp_path, "t-noresult").exists()  # fail-closed
    assert report["kept"] == 2


@pytest.mark.parametrize("valid", [True, False], ids=["saved-read", "malformed-receipt"])
def test_acks_file_travels_with_its_mailbox(tmp_path, valid):
    from ouroboros import working_checkpoint
    from tests._question_answer_shared import stage_question_mail

    assert write_owner_message(tmp_path, "hi", "t-done", msg_id="greeting")
    ack = _ack_path(tmp_path, "t-done")
    if valid:
        _, limit = stage_question_mail(tmp_path, "t-done")
        working_checkpoint.save_ready(limit)
        assert json.loads(ack.read_text(encoding="utf-8"))["msg_id"] == "greeting"
    else:
        ack.write_text("{}\n", encoding="utf-8")
    _settle(tmp_path, "t-done")

    report = sweep_settled_owner_mailboxes(tmp_path)

    assert report["removed"] == (["t-done"] if valid else [])
    assert ack.exists() is (not valid)
    assert _mailbox_path(tmp_path, "t-done").exists() is (not valid)


def test_orphan_acks_without_mailbox_left_alone(tmp_path):
    # No result establishes that this name belongs to a settled task.
    _ack_path(tmp_path, "t-ghost").parent.mkdir(parents=True, exist_ok=True)
    _ack_path(tmp_path, "t-ghost").write_text("{}\n", encoding="utf-8")

    report = sweep_settled_owner_mailboxes(tmp_path)

    assert report == {"removed": [], "kept": 0}
    assert _ack_path(tmp_path, "t-ghost").exists()


@pytest.mark.parametrize("condition", ["torn", "overlapping-mailbox", "active"])
def test_sweep_retains_unverified_or_active_sidecar(tmp_path, condition):
    from ouroboros import working_checkpoint
    from ouroboros.task_results import write_task_result
    from tests._question_answer_shared import stage_question_mail

    task_id = "task.with.dots"
    write_task_result(tmp_path, task_id, status="running")
    assert write_owner_message(tmp_path, "Hello", task_id, msg_id="hello")
    _, limit = stage_question_mail(tmp_path, task_id)
    working_checkpoint.save_ready(limit)
    ack = _ack_path(tmp_path, task_id)
    if condition != "active":
        settled = write_task_result(tmp_path, task_id, status="completed")
        assert settled["owner_mailbox"]["read_complete"]
    _mailbox_path(tmp_path, task_id).unlink()
    if condition == "torn":
        with ack.open("ab") as stream:
            stream.write(b'{"msg_id":"other",')
    elif condition == "overlapping-mailbox":
        assert write_owner_message(tmp_path, "Still needed", task_id + ".acks", msg_id="later")
    saved = ack.read_bytes()
    assert sweep_settled_owner_mailboxes(tmp_path) == {"removed": [], "kept": 1}
    assert ack.read_bytes() == saved


@pytest.mark.parametrize("receipts_only", [False, True], ids=["mixed-mail", "receipts-only"])
def test_settling_suffixed_recipient_preserves_both_owners_records(tmp_path, receipts_only):
    from ouroboros.owner_mailbox import acknowledge_task_messages
    from ouroboros.task_results import write_task_result
    from supervisor.terminal_delivery import cleanup_settled_owner_mailbox

    base, recipient = "sample", "sample.acks"
    shared = _mailbox_path(tmp_path, recipient)
    assert shared == _ack_path(tmp_path, base)
    for task in (base, recipient):
        write_task_result(tmp_path, task, "running")
        assert write_owner_message(tmp_path, f"Letter for {task}", task, msg_id=task)
    assert acknowledge_task_messages(tmp_path, recipient, [recipient], wake_id="loop_delivery")
    write_task_result(tmp_path, recipient, "completed")
    if receipts_only:
        # An earlier cleanup stopped between the recipient's mailbox and its receipts.
        shared.unlink()
    assert acknowledge_task_messages(tmp_path, base, [base], wake_id="loop_delivery")
    write_task_result(tmp_path, base, "completed")
    if receipts_only:
        _mailbox_path(tmp_path, base).unlink()

    owned = {path: path.read_bytes() for path in shared.parent.glob("*.jsonl")}
    cleanup_settled_owner_mailbox(tmp_path, recipient, {"drive_root": str(tmp_path)})
    assert {path: path.read_bytes() for path in owned} == owned
    if receipts_only:
        # Each owner can still retire its own verified orphan receipt.
        assert sweep_settled_owner_mailboxes(tmp_path)["removed"] == [base]
        assert _ack_path(tmp_path, recipient).read_bytes() == owned[_ack_path(tmp_path, recipient)]
        assert sweep_settled_owner_mailboxes(tmp_path)["removed"] == [recipient]
    else:
        cleanup_settled_owner_mailbox(tmp_path, base, {"drive_root": str(tmp_path)})
        assert sweep_settled_owner_mailboxes(tmp_path)["removed"] == []
        assert {path: path.read_bytes() for path in owned} == owned


@pytest.mark.parametrize("retry", [False, True], ids=["canonical", "predecessor-drive"])
def test_suffixed_recipient_retires_its_own_letters(tmp_path, retry):
    from ouroboros.owner_mailbox import acknowledge_task_messages
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.terminal_delivery import cleanup_settled_owner_mailbox

    task = "sample.acks"
    drive = tmp_path / "task_drives" / "predecessor" if retry else tmp_path
    drive.mkdir(parents=True, exist_ok=True)
    write_task_result(tmp_path, task, "running", child_drive_root=str(drive))
    assert write_owner_message(drive, "Keep the café draft", task, msg_id="draft")
    path = _mailbox_path(drive, task)
    [letter] = path.read_bytes().splitlines()
    if retry:
        assert acknowledge_task_messages(drive, task, ["draft"], wake_id="loop_delivery")
    write_task_result(tmp_path, task, "completed")
    cleanup_settled_owner_mailbox(tmp_path, task)
    assert not path.exists() and not _ack_path(drive, task).exists()
    [saved] = load_task_result(tmp_path, task)["owner_mailbox"]["rows"]
    assert saved.encode("utf-8") == letter


def test_the_drive_custody_pass_runs_the_mailbox_sweep():
    import inspect

    import ouroboros.server_maintenance as sm

    assert "sweep_settled_owner_mailboxes" in inspect.getsource(sm._run_drive_custody_pass)
    assert "sweep_settled_owner_mailboxes" not in inspect.getsource(sm._run_reconcile_prunes)


def test_drive_pass_records_an_event_when_only_a_mailbox_was_removed(tmp_path, monkeypatch):
    store_import.ensure_current_store(tmp_path)
    write_owner_message(tmp_path, "goodbye", "t-done")
    _settle(tmp_path, "t-done")
    monkeypatch.setattr(server_maintenance, "DATA_DIR", tmp_path)
    server_maintenance._run_drive_custody_pass()
    row = maintenance_jobs.job(tmp_path, "drive_settlement")
    events = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert row and events and events[-1]["type"] == "headless_task_drive_prune"
    assert events[-1]["report"]["mailboxes"]["removed"] == ["t-done"]
