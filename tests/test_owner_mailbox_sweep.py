"""CPL4-C18 pins: startup sweep of mailboxes whose task settled off-path.

A mailbox (and its acks file) goes only when the task's durable result is
SETTLED; no result, a non-terminal result, or an unclassifiable filename
keeps it — an undelivered owner directive must survive any ambiguity.
"""

from __future__ import annotations

import json

from ouroboros.owner_mailbox import (
    _ack_path,
    _mailbox_path,
    sweep_settled_owner_mailboxes,
    write_owner_message,
)
from ouroboros.task_result_schema import stamp_task_result_schema


def _settle(root, task_id, status="completed"):
    results = root / "task_results"
    results.mkdir(parents=True, exist_ok=True)
    row = stamp_task_result_schema({"task_id": task_id, "status": status})
    (results / f"{task_id}.json").write_text(json.dumps(row), encoding="utf-8")


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


def test_acks_file_travels_with_its_mailbox(tmp_path):
    write_owner_message(tmp_path, "hi", "t-done")
    _ack_path(tmp_path, "t-done").write_text("{}\n", encoding="utf-8")
    _settle(tmp_path, "t-done")

    sweep_settled_owner_mailboxes(tmp_path)

    assert not _ack_path(tmp_path, "t-done").exists()


def test_orphan_acks_without_mailbox_left_alone(tmp_path):
    # The .acks.jsonl glob entry is skipped as a mailbox candidate; nothing
    # else in the dir is guessed at.
    _ack_path(tmp_path, "t-ghost").parent.mkdir(parents=True, exist_ok=True)
    _ack_path(tmp_path, "t-ghost").write_text("{}\n", encoding="utf-8")

    report = sweep_settled_owner_mailboxes(tmp_path)

    assert report == {"removed": [], "kept": 0}
    assert _ack_path(tmp_path, "t-ghost").exists()


def test_startup_prune_sweeps_run_the_mailbox_sweep():
    import inspect

    import ouroboros.server_maintenance as sm

    assert "sweep_settled_owner_mailboxes" in inspect.getsource(sm._startup_prune_sweeps)
