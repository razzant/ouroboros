"""TZ-1 V10 and A: mail written to a task that its model never read is kept by the
accepted terminal write itself (no ACK, no second writer per terminal path), custody only
grows across both canonical/replica seams, forward_to_worker reaches a queued task with an
honest receipt, and every effective read stays pure."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import artifacts, headless, owner_mailbox, task_custody
from ouroboros.task_results import load_task_result, write_task_result

TASK = "mailtask"


def _msg_ids(value) -> list:
    return [json.loads(row)["msg_id"] for row in (value or {}).get("rows") or []]


@pytest.mark.parametrize("terminal", ["cancelled", "failed", "completed"])
def test_an_accepted_terminal_transition_keeps_every_unread_row_without_acknowledging(tmp_path, terminal):
    data = tmp_path / "data"
    drive = headless.prepare_task_drive(data, TASK, "forked")
    write_task_result(data, TASK, "scheduled", child_drive_root=str(drive))
    owner_mailbox.write_owner_message(drive, "owner words", TASK, msg_id="o1")
    owner_mailbox.write_task_message(drive, "x" * 7000, TASK, source_task_id="parent1", msg_id="t1")
    owner_mailbox.write_owner_message(drive, "hurry", TASK, msg_id="h1", kind=owner_mailbox.KIND_HURRY)
    owner_mailbox.write_owner_message(data, "canonical words", TASK, msg_id="c1")
    owner_mailbox.acknowledge_task_messages(drive, TASK, ["o1"], wake_id="test")  # o1 was read

    stored = write_task_result(data, TASK, terminal, result="ended")

    custody = stored["unread_mailbox"]
    assert sorted(_msg_ids(custody)) == ["c1", "t1"] and custody["read_complete"] is True
    assert json.loads(next(row for row in custody["rows"] if '"t1"' in row))["text"] == "x" * 7000  # full bytes
    assert owner_mailbox.acknowledged_task_message_ids(drive, TASK) == {"o1"}  # capture never ACKs
    # A rejected regression and a same-status enrichment fabricate no new transition.
    owner_mailbox.write_task_message(drive, "too late", TASK, source_task_id="parent1", msg_id="t2")
    write_task_result(data, TASK, "running", result="stale mirror")
    write_task_result(data, TASK, terminal, cost_note="enrichment")
    assert sorted(_msg_ids(load_task_result(data, TASK)["unread_mailbox"])) == ["c1", "t1"]


def test_get_task_result_shows_unread_mail_and_the_exact_rows_ride_the_authority(tmp_path):
    from ouroboros.tools.control_task_results import _get_task_result

    data = tmp_path / "data"
    owner_mailbox.write_task_message(data, "please also check the logs", TASK, source_task_id="parent1", msg_id="t1")
    write_task_result(data, TASK, "cancelled", result="Cancelled before start.", parent_task_id="parent1",
                      root_task_id="parent1", delegation_role="subagent")
    ctx = SimpleNamespace(drive_root=data, task_id="parent1", task_metadata={})

    text = _get_task_result(ctx, TASK)
    assert "[UNREAD_MAILBOX] 1 message(s)" in text and "please also check the logs" in text
    authority = json.loads(_get_task_result(ctx, TASK, include_authority=True))["authority"]
    assert _msg_ids(authority["unread_mailbox"]) == ["t1"]


def test_a_torn_mailbox_read_is_disclosed_and_never_proof_of_emptiness(tmp_path):
    data = tmp_path / "data"
    mailbox = owner_mailbox._mailbox_path(data, TASK)
    mailbox.parent.mkdir(parents=True)
    mailbox.write_text('{"msg_id": "a", "kind": "owner_text", "text": "whole"}\n{"msg_id": "b", "te', encoding="utf-8")

    custody = write_task_result(data, TASK, "failed", result="x")["unread_mailbox"]

    assert custody["read_complete"] is False
    assert not owner_mailbox.cleanup_task_mailbox(data, TASK) and mailbox.is_file()


def test_exact_rows_keep_unicode_line_separators(tmp_path):
    data = tmp_path / "data"
    text = "first second third"
    owner_mailbox.write_owner_message(data, text, TASK, msg_id="u1")
    custody = write_task_result(data, TASK, "cancelled", result="x")["unread_mailbox"]
    assert custody["read_complete"] is True and json.loads(custody["rows"][0])["text"] == text


def test_unread_custody_is_a_union_at_both_replica_seams(tmp_path):
    from ouroboros.post_task_checkpoint import project_replica_task_result_fields
    from ouroboros.task_status import load_effective_task_result

    data = tmp_path / "data"
    drive = headless.prepare_task_drive(data, TASK, "empty")
    owner_mailbox.write_owner_message(drive, "one", TASK, msg_id="m1")
    write_task_result(drive, TASK, "completed", result="child answer")  # the child's own capture
    write_task_result(data, TASK, "running", child_drive_root=str(drive))
    owner_mailbox.write_owner_message(data, "two", TASK, msg_id="m2")
    canonical = write_task_result(data, TASK, "failed", result="orphaned")  # canonical capture: m1 + m2
    assert sorted(_msg_ids(canonical["unread_mailbox"])) == ["m1", "m2"]
    stale = {"status": "completed", "unread_mailbox": {"rows": [], "read_complete": True}}

    # Effective-read seam and copy-back seam: a stale replica never drops a held row.
    assert sorted(_msg_ids(project_replica_task_result_fields(canonical, stale)["unread_mailbox"])) == ["m1", "m2"]
    assert sorted(_msg_ids(load_effective_task_result(data, TASK)["unread_mailbox"])) == ["m1", "m2"]
    copied = headless.copy_child_task_result(data, {"id": TASK, "drive_root": str(drive)})
    assert sorted(_msg_ids(copied["unread_mailbox"])) == ["m1", "m2"]
    write_task_result(data, TASK, "failed", unread_mailbox={"rows": [], "read_complete": True})
    assert sorted(_msg_ids(load_task_result(data, TASK)["unread_mailbox"])) == ["m1", "m2"]


def test_forward_to_a_queued_task_is_read_when_it_starts_and_kept_if_it_never_does(tmp_path):
    import queue as queue_mod

    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.tools.core import _forward_to_worker

    data = tmp_path / "data"
    drive = headless.prepare_task_drive(data, TASK, "forked")
    write_task_result(data, TASK, "scheduled", child_drive_root=str(drive), parent_task_id="parent1",
                      root_task_id="parent1", delegation_role="subagent")
    ctx = SimpleNamespace(drive_root=data, task_id="parent1", task_metadata={})

    receipt = _forward_to_worker(ctx, TASK, "start with the logs")

    assert "(queued)" in receipt and "has not started, so nothing has read it" in receipt
    rows, complete = task_custody.unread_mail_rows(drive, TASK)
    assert complete and json.loads(rows[0])["text"] == "start with the logs"
    msg_id = json.loads(rows[0])["msg_id"]
    assert owner_mailbox.mail_read_state(drive, TASK, msg_id) is False
    # Its first round reads (and acknowledges) it like any running delivery.
    messages: list = []
    _drain_incoming_messages(messages, queue_mod.Queue(), drive, TASK, None, set())
    assert "start with the logs" in json.dumps(messages) and owner_mailbox.mail_read_state(drive, TASK, msg_id) is True


def test_forward_to_a_queued_task_cancelled_before_start_is_held_unread_as_its_exact_row(tmp_path):
    """TZ-2 B5 acceptance: the real ``_forward_to_worker`` writes to a queued task and says
    only that (queued; nothing has read it); the supervisor's pending drop then cancels the
    task before it ever starts; the result reader shows the message as unread mail and the
    authority carries the exact mailbox row. Nothing acknowledged it and nothing says "read"."""
    from ouroboros.tools.control_task_results import _get_task_result
    from ouroboros.tools.core import _forward_to_worker

    data = tmp_path / "data"
    drive = headless.prepare_task_drive(data, TASK, "forked")
    write_task_result(data, TASK, "scheduled", child_drive_root=str(drive), parent_task_id="parent1",
                      root_task_id="parent1", delegation_role="subagent")
    ctx = SimpleNamespace(drive_root=data, task_id="parent1", task_metadata={})

    receipt = _forward_to_worker(ctx, TASK, "start with the logs — then the config")

    assert f"({owner_mailbox.MAIL_QUEUED})" in receipt and "has not started, so nothing has read it" in receipt
    assert "delivered" not in receipt and "next checkpoint" not in receipt
    raw = owner_mailbox._mailbox_path(drive, TASK).read_text(encoding="utf-8")
    assert raw.count("\n") == 1 and raw.endswith("\n")
    row = raw[:-1]
    msg_id = json.loads(row)["msg_id"]

    stored = write_task_result(data, TASK, "cancelled", strict_existing_dict=True, result="Cancelled before start.")

    assert stored["unread_mailbox"]["rows"] == [row] and stored["unread_mailbox"]["read_complete"] is True
    text = _get_task_result(ctx, TASK)
    assert "[UNREAD_MAILBOX] 1 message(s)" in text and "start with the logs — then the config" in text
    authority = json.loads(_get_task_result(ctx, TASK, include_authority=True))["authority"]
    assert authority["status"] == "cancelled" and authority["unread_mailbox"]["rows"] == [row]
    assert owner_mailbox.mail_read_state(drive, TASK, msg_id) is False
    assert owner_mailbox.acknowledged_task_message_ids(drive, TASK) == set()


def test_mail_write_receipt_vocabulary():
    assert owner_mailbox.mail_write_receipt("scheduled")["receipt"] == owner_mailbox.MAIL_QUEUED
    assert owner_mailbox.mail_write_receipt("running")["receipt"] == owner_mailbox.MAIL_DELIVERED
    assert owner_mailbox.mail_write_receipt("running", drain_ended=True)["receipt"] == owner_mailbox.MAIL_RETAINED_UNREAD
    assert all(owner_mailbox.mail_write_receipt(status)["read"] is False for status in ("scheduled", "running"))


def test_canonical_only_late_mail_after_cleanup_is_held_by_the_next_sweep(tmp_path):
    data = tmp_path / "data"
    write_task_result(data, TASK, "completed", result="done",
                      child_ref_promotion={"schema_version": 1, "status": "complete", "pending_refs": []},
                      root_phase_checkpoint={"post_task_synthesis": "completed"})
    assert owner_mailbox.cleanup_task_mailbox(data, TASK)  # nothing there
    owner_mailbox.write_owner_message(data, "after the old settlement", TASK, msg_id="late")
    report = owner_mailbox.sweep_settled_owner_mailboxes(data)
    assert report["removed"] == [TASK]
    assert _msg_ids(load_task_result(data, TASK)["unread_mailbox"]) == ["late"]


def _tree_snapshot(root: Path) -> dict:
    return {str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted(root.rglob("*")) if path.is_file()}


def test_every_observation_surface_is_pure(tmp_path, monkeypatch):
    """No read copies, hashes or registers a file, or writes any byte (the fail-soft
    quarantine of an inadmissible row stays the one owner-accepted exception)."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.tasks import api_task_get, api_tasks_list
    from ouroboros.task_status import find_child_tasks, load_effective_task_result, wait_for_effective_tasks
    from ouroboros.tools.control_task_results import _get_task_result

    data = tmp_path / "data"
    drive = headless.prepare_task_drive(data, TASK, "empty")
    store = artifacts.task_artifact_dir_path(drive, TASK, create=True)
    (store / "reports").mkdir()
    (store / "reports" / "summary.txt").write_text("nested", encoding="utf-8")
    (store / "top.txt").write_text("top", encoding="utf-8")
    write_task_result(drive, TASK, "completed", result="child done",
                      artifacts=artifacts.collect_task_artifact_records(drive, TASK))
    write_task_result(data, TASK, "running", child_drive_root=str(drive), parent_task_id="parent1",
                      root_task_id="parent1", delegation_role="subagent")
    (data / "state" / "queue_snapshot.json").write_text('{"pending": [], "running": []}', encoding="utf-8")
    before = _tree_snapshot(data)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("an observation surface touched artifact bytes or registrations")

    for name in ("stream_artifact_file", "copy_artifact_file", "copy_file_to_task_artifacts",
                 "_register_task_artifact_records", "store_actor_source_bytes"):
        monkeypatch.setattr(artifacts, name, forbidden)
    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_list, methods=["GET"]),
                            Route("/api/tasks/{task_id}", endpoint=api_task_get, methods=["GET"])])
    app.state.drive_root = data
    client = TestClient(app)
    ctx = SimpleNamespace(drive_root=data, task_id="parent1", task_metadata={})

    view = load_effective_task_result(data, TASK)
    assert sorted(row.get("relpath") or row["name"] for row in view["artifacts"]) == ["reports/summary.txt", "top.txt"]
    assert all(row.get("measured") is False or row.get("sha256") for row in view["artifacts"])
    load_effective_task_result(data, TASK, materialize_artifacts=False)
    find_child_tasks(data, parent_task_id="parent1")
    wait_for_effective_tasks(data, [TASK], timeout_sec=0)
    assert client.get(f"/api/tasks/{TASK}").status_code == 200
    assert client.get("/api/tasks").status_code == 200
    _get_task_result(ctx, TASK)
    assert _tree_snapshot(data) == before
    assert not artifacts.task_artifact_dir_path(data, TASK).exists()


def test_the_terminal_capture_reads_the_mailbox_before_the_row_lock(tmp_path, monkeypatch):
    """V10: the mailbox bytes are read outside the task-result row lock; only the bounded
    union runs under it (no file read inside a 4 s lock hold)."""
    from ouroboros import platform_layer, task_results

    data = tmp_path / "data"
    owner_mailbox.write_owner_message(data, "unread words", TASK, msg_id="u1")
    events = []
    real_capture, real_acquire = task_custody.capture_unread_mail, platform_layer.acquire_exclusive_file_lock

    def capture(*args, **kwargs):
        events.append("capture")
        return real_capture(*args, **kwargs)

    def acquire(path, **kwargs):
        if str(path).endswith(f"{TASK}.json.lock"):
            events.append("row_lock")
        return real_acquire(path, **kwargs)

    monkeypatch.setattr(task_results, "capture_unread_mail", capture, raising=False)
    monkeypatch.setattr(task_custody, "capture_unread_mail", capture)
    monkeypatch.setattr(platform_layer, "acquire_exclusive_file_lock", acquire)
    stored = write_task_result(data, TASK, "cancelled", result="x")
    assert _msg_ids(stored["unread_mailbox"]) == ["u1"]
    assert events == ["capture", "row_lock"], events


def test_forward_receipt_says_retained_unread_once_the_recipient_drain_ended(tmp_path, monkeypatch):
    """A message written after the recipient's own drain ended (TZ-2's ``mailbox_drain_ended``
    fact, read at the recipient's drive) is never promised to a next checkpoint: the receipt
    names the unread retention honestly; the row stays for the settlement to keep."""
    from ouroboros.tools.core import _forward_to_worker

    data = tmp_path / "data"
    drive = headless.prepare_task_drive(data, TASK, "forked")
    write_task_result(data, TASK, "running", child_drive_root=str(drive), parent_task_id="parent1",
                      root_task_id="parent1", delegation_role="subagent")
    ctx = SimpleNamespace(drive_root=data, task_id="parent1", task_metadata={})
    asked = []
    monkeypatch.setattr(owner_mailbox, "mailbox_drain_ended", lambda root, tid: asked.append((root, tid)) or True)

    receipt = _forward_to_worker(ctx, TASK, "too late for a checkpoint")

    assert asked == [(drive, TASK)]  # the recipient's own drive, never the sender's
    assert f"({owner_mailbox.MAIL_RETAINED_UNREAD})" in receipt and "next checkpoint" not in receipt
    rows, complete = task_custody.unread_mail_rows(drive, TASK)
    assert complete and json.loads(rows[0])["text"] == "too late for a checkpoint"


def test_the_seam_keeps_a_canonical_mailbox_with_uncarried_inputs_and_the_off_loop_sweep_carries_them(tmp_path):
    """The task-done seam runs on the loop thread and hashes nothing: a settled task's mailbox
    whose unread rows carry attachments waits for the off-loop sweep, which verifies and records
    the canonical closure per exact row (inline and >25-row manifest alike) before unlinking."""
    from ouroboros import artifacts

    data = tmp_path / "data"
    write_task_result(data, TASK, "completed", result="done", root_phase_checkpoint={"post_task_synthesis": "completed"})
    sources = []
    for index in range(30):
        source = tmp_path / f"input-{index}.txt"
        source.write_text(f"input {index}", encoding="utf-8")
        sources.append(str(source))
    manifest = artifacts.stage_task_attachments(data, TASK, sources)
    assert owner_mailbox.write_owner_message(data, "see the files", TASK, msg_id="late", attachment_manifest=manifest)
    row = task_custody.unread_mail_rows(data, TASK)[0][0]
    assert "attachment_manifest_ref" in json.loads(row)

    assert owner_mailbox.cleanup_task_mailbox(data, TASK, carry_inputs=False) is False
    assert owner_mailbox._mailbox_path(data, TASK).is_file()
    assert owner_mailbox.sweep_settled_owner_mailboxes(data)["removed"] == [TASK]
    custody = load_task_result(data, TASK)["unread_mailbox"]
    assert custody["rows"] == [row]
    resolved = artifacts.resolve_attachment_manifest(data, TASK, custody["inputs"][task_custody._row_key(row)])
    assert len(resolved) == 30
    for item in resolved:
        artifacts.stream_artifact_file(Path(item["abs_path"]), expected=item)


def test_the_drive_custody_pass_sweeps_the_canonical_mailboxes_the_seam_left(tmp_path, monkeypatch):
    """The off-loop drive-custody pass owns the mailbox sweep too, so a canonical mailbox the
    loop thread kept (unread inputs to carry) is carried and unlinked within a cadence."""
    from ouroboros import artifacts
    from ouroboros import server_maintenance as sm

    data = tmp_path / "data"
    from ouroboros.startup_migrations import prepare_startup_state
    prepare_startup_state(data)
    monkeypatch.setattr(sm, "DATA_DIR", data)
    monkeypatch.setattr(sm, "_DRIVE_PRUNE_CURSOR", {"headless": "", "direct": ""})
    write_task_result(data, TASK, "completed", result="done", root_phase_checkpoint={"post_task_synthesis": "completed"})
    source = tmp_path / "input.txt"
    source.write_text("input", encoding="utf-8")
    manifest = artifacts.stage_task_attachments(data, TASK, [str(source)])
    assert owner_mailbox.write_owner_message(data, "see the file", TASK, msg_id="late", attachment_manifest=manifest)
    assert owner_mailbox.cleanup_task_mailbox(data, TASK, carry_inputs=False) is False

    sm._run_drive_custody_pass()

    assert not owner_mailbox._mailbox_path(data, TASK).exists()
    custody = load_task_result(data, TASK)["unread_mailbox"]
    assert _msg_ids(custody) == ["late"] and len(custody["inputs"]) == 1
    closed = threading.Event()
    closed.set()
    assert owner_mailbox.write_owner_message(data, "again", TASK, msg_id="again", attachment_manifest=manifest)
    sm._run_drive_custody_pass(closed)
    assert owner_mailbox._mailbox_path(data, TASK).exists(), "a closed generation unlinks nothing"


def _staged(tmp_path: Path, drive: Path, count: int, prefix: str) -> list:
    sources = []
    for index in range(count):
        source = tmp_path / f"{prefix}-{index}.txt"
        source.write_text(f"{prefix} input {index}", encoding="utf-8")
        sources.append(str(source))
    manifest = artifacts.stage_task_attachments(drive, TASK, sources)
    assert len(manifest) == count and all(row["status"] == "staged" for row in manifest)
    return manifest


@pytest.mark.parametrize("count", [2, 30])
def test_an_acknowledged_owner_row_with_line_separators_keeps_its_inputs_through_drive_gc(tmp_path, count):
    """R1: every mailbox reader splits rows on "\\n" only (``owner_mailbox.mailbox_lines``). Owner
    text holding a literal U+2028/U+2029 that was delivered and acknowledged is no unread capture,
    so the background retention carries its follow-up inputs: it reads that row whole (inline
    and >25-row manifest alike) and the canonical store keeps the files after the drive goes."""
    data = tmp_path / "data"
    drive = headless.prepare_task_drive(data, TASK, "empty")
    write_task_result(data, TASK, "running", headless_child_drive_root=str(drive))
    manifest = _staged(tmp_path, drive, count, "follow-up")
    text = "first second third"
    assert owner_mailbox.write_owner_message(drive, text, TASK, msg_id="u1", attachment_manifest=manifest)
    assert [entry["text"] for entry in owner_mailbox.drain_owner_entries(drive, TASK)] == [text]
    assert owner_mailbox.acknowledge_task_messages(drive, TASK, ["u1"], wake_id="test")
    assert task_custody.unread_mail_rows(drive, TASK) == ([], True)  # read: no unread custody holds it
    write_task_result(drive, TASK, "completed", result="done")

    copied = headless.copy_child_task_result(data, {"id": TASK, "drive_root": str(drive)})
    assert copied["child_ref_promotion"]["status"] == "incomplete" and drive.exists()
    copied = headless.retry_child_task_refs(data, drive, TASK)
    assert copied["child_ref_promotion"]["status"] == "complete"
    report = headless.prune_headless_task_drives(data, retention_days=1, now=4_000_000_000, live=lambda _task: False)
    assert [row["task_id"] for row in report["pruned"]] == [TASK] and not drive.exists()
    store = artifacts.task_artifact_dir_path(data, TASK)
    for row in manifest:
        artifacts.stream_artifact_file(store / row["relpath"], expected=row)


@pytest.mark.parametrize("count", [2, 30])
def test_an_unreadable_owner_history_row_fails_input_promotion_closed_until_repaired(tmp_path, count):
    """R1: a torn append followed by the next row leaves ONE unreadable line that swallowed an owner
    row with inputs. Its kind cannot be proven, so the input history read fails closed: the
    promotion keeps a pending ref (retry evidence), the drive stays, and a repair converges."""
    from ouroboros import observability

    data = tmp_path / "data"
    drive = headless.prepare_task_drive(data, TASK, "empty")
    write_task_result(data, TASK, "running", headless_child_drive_root=str(drive))
    mailbox = owner_mailbox._mailbox_path(drive, TASK)
    mailbox.parent.mkdir(parents=True, exist_ok=True)
    mailbox.write_text('{"msg_id": "torn", "kind": "owner_text", "te', encoding="utf-8")  # a crashed append
    manifest = _staged(tmp_path, drive, count, "swallowed")
    assert owner_mailbox.write_owner_message(drive, "see the files", TASK, msg_id="m1", attachment_manifest=manifest)
    assert len(mailbox.read_text(encoding="utf-8").split("\n")) == 2  # one unreadable line, one terminator
    write_task_result(drive, TASK, "completed", result="done")

    copied = headless.copy_child_task_result(data, {"id": TASK, "drive_root": str(drive)})
    assert copied["child_ref_promotion"]["status"] == "incomplete"
    assert any(ref.get("path") == str(mailbox) for ref in copied["child_ref_promotion"]["pending_refs"])
    later = 4_000_000_000
    assert not headless.prune_headless_task_drives(data, retention_days=1, now=later, live=lambda _t: False)["pruned"]
    assert drive.is_dir()

    content = mailbox.read_text(encoding="utf-8")
    mailbox.write_text(content[content.index("{", 1):], encoding="utf-8")  # the torn prefix repaired away
    assert observability.retry_pending_child_ref_promotions(data)["completed"] == [TASK]
    report = headless.prune_headless_task_drives(data, retention_days=1, now=later, live=lambda _task: False)
    assert [row["task_id"] for row in report["pruned"]] == [TASK]
    store = artifacts.task_artifact_dir_path(data, TASK)
    for row in manifest:
        artifacts.stream_artifact_file(store / row["relpath"], expected=row)
    custody = load_task_result(data, TASK)["unread_mailbox"]
    assert _msg_ids(custody) == ["m1"] and len(custody["inputs"]) == 1


@pytest.mark.parametrize("boundary", ["inside_carry", "mail_lock_wait"])
def test_a_generation_closed_inside_a_mailbox_cleanup_writes_and_unlinks_nothing_more(tmp_path, monkeypatch, boundary):
    """R3: the off-loop cleanup (the drive-custody sweep and the ref retry) threads the generation
    into its mutation owners: a close observed inside the input carry places no further copy, and
    one observed while the mail lock was awaited writes no row and unlinks nothing. The mailbox
    and the row stay exactly as retry evidence; the next generation converges."""
    data = tmp_path / "data"
    if boundary == "inside_carry":
        drive = headless.prepare_task_drive(data, TASK, "empty")
        write_task_result(data, TASK, "scheduled", child_drive_root=str(drive))
        manifest = _staged(tmp_path, drive, 30, "carried")
        write_task_result(data, TASK, "cancelled", result="Cancelled before start.")
    else:
        drive = data
        write_task_result(data, TASK, "completed", result="done", root_phase_checkpoint={"post_task_synthesis": "completed"})
        manifest = _staged(tmp_path, data, 2, "canonical")
    assert owner_mailbox.write_owner_message(drive, "see the files", TASK, msg_id="m1", attachment_manifest=manifest)
    mailbox = owner_mailbox._mailbox_path(drive, TASK)
    before = load_task_result(data, TASK)
    closed, written_after = [], []
    if boundary == "inside_carry":
        store = artifacts.task_artifact_dir_path(data, TASK).resolve()
        real_copy = artifacts.copy_artifact_file

        def copy_then_close(source, destination, **kwargs):
            placed = Path(destination).resolve().is_relative_to(store) and Path(source).resolve() != Path(destination).resolve()
            was_closed = bool(closed)
            measured = real_copy(source, destination, **kwargs)
            if placed:
                written_after.extend([str(destination)] if was_closed else [])
                closed.append(1)  # the generation closes right after the first placed copy
            return measured
        monkeypatch.setattr(artifacts, "copy_artifact_file", copy_then_close)
        assert owner_mailbox.cleanup_task_mailbox(drive, TASK, canonical_root=data, stop=lambda: bool(closed)) is False
        assert closed and written_after == [], "a copy was placed after the close"
    else:
        real_lock = task_custody.task_mail_lock

        def lock_then_close(*args, **kwargs):
            closed.append(1)  # the generation closes while this lock is awaited
            return real_lock(*args, **kwargs)
        monkeypatch.setattr(task_custody, "task_mail_lock", lock_then_close)
        assert owner_mailbox.sweep_settled_owner_mailboxes(data, stop=lambda: bool(closed)) == {"removed": [], "kept": 1}
        assert closed
    assert mailbox.is_file() and load_task_result(data, TASK) == before

    monkeypatch.undo()
    assert owner_mailbox.cleanup_task_mailbox(drive, TASK, canonical_root=data) is True
    custody = load_task_result(data, TASK)["unread_mailbox"]
    resolved = artifacts.resolve_attachment_manifest(data, TASK, next(iter(custody["inputs"].values())))
    assert len(resolved) == len(manifest)
    for item in resolved:
        artifacts.stream_artifact_file(Path(item["abs_path"]), expected=item)
