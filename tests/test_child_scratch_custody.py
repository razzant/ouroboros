"""A finished child's external scratch follows its parent's exact decision through handoff and Continue."""
from __future__ import annotations

import json
import pathlib
import tempfile

import pytest

from ouroboros import delegate_custody, headless, owner_keep, session_scratch, task_custody, test_environment
from ouroboros.workspace_preflight import SESSION_SCRATCH_DIRNAME, release_finished_scratch, session_service_directories
from tests._shared import retire_synthetic_retention
from tests._store_shared import read_result, seed_result
from tests.test_session_scratch_release import OVER, _hold, _record, _started
from tests.test_session_service_directories import GRANTED, _Catalog, _synthetic_temp_parent  # noqa: F401

LEAD, CHILD, NEXT = "lead-1", "child-1", "lead-2"
PENDING, UNREADABLE = session_scratch.PARENT_DECISION_PENDING, session_scratch.PARENT_UNREADABLE
STALE = "2026-01-01T00:00:00+00:00"


def _child(tmp_path, lead_status="running", **fields):
    """A finished direct child of LEAD whose start's scratch holds an environment; LEAD's row in LEAD_STATUS."""
    data, scratch = _started(tmp_path, task_id=CHILD)
    seed_result(data, LEAD, {"status": lead_status})
    seed_result(data, CHILD, {"status": "completed", "parent_task_id": LEAD, "root_task_id": LEAD,
                              "delegation_role": "subagent", "result": "measured", **fields})
    return data, scratch


def _decide(data, disposition, child=CHILD):
    """LEAD's decision on CHILD's current exact result, through the one ledger authority ``tree_note`` writes."""
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.task_tree_ledger import CHILD_RESULT_DISPOSITION_TYPE, tree_ledger_append
    from ouroboros.tools.join_ledger import _child_result_sha256

    digest = _child_result_sha256(load_effective_task_result(data, child))
    out = tree_ledger_append(LEAD, "decision", f"{disposition} here", task_id=LEAD, payload={
        "type": CHILD_RESULT_DISPOSITION_TYPE, "child_task_id": child, "disposition": disposition,
        "child_result_sha256": digest}, allow_child_result_disposition=True, data_root=data)
    assert not out.startswith("⚠️"), out


def _next(data, status):
    """The successor root the owner's Continue of LEAD admitted, in STATUS."""
    seed_result(data, NEXT, {"status": status, "root_task_id": NEXT, "delegation_role": "root",
                             "metadata": {"continuation": {"predecessor_task_id": LEAD, "admission": {"by": {}}}}})


def _continue_lead(data, status="running"):
    """The owner's Continue of LEAD: its recorded, verifiable claim and the successor root it admitted."""
    from ouroboros.owner_continue import binding_sha

    binding = {"predecessor_task_id": LEAD, "successor_task_id": NEXT, "action_nonce": "press-0001-abcdef"}
    seed_result(data, LEAD, {"status": "cancelled", "continued_by": {
        "by": {}, "successor_task_id": NEXT, "action_nonce": "press-0001-abcdef", "binding": binding,
        "binding_sha256": binding_sha(binding), "state": "admitted"}})
    _next(data, status)


def _queue(data, running=(), ts=None):
    """The queue snapshot the supervisor publishes: the trash pass's probe in the maintenance child."""
    from ouroboros.utils import utc_now_iso
    from tests._shared import write_queue_document

    (data / "state").mkdir(parents=True, exist_ok=True)
    write_queue_document(data / "state" / "queue_snapshot.json", {"ts": ts or utc_now_iso(), "pending": [],
                         "running": [{"id": task, "task": {"id": task}} for task in running]})


def _classes(report):
    return [hold["class"] for hold in report["holds"]]


@pytest.mark.parametrize("disposition", ["integrated", "irrelevant"])
def test_a_finished_childs_scratch_waits_for_its_live_parents_decision(tmp_path, disposition):
    data, scratch = _child(tmp_path)
    assert release_finished_scratch(data, OVER) == {"released": [], "holds": [_hold(data, PENDING, CHILD)],
                                                     "errors": []}
    assert _record(data, CHILD).is_file() and (scratch / "env" / "pyvenv.cfg").is_file()
    assert [(row["task_id"], row["hold"]) for row in session_scratch.scratch_records(data, OVER)] == [(CHILD, PENDING)]
    _decide(data, "deferred")  # a later step or a sibling still reads it by a path the parent handed on
    assert _classes(release_finished_scratch(data, OVER)) == [PENDING]
    _decide(data, disposition)
    assert release_finished_scratch(data, OVER)["released"] == [str(_record(data, CHILD).parent.resolve())]
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not scratch.exists()


def test_only_the_current_exact_result_releases_never_an_older_one_or_a_raw_mirror(tmp_path):
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256, _current_child_result_disposition

    data, scratch = _child(tmp_path)
    _decide(data, "integrated")
    changed = {"status": "completed", "parent_task_id": LEAD, "root_task_id": LEAD, "delegation_role": "subagent",
               "result": "measured again, differently"}
    seed_result(data, CHILD, changed)
    assert session_scratch.parent_hold(data, CHILD, OVER) == PENDING
    digest = _child_result_sha256(load_task_result(data, CHILD, strict=True))  # what a reader of the raw row compares
    seed_result(data, CHILD, {**changed, "child_result_disposition": "integrated",
                              "child_result_disposition_sha256": digest,
                              "child_result_disposition_source": "task_tree_ledger"})
    raw = load_task_result(data, CHILD, strict=True)
    assert _current_child_result_disposition(raw) == "integrated", "the raw mirror would release it if it were read"
    assert session_scratch.parent_hold(data, CHILD, OVER) == PENDING
    assert _classes(release_finished_scratch(data, OVER)) == [PENDING] and scratch.is_dir()
    _decide(data, "integrated")
    assert release_finished_scratch(data, OVER)["released"]


@pytest.mark.parametrize("lead_status, probe, expected", [
    ("completed", False, ""), ("failed", False, ""), ("completed", True, PENDING), ("completed", None, UNREADABLE),
    ("running", False, PENDING),
], ids=["completed-over", "failed-over", "terminal-but-live", "liveness-unknown", "running"])
def test_the_hold_lives_with_the_parent_line_whatever_the_decision(tmp_path, lead_status, probe, expected):
    """Once the parent is proven over nothing waits on the scratch, decided or not; a parent its probe finds live
    holds it, and one whose liveness cannot be proven holds it as unreadable, never presumed over."""
    data, scratch = _child(tmp_path, lead_status=lead_status)
    live = lambda task_id: probe if task_id == LEAD else False  # noqa: E731 -- the child itself is over
    assert session_scratch.parent_hold(data, CHILD, live) == expected
    report = release_finished_scratch(data, live)
    assert _classes(report) == ([expected] if expected else []) and bool(report["released"]) is not bool(expected)
    assert scratch.is_dir()  # the trash pass deletes it, never the release pass


@pytest.mark.parametrize("fault", ["absent", "unreadable"])
def test_a_parent_row_that_cannot_be_read_is_unknown_never_an_end(tmp_path, monkeypatch, fault):
    """A root records no parent and waits for none, as before; a child whose recorded parent's row is absent or
    unreadable waits as ``parent_unreadable``: neither proves the parent over."""
    from tests._store_shared import unreadable_result

    root_data, root_scratch = _started(tmp_path / "root")
    assert session_scratch.parent_hold(root_data, "task-1", OVER) == ""
    data, scratch = _child(tmp_path)
    if fault == "absent":
        seed_result(data, CHILD, {"status": "completed", "parent_task_id": "lead-gone", "root_task_id": "lead-gone",
                                  "delegation_role": "subagent"})
    else:
        unreadable_result(monkeypatch, LEAD, root=data)
    assert session_scratch.parent_hold(data, CHILD, OVER) == UNREADABLE
    assert _classes(release_finished_scratch(data, OVER)) == [UNREADABLE] and scratch.is_dir()


def test_a_continued_parent_line_holds_until_its_successor_is_over(tmp_path):
    data, scratch = _child(tmp_path, lead_status="cancelled")
    _continue_lead(data)
    assert _classes(release_finished_scratch(data, OVER)) == [PENDING] and scratch.is_dir()
    _next(data, "completed")
    assert release_finished_scratch(data, OVER)["released"]


@pytest.mark.parametrize("member", [LEAD, NEXT])
@pytest.mark.parametrize("fact, expected", [
    ("run", PENDING), ("start_request", PENDING), ("unreadable", UNREADABLE),
])
def test_a_terminal_parent_line_holds_child_scratch_until_delegated_custody_closes(tmp_path, monkeypatch, member,
                                                                                    fact, expected):
    from ouroboros import delegate_custody_current

    data, scratch = _child(tmp_path, lead_status="failed")
    if member == NEXT:
        _continue_lead(data, status="failed")
    original = delegate_custody_current.task_snapshot

    def unreadable(root, task_id, **kwargs):
        if task_id == member:
            raise OSError("synthetic parent custody read failure")
        return original(root, task_id, **kwargs)

    if fact == "run":
        assert delegate_custody.emit(data, delegate_custody.STARTED, {"task_id": member, "run_id": "run-parent"})
    elif fact == "start_request":
        assert delegate_custody.emit(data, delegate_custody.START_REQUESTED, {
            "task_id": member, "invocation_id": "inv-parent", "request": {"prompt": "synthetic"}})
    else:
        monkeypatch.setattr(delegate_custody_current, "task_snapshot", unreadable)
    report = release_finished_scratch(data, OVER)
    assert report == {"released": [], "holds": [_hold(data, expected, CHILD)], "errors": []}
    assert _record(data, CHILD).is_file() and (scratch / "env" / "pyvenv.cfg").is_file()

    if fact == "run":
        assert delegate_custody.emit(data, delegate_custody.SETTLED, {
            "task_id": member, "run_id": "run-parent", "state": "succeeded"})
    elif fact == "start_request":
        assert delegate_custody.emit(data, delegate_custody.START_FAILED, {
            "task_id": member, "invocation_id": "inv-parent", "definite": True, "error": "refused"})
    else:
        monkeypatch.setattr(delegate_custody_current, "task_snapshot", original)
    assert release_finished_scratch(data, OVER)["released"]
    _queue(data)
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not scratch.exists()


def test_a_record_in_the_trash_waits_again_for_a_late_continue_of_its_parent(tmp_path):
    """The record was handed on while the parent line was over; then the owner continued the parent. The trash pass
    in the maintenance child asks the same question with the published queue's probe: a successor still working, one
    that wrote its terminal row while its process runs, or a queue that proves nothing keeps the record in its own
    entry; the line proven over lets it go."""
    data, scratch = _child(tmp_path, lead_status="cancelled")
    assert release_finished_scratch(data, OVER)["released"]
    _continue_lead(data)
    _queue(data)  # nothing running: the parent itself is proven over; its successor's row is not final
    report = task_custody.remove_custody_trash(data)
    assert (report["removed"], report["stuck"], _classes(report)) == (0, {}, [PENDING])
    [entry] = (data / "state" / "custody_trash").iterdir()
    assert entry.name.startswith(f"{CHILD}+scratch-") and entry.name in report["held"]
    _next(data, "completed")
    _queue(data, running=[NEXT])
    assert _classes(task_custody.remove_custody_trash(data)) == [PENDING]  # its terminal row written, its process runs
    row = next(row for row in session_scratch.scratch_records(data, OVER) if row["holder"] == "trash")
    assert (row["task_id"], row["hold"]) == (CHILD, PENDING)  # Storage names what the trash pass would
    _queue(data, ts=STALE)
    assert _classes(task_custody.remove_custody_trash(data)) == [UNREADABLE]  # a stale queue proves nothing
    (data / "state" / "queue_snapshot.json").unlink()
    assert _classes(task_custody.remove_custody_trash(data)) == [UNREADABLE]  # nor does a missing one
    assert (scratch / "env" / "pyvenv.cfg").is_file()
    _queue(data)
    assert task_custody.remove_custody_trash(data) == {"removed": 1, "stuck": {}, "holds": []}
    assert not scratch.exists()


def test_an_accepted_result_releases_a_reopened_lines_trash_record(tmp_path):
    data, scratch = _child(tmp_path, lead_status="cancelled")
    assert release_finished_scratch(data, OVER)["released"]
    _continue_lead(data)
    _decide(data, "irrelevant")
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not scratch.exists()


def test_exact_child_disposition_releases_despite_parent_run_custody(tmp_path):
    data, scratch = _child(tmp_path, lead_status="failed")
    assert delegate_custody.emit(data, delegate_custody.STARTED, {"task_id": LEAD, "run_id": "run-parent"})
    _decide(data, "integrated")
    assert release_finished_scratch(data, OVER)["released"]
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not scratch.exists()


@pytest.mark.parametrize("release", ["integrated", "explicit_cancel", "lead_over"])
def test_an_unaccepted_cancelled_childs_drive_settles_while_the_trash_keeps_its_scratch(tmp_path, release):
    """A child cancelled without its parent's word is not accepted. Its drive settles (its own bytes are custodied),
    which never deletes the scratch: the trash pass keeps it in its own entry until the parent's exact-result
    decision, its explicit cancel or its proven end."""
    from types import SimpleNamespace

    from ouroboros import artifacts
    from ouroboros.task_results import write_task_result

    data = tmp_path / "data"
    drive = headless.prepare_task_drive(data, CHILD, "empty")
    target = tmp_path / "target"
    target.mkdir()
    made = session_service_directories(_Catalog(GRANTED), "route", access="full", mode="agent", data_root=data,
                                       drive_root=drive, task_id=CHILD, key="in-place-inv", execution_root=target)
    scratch = pathlib.Path(made.scratch)
    (scratch / "snapshot.sqlite").write_bytes(b"frozen for one measurement")
    source = tmp_path / "report.txt"
    source.write_text("method, source and sha256 of the snapshot", encoding="utf-8")
    capture = artifacts.copy_file_to_task_artifacts(SimpleNamespace(drive_root=drive, task_id=CHILD), source,
                                                    immutable=True)
    write_task_result(drive, CHILD, "running", artifacts=[capture])
    write_task_result(data, LEAD, "running")
    write_task_result(data, CHILD, "cancelled", result="Cancelled.", child_drive_root=str(drive),
                      parent_task_id=LEAD, root_task_id=LEAD, delegation_role="subagent")
    assert _classes(release_finished_scratch(data, OVER)) == [PENDING]
    assert task_custody.settle_child_drive(data, CHILD, drive, live=OVER)["status"] == "removed"
    report = task_custody.remove_custody_trash(data)
    assert (report["removed"], _classes(report)) == (1, [PENDING]) and (scratch / "snapshot.sqlite").is_file()
    [entry] = (data / "state" / "custody_trash").iterdir()
    assert entry.name.startswith(f"{CHILD}+scratch-")
    if release == "integrated":
        _decide(data, "integrated")
    elif release == "explicit_cancel":
        write_task_result(data, CHILD, "cancelled", parent_decision="cancelled")
    else:
        write_task_result(data, LEAD, "completed")
        _queue(data)
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not scratch.exists()


@pytest.mark.parametrize("child", [CHILD, "data"])
def test_a_nested_layouts_record_is_held_for_its_own_task_and_handed_on_under_its_id(tmp_path, child):
    """A moved drive may carry a deeper child's layout: its record is judged by that task's parent line, and its
    handoff entry carries that task, so the next pass asks the same question again."""
    data = tmp_path / "data"
    seed_result(data, LEAD, {"status": "running"})
    seed_result(data, child, {"status": "completed", "parent_task_id": LEAD, "root_task_id": LEAD})
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="obs-", dir=test_environment.session_scratch_parent()))
    (scratch / "frozen.sqlite").write_bytes(b"snapshot")
    seen = scratch.lstat()
    entry = data / "state" / "custody_trash" / "outer-1-0123456789ab"
    owner = entry.joinpath("data", "state", "headless_tasks", child, "data", "task_drives", child,
                           SESSION_SCRATCH_DIRNAME, "k")
    owner.mkdir(parents=True)
    (owner / "scratch.json").write_text(json.dumps({"schema_version": 1, "path": str(scratch),
                                                   "identity": [seen.st_dev, seen.st_ino]}), encoding="utf-8")
    report = task_custody.remove_custody_trash(data)
    assert (report["removed"], _classes(report)) == (1, [PENDING]) and scratch.is_dir()
    [handoff] = (data / "state" / "custody_trash").iterdir()
    assert handoff.name.startswith(f"{child}+scratch-"), "its own task, not the outer drive's"
    assert task_custody.remove_custody_trash(data)["holds"] == report["holds"]
    _decide(data, "integrated", child=child)
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not scratch.exists()


def test_a_nested_record_another_hold_kept_first_still_waits_for_its_own_parent(tmp_path):
    """A producer's marker holds the deeper child's record first: its handoff still carries the child, so once
    the marker goes the next pass asks the child's parent line, never the outer drive's task, and only the
    parent's exact acceptance lets the scratch go."""
    data = tmp_path / "data"
    seed_result(data, LEAD, {"status": "running"})
    seed_result(data, CHILD, {"status": "completed", "parent_task_id": LEAD, "root_task_id": LEAD})
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="obs-", dir=test_environment.session_scratch_parent()))
    (scratch / "frozen.sqlite").write_bytes(b"snapshot")
    reason = "debug retention"
    assert test_environment.retain_tree(scratch, reason)
    seen = scratch.lstat()
    owner = data.joinpath("state", "custody_trash", "outer-1-0123456789ab", "data", "state", "headless_tasks", CHILD,
                          "data", "task_drives", CHILD, SESSION_SCRATCH_DIRNAME, "k")
    owner.mkdir(parents=True)
    (owner / "scratch.json").write_text(json.dumps({"schema_version": 1, "path": str(scratch),
                                                   "identity": [seen.st_dev, seen.st_ino]}), encoding="utf-8")
    report = task_custody.remove_custody_trash(data)
    assert (report["removed"], _classes(report)) == (1, ["producer_retained"])
    [handoff] = (data / "state" / "custody_trash").iterdir()
    assert handoff.name.startswith(f"{CHILD}+scratch-")
    assert retire_synthetic_retention(scratch, reason)
    assert _classes(task_custody.remove_custody_trash(data)) == [PENDING] and (scratch / "frozen.sqlite").is_file()
    _decide(data, "integrated")
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not scratch.exists()


def test_a_seat_record_follows_its_seat_whatever_its_tasks_parent_does(tmp_path):
    data = tmp_path / "data"
    seed_result(data, LEAD, {"status": "running"})
    seed_result(data, CHILD, {"status": "completed", "parent_task_id": LEAD, "root_task_id": LEAD})
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="obs-", dir=test_environment.session_scratch_parent()))
    seen = scratch.lstat()
    owner = data / "state" / "custody_trash" / f"{CHILD}+scratch-0123456789ab" / SESSION_SCRATCH_DIRNAME / "seat-home-1"
    owner.mkdir(parents=True)
    (owner / "scratch.json").write_text(json.dumps({"schema_version": 1, "path": str(scratch),
                                                   "identity": [seen.st_dev, seen.st_ino]}), encoding="utf-8")
    assert session_scratch.parent_hold(data, CHILD, OVER) == PENDING
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not scratch.exists()


def test_the_owners_keep_still_holds_a_decided_childs_scratch(tmp_path):
    data, scratch = _child(tmp_path)
    assert owner_keep.set_owner_keep(data, CHILD, keep=True, source={"basis": "test"})["keep"] is True
    assert _classes(release_finished_scratch(data, OVER)) == [PENDING]
    _decide(data, "integrated")
    assert _classes(release_finished_scratch(data, OVER)) == ["owner_keep"] and scratch.is_dir()
    assert owner_keep.set_owner_keep(data, CHILD, keep=False, source={"basis": "test"})["keep"] is False
    assert release_finished_scratch(data, OVER)["released"]


@pytest.mark.parametrize("retain", ["report", "exact", "producer_hold"])
def test_measurement_recipe_survives_scratch_release(tmp_path, retain):
    import hashlib

    from ouroboros import artifacts

    data, scratch = _child(tmp_path)
    snapshot = scratch / "sample.bin"
    sample = b"synthetic measurement input"
    snapshot.write_bytes(sample)
    saved = artifacts.task_artifact_dir_path(data, CHILD, create=True)
    digest = hashlib.sha256(sample).hexdigest()
    report = f"source=synthetic-sample sha256={digest} result={len(sample)} bytes"
    script = "from pathlib import Path\nprint(len(Path('sample.bin').read_bytes()))\n"
    (saved / "measurement.txt").write_text(report)
    (saved / "measure.py").write_text(script)
    if retain == "exact":
        artifacts.copy_artifact_file(snapshot, saved / snapshot.name)
    if retain == "producer_hold":
        assert test_environment.retain_tree(scratch, "open investigation")

    _decide(data, "integrated")
    assert release_finished_scratch(data, OVER)["released"]
    trash = task_custody.remove_custody_trash(data)
    if retain == "producer_hold":
        assert _classes(trash) == ["producer_retained"] and snapshot.read_bytes() == sample
        assert retire_synthetic_retention(scratch, "open investigation")
        assert task_custody.remove_custody_trash(data)["removed"] == 1 and not scratch.exists()
    else:
        assert trash["removed"] == 1 and not scratch.exists()
    assert (saved / "measurement.txt").read_text() == report
    assert (saved / "measure.py").read_text() == script
    if retain == "exact":
        assert (saved / snapshot.name).read_bytes() == sample
    else:
        assert not (saved / snapshot.name).exists()


def _claim_lead(data):
    """Write the real Continue claim (the admission effect that must share the release interlock)."""
    from ouroboros.owner_continue import claim_on_predecessor

    return claim_on_predecessor(data, LEAD, "press-race-0001", {
        "predecessor_task_id": LEAD, "successor_task_id": NEXT, "action_nonce": "press-race-0001"})


def _handed_child(tmp_path):
    data, scratch = _child(tmp_path, lead_status="cancelled")
    seed_result(data, LEAD, {"status": "cancelled", "root_task_id": LEAD,
                             "cancel_origin": {"source": "snapshot_restore", "reason": "server_shutdown"}})
    assert release_finished_scratch(data, OVER)["released"]
    _queue(data)
    [entry] = (data / "state" / "custody_trash").iterdir()
    return data, scratch, entry


def test_trash_acceptance_rechecks_parent_run_custody_without_waiting(tmp_path, monkeypatch):
    from ouroboros import delegate_custody_current

    data, scratch, entry = _handed_child(tmp_path)
    original = delegate_custody_current.task_snapshot
    calls = []

    def interleave(root, task_id, *, wait_sec):
        assert task_id == LEAD
        calls.append(wait_sec)
        held = original(root, task_id, wait_sec=wait_sec)
        if len(calls) == 1:
            # Give the preliminary proof stale facts; acceptance must read the store again.
            assert delegate_custody.emit(data, delegate_custody.STARTED, {"task_id": LEAD, "run_id": "run-parent"})
        return held

    with monkeypatch.context() as patch:
        patch.setattr(delegate_custody_current, "task_snapshot", interleave)
        report = task_custody.remove_custody_trash(data)
        assert report["removed"] == 0 and _classes(report) == [PENDING]
        assert len(calls) == 2 and calls[1] == 0
    record = entry / SESSION_SCRATCH_DIRNAME / "snap-1" / "scratch.json"
    assert not json.loads(record.read_text()).get(session_scratch.RELEASE_ACCEPTED)
    assert (scratch / "env" / "pyvenv.cfg").is_file()
    assert delegate_custody.emit(data, delegate_custody.SETTLED, {
        "task_id": LEAD, "run_id": "run-parent", "state": "succeeded"})
    _queue(data)
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not scratch.exists()


@pytest.mark.parametrize("admitted", [False, True], ids=["claim-before-successor-row", "successor-admitted"])
def test_continue_claim_after_chain_read_preserves_the_scratch(tmp_path, monkeypatch, admitted):
    from ouroboros import review_thread_retention

    data, scratch, entry = _handed_child(tmp_path)
    original, reads = review_thread_retention._continued_roots, []

    def interleave(root, task):
        result = original(root, task)
        reads.append(task)
        if len(reads) == 1:
            assert _claim_lead(data)[1] is True
            if admitted:
                _next(data, "running")
        return result  # deliberately the old empty chain

    monkeypatch.setattr(review_thread_retention, "_continued_roots", interleave)
    report = task_custody.remove_custody_trash(data)
    assert report["removed"] == 0 and _classes(report) == [PENDING if admitted else UNREADABLE]
    record = entry / SESSION_SCRATCH_DIRNAME / "snap-1" / "scratch.json"
    assert not json.loads(record.read_text()).get(session_scratch.RELEASE_ACCEPTED)
    assert (scratch / "env" / "pyvenv.cfg").read_text() == "home = /usr\n"


def test_continue_claim_cannot_enter_after_the_final_chain_read(tmp_path, monkeypatch):
    from ouroboros import force_remove, review_thread_retention, subagent_worktrees

    data, scratch, entry = _handed_child(tmp_path)
    original, lock = review_thread_retention._continued_roots, task_custody.task_custody_lock
    reads, blocked, removed = [], [], []
    prove = subagent_worktrees._task_proven_over

    def prove_off_lock(task, root, live):
        with lock(root, task, timeout_sec=0) as held:
            assert held  # process census must not run inside the deletion/Continue interlock
        return prove(task, root, live)

    monkeypatch.setattr(subagent_worktrees, "_task_proven_over", prove_off_lock)

    def interleave(root, task):
        result = original(root, task)
        reads.append(task)
        if len(reads) == 2:  # final read under the predecessor's custody lock
            with pytest.raises(ValueError, match="continuation_custody_busy"):
                _claim_lead(data)
            blocked.append(True)
        return result

    remove = force_remove.force_rmtree

    def remove_off_lock(path):
        if pathlib.Path(path) == scratch.parent:
            record = entry / SESSION_SCRATCH_DIRNAME / "snap-1" / "scratch.json"
            assert json.loads(record.read_text())[session_scratch.RELEASE_ACCEPTED]
            # Physical deletion holds neither lock, and a later Continue is allowed.
            assert _claim_lead(data)[1] is True
            with lock(data, CHILD, timeout_sec=0) as held:
                assert held
            removed.append(True)
        return remove(path)

    monkeypatch.setattr(review_thread_retention, "_continued_roots", interleave)
    monkeypatch.setattr(force_remove, "force_rmtree", remove_off_lock)
    assert task_custody.remove_custody_trash(data)["removed"] == 1
    assert blocked == removed == [True] and not scratch.exists()


@pytest.mark.parametrize("release", ["integrated", "line_over"])
def test_keep_after_handoff_holds_until_the_owner_releases_it(tmp_path, release):
    data, scratch, entry = _handed_child(tmp_path)
    _continue_lead(data)
    container = str(scratch.parent)
    answer = owner_keep.set_owner_keep(data, CHILD, keep=True, source={"basis": "test"})
    assert container in answer["retained"] and container not in answer["retired"]
    if release == "integrated":
        _decide(data, "integrated")
    else:
        _next(data, "completed")
    assert _classes(task_custody.remove_custody_trash(data)) == ["owner_keep"]
    row = next(row for row in session_scratch.scratch_records(data, OVER) if row["path"] == container)
    assert row["hold"] == "owner_keep" and (scratch / "env" / "pyvenv.cfg").is_file()
    assert owner_keep.set_owner_keep(data, CHILD, keep=False, source={"basis": "test"})["status"] == "released"
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not entry.exists()


def test_keep_refreshes_the_child_record_moved_after_inventory(tmp_path, monkeypatch):
    data, scratch = _child(tmp_path, lead_status="cancelled")
    original, moved = owner_keep.inventory, []

    def interleave(*args):
        before = original(*args)
        assert release_finished_scratch(data, OVER)["released"]
        moved.append(True)
        return before

    monkeypatch.setattr(owner_keep, "inventory", interleave)
    answer = owner_keep.set_owner_keep(data, CHILD, keep=True, source={"basis": "test"})
    assert moved == [True] and str(scratch.parent) in answer["retained"]
    _queue(data)
    assert _classes(task_custody.remove_custody_trash(data)) == ["owner_keep"]


def test_keep_wins_between_parent_proof_and_release_acceptance(tmp_path, monkeypatch):
    data, scratch, entry = _handed_child(tmp_path)
    original, answers = session_scratch.parent_hold, []

    def interleave(*args, **kwargs):
        result = original(*args, **kwargs)
        if not answers:
            answers.append(owner_keep.set_owner_keep(data, CHILD, keep=True, source={"basis": "test"}))
        return result

    monkeypatch.setattr(session_scratch, "parent_hold", interleave)
    assert _classes(task_custody.remove_custody_trash(data)) == ["owner_keep"]
    assert str(scratch.parent) in answers[0]["retained"] and entry.is_dir()


def test_accepted_child_removal_retries_without_claiming_keep_can_rescue_it(tmp_path, monkeypatch):
    from ouroboros import force_remove

    data, scratch, entry = _handed_child(tmp_path)
    original, answers = force_remove.force_rmtree, []

    def failed(path):
        if pathlib.Path(path) == scratch.parent:
            answers.append(owner_keep.set_owner_keep(data, CHILD, keep=True, source={"basis": "test"}))
            raise PermissionError("synthetic interrupted removal")
        return original(path)

    monkeypatch.setattr(force_remove, "force_rmtree", failed)
    assert _classes(task_custody.remove_custody_trash(data)) == ["removal_failed"]
    assert str(scratch.parent) in answers[0]["retired"] and str(scratch.parent) not in answers[0]["retained"]
    assert scratch.is_dir() and entry.is_dir()
    _claim_lead(data)  # a claim after acceptance cannot undo it, including after a remover restart
    _next(data, "running")
    monkeypatch.setattr(force_remove, "force_rmtree", original)
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not scratch.exists()


@pytest.mark.parametrize("layout", ["moved_drive", "nested_drive", "staging"])
def test_storage_and_keep_find_child_scratch_before_its_separate_handoff(tmp_path, layout):
    data, scratch = _child(tmp_path)
    container = str(scratch.parent)
    source = _record(data, CHILD).parent
    area = "custody_staging" if layout == "staging" else "custody_trash"
    outer = LEAD if layout == "nested_drive" else CHILD
    entry = data / "state" / area / f"{outer}-0123456789ab"
    holder = entry if layout != "nested_drive" else entry / "data" / "task_drives" / CHILD
    target = holder / SESSION_SCRATCH_DIRNAME / source.name
    target.parent.mkdir(parents=True)
    source.rename(target)
    _queue(data)
    row = next(row for row in session_scratch.scratch_records(data, OVER) if row["path"] == container)
    assert row["hold"] == PENDING and row["task_id"] == CHILD
    if layout == "nested_drive":
        assert container not in owner_keep.set_owner_keep(
            data, LEAD, keep=True, source={"basis": "test"})["retained"]
    answer = owner_keep.set_owner_keep(data, CHILD, keep=True, source={"basis": "test"})
    assert container in answer["retained"]
    _decide(data, "integrated")
    if layout == "staging":
        trash = data / "state" / "custody_trash"
        trash.mkdir()
        entry.rename(trash / entry.name)
    report = task_custody.remove_custody_trash(data)
    assert _classes(report) == ["owner_keep"] and scratch.is_dir()
    [handoff] = (data / "state" / "custody_trash").iterdir()
    assert handoff.name.startswith(f"{CHILD}+scratch-")
    if layout == "nested_drive":
        owner_keep.set_owner_keep(data, CHILD, keep=False, source={"basis": "test"})
        assert task_custody.remove_custody_trash(data)["removed"] == 1
        assert owner_keep.is_kept(data, LEAD) and not scratch.exists()


def test_a_trash_pass_reads_one_queue_and_one_shared_parent_view(tmp_path, monkeypatch):
    from ouroboros import task_status
    from ouroboros.workspace_preflight import _retire_record

    data, scratch = _child(tmp_path)
    children = [CHILD, "child-2", "child-3"]
    for child in children[1:]:
        _started(tmp_path, task_id=child)
        seed_result(data, child, {"status": "completed", "parent_task_id": LEAD, "root_task_id": LEAD})
    for child in children:
        assert _retire_record(data, child, _record(data, child).parent)
    _queue(data, running=[LEAD, *[f"unrelated-{i}" for i in range(200)]])
    original_queue, original_member = task_status._load_queue_snapshot, session_scratch._line_member_open
    calls = {"queue": 0, "parent": 0}

    def queue(*args, **kwargs):
        calls["queue"] += 1
        return original_queue(*args, **kwargs)

    def member(*args, **kwargs):
        calls["parent"] += 1
        return original_member(*args, **kwargs)

    monkeypatch.setattr(task_status, "_load_queue_snapshot", queue)
    monkeypatch.setattr(session_scratch, "_line_member_open", member)
    assert _classes(task_custody.remove_custody_trash(data)) == [PENDING]
    assert calls == {"queue": 1, "parent": 1} and scratch.is_dir()


def test_child_release_requires_a_durable_acceptance_before_removal(tmp_path, monkeypatch):
    from ouroboros import utils

    data, scratch, entry = _handed_child(tmp_path)
    original = utils.atomic_write_json

    def fail_acceptance(path, value, **kwargs):
        if isinstance(value, dict) and session_scratch.RELEASE_ACCEPTED in value:
            raise PermissionError("synthetic acceptance write failure")
        return original(path, value, **kwargs)

    monkeypatch.setattr(utils, "atomic_write_json", fail_acceptance)
    assert _classes(task_custody.remove_custody_trash(data)) == [UNREADABLE]
    assert scratch.is_dir() and entry.is_dir()
    assert str(scratch.parent) in owner_keep.set_owner_keep(
        data, CHILD, keep=True, source={"basis": "test"})["retained"]


def test_unreadable_queue_holds_a_child_without_aborting_the_trash_pass(tmp_path, monkeypatch):
    from ouroboros import task_status

    data, scratch, entry = _handed_child(tmp_path)

    def unavailable(*args):
        raise OSError("synthetic queue read failure")

    monkeypatch.setattr(task_status, "_load_queue_snapshot", unavailable)
    assert _classes(task_custody.remove_custody_trash(data)) == [UNREADABLE]
    assert scratch.is_dir() and entry.is_dir()


# ---- the disposition writer's interlock, the full decision source, durable acceptance ----
# ---- and the Continue admission interlock (the composed A+B seam) ------------------------

RETRY, PRED = "retry-1", "pred-1"


def _tree_note_decide(data, disposition, child=CHILD, parent=LEAD):
    """The parent's decision through the REAL tree_note writer the mind calls (registry and all)."""
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    digest = _child_result_sha256(load_effective_task_result(data, child))
    repo = pathlib.Path(__file__).resolve().parents[1]
    workspace = data.parent / "copy"  # the distinct workspace made by _started
    assert workspace.is_dir()
    ctx = ToolContext(repo_dir=repo, drive_root=data, task_id=parent, budget_drive_root=str(data),
                      workspace_root=workspace, workspace_mode="external", task_metadata={"root_task_id": parent})
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    registry.set_context(ctx)
    return registry.execute_result("tree_note", {
        "kind": "decision", "text": f"{disposition} through the real writer",
        "payload": {"type": "child_result_disposition", "child_task_id": child,
                    "disposition": disposition, "child_result_sha256": digest}}).text


def test_a_deferred_recorded_by_tree_note_first_holds_the_scratch(tmp_path, monkeypatch):
    data, scratch, entry = _handed_child(tmp_path)
    _continue_lead(data, "running")  # the parent line lives; nobody decided yet
    asked, original = [], session_scratch.child_parent

    def interleave(root, task):
        why = original(root, task)
        if not asked:  # before any lock: the REAL writer records the parent's deferred
            asked.append(_tree_note_decide(data, "deferred", child=CHILD))
        return why

    monkeypatch.setattr(session_scratch, "child_parent", interleave)
    report = task_custody.remove_custody_trash(data)
    assert report["removed"] == 0 and _classes(report) == [PENDING]
    assert asked[0].startswith("OK:") and (scratch / "env" / "pyvenv.cfg").is_file()


def test_tree_note_deferred_cannot_enter_the_release_acceptance_window(tmp_path, monkeypatch):
    from ouroboros import utils

    data, scratch, entry = _handed_child(tmp_path)
    original, seen = utils.atomic_write_json, []

    def at_the_bit(path, value, **kwargs):
        if isinstance(value, dict) and session_scratch.RELEASE_ACCEPTED in value:
            seen.append(_tree_note_decide(data, "deferred", child=CHILD))  # inside the window
        return original(path, value, **kwargs)

    monkeypatch.setattr(utils, "atomic_write_json", at_the_bit)
    report = task_custody.remove_custody_trash(data)
    assert report["removed"] == 1 and not scratch.exists()  # an accepted deletion is irreversible
    assert seen and seen[0].startswith("⚠️ CHILD_RESULT_DISPOSITION_BUSY")
    assert _tree_note_decide(data, "deferred", child=CHILD).startswith("OK:")  # after: records fine
    assert not scratch.exists()  # a late decision never resurrects the released scratch


def test_the_decision_check_reads_the_full_result_behind_a_retry_transition(tmp_path):
    from ouroboros.task_reads import TaskReads

    data, scratch = _child(tmp_path)  # LEAD's row runs: the line lives
    seed_result(data, CHILD, {"status": "completed", "parent_task_id": LEAD, "root_task_id": LEAD,
                             "delegation_role": "subagent", "result": "first", "superseded_by": RETRY})
    seed_result(data, RETRY, {"status": "completed", "parent_task_id": LEAD, "root_task_id": LEAD,
                              "result": "measured-retry"})
    _decide(data, "integrated", child=RETRY)
    # A shared display view of the retry drops result/trace_summary and hashes a different
    # document: this must not hold an already-decided child (false parent_decision_pending).
    assert session_scratch.parent_hold(data, CHILD, OVER, reads=TaskReads(data)) == ""
    assert session_scratch.parent_hold(data, CHILD, OVER) == ""  # the fresh full read agrees


def _handed_retry_child(tmp_path):
    """CHILD's row superseded by the physical retry RETRY; the releasing decision sits on RETRY's own row."""
    data, scratch = _started(tmp_path, task_id=CHILD)
    seed_result(data, LEAD, {"status": "cancelled", "root_task_id": LEAD,
                             "cancel_origin": {"source": "snapshot_restore", "reason": "server_shutdown"}})
    seed_result(data, CHILD, {"status": "completed", "parent_task_id": LEAD, "root_task_id": LEAD,
                              "delegation_role": "subagent", "result": "first", "superseded_by": RETRY})
    seed_result(data, RETRY, {"status": "completed", "parent_task_id": LEAD, "root_task_id": LEAD,
                              "delegation_role": "subagent", "result": "measured-retry"})
    _decide(data, "integrated", child=RETRY)
    assert release_finished_scratch(data, OVER)["released"]
    _queue(data)
    [entry] = (data / "state" / "custody_trash").iterdir()
    return data, scratch, entry


def test_a_tree_note_on_the_retry_successor_gets_busy_inside_the_acceptance_window(tmp_path, monkeypatch):
    """Retry variant of the disposition-writer race: the decision is recorded on the PHYSICAL
    retry result R, so acceptance must hold R's custody lock too — a real tree_note for R inside
    the window is refused busy and records nothing; the accepted deletion stands."""
    from ouroboros import utils

    data, scratch, entry = _handed_retry_child(tmp_path)
    original, seen = utils.atomic_write_json, []

    def at_the_bit(path, value, **kwargs):
        if isinstance(value, dict) and session_scratch.RELEASE_ACCEPTED in value:
            seen.append(_tree_note_decide(data, "deferred", child=RETRY, parent=LEAD))  # inside the window
        return original(path, value, **kwargs)

    monkeypatch.setattr(utils, "atomic_write_json", at_the_bit)
    report = task_custody.remove_custody_trash(data)
    assert report["removed"] == 1 and not scratch.exists()  # an accepted deletion is irreversible
    assert seen and seen[0].startswith("⚠️ CHILD_RESULT_DISPOSITION_BUSY")  # R's lock was held: nothing recorded
    assert _tree_note_decide(data, "deferred", child=RETRY, parent=LEAD).startswith("OK:")  # after: records fine
    assert not scratch.exists()  # a late decision never resurrects the released scratch


def test_a_changed_resolving_identity_holds_the_acceptance_pass(tmp_path, monkeypatch):
    """Under the custody locks, if the deciding result's physical id changed (C now supersedes to
    R2), acceptance holds the pass instead of writing against the stale identity; the next pass
    locks the new id and completes."""
    data, scratch, entry = _handed_retry_child(tmp_path)
    retry2 = "retry-2"
    real_lock, reseeded = task_custody.task_custody_lock, {"done": False}

    def reseeding_lock(root, task, **kw):
        if task == RETRY and not reseeded["done"]:
            reseeded["done"] = True
            seed_result(data, retry2, {"status": "completed", "parent_task_id": LEAD, "root_task_id": LEAD,
                                       "delegation_role": "subagent", "result": "second-retry"})
            seed_result(data, CHILD, {"status": "completed", "parent_task_id": LEAD, "root_task_id": LEAD,
                                      "delegation_role": "subagent", "result": "first", "superseded_by": retry2})
            _decide(data, "integrated", child=retry2)
        return real_lock(root, task, timeout_sec=0)

    monkeypatch.setattr(task_custody, "task_custody_lock", reseeding_lock)
    first = task_custody.remove_custody_trash(data)
    assert first["removed"] == 0 and _classes(first) == [PENDING]  # the identity changed under the lock
    record = entry / SESSION_SCRATCH_DIRNAME / "snap-1" / "scratch.json"
    assert not json.loads(record.read_text()).get(session_scratch.RELEASE_ACCEPTED)
    assert scratch.is_dir() and (scratch / "env" / "pyvenv.cfg").is_file()
    second = task_custody.remove_custody_trash(data)
    assert second["removed"] == 1 and not scratch.exists()  # the new resolving id locks and completes


def test_the_decision_check_reads_the_full_replica_before_copyback(tmp_path):
    from ouroboros.task_reads import TaskReads

    data, scratch = _child(tmp_path)
    drive = data / "task_drives" / f"{CHILD}-replica"
    seed_result(drive, CHILD, {"status": "completed", "result": "measured-replica"})
    seed_result(data, CHILD, {"status": "completed", "parent_task_id": LEAD, "root_task_id": LEAD,
                              "delegation_role": "subagent", "result": "stale",
                              "child_drive_root": str(drive)})
    _decide(data, "integrated", child=CHILD)  # binds the merged view: the replica owns result
    assert session_scratch.parent_hold(data, CHILD, OVER, reads=TaskReads(data)) == ""
    assert session_scratch.parent_hold(data, CHILD, OVER) == ""


def test_a_failed_folder_sync_after_the_acceptance_rename_retries_and_completes(tmp_path, monkeypatch):
    from ouroboros import utils

    data, scratch, entry = _handed_child(tmp_path)
    record = entry / SESSION_SCRATCH_DIRNAME / "snap-1" / "scratch.json"
    original, calls = utils.fsync_folder, []
    seen = {"bit": False}

    def flaky(path):
        if pathlib.Path(path) == record.parent:
            calls.append(True)
            if len(calls) == 1:
                raise OSError("synthetic first sync failure")
        return original(path)

    monkeypatch.setattr(utils, "fsync_folder", flaky)
    real_remove = task_custody._remove_entry

    def removing(what):
        # the real remover boundary: the acceptance bit is read from the record BEFORE the
        # entry goes — after the call the record and the entry are simply absent
        assert json.loads(record.read_text())[session_scratch.RELEASE_ACCEPTED]
        seen["bit"] = True
        return real_remove(what)

    monkeypatch.setattr(task_custody, "_remove_entry", removing)
    report = task_custody.remove_custody_trash(data)
    assert report["removed"] == 1 and len(calls) == 3 and not scratch.exists()
    assert seen["bit"] and not record.exists() and not entry.exists()


@pytest.mark.parametrize("new_hold", ["none", "keep", "continue"])
def test_a_persistently_unsynced_acceptance_holds_removal_until_the_sync_completes(tmp_path, monkeypatch, new_hold):
    from ouroboros import utils, workspace_preflight

    data, scratch, entry = _handed_child(tmp_path)
    record = entry / SESSION_SCRATCH_DIRNAME / "snap-1" / "scratch.json"
    original, repaired = utils.fsync_folder, {"done": False}

    def unavailable(path):
        if pathlib.Path(path) == record.parent and not repaired["done"]:
            raise OSError("synthetic persistent sync failure")
        return original(path)

    monkeypatch.setattr(utils, "fsync_folder", unavailable)
    first = task_custody.remove_custody_trash(data)
    assert first["removed"] == 0 and _classes(first) == [UNREADABLE] and scratch.is_dir()
    row = json.loads(record.read_text())
    assert row[session_scratch.RELEASE_ACCEPTED] and not session_scratch.release_finalized(row)
    assert json.loads(record.with_name(workspace_preflight.HANDOFF_RECORD).read_text(encoding="utf-8")) == {"task_id": CHILD}
    second = task_custody.remove_custody_trash(data)
    assert second["removed"] == 0 and _classes(second) == [UNREADABLE]  # ...but never bypasses the sync
    repaired["done"] = True
    if new_hold == "keep":
        assert str(scratch.parent) in owner_keep.set_owner_keep(data, CHILD, keep=True, source={"basis": "test"})["retained"]
        assert _classes(task_custody.remove_custody_trash(data)) == ["owner_keep"] and scratch.exists()
        owner_keep.set_owner_keep(data, CHILD, keep=False, source={"basis": "test"})
    elif new_hold == "continue":
        assert _claim_lead(data)[1] is True
        _next(data, "running")
        assert _classes(task_custody.remove_custody_trash(data)) == [PENDING] and scratch.exists()
        _next(data, "completed")
        _queue(data)
    assert task_custody.remove_custody_trash(data)["removed"] == 1 and not scratch.exists()


@pytest.mark.parametrize("proof_bytes", [None, "{", '{"task_id": "other-task"}'])
def test_a_direct_handoff_with_unconfirmed_owner_keeps_its_scratch(tmp_path, proof_bytes):
    from ouroboros.workspace_preflight import HANDOFF_RECORD

    data, scratch, entry = _handed_child(tmp_path)
    record = entry / SESSION_SCRATCH_DIRNAME / "snap-1" / "scratch.json"
    sidecar, before = record.with_name(HANDOFF_RECORD), record.read_bytes()
    assert json.loads(sidecar.read_text(encoding="utf-8")) == {"task_id": CHILD}
    if proof_bytes is None:
        sidecar.unlink()
    else:
        sidecar.write_text(proof_bytes, encoding="utf-8")
    assert _classes(task_custody.remove_custody_trash(data)) == [session_scratch.OWNER_UNPROVEN]
    assert scratch.is_dir() and record.read_bytes() == before
    assert {row["hold"] for row in session_scratch.scratch_records(data, OVER)} == {session_scratch.OWNER_UNPROVEN}


def test_a_later_exact_deferral_keeps_a_cancelled_childs_scratch_while_parent_lives(tmp_path):
    data, scratch = _child(tmp_path)
    seed_result(data, CHILD, {"status": "cancelled", "parent_decision": "cancelled", "parent_task_id": LEAD,
                              "root_task_id": LEAD, "delegation_role": "subagent"})
    assert session_scratch.parent_hold(data, CHILD, OVER) == ""
    _decide(data, "deferred")
    assert _classes(release_finished_scratch(data, OVER)) == [PENDING] and scratch.is_dir()
    _decide(data, "integrated")
    assert release_finished_scratch(data, OVER)["released"]


def _pred_child(tmp_path, monkeypatch):
    """A continuable root PRED (recorded owner opening) whose finished child's scratch is in the trash."""
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_owner_continue import _interrupted, _owner_mail

    data, scratch = _started(tmp_path, task_id=CHILD)
    _install_queue(data, monkeypatch)
    _interrupted(data, task_id=PRED)
    _owner_mail(data, task_id=PRED)
    seed_result(data, CHILD, {"status": "completed", "parent_task_id": PRED, "root_task_id": PRED,
                              "delegation_role": "subagent", "result": "measured"})
    _queue(data)  # a fresh empty census: PRED is proven over, so the child's record hands off
    assert release_finished_scratch(data, OVER)["released"]
    [entry] = (data / "state" / "custody_trash").iterdir()
    return data, scratch, entry


def test_a_root_continue_taken_first_holds_the_childs_scratch(tmp_path, monkeypatch):
    from supervisor.continuation_admission import admit_continuation
    from tests.test_owner_continue import NONCE

    data, scratch, entry = _pred_child(tmp_path, monkeypatch)
    assert admit_continuation(PRED, action_nonce=NONCE)["ok"]
    report = task_custody.remove_custody_trash(data)
    assert report["removed"] == 0 and _classes(report) == [PENDING]
    assert (scratch / "env" / "pyvenv.cfg").is_file()  # the extended line holds the undecided child


def test_the_claim_takes_predecessor_custody_alone(tmp_path, monkeypatch):
    import contextlib

    from supervisor import queue as queue_module
    from supervisor.continuation_admission import admit_continuation
    from tests.test_owner_continue import NONCE

    _pred_child(tmp_path, monkeypatch)
    real_custody, seen = task_custody.task_custody_lock, []

    @contextlib.contextmanager
    def custody(root, task, **kwargs):
        if task == PRED:
            seen.append((kwargs.get("timeout_sec"), queue_module._queue_lock._is_owned()))
        with real_custody(root, task, **kwargs) as held:
            yield held

    monkeypatch.setattr(task_custody, "task_custody_lock", custody)
    assert admit_continuation(PRED, action_nonce=NONCE)["ok"]
    assert seen and all(timeout == 0 and not owned for timeout, owned in seen)


def test_a_root_continue_cannot_claim_inside_the_release_window(tmp_path, monkeypatch):
    """Real competition, no staged interlock: inside the acceptance window the pass holds the
    child's and the line tail's (the claimable predecessor's) custody locks, so the Continue
    claim's predecessor-custody RMW refuses on the real lock — and the same press retries after."""
    from ouroboros import utils
    from supervisor.continuation_admission import admit_continuation
    from tests.test_owner_continue import NONCE

    data, scratch, entry = _pred_child(tmp_path, monkeypatch)
    peer = "unrelated-held"
    seed_result(data, peer, {"status": "scheduled", "_budget_pause_hold": {"reason": "owner_pause"}})
    before = read_result(data, peer)
    original_write, inside = utils.atomic_write_json, []

    def at_the_bit(path, value, **kwargs):
        if isinstance(value, dict) and session_scratch.RELEASE_ACCEPTED in value:
            inside.append(admit_continuation(PRED, action_nonce=NONCE))  # inside the window
        return original_write(path, value, **kwargs)

    monkeypatch.setattr(utils, "atomic_write_json", at_the_bit)
    report = task_custody.remove_custody_trash(data)
    assert report["removed"] == 1 and not scratch.exists()  # the acceptance is irreversible
    assert inside and not inside[0]["ok"]  # the claim refused on the held predecessor custody
    assert inside[0].get("error") == "continuation_custody_busy"
    assert not read_result(data, PRED).get("continued_by")
    assert admit_continuation(PRED, action_nonce=NONCE)["ok"]  # the same press retries after it
    assert read_result(data, PRED)["continued_by"]["action_nonce"] == NONCE
    assert not scratch.exists()  # a claim after acceptance never resurrects the scratch
    assert read_result(data, peer) == before
