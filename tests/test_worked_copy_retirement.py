"""A finished work's saved states for exact return leave after the owner's window, each with a line saying what it was.

Owner decision 2026-11-18 (4A): Ouroboros itself removes worked-out copies of finished works; whatever the owner keeps
stays. The sources that share the folders (cuts, room views, reviews, plans, focus sources) never match.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import pytest

from ouroboros.startup_migrations import prepare_startup_state
from ouroboros.task_results import write_task_result

DAYS = 2
SHA = "a" * 64


def _iso(stamp):
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat()


@pytest.fixture
def root(tmp_path, monkeypatch):
    data = tmp_path / "data"
    prepare_startup_state(data)
    monkeypatch.setenv("OUROBOROS_GC_RETENTION_DAYS", str(DAYS))
    _queue(data)
    return data


def _queue(root, running=()):
    from ouroboros.utils import atomic_write_json
    stamp = {"generation": "copies-test", "ts": _iso(time.time())}
    atomic_write_json(root / "state/queue_snapshot.json", {**stamp, "running": [{"id": task} for task in running]})
    atomic_write_json(root / "state/queue.alive", stamp)


def _file(path, text="state"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _task(root, task_id, status="completed", *, ended_days_ago=DAYS + 1, calls=True):
    write_task_result(root, task_id, status, completed_at=_iso(time.time() - ended_days_ago * 86400))
    if calls:
        _file(root / "observability" / "calls" / task_id / "llm-1.json", "{}")
    return root / "task_results" / "artifacts" / task_id / "source_handles"


def _copies(handles):
    checkpoints = handles / "context_checkpoints"
    return [_file(checkpoints / f"owner-wait-{'1' * 32}-{SHA}.json"), _file(checkpoints / f"budget-pause-sleep-{'2' * 32}-{SHA}.json"),
            _file(checkpoints / f"budget-pause-owner-pause-{'3' * 32}-{SHA}.json"), _file(checkpoints / f"working-a1-s3-{SHA}.json"),
            _file(handles / "working" / "working_checkpoint-a1.json")]


def _sources(handles):
    checkpoints = handles / "context_checkpoints"
    return [_file(checkpoints / f"{'b' * 16}-{SHA}.json", "cut"), _file(checkpoints / f"room-r1-{SHA}.json", "room"),
            _file(checkpoints / f"review-history-view-{SHA}.json", "review"),
            _file(checkpoints / f"focus_source_main-{SHA}.md", "focus"), _file(checkpoints / f"plan-author-x-{SHA}.json", "plan")]


def _ledger(handles):
    path = handles / "retired_copies.jsonl"
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines() if path.exists() else ():
        try:
            lines.append(json.loads(raw))
        except ValueError:
            continue  # a line torn by a crash is not a record
    return lines


def _prune(root):
    from ouroboros.maintenance_retirement import prune_worked_copies
    return prune_worked_copies(root)


def test_only_saved_states_of_a_finished_work_leave_and_each_leaves_its_line(root):
    handles = _task(root, "done")
    copies, sources = _copies(handles), _sources(handles)
    report = _prune(root)
    assert report["removed"] == len(copies) and not any(path.exists() for path in copies)
    assert all(path.exists() for path in sources), "sources sharing the folder are never saved states"
    lines = _ledger(handles)
    assert sorted(line["name"] for line in lines) == sorted(path.name for path in copies)
    owner_wait = next(line for line in lines if line["kind"] == "owner_wait")
    assert owner_wait["sha256"] == SHA and owner_wait["bytes"] == len("state")
    assert owner_wait["call_records"] == ["observability/calls/done"] and owner_wait["window_days"] == DAYS
    rolling = next(line for line in lines if line["kind"] == "working_rolling")
    assert rolling["sha256"] is None and rolling["folder"] == "working"


def test_a_running_work_keeps_its_saved_states_until_it_is_over(root):
    handles = _task(root, "live", status="running")
    copies = _copies(handles)
    assert _prune(root)["removed"] == 0 and all(path.exists() for path in copies), "a running work owes nothing yet"
    write_task_result(root, "live", "failed", completed_at=_iso(time.time() - (DAYS + 1) * 86400))
    assert _prune(root)["removed"] == len(copies)


def test_the_window_counts_from_the_work_end_not_from_the_file(root):
    handles = _task(root, "recent", ended_days_ago=DAYS - 1)
    copies = _copies(handles)
    for path in copies:  # a state saved long ago still waits the whole window after the work ended
        os.utime(path, (time.time() - 30 * 86400,) * 2)
    report = _prune(root)
    assert report["held"] == {"younger_than_retention": len(copies)} and all(path.exists() for path in copies)
    ended = time.time() - (DAYS - 1) * 86400
    assert abs(report["next_due"] - (ended + DAYS * 86400)) < 5


def test_the_owners_keep_holds_and_its_release_lets_them_go(root):
    from ouroboros.owner_keep import set_owner_keep

    handles = _task(root, "kept")
    copies = _copies(handles)
    set_owner_keep(root, "kept", keep=True, source={"basis": "test"})
    assert _prune(root)["held"] == {"owner_keep": len(copies)} and all(path.exists() for path in copies)
    set_owner_keep(root, "kept", keep=False, source={"basis": "test"})
    assert _prune(root)["removed"] == len(copies)


def test_without_call_records_the_record_would_name_nothing_so_the_states_stay(root):
    handles = _task(root, "bare", calls=False)
    copies = _copies(handles)
    (root / "observability" / "calls" / "bare").mkdir(parents=True)  # an empty call folder records nothing
    assert _prune(root)["held"] == {"no_call_records": len(copies)} and all(path.exists() for path in copies)
    _file(root / "observability" / "calls" / "bare" / "llm-1.json", "{}")
    assert _prune(root)["removed"] == len(copies)


def test_live_ownership_or_an_unknown_probe_keeps_them(root, monkeypatch):
    handles = _task(root, "owned")
    copies = _copies(handles)
    _queue(root, running=["owned"])
    assert _prune(root)["held"] == {"not_proven_over": len(copies)} and all(path.exists() for path in copies)
    _queue(root)
    monkeypatch.setattr("ouroboros.body_candidate.live_lineage_processes", lambda *_: [123])
    assert _prune(root)["held"] == {"not_proven_over": len(copies)}
    monkeypatch.setattr("ouroboros.body_candidate.live_lineage_processes", lambda *_: [])
    assert _prune(root)["removed"] == len(copies)


def test_a_link_with_a_saved_state_name_is_never_followed_or_removed(root, tmp_path):
    handles = _task(root, "linked")
    outside = _file(tmp_path / "outside.json", "outside")
    link = handles / "context_checkpoints" / f"owner-wait-{'9' * 32}-{SHA}.json"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside)
    _prune(root)
    assert link.is_symlink() and outside.read_text(encoding="utf-8") == "outside"


def test_packing_leaves_saved_states_loose_for_their_retirer(root):
    from ouroboros.evidence_packs import loose_members

    handles = _task(root, "packed")
    copies, sources = _copies(handles), _sources(handles)
    names = {pathlib_name(member["path"]) for member in loose_members(root, "packed") if member.get("path")}
    assert names == {path.name for path in sources}
    assert not names & {path.name for path in copies}


def pathlib_name(path):
    return str(path).rsplit("/", 1)[-1]


def test_a_finished_write_schedules_the_pass_at_the_end_of_the_window(root):
    from ouroboros import maintenance_jobs

    handles = _task(root, "scheduled", ended_days_ago=0)
    _copies(handles)
    write_task_result(root, "scheduled", "completed", completed_at=_iso(time.time()))
    due = maintenance_jobs.job(root, "runtime_retirement")["next_due"]
    assert due <= time.time() + DAYS * 86400 + 5


def test_a_queued_receiver_that_may_load_a_finished_source_keeps_it(root):
    from ouroboros.utils import atomic_write_json

    handles = _task(root, "source")
    copies = _copies(handles)
    stamp = {"generation": "copies-test", "ts": _iso(time.time())}
    receiver = {"id": "retry", "task": {"id": "retry", "_working_recovery": {"source_task_id": "source", "source_ref": {
        "path": f"source_handles/context_checkpoints/working-a1-s3-{SHA}.json"}}}}
    atomic_write_json(root / "state/queue_snapshot.json", {**stamp, "pending": [receiver], "running": []})
    atomic_write_json(root / "state/queue.alive", stamp)
    assert _prune(root)["held"] == {"lent_to_receiver": len(copies)} and all(path.exists() for path in copies)
    _queue(root)
    assert _prune(root)["removed"] == len(copies)


def test_a_child_result_still_being_promoted_holds_the_copies(root):
    from ouroboros import store

    handles = _task(root, "promoting")
    copies = _copies(handles)
    with store.open_store(root, write=True) as txn:
        txn.conn.execute("UPDATE tasks SET has_pending_promotion=1 WHERE task_id=?", ("promoting",))
    assert _prune(root)["held"] == {"child_refs_pending": len(copies)}
    with store.open_store(root, write=True) as txn:
        txn.conn.execute("UPDATE tasks SET has_pending_promotion=0 WHERE task_id=?", ("promoting",))
    assert _prune(root)["removed"] == len(copies)


def test_a_crash_between_the_line_and_the_removal_writes_no_second_line(root, monkeypatch):
    handles = _task(root, "crashed")
    copy = _file(handles / "context_checkpoints" / f"owner-wait-{'4' * 32}-{SHA}.json")
    real_unlink = os.unlink if False else type(copy).unlink
    calls = {"n": 0}

    def failing_unlink(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("shared on Windows")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(type(copy), "unlink", failing_unlink)
    first = _prune(root)
    assert copy.exists() and first["errors"] == 1 and first["held"] == {"removal_failed": 1} and len(_ledger(handles)) == 1
    assert _prune(root)["removed"] == 1 and not copy.exists()
    assert [line["name"] for line in _ledger(handles)] == [copy.name]


def test_keep_on_a_work_whose_only_material_left_is_its_saved_states_is_saved(root):
    from ouroboros.owner_keep import set_owner_keep

    handles = _task(root, "lean")
    # Its recorded child drive is already settled away: without the saved states named, Keep would be answered
    # ``already_retired`` and never saved, and the pass would then remove them.
    write_task_result(root, "lean", "completed", child_drive_root=str(root / "task_drives" / "lean-settled"))
    copies = _copies(handles)
    answer = set_owner_keep(root, "lean", keep=True, source={"basis": "test"})
    assert answer["status"] in ("kept", "partly_retired") and answer["keep"] is True
    assert {str(path) for path in copies} <= set(answer["retained"])
    assert _prune(root)["held"] == {"owner_keep": len(copies)}


def test_a_carrier_kept_in_the_shared_table_still_lends_its_source(root):
    from ouroboros.utils import atomic_write_json

    handles = _task(root, "shared")
    copies = _copies(handles)
    stamp = {"generation": "copies-test", "ts": _iso(time.time())}
    carrier = {"source_task_id": "shared", "source_ref": {"path": f"source_handles/context_checkpoints/x-{SHA}.json"}}
    digest = "d" * 64
    document = {**stamp, "pending": [{"id": "retry", "task": {"id": "retry", "_working_recovery": {"$shared": digest}}}],
                "running": [], "$shared": {digest: carrier}}
    from ouroboros import queue_observation
    atomic_write_json(root / "state/queue_snapshot.json", document)
    atomic_write_json(root / "state/queue.alive", stamp)
    expanded = queue_observation.load_queue_document(root)
    if (expanded.get("pending") or [{}])[0].get("task", {}).get("_working_recovery", {}).get("source_task_id") != "shared":
        pytest.skip("the published queue does not use this shared-table spelling")
    assert _prune(root)["held"] == {"lent_to_receiver": len(copies)}


def test_an_unreadable_or_missing_queue_lends_everything(root):
    handles = _task(root, "nolend")
    copies = _copies(handles)
    (root / "state/queue_snapshot.json").write_text("{", encoding="utf-8")
    assert _prune(root)["held"] == {"lent_to_receiver": len(copies)}
    (root / "state/queue_snapshot.json").unlink()  # no published queue: nothing proves no receiver will load it
    assert _prune(root)["held"] == {"lent_to_receiver": len(copies)}
    _queue(root)
    assert _prune(root)["removed"] == len(copies)


def test_a_near_miss_name_is_a_source_kept_and_packed(root):
    from ouroboros.evidence_packs import loose_members

    handles = _task(root, "near")
    near = _file(handles / "context_checkpoints" / f"owner-wait-notes-x-{SHA}.json", "notes")
    notes = _file(handles / "context_checkpoints" / f"working-notes-{SHA}.json", "working notes")
    _prune(root)
    assert near.exists() and notes.exists()
    packed = {pathlib_name(member["path"]) for member in loose_members(root, "near") if member.get("path")}
    assert {near.name, notes.name} <= packed


def test_a_read_of_a_retired_saved_state_names_its_record(root):
    from ouroboros.source_retention import read_retained_task_source

    handles = _task(root, "reread")
    copy = _copies(handles)[0]
    ref = {"path": f"source_handles/context_checkpoints/{copy.name}", "size": 5,
           "sha256": SHA, "kind": "task_source", "root": "artifact_store"}
    _prune(root)
    with pytest.raises(FileNotFoundError, match="was retired at .* after its work ended; what it held is in observability/calls/reread"):
        read_retained_task_source(root, "reread", ref)
    missing = dict(ref, path=f"source_handles/context_checkpoints/owner-wait-{'c' * 32}-{SHA}.json")
    with pytest.raises(FileNotFoundError) as never:
        read_retained_task_source(root, "reread", missing)
    assert "retired" not in str(never.value)


def test_releasing_a_keep_makes_the_retirement_step_owed_now(root):
    from ouroboros import maintenance_jobs
    from ouroboros.owner_keep import set_owner_keep

    handles = _task(root, "release")
    _copies(handles)
    set_owner_keep(root, "release", keep=True, source={"basis": "test"})
    before = time.time()
    set_owner_keep(root, "release", keep=False, source={"basis": "test"})
    assert maintenance_jobs.job(root, "runtime_retirement")["next_due"] <= before + 1


def test_storage_shows_what_the_pass_released_and_what_waits_its_days(root):
    from ouroboros.storage_removers import place_removers

    _copies(_task(root, "gone"))
    waiting = _copies(_task(root, "waiting", ended_days_ago=DAYS - 1))
    _prune(root)
    line = next(fact for fact in place_removers(root, engine=False)["task_results"] if fact["id"] == "saved_states")
    assert line["released"] == 5 and line["held"] == len(waiting)
    assert line["holds"][0]["class"] == "younger_than_retention" and line["holds"][0]["example"] == "waiting"


def test_a_finished_write_puts_its_row_and_the_pass_drops_it_once_nothing_is_left(root):
    from ouroboros import maintenance_jobs, store
    from ouroboros.maintenance_retirement import WORKED_COPIES

    _prune(root)  # the one seeding walk of an empty history
    assert maintenance_jobs.job(root, "runtime_retirement").get("worked_copies_seeded")
    handles = root / "task_results" / "artifacts" / "rowed" / "source_handles"
    copies = _copies(handles)
    _file(root / "observability" / "calls" / "rowed" / "llm-1.json", "{}")
    write_task_result(root, "rowed", "completed", completed_at=_iso(time.time() - (DAYS + 1) * 86400))
    with store.open_store(root, write=False) as txn:
        assert txn.get_row(WORKED_COPIES, "rowed") == {}
    assert _prune(root)["removed"] == len(copies)
    with store.open_store(root, write=False) as txn:
        assert txn.get_row(WORKED_COPIES, "rowed") is None


def test_a_torn_last_line_gets_its_newline_before_the_next_line(root):
    handles = _task(root, "torn")
    ledger = handles / "retired_copies.jsonl"
    _file(ledger, '{"name": "half')  # a crash mid-write left no newline
    copy = _file(handles / "context_checkpoints" / f"owner-wait-{'5' * 32}-{SHA}.json")
    assert _prune(root)["removed"] == 1 and not copy.exists()
    assert [line["name"] for line in _ledger(handles)] == [copy.name]


def test_a_new_generation_of_the_rolling_file_gets_its_own_line(root):
    handles = _task(root, "rolling")
    rolling = _file(handles / "working" / "working_checkpoint-a1.json", "first")
    assert _prune(root)["removed"] == 1
    _file(rolling, "second generation")  # a later write of the same name, e.g. a reopened attempt's leftover
    write_task_result(root, "rolling", "completed", completed_at=_iso(time.time() - (DAYS + 1) * 86400), title="again")
    assert _prune(root)["removed"] == 1
    assert [line["bytes"] for line in _ledger(handles)] == [len("first"), len("second generation")]


def test_read_files_fallback_names_the_record_of_a_retired_state(root):
    from ouroboros.evidence_packs import materialize_path

    handles = _task(root, "readfile")
    copy = _copies(handles)[0]
    _prune(root)
    with pytest.raises(FileNotFoundError, match="was retired at .* after its work ended"):
        materialize_path(copy)
    assert materialize_path(copy.with_name(f"owner-wait-{'e' * 32}-{SHA}.json")) is None


def test_a_queued_retry_holds_its_finished_predecessor(root):
    from ouroboros.utils import atomic_write_json

    handles = _task(root, "pred")
    copies = _copies(handles)
    stamp = {"generation": "copies-test", "ts": _iso(time.time())}
    retry = {"id": "succ", "task": {"id": "succ", "timeout_retry_from": "pred", "original_task_id": "pred"}}
    atomic_write_json(root / "state/queue_snapshot.json", {**stamp, "pending": [retry], "running": []})
    atomic_write_json(root / "state/queue.alive", stamp)
    assert _prune(root)["held"] == {"lent_to_receiver": len(copies)}


def test_a_creation_time_is_no_end(root):
    old = _iso(time.time() - 30 * 86400)
    write_task_result(root, "undated", "completed", ts=old)
    _file(root / "observability" / "calls" / "undated" / "llm-1.json", "{}")
    copies = _copies(root / "task_results" / "artifacts" / "undated" / "source_handles")
    held = _prune(root)["held"]
    assert held in ({"finish_time_unknown": len(copies)}, {"younger_than_retention": len(copies)})
    assert all(path.exists() for path in copies)
