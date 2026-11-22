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
    # A live publisher renews its stamp every few seconds; on a loaded runner one pass can outlast the 10-second
    # freshness, so the test's queue stays fresh as long as a stale-queue test does not age it on purpose.
    monkeypatch.setattr("ouroboros.task_status._SNAPSHOT_OWNERSHIP_FRESH_SEC", 600.0)
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
    """One pass, the test's published queue stamped again first: a slow runner must not see it go stale."""
    from ouroboros.saved_state_retirement import prune_worked_copies
    from ouroboros.utils import atomic_write_json
    try:
        document = json.loads((root / "state/queue_snapshot.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        document = None  # a missing or torn queue stays as the test left it
    if isinstance(document, dict):
        stamp = {"generation": document.get("generation") or "copies-test", "ts": _iso(time.time())}
        atomic_write_json(root / "state/queue_snapshot.json", {**document, **stamp})
        atomic_write_json(root / "state/queue.alive", stamp)
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


def test_a_running_works_first_pass_only_dates_and_its_end_lets_every_saved_state_go(root):
    """While the work runs, a pass removes nothing the same day: the two pause copies its row never named are proven
    and wait the owner's days, the rest is not proven (its rules: ``test_running_saved_states``). Once the work has
    ended longer ago than those days, every saved state of it goes, whatever the running dates said."""
    handles = _task(root, "live", status="running")
    copies = _copies(handles)
    report = _prune(root)
    assert report["removed"] == 0 and report["waiting"] == 2 and all(path.exists() for path in copies), report
    write_task_result(root, "live", "failed", completed_at=_iso(time.time() - (DAYS + 1) * 86400))
    assert _prune(root)["removed"] == len(copies)


def test_the_window_counts_from_the_work_end_not_from_the_file(root):
    handles = _task(root, "recent", ended_days_ago=DAYS - 1)
    copies = _copies(handles)
    for path in copies:  # a state saved long ago still waits the whole window after the work ended
        os.utime(path, (time.time() - 30 * 86400,) * 2)
    report = _prune(root)
    assert report["waiting"] == len(copies) and all(path.exists() for path in copies)
    assert report["held"] == {}, "the owner's days are the rule, not a keep (maintenance_jobs: age waits are not holds)"
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
    _queue(root, running=["owned"])  # the queue runs it: a receiver of its own saved states
    assert _prune(root)["held"] == {"lent_to_receiver": len(copies)} and all(path.exists() for path in copies)
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
    ended = time.time()
    write_task_result(root, "scheduled", "completed", completed_at=_iso(ended))
    due = maintenance_jobs.job(root, "runtime_retirement")["next_due"]
    assert abs(due - (ended + DAYS * 86400)) < 5, "due when the window ends, neither now nor later"


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
    assert {f"saved_states:{path.parent}" for path in copies} <= set(answer["retained"]), "the folders, never a path per file"
    assert not {str(path) for path in copies} & set(answer["retained"])
    assert _prune(root)["held"] == {"owner_keep": len(copies)}


@pytest.mark.parametrize("operation", ["unreadable", {"state": "accounted", "files": 1}])
def test_a_malformed_release_record_still_saves_keep_and_the_next_pass_leaves_the_saved_states(root, operation):
    """A protective action does not depend on the evidence it protects: whatever the release record holds, an explicit
    Keep is saved and the next eligible pass leaves the saved states; a release on the same record raises."""
    from ouroboros.owner_keep import set_owner_keep
    from ouroboros.payload_release_state import KEY
    from ouroboros.task_store import patch_document, read_document

    handles = _task(root, "malformed")
    copies = _copies(handles)
    patch_document(root, "malformed", lambda row: {**row, KEY: {"operations": {"old-record": operation}}})
    with pytest.raises(ValueError):
        set_owner_keep(root, "malformed", keep=False, source={"basis": "test"})
    answer = set_owner_keep(root, "malformed", keep=True, source={"basis": "test"})
    assert answer["keep"] and answer["unknown"] == ["canonical_payloads"], answer
    assert read_document(root, "malformed").get("owner_keep")
    report = _prune(root)
    assert report["held"] == {"owner_keep": len(copies)} and report["removed"] == 0, report
    assert all(path.is_file() for path in copies)


def test_a_first_saved_state_arriving_after_the_folder_walk_is_kept_by_its_folder(root, monkeypatch):
    """The only material of a work is a saved state written after Keep's first folder walk: the canonical walk names
    its folder, so Keep is saved (never already_retired) and the next pass leaves the file."""
    from ouroboros import saved_state_retirement
    from ouroboros.owner_keep import set_owner_keep
    from ouroboros.task_store import read_document

    handles = _task(root, "arrival")
    write_task_result(root, "arrival", "completed", child_drive_root=str(root / "task_drives" / "arrival-settled"))
    copy = _file(handles / "working" / "working_checkpoint-a1.json")
    with monkeypatch.context() as patch:
        patch.setattr(saved_state_retirement, "worked_copies_in", lambda _folder: [])  # the first walk saw nothing
        answer = set_owner_keep(root, "arrival", keep=True, source={"basis": "test"})
    assert answer["status"] != "already_retired" and f"saved_states:{copy.parent}" in answer["retained"], answer
    assert str(copy) not in answer["retained"] and read_document(root, "arrival").get("owner_keep")
    report = _prune(root)
    assert report["held"] == {"owner_keep": 1} and copy.is_file(), report


@pytest.mark.parametrize("walk", ["missed", "raised"])
def test_keep_names_a_saved_state_by_what_it_is_even_when_the_folder_walk_misses_it(root, monkeypatch, walk):
    """Keep's per-file listing leaves a saved state out by its folder and its writer's name, not by the earlier walk:
    one that lands after that walk, or a walk that raises, still never appears per file. Sources beside it and an
    ordinary stored file that merely shares a saved state's name are listed."""
    from ouroboros import saved_state_retirement
    from ouroboros.owner_keep import inventory

    handles = _task(root, "late")
    copies, sources = _copies(handles), _sources(handles)
    namesake = _file(handles.parent / "exports" / "working" / "working_checkpoint-a1.json")
    if walk == "missed":
        monkeypatch.setattr(saved_state_retirement, "worked_copies_in", lambda _folder: [])
    else:
        def unreadable(_folder):
            raise PermissionError("folder not searchable")

        monkeypatch.setattr(saved_state_retirement, "worked_copies_in", unreadable)
    places, unknown, _pins = inventory(root, "late", {})
    assert not {str(path) for path in copies} & set(places), places
    assert {str(path) for path in sources} | {str(namesake)} <= set(places), places
    assert ("saved_states" in unknown) == (walk == "raised")


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
    from ouroboros.saved_state_retirement import prune_worked_copies
    from ouroboros.utils import atomic_write_json
    old = {"generation": "copies-test", "ts": _iso(time.time() - 3600)}  # a queue that stopped publishing an hour ago
    atomic_write_json(root / "state/queue_snapshot.json", {**old, "running": []})
    atomic_write_json(root / "state/queue.alive", old)
    assert prune_worked_copies(root)["held"] == {"lent_to_receiver": len(copies)}
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
    with pytest.raises(FileNotFoundError, match="was retired at .* after its work ended; the work's model calls are in observability/calls/reread; its exact bytes are not kept"):
        read_retained_task_source(root, "reread", ref)
    missing = dict(ref, path=f"source_handles/context_checkpoints/owner-wait-{'c' * 32}-{SHA}.json")
    with pytest.raises(FileNotFoundError) as never:
        read_retained_task_source(root, "reread", missing)
    assert "retired" not in str(never.value)


def test_releasing_a_keep_makes_the_retirement_step_owed_now(root):
    from ouroboros import maintenance_jobs
    from ouroboros.owner_keep import set_owner_keep

    handles = _task(root, "release", ended_days_ago=0)  # its own window ends in two days: no deadline is due yet
    _copies(handles)
    write_task_result(root, "release", "completed", completed_at=_iso(time.time()), title="with its files")
    set_owner_keep(root, "release", keep=True, source={"basis": "test"})
    assert maintenance_jobs.job(root, "runtime_retirement")["next_due"] > time.time() + DAYS * 86400 - 60
    before = time.time()
    set_owner_keep(root, "release", keep=False, source={"basis": "test"})
    assert maintenance_jobs.job(root, "runtime_retirement")["next_due"] <= before + 1


def test_storage_shows_what_the_pass_released_and_what_it_keeps_never_the_window(root):
    from ouroboros.owner_keep import set_owner_keep
    from ouroboros.storage_removers import place_removers

    _copies(_task(root, "gone"))
    _copies(_task(root, "waiting", ended_days_ago=DAYS - 1))  # inside the owner's days: the rule, never a hold
    _copies(_task(root, "kept"))
    set_owner_keep(root, "kept", keep=True, source={"basis": "test"})
    _prune(root)
    line = next(fact for fact in place_removers(root, engine=False)["task_results"] if fact["id"] == "saved_states")
    assert line["released"] == 5 and line["held"] == 5
    assert [(hold["class"], hold["example"]) for hold in line["holds"]] == [("owner_keep", "kept")]


def test_a_finished_write_puts_its_row_and_the_pass_drops_it_once_nothing_is_left(root):
    from ouroboros import maintenance_jobs, store
    from ouroboros.saved_state_retirement import WORKED_COPIES

    _prune(root)  # the one seeding walk of an empty history
    assert maintenance_jobs.job(root, "runtime_retirement").get("worked_copies_seeded", {}).get("at")
    handles = root / "task_results" / "artifacts" / "rowed" / "source_handles"
    copies = _copies(handles)
    _file(root / "observability" / "calls" / "rowed" / "llm-1.json", "{}")
    write_task_result(root, "rowed", "completed", completed_at=_iso(time.time() - (DAYS + 1) * 86400))
    with store.open_store(root, write=False) as txn:
        assert txn.get_row(WORKED_COPIES, "rowed") is not None
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


def test_a_queued_retry_holds_its_finished_predecessor(root):
    from ouroboros.utils import atomic_write_json

    handles = _task(root, "pred")
    copies = _copies(handles)
    stamp = {"generation": "copies-test", "ts": _iso(time.time())}
    retry = {"id": "succ", "task": {"id": "succ", "timeout_retry_from": "pred", "original_task_id": "pred"}}
    atomic_write_json(root / "state/queue_snapshot.json", {**stamp, "pending": [], "running": [retry]})
    atomic_write_json(root / "state/queue.alive", stamp)
    assert _prune(root)["held"] == {"lent_to_receiver": len(copies)}
    _queue(root)
    assert _prune(root)["removed"] == len(copies)


def test_a_creation_time_is_no_end(root):
    old = _iso(time.time() - 30 * 86400)
    write_task_result(root, "undated", "completed", ts=old)
    _file(root / "observability" / "calls" / "undated" / "llm-1.json", "{}")
    copies = _copies(root / "task_results" / "artifacts" / "undated" / "source_handles")
    report = _prune(root)  # its window runs from when the host first saw it settled (the walk), never from ``ts``
    assert report["waiting"] == len(copies) and report["held"] == {} and all(path.exists() for path in copies)
    from ouroboros.saved_state_retirement import prune_worked_copies
    assert prune_worked_copies(root, now=time.time() + (DAYS + 1) * 86400)["removed"] == len(copies)


def test_the_maintenance_child_waits_the_owners_days_not_the_default(root, monkeypatch):
    from ouroboros import startup_historical_audit

    seen = {}
    monkeypatch.setattr("ouroboros.process_custody.spawn_supervised", lambda argv, **kwargs: seen.update(kwargs["env"]))
    startup_historical_audit.spawn_maintenance_job(root, root, "maintenance")
    assert seen["OUROBOROS_GC_RETENTION_DAYS"] == str(DAYS)
    monkeypatch.setenv("OUROBOROS_GC_RETENTION_DAYS", "11")
    startup_historical_audit.spawn_maintenance_job(root, root, "maintenance")
    assert seen["OUROBOROS_GC_RETENTION_DAYS"] == "11"


def test_a_whole_retry_line_of_a_queued_task_is_lent(root):
    from ouroboros.utils import atomic_write_json

    first, middle = _copies(_task(root, "first")), _copies(_task(root, "middle"))
    write_task_result(root, "first", "completed", completed_at=_iso(time.time() - (DAYS + 1) * 86400), retry_task_id="middle")
    write_task_result(root, "middle", "completed", completed_at=_iso(time.time() - (DAYS + 1) * 86400),
                      timeout_retry_from="first", original_task_id="first", retry_task_id="last")
    write_task_result(root, "last", "running", timeout_retry_from="middle", original_task_id="middle")
    stamp = {"generation": "copies-test", "ts": _iso(time.time())}
    atomic_write_json(root / "state/queue_snapshot.json", {**stamp, "pending": [], "running": [{"id": "last", "task": {"id": "last"}}]})
    atomic_write_json(root / "state/queue.alive", stamp)
    assert _prune(root)["held"] == {"lent_to_receiver": len(first) + len(middle)}
    assert all(path.exists() for path in first + middle)


def test_another_body_on_the_store_makes_the_history_walk_again(root):
    from ouroboros import store

    _prune(root)
    handles = _task(root, "older")  # finished under a body that puts no rows: no row before the files appeared
    copies = _copies(handles)
    assert _prune(root)["removed"] == 0, "the walk ran under this body already"
    with store.open_store(root, write=True) as txn:
        txn.set_meta("last_store_body", {"sha": "0" * 40, "at": _iso(time.time())})
    assert _prune(root)["removed"] == len(copies)
    later = _copies(_task(root, "older-again"))  # an older body booted, finished this, and the same body came back:
    with store.open_store(root, write=True) as txn:  # the sha is the same, the fact's ``at`` is not
        txn.set_meta("last_store_body", {"sha": "0" * 40, "at": _iso(time.time() + 60)})
    assert _prune(root)["removed"] == len(later)


def test_a_call_folder_holding_only_a_parts_manifest_records_no_call(root):
    handles = _task(root, "nocalls", calls=False)
    copies = _copies(handles)
    _file(root / "observability" / "calls" / "nocalls" / "working-a1.json", "{}")
    assert _prune(root)["held"] == {"no_call_records": len(copies)}
    _file(root / "observability" / "calls" / "nocalls" / "llm-1.json", "{}")
    assert _prune(root)["removed"] == len(copies)


def test_only_read_file_asks_for_the_record_of_a_missed_name(root):
    from ouroboros.evidence_packs import materialize_path

    copy = _copies(_task(root, "quiet"))[0]
    _prune(root)
    assert materialize_path(copy) is None, "other callers keep the plain miss they handle"
    with pytest.raises(FileNotFoundError, match="was retired at .* after its work ended; the work's model calls are in"):
        materialize_path(copy, explain=True)
    assert materialize_path(copy.with_name(f"owner-wait-{'e' * 32}-{SHA}.json"), explain=True) is None


def test_a_saved_state_an_earlier_pack_holds_is_the_packers_to_unlink(root, monkeypatch):
    from ouroboros import evidence_packs, saved_state_retirement

    import hashlib

    def addressed(text, wait):  # a real content-addressed name: the packer verifies the digest
        return handles / "context_checkpoints" / f"owner-wait-{wait * 32}-{hashlib.sha256(text.encode()).hexdigest()}.json"

    handles = _task(root, "prepacked")
    packed = _file(addressed("packed before", "6"), "packed before")
    with monkeypatch.context() as older_body:  # a body before this change packed every handle
        older_body.setattr(saved_state_retirement, "worked_copy", lambda folder, name: None)
        evidence_packs.pack_task(root, "prepacked")
    assert not packed.exists()
    from ouroboros import store
    from ouroboros.saved_state_retirement import WORKED_COPIES

    _prune(root)  # the folder holds no loose saved state: its row goes
    assert evidence_packs.materialize_handle(root, "prepacked", f"source_handles/context_checkpoints/{packed.name}")
    with store.open_store(root, write=False) as txn:
        assert (txn.get_row(WORKED_COPIES, "prepacked") or {}).get("settled_at"), "written back: owed again, with a date"
    evidence_packs.pack_task(root, "prepacked")
    assert not packed.exists(), "the packer unlinks a loose copy of a member it already holds, as before"
    assert evidence_packs.materialize_handle(root, "prepacked", f"source_handles/context_checkpoints/{packed.name}")
    fresh = _file(addressed("never packed", "7"), "never packed")
    assert _prune(root)["removed"] == 2 and not packed.exists() and not fresh.exists()
    lines = {line["name"]: line for line in _ledger(handles)}
    assert lines[packed.name]["packed"] == ["observability/packs/prepacked.xz"] and "stay there" in lines[packed.name]["rebuild"]
    assert "packed" not in lines[fresh.name] and "are not kept" in lines[fresh.name]["rebuild"]
    said = saved_state_retirement.retired_message
    assert said(root, "prepacked", packed.name).endswith("its exact bytes stay in observability/packs/prepacked.xz")
    assert said(root, "prepacked", fresh.name).endswith("its exact bytes are not kept")


def test_read_file_of_a_retired_saved_state_says_what_became_of_it(root):
    from ouroboros.tools.core_file_tools import _read_file
    from ouroboros.tools.tool_context import ToolContext

    handles = _task(root, "reader")
    copy = _copies(handles)[0]
    _prune(root)
    ctx = ToolContext(repo_dir=root, drive_root=root, task_id="reader", budget_drive_root=str(root))
    answer = _read_file(ctx, f"source_handles/context_checkpoints/{copy.name}", root="artifact_store")
    assert "NOT_FOUND" in answer and "was retired at" in answer and "observability/calls/reader" in answer
    plain = _read_file(ctx, f"source_handles/context_checkpoints/owner-wait-{'f' * 32}-{SHA}.json", root="artifact_store")
    assert "NOT_FOUND" in plain and "was retired at" not in plain
    data = _read_file(ctx, f"task_results/artifacts/reader/source_handles/context_checkpoints/{copy.name}", root="runtime_data")
    assert "was retired at" in data and "does not exist yet" not in data, "the data root's read names the record too"


def test_a_receiver_published_during_the_pass_is_seen_before_the_next_task(root):
    from ouroboros import saved_state_retirement
    from ouroboros.utils import atomic_write_json

    first, second = _copies(_task(root, "a-first")), _copies(_task(root, "b-second"))
    real = saved_state_retirement._retire
    published = []

    def publish_after_first(*args, **kwargs):
        real(*args, **kwargs)
        if not published:  # a queue publication lands while the pass works through its rows
            stamp = {"generation": "copies-test-2", "ts": _iso(time.time())}
            receiver = {"id": "retry", "task": {"id": "retry", "_working_recovery": {"source_task_id": "b-second"}}}
            atomic_write_json(root / "state/queue_snapshot.json", {**stamp, "pending": [receiver], "running": []})
            atomic_write_json(root / "state/queue.alive", stamp)
            published.append(True)

    saved_state_retirement._retire = publish_after_first
    try:
        report = _prune(root)
    finally:
        saved_state_retirement._retire = real
    assert not any(path.exists() for path in first) and all(path.exists() for path in second)
    assert report["held"] == {"lent_to_receiver": len(second)}


def test_a_row_put_again_during_the_pass_stays_owed(root):
    from ouroboros import saved_state_retirement, store
    from ouroboros.saved_state_retirement import WORKED_COPIES

    handles = _task(root, "again")
    _copies(handles)
    real = saved_state_retirement._retire

    def a_new_terminal_write(*args, **kwargs):
        real(*args, **kwargs)
        with store.open_store(root, write=True) as txn:
            txn.put_row(WORKED_COPIES, "again", {"write_seq": 999})

    saved_state_retirement._retire = a_new_terminal_write
    try:
        _prune(root)
    finally:
        saved_state_retirement._retire = real
    with store.open_store(root, write=False) as txn:
        assert txn.get_row(WORKED_COPIES, "again") == {"write_seq": 999}


def test_a_write_of_an_old_finished_work_makes_no_run_of_its_own(root):
    from ouroboros import maintenance_jobs

    handles = _task(root, "old", ended_days_ago=30)
    _sources(handles)  # its folder keeps its sources forever: nothing in it is a saved state
    _prune(root)
    before = maintenance_jobs.job(root, "runtime_retirement").get("next_due")
    write_task_result(root, "old", "completed", completed_at=_iso(time.time() - 30 * 86400), title="patched later")
    due = maintenance_jobs.job(root, "runtime_retirement").get("next_due")
    assert due == before or due > time.time(), "a past end is the next pass's, never a run of its own"
    from ouroboros import store
    from ouroboros.saved_state_retirement import WORKED_COPIES
    with store.open_store(root, write=False) as txn:
        assert txn.get_row(WORKED_COPIES, "old") is None, "a folder of sources only owes the retirer nothing"
    fresh = _task(root, "fresh", ended_days_ago=0)  # the same write of a work that just ended schedules its window
    _copies(fresh)
    write_task_result(root, "fresh", "completed", completed_at=_iso(time.time()), title="with its files")
    assert maintenance_jobs.job(root, "runtime_retirement")["next_due"] > time.time() + DAYS * 86400 - 60


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="POSIX folder permissions")
def test_one_unreadable_folder_is_owed_and_the_others_go_in_the_same_pass(root):
    from ouroboros import store
    from ouroboros.saved_state_retirement import WORKED_COPIES

    readable = _copies(_task(root, "readable"))
    locked = _task(root, "locked")
    copies = _copies(locked)
    os.chmod(locked / "context_checkpoints", 0)
    try:
        report = _prune(root)
        assert report["removed"] == len(readable) and report["errors"] == 1
        with store.open_store(root, write=False) as txn:
            assert txn.get_row(WORKED_COPIES, "locked") is not None, "the unreadable folder stays owed"
    finally:
        os.chmod(locked / "context_checkpoints", 0o755)
    assert _prune(root)["removed"] == len(copies)



@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="POSIX folder permissions")
def test_a_terminal_write_beside_an_unreadable_folder_lands_and_stays_owed(root):
    from ouroboros import store
    from ouroboros.saved_state_retirement import WORKED_COPIES
    from ouroboros.task_results import load_task_result

    handles = root / "task_results" / "artifacts" / "shut" / "source_handles"
    (handles / "context_checkpoints").mkdir(parents=True)
    os.chmod(handles, 0)  # nothing under it can be listed or even looked up
    try:
        write_task_result(root, "shut", "completed", completed_at=_iso(time.time()))
        assert (load_task_result(root, "shut") or {}).get("status") == "completed", "the write itself never fails"
        with store.open_store(root, write=False) as txn:
            assert txn.get_row(WORKED_COPIES, "shut") is not None, "what cannot be read may hold saved states"
    finally:
        os.chmod(handles, 0o755)



def test_keep_after_a_pass_took_the_saved_states_says_so_never_kept(root, monkeypatch):
    from ouroboros import owner_keep
    from tests._owner_keep_answer import outside_store

    handles = _task(root, "raced")
    write_task_result(root, "raced", "completed", child_drive_root=str(root / "task_drives" / "raced-settled"))
    copies = _copies(handles)
    sources = _sources(handles)  # the folders stay with their sources after the pass
    real = owner_keep.inventory

    def pass_between_inventory_and_lock(*args, **kwargs):  # the pass holds the custody lock Keep then waits for
        found = real(*args, **kwargs)
        report = _prune(root)
        assert report["removed"] == len(copies), report
        return found

    monkeypatch.setattr(owner_keep, "inventory", pass_between_inventory_and_lock)
    answer = owner_keep.set_owner_keep(root, "raced", keep=True, source={"basis": "test"})
    assert not any(path.exists() for path in copies)
    # #3981: Keep also names every file of the canonical store; the folders' sources stay there, so the answer is
    # partly retired (the saved states gone, the sources kept), and no saved state is ever reported retained.
    rest = outside_store(answer, root, "raced")
    assert answer["status"] == "partly_retired" and not rest["retained"], answer
    assert not {str(path) for path in copies} & set(answer["retained"]), answer
    assert {str(path) for path in sources} <= set(answer["retained"]), answer  # other stored files are listed
    assert {place for place in answer["retired"] if place.startswith("saved_states:")} == {
        f"saved_states:{handles / 'context_checkpoints'}", f"saved_states:{handles / 'working'}"}



def test_a_retry_walk_that_cannot_finish_lends_everything(root, monkeypatch):
    from ouroboros import source_retention, store

    copies = _copies(_task(root, "ancestor"))
    write_task_result(root, "receiver", "running")
    _queue(root, running=["receiver"])
    real = source_retention._source_locations

    def busy(root_, task_id, **kwargs):  # a competing transaction on one of the walk's reads
        yield from list(real(root_, task_id, **kwargs))[:1]
        raise store.StoreBusy("store busy")

    monkeypatch.setattr(source_retention, "_source_locations", busy)
    assert _prune(root)["held"] == {"lent_to_receiver": len(copies)}, "an unfinished walk proves no predecessor free"
    monkeypatch.setattr(source_retention, "_source_locations", real)
    assert _prune(root)["removed"] == len(copies)


def test_a_task_written_after_the_pass_read_it_waits_for_the_next_pass(root, monkeypatch):
    from ouroboros import saved_state_retirement

    copies = _copies(_task(root, "finalized"))
    real = saved_state_retirement._call_records

    def finalize_meanwhile(root_, task_id):  # its artifacts are published after the pass read the row: the end moves
        write_task_result(root_, task_id, "completed", completed_at=_iso(time.time() - (DAYS + 1) * 86400),
                          artifact_finalized_at=_iso(time.time()))
        return real(root_, task_id)

    monkeypatch.setattr(saved_state_retirement, "_call_records", finalize_meanwhile)
    report = _prune(root)
    assert report["removed"] == 0 and report["held"] == {"changed": len(copies)} and all(path.exists() for path in copies)
    monkeypatch.setattr(saved_state_retirement, "_call_records", real)
    report = _prune(root)
    assert report["removed"] == 0 and report["waiting"] == len(copies), "the new end starts the owner's days again"


def test_a_read_that_moves_only_the_access_time_is_no_change_and_a_new_write_is(root, monkeypatch):
    from ouroboros import saved_state_retirement

    read, rewritten = _copies(_task(root, "read")), _copies(_task(root, "rewritten"))
    real = saved_state_retirement._call_records

    def meanwhile(root_, task_id):  # after the listing: one task's files are read (Linux relatime), the other's written
        for path in read if task_id == "read" else rewritten:
            info = os.stat(path)
            moved = (info.st_atime_ns + 10**12, info.st_mtime_ns) if task_id == "read" else (info.st_atime_ns, info.st_mtime_ns + 10**9)
            os.utime(path, ns=moved)
        return real(root_, task_id)

    monkeypatch.setattr(saved_state_retirement, "_call_records", meanwhile)
    report = _prune(root)
    assert report["removed"] == len(read) and not any(path.exists() for path in read), report
    assert report["held"] == {"changed": len(rewritten)} and all(path.exists() for path in rewritten)


def test_every_removal_follows_a_folder_sync_of_its_record(root, monkeypatch):
    import errno

    from ouroboros import saved_state_retirement

    copies = _copies(_task(root, "synced"))
    synced = []
    real = saved_state_retirement.fsync_folder

    def first_fails(folder):
        if not synced:
            synced.append(False)
            raise OSError(errno.EIO, "disk error")
        real(folder)
        synced.append(True)

    monkeypatch.setattr(saved_state_retirement, "fsync_folder", first_fails)
    report = _prune(root)
    assert synced[0] is False and report["removed"] >= 1
    assert synced.count(True) >= report["removed"], "no file goes before its record's folder entry is synced"
    assert sum(path.exists() for path in copies) == len(copies) - report["removed"]


def test_the_queue_is_read_once_per_publication_or_certificate(root, monkeypatch):
    from ouroboros import saved_state_retirement
    from ouroboros.saved_state_retirement import prune_worked_copies
    from ouroboros.utils import atomic_write_json

    for name in ("q-a", "q-b", "q-c"):
        _copies(_task(root, name))
    (root / "state/queue_snapshot.json").unlink()
    parsed = []
    real = saved_state_retirement._published_lent
    monkeypatch.setattr(saved_state_retirement, "_published_lent", lambda root_: parsed.append(1) or real(root_))
    assert _prune(root)["held"] == {"lent_to_receiver": 15} and len(parsed) == 1, "a missing queue is read once a pass"
    old = {"generation": "copies-test", "ts": _iso(time.time() - 3600)}
    atomic_write_json(root / "state/queue_snapshot.json", {**old, "running": []})
    atomic_write_json(root / "state/queue.alive", old)
    ends = []
    real_end = saved_state_retirement._work_end

    def renewed_after_the_first(document):  # the publisher renews the same generation's certificate between two tasks
        ends.append(1)
        if len(ends) == 2:
            atomic_write_json(root / "state/queue.alive", {"generation": "copies-test", "ts": _iso(time.time())})
        return real_end(document)

    monkeypatch.setattr(saved_state_retirement, "_work_end", renewed_after_the_first)
    report = prune_worked_copies(root)
    assert report["held"] == {"lent_to_receiver": 5} and report["removed"] == 10, "a renewed certificate is read at once"
    for name in ("q-d", "q-e", "q-f"):  # finished after the walk: their own terminal writes put their rows
        _copies(_task(root, name))
        write_task_result(root, name, "completed", completed_at=_iso(time.time() - (DAYS + 1) * 86400))
    fresh = {"generation": "copies-test", "ts": _iso(time.time())}
    atomic_write_json(root / "state/queue_snapshot.json", {**fresh, "running": []})
    atomic_write_json(root / "state/queue.alive", fresh)
    parsed.clear()
    ends.clear()

    def renewed_each_task(document):  # the publisher renews its certificate every few seconds of a busy pass
        atomic_write_json(root / "state/queue.alive", {"generation": "copies-test", "ts": _iso(time.time())})
        return real_end(document)

    monkeypatch.setattr(saved_state_retirement, "_work_end", renewed_each_task)
    report = prune_worked_copies(root)
    assert report["removed"] == 15 + 5 and len(parsed) == 1, "a fresh answer stands while only the certificate moves"


def test_a_retired_line_names_the_pack_generation_that_holds_the_bytes(root, monkeypatch):
    import hashlib

    from ouroboros import evidence_packs, saved_state_retirement

    handles = _task(root, "multigen")

    def addressed(text, wait):
        return handles / "context_checkpoints" / f"owner-wait-{wait * 32}-{hashlib.sha256(text.encode()).hexdigest()}.json"

    with monkeypatch.context() as older_body:  # a body before this change packed every handle, one generation a pass
        older_body.setattr(saved_state_retirement, "worked_copy", lambda folder, name: None)
        _file(addressed("first generation", "8"), "first generation")
        evidence_packs.pack_task(root, "multigen")
        second = _file(addressed("second generation", "9"), "second generation")
        evidence_packs.pack_task(root, "multigen")
    assert (root / "observability" / "packs" / "multigen.2.xz").exists() and not second.exists()
    assert evidence_packs.materialize_handle(root, "multigen", f"source_handles/context_checkpoints/{second.name}")
    assert _prune(root)["removed"] == 1 and not second.exists()
    line = next(line for line in _ledger(handles) if line["name"] == second.name)
    assert line["packed"] == ["observability/packs/multigen.2.xz"], "the generation that holds these bytes"


def test_a_partial_pack_listing_holds_the_saved_states_and_a_whole_one_names_the_generation(root, monkeypatch):
    """One generation's sidecar is unreadable while the pack index is busy: the listing then misses that generation,
    and a line saying "its exact bytes are not kept" would be false (``handle_bytes`` still returns them)."""
    import hashlib

    from ouroboros import evidence_packs, saved_state_retirement, store

    handles = _task(root, "partial")

    def addressed(text, wait):
        return handles / "context_checkpoints" / f"owner-wait-{wait * 32}-{hashlib.sha256(text.encode()).hexdigest()}.json"

    with monkeypatch.context() as older_body:  # a body before #3789 packed every handle, one generation a pass
        older_body.setattr(saved_state_retirement, "worked_copy", lambda folder, name: None)
        first = _file(addressed("first generation", "8"), "first generation")
        evidence_packs.pack_task(root, "partial")
        _file(addressed("second generation", "9"), "second generation")
        evidence_packs.pack_task(root, "partial")
    rel = f"source_handles/context_checkpoints/{first.name}"
    assert evidence_packs.materialize_handle(root, "partial", rel) and first.exists()
    sidecars = [evidence_packs.packs_dir(root) / evidence_packs.sidecar_name("partial", generation) for generation in (1, 2)]
    intact, index = [path.read_bytes() for path in sidecars], evidence_packs.packs_dir(root) / evidence_packs.INDEX_NAME
    sidecars[0].write_text("{", encoding="utf-8")
    with store.name_lock(root, evidence_packs.INDEX_LOCK_NAME, timeout_sec=0), store.acquisition_budget(0):
        report = _prune(root)
    assert report["held"] == {"packs_unknown": 1} and report["removed"] == 0, report
    assert first.exists() and not _ledger(handles), "an unknown pack is never written down as bytes not kept"
    with evidence_packs.open_index(root, write=True) as conn:  # an index rebuilt while a sidecar was unreadable says so
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('sidecar_errors', ?)", (json.dumps(["unreadable sidecar"]),))
    assert _prune(root)["held"] == {"packs_unknown": 1} and first.exists(), "its rows may miss that generation"
    index.unlink()
    evidence_packs.ensure_index(root)  # rebuilt while generation 1's sidecar is unreadable: generation 2's rows only
    sidecars[1].write_text("{", encoding="utf-8")  # then generation 2's sidecar fails too: no sidecar of it reads
    assert _prune(root)["held"] == {"packs_unknown": 1} and first.exists() and not _ledger(handles), \
        "a faulted index's rows are not the whole listing"
    for path, data in zip(sidecars, intact):
        path.write_bytes(data)
    index.unlink()
    evidence_packs.ensure_index(root)  # rebuilt from readable sidecars: every generation, no fault recorded
    sidecars[0].write_text("{", encoding="utf-8")
    again = _prune(root)  # an index that recorded no unreadable sidecar makes the listing whole
    assert again["removed"] == 1 and not first.exists(), again["held"]
    line = next(line for line in _ledger(handles) if line["name"] == first.name)
    assert line["packed"] == ["observability/packs/partial.xz"], "the generation that holds these bytes"
    assert evidence_packs.handle_bytes(root, "partial", rel) == b"first generation"


def test_an_index_that_recorded_an_unreadable_sidecar_answers_no_whole_listing_though_it_holds_rows(root, monkeypatch):
    """An index rebuilt while generation 1's sidecar was unreadable records that fault and holds only generation 2's
    rows: while the sidecar stays unreadable those rows are not the whole listing (``whole``); once it reads, the
    task's own sidecars are."""
    import hashlib

    from ouroboros import evidence_packs, saved_state_retirement

    handles = _task(root, "validated")
    text = "packed by an older body"
    first = handles / "context_checkpoints" / f"owner-wait-{'8' * 32}-{hashlib.sha256(text.encode()).hexdigest()}.json"
    with monkeypatch.context() as older_body:  # a body before #3789 packed every handle
        older_body.setattr(saved_state_retirement, "worked_copy", lambda folder, name: None)
        _file(first, text)
        evidence_packs.pack_task(root, "validated")
    notes = hashlib.sha256(b"notes").hexdigest()
    _file(handles / "context_checkpoints" / f"working-notes-{notes}.json", "notes")  # a source: generation 2
    evidence_packs.pack_task(root, "validated")
    assert evidence_packs.materialize_handle(root, "validated", f"source_handles/context_checkpoints/{first.name}")
    sidecar = evidence_packs.packs_dir(root) / evidence_packs.sidecar_name("validated", 1)
    intact = sidecar.read_bytes()

    def rebuilt():
        (evidence_packs.packs_dir(root) / evidence_packs.INDEX_NAME).unlink()
        evidence_packs.ensure_index(root)  # the packing pass's rebuild, from the sidecars as they read now

    sidecar.write_text("{", encoding="utf-8")
    rebuilt()
    report = _prune(root)
    assert report["held"] == {"packs_unknown": 1} and first.exists() and not _ledger(handles), report
    sidecar.write_bytes(intact)
    rebuilt()
    assert _prune(root)["removed"] == 1 and not first.exists()
    assert _ledger(handles)[0]["packed"] == ["observability/packs/validated.xz"]



def test_a_recovered_crash_with_no_recorded_end_goes_its_days_after_it_was_written_settled(root):
    from ouroboros.saved_state_retirement import prune_worked_copies

    handles = root / "task_results" / "artifacts" / "crashed" / "source_handles"
    copies = _copies(handles)
    _file(root / "observability" / "calls" / "crashed" / "llm-1.json", "{}")
    _prune(root)  # the walk of an empty history
    settled = time.time()
    write_task_result(root, "crashed", "failed", reason_code="worker_crash_owner_wait")  # recovery: no observed end
    report = _prune(root)
    assert report["waiting"] == len(copies) and report["held"] == {}, "counted from the settled write, never unknown"
    assert prune_worked_copies(root, now=settled + (DAYS + 1) * 86400)["removed"] == len(copies)
    line = _ledger(handles)[0]
    assert abs(datetime.fromisoformat(line["finished_at"]).timestamp() - settled) < 60


def test_a_new_obligation_is_owed_now_and_a_later_write_only_by_an_end_ahead(root):
    from ouroboros import maintenance_jobs, store

    _prune(root)  # the walk of an empty history
    handles = root / "task_results" / "artifacts" / "long" / "source_handles"
    _copies(handles)
    _file(root / "observability" / "calls" / "long" / "llm-1.json", "{}")
    assert maintenance_jobs.job(root, "runtime_retirement").get("next_due") is None
    before = time.time()  # it ran longer than the owner's days: its projected date is its creation
    write_task_result(root, "long", "completed", ts=_iso(time.time() - (DAYS + 5) * 86400))
    assert maintenance_jobs.job(root, "runtime_retirement")["next_due"] <= before + 5, "a new obligation is owed now"
    assert _prune(root)["waiting"] == 5  # no recorded end: its days run from the settled write, and the row stays
    with store.open_store(root, write=True) as txn:
        maintenance_jobs.record_in(txn, "runtime_retirement", next_due=None)
    write_task_result(root, "long", "completed", title="patched later")
    assert maintenance_jobs.job(root, "runtime_retirement").get("next_due") is None, "an old work's patch wakes nothing"


def test_a_promotion_published_after_the_first_read_holds_under_the_lock(root, monkeypatch):
    from ouroboros import saved_state_retirement, store

    copies = _copies(_task(root, "late-child"))
    real = saved_state_retirement._call_records

    def promotion_meanwhile(root_, task_id):
        with store.open_store(root_, write=True) as txn:
            txn.conn.execute("UPDATE tasks SET has_pending_promotion=1 WHERE task_id=?", (task_id,))
        return real(root_, task_id)

    monkeypatch.setattr(saved_state_retirement, "_call_records", promotion_meanwhile)
    assert _prune(root)["held"] == {"child_refs_pending": len(copies)} and all(path.exists() for path in copies)


def test_a_pooled_retry_reading_its_predecessors_retired_state_is_told_what_became_of_it(root):
    from ouroboros.tools.core_file_tools import _read_file
    from ouroboros.tools.tool_context import ToolContext

    copy = _copies(_task(root, "pred"))[0]
    write_task_result(root, "pred", "completed", completed_at=_iso(time.time() - (DAYS + 1) * 86400), retry_task_id="succ")
    write_task_result(root, "succ", "completed", completed_at=_iso(time.time() - (DAYS + 1) * 86400),
                      timeout_retry_from="pred", original_task_id="pred")
    _prune(root)
    assert not copy.exists()
    child = root / "state" / "headless_tasks" / "succ" / "data"  # the pool runs it on its own drive
    child.mkdir(parents=True)
    ctx = ToolContext(repo_dir=root, drive_root=child, task_id="succ", budget_drive_root=str(root))
    answer = _read_file(ctx, f"source_handles/context_checkpoints/{copy.name}", root="artifact_store")
    assert "NOT_FOUND" in answer and "was retired at" in answer and "observability/calls/pred" in answer, answer



def test_a_new_walk_keeps_when_the_host_first_saw_a_work_settled(root):
    from ouroboros import store
    from ouroboros.saved_state_retirement import WORKED_COPIES, prune_worked_copies

    write_task_result(root, "nodate", "completed", ts=_iso(time.time() - 30 * 86400))  # no recorded end
    _file(root / "observability" / "calls" / "nodate" / "llm-1.json", "{}")
    copies = _copies(root / "task_results" / "artifacts" / "nodate" / "source_handles")
    assert _prune(root)["waiting"] == len(copies)  # the walk dates it now
    with store.open_store(root, write=False) as txn:
        first = txn.get_row(WORKED_COPIES, "nodate")["settled_at"]
    with store.open_store(root, write=True) as txn:  # another body booted: the walk runs again
        txn.set_meta("last_store_body", {"sha": "1" * 40, "at": _iso(time.time() + 60)})
    _prune(root)
    with store.open_store(root, write=False) as txn:
        assert txn.get_row(WORKED_COPIES, "nodate")["settled_at"] == first, "an install that updates often still gets there"
    assert prune_worked_copies(root, now=first + (DAYS + 1) * 86400)["removed"] == len(copies)
